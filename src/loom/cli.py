"""Loom CLI — wire all pipeline stages together.

Commands (each stage runnable independently):
  ingest     — Run all source adapters, produce Events
  resolve    — Run person resolution (union-find)
  score      — Heuristic event scoring (Stage 2)
  triage     — Run LLM triage on events
  propagate  — Run message passing on triage claims
  extract    — Run deep commitment extraction
  synthesize — Run claims -> beliefs compiler
  commitments — Show commitments with full provenance trace
  brief      — Generate the blind spot briefing
  stats      — Show database stats
  pipeline   — Run the full pipeline end-to-end
  eval       — Data quality evaluation (check, sample, review, stats, run)

Usage:
  python -m loom.cli ingest --hours 168
  python -m loom.cli pipeline --dry-run
  python -m loom.cli brief --days 7
  python -m loom.cli stats
  python -m loom.cli eval check --stage 0
  python -m loom.cli eval sample --stage 4 --n 10
  python -m loom.cli eval review --stage 4
  python -m loom.cli eval stats --stage 4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from loom.constants import (
    CLOUD_DEEP_MODEL,
    CLOUD_LITE_MODEL,
    CLOUD_REASONING_MODEL,
    HOURS_ALL,
    LOOM_DIR,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    USER_TIMEZONE,
    WATCH_DEBOUNCE_SECONDS,
    WATCH_POLL_CALENDAR_SECONDS,
    WATCH_POLL_FILE_SOURCES_SECONDS,
    WATCH_POLL_GRANOLA_SECONDS,
    WATCH_POLL_SLACK_SECONDS,
    safe_timezone,
)
from loom.profile import flatten_profile, load_profile
from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)

# Sources in default ingest order
DEFAULT_SOURCES = [
    "mail", "imessage", "calendar", "slack",
    "granola", "whatsapp", "contacts",
    "calls_macos", "calls_whatsapp",
]

# Ambient sources — opt-in via --ambient flag
AMBIENT_SOURCES = [
    "knowledgec", "safari", "chrome", "notes", "shell_history",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_llm_client(dry_run: bool, llm_provider: str = "ollama"):
    """Get the appropriate LLM client based on flags.

    Args:
        dry_run: If True, return MockLLMClient regardless of provider.
        llm_provider: One of 'ollama', 'gemini', 'mock'.
    """
    if dry_run or llm_provider == "mock":
        from loom.llm.mock import MockLLMClient
        return MockLLMClient()

    if llm_provider == "gemini":
        from loom.llm.gemini import GeminiClient
        return GeminiClient()

    # Default: ollama
    try:
        from loom.llm.ollama import OllamaClient
        return OllamaClient()
    except ImportError:
        logger.warning("OllamaClient not available, falling back to mock")
        from loom.llm.mock import MockLLMClient
        return MockLLMClient()


def _since_timestamp(hours: int | None, since: str | None) -> int:
    """Convert --hours or --since to a Unix timestamp."""
    if since:
        from datetime import datetime, timezone
        try:
            dt = datetime.fromisoformat(since)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            print(f"Error: --since must be ISO format (e.g. 2026-02-01T00:00:00), got: {since}")
            sys.exit(1)

    if hours is not None:
        return int(time.time()) - hours * SECONDS_PER_HOUR

    # Default: all available data
    return 0


def _load_user_config() -> dict:
    """Load user config from profile.yaml or config.json.

    Delegates to loom.profile.load_profile() and flattens for backward compat.
    """
    from loom.profile import flatten_profile, load_profile
    return flatten_profile(load_profile())


def _detect_user_email_from_sent(store) -> str | None:
    """Auto-detect user's email from sent mail messages.

    Queries events from the 'mail' source where is_from_me=True and extracts
    the sender's email. This is a fallback for when profile.yaml has no emails.
    """
    try:
        row = store.conn.execute("""
            SELECT participants FROM events
            WHERE source = 'mail'
              AND json_extract(metadata, '$.is_from_me') = 1
            ORDER BY timestamp DESC LIMIT 1
        """).fetchone()
        if not row:
            return None
        import json as _json
        participants = _json.loads(row["participants"] or "[]")
        if not participants:
            return None
        # First participant in a sent message is the sender (user)
        sender = participants[0]
        # Parse "Name <email>" format
        if "<" in sender and ">" in sender:
            email = sender.split("<")[1].split(">")[0].strip()
        elif "@" in sender:
            email = sender.strip()
        else:
            return None
        return email.lower() if email else None
    except Exception:
        return None


def _print_header(title: str) -> None:
    """Print a section header."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def _print_stats_table(stats: dict) -> None:
    """Print a dict as a formatted key-value table."""
    max_key_len = max(len(str(k)) for k in stats) if stats else 0
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k:>{max_key_len}}: {v}")
        else:
            print(f"  {str(key):>{max_key_len}}: {value}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: ingest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_ingest(args: argparse.Namespace) -> None:
    """Run all source adapters, produce Events."""
    _print_header("Stage 0: Ingest")

    store = LayeredGraphStore(args.db_path)
    since = _since_timestamp(args.hours, args.since)
    sources = args.sources if args.sources else DEFAULT_SOURCES
    if getattr(args, "ambient", False):
        sources = list(sources) + AMBIENT_SOURCES

    try:
        from loom.adapters import get_adapter
    except ImportError:
        print("Error: loom.adapters module not available yet.")
        store.close()
        return

    total_fetched = 0
    total_inserted = 0

    for source_name in sources:
        try:
            adapter = get_adapter(source_name)
        except (ImportError, KeyError, AttributeError):
            print(f"  {source_name}: adapter not available, skipping")
            continue

        if adapter is None:
            print(f"  {source_name}: unknown source")
            continue

        try:
            availability = adapter.check_availability()
            if not availability.available:
                msg = availability.user_action or availability.reason or "unavailable"
                print(f"  {source_name}: {msg}")
                continue
        except Exception:
            pass

        print(f"  {source_name}: ingesting...", end=" ", flush=True)

        try:
            # Contacts are identity bridges — always ingest fully
            # regardless of --limit so the resolver can merge persons.
            source_limit = 0 if source_name == "contacts" else (args.limit or 0)
            result = adapter.ingest(since_ts=since, limit=source_limit)
            fetched = len(result.events)
            inserted = store.put_events_batch(result.events) if result.events else 0
            total_fetched += fetched
            total_inserted += inserted
            errors_msg = f" ({len(result.errors)} errors)" if result.errors else ""
            print(f"{fetched} fetched, {inserted} new{errors_msg}")
            for err in result.errors[:3]:
                print(f"    warning: {err}")
        except Exception as exc:
            print(f"error: {exc}")

    print(f"\n  Total: {total_fetched} fetched, {total_inserted} new")

    store.update_sync_state(
        source="ingest",
        last_event_ts=int(time.time()),
        event_count=total_inserted,
        status="complete",
    )
    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: resolve
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_resolve(args: argparse.Namespace) -> None:
    """Run person resolution (union-find)."""
    _print_header("Stage 0b: Person Resolution")

    store = LayeredGraphStore(args.db_path)

    try:
        from loom.resolver import persist_persons, resolve_persons
    except ImportError:
        print("Error: loom.resolver module not available yet.")
        store.close()
        return

    try:
        user_config = _load_user_config()
        # Auto-detect user email from sent mail if profile has none
        if not user_config.get("emails"):
            detected = _detect_user_email_from_sent(store)
            if detected:
                user_config["emails"] = [detected]
                print(f"  Auto-detected user email: {detected}")
        persons = resolve_persons(store, user_config=user_config)
        result = persist_persons(store, persons)
        print(f"  Persons resolved: {len(persons)}")
        print(f"  Persons written: {result.get('persons_written', 0)}")
        print(f"  Identifiers written: {result.get('identifiers_written', 0)}")
        user_found = result.get('user_found', False)
        print(f"  User found: {user_found}")
        if not user_found:
            print("  ⚠ WARNING: Could not identify your user account.")
            print("    Add your email to ~/.loom/profile.yaml under 'emails:' to fix this.")
    except Exception as exc:
        print(f"Error during resolution: {exc}")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: dedup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_dedup(args: argparse.Namespace) -> None:
    """Stage 0b2: Person deduplication."""
    _print_header("Stage 0b2: Person Deduplication")

    store = LayeredGraphStore(args.db_path)

    try:
        from loom.dedup import run_dedup
    except ImportError:
        print("Error: loom.dedup module not available.")
        store.close()
        return

    skip_llm = getattr(args, "skip_llm", False)
    llm_client = None
    if not skip_llm:
        llm_provider = getattr(args, "llm", "gemini")
        llm_client = _get_llm_client(dry_run=False, llm_provider=llm_provider)

    save_dir = str(LOOM_DIR / "dedup_results")

    try:
        result = run_dedup(
            store,
            llm_client=llm_client,
            skip_llm=skip_llm,
            save_dir=save_dir,
        )
        print(f"  Layer 1: {result['layer1_groups']} groups, {result['layer1_merged']} merged")
        print(f"  Layer 2: {result['layer2_clusters']} clusters, {result['layer2_merged']} merged")
        print(f"  Total: {result['before']:,} → {result['after']:,} persons")
    except Exception as exc:
        print(f"Error during dedup: {exc}")
        import traceback
        traceback.print_exc()

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: link
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_link(args: argparse.Namespace) -> None:
    """Link events to persons."""
    _print_header("Stage 0c: Event-Person Linking")

    store = LayeredGraphStore(args.db_path)

    try:
        from loom.linker import link_events_to_persons
    except ImportError:
        print("Error: loom.linker module not available yet.")
        store.close()
        return

    since = _since_timestamp(args.hours, args.since)
    try:
        result = link_events_to_persons(store, since_ts=since)
        print(f"  Edges created: {result.get('edges_created', 0)}")
        print(f"  Events linked: {result.get('events_linked', 0)}")
        print(f"  Events unlinked: {result.get('events_unlinked', 0)}")
        print(f"  Persons referenced: {result.get('persons_referenced', 0)}")
    except Exception as exc:
        print(f"Error during linking: {exc}")

    store.close()


def cmd_relink(args: argparse.Namespace) -> None:
    """Delete and rebuild all event_persons edges with thread-based linking."""
    _print_header("Relink: Rebuild Event-Person Edges")

    store = LayeredGraphStore(args.db_path)

    from loom.linker import link_events_to_persons

    # Count before
    before = store.conn.execute("SELECT COUNT(*) FROM event_persons").fetchone()[0]
    print(f"  Edges before: {before:,}")

    # Delete all existing edges
    print("  Deleting all event_persons edges...")
    store.conn.execute("DELETE FROM event_persons")
    store.conn.commit()

    # Rebuild
    print("  Rebuilding with thread-based linking...")
    t0 = time.time()
    result = link_events_to_persons(store)
    store.conn.commit()
    elapsed = time.time() - t0

    after = store.conn.execute("SELECT COUNT(*) FROM event_persons").fetchone()[0]
    print(f"  Edges after:  {after:,}")
    print(f"  Reduction:    {before - after:,} edges ({(before - after) / before * 100:.1f}%)")
    print(f"  Elapsed:      {elapsed:.1f}s")
    print(f"  Events linked:      {result.get('events_linked', 0):,}")
    print(f"  Membership edges:   {result.get('membership_edges', 0):,}")
    print(f"  Mention edges:      {result.get('mention_edges', 0):,}")

    # Rebuild Stage 1 claims that depend on event_persons counts
    if not args.skip_claims:
        print("\n  Rebuilding Stage 1 claims (communication frequency, etc.)...")
        try:
            from loom.claims_stage1 import extract_stage1_claims
            # Delete existing stage1 claims
            stage1_types = [
                "communication_frequency", "communication_channel",
                "directionality", "timing_pattern", "recency",
                "thread_activity",
            ]
            placeholders = ",".join("?" for _ in stage1_types)
            store.conn.execute(
                f"DELETE FROM claims WHERE claim_type IN ({placeholders})",
                stage1_types,
            )
            store.conn.commit()
            claims_result = extract_stage1_claims(store)
            print(f"  Stage 1 claims rebuilt: {claims_result.get('claims_created', 0):,}")
        except Exception as exc:
            print(f"  Warning: Stage 1 rebuild failed: {exc}")

    store.close()
    print("\n  Done.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: claims
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_claims(args: argparse.Namespace) -> None:
    """Extract deterministic Stage 1 claims."""
    _print_header("Stage 1: Deterministic Claims")

    store = LayeredGraphStore(args.db_path)
    since = _since_timestamp(args.hours, args.since)

    try:
        from loom.claims_stage1 import extract_stage1_claims, populate_person_profiles
    except ImportError:
        print("Error: loom.claims_stage1 module not available yet.")
        store.close()
        return

    # Default to 30 days if no --hours/--since provided
    if not since:
        since = int(time.time()) - 30 * SECONDS_PER_DAY

    try:
        result = extract_stage1_claims(store, since_ts=since)
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Total claims: {result.get('total_claims', 'n/a')}")
            print(f"  New claims: {result.get('new_claims', 'n/a')}")
            by_type = result.get("by_type", {})
            if by_type:
                print("  By type:")
                for ctype, count in sorted(by_type.items()):
                    print(f"    {ctype}: {count}")

            # Populate person profiles from Stage 1 claims
            pp_result = populate_person_profiles(store)
            print(f"  Person profiles: {pp_result['profiles_written']} populated")
            if pp_result["tier_changes"]:
                print(f"  Tier changes: {pp_result['tier_changes']}")
    except Exception as exc:
        print(f"Error during claims extraction: {exc}")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: annotate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_annotate(args: argparse.Namespace) -> None:
    """Extract structural annotations from events (Stage 1.5)."""
    _print_header("Stage 1.5: Structural Annotations")

    store = LayeredGraphStore(args.db_path)
    since = _since_timestamp(args.hours, args.since)

    from loom.annotate import annotate_structural
    from loom.constants import LIMIT_ALL

    events = store.get_events(since=since, limit=LIMIT_ALL)
    result = annotate_structural(store, events)
    print(f"  Events processed: {result['events_processed']}")
    print(f"  Annotations extracted: {result['annotations_total']}")
    print(f"  Annotations written: {result['annotations_written']}")
    print(f"  Elapsed: {result['elapsed_seconds']:.1f}s")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_score(args: argparse.Namespace) -> None:
    """Run heuristic event scoring (Stage 2)."""
    _print_header("Stage 2: Heuristic Scoring")

    store = LayeredGraphStore(args.db_path)
    since = _since_timestamp(args.hours, args.since)
    lens = getattr(args, "lens", "chief_of_staff")

    from loom.score import run_scoring

    result = run_scoring(store, since_ts=since, lens=lens)
    print(f"  Lens: {result['lens']}")
    print(f"  Events scored: {result['scored']}")
    print(f"  Projections written: {result['projections_written']}")
    print(f"  Routes: skip={result['skip']}, low_priority={result['low_priority']}, "
          f"full_triage={result['full_triage']}")
    by_source = result.get("by_source", {})
    if by_source:
        print("  By source:")
        for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
            print(f"    {source:>12s}: {count:>4d}")
    print(f"  Elapsed: {result['elapsed_seconds']:.1f}s")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: triage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_triage(args: argparse.Namespace) -> None:
    """Run LLM triage on events."""
    _print_header("Stage 4: LLM Triage")

    store = LayeredGraphStore(args.db_path)
    llm = _get_llm_client(args.dry_run, getattr(args, "llm", "ollama"))
    since = _since_timestamp(args.hours, args.since)

    try:
        from loom.triage import run_triage
    except ImportError:
        print("Error: loom.triage module not available yet.")
        store.close()
        return

    lens = getattr(args, "lens", "chief_of_staff")
    user_config = _load_user_config()
    cloud_all = user_config.get("sensitivity_mode", "cloud_all") == "cloud_all"

    from loom.profile import format_profile_context, load_profile
    profile = load_profile()
    profile_ctx = format_profile_context(profile)
    try:
        result = run_triage(store, llm, since_ts=since, lens=lens, cloud_all=cloud_all, profile_context=profile_ctx)
        print(f"  Lens: {result.get('lens', 'none')}")
        print(f"  Events triaged: {result.get('triaged', 'n/a')}")
        print(f"  Claims written: {result.get('claims_written', 'n/a')}")
        print(f"  Failed: {result.get('failed', 0)}")
        print(f"  Threads: {result.get('threads', 0)} "
              f"(cloud={result.get('thread_full', 0)}, "
              f"local={result.get('routed_to_local', 0)})")
        elapsed = result.get("elapsed_seconds", 0)
        print(f"  Elapsed: {elapsed:.1f}s")
        print("  Tier distribution:")
        for tier in ("ignore", "lightweight", "deep"):
            print(f"    {tier:>12s}: {result.get(tier, 0)}")
    except Exception as exc:
        print(f"Error during triage: {exc}")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: propagate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_propagate(args: argparse.Namespace) -> None:
    """Run message passing on triage claims."""
    _print_header("Stage 5: Score Propagation")

    store = LayeredGraphStore(args.db_path)

    try:
        from loom.propagate import run_propagation
    except ImportError:
        print("Error: loom.propagate module not available yet.")
        store.close()
        return

    from loom.profile import load_profile
    profile = load_profile()
    try:
        result = run_propagation(store, profile=profile)
        print(f"  Rounds: {result.get('rounds', 'n/a')}")
        print(f"  Claims adjusted: {result.get('adjusted', 'n/a')}")
        pre = result.get("pre_tiers", {})
        post = result.get("post_tiers", {})
        if pre and post:
            print("  Tier changes:")
            for tier in ("ignore", "lightweight", "deep"):
                p = pre.get(tier, 0)
                q = post.get(tier, 0)
                print(f"    {tier:>12s}: {p:>4d} -> {q:>4d} ({q - p:+d})")
    except Exception as exc:
        print(f"Error during propagation: {exc}")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: extract
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_extract(args: argparse.Namespace) -> None:
    """Run deep commitment extraction."""
    _print_header("Stage 6: Deep Extraction")

    store = LayeredGraphStore(args.db_path)
    llm = _get_llm_client(args.dry_run, getattr(args, "llm", "ollama"))

    try:
        from loom.extract import run_extraction
    except ImportError:
        print("Error: loom.extract module not available yet.")
        store.close()
        return

    from loom.profile import format_profile_context, load_profile
    profile = load_profile()
    profile_ctx = format_profile_context(profile)
    try:
        is_gemini = getattr(args, "llm", "") == "gemini"
        cloud_model = CLOUD_LITE_MODEL if is_gemini else ""
        user_config = _load_user_config()
        user_email = (user_config.get("emails") or [""])[0]
        result = run_extraction(
            store, local_llm=llm, cloud_llm=llm,
            cloud_model=cloud_model, user_email=user_email,
            profile_context=profile_ctx,
        )
        print(f"  Threads processed: {result.get('threads_processed', 'n/a')}")
        print(f"  Actionable: {result.get('actionable', 0)}")
        print(f"  Logistics: {result.get('logistics', 0)}")
        print(f"  Relational: {result.get('relational', 0)}")
        print(f"  Errors: {result.get('errors', 0)}")
        print(f"  Skipped: {result.get('skipped', 0)}")
    except Exception as exc:
        print(f"Error during extraction: {exc}")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: synthesize
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_synthesize(args: argparse.Namespace) -> None:
    """Run claims -> beliefs compiler."""
    _print_header("Stage 7: Belief Synthesis")

    store = LayeredGraphStore(args.db_path)
    llm = _get_llm_client(args.dry_run, getattr(args, "llm", "ollama"))

    try:
        from loom.beliefs import run_synthesis
    except ImportError:
        print("Error: loom.beliefs module not available yet.")
        store.close()
        return

    from loom.profile import format_profile_context, load_profile
    profile = load_profile()
    profile_ctx = format_profile_context(profile)
    try:
        is_gemini = getattr(args, "llm", "") == "gemini"
        model = CLOUD_DEEP_MODEL if is_gemini else ""
        lite_model = CLOUD_LITE_MODEL if is_gemini else ""
        user_config = _load_user_config()
        user_email = (user_config.get("emails") or [""])[0]
        result = run_synthesis(
            store, llm, model=model, lite_model=lite_model,
            user_email=user_email,
            profile_context=profile_ctx,
        )
        print(f"  Actionable threads: {result.get('actionable_threads', 0)}")
        print(f"  Logistics threads: {result.get('logistics_threads', 0)}")
        print(f"  Relational threads: {result.get('relational_threads', 0)}")
        print(f"  Bundles processed: {result.get('bundles_processed', 0)}")
        print(f"  Commitments: {result.get('total_commitments', 0)}")
        print(f"  Logistics facts: {result.get('total_logistics', 0)}")
        print(f"  Relational contexts: {result.get('total_relational', 0)}")
        print(f"  Dedup merged: {result.get('dedup_merged', 0)}")
        print(f"  Entity beliefs: {result.get('entity_beliefs', 0)}")
        pib = result.get('participant_inference_beliefs', 0)
        if pib:
            print(f"  Participant inference beliefs: {pib}")
        print(f"  Relation beliefs: {result.get('relation_beliefs', 0)}")
        rcb = result.get('relational_context_beliefs', 0)
        if rcb:
            print(f"  Relational context beliefs (FOAF): {rcb}")
        cfb = result.get('commitment_fact_beliefs', 0)
        cfm = result.get('commitment_fact_merged', 0)
        print(f"  Commitment FACT beliefs: {cfb} ({cfm} merged)")
        print(f"  Logistics FACT beliefs: {result.get('logistics_fact_beliefs', 0)}")
        print(f"  Elapsed: {result.get('elapsed_seconds', 0):.1f}s")
        if result.get("synthesis_errors"):
            print(f"  Synthesis errors: {result['synthesis_errors']}")
        if result.get("logistics_errors"):
            print(f"  Logistics errors: {result['logistics_errors']}")
        if result.get("relational_errors"):
            print(f"  Relational errors: {result['relational_errors']}")
    except Exception as exc:
        print(f"Error during synthesis: {exc}")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: commitments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_commitments(args: argparse.Namespace) -> None:
    """Show commitments with full provenance trace."""
    import json as _json
    from datetime import datetime, timezone

    store = LayeredGraphStore(args.db_path)

    show_overdue = getattr(args, "overdue", False)
    show_all = getattr(args, "all", False)
    person_filter = getattr(args, "person", None)
    limit = getattr(args, "n", 20)
    show_trace = getattr(args, "trace", True)

    _print_header("Commitments")

    # Query commitment claims
    # Include claims with empty superseded_by (incorrectly superseded)
    if show_all:
        rows = store.conn.execute(
            """SELECT id, subject, object, confidence, created_at, superseded_by
               FROM claims
               WHERE claim_type = 'commitment'
                 AND json_extract(object, '$.status') = 'open'
               ORDER BY json_extract(object, '$.priority') ASC,
                        confidence DESC"""
        ).fetchall()
    else:
        rows = store.conn.execute(
            """SELECT id, subject, object, confidence, created_at, superseded_by
               FROM claims
               WHERE claim_type = 'commitment'
                 AND (superseded_by IS NULL OR superseded_by = '')
                 AND json_extract(object, '$.status') = 'open'
               ORDER BY json_extract(object, '$.priority') ASC,
                        confidence DESC"""
        ).fetchall()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    shown = 0

    for r in rows:
        if shown >= limit:
            break

        try:
            obj = _json.loads(r["object"])
        except (_json.JSONDecodeError, TypeError):
            continue

        deadline = obj.get("deadline")
        is_overdue = deadline and deadline < today
        if show_overdue and not is_overdue:
            continue

        who = obj.get("who", "user")
        to_whom = obj.get("to_whom", "")
        if person_filter:
            pf = person_filter.lower()
            if pf not in (who or "").lower() and pf not in (to_whom or "").lower():
                continue

        shown += 1
        what = obj.get("what", "(n/a)")
        ctype = obj.get("type", "unknown")
        priority = obj.get("priority", 3)
        confidence = r["confidence"]
        direction = obj.get("direction", "?")

        # Header
        overdue_tag = " [OVERDUE]" if is_overdue else ""
        print(f"  [{shown}] {what}{overdue_tag}")
        print(f"      Type: {ctype}  Priority: {priority}  Confidence: {confidence:.1f}")
        if deadline:
            print(f"      Deadline: {deadline}")
        if who:
            owner = f"{who}" + (f" -> {to_whom}" if to_whom else "")
            print(f"      Owner: {owner}  Direction: {direction}")

        evidence = obj.get("evidence_quote")
        if evidence:
            eq = evidence[:120]
            print(f'      Evidence: "{eq}"')

        if not show_trace:
            print()
            continue

        # ── Provenance trace ──
        source_event_id = obj.get("source_event_id") or r["subject"]
        evt = store.conn.execute(
            """SELECT id, source, event_type, timestamp, participants, metadata
               FROM events WHERE id = ?""",
            (source_event_id,),
        ).fetchone()

        if evt:
            ts = datetime.fromtimestamp(
                evt["timestamp"], tz=timezone.utc,
            ).strftime("%Y-%m-%d %H:%M") if evt["timestamp"] else "?"
            meta = _json.loads(evt["metadata"] or "{}")
            subj = meta.get("subject", meta.get("title", evt["event_type"]))
            print(f"      Source: {evt['source']} | {str(subj)[:60]}")
            print(f"      Time:   {ts}")

            # Resolved people
            people = store.conn.execute(
                """SELECT p.canonical_name, ep.role
                   FROM event_persons ep
                   JOIN persons p ON ep.person_id = p.person_id
                   WHERE ep.event_id = ?""",
                (evt["id"],),
            ).fetchall()
            if people:
                parts = [
                    f'{p["canonical_name"]} ({p["role"]})'
                    for p in people[:5]
                ]
                print(f"      People: {', '.join(parts)}")

            # Triage claim
            triage = store.conn.execute(
                """SELECT object FROM claims
                   WHERE subject = ? AND claim_type = 'triage'
                     AND superseded_by IS NULL
                   LIMIT 1""",
                (evt["id"],),
            ).fetchone()
            if triage:
                tv = _json.loads(triage["object"] or "{}")
                reason = str(tv.get("reason", ""))[:60]
                print(
                    f"      Triage: score={tv.get('score', '?')} "
                    f"domain={tv.get('domain', '?')} "
                    f'reason="{reason}"'
                )
        else:
            print(f"      Source: event {str(source_event_id)[:16]}... (not found)")

        # Beliefs referencing this claim
        beliefs = store.conn.execute(
            """SELECT belief_type, summary, confidence
               FROM beliefs
               WHERE source_claims LIKE ?
               LIMIT 3""",
            (f'%{r["id"]}%',),
        ).fetchall()
        if beliefs:
            for b in beliefs:
                print(
                    f"      Belief: [{b['belief_type']}] "
                    f"conf={b['confidence']:.1f} — {b['summary'][:60]}"
                )

        print()

    # Summary
    total = len(rows)
    overdue_count = sum(
        1 for r in rows
        if (d := _json.loads(r["object"]).get("deadline")) and d < today
    )
    print(f"  Total open: {total}  Overdue: {overdue_count}  Shown: {shown}")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: brief
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_brief(args: argparse.Namespace) -> None:
    """Generate the blind spot briefing."""
    _print_header("Stage 8: Blind Spot Briefing")

    store = LayeredGraphStore(args.db_path)
    llm = _get_llm_client(args.dry_run, getattr(args, "llm", "ollama"))

    from loom.briefing import run_briefing

    is_gemini = getattr(args, "llm", "") == "gemini"
    model = CLOUD_DEEP_MODEL if is_gemini else ""

    interactive = not getattr(args, "no_interactive", False)

    result = run_briefing(
        store, llm,
        days_ahead=args.days,
        user_tz=args.tz,
        model=model,
        interactive=interactive,
        thinking_level=getattr(args, "thinking", None),
    )

    # Print the markdown briefing
    print(result["briefing"])

    # Print stats
    print("\n---")
    cstats = result.get("commitments", {})
    astats = result.get("anticipation", {})
    print(f"Generated in {result['elapsed_s']:.1f}s")
    print(f"Calendar events: {result['events_count']}")
    print(f"Commitments: {cstats.get('total_open', 0)} open, "
          f"{cstats.get('matched_to_calendar', 0)} matched, "
          f"{cstats.get('unscheduled', 0)} orphaned, "
          f"{cstats.get('overdue', 0)} overdue")
    print(f"Anticipation: {astats.get('system_queries', 0)} graph queries "
          f"({astats.get('system_results', 0)} results), "
          f"{astats.get('web_searches', 0)} web searches "
          f"({astats.get('web_results', 0)} results), "
          f"{astats.get('user_questions', 0)} user questions "
          f"({astats.get('user_answers', 0)} answered), "
          f"{astats.get('reassurances', 0)} reassurances")
    print(f"Blind spots: {astats.get('blind_spot_candidates', 0)} candidates → "
          f"{astats.get('blind_spot_final', 0)} ranked")
    print(f"Prompt: {result['prompt_length']} chars")

    # Save raw I/O for analysis
    import json as _json
    ts = int(time.time())
    save_dir = LOOM_DIR / "briefings"
    save_dir.mkdir(exist_ok=True)

    # Save briefing markdown
    briefing_path = save_dir / f"briefing_{ts}.md"
    briefing_path.write_text(result["briefing"])
    print(f"\nBriefing saved to {briefing_path}")

    # Save all raw I/O as JSON
    raw = result.get("raw", {})
    raw_path = save_dir / f"briefing_{ts}_raw.json"
    raw_path.write_text(_json.dumps(raw, indent=2, default=str))
    print(f"Raw I/O saved to {raw_path}")

    # Also save if --save specified
    if args.save:
        Path(args.save).write_text(result["briefing"])
        print(f"Also saved to {args.save}")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: topic-normalize
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_topic_normalize(args: argparse.Namespace) -> None:
    """Run topic normalization pipeline."""
    _print_header("Topic Normalization")

    store = LayeredGraphStore(args.db_path)

    from loom.topic_normalize import run_normalization
    stats = run_normalization(store)

    print(f"  Raw unique topics:      {stats['raw_unique_topics']}")
    print(f"  Synonym mappings:       {stats['synonym_mappings']}")
    print(f"  Canonical topics:       {stats['canonical_topics']}")
    print(f"  Batch artifact combos:  {stats['batch_artifact_combos']}")
    print(f"  Artifacts stripped:     {stats['batch_artifacts_stripped']}")
    print(f"  LLM groups:             {stats['llm_groups']}")
    print(f"  Annotations updated:    {stats['annotations_renormalized']}")
    print(f"  Duration:               {stats['duration_seconds']}s")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: topic-stats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_topic_stats(args: argparse.Namespace) -> None:
    """Show topic frequency distribution from annotations."""
    _print_header("Topic Statistics")

    store = LayeredGraphStore(args.db_path)

    from loom.topic_normalize import topic_stats
    stats = topic_stats(store)

    print(f"  Total annotations:  {stats['total_annotations']}")
    print(f"  Unique topics:      {stats['unique_topics']}")
    print(f"  Singletons:         {stats['singletons']}")
    print()
    print("  Top 20 topics:")
    for topic, count in stats["top_20"].items():
        print(f"    {count:4d}  {topic}")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: graph-ls
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_graph_ls(args: argparse.Namespace) -> None:
    """Generate stratified graph summary for LLM reasoning (Pro-Lite-Pro pipeline)."""
    import json as _json

    store = LayeredGraphStore(args.db_path)

    from loom.graph_ls import generate_graph_ls

    result = generate_graph_ls(store)
    store.close()

    if getattr(args, "json_output", False):
        print(_json.dumps(result, indent=2, default=str))
    else:
        _print_header("Graph ls — Stratified Summary")
        print(_json.dumps(result, indent=2, default=str))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: sandwich
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_sandwich(args: argparse.Namespace) -> None:
    """Run the Pro-Lite-Pro sandwich pipeline."""
    import json as _json

    _print_header("Pro-Lite-Pro Sandwich Pipeline")

    store = LayeredGraphStore(args.db_path)
    llm = _get_llm_client(args.dry_run, getattr(args, "llm", "gemini"))

    from loom.sandwich import run_sandwich

    result = run_sandwich(
        store, llm,
        model=getattr(args, "model", ""),
        thinking_level=getattr(args, "thinking", None),
    )

    # Print formatted output
    print(result.get("output", "No output"))

    # Save raw I/O
    save_dir = LOOM_DIR / "sandwich"
    save_dir.mkdir(exist_ok=True)
    ts = int(time.time())

    raw_path = save_dir / f"sandwich_{ts}.json"
    # Remove non-serializable items
    save_data = {k: v for k, v in result.items() if k != "output"}
    raw_path.write_text(_json.dumps(save_data, indent=2, default=str))
    print(f"\nRaw I/O saved to {raw_path}")

    if getattr(args, "save", None):
        from pathlib import Path
        Path(args.save).write_text(result.get("output", ""))
        print(f"Output saved to {args.save}")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: oracle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_oracle(args: argparse.Namespace) -> None:
    """Interactive retrieval mode — ask questions about your graph."""
    import json as _json

    store = LayeredGraphStore(args.db_path)
    llm = _get_llm_client(args.dry_run, getattr(args, "llm", "gemini"))

    from loom.oracle import ask_oracle, format_oracle_output

    question = " ".join(args.question)
    if not question:
        print("Usage: loom oracle 'Where did I promise to send the proposal?'")
        store.close()
        return

    _print_header("Oracle")
    result = ask_oracle(
        store, llm, question,
        model=getattr(args, "model", ""),
        thinking_level=getattr(args, "thinking", None),
    )

    print(format_oracle_output(result))

    # Save raw I/O
    if getattr(args, "save_raw", False):
        save_dir = LOOM_DIR / "oracle"
        save_dir.mkdir(exist_ok=True)
        ts = int(time.time())
        raw_path = save_dir / f"oracle_{ts}.json"
        raw_path.write_text(_json.dumps(result, indent=2, default=str))
        print(f"\nRaw I/O saved to {raw_path}")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: estimate-person-model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_estimate_person_model(args: argparse.Namespace) -> None:
    """Estimate the 11-dimension Person Model from all available data."""
    import json as _json

    _print_header("Person Model Estimation")

    store = LayeredGraphStore(args.db_path)
    llm = _get_llm_client(args.dry_run, getattr(args, "llm", "gemini"))

    from loom.person_model import estimate_person_model, get_model_gaps

    force = getattr(args, "force", False)
    model = estimate_person_model(store, llm=llm, force=force)

    # Print confidences
    print("  Per-dimension confidence:")
    for dim_name, dim_data in sorted(model.items()):
        if isinstance(dim_data, dict) and "confidence" in dim_data:
            conf = dim_data["confidence"]
            bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
            print(f"    {dim_name:<30s} {bar} {conf:.2f}")

    # Print gaps
    gaps = get_model_gaps(model)
    if gaps:
        print(f"\n  {len(gaps)} gap(s) detected (confidence < 0.3):")
        for g in gaps:
            print(f"    - {g['dimension']}: {g['confidence']:.2f}")
        print("\n  Run the person model chat to fill in gaps.")

    # Save JSON
    save_dir = Path.home() / ".loom" / "person_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = save_dir / f"model_{ts}.json"
    out_path.write_text(_json.dumps(model, indent=2, default=str))
    print(f"\n  Model saved to {out_path}")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: stats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_stats(args: argparse.Namespace) -> None:
    """Show database stats."""
    _print_header("Database Statistics")

    store = LayeredGraphStore(args.db_path)
    stats = store.stats()

    print(f"  Events:  {stats['events_count']}")
    print(f"  Claims:  {stats['claims_count']} (active: {stats.get('active_claims', 'n/a')})")
    print(f"  Beliefs: {stats['beliefs_count']}")
    print(f"  Persons: {stats['persons_count']}")

    events_by_source = stats.get("events_by_source", {})
    if events_by_source:
        print("\n  Events by source:")
        for source, count in sorted(events_by_source.items(), key=lambda x: -x[1]):
            print(f"    {source:>12s}: {count:>6d}")

    events_by_type = stats.get("events_by_type", {})
    if events_by_type:
        print("\n  Events by type:")
        for etype, count in sorted(events_by_type.items(), key=lambda x: -x[1]):
            print(f"    {etype:>16s}: {count:>6d}")

    # Triage tier distribution
    try:
        triage_rows = store.conn.execute(
            """SELECT
                 SUM(CASE WHEN confidence < 0.3 THEN 1 ELSE 0 END) as ignore_n,
                 SUM(CASE WHEN confidence >= 0.3 AND confidence < 0.7 THEN 1 ELSE 0 END) as lightweight_n,
                 SUM(CASE WHEN confidence >= 0.7 THEN 1 ELSE 0 END) as deep_n,
                 COUNT(*) as total
               FROM claims
               WHERE claim_type = 'triage' AND superseded_by IS NULL"""
        ).fetchone()

        if triage_rows and triage_rows["total"] > 0:
            print("\n  Triage tiers:")
            print(f"    {'ignore (<0.3)':>20s}: {triage_rows['ignore_n']:>6d}")
            print(f"    {'lightweight (0.3-0.7)':>20s}: {triage_rows['lightweight_n']:>6d}")
            print(f"    {'deep (>=0.7)':>20s}: {triage_rows['deep_n']:>6d}")
            print(f"    {'total':>20s}: {triage_rows['total']:>6d}")
    except Exception:
        pass

    # Belief breakdown
    try:
        belief_rows = store.conn.execute(
            """SELECT belief_type, status, COUNT(*) as cnt
               FROM beliefs
               GROUP BY belief_type, status
               ORDER BY belief_type, status"""
        ).fetchall()

        if belief_rows:
            print("\n  Beliefs by type/status:")
            for row in belief_rows:
                print(f"    {row['belief_type']:>12s} / {row['status']:<12s}: {row['cnt']:>4d}")
    except Exception:
        pass

    # Sync state
    sync = stats.get("sync_state", {})
    if sync:
        print("\n  Sync state:")
        for source, state in sorted(sync.items()):
            last_sync = state.get("last_sync", 0)
            event_count = state.get("event_count", 0)
            status = state.get("status", "unknown")
            if last_sync:
                from datetime import datetime, timezone
                dt = datetime.fromtimestamp(last_sync, tz=timezone.utc)
                print(f"    {source:>12s}: {event_count:>6d} events | "
                      f"last sync {dt.strftime('%Y-%m-%d %H:%M')} | {status}")
            else:
                print(f"    {source:>12s}: {event_count:>6d} events | never synced | {status}")

    store.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: watch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_watch(args: argparse.Namespace) -> None:
    """Run the continuous watch daemon."""
    from loom.watcher import WatchDaemon

    # Default to 168h (7 days) for initial catchup if no --hours specified
    if args.hours is None and args.since is None:
        args.hours = 168

    poll_intervals = {
        "imessage": WATCH_POLL_FILE_SOURCES_SECONDS,  # FSEvents unreliable for chat.db
        "calendar": getattr(args, "poll_calendar", WATCH_POLL_CALENDAR_SECONDS),
        "slack": getattr(args, "poll_slack", WATCH_POLL_SLACK_SECONDS),
        "granola": getattr(args, "poll_granola", WATCH_POLL_GRANOLA_SECONDS),
    }

    enabled_sources = args.sources if args.sources else None

    daemon = WatchDaemon(
        args=args,
        debounce_seconds=getattr(args, "debounce", WATCH_DEBOUNCE_SECONDS),
        poll_intervals=poll_intervals,
        enabled_sources=enabled_sources,
    )
    daemon.run()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _run_onboarding_checkpoint(args: argparse.Namespace, summary: dict) -> None:
    """Run person model estimation and optional inline onboarding between triage and extraction.

    Skips if:
    - Person model already exists and was estimated < 1 hour ago
    - Running in dry-run mode (uses mock LLM)
    """
    import time as _time

    db_path = getattr(args, "db_path", None) or str(LOOM_DIR / "graph.db")
    store = LayeredGraphStore(db_path)

    from loom.person_model import estimate_person_model, get_model_gaps, get_person_model

    existing = get_person_model(store)
    if existing:
        estimated_at = existing.get("updated_at") or existing.get("created_at", 0)
        if _time.time() - estimated_at < 3600:
            logger.info("Person model recently estimated (%ds ago), skipping onboarding",
                        int(_time.time() - estimated_at))
            return

    print("\n  ── Person Model Checkpoint ──")

    # Get LLM client
    llm = None
    if not getattr(args, "dry_run", False):
        try:
            from loom.llm.gemini import GeminiClient
            llm = GeminiClient()
        except Exception as exc:
            logger.warning("Gemini not available for onboarding: %s", exc)
    else:
        from loom.llm.mock import MockLLMClient
        llm = MockLLMClient()

    model = estimate_person_model(store, llm=llm, force=False)
    gaps = get_model_gaps(model)

    if not gaps:
        print("  Person model complete — no gaps to fill")
        return

    print(f"  Person model has {len(gaps)} low-confidence dimensions")

    # Generate questions
    from loom.mcp_tools.person_model_tools import generate_onboarding_questions
    questions = generate_onboarding_questions(model, gaps, llm, store, max_questions=5)

    if not questions:
        return

    if sys.stdin.isatty():
        _run_inline_onboarding(store, model, questions)
    else:
        print("  Run 'loom person-model-chat' to fill profile gaps")


def _run_inline_onboarding(
    store: LayeredGraphStore, model: dict, questions: list[dict],
) -> None:
    """Present quick onboarding questions inline during pipeline run."""
    from loom.person_model import update_person_model_field

    print(f"\n  Quick onboarding ({len(questions)} questions, Enter to skip):\n")

    answered = 0
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q['question']}")
        try:
            answer = input("     → ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Skipping remaining questions")
            break

        if not answer:
            continue

        dimension = q.get("dimension", "")
        if dimension:
            update_person_model_field(store, dimension, "user_input", answer)
            answered += 1

    if answered:
        print(f"\n  Saved {answered} answer(s) to person model")
    print()


def run_pipeline_stages(args: argparse.Namespace) -> dict:
    """Run pipeline stages as a reusable function.

    Called by both cmd_pipeline() and WatchDaemon.
    Runs stages 0 through 7 (ingest through synthesis).
    Stage 8 (briefing) is NOT included — it stays on-demand.

    Ingestion stages (0-1) always process full history so that person
    resolution sees all contacts and events. LLM-heavy stages (2+)
    respect --hours/--since to limit expensive processing.

    Returns a summary dict with per-stage status.
    """
    summary: dict = {"stages_run": 0, "stages_failed": 0, "errors": []}

    # ── Pre-flight: validate profile exists with emails ──
    profile_raw = load_profile()
    profile_flat = flatten_profile(profile_raw)
    profile_emails = profile_flat.get("emails", [])
    profile_name = profile_flat.get("name", "")
    tz = safe_timezone()
    print(f"\n  Profile check: name={profile_name!r}, emails={profile_emails}, tz={tz}")
    if not profile_emails:
        msg = (
            "FATAL: No user email found in profile.\n"
            "  Create ~/.loom/profile.yaml with at least one email address.\n"
            "  The pipeline cannot identify your messages without this.\n"
            "  Example:\n"
            "    name: Your Name\n"
            "    emails:\n"
            "      - you@example.com\n"
        )
        print(f"\n  {msg}")
        summary["stages_failed"] += 1
        summary["errors"].append("missing_profile_emails")
        return summary

    # Ingestion stages: always full history (identity needs everything)
    ingest_stages = [
        ("0_ingest", cmd_ingest),
        ("0b_resolve", cmd_resolve),
        ("0b2_dedup", cmd_dedup),
        ("0c_link", cmd_link),
        ("1_claims", cmd_claims),
        ("1.5_annotate", cmd_annotate),
    ]

    # LLM-heavy stages split around onboarding checkpoint
    pre_onboarding_stages = [
        ("2_score", cmd_score),
        ("4_triage", cmd_triage),
    ]

    post_onboarding_stages = [
        ("5_propagate", cmd_propagate),
        ("6_extract", cmd_extract),
        ("7_synthesize", cmd_synthesize),
    ]

    # Save user's date filter, run ingestion with full history
    saved_hours = args.hours
    saved_since = args.since

    args.hours = None
    args.since = None
    for stage_name, stage_fn in ingest_stages:
        try:
            stage_fn(args)
            summary["stages_run"] += 1
        except Exception as exc:
            summary["stages_failed"] += 1
            summary["errors"].append(f"{stage_name}: {exc}")
            print(f"  {stage_name} failed: {exc}")

    # Pre-onboarding LLM stages — default to 30 days if no --hours/--since
    args.hours = saved_hours
    args.since = saved_since
    if args.hours is None and args.since is None:
        args.hours = 720  # 30 days
        print(f"\n  LLM stages: defaulting to 30 days (720h)\n")
    for stage_name, stage_fn in pre_onboarding_stages:
        try:
            stage_fn(args)
            summary["stages_run"] += 1
        except Exception as exc:
            summary["stages_failed"] += 1
            summary["errors"].append(f"{stage_name}: {exc}")
            print(f"  {stage_name} failed: {exc}")

    # Onboarding checkpoint (between triage and propagation)
    if not getattr(args, "skip_onboarding", False):
        _run_onboarding_checkpoint(args, summary)

    # Post-onboarding LLM stages
    for stage_name, stage_fn in post_onboarding_stages:
        try:
            stage_fn(args)
            summary["stages_run"] += 1
        except Exception as exc:
            summary["stages_failed"] += 1
            summary["errors"].append(f"{stage_name}: {exc}")
            print(f"  {stage_name} failed: {exc}")

    return summary


BUILD_VERSION = "2026-02-24e"
"""Build version stamp — bump on each DMG build so we can tell which binary is running."""


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Run the full pipeline end-to-end."""
    t0 = time.time()
    print("\n" + "=" * 60)
    print(f"  Loom Full Pipeline  (build {BUILD_VERSION})")
    print("=" * 60)

    if args.dry_run:
        print("  [DRY RUN] Using mock LLM client for all LLM stages\n")

    summary = run_pipeline_stages(args)

    # Stage 8: Briefing (only if --brief is set)
    if args.brief:
        try:
            cmd_brief(args)
        except Exception as exc:
            print(f"  Briefing failed: {exc}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Stages run: {summary['stages_run']}, failed: {summary['stages_failed']}")
    print(f"{'=' * 60}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Command: eval
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EVAL_STAGE_MAP = {
    "0": "0_ingest",
    "1": "1_resolve",
    "2": "2_link",
    "3": "3_claims",
    "4": "4_triage",
    "5": "5_propagate",
    "6": "6_extract",
    "7": "7_synthesize",
}


def _resolve_eval_stage(stage_arg: str) -> str:
    """Map short stage numbers to full stage names."""
    return EVAL_STAGE_MAP.get(stage_arg, stage_arg)


def cmd_eval(args: argparse.Namespace) -> None:
    """Route eval subcommands."""
    sub = getattr(args, "eval_sub", None)
    if not sub:
        print("Usage: loom eval {check,sample,review,stats,run,freeze}")
        return

    dispatch = {
        "check": cmd_eval_check,
        "sample": cmd_eval_sample,
        "review": cmd_eval_review,
        "stats": cmd_eval_stats,
        "run": cmd_eval_run,
        "freeze": cmd_eval_freeze,
    }
    handler = dispatch.get(sub)
    if handler:
        handler(args)
    else:
        print(f"Unknown eval subcommand: {sub}")


def cmd_eval_check(args: argparse.Namespace) -> None:
    """Run automated completeness checks."""
    _print_header("Eval: Completeness Check")

    from loom.eval.checks import format_report, run_stage0_checks

    db_path = args.db_path or str(LOOM_DIR / "graph.db")
    report = run_stage0_checks(db_path)
    print(format_report(report))


def cmd_eval_sample(args: argparse.Namespace) -> None:
    """Sample items for review."""
    stage = _resolve_eval_stage(args.stage)
    n = getattr(args, "n", 10)
    stratify = getattr(args, "stratify", "default")

    _print_header(f"Eval: Sample (stage={stage}, n={n})")

    from loom.eval.sampler import sample

    db_path = args.db_path or str(LOOM_DIR / "graph.db")
    try:
        items = sample(db_path, stage, n, stratify)
    except ValueError as exc:
        print(f"  Error: {exc}")
        return

    import json as _json
    for i, item in enumerate(items, 1):
        print(f"\n--- Sample {i}/{len(items)} ---")
        # Compact display
        item_id = item.get("id", "N/A")
        source = item.get("source", item.get("belief_type", "N/A"))
        conf = item.get("confidence", "")
        subj = item.get("subject", "")
        print(f"  ID: {item_id}  Source: {source}")
        if conf:
            print(f"  Confidence: {conf}")
        if subj:
            print(f"  Subject: {subj}")
        content = item.get("raw_content", item.get("summary", ""))
        if content:
            preview = content[:150] + "..." if len(str(content)) > 150 else content
            print(f"  Content: {preview}")

    print(f"\n  Total samples: {len(items)}")


def cmd_eval_review(args: argparse.Namespace) -> None:
    """Interactive review session."""
    stage = _resolve_eval_stage(args.stage)
    n = getattr(args, "n", 10)
    stratify = getattr(args, "stratify", "default")
    reviewer_name = getattr(args, "reviewer", "")
    golden_dir = getattr(args, "golden_dir", None)
    frozen = getattr(args, "frozen", False)

    mode = "frozen" if frozen else "sample"
    _print_header(f"Eval: Review (stage={stage}, n={n}, mode={mode})")

    from loom.eval.reviewer import review_frozen, review_session

    db_path = args.db_path or str(LOOM_DIR / "graph.db")
    try:
        if frozen:
            review_frozen(stage, n, reviewer_name, golden_dir)
        else:
            review_session(db_path, stage, n, stratify, reviewer_name, golden_dir)
    except ValueError as exc:
        print(f"  Error: {exc}")


def cmd_eval_stats(args: argparse.Namespace) -> None:
    """Show evaluation statistics."""
    stage = _resolve_eval_stage(args.stage) if args.stage else None
    golden_dir = getattr(args, "golden_dir", None)

    _print_header("Eval: Statistics")

    from loom.eval.stats import compute_stats, format_stats

    stats = compute_stats(golden_dir, stage)
    print(format_stats(stats))


def cmd_eval_run(args: argparse.Namespace) -> None:
    """Re-run a stage against golden records."""
    stage = _resolve_eval_stage(args.stage)
    golden_dir = getattr(args, "golden_dir", None)

    _print_header(f"Eval: Run (stage={stage})")

    from loom.eval.runner import format_eval_report, run_stage_eval

    db_path = args.db_path or str(LOOM_DIR / "graph.db")
    results = run_stage_eval(db_path, stage, golden_dir)
    print(format_eval_report(results))


def cmd_eval_freeze(args: argparse.Namespace) -> None:
    """Freeze current sample as golden records (auto-approve from DB state)."""
    stage = _resolve_eval_stage(args.stage)
    n = getattr(args, "n", 50)
    golden_dir = getattr(args, "golden_dir", None)

    _print_header(f"Eval: Freeze (stage={stage}, n={n})")

    from loom.eval.golden import GoldenRecord, GoldenStore
    from loom.eval.sampler import sample

    db_path = args.db_path or str(LOOM_DIR / "graph.db")
    golden = GoldenStore(golden_dir)

    try:
        items = sample(db_path, stage, n)
    except ValueError as exc:
        print(f"  Error: {exc}")
        return

    count = 0
    for item in items:
        item_id = item.get("id", "")
        record = GoldenRecord(
            id="",
            stage=stage,
            item_id=item_id,
            input_data=item,
            expected_output=item,
            judgment="approve",
            notes="auto-frozen from DB state",
            reviewer="freeze",
        )
        golden.add(record)
        count += 1

    print(f"  Frozen {count} records to golden store")
    print(f"  Path: {golden.base_dir / f'{stage}.jsonl'}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Argument parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="loom",
        description="Loom: a computational autobiography pipeline",
    )
    parser.add_argument(
        "--db-path", default=None,
        help=f"Database path (default: {LOOM_DIR / 'graph.db'})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use mock LLM client for all LLM stages",
    )
    parser.add_argument(
        "--llm", choices=["ollama", "gemini", "mock"], default="gemini",
        help="LLM provider to use (default: gemini)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline stage to run")

    # ── ingest ──
    p_ingest = subparsers.add_parser("ingest", help="Ingest events from sources")
    p_ingest.add_argument("--hours", type=int, default=None, help="Lookback hours")
    p_ingest.add_argument("--since", default=None, help="Start timestamp (ISO format)")
    p_ingest.add_argument("--limit", type=int, default=None, help="Max items per source")
    p_ingest.add_argument(
        "--sources", nargs="*", default=None,
        help=f"Sources to ingest (default: {' '.join(DEFAULT_SOURCES)})",
    )
    p_ingest.add_argument(
        "--ambient", action="store_true",
        help="Also ingest ambient sources (knowledgec, safari, chrome, notes, shell_history)",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    # ── resolve ──
    p_resolve = subparsers.add_parser("resolve", help="Run person resolution")
    p_resolve.set_defaults(func=cmd_resolve)

    # ── dedup ──
    p_dedup = subparsers.add_parser("dedup", help="Stage 0b2: Person deduplication (deterministic + LLM)")
    p_dedup.add_argument("--skip-llm", action="store_true", help="Skip Layer 3 LLM verification (deterministic only)")
    p_dedup.set_defaults(func=cmd_dedup)

    # ── link ──
    p_link = subparsers.add_parser("link", help="Link events to persons")
    p_link.add_argument("--hours", type=int, default=None, help="Lookback hours (default: all)")
    p_link.add_argument("--since", default=None, help="Start timestamp (ISO format)")
    p_link.set_defaults(func=cmd_link)

    # ── relink ──
    p_relink = subparsers.add_parser("relink", help="Rebuild all event-person edges with thread-based linking")
    p_relink.add_argument("--skip-claims", action="store_true", help="Skip Stage 1 claims rebuild")
    p_relink.set_defaults(func=cmd_relink)

    # ── claims ──
    p_claims = subparsers.add_parser("claims", help="Extract Stage 1 deterministic claims")
    p_claims.add_argument("--hours", type=int, default=None, help="Lookback hours")
    p_claims.add_argument("--since", default=None, help="Start timestamp (ISO format)")
    p_claims.set_defaults(func=cmd_claims)

    # ── annotate ──
    p_annotate = subparsers.add_parser("annotate", help="Extract structural annotations (Stage 1.5)")
    p_annotate.add_argument("--hours", type=int, default=None, help="Lookback hours")
    p_annotate.add_argument("--since", default=None, help="Start timestamp (ISO format)")
    p_annotate.set_defaults(func=cmd_annotate)

    # ── score ──
    p_score = subparsers.add_parser("score", help="Run heuristic event scoring (Stage 2)")
    p_score.add_argument("--hours", type=int, default=None, help="Lookback hours")
    p_score.add_argument("--since", default=None, help="Start timestamp (ISO format)")
    p_score.add_argument("--lens", default="chief_of_staff", help="Scoring lens (default: chief_of_staff)")
    p_score.set_defaults(func=cmd_score)

    # ── triage ──
    p_triage = subparsers.add_parser("triage", help="Run LLM triage on events")
    p_triage.add_argument("--hours", type=int, default=None, help="Lookback hours")
    p_triage.add_argument("--since", default=None, help="Start timestamp (ISO format)")
    p_triage.add_argument("--lens", default="chief_of_staff", help="Scoring lens for route filtering (default: chief_of_staff)")
    p_triage.set_defaults(func=cmd_triage)

    # ── propagate ──
    p_propagate = subparsers.add_parser("propagate", help="Run score propagation")
    p_propagate.set_defaults(func=cmd_propagate)

    # ── extract ──
    p_extract = subparsers.add_parser("extract", help="Run deep commitment extraction")
    p_extract.set_defaults(func=cmd_extract)

    # ── synthesize ──
    p_synth = subparsers.add_parser("synthesize", help="Compile claims into beliefs")
    p_synth.set_defaults(func=cmd_synthesize)

    # ── commitments ──
    p_commit = subparsers.add_parser("commitments", help="Show commitments with provenance trace")
    p_commit.add_argument("--overdue", action="store_true", help="Show only overdue commitments")
    p_commit.add_argument("--all", action="store_true", help="Include superseded commitments")
    p_commit.add_argument("--person", default=None, help="Filter by person name")
    p_commit.add_argument("--n", type=int, default=20, help="Max items to show (default: 20)")
    p_commit.add_argument("--no-trace", dest="trace", action="store_false",
                          help="Hide provenance trace (compact view)")
    p_commit.set_defaults(func=cmd_commitments)

    # ── brief ──
    p_brief = subparsers.add_parser("brief", help="Generate blind spot briefing")
    p_brief.add_argument("--days", type=int, default=7, help="Days ahead (default: 7)")
    p_brief.add_argument("--lookback", type=int, default=30, help="Lookback days for context")
    p_brief.add_argument("--tz", default=USER_TIMEZONE, help="Timezone")
    p_brief.add_argument("--save", default=None, help="Save briefing to file")
    p_brief.add_argument("--thinking", default=None, choices=["minimal", "low", "medium", "high"],
                          help="Thinking level for Gemini 3 (default: model decides)")
    p_brief.add_argument("--no-interactive", action="store_true",
                          help="Skip user questions (non-interactive mode)")
    p_brief.set_defaults(func=cmd_brief)

    # ── topic-normalize ──
    p_topic_norm = subparsers.add_parser("topic-normalize", help="Run topic normalization pipeline")
    p_topic_norm.set_defaults(func=cmd_topic_normalize)

    # ── topic-stats ──
    p_topic_stats = subparsers.add_parser("topic-stats", help="Show topic frequency distribution")
    p_topic_stats.set_defaults(func=cmd_topic_stats)

    # ── graph-ls ──
    p_graph_ls = subparsers.add_parser("graph-ls", help="Generate stratified graph summary for LLM reasoning")
    p_graph_ls.add_argument("--json", dest="json_output", action="store_true",
                            help="Output raw JSON (default: pretty-print)")
    p_graph_ls.set_defaults(func=cmd_graph_ls)

    # ── sandwich ──
    p_sandwich = subparsers.add_parser("sandwich", help="Run Pro-Lite-Pro sandwich pipeline for blind spot discovery")
    p_sandwich.add_argument("--model", default="", help="Override model for Surveyor/Consigliere")
    p_sandwich.add_argument("--thinking", default=None, choices=["minimal", "low", "medium", "high"],
                            help="Thinking level for Gemini 3 (default: model decides)")
    p_sandwich.add_argument("--save", default=None, help="Save formatted output to file")
    p_sandwich.set_defaults(func=cmd_sandwich)

    # ── oracle ──
    p_oracle = subparsers.add_parser("oracle", help="Ask questions about your knowledge graph")
    p_oracle.add_argument("question", nargs="+", help="Your question (e.g., 'Where did I promise to send the proposal?')")
    p_oracle.add_argument("--model", default="", help="Override model")
    p_oracle.add_argument("--thinking", default=None, choices=["minimal", "low", "medium", "high"],
                          help="Thinking level for Gemini 3")
    p_oracle.add_argument("--save-raw", action="store_true", help="Save raw evidence to ~/.loom/oracle/")
    p_oracle.set_defaults(func=cmd_oracle)

    # ── stats ──
    p_stats = subparsers.add_parser("stats", help="Show database statistics")
    p_stats.set_defaults(func=cmd_stats)

    # ── pipeline ──
    p_pipeline = subparsers.add_parser("pipeline", help="Run full pipeline end-to-end")
    p_pipeline.add_argument("--hours", type=int, default=None, help="Lookback hours")
    p_pipeline.add_argument("--since", default=None, help="Start timestamp (ISO format)")
    p_pipeline.add_argument("--limit", type=int, default=None, help="Max items per source")
    p_pipeline.add_argument(
        "--sources", nargs="*", default=None,
        help="Sources to ingest",
    )
    p_pipeline.add_argument(
        "--ambient", action="store_true",
        help="Also ingest ambient sources (knowledgec, safari, chrome, notes, shell_history)",
    )
    p_pipeline.add_argument("--lens", default="chief_of_staff", help="Scoring lens (default: chief_of_staff)")
    p_pipeline.add_argument("--days", type=int, default=7, help="Briefing days ahead")
    p_pipeline.add_argument("--lookback", type=int, default=30, help="Briefing lookback days")
    p_pipeline.add_argument("--tz", default=USER_TIMEZONE, help="Timezone")
    p_pipeline.add_argument("--save", default=None, help="Save briefing to file")
    p_pipeline.add_argument(
        "--brief", action="store_true",
        help="Also generate briefing at the end",
    )
    p_pipeline.add_argument(
        "--skip-onboarding", action="store_true",
        help="Skip person model onboarding checkpoint",
    )
    p_pipeline.set_defaults(func=cmd_pipeline)

    # ── watch ──
    p_watch = subparsers.add_parser("watch", help="Run continuous watch daemon")
    p_watch.add_argument("--hours", type=int, default=None, help="Lookback hours for initial catchup")
    p_watch.add_argument("--since", default=None, help="Start timestamp (ISO format)")
    p_watch.add_argument("--limit", type=int, default=None, help="Max items per source")
    p_watch.add_argument(
        "--sources", nargs="*", default=None,
        help="Sources to watch (default: all available)",
    )
    p_watch.add_argument("--lens", default="chief_of_staff", help="Scoring lens (default: chief_of_staff)")
    p_watch.add_argument(
        "--debounce", type=float, default=WATCH_DEBOUNCE_SECONDS,
        help=f"Debounce delay in seconds (default: {WATCH_DEBOUNCE_SECONDS})",
    )
    p_watch.add_argument(
        "--poll-calendar", type=float, default=WATCH_POLL_CALENDAR_SECONDS,
        help=f"Calendar poll interval in seconds (default: {WATCH_POLL_CALENDAR_SECONDS})",
    )
    p_watch.add_argument(
        "--poll-slack", type=float, default=WATCH_POLL_SLACK_SECONDS,
        help=f"Slack poll interval in seconds (default: {WATCH_POLL_SLACK_SECONDS})",
    )
    p_watch.add_argument(
        "--poll-granola", type=float, default=WATCH_POLL_GRANOLA_SECONDS,
        help=f"Granola poll interval in seconds (default: {WATCH_POLL_GRANOLA_SECONDS})",
    )
    p_watch.set_defaults(func=cmd_watch)

    # ── estimate-person-model ──
    p_pm = subparsers.add_parser("estimate-person-model", help="Estimate the 11-dimension Person Model")
    p_pm.add_argument("--force", action="store_true", help="Force re-estimation even if recent model exists")
    p_pm.set_defaults(func=cmd_estimate_person_model)

    # ── eval ──
    p_eval = subparsers.add_parser("eval", help="Data quality evaluation")
    p_eval.set_defaults(func=cmd_eval)
    eval_sub = p_eval.add_subparsers(dest="eval_sub", help="Eval subcommand")

    # eval check
    p_eval_check = eval_sub.add_parser("check", help="Run completeness checks")
    p_eval_check.add_argument("--stage", default="0", help="Stage to check (default: 0)")
    p_eval_check.set_defaults(func=cmd_eval, eval_sub="check")

    # eval sample
    p_eval_sample = eval_sub.add_parser("sample", help="Sample items for review")
    p_eval_sample.add_argument("--stage", required=True, help="Stage to sample (0-7)")
    p_eval_sample.add_argument("--n", type=int, default=10, help="Number of samples")
    p_eval_sample.add_argument("--stratify", default="default", help="Stratification method")
    p_eval_sample.set_defaults(func=cmd_eval, eval_sub="sample")

    # eval review
    p_eval_review = eval_sub.add_parser("review", help="Interactive review session")
    p_eval_review.add_argument("--stage", required=True, help="Stage to review (0-7)")
    p_eval_review.add_argument("--n", type=int, default=10, help="Number of items")
    p_eval_review.add_argument("--stratify", default="default", help="Stratification method")
    p_eval_review.add_argument("--reviewer", default="", help="Reviewer name")
    p_eval_review.add_argument("--frozen", action="store_true", help="Review already-frozen golden records instead of sampling new ones")
    p_eval_review.add_argument("--golden-dir", default=None, help="Golden store directory")
    p_eval_review.set_defaults(func=cmd_eval, eval_sub="review")

    # eval stats
    p_eval_stats = eval_sub.add_parser("stats", help="Show evaluation statistics")
    p_eval_stats.add_argument("--stage", default=None, help="Stage to show stats for (all if omitted)")
    p_eval_stats.add_argument("--golden-dir", default=None, help="Golden store directory")
    p_eval_stats.set_defaults(func=cmd_eval, eval_sub="stats")

    # eval run
    p_eval_run = eval_sub.add_parser("run", help="Re-run stage against golden records")
    p_eval_run.add_argument("--stage", required=True, help="Stage to run (0-7)")
    p_eval_run.add_argument("--golden-dir", default=None, help="Golden store directory")
    p_eval_run.set_defaults(func=cmd_eval, eval_sub="run")

    # eval freeze
    p_eval_freeze = eval_sub.add_parser("freeze", help="Freeze current DB state as golden records")
    p_eval_freeze.add_argument("--stage", required=True, help="Stage to freeze (0-7)")
    p_eval_freeze.add_argument("--n", type=int, default=50, help="Number of items to freeze")
    p_eval_freeze.add_argument("--golden-dir", default=None, help="Golden store directory")
    p_eval_freeze.set_defaults(func=cmd_eval, eval_sub="freeze")

    # ── agent ──
    p_agent = subparsers.add_parser("agent", help="Agent commands (VC outreach)")
    agent_sub = p_agent.add_subparsers(dest="agent_command")

    p_agent_pipeline = agent_sub.add_parser("pipeline", help="Show investor pipeline")
    p_agent_pipeline.add_argument("--json", dest="json_output", action="store_true",
                                   help="Output as JSON")

    p_agent_research = agent_sub.add_parser("research", help="Research a VC")
    p_agent_research.add_argument("--name", required=True, help="VC name")
    p_agent_research.add_argument("--firm", required=True, help="Firm name")
    p_agent_research.add_argument("--introduced-by", dest="introduced_by", default="")
    p_agent_research.add_argument("--introducer-agent", dest="introducer_agent", default="",
                                   choices=["cto", "ceo"])
    p_agent_research.add_argument("--json", dest="json_output", action="store_true",
                                   help="Output as JSON")

    p_agent_drafts = agent_sub.add_parser("drafts", help="Review outreach drafts")
    p_agent_drafts.add_argument("--role", choices=["cto", "ceo"], help="Filter by agent")
    p_agent_drafts.add_argument("--json", dest="json_output", action="store_true",
                                 help="Output as JSON")

    p_agent_approve = agent_sub.add_parser("approve", help="Approve a draft")
    p_agent_approve.add_argument("draft_id", help="Draft ID")
    p_agent_approve.add_argument("--json", dest="json_output", action="store_true")

    p_agent_reject = agent_sub.add_parser("reject", help="Reject a draft")
    p_agent_reject.add_argument("draft_id", help="Draft ID")
    p_agent_reject.add_argument("--json", dest="json_output", action="store_true")

    p_agent_discover = agent_sub.add_parser("discover", help="Scan KG to discover VCs")
    p_agent_discover.add_argument("--days", type=int, default=180,
                                   help="Days back to scan")
    p_agent_discover.add_argument("--add-all", action="store_true",
                                   help="Add all discovered VCs to pipeline")
    p_agent_discover.add_argument("--json", dest="json_output", action="store_true",
                                   help="Output as JSON")

    p_agent_run = agent_sub.add_parser("run", help="Run an agent with a task")
    p_agent_run.add_argument("--role", choices=["cto", "ceo"], default="cto")
    p_agent_run.add_argument("task", nargs="+", help="Task description")

    p_agent.set_defaults(func=cmd_agent_dispatch)

    # ── mcp-server ──
    p_mcp = subparsers.add_parser("mcp-server", help="Start the MCP server")
    p_mcp.add_argument("--port", type=int, default=None, help="Port (default: 9119)")
    p_mcp.add_argument("--read-only", action="store_true", help="Only expose read tools")
    p_mcp.add_argument("--stdio", action="store_true", help="Use stdio transport (for Claude Desktop)")
    p_mcp.set_defaults(func=cmd_mcp_server)

    return parser


def cmd_agent_dispatch(args: argparse.Namespace) -> None:
    """Dispatch agent subcommands to loom.agents.cli."""
    sub = getattr(args, "agent_command", None)
    if not sub:
        print("Usage: loom agent {pipeline,research,discover,drafts,approve,reject,run}")
        return

    from loom.agents import cli as agent_cli

    # Map workspace path from main CLI's db_path
    workspace_path = str(LOOM_DIR / "workspace.db")
    args.workspace = workspace_path

    if sub == "pipeline":
        agent_cli.cmd_pipeline(args)
    elif sub == "research":
        args.db_path = args.db_path or str(LOOM_DIR / "graph.db")
        agent_cli.cmd_research(args)
    elif sub == "drafts":
        agent_cli.cmd_drafts(args)
    elif sub == "approve":
        agent_cli.cmd_approve(args)
    elif sub == "reject":
        agent_cli.cmd_reject(args)
    elif sub == "discover":
        args.db_path = args.db_path or str(LOOM_DIR / "graph.db")
        agent_cli.cmd_discover(args)
    elif sub == "run":
        args.db_path = args.db_path or str(LOOM_DIR / "graph.db")
        agent_cli.cmd_agent(args)


def cmd_mcp_server(args: argparse.Namespace) -> None:
    """Start the MCP server."""
    import asyncio
    from loom.mcp_server import run_sse_server, run_stdio_server
    from loom.constants import MCP_DEFAULT_PORT

    db_path = args.db_path or str(LOOM_DIR / "graph.db")
    store = LayeredGraphStore(db_path)
    port = args.port or MCP_DEFAULT_PORT

    if args.stdio:
        asyncio.run(run_stdio_server(store, read_only=args.read_only))
    else:
        asyncio.run(run_sse_server(port, store, read_only=args.read_only))


def _check_first_run_config() -> None:
    """Check for config.json on first run.

    Uses load_config() which searches ~/.loom/, ~/Downloads/, ~/Desktop/.
    If found outside ~/.loom/, it's auto-copied there.
    If not found anywhere, prints friendly instructions.
    """
    from loom.constants import load_config

    cfg = load_config()
    if cfg.get("gemini_api_key"):
        return  # Good to go

    # Also check Keychain (where the bundled app stores it)
    import subprocess
    for service in ("alteris-listener", "loom"):
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-a", "gemini", "-s", service, "-w"],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return  # Found in Keychain
        except FileNotFoundError:
            pass

    # No config found — print friendly instructions
    print("\n" + "=" * 60)
    print("  Loom — Config File Needed")
    print("=" * 60)
    print()
    print("  Loom needs a config.json file with your API keys.")
    print()
    print("  You should have received a config.json file.")
    print("  Drop it in your Downloads folder and re-run this command.")
    print()
    print("  Loom checks these locations (in order):")
    print("    1. ~/.loom/config.json")
    print("    2. ~/Downloads/config.json")
    print("    3. ~/Desktop/config.json")
    print()
    print("  The file should look like:")
    print('    {')
    print('      "gemini_api_key": "your-key-here",')
    print('      "anthropic_api_key": "your-key-here"')
    print('    }')
    print()
    sys.exit(1)


def main() -> None:
    """CLI entry point."""
    # Force line-buffered stdout so piped readers (the macOS app) see output
    # immediately. Without this, Python's default full buffering on pipes
    # delays output until the buffer fills (~8KB), breaking interactive Q&A.
    if not sys.stdout.isatty():
        sys.stdout.reconfigure(line_buffering=True)

    parser = build_parser()
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # First-run: prompt for API keys if config is missing
    _check_first_run_config()

    # Run the command
    args.func(args)


if __name__ == "__main__":
    main()
