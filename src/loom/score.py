"""Stage 2: Heuristic event scoring.

Scores every event using metadata signals and routes them to:
  skip         (< 0.1)  -- noise, do not triage
  low_priority (0.1-0.4) -- cheap LLM path (Qwen batch)
  full_triage  (>= 0.4) -- expensive LLM path (Gemini)

Three-layer architecture (portable across users):
  Layer 1: Universal noise filtering — per-source noise signal extraction
           with corroboration model (2+ signals = skip, 1 = low_priority).
           Noise signals come from platform classifiers, not user preference.
  Layer 2: Person engagement floors — the primary importance signal. Computed
           from the user's own communication graph (thread count, source count,
           from_me ratio). Replaces per-source base scores as the importance
           signal. A universal base of 0.3 (low_priority) is the fallback for
           unknown persons.
  Layer 3: Structural overrides — identity/reaction skip, time-sensitive floor,
           from_me floor. Truly universal, not preference-based.

Signal abstraction: email scoring uses helper functions (_get_email_category,
_is_time_sensitive, _is_automated, _get_junk_level) instead of raw Apple metadata
field names. When a Gmail adapter lands, add cases to those helpers; the scorer
doesn't change.

Pure Python, no LLM calls. Produces one projection per (event, lens) pair.
Projections are disposable: DELETE + re-score is safe. Different lenses
produce different projections for the same events.

Pipeline position: After Stage 0 (ingest), Stage 1 (structural claims), and
person resolution. Before Stage 3/4 (embedding + LLM triage).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

from loom.constants import (
    HEURISTIC_FULL_TRIAGE_THRESHOLD,
    HEURISTIC_HIGH_IMPACT_FLOOR,
    HEURISTIC_SKIP_THRESHOLD,
    HEURISTIC_UNIVERSAL_BASE,
    LIMIT_ALL,
)
from loom.models import Annotation, Event, Projection
from loom.prefilter import is_machine_generated
from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)

DEFAULT_LENS = "chief_of_staff"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class PersonEngagement:
    """Lightweight engagement profile for a person in the scoring window."""
    thread_count: int      # events involving this person (excluding identity)
    source_count: int      # distinct sources (mail, imessage, whatsapp, etc.)
    from_me_ratio: float   # fraction of events authored by the user
    last_seen_ts: int      # most recent event timestamp


@dataclass(frozen=True)
class ScoreResult:
    """Outcome of scoring a single event.

    components tracks the cumulative running score at each step, making
    the scoring chain self-documenting. Example:
        {"base": 0.3, "after_automated": 0.0, "after_clamp": 0.0,
         "after_high_impact_floor": 0.4, "final": 0.4}

    signals captures the raw metadata values the scorer read, so downstream
    stages can reason about why without re-querying the event.
    """
    score: float
    route: str       # "skip" | "low_priority" | "full_triage"
    components: dict
    signals: dict


def _route(score: float) -> str:
    """Map score to routing tier."""
    score = round(score, 3)
    if score < HEURISTIC_SKIP_THRESHOLD:
        return "skip"
    if score < HEURISTIC_FULL_TRIAGE_THRESHOLD:
        return "low_priority"
    return "full_triage"


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Signal abstraction — provider-independent email signal extraction.
# Isolates Apple Mail metadata field names from scoring logic. When a
# Gmail adapter lands, add cases here; the scorer doesn't change.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_email_category(meta: dict) -> int | None:
    """Normalize email category. 0=primary, 1=transactions, 2=updates, 3=promotions.

    Apple Mail: model_category from on-device classifier.
    TODO: Gmail → map CATEGORY_PRIMARY/SOCIAL/PROMOTIONS/UPDATES/FORUMS.
    """
    if "model_category" in meta:
        return meta["model_category"]
    return None


def _is_time_sensitive(meta: dict) -> bool:
    """Time-sensitive email flag.

    Apple Mail: high_impact from Apple Intelligence.
    TODO: Gmail → gmail_important (weak proxy, trained on open/reply not urgency).
    """
    return bool(meta.get("high_impact"))


def _is_automated(meta: dict) -> bool:
    """Machine-generated email detection.

    Apple Mail: automated_conversation flag.
    TODO: Gmail → Precedence: bulk header + sender pattern heuristics.
    """
    return bool(meta.get("automated"))


def _get_junk_level(meta: dict) -> int:
    """Spam confidence. 0=not spam, higher=more confident spam.

    Apple Mail: graduated junk_level from server-side scoring.
    TODO: Gmail → 1 if SPAM label, else 0.
    """
    return meta.get("junk_level", 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Person engagement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_person_engagement(
    store: LayeredGraphStore,
    person_id: str,
    since_ts: int = 0,
    user_person_id: str | None = None,
) -> PersonEngagement:
    """Compute engagement profile for a person from the store.

    Fast path: reads from person_profiles table when available (pre-computed
    by Stage 1). Falls back to event-based computation when the table is
    empty or the person has no profile.

    Slow path queries events linked to this person (excluding identity events),
    counts distinct sources, from_me ratio, and recency.

    from_me detection uses two signals:
      1. event.metadata["is_from_me"] (set by iMessage/WhatsApp adapters)
      2. user_person_id linked as "sender" to the same event (works for mail
         where the adapter doesn't set is_from_me)
    """
    # Fast path: use pre-computed person_profiles table
    profile = store.get_person_profile(person_id)
    if profile and profile.get("message_count", 0) > 0:
        channels = profile.get("channels", [])
        source_count = len(channels) if isinstance(channels, list) else profile.get("channel_count", 0)
        return PersonEngagement(
            thread_count=profile["message_count"],
            source_count=source_count,
            from_me_ratio=round(profile.get("user_initiated_ratio") or 0.0, 3),
            last_seen_ts=profile.get("last_contact_ts") or 0,
        )

    # Slow path: compute from events
    events = store.get_events_for_person(person_id, since=since_ts, limit=LIMIT_ALL)
    events = [ev for ev in events if ev.event_type != "identity"]

    if not events:
        return PersonEngagement(0, 0, 0.0, 0)

    sources = {ev.source for ev in events}
    last_seen = max(ev.timestamp for ev in events)

    from_me_count = 0
    for ev in events:
        if ev.metadata.get("is_from_me"):
            from_me_count += 1
        elif user_person_id:
            co_persons = store.get_persons_for_event(ev.id)
            target_roles = [role for pid, role in co_persons if pid == person_id]
            if any(r in ("recipient", "cc") for r in target_roles):
                from_me_count += 1

    return PersonEngagement(
        thread_count=len(events),
        source_count=len(sources),
        from_me_ratio=round(from_me_count / len(events), 3),
        last_seen_ts=last_seen,
    )


def _engagement_floor(eng: PersonEngagement) -> float:
    """Compute the minimum score floor from person engagement.

    Floors prevent filtering events involving people the user actively
    communicates with across multiple channels.
    """
    if eng.source_count >= 3:
        return 0.45
    if eng.source_count >= 2:
        return 0.40
    if eng.thread_count >= 5 and eng.from_me_ratio > 0:
        return 0.35
    return 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-source scorers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _score_mail(event: Event) -> ScoreResult:
    meta = event.metadata
    c: dict = {}

    automated = _is_automated(meta)
    category = _get_email_category(meta)
    junk = _get_junk_level(meta)
    time_sensitive = _is_time_sensitive(meta)
    noreply = bool(meta.get("sender_is_noreply"))

    signals = {
        "automated": automated,
        "email_category": category,
        "time_sensitive": time_sensitive,
        "replied": bool(meta.get("replied")),
        "flagged": bool(meta.get("flagged")),
        "list_id": bool(meta.get("list_id")),
        "junk_level": junk,
        "urgent": bool(meta.get("urgent")),
        "noreply": noreply,
    }

    score = HEURISTIC_UNIVERSAL_BASE
    c["base"] = score

    # Boosts (positive content signals — universal across providers)
    if meta.get("replied"):
        score += 0.4
        c["after_replied"] = round(score, 3)
    if meta.get("flagged"):
        score += 0.3
        c["after_flagged"] = round(score, 3)
    if category == 0:
        score += 0.15
        c["after_primary_category"] = round(score, 3)
    if meta.get("urgent"):
        score += 0.15
        c["after_urgent"] = round(score, 3)

    # Penalties (noise signals — platform classifiers, not user preference)
    if automated:
        # Primary + automated = auto-generated on behalf of a real person
        # (calendar invites, shared docs, delegation notifications).
        if category == 0:
            score -= 0.15
        else:
            score -= 0.3
        c["after_automated"] = round(score, 3)
    if meta.get("list_id"):
        score -= 0.2
        c["after_list_id"] = round(score, 3)
    if category in (2, 3):
        score -= 0.2
        c["after_model_category_penalty"] = round(score, 3)
    if junk > 0:
        score -= 0.3
        c["after_junk"] = round(score, 3)
    if noreply:
        score -= 0.1
        c["after_noreply"] = round(score, 3)

    score = round(_clamp(score), 3)
    c["after_clamp"] = score

    # Layer 1: Corroboration — single noise signal should not produce skip.
    # Require 2+ independent noise indicators for confident filtering.
    noise_count = sum([
        automated,
        bool(meta.get("list_id")),
        category in (2, 3),
        junk > 0,
        noreply,
    ])
    c["noise_signals"] = noise_count
    if noise_count == 1 and score < HEURISTIC_SKIP_THRESHOLD:
        score = HEURISTIC_SKIP_THRESHOLD
        c["after_corroboration_floor"] = round(score, 3)

    # Layer 3: time-sensitive floor override.
    # Conditional: don't apply when automated=true — Apple Intelligence marks
    # security alerts as high_impact because they contain urgent language, but
    # these are ephemeral (2FA prompts expire) and shouldn't override noise.
    if time_sensitive and not automated:
        score = max(score, HEURISTIC_HIGH_IMPACT_FLOOR)
        c["after_high_impact_floor"] = round(score, 3)

    c["final"] = round(score, 3)
    return ScoreResult(round(score, 3), _route(score), c, signals)


def _score_imessage(event: Event) -> ScoreResult:
    meta = event.metadata
    c: dict = {}

    is_filtered = bool(meta.get("is_filtered"))
    delivered_quietly = bool(meta.get("delivered_quietly"))
    is_auto_reply = bool(meta.get("is_auto_reply"))

    signals = {
        "is_filtered": is_filtered,
        "delivered_quietly": delivered_quietly,
        "is_auto_reply": is_auto_reply,
    }

    score = HEURISTIC_UNIVERSAL_BASE
    c["base"] = score

    if is_filtered:
        score -= 0.1
        c["after_filtered"] = round(score, 3)
    if delivered_quietly:
        score -= 0.1
        c["after_quiet"] = round(score, 3)
    if is_auto_reply:
        score -= 0.15
        c["after_auto_reply"] = round(score, 3)

    score = round(_clamp(score), 3)
    c["after_clamp"] = score

    # Layer 1: noise counting — filtered, quiet, auto_reply are Apple platform signals
    noise_count = sum([is_filtered, delivered_quietly, is_auto_reply])
    c["noise_signals"] = noise_count
    if noise_count == 1 and score < HEURISTIC_SKIP_THRESHOLD:
        score = HEURISTIC_SKIP_THRESHOLD
        c["after_corroboration_floor"] = round(score, 3)

    c["final"] = round(score, 3)
    return ScoreResult(round(score, 3), _route(score), c, signals)


def _score_whatsapp(event: Event) -> ScoreResult:
    meta = event.metadata
    c: dict = {}
    signals = {
        "is_group": bool(meta.get("is_group")),
        "participant_count": len(event.participants),
        "shared_url": bool(meta.get("shared_url")),
        "shared_document": bool(meta.get("shared_document")),
    }

    score = HEURISTIC_UNIVERSAL_BASE
    c["base"] = score

    # Large group noise penalty
    if meta.get("is_group") and len(event.participants) > 10:
        score -= 0.15
        c["after_large_group"] = round(score, 3)

    # Content type boosts
    if meta.get("shared_url"):
        score += 0.1
        c["after_shared_url"] = round(score, 3)
    if meta.get("shared_document"):
        score += 0.1
        c["after_shared_document"] = round(score, 3)

    score = _clamp(score)
    c["after_clamp"] = round(score, 3)
    c["final"] = round(score, 3)
    return ScoreResult(round(score, 3), _route(score), c, signals)


def _score_calendar(event: Event) -> ScoreResult:
    meta = event.metadata
    c: dict = {}

    is_holiday_birthday = meta.get("calendar_type") in ("holiday", "birthday")

    signals = {
        "calendar_type": meta.get("calendar_type"),
        "attendee_count": len(meta.get("attendees", [])),
    }

    score = HEURISTIC_UNIVERSAL_BASE
    c["base"] = score

    # Boosts
    if len(meta.get("attendees", [])) > 0:
        score += 0.1
        c["after_has_attendees"] = round(score, 3)

    # Penalties (noise signals)
    if is_holiday_birthday:
        score -= 0.3
        c["after_holiday_birthday"] = round(score, 3)

    score = round(_clamp(score), 3)
    c["after_clamp"] = score

    # Layer 1: corroboration for calendar noise signals
    noise_count = sum([is_holiday_birthday])
    c["noise_signals"] = noise_count
    if noise_count == 1 and score < HEURISTIC_SKIP_THRESHOLD:
        score = HEURISTIC_SKIP_THRESHOLD
        c["after_corroboration_floor"] = round(score, 3)

    c["final"] = round(score, 3)
    return ScoreResult(round(score, 3), _route(score), c, signals)


def _score_granola(event: Event) -> ScoreResult:
    meta = event.metadata
    c: dict = {}
    signals = {
        "has_transcript": bool(meta.get("has_transcript")),
    }

    score = HEURISTIC_UNIVERSAL_BASE
    c["base"] = score

    if meta.get("has_transcript"):
        score += 0.1
        c["after_transcript"] = round(score, 3)

    c["final"] = round(score, 3)
    return ScoreResult(round(score, 3), _route(score), c, signals)


def _score_slack(event: Event) -> ScoreResult:
    meta = event.metadata
    c: dict = {}
    signals = {
        "has_thread": bool(meta.get("has_thread")),
        "reply_count": meta.get("reply_count") or 0,
    }

    score = HEURISTIC_UNIVERSAL_BASE
    c["base"] = score

    if meta.get("has_thread"):
        score += 0.1
        c["after_thread"] = round(score, 3)
    if (meta.get("reply_count") or 0) > 2:
        score += 0.1
        c["after_reply_count"] = round(score, 3)

    c["final"] = round(score, 3)
    return ScoreResult(round(score, 3), _route(score), c, signals)


def _score_default(event: Event) -> ScoreResult:
    c = {"base": HEURISTIC_UNIVERSAL_BASE, "final": HEURISTIC_UNIVERSAL_BASE}
    return ScoreResult(HEURISTIC_UNIVERSAL_BASE, _route(HEURISTIC_UNIVERSAL_BASE), c, {})


_SCORERS = {
    "mail": _score_mail,
    "imessage": _score_imessage,
    "whatsapp": _score_whatsapp,
    "calendar": _score_calendar,
    "granola": _score_granola,
    "slack": _score_slack,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dispatcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def score_event(
    event: Event,
    person_context: PersonEngagement | None = None,
) -> ScoreResult:
    """Score a single event.

    Three-layer scoring:
      Layer 1: Per-source noise filtering with corroboration (inside source scorers)
      Layer 2: Person engagement floor — the primary importance signal
      Layer 3: Structural overrides (identity/reaction skip, time-sensitive floor,
               from_me floor)
    """
    # Layer 3: Universal skip — identity events don't flow through triage
    if event.event_type == "identity":
        return ScoreResult(0.0, "skip", {"reason": "identity_event", "final": 0.0}, {})

    # Layer 3: Universal skip — reaction events (iMessage tapbacks, WhatsApp emoji)
    if event.event_type == "reaction":
        return ScoreResult(0.0, "skip", {"reason": "reaction_event", "final": 0.0}, {})

    # Layer 0: Machine-generated pre-filter — deterministic, zero tokens.
    # Kills automated/notification emails before any scoring or LLM.
    is_machine, machine_reason = is_machine_generated(event)
    if is_machine:
        return ScoreResult(
            0.0, "skip",
            {"reason": "machine_generated", "machine_reason": machine_reason, "final": 0.0},
            {"machine_generated": True, "machine_reason": machine_reason},
        )

    # Layer 1: source-specific noise filtering + content boosts
    scorer = _SCORERS.get(event.source, _score_default)
    result = scorer(event)
    score = result.score
    c = dict(result.components)
    changed = False

    # Layer 2: Person engagement floor — the primary importance signal.
    # Events involving active cross-source contacts are worth triaging
    # even if metadata signals are weak. This is what makes the scorer
    # portable across users: it adapts to each user's communication graph.
    if person_context:
        floor = _engagement_floor(person_context)
        if floor > 0 and score < floor:
            score = floor
            c["after_engagement_floor"] = round(score, 3)
            changed = True
        # Cross-source corroboration: person active in 2+ source types
        # is independently more important. This differentiates Sam's
        # Granola meetings (transcript + cross-source) from solo meetings.
        if person_context.source_count >= 2:
            score += 0.05
            c["after_cross_source_bonus"] = round(score, 3)
            changed = True

    # Layer 3: from_me floor — the user chose to send this, so it's
    # inherently relevant. Never score user-authored content below base.
    if event.metadata.get("is_from_me") and score < HEURISTIC_UNIVERSAL_BASE:
        score = HEURISTIC_UNIVERSAL_BASE
        c["after_from_me_floor"] = HEURISTIC_UNIVERSAL_BASE
        changed = True

    if changed:
        # Ensure final is always the last key
        if "final" in c:
            del c["final"]
        c["final"] = round(score, 3)
        return ScoreResult(round(score, 3), _route(score), c, result.signals)

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_scoring(
    store: LayeredGraphStore,
    since_ts: int = 0,
    lens: str = DEFAULT_LENS,
) -> dict:
    """Score all events and write projections for a lens.

    Pre-computes person engagement from existing event-person edges,
    then scores each event with engagement context. Projections are
    upserted: re-scoring the same lens overwrites previous projections.

    Returns:
        {"scored": N, "projections_written": N,
         "skip": N, "low_priority": N, "full_triage": N,
         "lens": str, "elapsed_seconds": float, "by_source": {source: count}}
    """
    t0 = time.time()
    events = store.get_events(since=since_ts, limit=LIMIT_ALL)

    # Find the user person_id so we skip it — the user's own engagement
    # is trivially high (present in every event) and uninformative.
    user_row = store.conn.execute(
        "SELECT person_id FROM persons WHERE is_user = 1 LIMIT 1"
    ).fetchone()
    user_person_id = user_row["person_id"] if user_row else None

    # Layer 2: pre-compute person engagement (cached per person_id).
    # For each event, find the linked person with the highest engagement
    # floor. This becomes the event's person_context for scoring.
    engagement_cache: dict[str, PersonEngagement] = {}
    event_person_context: dict[str, PersonEngagement | None] = {}

    for event in events:
        persons = store.get_persons_for_event(event.id)
        best_eng: PersonEngagement | None = None
        best_floor = 0.0
        for person_id, role in persons:
            if role == "identity":
                continue
            if person_id == user_person_id:
                continue
            if person_id not in engagement_cache:
                engagement_cache[person_id] = compute_person_engagement(
                    store, person_id, since_ts,
                    user_person_id=user_person_id,
                )
            eng = engagement_cache[person_id]
            floor = _engagement_floor(eng)
            if floor > best_floor:
                best_floor = floor
                best_eng = eng
        event_person_context[event.id] = best_eng

    stats: dict = {
        "scored": 0, "projections_written": 0,
        "skip": 0, "low_priority": 0, "full_triage": 0,
        "lens": lens, "by_source": {},
    }

    projections: list[Projection] = []
    score_annotations: list[Annotation] = []
    now = int(time.time())

    for event in events:
        result = score_event(event, person_context=event_person_context.get(event.id))
        stats["scored"] += 1
        stats[result.route] += 1
        stats["by_source"][event.source] = stats["by_source"].get(event.source, 0) + 1

        projections.append(Projection(
            event_id=event.id,
            lens=lens,
            score=result.score,
            route=result.route,
            components={
                "components": result.components,
                "signals": result.signals,
            },
            computed_at=now,
        ))

        # Write noise/corroboration annotations from scorer signals
        noise_count = result.components.get("noise_count")
        if noise_count is not None:
            score_annotations.append(Annotation(
                event_id=event.id, facet="noise_count",
                value=str(noise_count), source="scorer_v1", created_at=now,
            ))
        if result.components.get("after_corroboration_floor") is not None:
            score_annotations.append(Annotation(
                event_id=event.id, facet="corroboration_rescued",
                value="true", source="scorer_v1", created_at=now,
            ))

    stats["projections_written"] = store.put_projections_batch(projections)
    if score_annotations:
        stats["score_annotations_written"] = store.put_annotations_batch(score_annotations)
    stats["elapsed_seconds"] = round(time.time() - t0, 2)
    logger.info(
        "Stage 2: %d events scored (%d projections written) in %.1fs [lens=%s]. "
        "Routes: skip=%d, low_priority=%d, full_triage=%d",
        stats["scored"], stats["projections_written"],
        stats["elapsed_seconds"], lens,
        stats["skip"], stats["low_priority"], stats["full_triage"],
    )
    return stats
