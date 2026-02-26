"""Interactive CLI review tool for eval golden store.

Presents sampled items one at a time and collects human judgments.
Writes results to the golden store as JSONL.

Usage:
    from loom.eval.reviewer import review_session
    review_session(db_path, stage="4_triage", n=10)
"""

from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime, timezone
from typing import Any

from loom.eval.golden import GoldenRecord, GoldenStore
from loom.eval.sampler import sample


JUDGMENTS = {
    "a": "approve",
    "c": "correct",
    "r": "reject",
    "f": "flag",
    "s": "skip",
    "q": "quit",
}


def _truncate(text: str | None, max_len: int = 200) -> str:
    if not text:
        return "(empty)"
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _format_timestamp(ts: int | None) -> str:
    if not ts:
        return "N/A"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _display_event(item: dict[str, Any]) -> None:
    """Display an event item for review."""
    print(f"  ID:      {item.get('id', 'N/A')}")
    print(f"  Source:  {item.get('source', 'N/A')}")
    print(f"  Type:    {item.get('event_type', 'N/A')}")
    print(f"  Time:    {_format_timestamp(item.get('timestamp'))}")

    participants = item.get("participants", "[]")
    if isinstance(participants, str):
        try:
            participants = json.loads(participants)
        except (json.JSONDecodeError, TypeError):
            pass
    if participants and participants != "[]":
        print(f"  Partic:  {participants}")

    content = item.get("raw_content")
    print(f"  Content: {_truncate(content, 300)}")

    metadata = item.get("metadata", "{}")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    if metadata and metadata != {}:
        meta_keys = list(metadata.keys())[:5]
        print(f"  Meta:    {', '.join(meta_keys)}")


def _display_triage(item: dict[str, Any]) -> None:
    """Display a triage claim for review."""
    print(f"  Claim ID:    {item.get('id', 'N/A')}")
    print(f"  Subject:     {item.get('subject', 'N/A')}")
    print(f"  Predicate:   {item.get('predicate', 'N/A')}")
    print(f"  Confidence:  {item.get('confidence', 'N/A')}")
    tier = item.get("_tier", "N/A")
    print(f"  Tier:        {tier}")

    obj = item.get("object", "")
    if isinstance(obj, str):
        try:
            obj_data = json.loads(obj)
            if isinstance(obj_data, dict):
                for k in ("domain", "topics", "entities", "commitment_type", "reason"):
                    if k in obj_data:
                        print(f"  {k.title():12s}: {obj_data[k]}")
        except (json.JSONDecodeError, TypeError):
            print(f"  Object:      {_truncate(obj, 200)}")

    event = item.get("_event")
    if event:
        print(f"  --- Source Event ---")
        print(f"  Source:  {event.get('source', 'N/A')}")
        print(f"  Time:    {_format_timestamp(event.get('timestamp'))}")
        content = event.get("raw_content")
        print(f"  Content: {_truncate(content, 300)}")


def _display_person(item: dict[str, Any]) -> None:
    """Display a person record for review."""
    print(f"  Person:  {item.get('person_id', 'N/A')}")
    print(f"  Name:    {item.get('canonical_name', '(none)')}")
    print(f"  Is user: {bool(item.get('is_user', 0))}")
    sources = item.get("sources", "[]")
    if isinstance(sources, str):
        try:
            sources = json.loads(sources)
        except (json.JSONDecodeError, TypeError):
            pass
    print(f"  Sources: {sources}")
    tier = item.get("_tier", "N/A")
    print(f"  Tier:    {tier}")

    identifiers = item.get("_identifiers", [])
    if identifiers:
        print(f"  Identifiers ({len(identifiers)}):")
        for ident in identifiers[:10]:
            id_type = ident.get("identifier_type", "?")
            id_val = ident.get("identifier", "?")
            display = ident.get("display_name", "")
            src = ident.get("source", "")
            extras = f"  ({display})" if display else ""
            extras += f"  [from {src}]" if src else ""
            print(f"    {id_type}: {id_val}{extras}")

    ev_count = item.get("_event_count", 0)
    linked_src = item.get("_linked_sources", [])
    print(f"  Events:  {ev_count} linked, sources: {linked_src}")


def _display_link(item: dict[str, Any]) -> None:
    """Display an event-person link for review."""
    print(f"  Event:   {item.get('event_id', 'N/A')}")
    print(f"  Person:  {item.get('person_id', 'N/A')}")
    print(f"  Name:    {item.get('canonical_name', '(none)')}")
    print(f"  Role:    {item.get('role', 'N/A')}")
    print(f"  Is user: {bool(item.get('is_user', 0))}")
    print(f"  Source:  {item.get('source', 'N/A')}")
    print(f"  Type:    {item.get('event_type', 'N/A')}")
    print(f"  Time:    {_format_timestamp(item.get('timestamp'))}")

    participants = item.get("participants", "[]")
    if isinstance(participants, str):
        try:
            participants = json.loads(participants)
        except (json.JSONDecodeError, TypeError):
            pass
    if participants and participants != "[]":
        print(f"  Partic:  {str(participants)[:150]}")

    content = item.get("content_preview") or item.get("raw_content")
    if content:
        print(f"  Content: {_truncate(content, 200)}")


def _display_deterministic_claim(item: dict[str, Any]) -> None:
    """Display a deterministic (Stage 1) claim for review."""
    print(f"  Claim ID:    {item.get('id', 'N/A')}")
    print(f"  Type:        {item.get('claim_type', 'N/A')}")
    print(f"  Subject:     {item.get('subject', 'N/A')}")
    person_name = item.get("_person_name")
    if person_name:
        print(f"  Person:      {person_name}")
    print(f"  Predicate:   {item.get('predicate', 'N/A')}")
    print(f"  Object:      {_truncate(str(item.get('object', '')), 200)}")
    print(f"  Confidence:  {item.get('confidence', 'N/A')}")

    source_events = item.get("_source_events", [])
    if source_events:
        print(f"  Source events ({len(source_events)}):")
        for ev in source_events:
            src = ev.get("source", "?")
            ts = _format_timestamp(ev.get("timestamp"))
            preview = ev.get("content_preview", "")
            print(f"    {src} / {ts}: {_truncate(str(preview), 100)}")


def _display_propagated(item: dict[str, Any]) -> None:
    """Display a propagated triage claim for review."""
    print(f"  Claim ID:    {item.get('id', 'N/A')}")
    print(f"  Confidence:  {item.get('confidence', 'N/A')}")
    tier = item.get("_tier", "N/A")
    print(f"  Tier:        {tier}")

    obj = item.get("object", "")
    if isinstance(obj, str):
        try:
            obj_data = json.loads(obj)
            if isinstance(obj_data, dict):
                for k in ("domain", "topics", "reason"):
                    if k in obj_data:
                        print(f"  {k.title():12s}: {obj_data[k]}")
        except (json.JSONDecodeError, TypeError):
            print(f"  Object:      {_truncate(obj, 200)}")

    sender = item.get("_sender")
    if sender:
        name = sender.get("canonical_name", "?")
        is_user = bool(sender.get("is_user", 0))
        ev_count = item.get("_sender_event_count", 0)
        print(f"  Sender:      {name} {'(USER)' if is_user else ''} ({ev_count} events)")

    event = item.get("_event")
    if event:
        print(f"  --- Source Event ---")
        print(f"  Source:  {event.get('source', 'N/A')}")
        print(f"  Time:    {_format_timestamp(event.get('timestamp'))}")
        content = event.get("raw_content")
        print(f"  Content: {_truncate(content, 300)}")


def _display_commitment(item: dict[str, Any]) -> None:
    """Display a commitment claim for review."""
    print(f"  Claim ID:    {item.get('id', 'N/A')}")
    print(f"  Subject:     {item.get('subject', 'N/A')}")
    print(f"  Predicate:   {item.get('predicate', 'N/A')}")
    print(f"  Confidence:  {item.get('confidence', 'N/A')}")

    obj = item.get("object", "")
    if isinstance(obj, str):
        try:
            obj_data = json.loads(obj)
            if isinstance(obj_data, dict):
                for k in ("what", "who", "deadline", "commitment_type", "direction"):
                    if k in obj_data:
                        print(f"  {k.title():12s}: {obj_data[k]}")
        except (json.JSONDecodeError, TypeError):
            print(f"  Object:      {_truncate(obj, 200)}")

    event = item.get("_event")
    if event:
        print(f"  --- Source Event ---")
        print(f"  Source:  {event.get('source', 'N/A')}")
        print(f"  Content: {_truncate(event.get('raw_content'), 300)}")


def _display_belief(item: dict[str, Any]) -> None:
    """Display a belief for review."""
    print(f"  Belief ID:   {item.get('id', 'N/A')}")
    print(f"  Type:        {item.get('belief_type', 'N/A')}")
    print(f"  Subject:     {item.get('subject', 'N/A')}")
    print(f"  Summary:     {item.get('summary', 'N/A')}")
    print(f"  Confidence:  {item.get('confidence', 'N/A')}")
    print(f"  Epistemic:   {item.get('epistemic_level', 'N/A')}")
    print(f"  Status:      {item.get('status', 'N/A')}")

    data = item.get("data", "{}")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            data = {}
    if data:
        print(f"  Data:        {json.dumps(data, indent=2)[:300]}")

    claims = item.get("source_claims", "[]")
    if isinstance(claims, str):
        try:
            claims = json.loads(claims)
        except (json.JSONDecodeError, TypeError):
            claims = []
    if claims:
        print(f"  Source claims: {len(claims)} claim(s)")


DISPLAY_FUNCTIONS = {
    "0_ingest": _display_event,
    "1_resolve": _display_person,
    "2_link": _display_link,
    "3_claims": _display_deterministic_claim,
    "4_triage": _display_triage,
    "5_propagate": _display_propagated,
    "6_extract": _display_commitment,
    "7_synthesize": _display_belief,
}


def review_frozen(
    stage: str,
    n: int = 0,
    reviewer_name: str = "",
    golden_dir: str | None = None,
) -> dict[str, Any]:
    """Review already-frozen golden records.

    Loads existing records from the golden store and presents them
    for human judgment. Updates are appended (last write wins).

    Args:
        n: Max records to review. 0 = all.

    Returns summary: {reviewed, approved, corrected, rejected, flagged, skipped}
    """
    golden = GoldenStore(golden_dir)
    display_fn = DISPLAY_FUNCTIONS.get(stage, _display_event)

    records = golden.list_records(stage)
    if not records:
        print(f"No frozen records for stage {stage}.")
        return {"reviewed": 0}

    # Filter to only auto-frozen / pending (skip already human-reviewed)
    pending = [r for r in records if r.reviewer == "freeze" or r.judgment == "pending"]
    if not pending:
        print(f"All {len(records)} records already reviewed. Use without --frozen to sample new items.")
        return {"reviewed": 0}

    if n > 0:
        pending = pending[:n]

    print(f"\n{len(pending)} frozen records to review (of {len(records)} total for stage {stage}).\n")
    print("Judgments: [a]pprove  [c]orrect  [r]eject  [f]lag  [s]kip  [q]uit")
    print("=" * 60)

    stats = {"reviewed": 0, "approve": 0, "correct": 0, "reject": 0, "flag": 0, "skip": 0}

    for i, record in enumerate(pending, 1):
        print(f"\n--- Record {i}/{len(pending)} (id={record.item_id[:16]}) ---")
        display_fn(record.input_data)
        print()

        while True:
            try:
                choice = input("Judgment [a/c/r/f/s/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended.")
                _print_review_summary(stats)
                return stats

            if choice in JUDGMENTS:
                break
            print(f"  Invalid choice '{choice}'. Use: a, c, r, f, s, q")

        judgment = JUDGMENTS[choice]

        if judgment == "quit":
            print("Session ended by user.")
            break

        if judgment == "skip":
            stats["skip"] += 1
            continue

        notes = ""
        if judgment in ("correct", "reject", "flag"):
            try:
                notes = input("Notes (optional): ").strip()
            except (EOFError, KeyboardInterrupt):
                pass

        golden.update_judgment(
            stage=stage,
            item_id=record.item_id,
            judgment=judgment,
            notes=notes,
            reviewer=reviewer_name or "human",
        )

        stats["reviewed"] += 1
        stats[judgment] += 1
        print(f"  -> {judgment}" + (f" ({notes})" if notes else ""))

    _print_review_summary(stats)
    return stats


def _print_review_summary(stats: dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"Review complete: {stats['reviewed']} reviewed")
    for k in ("approve", "correct", "reject", "flag", "skip"):
        if stats.get(k, 0) > 0:
            print(f"  {k}: {stats[k]}")


def review_session(
    db_path: str,
    stage: str,
    n: int = 10,
    stratify_by: str = "default",
    reviewer_name: str = "",
    golden_dir: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run an interactive review session.

    Returns summary: {reviewed, approved, corrected, rejected, flagged, skipped}
    """
    golden = GoldenStore(golden_dir)
    display_fn = DISPLAY_FUNCTIONS.get(stage, _display_event)

    print(f"\nSampling {n} items for stage {stage}...")
    items = sample(db_path, stage, n, stratify_by, seed)

    if not items:
        print("No items to review.")
        return {"reviewed": 0}

    print(f"Got {len(items)} items. Starting review.\n")
    print("Judgments: [a]pprove  [c]orrect  [r]eject  [f]lag  [s]kip  [q]uit")
    print("=" * 60)

    stats = {"reviewed": 0, "approve": 0, "correct": 0, "reject": 0, "flag": 0, "skip": 0}

    for i, item in enumerate(items, 1):
        print(f"\n--- Item {i}/{len(items)} ---")
        display_fn(item)
        print()

        while True:
            try:
                choice = input("Judgment [a/c/r/f/s/q]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nSession ended.")
                return stats

            if choice in JUDGMENTS:
                break
            print(f"  Invalid choice '{choice}'. Use: a, c, r, f, s, q")

        judgment = JUDGMENTS[choice]

        if judgment == "quit":
            print("Session ended by user.")
            break

        if judgment == "skip":
            stats["skip"] += 1
            continue

        notes = ""
        if judgment in ("correct", "reject", "flag"):
            try:
                notes = input("Notes (optional): ").strip()
            except (EOFError, KeyboardInterrupt):
                pass

        item_id = item.get("id", "")
        record = GoldenRecord(
            id="",
            stage=stage,
            item_id=item_id,
            input_data=item,
            judgment=judgment,
            notes=notes,
            reviewer=reviewer_name,
        )
        golden.add(record)

        stats["reviewed"] += 1
        stats[judgment] += 1
        print(f"  -> {judgment}" + (f" ({notes})" if notes else ""))

    print(f"\n{'=' * 60}")
    print(f"Review complete: {stats['reviewed']} reviewed")
    for k in ("approve", "correct", "reject", "flag", "skip"):
        if stats[k] > 0:
            print(f"  {k}: {stats[k]}")

    return stats
