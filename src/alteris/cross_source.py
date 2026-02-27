"""Cross-source content linking: deterministic annotations and claims.

Stage 1.5 in the pipeline. Runs after ingestion + person resolution, before
LLM triage. No LLM calls — pure regex, SQL aggregation, and temporal math.

Two outputs:
  1. Annotations: faceted observations on individual events.
     These become a queryable index for synthesis and LLM context retrieval.
  2. Claims: cross-source linking assertions when matching signals appear in
     events from different sources within a time window.

Annotation facets (inclusion-based, not exclusion):
  - dollar_amount: "$472.50" extracted via regex from raw_content
  - person_mention: known name from contacts/profile found in event text

Claim types:
  - cross_source_dollar_cluster: same dollar amount within time window across sources
  - cross_source_temporal_burst: 3+ events from 2+ sources in a short window
  - cross_source_entity_bridge: same non-user person in event_persons from 2+ sources
  - cross_source_name_bridge: same known name mentioned in text from 2+ sources
  - calendar_corroboration: calendar event + messages from other sources with
    shared name/dollar mentions within 30min
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

from alteris.confidence import compute_confidence
from alteris.models import (
    Annotation,
    Claim,
    ExtractionMethod,
    ExtractionProvenance,
    Modality,
)
from alteris.privacy import SensitivityLevel
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DOLLAR_RE = re.compile(r"\$[\d,]+\.\d{2}")
"""Match dollar amounts like $472.50, $1,200.00."""

TEMPORAL_BURST_WINDOW_S = 300
"""5 minutes: events within this window form a burst."""

TEMPORAL_BURST_MIN_EVENTS = 3
"""Minimum events from 2+ sources to qualify as a burst."""

DOLLAR_CLUSTER_WINDOW_S = 3600
"""1 hour: dollar amounts within this window may be related."""

CALENDAR_CORROBORATION_WINDOW_S = 1800
"""30 minutes: messages within this window of a calendar event may be related."""

NAME_MIN_LEN = 3
"""Minimum name length for full-name matching."""

FIRST_NAME_MIN_LEN = 5
"""Minimum first-name length for standalone matching.
Prevents "The", "Will", "Mark", "Team", "Meta" etc. from matching
when they appear as first names of multi-word contact entries."""

_PROVENANCE = ExtractionProvenance(
    model_id="deterministic",
    prompt_version="cross_source_v2",
    extraction_method=ExtractionMethod.DETERMINISTIC,
)


def _claim_id(claim_type: str, subject: str, predicate: str) -> str:
    """Deterministic claim ID from semantic key."""
    raw = f"{claim_type}:{subject}:{predicate}"
    return f"claim:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Name index (inclusion list from contacts + profile)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_name_index(store: LayeredGraphStore) -> dict[str, str]:
    """Build a lookup of searchable name -> person_id/label.

    Sources:
      1. Persons table: canonical_name -> first name, full name
      2. Profile.yaml: family, care providers, nicknames

    Returns: {lowercase_name: person_id_or_label}
    Only includes names with 3+ chars to avoid false matches.
    """
    name_map: dict[str, str] = {}

    # 1. From persons table
    rows = store.conn.execute(
        """SELECT person_id, canonical_name FROM persons
           WHERE canonical_name != '' AND is_user = 0"""
    ).fetchall()

    for r in rows:
        full_name = r["canonical_name"].strip()
        person_id = r["person_id"]

        # Skip entries that look like businesses/brands (all caps, numbers)
        if full_name.isupper() or any(c.isdigit() for c in full_name):
            continue

        # Full name with 2+ words (e.g., "Sam Park") — high precision
        parts = full_name.split()
        if len(parts) >= 2 and len(full_name) >= NAME_MIN_LEN:
            name_map[full_name.lower()] = person_id

        # First name only — require 5+ chars to avoid common-word collisions
        # (skips "The", "Will", "Mark", "Team", "Meta", "Home", etc.)
        if parts:
            first = parts[0]
            if len(first) >= FIRST_NAME_MIN_LEN and first[0].isupper():
                name_map[first.lower()] = person_id

    # 2. From profile.yaml
    profile_names = _load_profile_names()
    for name, label in profile_names.items():
        if len(name) >= NAME_MIN_LEN:
            name_map[name.lower()] = label

    return name_map


def _load_profile_names() -> dict[str, str]:
    """Load named entities from profile.yaml.

    Returns: {name: label} for family, care providers, etc.
    """
    from alteris.profile import get_user_name, load_profile

    profile = load_profile()
    if not profile:
        return {}

    names: dict[str, str] = {}

    # User's own name
    user_name = get_user_name(profile)
    if user_name:
        names[user_name] = "user"

    # v2 hierarchical format: family_and_relationships.immediate_family
    far = profile.get("family_and_relationships", {})
    if isinstance(far, dict):
        for member in far.get("immediate_family", []):
            if isinstance(member, str):
                # Extract name (before parenthetical)
                name_part = member.split("(")[0].strip().rstrip(" -")
                if name_part:
                    names[name_part] = "family"
        for member in far.get("extended_network", []):
            if isinstance(member, str):
                name_part = member.split("(")[0].strip().rstrip(" -")
                if name_part:
                    names[name_part] = "family:extended"

    # v1 flat format: family.spouse, family.children, family.care_providers
    family = profile.get("family", {})
    if isinstance(family, dict):
        if family.get("spouse"):
            names[family["spouse"]] = "family:spouse"
        children = family.get("children", [])
        for child in children:
            if isinstance(child, dict):
                if child.get("name"):
                    names[child["name"]] = "family:child"
                for nick in child.get("nicknames", []):
                    names[nick] = "family:child:nickname"
        for provider in family.get("care_providers", []):
            if isinstance(provider, dict) and provider.get("name"):
                names[provider["name"]] = f"family:care:{provider.get('role', '')}"

    return names


def find_names_in_text(text: str, name_index: dict[str, str]) -> list[tuple[str, str]]:
    """Find known names mentioned in text.

    Returns: [(matched_name, person_id_or_label), ...]
    Uses word-boundary matching to avoid partial matches.
    """
    if not text:
        return []

    text_lower = text.lower()
    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Sort by length descending so "Sam Park" matches before "Sam"
    for name in sorted(name_index.keys(), key=len, reverse=True):
        if name in seen:
            continue

        # Word-boundary check: name must be surrounded by non-alpha chars
        # Use regex for precise word boundary matching
        pattern = r'(?<![a-zA-Z])' + re.escape(name) + r'(?![a-zA-Z])'
        if re.search(pattern, text_lower):
            found.append((name, name_index[name]))
            seen.add(name)
            # If we matched a full name, skip matching its first name
            parts = name.split()
            if len(parts) > 1:
                for part in parts:
                    seen.add(part.lower())

    return found


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dollar amount extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_dollar_amounts(text: str) -> list[str]:
    """Extract normalized dollar amounts from text.

    Returns amounts as strings like '472.50' (no $ prefix, no commas).
    """
    matches = DOLLAR_RE.findall(text or "")
    amounts = []
    for m in matches:
        normalized = m.replace("$", "").replace(",", "")
        if normalized not in amounts:
            amounts.append(normalized)
    return amounts


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Annotation extraction (per-event content signals)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def annotate_events(
    store: LayeredGraphStore,
    since_ts: int | None = None,
) -> dict[str, int]:
    """Annotate all events with content signals.

    Two signal types (inclusion-based):
      1. Dollar amounts: regex match → dollar_amount facet
      2. Name mentions: known names from contacts/profile → person_mention facet

    Idempotent (INSERT OR IGNORE).
    Returns: {dollar_amounts: N, name_mentions: N, events_scanned: N}
    """
    t0 = time.time()
    now_ts = int(time.time())

    # Build the name lookup from contacts + profile
    name_index = build_name_index(store)
    logger.info("Name index: %d searchable names", len(name_index))

    conditions = []
    params: list = []
    if since_ts:
        conditions.append("timestamp >= ?")
        params.append(since_ts)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    rows = store.conn.execute(
        f"SELECT id, source, raw_content, metadata FROM events {where}",  # noqa: S608
        params,
    ).fetchall()

    dollar_anns: list[Annotation] = []
    name_anns: list[Annotation] = []

    for row in rows:
        event_id = row["id"]
        raw_content = row["raw_content"] or ""
        metadata = json.loads(row["metadata"] or "{}")

        # Combine subject + content for matching
        subject = metadata.get("subject", "")
        full_text = f"{subject} {raw_content}" if subject else raw_content

        # 1. Dollar amounts
        amounts = extract_dollar_amounts(full_text)
        for amt in amounts:
            dollar_anns.append(Annotation(
                event_id=event_id,
                facet="dollar_amount",
                value=amt,
                confidence=1.0,
                source="structural",
                created_at=now_ts,
            ))

        # 2. Name mentions — scan first 1000 chars (enough for detection,
        # avoids scanning full email bodies which may contain signatures/footers)
        text_to_scan = full_text[:1000]
        mentions = find_names_in_text(text_to_scan, name_index)
        for name, label in mentions:
            name_anns.append(Annotation(
                event_id=event_id,
                facet="person_mention",
                value=name,
                confidence=0.9,
                source="structural",
                created_at=now_ts,
            ))

    # Batch insert
    dollar_count = store.put_annotations_batch(dollar_anns) if dollar_anns else 0
    name_count = store.put_annotations_batch(name_anns) if name_anns else 0

    elapsed = time.time() - t0
    logger.info(
        "Annotated %d events: %d dollar amounts, %d name mentions in %.1fs",
        len(rows), dollar_count, name_count, elapsed,
    )

    return {
        "events_scanned": len(rows),
        "dollar_amounts": dollar_count,
        "name_mentions": name_count,
        "names_in_index": len(name_index),
        "duration_seconds": round(elapsed, 2),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Cross-source claims (match signals across sources)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_dollar_clusters(store: LayeredGraphStore) -> list[Claim]:
    """Find events with the same dollar amount within a time window.

    Groups by normalized amount, then checks if events from different
    sources fall within DOLLAR_CLUSTER_WINDOW_S of each other.
    """
    rows = store.conn.execute(
        """SELECT a.event_id, a.value AS amount, e.source, e.timestamp
           FROM annotations a
           JOIN events e ON a.event_id = e.id
           WHERE a.facet = 'dollar_amount'
           ORDER BY a.value, e.timestamp"""
    ).fetchall()

    by_amount: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_amount[r["amount"]].append({
            "event_id": r["event_id"],
            "source": r["source"],
            "timestamp": r["timestamp"],
        })

    claims: list[Claim] = []
    for amount, events in by_amount.items():
        if len(events) < 2:
            continue

        sources = {e["source"] for e in events}
        if len(sources) < 2:
            continue

        events_sorted = sorted(events, key=lambda x: x["timestamp"])
        clusters = _cluster_by_time(events_sorted, DOLLAR_CLUSTER_WINDOW_S)

        for cluster in clusters:
            cluster_sources = {e["source"] for e in cluster}
            if len(cluster_sources) < 2:
                continue

            event_ids = [e["event_id"] for e in cluster]
            min_ts = min(e["timestamp"] for e in cluster)
            max_ts = max(e["timestamp"] for e in cluster)
            confidence = compute_confidence(
                "cross_source_inference", len(cluster), evidence_scale=5,
            )

            claims.append(Claim(
                id=_claim_id("cross_source_dollar_cluster", amount, f"{min_ts}"),
                event_ids=event_ids,
                claim_type="cross_source_dollar_cluster",
                subject=f"${amount}",
                predicate=f"amount_cluster:{amount}:{min_ts}",
                object=json.dumps({
                    "amount": amount,
                    "event_count": len(cluster),
                    "sources": sorted(cluster_sources),
                    "span_seconds": max_ts - min_ts,
                    "min_ts": min_ts,
                    "max_ts": max_ts,
                }),
                confidence=confidence,
                modality=Modality.OBSERVED,
                provenance=_PROVENANCE,
                sensitivity=SensitivityLevel.SENSITIVE,
            ))

    return claims


def _extract_temporal_bursts(store: LayeredGraphStore) -> list[Claim]:
    """Find multi-source activity bursts within TEMPORAL_BURST_WINDOW_S.

    A burst is 3+ events from 2+ sources within a 5-minute window.
    These suggest a real-world event triggered activity across channels.
    """
    rows = store.conn.execute(
        """SELECT id, source, timestamp, event_type
           FROM events
           WHERE event_type != 'identity'
           ORDER BY timestamp"""
    ).fetchall()

    events = [dict(r) for r in rows]
    claims: list[Claim] = []
    i = 0
    seen_windows: set[int] = set()

    while i < len(events):
        window = [events[i]]
        j = i + 1
        while j < len(events) and events[j]["timestamp"] - events[i]["timestamp"] <= TEMPORAL_BURST_WINDOW_S:
            window.append(events[j])
            j += 1

        sources = {e["source"] for e in window}
        if len(window) >= TEMPORAL_BURST_MIN_EVENTS and len(sources) >= 2:
            window_key = events[i]["timestamp"] // TEMPORAL_BURST_WINDOW_S
            if window_key not in seen_windows:
                seen_windows.add(window_key)
                event_ids = [e["id"] for e in window]
                min_ts = min(e["timestamp"] for e in window)
                max_ts = max(e["timestamp"] for e in window)

                claims.append(Claim(
                    id=_claim_id("cross_source_temporal_burst", "burst", f"{window_key}"),
                    event_ids=event_ids,
                    claim_type="cross_source_temporal_burst",
                    subject="temporal_burst",
                    predicate=f"burst:{window_key}",
                    object=json.dumps({
                        "event_count": len(window),
                        "sources": sorted(sources),
                        "source_count": len(sources),
                        "span_seconds": max_ts - min_ts,
                        "min_ts": min_ts,
                        "max_ts": max_ts,
                        "event_types": sorted({e["event_type"] for e in window}),
                    }),
                    confidence=compute_confidence(
                        "cross_source_inference",
                        len(window) + len(sources),
                        evidence_scale=10,
                    ),
                    modality=Modality.OBSERVED,
                    provenance=_PROVENANCE,
                    sensitivity=SensitivityLevel.SENSITIVE,
                ))
        i += 1

    return claims


def _extract_entity_bridges(store: LayeredGraphStore) -> list[Claim]:
    """Find non-user persons who appear in events from 2+ sources.

    These persons are natural bridges for cross-source linking.
    """
    rows = store.conn.execute(
        """SELECT ep.person_id, p.canonical_name,
                  GROUP_CONCAT(DISTINCT e.source) AS sources,
                  COUNT(DISTINCT e.source) AS source_count,
                  COUNT(DISTINCT e.id) AS event_count
           FROM event_persons ep
           JOIN events e ON ep.event_id = e.id
           JOIN persons p ON ep.person_id = p.person_id
           WHERE p.is_user = 0
             AND e.event_type != 'identity'
           GROUP BY ep.person_id
           HAVING COUNT(DISTINCT e.source) >= 2"""
    ).fetchall()

    claims: list[Claim] = []
    for r in rows:
        person_id = r["person_id"]
        sources = r["sources"].split(",")

        event_rows = store.conn.execute(
            """SELECT DISTINCT e.id, e.source
               FROM event_persons ep
               JOIN events e ON ep.event_id = e.id
               WHERE ep.person_id = ?
                 AND e.event_type != 'identity'
               LIMIT 100""",
            (person_id,),
        ).fetchall()
        event_ids = [er["id"] for er in event_rows]

        claims.append(Claim(
            id=_claim_id("cross_source_entity_bridge", person_id,
                         f"bridge:{','.join(sorted(sources))}"),
            event_ids=event_ids,
            claim_type="cross_source_entity_bridge",
            subject=person_id,
            predicate=f"entity_bridge:{','.join(sorted(sources))}",
            object=json.dumps({
                "person_name": r["canonical_name"],
                "person_id": person_id,
                "sources": sorted(sources),
                "source_count": r["source_count"],
                "event_count": r["event_count"],
            }),
            confidence=compute_confidence(
                "cross_source_inference", r["source_count"], evidence_scale=5,
            ),
            modality=Modality.OBSERVED,
            provenance=_PROVENANCE,
            sensitivity=SensitivityLevel.SENSITIVE,
        ))

    return claims


def _extract_name_bridges(store: LayeredGraphStore) -> list[Claim]:
    """Find known names mentioned in text from 2+ sources.

    Uses person_mention annotations to find cases where a person is
    discussed (mentioned in body text) across multiple sources, even if
    they're not a direct participant in the event.
    """
    rows = store.conn.execute(
        """SELECT a.value AS name,
                  COUNT(DISTINCT e.source) AS source_count,
                  COUNT(DISTINCT a.event_id) AS event_count,
                  GROUP_CONCAT(DISTINCT e.source) AS sources
           FROM annotations a
           JOIN events e ON a.event_id = e.id
           WHERE a.facet = 'person_mention'
             AND e.event_type != 'identity'
           GROUP BY a.value
           HAVING COUNT(DISTINCT e.source) >= 2"""
    ).fetchall()

    claims: list[Claim] = []
    for r in rows:
        name = r["name"]
        sources = r["sources"].split(",")

        event_rows = store.conn.execute(
            """SELECT a.event_id, e.source, e.timestamp
               FROM annotations a
               JOIN events e ON a.event_id = e.id
               WHERE a.facet = 'person_mention' AND a.value = ?
                 AND e.event_type != 'identity'
               ORDER BY e.timestamp
               LIMIT 50""",
            (name,),
        ).fetchall()

        event_ids = [er["event_id"] for er in event_rows]
        confidence = compute_confidence(
            "cross_source_inference", r["source_count"], evidence_scale=5,
        )

        claims.append(Claim(
            id=_claim_id("cross_source_name_bridge", name,
                         f"bridge:{','.join(sorted(sources))}"),
            event_ids=event_ids,
            claim_type="cross_source_name_bridge",
            subject=name,
            predicate=f"name_bridge:{','.join(sorted(sources))}",
            object=json.dumps({
                "name": name,
                "sources": sorted(sources),
                "source_count": r["source_count"],
                "event_count": r["event_count"],
            }),
            confidence=confidence,
            modality=Modality.OBSERVED,
            provenance=_PROVENANCE,
            sensitivity=SensitivityLevel.SENSITIVE,
        ))

    return claims


def _extract_calendar_corroboration(store: LayeredGraphStore) -> list[Claim]:
    """Find calendar events corroborated by messages from other sources.

    For each calendar event, check if messages from other sources appeared
    within CALENDAR_CORROBORATION_WINDOW_S and share person_mention or
    dollar_amount annotations with the calendar event.
    """
    cal_rows = store.conn.execute(
        """SELECT id, timestamp,
                  json_extract(metadata, '$.subject') AS subject,
                  json_extract(metadata, '$.location') AS location
           FROM events
           WHERE source = 'calendar'
             AND event_type = 'calendar_event'"""
    ).fetchall()

    claims: list[Claim] = []

    for cal in cal_rows:
        cal_id = cal["id"]
        cal_ts = cal["timestamp"]
        cal_subject = cal["subject"] or ""

        # Get this calendar event's annotations (names + dollars)
        cal_anns = store.conn.execute(
            """SELECT facet, value FROM annotations
               WHERE event_id = ? AND facet IN ('person_mention', 'dollar_amount')""",
            (cal_id,),
        ).fetchall()
        cal_signals = {(r["facet"], r["value"]) for r in cal_anns}
        if not cal_signals:
            continue

        # Find messages within window from other sources
        msg_rows = store.conn.execute(
            """SELECT id, source, timestamp
               FROM events
               WHERE source != 'calendar'
                 AND event_type != 'identity'
                 AND timestamp BETWEEN ? AND ?""",
            (cal_ts - CALENDAR_CORROBORATION_WINDOW_S,
             cal_ts + CALENDAR_CORROBORATION_WINDOW_S),
        ).fetchall()

        corroborating: list[dict] = []
        for msg in msg_rows:
            msg_anns = store.conn.execute(
                """SELECT facet, value FROM annotations
                   WHERE event_id = ?
                     AND facet IN ('person_mention', 'dollar_amount')""",
                (msg["id"],),
            ).fetchall()
            msg_signals = {(r["facet"], r["value"]) for r in msg_anns}
            overlap = cal_signals & msg_signals
            if overlap:
                corroborating.append({
                    "event_id": msg["id"],
                    "source": msg["source"],
                    "timestamp": msg["timestamp"],
                    "shared_signals": [
                        {"facet": f, "value": v} for f, v in sorted(overlap)
                    ],
                    "offset_seconds": msg["timestamp"] - cal_ts,
                })

        if not corroborating:
            continue

        corr_sources = {c["source"] for c in corroborating}
        event_ids = [cal_id] + [c["event_id"] for c in corroborating]
        confidence = compute_confidence(
            "cross_source_inference",
            len(corroborating) + len(corr_sources),
            evidence_scale=5,
        )

        all_shared = set()
        for c in corroborating:
            for s in c["shared_signals"]:
                all_shared.add(s["value"])

        claims.append(Claim(
            id=_claim_id("calendar_corroboration", cal_id, "corroboration"),
            event_ids=event_ids,
            claim_type="calendar_corroboration",
            subject=cal_id,
            predicate=f"corroboration:{cal_id}",
            object=json.dumps({
                "calendar_subject": cal_subject,
                "calendar_ts": cal_ts,
                "shared_signals": sorted(all_shared),
                "corroborating_sources": sorted(corr_sources),
                "corroborating_count": len(corroborating),
            }),
            confidence=confidence,
            modality=Modality.OBSERVED,
            provenance=_PROVENANCE,
            sensitivity=SensitivityLevel.SENSITIVE,
        ))

    return claims


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cluster_by_time(events: list[dict], window_s: int) -> list[list[dict]]:
    """Cluster a sorted list of events by temporal proximity."""
    if not events:
        return []
    clusters: list[list[dict]] = [[events[0]]]
    for e in events[1:]:
        if e["timestamp"] - clusters[-1][0]["timestamp"] <= window_s:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    return clusters


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Context retrieval (for LLM extraction augmentation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_related_events(
    store: LayeredGraphStore,
    event_id: str,
    max_results: int = 10,
) -> list[dict]:
    """Find events from other sources related to a given event.

    Uses annotations (dollar amounts, name mentions) and person graph to
    find cross-source matches. Called during extraction to provide the LLM
    with cross-source context.

    Returns list of {event_id, source, timestamp, relationship, shared_signal}.
    """
    event_row = store.conn.execute(
        "SELECT source, timestamp FROM events WHERE id = ?", (event_id,)
    ).fetchone()
    if not event_row:
        return []

    event_source = event_row["source"]
    event_ts = event_row["timestamp"]

    results: list[dict] = []
    seen_ids: set[str] = set()

    # 1. Match by dollar amount
    dollar_anns = store.get_annotations(event_id=event_id, facet="dollar_amount")
    for ann in dollar_anns:
        matching = store.conn.execute(
            """SELECT a.event_id, e.source, e.timestamp,
                      json_extract(e.metadata, '$.subject') AS subject
               FROM annotations a
               JOIN events e ON a.event_id = e.id
               WHERE a.facet = 'dollar_amount' AND a.value = ?
                 AND e.source != ?
                 AND a.event_id != ?
               ORDER BY ABS(e.timestamp - ?) LIMIT ?""",
            (ann.value, event_source, event_id, event_ts, max_results),
        ).fetchall()
        for m in matching:
            if m["event_id"] not in seen_ids:
                seen_ids.add(m["event_id"])
                results.append({
                    "event_id": m["event_id"],
                    "source": m["source"],
                    "timestamp": m["timestamp"],
                    "relationship": "same_dollar_amount",
                    "shared_signal": f"${ann.value}",
                    "subject": m["subject"],
                })

    # 2. Match by person name mentions
    name_anns = store.get_annotations(event_id=event_id, facet="person_mention")
    for ann in name_anns:
        matching = store.conn.execute(
            """SELECT a.event_id, e.source, e.timestamp,
                      json_extract(e.metadata, '$.subject') AS subject
               FROM annotations a
               JOIN events e ON a.event_id = e.id
               WHERE a.facet = 'person_mention' AND a.value = ?
                 AND e.source != ?
                 AND a.event_id != ?
               ORDER BY ABS(e.timestamp - ?) LIMIT 3""",
            (ann.value, event_source, event_id, event_ts),
        ).fetchall()
        for m in matching:
            if m["event_id"] not in seen_ids:
                seen_ids.add(m["event_id"])
                results.append({
                    "event_id": m["event_id"],
                    "source": m["source"],
                    "timestamp": m["timestamp"],
                    "relationship": "same_person_mentioned",
                    "shared_signal": ann.value,
                    "subject": m["subject"],
                })

    # 3. Match by shared persons in event_persons (direct participants)
    person_rows = store.conn.execute(
        """SELECT ep.person_id, p.canonical_name
           FROM event_persons ep
           JOIN persons p ON ep.person_id = p.person_id
           WHERE ep.event_id = ? AND p.is_user = 0""",
        (event_id,),
    ).fetchall()
    for pr in person_rows:
        matching = store.conn.execute(
            """SELECT DISTINCT e.id AS event_id, e.source, e.timestamp,
                      json_extract(e.metadata, '$.subject') AS subject
               FROM event_persons ep
               JOIN events e ON ep.event_id = e.id
               WHERE ep.person_id = ?
                 AND e.source != ?
                 AND e.id != ?
                 AND e.event_type != 'identity'
               ORDER BY ABS(e.timestamp - ?) LIMIT 3""",
            (pr["person_id"], event_source, event_id, event_ts),
        ).fetchall()
        for m in matching:
            if m["event_id"] not in seen_ids:
                seen_ids.add(m["event_id"])
                results.append({
                    "event_id": m["event_id"],
                    "source": m["source"],
                    "timestamp": m["timestamp"],
                    "relationship": "shared_person",
                    "shared_signal": pr["canonical_name"],
                    "subject": m["subject"],
                })

    results.sort(key=lambda x: abs(x["timestamp"] - event_ts))
    return results[:max_results]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_cross_source_linking(
    store: LayeredGraphStore,
    since_ts: int | None = None,
) -> dict[str, int | float | dict]:
    """Run the full cross-source linking pipeline.

    Phase 1: Annotate events with dollar amounts + name mentions
    Phase 2: Extract cross-source linking claims

    Returns summary stats.
    """
    t0 = time.time()

    # Phase 1: Content annotations
    ann_stats = annotate_events(store, since_ts=since_ts)

    # Phase 2: Cross-source claims
    extractors = [
        ("dollar_clusters", _extract_dollar_clusters),
        ("temporal_bursts", _extract_temporal_bursts),
        ("entity_bridges", _extract_entity_bridges),
        ("name_bridges", _extract_name_bridges),
        ("calendar_corroboration", _extract_calendar_corroboration),
    ]

    all_claims: list[Claim] = []
    by_type: dict[str, int] = {}

    for name, fn in extractors:
        claims = fn(store)
        all_claims.extend(claims)
        by_type[name] = len(claims)
        logger.info("Cross-source %s: %d claims", name, len(claims))

    new_claims = 0
    existing_claims = 0
    for claim in all_claims:
        if store.put_claim(claim):
            new_claims += 1
        else:
            existing_claims += 1

    elapsed = time.time() - t0
    logger.info(
        "Cross-source linking: %d claims (%d new, %d existing) in %.1fs",
        len(all_claims), new_claims, existing_claims, elapsed,
    )

    return {
        "annotations": ann_stats,
        "total_claims": len(all_claims),
        "by_type": by_type,
        "new_claims": new_claims,
        "existing_claims": existing_claims,
        "duration_seconds": round(elapsed, 2),
    }
