"""Stage 8: Weekly briefing — calendar events + commitments + blind spots.

Architecture (following briefing_v2 from the previous codebase):
  Phase 1 (SQL, no LLM):
    - Upcoming calendar events with attendee resolution
    - ALL open commitments from claims table (direct, not via beliefs)
    - Recently resolved commitments (closed in last 7 days)
    - Match commitments → calendar events via person graph overlap
    - Identify orphaned commitments (not tied to any upcoming event)

  Phase 2 (SQL, no LLM):
    - Per-event context: recent communications with attendees
    - Source thread snippets for orphaned commitments
    - Cross-source annotations (dollar amounts, name mentions)
    - Staleness signals (days overdue, group message tags)

  Phase 2.5 (LLM):
    - Agentic triage: LLM reviews context, requests additional queries

  Phase 3 (LLM):
    - Synthesize briefing via mental simulation framework
    - Per-event lifecycle: Approach → Room → Materials → Risks → Exit

  Phase 4 (LLM):
    - Anticipatory questions: surface logistics gaps the briefing missed
    - 4-step walkthrough: Journey → Context → Objective → Exit
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from alteris.constants import CLOUD_FAST_MODEL, SECONDS_PER_DAY, USER_TIMEZONE, safe_timezone
from alteris.llm.base import LLMClient
from alteris.prompts.briefing import (
    ANTICIPATION_SYSTEM_PROMPT,
    BRIEFING_SYSTEM_PROMPT,
    CANDIDATE_GENERATION_PROMPT,
)
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)

BRIEFING_DAYS_AHEAD = 7
BRIEFING_LOOKBACK_DAYS = 30
BRIEFING_MAX_CONTEXT_PER_PERSON = 5
TRIAGE_MAX_QUERIES = 5
BLIND_SPOT_CANDIDATES = 10
BLIND_SPOT_FINAL = 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data classes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class CalendarEvent:
    """Upcoming calendar event with resolved attendees."""

    __slots__ = (
        "event_id", "title", "start_ts", "end_ts", "location",
        "description", "attendees", "metadata",
    )

    def __init__(
        self,
        event_id: str,
        title: str,
        start_ts: int,
        end_ts: int,
        location: str = "",
        description: str = "",
        attendees: list[dict] | None = None,
        metadata: dict | None = None,
    ):
        self.event_id = event_id
        self.title = title
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.location = location
        self.description = description
        self.attendees = attendees or []
        self.metadata = metadata or {}

    @property
    def attendee_person_ids(self) -> set[str]:
        return {a["person_id"] for a in self.attendees if a.get("person_id")}

    @property
    def duration_minutes(self) -> int:
        return max(0, (self.end_ts - self.start_ts) // 60)

    def format_time(self, tz_name: str = USER_TIMEZONE) -> str:
        tz = safe_timezone(tz_name)
        dt = datetime.fromtimestamp(self.start_ts, tz=tz)
        end_dt = datetime.fromtimestamp(self.end_ts, tz=tz)
        return f"{dt.strftime('%a %b %d, %I:%M %p')} – {end_dt.strftime('%I:%M %p')}"


class Commitment:
    """Open commitment extracted from communications."""

    __slots__ = (
        "claim_id", "type", "who", "what", "to_whom",
        "deadline", "status", "priority", "note",
        "source_thread_id", "source_person_ids", "confidence",
        "direction", "staleness_signal",
        "speech_act", "proposed_by", "response_from",
        "next_action", "evidence_quote",
    )

    def __init__(
        self,
        claim_id: str,
        type: str,
        who: str,
        what: str,
        to_whom: str = "",
        deadline: str | None = None,
        status: str = "open",
        priority: int = 3,
        note: str = "",
        source_thread_id: str = "",
        source_person_ids: set[str] | None = None,
        confidence: float = 0.8,
        direction: str = "ambiguous",
        staleness_signal: str = "none",
        speech_act: str = "request",
        proposed_by: str | None = None,
        response_from: str | None = None,
        next_action: str | None = None,
        evidence_quote: str | None = None,
    ):
        self.claim_id = claim_id
        self.type = type
        self.who = who
        self.what = what
        self.to_whom = to_whom
        self.deadline = deadline
        self.status = status
        self.priority = priority
        self.note = note
        self.source_thread_id = source_thread_id
        self.source_person_ids = source_person_ids or set()
        self.confidence = confidence
        self.direction = direction
        self.staleness_signal = staleness_signal
        self.speech_act = speech_act
        self.proposed_by = proposed_by
        self.response_from = response_from
        self.next_action = next_action
        self.evidence_quote = evidence_quote

    @property
    def is_overdue(self) -> bool:
        if not self.deadline:
            return False
        try:
            dl = datetime.strptime(self.deadline, "%Y-%m-%d").date()
            return dl < datetime.now(timezone.utc).date()
        except ValueError:
            return False

    @property
    def deadline_in_days(self) -> int | None:
        if not self.deadline:
            return None
        try:
            dl = datetime.strptime(self.deadline, "%Y-%m-%d").date()
            return (dl - datetime.now(timezone.utc).date()).days
        except ValueError:
            return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Gather data (pure SQL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _get_upcoming_events(
    store: LayeredGraphStore,
    days_ahead: int = BRIEFING_DAYS_AHEAD,
    user_tz: str = USER_TIMEZONE,
) -> list[CalendarEvent]:
    """Get upcoming calendar events with resolved attendees."""
    import zoneinfo

    tz = safe_timezone(user_tz)
    now = datetime.now(tz)
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_window = start_of_today + timedelta(days=days_ahead)

    start_ts = int(start_of_today.timestamp())
    end_ts = int(end_window.timestamp())

    rows = store.conn.execute(
        """SELECT id, metadata, timestamp, raw_content
           FROM events
           WHERE source = 'calendar'
             AND timestamp >= ? AND timestamp <= ?
           ORDER BY timestamp ASC""",
        (start_ts, end_ts),
    ).fetchall()

    events = []
    for row in rows:
        meta = json.loads(row["metadata"] or "{}")
        event_id = row["id"]

        title = meta.get("title", "") or meta.get("subject", "") or "(untitled)"
        start = row["timestamp"]
        duration_mins = meta.get("duration_minutes", 60)
        end = start + duration_mins * 60
        location = meta.get("location", "") or ""
        description = meta.get("notes", "") or meta.get("description", "") or ""

        attendee_rows = store.conn.execute(
            """SELECT ep.person_id, ep.role, p.canonical_name
               FROM event_persons ep
               JOIN persons p ON ep.person_id = p.person_id
               WHERE ep.event_id = ?
                 AND ep.role != 'self'""",
            (event_id,),
        ).fetchall()

        attendees = [
            {
                "person_id": r["person_id"],
                "name": r["canonical_name"],
                "role": r["role"],
            }
            for r in attendee_rows
        ]

        events.append(CalendarEvent(
            event_id=event_id,
            title=title,
            start_ts=start,
            end_ts=end,
            location=location,
            description=description,
            attendees=attendees,
            metadata=meta,
        ))

    return events


def _get_open_commitments(store: LayeredGraphStore) -> list[Commitment]:
    """Get all open commitments directly from claims table.

    Resolves the person graph for each commitment's source events
    so we can match commitments to calendar events.
    """
    rows = store.conn.execute(
        """SELECT c.id, c.subject, c.predicate, c.object, c.confidence
           FROM claims c
           WHERE c.claim_type = 'commitment'
             AND c.superseded_by IS NULL
             AND json_extract(c.object, '$.status') = 'open'
           ORDER BY json_extract(c.object, '$.priority') ASC,
                    c.confidence DESC"""
    ).fetchall()

    commitments = []
    for row in rows:
        obj = json.loads(row["object"])
        claim_id = row["id"]

        # Find person_ids involved in this commitment's source events
        person_rows = store.conn.execute(
            """SELECT DISTINCT ep.person_id
               FROM claim_events ce
               JOIN event_persons ep ON ce.event_id = ep.event_id
               WHERE ce.claim_id = ?
                 AND ep.role != 'self'""",
            (claim_id,),
        ).fetchall()
        source_person_ids = {r["person_id"] for r in person_rows}

        commitments.append(Commitment(
            claim_id=claim_id,
            type=obj.get("type", row["predicate"]),
            who=obj.get("who", ""),
            what=obj.get("what", ""),
            to_whom=obj.get("to_whom", ""),
            deadline=obj.get("deadline"),
            status=obj.get("status", "open"),
            priority=obj.get("priority", 3),
            note=obj.get("note", ""),
            source_thread_id=row["subject"],
            source_person_ids=source_person_ids,
            confidence=row["confidence"],
            direction=obj.get("direction", "ambiguous"),
            staleness_signal=obj.get("staleness_signal", "none"),
            speech_act=obj.get("speech_act", "request"),
            proposed_by=obj.get("proposed_by"),
            response_from=obj.get("response_from"),
            next_action=obj.get("next_action"),
            evidence_quote=obj.get("evidence_quote"),
        ))

    return commitments


def _match_commitments_to_events(
    events: list[CalendarEvent],
    commitments: list[Commitment],
) -> tuple[dict[str, list[Commitment]], list[Commitment]]:
    """Match commitments to calendar events via participant-aware overlap.

    When proposed_by/response_from are available, match only to events where
    those specific people are attendees (prevents context bleeding from broad
    thread participant overlap). Falls back to source_person_ids when the
    structural fields aren't populated.

    Returns:
        (event_commitments, orphaned_commitments)
        event_commitments: {event_id: [matching commitments]}
        orphaned_commitments: commitments not linked to any upcoming event
    """
    event_commitments: dict[str, list[Commitment]] = {
        e.event_id: [] for e in events
    }
    matched_ids: set[str] = set()

    # Build attendee name → event mapping (lowercase for fuzzy name matching)
    name_to_events: dict[str, list[str]] = {}
    for event in events:
        for a in event.attendees:
            name_lower = a.get("name", "").lower().strip()
            if name_lower:
                name_to_events.setdefault(name_lower, []).append(event.event_id)

    # Build person_id → event mapping (fallback for legacy commitments)
    person_to_events: dict[str, list[str]] = {}
    for event in events:
        for pid in event.attendee_person_ids:
            person_to_events.setdefault(pid, []).append(event.event_id)

    for commitment in commitments:
        # Prefer proposed_by/response_from for participant-aware matching
        relevant_names = set()
        if commitment.proposed_by and commitment.proposed_by != "user":
            relevant_names.add(commitment.proposed_by.lower().strip())
        if commitment.response_from and commitment.response_from != "user":
            relevant_names.add(commitment.response_from.lower().strip())
        if commitment.to_whom and commitment.to_whom != "user":
            to = commitment.to_whom.lower().strip()
            if not to.startswith("group:") and to != "unresolved":
                relevant_names.add(to)

        if relevant_names:
            # Match by commitment participant names → event attendee names
            for name in relevant_names:
                for event_name, event_ids in name_to_events.items():
                    # Fuzzy: check if commitment name is substring of event
                    # attendee name or vice versa (handles "Sarah" vs "Sarah Johnson")
                    if name in event_name or event_name in name:
                        for eid in event_ids:
                            if commitment.claim_id not in {
                                c.claim_id for c in event_commitments[eid]
                            }:
                                event_commitments[eid].append(commitment)
                        matched_ids.add(commitment.claim_id)
        else:
            # Fallback: broad person_id overlap (legacy commitments without
            # structural fields, or user-only commitments)
            for pid in commitment.source_person_ids:
                if pid in person_to_events:
                    for event_id in person_to_events[pid]:
                        if commitment.claim_id not in {
                            c.claim_id for c in event_commitments[event_id]
                        }:
                            event_commitments[event_id].append(commitment)
                    matched_ids.add(commitment.claim_id)

    orphaned = [c for c in commitments if c.claim_id not in matched_ids]
    return event_commitments, orphaned


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Gather context per event (pure SQL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _gather_event_context(
    store: LayeredGraphStore,
    event: CalendarEvent,
    lookback_days: int = BRIEFING_LOOKBACK_DAYS,
    max_per_person: int = BRIEFING_MAX_CONTEXT_PER_PERSON,
) -> list[dict]:
    """Get recent communications with event attendees.

    For each attendee, find the most recent emails/messages involving them.
    Returns list of {person_name, source, subject, snippet, timestamp, direction}.
    """
    cutoff_ts = int(time.time()) - lookback_days * SECONDS_PER_DAY
    context_items = []
    seen_event_ids: set[str] = set()

    for attendee in event.attendees:
        pid = attendee.get("person_id")
        if not pid:
            continue

        rows = store.conn.execute(
            """SELECT e.id, e.source, e.timestamp, e.metadata, e.raw_content
               FROM events e
               JOIN person_events pe ON e.id = pe.event_id
               WHERE pe.person_id = ?
                 AND e.source IN ('mail', 'imessage', 'slack', 'whatsapp', 'granola')
                 AND e.timestamp >= ?
               ORDER BY e.timestamp DESC
               LIMIT ?""",
            (pid, cutoff_ts, max_per_person),
        ).fetchall()

        for row in rows:
            if row["id"] in seen_event_ids:
                continue
            seen_event_ids.add(row["id"])

            meta = json.loads(row["metadata"] or "{}")
            content = row["raw_content"] or ""
            snippet = content[:300].replace("\n", " ").strip()

            context_items.append({
                "person_name": attendee["name"],
                "source": row["source"],
                "subject": meta.get("subject", ""),
                "snippet": snippet,
                "timestamp": row["timestamp"],
                "is_from_me": meta.get("is_from_me", False),
            })

    context_items.sort(key=lambda x: x["timestamp"], reverse=True)
    return context_items


def _get_commitment_source_snippet(
    store: LayeredGraphStore,
    commitment: Commitment,
) -> str:
    """Get a snippet from the commitment's source thread for context."""
    row = store.conn.execute(
        """SELECT e.raw_content, e.metadata, e.source
           FROM claim_events ce
           JOIN events e ON ce.event_id = e.id
           WHERE ce.claim_id = ?
           ORDER BY e.timestamp DESC
           LIMIT 1""",
        (commitment.claim_id,),
    ).fetchone()

    if not row:
        return ""

    meta = json.loads(row["metadata"] or "{}")
    content = row["raw_content"] or ""
    subject = meta.get("subject", "")
    source = row["source"]
    snippet = content[:200].replace("\n", " ").strip()

    parts = []
    if source:
        parts.append(f"[{source}]")
    if subject:
        parts.append(subject)
    if snippet:
        parts.append(f"— {snippet}")
    return " ".join(parts)


def _load_profile() -> dict:
    """Load user profile from profile.yaml."""
    from alteris.profile import load_profile
    return load_profile()


def _get_recently_resolved(
    store: LayeredGraphStore,
    lookback_days: int = 7,
) -> list[dict]:
    """Get commitments resolved in the last N days.

    Returns list of {what, who, to_whom, type, resolved_date} for display
    in a "completed this week" section.
    """
    cutoff = int(time.time()) - lookback_days * SECONDS_PER_DAY
    rows = store.conn.execute(
        """SELECT c.object, c.created_at
           FROM claims c
           WHERE c.claim_type = 'commitment'
             AND json_extract(c.object, '$.status') != 'open'
             AND c.created_at >= ?
           ORDER BY c.created_at DESC
           LIMIT 20""",
        (cutoff,),
    ).fetchall()

    resolved = []
    for r in rows:
        obj = json.loads(r["object"])
        resolved.append({
            "what": obj.get("what", ""),
            "who": obj.get("who", ""),
            "to_whom": obj.get("to_whom", ""),
            "type": obj.get("type", ""),
            "status": obj.get("status", "done"),
        })
    return resolved



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2b: Cross-source enrichment (annotations + linked events)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _get_event_facets(
    store: LayeredGraphStore,
    event_id: str,
) -> dict[str, list[str]]:
    """Get annotations/facets for a specific event.

    Returns: {facet_name: [values]} e.g. {"dollar_amount": ["472.50"], ...}
    """
    rows = store.conn.execute(
        "SELECT facet, value FROM annotations WHERE event_id = ?",
        (event_id,),
    ).fetchall()

    facets: dict[str, list[str]] = {}
    for r in rows:
        facets.setdefault(r["facet"], []).append(r["value"])
    return facets


def _get_cross_source_links(
    store: LayeredGraphStore,
    event_id: str,
    max_results: int = 8,
) -> list[dict]:
    """Find cross-source related events using annotations and person graph.

    Wraps find_related_events() from cross_source module.
    Returns: [{event_id, source, timestamp, relationship, shared_signal, subject}]
    """
    from alteris.cross_source import find_related_events

    return find_related_events(store, event_id, max_results=max_results)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2.5: Agentic triage — LLM reviews context and requests more
def _execute_graph_query(
    query: dict,
    store: LayeredGraphStore,
    user_tz: str = USER_TIMEZONE,
) -> list[dict]:
    """Execute a single graph query requested by the triage LLM."""
    import zoneinfo

    tz = safe_timezone(user_tz)
    qtype = query.get("query_type", "")
    params = query.get("params", {})
    results: list[dict] = []

    try:
        # Aliases for v2 query type vocabulary
        if qtype == "recent_from_person":
            qtype = "person_events"

        if qtype == "person_emails":
            # Emails only — filter person_events to source='mail'
            name = (params.get("name") or params.get("person_name") or "").lower()
            email = (params.get("email") or "").lower()
            days = params.get("days", 14)
            if not name and not email:
                return []
            cutoff = int(time.time()) - days * SECONDS_PER_DAY
            like_clause = f"%{email}%" if email else f"%{name}%"
            rows = store.conn.execute(
                """SELECT DISTINCT e.id, e.source, e.timestamp, e.metadata,
                          e.raw_content
                   FROM events e
                   JOIN person_events pe ON e.id = pe.event_id
                   JOIN persons p ON pe.person_id = p.person_id
                   WHERE (LOWER(p.canonical_name) LIKE ? OR LOWER(p.person_id) LIKE ?)
                     AND e.source = 'mail'
                     AND e.timestamp >= ?
                   ORDER BY e.timestamp DESC
                   LIMIT 10""",
                (like_clause, like_clause, cutoff),
            ).fetchall()
            for r in rows:
                meta = json.loads(r["metadata"] or "{}")
                dt = datetime.fromtimestamp(r["timestamp"], tz=tz)
                content = r["raw_content"] or ""
                results.append({
                    "id": r["id"],
                    "source": r["source"],
                    "date": dt.strftime("%Y-%m-%d %H:%M"),
                    "subject": meta.get("subject", ""),
                    "snippet": content[:300].replace("\n", " ").strip(),
                    "query_source": f"person_emails:{name or email}",
                })

        elif qtype == "person_messages":
            # iMessage/Slack/WhatsApp only
            name = (params.get("name") or params.get("person_name") or "").lower()
            days = params.get("days", 14)
            if not name:
                return []
            cutoff = int(time.time()) - days * SECONDS_PER_DAY
            rows = store.conn.execute(
                """SELECT DISTINCT e.id, e.source, e.timestamp, e.metadata,
                          e.raw_content
                   FROM events e
                   JOIN person_events pe ON e.id = pe.event_id
                   JOIN persons p ON pe.person_id = p.person_id
                   WHERE LOWER(p.canonical_name) LIKE ?
                     AND e.source IN ('imessage', 'slack', 'whatsapp')
                     AND e.timestamp >= ?
                   ORDER BY e.timestamp DESC
                   LIMIT 10""",
                (f"%{name}%", cutoff),
            ).fetchall()
            for r in rows:
                meta = json.loads(r["metadata"] or "{}")
                dt = datetime.fromtimestamp(r["timestamp"], tz=tz)
                content = r["raw_content"] or ""
                results.append({
                    "id": r["id"],
                    "source": r["source"],
                    "date": dt.strftime("%Y-%m-%d %H:%M"),
                    "subject": meta.get("subject", ""),
                    "snippet": content[:300].replace("\n", " ").strip(),
                    "query_source": f"person_messages:{name}",
                })

        elif qtype == "meeting_lookup":
            # Granola meeting notes by keyword or ID
            meeting_id = params.get("meeting_id", "")
            keywords = params.get("keywords", [])
            if meeting_id:
                rows = store.conn.execute(
                    """SELECT id, source, timestamp, metadata, raw_content
                       FROM events
                       WHERE id = ? AND source = 'granola'""",
                    (meeting_id,),
                ).fetchall()
            elif keywords:
                kw = keywords[0].lower() if keywords else ""
                rows = store.conn.execute(
                    """SELECT id, source, timestamp, metadata, raw_content
                       FROM events
                       WHERE source = 'granola'
                         AND (LOWER(raw_content) LIKE ?
                              OR LOWER(json_extract(metadata, '$.subject')) LIKE ?)
                       ORDER BY timestamp DESC
                       LIMIT 5""",
                    (f"%{kw}%", f"%{kw}%"),
                ).fetchall()
            else:
                rows = []
            for r in rows:
                meta = json.loads(r["metadata"] or "{}")
                dt = datetime.fromtimestamp(r["timestamp"], tz=tz)
                content = r["raw_content"] or ""
                results.append({
                    "id": r["id"],
                    "source": r["source"],
                    "date": dt.strftime("%Y-%m-%d %H:%M"),
                    "subject": meta.get("subject", ""),
                    "snippet": content[:500].replace("\n", " ").strip(),
                    "query_source": f"meeting_lookup:{meeting_id or keywords}",
                })

        elif qtype == "person_events":
            name = (params.get("person_name") or "").lower()
            days = params.get("days", 14)
            if not name:
                return []
            cutoff = int(time.time()) - days * SECONDS_PER_DAY
            rows = store.conn.execute(
                """SELECT DISTINCT e.id, e.source, e.timestamp, e.metadata,
                          e.raw_content
                   FROM events e
                   JOIN person_events pe ON e.id = pe.event_id
                   JOIN persons p ON pe.person_id = p.person_id
                   WHERE LOWER(p.canonical_name) LIKE ?
                     AND e.timestamp >= ?
                     AND e.event_type != 'identity'
                   ORDER BY e.timestamp DESC
                   LIMIT 10""",
                (f"%{name}%", cutoff),
            ).fetchall()
            for r in rows:
                meta = json.loads(r["metadata"] or "{}")
                dt = datetime.fromtimestamp(r["timestamp"], tz=tz)
                content = r["raw_content"] or ""
                results.append({
                    "id": r["id"],
                    "source": r["source"],
                    "date": dt.strftime("%Y-%m-%d %H:%M"),
                    "subject": meta.get("subject", ""),
                    "snippet": content[:300].replace("\n", " ").strip(),
                    "query_source": f"person_events:{name}",
                })

        elif qtype == "topic_search":
            keywords = params.get("keywords", [])
            if not keywords:
                return []
            for kw in keywords[:3]:
                kw_lower = kw.lower()
                rows = store.conn.execute(
                    """SELECT id, source, timestamp, metadata, raw_content
                       FROM events
                       WHERE (LOWER(raw_content) LIKE ?
                              OR LOWER(json_extract(metadata, '$.subject')) LIKE ?)
                         AND event_type != 'identity'
                       ORDER BY timestamp DESC
                       LIMIT 5""",
                    (f"%{kw_lower}%", f"%{kw_lower}%"),
                ).fetchall()
                for r in rows:
                    meta = json.loads(r["metadata"] or "{}")
                    dt = datetime.fromtimestamp(r["timestamp"], tz=tz)
                    content = r["raw_content"] or ""
                    results.append({
                        "id": r["id"],
                        "source": r["source"],
                        "date": dt.strftime("%Y-%m-%d %H:%M"),
                        "subject": meta.get("subject", ""),
                        "snippet": content[:300].replace("\n", " ").strip(),
                        "query_source": f"topic_search:{kw}",
                    })

        elif qtype in ("commitment_search", "commitments_search"):
            keywords = params.get("keywords", [])
            if not keywords:
                return []
            rows = store.conn.execute(
                """SELECT id, subject, object, confidence
                   FROM claims
                   WHERE claim_type = 'commitment'
                     AND superseded_by IS NULL
                     AND json_extract(object, '$.status') = 'open'"""
            ).fetchall()
            for r in rows:
                obj = json.loads(r["object"])
                text = (
                    f"{obj.get('who', '')} {obj.get('to_whom', '')} "
                    f"{obj.get('what', '')} {obj.get('note', '')}"
                ).lower()
                for kw in keywords:
                    if kw.lower() in text:
                        results.append({
                            "id": r["id"],
                            "type": obj.get("type", ""),
                            "what": obj.get("what", ""),
                            "who": obj.get("who", ""),
                            "to_whom": obj.get("to_whom", ""),
                            "deadline": obj.get("deadline"),
                            "priority": obj.get("priority", 3),
                            "query_source": f"commitment_search:{kw}",
                        })
                        break

        elif qtype == "facet_query":
            facet = params.get("facet", "")
            value = params.get("value")
            person_name = params.get("person_name")
            if not facet:
                return []

            if value:
                rows = store.conn.execute(
                    """SELECT a.event_id, a.value, e.source, e.timestamp,
                              json_extract(e.metadata, '$.subject') AS subject
                       FROM annotations a
                       JOIN events e ON a.event_id = e.id
                       WHERE a.facet = ? AND a.value = ?
                       ORDER BY e.timestamp DESC LIMIT 10""",
                    (facet, value),
                ).fetchall()
            elif person_name:
                rows = store.conn.execute(
                    """SELECT DISTINCT a.event_id, a.value, e.source, e.timestamp,
                              json_extract(e.metadata, '$.subject') AS subject
                       FROM annotations a
                       JOIN events e ON a.event_id = e.id
                       JOIN person_events pe ON e.id = pe.event_id
                       JOIN persons p ON pe.person_id = p.person_id
                       WHERE a.facet = ? AND LOWER(p.canonical_name) LIKE ?
                       ORDER BY e.timestamp DESC LIMIT 10""",
                    (facet, f"%{person_name.lower()}%"),
                ).fetchall()
            else:
                rows = store.conn.execute(
                    """SELECT a.event_id, a.value, e.source, e.timestamp,
                              json_extract(e.metadata, '$.subject') AS subject
                       FROM annotations a
                       JOIN events e ON a.event_id = e.id
                       WHERE a.facet = ?
                       ORDER BY e.timestamp DESC LIMIT 10""",
                    (facet,),
                ).fetchall()

            for r in rows:
                dt = datetime.fromtimestamp(r["timestamp"], tz=tz)
                results.append({
                    "event_id": r["event_id"],
                    "facet": facet,
                    "value": r["value"],
                    "source": r["source"],
                    "date": dt.strftime("%Y-%m-%d %H:%M"),
                    "subject": r["subject"] or "",
                    "query_source": f"facet_query:{facet}",
                })

        elif qtype == "cross_source_links":
            person_name = (params.get("person_name") or "").lower()
            if not person_name:
                return []
            rows = store.conn.execute(
                """SELECT id, claim_type, subject, object, confidence
                   FROM claims
                   WHERE (claim_type LIKE 'cross_source_%'
                          OR claim_type = 'calendar_corroboration')
                     AND (LOWER(subject) LIKE ? OR LOWER(object) LIKE ?)
                   ORDER BY confidence DESC LIMIT 10""",
                (f"%{person_name}%", f"%{person_name}%"),
            ).fetchall()
            for r in rows:
                obj = json.loads(r["object"])
                results.append({
                    "claim_id": r["id"],
                    "claim_type": r["claim_type"],
                    "detail": obj,
                    "confidence": r["confidence"],
                    "query_source": f"cross_source_links:{person_name}",
                })

        elif qtype in ("thread_events", "thread_lookup"):
            thread_id = params.get("thread_id", "")
            if not thread_id:
                return []
            rows = store.conn.execute(
                """SELECT id, source, timestamp, metadata, raw_content
                   FROM events
                   WHERE json_extract(metadata, '$.thread_id') = ?
                   ORDER BY timestamp ASC LIMIT 20""",
                (thread_id,),
            ).fetchall()
            for r in rows:
                meta = json.loads(r["metadata"] or "{}")
                dt = datetime.fromtimestamp(r["timestamp"], tz=tz)
                content = r["raw_content"] or ""
                results.append({
                    "id": r["id"],
                    "source": r["source"],
                    "date": dt.strftime("%Y-%m-%d %H:%M"),
                    "subject": meta.get("subject", ""),
                    "snippet": content[:300].replace("\n", " ").strip(),
                    "query_source": f"thread_events:{thread_id}",
                })

    except Exception as e:
        logger.warning("Graph query failed (%s): %s", qtype, e)

    # Deduplicate by ID
    seen_ids: set[str] = set()
    deduped = []
    for r in results:
        rid = r.get("id") or r.get("event_id") or r.get("claim_id", "")
        if rid and rid not in seen_ids:
            seen_ids.add(rid)
            deduped.append(r)
        elif not rid:
            deduped.append(r)
    return deduped


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gemini structured output schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_anticipation_response_schema():
    """Gemini structured output schema for the anticipation engine."""
    from google.genai import types

    query_schema = types.Schema(
        type="OBJECT",
        properties={
            "query_type": types.Schema(type="STRING", enum=[
                "person_emails", "person_messages", "topic_search",
                "recent_from_person", "commitments_search",
                "thread_lookup", "meeting_lookup",
            ]),
            "params": types.Schema(type="OBJECT", properties={}),
            "reason": types.Schema(type="STRING"),
        },
        required=["query_type", "reason"],
    )

    web_search_schema = types.Schema(
        type="OBJECT",
        properties={
            "query": types.Schema(type="STRING"),
            "reason": types.Schema(type="STRING"),
        },
        required=["query", "reason"],
    )

    user_question_schema = types.Schema(
        type="OBJECT",
        properties={
            "event_subject": types.Schema(type="STRING"),
            "question": types.Schema(type="STRING"),
            "category": types.Schema(type="STRING", enum=[
                "LOGISTICS", "DECISIONS", "CONTEXT",
                "MATERIALS", "VERIFICATION",
            ]),
            "confidence": types.Schema(type="STRING"),
            "context_if_answered": types.Schema(type="STRING"),
        },
        required=["event_subject", "question", "category"],
    )

    reassurance_schema = types.Schema(
        type="OBJECT",
        properties={
            "event_subject": types.Schema(type="STRING"),
            "note": types.Schema(type="STRING"),
        },
        required=["event_subject", "note"],
    )

    return types.Schema(
        type="OBJECT",
        properties={
            "system_queries": types.Schema(
                type="ARRAY", items=query_schema,
            ),
            "web_searches": types.Schema(
                type="ARRAY", items=web_search_schema,
            ),
            "user_questions": types.Schema(
                type="ARRAY", items=user_question_schema,
            ),
            "reassurances": types.Schema(
                type="ARRAY", items=reassurance_schema,
            ),
        },
        required=[
            "system_queries", "web_searches",
            "user_questions", "reassurances",
        ],
    )


def _build_candidate_generation_schema():
    """Gemini structured output schema for blind spot candidate generation."""
    from google.genai import types

    return types.Schema(
        type="OBJECT",
        properties={
            "candidates": types.Schema(
                type="ARRAY",
                items=types.Schema(
                    type="OBJECT",
                    properties={
                        "insight": types.Schema(type="STRING"),
                        "evidence": types.Schema(type="STRING"),
                        "novelty": types.Schema(type="INTEGER"),
                        "layer": types.Schema(type="STRING", enum=[
                            "external", "cross_source", "single_source",
                        ]),
                        "actionable": types.Schema(type="BOOLEAN"),
                        "category": types.Schema(type="STRING", enum=[
                            "COLLISION", "WEATHER_TRAFFIC", "CONTRADICTION",
                            "COMMITMENT_LOAD", "SOCIAL_DYNAMICS", "AVOIDANCE",
                            "EXTERNAL_FACT", "REASSURANCE", "OTHER",
                        ]),
                    },
                    required=[
                        "insight", "evidence", "novelty",
                        "layer", "actionable", "category",
                    ],
                ),
            ),
        },
        required=["candidates"],
    )


def _run_anticipation_pass(
    prompt: str,
    llm_client: LLMClient,
    store: LayeredGraphStore,
    model: str = "",
    user_tz: str = USER_TIMEZONE,
    fallback_model: str = "",
) -> dict:
    """Run Pass 1 Anticipation Engine: LLM requests system queries, web searches,
    user questions, and surfaces reassurances.

    Returns: {
        system_queries: [{query_type, params, reason}],
        system_results: [graph query results],
        web_searches: [{query, reason}],
        web_results: [{query, reason, result}],
        user_questions: [{event_subject, question, category, ...}],
        reassurances: [{event_subject, note}],
    }
    """
    logger.info("Pass 1: Anticipation Engine")

    try:
        schema = _build_anticipation_response_schema()
    except ImportError:
        schema = None

    raw_str = llm_client.generate(
        prompt=prompt,
        system=ANTICIPATION_SYSTEM_PROMPT,
        model=model,
        temperature=0.2,
        max_tokens=4096,
        response_schema=schema,
        format_json=schema is None,
    )
    if raw_str is None and fallback_model and fallback_model != model:
        logger.warning("Primary model %s failed, falling back to %s", model, fallback_model)
        raw_str = llm_client.generate(
            prompt=prompt,
            system=ANTICIPATION_SYSTEM_PROMPT,
            model=fallback_model,
            temperature=0.2,
            max_tokens=4096,
            response_schema=schema,
            format_json=schema is None,
        )

    try:
        result = json.loads(raw_str) if raw_str else None
    except (json.JSONDecodeError, TypeError):
        result = None

    if result is None:
        logger.warning("Anticipation pass returned None (LLM failure)")
        print("  \u26a0 Anticipation engine unavailable \u2014 briefing will lack questions/reassurances")
        return {
            "system_queries": [], "system_results": [],
            "web_searches": [], "web_results": [],
            "user_questions": [], "reassurances": [],
            "_raw_llm_output": None,
        }

    # Handle list responses — Gemini sometimes wraps output in an array
    if isinstance(result, list):
        logger.info("Anticipation pass returned list (%d items), unwrapping", len(result))
        if len(result) == 1 and isinstance(result[0], dict):
            result = result[0]
        elif result and all(isinstance(item, dict) for item in result):
            # Merge multiple dicts — combine list values under the same keys
            merged: dict[str, Any] = {}
            for item in result:
                for key, value in item.items():
                    if key in merged and isinstance(merged[key], list) and isinstance(value, list):
                        merged[key].extend(value)
                    elif key not in merged:
                        merged[key] = value
            result = merged
            logger.info("Merged %d list items into dict with keys: %s",
                        len(result), list(result.keys()))
        else:
            logger.warning("Anticipation pass returned non-dict list elements: %s",
                           [type(x).__name__ for x in result[:3]])
            return {
                "system_queries": [], "system_results": [],
                "web_searches": [], "web_results": [],
                "user_questions": [], "reassurances": [],
                "_raw_llm_output": result,
            }

    if not isinstance(result, dict):
        logger.warning("Anticipation pass returned unexpected type: %s", type(result))
        return {
            "system_queries": [], "system_results": [],
            "web_searches": [], "web_results": [],
            "user_questions": [], "reassurances": [],
            "_raw_llm_output": result,
        }

    system_queries = result.get("system_queries", [])[:TRIAGE_MAX_QUERIES]
    web_searches = result.get("web_searches", [])[:TRIAGE_MAX_QUERIES]
    user_questions = result.get("user_questions", [])[:5]
    reassurances = result.get("reassurances", [])

    # Phase 2.6: Execute system graph queries
    system_results: list[dict] = []
    if system_queries:
        logger.info("Phase 2.6: Executing %d system queries", len(system_queries))
        for i, query in enumerate(system_queries):
            qtype = query.get("query_type", "?")
            reason = query.get("reason", "")
            logger.info("  [%d] %s: %s", i + 1, qtype, reason[:80])

            results = _execute_graph_query(query, store, user_tz=user_tz)
            if results:
                logger.info("      -> %d results", len(results))
                system_results.extend(results)
            else:
                logger.info("      -> no results")

    # Phase 2.7: Execute web searches
    web_results: list[dict] = []
    if web_searches:
        logger.info("Phase 2.7: Executing %d web searches", len(web_searches))
        for i, search in enumerate(web_searches):
            query_text = search.get("query", "")
            reason = search.get("reason", "")
            logger.info("  [%d] %s", i + 1, query_text[:80])

            search_result = llm_client.web_search(query=query_text, model=model)
            if search_result:
                logger.info("      -> got result (%d chars)", len(search_result))
                web_results.append({
                    "query": query_text,
                    "reason": reason,
                    "result": search_result,
                })
            else:
                logger.info("      -> no result (web search not supported or failed)")

    # Log user questions and reassurances
    if user_questions:
        logger.info("Phase 2.8: %d user questions generated", len(user_questions))
        for q in user_questions:
            logger.info("  [%s] %s: %s",
                         q.get("category", "?"),
                         q.get("event_subject", "?"),
                         q.get("question", "")[:80])

    if reassurances:
        logger.info("Reassurances: %d", len(reassurances))
        for r in reassurances:
            logger.info("  %s: %s",
                         r.get("event_subject", "?"),
                         r.get("note", "")[:80])

    return {
        "system_queries": system_queries,
        "system_results": system_results,
        "web_searches": web_searches,
        "web_results": web_results,
        "user_questions": user_questions,
        "reassurances": reassurances,
        "_raw_llm_output": result,
    }


def _collect_user_answers(
    questions: list[dict],
    store: LayeredGraphStore | None = None,
) -> list[dict]:
    """Present questions to user via CLI and collect answers.

    Each answer is stored as a user_answer claim if store is provided.

    Returns: [{event_subject, question, category, answer, ...}]
    """
    if not questions:
        return []

    answered: list[dict] = []
    print("\n" + "=" * 60, flush=True)
    print("  Questions for you (press Enter to skip)", flush=True)
    print("=" * 60, flush=True)

    for i, q in enumerate(questions, 1):
        event = q.get("event_subject", "")
        category = q.get("category", "")
        question = q.get("question", "")
        confidence = q.get("confidence", "")

        header = f"[{category}]" if category else ""
        if event:
            header = f"{header} {event}" if header else event
        print(f"\n  {i}. {header}", flush=True)
        print(f"     {question}", flush=True)

        try:
            # Print prompt on its own line and flush so piped readers (the app)
            # see the → marker before we block on stdin.
            print("     →", flush=True)
            sys.stdout.flush()
            # Small delay to ensure the pipe delivers the → marker to the
            # reader (the macOS app) before we block on stdin.readline().
            time.sleep(0.05)

            # Use select() with timeout so a broken pipe doesn't block forever.
            import select
            ready, _, _ = select.select([sys.stdin], [], [], 120)
            if not ready:
                print("  (timed out waiting for answer, skipping)", flush=True)
                continue
            raw = sys.stdin.readline()
            if not raw:
                # EOF — stdin closed (e.g., non-interactive pipe)
                print("  (skipping remaining questions)", flush=True)
                break
            answer = raw.strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  (skipping remaining questions)", flush=True)
            break

        if answer:
            entry = {
                "event_subject": event,
                "question": question,
                "category": category,
                "answer": answer,
                "confidence": confidence,
            }
            answered.append(entry)

            # Persist as a claim for future context
            if store:
                _store_user_answer(store, entry)

    if answered:
        print(f"\n  Recorded {len(answered)} answer(s)")
    print()
    return answered


def _store_user_answer(store: LayeredGraphStore, entry: dict) -> None:
    """Persist a user answer as a claim in the store."""
    import hashlib

    answer_data = json.dumps({
        "question": entry["question"],
        "answer": entry["answer"],
        "category": entry.get("category", ""),
        "event_subject": entry.get("event_subject", ""),
        "confidence": entry.get("confidence", ""),
    }, sort_keys=True)

    claim_id = hashlib.sha256(
        f"user_answer:{entry['question']}:{entry['answer']}".encode()
    ).hexdigest()[:24]

    now = int(time.time())
    try:
        store.conn.execute(
            """INSERT OR REPLACE INTO claims
               (id, claim_type, subject, predicate, object, confidence,
                prompt_version, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                claim_id,
                "user_answer",
                entry.get("event_subject", "briefing"),
                "user_stated",
                answer_data,
                1.0,  # User answers are ground truth
                "briefing_v2",
                now,
            ),
        )
        store.conn.commit()
        logger.info("Stored user answer claim: %s", claim_id[:12])
    except Exception as exc:
        logger.warning("Failed to store user answer: %s", exc)




# (Phase 4 anticipatory questions removed — subsumed by Pass 1 Anticipation Engine)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2.9: Blind Spot Candidate Generation + Ranking
def _generate_blind_spot_candidates(
    prompt: str,
    llm_client: LLMClient,
    model: str = "",
    n_candidates: int = BLIND_SPOT_CANDIDATES,
) -> list[dict]:
    """Phase 2.9a: Generate a wide pool of blind spot candidates.

    Returns a list of candidate dicts with insight, evidence, novelty, layer,
    actionable, category fields.
    """
    logger.info("Phase 2.9a: Generating %d blind spot candidates", n_candidates)

    system = CANDIDATE_GENERATION_PROMPT.format(n_candidates=n_candidates)

    try:
        cand_schema = _build_candidate_generation_schema()
    except ImportError:
        cand_schema = None

    raw_str = llm_client.generate(
        prompt=prompt,
        system=system,
        model=model,
        temperature=0.7,  # High temp for divergent brainstorming; ranking pass filters
        max_tokens=4096,
        response_schema=cand_schema,
        format_json=cand_schema is None,
    )

    try:
        result = json.loads(raw_str) if raw_str else None
    except (json.JSONDecodeError, TypeError):
        result = None

    if result is None:
        logger.warning("Candidate generation returned None")
        return []

    # Handle list wrapping
    if isinstance(result, list):
        if len(result) == 1 and isinstance(result[0], dict):
            result = result[0]
        else:
            logger.warning("Candidate generation returned unexpected list")
            return []

    if not isinstance(result, dict):
        logger.warning("Candidate generation returned %s, expected dict", type(result))
        return []

    candidates = result.get("candidates", [])
    if not isinstance(candidates, list):
        logger.warning("candidates field is %s, expected list", type(candidates))
        return []

    # Validate each candidate has required fields
    valid = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        if not c.get("insight"):
            continue
        # Ensure defaults
        c.setdefault("novelty", 3)
        c.setdefault("layer", "single_source")
        c.setdefault("actionable", True)
        c.setdefault("category", "OTHER")
        c.setdefault("evidence", "")
        valid.append(c)

    logger.info("Generated %d valid candidates (of %d)", len(valid), len(candidates))
    return valid


def _rank_and_dedup_candidates(
    candidates: list[dict],
    n_final: int = BLIND_SPOT_FINAL,
) -> list[dict]:
    """Phase 2.9b: Rank candidates by composite score and deduplicate.

    Scoring formula:
        score = novelty_weight + layer_weight + actionable_bonus - category_penalty

    Where:
        novelty_weight: novelty * 2.0 (most important — we want NON-OBVIOUS)
        layer_weight:   external=3, cross_source=2, single_source=0
        actionable:     +1 if actionable
        category_penalty: -1 for REASSURANCE (still valuable, but not a blind spot)

    Dedup: If two candidates share >60% word overlap in their insight text,
    keep the higher-scoring one.
    """
    if not candidates:
        return []

    logger.info("Phase 2.9b: Ranking %d candidates, selecting top %d",
                len(candidates), n_final)

    # Score each candidate
    layer_scores = {"external": 3.0, "cross_source": 2.0, "single_source": 0.0}
    for c in candidates:
        novelty = min(5, max(1, c.get("novelty", 3)))
        layer = layer_scores.get(c.get("layer", "single_source"), 0.0)
        actionable = 1.0 if c.get("actionable", True) else 0.0
        category_penalty = -1.0 if c.get("category") == "REASSURANCE" else 0.0

        c["_score"] = (novelty * 2.0) + layer + actionable + category_penalty
        c["_novelty_raw"] = novelty

    # Sort by score descending
    candidates.sort(key=lambda c: c["_score"], reverse=True)

    # Log scores for debugging
    for i, c in enumerate(candidates):
        logger.info(
            "  [%d] score=%.1f novelty=%d layer=%s cat=%s: %s",
            i + 1, c["_score"], c["_novelty_raw"],
            c.get("layer", "?"), c.get("category", "?"),
            c.get("insight", "")[:80],
        )

    # Dedup by word overlap
    def word_set(text: str) -> set[str]:
        return set(text.lower().split())

    deduped: list[dict] = []
    for c in candidates:
        c_words = word_set(c.get("insight", ""))
        if not c_words:
            continue

        is_dup = False
        for existing in deduped:
            existing_words = word_set(existing.get("insight", ""))
            if not existing_words:
                continue
            overlap = len(c_words & existing_words) / min(len(c_words), len(existing_words))
            if overlap > 0.6:
                is_dup = True
                break

        if not is_dup:
            deduped.append(c)

    logger.info("After dedup: %d candidates (removed %d dupes)",
                len(deduped), len(candidates) - len(deduped))

    # Ensure category diversity: at least 1 from each valuable category if available
    final = deduped[:n_final]
    remaining = deduped[n_final:]

    # Check if we're missing key categories
    final_cats = {c.get("category") for c in final}
    priority_cats = {"WEATHER_TRAFFIC", "EXTERNAL_FACT"}
    missing = priority_cats - final_cats

    if missing and remaining:
        for cat in missing:
            for i, c in enumerate(remaining):
                if c.get("category") == cat:
                    # Swap in: replace the lowest-scoring item in final
                    final[-1] = c
                    remaining.pop(i)
                    logger.info("  Diversity swap: added %s candidate", cat)
                    break

    # Clean up internal scoring fields
    for c in final:
        c.pop("_score", None)
        c.pop("_novelty_raw", None)

    logger.info("Final blind spots: %d", len(final))
    return final


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: LLM synthesis
def _get_logistics_facts(store: LayeredGraphStore) -> list[dict]:
    """Query logistics claims from the store."""
    rows = store.conn.execute(
        """SELECT id, subject, predicate, object, confidence
           FROM claims
           WHERE claim_type = 'logistics'
             AND superseded_by IS NULL
           ORDER BY confidence DESC"""
    ).fetchall()

    results = []
    for r in rows:
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue
        obj["_claim_id"] = r["id"]
        obj["_thread_id"] = r["subject"]
        results.append(obj)
    return results


def _get_relational_context(store: LayeredGraphStore) -> list[dict]:
    """Query relational context claims from the store."""
    rows = store.conn.execute(
        """SELECT id, subject, predicate, object, confidence
           FROM claims
           WHERE claim_type = 'relational_context'
             AND superseded_by IS NULL
           ORDER BY confidence DESC"""
    ).fetchall()

    results = []
    for r in rows:
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue
        obj["_claim_id"] = r["id"]
        obj["_thread_id"] = r["subject"]
        results.append(obj)
    return results


def _build_briefing_prompt(
    events: list[CalendarEvent],
    event_commitments: dict[str, list[Commitment]],
    event_contexts: dict[str, list[dict]],
    orphaned: list[Commitment],
    orphan_snippets: dict[str, str],
    user_tz: str,
    profile: dict,
    event_facets: dict[str, dict[str, list[str]]] | None = None,
    cross_source_links: dict[str, list[dict]] | None = None,
    triage_assessment: str = "",
    additional_context: list[dict] | None = None,
    recently_resolved: list[dict] | None = None,
    logistics_facts: list[dict] | None = None,
    relational_context: list[dict] | None = None,
    web_results: list[dict] | None = None,
    user_answers: list[dict] | None = None,
    reassurances: list[dict] | None = None,
    blind_spots: list[dict] | None = None,
) -> str:
    """Build the user prompt with all gathered data.

    Optional enrichment params (None = skip those sections):
        event_facets: {event_id: {facet: [values]}} — annotations per event
        cross_source_links: {event_id: [related_event_dicts]} — cross-source links
        triage_assessment: LLM analyst's assessment of context gaps
        additional_context: Extra items retrieved by agentic triage queries
        recently_resolved: [{what, who, to_whom, type, status}] — closed items
        logistics_facts: Structured logistics facts from multi-gate extraction
        relational_context: Person/relationship context from multi-gate extraction
    """
    import zoneinfo

    tz = safe_timezone(user_tz)
    now = datetime.now(tz)
    parts = []

    # Header
    parts.append(f"TODAY: {now.strftime('%A, %B %d, %Y')}")
    if profile.get("name"):
        parts.append(f"USER: {profile['name']}")
    if profile.get("role"):
        parts.append(f"ROLE: {profile['role']}")
    if profile.get("context"):
        parts.append(f"CONTEXT: {profile['context']}")

    # User profile details (family, home, local knowledge)
    if profile.get("home"):
        home = profile["home"]
        loc_parts = [home.get("neighborhood", ""), home.get("city", ""), home.get("state", "")]
        parts.append(f"HOME: {', '.join(p for p in loc_parts if p)}")
    if profile.get("family"):
        fam = profile["family"]
        fam_parts = []
        if isinstance(fam, list):
            # Simple list of family member names
            fam_parts = [str(name) for name in fam]
        elif isinstance(fam, dict):
            if fam.get("spouse"):
                fam_parts.append(f"Spouse: {fam['spouse']}")
            for child in fam.get("children", []):
                school = child.get("school") or child.get("daycare") or ""
                school_str = f" ({school})" if school else ""
                fam_parts.append(f"Child: {child['name']}{school_str}")
            for cp in fam.get("care_providers", []):
                fam_parts.append(f"Care: {cp['name']} — {cp.get('role', '')}")
        if fam_parts:
            parts.append(f"FAMILY: {'; '.join(fam_parts)}")
    if profile.get("local_knowledge"):
        parts.append("LOCAL KNOWLEDGE:")
        for item in profile["local_knowledge"]:
            parts.append(f"  - {item}")

    parts.append("")

    # Section 1: Calendar events
    parts.append("=" * 60)
    parts.append("SECTION 1: CALENDAR EVENTS")
    parts.append("=" * 60)

    if not events:
        parts.append("No upcoming calendar events found.")
        parts.append("")
        parts.append("NOTE: Since there are no calendar events, the user's week is")
        parts.append("structured around their open commitments and deadlines below.")
        parts.append("Focus the briefing on what's due, what's overdue, and what")
        parts.append("needs attention based on communications and commitments.")
    else:
        for event in events:
            parts.append("")
            parts.append(f"EVENT: {event.title}")
            parts.append(f"  When: {event.format_time(user_tz)} ({event.duration_minutes}min)")
            if event.location:
                parts.append(f"  Where: {event.location}")
            if event.attendees:
                names = [a["name"] for a in event.attendees if a["name"]]
                parts.append(f"  With: {', '.join(names)}")
            if event.description:
                parts.append(f"  Description: {event.description[:500]}")

            # Cross-source annotations (facets)
            if event_facets:
                facets = event_facets.get(event.event_id, {})
                if facets:
                    facet_parts = []
                    if "dollar_amount" in facets:
                        amounts = ", ".join(f"${v}" for v in facets["dollar_amount"])
                        facet_parts.append(f"Dollar amounts: {amounts}")
                    if "person_mention" in facets:
                        mentions = ", ".join(facets["person_mention"][:8])
                        facet_parts.append(f"People mentioned: {mentions}")
                    if "time_sensitive" in facets:
                        facet_parts.append("Time-sensitive: yes")
                    if facet_parts:
                        parts.append(f"  Annotations: {'; '.join(facet_parts)}")

            # Cross-source links (events from other sources sharing signals)
            if cross_source_links:
                links = cross_source_links.get(event.event_id, [])
                if links:
                    parts.append(f"  Cross-source links ({len(links)} related events):")
                    for link in links[:5]:
                        subj = link.get("subject", "")
                        subj_str = f": {subj}" if subj else ""
                        parts.append(
                            f"    [{link['source']}] {link['relationship']} "
                            f"\"{link['shared_signal']}\"{subj_str}"
                        )

            # Recent comms with attendees
            ctx = event_contexts.get(event.event_id, [])
            if ctx:
                parts.append(f"  Recent communications ({len(ctx)} items):")
                for item in ctx[:8]:
                    direction = "→ sent" if item["is_from_me"] else "← received"
                    dt = datetime.fromtimestamp(item["timestamp"], tz=tz)
                    date_str = dt.strftime("%b %d")
                    parts.append(
                        f"    [{item['source']}] {date_str} {direction} "
                        f"({item['person_name']}): "
                        f"{item['subject'] or item['snippet'][:80]}"
                    )

            # Matched commitments
            matched = event_commitments.get(event.event_id, [])
            if matched:
                parts.append(f"  Open commitments with these people ({len(matched)}):")
                for c in matched:
                    dl = f" (due {c.deadline})" if c.deadline else ""
                    overdue = " [OVERDUE]" if c.is_overdue else ""
                    parts.append(
                        f"    - [{c.type}] {c.what} "
                        f"(who: {c.who}, to: {c.to_whom}){dl}{overdue}"
                    )

            parts.append("-" * 40)

    # Section 2: Orphaned commitments
    parts.append("")
    parts.append("=" * 60)
    parts.append("SECTION 2: UNSCHEDULED COMMITMENTS")
    parts.append("(open items from communications NOT tied to any upcoming calendar event)")
    parts.append("=" * 60)

    if not orphaned:
        parts.append("No unscheduled commitments found.")
    else:
        # Group by temporal urgency with finer granularity
        overdue_last_week = [
            c for c in orphaned
            if c.is_overdue and c.deadline_in_days is not None
            and abs(c.deadline_in_days) <= 7
        ]
        overdue_older = [
            c for c in orphaned
            if c.is_overdue and (
                c.deadline_in_days is None or abs(c.deadline_in_days) > 7
            )
        ]
        due_this_week = [
            c for c in orphaned
            if c.deadline and not c.is_overdue
            and c.deadline_in_days is not None and c.deadline_in_days <= 7
        ]
        due_next_week = [
            c for c in orphaned
            if c.deadline and not c.is_overdue
            and c.deadline_in_days is not None
            and 7 < c.deadline_in_days <= 14
        ]
        no_deadline = [c for c in orphaned if not c.deadline]
        future = [
            c for c in orphaned
            if c.deadline and not c.is_overdue
            and (c.deadline_in_days is None or c.deadline_in_days > 14)
        ]

        for label, group in [
            ("OVERDUE FROM LAST WEEK — needs immediate attention", overdue_last_week),
            ("OVERDUE (older)", overdue_older),
            ("DUE THIS WEEK", due_this_week),
            ("DUE NEXT WEEK", due_next_week),
            ("NO DEADLINE (recent promises)", no_deadline),
            ("FUTURE DEADLINES (>2 weeks)", future),
        ]:
            if group:
                parts.append(f"\n  {label} ({len(group)} items):")
                for c in group:
                    dl = f" (due {c.deadline})" if c.deadline else ""
                    snippet = orphan_snippets.get(c.claim_id, "")
                    source = f"\n      Source: {snippet}" if snippet else ""

                    # Staleness tag for overdue items
                    staleness = ""
                    if c.is_overdue and c.deadline_in_days is not None:
                        days_overdue = abs(c.deadline_in_days)
                        staleness = f" [{days_overdue} days overdue]"

                    # Direction tag (from synthesis)
                    direction_tag = ""
                    if c.direction == "group_ask":
                        to_whom = c.to_whom or "group chat"
                        direction_tag = f" [GROUP ASK — may not be user's responsibility: {to_whom}]"
                    elif c.direction == "self_directed":
                        direction_tag = " [SELF-DIRECTED]"

                    # Staleness signal tag (from synthesis)
                    signal_tag = ""
                    if c.staleness_signal == "overdue_no_followup":
                        signal_tag = " [POSSIBLY RESOLVED — overdue with no follow-up]"
                    elif c.staleness_signal == "group_broadcast":
                        signal_tag = " [GROUP BROADCAST]"
                    elif c.staleness_signal == "old_thread":
                        signal_tag = " [OLD THREAD]"

                    parts.append(
                        f"    P{c.priority} [{c.type}] {c.what}"
                        f"\n      Who: {c.who} → To: {c.to_whom}{dl}"
                        f"{staleness}{direction_tag}{signal_tag}{source}"
                    )

    # Section 2b: Logistics facts (from multi-gate)
    if logistics_facts:
        parts.append("")
        parts.append("=" * 60)
        parts.append("LOGISTICS INTELLIGENCE")
        parts.append("(extracted from reservations, travel, care provider, appointment threads)")
        parts.append("=" * 60)

        by_type: dict[str, list[dict]] = {}
        for fact in logistics_facts:
            ft = fact.get("type", "other")
            by_type.setdefault(ft, []).append(fact)

        for fact_type, facts in by_type.items():
            label = fact_type.replace("_", " ").title()
            parts.append(f"\n  {label}:")
            for f in facts:
                if fact_type == "reservation":
                    parts.append(
                        f"    - {f.get('venue', '?')}: {f.get('date', '?')} "
                        f"at {f.get('time', '?')} (party of {f.get('party_size', '?')})"
                    )
                elif fact_type == "travel":
                    parts.append(
                        f"    - {f.get('destination', '?')}: {f.get('date', '?')} "
                        f"via {f.get('airline', '?')} "
                        f"[{f.get('confirmation', 'no conf')}]"
                    )
                elif fact_type == "care_provider":
                    parts.append(
                        f"    - {f.get('provider', '?')}: {f.get('date', '?')} "
                        f"{f.get('hours', '')} ({f.get('rate', '?')})"
                    )
                elif fact_type == "appointment":
                    parts.append(
                        f"    - {f.get('provider', '?')}: {f.get('date', '?')} "
                        f"at {f.get('location', '?')}"
                    )
                elif fact_type == "childcare":
                    parts.append(
                        f"    - {f.get('child', '?')} at {f.get('facility', '?')}: "
                        f"{f.get('date', '?')} pickup={f.get('pickup_time', '?')}"
                    )
                else:
                    parts.append(f"    - {f.get('summary', json.dumps(f)[:120])}")

    # Section 2c: Relational context (from multi-gate)
    if relational_context:
        parts.append("")
        parts.append("=" * 60)
        parts.append("RELATIONAL INTELLIGENCE")
        parts.append("(roles, relationships, competitive context from message threads)")
        parts.append("=" * 60)

        for person in relational_context:
            name = person.get("name", "?")
            role = person.get("role", "unknown")
            org = person.get("organization", "")
            ctx = person.get("context", "")
            org_str = f" ({org})" if org else ""
            parts.append(f"  - {name}: {role}{org_str}")
            if ctx:
                parts.append(f"    Context: {ctx}")

    # Section 3: Additional context from triage
    if triage_assessment or additional_context:
        parts.append("")
        parts.append("=" * 60)
        parts.append("SECTION 3: ADDITIONAL CONTEXT (from intelligence analyst)")
        parts.append("=" * 60)

        if triage_assessment:
            parts.append(f"\nAnalyst assessment: {triage_assessment}")

        if additional_context:
            parts.append(f"\nAdditional retrieved items ({len(additional_context)}):")
            for item in additional_context:
                qs = item.get("query_source", "")
                if "commitment_search" in qs:
                    parts.append(
                        f"  [{qs}] P{item.get('priority', 3)} [{item.get('type', '')}] "
                        f"{item.get('what', '')} (who: {item.get('who', '')} "
                        f"→ {item.get('to_whom', '')})"
                    )
                elif "facet_query" in qs:
                    parts.append(
                        f"  [{qs}] [{item.get('source', '')}] {item.get('date', '')} "
                        f"{item.get('facet', '')}={item.get('value', '')} "
                        f"| {item.get('subject', '')}"
                    )
                elif "cross_source_links" in qs:
                    detail = item.get("detail", {})
                    parts.append(
                        f"  [{qs}] {item.get('claim_type', '')} "
                        f"conf={item.get('confidence', 0):.2f} | "
                        f"{json.dumps(detail)[:200]}"
                    )
                else:
                    snippet = item.get("snippet", item.get("subject", ""))
                    parts.append(
                        f"  [{qs}] [{item.get('source', '')}] {item.get('date', '')} "
                        f"| {snippet[:120]}"
                    )

    # Section 4: Recently resolved commitments
    if recently_resolved:
        parts.append("")
        parts.append("=" * 60)
        parts.append("SECTION 4: RECENTLY RESOLVED COMMITMENTS")
        parts.append("(closed in the last 7 days — show these to build trust)")
        parts.append("=" * 60)

        for item in recently_resolved[:10]:
            what = item.get("what", "")
            who = item.get("who", "")
            to_whom = item.get("to_whom", "")
            status = item.get("status", "done")
            parts.append(f"  [{status}] {what} (who: {who} → to: {to_whom})")

    # Section 5: Web search results (from Pass 1 Anticipation Engine)
    if web_results:
        parts.append("")
        parts.append("=" * 60)
        parts.append("SECTION 5: WEB SEARCH RESULTS")
        parts.append("(external facts retrieved by the Anticipation Engine)")
        parts.append("=" * 60)

        for wr in web_results:
            parts.append(f"\n  Query: {wr.get('query', '')}")
            parts.append(f"  Reason: {wr.get('reason', '')}")
            result_text = wr.get("result", "")
            if result_text:
                # Truncate very long search results
                if len(result_text) > 1000:
                    result_text = result_text[:1000] + "..."
                parts.append(f"  Result: {result_text}")

    # Section 6: User-provided answers (highest trust)
    if user_answers:
        parts.append("")
        parts.append("=" * 60)
        parts.append("SECTION 6: USER-PROVIDED ANSWERS")
        parts.append("(ground truth — override all other sources)")
        parts.append("=" * 60)

        for ua in user_answers:
            event = ua.get("event_subject", "")
            cat = ua.get("category", "")
            header = f"[{cat}] {event}" if cat else event
            parts.append(f"\n  {header}")
            parts.append(f"  Q: {ua.get('question', '')}")
            parts.append(f"  A: {ua.get('answer', '')}")

    # Section 7: Reassurances (things confirmed as handled)
    if reassurances:
        parts.append("")
        parts.append("=" * 60)
        parts.append("SECTION 7: REASSURANCES")
        parts.append("(things the user might worry about that are already handled)")
        parts.append("=" * 60)

        for r in reassurances:
            event = r.get("event_subject", "")
            note = r.get("note", "")
            if event:
                parts.append(f"  [{event}] {note}")
            else:
                parts.append(f"  {note}")

    # Section 8: Pre-ranked blind spots (from generate+rank pass)
    if blind_spots:
        parts.append("")
        parts.append("=" * 60)
        parts.append("SECTION 8: PRE-RANKED BLIND SPOTS")
        parts.append("(generated by a separate analysis pass, ranked by novelty)")
        parts.append("You MUST include ALL of these in 'What You Might Not Be")
        parts.append("Thinking About'. Rewrite for clarity but do NOT drop any.")
        parts.append("=" * 60)

        for i, bs in enumerate(blind_spots, 1):
            parts.append(f"\n  [{i}] {bs.get('category', 'OTHER')}"
                         f" (novelty={bs.get('novelty', 3)}/5,"
                         f" layer={bs.get('layer', '?')})")
            parts.append(f"  Insight: {bs.get('insight', '')}")
            parts.append(f"  Evidence: {bs.get('evidence', '')}")

    return "\n".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_briefing(
    store: LayeredGraphStore,
    llm_client: LLMClient,
    days_ahead: int = BRIEFING_DAYS_AHEAD,
    user_tz: str = USER_TIMEZONE,
    model: str = "",
    profile: dict | None = None,
    skip_anticipation: bool = False,
    interactive: bool = True,
    thinking_level: str | None = None,
) -> dict[str, Any]:
    """Run the v2 briefing pipeline.

    Pipeline:
        Phase 1:    Gather (SQL) — calendar, commitments, context
        Phase 2:    Context + cross-source enrichment (SQL)
        Phase 2.5:  Pass 1 — Anticipation Engine (system queries + web + user Qs)
        Phase 2.6:  Execute system graph queries
        Phase 2.7:  Execute web searches
        Phase 2.8:  Present user questions, collect answers
        Phase 3:    Pass 2 — Briefing synthesis (with all gathered context)

    Args:
        store: LayeredGraphStore with ingested + triaged + extracted data
        llm_client: LLM client for synthesis
        days_ahead: How many days of calendar to look ahead
        user_tz: User's timezone
        model: LLM model for synthesis
        profile: User profile dict. Loaded from profile.yaml if not provided.
        skip_anticipation: If True, skip the Anticipation Engine (Phase 2.5+).
        interactive: If True, present user questions via CLI.
            If False, skip user questions (non-interactive mode).
        thinking_level: Reasoning level for Gemini 3 models.

    Returns:
        {briefing, events_count, commitments, anticipation, prompt_length,
         elapsed_s, raw}
    """
    t0 = time.time()
    if profile is None:
        profile = _load_profile()

    # ── Phase 1: Gather ──
    logger.info("Phase 1: Gathering data")

    events = _get_upcoming_events(store, days_ahead=days_ahead, user_tz=user_tz)
    logger.info("Calendar events (next %d days): %d", days_ahead, len(events))

    all_commitments = _get_open_commitments(store)
    logger.info("Open commitments: %d", len(all_commitments))

    overdue_count = sum(1 for c in all_commitments if c.is_overdue)
    if overdue_count:
        logger.info("  %d overdue", overdue_count)

    event_commitments, orphaned = _match_commitments_to_events(
        events, all_commitments,
    )
    matched_count = len(all_commitments) - len(orphaned)
    logger.info(
        "Matched to calendar: %d  |  Unscheduled: %d",
        matched_count, len(orphaned),
    )

    # ── Phase 2: Context ──
    logger.info("Phase 2: Gathering context")

    event_contexts: dict[str, list[dict]] = {}
    for event in events:
        ctx = _gather_event_context(store, event, lookback_days=BRIEFING_LOOKBACK_DAYS)
        event_contexts[event.event_id] = ctx

    orphan_snippets: dict[str, str] = {}
    for c in orphaned[:50]:
        snippet = _get_commitment_source_snippet(store, c)
        if snippet:
            orphan_snippets[c.claim_id] = snippet

    recently_resolved = _get_recently_resolved(store, lookback_days=7)
    logger.info("Recently resolved: %d", len(recently_resolved))

    logistics_facts = _get_logistics_facts(store)
    relational_context = _get_relational_context(store)
    logger.info(
        "Multi-gate data: %d logistics facts, %d relational contexts",
        len(logistics_facts), len(relational_context),
    )

    # ── Phase 2b: Cross-source enrichment ──
    logger.info("Phase 2b: Cross-source enrichment")

    ef: dict[str, dict[str, list[str]]] = {}
    csl: dict[str, list[dict]] = {}
    for event in events:
        facets = _get_event_facets(store, event.event_id)
        if facets:
            ef[event.event_id] = facets
        links = _get_cross_source_links(store, event.event_id)
        if links:
            csl[event.event_id] = links

    facet_count = sum(sum(len(v) for v in f.values()) for f in ef.values())
    link_count = sum(len(v) for v in csl.values())
    logger.info("  Annotations: %d facets across %d events", facet_count, len(ef))
    logger.info("  Cross-source links: %d links across %d events", link_count, len(csl))

    # ── Phase 2.5–2.8: Anticipation Engine ──
    anticipation_result: dict = {
        "system_queries": [], "system_results": [],
        "web_searches": [], "web_results": [],
        "user_questions": [], "reassurances": [],
    }
    user_answers: list[dict] = []
    base_prompt = ""

    # Run anticipation even without calendar — deadline-based commitments
    # (overdue, due this week, due next week) serve as temporal anchors
    has_temporal_anchor = bool(events) or any(
        c.deadline for c in all_commitments
    )

    if not skip_anticipation and has_temporal_anchor:
        # Build base prompt for anticipation engine to review
        base_prompt = _build_briefing_prompt(
            events=events,
            event_commitments=event_commitments,
            event_contexts=event_contexts,
            orphaned=orphaned,
            orphan_snippets=orphan_snippets,
            user_tz=user_tz,
            profile=profile,
            event_facets=ef,
            cross_source_links=csl,
            recently_resolved=recently_resolved,
            logistics_facts=logistics_facts or None,
            relational_context=relational_context or None,
        )

        anticipation_result = _run_anticipation_pass(
            prompt=base_prompt,
            llm_client=llm_client,
            store=store,
            model=model,
            user_tz=user_tz,
            fallback_model=CLOUD_FAST_MODEL,
        )

        # Phase 2.8: User questions (if interactive)
        if interactive and anticipation_result["user_questions"]:
            user_answers = _collect_user_answers(
                anticipation_result["user_questions"],
                store=store,
            )

    # ── Phase 2.9: Blind spot candidate generation + ranking ──
    ranked_blind_spots: list[dict] = []
    all_candidates: list[dict] = []
    if not skip_anticipation and has_temporal_anchor:
        # Build enriched prompt with all anticipation results for candidate gen
        candidate_prompt = _build_briefing_prompt(
            events=events,
            event_commitments=event_commitments,
            event_contexts=event_contexts,
            orphaned=orphaned,
            orphan_snippets=orphan_snippets,
            user_tz=user_tz,
            profile=profile,
            event_facets=ef,
            cross_source_links=csl,
            additional_context=(
                anticipation_result["system_results"]
                if anticipation_result["system_results"]
                else None
            ),
            recently_resolved=recently_resolved,
            logistics_facts=logistics_facts or None,
            relational_context=relational_context or None,
            web_results=anticipation_result["web_results"] or None,
            user_answers=user_answers or None,
            reassurances=anticipation_result["reassurances"] or None,
        )

        all_candidates = _generate_blind_spot_candidates(
            prompt=candidate_prompt,
            llm_client=llm_client,
            model=model,
            n_candidates=BLIND_SPOT_CANDIDATES,
        )

        if all_candidates:
            ranked_blind_spots = _rank_and_dedup_candidates(
                all_candidates, n_final=BLIND_SPOT_FINAL,
            )

    # ── Phase 3: Briefing synthesis ──
    logger.info("Phase 3: Synthesizing briefing")

    prompt = _build_briefing_prompt(
        events=events,
        event_commitments=event_commitments,
        event_contexts=event_contexts,
        orphaned=orphaned,
        orphan_snippets=orphan_snippets,
        user_tz=user_tz,
        profile=profile,
        event_facets=ef,
        cross_source_links=csl,
        additional_context=(
            anticipation_result["system_results"]
            if anticipation_result["system_results"]
            else None
        ),
        recently_resolved=recently_resolved,
        logistics_facts=logistics_facts or None,
        relational_context=relational_context or None,
        web_results=anticipation_result["web_results"] or None,
        user_answers=user_answers or None,
        reassurances=anticipation_result["reassurances"] or None,
        blind_spots=ranked_blind_spots or None,
    )

    logger.info("Prompt: %d chars", len(prompt))

    # Enable google_search grounding as fallback for anything Pass 1 missed
    briefing_md = llm_client.generate(
        prompt=prompt,
        system=BRIEFING_SYSTEM_PROMPT,
        model=model,
        temperature=0.3,
        max_tokens=8192,
        thinking_level=thinking_level,
        google_search=True,
    )

    if not briefing_md:
        logger.warning("LLM returned empty response for briefing")
        briefing_md = "Briefing generation failed — no response from LLM."

    elapsed = time.time() - t0
    logger.info("Briefing ready (%.1fs)", elapsed)

    return {
        "briefing": briefing_md,
        "events_count": len(events),
        "commitments": {
            "total_open": len(all_commitments),
            "matched_to_calendar": matched_count,
            "unscheduled": len(orphaned),
            "overdue": overdue_count,
            "recently_resolved": len(recently_resolved),
        },
        "anticipation": {
            "system_queries": len(anticipation_result["system_queries"]),
            "system_results": len(anticipation_result["system_results"]),
            "web_searches": len(anticipation_result["web_searches"]),
            "web_results": len(anticipation_result["web_results"]),
            "user_questions": len(anticipation_result["user_questions"]),
            "user_answers": len(user_answers),
            "reassurances": len(anticipation_result["reassurances"]),
            "blind_spot_candidates": len(all_candidates),
            "blind_spot_final": len(ranked_blind_spots),
        },
        "prompt_length": len(prompt),
        "elapsed_s": round(elapsed, 2),
        # Raw I/O for analysis
        "raw": {
            "anticipation_system_prompt": ANTICIPATION_SYSTEM_PROMPT,
            "anticipation_input_prompt": base_prompt,
            "anticipation_output": anticipation_result,
            "user_answers": user_answers,
            "blind_spot_candidates": all_candidates,
            "blind_spot_ranked": ranked_blind_spots,
            "briefing_system_prompt": BRIEFING_SYSTEM_PROMPT,
            "briefing_input_prompt": prompt,
            "briefing_output": briefing_md,
            "profile": profile,
            "model": model,
        },
    }
