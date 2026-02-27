"""Knowledge Graph read tools for the MCP server.

All tools in this module are tagged 'read' — safe for user agents.
They query the LayeredGraphStore and return JSON-serializable dicts.
"""

from __future__ import annotations

import json
import logging
import time

from alteris.mcp_tools import ToolDef, ToolParam, register_tool
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool implementations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def handle_alteris_stats(store: LayeredGraphStore, **kwargs) -> dict:
    """Return database statistics."""
    return store.stats()


def handle_alteris_query_events(
    store: LayeredGraphStore,
    source: str | None = None,
    since: int | None = None,
    until: int | None = None,
    event_type: str | None = None,
    limit: int = 100,
    **kwargs,
) -> dict:
    """Query events with flexible filters."""
    events = store.get_events(
        since=since or 0,
        until=until or 0,
        source=source,
        event_type=event_type,
        limit=limit,
    )
    return {
        "count": len(events),
        "events": [_event_to_dict(e) for e in events],
    }


def handle_alteris_query_beliefs(
    store: LayeredGraphStore,
    subject: str | None = None,
    belief_type: str | None = None,
    status: str = "active",
    min_confidence: float = 0.0,
    limit: int = 100,
    **kwargs,
) -> dict:
    """Query beliefs with flexible filters."""
    beliefs = store.get_beliefs(
        subject=subject,
        belief_type=belief_type,
        status=status,
        min_confidence=min_confidence,
        limit=limit,
    )
    return {
        "count": len(beliefs),
        "beliefs": [_belief_to_dict(b) for b in beliefs],
    }


def handle_alteris_query_persons(
    store: LayeredGraphStore,
    limit: int = 200,
    **kwargs,
) -> dict:
    """List all resolved persons."""
    persons = store.get_all_persons()
    result = []
    for p in persons[:limit]:
        identifiers = store.get_person_identifiers(p["person_id"])
        result.append({
            "person_id": p["person_id"],
            "canonical_name": p["canonical_name"],
            "is_user": bool(p.get("is_user", 0)),
            "sources": json.loads(p.get("sources", "[]")) if isinstance(p.get("sources"), str) else p.get("sources", []),
            "identifiers": [
                {"type": i["identifier_type"], "value": i["identifier"], "display_name": i.get("display_name", "")}
                for i in identifiers
            ],
        })
    return {"count": len(result), "persons": result}


def handle_alteris_query_commitments(
    store: LayeredGraphStore,
    overdue: bool = False,
    person: str | None = None,
    limit: int = 50,
    **kwargs,
) -> dict:
    """Query active commitment claims."""
    claims = store.get_active_claims(claim_type="commitment", limit=500)

    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    results = []
    for c in claims:
        try:
            obj = json.loads(c.object) if isinstance(c.object, str) else c.object
        except (json.JSONDecodeError, TypeError):
            continue

        if obj.get("status") != "open":
            continue

        deadline = obj.get("deadline")
        is_overdue = bool(deadline and deadline < today)
        if overdue and not is_overdue:
            continue

        if person:
            pf = person.lower()
            who = (obj.get("who") or "").lower()
            to_whom = (obj.get("to_whom") or "").lower()
            if pf not in who and pf not in to_whom:
                continue

        results.append({
            "id": c.id,
            "what": obj.get("what", ""),
            "who": obj.get("who", ""),
            "to_whom": obj.get("to_whom", ""),
            "type": obj.get("type", ""),
            "deadline": deadline,
            "priority": obj.get("priority", 3),
            "direction": obj.get("direction", ""),
            "confidence": c.confidence,
            "overdue": is_overdue,
            "evidence_quote": obj.get("evidence_quote", ""),
        })
        if len(results) >= limit:
            break

    return {"count": len(results), "commitments": results}


def handle_alteris_person_detail(
    store: LayeredGraphStore,
    person_id: str = "",
    **kwargs,
) -> dict:
    """Get detailed info about a person: profile, identifiers, recent events, beliefs."""
    if not person_id:
        return {"error": "person_id is required"}

    person = store.get_person(person_id)
    if not person:
        return {"error": f"Person {person_id} not found"}

    identifiers = store.get_person_identifiers(person_id)
    recent_events = store.get_events_for_person(person_id, limit=20)

    # Find beliefs mentioning this person
    beliefs = store.get_beliefs(subject=person_id, limit=20)
    # Also check by canonical name
    name = person.get("canonical_name", "")
    if name:
        name_beliefs = store.get_beliefs(subject=name, limit=20)
        seen = {b.id for b in beliefs}
        for b in name_beliefs:
            if b.id not in seen:
                beliefs.append(b)

    return {
        "person_id": person_id,
        "canonical_name": name,
        "is_user": bool(person.get("is_user", 0)),
        "sources": json.loads(person.get("sources", "[]")) if isinstance(person.get("sources"), str) else person.get("sources", []),
        "identifiers": [
            {"type": i["identifier_type"], "value": i["identifier"]}
            for i in identifiers
        ],
        "recent_events": [_event_to_dict(e) for e in recent_events],
        "beliefs": [_belief_to_dict(b) for b in beliefs],
    }


def handle_alteris_search(
    store: LayeredGraphStore,
    query: str = "",
    sources: list[str] | None = None,
    limit: int = 50,
    **kwargs,
) -> dict:
    """Full-text search over event raw_content."""
    if not query:
        return {"error": "query is required"}

    # SQLite LIKE-based search (no FTS5 for now)
    conditions = ["raw_content LIKE ?"]
    params: list = [f"%{query}%"]

    if sources:
        placeholders = ",".join("?" * len(sources))
        conditions.append(f"source IN ({placeholders})")
        params.extend(sources)

    params.append(limit)
    where = " AND ".join(conditions)

    rows = store.conn.execute(
        f"SELECT * FROM events WHERE {where} ORDER BY timestamp DESC LIMIT ?",
        params,
    ).fetchall()

    from alteris.privacy import SensitivityLevel
    from alteris.models import Event

    events = []
    for r in rows:
        events.append({
            "id": r["id"],
            "source": r["source"],
            "event_type": r["event_type"],
            "timestamp": r["timestamp"],
            "preview": (r["raw_content"] or "")[:200],
            "metadata": json.loads(r["metadata"] or "{}"),
        })

    return {"count": len(events), "query": query, "events": events}


def handle_alteris_get_briefing(
    store: LayeredGraphStore,
    days: int = 7,
    lookback: int = 30,
    **kwargs,
) -> dict:
    """Get the most recent briefing or generate a new one."""
    from pathlib import Path
    from alteris.constants import ALTERIS_DIR

    # Check for saved briefings
    briefing_dir = ALTERIS_DIR / "briefings"
    if briefing_dir.exists():
        md_files = sorted(briefing_dir.glob("briefing_*.md"), reverse=True)
        if md_files:
            latest = md_files[0]
            return {
                "briefing": latest.read_text(),
                "source": "cached",
                "file": str(latest),
                "generated_at": int(latest.stat().st_mtime),
            }

    return {
        "briefing": "",
        "source": "none",
        "message": "No briefings found. Run the pipeline first.",
    }


def handle_alteris_get_spend(
    store: LayeredGraphStore,
    days: int = 7,
    **kwargs,
) -> dict:
    """Get API spend summary."""
    try:
        from alteris.spend import get_daily_spend, get_spend_summary, check_onboarding_budget
        summary = get_spend_summary(store, days=days)
        today = get_daily_spend(store)
        onboarding = check_onboarding_budget(store)
        return {
            "today": today,
            "onboarding": onboarding,
            "history": summary,
        }
    except ImportError:
        return {"error": "spend module not yet available"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _event_to_dict(event) -> dict:
    """Convert an Event to a JSON-serializable dict."""
    meta = event.metadata if isinstance(event.metadata, dict) else {}
    return {
        "id": event.id,
        "source": event.source,
        "source_id": event.source_id,
        "event_type": event.event_type,
        "timestamp": event.timestamp,
        "participants": list(event.participants),
        "preview": (event.raw_content or "")[:300],
        "metadata": meta,
        "sensitivity": event.sensitivity.value,
    }


def _belief_to_dict(belief) -> dict:
    """Convert a Belief to a JSON-serializable dict."""
    return {
        "id": belief.id,
        "belief_type": belief.belief_type.value,
        "subject": belief.subject,
        "summary": belief.summary,
        "data": belief.data,
        "epistemic_level": belief.epistemic_level.value,
        "confidence": belief.confidence,
        "status": belief.status.value,
        "source_claims": belief.source_claims,
        "created_at": belief.created_at,
        "updated_at": belief.updated_at,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

register_tool(ToolDef(
    name="alteris_stats",
    description="Get database statistics: event counts by source, claim counts, belief counts, person counts, sync state.",
    permission="read",
    handler=handle_alteris_stats,
))

register_tool(ToolDef(
    name="alteris_query_events",
    description="Query events from the knowledge graph with optional filters for source, time range, event type, and limit.",
    permission="read",
    params=[
        ToolParam("source", "string", "Filter by source (mail, imessage, whatsapp, calendar, etc.)"),
        ToolParam("since", "integer", "Unix timestamp: only events after this time"),
        ToolParam("until", "integer", "Unix timestamp: only events before this time"),
        ToolParam("event_type", "string", "Filter by event type (message, email, calendar_event, etc.)"),
        ToolParam("limit", "integer", "Max results (default 100)", default=100),
    ],
    handler=handle_alteris_query_events,
))

register_tool(ToolDef(
    name="alteris_query_beliefs",
    description="Query synthesized beliefs. Beliefs are high-level inferences from claims: entities, relations, facts, observations.",
    permission="read",
    params=[
        ToolParam("subject", "string", "Filter beliefs about a specific subject (person name, topic)"),
        ToolParam("belief_type", "string", "Filter by type: entity, relation, fact, observation", enum=["entity", "relation", "fact", "observation"]),
        ToolParam("status", "string", "Filter by status (default: active)", default="active", enum=["active", "resolved", "stale", "retracted", "superseded"]),
        ToolParam("min_confidence", "number", "Minimum confidence score (0.0-1.0)", default=0.0),
        ToolParam("limit", "integer", "Max results (default 100)", default=100),
    ],
    handler=handle_alteris_query_beliefs,
))

register_tool(ToolDef(
    name="alteris_query_persons",
    description="List all resolved persons in the knowledge graph, with their identifiers (email, phone, etc.).",
    permission="read",
    params=[
        ToolParam("limit", "integer", "Max results (default 200)", default=200),
    ],
    handler=handle_alteris_query_persons,
))

register_tool(ToolDef(
    name="alteris_query_commitments",
    description="Query active commitments (things the user or others committed to do). Includes deadlines, priorities, and overdue status.",
    permission="read",
    params=[
        ToolParam("overdue", "boolean", "Only show overdue commitments", default=False),
        ToolParam("person", "string", "Filter by person name"),
        ToolParam("limit", "integer", "Max results (default 50)", default=50),
    ],
    handler=handle_alteris_query_commitments,
))

register_tool(ToolDef(
    name="alteris_person_detail",
    description="Get detailed information about a specific person: their identifiers, recent events, and beliefs about them.",
    permission="read",
    params=[
        ToolParam("person_id", "string", "The person_id to look up", required=True),
    ],
    handler=handle_alteris_person_detail,
))

register_tool(ToolDef(
    name="alteris_search",
    description="Full-text search over event content. Searches email bodies, message text, meeting notes, etc.",
    permission="read",
    params=[
        ToolParam("query", "string", "Search query", required=True),
        ToolParam("sources", "array", "Filter by sources (e.g. ['mail', 'imessage'])"),
        ToolParam("limit", "integer", "Max results (default 50)", default=50),
    ],
    handler=handle_alteris_search,
))

register_tool(ToolDef(
    name="alteris_get_briefing",
    description="Get the most recent blind spot briefing. Returns cached briefing if available.",
    permission="read",
    params=[
        ToolParam("days", "integer", "Days ahead to cover (default 7)", default=7),
    ],
    handler=handle_alteris_get_briefing,
))

register_tool(ToolDef(
    name="alteris_get_spend",
    description="Get API spend summary including daily totals, onboarding budget status, and historical usage.",
    permission="read",
    params=[
        ToolParam("days", "integer", "Days of history to include (default 7)", default=7),
    ],
    handler=handle_alteris_get_spend,
))
