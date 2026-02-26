"""Onboarding tools for the MCP server.

Provides a rich KG summary to power the onboarding chat experience.
The onboarding chat shows new users what Loom discovered about their
digital life and helps them configure their Clarity Queue.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from loom.mcp_tools import ToolDef, ToolParam, register_tool
from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)


def handle_loom_onboarding_summary(
    store: LayeredGraphStore,
    **kwargs,
) -> dict:
    """Return a rich KG summary for the onboarding chat.

    Aggregates stats, top contacts, commitments, beliefs, calendar events,
    and suggests sender rules + categories based on the KG data.
    """
    now = int(time.time())
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 1. Basic stats
    stats = store.stats()

    # 2. Top contacts — people with most events, showing cross-source presence
    top_contacts = _get_top_contacts(store, limit=10)

    # 3. Open commitments summary
    commitments = _get_commitment_summary(store, today)

    # 4. Active beliefs (high confidence entities + facts)
    notable_beliefs = _get_notable_beliefs(store, limit=10)

    # 5. Upcoming calendar events (next 7 days)
    upcoming_events = _get_upcoming_events(store, now, days=7)

    # 6. Source coverage — which channels are connected and how much data
    source_coverage = _get_source_coverage(stats)

    # 7. Suggested sender rules based on email patterns
    suggested_rules = _suggest_sender_rules(store)

    # 8. Suggested categories based on belief topics
    suggested_categories = _suggest_categories(store)

    # 9. Interesting cross-source discoveries
    cross_source = _get_cross_source_discoveries(store)

    return {
        "stats": {
            "total_events": stats.get("events_count", 0),
            "total_persons": stats.get("persons_count", 0),
            "total_claims": stats.get("active_claims", 0),
            "total_beliefs": stats.get("beliefs_count", 0),
            "events_by_source": stats.get("events_by_source", {}),
        },
        "top_contacts": top_contacts,
        "commitments": commitments,
        "notable_beliefs": notable_beliefs,
        "upcoming_events": upcoming_events,
        "source_coverage": source_coverage,
        "suggested_sender_rules": suggested_rules,
        "suggested_categories": suggested_categories,
        "cross_source_discoveries": cross_source,
    }


def _get_top_contacts(store: LayeredGraphStore, limit: int = 10) -> list[dict]:
    """Get people the user interacts with most, with cross-source breakdown."""
    try:
        rows = store.conn.execute("""
            SELECT ep.person_id, p.canonical_name, COUNT(*) as event_count,
                   GROUP_CONCAT(DISTINCT e.source) as sources
            FROM event_persons ep
            JOIN persons p ON p.person_id = ep.person_id
            JOIN events e ON e.id = ep.event_id
            WHERE p.is_user = 0
            GROUP BY ep.person_id
            ORDER BY event_count DESC
            LIMIT ?
        """, (limit,)).fetchall()
    except Exception:
        return []

    contacts = []
    for r in rows:
        sources = (r["sources"] or "").split(",")
        contacts.append({
            "person_id": r["person_id"],
            "name": r["canonical_name"] or r["person_id"],
            "event_count": r["event_count"],
            "sources": [s.strip() for s in sources if s.strip()],
            "cross_source": len(set(s.strip() for s in sources if s.strip())) > 1,
        })
    return contacts


def _get_commitment_summary(store: LayeredGraphStore, today: str) -> dict:
    """Summarize open commitments: total, overdue, high priority items."""
    claims = store.get_active_claims(claim_type="commitment", limit=500)

    open_items = []
    overdue_count = 0

    for c in claims:
        try:
            obj = json.loads(c.object) if isinstance(c.object, str) else c.object
        except (json.JSONDecodeError, TypeError):
            continue

        if obj.get("status") != "open":
            continue

        deadline = obj.get("deadline")
        is_overdue = bool(deadline and deadline < today)
        if is_overdue:
            overdue_count += 1

        open_items.append({
            "what": obj.get("what", ""),
            "who": obj.get("who", ""),
            "to_whom": obj.get("to_whom", ""),
            "deadline": deadline,
            "priority": obj.get("priority", 3),
            "direction": obj.get("direction", ""),
            "overdue": is_overdue,
            "confidence": c.confidence,
        })

    # Sort by priority then deadline
    open_items.sort(key=lambda x: (x.get("priority", 3), x.get("deadline") or "9999"))

    return {
        "total_open": len(open_items),
        "overdue_count": overdue_count,
        "high_priority": [i for i in open_items if i.get("priority", 3) <= 2][:5],
        "sample": open_items[:8],
    }


def _get_notable_beliefs(store: LayeredGraphStore, limit: int = 10) -> list[dict]:
    """Get high-confidence beliefs that would be interesting to show a new user."""
    beliefs = store.get_beliefs(status="active", min_confidence=0.6, limit=50)

    # Prefer entities and facts over observations
    type_order = {"entity": 0, "fact": 1, "relation": 2, "observation": 3}
    beliefs.sort(key=lambda b: (type_order.get(b.belief_type.value, 4), -b.confidence))

    results = []
    for b in beliefs[:limit]:
        results.append({
            "type": b.belief_type.value,
            "subject": b.subject,
            "summary": b.summary,
            "confidence": b.confidence,
        })
    return results


def _get_upcoming_events(store: LayeredGraphStore, now: int, days: int = 7) -> list[dict]:
    """Get upcoming calendar events with attendee information."""
    end = now + (days * 86400)
    events = store.get_events(since=now, until=end, source="calendar", limit=30)

    results = []
    for e in events:
        meta = e.metadata if isinstance(e.metadata, dict) else {}
        results.append({
            "title": meta.get("title") or meta.get("subject") or (e.raw_content or "")[:80],
            "timestamp": e.timestamp,
            "location": meta.get("location", ""),
            "duration_minutes": meta.get("duration_minutes"),
            "attendees": list(e.participants)[:5],
        })
    return results


def _get_source_coverage(stats: dict) -> list[dict]:
    """Describe what each connected source contributed."""
    by_source = stats.get("events_by_source", {})
    source_labels = {
        "mail": ("Mail", "envelope.fill", "Emails from Apple Mail"),
        "imessage": ("iMessage", "message.fill", "Text messages"),
        "whatsapp": ("WhatsApp", "phone.fill", "WhatsApp messages"),
        "calendar": ("Calendar", "calendar", "Calendar events"),
        "contacts": ("Contacts", "person.2.fill", "Contact cards"),
        "granola": ("Granola", "waveform", "Meeting transcripts"),
        "slack": ("Slack", "number", "Slack messages"),
    }

    coverage = []
    for source, count in sorted(by_source.items(), key=lambda x: -x[1]):
        label, icon, desc = source_labels.get(source, (source.title(), "doc", source))
        coverage.append({
            "source": source,
            "label": label,
            "icon": icon,
            "description": desc,
            "event_count": count,
        })
    return coverage


def _suggest_sender_rules(store: LayeredGraphStore) -> list[dict]:
    """Suggest P1/P2 sender rules based on interaction frequency."""
    try:
        # Find most-emailed senders (by event count)
        rows = store.conn.execute("""
            SELECT json_extract(metadata, '$.sender') as sender, COUNT(*) as cnt
            FROM events
            WHERE source IN ('mail', 'imessage', 'whatsapp')
              AND json_extract(metadata, '$.sender') IS NOT NULL
              AND COALESCE(json_extract(metadata, '$.is_from_me'), 0) != 1
            GROUP BY sender
            ORDER BY cnt DESC
            LIMIT 15
        """).fetchall()
    except Exception:
        return []

    suggestions = []
    for i, r in enumerate(rows):
        sender = r["sender"]
        if not sender:
            continue
        priority = "P1" if i < 5 else "P2"
        suggestions.append({
            "pattern": sender,
            "priority": priority,
            "event_count": r["cnt"],
            "reason": f"{'Top' if i < 5 else 'Frequent'} contact: {r['cnt']} messages",
        })
    return suggestions[:10]


def _suggest_categories(store: LayeredGraphStore) -> list[dict]:
    """Suggest CQ categories based on belief domains and commitment types."""
    from loom.constants import CQ_DEFAULT_CATEGORIES

    # Check what domains appear in beliefs
    beliefs = store.get_beliefs(status="active", limit=200)

    domain_counts: Counter = Counter()
    for b in beliefs:
        data = b.data if isinstance(b.data, dict) else {}
        domain = data.get("domain", "")
        if domain:
            domain_counts[domain] += 1
        # Also check subject for topic hints
        subject = b.subject.lower()
        for cat in CQ_DEFAULT_CATEGORIES:
            if cat in subject:
                domain_counts[cat] += 1

    # Merge with defaults
    suggestions = []
    for cat in CQ_DEFAULT_CATEGORIES:
        count = domain_counts.get(cat, 0)
        icons = {
            "work": "briefcase", "personal": "person", "finance": "dollarsign.circle",
            "health": "heart", "home": "house", "errands": "cart",
            "waiting-on": "hourglass", "someday": "star",
        }
        suggestions.append({
            "name": cat,
            "icon": icons.get(cat, "tag"),
            "kg_evidence_count": count,
            "reason": f"Found {count} related beliefs" if count > 0 else "Common category",
        })

    # Sort by evidence count (items with KG backing first)
    suggestions.sort(key=lambda x: -x["kg_evidence_count"])
    return suggestions


def _get_cross_source_discoveries(store: LayeredGraphStore) -> list[dict]:
    """Find interesting cross-source patterns (people appearing in multiple channels)."""
    try:
        rows = store.conn.execute("""
            SELECT p.canonical_name, p.person_id,
                   GROUP_CONCAT(DISTINCT e.source) as sources,
                   COUNT(DISTINCT e.source) as source_count,
                   COUNT(*) as total_events
            FROM event_persons ep
            JOIN persons p ON p.person_id = ep.person_id
            JOIN events e ON e.id = ep.event_id
            WHERE p.is_user = 0
            GROUP BY ep.person_id
            HAVING source_count > 1
            ORDER BY source_count DESC, total_events DESC
            LIMIT 5
        """).fetchall()
    except Exception:
        return []

    discoveries = []
    for r in rows:
        sources = [s.strip() for s in (r["sources"] or "").split(",") if s.strip()]
        source_labels = {"mail": "email", "imessage": "iMessage", "whatsapp": "WhatsApp",
                         "calendar": "calendar", "slack": "Slack", "granola": "meetings"}
        labeled = [source_labels.get(s, s) for s in sources]
        discoveries.append({
            "person": r["canonical_name"] or r["person_id"],
            "sources": sources,
            "total_events": r["total_events"],
            "description": f"Appears across {', '.join(labeled)} ({r['total_events']} events)",
        })
    return discoveries


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

register_tool(ToolDef(
    name="loom_onboarding_summary",
    description="Get a rich Knowledge Graph summary for the onboarding chat: stats, top contacts, commitments, beliefs, calendar events, suggested sender rules, and suggested categories.",
    permission="read",
    handler=handle_loom_onboarding_summary,
))
