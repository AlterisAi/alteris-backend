"""Onboarding tools for the MCP server.

Provides a rich KG summary to power the onboarding chat experience.
The onboarding chat shows new users what Alteris discovered about their
digital life and helps them configure their Clarity Queue.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from alteris.mcp_tools import ToolDef, ToolParam, register_tool
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)


def handle_alteris_onboarding_summary(
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
    """Get people the user interacts with most, ranked by relationship quality.

    Uses person_profiles (from Stage 1 claims) to rank by a composite score
    that favors bidirectional, multi-channel, real human contacts over
    automated senders, newsletters, and spam:

    - Bidirectional communication (user_initiated_ratio near 0.5) scores highest
    - Multi-channel presence (email + WhatsApp + calendar) boosts rank
    - Pure inbound-only contacts (ratio ≈ 0.0) are heavily penalized
    - Single-channel, high-volume contacts (newsletters, banks) rank lower
    """
    try:
        # Use person_profiles which has pre-computed engagement metrics.
        # Only include contacts that have a resolved name — contacts known
        # only by phone number or hash aren't meaningful "top contacts".
        rows = store.conn.execute("""
            SELECT pp.person_id, pp.canonical_name, pp.message_count,
                   pp.direct_count, pp.group_count,
                   pp.user_initiated_ratio, pp.channel_count, pp.channels,
                   pp.days_since_last, pp.tier, pp.is_user
            FROM person_profiles pp
            WHERE pp.is_user = 0
              AND pp.message_count >= 5
              AND pp.canonical_name IS NOT NULL
              AND pp.canonical_name <> ''
            ORDER BY pp.message_count DESC
            LIMIT 200
        """).fetchall()
    except Exception:
        return []

    scored = []
    for r in rows:
        name = (r["canonical_name"] or "").strip()
        # Skip contacts whose "name" is just a phone number or email
        if not name or name.startswith("+") or name.startswith("("):
            continue

        msg_count = r["message_count"] or 0
        direct_count = r["direct_count"] or 0
        group_count = r["group_count"] or 0
        ratio = r["user_initiated_ratio"]
        if ratio is None:
            ratio = 0.0
        channel_count = r["channel_count"] or 1
        channels_raw = r["channels"] or "[]"

        try:
            channels = json.loads(channels_raw) if isinstance(channels_raw, str) else channels_raw
        except (json.JSONDecodeError, TypeError):
            channels = []

        # Bidirectionality score: peaks at 0.5 (balanced), drops toward 0 or 1
        # ratio=0.0 means pure inbound (newsletters, automated) → score 0.0
        # ratio=0.5 means balanced conversation → score 1.0
        # ratio=1.0 means user always initiates → score 0.5 (still a real contact)
        if ratio < 0.01:
            bidir_score = 0.0  # Pure inbound — automated sender
        elif ratio > 0.99:
            bidir_score = 0.3  # Pure outbound — user always initiates
        else:
            bidir_score = 1.0 - abs(ratio - 0.5) * 2  # Peaks at 0.5

        # Multi-channel bonus: each additional channel adds weight
        channel_score = min(channel_count / 3.0, 1.0)  # 3+ channels = max

        # Volume score: use effective count (direct + 5% group) so group-only
        # contacts don't rank above real 1:1 conversation partners
        effective_count = direct_count + int(group_count * 0.05)
        volume_score = math.log10(max(effective_count, 1)) / 5.0  # log10(100K)≈5

        # Dampen bidir score for low-volume contacts — a perfect 0.5 ratio
        # on 5 messages is noise, not signal (e.g. replying to one recruiter email)
        if effective_count < 10:
            bidir_score *= 0.3
        elif effective_count < 20:
            bidir_score *= 0.6

        # Composite: bidirectionality is the primary signal
        # Volume without bidirectionality is noise (Chase, USPS)
        composite = (
            bidir_score * 0.45
            + channel_score * 0.35
            + volume_score * 0.20
        )

        scored.append({
            "person_id": r["person_id"],
            "name": name,
            "event_count": msg_count,
            "sources": channels if isinstance(channels, list) else [],
            "cross_source": channel_count > 1,
            "_score": composite,
        })

    # Sort by composite score descending, take top N
    scored.sort(key=lambda x: x["_score"], reverse=True)
    result = scored[:limit]
    # Remove internal score from output
    for r in result:
        del r["_score"]
    return result


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
    from alteris.constants import CQ_DEFAULT_CATEGORIES

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
    name="alteris_onboarding_summary",
    description="Get a rich Knowledge Graph summary for the onboarding chat: stats, top contacts, commitments, beliefs, calendar events, suggested sender rules, and suggested categories.",
    permission="read",
    handler=handle_alteris_onboarding_summary,
))
