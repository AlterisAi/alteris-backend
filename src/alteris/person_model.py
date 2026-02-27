"""Person Model — 11-dimension estimated user profile.

Builds a rich model of the user from all available knowledge graph data.
Three phases:
  Phase 1 (Scout): 11 deterministic SQL scout functions + Apple Intelligence scouts
  Phase 2 (Surveyor): Pro model synthesizes scout data into structured JSON
  Phase 3 (Save): Persist to DB with version tracking

The model is re-estimable: user_corrections are preserved across re-estimation.

Usage:
    from alteris.person_model import estimate_person_model, get_person_model
    model = estimate_person_model(store, llm)
    latest = get_person_model(store)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from alteris.constants import (
    PERSON_MODEL_LOW_CONFIDENCE_THRESHOLD,
    PERSON_MODEL_MAX_SCOUT_RESULTS,
    PERSON_MODEL_SURVEYOR_MODEL,
)
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)

_7D = 7 * 86400
_30D = 30 * 86400
_90D = 90 * 86400


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Empty model template
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _empty_model() -> dict:
    """Return the 11-dimension empty person model template."""
    return {
        "identity": {
            "name": "", "timezone": "", "location": "",
            "emails": [], "phones": [],
            "confidence": 0.0, "sources": [],
        },
        "professional": {
            "role": "", "organization": "", "industry": "",
            "responsibilities": [], "active_projects": [],
            "confidence": 0.0, "sources": [],
        },
        "communication_fingerprint": {
            "primary_channels": [],
            "response_time_hours": {},
            "peak_hours": [], "quiet_hours": [],
            "message_volume_weekly": 0,
            "formality_level": "",
            "confidence": 0.0, "sources": [],
        },
        "relationship_map": {
            "inner_circle": [],
            "professional_contacts": [],
            "organizations": [],
            "relationship_dynamics": [],
            "confidence": 0.0, "sources": [],
        },
        "active_workstreams": {
            "threads": [],
            "open_commitments_inbound": 0,
            "open_commitments_outbound": 0,
            "overdue_count": 0,
            "confidence": 0.0, "sources": [],
        },
        "goals_and_values": {
            "inferred_goals": [],
            "inferred_values": [],
            "priorities": [],
            "confidence": 0.0, "sources": [],
        },
        "life_architecture": {
            "family": [],
            "dependents": [],
            "routines": [],
            "non_negotiable_blocks": [],
            "confidence": 0.0, "sources": [],
        },
        "temporal_patterns": {
            "work_hours": {},
            "busiest_day": "",
            "quietest_day": "",
            "meeting_load_weekly": 0,
            "avg_meetings_per_day": 0.0,
            "confidence": 0.0, "sources": [],
        },
        "domain_expertise": {
            "topics_discussed": [],
            "tools_used": [],
            "technical_depth": "",
            "confidence": 0.0, "sources": [],
        },
        "financial_landscape": {
            "recurring_obligations": [],
            "financial_contacts": [],
            "confidence": 0.0, "sources": [],
        },
        "visibility_gaps": {
            "low_visibility_domains": [],
            "uninstrumented_channels": [],
            "confidence": 0.0, "sources": [],
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def estimate_person_model(
    store: LayeredGraphStore,
    llm=None,
    force: bool = False,
) -> dict:
    """Estimate the 11-dimension person model from all available data.

    Phase 1: Run all scout functions (deterministic SQL)
    Phase 2: Surveyor LLM synthesizes into structured model
    Phase 3: Save to DB, preserving user_corrections

    Returns the full model dict.
    """
    # Load existing user corrections (preserved across re-estimation)
    existing = get_person_model(store)
    user_corrections = {}
    prev_version = 0
    if existing:
        user_corrections = existing.get("user_corrections", {})
        prev_version = existing.get("version", 0)
        if not force and existing.get("estimated_at"):
            age = int(time.time()) - existing["estimated_at"]
            if age < 3600:  # Don't re-estimate within 1 hour
                logger.info("Person model estimated %ds ago, skipping (use --force to override)", age)
                return existing.get("model", _empty_model())

    # Phase 1: Scout
    logger.info("Phase 1: Running scouts...")
    scout_data = phase1_scout_all(store)

    # Phase 2: Surveyor
    if llm is not None:
        logger.info("Phase 2: Surveyor synthesis...")
        model = phase2_surveyor(llm, scout_data, user_corrections)
    else:
        logger.info("Phase 2: No LLM — using scout-only model")
        model = phase2_scout_only(scout_data, user_corrections)

    # Phase 3: Save
    new_version = prev_version + 1
    method = "scout" if llm is None else "llm"
    model_id = phase3_save(store, model, user_corrections, new_version, method)
    logger.info("Saved person model v%d (id=%d, method=%s)", new_version, model_id, method)

    return model


def get_person_model(store: LayeredGraphStore) -> dict | None:
    """Return the latest person model from the DB, or None."""
    row = store.conn.execute(
        "SELECT * FROM person_model ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["model"] = json.loads(d["model_json"])
    d["user_corrections"] = json.loads(d.get("user_corrections") or "{}")
    return d


def update_person_model_field(
    store: LayeredGraphStore,
    dimension: str,
    field: str,
    value,
) -> dict:
    """Update a specific field in user_corrections (persisted across re-estimation)."""
    existing = get_person_model(store)
    if not existing:
        return {"error": "No person model exists. Run estimate first."}

    corrections = existing.get("user_corrections", {})
    if dimension not in corrections:
        corrections[dimension] = {}
    corrections[dimension][field] = value

    store.conn.execute(
        "UPDATE person_model SET user_corrections = ? WHERE id = ?",
        (json.dumps(corrections), existing["id"]),
    )
    store.conn.commit()

    # Apply correction to model
    model = existing["model"]
    if dimension in model and field in model[dimension]:
        model[dimension][field] = value

    return {"status": "ok", "dimension": dimension, "field": field, "value": value}


def get_model_gaps(model: dict) -> list[dict]:
    """Return dimensions with confidence below threshold."""
    gaps = []
    for dim_name, dim_data in model.items():
        if isinstance(dim_data, dict) and "confidence" in dim_data:
            conf = dim_data["confidence"]
            if conf < PERSON_MODEL_LOW_CONFIDENCE_THRESHOLD:
                gaps.append({
                    "dimension": dim_name,
                    "confidence": conf,
                    "sources": dim_data.get("sources", []),
                })
    gaps.sort(key=lambda g: g["confidence"])
    return gaps


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Scout functions (all deterministic SQL)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_user_names(conn) -> list[str]:
    """Load user's canonical name and display names for filtering beliefs."""
    names: list[str] = []
    user_row = conn.execute(
        "SELECT person_id, canonical_name FROM persons WHERE is_user = 1 LIMIT 1"
    ).fetchone()
    if user_row:
        if user_row["canonical_name"]:
            names.append(user_row["canonical_name"])
        # Also grab display names from identifiers
        idents = conn.execute(
            "SELECT DISTINCT display_name FROM person_identifiers WHERE person_id = ? AND display_name != ''",
            (user_row["person_id"],),
        ).fetchall()
        for row in idents:
            if row["display_name"] and row["display_name"] not in names:
                names.append(row["display_name"])
    return names


def phase1_scout_all(store: LayeredGraphStore) -> dict:
    """Run all 11 scout functions + Apple Intelligence scouts."""
    conn = store.conn
    now = int(time.time())

    # Load user names for filtering relationship/family beliefs to user-only
    user_names = _load_user_names(conn)

    return {
        "identity": _scout_identity(conn, now),
        "professional": _scout_professional(conn, now),
        "communication_fingerprint": _scout_communication_fingerprint(conn, now),
        "relationship_map": _scout_relationship_map(conn, now, user_names=user_names),
        "active_workstreams": _scout_active_workstreams(conn, now),
        "goals_and_values": _scout_goals_and_values(conn, now),
        "life_architecture": _scout_life_architecture(conn, now, user_names=user_names),
        "temporal_patterns": _scout_temporal_patterns(conn, now),
        "domain_expertise": _scout_domain_expertise(conn, now),
        "financial_landscape": _scout_financial_landscape(conn, now),
        "visibility_gaps": _scout_visibility_gaps(conn, now),
        "apple_mail_categories": _scout_apple_mail_categories(conn, now),
        "apple_intelligence_signals": _scout_apple_intelligence_signals(conn, now),
    }


def _scout_identity(conn, now: int) -> dict:
    """Pull user identity from config + persons table + identifiers."""
    result: dict = {"emails": [], "phones": [], "name": "", "timezone": "", "location": ""}

    # User person
    user_row = conn.execute(
        "SELECT person_id, canonical_name FROM persons WHERE is_user = 1 LIMIT 1"
    ).fetchone()
    if user_row:
        result["name"] = user_row["canonical_name"]
        result["person_id"] = user_row["person_id"]

        # All user identifiers
        idents = conn.execute(
            "SELECT identifier_type, identifier FROM person_identifiers WHERE person_id = ?",
            (user_row["person_id"],),
        ).fetchall()
        for row in idents:
            if row["identifier_type"] == "email":
                result["emails"].append(row["identifier"])
            elif row["identifier_type"] == "phone":
                result["phones"].append(row["identifier"])

    # Try loading from profile.yaml / config.json
    config = _load_user_config_safe()
    if config:
        if not result["name"] and config.get("name"):
            result["name"] = config["name"]
        for e in config.get("emails", []):
            if e not in result["emails"]:
                result["emails"].append(e)
        for p in config.get("phones", []):
            if p not in result["phones"]:
                result["phones"].append(p)
        if config.get("timezone"):
            result["timezone"] = config["timezone"]
        if config.get("location"):
            result["location"] = config["location"]

    # Infer timezone from constants
    if not result["timezone"]:
        from alteris.constants import USER_TIMEZONE
        result["timezone"] = USER_TIMEZONE

    return result


def _scout_professional(conn, now: int) -> dict:
    """Infer professional role from calendar patterns, beliefs, Granola transcripts."""
    result: dict = {"recurring_meetings": [], "org_signals": [], "role_signals": [], "sources": []}

    # Recurring calendar events (standup, 1:1, retro → team structure)
    recurring = conn.execute("""
        SELECT metadata->>'subject' as subject, COUNT(*) as cnt
        FROM events
        WHERE event_type = 'calendar_event'
          AND timestamp > ?
          AND metadata->>'subject' IS NOT NULL
        GROUP BY LOWER(metadata->>'subject')
        HAVING cnt >= 3
        ORDER BY cnt DESC
        LIMIT ?
    """, (now - _90D, PERSON_MODEL_MAX_SCOUT_RESULTS)).fetchall()
    result["recurring_meetings"] = [
        {"subject": r["subject"], "count": r["cnt"]} for r in recurring
    ]
    if recurring:
        result["sources"].append("calendar")

    # Entity beliefs about organizations
    org_beliefs = conn.execute("""
        SELECT summary, confidence, json_extract(data, '$.context') as context
        FROM beliefs
        WHERE status = 'active'
          AND (belief_type = 'entity' OR belief_type = 'relation')
          AND (LOWER(summary) LIKE '%organization%' OR LOWER(summary) LIKE '%company%'
               OR LOWER(summary) LIKE '%team%' OR LOWER(summary) LIKE '%role%')
        ORDER BY confidence DESC
        LIMIT 10
    """).fetchall()
    result["org_signals"] = [dict(b) for b in org_beliefs]
    if org_beliefs:
        result["sources"].append("beliefs")

    # Granola transcripts: role-revealing context
    granola_events = conn.execute("""
        SELECT SUBSTR(raw_content, 1, 500) as body, metadata->>'subject' as subject
        FROM events
        WHERE source = 'granola' AND timestamp > ?
        ORDER BY timestamp DESC
        LIMIT 10
    """, (now - _30D,)).fetchall()
    if granola_events:
        result["role_signals"] = [
            {"subject": g["subject"], "body_preview": g["body"]}
            for g in granola_events
        ]
        result["sources"].append("granola")

    return result


def _scout_communication_fingerprint(conn, now: int) -> dict:
    """Hourly histogram, channel breakdown, response times."""
    result: dict = {"hourly_histogram": {}, "channel_volumes": {}, "total_7d": 0, "sources": []}

    # Hourly histogram (all sources, 30d)
    hourly = conn.execute("""
        SELECT CAST(strftime('%H', timestamp, 'unixepoch', 'localtime') AS INTEGER) as hour,
               COUNT(*) as cnt
        FROM events
        WHERE timestamp > ?
        GROUP BY hour
        ORDER BY hour
    """, (now - _30D,)).fetchall()
    result["hourly_histogram"] = {r["hour"]: r["cnt"] for r in hourly}
    if hourly:
        result["sources"].append("events")

    # Channel volumes (by source, 30d)
    channels = conn.execute("""
        SELECT source, COUNT(*) as cnt
        FROM events
        WHERE timestamp > ?
        GROUP BY source
        ORDER BY cnt DESC
    """, (now - _30D,)).fetchall()
    result["channel_volumes"] = {r["source"]: r["cnt"] for r in channels}

    # 7-day message volume
    vol7 = conn.execute(
        "SELECT COUNT(*) as cnt FROM events WHERE timestamp > ?",
        (now - _7D,),
    ).fetchone()
    result["total_7d"] = vol7["cnt"] if vol7 else 0

    # Person profiles channel breakdown
    profiles = conn.execute("""
        SELECT channels, message_count
        FROM person_profiles
        WHERE is_user = 1
        LIMIT 1
    """).fetchall()
    if profiles:
        result["user_profile_channels"] = profiles[0]["channels"]
        result["sources"].append("person_profiles")

    return result


def _scout_relationship_map(conn, now: int, user_names: list[str] | None = None) -> dict:
    """Inner circle, professional contacts, co-occurrence."""
    result: dict = {"inner_circle": [], "frequent_contacts": [], "co_occurrence": [], "sources": []}

    # Tier 1-2 from person_profiles
    inner = conn.execute("""
        SELECT pp.person_id, pp.canonical_name, pp.tier, pp.message_count,
               pp.channels, pp.user_initiated_ratio
        FROM person_profiles pp
        WHERE pp.tier <= 2 AND pp.is_user = 0
        ORDER BY pp.message_count DESC
        LIMIT ?
    """, (PERSON_MODEL_MAX_SCOUT_RESULTS,)).fetchall()
    result["inner_circle"] = [
        {
            "name": r["canonical_name"],
            "tier": r["tier"],
            "messages": r["message_count"],
            "channels": r["channels"],
        }
        for r in inner
    ]
    if inner:
        result["sources"].append("person_profiles")

    # Top contacts by event volume (30d)
    freq = conn.execute("""
        SELECT p.canonical_name, COUNT(DISTINCT pe.event_id) as event_count
        FROM person_events pe
        JOIN persons p ON pe.person_id = p.person_id
        JOIN events e ON pe.event_id = e.id
        WHERE p.is_user = 0 AND e.timestamp > ?
        GROUP BY pe.person_id
        ORDER BY event_count DESC
        LIMIT 20
    """, (now - _30D,)).fetchall()
    result["frequent_contacts"] = [
        {"name": r["canonical_name"], "events_30d": r["event_count"]}
        for r in freq
    ]
    if freq:
        result["sources"].append("events")

    # Relation beliefs — scope to user's relationships to avoid leaking
    # third-party relations extracted from conversations
    if user_names:
        name_clauses = " OR ".join("LOWER(summary) LIKE ?" for _ in user_names)
        params = [f"%{n.lower()}%" for n in user_names]
        relations = conn.execute(f"""
            SELECT summary, confidence, json_extract(data, '$.context') as context
            FROM beliefs
            WHERE status = 'active' AND belief_type = 'relation'
              AND ({name_clauses})
            ORDER BY confidence DESC
            LIMIT 15
        """, params).fetchall()
    else:
        relations = []
    result["relation_beliefs"] = [dict(r) for r in relations]
    if relations:
        result["sources"].append("beliefs")

    return result


def _scout_active_workstreams(conn, now: int) -> dict:
    """Open commitments, CQ tasks, active threads."""
    result: dict = {"commitments_inbound": 0, "commitments_outbound": 0, "overdue": 0,
                    "active_threads": [], "cq_tasks": [], "sources": []}

    # Open commitments by direction
    commits = conn.execute("""
        SELECT json_extract(object, '$.direction') as direction,
               json_extract(object, '$.deadline') as deadline,
               json_extract(object, '$.what') as what,
               json_extract(object, '$.status') as status
        FROM claims
        WHERE claim_type = 'commitment'
          AND (superseded_by IS NULL OR superseded_by = '')
          AND json_extract(object, '$.status') = 'open'
    """).fetchall()
    for c in commits:
        d = c["direction"] or "unknown"
        if d == "inbound":
            result["commitments_inbound"] += 1
        elif d == "outbound":
            result["commitments_outbound"] += 1
        if c["deadline"]:
            try:
                dl = datetime.fromisoformat(c["deadline"].replace("Z", "+00:00"))
                if dl.timestamp() < now:
                    result["overdue"] += 1
            except (ValueError, AttributeError):
                pass
    if commits:
        result["sources"].append("claims")

    # CQ tasks (immediate bucket)
    cq = conn.execute("""
        SELECT title, bucket, due_date
        FROM cq_tasks
        WHERE done = 0
        ORDER BY position
        LIMIT 20
    """).fetchall()
    result["cq_tasks"] = [
        {"title": t["title"], "bucket": t["bucket"], "due_date": t["due_date"]}
        for t in cq
    ]
    if cq:
        result["sources"].append("cq_tasks")

    return result


def _scout_goals_and_values(conn, now: int) -> dict:
    """Infer goals from commitment patterns, beliefs, calendar focus areas."""
    result: dict = {"commitment_themes": [], "calendar_categories": [], "sources": []}

    # Commitment themes (what field patterns)
    themes = conn.execute("""
        SELECT json_extract(object, '$.what') as what,
               json_extract(object, '$.direction') as direction
        FROM claims
        WHERE claim_type = 'commitment'
          AND (superseded_by IS NULL OR superseded_by = '')
          AND json_extract(object, '$.status') = 'open'
        ORDER BY confidence DESC
        LIMIT ?
    """, (PERSON_MODEL_MAX_SCOUT_RESULTS,)).fetchall()
    result["commitment_themes"] = [
        {"what": t["what"], "direction": t["direction"]}
        for t in themes
    ]
    if themes:
        result["sources"].append("claims")

    # Fact beliefs (patterns, habits)
    facts = conn.execute("""
        SELECT summary, confidence
        FROM beliefs
        WHERE status = 'active' AND belief_type = 'fact'
        ORDER BY confidence DESC
        LIMIT 15
    """).fetchall()
    result["fact_beliefs"] = [dict(f) for f in facts]
    if facts:
        result["sources"].append("beliefs")

    return result


def _scout_life_architecture(conn, now: int, user_names: list[str] | None = None) -> dict:
    """Family, routines, non-negotiable blocks from calendar + contacts."""
    result: dict = {"recurring_personal": [], "family_signals": [], "sources": []}

    # Personal calendar events (weekends, evenings, recurring)
    personal = conn.execute("""
        SELECT metadata->>'subject' as subject, COUNT(*) as cnt,
               AVG(CAST(strftime('%H', timestamp, 'unixepoch', 'localtime') AS INTEGER)) as avg_hour
        FROM events
        WHERE event_type = 'calendar_event'
          AND timestamp > ?
          AND metadata->>'subject' IS NOT NULL
        GROUP BY LOWER(metadata->>'subject')
        HAVING cnt >= 2 AND (avg_hour < 9 OR avg_hour > 17)
        ORDER BY cnt DESC
        LIMIT 15
    """, (now - _90D,)).fetchall()
    result["recurring_personal"] = [
        {"subject": r["subject"], "count": r["cnt"], "avg_hour": round(r["avg_hour"], 1)}
        for r in personal
    ]
    if personal:
        result["sources"].append("calendar")

    # Family beliefs — only return those involving the USER to avoid
    # leaking third-party family relationships extracted from conversations
    if user_names:
        name_clauses = " OR ".join("LOWER(summary) LIKE ?" for _ in user_names)
        params = [f"%{n.lower()}%" for n in user_names]
        family_beliefs = conn.execute(f"""
            SELECT summary, confidence, json_extract(data, '$.relationship_type') as rel_type
            FROM beliefs
            WHERE status = 'active' AND belief_type = 'relation'
              AND (LOWER(summary) LIKE '%family%' OR LOWER(summary) LIKE '%spouse%'
                   OR LOWER(summary) LIKE '%child%' OR LOWER(summary) LIKE '%parent%')
              AND ({name_clauses})
            ORDER BY confidence DESC
            LIMIT 10
        """, params).fetchall()
    else:
        # No user identity known — return nothing rather than leaking all family beliefs
        family_beliefs = []
    result["family_signals"] = [dict(f) for f in family_beliefs]
    if family_beliefs:
        result["sources"].append("beliefs")

    return result


def _scout_temporal_patterns(conn, now: int) -> dict:
    """Hourly/daily event patterns, meeting load, work hours."""
    result: dict = {"daily_histogram": {}, "meeting_load_7d": 0, "first_last_activity": {},
                    "sources": []}

    # Day-of-week histogram (30d)
    daily = conn.execute("""
        SELECT CAST(strftime('%w', timestamp, 'unixepoch', 'localtime') AS INTEGER) as dow,
               COUNT(*) as cnt
        FROM events
        WHERE timestamp > ?
        GROUP BY dow
        ORDER BY dow
    """, (now - _30D,)).fetchall()
    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    result["daily_histogram"] = {day_names[r["dow"]]: r["cnt"] for r in daily}
    if daily:
        result["sources"].append("events")

    # Meeting load (7d)
    meetings = conn.execute(
        "SELECT COUNT(*) as cnt FROM events WHERE event_type = 'calendar_event' AND timestamp > ? AND timestamp < ?",
        (now - _7D, now + _7D),
    ).fetchone()
    result["meeting_load_7d"] = meetings["cnt"] if meetings else 0

    # First/last activity per day (7d)
    activity = conn.execute("""
        SELECT strftime('%Y-%m-%d', timestamp, 'unixepoch', 'localtime') as day,
               MIN(strftime('%H:%M', timestamp, 'unixepoch', 'localtime')) as first_activity,
               MAX(strftime('%H:%M', timestamp, 'unixepoch', 'localtime')) as last_activity,
               COUNT(*) as event_count
        FROM events
        WHERE timestamp > ?
        GROUP BY day
        ORDER BY day DESC
        LIMIT 7
    """, (now - _7D,)).fetchall()
    result["first_last_activity"] = {
        r["day"]: {"first": r["first_activity"], "last": r["last_activity"], "events": r["event_count"]}
        for r in activity
    }

    return result


def _scout_domain_expertise(conn, now: int) -> dict:
    """Topics discussed, tools used from beliefs and annotations."""
    result: dict = {"topics": [], "tools": [], "sources": []}

    # Topic annotations (top topics from annotations table)
    topics = conn.execute("""
        SELECT value, COUNT(*) as cnt
        FROM annotations
        WHERE facet = 'topic'
        GROUP BY value
        ORDER BY cnt DESC
        LIMIT ?
    """, (PERSON_MODEL_MAX_SCOUT_RESULTS,)).fetchall()
    result["topics"] = [{"topic": t["value"], "count": t["cnt"]} for t in topics]
    if topics:
        result["sources"].append("annotations")

    # Tool signals from app_focus events
    tools = conn.execute("""
        SELECT metadata->>'app_name' as app, COUNT(*) as cnt
        FROM events
        WHERE event_type = 'app_focus' AND timestamp > ?
          AND metadata->>'app_name' IS NOT NULL
        GROUP BY app
        ORDER BY cnt DESC
        LIMIT 20
    """, (now - _30D,)).fetchall()
    result["tools"] = [{"app": t["app"], "sessions": t["cnt"]} for t in tools]
    if tools:
        result["sources"].append("knowledgec")

    return result


def _scout_financial_landscape(conn, now: int) -> dict:
    """Financial contacts, recurring obligations from commitments + mail."""
    result: dict = {"financial_commitments": [], "financial_contacts": [], "sources": []}

    # Commitments with financial signals
    fin = conn.execute("""
        SELECT json_extract(object, '$.what') as what,
               json_extract(object, '$.who') as who,
               json_extract(object, '$.deadline') as deadline,
               confidence
        FROM claims
        WHERE claim_type = 'commitment'
          AND (superseded_by IS NULL OR superseded_by = '')
          AND json_extract(object, '$.status') = 'open'
          AND (LOWER(json_extract(object, '$.what')) LIKE '%payment%'
               OR LOWER(json_extract(object, '$.what')) LIKE '%bill%'
               OR LOWER(json_extract(object, '$.what')) LIKE '%invoice%'
               OR LOWER(json_extract(object, '$.what')) LIKE '%tuition%'
               OR LOWER(json_extract(object, '$.what')) LIKE '%rent%'
               OR LOWER(json_extract(object, '$.what')) LIKE '%insurance%')
        ORDER BY confidence DESC
        LIMIT 10
    """).fetchall()
    result["financial_commitments"] = [dict(f) for f in fin]
    if fin:
        result["sources"].append("claims")

    # Transaction emails (classified by Apple Intelligence's on-device model, not Alteris)
    fin_mail = conn.execute("""
        SELECT a.value, COUNT(*) as cnt
        FROM annotations a
        WHERE a.facet = 'email_category' AND a.value = '1'
        GROUP BY a.value
    """).fetchall()
    if fin_mail:
        result["transaction_email_count"] = sum(r["cnt"] for r in fin_mail)
        result["transaction_email_classified_by"] = "Apple Intelligence (Mail.app model_category=1)"
        result["sources"].append("annotations")

    return result


def _scout_visibility_gaps(conn, now: int) -> dict:
    """What the system can't see — uninstrumented channels, low-data domains."""
    result: dict = {"source_event_counts": {}, "stale_sources": [], "sources": []}

    # Event counts per source
    sources = conn.execute("""
        SELECT source, COUNT(*) as cnt,
               MAX(timestamp) as latest_ts
        FROM events
        GROUP BY source
        ORDER BY cnt DESC
    """).fetchall()
    for s in sources:
        result["source_event_counts"][s["source"]] = s["cnt"]
        age_days = (now - s["latest_ts"]) / 86400 if s["latest_ts"] else 999
        if age_days > 7:
            result["stale_sources"].append({
                "source": s["source"],
                "days_since_last": round(age_days, 1),
            })

    # Known uninstrumented channels
    result["uninstrumented"] = [
        "SMS (non-iMessage)", "Signal", "Telegram", "Discord",
        "LinkedIn DMs", "Twitter/X DMs", "personal journal/diary",
    ]
    result["sources"].append("events")

    return result


def _scout_apple_mail_categories(conn, now: int) -> dict:
    """Aggregate Apple Intelligence email category distribution.

    These categories are assigned by Apple's on-device ML model (Apple Intelligence)
    running in Mail.app — NOT by Alteris. The model_category field is written by macOS
    into the Mail Envelope Index database. Categories: 0=Primary, 1=Transactions,
    2=Updates, 3=Promotions.
    """
    result: dict = {
        "classified_by": "Apple Intelligence (on-device ML in Mail.app)",
        "categories": {}, "total": 0, "sources": [],
    }

    category_names = {
        "0": "Primary",
        "1": "Transactions",
        "2": "Updates",
        "3": "Promotions",
    }

    cats = conn.execute("""
        SELECT value, COUNT(*) as cnt
        FROM annotations
        WHERE facet = 'email_category'
        GROUP BY value
        ORDER BY cnt DESC
    """).fetchall()
    for c in cats:
        label = category_names.get(str(c["value"]), f"Category_{c['value']}")
        result["categories"][label] = c["cnt"]
        result["total"] += c["cnt"]
    if cats:
        result["sources"].append("mail_annotations")

    return result


def _scout_apple_intelligence_signals(conn, now: int) -> dict:
    """Aggregate urgency/importance signals from Apple Intelligence.

    These signals are produced by Apple's on-device ML models, NOT by Alteris:
    - time_sensitive: Apple Mail flags emails it considers time-critical
    - high_impact: Apple Mail's model_high_impact field for important senders
    - is_filtered: iMessage's junk/unknown sender filter
    - delivered_quietly: iMessage notification suppression for low-priority messages
    All classification happens on-device in macOS — Alteris just reads the results.
    """
    result: dict = {
        "classified_by": "Apple Intelligence (on-device ML in Mail.app and iMessage)",
        "signals": {}, "sources": [],
    }

    # Time sensitive annotations
    ts = conn.execute(
        "SELECT COUNT(*) as cnt FROM annotations WHERE facet = 'time_sensitive' AND value = '1'"
    ).fetchone()
    result["signals"]["time_sensitive_count"] = ts["cnt"] if ts else 0

    # High impact
    hi = conn.execute(
        "SELECT COUNT(*) as cnt FROM annotations WHERE facet = 'high_impact' AND value = '1'"
    ).fetchone()
    result["signals"]["high_impact_count"] = hi["cnt"] if hi else 0

    # iMessage filtered/quiet
    filtered = conn.execute(
        "SELECT COUNT(*) as cnt FROM annotations WHERE facet = 'is_filtered' AND value = '1'"
    ).fetchone()
    result["signals"]["imessage_filtered_count"] = filtered["cnt"] if filtered else 0

    quiet = conn.execute(
        "SELECT COUNT(*) as cnt FROM annotations WHERE facet = 'delivered_quietly' AND value = '1'"
    ).fetchone()
    result["signals"]["delivered_quietly_count"] = quiet["cnt"] if quiet else 0

    if any(v > 0 for v in result["signals"].values()):
        result["sources"].append("apple_intelligence")

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Surveyor synthesis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SURVEYOR_SYSTEM = """\
You are the Person Model Surveyor. Your job is to synthesize raw scout data
from a knowledge graph into a structured 11-dimension Person Model.

## Instructions

1. For each dimension, fill in fields from the scout evidence.
2. Set confidence per dimension (0.0-1.0):
   - High (>0.7): Multiple independent sources agree
   - Medium (0.4-0.7): Single source or inferred from patterns
   - Low (<0.3): Speculative or no data
3. List which sources contributed to each dimension.
4. For dimensions with very low data, leave fields as defaults and set confidence near 0.
5. Privacy: Use first name + last initial for contacts. Abstract dollar amounts.
   Do NOT include raw email addresses or phone numbers in the model.
6. For goals_and_values: Infer from commitment patterns, calendar priorities,
   and how the user allocates time. What do they consistently invest effort in?

## Apple Intelligence Signals (classified by Apple, NOT by Alteris)
The scout data includes two sections produced by Apple's on-device ML models
(Apple Intelligence), not by Alteris's own analysis:

- **apple_mail_categories**: Apple Mail's on-device model classifies every email
  into Primary (0), Transactions (1), Updates (2), or Promotions (3). This is the
  `model_category` field from Mail.app's Envelope Index. Use the distribution to
  understand what fraction of the user's email is signal vs noise.

- **apple_intelligence_signals**: Aggregated counts of Apple's urgency flags:
  `time_sensitive` (Apple Mail), `high_impact` (Apple Mail), `is_filtered`
  (iMessage junk filter), `delivered_quietly` (iMessage notification suppression).
  These reflect Apple's ML judgment about importance — use them to calibrate
  the communication_fingerprint dimension but attribute them to Apple Intelligence.

When referencing these in the model, note the classifier (e.g., "Apple Intelligence
classifies 60% of email as Promotions/Updates").

## Output Format
Return valid JSON matching the 11-dimension schema. Each dimension has its own
confidence and sources array.
"""

_SURVEYOR_PROMPT_TEMPLATE = """\
Here is the raw scout data from the user's knowledge graph:

{scout_json}

{corrections_section}

Produce the 11-dimension Person Model as a single JSON object.
"""


def phase2_surveyor(llm, scout_data: dict, user_corrections: dict) -> dict:
    """Use Pro model to synthesize scout data into person model."""
    scout_json = json.dumps(scout_data, indent=2, default=str)

    corrections_section = ""
    if user_corrections:
        corrections_section = (
            "The user has previously corrected these fields (ALWAYS preserve them):\n"
            + json.dumps(user_corrections, indent=2)
        )

    prompt = _SURVEYOR_PROMPT_TEMPLATE.format(
        scout_json=scout_json,
        corrections_section=corrections_section,
    )

    raw = llm.generate(
        prompt=prompt,
        system=_SURVEYOR_SYSTEM,
        model=PERSON_MODEL_SURVEYOR_MODEL,
        temperature=0.2,
        max_tokens=8192,
        format_json=True,
    )

    if not raw:
        logger.warning("Surveyor returned empty response, falling back to scout-only")
        return phase2_scout_only(scout_data, user_corrections)

    try:
        model = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Surveyor returned invalid JSON, falling back to scout-only")
        return phase2_scout_only(scout_data, user_corrections)

    # Merge user corrections (override surveyor output)
    model = _apply_corrections(model, user_corrections)

    # Ensure all 11 dimensions exist
    template = _empty_model()
    for dim in template:
        if dim not in model:
            model[dim] = template[dim]

    return model


def phase2_scout_only(scout_data: dict, user_corrections: dict) -> dict:
    """Build model directly from scout data without LLM (fallback/dry-run)."""
    model = _empty_model()

    # Identity
    ident = scout_data.get("identity", {})
    model["identity"]["name"] = ident.get("name", "")
    model["identity"]["timezone"] = ident.get("timezone", "")
    model["identity"]["location"] = ident.get("location", "")
    model["identity"]["emails"] = ident.get("emails", [])
    model["identity"]["phones"] = ident.get("phones", [])
    model["identity"]["confidence"] = 0.8 if ident.get("name") else 0.1
    model["identity"]["sources"] = ["config", "persons"]

    # Communication fingerprint
    comm = scout_data.get("communication_fingerprint", {})
    if comm.get("channel_volumes"):
        model["communication_fingerprint"]["primary_channels"] = list(comm["channel_volumes"].keys())[:5]
    model["communication_fingerprint"]["message_volume_weekly"] = comm.get("total_7d", 0)
    if comm.get("hourly_histogram"):
        sorted_hours = sorted(comm["hourly_histogram"].items(), key=lambda x: x[1], reverse=True)
        model["communication_fingerprint"]["peak_hours"] = [h for h, _ in sorted_hours[:3]]
        model["communication_fingerprint"]["quiet_hours"] = [h for h, _ in sorted_hours[-3:]]
    model["communication_fingerprint"]["confidence"] = 0.6 if comm.get("total_7d", 0) > 50 else 0.2
    model["communication_fingerprint"]["sources"] = comm.get("sources", [])

    # Relationship map
    rel = scout_data.get("relationship_map", {})
    model["relationship_map"]["inner_circle"] = [
        r.get("name", "") for r in rel.get("inner_circle", [])
    ]
    model["relationship_map"]["professional_contacts"] = [
        r.get("name", "") for r in rel.get("frequent_contacts", [])[:10]
    ]
    model["relationship_map"]["confidence"] = 0.5 if rel.get("inner_circle") else 0.1
    model["relationship_map"]["sources"] = rel.get("sources", [])

    # Active workstreams
    ws = scout_data.get("active_workstreams", {})
    model["active_workstreams"]["open_commitments_inbound"] = ws.get("commitments_inbound", 0)
    model["active_workstreams"]["open_commitments_outbound"] = ws.get("commitments_outbound", 0)
    model["active_workstreams"]["overdue_count"] = ws.get("overdue", 0)
    model["active_workstreams"]["confidence"] = 0.6 if ws.get("sources") else 0.1
    model["active_workstreams"]["sources"] = ws.get("sources", [])

    # Temporal patterns
    tp = scout_data.get("temporal_patterns", {})
    model["temporal_patterns"]["meeting_load_weekly"] = tp.get("meeting_load_7d", 0)
    if tp.get("daily_histogram"):
        hist = tp["daily_histogram"]
        if hist:
            model["temporal_patterns"]["busiest_day"] = max(hist, key=hist.get)
            model["temporal_patterns"]["quietest_day"] = min(hist, key=hist.get)
    model["temporal_patterns"]["confidence"] = 0.5 if tp.get("sources") else 0.1
    model["temporal_patterns"]["sources"] = tp.get("sources", [])

    # Domain expertise
    de = scout_data.get("domain_expertise", {})
    model["domain_expertise"]["topics_discussed"] = [
        t["topic"] for t in de.get("topics", [])[:15]
    ]
    model["domain_expertise"]["tools_used"] = [
        t["app"] for t in de.get("tools", [])[:10]
    ]
    model["domain_expertise"]["confidence"] = 0.4 if de.get("topics") else 0.1
    model["domain_expertise"]["sources"] = de.get("sources", [])

    # Visibility gaps
    vg = scout_data.get("visibility_gaps", {})
    model["visibility_gaps"]["uninstrumented_channels"] = vg.get("uninstrumented", [])
    model["visibility_gaps"]["low_visibility_domains"] = [
        s["source"] for s in vg.get("stale_sources", [])
    ]
    model["visibility_gaps"]["confidence"] = 0.3
    model["visibility_gaps"]["sources"] = vg.get("sources", [])

    # Low-confidence dimensions filled with minimal data
    for dim in ["professional", "goals_and_values", "life_architecture", "financial_landscape"]:
        scout = scout_data.get(dim, {})
        model[dim]["confidence"] = 0.2 if scout.get("sources") else 0.0
        model[dim]["sources"] = scout.get("sources", [])

    # Apply user corrections
    model = _apply_corrections(model, user_corrections)

    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Save to DB
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase3_save(
    store: LayeredGraphStore,
    model: dict,
    user_corrections: dict,
    version: int,
    method: str,
) -> int:
    """Save the person model to the DB. Returns the row ID."""
    now = int(time.time())
    cursor = store.conn.execute(
        """INSERT INTO person_model (version, model_json, user_corrections, estimated_at, estimation_method)
           VALUES (?, ?, ?, ?, ?)""",
        (version, json.dumps(model), json.dumps(user_corrections), now, method),
    )
    store.conn.commit()
    return cursor.lastrowid


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _apply_corrections(model: dict, corrections: dict) -> dict:
    """Merge user_corrections into model (corrections always win)."""
    if not corrections:
        return model
    for dim, fields in corrections.items():
        if dim in model and isinstance(fields, dict):
            for field, value in fields.items():
                model[dim][field] = value
    return model


def _load_user_config_safe() -> dict:
    """Load user config from profile.yaml or config.json, silently returning {} on failure."""
    from alteris.profile import load_profile
    return load_profile()
