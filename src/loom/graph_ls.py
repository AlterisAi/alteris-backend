"""Graph ls — stratified summary of the knowledge graph for LLM reasoning.

Produces a 7-tier JSON summary designed to be the input to the
Pro-Lite-Pro sandwich pipeline:

  Tier 1: Inner Circle    — multi-source, active, commitment-linked persons
  Tier 2: Rising Signals  — temporal bursts, new/reactivated contacts
  Tier 3: Action Matrix   — open commitments by urgency
  Tier 4: Relationships   — FOAF-style relation beliefs with context
  Tier 5: Upcoming Events — calendar events with resolved participants
  Tier 6: Structural Anomalies — data quality issues, noise, ghosts
  Tier 7: Epistemic Gaps  — what the graph CANNOT answer

The output is deterministic (pure SQL aggregations, no LLM calls) and
designed to fit in ~8K tokens so a frontier model can survey the entire
graph topology in a single context window.

Usage:
    from loom.graph_ls import generate_graph_ls
    ls = generate_graph_ls(store)
    # Feed ls to Gemini 3.1 Pro as the Surveyor input
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)

_7D = 7 * 86400
_30D = 30 * 86400


def generate_graph_ls(store: LayeredGraphStore) -> dict:
    """Generate the full stratified graph summary.

    Returns a dict with keys: meta, schema, TIER_1 through TIER_7.
    """
    now = int(time.time())
    conn = store.conn
    conn.row_factory = _dict_factory

    result = {
        "meta": _meta(conn, now),
        "schema": _schema(conn),
        "TIER_1_INNER_CIRCLE": _tier1_inner_circle(conn, now),
        "TIER_2_RISING_SIGNALS": _tier2_rising(conn, now),
        "TIER_3_ACTION_MATRIX": _tier3_action_matrix(conn, now),
        "TIER_4_RELATIONSHIPS": _tier4_relationships(conn, now),
        "TIER_5_UPCOMING_EVENTS": _tier5_upcoming(conn, now),
        "TIER_6_STRUCTURAL_ANOMALIES": _tier6_anomalies(conn, now),
        "TIER_7_EPISTEMIC_GAPS": _tier7_gaps(conn, now),
    }

    return result


def _dict_factory(cursor, row):
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Meta & Schema
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _meta(conn, now: int) -> dict:
    """High-level volume counts."""
    event_counts = {}
    for row in conn.execute(
        "SELECT source, COUNT(*) as cnt FROM events GROUP BY source ORDER BY cnt DESC"
    ).fetchall():
        event_counts[row["source"]] = row["cnt"]

    belief_counts = {}
    for row in conn.execute(
        "SELECT belief_type || '_' || status as key, COUNT(*) as cnt "
        "FROM beliefs GROUP BY belief_type, status ORDER BY cnt DESC"
    ).fetchall():
        belief_counts[row["key"]] = row["cnt"]

    persons_count = conn.execute("SELECT COUNT(*) as c FROM persons").fetchone()["c"]

    # Claim breakdown
    claim_counts = {}
    for row in conn.execute(
        "SELECT claim_type, COUNT(*) as cnt FROM claims "
        "GROUP BY claim_type ORDER BY cnt DESC"
    ).fetchall():
        claim_counts[row["claim_type"]] = row["cnt"]

    open_commitments = conn.execute("""
        SELECT COUNT(*) as c FROM claims
        WHERE claim_type = 'commitment'
          AND (superseded_by IS NULL OR superseded_by = '')
          AND json_extract(object, '$.status') = 'open'
    """).fetchone()["c"]

    return {
        "system": "Loom — computational autobiography. Constructs a living model of a person's digital existence from scattered traces across email, calendar, messaging, browser, shell, and app usage.",
        "data_dictionary": {
            "events": "Raw ingested data points from Mac-native databases (Mail, iMessage, WhatsApp, Calendar, Contacts, Granola meetings, Slack, knowledgeC app usage, Safari/Chrome history, Apple Notes, shell history). Each event has: source, timestamp, participants, raw_content, metadata, sensitivity level.",
            "claims": "Extracted assertions about the world. Types: 'triage' (LLM relevance scoring), 'commitment' (promises, requests, deadlines), 'structural' (deterministic patterns). Each claim has confidence (0.0-1.0) and provenance to source events.",
            "beliefs": "Synthesized higher-order knowledge compiled from clusters of claims. Types: 'entity' (people, orgs), 'relation' (connections between entities with FOAF-style tiers), 'fact' (verified assertions), 'observation' (behavioral patterns). Each belief has: confidence, epistemic_level (observation/inference/judgment/computed), inference_chain (reasoning steps), source_claims, and Admiralty scores (source_reliability A-F, info_credibility 1-6).",
            "persons": "Resolved person entities via union-find across sources. A single person may have email, phone, Slack handle identifiers merged. Each has a contact tier (1-4) and communication profile.",
            "ambient_sources": "Behavioral data that shows what the user ACTUALLY DID (vs what they said they would do). knowledgeC=app focus times, safari/chrome=sites visited, shell_history=commands run, notes=notes written. Critical for closing the Intention-Action Gap.",
            "visibility_index": "Per-domain visibility coefficient V = commitments_with_ambient_trace / total_commitments. V < 0.2 means the system is structurally blind to that domain (e.g., finance transactions done via mobile app).",
            "work_rhythm": "Daily first/last interaction times from knowledgeC app_focus events. Reveals wake patterns, late nights, weekend work.",
            "orphan": "A commitment whose source thread has had zero activity in 7 days. High orphan rate suggests either: (a) tasks completed via uninstrumented channel, (b) tasks genuinely forgotten.",
        },
        "generated": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "total_events": sum(event_counts.values()),
        "total_persons": persons_count,
        "total_claims": sum(claim_counts.values()),
        "total_beliefs": sum(belief_counts.values()),
        "open_commitments": open_commitments,
        "events_by_source": event_counts,
        "claims_by_type": claim_counts,
        "beliefs_by_type_status": belief_counts,
        "ambient_sources_present": _detect_ambient_sources(conn),
        "work_rhythm": _detect_work_rhythm(conn, now),
    }


def _detect_work_rhythm(conn, now: int) -> dict:
    """Detect daily first/last interaction times from knowledgeC app_focus events."""
    rows = conn.execute("""
        SELECT date(timestamp, 'unixepoch', 'localtime') as day,
               time(MIN(timestamp), 'unixepoch', 'localtime') as first_active,
               time(MAX(timestamp), 'unixepoch', 'localtime') as last_active,
               COUNT(*) as interactions,
               CAST((MAX(timestamp) - MIN(timestamp)) / 3600.0 AS INTEGER) as active_hours
        FROM events
        WHERE source = 'knowledgec' AND timestamp > ?
        GROUP BY day
        ORDER BY day DESC
        LIMIT 7
    """, (now - _7D,)).fetchall()

    if not rows:
        return {"status": "NO_DATA — knowledgeC not ingested"}

    days = []
    for r in rows:
        days.append({
            "date": r["day"],
            "wake": r["first_active"],
            "last_active": r["last_active"],
            "interactions": r["interactions"],
            "active_hours": r["active_hours"],
        })

    # Compute averages
    avg_wake = None
    avg_last = None
    if days:
        wake_minutes = []
        last_minutes = []
        for d in days:
            if d["wake"]:
                parts = d["wake"].split(":")
                wake_minutes.append(int(parts[0]) * 60 + int(parts[1]))
            if d["last_active"]:
                parts = d["last_active"].split(":")
                last_minutes.append(int(parts[0]) * 60 + int(parts[1]))
        if wake_minutes:
            avg_m = sum(wake_minutes) // len(wake_minutes)
            avg_wake = f"{avg_m // 60:02d}:{avg_m % 60:02d}"
        if last_minutes:
            avg_m = sum(last_minutes) // len(last_minutes)
            avg_last = f"{avg_m // 60:02d}:{avg_m % 60:02d}"

    return {
        "avg_first_interaction": avg_wake,
        "avg_last_interaction": avg_last,
        "days": days,
    }


def _detect_ambient_sources(conn) -> dict:
    """Check which ambient sources have been ingested."""
    ambient = {}
    for src in ("knowledgec", "safari", "chrome", "arc", "brave", "edge",
                "vivaldi", "notes", "shell_history"):
        row = conn.execute(
            "SELECT COUNT(*) as c FROM events WHERE source = ?", (src,)
        ).fetchone()
        if row["c"] > 0:
            ambient[src] = row["c"]
    return ambient if ambient else {"status": "NONE — ambient adapters not yet ingested"}


def _schema(conn) -> dict:
    """Compact schema summary for each key table."""
    tables = ["events", "claims", "beliefs", "persons", "event_persons",
              "person_identifiers", "person_profiles", "annotations", "projections"]
    schema = {}
    for tbl in tables:
        try:
            cols = conn.execute(f"PRAGMA table_info({tbl})").fetchall()
            schema[tbl] = ", ".join(c["name"] for c in cols)
        except Exception:
            pass
    return schema


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 1: Inner Circle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _tier1_inner_circle(conn, now: int) -> dict:
    """Multi-source, active persons with commitment links."""
    rows = conn.execute("""
        SELECT p.canonical_name, p.person_id,
               COUNT(DISTINCT e.source) as source_count,
               COUNT(DISTINCT ep.event_id) as total_edges,
               SUM(CASE WHEN e.timestamp > ? THEN 1 ELSE 0 END) as events_7d,
               SUM(CASE WHEN e.timestamp > ? THEN 1 ELSE 0 END) as events_30d,
               GROUP_CONCAT(DISTINCT e.source) as sources,
               pp.tier as contact_tier,
               pp.user_initiated_ratio,
               pp.channels
        FROM persons p
        JOIN event_persons ep ON p.person_id = ep.person_id
        JOIN events e ON ep.event_id = e.id
        LEFT JOIN person_profiles pp ON p.person_id = pp.person_id
        WHERE p.is_user = 0
          AND e.timestamp > ?
        GROUP BY p.person_id
        HAVING source_count >= 3 AND events_30d >= 10
        ORDER BY source_count DESC, events_7d DESC
        LIMIT 15
    """, (now - _7D, now - _30D, now - _30D)).fetchall()

    nodes = []
    for r in rows:
        # Get open commitments referencing this person with full details
        name_lower = (r["canonical_name"] or "").lower()
        commitments = []
        if name_lower:
            commit_rows = conn.execute("""
                SELECT json_extract(object, '$.what') as what,
                       json_extract(object, '$.who') as who,
                       json_extract(object, '$.to_whom') as to_whom,
                       json_extract(object, '$.deadline') as deadline,
                       json_extract(object, '$.type') as ctype,
                       json_extract(object, '$.direction') as direction,
                       json_extract(object, '$.priority') as priority,
                       json_extract(object, '$.staleness_signal') as staleness,
                       json_extract(object, '$.evidence_quote') as evidence_quote,
                       confidence
                FROM claims
                WHERE claim_type = 'commitment'
                  AND (superseded_by IS NULL OR superseded_by = '')
                  AND json_extract(object, '$.status') = 'open'
                  AND (LOWER(json_extract(object, '$.who')) LIKE ?
                       OR LOWER(json_extract(object, '$.to_whom')) LIKE ?)
                ORDER BY confidence DESC
                LIMIT 5
            """, (f"%{name_lower}%", f"%{name_lower}%")).fetchall()
            for cr in commit_rows:
                commitments.append({
                    "what": cr["what"],
                    "to_whom": cr["to_whom"],
                    "deadline": cr["deadline"],
                    "type": cr["ctype"],
                    "direction": cr["direction"],
                    "priority": cr["priority"],
                    "confidence": cr["confidence"],
                    "staleness": cr["staleness"],
                    "evidence_quote": (cr["evidence_quote"] or "")[:150] or None,
                })

        # Get relation beliefs with full data
        relation_beliefs = []
        rel_rows = conn.execute("""
            SELECT summary, confidence, data,
                   inference_chain, evidence_log
            FROM beliefs
            WHERE belief_type = 'relation' AND status = 'active'
              AND subject LIKE ?
            ORDER BY confidence DESC LIMIT 3
        """, (f"%{r['person_id']}%",)).fetchall()
        for rb in rel_rows:
            belief_entry = {
                "summary": rb["summary"],
                "confidence": rb["confidence"],
            }
            if rb["data"]:
                try:
                    belief_entry["data"] = json.loads(rb["data"])
                except (json.JSONDecodeError, TypeError):
                    pass
            if rb["inference_chain"]:
                try:
                    belief_entry["inference_chain"] = json.loads(rb["inference_chain"])
                except (json.JSONDecodeError, TypeError):
                    pass
            relation_beliefs.append(belief_entry)

        node = {
            "name": r["canonical_name"],
            "person_id": r["person_id"],
            "sources": r["source_count"],
            "events_7d": r["events_7d"],
            "events_30d": r["events_30d"],
            "channels": r["sources"],
            "tier": r["contact_tier"],
            "user_initiated_ratio": round(r["user_initiated_ratio"], 2) if r["user_initiated_ratio"] else None,
            "open_commitments": commitments,
        }
        if relation_beliefs:
            node["relation_beliefs"] = relation_beliefs

        nodes.append(node)

    return {
        "weight": "CRITICAL",
        "description": "Multi-source (3+), active in last 30d, commitment-linked",
        "field_glossary": {
            "sources": "Number of distinct data sources (mail, whatsapp, calendar, etc.) this person appears in",
            "events_7d": "Total events involving this person in the last 7 days",
            "events_30d": "Total events involving this person in the last 30 days",
            "channels": "Comma-separated list of data sources",
            "tier": "Contact tier (1=inner circle, 2=active, 3=peripheral, 4=distant)",
            "user_initiated_ratio": "Fraction of messages where the user initiated (0.0-1.0). High = user drives relationship",
            "open_commitments": "List of open commitments involving this person, with full fields",
            "relation_beliefs": "Relation beliefs about this person from the beliefs table, including inference chains",
        },
        "nodes": nodes,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 2: Rising Signals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _tier2_rising(conn, now: int) -> dict:
    """Temporal burst contacts — disproportionate 7d vs prior 23d activity."""
    rows = conn.execute("""
        SELECT p.canonical_name,
               COUNT(CASE WHEN e.timestamp > ? THEN 1 END) as events_7d,
               COUNT(CASE WHEN e.timestamp <= ? AND e.timestamp > ? THEN 1 END) as events_prev_23d,
               GROUP_CONCAT(DISTINCT e.source) as sources,
               GROUP_CONCAT(DISTINCT e.event_type) as event_types,
               datetime(MAX(e.timestamp), 'unixepoch') as last_active
        FROM persons p
        JOIN event_persons ep ON p.person_id = ep.person_id
        JOIN events e ON ep.event_id = e.id
        WHERE p.is_user = 0 AND e.timestamp > ?
        GROUP BY p.person_id
        HAVING events_7d > 3
          AND (events_prev_23d = 0 OR CAST(events_7d AS REAL)/events_prev_23d > 3.0)
        ORDER BY CAST(events_7d AS REAL) / MAX(1, events_prev_23d) DESC
        LIMIT 10
    """, (now - _7D, now - _7D, now - _30D, now - _30D)).fetchall()

    nodes = []
    for r in rows:
        prev = r["events_prev_23d"]
        if prev == 0:
            ratio = "NEW"
        else:
            ratio = f"{r['events_7d'] / prev:.0f}x"

        nodes.append({
            "name": r["canonical_name"],
            "events_7d": r["events_7d"],
            "prior_23d": prev,
            "burst_ratio": ratio,
            "sources": r["sources"],
            "event_types": r["event_types"],
            "last_active": r["last_active"],
        })

    return {
        "weight": "HIGH",
        "description": "Temporal burst — disproportionate 7d vs prior 23d activity",
        "field_glossary": {
            "events_7d": "Events in last 7 days",
            "prior_23d": "Events in the 23 days before that (days 8-30)",
            "burst_ratio": "Ratio of 7d to prior 23d activity. 'NEW' = zero prior activity. '7x' = 7 times more active than baseline",
            "event_types": "Types of events (email, message, calendar_event, etc.)",
            "last_active": "Timestamp of most recent event",
        },
        "nodes": nodes,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 3: Action Matrix
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _tier3_action_matrix(conn, now: int) -> dict:
    """Open commitments grouped by urgency and type."""
    today = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
    today_plus_3 = datetime.fromtimestamp(now + 3 * 86400, tz=timezone.utc).strftime("%Y-%m-%d")
    today_plus_7 = datetime.fromtimestamp(now + 7 * 86400, tz=timezone.utc).strftime("%Y-%m-%d")

    rows = conn.execute("""
        SELECT id, object, confidence, subject as source_event_id
        FROM claims
        WHERE claim_type = 'commitment'
          AND (superseded_by IS NULL OR superseded_by = '')
          AND json_extract(object, '$.status') = 'open'
        ORDER BY json_extract(object, '$.priority') ASC, confidence DESC
    """).fetchall()

    overdue = []
    imminent = []
    this_week = []
    undeadlined = []

    by_type: dict[str, dict] = {}

    for r in rows:
        # Parse full object JSON
        try:
            obj = json.loads(r["object"]) if isinstance(r["object"], str) else r["object"]
        except (json.JSONDecodeError, TypeError):
            obj = {}

        ctype = obj.get("type") or "unknown"
        if ctype not in by_type:
            by_type[ctype] = {"total": 0, "overdue": 0}
        by_type[ctype]["total"] += 1

        # Check thread activity for orphan detection
        thread_active = _has_recent_thread_activity(conn, r["source_event_id"], now - _7D)

        entry = {
            "what": obj.get("what"),
            "who": obj.get("who"),
            "to_whom": obj.get("to_whom"),
            "type": ctype,
            "direction": obj.get("direction"),
            "priority": obj.get("priority"),
            "confidence": r["confidence"],
            "orphan": not thread_active,
            "staleness_signal": obj.get("staleness_signal"),
            "provenance": obj.get("provenance"),
            "speech_act": obj.get("speech_act"),
            "has_named_actor": obj.get("has_named_actor"),
            "has_concrete_deliverable": obj.get("has_concrete_deliverable"),
            "has_temporal_constraint": obj.get("has_temporal_constraint"),
        }
        # Include evidence quote if present (truncated)
        eq = obj.get("evidence_quote")
        if eq:
            entry["evidence_quote"] = eq[:200]
        # Include next_action if present
        na = obj.get("next_action")
        if na:
            entry["next_action"] = na[:200]

        deadline = obj.get("deadline")
        if deadline:
            entry["deadline"] = deadline
            if deadline < today:
                by_type[ctype]["overdue"] += 1
                overdue.append(entry)
            elif deadline <= today_plus_3:
                imminent.append(entry)
            elif deadline <= today_plus_7:
                this_week.append(entry)
            else:
                if (obj.get("priority") or 5) <= 2:
                    this_week.append(entry)
        else:
            if r["confidence"] >= 0.9:
                undeadlined.append(entry)

    return {
        "weight": "HIGH",
        "description": f"Open commitments: {len(overdue)} overdue, {len(imminent)} imminent, {len(this_week)} this week",
        "field_glossary": {
            "what": "The commitment action (e.g., 'send updated proposal')",
            "who": "Who owns the action (usually 'user')",
            "to_whom": "The counterparty (person or org the action is directed at)",
            "type": "Commitment type (inbound_request, payment_due, self_directed, etc.)",
            "direction": "How the commitment was created (direct_ask, self_directed, inferred)",
            "priority": "Priority level (1=highest, 5=lowest)",
            "confidence": "System confidence in this commitment (0.0-1.0)",
            "orphan": "True if the source thread has had NO activity in 7 days — suggests the commitment may be forgotten or silently resolved",
            "staleness_signal": "Signal from extraction about whether this commitment is going stale (none, aging, stale)",
            "provenance": "How this commitment was identified (assigned_to_user, user_volunteered, inferred_from_context)",
            "speech_act": "The speech act that created it (request, promise, offer, etc.)",
            "has_named_actor": "Whether a specific person is responsible (vs. vague 'someone')",
            "has_concrete_deliverable": "Whether the outcome is specific and verifiable",
            "has_temporal_constraint": "Whether there is an explicit deadline or timeframe",
            "evidence_quote": "The source text that triggered extraction (verbatim quote from email/message)",
            "next_action": "Suggested next physical action to fulfill the commitment",
        },
        "OVERDUE": overdue[:10],
        "IMMINENT": imminent[:5],
        "THIS_WEEK": this_week[:5],
        "UNDEADLINED_HIGH_CONF": undeadlined[:8],
        "by_type": by_type,
    }


def _has_recent_thread_activity(conn, source_event_id: str, since_ts: int) -> bool:
    """Check if the source event's thread has any recent activity."""
    if not source_event_id:
        return False
    evt = conn.execute(
        "SELECT metadata->>'thread_id' as tid FROM events WHERE id = ?",
        (source_event_id,)
    ).fetchone()
    if not evt or not evt.get("tid"):
        return False
    recent = conn.execute(
        "SELECT 1 FROM events WHERE metadata->>'thread_id' = ? AND timestamp > ? LIMIT 1",
        (evt["tid"], since_ts)
    ).fetchone()
    return recent is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 4: Relationships
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _tier4_relationships(conn, now: int) -> dict:
    """Active relation beliefs with full data + domain bleed detection."""
    rows = conn.execute("""
        SELECT summary, confidence, data, epistemic_level,
               source_reliability, info_credibility,
               source_claims, inference_chain, evidence_log,
               created_at, updated_at
        FROM beliefs
        WHERE belief_type = 'relation' AND status = 'active'
        ORDER BY confidence DESC
        LIMIT 15
    """).fetchall()

    relations = []
    for r in rows:
        # Parse full data blob
        data = {}
        try:
            data = json.loads(r["data"]) if isinstance(r["data"], str) else (r["data"] or {})
        except (json.JSONDecodeError, TypeError):
            pass

        entry = {
            "summary": r["summary"],
            "confidence": r["confidence"],
            "epistemic_level": r["epistemic_level"],
            "source_reliability": r["source_reliability"],
            "info_credibility": r["info_credibility"],
            "data": data,
        }

        # Include inference chain if present
        if r["inference_chain"]:
            try:
                entry["inference_chain"] = json.loads(r["inference_chain"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Include source claims count
        if r["source_claims"]:
            try:
                sc = json.loads(r["source_claims"])
                entry["source_claim_count"] = len(sc) if isinstance(sc, list) else 0
            except (json.JSONDecodeError, TypeError):
                pass

        # Domain Bleed detection: vocational person appearing in personal calendar
        person_name = data.get("name") or ""
        rel_tier = data.get("relationship_tier") or ""
        if person_name and "vocational" in rel_tier:
            bleed = conn.execute("""
                SELECT COUNT(*) as c
                FROM events e
                JOIN event_persons ep ON e.id = ep.event_id
                JOIN persons p ON ep.person_id = p.person_id
                WHERE p.canonical_name LIKE ?
                  AND e.source = 'calendar'
                  AND e.timestamp > ?
                  AND (e.metadata LIKE '%personal%' OR e.metadata LIKE '%family%'
                       OR e.metadata LIKE '%Home%')
            """, (f"%{person_name}%", now - _30D)).fetchone()
            if bleed and bleed["c"] > 0:
                entry["domain_bleed"] = True
                entry["bleed_detail"] = (
                    f"Vocational contact appearing in {bleed['c']} "
                    f"personal/family calendar events — cross-domain dependency"
                )

        relations.append(entry)

    # Sort: domain bleed contacts first, then by confidence
    relations.sort(key=lambda r: (not r.get("domain_bleed", False), -r["confidence"]))

    return {
        "weight": "MEDIUM",
        "description": "FOAF-style relation beliefs with full data, inference chains + domain bleed detection",
        "field_glossary": {
            "summary": "Human-readable one-liner about the relationship",
            "confidence": "Belief confidence (0.0-1.0)",
            "epistemic_level": "How this belief was formed (observation=direct data, inference=derived, judgment=LLM reasoning, computed=algorithmic)",
            "source_reliability": "Admiralty source reliability grade (A=completely reliable, F=unrated). Currently placeholder.",
            "info_credibility": "Admiralty information credibility (1=confirmed, 6=unrated). Currently placeholder.",
            "data": "Full structured data blob containing: name, relationship_type, relationship_tier, role, organization, context, relationship_strength",
            "data.relationship_tier": "FOAF-style tier (intimate_family, intimate_partner, vocational_core_team, vocational_network, community, acquaintance)",
            "inference_chain": "Array of reasoning steps that produced this belief (e.g., ['Seen in 3 email threads', 'Co-attendee in 2 meetings'])",
            "source_claim_count": "Number of source claims that support this belief",
            "domain_bleed": "True if a vocational contact appears in personal/family calendar events — signals cross-domain dependency",
        },
        "relations": relations,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 5: Upcoming Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _tier5_upcoming(conn, now: int) -> dict:
    """Calendar events in the next 7 days with resolved participants."""
    rows = conn.execute("""
        SELECT e.id, e.metadata->>'subject' as title,
               datetime(e.timestamp, 'unixepoch') as event_time,
               e.timestamp,
               e.metadata->>'calendar' as calendar,
               e.metadata->>'location' as location,
               e.metadata->>'is_recurring' as recurring,
               e.metadata->>'is_all_day' as all_day
        FROM events e
        WHERE e.source = 'calendar'
          AND e.timestamp >= ?
          AND e.timestamp <= ?
        ORDER BY e.timestamp
    """, (now, now + _7D)).fetchall()

    events = []
    for r in rows:
        # Resolve participants with tier and commitment context
        participants = conn.execute("""
            SELECT p.canonical_name, p.person_id,
                   pp.tier, pp.message_count, pp.user_initiated_ratio
            FROM event_persons ep
            JOIN persons p ON ep.person_id = p.person_id
            LEFT JOIN person_profiles pp ON p.person_id = pp.person_id
            WHERE ep.event_id = ? AND p.is_user = 0
        """, (r["id"],)).fetchall()

        participant_list = []
        for p in participants:
            if not p["canonical_name"]:
                continue
            pentry = {"name": p["canonical_name"], "tier": p["tier"]}
            # Count open commitments with this person
            pname_lower = p["canonical_name"].lower()
            pc = conn.execute("""
                SELECT COUNT(*) as c FROM claims
                WHERE claim_type = 'commitment'
                  AND (superseded_by IS NULL OR superseded_by = '')
                  AND json_extract(object, '$.status') = 'open'
                  AND (LOWER(json_extract(object, '$.who')) LIKE ?
                       OR LOWER(json_extract(object, '$.to_whom')) LIKE ?)
            """, (f"%{pname_lower}%", f"%{pname_lower}%")).fetchone()
            if pc["c"] > 0:
                pentry["open_commitments"] = pc["c"]
            participant_list.append(pentry)

        # Check for commitment collisions
        commitments_near = conn.execute("""
            SELECT json_extract(object, '$.what') as what,
                   json_extract(object, '$.deadline') as deadline,
                   json_extract(object, '$.who') as who,
                   confidence
            FROM claims
            WHERE claim_type = 'commitment'
              AND (superseded_by IS NULL OR superseded_by = '')
              AND json_extract(object, '$.status') = 'open'
              AND json_extract(object, '$.deadline') IS NOT NULL
              AND json_extract(object, '$.deadline') >= date(?, 'unixepoch', '-1 day')
              AND json_extract(object, '$.deadline') <= date(?, 'unixepoch', '+1 day')
            LIMIT 5
        """, (r["timestamp"], r["timestamp"])).fetchall()

        event = {
            "title": r["title"],
            "when": r["event_time"],
            "calendar": r["calendar"],
            "location": r["location"] or None,
            "recurring": bool(r["recurring"]),
            "participants": participant_list[:10] if participant_list else [],
        }
        if commitments_near:
            event["nearby_commitments"] = [
                {"what": c["what"], "deadline": c["deadline"],
                 "who": c["who"], "confidence": c["confidence"]}
                for c in commitments_near
            ]

        events.append(event)

    return {
        "weight": "MEDIUM",
        "description": f"{len(events)} events in next 7 days",
        "field_glossary": {
            "title": "Calendar event title",
            "when": "Event start time (UTC)",
            "calendar": "Which calendar (Work, Personal, Family, etc.)",
            "location": "Event location if specified",
            "recurring": "Whether this is a recurring event",
            "participants": "Resolved attendees with their contact tier and open commitment count",
            "participants[].tier": "Contact tier (1=inner circle, 2=active, 3=peripheral, 4=distant)",
            "participants[].open_commitments": "Number of open commitments the user has with this person",
            "nearby_commitments": "Open commitments with deadlines within +/- 1 day of this event — potential preparation needed",
        },
        "events": events,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 6: Structural Anomalies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _tier6_anomalies(conn, now: int) -> dict:
    """Data quality issues, noise sources, ghost nodes."""
    anomalies = []

    # 1. Newsletter noise — high-edge automated senders
    noise_rows = conn.execute("""
        SELECT p.canonical_name, COUNT(DISTINCT ep.event_id) as edges
        FROM persons p
        JOIN event_persons ep ON p.person_id = ep.person_id
        WHERE p.is_user = 0
        GROUP BY p.person_id
        HAVING edges > 500
        ORDER BY edges DESC
        LIMIT 10
    """).fetchall()

    # Filter to likely automated senders (names suggesting newsletters/services)
    noise_keywords = {"team", "newsletter", "update", "noreply", "notification",
                      "service", "support", "account", "admin", "info",
                      "delivery", "snacks", "robinhood", "usps", "usaa",
                      "amazon", "product safety", "hide my email"}
    noise_nodes = []
    for r in noise_rows:
        name_lower = (r["canonical_name"] or "").lower()
        if any(kw in name_lower for kw in noise_keywords):
            noise_nodes.append({"name": r["canonical_name"], "edges": r["edges"]})

    if noise_nodes:
        anomalies.append({
            "type": "NEWSLETTER_NOISE",
            "detail": f"{len(noise_nodes)} automated senders with >500 edges dominate graph density",
            "nodes": noise_nodes[:5],
        })

    # 2. Ghost nodes — previously tier-1/2 but now dormant
    ghost_rows = conn.execute("""
        SELECT p.canonical_name, pp.tier, pp.message_count,
               CAST((? - MAX(e.timestamp)) / 86400 AS INTEGER) as days_silent
        FROM persons p
        JOIN event_persons ep ON p.person_id = ep.person_id
        JOIN events e ON ep.event_id = e.id
        JOIN person_profiles pp ON p.person_id = pp.person_id
        WHERE p.is_user = 0
          AND pp.tier IN (1, 2)
        GROUP BY p.person_id
        HAVING days_silent > 14
        ORDER BY pp.message_count DESC
        LIMIT 5
    """, (now,)).fetchall()

    if ghost_rows:
        anomalies.append({
            "type": "GHOST_TIER1",
            "detail": f"{len(ghost_rows)} tier-1/2 contacts silent for 14+ days",
            "nodes": [{"name": r["canonical_name"], "days_silent": r["days_silent"],
                       "message_count": r["message_count"]} for r in ghost_rows],
        })

    # 3. Orphan commitments — no recent thread activity
    orphan_count = conn.execute("""
        SELECT COUNT(*) as c FROM claims
        WHERE claim_type = 'commitment'
          AND (superseded_by IS NULL OR superseded_by = '')
          AND json_extract(object, '$.status') = 'open'
          AND json_extract(object, '$.deadline') IS NOT NULL
          AND json_extract(object, '$.deadline') < date('now', '+7 days')
    """).fetchone()["c"]
    if orphan_count > 0:
        anomalies.append({
            "type": "ORPHAN_COMMITMENTS",
            "detail": f"{orphan_count} urgent commitments with no recent thread follow-up",
        })

    # 4. Missing ambient evidence
    ambient_count = conn.execute(
        "SELECT COUNT(*) as c FROM events WHERE source IN "
        "('knowledgec','safari','chrome','arc','brave','edge','vivaldi','notes','shell_history')"
    ).fetchone()["c"]
    if ambient_count == 0:
        anomalies.append({
            "type": "NO_BEHAVIORAL_EVIDENCE",
            "detail": "0 ambient events ingested. Cannot corroborate any commitment with behavioral data.",
        })
    else:
        anomalies.append({
            "type": "AMBIENT_AVAILABLE",
            "detail": f"{ambient_count} ambient events available for intention-action triangulation",
        })

    # 5. Stale high-edge nodes (from distant past)
    stale_rows = conn.execute("""
        SELECT p.canonical_name, COUNT(DISTINCT ep.event_id) as edges,
               datetime(MAX(e.timestamp), 'unixepoch') as last_active,
               CAST((? - MAX(e.timestamp)) / 86400 AS INTEGER) as days_silent
        FROM persons p
        JOIN event_persons ep ON p.person_id = ep.person_id
        JOIN events e ON ep.event_id = e.id
        WHERE p.is_user = 0
        GROUP BY p.person_id
        HAVING edges > 200 AND days_silent > 365
        ORDER BY edges DESC
        LIMIT 5
    """, (now,)).fetchall()

    if stale_rows:
        anomalies.append({
            "type": "STALE_HIGH_EDGE",
            "detail": f"{len(stale_rows)} historical high-edge nodes (>200 edges, >1yr dormant)",
            "nodes": [{"name": r["canonical_name"], "edges": r["edges"],
                       "last_active": r["last_active"]} for r in stale_rows],
        })

    return {
        "weight": "MEDIUM",
        "description": "Data quality issues, noise, ghost nodes",
        "anomalies": anomalies,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier 7: Epistemic Gaps
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _detect_shadow_activity(conn, now: int) -> list[dict]:
    """Find high-frequency activity not matching any open commitment.

    Compares shell_history project names and browser domains against
    commitment keywords to find the 'Missing 20%' — things the user
    is actively working on but hasn't articulated as commitments.
    """
    # Get commitment keywords for cross-reference
    commitment_words: set[str] = set()
    commit_rows = conn.execute("""
        SELECT json_extract(object, '$.what') as what
        FROM claims
        WHERE claim_type = 'commitment'
          AND (superseded_by IS NULL OR superseded_by = '')
          AND json_extract(object, '$.status') = 'open'
    """).fetchall()
    for r in commit_rows:
        if r["what"]:
            for word in r["what"].lower().split():
                if len(word) > 3:
                    commitment_words.add(word)

    unmatched = []

    # Check shell_history for project directories
    shell_rows = conn.execute("""
        SELECT raw_content, COUNT(*) as cnt
        FROM events
        WHERE source = 'shell_history' AND timestamp > ?
        GROUP BY raw_content
        HAVING cnt >= 3
        ORDER BY cnt DESC
        LIMIT 50
    """, (now - _7D,)).fetchall()

    for r in shell_rows:
        cmd = (r["raw_content"] or "").lower()
        # Extract project hints from cd commands, git repos, etc.
        tokens = set(cmd.replace("/", " ").replace("-", " ").replace("_", " ").split())
        tokens = {t for t in tokens if len(t) > 3}
        if tokens and not tokens & commitment_words:
            unmatched.append({
                "source": "shell_history",
                "activity": r["raw_content"][:80] if r["raw_content"] else "",
                "frequency": r["cnt"],
            })

    # Check browser history for frequent domains
    browser_rows = conn.execute("""
        SELECT json_extract(metadata, '$.domain') as domain, COUNT(*) as cnt
        FROM events
        WHERE source IN ('safari', 'chrome', 'arc', 'brave', 'edge', 'vivaldi')
          AND timestamp > ?
        GROUP BY domain
        HAVING cnt >= 3
        ORDER BY cnt DESC
        LIMIT 30
    """, (now - _7D,)).fetchall()

    skip_domains = {"google.com", "github.com", "stackoverflow.com", "youtube.com",
                    "reddit.com", "twitter.com", "x.com", "mail.google.com",
                    "docs.google.com", "calendar.google.com", "linkedin.com"}
    for r in browser_rows:
        domain = (r["domain"] or "").lower()
        if domain and domain not in skip_domains:
            domain_tokens = set(domain.replace(".", " ").replace("-", " ").split())
            domain_tokens = {t for t in domain_tokens if len(t) > 3}
            if domain_tokens and not domain_tokens & commitment_words:
                unmatched.append({
                    "source": "browser",
                    "activity": domain,
                    "frequency": r["cnt"],
                })

    # Sort by frequency descending
    unmatched.sort(key=lambda x: -x["frequency"])
    return unmatched


def _tier7_gaps(conn, now: int) -> dict:
    """What the graph CANNOT answer — highest value for Pro Phase 1."""
    gaps = []

    # 1. Intention-Action gap
    open_commitments = conn.execute("""
        SELECT COUNT(*) as c FROM claims
        WHERE claim_type = 'commitment'
          AND (superseded_by IS NULL OR superseded_by = '')
          AND json_extract(object, '$.status') = 'open'
    """).fetchone()["c"]

    ambient_count = conn.execute(
        "SELECT COUNT(*) as c FROM events WHERE source IN "
        "('knowledgec','safari','chrome','notes','shell_history')"
    ).fetchone()["c"]

    gaps.append({
        "gap": "Intention-Action Gap",
        "severity": "CRITICAL" if ambient_count == 0 else "HIGH",
        "detail": (
            f"{open_commitments} open commitments but "
            f"{'ZERO behavioral evidence to corroborate' if ambient_count == 0 else f'{ambient_count} ambient events available for triangulation'}. "
            "Cannot distinguish genuine dropped balls from quietly resolved tasks."
        ),
        "scout_instructions": (
            "For each OVERDUE commitment: search knowledgeC for app usage matching keywords, "
            "search browser history for related URLs, search shell_history for related git/code activity. "
            "Cross-reference timestamps within 4 hours of commitment deadline."
        ),
    })

    # 2. Unresolved interview/career signals
    meta_threads = conn.execute("""
        SELECT COUNT(*) as c FROM events
        WHERE source = 'mail'
          AND timestamp > ?
          AND (metadata->>'subject' LIKE '%Meta%' OR metadata->>'subject' LIKE '%interview%')
    """, (now - _7D,)).fetchone()["c"]

    if meta_threads > 0:
        career_commitments = conn.execute("""
            SELECT json_extract(object, '$.what') as what FROM claims
            WHERE claim_type = 'commitment'
              AND (superseded_by IS NULL OR superseded_by = '')
              AND json_extract(object, '$.status') = 'open'
              AND (json_extract(object, '$.what') LIKE '%Meta%'
                   OR json_extract(object, '$.what') LIKE '%interview%'
                   OR json_extract(object, '$.what') LIKE '%Career Profile%')
        """).fetchall()

        gaps.append({
            "gap": "Career Trajectory Ambiguity",
            "severity": "HIGH",
            "detail": (
                f"{meta_threads} career-related emails in last 7 days. "
                f"{len(career_commitments)} related open commitments. "
                "Is the user actively pursuing this or passively ignoring?"
            ),
            "related_commitments": [c["what"] for c in career_commitments[:5]],
            "scout_instructions": (
                "Search browser history for 'Meta careers', 'Surreal', 'interview prep'. "
                "Search knowledgeC for LinkedIn/Greenhouse/Lever app usage. "
                "Search shell_history for resume/CV-related activity."
            ),
        })

    # 3. Team dynamics uncertainty
    team_beliefs = conn.execute("""
        SELECT COUNT(*) as c FROM beliefs
        WHERE belief_type = 'relation' AND status = 'active'
          AND (summary LIKE '%Alteris%' OR summary LIKE '%Altarus%')
    """).fetchone()["c"]

    recent_meetings = conn.execute("""
        SELECT COUNT(*) as c FROM events
        WHERE source = 'granola' AND timestamp > ?
    """, (now - _7D,)).fetchone()["c"]

    if team_beliefs > 0:
        gaps.append({
            "gap": "Alteris Team Decision State",
            "severity": "MEDIUM",
            "detail": (
                f"{team_beliefs} Alteris-related relation beliefs, "
                f"{recent_meetings} meetings in last 7 days. "
                "What is the current decision state on: YC application, "
                "product pivot, open-source strategy, fundraising?"
            ),
            "scout_instructions": (
                "Fetch Granola meeting transcripts from last 7 days. "
                "Search whatsapp/imessage for 'YC', 'pivot', 'fundrais', 'open source'. "
                "Cross-reference with calendar events for upcoming team syncs."
            ),
        })

    # 4. Financial exposure
    payment_commitments = conn.execute("""
        SELECT json_extract(object, '$.what') as what,
               json_extract(object, '$.deadline') as deadline
        FROM claims
        WHERE claim_type = 'commitment'
          AND (superseded_by IS NULL OR superseded_by = '')
          AND json_extract(object, '$.status') = 'open'
          AND json_extract(object, '$.type') = 'payment_due'
        ORDER BY json_extract(object, '$.deadline')
    """).fetchall()

    if payment_commitments:
        overdue_payments = [p for p in payment_commitments
                           if p["deadline"] and p["deadline"] < datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")]
        gaps.append({
            "gap": "Financial Exposure",
            "severity": "HIGH" if overdue_payments else "MEDIUM",
            "detail": (
                f"{len(payment_commitments)} open payment commitments, "
                f"{len(overdue_payments)} overdue. "
                "Absence-of-evidence reasoning not yet implemented — "
                "cannot determine if payments were made."
            ),
            "payments": [{"what": p["what"], "deadline": p["deadline"]} for p in payment_commitments],
            "scout_instructions": (
                "Search mail for payment confirmation emails (receipt, thank you, payment received). "
                "Search browser history for bank/payment portal visits. "
                "Search knowledgeC for banking app usage."
            ),
        })

    # 5. Family logistics collision detection
    family_signals = conn.execute("""
        SELECT COUNT(*) as c FROM claims
        WHERE claim_type = 'commitment'
          AND (superseded_by IS NULL OR superseded_by = '')
          AND json_extract(object, '$.status') = 'open'
          AND (json_extract(object, '$.what') LIKE '%child%'
               OR json_extract(object, '$.what') LIKE '%kid%'
               OR json_extract(object, '$.what') LIKE '%camp%'
               OR json_extract(object, '$.what') LIKE '%school%'
               OR json_extract(object, '$.what') LIKE '%preschool%'
               OR json_extract(object, '$.what') LIKE '%tuition%'
               OR json_extract(object, '$.what') LIKE '%incident%')
    """).fetchone()["c"]

    if family_signals > 0:
        gaps.append({
            "gap": "Family Logistics Collision",
            "severity": "MEDIUM",
            "detail": (
                f"{family_signals} family-related commitments open. "
                "Potential collision with work commitments not yet analyzed."
            ),
            "scout_instructions": (
                "Fetch calendar events tagged 'personal' or in family calendar. "
                "Cross-reference with work commitments for temporal overlaps. "
                "Check for registration deadlines (summer camp Mar 17)."
            ),
        })

    # 6. Shadow Activity detection (once ambient sources exist)
    if ambient_count > 0:
        # Keyword Unmatch: find high-frequency shell/browser terms not in commitments
        unmatched_projects = _detect_shadow_activity(conn, now)

        gaps.append({
            "gap": "Shadow Activity Detection",
            "severity": "HIGH" if unmatched_projects else "MEDIUM",
            "detail": (
                f"Ambient data available. "
                + (f"{len(unmatched_projects)} high-frequency activity clusters "
                   f"not matching any open commitment — potential 'Missing 20%'."
                   if unmatched_projects
                   else "No obvious unmatched activity detected.")
            ),
            "unmatched_activity": unmatched_projects[:5] if unmatched_projects else None,
            "scout_instructions": (
                "Group knowledgeC by bundle_id, find apps used >30min/day. "
                "Cross-reference with commitment keywords. "
                "Flag browser domains with 3+ visits not matching any graph entity. "
                "Check shell_history for project directories not in commitments."
            ),
        })

    return {
        "weight": "HIGHEST — primary input for Pro Phase 1 Surveyor",
        "description": "What the graph CANNOT answer right now",
        "field_glossary": {
            "gap": "Short title of the epistemic gap",
            "severity": "CRITICAL (system is blind), HIGH (evidence exists but is ambiguous), MEDIUM (low priority uncertainty)",
            "detail": "Explanation of what is unknown and why it matters",
            "related_commitments": "List of open commitments related to this gap",
            "scout_instructions": "Specific queries to run in Phase 2 to investigate this gap",
            "shadow_activity": "Activities detected in shell/browser that do NOT match any open commitment — the 'Missing 20%' of untracked work",
        },
        "gaps": gaps,
    }
