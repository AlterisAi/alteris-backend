"""Stage 1: Deterministic claims extraction.

Produces claims from events using only SQL aggregation and Python logic.
No LLM calls. Every claim links back to its source events for paper trail.

Claim types extracted:
  - communication_frequency: how often user communicates with a person
  - communication_channel: which sources a person appears in
  - directionality: who initiates vs responds
  - timing_pattern: when communication happens (hour-of-day, day-of-week)
  - recency: when the user last communicated with someone
  - thread_activity: active email threads with participants
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone

from loom.confidence import compute_confidence
from loom.constants import (
    DEFAULT_CLAIM_CONFIDENCE,
    SECONDS_PER_DAY,
    USER_TIMEZONE,
)
from loom.models import (
    Claim,
    ExtractionMethod,
    ExtractionProvenance,
    Modality,
)
from loom.privacy import SensitivityLevel
from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_PROVENANCE = ExtractionProvenance(
    model_id="deterministic",
    prompt_version="stage1_v1",
    extraction_method=ExtractionMethod.DETERMINISTIC,
)


def _claim_id(claim_type: str, subject: str, predicate: str) -> str:
    """Deterministic claim ID from its semantic key."""
    raw = f"{claim_type}:{subject}:{predicate}"
    return f"claim:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


_THREAD_SOURCES = ("whatsapp", "imessage", "mail")


def _event_ids_for_person(
    store: LayeredGraphStore,
    person_id: str,
    limit: int = 50,
) -> list[str]:
    """Get event IDs where a person participates (including thread expansion)."""
    rows = store.conn.execute(
        "SELECT event_id FROM person_events WHERE person_id = ? LIMIT ?",
        (person_id, limit),
    ).fetchall()
    return [r["event_id"] for r in rows]


def _setup_user_events_table(
    store: LayeredGraphStore, user_id: str, since_ts: int = 0,
) -> None:
    """Create a temp table of the user's event IDs for efficient joins.

    User participates via:
    - 'self' edges: calendar, granola, calls, slack (per-message)
    - Implicit: WhatsApp, iMessage, mail use thread-level membership,
      so ALL events from these sources in the user's inbox are user events.
    """
    store.conn.execute("DROP TABLE IF EXISTS _user_events")
    if since_ts:
        store.conn.execute(
            """CREATE TEMP TABLE _user_events AS
               SELECT ep.event_id
               FROM event_persons ep
               JOIN events e ON ep.event_id = e.id
               WHERE ep.person_id = ? AND ep.role = 'self'
                 AND e.timestamp >= ?
               UNION
               SELECT id FROM events
               WHERE source IN ('whatsapp', 'imessage', 'mail')
                 AND timestamp >= ?""",
            (user_id, since_ts, since_ts),
        )
    else:
        store.conn.execute(
            """CREATE TEMP TABLE _user_events AS
               SELECT event_id
               FROM event_persons
               WHERE person_id = ? AND role = 'self'
               UNION
               SELECT id FROM events
               WHERE source IN ('whatsapp', 'imessage', 'mail')""",
            (user_id,),
        )
    store.conn.execute(
        "CREATE INDEX IF NOT EXISTS _idx_ue ON _user_events(event_id)"
    )
    count = store.conn.execute("SELECT COUNT(*) FROM _user_events").fetchone()[0]
    logger.info("User events temp table: %d events (since_ts=%d)", count, since_ts)



# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Claim extractors
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_communication_frequency(
    store: LayeredGraphStore, user_id: str,
) -> list[Claim]:
    """Claim: user communicates with person X at frequency Y."""
    rows = store.conn.execute(
        """SELECT pe.person_id, p.canonical_name,
                  COUNT(DISTINCT pe.event_id) as event_count
           FROM person_events pe
           JOIN _user_events ue ON pe.event_id = ue.event_id
           JOIN persons p ON pe.person_id = p.person_id
           WHERE pe.person_id != ?
           GROUP BY pe.person_id
           ORDER BY event_count DESC""",
        (user_id,),
    ).fetchall()

    claims = []
    for r in rows:
        count = r["event_count"]
        name = r["canonical_name"]
        person_id = r["person_id"]

        confidence = compute_confidence("system_sql", count, evidence_scale=20)
        event_ids = _event_ids_for_person(store, person_id)

        claims.append(Claim(
            id=_claim_id("communication_frequency", user_id,
                         f"freq_with:{person_id}"),
            event_ids=event_ids,
            claim_type="communication_frequency",
            subject=user_id,
            predicate=f"communicates_with:{person_id}",
            object=json.dumps({"person_name": name, "event_count": count}),
            confidence=confidence,
            modality=Modality.OBSERVED,
            provenance=_PROVENANCE,
            sensitivity=SensitivityLevel.SENSITIVE,
        ))

    return claims


def _extract_communication_channels(
    store: LayeredGraphStore, user_id: str,
) -> list[Claim]:
    """Claim: user communicates with person X via channels [Y, Z]."""
    rows = store.conn.execute(
        """SELECT pe.person_id, p.canonical_name,
                  GROUP_CONCAT(DISTINCT e.source) as sources,
                  COUNT(DISTINCT e.source) as source_count
           FROM person_events pe
           JOIN _user_events ue ON pe.event_id = ue.event_id
           JOIN persons p ON pe.person_id = p.person_id
           JOIN events e ON pe.event_id = e.id
           WHERE pe.person_id != ?
           GROUP BY pe.person_id
           HAVING source_count >= 1""",
        (user_id,),
    ).fetchall()

    claims = []
    for r in rows:
        sources = r["sources"].split(",")
        person_id = r["person_id"]
        confidence = compute_confidence("system_sql", r["source_count"], evidence_scale=3)
        event_ids = _event_ids_for_person(store, person_id)

        claims.append(Claim(
            id=_claim_id("communication_channel", user_id,
                         f"channels_with:{person_id}"),
            event_ids=event_ids,
            claim_type="communication_channel",
            subject=user_id,
            predicate=f"channels_with:{person_id}",
            object=json.dumps({
                "person_name": r["canonical_name"],
                "channels": sources,
                "channel_count": r["source_count"],
            }),
            confidence=confidence,
            modality=Modality.OBSERVED,
            provenance=_PROVENANCE,
            sensitivity=SensitivityLevel.PRIVATE,
        ))

    return claims


def _extract_directionality(
    store: LayeredGraphStore, user_id: str,
) -> list[Claim]:
    """Claim: communication with person X is N% initiated by user.

    Non-thread sources (calendar, granola, slack): use sender/recipient roles.
    Thread sources (WhatsApp, iMessage, mail): sender role = they sent it,
    is_from_me metadata = user sent it. No recipient edges exist.
    """
    person_dir: dict[str, dict] = defaultdict(
        lambda: {"they_sent": 0, "user_sent": 0, "name": ""}
    )

    # Non-thread sources: existing sender/recipient role logic
    placeholders = ",".join("?" for _ in _THREAD_SOURCES)
    for r in store.conn.execute(
        f"""SELECT ep.person_id, p.canonical_name, ep.role,
                   COUNT(DISTINCT ep.event_id) as cnt
            FROM event_persons ep
            JOIN _user_events ue ON ep.event_id = ue.event_id
            JOIN persons p ON ep.person_id = p.person_id
            JOIN events e ON ep.event_id = e.id
            WHERE ep.person_id != ?
              AND ep.role IN ('sender', 'recipient')
              AND e.source NOT IN ({placeholders})
            GROUP BY ep.person_id, ep.role""",
        [user_id] + list(_THREAD_SOURCES),
    ).fetchall():
        pid = r["person_id"]
        person_dir[pid]["name"] = r["canonical_name"]
        if r["role"] == "sender":
            person_dir[pid]["they_sent"] += r["cnt"]
        else:
            person_dir[pid]["user_sent"] += r["cnt"]

    # Thread sources: they_sent = events where person has sender role
    for r in store.conn.execute(
        f"""SELECT ep.person_id, p.canonical_name,
                   COUNT(DISTINCT ep.event_id) as cnt
            FROM event_persons ep
            JOIN _user_events ue ON ep.event_id = ue.event_id
            JOIN persons p ON ep.person_id = p.person_id
            JOIN events e ON ep.event_id = e.id
            WHERE ep.person_id != ?
              AND ep.role = 'sender'
              AND e.source IN ({placeholders})
            GROUP BY ep.person_id""",
        [user_id] + list(_THREAD_SOURCES),
    ).fetchall():
        pid = r["person_id"]
        person_dir[pid]["they_sent"] += r["cnt"]
        person_dir[pid]["name"] = r["canonical_name"]

    # Thread sources: user_sent = events in shared threads where is_from_me = 1
    for r in store.conn.execute(
        f"""SELECT pe.person_id, COUNT(*) as cnt
            FROM person_events pe
            JOIN events e ON pe.event_id = e.id
            WHERE pe.person_id != ?
              AND e.source IN ({placeholders})
              AND json_extract(e.metadata, '$.is_from_me') = 1
            GROUP BY pe.person_id""",
        [user_id] + list(_THREAD_SOURCES),
    ).fetchall():
        person_dir[r["person_id"]]["user_sent"] += r["cnt"]

    claims = []
    for person_id, data in person_dir.items():
        they_sent = data["they_sent"]
        user_sent = data["user_sent"]
        total = they_sent + user_sent
        if total == 0:
            continue

        user_initiated_ratio = user_sent / total
        event_ids = _event_ids_for_person(store, person_id)

        claims.append(Claim(
            id=_claim_id("directionality", user_id,
                         f"direction_with:{person_id}"),
            event_ids=event_ids,
            claim_type="directionality",
            subject=user_id,
            predicate=f"direction_with:{person_id}",
            object=json.dumps({
                "person_name": data["name"],
                "user_sent": user_sent,
                "they_sent": they_sent,
                "total": total,
                "user_initiated_ratio": round(user_initiated_ratio, 2),
            }),
            confidence=compute_confidence("system_sql", total, evidence_scale=15),
            modality=Modality.OBSERVED,
            provenance=_PROVENANCE,
            sensitivity=SensitivityLevel.SENSITIVE,
        ))

    return claims


def _extract_timing_patterns(
    store: LayeredGraphStore, user_id: str,
) -> list[Claim]:
    """Claim: user communicates with person X primarily during hours [H1, H2]."""
    rows = store.conn.execute(
        """SELECT pe.person_id, p.canonical_name, e.timestamp
           FROM person_events pe
           JOIN _user_events ue ON pe.event_id = ue.event_id
           JOIN persons p ON pe.person_id = p.person_id
           JOIN events e ON pe.event_id = e.id
           WHERE pe.person_id != ?""",
        (user_id,),
    ).fetchall()

    try:
        from zoneinfo import ZoneInfo
        local_tz = ZoneInfo(USER_TIMEZONE)
    except (ImportError, KeyError):
        local_tz = timezone.utc

    person_times: dict[str, dict] = defaultdict(lambda: {
        "name": "", "hours": defaultdict(int),
        "days": defaultdict(int), "count": 0
    })
    for r in rows:
        pid = r["person_id"]
        person_times[pid]["name"] = r["canonical_name"]
        person_times[pid]["count"] += 1
        dt = datetime.fromtimestamp(
            r["timestamp"], tz=timezone.utc,
        ).astimezone(local_tz)
        person_times[pid]["hours"][dt.hour] += 1
        person_times[pid]["days"][dt.strftime("%a")] += 1

    claims = []
    for person_id, data in person_times.items():
        if data["count"] < 3:
            continue

        sorted_hours = sorted(data["hours"].items(), key=lambda x: -x[1])
        peak_hours = [h for h, _ in sorted_hours[:3]]

        sorted_days = sorted(data["days"].items(), key=lambda x: -x[1])
        peak_days = [d for d, _ in sorted_days[:3]]

        business_count = sum(
            c for h, c in data["hours"].items() if 9 <= h <= 17
        )
        total = data["count"]
        business_ratio = business_count / total if total > 0 else 0
        event_ids = _event_ids_for_person(store, person_id)[:50]

        claims.append(Claim(
            id=_claim_id("timing_pattern", user_id,
                         f"timing_with:{person_id}"),
            event_ids=event_ids,
            claim_type="timing_pattern",
            subject=user_id,
            predicate=f"timing_with:{person_id}",
            object=json.dumps({
                "person_name": data["name"],
                "peak_hours": peak_hours,
                "peak_days": peak_days,
                "business_ratio": round(business_ratio, 2),
                "event_count": total,
            }),
            confidence=compute_confidence("system_sql", total, evidence_scale=15),
            modality=Modality.OBSERVED,
            provenance=_PROVENANCE,
            sensitivity=SensitivityLevel.SENSITIVE,
        ))

    return claims


def _extract_recency(
    store: LayeredGraphStore, user_id: str,
) -> list[Claim]:
    """Claim: user last communicated with person X at time T."""
    now = int(time.time())

    # Cap at "now" to exclude future recurring calendar events.
    rows = store.conn.execute(
        """SELECT pe.person_id, p.canonical_name,
                  MAX(CASE WHEN e.timestamp <= ? THEN e.timestamp END) as last_ts,
                  MIN(e.timestamp) as first_ts,
                  COUNT(*) as event_count
           FROM person_events pe
           JOIN _user_events ue ON pe.event_id = ue.event_id
           JOIN persons p ON pe.person_id = p.person_id
           JOIN events e ON pe.event_id = e.id
           WHERE pe.person_id != ?
           GROUP BY pe.person_id""",
        (now, user_id),
    ).fetchall()

    # For people with many events, use 5th-percentile timestamp as "first real
    # contact" to exclude stale calendar imports from years ago. Only consider
    # events within the observation window (timestamp <= now).
    def _robust_first_ts(person_id: str, event_count: int) -> int | None:
        # Count events in observation window only
        cnt_row = store.conn.execute(
            """SELECT COUNT(*) as cnt FROM person_events pe
               JOIN events e ON pe.event_id = e.id
               WHERE pe.person_id = ? AND e.timestamp <= ?""",
            (person_id, now),
        ).fetchone()
        cnt = cnt_row["cnt"]
        if cnt < 20:
            # Few events — use the actual earliest within window
            row = store.conn.execute(
                """SELECT MIN(e.timestamp) as ts FROM person_events pe
                   JOIN events e ON pe.event_id = e.id
                   WHERE pe.person_id = ? AND e.timestamp <= ?""",
                (person_id, now),
            ).fetchone()
            return row["ts"] if row else None
        # 5th-percentile within the observation window
        offset = max(1, cnt // 20)
        row = store.conn.execute(
            """SELECT e.timestamp FROM person_events pe
               JOIN events e ON pe.event_id = e.id
               WHERE pe.person_id = ? AND e.timestamp <= ?
               ORDER BY e.timestamp ASC LIMIT 1 OFFSET ?""",
            (person_id, now, offset),
        ).fetchone()
        return row["timestamp"] if row else None

    claims = []
    for r in rows:
        person_id = r["person_id"]
        last_ts = r["last_ts"]
        if last_ts is None:
            continue
        first_ts = _robust_first_ts(person_id, r["event_count"])
        if first_ts is None:
            continue
        days_since = (now - last_ts) / SECONDS_PER_DAY
        span_days = (last_ts - first_ts) / SECONDS_PER_DAY

        # Recency is a well-evidenced fact (we know when the last contact
        # was). High base confidence that decays with time.
        confidence = compute_confidence(
            "system_sql", 50, recency_days=days_since, half_life_days=30,
        )

        # Most recent events for this person (recency = recent events matter most)
        recent_rows = store.conn.execute(
            """SELECT pe.event_id FROM person_events pe
               JOIN events e ON pe.event_id = e.id
               WHERE pe.person_id = ?
               ORDER BY e.timestamp DESC LIMIT 5""",
            (person_id,),
        ).fetchall()
        event_ids = [rr["event_id"] for rr in recent_rows]

        claims.append(Claim(
            id=_claim_id("recency", user_id,
                         f"recency_with:{person_id}"),
            event_ids=event_ids,
            claim_type="recency",
            subject=user_id,
            predicate=f"last_contact_with:{person_id}",
            object=json.dumps({
                "person_name": r["canonical_name"],
                "last_contact_ts": last_ts,
                "last_contact_iso": datetime.fromtimestamp(
                    last_ts, tz=timezone.utc
                ).isoformat(),
                "first_contact_ts": first_ts,
                "days_since_last": round(days_since, 1),
                "relationship_span_days": round(span_days, 1),
            }),
            confidence=confidence,
            modality=Modality.OBSERVED,
            provenance=_PROVENANCE,
            sensitivity=SensitivityLevel.PRIVATE,
        ))

    return claims


def _extract_thread_activity(
    store: LayeredGraphStore, user_id: str,
) -> list[Claim]:
    """Claim: active email thread about subject X with participants [Y, Z]."""
    # Pre-fetch person name lookup (avoids per-thread queries)
    name_lookup: dict[str, str] = {}
    for r in store.conn.execute(
        "SELECT person_id, canonical_name FROM persons"
    ).fetchall():
        name_lookup[r["person_id"]] = r["canonical_name"]

    rows = store.conn.execute(
        """SELECT json_extract(e.metadata, '$.thread_id') as thread_id,
                  json_extract(e.metadata, '$.subject') as subject,
                  COUNT(*) as msg_count,
                  GROUP_CONCAT(DISTINCT ep.person_id) as participant_ids,
                  MAX(e.timestamp) as last_ts,
                  MIN(e.timestamp) as first_ts,
                  GROUP_CONCAT(e.id) as event_id_list
           FROM events e
           JOIN _user_events ue ON e.id = ue.event_id
           LEFT JOIN event_persons ep ON e.id = ep.event_id
             AND ep.person_id != ?
             AND ep.role != 'identity'
           WHERE e.source = 'mail'
             AND json_extract(e.metadata, '$.thread_id') IS NOT NULL
             AND json_extract(e.metadata, '$.thread_id') != ''
           GROUP BY json_extract(e.metadata, '$.thread_id')
           HAVING msg_count >= 2
           ORDER BY last_ts DESC""",
        (user_id,),
    ).fetchall()

    claims = []
    for r in rows:
        thread_id = r["thread_id"] or ""
        subject = r["subject"] or "(no subject)"
        participant_ids = [
            p for p in (r["participant_ids"] or "").split(",") if p
        ]

        participant_names = [
            name_lookup[pid]
            for pid in participant_ids[:10]
            if pid in name_lookup
        ]

        # Use event IDs from the GROUP_CONCAT (already fetched)
        all_eids = (r["event_id_list"] or "").split(",")
        event_ids = list(dict.fromkeys(all_eids))[:50]  # dedup, keep order

        claims.append(Claim(
            id=_claim_id("thread_activity", user_id,
                         f"thread:{thread_id}"),
            event_ids=event_ids,
            claim_type="thread_activity",
            subject=user_id,
            predicate=f"active_thread:{thread_id[:40]}",
            object=json.dumps({
                "subject": subject,
                "message_count": r["msg_count"],
                "participants": participant_names,
                "last_activity_ts": r["last_ts"],
                "first_activity_ts": r["first_ts"],
                "thread_id": thread_id,
            }),
            confidence=compute_confidence("system_sql", r["msg_count"], evidence_scale=10),
            modality=Modality.OBSERVED,
            provenance=_PROVENANCE,
            sensitivity=SensitivityLevel.SENSITIVE,
        ))

    return claims


def _extract_topic_affinity(
    store: LayeredGraphStore, user_id: str,
) -> list[Claim]:
    """Claim: person X has topic affinity Y based on shared event annotations.

    Pure SQL: for each (person, topic) pair with >=2 shared events, emit a claim.
    Topics come from annotations table (facet='topic').
    """
    rows = store.conn.execute(
        """SELECT pe.person_id, p.canonical_name,
                  a.value AS topic,
                  COUNT(DISTINCT a.event_id) AS event_count,
                  GROUP_CONCAT(DISTINCT e.source) AS sources
           FROM person_events pe
           JOIN _user_events ue ON pe.event_id = ue.event_id
           JOIN persons p ON pe.person_id = p.person_id
           JOIN annotations a ON pe.event_id = a.event_id
           JOIN events e ON pe.event_id = e.id
           WHERE pe.person_id != ?
             AND a.facet = 'topic'
           GROUP BY pe.person_id, a.value
           HAVING event_count >= 2
           ORDER BY event_count DESC""",
        (user_id,),
    ).fetchall()

    claims = []
    for r in rows:
        person_id = r["person_id"]
        topic = r["topic"]
        event_count = r["event_count"]
        sources = (r["sources"] or "").split(",")

        event_ids = _event_ids_for_person(store, person_id)

        claims.append(Claim(
            id=_claim_id("topic_affinity", person_id, f"topic:{topic}"),
            event_ids=event_ids[:20],
            claim_type="topic_affinity",
            subject=person_id,
            predicate=f"has_topic_affinity:{topic}",
            object=json.dumps({
                "person_name": r["canonical_name"],
                "topic": topic,
                "event_count": event_count,
                "sources": sources,
            }),
            confidence=compute_confidence("system_sql", event_count, evidence_scale=10),
            modality=Modality.OBSERVED,
            provenance=_PROVENANCE,
            sensitivity=SensitivityLevel.PRIVATE,
        ))

    return claims


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main extraction pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_stage1_claims(
    store: LayeredGraphStore,
    user_person_id: str | None = None,
    since_ts: int = 0,
) -> dict[str, int | dict | str | float]:
    """Run all Stage 1 deterministic claim extractors.

    Args:
        store: Graph store with events, persons, and event_persons populated
        user_person_id: The user's person_id. Auto-detected if not provided.
        since_ts: Only consider events with timestamp >= this value (0 = all).
            Default 30 days is applied by the CLI, not here.

    Returns:
        {total_claims, by_type: {type: count}, new_claims, existing_claims}
    """
    if not user_person_id:
        row = store.conn.execute(
            "SELECT person_id FROM persons WHERE is_user = 1 LIMIT 1"
        ).fetchone()
        if row:
            user_person_id = row["person_id"]
    if not user_person_id:
        return {"error": "User not found in persons table"}

    t0 = time.time()

    # Pre-materialize user's event IDs into a temp table.
    # Eliminates the O(E²) self-join in every extractor.
    _setup_user_events_table(store, user_person_id, since_ts)

    # Verify person_events table is populated (built by linker)
    pe_count = store.conn.execute("SELECT COUNT(*) FROM person_events").fetchone()[0]
    if pe_count == 0:
        from loom.linker import rebuild_person_events
        logger.warning("person_events empty, rebuilding...")
        rebuild_person_events(store)

    extractors = [
        ("communication_frequency", _extract_communication_frequency),
        ("communication_channel", _extract_communication_channels),
        ("directionality", _extract_directionality),
        ("timing_pattern", _extract_timing_patterns),
        ("recency", _extract_recency),
        ("thread_activity", _extract_thread_activity),
        ("topic_affinity", _extract_topic_affinity),
    ]

    all_claims: list[Claim] = []
    by_type: dict[str, int] = {}

    for name, fn in extractors:
        claims = fn(store, user_person_id)
        all_claims.extend(claims)
        by_type[name] = len(claims)
        logger.info("Extractor %s: %d claims", name, len(claims))

    new_claims = 0
    existing_claims = 0
    for claim in all_claims:
        if store.put_claim(claim, commit=False):
            new_claims += 1
        else:
            existing_claims += 1
    store.conn.commit()

    elapsed = time.time() - t0
    logger.info(
        "Stage 1: %d claims extracted (%d new, %d existing) in %.1fs",
        len(all_claims), new_claims, existing_claims, elapsed,
    )

    return {
        "total_claims": len(all_claims),
        "by_type": by_type,
        "new_claims": new_claims,
        "existing_claims": existing_claims,
        "duration_seconds": round(elapsed, 2),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Person profiles
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def populate_person_profiles(store: LayeredGraphStore) -> dict[str, int]:
    """Build person profiles from Stage 1 claims and persist to person_profiles table.

    Reads the same 4 claim types that extract.build_person_profiles() used to
    reconstruct ephemerally. Writes directly to the person_profiles table so
    downstream modules can query cheaply.

    Returns:
        {profiles_written, tier_changes} — count of profiles written and
        tier transitions detected.
    """
    now_ts = int(time.time())

    def _extract_person_id(predicate: str) -> str | None:
        if ":" not in predicate:
            return None
        parts = predicate.split(":", 1)
        return parts[1] if len(parts) > 1 else None

    profiles: dict[str, dict] = {}

    # Communication frequency → msg_count, tier, name
    freq_rows = store.conn.execute(
        """SELECT predicate, object FROM claims
           WHERE claim_type = 'communication_frequency'
             AND (superseded_by IS NULL OR superseded_by = '')"""
    ).fetchall()
    for r in freq_rows:
        pid = _extract_person_id(r["predicate"])
        if not pid:
            continue
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue
        msg_count = obj.get("event_count", 0)
        if msg_count == 0:
            continue

        if msg_count >= 50:
            tier = 1
        elif msg_count >= 20:
            tier = 2
        elif msg_count >= 5:
            tier = 3
        else:
            tier = 4

        profiles[pid] = {
            "person_id": pid,
            "canonical_name": obj.get("person_name", pid),
            "message_count": msg_count,
            "tier": tier,
        }

    # Directionality → user_initiated_ratio
    dir_rows = store.conn.execute(
        """SELECT predicate, object FROM claims
           WHERE claim_type = 'directionality'
             AND (superseded_by IS NULL OR superseded_by = '')"""
    ).fetchall()
    for r in dir_rows:
        pid = _extract_person_id(r["predicate"])
        if not pid or pid not in profiles:
            continue
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue
        profiles[pid]["user_initiated_ratio"] = obj.get("user_initiated_ratio", 0.5)

    # Recency → days_since_last, first_contact_ts, last_contact_ts, relationship_span_days
    rec_rows = store.conn.execute(
        """SELECT predicate, object FROM claims
           WHERE claim_type = 'recency'
             AND (superseded_by IS NULL OR superseded_by = '')"""
    ).fetchall()
    for r in rec_rows:
        pid = _extract_person_id(r["predicate"])
        if not pid or pid not in profiles:
            continue
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue
        profiles[pid]["days_since_last"] = obj.get("days_since_last")
        profiles[pid]["first_contact_ts"] = obj.get("first_contact_ts")
        profiles[pid]["last_contact_ts"] = obj.get("last_contact_ts")
        profiles[pid]["relationship_span_days"] = obj.get("relationship_span_days")

    # Communication channel → channels[], channel_count
    chan_rows = store.conn.execute(
        """SELECT predicate, object FROM claims
           WHERE claim_type = 'communication_channel'
             AND (superseded_by IS NULL OR superseded_by = '')"""
    ).fetchall()
    for r in chan_rows:
        pid = _extract_person_id(r["predicate"])
        if not pid or pid not in profiles:
            continue
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue
        channels = obj.get("channels", [])
        profiles[pid]["channels"] = channels
        profiles[pid]["channel_count"] = len(channels)

    # Mark user profiles
    user_row = store.conn.execute(
        "SELECT person_id FROM persons WHERE is_user = 1 LIMIT 1"
    ).fetchone()
    user_pid = user_row["person_id"] if user_row else None

    for pid, prof in profiles.items():
        prof["is_user"] = (pid == user_pid)

    # Batch upsert
    profile_list = list(profiles.values())
    written = store.upsert_person_profiles_batch(profile_list) if profile_list else 0

    # Count tier changes
    tier_changes = 0
    for prof in store.get_person_profiles():
        if prof["previous_tier"] is not None and prof["previous_tier"] != prof["tier"]:
            tier_changes += 1
            logger.info(
                "Tier change: %s (%s) tier %d → %d",
                prof["canonical_name"], prof["person_id"],
                prof["previous_tier"], prof["tier"],
            )

    logger.info("Person profiles: %d populated (%d tier changes)", written, tier_changes)
    return {"profiles_written": written, "tier_changes": tier_changes}
