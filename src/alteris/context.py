"""Layer 1 pre-triage context enrichment.

Builds deterministic context blocks from events, persons, and person_events.
No LLM calls, no triage claims as input (that would be circular).

Three enrichment functions:
  build_contact_dossier()  — who is this person (behavioral signals)
  build_thread_snapshot()  — thread state header (velocity, response state)
  build_day_context()      — what the user's day looks like

Plus compact variants for local model (Qwen 2K context budget):
  build_compact_contact()  — one-line contact summary
  build_compact_thread()   — one-line thread summary

Philosophy: present observable signals, let the LLM infer relationship
type and importance. No relationship labels, no topic inference — those
come from triage output (Layer 2).
"""

from __future__ import annotations

import json
import logging
import time
import zoneinfo
from collections import defaultdict
from datetime import datetime

from alteris.constants import (
    CONTACT_DOSSIER_TIMING_LIMIT,
    DAY_CONTEXT_MAX_EVENTS,
    THREAD_SNAPSHOT_MAX_NODES,
    USER_TIMEZONE,
    safe_timezone,
)
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)

# Source abbreviations for token-efficient timing histograms
_SOURCE_ABBREV = {
    "mail": "em", "imessage": "im", "whatsapp": "wa",
    "slack": "sl", "granola": "mtg", "calendar": "cal",
}

# Compact hour slots (7 buckets, not 24 — token efficient)
_HOUR_SLOTS = [
    (0, 6, "12a-6a"),
    (6, 9, "6-9a"),
    (9, 12, "9a-12p"),
    (12, 15, "12-3p"),
    (15, 18, "3-6p"),
    (18, 21, "6-9p"),
    (21, 24, "9p-12a"),
]

_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

_GENERIC_DOMAINS = frozenset({
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "icloud.com", "me.com", "aol.com", "protonmail.com",
    "live.com", "msn.com",
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Internal helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_user_person_id(store: LayeredGraphStore) -> str | None:
    row = store.conn.execute(
        "SELECT person_id FROM persons WHERE is_user = 1 LIMIT 1"
    ).fetchone()
    return row["person_id"] if row else None


def _format_span(days: float) -> str:
    """Format a day count as human-readable span."""
    if days < 1:
        return f"{max(1, int(days * 24))}h"
    if days < 30:
        return f"{int(days)}d"
    months = int(days / 30)
    if months >= 12:
        years = months // 12
        rem = months % 12
        return f"{years}y {rem}m" if rem else f"{years}y"
    return f"{months}m"


def _format_ago(hours: float) -> str:
    """Format hours-ago as human-readable string."""
    if hours < 1:
        return f"{max(1, int(hours * 60))}m ago"
    if hours < 48:
        return f"{hours:.0f}h ago"
    return f"{hours / 24:.0f}d ago"


def _resolve_sender_name(
    event_id: str,
    store: LayeredGraphStore,
    user_pid: str | None,
) -> str:
    """Resolve the non-user sender name for an event."""
    persons = store.get_persons_for_event(event_id)
    for pid, role in persons:
        if pid != user_pid and role == "sender":
            p = store.get_person(pid)
            return (p or {}).get("canonical_name", pid[:12])
    # Fallback: any non-user participant
    for pid, role in persons:
        if pid != user_pid:
            p = store.get_person(pid)
            return (p or {}).get("canonical_name", pid[:12])
    return "them"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Timing histogram
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_timing_histogram(
    interactions: list[dict],
    user_tz: str = USER_TIMEZONE,
) -> str | None:
    """Build adaptive temporal interaction histograms.

    interactions: list of {timestamp, source, is_outbound}.
    Returns multi-line text or None if too few data points.

    Adaptive granularity:
      always   → hourly (7 slots) + daily (Mon-Sun)
      ≥30 days → + monthly
      ≥365 days → + yearly
    """
    if len(interactions) < 3:
        return None

    tz = safe_timezone(user_tz)

    first_ts = interactions[0]["timestamp"]
    last_ts = interactions[-1]["timestamp"]
    span_days = max(1, (last_ts - first_ts) / 86400)

    hourly: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    daily: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    monthly: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    yearly: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for r in interactions:
        src_abbr = _SOURCE_ABBREV.get(r["source"], r["source"][:3])
        direction = "↑" if r["is_outbound"] else "↓"
        dt = datetime.fromtimestamp(r["timestamp"], tz=tz)

        hour = dt.hour
        for start_h, end_h, label in _HOUR_SLOTS:
            if start_h <= hour < end_h:
                hourly[label][src_abbr] += 1
                hourly[label][direction] += 1
                hourly[label]["_total"] += 1
                break

        dow_name = _DAY_NAMES[dt.weekday()]
        daily[dow_name][src_abbr] += 1
        daily[dow_name][direction] += 1
        daily[dow_name]["_total"] += 1

        mon_name = _MONTH_NAMES[dt.month - 1]
        monthly[mon_name][src_abbr] += 1
        monthly[mon_name][direction] += 1
        monthly[mon_name]["_total"] += 1

        yr_key = str(dt.year)
        yearly[yr_key][src_abbr] += 1
        yearly[yr_key][direction] += 1
        yearly[yr_key]["_total"] += 1

    def _fmt_line(
        buckets: dict[str, dict[str, int]],
        keys: list[str],
        label: str,
    ) -> str | None:
        parts = []
        for k in keys:
            b = buckets.get(k)
            if not b or b.get("_total", 0) == 0:
                continue
            total = b["_total"]
            channels = []
            for ck, cv in sorted(b.items(), key=lambda x: -x[1]):
                if ck in ("↑", "↓", "_total"):
                    continue
                channels.append(f"{ck}:{cv}")
            ch_str = ",".join(channels)
            out_n = b.get("↑", 0)
            in_n = b.get("↓", 0)
            parts.append(f"{k}={total}({ch_str} ↑{out_n}↓{in_n})")
        return f"    {label}: {' | '.join(parts)}" if parts else None

    lines: list[str] = []

    h_line = _fmt_line(hourly, [lbl for _, _, lbl in _HOUR_SLOTS], "By hour")
    if h_line:
        lines.append(h_line)

    d_line = _fmt_line(daily, _DAY_NAMES, "By day")
    if d_line:
        lines.append(d_line)

    if span_days >= 30:
        m_line = _fmt_line(monthly, _MONTH_NAMES, "By month")
        if m_line:
            lines.append(m_line)

    if span_days >= 365:
        y_line = _fmt_line(yearly, sorted(yearly.keys()), "By year")
        if y_line:
            lines.append(y_line)

    if not lines:
        return None

    # Build legend for abbreviations actually used
    all_abbrevs: set[str] = set()
    for bucket_dict in (hourly, daily, monthly, yearly):
        for b in bucket_dict.values():
            for k in b:
                if k not in ("↑", "↓", "_total"):
                    all_abbrevs.add(k)

    full_labels = {
        "em": "email", "im": "iMessage", "wa": "WhatsApp",
        "sl": "Slack", "mtg": "meeting", "cal": "calendar",
    }
    legend_parts = [
        f"{a}={full_labels.get(a, a)}" for a in sorted(all_abbrevs)
    ]
    legend = f"  Timing ({', '.join(legend_parts)}, ↑=user sent ↓=received):"
    lines.insert(0, legend)

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Contact Dossier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_contact_dossier(
    person_id: str,
    store: LayeredGraphStore,
    *,
    now: int | None = None,
    window_days: int | None = None,
) -> str | None:
    """Build a compact contact dossier for LLM injection.

    Returns multi-line text block of observable behavioral signals,
    or None if person is the user or unknown.

    No relationship inference — the LLM does that from these signals.

    Args:
        window_days: Limit stats to the most recent N days. Defaults to
            None (all-time). Pass e.g. 30 for eval comparisons.
            History span and timing histogram always use all-time data.

    Example output::

        CONTACT: Sam Park (tier 1)
          Channels: WhatsApp 4086 | iMessage 45 | email 38 | Slack 12
          Pattern: daily, balanced (48% outbound), 4181 total msgs
          Recent: 23 msgs in last 48h (elevated vs 8/day baseline)
          Domains: loom.example.com
          History: first contact 11m ago
          Timing (em=email, wa=WhatsApp, ↑=user sent ↓=received):
            By hour: 6-9a=8(...) | 9a-12p=15(...) | ...
            By day: Mon=45(...) | Tue=52(...) | ...
    """
    now = now or int(time.time())

    person = store.get_person(person_id)
    if not person or person.get("is_user"):
        return None

    name = person.get("canonical_name") or person_id[:16]
    user_pid = _get_user_person_id(store)

    # ── Aggregated stats via SQL (no row-count cap) ──
    comm_sources = ("mail", "imessage", "whatsapp", "slack", "granola")
    placeholders = ",".join("?" for _ in comm_sources)

    # Window filter: limit stats to recent N days (tier/freq/baseline),
    # but always fetch all-time first_ts for history span.
    window_cutoff = (now - window_days * 86400) if window_days else 0

    agg_rows = store.conn.execute(
        f"""SELECT e.source,
                   COUNT(*) as cnt,
                   SUM(CASE WHEN json_extract(e.metadata, '$.is_from_me') THEN 1 ELSE 0 END) as sent,
                   MIN(e.timestamp) as first_ts,
                   MAX(e.timestamp) as last_ts
            FROM events e
            JOIN person_events pe ON e.id = pe.event_id
            WHERE pe.person_id = ?
              AND e.source IN ({placeholders})
              AND e.timestamp >= ?
            GROUP BY e.source""",
        (person_id, *comm_sources, window_cutoff),
    ).fetchall()

    if not agg_rows:
        return None

    source_counts: dict[str, int] = {}
    source_sent: dict[str, int] = {}
    total = 0
    sent = 0
    for r in agg_rows:
        source_counts[r["source"]] = r["cnt"]
        source_sent[r["source"]] = r["sent"]
        total += r["cnt"]
        sent += r["sent"]
    recv = total - sent

    # ── Tier + history span from person_profiles (consistent with gate) ──
    profile = store.get_person_profile(person_id)
    if profile:
        tier = profile["tier"]
        first_ever_ts = profile.get("first_contact_ts") or now
        history_days = max(1, (now - first_ever_ts) / 86400)
    else:
        if total >= 50:
            tier = 1
        elif total >= 10:
            tier = 2
        else:
            tier = 3
        first_ever_ts = store.conn.execute(
            f"""SELECT MIN(e.timestamp)
                FROM events e
                JOIN person_events pe ON e.id = pe.event_id
                WHERE pe.person_id = ?
                  AND e.source IN ({placeholders})""",
            (person_id, *comm_sources),
        ).fetchone()[0] or now
        history_days = max(1, (now - first_ever_ts) / 86400)

    # ── Communication pattern (based on window, not all-time) ──
    stats_span_days = window_days if window_days else max(1, history_days)
    daily_baseline = total / stats_span_days
    if daily_baseline >= 1.0:
        frequency = "daily"
    elif daily_baseline >= 0.15:
        frequency = "weekly"
    elif daily_baseline >= 0.03:
        frequency = "monthly"
    else:
        frequency = "rare"

    outbound_pct = (sent / total * 100) if total > 0 else 0
    if 35 <= outbound_pct <= 65:
        balance = f"balanced ({outbound_pct:.0f}% outbound)"
    elif outbound_pct > 65:
        balance = f"mostly outbound ({outbound_pct:.0f}%)"
    else:
        balance = f"mostly inbound ({100 - outbound_pct:.0f}% from them)"

    # ── Recent activity (48h) ──
    cutoff_48h = now - 48 * 3600
    recent_count = store.conn.execute(
        f"""SELECT COUNT(*) FROM events e
            JOIN person_events pe ON e.id = pe.event_id
            WHERE pe.person_id = ?
              AND e.source IN ({placeholders})
              AND e.timestamp >= ?""",
        (person_id, *comm_sources, cutoff_48h),
    ).fetchone()[0]
    recent_daily = recent_count / 2.0
    activity_level = (
        "elevated"
        if recent_daily > daily_baseline * 1.5 and recent_count > 3
        else "normal"
    )

    # ── Domains and aliases from person_identifiers ──
    id_rows = store.get_person_identifiers(person_id)
    domains: set[str] = set()
    aliases: set[str] = set()
    for idr in id_rows:
        if idr["identifier_type"] == "email":
            domain = idr["identifier"].split("@")[-1].lower()
            if domain not in _GENERIC_DOMAINS:
                domains.add(domain)
        if idr["identifier_type"] == "display_name":
            alias = idr["identifier"]
            if alias and alias != name:
                aliases.add(alias)

    # ── Shared surname ──
    user_person = store.get_person(user_pid) if user_pid else None
    user_name = (user_person or {}).get("canonical_name", "")
    user_last = user_name.split()[-1].lower() if user_name else ""
    shared_surname = bool(user_last and user_last in name.lower().split())

    # ── Format channels ──
    channel_parts = []
    for src in ("whatsapp", "imessage", "mail", "slack", "granola"):
        cnt = source_counts.get(src, 0)
        if cnt > 0:
            label = {"mail": "email", "imessage": "iMessage"}.get(
                src, src.capitalize()
            )
            channel_parts.append(f"{label} {cnt}")
    channels_str = " | ".join(channel_parts) if channel_parts else "unknown"

    # ── Build output ──
    lines = [f"CONTACT: {name} (tier {tier})"]

    if aliases:
        lines.append(f"  Also known as: {', '.join(sorted(aliases))}")

    lines.append(f"  Channels: {channels_str}")
    window_note = f" (last {window_days}d)" if window_days else ""
    lines.append(f"  Pattern: {frequency}, {balance}, {total} msgs{window_note}")

    if recent_count > 0:
        recent_str = f"  Recent: {recent_count} msgs in last 48h ({activity_level}"
        if activity_level == "elevated":
            recent_str += f" vs {daily_baseline:.0f}/day baseline"
        recent_str += ")"
        lines.append(recent_str)

    if domains:
        lines.append(f"  Domains: {', '.join(sorted(domains))}")
    if shared_surname:
        lines.append(f"  Note: shares surname with user ({user_last.title()})")
    lines.append(f"  History: first contact {_format_span(history_days)} ago")

    # ── Timing histogram (separate query for full timestamp coverage) ──
    timing_rows = store.conn.execute(
        f"""SELECT e.timestamp, e.source,
                   json_extract(e.metadata, '$.is_from_me') as is_from_me
            FROM events e
            JOIN person_events pe ON e.id = pe.event_id
            WHERE pe.person_id = ?
              AND e.source IN ({placeholders})
            ORDER BY e.timestamp ASC
            LIMIT ?""",
        (person_id, *comm_sources, CONTACT_DOSSIER_TIMING_LIMIT),
    ).fetchall()
    histogram_data = [
        {
            "timestamp": r["timestamp"],
            "source": r["source"],
            "is_outbound": bool(r["is_from_me"]),
        }
        for r in timing_rows
    ]
    histogram = _build_timing_histogram(histogram_data)
    if histogram:
        lines.append(histogram)

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thread Snapshot (header / orientation block)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_thread_snapshot(
    thread_id: str,
    store: LayeredGraphStore,
    *,
    now: int | None = None,
) -> str | None:
    """Build a thread state snapshot for LLM orientation.

    Returns a stats header block: velocity, response state, participants,
    last 5 messages as preview. This is the HEADER only — for THREAD_FULL
    strategy, the complete message history follows below this block.

    Example output::

        THREAD: 230 msgs over 14 days, 2 participants
          Participants: USER, Sam Park
          Velocity: 10.6 msgs/day this week (elevated, baseline: 3.4/day)
          Last USER msg: 31h ago
          Awaiting response: last msg from Sam Park (25h ago)
          Recent (last 5):
            [02/12 09:14] Sam Park: Can you review the deck before the call?
            [02/12 09:15] USER: Looking at it now
            ...
    """
    now = now or int(time.time())

    if not thread_id:
        return None

    # Total count (cheap index scan once thread_id is a column;
    # for now uses json_extract on metadata)
    total_msgs = store.conn.execute(
        """SELECT COUNT(*) FROM events
           WHERE json_extract(metadata, '$.thread_id') = ?""",
        (thread_id,),
    ).fetchone()[0]

    if total_msgs < 2:
        return None

    # Fetch recent messages for stats + preview
    nodes = store.conn.execute(
        """SELECT e.id, e.timestamp, e.source, e.raw_content,
                  json_extract(e.metadata, '$.is_from_me') as is_from_me
           FROM events e
           WHERE json_extract(e.metadata, '$.thread_id') = ?
           ORDER BY e.timestamp DESC
           LIMIT ?""",
        (thread_id, THREAD_SNAPSHOT_MAX_NODES),
    ).fetchall()
    nodes = list(reversed(nodes))  # chronological

    user_pid = _get_user_person_id(store)

    # ── Timeline ──
    first_ts = store.conn.execute(
        """SELECT MIN(timestamp) FROM events
           WHERE json_extract(metadata, '$.thread_id') = ?""",
        (thread_id,),
    ).fetchone()[0] or now
    last_ts = nodes[-1]["timestamp"] if nodes else now
    span_days = max(1, (last_ts - first_ts) / 86400)

    # ── Velocity ──
    cutoff_7d = now - 7 * 86400
    recent_msgs = store.conn.execute(
        """SELECT COUNT(*) FROM events
           WHERE json_extract(metadata, '$.thread_id') = ?
             AND timestamp >= ?""",
        (thread_id, cutoff_7d),
    ).fetchone()[0]
    recent_velocity = recent_msgs / 7.0
    baseline_velocity = total_msgs / span_days

    velocity_str = f"{recent_velocity:.1f} msgs/day this week"
    if baseline_velocity > 0 and recent_velocity > baseline_velocity * 1.5:
        velocity_str += f" (elevated, baseline: {baseline_velocity:.1f}/day)"
    elif baseline_velocity > 0.1:
        velocity_str += f" (baseline: {baseline_velocity:.1f}/day)"

    # ── Participants (resolved names via person_events) ──
    part_rows = store.conn.execute(
        """SELECT p.canonical_name, p.is_user, pe.person_id,
                  COUNT(DISTINCT pe.event_id) as cnt
           FROM person_events pe
           JOIN persons p ON pe.person_id = p.person_id
           JOIN events e ON pe.event_id = e.id
           WHERE json_extract(e.metadata, '$.thread_id') = ?
           GROUP BY pe.person_id
           ORDER BY cnt DESC
           LIMIT 10""",
        (thread_id,),
    ).fetchall()

    participant_names = []
    for pr in part_rows:
        if pr["is_user"]:
            participant_names.append("USER")
        else:
            participant_names.append(pr["canonical_name"] or pr["person_id"][:12])

    # ── Response state ──
    last_user_ts = None
    last_other_ts = None
    last_other_name = None

    for n in reversed(nodes):
        is_from_me = bool(n["is_from_me"])
        if is_from_me and last_user_ts is None:
            last_user_ts = n["timestamp"]
        elif not is_from_me and last_other_ts is None:
            last_other_ts = n["timestamp"]
            last_other_name = _resolve_sender_name(n["id"], store, user_pid)
        if last_user_ts is not None and last_other_ts is not None:
            break

    # Determine thread status
    last_is_from_me = bool(nodes[-1]["is_from_me"]) if nodes else False

    # ── Build output ──
    span_label = (
        f"{span_days:.0f} days" if span_days >= 1
        else f"{max(1, int(span_days * 24))} hours"
    )

    lines = [
        f"THREAD: {total_msgs} msgs over {span_label}, "
        f"{len(participant_names)} participants",
    ]
    if participant_names:
        lines.append(f"  Participants: {', '.join(participant_names[:6])}")
    lines.append(f"  Velocity: {velocity_str}")

    if last_user_ts is not None:
        user_hours_ago = (now - last_user_ts) / 3600
        lines.append(f"  Last USER msg: {_format_ago(user_hours_ago)}")

    if not last_is_from_me and last_other_ts is not None:
        other_hours_ago = (now - last_other_ts) / 3600
        who = last_other_name or "them"
        lines.append(
            f"  Awaiting response: last msg from {who} "
            f"({_format_ago(other_hours_ago)})"
        )

    # ── Last 5 messages (orientation preview, not content limit) ──
    preview_nodes = nodes[-5:] if len(nodes) >= 5 else list(nodes)
    if preview_nodes:
        tz = safe_timezone()
        lines.append("  Recent (last 5):")
        for n in preview_nodes:
            dt = datetime.fromtimestamp(n["timestamp"], tz=tz)
            time_str = dt.strftime("%m/%d %H:%M")
            if bool(n["is_from_me"]):
                who = "USER"
            else:
                who = _resolve_sender_name(n["id"], store, user_pid)
            body = (n["raw_content"] or "")[:80].replace("\n", " ").strip()
            if body:
                lines.append(f"    [{time_str}] {who}: {body}")

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Day Context
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_day_context(
    store: LayeredGraphStore,
    target_date: int | None = None,
    *,
    user_tz: str = USER_TIMEZONE,
) -> str:
    """Build a snapshot of the user's day for LLM orientation.

    Computed once per triage batch (not per event).

    Example output::

        DAY CONTEXT (Wed Feb 12):
          Calendar: 3 events (next: "1:1 with Sam" at 10:30 AM)
          Inbox: 12 emails, 8 iMessages, 5 WhatsApp today
          Active threads: 7 with activity in last 24h
    """
    tz = safe_timezone(user_tz)

    if target_date:
        now_local = datetime.fromtimestamp(target_date, tz=tz)
    else:
        now_local = datetime.now(tz)

    now_ts = int(now_local.timestamp())

    today_start = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_ts = int(today_start.timestamp())
    today_end_ts = today_start_ts + 86400

    # ── Calendar events today ──
    cal_events = store.conn.execute(
        """SELECT raw_content, timestamp, metadata FROM events
           WHERE source = 'calendar'
             AND timestamp >= ? AND timestamp < ?
           ORDER BY timestamp ASC
           LIMIT ?""",
        (today_start_ts, today_end_ts, DAY_CONTEXT_MAX_EVENTS),
    ).fetchall()

    next_event_str = None
    for e in cal_events:
        if e["timestamp"] >= now_ts:
            meta = json.loads(e["metadata"] or "{}")
            subject = meta.get("subject", "") or (e["raw_content"] or "")[:50]
            evt_time = datetime.fromtimestamp(e["timestamp"], tz=tz)
            next_event_str = f'"{subject}" at {evt_time.strftime("%I:%M %p")}'
            break

    # ── Inbox volume by source ──
    volume = store.conn.execute(
        """SELECT source, COUNT(*) as cnt FROM events
           WHERE timestamp >= ? AND timestamp < ?
             AND source NOT IN ('calendar', 'contacts')
           GROUP BY source""",
        (today_start_ts, today_end_ts),
    ).fetchall()

    source_labels = {
        "mail": "emails", "imessage": "iMessages", "whatsapp": "WhatsApp",
        "slack": "Slack", "granola": "meetings",
    }
    volume_parts = []
    for v in volume:
        label = source_labels.get(v["source"], v["source"])
        volume_parts.append(f"{v['cnt']} {label}")

    # ── Active threads (24h) ──
    cutoff_24h = now_ts - 24 * 3600
    active_threads = store.conn.execute(
        """SELECT COUNT(DISTINCT json_extract(metadata, '$.thread_id'))
           FROM events
           WHERE json_extract(metadata, '$.thread_id') IS NOT NULL
             AND json_extract(metadata, '$.thread_id') != ''
             AND timestamp >= ?""",
        (cutoff_24h,),
    ).fetchone()[0]

    # ── Build output ──
    day_name = now_local.strftime("%a %b %d")
    lines = [f"DAY CONTEXT ({day_name}):"]

    if cal_events:
        cal_str = (
            f"  Calendar: {len(cal_events)} "
            f"event{'s' if len(cal_events) != 1 else ''}"
        )
        if next_event_str:
            cal_str += f" (next: {next_event_str})"
        lines.append(cal_str)
    else:
        lines.append("  Calendar: no events today")

    if volume_parts:
        lines.append(f"  Inbox: {', '.join(volume_parts)} today")

    lines.append(f"  Active threads: {active_threads} with activity in last 24h")

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Compact headers (for local model / Qwen 2K budget)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_compact_contact(
    person_id: str,
    store: LayeredGraphStore,
    *,
    now: int | None = None,
    window_days: int | None = None,
) -> str | None:
    """One-line contact summary for local model (~50 tokens).

    Args:
        window_days: Limit stats to most recent N days. Defaults to
            None (all-time). Pass e.g. 30 for eval comparisons.

    Example::

        CONTACT: Sam Park | tier:1 | msgs:4181 | channels:wa,im,em,sl
        | outbound:48% | recent:23/48h | span:11mo | freq:daily
    """
    now = now or int(time.time())

    person = store.get_person(person_id)
    if not person or person.get("is_user"):
        return None

    name = person.get("canonical_name") or person_id[:16]

    comm_sources = ("mail", "imessage", "whatsapp", "slack", "granola")
    placeholders = ",".join("?" for _ in comm_sources)
    window_cutoff = (now - window_days * 86400) if window_days else 0

    agg = store.conn.execute(
        f"""SELECT COUNT(*) as total,
                   SUM(CASE WHEN json_extract(e.metadata, '$.is_from_me') THEN 1 ELSE 0 END) as sent,
                   GROUP_CONCAT(DISTINCT e.source) as sources
            FROM events e
            JOIN person_events pe ON e.id = pe.event_id
            WHERE pe.person_id = ?
              AND e.source IN ({placeholders})
              AND e.timestamp >= ?""",
        (person_id, *comm_sources, window_cutoff),
    ).fetchone()

    total = agg["total"] or 0
    if not total:
        return None

    sent = agg["sent"] or 0
    outbound_pct = int(sent / total * 100) if total else 0

    sources = sorted(agg["sources"].split(",")) if agg["sources"] else []
    ch_short = ",".join(_SOURCE_ABBREV.get(s, s[:2]) for s in sources)

    # Tier + history span from person_profiles (consistent with gate)
    profile = store.get_person_profile(person_id)
    if profile:
        tier = profile["tier"]
        first_ts = profile.get("first_contact_ts") or now
    else:
        tier = 1 if total >= 50 else (2 if total >= 10 else 3)
        first_ts = store.conn.execute(
            f"""SELECT MIN(e.timestamp)
                FROM events e
                JOIN person_events pe ON e.id = pe.event_id
                WHERE pe.person_id = ?
                  AND e.source IN ({placeholders})""",
            (person_id, *comm_sources),
        ).fetchone()[0] or now
    span = _format_span(max(1, (now - first_ts) / 86400))

    stats_span_days = window_days if window_days else max(1, (now - first_ts) / 86400)
    daily_baseline = total / stats_span_days
    if daily_baseline >= 1.0:
        freq = "daily"
    elif daily_baseline >= 0.15:
        freq = "weekly"
    elif daily_baseline >= 0.03:
        freq = "monthly"
    else:
        freq = "rare"

    cutoff_48h = now - 48 * 3600
    recent = store.conn.execute(
        f"""SELECT COUNT(*) FROM events e
            JOIN person_events pe ON e.id = pe.event_id
            WHERE pe.person_id = ?
              AND e.source IN ({placeholders})
              AND e.timestamp >= ?""",
        (person_id, *comm_sources, cutoff_48h),
    ).fetchone()[0]

    return (
        f"CONTACT: {name} | tier:{tier} | msgs:{total} | channels:{ch_short}"
        f" | outbound:{outbound_pct}% | recent:{recent}/48h"
        f" | span:{span} | freq:{freq}"
    )


def build_compact_thread(
    thread_id: str,
    store: LayeredGraphStore,
    *,
    now: int | None = None,
) -> str | None:
    """One-line thread summary for local model (~50 tokens).

    Example::

        THREAD: 47 msgs/14d | velocity:10.6/day | last_user:6h
        | last_them:2h | status:awaiting_user | participants:2
    """
    now = now or int(time.time())

    if not thread_id:
        return None

    total = store.conn.execute(
        """SELECT COUNT(*) FROM events
           WHERE json_extract(metadata, '$.thread_id') = ?""",
        (thread_id,),
    ).fetchone()[0]

    if total < 2:
        return None

    # Span
    ts_row = store.conn.execute(
        """SELECT MIN(timestamp) as first_ts, MAX(timestamp) as last_ts
           FROM events
           WHERE json_extract(metadata, '$.thread_id') = ?""",
        (thread_id,),
    ).fetchone()
    first_ts = ts_row["first_ts"] or now
    last_ts = ts_row["last_ts"] or now
    span_days = max(1, (last_ts - first_ts) / 86400)
    span_label = _format_span(span_days)

    # Velocity
    cutoff_7d = now - 7 * 86400
    recent_msgs = store.conn.execute(
        """SELECT COUNT(*) FROM events
           WHERE json_extract(metadata, '$.thread_id') = ?
             AND timestamp >= ?""",
        (thread_id, cutoff_7d),
    ).fetchone()[0]
    velocity = recent_msgs / 7.0

    # Last messages
    last_two = store.conn.execute(
        """SELECT timestamp, json_extract(metadata, '$.is_from_me') as is_from_me
           FROM events
           WHERE json_extract(metadata, '$.thread_id') = ?
           ORDER BY timestamp DESC LIMIT 10""",
        (thread_id,),
    ).fetchall()

    last_user_ago = ""
    last_them_ago = ""
    status = "ongoing"
    for r in last_two:
        if bool(r["is_from_me"]) and not last_user_ago:
            last_user_ago = _format_span((now - r["timestamp"]) / 86400)
        elif not bool(r["is_from_me"]) and not last_them_ago:
            last_them_ago = _format_span((now - r["timestamp"]) / 86400)
        if last_user_ago and last_them_ago:
            break

    if last_two and not bool(last_two[0]["is_from_me"]):
        status = "awaiting_user"
    elif last_two and bool(last_two[0]["is_from_me"]):
        status = "awaiting_them"

    # Participant count
    n_parts = store.conn.execute(
        """SELECT COUNT(DISTINCT pe.person_id)
           FROM person_events pe
           JOIN events e ON pe.event_id = e.id
           WHERE json_extract(e.metadata, '$.thread_id') = ?""",
        (thread_id,),
    ).fetchone()[0]

    return (
        f"THREAD: {total} msgs/{span_label} | velocity:{velocity:.1f}/day"
        f" | last_user:{last_user_ago or '?'} | last_them:{last_them_ago or '?'}"
        f" | status:{status} | participants:{n_parts}"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Composition helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_full_context(
    person_id: str,
    thread_id: str,
    store: LayeredGraphStore,
    target_date: int | None = None,
    *,
    now: int | None = None,
) -> str:
    """Build full enrichment for cloud model (Gemini).

    Combines contact dossier + thread snapshot + day context.
    """
    parts: list[str] = []

    dossier = build_contact_dossier(person_id, store, now=now)
    if dossier:
        parts.append(dossier)

    snapshot = build_thread_snapshot(thread_id, store, now=now)
    if snapshot:
        parts.append(snapshot)

    day = build_day_context(store, target_date)
    parts.append(day)

    return "\n\n".join(parts)


def build_compact_header(
    person_id: str,
    thread_id: str,
    store: LayeredGraphStore,
    *,
    now: int | None = None,
) -> str:
    """Build ~200 token context header for local model (Qwen).

    Combines compact contact + compact thread on two lines.
    """
    parts: list[str] = []

    contact = build_compact_contact(person_id, store, now=now)
    if contact:
        parts.append(contact)

    thread = build_compact_thread(thread_id, store, now=now)
    if thread:
        parts.append(thread)

    return "\n".join(parts) if parts else ""
