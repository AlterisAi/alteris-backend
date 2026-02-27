"""Clarity Queue MCP tools.

CQ is a **thin view layer** on top of the Knowledge Graph. Most data comes
from KG commitments, beliefs, and events — CQ only stores:
  - User bucket overrides (which priority bucket a KG item belongs to)
  - Done/undone state for KG items
  - Manually-added tasks (not from KG)
  - Chat session history

The "inbox" is just unbucketed KG commitments, surfaced by querying the KG
and filtering out items the user has already bucketed or dismissed.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone

from pathlib import Path

from alteris.mcp_tools import ToolDef, ToolParam, register_tool
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configurable horizon helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _load_profile_yaml() -> dict:
    """Load profile.yaml, returning empty dict on failure."""
    from alteris.profile import load_profile
    return load_profile()


def _get_lookback_days() -> int:
    """Read CQ lookback days from profile.yaml, fall back to constant."""
    from alteris.constants import CQ_DEFAULT_LOOKBACK_DAYS
    profile = _load_profile_yaml()
    val = profile.get("cq_lookback_days")
    if isinstance(val, int) and 1 <= val <= 365:
        return val
    return CQ_DEFAULT_LOOKBACK_DAYS


def _get_lookahead_days() -> int:
    """Read CQ lookahead days from profile.yaml, fall back to constant."""
    from alteris.constants import CQ_DEFAULT_LOOKAHEAD_DAYS
    profile = _load_profile_yaml()
    val = profile.get("cq_lookahead_days")
    if isinstance(val, int) and 1 <= val <= 365:
        return val
    return CQ_DEFAULT_LOOKAHEAD_DAYS

# Cached set of user identifiers (emails, phones, display names, person_id).
# Built once per process from the persons + person_identifiers tables.
_user_identifiers: set[str] | None = None


def _get_user_identifiers(store: LayeredGraphStore) -> set[str]:
    """Return a set of lowercase strings that identify the user.

    Includes emails, phones, display names, canonical name, and the
    person_id itself. Used to replace user references with "" in task
    fields so the UI doesn't show the user's own email.
    """
    global _user_identifiers
    if _user_identifiers is not None:
        return _user_identifiers

    ids: set[str] = {"user", "me", ""}
    try:
        rows = store.conn.execute(
            "SELECT person_id, canonical_name FROM persons WHERE is_user = 1"
        ).fetchall()
        for r in rows:
            pid = r["person_id"]
            ids.add(pid.lower())
            ids.add(r["canonical_name"].lower())
            # All identifiers for this person
            id_rows = store.conn.execute(
                "SELECT identifier FROM person_identifiers WHERE person_id = ?",
                (pid,),
            ).fetchall()
            for ir in id_rows:
                ids.add(ir["identifier"].lower().strip())
    except Exception:
        pass

    _user_identifiers = ids
    return ids


def _clean_person_field(store: LayeredGraphStore, value: str | None) -> str:
    """Clean a who/to_whom/assignee field.

    - Returns "" if the value is a user identifier (email, phone, name).
    - Returns "" for 'unresolved'.
    - Otherwise returns the value unchanged.
    """
    if not value or not value.strip():
        return ""
    v = value.strip()
    if v.lower() in ("unresolved", "unknown", "none"):
        return ""
    user_ids = _get_user_identifiers(store)
    # Strip angle brackets for email matching: "<email>" → "email"
    test = v.lower().strip("<>").strip()
    if test in user_ids:
        return ""
    return v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Auto-bucketing logic: assigns KG commitments to CQ buckets by deadline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _auto_bucket(deadline: str | None, now_ts: int) -> str:
    """Assign a CQ bucket based on deadline proximity.

    Returns:
        'immediate' if due today or overdue,
        'review' if due within 7 days,
        'background' otherwise or if no deadline.
    """
    if not deadline:
        return "background"

    try:
        # Try common date formats
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dl = datetime.strptime(deadline, fmt).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        else:
            return "background"

        now = datetime.fromtimestamp(now_ts, tz=timezone.utc)
        days_until = (dl - now).days

        if days_until <= 0:
            return "immediate"  # overdue or due today
        elif days_until <= 7:
            return "review"
        else:
            return "background"
    except Exception:
        return "background"


def _source_channel_for_claim(store: LayeredGraphStore, claim_id: str | None) -> str:
    """Derive source channel (mail/imessage/whatsapp/calendar) from claim's events."""
    if not claim_id:
        return ""
    events = store.get_events_for_claim(claim_id)
    if events:
        return events[0].source
    return ""


def _within_relevance_window(
    deadline: str | None,
    now_ts: int,
    lookback_days: int | None = None,
    lookahead_days: int | None = None,
) -> bool:
    """Check if a deadline falls within [-lookback_days, +lookahead_days] of now.

    Items with no deadline are always relevant (return True).
    If lookback/lookahead not provided, reads from profile or uses defaults.
    """
    if not deadline:
        return True

    if lookback_days is None:
        lookback_days = _get_lookback_days()
    if lookahead_days is None:
        lookahead_days = _get_lookahead_days()

    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dl_ts = int(datetime.strptime(deadline, fmt).replace(tzinfo=timezone.utc).timestamp())
            break
        except ValueError:
            continue
    else:
        return True  # unparseable → keep

    diff_days = (dl_ts - now_ts) / 86400
    return -lookback_days <= diff_days <= lookahead_days


def _is_stale_event(what: str, commitment_type: str, source_channel: str, deadline: str | None, now_ts: int) -> bool:
    """Check if a past-dated task is a stale time-bound event (no longer actionable).

    Time-bound events (attend, leave, go, meet) that are >24h past their deadline
    are stale. Deliverables (send, prepare, review, respond) remain actionable
    as overdue items even after their deadline.
    """
    if not deadline:
        return False

    # Parse deadline
    dl_ts = None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dl_ts = int(datetime.strptime(deadline, fmt).replace(tzinfo=timezone.utc).timestamp())
            break
        except ValueError:
            continue
    if dl_ts is None:
        return False

    # Not past → not stale
    hours_past = (now_ts - dl_ts) / 3600
    if hours_past < 24:
        return False

    # Calendar events older than 24h are stale
    if source_channel == "calendar":
        return True

    # Time-bound commitment types/verbs are stale once past
    stale_verbs = {"attend", "leave", "go", "meet", "arrive", "depart", "pick up", "drop off", "call"}
    what_lower = what.lower()
    ctype_lower = (commitment_type or "").lower()
    for verb in stale_verbs:
        if what_lower.startswith(verb) or verb in ctype_lower:
            return True

    return False


def _commitments_to_tasks(store: LayeredGraphStore) -> list[dict]:
    """Query KG for active commitments and format as CQ task dicts.

    Surfaces all rich fields from the claim's object JSON (priority, next_action,
    evidence_quote, note, deferred_until, assignee, custom_fields, etc.).
    P1 → immediate, P2 → review, P3+ → background (unless deadline overrides).
    Filters out stale time-bound events (past calendar events, past "attend/leave" tasks).
    """
    now = int(time.time())
    commitments = store.get_active_claims(claim_type="commitment", limit=100)

    tasks = []
    for c in commitments:
        try:
            obj = json.loads(c.object) if isinstance(c.object, str) else c.object
        except (json.JSONDecodeError, TypeError):
            continue

        if obj.get("status") not in ("open", None, ""):
            continue

        deadline = obj.get("deadline")
        what = obj.get("what", "")
        if not what:
            continue

        source_channel = _source_channel_for_claim(store, c.id)
        commitment_type = obj.get("type", "")

        # Skip stale time-bound events (e.g., past "leave for church" or calendar events)
        if _is_stale_event(what, commitment_type, source_channel, deadline, now):
            continue

        # Only show items within ±7 day relevance window
        if not _within_relevance_window(deadline, now):
            continue

        note = obj.get("note") or ""
        priority = obj.get("priority", 3)
        bucket = _auto_bucket(deadline, now)

        # Priority overrides: P1 → immediate, P2 → review (at minimum)
        if priority == 1:
            bucket = "immediate"
        elif priority == 2 and bucket == "background":
            bucket = "review"

        tasks.append({
            "id": c.id,
            "title": what,
            "note": note,
            "source": "kg_commitment",
            "due_date": deadline,
            "done": False,
            "bucket": bucket,
            "claim_id": c.id,
            "labels": None,
            "position": 0,
            "source_channel": source_channel,
            "updated_at": getattr(c, "created_at", 0) or now,
            # Rich fields from synthesis
            "priority": priority,
            "commitment_type": obj.get("type", ""),
            "next_action": obj.get("next_action"),
            "assignee": _clean_person_field(store, obj.get("assignee")),
            "defer_until": obj.get("deferred_until"),
            "custom_fields": obj.get("custom_fields"),
            "evidence_quote": obj.get("evidence_quote"),
            "who": _clean_person_field(store, obj.get("who", "")),
            "to_whom": _clean_person_field(store, obj.get("to_whom", "")),
        })

    return tasks


def _beliefs_to_tasks(store: LayeredGraphStore) -> list[dict]:
    """Surface high-confidence actionable beliefs as potential CQ items.

    Surfaces rich fields from belief data (note, priority, next_action,
    assignee, evidence_quote, etc.) — mirroring _commitments_to_tasks().
    Filters out stale time-bound events.
    Skips beliefs that duplicate an active commitment claim (dedup).
    """
    now = int(time.time())
    beliefs = store.get_beliefs(status="active", belief_type="fact", limit=50)

    # Build set of active commitment claim IDs for dedup.
    # If a belief's source_claims include an active commitment, the commitment
    # is already surfaced — skip the belief to avoid double-surfacing.
    active_commitment_ids = {
        r["id"] for r in store.conn.execute(
            "SELECT id FROM claims WHERE claim_type = 'commitment' AND superseded_by IS NULL"
        ).fetchall()
    }

    tasks = []
    for b in beliefs:
        if b.confidence < 0.6:
            continue

        data = json.loads(b.data) if isinstance(b.data, str) else b.data
        deadline = data.get("deadline") or data.get("date")
        if not deadline:
            continue

        # Skip beliefs that are derived from active commitment claims —
        # the commitment itself is already in the CQ.
        src_claims = b.source_claims if isinstance(b.source_claims, list) else (
            json.loads(b.source_claims) if isinstance(b.source_claims, str) and b.source_claims else []
        )
        if any(cid in active_commitment_ids for cid in src_claims):
            continue

        commitment_type = data.get("type", "")
        # Skip stale time-bound events
        if _is_stale_event(b.summary, commitment_type, "", deadline, now):
            continue

        # Only show items within ±7 day relevance window
        if not _within_relevance_window(deadline, now):
            continue

        priority = data.get("priority", 3)
        bucket = _auto_bucket(deadline, now)
        if priority == 1:
            bucket = "immediate"
        elif priority == 2 and bucket == "background":
            bucket = "review"

        # Use rich note from belief data, fall back to summary
        note = data.get("note") or ""

        tasks.append({
            "id": f"belief_{b.id}",
            "title": b.summary,
            "note": note,
            "source": "kg_belief",
            "due_date": deadline,
            "done": False,
            "bucket": bucket,
            "claim_id": None,
            "labels": None,
            "position": 0,
            "source_channel": "",
            "updated_at": getattr(b, "updated_at", 0) or now,
            # Rich fields from belief data
            "priority": priority,
            "commitment_type": data.get("type", ""),
            "next_action": data.get("next_action"),
            "assignee": _clean_person_field(store, data.get("assignee")),
            "defer_until": data.get("deferred_until"),
            "custom_fields": data.get("custom_fields"),
            "evidence_quote": data.get("evidence_quote"),
            "who": _clean_person_field(store, data.get("who", "")),
            "to_whom": _clean_person_field(store, data.get("to_whom", "")),
        })

    return tasks


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Recurrence helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _compute_next_due(current_due: str | None, recurrence: dict) -> str | None:
    """Compute next due date from current due + recurrence rule.

    recurrence: {"freq": "daily"|"weekly"|"monthly"|"yearly", "interval": int}
    Returns ISO date string or None.
    """
    if not current_due or not recurrence:
        return None

    freq = recurrence.get("freq", "weekly")
    interval = recurrence.get("interval", 1)

    try:
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                base = datetime.strptime(current_due, fmt)
                break
            except ValueError:
                continue
        else:
            return None

        if freq == "daily":
            next_dt = base + timedelta(days=interval)
        elif freq == "weekly":
            next_dt = base + timedelta(weeks=interval)
        elif freq == "monthly":
            month = base.month + interval
            year = base.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            day = min(base.day, 28)  # Safe for all months
            next_dt = base.replace(year=year, month=month, day=day)
        elif freq == "yearly":
            next_dt = base.replace(year=base.year + interval)
        else:
            return None

        return next_dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def _resolve_kg_commitment(store: LayeredGraphStore, task_id: str) -> None:
    """Propagate done=True from CQ to the underlying KG commitment + belief.

    Records an immutable user action event (the fact that the user marked this
    done), then resolves the claim and belief referencing that event.
    """
    from alteris.models import Event, SensitivityLevel

    now = int(time.time())
    row = store.conn.execute(
        "SELECT id, object FROM claims WHERE id = ? AND claim_type = 'commitment'",
        (task_id,),
    ).fetchone()
    if not row:
        return

    try:
        obj = json.loads(row["object"])
    except (json.JSONDecodeError, TypeError):
        return

    if obj.get("status") == "done":
        return  # already resolved

    # Record immutable user action event
    action_event = Event(
        id=Event.make_id("user", f"resolve:{task_id}:{now}"),
        source="user",
        source_id=f"resolve:{task_id}:{now}",
        event_type="action",
        timestamp=now,
        raw_content="",
        metadata={
            "action": "resolve_commitment",
            "claim_id": task_id,
            "what": obj.get("what", ""),
            "to_whom": obj.get("to_whom", ""),
            "deadline": obj.get("deadline", ""),
        },
        sensitivity=SensitivityLevel.PUBLIC,
        created_at=now,
    )
    store.put_event(action_event)

    # Resolve the claim, referencing the action event
    obj["status"] = "done"
    obj["resolution_evidence"] = "Marked done by user"
    obj["resolution_event_id"] = action_event.id
    obj["resolution_checked_until"] = now
    store.conn.execute(
        "UPDATE claims SET object = ? WHERE id = ?",
        (json.dumps(obj), task_id),
    )

    store.conn.execute(
        """UPDATE beliefs SET status = 'resolved', updated_at = ?
           WHERE status = 'active'
             AND belief_type = 'fact'
             AND source_claims LIKE ?""",
        (now, f'%{task_id}%'),
    )
    store.conn.commit()
    logger.info("User resolved commitment %s → event %s", task_id[:16], action_event.id[:16])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool implementations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def handle_cq_get_state(store: LayeredGraphStore, date: str | None = None, **kwargs) -> dict:
    """Desk focused view: only user-accepted tasks + manual tasks.

    Rules:
    - KG items only appear if the user explicitly accepted them.
    - Done items only appear if completed today (user's local date).
    - Manual tasks (source='manual') always appear.
    - The All view uses ``handle_cq_get_all`` instead.
    """
    from alteris.constants import CQ_BUCKETS

    now = int(time.time())
    today_str = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")

    # ── Gather user overrides ──────────────────────────────────────────
    user_tasks = store.get_cq_tasks(include_done=True)

    accepted_ids: set[str] = set()
    override_map: dict[str, dict] = {}
    manual_tasks: list[dict] = []

    for t in user_tasks:
        uid = t.get("claim_id") or t["id"]
        if t.get("source") == "manual":
            manual_tasks.append(t)
        elif t.get("accepted"):
            accepted_ids.add(uid)
            override_map[uid] = t

    # Fast path: nothing accepted and no manual tasks → empty desk.
    if not accepted_ids and not manual_tasks:
        empty: dict = {bucket: [] for bucket in CQ_BUCKETS}
        undo_entry = store.conn.execute(
            "SELECT COUNT(*) FROM cq_undo_log"
        ).fetchone()
        empty["done"] = []
        empty["done_count"] = 0
        empty["dismissed"] = []
        empty["dismissed_count"] = 0
        empty["deferred_count"] = 0
        empty["kg_items"] = 0
        empty["canUndo"] = (undo_entry[0] or 0) > 0
        return empty

    # ── Build accepted KG tasks ────────────────────────────────────────
    kg_tasks = _commitments_to_tasks(store)
    kg_tasks.extend(_beliefs_to_tasks(store))
    kg_map = {t["id"]: t for t in kg_tasks}

    merged: list[dict] = []

    for aid in accepted_ids:
        kg_item = kg_map.get(aid)
        if not kg_item:
            # Accepted but claim no longer in active KG — try to build from
            # the override + raw claim data (e.g. user resolved it).
            ov = override_map[aid]
            claim_id = ov.get("claim_id") or aid
            title = ov.get("title", "")
            if not title and claim_id:
                row = store.conn.execute(
                    "SELECT object FROM claims WHERE id = ?", (claim_id,),
                ).fetchone()
                if row:
                    try:
                        obj = json.loads(row["object"])
                        title = obj.get("what", "")
                    except (json.JSONDecodeError, TypeError):
                        pass
            if not title:
                continue
            kg_item = {
                "id": claim_id,
                "title": title,
                "note": ov.get("note", ""),
                "source": "kg_commitment",
                "due_date": ov.get("due_date"),
                "done": ov.get("done", 0),
                "bucket": ov.get("bucket", "background"),
                "claim_id": claim_id,
                "labels": None,
                "position": 0,
                "source_channel": "",
                "updated_at": ov.get("updated_at", 0) or now,
                "priority": 3,
                "commitment_type": "",
            }

        # Apply user overrides.
        ov = override_map.get(aid, {})
        if ov:
            kg_item["bucket"] = ov.get("bucket", kg_item.get("bucket", "background"))
            kg_item["done"] = ov.get("done", 0)
            kg_item["position"] = ov.get("position", 0)
            kg_item["defer_until"] = ov.get("defer_until")
            kg_item["recurrence"] = ov.get("recurrence")
            kg_item["custom_fields"] = ov.get("custom_fields")
            kg_item["category"] = ov.get("category")

        merged.append(kg_item)

    # Manual tasks always in Desk.
    for mt in manual_tasks:
        if "source_channel" not in mt:
            mt["source_channel"] = ""
        merged.append(mt)

    # ── Bucket into focused view ───────────────────────────────────────
    result: dict = {bucket: [] for bucket in CQ_BUCKETS}
    done_today: list[dict] = []
    deferred_count = 0

    for item in merged:
        if item.get("done"):
            # Only show done items completed today.
            updated = item.get("updated_at", 0)
            done_date = datetime.fromtimestamp(updated, tz=timezone.utc).strftime("%Y-%m-%d") if updated else ""
            if done_date == today_str:
                done_today.append(item)
            continue

        defer_until = item.get("defer_until")
        if defer_until and defer_until > today_str:
            deferred_count += 1
            continue

        bucket = item.get("bucket", "background")
        if bucket not in result:
            bucket = "background"
        result[bucket].append(item)

    for bucket in CQ_BUCKETS:
        result[bucket].sort(key=lambda t: (t.get("position", 0), t.get("due_date") or "9999"))

    if len(result["immediate"]) > 10:
        immediate = result["immediate"]
        immediate.sort(key=lambda t: (t.get("priority", 3), t.get("due_date") or "9999"))
        result["review"] = immediate[10:] + result["review"]
        result["immediate"] = immediate[:10]

    undo_entry = store.conn.execute(
        "SELECT COUNT(*) FROM cq_undo_log"
    ).fetchone()

    result["done"] = done_today
    result["done_count"] = len(done_today)
    result["dismissed"] = []
    result["dismissed_count"] = 0
    result["deferred_count"] = deferred_count
    result["kg_items"] = len(kg_tasks) if accepted_ids else 0
    result["canUndo"] = (undo_entry[0] or 0) > 0

    return result


def handle_cq_add_task(
    store: LayeredGraphStore,
    bucket: str = "review",
    title: str = "",
    note: str = "",
    due_date: str | None = None,
    labels: list[str] | None = None,
    defer_until: str | None = None,
    recurrence: dict | None = None,
    custom_fields: dict | None = None,
    category: str | None = None,
    **kwargs,
) -> dict:
    """Add a manual task to a CQ bucket (not from KG)."""
    if not title:
        return {"error": "title is required"}

    from alteris.constants import CQ_BUCKETS
    if bucket not in CQ_BUCKETS:
        return {"error": f"bucket must be one of {CQ_BUCKETS}"}

    task_id = f"cq_{uuid.uuid4().hex[:12]}"
    existing = store.get_cq_tasks(bucket=bucket)
    position = max((t.get("position", 0) for t in existing), default=-1) + 1

    store.put_cq_task(
        task_id=task_id,
        bucket=bucket,
        title=title,
        note=note,
        source="manual",
        due_date=due_date,
        labels=labels,
        position=position,
        defer_until=defer_until,
        recurrence=recurrence,
        custom_fields=custom_fields,
        category=category,
    )

    # Push undo for add (prev_state is empty — undo = delete)
    store.push_cq_undo("add", task_id, {})

    return {"task_id": task_id, "bucket": bucket}


def handle_cq_update_task(
    store: LayeredGraphStore,
    task_id: str = "",
    title: str | None = None,
    note: str | None = None,
    done: bool | None = None,
    due_date: str | None = None,
    labels: list[str] | None = None,
    defer_until: str | None = None,
    recurrence: dict | None = None,
    custom_fields: dict | None = None,
    category: str | None = None,
    accepted: bool | None = None,
    **kwargs,
) -> dict:
    """Update a CQ task. For KG items, this creates/updates an override."""
    if not task_id:
        return {"error": "task_id is required"}

    fields = {}
    if title is not None:
        fields["title"] = title
    if note is not None:
        fields["note"] = note
    if done is not None:
        fields["done"] = int(done)
    if due_date is not None:
        fields["due_date"] = due_date
    if labels is not None:
        fields["labels"] = labels
    if defer_until is not None:
        fields["defer_until"] = defer_until
    if recurrence is not None:
        fields["recurrence"] = recurrence
    if custom_fields is not None:
        fields["custom_fields"] = custom_fields
    if category is not None:
        fields["category"] = category
    if accepted is not None:
        fields["accepted"] = int(accepted)

    if not fields:
        return {"error": "no fields to update"}

    # Snapshot for undo before mutation
    prev = store.snapshot_cq_task(task_id)
    if prev:
        store.push_cq_undo("update", task_id, prev)

    # Check if this is a KG item that doesn't have a cq_tasks row yet
    existing = store.get_cq_tasks(include_done=True)
    has_row = any(t["id"] == task_id for t in existing)

    if has_row:
        ok = store.update_cq_task(task_id, **fields)
    else:
        bucket = fields.pop("bucket", "background")
        # put_cq_task doesn't accept 'done' or 'accepted' — pop them, apply via update after
        done_val = fields.pop("done", None)
        accepted_val = fields.pop("accepted", None)
        put_fields = {k: v for k, v in fields.items() if k not in ("title", "note")}
        store.put_cq_task(
            task_id=task_id,
            bucket=bucket,
            title=fields.get("title", ""),
            note=fields.get("note", ""),
            source="kg_override",
            claim_id=task_id,
            **put_fields,
        )
        if done_val is not None:
            store.update_cq_task(task_id, done=done_val)
        if accepted_val is not None:
            store.update_cq_task(task_id, accepted=accepted_val)
        ok = True

    result = {"updated": ok, "task_id": task_id}

    # Propagate done=True to the underlying KG commitment claim + belief
    if done and ok:
        _resolve_kg_commitment(store, task_id)

    # Recurrence: on done=True with recurrence, auto-create next instance
    if done and ok:
        # Re-read the task to get recurrence info
        tasks = store.get_cq_tasks(include_done=True)
        task_data = next((t for t in tasks if t["id"] == task_id), None)
        rec = task_data.get("recurrence") if task_data else recurrence
        if rec:
            current_due = (task_data or {}).get("due_date") or due_date
            next_due = _compute_next_due(current_due, rec)
            if next_due:
                new_id = f"cq_{uuid.uuid4().hex[:12]}"
                store.put_cq_task(
                    task_id=new_id,
                    bucket=(task_data or {}).get("bucket", "review"),
                    title=(task_data or {}).get("title", title or ""),
                    note=(task_data or {}).get("note", note or ""),
                    source="manual",
                    due_date=next_due,
                    recurrence=rec,
                    custom_fields=(task_data or {}).get("custom_fields"),
                    category=(task_data or {}).get("category"),
                )
                result["next_instance"] = {"task_id": new_id, "due_date": next_due}

    return result


def handle_cq_move_task(
    store: LayeredGraphStore,
    task_id: str = "",
    target_bucket: str = "",
    position: int = 0,
    **kwargs,
) -> dict:
    """Move a task to a different bucket. Creates an override for KG items."""
    if not task_id or not target_bucket:
        return {"error": "task_id and target_bucket are required"}

    from alteris.constants import CQ_BUCKETS
    if target_bucket not in CQ_BUCKETS:
        return {"error": f"target_bucket must be one of {CQ_BUCKETS}"}

    # Snapshot for undo
    prev = store.snapshot_cq_task(task_id)
    if prev:
        store.push_cq_undo("move", task_id, prev)

    existing = store.get_cq_tasks(include_done=True)
    has_row = any(t["id"] == task_id for t in existing)

    if has_row:
        ok = store.move_cq_task(task_id, target_bucket, position)
    else:
        store.put_cq_task(
            task_id=task_id,
            bucket=target_bucket,
            title="",
            source="kg_override",
            claim_id=task_id,
            position=position,
        )
        ok = True

    return {"moved": ok, "task_id": task_id, "bucket": target_bucket}


def handle_cq_dismiss(
    store: LayeredGraphStore,
    task_id: str = "",
    **kwargs,
) -> dict:
    """Dismiss a KG item from CQ (won't show up again)."""
    if not task_id:
        return {"error": "task_id is required"}

    # Snapshot for undo
    prev = store.snapshot_cq_task(task_id)
    if prev:
        store.push_cq_undo("dismiss", task_id, prev)

    existing = store.get_cq_tasks(include_done=True)
    for t in existing:
        if t["id"] == task_id and t.get("source") == "manual":
            ok = store.delete_cq_task(task_id)
            return {"deleted": ok, "task_id": task_id}

    store.put_cq_task(
        task_id=f"dismiss_{task_id}",
        bucket="dismissed",
        title="",
        source="dismissed",
        claim_id=task_id,
    )
    return {"dismissed": True, "task_id": task_id}


def handle_cq_restore(
    store: LayeredGraphStore,
    task_id: str = "",
    **kwargs,
) -> dict:
    """Restore a dismissed task — deletes the dismiss override so the KG item reappears."""
    if not task_id:
        return {"error": "task_id is required"}

    dismiss_id = f"dismiss_{task_id}"
    ok = store.delete_cq_task(dismiss_id)
    if not ok:
        # Maybe the dismiss record uses the task_id directly
        ok = store.delete_cq_task(task_id)

    return {"restored": ok, "task_id": task_id}


def handle_cq_reorder(
    store: LayeredGraphStore,
    bucket: str = "",
    task_ids: list[str] | None = None,
    **kwargs,
) -> dict:
    """Reorder tasks within a bucket by providing ordered task_ids."""
    if not bucket or not task_ids:
        return {"error": "bucket and task_ids are required"}

    ok = store.reorder_cq_tasks(bucket, task_ids)
    return {"reordered": ok, "bucket": bucket}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# New CQ v2 tools
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def handle_cq_undo(store: LayeredGraphStore, **kwargs) -> dict:
    """Pop the last undo entry and restore previous state."""
    entry = store.pop_cq_undo()
    if not entry:
        return {"error": "nothing to undo"}

    action = entry["action"]
    task_id = entry["task_id"]
    prev = entry["prev_state"]

    if action == "add":
        # Undo an add = delete the task
        store.delete_cq_task(task_id)
        return {"undone": "add", "task_id": task_id, "action": "deleted"}

    if action == "delete":
        # Undo a delete = re-insert from snapshot
        store.put_cq_task(
            task_id=prev["id"],
            bucket=prev.get("bucket", "background"),
            title=prev.get("title", ""),
            note=prev.get("note", ""),
            source=prev.get("source", "manual"),
            due_date=prev.get("due_date"),
            labels=json.loads(prev["labels"]) if prev.get("labels") else None,
            position=prev.get("position", 0),
            claim_id=prev.get("claim_id"),
            defer_until=prev.get("defer_until"),
            recurrence=json.loads(prev["recurrence"]) if prev.get("recurrence") else None,
            custom_fields=json.loads(prev["custom_fields"]) if prev.get("custom_fields") else None,
            category=prev.get("category"),
        )
        if prev.get("done"):
            store.update_cq_task(prev["id"], done=prev["done"])
        return {"undone": "delete", "task_id": task_id, "action": "restored"}

    # For dismiss, update, move, toggle_done: restore full previous state
    if prev:
        store.put_cq_task(
            task_id=prev["id"],
            bucket=prev.get("bucket", "background"),
            title=prev.get("title", ""),
            note=prev.get("note", ""),
            source=prev.get("source", "manual"),
            due_date=prev.get("due_date"),
            labels=json.loads(prev["labels"]) if prev.get("labels") else None,
            position=prev.get("position", 0),
            claim_id=prev.get("claim_id"),
            defer_until=prev.get("defer_until"),
            recurrence=json.loads(prev["recurrence"]) if prev.get("recurrence") else None,
            custom_fields=json.loads(prev["custom_fields"]) if prev.get("custom_fields") else None,
            category=prev.get("category"),
        )
        if prev.get("done"):
            store.update_cq_task(prev["id"], done=prev["done"])
        # If this was a dismiss, also remove the dismiss record
        if action == "dismiss":
            store.delete_cq_task(f"dismiss_{task_id}")

    return {"undone": action, "task_id": task_id, "action": "restored"}


def handle_cq_get_all(store: LayeredGraphStore, **kwargs) -> dict:
    """Get ALL tasks across all statuses: active, deferred, dismissed, done, stale, superseded."""
    now = int(time.time())
    today_str = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")

    # Get KG items
    kg_tasks = _commitments_to_tasks(store)
    kg_tasks.extend(_beliefs_to_tasks(store))

    # Get all user tasks
    user_tasks = store.get_cq_tasks(include_done=True)
    override_map = {}
    manual_tasks = []
    dismissed_map = {}  # claim_id -> dismiss record

    for t in user_tasks:
        if t.get("source") == "manual":
            manual_tasks.append(t)
        elif t.get("source") == "dismissed":
            dismissed_map[t.get("claim_id") or t["id"]] = t
        else:
            override_map[t.get("claim_id") or t["id"]] = t

    # Categorize
    active = []
    deferred = []
    dismissed = []
    done = []

    seen_ids = set()

    for kg_item in kg_tasks:
        item_id = kg_item["id"]
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        override = override_map.get(item_id)
        if override:
            kg_item["bucket"] = override.get("bucket", kg_item["bucket"])
            kg_item["done"] = override.get("done", 0)
            kg_item["position"] = override.get("position", 0)
            kg_item["defer_until"] = override.get("defer_until")
            kg_item["recurrence"] = override.get("recurrence")
            kg_item["custom_fields"] = override.get("custom_fields")
            kg_item["category"] = override.get("category")

        if item_id in dismissed_map:
            kg_item["status"] = "dismissed"
            dismissed.append(kg_item)
        elif kg_item.get("done"):
            kg_item["status"] = "done"
            done.append(kg_item)
        elif kg_item.get("defer_until") and kg_item["defer_until"] > today_str:
            kg_item["status"] = "deferred"
            deferred.append(kg_item)
        else:
            kg_item["status"] = "active"
            active.append(kg_item)

    for mt in manual_tasks:
        mt["source_channel"] = ""
        if mt.get("done"):
            mt["status"] = "done"
            done.append(mt)
        elif mt.get("defer_until") and mt["defer_until"] > today_str:
            mt["status"] = "deferred"
            deferred.append(mt)
        else:
            mt["status"] = "active"
            active.append(mt)

    return {
        "active": active,
        "deferred": deferred,
        "dismissed": dismissed,
        "done": done,
        "counts": {
            "active": len(active),
            "deferred": len(deferred),
            "dismissed": len(dismissed),
            "done": len(done),
            "total": len(active) + len(deferred) + len(dismissed) + len(done),
        },
    }


def handle_cq_get_thread(
    store: LayeredGraphStore,
    task_id: str = "",
    **kwargs,
) -> dict:
    """Given a task_id, traverse claim_id -> events -> thread.

    Returns the full message thread with sender, timestamp, subject, body, source.
    """
    if not task_id:
        return {"error": "task_id is required"}

    # Find the claim_id for this task
    claim_id = task_id  # For KG items, task_id IS the claim_id

    # Check if it's a manual task with a claim_id reference
    tasks = store.get_cq_tasks(include_done=True)
    for t in tasks:
        if t["id"] == task_id and t.get("claim_id"):
            claim_id = t["claim_id"]
            break

    # Get events for the claim
    events = store.get_events_for_claim(claim_id)
    if not events:
        return {"task_id": task_id, "messages": [], "note": "No events found for this task"}

    # Try to get thread_id from first event's metadata
    thread_events = events
    first_meta = events[0].metadata if isinstance(events[0].metadata, dict) else {}
    thread_id = first_meta.get("thread_id")

    if thread_id:
        # Get all events in this thread
        thread_events = store.get_events_by_thread(thread_id, source=events[0].source)
        if not thread_events:
            thread_events = events

    messages = []
    for e in thread_events:
        meta = e.metadata if isinstance(e.metadata, dict) else {}
        messages.append({
            "event_id": e.id,
            "source": e.source,
            "timestamp": e.timestamp,
            "date": datetime.fromtimestamp(e.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "sender": meta.get("sender", meta.get("from", "")),
            "subject": meta.get("subject", meta.get("title", "")),
            "body": (e.raw_content or "")[:2000],  # Cap at 2K chars
            "participants": list(e.participants),
        })

    return {
        "task_id": task_id,
        "claim_id": claim_id,
        "thread_id": thread_id,
        "source": events[0].source if events else "",
        "message_count": len(messages),
        "messages": messages,
    }


def handle_cq_manage_sender_rules(
    store: LayeredGraphStore,
    action: str = "list",
    pattern: str = "",
    priority: str = "",
    source: str = "",
    note: str = "",
    rule_id: int | None = None,
    **kwargs,
) -> dict:
    """Add, remove, or list sender priority rules."""
    if action == "list":
        rules = store.get_sender_rules()
        return {"count": len(rules), "rules": rules}

    if action == "add":
        if not pattern or not priority:
            return {"error": "pattern and priority are required"}
        if priority not in ("P1", "P2", "P3", "block"):
            return {"error": "priority must be P1, P2, P3, or block"}
        store.put_sender_rule(pattern, priority, source, note)
        return {"added": True, "pattern": pattern, "priority": priority}

    if action == "delete":
        if rule_id is None:
            return {"error": "rule_id is required for delete"}
        store.delete_sender_rule(rule_id)
        return {"deleted": True, "rule_id": rule_id}

    return {"error": f"unknown action: {action}"}


def handle_cq_manage_categories(
    store: LayeredGraphStore,
    action: str = "list",
    name: str = "",
    color: str = "",
    icon: str = "",
    **kwargs,
) -> dict:
    """Add, remove, or list user-defined categories."""
    if action == "list":
        cats = store.get_cq_categories()
        return {"count": len(cats), "categories": cats}

    if action == "add":
        if not name:
            return {"error": "name is required"}
        store.put_cq_category(name, color, icon)
        return {"added": True, "name": name}

    if action == "delete":
        if not name:
            return {"error": "name is required for delete"}
        store.delete_cq_category(name)
        return {"deleted": True, "name": name}

    return {"error": f"unknown action: {action}"}


def handle_cq_manage_extractable_fields(
    store: LayeredGraphStore,
    action: str = "list",
    name: str = "",
    description: str = "",
    example: str = "",
    **kwargs,
) -> dict:
    """Add, remove, or list user-defined extractable field definitions.

    These fields are injected into the synthesis prompt for best-effort extraction
    into the custom_fields object on each commitment.
    """
    if action == "list":
        fields = store.get_cq_extractable_fields()
        return {"count": len(fields), "fields": fields}

    if action == "add":
        if not name:
            return {"error": "name is required"}
        store.put_cq_extractable_field(name, description, example)
        return {"added": True, "name": name}

    if action == "delete":
        if not name:
            return {"error": "name is required for delete"}
        store.delete_cq_extractable_field(name)
        return {"deleted": True, "name": name}

    return {"error": f"unknown action: {action}"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enriched Agenda surface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _compute_time_bands(
    now: datetime, lookahead_days: int
) -> list[dict]:
    """Compute time bands for the agenda with exponential collapse.

    Returns list of dicts with: band_key, label, start_date, end_date,
    default_expanded, detail_level.
    """
    import calendar as cal_mod

    end_horizon = now + timedelta(days=lookahead_days)
    bands: list[dict] = []

    today = now.date()
    tomorrow = today + timedelta(days=1)

    # Band 1: Today
    bands.append({
        "band_key": "today",
        "label": f"Today - {now.strftime('%A, %b %d')}",
        "start_date": today.isoformat(),
        "end_date": today.isoformat(),
        "default_expanded": True,
        "detail_level": "full",
    })

    if lookahead_days < 1:
        return bands

    # Band 2: Tomorrow
    bands.append({
        "band_key": "tomorrow",
        "label": f"Tomorrow - {tomorrow.strftime('%a %b %d')}",
        "start_date": tomorrow.isoformat(),
        "end_date": tomorrow.isoformat(),
        "default_expanded": True,
        "detail_level": "full",
    })

    # Band 3: Rest of this week (day after tomorrow through Sunday)
    day_after_tomorrow = today + timedelta(days=2)
    # Sunday of this week
    days_to_sunday = 6 - today.weekday()  # Monday=0, Sunday=6
    end_of_week = today + timedelta(days=days_to_sunday)

    if day_after_tomorrow <= end_of_week and day_after_tomorrow <= end_horizon.date():
        actual_end = min(end_of_week, end_horizon.date())
        bands.append({
            "band_key": "this_week",
            "label": f"This Week ({day_after_tomorrow.strftime('%b %d')}-{actual_end.strftime('%b %d')})",
            "start_date": day_after_tomorrow.isoformat(),
            "end_date": actual_end.isoformat(),
            "default_expanded": True,
            "detail_level": "compact",
        })

    # Band 4: Next week (Monday to Sunday)
    next_monday = end_of_week + timedelta(days=1)
    next_sunday = next_monday + timedelta(days=6)
    if next_monday <= end_horizon.date():
        actual_end = min(next_sunday, end_horizon.date())
        bands.append({
            "band_key": "next_week",
            "label": f"Next Week ({next_monday.strftime('%b %d')}-{actual_end.strftime('%b %d')})",
            "start_date": next_monday.isoformat(),
            "end_date": actual_end.isoformat(),
            "default_expanded": False,
            "detail_level": "compact",
        })

    # Band 5: Week after next
    week_after_monday = next_sunday + timedelta(days=1)
    week_after_sunday = week_after_monday + timedelta(days=6)
    if week_after_monday <= end_horizon.date():
        actual_end = min(week_after_sunday, end_horizon.date())
        bands.append({
            "band_key": "week_after_next",
            "label": f"Week of {week_after_monday.strftime('%b %d')}",
            "start_date": week_after_monday.isoformat(),
            "end_date": actual_end.isoformat(),
            "default_expanded": False,
            "detail_level": "compact",
        })

    # Band 6+: Rest of current month, then each subsequent month
    rest_start = week_after_sunday + timedelta(days=1)
    if rest_start <= end_horizon.date():
        # Rest of current month
        current_month_end_day = cal_mod.monthrange(rest_start.year, rest_start.month)[1]
        current_month_end = rest_start.replace(day=current_month_end_day)
        if rest_start <= current_month_end and rest_start.month == now.month:
            actual_end = min(current_month_end, end_horizon.date())
            bands.append({
                "band_key": f"rest_of_{rest_start.strftime('%Y_%m')}",
                "label": f"Rest of {rest_start.strftime('%B')}",
                "start_date": rest_start.isoformat(),
                "end_date": actual_end.isoformat(),
                "default_expanded": False,
                "detail_level": "compact",
            })
            rest_start = actual_end + timedelta(days=1)

        # Subsequent months
        while rest_start <= end_horizon.date():
            month_end_day = cal_mod.monthrange(rest_start.year, rest_start.month)[1]
            month_end = rest_start.replace(day=month_end_day)
            actual_end = min(month_end, end_horizon.date())
            bands.append({
                "band_key": f"month_{rest_start.strftime('%Y_%m')}",
                "label": rest_start.strftime("%B %Y") if rest_start.year != now.year else rest_start.strftime("%B"),
                "start_date": rest_start.isoformat(),
                "end_date": actual_end.isoformat(),
                "default_expanded": False,
                "detail_level": "compact",
            })
            rest_start = actual_end + timedelta(days=1)

    return bands


def _enrich_attendees(
    store: LayeredGraphStore,
    attendee_rows: list[dict],
    detail_level: str,
    user_ids: set[str],
    max_enriched: int = 5,
) -> list[dict]:
    """Enrich attendees with KG context. Full enrichment only for 'full' detail."""
    now = int(time.time())
    thirty_days_ago = now - 30 * 86400
    enriched = []

    for i, att in enumerate(attendee_rows):
        person_id = att.get("person_id", "")
        name = att.get("name", "")
        role = att.get("role", "")

        # Skip user's own entries
        if name.lower() in user_ids or person_id.lower() in user_ids:
            continue

        entry: dict = {
            "person_id": person_id,
            "name": name or person_id,
            "role": role,
            "is_organizer": role == "organizer",
        }

        # Full enrichment: last interaction, commitments, comm frequency
        if detail_level == "full" and i < max_enriched and person_id:
            # Last interaction (single query)
            last_event = store.conn.execute(
                """SELECT e.timestamp, e.source, e.raw_content
                   FROM events e
                   JOIN person_events pe ON e.id = pe.event_id
                   WHERE pe.person_id = ? AND e.source != 'calendar'
                   ORDER BY e.timestamp DESC LIMIT 1""",
                (person_id,),
            ).fetchone()

            if last_event:
                snippet = (last_event["raw_content"] or "")[:80]
                entry["last_interaction"] = {
                    "timestamp": last_event["timestamp"],
                    "source": last_event["source"],
                    "snippet": snippet,
                }

            # Open commitments involving this person (GROUP BY to dedup
            # when a claim has multiple events with the same person)
            commit_rows = store.conn.execute(
                """SELECT c.object, c.confidence
                   FROM claims c
                   JOIN claim_events ce ON c.id = ce.claim_id
                   JOIN person_events pe ON ce.event_id = pe.event_id
                   WHERE pe.person_id = ?
                     AND c.claim_type = 'commitment'
                     AND c.superseded_by IS NULL
                   GROUP BY c.id
                   ORDER BY c.confidence DESC LIMIT 3""",
                (person_id,),
            ).fetchall()

            commitments = []
            for cr in commit_rows:
                try:
                    obj = json.loads(cr["object"]) if isinstance(cr["object"], str) else cr["object"]
                    if obj.get("status") not in ("open", None, ""):
                        continue
                    commitments.append({
                        "what": obj.get("what", ""),
                        "deadline": obj.get("deadline"),
                        "type": obj.get("type", ""),
                        "priority": obj.get("priority", 3),
                    })
                except (json.JSONDecodeError, TypeError):
                    continue

            if commitments:
                entry["open_commitments"] = commitments

            # Communication frequency (message count last 30 days)
            msg_count = store.conn.execute(
                """SELECT COUNT(*) FROM events e
                   JOIN person_events pe ON e.id = pe.event_id
                   WHERE pe.person_id = ? AND e.timestamp >= ?
                     AND e.source != 'calendar'""",
                (person_id, thirty_days_ago),
            ).fetchone()[0]
            entry["communication_count"] = msg_count

        enriched.append(entry)

    # If more attendees than shown, add overflow count
    overflow = len(attendee_rows) - len(enriched)
    # Note: overflow tracked by caller via total count vs enriched count

    return enriched


def _get_prep_items(
    store: LayeredGraphStore,
    event_start_ts: int,
    attendee_person_ids: set[str],
    user_ids: set[str],
) -> list[dict]:
    """Find CQ tasks/commitments due before an event that involve its attendees."""
    now = int(time.time())
    commitments = store.get_active_claims(claim_type="commitment", limit=50)
    prep = []

    for c in commitments:
        try:
            obj = json.loads(c.object) if isinstance(c.object, str) else c.object
        except (json.JSONDecodeError, TypeError):
            continue

        if obj.get("status") not in ("open", None, ""):
            continue

        deadline = obj.get("deadline")
        if not deadline:
            continue

        # Parse deadline
        dl_ts = None
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dl_ts = int(datetime.strptime(deadline, fmt).replace(tzinfo=timezone.utc).timestamp())
                break
            except ValueError:
                continue
        if dl_ts is None or dl_ts > event_start_ts:
            continue
        if dl_ts < now - 7 * 86400:
            continue  # skip very old overdue items

        what = obj.get("what", "")
        commitment_type = obj.get("type", "")
        source_channel = _source_channel_for_claim(store, c.id)

        # Skip stale time-bound events (past "attend", "leave", calendar events)
        if _is_stale_event(what, commitment_type, source_channel, deadline, now):
            continue

        # Check if any attendee is involved
        events = store.get_events_for_claim(c.id)
        involved_persons: set[str] = set()
        for ev in events:
            persons = store.get_persons_for_event(ev.id)
            for pid, _ in persons:
                involved_persons.add(pid)

        # Exclude user from both sets so the user's own person ID
        # (present in every commitment AND every calendar event) doesn't
        # cause false matches like "Pay Bank of America" in a Josie meeting.
        non_user_involved = involved_persons - user_ids
        non_user_attendees = attendee_person_ids - user_ids
        if non_user_involved & non_user_attendees:
            what = obj.get("what", "")
            if what:
                prep.append({
                    "id": c.id,
                    "title": what,
                    "deadline": deadline,
                    "priority": obj.get("priority", 3),
                })

    return prep[:5]  # Cap at 5 prep items per event


def handle_cq_get_agenda(
    store: LayeredGraphStore,
    lookahead_days: int | None = None,
    **kwargs,
) -> dict:
    """Get enriched agenda: calendar events grouped by time bands with KG context."""
    from alteris.constants import safe_timezone

    if lookahead_days is None:
        lookahead_days = _get_lookahead_days()
    lookback_days = _get_lookback_days()

    tz = safe_timezone()
    now = datetime.now(tz)
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_window = start_of_today + timedelta(days=lookahead_days + 1)

    start_ts = int(start_of_today.timestamp())
    end_ts = int(end_window.timestamp())

    user_ids = _get_user_identifiers(store)

    # Compute time bands
    bands = _compute_time_bands(now, lookahead_days)

    # Fetch all calendar events in the window
    rows = store.conn.execute(
        """SELECT id, metadata, timestamp, raw_content
           FROM events
           WHERE source = 'calendar'
             AND timestamp >= ? AND timestamp <= ?
           ORDER BY timestamp ASC""",
        (start_ts, end_ts),
    ).fetchall()

    # Parse events
    all_events: list[dict] = []
    for row in rows:
        meta = json.loads(row["metadata"] or "{}")
        event_id = row["id"]
        event_ts = row["timestamp"]

        title = meta.get("title", "") or meta.get("subject", "") or "(untitled)"
        duration_mins = meta.get("duration_minutes", 60)
        end_event_ts = event_ts + duration_mins * 60
        location = meta.get("location", "") or ""
        is_all_day = meta.get("all_day", False) or meta.get("is_all_day", False)
        is_recurring = meta.get("is_recurring", False) or meta.get("recurrence_rule") is not None
        event_url = meta.get("url", "") or meta.get("meeting_url", "") or ""
        calendar_name = meta.get("calendar_name", "") or meta.get("calendar", "") or ""
        # user_acceptance stripped — calendar defaults are unreliable
        # TODO: re-enable once we can detect actual vs default RSVP responses

        # Resolve attendees
        attendee_rows = store.conn.execute(
            """SELECT ep.person_id, ep.role, p.canonical_name
               FROM event_persons ep
               JOIN persons p ON ep.person_id = p.person_id
               WHERE ep.event_id = ?""",
            (event_id,),
        ).fetchall()

        raw_attendees = [
            {"person_id": r["person_id"], "name": r["canonical_name"], "role": r["role"]}
            for r in attendee_rows
        ]

        # Get date for band assignment
        event_dt = datetime.fromtimestamp(event_ts, tz=tz)
        event_date = event_dt.date().isoformat()

        # Format times in local tz
        start_time = event_dt.strftime("%I:%M %p").lstrip("0")
        end_dt = datetime.fromtimestamp(end_event_ts, tz=tz)
        end_time = end_dt.strftime("%I:%M %p").lstrip("0")

        all_events.append({
            "event_id": event_id,
            "title": title,
            "start_ts": event_ts,
            "end_ts": end_event_ts,
            "start_time": start_time,
            "end_time": end_time,
            "date": event_date,
            "location": location,
            "duration_minutes": duration_mins,
            "is_all_day": is_all_day,
            "is_recurring": is_recurring,
            "event_url": event_url,
            "calendar_name": calendar_name,
            "raw_attendees": raw_attendees,
            "attendee_count": len([a for a in raw_attendees if a["name"].lower() not in user_ids and a["person_id"].lower() not in user_ids]),
        })

    # Assign events to bands and enrich
    result_bands: list[dict] = []
    total_events = 0

    for band in bands:
        band_start = band["start_date"]
        band_end = band["end_date"]
        detail_level = band["detail_level"]

        band_events = [
            e for e in all_events
            if band_start <= e["date"] <= band_end
        ]

        enriched_events = []
        for ev in band_events:
            event_out: dict = {
                "event_id": ev["event_id"],
                "title": ev["title"],
                "start_time": ev["start_time"],
                "end_time": ev["end_time"],
                "date": ev["date"],
                "location": ev["location"],
                "duration_minutes": ev["duration_minutes"],
                "is_all_day": ev["is_all_day"],
                "is_recurring": ev["is_recurring"],
                "event_url": ev["event_url"],
                "calendar_name": ev["calendar_name"],
                "attendee_count": ev["attendee_count"],
            }

            # Enrich attendees based on detail level
            attendees = _enrich_attendees(
                store, ev["raw_attendees"], detail_level, user_ids,
            )
            event_out["attendees"] = attendees

            # Prep items for full-detail bands only
            if detail_level == "full":
                attendee_pids = {a["person_id"] for a in ev["raw_attendees"] if a.get("person_id")}
                prep = _get_prep_items(store, ev["start_ts"], attendee_pids, user_ids)
                if prep:
                    event_out["preparation_items"] = prep

            enriched_events.append(event_out)

        total_events += len(enriched_events)

        result_bands.append({
            "band_key": band["band_key"],
            "label": band["label"],
            "start_date": band_start,
            "end_date": band_end,
            "default_expanded": band["default_expanded"],
            "detail_level": detail_level,
            "events": enriched_events,
            "event_count": len(enriched_events),
        })

    return {
        "lookback_days": lookback_days,
        "lookahead_days": lookahead_days,
        "bands": result_bands,
        "total_events": total_events,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Horizon configuration tools
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def handle_cq_get_horizons(store: LayeredGraphStore, **kwargs) -> dict:
    """Get the current CQ horizon settings."""
    return {
        "lookback_days": _get_lookback_days(),
        "lookahead_days": _get_lookahead_days(),
    }


def handle_cq_set_horizons(
    store: LayeredGraphStore,
    lookback_days: int | None = None,
    lookahead_days: int | None = None,
    **kwargs,
) -> dict:
    """Set CQ horizon values in profile.yaml."""
    try:
        import yaml
    except ImportError:
        return {"error": "PyYAML not installed"}

    from alteris.constants import ALTERIS_DIR
    profile_path = ALTERIS_DIR / "profile.yaml"
    ALTERIS_DIR.mkdir(parents=True, exist_ok=True)

    existing = {}
    if profile_path.exists():
        try:
            data = yaml.safe_load(profile_path.read_text())
            existing = data if isinstance(data, dict) else {}
        except Exception:
            pass

    if lookback_days is not None:
        if not (1 <= lookback_days <= 365):
            return {"error": "lookback_days must be between 1 and 365"}
        existing["cq_lookback_days"] = lookback_days

    if lookahead_days is not None:
        if not (1 <= lookahead_days <= 365):
            return {"error": "lookahead_days must be between 1 and 365"}
        existing["cq_lookahead_days"] = lookahead_days

    profile_path.write_text(yaml.dump(existing, default_flow_style=False))

    return {
        "lookback_days": existing.get("cq_lookback_days", _get_lookback_days()),
        "lookahead_days": existing.get("cq_lookahead_days", _get_lookahead_days()),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chat tools (unchanged)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _cq_coach_tools() -> list[dict]:
    """Function declarations for the CQ coaching agent."""
    return [
        {
            "name": "move_task",
            "description": "Move a CQ task to a different bucket.",
            "parameters": {
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to move"},
                    "bucket": {"type": "string", "description": "Target bucket", "enum": ["immediate", "review", "background"]},
                },
                "required": ["task_id", "bucket"],
            },
        },
        {
            "name": "defer_task",
            "description": "Defer a CQ task until a future date.",
            "parameters": {
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to defer"},
                    "defer_until": {"type": "string", "description": "Date to defer until (YYYY-MM-DD)"},
                },
                "required": ["task_id", "defer_until"],
            },
        },
        {
            "name": "mark_done",
            "description": "Mark a CQ task as done.",
            "parameters": {
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to mark done"},
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "add_task",
            "description": "Add a new task to the CQ.",
            "parameters": {
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "bucket": {"type": "string", "description": "Target bucket", "enum": ["immediate", "review", "background"]},
                    "note": {"type": "string", "description": "Optional note"},
                    "due_date": {"type": "string", "description": "Due date (YYYY-MM-DD)"},
                },
                "required": ["title", "bucket"],
            },
        },
    ]


def _cq_tool_executor(store: LayeredGraphStore):
    """Returns a tool executor closure bound to the store."""
    def execute(name: str, args: dict) -> dict:
        try:
            if name == "move_task":
                handle_cq_move_task(store, task_id=args["task_id"], target_bucket=args["bucket"])
                return {"status": "ok", "action": f"Moved to {args['bucket']}"}
            elif name == "defer_task":
                handle_cq_update_task(store, task_id=args["task_id"], defer_until=args.get("defer_until"))
                return {"status": "ok", "action": f"Deferred until {args.get('defer_until')}"}
            elif name == "mark_done":
                handle_cq_update_task(store, task_id=args["task_id"], done=True)
                return {"status": "ok", "action": "Marked done"}
            elif name == "add_task":
                handle_cq_add_task(
                    store,
                    bucket=args.get("bucket", "review"),
                    title=args["title"],
                    note=args.get("note", ""),
                    due_date=args.get("due_date"),
                )
                return {"status": "ok", "action": f"Added '{args['title']}' to {args.get('bucket', 'review')}"}
            else:
                return {"error": f"Unknown tool: {name}"}
        except Exception as exc:
            return {"error": str(exc)}
    return execute


def handle_cq_chat(
    store: LayeredGraphStore,
    message: str = "",
    session_id: str | None = None,
    session_type: str = "clarity",
    **kwargs,
) -> dict:
    """AI coaching chat with full KG context and tool calling.

    The coach can directly modify CQ state (move, defer, done, add) via
    Gemini function calling — actions are executed in a loop before the
    final text response is returned.
    """
    if not message:
        return {"error": "message is required"}

    now = int(time.time())

    if session_id:
        session = store.get_cq_session(session_id)
        if session:
            messages = json.loads(session.get("messages", "[]"))
        else:
            messages = []
    else:
        session_id = f"cqs_{uuid.uuid4().hex[:12]}"
        messages = []

    messages.append({"role": "user", "content": message, "timestamp": now})
    context = _build_chat_context(store)

    actions_taken = []

    try:
        from alteris.llm.gemini import GeminiClient
        from alteris.constants import CQ_COACHING_MODEL

        llm = GeminiClient()
        system_prompt = _build_coaching_system_prompt(context)
        chat_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

        # Track actions via wrapper
        executor = _cq_tool_executor(store)
        def tracking_executor(name: str, args: dict) -> dict:
            result = executor(name, args)
            if result.get("status") == "ok":
                actions_taken.append(result["action"])
            return result

        response = llm.chat_with_tools(
            messages=chat_messages,
            tools=_cq_coach_tools(),
            system=system_prompt,
            model=CQ_COACHING_MODEL,
            temperature=0.4,
            max_tokens=1024,
            tool_executor=tracking_executor,
            max_tool_rounds=3,
        )
        if not response:
            response = "I'm having trouble connecting to the AI service. Please try again."

    except Exception as exc:
        logger.warning("CQ chat LLM error: %s", exc)
        response = f"AI coaching is temporarily unavailable: {exc}"

    messages.append({"role": "assistant", "content": response, "timestamp": int(time.time())})

    title = message[:50] if len(messages) <= 2 else ""
    store.put_cq_session(session_id, messages, title=title, session_type=session_type)

    result = {
        "session_id": session_id,
        "response": response,
        "message_count": len(messages),
    }
    if actions_taken:
        result["actions"] = actions_taken

    return result


def handle_cq_list_sessions(
    store: LayeredGraphStore,
    limit: int = 20,
    **kwargs,
) -> dict:
    """List recent chat sessions."""
    sessions = store.get_cq_sessions(limit=limit)
    result = []
    for s in sessions:
        msgs = json.loads(s.get("messages", "[]"))
        result.append({
            "id": s["id"],
            "title": s.get("title", ""),
            "session_type": s.get("session_type", "clarity"),
            "message_count": len(msgs),
            "created_at": s.get("created_at"),
            "updated_at": s.get("updated_at"),
        })
    return {"count": len(result), "sessions": result}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chat context helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_chat_context(store: LayeredGraphStore) -> dict:
    """Build KG context for the coaching chat.

    Includes the current CQ state (tasks with IDs in each bucket) so the
    LLM can reference specific tasks when using tools.
    """
    from alteris.constants import CQ_MAX_CONTEXT_BELIEFS

    now = int(time.time())

    # Current CQ state — the LLM needs task IDs to act on them
    cq_state = handle_cq_get_state(store)
    bucket_summaries = {}
    for bucket in ("immediate", "review", "background"):
        items = cq_state.get(bucket, [])
        lines = []
        for t in items:
            due = t.get("due_date") or "no deadline"
            priority = t.get("priority", "")
            p_str = f" P{priority}" if priority else ""
            lines.append(f"- [{t['id'][:16]}]{p_str} {t.get('title', '?')} (due: {due})")
        bucket_summaries[bucket] = lines

    # Upcoming calendar
    upcoming_events = store.get_events(since=now, limit=10)
    calendar_summaries = []
    for e in upcoming_events:
        if e.source == "calendar":
            meta = e.metadata if isinstance(e.metadata, dict) else {}
            title = meta.get("title", meta.get("subject", "Untitled"))
            ts = datetime.fromtimestamp(e.timestamp, tz=timezone.utc).strftime("%m/%d %H:%M")
            calendar_summaries.append(f"- {title} ({ts})")

    # Active beliefs (compact)
    beliefs = store.get_beliefs(status="active", limit=CQ_MAX_CONTEXT_BELIEFS)
    belief_summaries = [
        f"- {b.summary} ({b.confidence:.1f})"
        for b in beliefs[:15]
    ]

    return {
        "cq_immediate": bucket_summaries.get("immediate", []),
        "cq_review": bucket_summaries.get("review", []),
        "cq_background": bucket_summaries.get("background", []),
        "beliefs": belief_summaries,
        "upcoming_calendar": calendar_summaries,
    }


def _build_coaching_system_prompt(context: dict) -> str:
    """Build the system prompt for the CQ coaching chat."""
    immediate = "\n".join(context.get("cq_immediate", [])) or "Empty."
    review = "\n".join(context.get("cq_review", [])) or "Empty."
    background = "\n".join(context.get("cq_background", [])) or "Empty."
    beliefs = "\n".join(context.get("beliefs", [])) or "No beliefs yet."
    calendar = "\n".join(context.get("upcoming_calendar", [])) or "No upcoming events."

    return f"""You are a personal productivity coach. You have full visibility into the user's Clarity Queue and knowledge graph.

## Clarity Queue (current state)

### Immediate (P1 — act today)
{immediate}

### Review (P2 — this week)
{review}

### Background (P3+ — can wait)
{background}

### Upcoming Calendar
{calendar}

### Active Beliefs
{beliefs}

## Tools
You have tools to directly modify the Clarity Queue. USE THEM when the user asks you to reorganize, move, defer, complete, or add tasks. Don't just suggest — act.
- **move_task**(task_id, bucket): Move between immediate / review / background
- **defer_task**(task_id, defer_until): Hide until a future date (YYYY-MM-DD)
- **mark_done**(task_id): Mark complete
- **add_task**(title, bucket, note?, due_date?): Create a new task

Task IDs are shown in brackets like [abc123...]. Use the full ID shown when calling tools.

## Guidelines
- Be concise (2-4 sentences) unless asked for detail
- When you take actions, briefly confirm what you did
- Flag things falling through the cracks
- Reference specific people or topics from the knowledge graph when relevant"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

register_tool(ToolDef(
    name="cq_get_state",
    description="Get the Clarity Queue state: merges KG commitments/beliefs with user overrides into priority buckets. Filters deferred tasks, includes canUndo and source_channel per task.",
    permission="write",
    params=[
        ToolParam("date", "string", "Date to query (ISO format, default: today)"),
    ],
    handler=handle_cq_get_state,
))

register_tool(ToolDef(
    name="cq_add_task",
    description="Add a manual task to a Clarity Queue bucket (for items not already in the KG).",
    permission="write",
    params=[
        ToolParam("bucket", "string", "Target bucket", required=True, enum=["immediate", "background", "review"]),
        ToolParam("title", "string", "Task title", required=True),
        ToolParam("note", "string", "Optional note/details"),
        ToolParam("due_date", "string", "Due date (ISO format)"),
        ToolParam("labels", "array", "Labels/tags"),
        ToolParam("defer_until", "string", "Hide until this date (ISO format)"),
        ToolParam("recurrence", "object", "Recurrence rule: {freq: daily|weekly|monthly|yearly, interval: int}"),
        ToolParam("custom_fields", "object", "User-defined key-value pairs"),
        ToolParam("category", "string", "Category/topic tag"),
    ],
    handler=handle_cq_add_task,
))

register_tool(ToolDef(
    name="cq_update_task",
    description="Update a CQ task. For KG items, creates an override. When marking a recurring task done, auto-creates next instance.",
    permission="write",
    params=[
        ToolParam("task_id", "string", "Task ID", required=True),
        ToolParam("title", "string", "New title"),
        ToolParam("note", "string", "New note"),
        ToolParam("done", "boolean", "Mark as done/undone"),
        ToolParam("due_date", "string", "New due date"),
        ToolParam("labels", "array", "New labels"),
        ToolParam("defer_until", "string", "Defer until date (ISO format)"),
        ToolParam("recurrence", "object", "Recurrence rule"),
        ToolParam("custom_fields", "object", "Custom key-value pairs"),
        ToolParam("category", "string", "Category tag"),
        ToolParam("accepted", "boolean", "Accept task into Desk"),
    ],
    handler=handle_cq_update_task,
))

register_tool(ToolDef(
    name="cq_move_task",
    description="Move a task to a different CQ bucket. Creates an override for KG items.",
    permission="write",
    params=[
        ToolParam("task_id", "string", "Task ID", required=True),
        ToolParam("target_bucket", "string", "Destination bucket", required=True, enum=["immediate", "background", "review"]),
        ToolParam("position", "integer", "Position in target bucket (default: end)", default=0),
    ],
    handler=handle_cq_move_task,
))

register_tool(ToolDef(
    name="cq_dismiss",
    description="Dismiss a KG item from CQ (won't show again). Deletes manual tasks.",
    permission="write",
    params=[
        ToolParam("task_id", "string", "Task ID to dismiss", required=True),
    ],
    handler=handle_cq_dismiss,
))

register_tool(ToolDef(
    name="cq_restore",
    description="Restore a dismissed task back to the Clarity Queue.",
    permission="write",
    params=[
        ToolParam("task_id", "string", "Task ID to restore", required=True),
    ],
    handler=handle_cq_restore,
))

register_tool(ToolDef(
    name="cq_reorder",
    description="Reorder tasks within a CQ bucket by providing the desired order of task IDs.",
    permission="write",
    params=[
        ToolParam("bucket", "string", "Bucket to reorder", required=True),
        ToolParam("task_ids", "array", "Ordered list of task IDs", required=True),
    ],
    handler=handle_cq_reorder,
))

register_tool(ToolDef(
    name="cq_undo",
    description="Undo the last CQ action (dismiss, toggle done, move, update, add).",
    permission="write",
    params=[],
    handler=handle_cq_undo,
))

register_tool(ToolDef(
    name="cq_get_all",
    description="Get ALL tasks across all statuses: active, deferred, dismissed, done. Includes merge chains and source attribution.",
    permission="write",
    params=[],
    handler=handle_cq_get_all,
))

register_tool(ToolDef(
    name="cq_get_thread",
    description="Get the originating email/message thread for a task. Traverses claim -> events -> thread.",
    permission="read",
    params=[
        ToolParam("task_id", "string", "Task ID to get thread for", required=True),
    ],
    handler=handle_cq_get_thread,
))

register_tool(ToolDef(
    name="cq_manage_sender_rules",
    description="Add, remove, or list sender priority rules. Priority tiers: P1 (immediate), P2 (important), P3 (normal), block.",
    permission="write",
    params=[
        ToolParam("action", "string", "Action to perform", required=True, enum=["list", "add", "delete"]),
        ToolParam("pattern", "string", "Email, phone, or domain pattern (for add)"),
        ToolParam("priority", "string", "Priority tier (for add)", enum=["P1", "P2", "P3", "block"]),
        ToolParam("source", "string", "Source filter: mail, imessage, whatsapp, or empty for all"),
        ToolParam("note", "string", "Note about why this rule exists"),
        ToolParam("rule_id", "integer", "Rule ID (for delete)"),
    ],
    handler=handle_cq_manage_sender_rules,
))

register_tool(ToolDef(
    name="cq_manage_categories",
    description="Add, remove, or list user-defined task categories.",
    permission="write",
    params=[
        ToolParam("action", "string", "Action to perform", required=True, enum=["list", "add", "delete"]),
        ToolParam("name", "string", "Category name"),
        ToolParam("color", "string", "Hex color for UI"),
        ToolParam("icon", "string", "SF Symbol name for icon"),
    ],
    handler=handle_cq_manage_categories,
))

register_tool(ToolDef(
    name="cq_manage_extractable_fields",
    description="Add, remove, or list user-defined extractable field definitions. These fields are injected into the synthesis prompt so the LLM extracts them as custom_fields on each commitment (best-effort).",
    permission="write",
    params=[
        ToolParam("action", "string", "Action to perform", required=True, enum=["list", "add", "delete"]),
        ToolParam("name", "string", "Field name (e.g. 'budget', 'location')"),
        ToolParam("description", "string", "What to extract (e.g. 'Dollar amount if mentioned')"),
        ToolParam("example", "string", "Example value (e.g. '$5,000')"),
    ],
    handler=handle_cq_manage_extractable_fields,
))

register_tool(ToolDef(
    name="cq_get_agenda",
    description="Get enriched agenda: calendar events grouped by time bands with attendee KG context, prep items, and communication history. Today/tomorrow get full enrichment; further out gets compact rows.",
    permission="read",
    params=[
        ToolParam("lookahead_days", "integer", "Override lookahead window (default: from profile or 30)"),
    ],
    handler=handle_cq_get_agenda,
))

register_tool(ToolDef(
    name="cq_get_horizons",
    description="Get the current Clarity Queue horizon settings (lookback and lookahead days).",
    permission="read",
    params=[],
    handler=handle_cq_get_horizons,
))

register_tool(ToolDef(
    name="cq_set_horizons",
    description="Set Clarity Queue horizon values. Lookback = how far back to show overdue items. Lookahead = how far forward to show upcoming items.",
    permission="write",
    params=[
        ToolParam("lookback_days", "integer", "Past window in days (1-365)"),
        ToolParam("lookahead_days", "integer", "Future window in days (1-365)"),
    ],
    handler=handle_cq_set_horizons,
))

register_tool(ToolDef(
    name="cq_chat",
    description="AI coaching chat for productivity. Has full knowledge graph context. Use for daily planning, task prioritization, and surfacing things the user might be missing.",
    permission="write",
    params=[
        ToolParam("message", "string", "User message", required=True),
        ToolParam("session_id", "string", "Session ID to continue (or new session if omitted)"),
        ToolParam("session_type", "string", "Session type", default="clarity", enum=["clarity", "coaching", "onboarding"]),
    ],
    handler=handle_cq_chat,
))

register_tool(ToolDef(
    name="cq_list_sessions",
    description="List recent CQ chat sessions.",
    permission="write",
    params=[
        ToolParam("limit", "integer", "Max sessions (default 20)", default=20),
    ],
    handler=handle_cq_list_sessions,
))
