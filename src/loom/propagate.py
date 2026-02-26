"""Stage 5: Graph message passing for triage score adjustment.

Runs after triage (Stage 4), before extraction (Stage 6).
Adjusts triage claim confidence using structural signals that
single-item LLM classification cannot see. Pure Python/SQL, no LLM.

Five propagation rules:
1. Sender reputation dampening -- catch newsletter urgency-bait
2. Thread score propagation -- coherence within conversation threads
3. Contact tier boost -- slight bump for recent inner-circle messages
4. Recipient dampening -- CC'd and mass emails get reduced
5. Source type prior -- user-participated threads and meetings bump

Architecture:
  - Input: triage Claims (claim_type="triage", confidence=score)
  - Working set: in-memory dicts built from claims + events + event_persons
  - Output: updated claims.confidence values
  - Downstream: get_deep_extraction_candidates() reads claims.confidence
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from loom.constants import (
    PROPAGATION_CONVERGENCE_THRESHOLD,
    PROPAGATION_MAX_ROUNDS,
    SECONDS_PER_DAY,
)
from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER_IGNORE = 0.3
TIER_DEEP = 0.7

# Rule 1: sender reputation
MIN_SENDER_MSGS = 5
OUTLIER_THRESHOLD = 0.4
OUTLIER_CAP_DELTA = 0.2
OUTLIER_FLOOR = 0.3

# Rule 2: thread coherence
LOW_OUTLIER_GAP = 0.2
THREAD_ACTIVE_MEAN = 0.3
STRONG_PULL_DELTA = 0.4
STANDARD_PULL_DELTA = 0.2
HIGH_OUTLIER_GAP = 0.4
HIGH_PULL_DOWN = 0.2

# Rule 3: contact tier boost
TIER_BOOST_LO = 0.3
TIER_BOOST_HI = 0.4
TIER_BOOST_DELTA = 0.1
TIER_BOOST_CAP = 0.5
TIER_BOOST_RECENCY_DAYS = 30

# Rule 4: recipient dampening
MASS_EMAIL_THRESHOLD = 5
RECIPIENT_DAMPEN = 0.2
RECIPIENT_FLOOR = 0.3

# Rule 5: source type prior
MSG_FLOOR_IF_USER_PARTICIPATED = 0.3
MEETING_FLOOR = 0.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TriagedItem:
    """Working representation of a triaged event during propagation."""
    event_id: str
    claim_id: str
    original_score: float
    effective_score: float
    event_type: str
    source: str
    timestamp: int
    thread_id: str
    sender_person_id: str
    sender_is_user: bool
    sender_tier: int
    cc_recipients: list[str] = field(default_factory=list)
    total_recipients: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class PropagationStats:
    """Track changes from a single propagation rule."""
    rule_name: str
    dampened: int = 0
    promoted: int = 0
    total_delta: float = 0.0

    @property
    def total_changed(self) -> int:
        return self.dampened + self.promoted

    @property
    def avg_delta(self) -> float:
        if self.total_changed == 0:
            return 0.0
        return self.total_delta / self.total_changed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Working set construction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_sender_cache(store: LayeredGraphStore) -> dict[str, dict]:
    """Build {person_id: {tier, is_user}} from frequency claims + persons."""
    cache: dict[str, dict] = {}

    rows = store.conn.execute(
        "SELECT person_id, is_user FROM persons"
    ).fetchall()
    for r in rows:
        cache[r["person_id"]] = {
            "tier": 3,
            "is_user": bool(r["is_user"]),
        }

    rows = store.conn.execute(
        """SELECT predicate, object FROM claims
           WHERE claim_type = 'communication_frequency'
             AND superseded_by IS NULL"""
    ).fetchall()

    for r in rows:
        pred = r["predicate"]
        if not pred.startswith("communicates_with:"):
            continue
        pid = pred.replace("communicates_with:", "")
        try:
            obj = json.loads(r["object"])
            total = obj.get("event_count", 0)
        except (json.JSONDecodeError, TypeError):
            continue

        if pid not in cache:
            cache[pid] = {"tier": 3, "is_user": False}

        if total >= 50:
            cache[pid]["tier"] = 1
        elif total >= 10:
            cache[pid]["tier"] = 2

    return cache


def _build_working_set(
    store: LayeredGraphStore,
    sender_cache: dict[str, dict],
) -> dict[str, TriagedItem]:
    """Load all triage claims and join with event/person data."""
    rows = store.conn.execute(
        """SELECT c.id AS claim_id, c.subject AS event_id,
                  c.confidence, c.object AS triage_json
           FROM claims c
           WHERE c.claim_type = 'triage'
             AND c.predicate = 'triage_result'
             AND c.superseded_by IS NULL"""
    ).fetchall()

    claim_data = {}
    for r in rows:
        try:
            obj = json.loads(r["triage_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        claim_data[r["event_id"]] = {
            "claim_id": r["claim_id"],
            "confidence": r["confidence"],
            "original_score": obj.get("score", r["confidence"]),
        }

    if not claim_data:
        return {}

    placeholders = ",".join("?" * len(claim_data))
    event_ids = list(claim_data.keys())

    event_rows = store.conn.execute(
        f"""SELECT id, event_type, source, timestamp, metadata
            FROM events WHERE id IN ({placeholders})""",
        event_ids,
    ).fetchall()

    event_meta = {}
    for r in event_rows:
        try:
            meta = json.loads(r["metadata"] or "{}")
        except (json.JSONDecodeError, TypeError):
            meta = {}
        event_meta[r["id"]] = {
            "event_type": r["event_type"],
            "source": r["source"],
            "timestamp": r["timestamp"],
            "thread_id": meta.get("thread_id", ""),
            "metadata": meta,
        }

    sender_rows = store.conn.execute(
        f"""SELECT ep.event_id, ep.person_id, p.is_user
            FROM event_persons ep
            JOIN persons p ON ep.person_id = p.person_id
            WHERE ep.role = 'sender'
              AND ep.event_id IN ({placeholders})""",
        event_ids,
    ).fetchall()

    event_sender = {}
    for r in sender_rows:
        event_sender[r["event_id"]] = {
            "person_id": r["person_id"],
            "is_user": bool(r["is_user"]),
        }

    recip_rows = store.conn.execute(
        f"""SELECT event_id, COUNT(*) as cnt
            FROM event_persons
            WHERE role IN ('recipient', 'cc_recipient', 'cc')
              AND event_id IN ({placeholders})
            GROUP BY event_id""",
        event_ids,
    ).fetchall()
    recip_counts = {r["event_id"]: r["cnt"] for r in recip_rows}

    cc_rows = store.conn.execute(
        f"""SELECT event_id, person_id
            FROM event_persons
            WHERE role IN ('cc_recipient', 'cc')
              AND event_id IN ({placeholders})""",
        event_ids,
    ).fetchall()
    cc_map: dict[str, list[str]] = {}
    for r in cc_rows:
        cc_map.setdefault(r["event_id"], []).append(r["person_id"])

    items: dict[str, TriagedItem] = {}
    for eid, cd in claim_data.items():
        em = event_meta.get(eid)
        if not em:
            continue

        sender = event_sender.get(
            eid, {"person_id": "", "is_user": False}
        )
        sender_pid = sender["person_id"]
        sender_info = sender_cache.get(
            sender_pid, {"tier": 3, "is_user": False}
        )

        items[eid] = TriagedItem(
            event_id=eid,
            claim_id=cd["claim_id"],
            original_score=cd["original_score"],
            effective_score=cd["confidence"],
            event_type=em["event_type"],
            source=em["source"],
            timestamp=em["timestamp"],
            thread_id=em["thread_id"],
            sender_person_id=sender_pid,
            sender_is_user=sender["is_user"] or sender_info["is_user"],
            sender_tier=sender_info["tier"],
            cc_recipients=cc_map.get(eid, []),
            total_recipients=recip_counts.get(eid, 0),
            metadata=em["metadata"],
        )

    return items


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Propagation rules
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def rule_sender_reputation(
    items: dict[str, TriagedItem],
) -> PropagationStats:
    """Rule 1: Dampen outlier scores from senders with low median.

    If a sender has 5+ triaged messages and their median score is low,
    any individual message scoring much higher is likely misclassified.
    Exempt: tier-1/2 senders and user's own messages.
    """
    stats = PropagationStats(rule_name="sender_reputation")

    by_sender: dict[str, list[TriagedItem]] = {}
    for item in items.values():
        if not item.sender_person_id:
            continue
        by_sender.setdefault(item.sender_person_id, []).append(item)

    for sender_pid, sender_items in by_sender.items():
        if len(sender_items) < MIN_SENDER_MSGS:
            continue

        sample = sender_items[0]
        if sample.sender_tier <= 2:
            continue
        if sample.sender_is_user:
            continue

        scores = sorted(item.effective_score for item in sender_items)
        med = median(scores)
        threshold = med + OUTLIER_THRESHOLD
        cap = max(med + OUTLIER_CAP_DELTA, OUTLIER_FLOOR)

        for item in sender_items:
            if item.effective_score > threshold:
                old = item.effective_score
                new = max(cap, OUTLIER_FLOOR)
                if new < old:
                    item.effective_score = new
                    stats.dampened += 1
                    stats.total_delta += (old - new)

    return stats


def rule_thread_coherence(
    items: dict[str, TriagedItem],
) -> PropagationStats:
    """Rule 2: Propagate scores within conversation threads.

    Low outlier in active thread -> pull up.
    High outlier in quiet thread -> pull down.
    User-sent messages are never dampened.
    """
    stats = PropagationStats(rule_name="thread_coherence")

    by_thread: dict[str, list[TriagedItem]] = {}
    for item in items.values():
        if not item.thread_id:
            continue
        by_thread.setdefault(item.thread_id, []).append(item)

    for thread_id, thread_items in by_thread.items():
        if len(thread_items) < 2:
            continue

        scores = [item.effective_score for item in thread_items]
        mean_score = sum(scores) / len(scores)
        max_score = max(scores)
        deep_count = sum(1 for s in scores if s >= TIER_DEEP)

        for item in thread_items:
            old = item.effective_score
            new = old

            # Low outlier in active thread -> pull up
            if (old < mean_score - LOW_OUTLIER_GAP
                    and mean_score >= THREAD_ACTIVE_MEAN):
                if deep_count >= 2 and old < TIER_DEEP:
                    new = min(
                        old + STRONG_PULL_DELTA,
                        max_score - 0.1,
                        TIER_DEEP,
                    )
                else:
                    new = min(old + STANDARD_PULL_DELTA, mean_score)

            # High outlier in quiet thread -> pull down
            if (old > mean_score + HIGH_OUTLIER_GAP
                    and old >= max_score
                    and not item.sender_is_user):
                new = max(old - HIGH_PULL_DOWN, mean_score + 0.1)

            if new != old:
                new = max(0.0, min(1.0, new))
                item.effective_score = new
                if new > old:
                    stats.promoted += 1
                else:
                    stats.dampened += 1
                stats.total_delta += abs(new - old)

    return stats


def rule_contact_tier_boost(
    items: dict[str, TriagedItem],
    colleague_pids: set[str] | None = None,
) -> PropagationStats:
    """Rule 3: Slight boost for recent messages from inner circle.

    Messages from tier-1 contacts scoring 0.3-0.4 that are less than
    30 days old get a +0.1 bump. Profile colleagues get the same boost.
    """
    stats = PropagationStats(rule_name="contact_tier_boost")
    now = int(time.time())
    cutoff = now - (TIER_BOOST_RECENCY_DAYS * SECONDS_PER_DAY)

    for item in items.values():
        if (item.sender_tier == 1
                and TIER_BOOST_LO <= item.effective_score <= TIER_BOOST_HI
                and item.timestamp > cutoff):
            old = item.effective_score
            new = min(old + TIER_BOOST_DELTA, TIER_BOOST_CAP)
            if new > old:
                item.effective_score = new
                stats.promoted += 1
                stats.total_delta += (new - old)

    # Boost items from profile colleagues (same logic as tier-1)
    if colleague_pids:
        for item in items.values():
            if (item.sender_person_id in colleague_pids
                    and TIER_BOOST_LO <= item.effective_score <= TIER_BOOST_HI
                    and item.timestamp > cutoff
                    and item.sender_tier != 1):  # avoid double-boost
                old = item.effective_score
                new = min(old + TIER_BOOST_DELTA, TIER_BOOST_CAP)
                if new > old:
                    item.effective_score = new
                    stats.promoted += 1
                    stats.total_delta += (new - old)

    return stats


def rule_recipient_dampening(
    items: dict[str, TriagedItem],
) -> PropagationStats:
    """Rule 4: Dampen CC'd and mass-email items.

    CC'd or 5+ recipients = likely mass email. Tier-1/2 senders exempt.
    """
    stats = PropagationStats(rule_name="recipient_dampening")

    for item in items.values():
        if item.effective_score < 0.5:
            continue
        if item.sender_tier <= 2:
            continue

        should_dampen = False

        if item.cc_recipients and item.event_type == "email":
            should_dampen = True

        if item.total_recipients >= MASS_EMAIL_THRESHOLD:
            should_dampen = True

        if should_dampen:
            old = item.effective_score
            new = max(old - RECIPIENT_DAMPEN, RECIPIENT_FLOOR)
            if new < old:
                item.effective_score = new
                stats.dampened += 1
                stats.total_delta += (old - new)

    return stats


def rule_source_type_prior(
    items: dict[str, TriagedItem],
) -> PropagationStats:
    """Rule 5: Adjust scores based on source type prior.

    - Messages in user-participated threads -> bump to 0.3
    - Meetings -> bump to 0.5
    """
    stats = PropagationStats(rule_name="source_type_prior")

    user_threads: set[str] = set()
    for item in items.values():
        if (item.event_type == "message"
                and item.sender_is_user
                and item.thread_id):
            user_threads.add(item.thread_id)

    for item in items.values():
        if (item.event_type == "message"
                and item.effective_score < MSG_FLOOR_IF_USER_PARTICIPATED):
            if item.thread_id and item.thread_id in user_threads:
                old = item.effective_score
                new = MSG_FLOOR_IF_USER_PARTICIPATED
                if new > old:
                    item.effective_score = new
                    stats.promoted += 1
                    stats.total_delta += (new - old)

    for item in items.values():
        if (item.event_type == "meeting"
                and item.effective_score < MEETING_FLOOR):
            old = item.effective_score
            new = MEETING_FLOOR
            if new > old:
                item.effective_score = new
                stats.promoted += 1
                stats.total_delta += (new - old)

    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Write-back
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _write_back(
    store: LayeredGraphStore, items: dict[str, TriagedItem],
) -> int:
    """Update claims.confidence for items whose score changed."""
    updated = 0
    for item in items.values():
        if abs(item.effective_score - item.original_score) < 1e-6:
            continue

        store.conn.execute(
            "UPDATE claims SET confidence = ? WHERE id = ?",
            (round(item.effective_score, 3), item.claim_id),
        )
        updated += 1

    store.conn.commit()
    return updated


def _get_tier_distribution(
    items: dict[str, TriagedItem],
) -> dict[str, int]:
    """Get current tier distribution from working set."""
    dist = {"ignore": 0, "lightweight": 0, "deep": 0}
    for item in items.values():
        s = item.effective_score
        if s < TIER_IGNORE:
            dist["ignore"] += 1
        elif s < TIER_DEEP:
            dist["lightweight"] += 1
        else:
            dist["deep"] += 1
    return dist


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_propagation(
    store: LayeredGraphStore,
    max_iterations: int = PROPAGATION_MAX_ROUNDS,
    convergence_threshold: float = PROPAGATION_CONVERGENCE_THRESHOLD,
    dry_run: bool = False,
    profile: dict | None = None,
) -> dict[str, Any]:
    """Run all message passing rules iteratively until convergence.

    Args:
        store: LayeredGraphStore with triage claims already stored.
        max_iterations: Maximum propagation rounds.
        convergence_threshold: Stop when changes < this fraction.
        dry_run: If True, don't write changes back to database.
        profile: Optional user profile dict. If provided, colleague names
            are resolved to person_ids for contact tier boosting.

    Returns dict with propagation statistics.
    """
    start_time = time.time()

    sender_cache = _build_sender_cache(store)
    items = _build_working_set(store, sender_cache)

    if not items:
        logger.warning("No triaged items found. Run triage first.")
        return {"error": "no_triaged_items", "count": 0}

    logger.info("Propagation: %d triaged items loaded", len(items))

    # Resolve profile colleague names to person_ids
    colleague_pids: set[str] = set()
    if profile:
        from loom.profile import get_colleague_names
        colleague_names = get_colleague_names(profile)
        if colleague_names:
            # Match colleague names against person canonical_names
            rows = store.conn.execute(
                "SELECT person_id, canonical_name FROM person_profiles WHERE canonical_name IS NOT NULL"
            ).fetchall()
            name_to_pid: dict[str, str] = {}
            for row in rows:
                cn = row["canonical_name"]
                if cn:
                    name_to_pid[cn.lower()] = row["person_id"]
            for cname in colleague_names:
                # Try exact match, then prefix match
                key = cname.lower().strip()
                if key in name_to_pid:
                    colleague_pids.add(name_to_pid[key])
                else:
                    # Fuzzy: check if colleague name is a prefix of any canonical name
                    for db_name, pid in name_to_pid.items():
                        if db_name.startswith(key) or key.startswith(db_name):
                            colleague_pids.add(pid)
            if colleague_pids:
                logger.info("Profile colleagues resolved to %d person_ids", len(colleague_pids))

    pre_tiers = _get_tier_distribution(items)

    rules = [
        ("sender_reputation", rule_sender_reputation),
        ("thread_coherence", rule_thread_coherence),
        ("contact_tier_boost", rule_contact_tier_boost),
        ("recipient_dampening", rule_recipient_dampening),
        ("source_type_prior", rule_source_type_prior),
    ]

    iterations_run = 0
    for iteration in range(max_iterations):
        iteration_stats = []
        for rule_name, rule_fn in rules:
            if rule_name == "contact_tier_boost":
                result = rule_fn(items, colleague_pids=colleague_pids or None)
            else:
                result = rule_fn(items)
            iteration_stats.append(result)
            logger.info(
                "  Iter %d | %s: %d dampened, %d promoted",
                iteration + 1, result.rule_name,
                result.dampened, result.promoted,
            )

        total_changed = sum(r.total_changed for r in iteration_stats)
        change_rate = total_changed / max(len(items), 1)
        iterations_run = iteration + 1

        logger.info(
            "  Iter %d total: %d changes (%.1f%% of items)",
            iteration + 1, total_changed, change_rate * 100,
        )

        if change_rate < convergence_threshold:
            logger.info("  Converged at iteration %d", iteration + 1)
            break

    post_tiers = _get_tier_distribution(items)

    updated_count = 0
    if not dry_run:
        updated_count = _write_back(store, items)
        logger.info("Wrote %d adjusted scores to claims", updated_count)

    elapsed = time.time() - start_time

    # Count tier transitions
    transitions: dict[str, int] = {}
    for item in items.values():
        if abs(item.effective_score - item.original_score) < 1e-6:
            continue
        old_tier = (
            "ignore" if item.original_score < TIER_IGNORE
            else "deep" if item.original_score >= TIER_DEEP
            else "lightweight"
        )
        new_tier = (
            "ignore" if item.effective_score < TIER_IGNORE
            else "deep" if item.effective_score >= TIER_DEEP
            else "lightweight"
        )
        if old_tier != new_tier:
            key = f"{old_tier}->{new_tier}"
            transitions[key] = transitions.get(key, 0) + 1

    return {
        "iterations": iterations_run,
        "triaged_count": len(items),
        "updated_count": updated_count,
        "pre_tiers": pre_tiers,
        "post_tiers": post_tiers,
        "transitions": transitions,
        "elapsed_seconds": round(elapsed, 2),
        "dry_run": dry_run,
    }
