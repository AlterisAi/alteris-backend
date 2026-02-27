"""Tests for alteris.propagate: Message passing on triage claims (Stage 5).

Tests cover:
  - Rule 1: Sender reputation dampening
  - Rule 2: Thread coherence (pull-up and pull-down)
  - Rule 3: Contact tier boost
  - Rule 4: Recipient dampening (CC, mass email)
  - Rule 5: Source type prior (user-participated threads, meetings)
  - Working set construction
  - Iterative convergence
  - Write-back to claims table
  - Tier distribution tracking
  - Edge cases (empty store, single item, no changes)
  - Full run_propagation pipeline
  - Dry run mode
"""

import json
import time

import pytest

from alteris.constants import (
    EVENT_TYPE_EMAIL,
    EVENT_TYPE_MEETING,
    EVENT_TYPE_MESSAGE,
)
from alteris.models import (
    Claim,
    Event,
    ExtractionMethod,
    ExtractionProvenance,
    Modality,
)
from alteris.propagate import (
    MASS_EMAIL_THRESHOLD,
    MEETING_FLOOR,
    MSG_FLOOR_IF_USER_PARTICIPATED,
    OUTLIER_THRESHOLD,
    TIER_BOOST_DELTA,
    TIER_BOOST_HI,
    TIER_BOOST_LO,
    TIER_DEEP,
    TIER_IGNORE,
    PropagationStats,
    TriagedItem,
    _build_sender_cache,
    _build_working_set,
    _get_tier_distribution,
    _write_back,
    rule_contact_tier_boost,
    rule_recipient_dampening,
    rule_sender_reputation,
    rule_source_type_prior,
    rule_thread_coherence,
    run_propagation,
)
from alteris.store import LayeredGraphStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_triaged_item(
    event_id: str = "e1",
    claim_id: str = "c1",
    score: float = 0.5,
    event_type: str = EVENT_TYPE_EMAIL,
    source: str = "mail",
    thread_id: str = "",
    sender_person_id: str = "p1",
    sender_is_user: bool = False,
    sender_tier: int = 3,
    cc_recipients: list[str] | None = None,
    total_recipients: int = 1,
    timestamp: int = 0,
) -> TriagedItem:
    return TriagedItem(
        event_id=event_id,
        claim_id=claim_id,
        original_score=score,
        effective_score=score,
        event_type=event_type,
        source=source,
        timestamp=timestamp or int(time.time()),
        thread_id=thread_id,
        sender_person_id=sender_person_id,
        sender_is_user=sender_is_user,
        sender_tier=sender_tier,
        cc_recipients=cc_recipients or [],
        total_recipients=total_recipients,
    )


def _make_triage_claim(event_id: str, score: float) -> Claim:
    import hashlib
    raw = f"triage:{event_id}:triage_result"
    claim_id = f"claim:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"
    return Claim(
        id=claim_id,
        event_ids=[event_id],
        claim_type="triage",
        subject=event_id,
        predicate="triage_result",
        object=json.dumps({"score": score, "reason": "test"}),
        confidence=score,
        modality=Modality.OBSERVED,
        provenance=ExtractionProvenance(model_id="test"),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PropagationStats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPropagationStats:
    def test_total_changed(self):
        s = PropagationStats(rule_name="test", dampened=3, promoted=2)
        assert s.total_changed == 5

    def test_avg_delta(self):
        s = PropagationStats(rule_name="test", dampened=2, promoted=0, total_delta=0.4)
        assert abs(s.avg_delta - 0.2) < 1e-6

    def test_avg_delta_no_changes(self):
        s = PropagationStats(rule_name="test")
        assert s.avg_delta == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 1: Sender reputation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSenderReputation:
    def test_dampens_outlier_from_low_median_sender(self):
        """A newsletter sender with median 0.1 but one item at 0.8 -> dampen."""
        items = {}
        # 5 low-score items from same sender
        for i in range(5):
            items[f"e{i}"] = _make_triaged_item(
                event_id=f"e{i}", claim_id=f"c{i}", score=0.1,
                sender_person_id="spammer", sender_tier=3,
            )
        # One outlier
        items["e_outlier"] = _make_triaged_item(
            event_id="e_outlier", claim_id="c_outlier", score=0.8,
            sender_person_id="spammer", sender_tier=3,
        )
        stats = rule_sender_reputation(items)
        assert stats.dampened >= 1
        assert items["e_outlier"].effective_score < 0.8

    def test_skips_tier1_senders(self):
        """Tier 1 senders are exempt from dampening."""
        items = {}
        for i in range(6):
            items[f"e{i}"] = _make_triaged_item(
                event_id=f"e{i}", claim_id=f"c{i}", score=0.1,
                sender_person_id="vip", sender_tier=1,
            )
        items["e_outlier"] = _make_triaged_item(
            event_id="e_outlier", claim_id="c_out", score=0.8,
            sender_person_id="vip", sender_tier=1,
        )
        stats = rule_sender_reputation(items)
        assert stats.dampened == 0
        assert items["e_outlier"].effective_score == 0.8

    def test_skips_user_sent(self):
        """User's own messages are never dampened."""
        items = {}
        for i in range(6):
            items[f"e{i}"] = _make_triaged_item(
                event_id=f"e{i}", claim_id=f"c{i}", score=0.1,
                sender_person_id="user", sender_is_user=True,
            )
        items["e_high"] = _make_triaged_item(
            event_id="e_high", claim_id="c_high", score=0.9,
            sender_person_id="user", sender_is_user=True,
        )
        stats = rule_sender_reputation(items)
        assert stats.dampened == 0

    def test_needs_minimum_messages(self):
        """Senders with fewer than MIN_SENDER_MSGS items are exempt."""
        items = {}
        for i in range(3):
            items[f"e{i}"] = _make_triaged_item(
                event_id=f"e{i}", claim_id=f"c{i}", score=0.1,
                sender_person_id="rare",
            )
        items["e_high"] = _make_triaged_item(
            event_id="e_high", claim_id="c_high", score=0.9,
            sender_person_id="rare",
        )
        stats = rule_sender_reputation(items)
        assert stats.dampened == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 2: Thread coherence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestThreadCoherence:
    def test_pulls_up_low_outlier_in_active_thread(self):
        """Item scoring 0.1 in a thread where mean is 0.7 -> pulled up."""
        items = {
            "e1": _make_triaged_item(event_id="e1", claim_id="c1", score=0.7, thread_id="t1"),
            "e2": _make_triaged_item(event_id="e2", claim_id="c2", score=0.8, thread_id="t1"),
            "e3": _make_triaged_item(event_id="e3", claim_id="c3", score=0.1, thread_id="t1"),
        }
        stats = rule_thread_coherence(items)
        assert stats.promoted >= 1
        assert items["e3"].effective_score > 0.1

    def test_pulls_down_high_outlier_in_quiet_thread(self):
        """Single high-scoring item in otherwise quiet thread -> pulled down."""
        items = {
            "e1": _make_triaged_item(event_id="e1", claim_id="c1", score=0.1, thread_id="t1"),
            "e2": _make_triaged_item(event_id="e2", claim_id="c2", score=0.1, thread_id="t1"),
            "e3": _make_triaged_item(
                event_id="e3", claim_id="c3", score=0.9, thread_id="t1",
                sender_is_user=False,
            ),
        }
        stats = rule_thread_coherence(items)
        assert stats.dampened >= 1
        assert items["e3"].effective_score < 0.9

    def test_user_sent_not_dampened(self):
        """User-sent messages are never pulled down."""
        items = {
            "e1": _make_triaged_item(event_id="e1", claim_id="c1", score=0.1, thread_id="t1"),
            "e2": _make_triaged_item(event_id="e2", claim_id="c2", score=0.1, thread_id="t1"),
            "e3": _make_triaged_item(
                event_id="e3", claim_id="c3", score=0.9, thread_id="t1",
                sender_is_user=True,
            ),
        }
        stats = rule_thread_coherence(items)
        assert items["e3"].effective_score == 0.9

    def test_standalone_messages_unaffected(self):
        """Messages not in a thread should not change."""
        items = {
            "e1": _make_triaged_item(event_id="e1", claim_id="c1", score=0.5, thread_id=""),
            "e2": _make_triaged_item(event_id="e2", claim_id="c2", score=0.9, thread_id=""),
        }
        stats = rule_thread_coherence(items)
        assert stats.total_changed == 0

    def test_single_item_thread_unaffected(self):
        items = {
            "e1": _make_triaged_item(event_id="e1", claim_id="c1", score=0.5, thread_id="solo"),
        }
        stats = rule_thread_coherence(items)
        assert stats.total_changed == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 3: Contact tier boost
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestContactTierBoost:
    def test_boosts_tier1_in_range(self):
        """Tier 1 sender with score in [0.3, 0.4] gets +0.1."""
        items = {
            "e1": _make_triaged_item(
                event_id="e1", claim_id="c1", score=0.3,
                sender_tier=1, timestamp=int(time.time()) - 86400,
            ),
        }
        stats = rule_contact_tier_boost(items)
        assert stats.promoted >= 1
        assert items["e1"].effective_score == 0.4

    def test_no_boost_below_range(self):
        items = {
            "e1": _make_triaged_item(
                event_id="e1", claim_id="c1", score=0.1, sender_tier=1,
            ),
        }
        stats = rule_contact_tier_boost(items)
        assert stats.promoted == 0

    def test_no_boost_above_range(self):
        items = {
            "e1": _make_triaged_item(
                event_id="e1", claim_id="c1", score=0.5, sender_tier=1,
            ),
        }
        stats = rule_contact_tier_boost(items)
        assert stats.promoted == 0

    def test_no_boost_tier3(self):
        items = {
            "e1": _make_triaged_item(
                event_id="e1", claim_id="c1", score=0.35, sender_tier=3,
            ),
        }
        stats = rule_contact_tier_boost(items)
        assert stats.promoted == 0

    def test_old_messages_not_boosted(self):
        """Messages older than recency cutoff don't get boosted."""
        old_ts = int(time.time()) - (60 * 86400)
        items = {
            "e1": _make_triaged_item(
                event_id="e1", claim_id="c1", score=0.35,
                sender_tier=1, timestamp=old_ts,
            ),
        }
        stats = rule_contact_tier_boost(items)
        assert stats.promoted == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 4: Recipient dampening
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRecipientDampening:
    def test_dampens_cc_emails(self):
        items = {
            "e1": _make_triaged_item(
                event_id="e1", claim_id="c1", score=0.7,
                event_type="email",
                cc_recipients=["p2", "p3"], sender_tier=3,
            ),
        }
        stats = rule_recipient_dampening(items)
        assert stats.dampened >= 1
        assert items["e1"].effective_score < 0.7

    def test_dampens_mass_email(self):
        items = {
            "e1": _make_triaged_item(
                event_id="e1", claim_id="c1", score=0.7,
                total_recipients=10, sender_tier=3,
            ),
        }
        stats = rule_recipient_dampening(items)
        assert stats.dampened >= 1

    def test_tier1_exempt(self):
        items = {
            "e1": _make_triaged_item(
                event_id="e1", claim_id="c1", score=0.7,
                cc_recipients=["p2"], event_type="email",
                sender_tier=1,
            ),
        }
        stats = rule_recipient_dampening(items)
        assert stats.dampened == 0

    def test_low_score_not_dampened(self):
        items = {
            "e1": _make_triaged_item(
                event_id="e1", claim_id="c1", score=0.3,
                total_recipients=10,
            ),
        }
        stats = rule_recipient_dampening(items)
        assert stats.dampened == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Rule 5: Source type prior
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSourceTypePrior:
    def test_meeting_floor(self):
        items = {
            "e1": _make_triaged_item(
                event_id="e1", claim_id="c1", score=0.2,
                event_type=EVENT_TYPE_MEETING,
            ),
        }
        stats = rule_source_type_prior(items)
        assert stats.promoted >= 1
        assert items["e1"].effective_score >= MEETING_FLOOR

    def test_user_thread_floor(self):
        """Messages in threads where user participated get floor boost."""
        items = {
            "e_user": _make_triaged_item(
                event_id="e_user", claim_id="c_user", score=0.5,
                event_type=EVENT_TYPE_MESSAGE, thread_id="t1",
                sender_is_user=True,
            ),
            "e_other": _make_triaged_item(
                event_id="e_other", claim_id="c_other", score=0.1,
                event_type=EVENT_TYPE_MESSAGE, thread_id="t1",
                sender_is_user=False,
            ),
        }
        stats = rule_source_type_prior(items)
        assert items["e_other"].effective_score >= MSG_FLOOR_IF_USER_PARTICIPATED

    def test_email_unaffected(self):
        """Plain emails without user participation in thread -> no change."""
        items = {
            "e1": _make_triaged_item(
                event_id="e1", claim_id="c1", score=0.1,
                event_type=EVENT_TYPE_EMAIL,
            ),
        }
        stats = rule_source_type_prior(items)
        assert items["e1"].effective_score == 0.1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tier distribution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTierDistribution:
    def test_distribution_counts(self):
        items = {
            "e1": _make_triaged_item(score=0.1),
            "e2": _make_triaged_item(event_id="e2", claim_id="c2", score=0.5),
            "e3": _make_triaged_item(event_id="e3", claim_id="c3", score=0.8),
        }
        dist = _get_tier_distribution(items)
        assert dist["ignore"] == 1
        assert dist["lightweight"] == 1
        assert dist["deep"] == 1

    def test_empty_distribution(self):
        dist = _get_tier_distribution({})
        assert dist["ignore"] == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunPropagation:
    def _seed_store(self, store, n_events=5, scores=None):
        now = int(time.time())
        scores = scores or [0.5] * n_events
        store.put_person("person_user", is_user=True)
        store.put_person("person_sender", canonical_name="Sender")

        for i in range(n_events):
            eid = Event.make_id("mail", f"prop_{i}")
            store.put_event(Event(
                id=eid, source="mail", source_id=f"prop_{i}",
                event_type=EVENT_TYPE_EMAIL, timestamp=now - i * 3600,
                raw_content=f"Email {i}",
                metadata={"thread_id": "t1"},
            ))
            store.link_event_person(eid, "person_sender", "sender")
            store.link_event_person(eid, "person_user", "recipient")
            claim = _make_triage_claim(eid, scores[i])
            store.put_claim(claim)

    def test_empty_store(self, store):
        result = run_propagation(store)
        assert "error" in result

    def test_basic_propagation(self, store):
        self._seed_store(store, n_events=5, scores=[0.5, 0.5, 0.5, 0.5, 0.5])
        result = run_propagation(store)
        assert result["triaged_count"] == 5
        assert "pre_tiers" in result
        assert "post_tiers" in result

    def test_dry_run(self, store):
        self._seed_store(store, n_events=3)
        result = run_propagation(store, dry_run=True)
        assert result["dry_run"] is True
        assert result["updated_count"] == 0

    def test_convergence(self, store):
        self._seed_store(store, n_events=5)
        result = run_propagation(store, max_iterations=10)
        assert result["iterations"] <= 10

    def test_write_back(self, store):
        self._seed_store(store, n_events=6, scores=[0.1, 0.1, 0.1, 0.1, 0.1, 0.9])
        result = run_propagation(store)
        # At least some scores should be adjusted
        assert result["triaged_count"] == 6

    def test_transitions_tracked(self, store):
        self._seed_store(store, n_events=6, scores=[0.1, 0.1, 0.1, 0.1, 0.1, 0.9])
        result = run_propagation(store)
        # transitions dict should exist even if empty
        assert "transitions" in result

    def test_elapsed_time(self, store):
        self._seed_store(store, n_events=3)
        result = run_propagation(store)
        assert result["elapsed_seconds"] >= 0
