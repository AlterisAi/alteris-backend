"""Tests for loom.score: Stage 2 heuristic event scoring.

Tests cover:
  - Per-source scoring (mail, imessage, whatsapp, calendar, granola, slack)
  - Route thresholds (skip < 0.1, low_priority 0.1-0.4, full_triage >= 0.4)
  - Floor overrides (high_impact, is_from_me)
  - Claim ID determinism and structure
  - run_scoring end-to-end with in-memory store
  - Idempotent re-scoring
  - Edge cases (empty metadata, unknown source, score clamping)
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest import make_event
from loom.constants import (
    EVENT_TYPE_CALENDAR,
    EVENT_TYPE_EMAIL,
    EVENT_TYPE_IDENTITY,
    EVENT_TYPE_MEETING,
    EVENT_TYPE_MESSAGE,
    EVENT_TYPE_REACTION,
)
from loom.score import (
    DEFAULT_LENS,
    PersonEngagement,
    ScoreResult,
    _engagement_floor,
    _score_mail,
    compute_person_engagement,
    run_scoring,
    score_event,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mail scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMailScoring:
    def test_base_score(self):
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL, metadata={})
        r = score_event(ev)
        assert r.score == 0.3
        assert r.route == "low_priority"

    def test_replied_boost(self):
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"replied": True})
        r = score_event(ev)
        assert r.score == 0.7
        assert r.route == "full_triage"

    def test_flagged_boost(self):
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"flagged": True})
        r = score_event(ev)
        assert r.score == 0.6
        assert r.route == "full_triage"

    def test_automated_penalty(self):
        """Single noise signal (automated only) → corroboration floor 0.1."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"automated": True})
        r = score_event(ev)
        assert r.score == 0.1
        assert r.route == "low_priority"
        assert "after_corroboration_floor" in r.components

    def test_list_id_penalty(self):
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"list_id": True})
        r = score_event(ev)
        assert r.score == 0.1
        assert r.route == "low_priority"

    def test_junk_penalty(self):
        """Single noise signal (junk only) → corroboration floor 0.1."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"junk_level": 1})
        r = score_event(ev)
        assert r.score == 0.1
        assert r.route == "low_priority"
        assert "after_corroboration_floor" in r.components

    def test_high_impact_floor_override(self):
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"high_impact": True})
        r = score_event(ev)
        assert r.score == 0.4
        assert r.route == "full_triage"
        assert "after_high_impact_floor" in r.components

    def test_automated_high_impact_no_longer_overrides(self):
        """automated + high_impact: high_impact_floor no longer applies when
        automated=True. Apple Intelligence marks security alerts as high_impact
        because they contain urgent language, but these are ephemeral noise.
        0.3 - 0.3(automated) = 0.0 -> clamp 0.0 -> corroboration 0.1."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"automated": True, "high_impact": True})
        r = score_event(ev)
        assert r.score == 0.1
        assert r.route == "low_priority"
        assert r.components["after_automated"] == 0.0
        assert r.components["after_clamp"] == 0.0
        assert "after_high_impact_floor" not in r.components

    def test_high_impact_with_all_penalties_prefiltered(self):
        """automated + list_id + model_category=2 + high_impact.
        Prefilter catches automated+list_id and returns machine_generated skip."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"automated": True, "list_id": True,
                                  "model_category": 2, "high_impact": True})
        r = score_event(ev)
        assert r.score == 0.0
        assert r.route == "skip"
        assert r.components.get("reason") == "machine_generated"

    def test_primary_category_boost(self):
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"model_category": 0})
        r = score_event(ev)
        assert r.score == 0.45
        assert r.route == "full_triage"

    def test_urgent_boost(self):
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"urgent": True})
        r = score_event(ev)
        assert r.score == 0.45
        assert r.route == "full_triage"

    def test_promo_category_penalty(self):
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"model_category": 3})
        r = score_event(ev)
        assert r.score == 0.1
        assert r.route == "low_priority"

    def test_combined_replied_and_automated(self):
        """replied(+0.4) + automated(-0.3) = 0.3 + 0.4 - 0.3 = 0.4."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"replied": True, "automated": True})
        r = score_event(ev)
        assert r.score == 0.4
        assert r.route == "full_triage"

    def test_primary_automated_reduced_penalty(self):
        """primary(cat=0) + automated: penalty is -0.15 not -0.3.
        The Villi calendar invite case: 0.3 + 0.15(primary) - 0.15(auto) = 0.30."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"model_category": 0, "automated": True})
        r = score_event(ev)
        assert r.score == 0.3
        assert r.route == "low_priority"
        assert r.components["after_primary_category"] == 0.45
        assert r.components["after_automated"] == 0.3

    def test_primary_automated_urgent(self):
        """primary + automated + urgent: 0.3 + 0.15 + 0.15 - 0.15 = 0.45.
        The Bright Horizons teacher case."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"model_category": 0, "automated": True,
                                  "urgent": True})
        r = score_event(ev)
        assert r.score == 0.45
        assert r.route == "full_triage"

    def test_nonprimary_automated_full_penalty(self):
        """Non-primary automated email: full -0.3 penalty, but corroboration
        floor catches single noise signal (automated only, cat=1 is not noise)."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"model_category": 1, "automated": True})
        r = score_event(ev)
        assert r.score == 0.1
        assert r.route == "low_priority"
        assert "after_corroboration_floor" in r.components

    def test_noreply_penalty(self):
        """noreply sender: -0.1 penalty, counts as noise signal."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"sender_is_noreply": True})
        r = score_event(ev)
        assert r.score == 0.2
        assert r.route == "low_priority"
        assert r.components["noise_signals"] == 1
        assert "after_noreply" in r.components

    def test_noreply_plus_automated_skip(self):
        """automated + noreply: 2 noise signals → can skip.
        0.3 - 0.3(automated) - 0.1(noreply) = -0.1 → clamp 0.0."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"automated": True, "sender_is_noreply": True})
        r = score_event(ev)
        assert r.score == 0.0
        assert r.route == "skip"
        assert r.components["noise_signals"] == 2

    def test_cumulative_components_format(self):
        """Components track running score at each step, not deltas.
        Use a non-prefiltered combo: replied + flagged + primary."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"replied": True, "flagged": True,
                                  "model_category": 0})
        r = score_event(ev)
        c = r.components
        assert c["base"] == 0.3
        assert c["after_replied"] == 0.7
        assert c["after_flagged"] == 1.0
        assert "after_primary_category" in c
        assert c["final"] == 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# iMessage scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIMessageScoring:
    def test_base_score(self):
        ev = make_event(source="imessage", event_type=EVENT_TYPE_MESSAGE,
                        metadata={})
        r = score_event(ev)
        assert r.score == 0.3
        assert r.route == "low_priority"

    def test_reaction_skip(self):
        ev = make_event(source="imessage", event_type=EVENT_TYPE_REACTION,
                        metadata={"is_reaction": True})
        r = score_event(ev)
        assert r.score == 0.0
        assert r.route == "skip"

    def test_filtered_penalty(self):
        """is_filtered costs -0.1 (not -0.2), so filtered alone = 0.2 = low_priority."""
        ev = make_event(source="imessage", event_type=EVENT_TYPE_MESSAGE,
                        metadata={"is_filtered": True})
        r = score_event(ev)
        assert r.score == 0.2
        assert r.route == "low_priority"

    def test_auto_reply_penalty(self):
        ev = make_event(source="imessage", event_type=EVENT_TYPE_MESSAGE,
                        metadata={"is_auto_reply": True})
        r = score_event(ev)
        assert r.score == 0.15
        assert r.route == "low_priority"

    def test_filtered_and_quiet_stacking(self):
        """filtered(-0.1) + quiet(-0.1) = 0.3 - 0.2 = 0.1, 2 noise but 0.1 >= threshold."""
        ev = make_event(source="imessage", event_type=EVENT_TYPE_MESSAGE,
                        metadata={"is_filtered": True, "delivered_quietly": True})
        r = score_event(ev)
        assert r.score == 0.1
        assert r.route == "low_priority"
        assert r.components["noise_signals"] == 2

    def test_all_noise_skip(self):
        """filtered + quiet + auto_reply: 3 noise signals, 0.3 - 0.35 → 0.0, skip."""
        ev = make_event(source="imessage", event_type=EVENT_TYPE_MESSAGE,
                        metadata={"is_filtered": True, "delivered_quietly": True,
                                  "is_auto_reply": True})
        r = score_event(ev)
        assert r.score == 0.0
        assert r.route == "skip"
        assert r.components["noise_signals"] == 3

    def test_noise_signals_in_components(self):
        """iMessage noise counting provides audit trail in components."""
        ev = make_event(source="imessage", event_type=EVENT_TYPE_MESSAGE,
                        metadata={"is_filtered": True})
        r = score_event(ev)
        assert r.components["noise_signals"] == 1
        # Single noise, score 0.2 >= 0.1, no corroboration floor needed
        assert "after_corroboration_floor" not in r.components


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WhatsApp scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestWhatsAppScoring:
    def test_base_score(self):
        ev = make_event(source="whatsapp", event_type=EVENT_TYPE_MESSAGE,
                        metadata={})
        r = score_event(ev)
        assert r.score == 0.3
        assert r.route == "low_priority"

    def test_reaction_skip(self):
        ev = make_event(source="whatsapp", event_type=EVENT_TYPE_REACTION,
                        metadata={"is_reaction": True})
        r = score_event(ev)
        assert r.score == 0.0
        assert r.route == "skip"

    def test_shared_url_boost(self):
        ev = make_event(source="whatsapp", event_type=EVENT_TYPE_MESSAGE,
                        metadata={"shared_url": True})
        r = score_event(ev)
        assert r.score == 0.4
        assert r.route == "full_triage"

    def test_shared_document_boost(self):
        ev = make_event(source="whatsapp", event_type=EVENT_TYPE_MESSAGE,
                        metadata={"shared_document": True})
        r = score_event(ev)
        assert r.score == 0.4
        assert r.route == "full_triage"

    def test_large_group_penalty(self):
        """Group with >10 participants gets -0.15 penalty."""
        participants = tuple(f"+1555000{i:04d}" for i in range(15))
        ev = make_event(source="whatsapp", event_type=EVENT_TYPE_MESSAGE,
                        participants=participants,
                        metadata={"is_group": True})
        r = score_event(ev)
        assert r.score == 0.15
        assert r.route == "low_priority"

    def test_small_group_no_penalty(self):
        """Group with <=10 participants is not penalized."""
        participants = tuple(f"+1555000{i:04d}" for i in range(5))
        ev = make_event(source="whatsapp", event_type=EVENT_TYPE_MESSAGE,
                        participants=participants,
                        metadata={"is_group": True})
        r = score_event(ev)
        assert r.score == 0.3
        assert r.route == "low_priority"

    def test_one_on_one_no_penalty(self):
        """1:1 WhatsApp message, not a group, gets base score."""
        ev = make_event(source="whatsapp", event_type=EVENT_TYPE_MESSAGE,
                        participants=("+14155551234", "+15550100001"),
                        metadata={"is_group": False})
        r = score_event(ev)
        assert r.score == 0.3
        assert r.route == "low_priority"

    def test_large_group_with_shared_url(self):
        """Large group penalty (-0.15) + shared_url boost (+0.1) = 0.25."""
        participants = tuple(f"+1555000{i:04d}" for i in range(20))
        ev = make_event(source="whatsapp", event_type=EVENT_TYPE_MESSAGE,
                        participants=participants,
                        metadata={"is_group": True, "shared_url": True})
        r = score_event(ev)
        assert r.score == 0.25
        assert r.route == "low_priority"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Calendar scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCalendarScoring:
    def test_base_score(self):
        """Calendar base is universal 0.3 — low_priority without engagement."""
        ev = make_event(source="calendar", event_type=EVENT_TYPE_CALENDAR,
                        metadata={})
        r = score_event(ev)
        assert r.score == 0.3
        assert r.route == "low_priority"

    def test_holiday_penalty(self):
        """Holiday alone: 0.3 - 0.3 = 0.0, single noise signal → corroboration floor 0.1."""
        ev = make_event(source="calendar", event_type=EVENT_TYPE_CALENDAR,
                        metadata={"calendar_type": "holiday"})
        r = score_event(ev)
        assert r.score == 0.1
        assert r.route == "low_priority"
        assert r.components["noise_signals"] == 1
        assert "after_corroboration_floor" in r.components

    def test_birthday_penalty(self):
        """Birthday same as holiday: corroboration rescue to 0.1."""
        ev = make_event(source="calendar", event_type=EVENT_TYPE_CALENDAR,
                        metadata={"calendar_type": "birthday"})
        r = score_event(ev)
        assert r.score == 0.1
        assert r.route == "low_priority"

    def test_acceptance_stripped(self):
        """Calendar acceptance status is stripped — no declined penalty."""
        ev = make_event(source="calendar", event_type=EVENT_TYPE_CALENDAR,
                        metadata={"user_acceptance": "declined"})
        r = score_event(ev)
        assert r.score == 0.3  # base only, no penalty

    def test_acceptance_boost_stripped(self):
        """Calendar acceptance status is stripped — no accepted boost."""
        ev = make_event(source="calendar", event_type=EVENT_TYPE_CALENDAR,
                        metadata={"user_acceptance": "accepted"})
        r = score_event(ev)
        assert r.score == 0.3  # base only, no boost

    def test_has_attendees_boost(self):
        """Attendees: 0.3 + 0.1 = 0.4, full_triage."""
        ev = make_event(source="calendar", event_type=EVENT_TYPE_CALENDAR,
                        metadata={"attendees": [{"name": "Sam", "email": "a@b.com"}]})
        r = score_event(ev)
        assert r.score == 0.4
        assert r.route == "full_triage"

    def test_attendees_only_boost(self):
        """Attendees boost only: 0.3 + 0.1(attendees) = 0.4."""
        ev = make_event(source="calendar", event_type=EVENT_TYPE_CALENDAR,
                        metadata={"user_acceptance": "accepted",
                                  "attendees": [{"name": "Sam"}]})
        r = score_event(ev)
        assert r.score == 0.4
        assert r.route == "full_triage"

    def test_holiday_only_noise(self):
        """Holiday alone: 0.3 - 0.3 = 0.0, single noise signal → floor at 0.1."""
        ev = make_event(source="calendar", event_type=EVENT_TYPE_CALENDAR,
                        metadata={"calendar_type": "holiday"})
        r = score_event(ev)
        assert r.score == 0.1  # corroboration floor
        assert r.components["noise_signals"] == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Granola scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGranolaScoring:
    def test_base_score(self):
        """Granola base is universal 0.3 — low_priority without engagement.
        Meetings with known contacts get rescued by person engagement floor."""
        ev = make_event(source="granola", event_type=EVENT_TYPE_MEETING,
                        metadata={})
        r = score_event(ev)
        assert r.score == 0.3
        assert r.route == "low_priority"

    def test_transcript_boost(self):
        """Transcript pushes to 0.4 → full_triage (solo brainstorm case)."""
        ev = make_event(source="granola", event_type=EVENT_TYPE_MEETING,
                        metadata={"has_transcript": True})
        r = score_event(ev)
        assert r.score == 0.4
        assert r.route == "full_triage"

    def test_no_transcript_low_priority(self):
        """Without transcript, Granola is 0.3 → low_priority.
        Person engagement is what rescues meetings with known contacts."""
        ev = make_event(source="granola", event_type=EVENT_TYPE_MEETING,
                        metadata={"has_transcript": False})
        r = score_event(ev)
        assert r.score == 0.3
        assert r.route == "low_priority"

    def test_engagement_rescues_granola(self):
        """Granola meeting with high-engagement co-founder → full_triage via floor + bonus."""
        ev = make_event(source="granola", event_type=EVENT_TYPE_MEETING,
                        metadata={})
        eng = PersonEngagement(thread_count=20, source_count=3,
                               from_me_ratio=0.4, last_seen_ts=1000)
        r = score_event(ev, person_context=eng)
        assert r.score == 0.5  # floor 0.45 + cross-source 0.05
        assert r.route == "full_triage"
        assert "after_engagement_floor" in r.components
        assert "after_cross_source_bonus" in r.components


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Slack scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSlackScoring:
    def test_base_score(self):
        """Slack base is universal 0.3 — low_priority without engagement.
        DMs with known contacts get rescued by person engagement floor."""
        ev = make_event(source="slack", event_type=EVENT_TYPE_MESSAGE,
                        metadata={})
        r = score_event(ev)
        assert r.score == 0.3
        assert r.route == "low_priority"

    def test_threaded_boost(self):
        """Threaded message: 0.3 + 0.1 = 0.4, full_triage."""
        ev = make_event(source="slack", event_type=EVENT_TYPE_MESSAGE,
                        metadata={"has_thread": True})
        r = score_event(ev)
        assert r.score == 0.4
        assert r.route == "full_triage"

    def test_high_reply_count(self):
        """Threaded + high reply: 0.3 + 0.1 + 0.1 = 0.5."""
        ev = make_event(source="slack", event_type=EVENT_TYPE_MESSAGE,
                        metadata={"has_thread": True, "reply_count": 5})
        r = score_event(ev)
        assert r.score == 0.5
        assert r.route == "full_triage"

    def test_low_reply_count_no_boost(self):
        """reply_count <= 2 doesn't trigger the boost."""
        ev = make_event(source="slack", event_type=EVENT_TYPE_MESSAGE,
                        metadata={"reply_count": 2})
        r = score_event(ev)
        assert r.score == 0.3

    def test_engagement_rescues_slack_dm(self):
        """Slack DM with co-founder (3+ sources) → full_triage via engagement + bonus."""
        ev = make_event(source="slack", event_type=EVENT_TYPE_MESSAGE,
                        metadata={})
        eng = PersonEngagement(thread_count=15, source_count=3,
                               from_me_ratio=0.5, last_seen_ts=1000)
        r = score_event(ev, person_context=eng)
        assert r.score == 0.5  # floor 0.45 + cross-source 0.05
        assert r.route == "full_triage"
        assert "after_engagement_floor" in r.components
        assert "after_cross_source_bonus" in r.components


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# General / routing / dispatch
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGeneralScoring:
    def test_contacts_always_skip(self):
        ev = make_event(source="contacts", event_type=EVENT_TYPE_IDENTITY,
                        metadata={})
        r = score_event(ev)
        assert r.score == 0.0
        assert r.route == "skip"
        assert r.components["reason"] == "identity_event"

    def test_reaction_always_skip(self):
        ev = make_event(source="imessage", event_type=EVENT_TYPE_REACTION,
                        metadata={})
        r = score_event(ev)
        assert r.score == 0.0
        assert r.route == "skip"
        assert r.components["reason"] == "reaction_event"

    def test_default_lens_is_chief_of_staff(self):
        """DEFAULT_LENS should be chief_of_staff."""
        assert DEFAULT_LENS == "chief_of_staff"

    def test_route_threshold_skip(self):
        """Score 0.09 routes to skip."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"automated": True, "junk_level": 1})
        r = score_event(ev)
        assert r.score < 0.1
        assert r.route == "skip"

    def test_route_threshold_low_priority(self):
        """Score 0.1 routes to low_priority."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"list_id": True})
        r = score_event(ev)
        assert r.score == 0.1
        assert r.route == "low_priority"

    def test_route_threshold_full_triage(self):
        """Score 0.4 routes to full_triage."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"high_impact": True})
        r = score_event(ev)
        assert r.score >= 0.4
        assert r.route == "full_triage"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# is_from_me floor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIsFromMeFloor:
    def test_from_me_prevents_penalty_below_base(self):
        """User-authored message with filter penalty stays >= base."""
        ev = make_event(source="imessage", event_type=EVENT_TYPE_MESSAGE,
                        metadata={"is_from_me": True, "is_filtered": True,
                                  "delivered_quietly": True})
        r = score_event(ev)
        assert r.score >= 0.3  # base for imessage
        assert "after_from_me_floor" in r.components

    def test_from_me_false_allows_penalty(self):
        """Non-user message with same penalties can drop below base."""
        ev = make_event(source="imessage", event_type=EVENT_TYPE_MESSAGE,
                        metadata={"is_from_me": False, "is_filtered": True,
                                  "delivered_quietly": True})
        r = score_event(ev)
        assert r.score == 0.1

    def test_from_me_no_effect_when_above_base(self):
        """is_from_me doesn't change score when it's already >= base."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"is_from_me": True, "replied": True})
        r = score_event(ev)
        assert r.score == 0.7
        assert "after_from_me_floor" not in r.components

    def test_from_me_whatsapp_large_group(self):
        """User message in large group: penalty would drop to 0.15, floor restores to 0.3."""
        participants = tuple(f"+1555000{i:04d}" for i in range(15))
        ev = make_event(source="whatsapp", event_type=EVENT_TYPE_MESSAGE,
                        participants=participants,
                        metadata={"is_from_me": True, "is_group": True})
        r = score_event(ev)
        assert r.score >= 0.3
        assert "after_from_me_floor" in r.components


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Edge cases
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEdgeCases:
    def test_empty_metadata(self):
        """Event with metadata={} gets base score for its source."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL, metadata={})
        r = score_event(ev)
        assert r.score == 0.3

    def test_unknown_source_default(self):
        """Unknown source routes through _score_default, gets 0.3."""
        ev = make_event(source="unknown_app", event_type="unknown_type",
                        metadata={})
        r = score_event(ev)
        assert r.score == 0.3
        assert r.route == "low_priority"

    def test_score_clamped_high(self):
        """Stacking all boosts doesn't exceed 1.0."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"replied": True, "flagged": True,
                                  "model_category": 0, "urgent": True})
        r = score_event(ev)
        assert r.score <= 1.0

    def test_score_clamped_low(self):
        """Stacking all penalties doesn't go below 0.0."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"automated": True, "list_id": True,
                                  "model_category": 2, "junk_level": 1})
        r = score_event(ev)
        assert r.score >= 0.0

    def test_components_has_final(self):
        """Every ScoreResult has 'final' in components."""
        for source in ["mail", "imessage", "whatsapp", "calendar", "granola", "slack"]:
            et = EVENT_TYPE_EMAIL if source == "mail" else EVENT_TYPE_MEETING if source == "granola" else EVENT_TYPE_CALENDAR if source == "calendar" else EVENT_TYPE_MESSAGE
            ev = make_event(source=source, event_type=et, metadata={})
            r = score_event(ev)
            assert "final" in r.components, f"Missing 'final' for source={source}"
            assert r.components["final"] == r.score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# End-to-end: run_scoring with in-memory store
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunScoring:
    def test_end_to_end(self, store):
        """Score multiple events and verify projections are written."""
        events = [
            make_event(source="mail", source_id="e2e_1",
                       event_type=EVENT_TYPE_EMAIL,
                       metadata={"replied": True}),
            make_event(source="imessage", source_id="e2e_2",
                       event_type=EVENT_TYPE_MESSAGE, metadata={}),
            make_event(source="contacts", source_id="e2e_3",
                       event_type=EVENT_TYPE_IDENTITY, metadata={}),
            make_event(source="granola", source_id="e2e_4",
                       event_type=EVENT_TYPE_MEETING,
                       metadata={"has_transcript": True}),
        ]
        for ev in events:
            store.put_event(ev)

        result = run_scoring(store)

        assert result["scored"] == 4
        assert result["projections_written"] == 4
        assert result["skip"] == 1      # contacts
        assert result["full_triage"] >= 2  # mail(replied), granola
        assert result["elapsed_seconds"] >= 0
        assert result["lens"] == "chief_of_staff"
        assert "mail" in result["by_source"]
        assert "contacts" in result["by_source"]

    def test_rescoring_overwrites_projections(self, store):
        """Running scoring twice overwrites projections (upsert, not insert)."""
        ev = make_event(source="mail", source_id="idem_1",
                        event_type=EVENT_TYPE_EMAIL, metadata={})
        store.put_event(ev)

        r1 = run_scoring(store)
        assert r1["projections_written"] == 1

        r2 = run_scoring(store)
        assert r2["projections_written"] == 1  # upsert, always writes

        # Only one projection per (event, lens)
        projs = store.get_projections(lens="chief_of_staff")
        assert len(projs) == 1

    def test_projection_structure(self, store):
        """Verify the written projection has correct fields."""
        ev = make_event(source="mail", source_id="struct_1",
                        event_type=EVENT_TYPE_EMAIL,
                        metadata={"replied": True, "model_category": 0})
        store.put_event(ev)
        run_scoring(store)

        proj = store.get_projection(ev.id, "chief_of_staff")
        assert proj is not None
        assert proj.event_id == ev.id
        assert proj.lens == "chief_of_staff"
        assert proj.score == 0.85  # replied(0.7) + primary(0.15)
        assert proj.route == "full_triage"
        assert "components" in proj.components
        assert proj.components["components"]["final"] == 0.85

    def test_empty_store(self, store):
        """Scoring an empty store returns zero counts."""
        result = run_scoring(store)
        assert result["scored"] == 0
        assert result["projections_written"] == 0

    def test_projection_includes_signals(self, store):
        """Written projection includes signals dict alongside components."""
        ev = make_event(source="mail", source_id="sig_1",
                        event_type=EVENT_TYPE_EMAIL,
                        metadata={"replied": True, "automated": False})
        store.put_event(ev)
        run_scoring(store)

        proj = store.get_projection(ev.id, "chief_of_staff")
        assert proj is not None
        assert "signals" in proj.components
        assert "replied" in proj.components["signals"]
        assert proj.components["signals"]["replied"] is True

    def test_lens_parameter(self, store):
        """Different lens produces separate projections."""
        ev = make_event(source="mail", source_id="lens_1",
                        event_type=EVENT_TYPE_EMAIL, metadata={})
        store.put_event(ev)

        run_scoring(store, lens="chief_of_staff")
        run_scoring(store, lens="financial_audit")

        p1 = store.get_projection(ev.id, "chief_of_staff")
        p2 = store.get_projection(ev.id, "financial_audit")
        assert p1 is not None
        assert p2 is not None
        assert p1.lens == "chief_of_staff"
        assert p2.lens == "financial_audit"

    def test_delete_projections_for_lens(self, store):
        """Deleting projections for one lens doesn't affect another."""
        ev = make_event(source="mail", source_id="del_1",
                        event_type=EVENT_TYPE_EMAIL, metadata={})
        store.put_event(ev)

        run_scoring(store, lens="chief_of_staff")
        run_scoring(store, lens="financial_audit")

        deleted = store.delete_projections("chief_of_staff")
        assert deleted == 1

        assert store.get_projection(ev.id, "chief_of_staff") is None
        assert store.get_projection(ev.id, "financial_audit") is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Corroboration model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCorroboration:
    """No single noise signal can skip; requires 2+ for confident filtering."""

    def test_single_automated_gets_floor(self):
        """automated-only: noise_count=1, score floored to 0.1."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"automated": True})
        r = score_event(ev)
        assert r.score == 0.1
        assert r.route == "low_priority"
        assert r.components["noise_signals"] == 1
        assert r.components["after_corroboration_floor"] == 0.1

    def test_single_list_id_no_floor_needed(self):
        """list_id-only: 0.3 - 0.2 = 0.1, already at threshold, no floor."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"list_id": True})
        r = score_event(ev)
        assert r.score == 0.1
        assert r.components["noise_signals"] == 1
        assert "after_corroboration_floor" not in r.components

    def test_single_junk_gets_floor(self):
        """junk-only: 0.3 - 0.3 = 0.0 → floor 0.1."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"junk_level": 1})
        r = score_event(ev)
        assert r.score == 0.1
        assert r.components["noise_signals"] == 1

    def test_single_promo_category_no_floor(self):
        """model_category=3: 0.3 - 0.2 = 0.1, at threshold, no floor needed."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"model_category": 3})
        r = score_event(ev)
        assert r.score == 0.1
        assert r.components["noise_signals"] == 1
        assert "after_corroboration_floor" not in r.components

    def test_double_noise_can_skip(self):
        """automated + junk: noise_count=2, no floor applied, can skip."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"automated": True, "junk_level": 1})
        r = score_event(ev)
        assert r.score == 0.0
        assert r.route == "skip"
        assert r.components["noise_signals"] == 2
        assert "after_corroboration_floor" not in r.components

    def test_triple_noise_prefiltered(self):
        """automated + list_id + promo category: prefilter catches this
        before noise counting even runs."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"automated": True, "list_id": True,
                                  "model_category": 2})
        r = score_event(ev)
        assert r.score == 0.0
        assert r.route == "skip"
        assert r.components.get("reason") == "machine_generated"

    def test_corroboration_automated_no_high_impact(self):
        """Single noise (automated) + high_impact: high_impact_floor no
        longer applies when automated=True. Corroboration floor 0.1."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"automated": True, "high_impact": True})
        r = score_event(ev)
        assert r.score == 0.1
        assert r.route == "low_priority"
        assert r.components["noise_signals"] == 1
        assert r.components["after_corroboration_floor"] == 0.1
        assert "after_high_impact_floor" not in r.components

    def test_high_impact_non_automated_still_works(self):
        """high_impact floor still applies to non-automated emails."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"high_impact": True, "model_category": 2})
        r = score_event(ev)
        # base 0.3 - 0.2(category) = 0.1, then high_impact floor → 0.4
        assert r.score == 0.4
        assert r.route == "full_triage"
        assert r.components["after_high_impact_floor"] == 0.4

    def test_no_noise_signals_zero_count(self):
        """Plain email: noise_count=0."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={})
        r = score_event(ev)
        assert r.components["noise_signals"] == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Signals dict
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSignals:
    """Each source scorer includes raw metadata signals in the result."""

    def test_mail_signals(self):
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"automated": True, "model_category": 0,
                                  "replied": True})
        r = score_event(ev)
        assert r.signals["automated"] is True
        assert r.signals["email_category"] == 0
        assert r.signals["replied"] is True
        assert "time_sensitive" in r.signals
        assert "flagged" in r.signals
        assert "list_id" in r.signals
        assert "junk_level" in r.signals
        assert "urgent" in r.signals
        assert "noreply" in r.signals

    def test_imessage_signals(self):
        ev = make_event(source="imessage", event_type=EVENT_TYPE_MESSAGE,
                        metadata={"is_filtered": True})
        r = score_event(ev)
        assert r.signals["is_filtered"] is True
        assert "delivered_quietly" in r.signals
        assert "is_auto_reply" in r.signals

    def test_whatsapp_signals(self):
        ev = make_event(source="whatsapp", event_type=EVENT_TYPE_MESSAGE,
                        metadata={"is_group": True, "shared_url": True})
        r = score_event(ev)
        assert r.signals["is_group"] is True
        assert r.signals["shared_url"] is True
        assert "shared_document" in r.signals
        assert "participant_count" in r.signals

    def test_calendar_signals(self):
        ev = make_event(source="calendar", event_type=EVENT_TYPE_CALENDAR,
                        metadata={"calendar_type": "holiday",
                                  "attendees": [{"name": "X"}]})
        r = score_event(ev)
        assert r.signals["calendar_type"] == "holiday"
        assert r.signals["attendee_count"] == 1

    def test_granola_signals(self):
        ev = make_event(source="granola", event_type=EVENT_TYPE_MEETING,
                        metadata={"has_transcript": True})
        r = score_event(ev)
        assert r.signals["has_transcript"] is True

    def test_slack_signals(self):
        ev = make_event(source="slack", event_type=EVENT_TYPE_MESSAGE,
                        metadata={"has_thread": True, "reply_count": 5})
        r = score_event(ev)
        assert r.signals["has_thread"] is True
        assert r.signals["reply_count"] == 5

    def test_identity_skip_has_empty_signals(self):
        ev = make_event(source="contacts", event_type=EVENT_TYPE_IDENTITY,
                        metadata={})
        r = score_event(ev)
        assert r.signals == {}

    def test_reaction_skip_has_empty_signals(self):
        ev = make_event(source="imessage", event_type=EVENT_TYPE_REACTION,
                        metadata={})
        r = score_event(ev)
        assert r.signals == {}

    def test_default_source_has_empty_signals(self):
        ev = make_event(source="unknown", event_type="unknown", metadata={})
        r = score_event(ev)
        assert r.signals == {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PersonEngagement and _engagement_floor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPersonEngagement:
    def test_engagement_floor_three_sources(self):
        eng = PersonEngagement(thread_count=10, source_count=3,
                               from_me_ratio=0.3, last_seen_ts=1000)
        assert _engagement_floor(eng) == 0.45

    def test_engagement_floor_two_sources(self):
        eng = PersonEngagement(thread_count=5, source_count=2,
                               from_me_ratio=0.5, last_seen_ts=1000)
        assert _engagement_floor(eng) == 0.40

    def test_engagement_floor_high_thread_count(self):
        """5+ threads from single source + user replied → 0.35."""
        eng = PersonEngagement(thread_count=5, source_count=1,
                               from_me_ratio=0.2, last_seen_ts=1000)
        assert _engagement_floor(eng) == 0.35

    def test_engagement_floor_high_threads_no_replies(self):
        """5+ threads but from_me_ratio=0 → no floor (not bidirectional)."""
        eng = PersonEngagement(thread_count=10, source_count=1,
                               from_me_ratio=0.0, last_seen_ts=1000)
        assert _engagement_floor(eng) == 0.0

    def test_engagement_floor_low_engagement(self):
        """Few threads, single source → no floor."""
        eng = PersonEngagement(thread_count=2, source_count=1,
                               from_me_ratio=0.5, last_seen_ts=1000)
        assert _engagement_floor(eng) == 0.0

    def test_engagement_floor_empty(self):
        eng = PersonEngagement(thread_count=0, source_count=0,
                               from_me_ratio=0.0, last_seen_ts=0)
        assert _engagement_floor(eng) == 0.0

    def test_engagement_floor_priority_three_sources_over_threads(self):
        """3 sources takes priority over thread_count check."""
        eng = PersonEngagement(thread_count=100, source_count=3,
                               from_me_ratio=0.5, last_seen_ts=1000)
        assert _engagement_floor(eng) == 0.45

    def test_compute_engagement_from_store(self, store):
        """Compute engagement from events linked to a person in the store."""
        now = int(__import__("time").time())
        events = [
            make_event(source="mail", source_id="eng_1",
                       event_type=EVENT_TYPE_EMAIL, timestamp=now - 100,
                       metadata={"is_from_me": True}),
            make_event(source="imessage", source_id="eng_2",
                       event_type=EVENT_TYPE_MESSAGE, timestamp=now - 50,
                       metadata={"is_from_me": False}),
            make_event(source="whatsapp", source_id="eng_3",
                       event_type=EVENT_TYPE_MESSAGE, timestamp=now,
                       metadata={"is_from_me": True}),
        ]
        for ev in events:
            store.put_event(ev)
        store.put_person("person_eng_test", canonical_name="Eng Test")
        for ev in events:
            store.link_event_person(ev.id, "person_eng_test", "sender")

        eng = compute_person_engagement(store, "person_eng_test")
        assert eng.thread_count == 3
        assert eng.source_count == 3
        assert eng.from_me_ratio == pytest.approx(2 / 3, abs=0.01)
        assert eng.last_seen_ts == now

    def test_compute_engagement_excludes_identity(self, store):
        """Identity events don't count toward engagement."""
        ev_mail = make_event(source="mail", source_id="eng_4",
                             event_type=EVENT_TYPE_EMAIL, metadata={})
        ev_contact = make_event(source="contacts", source_id="eng_5",
                                event_type=EVENT_TYPE_IDENTITY, metadata={})
        store.put_event(ev_mail)
        store.put_event(ev_contact)
        store.put_person("person_eng_id", canonical_name="Eng Id")
        store.link_event_person(ev_mail.id, "person_eng_id", "sender")
        store.link_event_person(ev_contact.id, "person_eng_id", "identity")

        eng = compute_person_engagement(store, "person_eng_id")
        assert eng.thread_count == 1
        assert eng.source_count == 1

    def test_compute_engagement_empty(self, store):
        """Person with no events returns zeroed engagement."""
        store.put_person("person_empty", canonical_name="Empty")
        eng = compute_person_engagement(store, "person_empty")
        assert eng.thread_count == 0
        assert eng.source_count == 0
        assert eng.from_me_ratio == 0.0
        assert eng.last_seen_ts == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Engagement floor in dispatcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEngagementFloorDispatcher:
    def test_engagement_raises_low_to_full_triage(self):
        """Event scoring 0.3 (low_priority) with 3-source contact → 0.50 (floor 0.45 + bonus 0.05)."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL, metadata={})
        eng = PersonEngagement(thread_count=10, source_count=3,
                               from_me_ratio=0.3, last_seen_ts=1000)
        r = score_event(ev, person_context=eng)
        assert r.score == 0.5
        assert r.route == "full_triage"
        assert r.components["after_engagement_floor"] == 0.45
        assert r.components["after_cross_source_bonus"] == 0.5

    def test_engagement_raises_automated_to_full_triage(self):
        """Automated-only mail (0.1 after corroboration) + 2-source contact → 0.45 (floor + bonus)."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"automated": True})
        eng = PersonEngagement(thread_count=8, source_count=2,
                               from_me_ratio=0.4, last_seen_ts=1000)
        r = score_event(ev, person_context=eng)
        assert r.score == 0.45
        assert r.route == "full_triage"
        assert r.components["after_engagement_floor"] == 0.4
        assert r.components["after_cross_source_bonus"] == 0.45

    def test_engagement_no_floor_but_cross_source_bonus(self):
        """Score already above floor still gets cross-source bonus."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL,
                        metadata={"replied": True})  # score=0.7
        eng = PersonEngagement(thread_count=10, source_count=3,
                               from_me_ratio=0.3, last_seen_ts=1000)
        r = score_event(ev, person_context=eng)
        assert r.score == 0.75
        assert "after_engagement_floor" not in r.components
        assert r.components["after_cross_source_bonus"] == 0.75

    def test_engagement_none_means_no_floor(self):
        """Without engagement data, no floor is applied."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL, metadata={})
        r = score_event(ev, person_context=None)
        assert r.score == 0.3
        assert "after_engagement_floor" not in r.components
        assert "after_cross_source_bonus" not in r.components

    def test_engagement_low_engagement_no_floor(self):
        """Person with low engagement (1 source, few threads) → no floor, no bonus."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL, metadata={})
        eng = PersonEngagement(thread_count=2, source_count=1,
                               from_me_ratio=0.5, last_seen_ts=1000)
        r = score_event(ev, person_context=eng)
        assert r.score == 0.3
        assert "after_engagement_floor" not in r.components
        assert "after_cross_source_bonus" not in r.components

    def test_single_source_high_threads_no_bonus(self):
        """1-source person with 5+ threads gets floor (0.35) but no cross-source bonus."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL, metadata={})
        eng = PersonEngagement(thread_count=10, source_count=1,
                               from_me_ratio=0.5, last_seen_ts=1000)
        r = score_event(ev, person_context=eng)
        assert r.score == 0.35
        assert "after_engagement_floor" in r.components
        assert "after_cross_source_bonus" not in r.components

    def test_engagement_with_from_me_floor_stacking(self):
        """Engagement floor + cross-source bonus, then from_me floor is a no-op."""
        participants = tuple(f"+1555000{i:04d}" for i in range(15))
        ev = make_event(source="whatsapp", event_type=EVENT_TYPE_MESSAGE,
                        participants=participants,
                        metadata={"is_from_me": True, "is_group": True})
        # Base 0.3, large_group -0.15 = 0.15, then engagement floor 0.40 + bonus 0.05 = 0.45
        eng = PersonEngagement(thread_count=8, source_count=2,
                               from_me_ratio=0.5, last_seen_ts=1000)
        r = score_event(ev, person_context=eng)
        assert r.score == 0.45
        assert "after_engagement_floor" in r.components
        assert r.components["after_cross_source_bonus"] == 0.45
        # from_me floor shouldn't trigger since 0.45 > base(0.3)
        assert "after_from_me_floor" not in r.components

    def test_cross_source_bonus_differentiates_granola(self):
        """Sam's Granola meeting (2-source) scores higher than solo meeting."""
        ev_solo = make_event(source="granola", event_type=EVENT_TYPE_MEETING,
                             metadata={"has_transcript": True})
        ev_sam = make_event(source="granola", source_id="g_sam",
                              event_type=EVENT_TYPE_MEETING,
                              metadata={"has_transcript": True})
        eng = PersonEngagement(thread_count=11, source_count=2,
                               from_me_ratio=0.36, last_seen_ts=1000)
        r_solo = score_event(ev_solo)
        r_sam = score_event(ev_sam, person_context=eng)
        assert r_solo.score == 0.4      # base + transcript
        assert r_sam.score == 0.45    # base + transcript + cross-source bonus
        assert r_sam.score > r_solo.score


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Final key ordering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFinalKeyOrdering:
    def test_final_always_last_key_basic(self):
        """Components dict has 'final' as the last key."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL, metadata={})
        r = score_event(ev)
        keys = list(r.components.keys())
        assert keys[-1] == "final"

    def test_final_last_after_engagement_floor(self):
        """When engagement floor modifies score, final is still last."""
        ev = make_event(source="mail", event_type=EVENT_TYPE_EMAIL, metadata={})
        eng = PersonEngagement(thread_count=10, source_count=3,
                               from_me_ratio=0.3, last_seen_ts=1000)
        r = score_event(ev, person_context=eng)
        keys = list(r.components.keys())
        assert keys[-1] == "final"

    def test_final_last_after_from_me_floor(self):
        """When from_me floor modifies score, final is still last."""
        participants = tuple(f"+1555000{i:04d}" for i in range(15))
        ev = make_event(source="whatsapp", event_type=EVENT_TYPE_MESSAGE,
                        participants=participants,
                        metadata={"is_from_me": True, "is_group": True})
        r = score_event(ev)
        keys = list(r.components.keys())
        assert keys[-1] == "final"

    def test_final_last_all_sources(self):
        """All sources produce 'final' as last key."""
        sources = [
            ("mail", EVENT_TYPE_EMAIL),
            ("imessage", EVENT_TYPE_MESSAGE),
            ("whatsapp", EVENT_TYPE_MESSAGE),
            ("calendar", EVENT_TYPE_CALENDAR),
            ("granola", EVENT_TYPE_MEETING),
            ("slack", EVENT_TYPE_MESSAGE),
        ]
        for source, et in sources:
            ev = make_event(source=source, event_type=et, metadata={})
            r = score_event(ev)
            keys = list(r.components.keys())
            assert keys[-1] == "final", f"source={source}: final not last, got {keys}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# End-to-end: engagement in run_scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunScoringWithEngagement:
    def test_engagement_applied_via_run_scoring(self, store):
        """run_scoring pre-computes engagement and applies floors."""
        import time as _time
        now = int(_time.time())

        # Create a person with events across 3 sources
        store.put_person("person_multi", canonical_name="Multi Source")

        ev_mail = make_event(source="mail", source_id="rs_1",
                             event_type=EVENT_TYPE_EMAIL, timestamp=now - 100,
                             metadata={"is_from_me": True})
        ev_msg = make_event(source="imessage", source_id="rs_2",
                            event_type=EVENT_TYPE_MESSAGE, timestamp=now - 50,
                            metadata={})
        ev_wa = make_event(source="whatsapp", source_id="rs_3",
                           event_type=EVENT_TYPE_MESSAGE, timestamp=now,
                           metadata={})
        # A mail event linked to this multi-source person
        ev_target = make_event(source="mail", source_id="rs_target",
                               event_type=EVENT_TYPE_EMAIL, timestamp=now,
                               metadata={})  # base 0.3 → should be raised to 0.45

        for ev in [ev_mail, ev_msg, ev_wa, ev_target]:
            store.put_event(ev)

        store.link_event_person(ev_mail.id, "person_multi", "sender")
        store.link_event_person(ev_msg.id, "person_multi", "sender")
        store.link_event_person(ev_wa.id, "person_multi", "sender")
        store.link_event_person(ev_target.id, "person_multi", "sender")

        result = run_scoring(store)
        assert result["scored"] == 4

        # Check the target event's projection
        proj = store.get_projection(ev_target.id, "chief_of_staff")
        assert proj is not None
        # 3 sources for person_multi → floor 0.45 + cross-source bonus 0.05 = 0.50
        assert proj.score == 0.5
        assert proj.route == "full_triage"
