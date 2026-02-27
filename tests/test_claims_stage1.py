"""Tests for alteris.claims_stage1: Deterministic claim extractors.

Tests cover:
  - _claim_id determinism
  - Communication frequency extraction
  - Communication channel extraction
  - Directionality extraction
  - Timing pattern extraction
  - Recency extraction
  - Thread activity extraction
  - extract_stage1_claims full pipeline
  - Idempotency (re-extraction produces zero new claims)
  - Empty/minimal data handling
  - User auto-detection
"""

import json
import time

import pytest

from alteris.claims_stage1 import (
    _claim_id,
    _extract_communication_channels,
    _extract_communication_frequency,
    _extract_directionality,
    _extract_recency,
    _extract_thread_activity,
    _extract_timing_patterns,
    _setup_user_events_table,
    extract_stage1_claims,
)
from alteris.models import Event, Modality
from alteris.store import LayeredGraphStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_populated_store():
    """Create a store with realistic event/person/link data for stage 1 testing."""
    store = LayeredGraphStore(db_path=":memory:")
    now = int(time.time())

    # Persons
    store.put_person("person_user", canonical_name="Alex", is_user=True, sources=["mail"])
    store.put_person("person_sam", canonical_name="Sam", sources=["mail", "whatsapp"])
    store.put_person("person_bob", canonical_name="Bob", sources=["mail"])

    # Events: emails between user and Sam
    for i in range(10):
        eid = Event.make_id("mail", f"email_{i}")
        from_me = (i % 3 == 0)
        store.put_event(Event(
            id=eid,
            source="mail",
            source_id=f"email_{i}",
            event_type="email",
            timestamp=now - (i * 3600),
            participants=("sam@example.com", "user@example.com"),
            raw_content=f"Email body {i}",
            metadata={
                "subject": "Thread Alpha",
                "thread_id": "thread_alpha",
                "is_from_me": from_me,
            },
        ))
        # Link sender/recipient
        if from_me:
            store.link_event_person(eid, "person_user", "sender")
            store.link_event_person(eid, "person_sam", "recipient")
        else:
            store.link_event_person(eid, "person_sam", "sender")
            store.link_event_person(eid, "person_user", "recipient")
        store.link_event_person(eid, "person_user", "self")

    # Events: WhatsApp messages between user and Sam
    for i in range(5):
        eid = Event.make_id("whatsapp", f"wa_{i}")
        store.put_event(Event(
            id=eid,
            source="whatsapp",
            source_id=f"wa_{i}",
            event_type="message",
            timestamp=now - (i * 7200),
            raw_content=f"WhatsApp message {i}",
        ))
        store.link_event_person(eid, "person_sam", "sender")
        store.link_event_person(eid, "person_user", "recipient")
        store.link_event_person(eid, "person_user", "self")

    # Events: emails between user and Bob (fewer)
    for i in range(3):
        eid = Event.make_id("mail", f"bob_email_{i}")
        store.put_event(Event(
            id=eid,
            source="mail",
            source_id=f"bob_email_{i}",
            event_type="email",
            timestamp=now - (i * 86400),
            raw_content=f"Email from Bob {i}",
            metadata={"subject": "Bob's thread", "thread_id": "thread_bob"},
        ))
        store.link_event_person(eid, "person_bob", "sender")
        store.link_event_person(eid, "person_user", "recipient")
        store.link_event_person(eid, "person_user", "self")

    # Pre-materialize user events temp table (required by all extractors)
    _setup_user_events_table(store, "person_user")
    return store


@pytest.fixture
def stage1_store():
    return _make_populated_store()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _claim_id
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestClaimId:
    def test_deterministic(self):
        id1 = _claim_id("frequency", "person_a", "freq_with:person_b")
        id2 = _claim_id("frequency", "person_a", "freq_with:person_b")
        assert id1 == id2

    def test_starts_with_claim_prefix(self):
        cid = _claim_id("test", "sub", "pred")
        assert cid.startswith("claim:")

    def test_different_inputs_different_ids(self):
        id1 = _claim_id("frequency", "person_a", "pred_a")
        id2 = _claim_id("frequency", "person_a", "pred_b")
        assert id1 != id2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Communication frequency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCommunicationFrequency:
    def test_produces_claims(self, stage1_store):
        claims = _extract_communication_frequency(stage1_store, "person_user")
        assert len(claims) > 0

    def test_claim_type(self, stage1_store):
        claims = _extract_communication_frequency(stage1_store, "person_user")
        for c in claims:
            assert c.claim_type == "communication_frequency"

    def test_claim_has_event_count(self, stage1_store):
        claims = _extract_communication_frequency(stage1_store, "person_user")
        for c in claims:
            data = json.loads(c.object)
            assert "event_count" in data
            assert data["event_count"] > 0

    def test_modality_is_observed(self, stage1_store):
        claims = _extract_communication_frequency(stage1_store, "person_user")
        for c in claims:
            assert c.modality == Modality.OBSERVED

    def test_confidence_scales_with_count(self, stage1_store):
        claims = _extract_communication_frequency(stage1_store, "person_user")
        # Sam should have higher confidence than Bob (more events)
        sam_claims = [c for c in claims if "person_sam" in c.predicate]
        bob_claims = [c for c in claims if "person_bob" in c.predicate]
        if sam_claims and bob_claims:
            assert sam_claims[0].confidence >= bob_claims[0].confidence

    def test_empty_store_no_claims(self, store):
        store.put_person("person_user", is_user=True)
        _setup_user_events_table(store, "person_user")
        claims = _extract_communication_frequency(store, "person_user")
        assert claims == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Communication channels
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCommunicationChannels:
    def test_produces_claims(self, stage1_store):
        claims = _extract_communication_channels(stage1_store, "person_user")
        assert len(claims) > 0

    def test_claim_type(self, stage1_store):
        claims = _extract_communication_channels(stage1_store, "person_user")
        for c in claims:
            assert c.claim_type == "communication_channel"

    def test_sam_has_multiple_channels(self, stage1_store):
        claims = _extract_communication_channels(stage1_store, "person_user")
        sam_claims = [c for c in claims if "person_sam" in c.predicate]
        assert len(sam_claims) > 0
        data = json.loads(sam_claims[0].object)
        assert data["channel_count"] >= 2
        assert "mail" in data["channels"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Directionality
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDirectionality:
    def test_produces_claims(self, stage1_store):
        claims = _extract_directionality(stage1_store, "person_user")
        assert len(claims) > 0

    def test_claim_type(self, stage1_store):
        claims = _extract_directionality(stage1_store, "person_user")
        for c in claims:
            assert c.claim_type == "directionality"

    def test_direction_data_fields(self, stage1_store):
        claims = _extract_directionality(stage1_store, "person_user")
        for c in claims:
            data = json.loads(c.object)
            assert "user_sent" in data
            assert "they_sent" in data
            assert "total" in data
            assert "user_initiated_ratio" in data
            assert 0.0 <= data["user_initiated_ratio"] <= 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Timing patterns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTimingPatterns:
    def test_produces_claims_for_frequent_contacts(self, stage1_store):
        claims = _extract_timing_patterns(stage1_store, "person_user")
        # Sam has 15 events, should produce a timing claim
        sam_claims = [c for c in claims if "person_sam" in c.predicate]
        assert len(sam_claims) > 0

    def test_skips_infrequent_contacts(self, stage1_store):
        """Contacts with < 3 events should not get timing claims."""
        claims = _extract_timing_patterns(stage1_store, "person_user")
        for c in claims:
            data = json.loads(c.object)
            assert data["event_count"] >= 3

    def test_timing_data_fields(self, stage1_store):
        claims = _extract_timing_patterns(stage1_store, "person_user")
        for c in claims:
            data = json.loads(c.object)
            assert "peak_hours" in data
            assert "peak_days" in data
            assert "business_ratio" in data
            assert isinstance(data["peak_hours"], list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Recency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRecency:
    def test_produces_claims(self, stage1_store):
        claims = _extract_recency(stage1_store, "person_user")
        assert len(claims) > 0

    def test_recency_data_fields(self, stage1_store):
        claims = _extract_recency(stage1_store, "person_user")
        for c in claims:
            data = json.loads(c.object)
            assert "last_contact_ts" in data
            assert "days_since_last" in data
            assert "relationship_span_days" in data
            assert data["last_contact_ts"] > 0

    def test_recent_contacts_higher_confidence(self, stage1_store):
        """Very recent contacts should have higher confidence than old ones."""
        claims = _extract_recency(stage1_store, "person_user")
        for c in claims:
            data = json.loads(c.object)
            if data["days_since_last"] < 1:
                assert c.confidence > 0.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thread activity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestThreadActivity:
    def test_produces_claims_for_active_threads(self, stage1_store):
        claims = _extract_thread_activity(stage1_store, "person_user")
        # thread_alpha has 10 messages, should produce a claim
        assert len(claims) > 0

    def test_thread_data_fields(self, stage1_store):
        claims = _extract_thread_activity(stage1_store, "person_user")
        for c in claims:
            data = json.loads(c.object)
            assert "subject" in data
            assert "message_count" in data
            assert "participants" in data
            assert data["message_count"] >= 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full extraction pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExtractStage1Claims:
    def test_auto_detects_user(self, stage1_store):
        result = extract_stage1_claims(stage1_store)
        assert "error" not in result
        assert result["total_claims"] > 0

    def test_explicit_user_id(self, stage1_store):
        result = extract_stage1_claims(stage1_store, user_person_id="person_user")
        assert "error" not in result

    def test_missing_user_returns_error(self, store):
        result = extract_stage1_claims(store)
        assert result == {"error": "User not found in persons table"}

    def test_result_structure(self, stage1_store):
        result = extract_stage1_claims(stage1_store)
        assert "total_claims" in result
        assert "by_type" in result
        assert "new_claims" in result
        assert "existing_claims" in result
        assert "duration_seconds" in result

    def test_by_type_has_all_extractors(self, stage1_store):
        result = extract_stage1_claims(stage1_store)
        by_type = result["by_type"]
        assert "communication_frequency" in by_type
        assert "communication_channel" in by_type
        assert "directionality" in by_type
        assert "timing_pattern" in by_type
        assert "recency" in by_type
        assert "thread_activity" in by_type

    def test_idempotent(self, stage1_store):
        result1 = extract_stage1_claims(stage1_store)
        assert result1["new_claims"] > 0
        result2 = extract_stage1_claims(stage1_store)
        assert result2["new_claims"] == 0
        assert result2["existing_claims"] == result1["total_claims"]

    def test_claims_stored_in_db(self, stage1_store):
        extract_stage1_claims(stage1_store)
        claims = stage1_store.get_active_claims(claim_type="communication_frequency")
        assert len(claims) > 0

    def test_no_events_produces_no_claims(self, store):
        store.put_person("person_user", is_user=True)
        result = extract_stage1_claims(store)
        assert result["total_claims"] == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Populate person profiles
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from alteris.claims_stage1 import populate_person_profiles


class TestPopulatePersonProfiles:
    def test_produces_profiles(self, stage1_store):
        # First extract claims so there's data to build profiles from
        extract_stage1_claims(stage1_store)
        result = populate_person_profiles(stage1_store)
        assert result["profiles_written"] > 0

    def test_result_structure(self, stage1_store):
        extract_stage1_claims(stage1_store)
        result = populate_person_profiles(stage1_store)
        assert "profiles_written" in result
        assert "tier_changes" in result
        assert isinstance(result["profiles_written"], int)
        assert isinstance(result["tier_changes"], int)

    def test_profiles_stored_in_db(self, stage1_store):
        extract_stage1_claims(stage1_store)
        populate_person_profiles(stage1_store)
        profiles = stage1_store.get_person_profiles()
        assert len(profiles) > 0

    def test_sam_has_correct_profile(self, stage1_store):
        extract_stage1_claims(stage1_store)
        populate_person_profiles(stage1_store)

        profile = stage1_store.get_person_profile("person_sam")
        assert profile is not None
        assert profile["message_count"] > 0
        assert profile["tier"] in (1, 2, 3, 4)
        assert isinstance(profile["channels"], list)
        assert len(profile["channels"]) > 0

    def test_tier_computation(self, stage1_store):
        extract_stage1_claims(stage1_store)
        populate_person_profiles(stage1_store)

        # Sam has 15 events (10 mail + 5 whatsapp), tier 3 (≥5)
        sam_person = stage1_store.get_person_profile("person_sam")
        assert sam_person is not None
        assert sam_person["tier"] == 3  # 15 msgs → tier 3

        # Bob has 3 events (3 mail), tier 4 (< 5)
        bob = stage1_store.get_person_profile("person_bob")
        assert bob is not None
        assert bob["tier"] == 4

    def test_idempotent(self, stage1_store):
        extract_stage1_claims(stage1_store)
        r1 = populate_person_profiles(stage1_store)
        r2 = populate_person_profiles(stage1_store)
        # Same data, no tier changes on second run
        assert r1["profiles_written"] == r2["profiles_written"]
        assert r2["tier_changes"] == 0

    def test_user_profile_marked(self, stage1_store):
        extract_stage1_claims(stage1_store)
        populate_person_profiles(stage1_store)

        user_profile = stage1_store.get_person_profile("person_user")
        if user_profile:
            assert user_profile["is_user"] == 1

    def test_no_claims_produces_no_profiles(self, store):
        store.put_person("person_user", is_user=True)
        result = populate_person_profiles(store)
        assert result["profiles_written"] == 0

    def test_channels_populated(self, stage1_store):
        extract_stage1_claims(stage1_store)
        populate_person_profiles(stage1_store)

        sam_person = stage1_store.get_person_profile("person_sam")
        assert sam_person is not None
        # Sam has events from mail and whatsapp
        assert "mail" in sam_person["channels"] or "whatsapp" in sam_person["channels"]
        assert sam_person["channel_count"] > 0

    def test_tier_change_detection(self, stage1_store):
        extract_stage1_claims(stage1_store)
        populate_person_profiles(stage1_store)

        # Manually downgrade Bob's tier and re-run
        stage1_store.conn.execute(
            "UPDATE person_profiles SET tier = 1, message_count = 100 WHERE person_id = 'person_bob'"
        )
        stage1_store.conn.commit()

        # Re-populate should detect the tier change (100→3 msgs, tier 1→4)
        result = populate_person_profiles(stage1_store)
        assert result["tier_changes"] > 0
