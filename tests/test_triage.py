"""Tests for alteris.triage: LLM triage (Stage 4).

Tests cover:
  - TriageResult Pydantic validation
  - Score floor rules
  - Claim construction and ID determinism
  - LLM response parsing (valid JSON, array, single, malformed, regex fallback)
  - Event selection (triageable events, resume mode)
  - Sender cache building
  - Temporal bucket assignment
  - Full run_triage pipeline with MockLLMClient
  - Downstream queries (get_triage_result, deep extraction candidates, sensitive events)
"""

import json
import time

import pytest

from alteris.constants import (
    EVENT_TYPE_CALENDAR,
    EVENT_TYPE_EMAIL,
    EVENT_TYPE_IDENTITY,
    EVENT_TYPE_MEETING,
    EVENT_TYPE_MESSAGE,
    SCORE_FLOOR_CALENDAR,
    SCORE_FLOOR_COMMITMENT,
    SCORE_FLOOR_MEETING,
    SCORE_FLOOR_TIER1_SENDER,
    SECONDS_PER_DAY,
)
from alteris.llm.mock import MOCK_TRIAGE_RESPONSE, MockLLMClient
from alteris.models import Claim, Event, ExtractionProvenance, Modality
from alteris.privacy import SensitivityLevel
from alteris.store import LayeredGraphStore
from alteris.triage import (
    PROMPT_VERSION,
    TriageResult,
    TriageStrategy,
    apply_score_floor,
    build_sender_cache,
    get_deep_extraction_candidates,
    get_sensitive_events,
    get_triage_result,
    get_triageable_events,
    parse_msg_batch_response,
    _per_event_claim_from_triage_result,
    run_triage,
    select_strategy,
    triage_claim_id,
    _assign_temporal_bucket,
    _failed_claim,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TriageResult validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTriageResult:
    def test_valid_result(self):
        r = TriageResult(
            score=0.75, reason="Test reason", domain="work",
            topics=["meeting"], entities=["Acme"], pii=[],
            sensitivity=[], commitment_type="deadline",
        )
        assert r.score == 0.8  # Rounded to 0.1 increments

    def test_score_clamped_high(self):
        r = TriageResult(score=1.5)
        assert r.score <= 1.0

    def test_score_clamped_low(self):
        r = TriageResult(score=-0.5)
        assert r.score >= 0.0

    def test_score_rounded(self):
        r = TriageResult(score=0.73)
        assert r.score == 0.7

    def test_invalid_domain_cleared(self):
        r = TriageResult(domain="nonexistent_domain")
        assert r.domain == ""

    def test_valid_domains(self):
        for d in ("work", "personal", "family", "financial", "health", "legal"):
            r = TriageResult(domain=d)
            assert r.domain == d

    def test_topics_truncated(self):
        r = TriageResult(topics=["a", "b", "c", "d", "e", "f", "g"])
        assert len(r.topics) <= 5

    def test_topics_lowercased(self):
        r = TriageResult(topics=["MEETING", "Project"])
        assert r.topics == ["meeting", "project"]

    def test_entities_truncated(self):
        r = TriageResult(entities=[f"entity_{i}" for i in range(15)])
        assert len(r.entities) <= 10

    def test_pii_validates(self):
        r = TriageResult(pii=["financial", "invalid", "medical"])
        assert r.pii == ["financial", "medical"]

    def test_pii_invalid_dropped(self):
        r = TriageResult(pii=["not_a_real_pii_type"])
        assert r.pii == []

    def test_sensitivity_validates(self):
        r = TriageResult(sensitivity=["health_discussion", "invalid_flag"])
        assert r.sensitivity == ["health_discussion"]

    def test_commitment_type_validates(self):
        r = TriageResult(commitment_type="deadline")
        assert r.commitment_type == "deadline"

    def test_commitment_type_null_cleared(self):
        r = TriageResult(commitment_type="null")
        assert r.commitment_type is None

    def test_commitment_type_none_preserved(self):
        r = TriageResult(commitment_type=None)
        assert r.commitment_type is None

    def test_commitment_type_invalid_cleared(self):
        r = TriageResult(commitment_type="not_a_type")
        assert r.commitment_type is None

    def test_reason_truncated(self):
        r = TriageResult(reason="x" * 500)
        assert len(r.reason) <= 200

    def test_model_validate_from_dict(self):
        data = {
            "score": 0.75,
            "reason": "Contains an action item",
            "domain": "work",
            "topics": ["project management"],
            "entities": ["Acme Corp"],
            "pii": [],
            "sensitivity": ["sensitive"],
            "commitment_type": "deadline",
        }
        r = TriageResult.model_validate(data)
        assert r.score == 0.8  # 0.75 rounded
        assert r.domain == "work"

    def test_sensitivity_string_coerced_to_list(self):
        """Pydantic before-validator coerces string to list, filtering invalid values."""
        r = TriageResult(sensitivity="not_a_list")
        assert r.sensitivity == []
        r2 = TriageResult(sensitivity="health_discussion")
        assert r2.sensitivity == ["health_discussion"]

    def test_topics_string_coerced_to_list(self):
        """Pydantic before-validator coerces string to list."""
        r = TriageResult(topics="some_topic")
        assert r.topics == ["some_topic"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Score floor rules
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestScoreFloor:
    def test_meeting_floor(self):
        score = apply_score_floor(0.1, EVENT_TYPE_MEETING, 3, None)
        assert score >= SCORE_FLOOR_MEETING

    def test_calendar_floor(self):
        score = apply_score_floor(0.1, EVENT_TYPE_CALENDAR, 3, None)
        assert score >= SCORE_FLOOR_CALENDAR

    def test_tier1_sender_floor(self):
        score = apply_score_floor(0.1, EVENT_TYPE_EMAIL, 1, None)
        assert score >= SCORE_FLOOR_TIER1_SENDER

    def test_commitment_floor(self):
        score = apply_score_floor(0.1, EVENT_TYPE_EMAIL, 3, "deadline")
        assert score >= SCORE_FLOOR_COMMITMENT

    def test_no_floor_applied(self):
        score = apply_score_floor(0.1, EVENT_TYPE_EMAIL, 3, None)
        assert score == 0.1

    def test_high_score_not_reduced(self):
        score = apply_score_floor(0.9, EVENT_TYPE_EMAIL, 3, None)
        assert score == 0.9

    def test_multiple_floors_highest_wins(self):
        score = apply_score_floor(0.1, EVENT_TYPE_MEETING, 1, "deadline")
        assert score >= SCORE_FLOOR_COMMITMENT

    def test_floor_rounds_to_one_decimal(self):
        score = apply_score_floor(0.33, EVENT_TYPE_EMAIL, 3, None)
        assert score == round(score, 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Claim construction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestClaimConstruction:
    def test_triage_claim_id_deterministic(self):
        id1 = triage_claim_id("event_001")
        id2 = triage_claim_id("event_001")
        assert id1 == id2

    def test_triage_claim_id_prefix(self):
        cid = triage_claim_id("event_001")
        assert cid.startswith("claim:")

    def test__per_event_claim_from_triage_result(self):
        result = TriageResult(
            score=0.7, reason="Action item",
            domain="work", topics=["project"],
            commitment_type="deadline",
        )
        claim = _per_event_claim_from_triage_result("event_001", result, "qwen3:30b-a3b")
        assert claim.claim_type == "triage"
        assert claim.subject == "event_001"
        assert claim.predicate == "triage_result"
        assert claim.confidence == 0.7
        data = json.loads(claim.object)
        assert data["score"] == 0.7
        assert data["commitment_type"] == "deadline"

    def test__per_event_claim_from_triage_result_provenance(self):
        result = TriageResult(score=0.5)
        claim = _per_event_claim_from_triage_result("event_001", result, "test-model")
        assert claim.provenance.model_id == "test-model"
        assert claim.provenance.prompt_version == PROMPT_VERSION

    def test__per_event_claim_from_triage_result_with_pii(self):
        result = TriageResult(
            score=0.5, pii=["financial"],
        )
        claim = _per_event_claim_from_triage_result("event_001", result, "model")
        assert claim.sensitivity == SensitivityLevel.CRITICAL

    def test_failed_claim(self):
        claim = _failed_claim("event_001", "model")
        assert claim.confidence == 0.0
        data = json.loads(claim.object)
        assert data["reason"] == "PARSE_FAILED"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM response parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestParsing:
    def test_parse_valid_json_array(self):
        raw = json.dumps([
            {"id": "e1", "score": 0.7, "reason": "test", "domain": "work",
             "topics": [], "entities": [], "pii": [], "sensitivity": [],
             "commitment_type": None},
        ])
        results = parse_msg_batch_response(raw, ["e1"])
        assert results["e1"] is not None
        assert results["e1"].score == 0.7

    def test_parse_single_object(self):
        raw = json.dumps({
            "id": "e1", "score": 0.5, "reason": "low priority",
            "domain": "personal",
        })
        results = parse_msg_batch_response(raw, ["e1"])
        assert results["e1"] is not None
        assert results["e1"].score == 0.5

    def test_parse_batch_matched_by_id(self):
        raw = json.dumps([
            {"id": "e1", "score": 0.3},
            {"id": "e2", "score": 0.7},
        ])
        results = parse_msg_batch_response(raw, ["e1", "e2"])
        assert results["e1"].score == 0.3
        assert results["e2"].score == 0.7

    def test_parse_batch_matched_by_position(self):
        raw = json.dumps([
            {"score": 0.3, "reason": "low"},
            {"score": 0.9, "reason": "high"},
        ])
        results = parse_msg_batch_response(raw, ["e1", "e2"])
        assert results["e1"].score == 0.3
        assert results["e2"].score == 0.9

    def test_parse_empty_response(self):
        results = parse_msg_batch_response("", ["e1", "e2"])
        assert results["e1"] is None
        assert results["e2"] is None

    def test_parse_with_thinking_tags(self):
        raw = '<think>some thoughts</think>[{"id": "e1", "score": 0.5}]'
        results = parse_msg_batch_response(raw, ["e1"])
        assert results["e1"] is not None

    def test_parse_with_markdown_fences(self):
        raw = '```json\n[{"id": "e1", "score": 0.8}]\n```'
        results = parse_msg_batch_response(raw, ["e1"])
        assert results["e1"] is not None

    def test_parse_regex_fallback(self):
        """Malformed JSON that still contains extractable objects."""
        raw = 'Some text {"id": "e1", "score": 0.6} more text'
        results = parse_msg_batch_response(raw, ["e1"])
        assert results["e1"] is not None
        assert results["e1"].score == 0.6

    def test_parse_completely_invalid(self):
        raw = "This is not JSON at all"
        results = parse_msg_batch_response(raw, ["e1"])
        assert results["e1"] is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Temporal bucket
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTemporalBucket:
    def test_hot_bucket(self):
        now = int(time.time())
        assert _assign_temporal_bucket(now - 3600, now) == 1

    def test_recent_bucket(self):
        now = int(time.time())
        assert _assign_temporal_bucket(now - 15 * SECONDS_PER_DAY, now) == 2

    def test_warm_bucket(self):
        now = int(time.time())
        assert _assign_temporal_bucket(now - 60 * SECONDS_PER_DAY, now) == 3

    def test_aging_bucket(self):
        now = int(time.time())
        assert _assign_temporal_bucket(now - 200 * SECONDS_PER_DAY, now) == 4

    def test_old_bucket(self):
        now = int(time.time())
        assert _assign_temporal_bucket(now - 400 * SECONDS_PER_DAY, now) == 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEventSelection:
    def test_triageable_excludes_identity(self, store):
        now = int(time.time())
        e1 = Event(
            id=Event.make_id("mail", "id_event"),
            source="mail", source_id="id_event",
            event_type=EVENT_TYPE_IDENTITY,
            timestamp=now, raw_content="identity",
        )
        e2 = Event(
            id=Event.make_id("mail", "real_event"),
            source="mail", source_id="real_event",
            event_type=EVENT_TYPE_EMAIL,
            timestamp=now, raw_content="real email",
        )
        store.put_event(e1)
        store.put_event(e2)
        events = get_triageable_events(store, resume=False)
        ids = {e["id"] for e in events}
        assert e1.id not in ids
        assert e2.id in ids

    def test_triageable_excludes_null_content(self, store):
        now = int(time.time())
        e = Event(
            id=Event.make_id("mail", "null_content"),
            source="mail", source_id="null_content",
            event_type=EVENT_TYPE_EMAIL,
            timestamp=now, raw_content=None,
        )
        store.put_event(e)
        events = get_triageable_events(store, resume=False)
        assert len(events) == 0

    def test_resume_skips_already_triaged(self, store):
        now = int(time.time())
        eid = Event.make_id("mail", "already_triaged")
        e = Event(
            id=eid, source="mail", source_id="already_triaged",
            event_type=EVENT_TYPE_EMAIL, timestamp=now,
            raw_content="test email",
        )
        store.put_event(e)

        # Store a triage claim for this event
        claim = _per_event_claim_from_triage_result(eid, TriageResult(score=0.5), "model")
        store.put_claim(claim)

        events = get_triageable_events(store, resume=True)
        assert len(events) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sender cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSenderCache:
    def test_builds_from_persons(self, store):
        store.put_person("p1", canonical_name="Alice")
        cache = build_sender_cache(store)
        assert "p1" in cache
        assert cache["p1"]["name"] == "Alice"

    def test_tier_from_frequency_claims(self, store):
        store.put_person("p1", canonical_name="Alice")
        freq_claim = Claim(
            id="claim:freq_1",
            event_ids=[],
            claim_type="communication_frequency",
            subject="user",
            predicate="communicates_with:p1",
            object=json.dumps({"event_count": 75}),
            confidence=0.9,
            provenance=ExtractionProvenance(model_id="deterministic"),
        )
        store.put_claim(freq_claim)
        cache = build_sender_cache(store)
        assert cache["p1"]["tier"] == 1

    def test_tier2_threshold(self, store):
        store.put_person("p1", canonical_name="Bob")
        freq_claim = Claim(
            id="claim:freq_2",
            event_ids=[],
            claim_type="communication_frequency",
            subject="user",
            predicate="communicates_with:p1",
            object=json.dumps({"event_count": 15}),
            confidence=0.7,
            provenance=ExtractionProvenance(model_id="deterministic"),
        )
        store.put_claim(freq_claim)
        cache = build_sender_cache(store)
        assert cache["p1"]["tier"] == 2

    def test_default_tier_3(self, store):
        store.put_person("p1", canonical_name="Nobody")
        cache = build_sender_cache(store)
        assert cache["p1"]["tier"] == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunTriage:
    def test_empty_store(self, store, mock_llm):
        result = run_triage(store, mock_llm)
        assert result["triaged"] == 0
        assert result["failed"] == 0

    def test_triages_events(self, store, mock_llm):
        now = int(time.time())
        e = Event(
            id=Event.make_id("mail", "triage_test"),
            source="mail", source_id="triage_test",
            event_type=EVENT_TYPE_EMAIL,
            timestamp=now,
            raw_content="Please send the proposal by Friday.",
        )
        store.put_event(e)
        store.put_person("person_user", is_user=True)

        result = run_triage(store, mock_llm)
        assert result["triaged"] >= 1
        assert mock_llm.generate_calls

    def test_stores_triage_claims(self, store, mock_llm):
        now = int(time.time())
        eid = Event.make_id("mail", "claim_check")
        e = Event(
            id=eid, source="mail", source_id="claim_check",
            event_type=EVENT_TYPE_EMAIL, timestamp=now,
            raw_content="Review the document.",
        )
        store.put_event(e)

        run_triage(store, mock_llm)

        claim = store.get_claim(triage_claim_id(eid))
        assert claim is not None
        assert claim.claim_type == "triage"

    def test_resume_mode_skips_triaged(self, store, mock_llm):
        now = int(time.time())
        eid = Event.make_id("mail", "resume_test")
        e = Event(
            id=eid, source="mail", source_id="resume_test",
            event_type=EVENT_TYPE_EMAIL, timestamp=now,
            raw_content="Already triaged email.",
        )
        store.put_event(e)

        run_triage(store, mock_llm, resume=True)
        call_count_1 = len(mock_llm.generate_calls)

        run_triage(store, mock_llm, resume=True)
        call_count_2 = len(mock_llm.generate_calls)

        # No new LLM calls in second run
        assert call_count_2 == call_count_1

    def test_lens_skips_skip_routed_events(self, store, mock_llm):
        """run_triage with lens excludes skip-routed events."""
        now = int(time.time())
        e_skip = Event(
            id=Event.make_id("mail", "lens_skip"),
            source="mail", source_id="lens_skip",
            event_type=EVENT_TYPE_EMAIL, timestamp=now,
            raw_content="This is automated junk.",
        )
        e_triage = Event(
            id=Event.make_id("mail", "lens_triage"),
            source="mail", source_id="lens_triage",
            event_type=EVENT_TYPE_EMAIL, timestamp=now,
            raw_content="Please review the proposal.",
        )
        store.put_event(e_skip)
        store.put_event(e_triage)

        from alteris.models import Projection
        store.put_projections_batch([
            Projection(event_id=e_skip.id, lens="chief_of_staff",
                       score=0.0, route="skip", components={}, computed_at=now),
            Projection(event_id=e_triage.id, lens="chief_of_staff",
                       score=0.5, route="full_triage", components={}, computed_at=now),
        ])

        result = run_triage(store, mock_llm, resume=False, lens="chief_of_staff")
        assert result["triaged"] >= 1

        # The skip event should not have a triage claim
        skip_claim = store.get_claim(triage_claim_id(e_skip.id))
        assert skip_claim is None

        # The triaged event should have a claim
        triage_claim = store.get_claim(triage_claim_id(e_triage.id))
        assert triage_claim is not None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Downstream queries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDownstreamQueries:
    def _insert_triaged_event(self, store, source_id, score, pii=None, commitment_type=None):
        now = int(time.time())
        eid = Event.make_id("mail", source_id)
        e = Event(
            id=eid, source="mail", source_id=source_id,
            event_type=EVENT_TYPE_EMAIL, timestamp=now,
            raw_content="Content",
        )
        store.put_event(e)
        result = TriageResult(
            score=score, domain="work",
            pii=pii or [], commitment_type=commitment_type,
        )
        claim = _per_event_claim_from_triage_result(eid, result, "model")
        store.put_claim(claim)
        return eid

    def test_get_triage_result(self, store):
        eid = self._insert_triaged_event(store, "dq1", 0.8)
        result = get_triage_result(store, eid)
        assert result is not None
        assert result["score"] == 0.8

    def test_get_triage_result_not_found(self, store):
        assert get_triage_result(store, "nonexistent") is None

    def test_deep_extraction_candidates(self, store):
        self._insert_triaged_event(store, "deep1", 0.8)
        self._insert_triaged_event(store, "low1", 0.2)
        candidates = get_deep_extraction_candidates(store)
        assert len(candidates) == 1
        assert candidates[0]["score"] == 0.8

    def test_sensitive_events(self, store):
        self._insert_triaged_event(store, "sens1", 0.5, pii=["financial"])
        self._insert_triaged_event(store, "safe1", 0.5)
        sensitive = get_sensitive_events(store)
        assert len(sensitive) == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Projection-based route filtering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestProjectionRouteFiltering:
    """get_triageable_events respects scoring projections."""

    def _make_and_store(self, store, source_id, raw_content="test"):
        now = int(time.time())
        e = Event(
            id=Event.make_id("mail", source_id),
            source="mail", source_id=source_id,
            event_type=EVENT_TYPE_EMAIL, timestamp=now,
            raw_content=raw_content,
        )
        store.put_event(e)
        return e

    def test_skip_routed_events_excluded(self, store):
        """Events with projection route='skip' are excluded when lens is provided."""
        e_skip = self._make_and_store(store, "skip_1")
        e_keep = self._make_and_store(store, "keep_1")

        from alteris.models import Projection
        now = int(time.time())
        store.put_projections_batch([
            Projection(event_id=e_skip.id, lens="chief_of_staff",
                       score=0.0, route="skip", components={}, computed_at=now),
            Projection(event_id=e_keep.id, lens="chief_of_staff",
                       score=0.5, route="full_triage", components={}, computed_at=now),
        ])

        events = get_triageable_events(store, resume=False, lens="chief_of_staff")
        ids = {e["id"] for e in events}
        assert e_skip.id not in ids
        assert e_keep.id in ids

    def test_no_lens_means_no_filtering(self, store):
        """Without lens, all events are returned regardless of projections."""
        e = self._make_and_store(store, "no_lens_1")

        from alteris.models import Projection
        now = int(time.time())
        store.put_projections_batch([
            Projection(event_id=e.id, lens="chief_of_staff",
                       score=0.0, route="skip", components={}, computed_at=now),
        ])

        events = get_triageable_events(store, resume=False, lens="")
        ids = {e["id"] for e in events}
        assert e.id in ids

    def test_events_without_projections_included(self, store):
        """Events that haven't been scored yet are still triaged."""
        e = self._make_and_store(store, "no_proj_1")
        events = get_triageable_events(store, resume=False, lens="chief_of_staff")
        ids = {e["id"] for e in events}
        assert e.id in ids

    def test_low_priority_included(self, store):
        """low_priority events are triaged (only skip is excluded)."""
        e = self._make_and_store(store, "low_pri_1")

        from alteris.models import Projection
        now = int(time.time())
        store.put_projections_batch([
            Projection(event_id=e.id, lens="chief_of_staff",
                       score=0.2, route="low_priority", components={}, computed_at=now),
        ])

        events = get_triageable_events(store, resume=False, lens="chief_of_staff")
        ids = {e["id"] for e in events}
        assert e.id in ids

    def test_skip_filter_with_resume(self, store):
        """Skip filtering works in resume mode too."""
        e_skip = self._make_and_store(store, "resume_skip_1")
        e_keep = self._make_and_store(store, "resume_keep_1")

        from alteris.models import Projection
        now = int(time.time())
        store.put_projections_batch([
            Projection(event_id=e_skip.id, lens="chief_of_staff",
                       score=0.0, route="skip", components={}, computed_at=now),
            Projection(event_id=e_keep.id, lens="chief_of_staff",
                       score=0.3, route="low_priority", components={}, computed_at=now),
        ])

        events = get_triageable_events(store, resume=True, lens="chief_of_staff")
        ids = {e["id"] for e in events}
        assert e_skip.id not in ids
        assert e_keep.id in ids

    def test_different_lens_not_affected(self, store):
        """Skip projection for one lens doesn't affect another lens."""
        e = self._make_and_store(store, "cross_lens_1")

        from alteris.models import Projection
        now = int(time.time())
        store.put_projections_batch([
            Projection(event_id=e.id, lens="chief_of_staff",
                       score=0.0, route="skip", components={}, computed_at=now),
            Projection(event_id=e.id, lens="financial_audit",
                       score=0.6, route="full_triage", components={}, computed_at=now),
        ])

        events_cos = get_triageable_events(store, resume=False, lens="chief_of_staff")
        events_fin = get_triageable_events(store, resume=False, lens="financial_audit")
        assert e.id not in {e["id"] for e in events_cos}
        assert e.id in {e["id"] for e in events_fin}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy selection with routes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestStrategySelection:
    """select_strategy factors in projection routes."""

    def _make_event_dict(self, eid, sensitivity=SensitivityLevel.SENSITIVE):
        return {"id": eid, "sensitivity": sensitivity}

    def test_default_thread_full(self):
        """Without routes, non-sensitive thread goes to THREAD_FULL (Gemini)."""
        events = [self._make_event_dict("e1"), self._make_event_dict("e2")]
        assert select_strategy(events) == TriageStrategy.THREAD_FULL

    def test_critical_overrides_route(self):
        """CRITICAL sensitivity always forces local, regardless of route."""
        events = [self._make_event_dict("e1", SensitivityLevel.CRITICAL)]
        routes = {"e1": "full_triage"}
        assert select_strategy(events, event_routes=routes) == TriageStrategy.MSG_COMPACT

    def test_critical_multi_event_batch(self):
        """Multi-event CRITICAL thread uses MSG_BATCH_COMPACT."""
        events = [
            self._make_event_dict("e1", SensitivityLevel.CRITICAL),
            self._make_event_dict("e2"),
        ]
        assert select_strategy(events) == TriageStrategy.MSG_BATCH_COMPACT

    def test_all_low_priority_single_event(self):
        """Single low_priority event → MSG_COMPACT (local model)."""
        events = [self._make_event_dict("e1")]
        routes = {"e1": "low_priority"}
        assert select_strategy(events, event_routes=routes) == TriageStrategy.MSG_COMPACT

    def test_all_low_priority_multi_event(self):
        """All-low_priority thread → MSG_BATCH_COMPACT (local model)."""
        events = [self._make_event_dict("e1"), self._make_event_dict("e2")]
        routes = {"e1": "low_priority", "e2": "low_priority"}
        assert select_strategy(events, event_routes=routes) == TriageStrategy.MSG_BATCH_COMPACT

    def test_one_full_triage_carries_thread(self):
        """If ANY event is full_triage, thread goes to THREAD_FULL (Gemini)."""
        events = [self._make_event_dict("e1"), self._make_event_dict("e2")]
        routes = {"e1": "low_priority", "e2": "full_triage"}
        assert select_strategy(events, event_routes=routes) == TriageStrategy.THREAD_FULL

    def test_no_routes_means_thread_full(self):
        """None event_routes → original behavior (THREAD_FULL for non-sensitive)."""
        events = [self._make_event_dict("e1")]
        assert select_strategy(events, event_routes=None) == TriageStrategy.THREAD_FULL

    def test_empty_routes_means_thread_full(self):
        """Empty routes dict → no event matched, falls through to THREAD_FULL."""
        events = [self._make_event_dict("e1")]
        routes: dict[str, str] = {}
        assert select_strategy(events, event_routes=routes) == TriageStrategy.THREAD_FULL

    def test_unknown_event_conservative_cloud(self):
        """Event not in routes dict → conservatively routed to cloud (THREAD_FULL).
        An event without a projection might be important, so we don't downgrade it."""
        events = [self._make_event_dict("e1")]
        routes = {"e_other": "full_triage"}  # e1 not in routes
        # e1 has no route → not confirmed as low_priority → cloud
        assert select_strategy(events, event_routes=routes) == TriageStrategy.THREAD_FULL
