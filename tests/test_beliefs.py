"""Tests for alteris.beliefs: Per-thread synthesis + claims-to-beliefs compiler (Stage 7).

Tests cover:
  - SynthesisCommitmentResult Pydantic validation
  - _belief_id determinism
  - Context builders (_build_person_context, _build_triage_context,
    _build_group_context, _build_thread_age_context)
  - _parse_synthesis_result
  - _commitment_claim_id determinism
  - _build_commitment_claim construction
  - synthesize_thread (per-thread LLM synthesis)
  - dedup_commitments (fuzzy dedup on commitment claims)
  - expire_stale_commitments
  - Downstream queries (get_open_commitments, get_overdue_commitments,
    get_commitments_for_thread)
  - _build_entity_beliefs (person -> entity)
  - _build_relation_beliefs (channels + directionality -> relations)
  - _expire_stale_beliefs
  - cluster_by_similarity (reusable fuzzy clustering)
  - _build_commitment_beliefs (commitment claims -> FACT beliefs)
  - _build_logistics_beliefs (logistics claims -> FACT beliefs)
  - run_synthesis full pipeline
  - Edge cases (empty store, no user, no claims)
"""

import json
import time

import pytest

from alteris.beliefs import (
    COMPATIBLE_TYPES,
    SynthesisCommitmentResult,
    VALID_COMMITMENT_TYPES,
    VALID_DIRECTIONS,
    _belief_id,
    _build_commitment_beliefs,
    _build_commitment_claim,
    _build_entity_beliefs,
    _build_group_context,
    _build_logistics_beliefs,
    _build_person_context,
    _build_relation_beliefs,
    _build_thread_age_context,
    _build_triage_context,
    _commitment_claim_id,
    _expire_stale_beliefs,
    _logistics_expiry,
    _logistics_summary,
    _parse_synthesis_result,
    cluster_by_similarity,
    dedup_commitments,
    expire_stale_commitments,
    get_commitments_for_thread,
    get_open_commitments,
    get_overdue_commitments,
    run_synthesis,
    synthesize_thread,
)
from alteris.constants import (
    SECONDS_PER_DAY,
    STALENESS_THREAD_AGE_DAYS,
)
from alteris.extract import ThreadBundle
from alteris.llm.mock import MOCK_SYNTHESIS_COMMITMENT_RESPONSE, MockLLMClient
from alteris.models import (
    Belief,
    BeliefStatus,
    BeliefType,
    Claim,
    EpistemicLevel,
    Event,
    ExtractionProvenance,
    Modality,
)
from alteris.store import LayeredGraphStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_claim(
    claim_id: str, claim_type: str, subject: str,
    predicate: str, obj: dict, confidence: float = 0.8,
    event_ids: list[str] | None = None,
    created_at: int | None = None,
) -> Claim:
    return Claim(
        id=claim_id,
        event_ids=event_ids or [],
        claim_type=claim_type,
        subject=subject,
        predicate=predicate,
        object=json.dumps(obj),
        confidence=confidence,
        modality=Modality.OBSERVED,
        provenance=ExtractionProvenance(model_id="test"),
        created_at=created_at or int(time.time()),
    )


def _make_event(
    source: str = "mail",
    source_id: str = "",
    event_type: str = "email",
    timestamp: int = 0,
    participants: tuple[str, ...] = ("sender@example.com", "user@example.com"),
    raw_content: str = "Test email body content.",
    metadata: dict | None = None,
) -> Event:
    if not source_id:
        source_id = f"test_{time.time_ns()}"
    if not timestamp:
        timestamp = int(time.time())
    eid = Event.make_id(source, source_id)
    return Event(
        id=eid,
        source=source,
        source_id=source_id,
        event_type=event_type,
        timestamp=timestamp,
        participants=participants,
        raw_content=raw_content,
        content_hash=Event.content_hash_of(raw_content) if raw_content else "",
        metadata=metadata or {},
    )


def _make_bundle(
    thread_id: str = "thread_test",
    events: list[Event] | None = None,
    triage_data: list[dict] | None = None,
) -> ThreadBundle:
    if events is None:
        events = [_make_event()]
    if triage_data is None:
        triage_data = [{}]
    return ThreadBundle(
        thread_id=thread_id,
        events=events,
        triage_data=triage_data,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SynthesisCommitmentResult validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSynthesisCommitmentResult:
    def test_valid_result(self):
        r = SynthesisCommitmentResult(
            type="inbound_request",
            who="user",
            what="send the updated proposal",
            to_whom="Sam",
            direction="direct_ask",
            deadline="2026-02-15",
            status="open",
            priority=2,
            confidence=0.85,
            staleness_signal="none",
            provenance="assigned_to_user",
            note="Sam asked in email thread",
            source_message_id="msg_0",
        )
        assert r.type == "inbound_request"
        assert r.who == "user"
        assert r.what == "send the updated proposal"
        assert r.to_whom == "Sam"
        assert r.direction == "direct_ask"
        assert r.deadline == "2026-02-15"
        assert r.status == "open"
        assert r.priority == 2
        assert r.confidence == 0.85
        assert r.staleness_signal == "none"
        assert r.provenance == "assigned_to_user"
        assert r.note == "Sam asked in email thread"
        assert r.source_message_id == "msg_0"

    def test_invalid_type_defaults(self):
        r = SynthesisCommitmentResult(type="garbage", what="do something")
        assert r.type == "inbound_request"

    def test_invalid_status_defaults(self):
        r = SynthesisCommitmentResult(status="garbage", what="do something")
        assert r.status == "open"

    def test_invalid_direction_defaults(self):
        r = SynthesisCommitmentResult(direction="garbage", what="do something")
        assert r.direction == "ambiguous"

    def test_invalid_staleness_defaults(self):
        r = SynthesisCommitmentResult(staleness_signal="garbage", what="do something")
        assert r.staleness_signal == "none"

    def test_priority_clamped_high(self):
        r = SynthesisCommitmentResult(priority=10, what="do something")
        assert r.priority == 3

    def test_priority_clamped_low(self):
        r = SynthesisCommitmentResult(priority=0, what="do something")
        assert r.priority == 1

    def test_confidence_clamped(self):
        r = SynthesisCommitmentResult(confidence=2.0, what="do something")
        assert r.confidence == 1.0

    def test_confidence_negative(self):
        r = SynthesisCommitmentResult(confidence=-1.0, what="do something")
        assert r.confidence == 0.0

    def test_what_truncated(self):
        r = SynthesisCommitmentResult(what="x" * 300)
        assert len(r.what) <= 200

    def test_deadline_null_cleared(self):
        r = SynthesisCommitmentResult(what="do something", deadline="null")
        assert r.deadline is None

    def test_deadline_invalid_format(self):
        r = SynthesisCommitmentResult(what="do something", deadline="February 15")
        assert r.deadline is None

    def test_all_valid_types(self):
        for t in VALID_COMMITMENT_TYPES:
            r = SynthesisCommitmentResult(type=t, what="do something")
            assert r.type == t

    def test_all_valid_directions(self):
        for d in VALID_DIRECTIONS:
            r = SynthesisCommitmentResult(direction=d, what="do something")
            assert r.direction == d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Belief ID
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBeliefId:
    def test_deterministic(self):
        id1 = _belief_id("entity", "person_a", "A is a colleague")
        id2 = _belief_id("entity", "person_a", "A is a colleague")
        assert id1 == id2

    def test_different_inputs(self):
        id1 = _belief_id("entity", "person_a", "summary A")
        id2 = _belief_id("entity", "person_b", "summary B")
        assert id1 != id2

    def test_prefix(self):
        bid = _belief_id("entity", "sub", "sum")
        assert bid.startswith("belief:")

    def test_hex_format(self):
        bid = _belief_id("entity", "sub", "sum")
        hex_part = bid.replace("belief:", "")
        assert all(c in "0123456789abcdef" for c in hex_part)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Context builders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestContextBuilders:
    def test_person_context_includes_names(self):
        store = LayeredGraphStore(db_path=":memory:")
        store.put_person("p_user", canonical_name="User", is_user=True)
        store.put_person("p_alice", canonical_name="Alice")

        e = _make_event(source_id="ctx_email_1")
        store.put_event(e)
        store.link_event_person(e.id, "p_alice", "sender")
        store.link_event_person(e.id, "p_user", "recipient")

        bundle = _make_bundle(events=[e])

        # Build persons_cache manually
        persons_cache = {
            e.id: [
                {"person_id": "p_alice", "name": "Alice", "is_user": False, "role": "sender"},
                {"person_id": "p_user", "name": "User", "is_user": True, "role": "recipient"},
            ]
        }

        result = _build_person_context(store, bundle, persons_cache)
        assert "Alice" in result
        # User entries are skipped
        assert "PERSON CONTEXT" in result

    def test_person_context_empty_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        bundle = _make_bundle()
        result = _build_person_context(store, bundle)
        assert "No known contacts" in result

    def test_triage_context_includes_scores(self):
        bundle = _make_bundle(
            triage_data=[
                {"score": 0.8, "domain": "work", "topics": ["project"]},
                {"score": 0.6, "domain": "personal", "topics": ["family"]},
            ]
        )
        result = _build_triage_context(bundle)
        assert "avg_score" in result
        assert "0.70" in result
        assert "work" in result

    def test_triage_context_empty(self):
        bundle = _make_bundle(triage_data=[])
        # triage_data is empty list, which is falsy
        bundle.triage_data = []
        result = _build_triage_context(bundle)
        assert "No triage data" in result

    def test_group_context_detects_group(self):
        e = _make_event(
            metadata={"is_group": True, "group_name": "Family Chat"},
            participants=("alice@test.com", "bob@test.com", "user@test.com"),
        )
        bundle = _make_bundle(events=[e])
        result = _build_group_context(bundle)
        assert "is_group=true" in result
        assert "Family Chat" in result

    def test_group_context_not_group(self):
        e = _make_event(metadata={"is_group": False})
        bundle = _make_bundle(events=[e])
        result = _build_group_context(bundle)
        assert "is_group=false" in result

    def test_thread_age_recent(self):
        now = int(time.time())
        e = _make_event(timestamp=now - 3600)  # 1 hour ago
        bundle = _make_bundle(events=[e])
        result = _build_thread_age_context(bundle)
        assert "THREAD AGE" in result
        assert "last_activity=0 days ago" in result
        # Should NOT have staleness warning
        assert "WARNING" not in result

    def test_thread_age_old(self):
        now = int(time.time())
        old_ts = now - (STALENESS_THREAD_AGE_DAYS + 5) * SECONDS_PER_DAY
        e = _make_event(timestamp=old_ts)
        bundle = _make_bundle(events=[e])
        result = _build_thread_age_context(bundle)
        assert "THREAD AGE" in result
        assert "WARNING" in result
        assert "stale" in result.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Parse synthesis result
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestParseSynthesisResult:
    def test_valid_commitments(self):
        raw = {
            "commitments": [
                {
                    "type": "inbound_request",
                    "who": "user",
                    "what": "send the updated proposal",
                    "direction": "direct_ask",
                    "deadline": "2026-02-15",
                    "status": "open",
                    "priority": 2,
                    "confidence": 0.85,
                    "staleness_signal": "none",
                }
            ]
        }
        results = _parse_synthesis_result(raw)
        assert len(results) == 1
        assert isinstance(results[0], SynthesisCommitmentResult)
        assert results[0].what == "send the updated proposal"

    def test_empty_commitments(self):
        results = _parse_synthesis_result({"commitments": []})
        assert results == []

    def test_none_input(self):
        results = _parse_synthesis_result(None)
        assert results == []

    def test_missing_key(self):
        results = _parse_synthesis_result({"other": []})
        assert results == []

    def test_skips_empty_what(self):
        raw = {
            "commitments": [
                {"type": "deadline", "what": ""},
                {"type": "deadline", "what": "   "},
                {"type": "deadline", "what": "valid task"},
            ]
        }
        results = _parse_synthesis_result(raw)
        # Empty-what entries are filtered out
        assert len(results) == 1
        assert results[0].what == "valid task"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Commitment claim ID
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCommitmentClaimId:
    def test_deterministic(self):
        id1 = _commitment_claim_id("t1", "do X", "user")
        id2 = _commitment_claim_id("t1", "do X", "user")
        assert id1 == id2

    def test_different(self):
        id1 = _commitment_claim_id("t1", "do X", "user")
        id2 = _commitment_claim_id("t2", "do Y", "alice")
        assert id1 != id2

    def test_hex_format(self):
        cid = _commitment_claim_id("t1", "do X", "user")
        assert all(c in "0123456789abcdef" for c in cid)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build commitment claim
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBuildCommitmentClaim:
    def _make_result(self, **kwargs) -> SynthesisCommitmentResult:
        defaults = {
            "type": "inbound_request",
            "who": "user",
            "what": "send the updated proposal",
            "to_whom": "Sam",
            "direction": "direct_ask",
            "deadline": "2026-02-15",
            "status": "open",
            "priority": 2,
            "confidence": 0.85,
            "staleness_signal": "none",
        }
        defaults.update(kwargs)
        return SynthesisCommitmentResult(**defaults)

    def test_basic_claim(self):
        result = self._make_result()
        claim = _build_commitment_claim(
            result, "thread_1", ["evt_1", "evt_2"], "gemini-flash",
        )
        assert claim.claim_type == "commitment"
        assert claim.subject == "thread_1"
        assert claim.predicate == "inbound_request"
        assert claim.confidence == 0.85

    def test_includes_direction(self):
        result = self._make_result(direction="group_ask")
        claim = _build_commitment_claim(
            result, "thread_1", ["evt_1"], "gemini-flash",
        )
        obj = json.loads(claim.object)
        assert obj["direction"] == "group_ask"

    def test_includes_staleness(self):
        result = self._make_result(staleness_signal="old_thread")
        claim = _build_commitment_claim(
            result, "thread_1", ["evt_1"], "gemini-flash",
        )
        obj = json.loads(claim.object)
        assert obj["staleness_signal"] == "old_thread"

    def test_includes_to_whom(self):
        result = self._make_result(to_whom="group:Family Chat")
        claim = _build_commitment_claim(
            result, "thread_1", ["evt_1"], "gemini-flash",
        )
        obj = json.loads(claim.object)
        assert obj["to_whom"] == "group:Family Chat"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Synthesize thread
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSynthesizeThread:
    def test_returns_claims(self):
        e = _make_event(
            raw_content="Please send the updated proposal by Friday.",
        )
        bundle = _make_bundle(thread_id="thread_synth", events=[e])
        mock_llm = MockLLMClient(
            json_responses={"THREAD CONTENT": MOCK_SYNTHESIS_COMMITMENT_RESPONSE},
        )
        claims, _, _ = synthesize_thread(bundle, mock_llm)
        assert len(claims) >= 1
        assert claims[0].claim_type == "commitment"

    def test_returns_msg_id_map(self):
        e = _make_event()
        bundle = _make_bundle(events=[e])
        mock_llm = MockLLMClient(
            json_responses={"THREAD CONTENT": MOCK_SYNTHESIS_COMMITMENT_RESPONSE},
        )
        _, msg_id_map, _ = synthesize_thread(bundle, mock_llm)
        assert isinstance(msg_id_map, dict)
        assert "msg_0" in msg_id_map
        assert msg_id_map["msg_0"] == e.id

    def test_empty_response(self):
        e = _make_event()
        bundle = _make_bundle(events=[e])
        mock_llm = MockLLMClient(
            json_responses={"THREAD CONTENT": {"commitments": []}},
        )
        claims, _, _ = synthesize_thread(bundle, mock_llm)
        assert claims == []

    def test_llm_called(self):
        e = _make_event()
        bundle = _make_bundle(events=[e])
        mock_llm = MockLLMClient(
            json_responses={"THREAD CONTENT": MOCK_SYNTHESIS_COMMITMENT_RESPONSE},
        )
        synthesize_thread(bundle, mock_llm)
        assert len(mock_llm.generate_calls) >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dedup commitments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDedupCommitments:
    def _put_commitment(
        self, store, claim_id, thread_id, what, who="user",
        ctype="inbound_request", confidence=0.8, deadline=None,
    ):
        obj = {
            "type": ctype, "who": who, "what": what,
            "status": "open", "direction": "direct_ask",
            "staleness_signal": "none",
        }
        if deadline:
            obj["deadline"] = deadline
        claim = Claim(
            id=claim_id,
            event_ids=[],
            claim_type="commitment",
            subject=thread_id,
            predicate=ctype,
            object=json.dumps(obj),
            confidence=confidence,
            modality=Modality.OBSERVED,
            provenance=ExtractionProvenance(model_id="test"),
        )
        store.put_claim(claim)

    def test_merges_same_what(self):
        store = LayeredGraphStore(db_path=":memory:")
        # Same "what" across different threads -> should merge
        self._put_commitment(store, "c1", "thread_a",
                             "send the updated proposal", confidence=0.9)
        self._put_commitment(store, "c2", "thread_b",
                             "send the updated proposal", confidence=0.7)
        result = dedup_commitments(store)
        assert result["merged"] >= 1

    def test_no_merge_different_what(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(store, "c1", "thread_a",
                             "send the updated proposal")
        self._put_commitment(store, "c2", "thread_b",
                             "review the contract draft")
        result = dedup_commitments(store)
        assert result["merged"] == 0

    def test_compatible_types_merge(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(store, "c1", "thread_a",
                             "send the updated proposal",
                             ctype="inbound_request", confidence=0.9)
        self._put_commitment(store, "c2", "thread_b",
                             "send the updated proposal",
                             ctype="user_commitment", confidence=0.7)
        result = dedup_commitments(store)
        # inbound_request + user_commitment are compatible
        assert frozenset({"inbound_request", "user_commitment"}) in COMPATIBLE_TYPES
        assert result["merged"] >= 1

    def test_single_claim_no_dedup(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(store, "c1", "thread_a", "send the proposal")
        result = dedup_commitments(store)
        assert result["merged"] == 0

    def test_empty_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        result = dedup_commitments(store)
        assert result["merged"] == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Expire stale commitments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExpireStaleCommitments:
    def test_expires_old_no_deadline(self):
        store = LayeredGraphStore(db_path=":memory:")
        # Create a commitment 40+ days old with no deadline
        old_time = int(time.time()) - (40 * SECONDS_PER_DAY)
        claim = Claim(
            id="com_old_no_dl",
            event_ids=[],
            claim_type="commitment",
            subject="thread_old",
            predicate="inbound_request",
            object=json.dumps({
                "what": "Old task no deadline", "who": "user",
                "status": "open", "type": "inbound_request",
                "direction": "direct_ask", "staleness_signal": "none",
            }),
            confidence=0.8,
            modality=Modality.OBSERVED,
            provenance=ExtractionProvenance(model_id="test"),
            created_at=old_time,
        )
        store.put_claim(claim)
        expired = expire_stale_commitments(store)
        assert expired >= 1

    def test_keeps_recent(self):
        store = LayeredGraphStore(db_path=":memory:")
        claim = Claim(
            id="com_recent",
            event_ids=[],
            claim_type="commitment",
            subject="thread_recent",
            predicate="inbound_request",
            object=json.dumps({
                "what": "Recent task", "who": "user",
                "status": "open", "type": "inbound_request",
                "direction": "direct_ask", "staleness_signal": "none",
            }),
            confidence=0.8,
            modality=Modality.OBSERVED,
            provenance=ExtractionProvenance(model_id="test"),
        )
        store.put_claim(claim)
        expired = expire_stale_commitments(store)
        assert expired == 0

    def test_empty_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        expired = expire_stale_commitments(store)
        assert expired == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Downstream queries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDownstreamQueries:
    def _put_commitment(
        self, store, claim_id, thread_id, what, status="open",
        deadline=None, direction="direct_ask",
    ):
        obj = {
            "type": "inbound_request", "who": "user", "what": what,
            "status": status, "direction": direction,
            "staleness_signal": "none", "priority": 2,
        }
        if deadline:
            obj["deadline"] = deadline
        claim = Claim(
            id=claim_id,
            event_ids=[],
            claim_type="commitment",
            subject=thread_id,
            predicate="inbound_request",
            object=json.dumps(obj),
            confidence=0.8,
            modality=Modality.OBSERVED,
            provenance=ExtractionProvenance(model_id="test"),
        )
        store.put_claim(claim)

    def test_get_open_commitments(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(store, "c_open", "t1", "open task", status="open")
        self._put_commitment(store, "c_done", "t2", "done task", status="done")
        results = get_open_commitments(store)
        assert len(results) == 1
        assert results[0]["what"] == "open task"

    def test_get_open_includes_direction(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(store, "c1", "t1", "task A",
                             direction="group_ask")
        results = get_open_commitments(store)
        assert len(results) == 1
        assert results[0]["direction"] == "group_ask"

    def test_get_overdue_commitments(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(store, "c_past", "t1", "past due task",
                             deadline="2020-01-01")
        self._put_commitment(store, "c_future", "t2", "future task",
                             deadline="2099-01-01")
        results = get_overdue_commitments(store)
        assert len(results) == 1
        assert results[0]["what"] == "past due task"

    def test_get_commitments_for_thread(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(store, "c1", "thread_alpha", "alpha task")
        self._put_commitment(store, "c2", "thread_beta", "beta task")
        results = get_commitments_for_thread(store, "thread_alpha")
        assert len(results) == 1
        assert results[0]["what"] == "alpha task"

    def test_empty_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        assert get_open_commitments(store) == []
        assert get_overdue_commitments(store) == []
        assert get_commitments_for_thread(store, "thread_x") == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entity beliefs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEntityBeliefs:
    def test_creates_for_each_person(self):
        store = LayeredGraphStore(db_path=":memory:")
        store.put_person("p1", canonical_name="Alice")
        store.put_person("p2", canonical_name="Bob")
        beliefs = _build_entity_beliefs(store)
        assert len(beliefs) == 2

    def test_user_confidence_1(self):
        store = LayeredGraphStore(db_path=":memory:")
        store.put_person("p_user", canonical_name="User", is_user=True)
        beliefs = _build_entity_beliefs(store)
        user_beliefs = [b for b in beliefs if b.data.get("is_user")]
        assert len(user_beliefs) == 1
        assert user_beliefs[0].confidence == 1.0

    def test_tier_from_message_count(self):
        store = LayeredGraphStore(db_path=":memory:")
        store.put_person("p1", canonical_name="Heavy Contact")
        c = _make_claim(
            "freq1", "communication_frequency", "user",
            "communicates_with:p1",
            {"event_count": 60, "person_name": "Heavy Contact"},
        )
        store.put_claim(c)
        beliefs = _build_entity_beliefs(store)
        heavy = [b for b in beliefs if b.subject == "p1"]
        assert len(heavy) == 1
        assert heavy[0].data["tier"] == 1

    def test_tier_3_for_infrequent(self):
        store = LayeredGraphStore(db_path=":memory:")
        store.put_person("p1", canonical_name="Rare Contact")
        beliefs = _build_entity_beliefs(store)
        rare = [b for b in beliefs if b.subject == "p1"]
        assert rare[0].data["tier"] == 3

    def test_empty_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        beliefs = _build_entity_beliefs(store)
        assert beliefs == []

    def test_entity_belief_type(self):
        store = LayeredGraphStore(db_path=":memory:")
        store.put_person("p1", canonical_name="Test")
        beliefs = _build_entity_beliefs(store)
        assert all(b.belief_type == BeliefType.ENTITY for b in beliefs)

    def test_includes_identifiers(self):
        store = LayeredGraphStore(db_path=":memory:")
        store.put_person("p1", canonical_name="Test")
        store.add_person_identifier("p1", "email", "test@test.com")
        beliefs = _build_entity_beliefs(store)
        assert "test@test.com" in beliefs[0].data.get("emails", [])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Relation beliefs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRelationBeliefs:
    def test_creates_from_channel_claims(self):
        store = LayeredGraphStore(db_path=":memory:")
        store.put_person("p_user", is_user=True)
        c = _make_claim(
            "ch1", "communication_channel", "user",
            "channels_with:p_contact",
            {"channels": ["mail", "whatsapp"], "channel_count": 2, "person_name": "Contact"},
        )
        store.put_claim(c)
        beliefs = _build_relation_beliefs(store)
        assert len(beliefs) == 1
        assert beliefs[0].belief_type == BeliefType.RELATION

    def test_multi_channel_classification(self):
        store = LayeredGraphStore(db_path=":memory:")
        store.put_person("p_user", is_user=True)
        c = _make_claim(
            "ch1", "communication_channel", "user",
            "channels_with:p_contact",
            {"channels": ["mail", "whatsapp", "slack"], "channel_count": 3, "person_name": "Contact"},
        )
        store.put_claim(c)
        beliefs = _build_relation_beliefs(store)
        assert beliefs[0].data["relation_type"] == "multi_channel_contact"

    def test_user_initiates_classification(self):
        store = LayeredGraphStore(db_path=":memory:")
        store.put_person("p_user", is_user=True)
        c_ch = _make_claim(
            "ch1", "communication_channel", "user",
            "channels_with:p_contact",
            {"channels": ["mail"], "channel_count": 1, "person_name": "Contact"},
        )
        c_dir = _make_claim(
            "dir1", "directionality", "user",
            "direction_with:p_contact",
            {"user_initiated_ratio": 0.8},
        )
        store.put_claim(c_ch)
        store.put_claim(c_dir)
        beliefs = _build_relation_beliefs(store)
        assert beliefs[0].data["relation_type"] == "user_initiates_with"

    def test_no_user_returns_empty(self):
        store = LayeredGraphStore(db_path=":memory:")
        beliefs = _build_relation_beliefs(store)
        assert beliefs == []

    def test_empty_claims(self):
        store = LayeredGraphStore(db_path=":memory:")
        store.put_person("p_user", is_user=True)
        beliefs = _build_relation_beliefs(store)
        assert beliefs == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Expire stale beliefs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExpireStale:
    def test_expires_past_deadline(self):
        store = LayeredGraphStore(db_path=":memory:")
        now = int(time.time())
        belief = Belief(
            id="belief:stale",
            belief_type=BeliefType.FACT,
            subject="test",
            summary="Should expire",
            data={},
            epistemic_level=EpistemicLevel.INFERENCE,
            source_claims=[],
            confidence=0.5,
            status=BeliefStatus.ACTIVE,
            expires_at=now - 1000,
        )
        store.put_belief(belief)
        expired = _expire_stale_beliefs(store)
        assert expired == 1

    def test_doesnt_expire_future(self):
        store = LayeredGraphStore(db_path=":memory:")
        now = int(time.time())
        belief = Belief(
            id="belief:fresh",
            belief_type=BeliefType.FACT,
            subject="test",
            summary="Still valid",
            data={},
            epistemic_level=EpistemicLevel.INFERENCE,
            source_claims=[],
            confidence=0.5,
            status=BeliefStatus.ACTIVE,
            expires_at=now + 100000,
        )
        store.put_belief(belief)
        expired = _expire_stale_beliefs(store)
        assert expired == 0

    def test_empty_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        expired = _expire_stale_beliefs(store)
        assert expired == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full synthesis pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunSynthesis:
    def _seed_store(self, store: LayeredGraphStore):
        """Seed a store with persons, identifiers, events, gate claims,
        stage1 claims, and triage claims for run_synthesis testing."""
        now = int(time.time())

        # Persons
        store.put_person("p_user", canonical_name="User", is_user=True)
        store.put_person("p_alice", canonical_name="Alice")
        store.add_person_identifier("p_user", "email", "user@example.com")
        store.add_person_identifier("p_alice", "email", "alice@test.com")

        # Events
        e1 = _make_event(
            source="mail", source_id="synth_email_1",
            timestamp=now - 3600,
            participants=("alice@test.com", "user@example.com"),
            raw_content="Please send me the report by Friday.",
            metadata={"subject": "Report", "thread_id": "thread_work", "is_from_me": False},
        )
        e2 = _make_event(
            source="mail", source_id="synth_email_2",
            timestamp=now - 1800,
            participants=("user@example.com", "alice@test.com"),
            raw_content="Sure, I will send it today.",
            metadata={"subject": "Re: Report", "thread_id": "thread_work", "is_from_me": True},
        )
        store.put_event(e1)
        store.put_event(e2)

        # Link events to persons
        store.link_event_person(e1.id, "p_alice", "sender")
        store.link_event_person(e1.id, "p_user", "recipient")
        store.link_event_person(e2.id, "p_user", "sender")
        store.link_event_person(e2.id, "p_alice", "recipient")

        # Stage 1 claims
        store.put_claim(_make_claim(
            "freq1", "communication_frequency", "p_user",
            "communicates_with:p_alice",
            {"event_count": 25, "person_name": "Alice"},
        ))
        store.put_claim(_make_claim(
            "ch1", "communication_channel", "p_user",
            "channels_with:p_alice",
            {"channels": ["mail", "whatsapp"], "channel_count": 2, "person_name": "Alice"},
        ))
        store.put_claim(_make_claim(
            "dir1", "directionality", "p_user",
            "direction_with:p_alice",
            {"user_initiated_ratio": 0.6, "user_sent": 6, "they_sent": 4, "total": 10},
        ))

        # Triage claims for events
        store.put_claim(_make_claim(
            "triage_e1", "triage", e1.id, "triage_result",
            {"score": 0.8, "domain": "work", "topics": ["report"]},
            event_ids=[e1.id],
        ))
        store.put_claim(_make_claim(
            "triage_e2", "triage", e2.id, "triage_result",
            {"score": 0.7, "domain": "work", "topics": ["report"]},
            event_ids=[e2.id],
        ))

        # Extraction gate claim (actionable)
        gate_claim = Claim(
            id="gate_thread_work",
            event_ids=[e1.id, e2.id],
            claim_type="extraction_gate",
            subject="thread_work",
            predicate="actionable",
            object=json.dumps({
                "actionable": True,
                "reason": "Direct request with deadline",
            }),
            confidence=1.0,
            modality=Modality.OBSERVED,
            provenance=ExtractionProvenance(
                model_id="gemini-flash",
                prompt_version="gate_v1",
            ),
        )
        store.put_claim(gate_claim)

        return e1, e2

    def test_empty_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        mock_llm = MockLLMClient()
        result = run_synthesis(store, mock_llm)
        assert result["total_beliefs"] == 0

    def test_produces_entity_beliefs(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_store(store)
        mock_llm = MockLLMClient(
            json_responses={"THREAD CONTENT": MOCK_SYNTHESIS_COMMITMENT_RESPONSE},
        )
        result = run_synthesis(store, mock_llm)
        assert result["entity_beliefs"] >= 2  # user + alice

    def test_produces_relation_beliefs(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_store(store)
        mock_llm = MockLLMClient(
            json_responses={"THREAD CONTENT": MOCK_SYNTHESIS_COMMITMENT_RESPONSE},
        )
        result = run_synthesis(store, mock_llm)
        assert result["relation_beliefs"] >= 1

    def test_result_structure(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_store(store)
        mock_llm = MockLLMClient(
            json_responses={"THREAD CONTENT": MOCK_SYNTHESIS_COMMITMENT_RESPONSE},
        )
        result = run_synthesis(store, mock_llm)
        expected_keys = {
            "total_commitments", "total_beliefs", "entity_beliefs",
            "relation_beliefs", "dedup_merged", "stale_commitments",
            "elapsed_seconds",
        }
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_beliefs_stored_in_db(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_store(store)
        mock_llm = MockLLMClient(
            json_responses={"THREAD CONTENT": MOCK_SYNTHESIS_COMMITMENT_RESPONSE},
        )
        run_synthesis(store, mock_llm)
        beliefs = store.get_beliefs(status="active")
        assert len(beliefs) > 0

    def test_result_includes_multigate_stats(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_store(store)
        mock_llm = MockLLMClient(
            json_responses={"THREAD CONTENT": MOCK_SYNTHESIS_COMMITMENT_RESPONSE},
        )
        result = run_synthesis(store, mock_llm)
        assert "total_logistics" in result
        assert "total_relational" in result
        assert "logistics_threads" in result
        assert "relational_threads" in result
        assert "logistics_errors" in result
        assert "relational_errors" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logistics extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from alteris.beliefs import (
    extract_logistics,
    extract_relational,
    _logistics_claim_id,
    _relational_claim_id,
)
from alteris.llm.mock import (
    MOCK_LOGISTICS_EXTRACTION_RESPONSE,
    MOCK_RELATIONAL_EXTRACTION_RESPONSE,
)


class TestLogisticsClaimId:
    def test_deterministic(self):
        id1 = _logistics_claim_id("t1", "Pink Salt 2026-02-14")
        id2 = _logistics_claim_id("t1", "Pink Salt 2026-02-14")
        assert id1 == id2

    def test_different_threads(self):
        id1 = _logistics_claim_id("t1", "Pink Salt")
        id2 = _logistics_claim_id("t2", "Pink Salt")
        assert id1 != id2

    def test_different_facts(self):
        id1 = _logistics_claim_id("t1", "Pink Salt")
        id2 = _logistics_claim_id("t1", "Alaska Airlines")
        assert id1 != id2


class TestRelationalClaimId:
    def test_deterministic(self):
        id1 = _relational_claim_id("t1", "Sam Park")
        id2 = _relational_claim_id("t1", "Sam Park")
        assert id1 == id2

    def test_different_threads(self):
        id1 = _relational_claim_id("t1", "Sam")
        id2 = _relational_claim_id("t2", "Sam")
        assert id1 != id2


class TestExtractLogistics:
    def test_extracts_facts(self):
        event = _make_event(raw_content="Reservation at The Garden Table, Feb 14, 4pm")
        bundle = ThreadBundle("t1", [event], [{"score": 0.8}])
        mock_llm = MockLLMClient(
            json_responses={"Garden Table": MOCK_LOGISTICS_EXTRACTION_RESPONSE}
        )
        claims = extract_logistics(bundle, mock_llm, model="test")
        assert len(claims) == 1
        assert claims[0].claim_type == "logistics"
        assert claims[0].predicate == "logistics_reservation"
        obj = json.loads(claims[0].object)
        assert obj["venue"] == "The Garden Table"
        assert obj["date"] == "2026-02-14"

    def test_empty_facts(self):
        event = _make_event(raw_content="Hello world")
        bundle = ThreadBundle("t1", [event], [{"score": 0.8}])
        mock_llm = MockLLMClient(
            json_responses={"Hello": {"facts": []}}
        )
        claims = extract_logistics(bundle, mock_llm, model="test")
        assert len(claims) == 0

    def test_no_facts_key(self):
        event = _make_event(raw_content="Hello world")
        bundle = ThreadBundle("t1", [event], [{"score": 0.8}])
        mock_llm = MockLLMClient(
            json_responses={"Hello": {"other": "data"}}
        )
        claims = extract_logistics(bundle, mock_llm, model="test")
        assert len(claims) == 0

    def test_care_provider_fact(self):
        event = _make_event(raw_content="Luci confirmed for Monday")
        bundle = ThreadBundle("t1", [event], [{"score": 0.8}])
        mock_llm = MockLLMClient(
            json_responses={"Luci": {"facts": [{
                "type": "care_provider",
                "provider": "Luciana",
                "date": "2026-02-17",
                "hours": "9am-3pm",
                "rate": "$150",
            }]}}
        )
        claims = extract_logistics(bundle, mock_llm, model="test")
        assert len(claims) == 1
        assert claims[0].predicate == "logistics_care_provider"


class TestExtractRelational:
    def test_extracts_people(self):
        event = _make_event(raw_content="Kai is co-founder of Example Corp")
        bundle = ThreadBundle("t1", [event], [{"score": 0.8}])
        mock_llm = MockLLMClient(
            json_responses={"Kai": MOCK_RELATIONAL_EXTRACTION_RESPONSE}
        )
        claims = extract_relational(bundle, mock_llm, model="test")
        assert len(claims) == 1
        assert claims[0].claim_type == "relational_context"
        assert claims[0].predicate == "person_context:Kai Tanaka"
        obj = json.loads(claims[0].object)
        assert obj["role"] == "co-founder"

    def test_empty_people(self):
        event = _make_event(raw_content="Hello world")
        bundle = ThreadBundle("t1", [event], [{"score": 0.8}])
        mock_llm = MockLLMClient(
            json_responses={"Hello": {"people": []}}
        )
        claims = extract_relational(bundle, mock_llm, model="test")
        assert len(claims) == 0

    def test_no_people_key(self):
        event = _make_event(raw_content="Hello world")
        bundle = ThreadBundle("t1", [event], [{"score": 0.8}])
        mock_llm = MockLLMClient(
            json_responses={"Hello": {"other": "data"}}
        )
        claims = extract_relational(bundle, mock_llm, model="test")
        assert len(claims) == 0

    def test_skips_empty_names(self):
        event = _make_event(raw_content="Someone said something")
        bundle = ThreadBundle("t1", [event], [{"score": 0.8}])
        mock_llm = MockLLMClient(
            json_responses={"Someone": {"people": [
                {"name": "", "role": "unknown"},
                {"name": "Valid Person", "role": "colleague"},
            ]}}
        )
        claims = extract_relational(bundle, mock_llm, model="test")
        assert len(claims) == 1
        obj = json.loads(claims[0].object)
        assert obj["name"] == "Valid Person"


class TestMultiGateRunSynthesis:
    """Test run_synthesis with all 3 gate types populated."""

    def _seed_with_all_gates(self, store):
        """Seed store with events and all 3 gate claim types."""
        from alteris.extract import (
            GateResult,
            LogisticsGateResult,
            RelationalGateResult,
            _build_gate_claim,
            _build_logistics_gate_claim,
            _build_relational_gate_claim,
        )

        event = _make_event(
            raw_content="Please send the updated proposal by Friday. Also Luci at 9am.",
            metadata={"subject": "Work", "thread_id": "thread_1", "is_from_me": False},
        )
        store.put_event(event)

        # Triage claim (required for bundle reconstruction)
        store.put_claim(_make_claim(
            "triage_1", "triage", event.id, "triage_result",
            {"score": 0.9, "reason": "test", "pii": []},
        ))

        # Actionable gate claim
        store.put_claim(_build_gate_claim(
            GateResult(actionable=True, reason="User commitment"),
            "thread_1", [event.id], "test",
        ))

        # Logistics gate claim
        store.put_claim(_build_logistics_gate_claim(
            LogisticsGateResult(logistics=True, reason="Care scheduling"),
            "thread_1", [event.id], "test",
        ))

        # Relational gate claim
        store.put_claim(_build_relational_gate_claim(
            RelationalGateResult(relational=True, reason="Co-founder context"),
            "thread_1", [event.id], "test",
        ))

        # Seed person for entity beliefs
        store.put_person("p_user", canonical_name="Test User", is_user=True)

    def test_processes_all_three_gate_types(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_with_all_gates(store)

        mock_llm = MockLLMClient(json_responses={
            "THREAD CONTENT": MOCK_SYNTHESIS_COMMITMENT_RESPONSE,
            "Sam": MOCK_LOGISTICS_EXTRACTION_RESPONSE,
        })
        result = run_synthesis(store, mock_llm)

        assert result["actionable_threads"] == 1
        assert result["logistics_threads"] == 1
        assert result["relational_threads"] == 1
        assert result["total_commitments"] >= 1
        # Logistics and relational counts depend on mock responses matching

    def test_deduplicates_bundles_across_gates(self):
        """Thread passing all 3 gates should only be reconstructed once."""
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_with_all_gates(store)

        mock_llm = MockLLMClient(json_responses={
            "THREAD CONTENT": MOCK_SYNTHESIS_COMMITMENT_RESPONSE,
        })
        result = run_synthesis(store, mock_llm)

        # Only 1 unique bundle should be reconstructed
        assert result["bundles_processed"] == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# cluster_by_similarity (reusable fuzzy clustering)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestClusterBySimilarity:
    def test_empty_input(self):
        clusters, logs = cluster_by_similarity([], key_fn=lambda x: x)
        assert clusters == []
        assert logs == []

    def test_single_item(self):
        clusters, logs = cluster_by_similarity(["hello"], key_fn=lambda x: x)
        assert clusters == [["hello"]]
        assert logs == [[]]

    def test_identical_items_cluster_together(self):
        items = ["send the proposal", "send the proposal"]
        clusters, _ = cluster_by_similarity(items, key_fn=lambda x: x)
        assert len(clusters) == 1
        assert len(clusters[0]) == 2

    def test_similar_items_cluster_together(self):
        items = [
            "send the updated proposal to Sam",
            "send the updated proposal to Sam by Friday",
        ]
        clusters, _ = cluster_by_similarity(items, key_fn=lambda x: x)
        assert len(clusters) == 1

    def test_dissimilar_items_separate(self):
        items = [
            "send the updated proposal",
            "review the contract draft",
            "schedule a meeting with Alice",
        ]
        clusters, _ = cluster_by_similarity(items, key_fn=lambda x: x)
        assert len(clusters) == 3

    def test_prefix_match_triggers_merge(self):
        """Items with same first 25 chars should merge even if tails differ."""
        items = [
            "send the updated proposal draft to Sam",
            "send the updated proposal draft to everyone",
        ]
        # First 25 chars: "send the updated proposal" — identical
        clusters, _ = cluster_by_similarity(items, key_fn=lambda x: x)
        assert len(clusters) == 1

    def test_custom_min_votes(self):
        items = ["send proposal", "send the proposal document"]
        # Require all 3 experts to agree: should NOT merge
        clusters_strict, _ = cluster_by_similarity(
            items, key_fn=lambda x: x, min_votes=3,
        )
        assert len(clusters_strict) == 2

        # Only require 1 vote: should merge (SeqMatch alone suffices)
        clusters_loose, _ = cluster_by_similarity(
            items, key_fn=lambda x: x, min_votes=1,
        )
        assert len(clusters_loose) == 1

    def test_dict_items_with_key_fn(self):
        items = [
            {"id": "a", "text": "send the updated proposal"},
            {"id": "b", "text": "send the updated proposal draft"},
            {"id": "c", "text": "review the contract"},
        ]
        clusters, _ = cluster_by_similarity(items, key_fn=lambda x: x["text"])
        assert len(clusters) == 2

    def test_transitive_clustering(self):
        """A~B and B~C should put all three in one cluster via union-find."""
        items = [
            "review the quarterly budget report for Q1",      # A
            "review the quarterly budget report for Q2",      # B (similar to A)
            "review the quarterly budget report for Q2 draft", # C (similar to B)
        ]
        # A~B share prefix, B~C share prefix → all merge transitively
        clusters, _ = cluster_by_similarity(items, key_fn=lambda x: x)
        assert len(clusters) == 1

    def test_seqmatch_catches_transcription_errors(self):
        """SequenceMatcher merges near-duplicates with misspellings."""
        items = [
            "package Altarus listener as installable app",
            "package Alteris listener as installable app",
        ]
        clusters, logs = cluster_by_similarity(items, key_fn=lambda x: x)
        assert len(clusters) == 1
        # Merge log should record which experts voted
        assert len(logs[0]) == 1
        experts = logs[0][0]["experts"]
        assert any("seqmatch" in e for e in experts)

    def test_stop_word_removal_improves_merge(self):
        """Removing stop words lets content-word Jaccard bridge variations."""
        items = [
            "leave for church by 9",
            "leave for church by 9:00 AM",
        ]
        clusters, _ = cluster_by_similarity(items, key_fn=lambda x: x)
        assert len(clusters) == 1

    def test_dissimilar_not_merged_by_seqmatch(self):
        """Completely different items shouldn't merge via any signal."""
        items = [
            "send CV to Elliott",
            "send architecture doc to Jordan",
        ]
        clusters, _ = cluster_by_similarity(items, key_fn=lambda x: x)
        assert len(clusters) == 2

    def test_before_vs_after_not_in_stop_words(self):
        """Temporal words like 'before'/'after' are NOT in STOP_WORDS."""
        from alteris.beliefs import STOP_WORDS
        assert "before" not in STOP_WORDS
        assert "after" not in STOP_WORDS
        assert "until" not in STOP_WORDS
        assert "during" not in STOP_WORDS

    def test_merge_log_records_expert_votes(self):
        """Merge log records which experts voted and their scores."""
        items = [
            "send the updated proposal draft to Sam",
            "send the updated proposal draft to Sam by Friday",
        ]
        clusters, logs = cluster_by_similarity(items, key_fn=lambda x: x)
        assert len(clusters) == 1
        assert len(logs) == 1
        merge_log = logs[0]
        assert len(merge_log) >= 1
        entry = merge_log[0]
        assert "experts" in entry
        assert "votes" in entry
        assert entry["votes"] >= 2
        assert "merged" in entry


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _build_commitment_beliefs (commitment claims -> FACT beliefs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBuildCommitmentBeliefs:
    @staticmethod
    def _put_commitment(
        store, claim_id, thread_id, what,
        who="user", deadline=None, confidence=0.8,
        priority=2, commitment_type="inbound_request",
    ):
        obj = {
            "type": commitment_type,
            "who": who,
            "what": what,
            "deadline": deadline,
            "status": "open",
            "priority": priority,
            "confidence": confidence,
            "direction": "direct_ask",
        }
        claim = _make_claim(
            claim_id, "commitment", thread_id,
            commitment_type, obj, confidence=confidence,
        )
        store.put_claim(claim)

    def test_single_claim_becomes_single_belief(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(
            store, "c1", "thread_a",
            "send the updated proposal", deadline="2026-02-20",
        )

        beliefs = _build_commitment_beliefs(store)
        assert len(beliefs) == 1
        b = beliefs[0]
        assert b.belief_type == BeliefType.FACT
        assert b.data["assertion_type"] == "commitment"
        assert b.data["what"] == "send the updated proposal"
        assert b.data["deadline"] == "2026-02-20"
        assert len(b.source_claims) == 1
        assert b.source_claims[0] == "c1"
        assert b.confidence == 0.8
        assert b.status == BeliefStatus.ACTIVE
        assert b.expires_at is not None  # has deadline → has expiry

    def test_similar_claims_merge_into_one_belief(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(
            store, "c1", "thread_a",
            "send the updated proposal to Sam", confidence=0.9,
        )
        self._put_commitment(
            store, "c2", "thread_b",
            "send the updated proposal to Sam by Friday", confidence=0.7,
        )

        beliefs = _build_commitment_beliefs(store)
        assert len(beliefs) == 1
        b = beliefs[0]
        assert len(b.source_claims) == 2
        assert "c1" in b.source_claims
        assert "c2" in b.source_claims
        assert b.confidence == 0.9  # max of merged claims
        assert b.data["claim_count"] == 2
        assert "merged_whats" in b.data

    def test_dissimilar_claims_stay_separate(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(
            store, "c1", "thread_a",
            "send the updated proposal",
        )
        self._put_commitment(
            store, "c2", "thread_b",
            "review the contract draft",
        )

        beliefs = _build_commitment_beliefs(store)
        assert len(beliefs) == 2

    def test_different_who_not_merged(self):
        """Same 'what' but different 'who' should NOT merge."""
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(
            store, "c1", "thread_a",
            "schedule the meeting", who="user",
        )
        self._put_commitment(
            store, "c2", "thread_b",
            "schedule the meeting", who="Alice",
        )

        beliefs = _build_commitment_beliefs(store)
        assert len(beliefs) == 2

    def test_different_deadline_not_merged(self):
        """Same 'what' and 'who' but different deadlines → separate beliefs."""
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(
            store, "c1", "thread_a",
            "send the report", deadline="2026-02-15",
        )
        self._put_commitment(
            store, "c2", "thread_b",
            "send the report", deadline="2026-03-01",
        )

        beliefs = _build_commitment_beliefs(store)
        assert len(beliefs) == 2

    def test_no_deadline_expiry_is_none(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(
            store, "c1", "thread_a",
            "look into the budget numbers",
        )

        beliefs = _build_commitment_beliefs(store)
        assert len(beliefs) == 1
        assert beliefs[0].expires_at is None

    def test_priority_takes_minimum(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(
            store, "c1", "thread_a",
            "send the updated proposal", priority=3, confidence=0.9,
        )
        self._put_commitment(
            store, "c2", "thread_b",
            "send the updated proposal", priority=1, confidence=0.7,
        )

        beliefs = _build_commitment_beliefs(store)
        assert len(beliefs) == 1
        assert beliefs[0].priority == 1  # most urgent

    def test_superseded_claims_excluded(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(
            store, "c1", "thread_a",
            "send the updated proposal",
        )
        # Supersede c1
        store.supersede_claim("c1", "c_new")

        beliefs = _build_commitment_beliefs(store)
        assert len(beliefs) == 0

    def test_empty_store_returns_empty(self):
        store = LayeredGraphStore(db_path=":memory:")
        beliefs = _build_commitment_beliefs(store)
        assert beliefs == []

    def test_belief_id_deterministic(self):
        """Same claims should produce same belief ID on re-run."""
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(
            store, "c1", "thread_a",
            "send the updated proposal",
        )

        beliefs_1 = _build_commitment_beliefs(store)
        beliefs_2 = _build_commitment_beliefs(store)
        assert beliefs_1[0].id == beliefs_2[0].id

    def test_inference_chain_records_merge(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(
            store, "c1", "thread_a",
            "clean out the carseats", confidence=0.9,
        )
        self._put_commitment(
            store, "c2", "thread_b",
            "clean out the carseats before trip", confidence=0.7,
        )
        self._put_commitment(
            store, "c3", "thread_c",
            "clean out the carseats this weekend", confidence=0.6,
        )

        beliefs = _build_commitment_beliefs(store)
        assert len(beliefs) == 1
        b = beliefs[0]
        assert len(b.source_claims) == 3
        chain = " ".join(b.inference_chain)
        assert "3 commitment claim(s)" in chain
        assert "Merged" in chain

    def test_evidence_log_tracks_compilation(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(
            store, "c1", "thread_a",
            "follow up with vendor",
        )

        beliefs = _build_commitment_beliefs(store)
        assert len(beliefs[0].evidence_log) == 1
        entry = beliefs[0].evidence_log[0]
        assert entry["event"] == "fact_compiled"
        assert entry["sources"] == 1

    def test_subject_includes_who(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_commitment(
            store, "c1", "thread_a",
            "send the invoice", who="Alice",
        )

        beliefs = _build_commitment_beliefs(store)
        assert beliefs[0].subject == "commitment:Alice"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _build_logistics_beliefs (logistics claims -> FACT beliefs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestBuildLogisticsBeliefs:
    @staticmethod
    def _put_logistics(
        store, claim_id, thread_id, obj, confidence=0.9,
    ):
        claim = _make_claim(
            claim_id, "logistics", thread_id,
            f"logistics_{obj.get('type', 'unknown')}", obj,
            confidence=confidence,
        )
        store.put_claim(claim)

    def test_reservation_becomes_fact_belief(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_logistics(store, "l1", "thread_a", {
            "type": "reservation",
            "venue": "The Garden Table",
            "date": "2026-02-14",
            "time": "7:30 PM",
            "party_size": 4,
        })

        beliefs = _build_logistics_beliefs(store)
        assert len(beliefs) == 1
        b = beliefs[0]
        assert b.belief_type == BeliefType.FACT
        assert b.data["assertion_type"] == "logistics_reservation"
        assert b.data["venue"] == "The Garden Table"
        assert "Reservation" in b.summary
        assert b.expires_at is not None

    def test_travel_becomes_fact_belief(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_logistics(store, "l1", "thread_a", {
            "type": "travel",
            "destination": "New York",
            "date": "2026-03-01",
        })

        beliefs = _build_logistics_beliefs(store)
        assert len(beliefs) == 1
        assert "Travel" in beliefs[0].summary
        assert beliefs[0].data["assertion_type"] == "logistics_travel"

    def test_duplicate_logistics_merged(self):
        """Same reservation discussed in two threads → one belief."""
        store = LayeredGraphStore(db_path=":memory:")
        self._put_logistics(store, "l1", "thread_a", {
            "type": "reservation",
            "venue": "The Garden Table",
            "date": "2026-02-14",
            "time": "7:30 PM",
        }, confidence=0.9)
        self._put_logistics(store, "l2", "thread_b", {
            "type": "reservation",
            "venue": "The Garden Table",
            "date": "2026-02-14",
            "time": "7:30 PM",
        }, confidence=0.8)

        beliefs = _build_logistics_beliefs(store)
        assert len(beliefs) == 1
        assert len(beliefs[0].source_claims) == 2
        assert beliefs[0].confidence == 0.9

    def test_different_logistics_stay_separate(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._put_logistics(store, "l1", "thread_a", {
            "type": "reservation",
            "venue": "The Garden Table",
            "date": "2026-02-14",
        })
        self._put_logistics(store, "l2", "thread_b", {
            "type": "travel",
            "destination": "New York",
            "date": "2026-03-01",
        })

        beliefs = _build_logistics_beliefs(store)
        assert len(beliefs) == 2

    def test_empty_store_returns_empty(self):
        store = LayeredGraphStore(db_path=":memory:")
        beliefs = _build_logistics_beliefs(store)
        assert beliefs == []

    def test_logistics_expiry_from_date(self):
        obj = {"type": "reservation", "date": "2026-02-14"}
        expiry = _logistics_expiry(obj)
        assert expiry is not None
        # Should be ~3 days after Feb 14
        from datetime import datetime, timezone
        expected = int(
            datetime(2026, 2, 14, tzinfo=timezone.utc).timestamp()
        ) + 3 * SECONDS_PER_DAY
        assert expiry == expected

    def test_logistics_expiry_no_date(self):
        assert _logistics_expiry({"type": "care_provider"}) is None

    def test_logistics_summary_all_types(self):
        assert "Garden" in _logistics_summary(
            {"venue": "Garden Table", "date": "2/14", "time": "7pm"},
            "reservation",
        )
        assert "New York" in _logistics_summary(
            {"destination": "New York", "date": "3/1"},
            "travel",
        )
        assert "Dr. Smith" in _logistics_summary(
            {"provider": "Dr. Smith", "date": "3/1"},
            "care_provider",
        )
        assert "Appt" in _logistics_summary(
            {"provider": "Dr. Smith", "date": "3/1"},
            "appointment",
        )
        assert "Soccer" in _logistics_summary(
            {"name": "Soccer camp", "dates": "6/1-6/5"},
            "activity",
        )
        assert "Kai" in _logistics_summary(
            {"child": "Kai", "facility": "ABC Daycare", "date": "3/1"},
            "childcare",
        )
        assert "fallback" in _logistics_summary(
            {"summary": "fallback text"},
            "unknown_type",
        )
