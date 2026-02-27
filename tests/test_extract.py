"""Tests for alteris.extract: Binary actionable gate (Stage 6).

Tests cover:
  - GateResult Pydantic validation (actionable, reason, coercion)
  - _is_sensitive detection
  - group_into_threads (thread grouping + iMessage synthetic threads)
  - format_event_as_markdown (email, message, calendar, meeting)
  - _gate_claim_id determinism
  - _build_gate_claim construction (actionable vs not_actionable)
  - _parse_gate_result (valid, none, missing fields)
  - run_gate (mock LLM interaction)
  - get_actionable_threads (downstream query)
  - Resume support (_is_thread_attempted, _record_extraction_run)
  - build_persons_cache
  - run_extraction full pipeline (empty, basic, resume, force, structure)
"""

import json
import time

import pytest

from alteris.extract import (
    GateResult,
    ThreadBundle,
    _build_gate_claim,
    _format_thread_for_llm,
    _gate_claim_id,
    _is_sensitive,
    _is_thread_attempted,
    _parse_gate_result,
    _record_extraction_run,
    build_persons_cache,
    format_event_as_markdown,
    get_actionable_threads,
    group_into_threads,
    run_extraction,
    run_gate,
)
from alteris.llm.mock import (
    MOCK_GATE_NOT_ACTIONABLE_RESPONSE,
    MOCK_GATE_RESPONSE,
    MockLLMClient,
)
from alteris.models import Claim, Event, ExtractionProvenance, Modality
from alteris.privacy import SensitivityLevel
from alteris.store import LayeredGraphStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_event(
    source_id: str = "e1",
    source: str = "mail",
    event_type: str = "email",
    content: str = "Test email body",
    thread_id: str = "thread_1",
    subject: str = "Test Subject",
    is_from_me: bool = False,
    timestamp: int = 0,
) -> Event:
    return Event(
        id=Event.make_id(source, source_id),
        source=source,
        source_id=source_id,
        event_type=event_type,
        timestamp=timestamp or int(time.time()),
        participants=("alice@test.com", "user@test.com"),
        raw_content=content,
        metadata={
            "subject": subject,
            "thread_id": thread_id,
            "is_from_me": is_from_me,
        },
    )


def _make_triage_claim(event_id: str, score: float = 0.8) -> Claim:
    import hashlib
    raw = f"triage:{event_id}:triage_result"
    cid = f"claim:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"
    return Claim(
        id=cid,
        event_ids=[event_id],
        claim_type="triage",
        subject=event_id,
        predicate="triage_result",
        object=json.dumps({"score": score, "reason": "test", "pii": []}),
        confidence=score,
        modality=Modality.OBSERVED,
        provenance=ExtractionProvenance(model_id="test"),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GateResult validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGateResult:
    def test_valid_actionable(self):
        r = GateResult(actionable=True, reason="Direct request")
        assert r.actionable is True
        assert r.reason == "Direct request"

    def test_valid_not_actionable(self):
        r = GateResult(actionable=False, reason="Newsletter")
        assert r.actionable is False
        assert r.reason == "Newsletter"

    def test_empty_reason(self):
        r = GateResult(actionable=True)
        assert r.actionable is True
        assert r.reason == ""

    def test_reason_truncated(self):
        r = GateResult(actionable=True, reason="x" * 300)
        assert len(r.reason) <= 200

    def test_reason_stripped(self):
        r = GateResult(actionable=True, reason="  spaces  ")
        assert r.reason == "spaces"

    def test_bool_coercion(self):
        r = GateResult(actionable=1, reason="test")
        assert r.actionable is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sensitivity detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSensitivity:
    def test_financial_is_sensitive(self):
        assert _is_sensitive({"pii": ["financial"]}) is True

    def test_medical_is_sensitive(self):
        assert _is_sensitive({"pii": ["medical"]}) is True

    def test_credentials_is_sensitive(self):
        assert _is_sensitive({"pii": ["credentials"]}) is True

    def test_legal_is_sensitive(self):
        assert _is_sensitive({"pii": ["legal"]}) is True

    def test_no_pii_not_sensitive(self):
        assert _is_sensitive({"pii": []}) is False

    def test_non_sensitive_pii(self):
        assert _is_sensitive({"pii": ["name", "email"]}) is False

    def test_missing_pii_key(self):
        assert _is_sensitive({}) is False

    def test_string_pii_not_sensitive(self):
        assert _is_sensitive({"pii": "financial"}) is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thread grouping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGroupIntoThreads:
    def test_groups_by_thread_id(self):
        store = LayeredGraphStore(db_path=":memory:")
        e1 = _make_event("e1", thread_id="t1")
        e2 = _make_event("e2", thread_id="t1")
        candidates = [
            {"event": e1, "triage": {"pii": []}},
            {"event": e2, "triage": {"pii": []}},
        ]
        threads, standalones = group_into_threads(candidates, store)
        assert len(threads) == 1
        assert len(threads[0].events) == 2

    def test_standalone_events(self):
        store = LayeredGraphStore(db_path=":memory:")
        e1 = _make_event("e1", thread_id="")
        candidates = [{"event": e1, "triage": {"pii": []}}]
        threads, standalones = group_into_threads(candidates, store)
        assert len(standalones) == 1

    def test_sensitive_thread_flagged(self):
        store = LayeredGraphStore(db_path=":memory:")
        e1 = _make_event("e1", thread_id="t1")
        candidates = [{"event": e1, "triage": {"pii": ["financial"]}}]
        threads, _ = group_into_threads(candidates, store)
        assert threads[0].sensitive is True

    def test_imessage_synthetic_thread(self):
        store = LayeredGraphStore(db_path=":memory:")
        e1 = Event(
            id=Event.make_id("imessage", "im1"),
            source="imessage", source_id="im1",
            event_type="message", timestamp=int(time.time()),
            participants=("+15550100002",),
            raw_content="Hey",
            metadata={},
        )
        candidates = [{"event": e1, "triage": {"pii": []}}]
        threads, standalones = group_into_threads(candidates, store)
        assert len(threads) == 1
        assert threads[0].thread_id.startswith("imessage:")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event formatting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFormatEvent:
    def test_email_format(self):
        event = _make_event("e1", event_type="email", subject="Important")
        md = format_event_as_markdown(event, index=1)
        assert "## Message 1" in md
        assert "**Subject:**" in md
        assert "Important" in md

    def test_message_inbound(self):
        event = _make_event(
            "m1", source="imessage", event_type="message",
            content="Hello", is_from_me=False,
        )
        md = format_event_as_markdown(event)
        assert "inbound" in md.lower()

    def test_message_outbound(self):
        event = _make_event(
            "m2", source="imessage", event_type="message",
            content="Reply", is_from_me=True,
        )
        md = format_event_as_markdown(event)
        assert "outbound" in md.lower()

    def test_no_content(self):
        event = _make_event("e1", content="")
        md = format_event_as_markdown(event)
        assert "(no content)" in md

    def test_body_section(self):
        event = _make_event("e1", content="Test body text")
        md = format_event_as_markdown(event)
        assert "### Body" in md
        assert "Test body text" in md


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gate claim ID
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGateClaimId:
    def test_deterministic(self):
        assert _gate_claim_id("t1") == _gate_claim_id("t1")

    def test_different(self):
        assert _gate_claim_id("t1") != _gate_claim_id("t2")

    def test_prefix(self):
        # The function returns a raw hex hash without a prefix
        cid = _gate_claim_id("t1")
        assert isinstance(cid, str)
        assert len(cid) == 16


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build gate claim
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBuildGateClaim:
    def test_actionable_claim(self):
        result = GateResult(actionable=True, reason="test")
        claim = _build_gate_claim(result, "t1", ["e1"], "test-model")
        assert claim.claim_type == "extraction_gate"
        assert claim.predicate == "actionable"
        assert claim.confidence == 1.0

    def test_not_actionable_claim(self):
        result = GateResult(actionable=False, reason="newsletter")
        claim = _build_gate_claim(result, "t1", ["e1"], "test-model")
        assert claim.predicate == "not_actionable"
        assert claim.confidence == 0.1

    def test_event_ids_passed(self):
        result = GateResult(actionable=True)
        claim = _build_gate_claim(result, "t1", ["e1", "e2"], "test-model")
        assert "e1" in claim.event_ids
        assert "e2" in claim.event_ids


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Parse gate result
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestParseGateResult:
    def test_valid_actionable(self):
        r = _parse_gate_result({"actionable": True, "reason": "test"})
        assert r.actionable is True
        assert r.reason == "test"

    def test_valid_not_actionable(self):
        r = _parse_gate_result({"actionable": False, "reason": "FYI"})
        assert r.actionable is False
        assert r.reason == "FYI"

    def test_none_input(self):
        r = _parse_gate_result(None)
        assert r.actionable is False

    def test_missing_fields(self):
        r = _parse_gate_result({"other": True})
        assert r.actionable is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run gate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunGate:
    def test_returns_actionable(self):
        e1 = _make_event("e1", content="Can you send the report?")
        bundle = ThreadBundle("t1", [e1], [{}])
        mock_llm = MockLLMClient(
            json_responses={"report": MOCK_GATE_RESPONSE},
        )
        result = run_gate(bundle, mock_llm)
        assert isinstance(result, GateResult)
        assert result.actionable is True

    def test_returns_not_actionable(self):
        e1 = _make_event("e1", content="Your weekly newsletter digest")
        bundle = ThreadBundle("t1", [e1], [{}])
        mock_llm = MockLLMClient(
            json_responses={"newsletter": MOCK_GATE_NOT_ACTIONABLE_RESPONSE},
        )
        result = run_gate(bundle, mock_llm)
        assert isinstance(result, GateResult)
        assert result.actionable is False

    def test_empty_thread(self):
        bundle = ThreadBundle("t1", [], [])
        mock_llm = MockLLMClient(
            json_responses={"empty": MOCK_GATE_NOT_ACTIONABLE_RESPONSE},
        )
        result = run_gate(bundle, mock_llm)
        assert isinstance(result, GateResult)

    def test_llm_called(self):
        e1 = _make_event("e1", content="Please review the contract")
        bundle = ThreadBundle("t1", [e1], [{}])
        mock_llm = MockLLMClient(
            json_responses={"contract": MOCK_GATE_RESPONSE},
        )
        run_gate(bundle, mock_llm)
        assert len(mock_llm.generate_calls) >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Get actionable threads
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetActionableThreads:
    def test_returns_actionable_threads(self):
        store = LayeredGraphStore(db_path=":memory:")
        gate_result = GateResult(actionable=True, reason="Direct request")
        claim = _build_gate_claim(gate_result, "thread_abc", ["e1", "e2"], "test")
        store.put_claim(claim)
        results = get_actionable_threads(store)
        assert len(results) == 1
        assert results[0]["thread_id"] == "thread_abc"
        assert results[0]["reason"] == "Direct request"

    def test_excludes_not_actionable(self):
        store = LayeredGraphStore(db_path=":memory:")
        gate_result = GateResult(actionable=False, reason="Newsletter")
        claim = _build_gate_claim(gate_result, "thread_news", ["e1"], "test")
        store.put_claim(claim)
        results = get_actionable_threads(store)
        assert len(results) == 0

    def test_empty_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        results = get_actionable_threads(store)
        assert results == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Resume support
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestResume:
    def test_not_attempted(self):
        store = LayeredGraphStore(db_path=":memory:")
        assert _is_thread_attempted(store, "t1") is False

    def test_record_and_check(self):
        store = LayeredGraphStore(db_path=":memory:")
        e1 = _make_event("e1")
        store.put_event(e1)
        bundle = ThreadBundle("t1", [e1], [{}])
        _record_extraction_run(store, bundle, "test", True, 500)
        assert _is_thread_attempted(store, "t1") is True

    def test_error_still_marks_attempted(self):
        store = LayeredGraphStore(db_path=":memory:")
        e1 = _make_event("e1")
        store.put_event(e1)
        bundle = ThreadBundle("t1", [e1], [{}])
        _record_extraction_run(
            store, bundle, "error", False, 100,
            status="error", error_msg="something broke",
        )
        assert _is_thread_attempted(store, "t1") is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Persons cache
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPersonsCache:
    def test_empty_bundles(self):
        store = LayeredGraphStore(db_path=":memory:")
        cache, email_to_name = build_persons_cache([], store)
        assert cache == {}

    def test_populates_from_event_persons(self):
        store = LayeredGraphStore(db_path=":memory:")
        store.put_person("p1", canonical_name="Alice")
        e = _make_event("e1")
        store.put_event(e)
        store.link_event_person(e.id, "p1", "sender")
        bundle = ThreadBundle("t1", [e], [{}])
        cache, email_to_name = build_persons_cache([bundle], store)
        assert e.id in cache
        assert cache[e.id][0]["name"] == "Alice"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunExtraction:
    def _seed_store(self, store, n_events=3):
        now = int(time.time())
        store.put_person("person_user", is_user=True)
        store.put_person("person_sender", canonical_name="Sender")

        for i in range(n_events):
            e = _make_event(
                f"ext_{i}", content=f"Please do thing {i}",
                thread_id="thread_ext", timestamp=now - i * 3600,
            )
            store.put_event(e)
            tc = _make_triage_claim(e.id, score=0.85)
            store.put_claim(tc)

    def test_empty_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        mock_llm = MockLLMClient()
        result = run_extraction(store, mock_llm)
        assert result["threads_processed"] == 0

    def test_basic_extraction(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_store(store)
        mock_llm = MockLLMClient()
        result = run_extraction(store, mock_llm)
        assert result["threads_processed"] >= 1

    def test_resume_skips_processed(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_store(store)
        mock_llm = MockLLMClient()
        run_extraction(store, mock_llm)
        result2 = run_extraction(store, mock_llm)
        assert result2["skipped"] >= 1

    def test_force_reprocesses(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_store(store)
        mock_llm = MockLLMClient()
        run_extraction(store, mock_llm)
        result2 = run_extraction(store, mock_llm, force=True)
        assert result2["skipped"] == 0

    def test_result_structure(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_store(store)
        mock_llm = MockLLMClient()
        result = run_extraction(store, mock_llm)
        assert "threads_processed" in result
        assert "errors" in result
        assert "skipped" in result
        assert "elapsed_seconds" in result

    def test_result_includes_multigate_counts(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_store(store)
        mock_llm = MockLLMClient()
        result = run_extraction(store, mock_llm)
        assert "logistics" in result
        assert "relational" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Multi-gate: Logistics + Relational
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from alteris.extract import (
    LogisticsGateResult,
    RelationalGateResult,
    _build_logistics_gate_claim,
    _build_relational_gate_claim,
    _logistics_gate_claim_id,
    _parse_logistics_gate_result,
    _parse_relational_gate_result,
    _relational_gate_claim_id,
    get_logistics_threads,
    get_relational_threads,
    run_logistics_gate,
    run_relational_gate,
)
from alteris.llm.mock import (
    MOCK_LOGISTICS_GATE_NOT_LOGISTICS_RESPONSE,
    MOCK_LOGISTICS_GATE_RESPONSE,
    MOCK_RELATIONAL_GATE_NOT_RELATIONAL_RESPONSE,
    MOCK_RELATIONAL_GATE_RESPONSE,
)


class TestLogisticsGateResult:
    def test_valid_logistics(self):
        r = LogisticsGateResult(logistics=True, reason="Reservation found")
        assert r.logistics is True
        assert r.reason == "Reservation found"

    def test_valid_not_logistics(self):
        r = LogisticsGateResult(logistics=False, reason="No scheduling")
        assert r.logistics is False

    def test_empty_reason(self):
        r = LogisticsGateResult(logistics=True)
        assert r.reason == ""

    def test_reason_truncated(self):
        r = LogisticsGateResult(logistics=True, reason="x" * 300)
        assert len(r.reason) <= 200


class TestRelationalGateResult:
    def test_valid_relational(self):
        r = RelationalGateResult(relational=True, reason="Person role found")
        assert r.relational is True
        assert r.reason == "Person role found"

    def test_valid_not_relational(self):
        r = RelationalGateResult(relational=False, reason="No person context")
        assert r.relational is False

    def test_empty_reason(self):
        r = RelationalGateResult(relational=True)
        assert r.reason == ""

    def test_reason_truncated(self):
        r = RelationalGateResult(relational=True, reason="x" * 300)
        assert len(r.reason) <= 200


class TestLogisticsGateClaimId:
    def test_deterministic(self):
        id1 = _logistics_gate_claim_id("t1")
        id2 = _logistics_gate_claim_id("t1")
        assert id1 == id2

    def test_different_threads(self):
        id1 = _logistics_gate_claim_id("t1")
        id2 = _logistics_gate_claim_id("t2")
        assert id1 != id2

    def test_differs_from_actionable_gate(self):
        id_logistics = _logistics_gate_claim_id("t1")
        id_actionable = _gate_claim_id("t1")
        assert id_logistics != id_actionable


class TestRelationalGateClaimId:
    def test_deterministic(self):
        id1 = _relational_gate_claim_id("t1")
        id2 = _relational_gate_claim_id("t1")
        assert id1 == id2

    def test_different_threads(self):
        id1 = _relational_gate_claim_id("t1")
        id2 = _relational_gate_claim_id("t2")
        assert id1 != id2


class TestBuildLogisticsGateClaim:
    def test_logistics_claim(self):
        result = LogisticsGateResult(logistics=True, reason="Reservation")
        claim = _build_logistics_gate_claim(result, "t1", ["e1"], "flash-lite")
        assert claim.claim_type == "logistics_gate"
        assert claim.predicate == "logistics"
        assert claim.confidence == 1.0
        obj = json.loads(claim.object)
        assert obj["logistics"] is True

    def test_not_logistics_claim(self):
        result = LogisticsGateResult(logistics=False, reason="No scheduling")
        claim = _build_logistics_gate_claim(result, "t1", ["e1"], "flash-lite")
        assert claim.predicate == "not_logistics"
        assert claim.confidence == 0.1


class TestBuildRelationalGateClaim:
    def test_relational_claim(self):
        result = RelationalGateResult(relational=True, reason="Person role")
        claim = _build_relational_gate_claim(result, "t1", ["e1"], "flash-lite")
        assert claim.claim_type == "relational_gate"
        assert claim.predicate == "relational"
        assert claim.confidence == 1.0
        obj = json.loads(claim.object)
        assert obj["relational"] is True

    def test_not_relational_claim(self):
        result = RelationalGateResult(relational=False, reason="No context")
        claim = _build_relational_gate_claim(result, "t1", ["e1"], "flash-lite")
        assert claim.predicate == "not_relational"
        assert claim.confidence == 0.1


class TestParseLogisticsGateResult:
    def test_valid(self):
        result = _parse_logistics_gate_result({"logistics": True, "reason": "test"})
        assert result.logistics is True

    def test_none_input(self):
        result = _parse_logistics_gate_result(None)
        assert result.logistics is False

    def test_string_coercion(self):
        result = _parse_logistics_gate_result({"logistics": "true", "reason": "test"})
        assert result.logistics is True


class TestParseRelationalGateResult:
    def test_valid(self):
        result = _parse_relational_gate_result({"relational": True, "reason": "test"})
        assert result.relational is True

    def test_none_input(self):
        result = _parse_relational_gate_result(None)
        assert result.relational is False

    def test_string_coercion(self):
        result = _parse_relational_gate_result({"relational": "true", "reason": "test"})
        assert result.relational is True


class TestRunLogisticsGate:
    def test_returns_result(self):
        event = _make_event(content="Reservation at Pink Salt, Feb 14, 4pm")
        bundle = ThreadBundle("t1", [event], [{"score": 0.8}])
        mock_llm = MockLLMClient(
            json_responses={"Pink Salt": MOCK_LOGISTICS_GATE_RESPONSE}
        )
        result = run_logistics_gate(bundle, mock_llm, model="test")
        assert isinstance(result, LogisticsGateResult)
        assert result.logistics is True


class TestRunRelationalGate:
    def test_returns_result(self):
        event = _make_event(content="Sam is co-founder of Alteris")
        bundle = ThreadBundle("t1", [event], [{"score": 0.8}])
        mock_llm = MockLLMClient(
            json_responses={"Sam": MOCK_RELATIONAL_GATE_RESPONSE}
        )
        result = run_relational_gate(bundle, mock_llm, model="test")
        assert isinstance(result, RelationalGateResult)
        assert result.relational is True


class TestGetLogisticsThreads:
    def test_empty_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        assert get_logistics_threads(store) == []

    def test_returns_logistics_threads(self):
        store = LayeredGraphStore(db_path=":memory:")
        event = _make_event()
        store.put_event(event)
        result = LogisticsGateResult(logistics=True, reason="Reservation")
        claim = _build_logistics_gate_claim(result, "t1", [event.id], "test")
        store.put_claim(claim)
        threads = get_logistics_threads(store)
        assert len(threads) == 1
        assert threads[0]["thread_id"] == "t1"

    def test_excludes_not_logistics(self):
        store = LayeredGraphStore(db_path=":memory:")
        event = _make_event()
        store.put_event(event)
        result = LogisticsGateResult(logistics=False, reason="No")
        claim = _build_logistics_gate_claim(result, "t1", [event.id], "test")
        store.put_claim(claim)
        assert get_logistics_threads(store) == []


class TestGetRelationalThreads:
    def test_empty_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        assert get_relational_threads(store) == []

    def test_returns_relational_threads(self):
        store = LayeredGraphStore(db_path=":memory:")
        event = _make_event()
        store.put_event(event)
        result = RelationalGateResult(relational=True, reason="Person role")
        claim = _build_relational_gate_claim(result, "t1", [event.id], "test")
        store.put_claim(claim)
        threads = get_relational_threads(store)
        assert len(threads) == 1
        assert threads[0]["thread_id"] == "t1"

    def test_excludes_not_relational(self):
        store = LayeredGraphStore(db_path=":memory:")
        event = _make_event()
        store.put_event(event)
        result = RelationalGateResult(relational=False, reason="No")
        claim = _build_relational_gate_claim(result, "t1", [event.id], "test")
        store.put_claim(claim)
        assert get_relational_threads(store) == []


class TestMultiGateRunExtraction:
    """Test that run_extraction stores all 3 gate types."""

    def _seed_store(self, store):
        event = _make_event(
            content="Reserve at Pink Salt. Sam is co-founder.",
        )
        store.put_event(event)
        store.put_claim(_make_triage_claim(event.id, 0.9))

    def test_stores_all_three_gate_claims(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_store(store)
        mock_llm = MockLLMClient()
        run_extraction(store, mock_llm)

        # Should have extraction_gate, logistics_gate, and relational_gate claims
        gate_types = store.conn.execute(
            "SELECT DISTINCT claim_type FROM claims WHERE claim_type LIKE '%gate'"
        ).fetchall()
        type_set = {r["claim_type"] for r in gate_types}
        assert "extraction_gate" in type_set
        assert "logistics_gate" in type_set
        assert "relational_gate" in type_set
