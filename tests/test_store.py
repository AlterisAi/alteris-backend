"""Tests for alteris.store: LayeredGraphStore backed by in-memory SQLite.

Tests cover:
  - Schema creation (tables, indexes)
  - Event CRUD: put, get, batch insert, idempotency, filters
  - Claim CRUD: put, get, active claims, supersession, confidence update
  - Claim-Event links: paper trail
  - Belief CRUD: put, get, upsert, supersession, status update
  - Person registry: put, identifiers, resolve
  - Event-Person links
  - Sync state tracking
  - Stats aggregation
  - Edge cases (empty store, missing records, duplicates)
"""

import json
import time

import pytest

from alteris.constants import DEFAULT_EVENT_QUERY_LIMIT, DEFAULT_QUERY_LIMIT
from alteris.models import (
    Annotation,
    Belief,
    BeliefStatus,
    BeliefType,
    Claim,
    EpistemicLevel,
    Event,
    ExtractionMethod,
    ExtractionProvenance,
    Modality,
    Projection,
)
from alteris.privacy import SensitivityLevel
from alteris.store import LayeredGraphStore


# Import helpers - pytest adds conftest.py's dir to path automatically
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from conftest import make_belief, make_claim, make_event


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Schema tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSchema:
    def test_tables_created(self, store):
        """All required tables exist after init."""
        tables = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        assert "events" in table_names
        assert "claims" in table_names
        assert "claim_events" in table_names
        assert "beliefs" in table_names
        assert "annotations" in table_names
        assert "projections" in table_names
        assert "persons" in table_names
        assert "person_identifiers" in table_names
        assert "event_persons" in table_names
        assert "sync_state" in table_names

    def test_wal_mode(self, tmp_path):
        """WAL mode is enabled for file-based databases."""
        db_path = tmp_path / "wal_test.db"
        s = LayeredGraphStore(db_path=db_path)
        row = s.conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"
        s.close()

    def test_foreign_keys_on(self, store):
        row = store.conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1

    def test_indexes_exist(self, store):
        indexes = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        idx_names = {r["name"] for r in indexes}
        assert "idx_events_source" in idx_names
        assert "idx_events_time" in idx_names
        assert "idx_claims_subject" in idx_names
        assert "idx_beliefs_confidence" in idx_names
        assert "idx_beliefs_subject" in idx_names
        assert "idx_annotations_facet" in idx_names
        assert "idx_annotations_event" in idx_names
        assert "idx_projections_lens_route" in idx_names
        assert "idx_projections_lens_score" in idx_names


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event CRUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEventCRUD:
    def test_put_event_returns_true_on_new(self, store):
        event = make_event(source_id="e1")
        assert store.put_event(event) is True

    def test_put_event_returns_false_on_duplicate(self, store):
        event = make_event(source_id="e1")
        store.put_event(event)
        assert store.put_event(event) is False

    def test_get_event_roundtrip(self, store):
        event = make_event(
            source="mail", source_id="rt1",
            raw_content="Hello world",
            participants=("a@b.com", "c@d.com"),
            metadata={"thread_id": "t1"},
        )
        store.put_event(event)
        got = store.get_event(event.id)
        assert got is not None
        assert got.id == event.id
        assert got.source == "mail"
        assert got.raw_content == "Hello world"
        assert got.participants == ("a@b.com", "c@d.com")
        assert got.metadata == {"thread_id": "t1"}

    def test_get_event_not_found(self, store):
        assert store.get_event("nonexistent") is None

    def test_put_events_batch(self, store):
        events = [make_event(source_id=f"batch_{i}") for i in range(10)]
        count = store.put_events_batch(events)
        assert count == 10

    def test_put_events_batch_idempotent(self, store):
        events = [make_event(source_id=f"idem_{i}") for i in range(5)]
        store.put_events_batch(events)
        count = store.put_events_batch(events)
        assert count == 0

    def test_put_events_batch_partial_new(self, store):
        e1 = make_event(source_id="partial_1")
        store.put_event(e1)
        e2 = make_event(source_id="partial_2")
        count = store.put_events_batch([e1, e2])
        assert count == 1

    def test_get_events_default(self, store):
        now = int(time.time())
        for i in range(5):
            store.put_event(make_event(source_id=f"list_{i}", timestamp=now - i * 100))
        events = store.get_events()
        assert len(events) == 5
        # Should be in descending timestamp order
        for i in range(len(events) - 1):
            assert events[i].timestamp >= events[i + 1].timestamp

    def test_get_events_since(self, store):
        now = int(time.time())
        store.put_event(make_event(source_id="old", timestamp=now - 10000))
        store.put_event(make_event(source_id="new", timestamp=now))
        events = store.get_events(since=now - 100)
        assert len(events) == 1
        assert events[0].source_id == "new"

    def test_get_events_until(self, store):
        now = int(time.time())
        store.put_event(make_event(source_id="old2", timestamp=now - 10000))
        store.put_event(make_event(source_id="new2", timestamp=now))
        events = store.get_events(until=now - 5000)
        assert len(events) == 1

    def test_get_events_by_source(self, store):
        now = int(time.time())
        store.put_event(make_event(source="mail", source_id="ms1", timestamp=now))
        store.put_event(make_event(source="imessage", source_id="ms2", timestamp=now))
        events = store.get_events(source="mail")
        assert len(events) == 1
        assert events[0].source == "mail"

    def test_get_events_by_type(self, store):
        now = int(time.time())
        store.put_event(make_event(source_id="type1", event_type="email", timestamp=now))
        store.put_event(make_event(source_id="type2", event_type="message", timestamp=now))
        events = store.get_events(event_type="email")
        assert len(events) == 1

    def test_get_events_limit(self, store):
        now = int(time.time())
        for i in range(20):
            store.put_event(make_event(source_id=f"lim_{i}", timestamp=now - i))
        events = store.get_events(limit=5)
        assert len(events) == 5

    def test_get_events_by_source_method(self, store):
        now = int(time.time())
        store.put_event(make_event(source="mail", source_id="bys1", timestamp=now))
        store.put_event(make_event(source="mail", source_id="bys2", timestamp=now - 1000))
        store.put_event(make_event(source="imessage", source_id="bys3", timestamp=now))
        events = store.get_events_by_source("mail")
        assert len(events) == 2
        assert all(e.source == "mail" for e in events)

    def test_count_events(self, store):
        store.put_event(make_event(source="mail", source_id="cnt1"))
        store.put_event(make_event(source="mail", source_id="cnt2"))
        store.put_event(make_event(source="imessage", source_id="cnt3"))
        assert store.count_events() == 3
        assert store.count_events(source="mail") == 2
        assert store.count_events(source="imessage") == 1
        assert store.count_events(source="slack") == 0

    def test_empty_store_count(self, store):
        assert store.count_events() == 0

    def test_sensitivity_roundtrip(self, store):
        event = make_event(source_id="sens", sensitivity=SensitivityLevel.CRITICAL)
        store.put_event(event)
        got = store.get_event(event.id)
        assert got.sensitivity == SensitivityLevel.CRITICAL

    def test_metadata_roundtrip(self, store):
        meta = {"thread_id": "t1", "is_from_me": True, "nested": {"key": "val"}}
        event = make_event(source_id="meta_rt", metadata=meta)
        store.put_event(event)
        got = store.get_event(event.id)
        assert got.metadata == meta


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Claim CRUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestClaimCRUD:
    def test_put_claim_new(self, store):
        claim = make_claim(claim_id="claim:new1")
        assert store.put_claim(claim) is True

    def test_put_claim_duplicate(self, store):
        claim = make_claim(claim_id="claim:dup1")
        store.put_claim(claim)
        assert store.put_claim(claim) is False

    def test_get_claim_roundtrip(self, store):
        event = make_event(source_id="ce1")
        store.put_event(event)
        claim = make_claim(
            claim_id="claim:rt1",
            event_ids=[event.id],
            claim_type="triage",
            subject="person_sam",
            predicate="triage_result",
            confidence=0.75,
            modality=Modality.ASSERTED,
        )
        store.put_claim(claim)
        got = store.get_claim("claim:rt1")
        assert got is not None
        assert got.id == "claim:rt1"
        assert got.claim_type == "triage"
        assert got.subject == "person_sam"
        assert got.confidence == 0.75
        assert got.modality == Modality.ASSERTED
        assert event.id in got.event_ids

    def test_get_claim_not_found(self, store):
        assert store.get_claim("nonexistent") is None

    def test_claim_event_links(self, store):
        e1 = make_event(source_id="link1")
        e2 = make_event(source_id="link2")
        store.put_event(e1)
        store.put_event(e2)
        claim = make_claim(claim_id="claim:links", event_ids=[e1.id, e2.id])
        store.put_claim(claim)
        got = store.get_claim("claim:links")
        assert set(got.event_ids) == {e1.id, e2.id}

    def test_get_events_for_claim(self, store):
        e1 = make_event(source_id="efc1")
        e2 = make_event(source_id="efc2")
        store.put_event(e1)
        store.put_event(e2)
        claim = make_claim(claim_id="claim:efc", event_ids=[e1.id, e2.id])
        store.put_claim(claim)
        events = store.get_events_for_claim("claim:efc")
        assert len(events) == 2
        assert {e.id for e in events} == {e1.id, e2.id}

    def test_get_claims_for_event(self, store):
        e1 = make_event(source_id="cfe1")
        store.put_event(e1)
        c1 = make_claim(claim_id="claim:cfe1", event_ids=[e1.id])
        c2 = make_claim(claim_id="claim:cfe2", event_ids=[e1.id])
        store.put_claim(c1)
        store.put_claim(c2)
        claims = store.get_claims_for_event(e1.id)
        assert len(claims) == 2

    def test_get_active_claims(self, store):
        c1 = make_claim(claim_id="claim:act1", confidence=0.8)
        c2 = make_claim(claim_id="claim:act2", confidence=0.6)
        store.put_claim(c1)
        store.put_claim(c2)
        store.supersede_claim("claim:act2", "claim:act1")
        active = store.get_active_claims()
        assert len(active) == 1
        assert active[0].id == "claim:act1"

    def test_get_active_claims_by_subject(self, store):
        c1 = make_claim(claim_id="claim:sub1", subject="person_a")
        c2 = make_claim(claim_id="claim:sub2", subject="person_b")
        store.put_claim(c1)
        store.put_claim(c2)
        result = store.get_active_claims(subject="person_a")
        assert len(result) == 1
        assert result[0].subject == "person_a"

    def test_get_active_claims_by_type(self, store):
        c1 = make_claim(claim_id="claim:type1", claim_type="triage")
        c2 = make_claim(claim_id="claim:type2", claim_type="commitment")
        store.put_claim(c1)
        store.put_claim(c2)
        result = store.get_active_claims(claim_type="triage")
        assert len(result) == 1
        assert result[0].claim_type == "triage"

    def test_get_active_claims_min_confidence(self, store):
        c1 = make_claim(claim_id="claim:conf1", confidence=0.9)
        c2 = make_claim(claim_id="claim:conf2", confidence=0.3)
        store.put_claim(c1)
        store.put_claim(c2)
        result = store.get_active_claims(min_confidence=0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_supersede_claim(self, store):
        c1 = make_claim(claim_id="claim:sup1")
        c2 = make_claim(claim_id="claim:sup2")
        store.put_claim(c1)
        store.put_claim(c2)
        store.supersede_claim("claim:sup1", "claim:sup2")
        got = store.get_claim("claim:sup1")
        assert got.superseded_by == "claim:sup2"

    def test_update_claim_confidence(self, store):
        claim = make_claim(claim_id="claim:upd_conf", confidence=0.5)
        store.put_claim(claim)
        store.update_claim_confidence("claim:upd_conf", 0.9)
        got = store.get_claim("claim:upd_conf")
        assert got.confidence == 0.9

    def test_claim_provenance_roundtrip(self, store):
        claim = make_claim(
            claim_id="claim:prov",
            extraction_method=ExtractionMethod.CLOUD_MODEL,
        )
        claim.provenance = ExtractionProvenance(
            model_id="gemini-flash",
            prompt_version="v3.0",
            context_hash="hash123",
            extraction_method=ExtractionMethod.CLOUD_MODEL,
        )
        store.put_claim(claim)
        got = store.get_claim("claim:prov")
        assert got.provenance.model_id == "gemini-flash"
        assert got.provenance.prompt_version == "v3.0"
        assert got.provenance.extraction_method == ExtractionMethod.CLOUD_MODEL

    def test_claim_user_verified_roundtrip(self, store):
        claim = make_claim(claim_id="claim:uv_true")
        claim.user_verified = True
        store.put_claim(claim)
        got = store.get_claim("claim:uv_true")
        assert got.user_verified is True

        claim2 = make_claim(claim_id="claim:uv_false")
        claim2.user_verified = False
        store.put_claim(claim2)
        got2 = store.get_claim("claim:uv_false")
        assert got2.user_verified is False

        claim3 = make_claim(claim_id="claim:uv_none")
        claim3.user_verified = None
        store.put_claim(claim3)
        got3 = store.get_claim("claim:uv_none")
        assert got3.user_verified is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Belief CRUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBeliefCRUD:
    def test_put_belief_new(self, store):
        belief = make_belief(belief_id="belief:new1")
        assert store.put_belief(belief) is True

    def test_get_belief_roundtrip(self, store):
        now = int(time.time())
        belief = Belief(
            id="belief:rt",
            belief_type=BeliefType.FACT,
            subject="user",
            summary="User has deadline on Feb 15",
            data={"deadline": "2026-02-15"},
            epistemic_level=EpistemicLevel.INFERENCE,
            source_claims=["claim:c1", "claim:c2"],
            source_reliability="B",
            info_credibility=3,
            confidence=0.85,
            inference_chain=["Step 1", "Step 2"],
            evidence_log=[{"timestamp": now, "event": "created", "delta": 0.85}],
            status=BeliefStatus.ACTIVE,
            priority=5,
            created_at=now,
            updated_at=now,
            expires_at=now + 86400,
        )
        store.put_belief(belief)
        got = store.get_belief("belief:rt")
        assert got is not None
        assert got.id == "belief:rt"
        assert got.belief_type == BeliefType.FACT
        assert got.subject == "user"
        assert got.summary == "User has deadline on Feb 15"
        assert got.data == {"deadline": "2026-02-15"}
        assert got.epistemic_level == EpistemicLevel.INFERENCE
        assert got.source_claims == ["claim:c1", "claim:c2"]
        assert got.source_reliability == "B"
        assert got.info_credibility == 3
        assert got.confidence == 0.85
        assert got.inference_chain == ["Step 1", "Step 2"]
        assert got.evidence_log[0]["event"] == "created"
        assert got.status == BeliefStatus.ACTIVE
        assert got.priority == 5
        assert got.expires_at == now + 86400

    def test_get_belief_not_found(self, store):
        assert store.get_belief("nonexistent") is None

    def test_belief_upsert(self, store):
        """Putting a belief with same ID should update it."""
        b1 = make_belief(belief_id="belief:upsert", summary="Version 1", confidence=0.5)
        store.put_belief(b1)
        b2 = make_belief(belief_id="belief:upsert", summary="Version 2", confidence=0.9)
        store.put_belief(b2)
        got = store.get_belief("belief:upsert")
        assert got.summary == "Version 2"
        assert got.confidence == 0.9

    def test_get_beliefs_default(self, store):
        b1 = make_belief(belief_id="belief:list1", confidence=0.9)
        b2 = make_belief(belief_id="belief:list2", confidence=0.7)
        store.put_belief(b1)
        store.put_belief(b2)
        beliefs = store.get_beliefs()
        assert len(beliefs) == 2
        # Should be ordered by confidence DESC
        assert beliefs[0].confidence >= beliefs[1].confidence

    def test_get_beliefs_by_subject(self, store):
        b1 = make_belief(belief_id="belief:sub1", subject="person_a")
        b2 = make_belief(belief_id="belief:sub2", subject="person_b")
        store.put_belief(b1)
        store.put_belief(b2)
        result = store.get_beliefs(subject="person_a")
        assert len(result) == 1
        assert result[0].subject == "person_a"

    def test_get_beliefs_by_type(self, store):
        b1 = make_belief(belief_id="belief:type1", belief_type=BeliefType.ENTITY)
        b2 = make_belief(belief_id="belief:type2", belief_type=BeliefType.FACT)
        store.put_belief(b1)
        store.put_belief(b2)
        result = store.get_beliefs(belief_type="entity")
        assert len(result) == 1
        assert result[0].belief_type == BeliefType.ENTITY

    def test_get_beliefs_by_status(self, store):
        b1 = make_belief(belief_id="belief:stat1", status=BeliefStatus.ACTIVE)
        b2 = make_belief(belief_id="belief:stat2", status=BeliefStatus.RESOLVED)
        store.put_belief(b1)
        store.put_belief(b2)
        active = store.get_beliefs(status="active")
        assert len(active) == 1
        resolved = store.get_beliefs(status="resolved")
        assert len(resolved) == 1

    def test_get_beliefs_min_confidence(self, store):
        b1 = make_belief(belief_id="belief:hc", confidence=0.9)
        b2 = make_belief(belief_id="belief:lc", confidence=0.3)
        store.put_belief(b1)
        store.put_belief(b2)
        result = store.get_beliefs(min_confidence=0.5)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_get_beliefs_by_claims(self, store):
        b1 = make_belief(
            belief_id="belief:bc1",
            source_claims=["claim:x", "claim:y"],
        )
        b2 = make_belief(
            belief_id="belief:bc2",
            source_claims=["claim:z"],
        )
        store.put_belief(b1)
        store.put_belief(b2)
        result = store.get_beliefs_by_claims(["claim:x"])
        assert len(result) == 1
        assert result[0].id == "belief:bc1"

    def test_get_beliefs_by_claims_empty(self, store):
        assert store.get_beliefs_by_claims([]) == []

    def test_update_belief_status(self, store):
        b = make_belief(belief_id="belief:stat_upd")
        store.put_belief(b)
        store.update_belief_status("belief:stat_upd", "stale")
        got = store.get_belief("belief:stat_upd")
        assert got.status == BeliefStatus.STALE

    def test_supersede_belief(self, store):
        b1 = make_belief(belief_id="belief:old")
        b2 = make_belief(belief_id="belief:new")
        store.put_belief(b1)
        store.put_belief(b2)
        store.supersede_belief("belief:old", "belief:new")
        old = store.get_belief("belief:old")
        new = store.get_belief("belief:new")
        assert old.superseded_by == "belief:new"
        assert old.status == BeliefStatus.SUPERSEDED
        assert new.supersedes == "belief:old"

    def test_belief_inference_chain_none(self, store):
        b = Belief(
            id="belief:noinf",
            belief_type=BeliefType.OBSERVATION,
            subject="test",
            summary="Observation without inference chain",
            data={},
            epistemic_level=EpistemicLevel.OBSERVATION,
            source_claims=[],
        )
        store.put_belief(b)
        got = store.get_belief("belief:noinf")
        assert got.inference_chain is None

    def test_belief_evidence_log_none(self, store):
        b = Belief(
            id="belief:noevlog",
            belief_type=BeliefType.OBSERVATION,
            subject="test",
            summary="No evidence log",
            data={},
            epistemic_level=EpistemicLevel.OBSERVATION,
            source_claims=[],
        )
        store.put_belief(b)
        got = store.get_belief("belief:noevlog")
        assert got.evidence_log is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Person registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPersonRegistry:
    def test_put_person(self, store):
        store.put_person("p1", canonical_name="Alice", sources=["mail"])
        person = store.get_person("p1")
        assert person is not None
        assert person["canonical_name"] == "Alice"

    def test_put_person_upsert(self, store):
        store.put_person("p1", canonical_name="Alice")
        store.put_person("p1", canonical_name="Alice B.")
        person = store.get_person("p1")
        assert person["canonical_name"] == "Alice B."

    def test_put_person_upsert_preserves_existing_name(self, store):
        """Upsert with empty name preserves existing."""
        store.put_person("p1", canonical_name="Alice")
        store.put_person("p1", canonical_name="")
        person = store.get_person("p1")
        assert person["canonical_name"] == "Alice"

    def test_put_person_is_user(self, store):
        store.put_person("p1", is_user=True)
        person = store.get_person("p1")
        assert person["is_user"] == 1

    def test_get_person_not_found(self, store):
        assert store.get_person("nonexistent") is None

    def test_get_all_persons(self, store):
        store.put_person("p1", canonical_name="Alice")
        store.put_person("p2", canonical_name="Bob")
        persons = store.get_all_persons()
        assert len(persons) == 2

    def test_add_person_identifier(self, store):
        store.put_person("p1")
        store.add_person_identifier("p1", "email", "a@b.com", display_name="Alice", source="mail")
        ids = store.get_person_identifiers("p1")
        assert len(ids) == 1
        assert ids[0]["identifier"] == "a@b.com"
        assert ids[0]["display_name"] == "Alice"

    def test_add_multiple_identifiers(self, store):
        store.put_person("p1")
        store.add_person_identifier("p1", "email", "a@b.com")
        store.add_person_identifier("p1", "phone", "+1111")
        store.add_person_identifier("p1", "email", "second@b.com")
        ids = store.get_person_identifiers("p1")
        assert len(ids) == 3

    def test_resolve_person_by_email(self, store):
        store.put_person("p1")
        store.add_person_identifier("p1", "email", "alice@example.com")
        assert store.resolve_person("email", "alice@example.com") == "p1"

    def test_resolve_person_by_phone(self, store):
        store.put_person("p1")
        store.add_person_identifier("p1", "phone", "+1234567890")
        assert store.resolve_person("phone", "+1234567890") == "p1"

    def test_resolve_person_not_found(self, store):
        assert store.resolve_person("email", "unknown@example.com") is None

    def test_identifier_replace_on_conflict(self, store):
        """Adding same identifier type+value again replaces the record."""
        store.put_person("p1")
        store.put_person("p2")
        store.add_person_identifier("p1", "email", "shared@example.com")
        store.add_person_identifier("p2", "email", "shared@example.com")
        assert store.resolve_person("email", "shared@example.com") == "p2"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event-Person links
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEventPersonLinks:
    def test_link_event_person(self, store):
        store.link_event_person("e1", "p1", "sender")
        pairs = store.get_persons_for_event("e1")
        assert len(pairs) == 1
        assert pairs[0] == ("p1", "sender")

    def test_link_idempotent(self, store):
        store.link_event_person("e1", "p1", "sender")
        store.link_event_person("e1", "p1", "sender")
        pairs = store.get_persons_for_event("e1")
        assert len(pairs) == 1

    def test_multiple_roles(self, store):
        store.link_event_person("e1", "p1", "sender")
        store.link_event_person("e1", "p1", "mentioned")
        pairs = store.get_persons_for_event("e1")
        assert len(pairs) == 2
        roles = {role for _, role in pairs}
        assert roles == {"sender", "mentioned"}

    def test_multiple_persons(self, store):
        store.link_event_person("e1", "p1", "sender")
        store.link_event_person("e1", "p2", "recipient")
        store.link_event_person("e1", "p3", "attendee")
        pairs = store.get_persons_for_event("e1")
        assert len(pairs) == 3

    def test_get_events_for_person(self, store):
        now = int(time.time())
        e1 = make_event(source_id="efp1", timestamp=now)
        e2 = make_event(source_id="efp2", timestamp=now - 100)
        store.put_event(e1)
        store.put_event(e2)
        store.link_event_person(e1.id, "p1", "sender")
        store.link_event_person(e2.id, "p1", "recipient")
        events = store.get_events_for_person("p1")
        assert len(events) == 2

    def test_get_events_for_person_by_role(self, store):
        now = int(time.time())
        e1 = make_event(source_id="efpr1", timestamp=now)
        e2 = make_event(source_id="efpr2", timestamp=now - 100)
        store.put_event(e1)
        store.put_event(e2)
        store.link_event_person(e1.id, "p1", "sender")
        store.link_event_person(e2.id, "p1", "recipient")
        sent = store.get_events_for_person("p1", role="sender")
        assert len(sent) == 1
        assert sent[0].id == e1.id

    def test_get_events_for_person_empty(self, store):
        events = store.get_events_for_person("nobody")
        assert events == []

    def test_get_persons_for_event_empty(self, store):
        pairs = store.get_persons_for_event("no_event")
        assert pairs == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sync state
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSyncState:
    def test_get_sync_state_empty(self, store):
        assert store.get_sync_state("mail") is None

    def test_update_and_get_sync_state(self, store):
        store.update_sync_state(
            source="mail",
            last_event_ts=1000,
            event_count=50,
            status="completed",
        )
        state = store.get_sync_state("mail")
        assert state is not None
        assert state["source"] == "mail"
        assert state["last_event_ts"] == 1000
        assert state["event_count"] == 50
        assert state["status"] == "completed"

    def test_sync_state_accumulates_count(self, store):
        store.update_sync_state(source="mail", event_count=50)
        store.update_sync_state(source="mail", event_count=30)
        state = store.get_sync_state("mail")
        assert state["event_count"] == 80

    def test_sync_state_max_timestamp(self, store):
        store.update_sync_state(source="mail", last_event_ts=1000)
        store.update_sync_state(source="mail", last_event_ts=500)
        state = store.get_sync_state("mail")
        assert state["last_event_ts"] == 1000

    def test_sync_state_cursor(self, store):
        store.update_sync_state(source="mail", cursor={"page": 5, "token": "abc"})
        state = store.get_sync_state("mail")
        cursor = json.loads(state["cursor"])
        assert cursor == {"page": 5, "token": "abc"}

    def test_sync_state_error(self, store):
        store.update_sync_state(source="mail", status="error", error_message="Permission denied")
        state = store.get_sync_state("mail")
        assert state["status"] == "error"
        assert state["error_message"] == "Permission denied"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Stats
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestStats:
    def test_empty_stats(self, store):
        s = store.stats()
        assert s["events_count"] == 0
        assert s["claims_count"] == 0
        assert s["beliefs_count"] == 0
        assert s["persons_count"] == 0
        assert s["annotations_count"] == 0
        assert s["projections_count"] == 0

    def test_populated_stats(self, populated_store):
        s = populated_store.stats()
        assert s["events_count"] == 6
        assert s["claims_count"] == 4
        assert s["beliefs_count"] == 4
        assert s["persons_count"] == 4
        assert "mail" in s["events_by_source"]
        assert s["events_by_source"]["mail"] == 2
        assert s["active_claims"] >= 1

    def test_events_by_type(self, populated_store):
        s = populated_store.stats()
        assert "email" in s["events_by_type"]

    def test_sync_state_in_stats(self, store):
        store.update_sync_state(source="mail", event_count=50)
        s = store.stats()
        assert "mail" in s["sync_state"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Store lifecycle
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestStoreLifecycle:
    def test_close_and_reopen(self, tmp_path):
        db_path = tmp_path / "lifecycle.db"
        s1 = LayeredGraphStore(db_path=db_path)
        e = make_event(source_id="life1")
        s1.put_event(e)
        s1.close()

        s2 = LayeredGraphStore(db_path=db_path)
        got = s2.get_event(e.id)
        assert got is not None
        assert got.id == e.id
        s2.close()

    def test_close_idempotent(self, store):
        store.close()
        store.close()

    def test_memory_db_no_file(self):
        s = LayeredGraphStore(db_path=":memory:")
        e = make_event(source_id="mem1")
        s.put_event(e)
        assert s.get_event(e.id) is not None
        s.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cross-layer paper trail
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPaperTrail:
    def test_belief_to_claims_to_events(self, populated_store):
        """Full provenance chain: Belief -> Claims -> Events."""
        belief = populated_store.get_belief("belief:fact_deadline")
        assert belief is not None
        assert len(belief.source_claims) > 0

        for cid in belief.source_claims:
            claim = populated_store.get_claim(cid)
            assert claim is not None

    def test_claim_to_events(self, populated_store):
        claim = populated_store.get_claim("claim:triage_email_001")
        assert claim is not None
        events = populated_store.get_events_for_claim(claim.id)
        assert len(events) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Annotation CRUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAnnotationCRUD:
    def test_put_annotation_new(self, store):
        ann = Annotation(event_id="ev1", facet="sender_domain", value="chase.com")
        assert store.put_annotation(ann) is True

    def test_put_annotation_duplicate(self, store):
        ann = Annotation(event_id="ev1", facet="sender_domain", value="chase.com")
        store.put_annotation(ann)
        assert store.put_annotation(ann) is False

    def test_get_annotations_by_event(self, store):
        store.put_annotation(Annotation(event_id="ev1", facet="sender_domain", value="chase.com"))
        store.put_annotation(Annotation(event_id="ev1", facet="is_automated", value="true"))
        store.put_annotation(Annotation(event_id="ev2", facet="sender_domain", value="gmail.com"))

        result = store.get_annotations(event_id="ev1")
        assert len(result) == 2

    def test_get_annotations_by_facet(self, store):
        store.put_annotation(Annotation(event_id="ev1", facet="sender_domain", value="chase.com"))
        store.put_annotation(Annotation(event_id="ev2", facet="sender_domain", value="gmail.com"))
        store.put_annotation(Annotation(event_id="ev3", facet="topic", value="finance"))

        result = store.get_annotations(facet="sender_domain")
        assert len(result) == 2

    def test_get_annotations_by_facet_and_value(self, store):
        store.put_annotation(Annotation(event_id="ev1", facet="sender_domain", value="chase.com"))
        store.put_annotation(Annotation(event_id="ev2", facet="sender_domain", value="chase.com"))
        store.put_annotation(Annotation(event_id="ev3", facet="sender_domain", value="gmail.com"))

        result = store.get_annotations(facet="sender_domain", value="chase.com")
        assert len(result) == 2

    def test_batch_insert(self, store):
        anns = [
            Annotation(event_id="ev1", facet="f1", value="v1"),
            Annotation(event_id="ev1", facet="f2", value="v2"),
            Annotation(event_id="ev2", facet="f1", value="v1"),
        ]
        inserted = store.put_annotations_batch(anns)
        assert inserted == 3

    def test_batch_insert_idempotent(self, store):
        anns = [Annotation(event_id="ev1", facet="f1", value="v1")]
        store.put_annotations_batch(anns)
        assert store.put_annotations_batch(anns) == 0

    def test_same_facet_different_sources(self, store):
        """Same facet/value from different sources creates distinct annotations."""
        store.put_annotation(Annotation(event_id="ev1", facet="life_domain", value="work",
                                        source="structural"))
        store.put_annotation(Annotation(event_id="ev1", facet="life_domain", value="work",
                                        source="triage_llm_v1"))
        result = store.get_annotations(event_id="ev1")
        assert len(result) == 2

    def test_multi_valued_facets(self, store):
        """An event can have multiple values for the same facet."""
        store.put_annotation(Annotation(event_id="ev1", facet="life_domain", value="work"))
        store.put_annotation(Annotation(event_id="ev1", facet="life_domain", value="financial"))
        result = store.get_annotations(event_id="ev1", facet="life_domain")
        values = {a.value for a in result}
        assert values == {"work", "financial"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Projection CRUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestProjectionCRUD:
    def test_put_projection(self, store):
        proj = Projection(event_id="ev1", lens="chief_of_staff",
                          score=0.45, route="full_triage",
                          components={"base": 0.3, "final": 0.45})
        assert store.put_projection(proj) is True

    def test_get_projection(self, store):
        proj = Projection(event_id="ev1", lens="chief_of_staff",
                          score=0.45, route="full_triage",
                          components={"base": 0.3})
        store.put_projection(proj)

        result = store.get_projection("ev1", "chief_of_staff")
        assert result is not None
        assert result.score == 0.45
        assert result.route == "full_triage"
        assert result.components["base"] == 0.3

    def test_get_projection_missing(self, store):
        assert store.get_projection("nonexistent", "chief_of_staff") is None

    def test_upsert_overwrites(self, store):
        """Re-inserting same (event_id, lens) updates the score."""
        store.put_projection(Projection(event_id="ev1", lens="chief_of_staff",
                                        score=0.3, route="low_priority"))
        store.put_projection(Projection(event_id="ev1", lens="chief_of_staff",
                                        score=0.6, route="full_triage"))
        result = store.get_projection("ev1", "chief_of_staff")
        assert result.score == 0.6
        assert result.route == "full_triage"

    def test_different_lenses_coexist(self, store):
        store.put_projection(Projection(event_id="ev1", lens="chief_of_staff",
                                        score=0.1, route="skip"))
        store.put_projection(Projection(event_id="ev1", lens="financial_audit",
                                        score=0.8, route="full_triage"))

        p1 = store.get_projection("ev1", "chief_of_staff")
        p2 = store.get_projection("ev1", "financial_audit")
        assert p1.score == 0.1
        assert p2.score == 0.8

    def test_get_projections_by_lens(self, store):
        store.put_projection(Projection(event_id="ev1", lens="chief_of_staff",
                                        score=0.3, route="low_priority"))
        store.put_projection(Projection(event_id="ev2", lens="chief_of_staff",
                                        score=0.5, route="full_triage"))
        store.put_projection(Projection(event_id="ev3", lens="financial_audit",
                                        score=0.9, route="full_triage"))

        result = store.get_projections(lens="chief_of_staff")
        assert len(result) == 2

    def test_get_projections_by_route(self, store):
        store.put_projection(Projection(event_id="ev1", lens="chief_of_staff",
                                        score=0.1, route="skip"))
        store.put_projection(Projection(event_id="ev2", lens="chief_of_staff",
                                        score=0.5, route="full_triage"))

        result = store.get_projections(lens="chief_of_staff", route="full_triage")
        assert len(result) == 1
        assert result[0].event_id == "ev2"

    def test_get_projections_min_score(self, store):
        store.put_projection(Projection(event_id="ev1", lens="chief_of_staff",
                                        score=0.1, route="skip"))
        store.put_projection(Projection(event_id="ev2", lens="chief_of_staff",
                                        score=0.5, route="full_triage"))

        result = store.get_projections(lens="chief_of_staff", min_score=0.4)
        assert len(result) == 1
        assert result[0].event_id == "ev2"

    def test_delete_projections_for_lens(self, store):
        store.put_projection(Projection(event_id="ev1", lens="chief_of_staff",
                                        score=0.3, route="low_priority"))
        store.put_projection(Projection(event_id="ev2", lens="chief_of_staff",
                                        score=0.5, route="full_triage"))
        store.put_projection(Projection(event_id="ev3", lens="financial_audit",
                                        score=0.9, route="full_triage"))

        deleted = store.delete_projections("chief_of_staff")
        assert deleted == 2

        # financial_audit still exists
        result = store.get_projections(lens="financial_audit")
        assert len(result) == 1

    def test_batch_insert(self, store):
        projs = [
            Projection(event_id="ev1", lens="chief_of_staff", score=0.3, route="low_priority"),
            Projection(event_id="ev2", lens="chief_of_staff", score=0.5, route="full_triage"),
        ]
        written = store.put_projections_batch(projs)
        assert written == 2

        result = store.get_projections(lens="chief_of_staff")
        assert len(result) == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Person Profiles CRUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPersonProfilesCRUD:
    def _make_profile(self, person_id="person_sam", **overrides):
        now = int(time.time())
        profile = {
            "person_id": person_id,
            "canonical_name": "Sam Park",
            "message_count": 100,
            "tier": 1,
            "user_initiated_ratio": 0.45,
            "channels": ["mail", "whatsapp"],
            "channel_count": 2,
            "days_since_last": 0.5,
            "first_contact_ts": now - 86400 * 30,
            "last_contact_ts": now - 3600,
            "relationship_span_days": 30.0,
            "is_user": 0,
            "computed_at": now,
        }
        profile.update(overrides)
        return profile

    def test_table_exists(self, store):
        tables = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='person_profiles'"
        ).fetchall()
        assert len(tables) == 1

    def test_indexes_exist(self, store):
        indexes = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        idx_names = {r["name"] for r in indexes}
        assert "idx_pp_tier" in idx_names
        assert "idx_pp_msg" in idx_names

    def test_upsert_single_profile(self, store):
        store.put_person("person_sam", canonical_name="Sam")
        profile = self._make_profile()
        changed = store.upsert_person_profile(profile)
        # First insert: no previous tier, so no tier change
        assert changed is False

        got = store.get_person_profile("person_sam")
        assert got is not None
        assert got["message_count"] == 100
        assert got["tier"] == 1
        assert got["channels"] == ["mail", "whatsapp"]
        assert got["channel_count"] == 2

    def test_upsert_detects_tier_change(self, store):
        store.put_person("person_bob", canonical_name="Bob")
        # First insert: tier 3
        p1 = self._make_profile(person_id="person_bob", message_count=8, tier=3)
        store.upsert_person_profile(p1)

        # Update: tier 2
        p2 = self._make_profile(person_id="person_bob", message_count=25, tier=2)
        changed = store.upsert_person_profile(p2)
        assert changed is True

        got = store.get_person_profile("person_bob")
        assert got["tier"] == 2
        assert got["previous_tier"] == 3
        assert got["message_count"] == 25
        assert got["previous_message_count"] == 8

    def test_upsert_no_change_same_tier(self, store):
        store.put_person("person_bob", canonical_name="Bob")
        p1 = self._make_profile(person_id="person_bob", message_count=100, tier=1)
        store.upsert_person_profile(p1)
        p2 = self._make_profile(person_id="person_bob", message_count=110, tier=1)
        changed = store.upsert_person_profile(p2)
        assert changed is False

    def test_batch_upsert(self, store):
        store.put_person("person_a", canonical_name="A")
        store.put_person("person_b", canonical_name="B")
        profiles = [
            self._make_profile(person_id="person_a", message_count=50, tier=1),
            self._make_profile(person_id="person_b", message_count=10, tier=3),
        ]
        count = store.upsert_person_profiles_batch(profiles)
        assert count == 2

        all_profiles = store.get_person_profiles()
        assert len(all_profiles) == 2

    def test_get_person_profile_not_found(self, store):
        assert store.get_person_profile("nonexistent") is None

    def test_get_person_profiles_min_messages(self, store):
        store.put_person("person_a", canonical_name="A")
        store.put_person("person_b", canonical_name="B")
        store.upsert_person_profile(
            self._make_profile(person_id="person_a", message_count=100, tier=1)
        )
        store.upsert_person_profile(
            self._make_profile(person_id="person_b", message_count=3, tier=4)
        )

        all_p = store.get_person_profiles()
        assert len(all_p) == 2

        filtered = store.get_person_profiles(min_messages=10)
        assert len(filtered) == 1
        assert filtered[0]["person_id"] == "person_a"

    def test_get_person_profiles_ordered_by_message_count(self, store):
        store.put_person("person_a", canonical_name="A")
        store.put_person("person_b", canonical_name="B")
        store.put_person("person_c", canonical_name="C")
        for pid, count in [("person_a", 10), ("person_b", 100), ("person_c", 50)]:
            store.upsert_person_profile(
                self._make_profile(person_id=pid, message_count=count)
            )

        profiles = store.get_person_profiles()
        counts = [p["message_count"] for p in profiles]
        assert counts == [100, 50, 10]

    def test_channels_json_roundtrip(self, store):
        store.put_person("person_a", canonical_name="A")
        store.upsert_person_profile(
            self._make_profile(
                person_id="person_a",
                channels=["mail", "whatsapp", "slack"],
                channel_count=3,
            )
        )
        got = store.get_person_profile("person_a")
        assert got["channels"] == ["mail", "whatsapp", "slack"]
        assert got["channel_count"] == 3

    def test_empty_profiles(self, store):
        profiles = store.get_person_profiles()
        assert profiles == []
