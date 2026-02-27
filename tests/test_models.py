"""Tests for alteris.models: Event, Claim, Belief dataclasses, enums, ID generation.

Tests cover:
  - Event ID determinism (same input -> same ID)
  - Content hash computation
  - Claim is_active logic
  - Belief defaults and field access
  - PersonIdentifiers merge and matches
  - Enum completeness and string values
  - Frozen/mutable behavior
"""

import json
import time

import pytest

from alteris.constants import CONTENT_HASH_PREFIX_LEN, HASH_PREFIX_LEN
from alteris.models import (
    Belief,
    BeliefStatus,
    BeliefType,
    Claim,
    EpistemicLevel,
    Event,
    ExtractionMethod,
    ExtractionProvenance,
    Modality,
    PersonIdentifiers,
)
from alteris.privacy import SensitivityLevel


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEventMakeId:
    def test_deterministic(self):
        """Same source + source_id always produces the same ID."""
        id1 = Event.make_id("mail", "email_001")
        id2 = Event.make_id("mail", "email_001")
        assert id1 == id2

    def test_different_source_different_id(self):
        """Different sources produce different IDs."""
        id_mail = Event.make_id("mail", "msg_001")
        id_imsg = Event.make_id("imessage", "msg_001")
        assert id_mail != id_imsg

    def test_different_source_id_different_id(self):
        """Different source_ids produce different IDs."""
        id1 = Event.make_id("mail", "email_001")
        id2 = Event.make_id("mail", "email_002")
        assert id1 != id2

    def test_id_length(self):
        """Event ID is a hex string of correct length."""
        eid = Event.make_id("mail", "test")
        assert len(eid) == HASH_PREFIX_LEN
        assert all(c in "0123456789abcdef" for c in eid)

    def test_id_is_sha256_prefix(self):
        """Event ID matches SHA-256 of 'source:source_id'."""
        import hashlib
        raw = "mail:email_001"
        expected = hashlib.sha256(raw.encode()).hexdigest()[:HASH_PREFIX_LEN]
        assert Event.make_id("mail", "email_001") == expected


class TestEventContentHash:
    def test_deterministic(self):
        """Same text produces the same hash."""
        h1 = Event.content_hash_of("Hello, world! This is a long enough test string.")
        h2 = Event.content_hash_of("Hello, world! This is a long enough test string.")
        assert h1 == h2

    def test_different_text_different_hash(self):
        h1 = Event.content_hash_of("Hello world, this is text A for hashing")
        h2 = Event.content_hash_of("Hello world, this is text B for hashing")
        assert h1 != h2

    def test_hash_length(self):
        h = Event.content_hash_of("this is a test string that is long enough")
        assert len(h) == CONTENT_HASH_PREFIX_LEN

    def test_unicode_handling(self):
        h = Event.content_hash_of("Caf\u00e9 na\u00efve r\u00e9sum\u00e9 with extra text here")
        assert len(h) == CONTENT_HASH_PREFIX_LEN

    def test_short_string_returns_empty(self):
        assert Event.content_hash_of("") == ""
        assert Event.content_hash_of("short") == ""
        assert Event.content_hash_of("x" * 19) == ""

    def test_min_length_boundary(self):
        assert Event.content_hash_of("x" * 20) != ""
        assert len(Event.content_hash_of("x" * 20)) == CONTENT_HASH_PREFIX_LEN


class TestEventCreation:
    def test_frozen_event(self):
        """Events are immutable (frozen dataclass)."""
        event = Event(
            id=Event.make_id("mail", "test"),
            source="mail",
            source_id="test",
            event_type="email",
            timestamp=int(time.time()),
        )
        with pytest.raises(AttributeError):
            event.source = "imessage"

    def test_default_values(self):
        event = Event(
            id="test_id",
            source="mail",
            source_id="test",
            event_type="email",
            timestamp=1000,
        )
        assert event.participants == ()
        assert event.raw_content is None
        assert event.content_hash == ""
        assert event.metadata == {}
        assert event.sensitivity == SensitivityLevel.SENSITIVE

    def test_with_all_fields(self):
        now = int(time.time())
        event = Event(
            id="full_id",
            source="imessage",
            source_id="msg_1",
            event_type="message",
            timestamp=now,
            participants=("user@example.com", "bob@example.com"),
            raw_content="Hello!",
            content_hash="abc123",
            metadata={"thread_id": "t1"},
            sensitivity=SensitivityLevel.CRITICAL,
            created_at=now,
        )
        assert event.participants == ("user@example.com", "bob@example.com")
        assert event.raw_content == "Hello!"
        assert event.sensitivity == SensitivityLevel.CRITICAL


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Claim tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestClaimIsActive:
    def test_active_by_default(self):
        claim = Claim(
            id="c1", event_ids=[], claim_type="test",
            subject="s", predicate="p", object="o",
        )
        assert claim.is_active is True

    def test_superseded_not_active(self):
        claim = Claim(
            id="c1", event_ids=[], claim_type="test",
            subject="s", predicate="p", object="o",
            superseded_by="c2",
        )
        assert claim.is_active is False

    def test_user_rejected_not_active(self):
        claim = Claim(
            id="c1", event_ids=[], claim_type="test",
            subject="s", predicate="p", object="o",
            user_verified=False,
        )
        assert claim.is_active is False

    def test_user_verified_is_active(self):
        claim = Claim(
            id="c1", event_ids=[], claim_type="test",
            subject="s", predicate="p", object="o",
            user_verified=True,
        )
        assert claim.is_active is True

    def test_user_verified_none_is_active(self):
        claim = Claim(
            id="c1", event_ids=[], claim_type="test",
            subject="s", predicate="p", object="o",
            user_verified=None,
        )
        assert claim.is_active is True


class TestClaimDefaults:
    def test_default_confidence(self):
        claim = Claim(
            id="c1", event_ids=[], claim_type="test",
            subject="s", predicate="p", object="o",
        )
        assert claim.confidence == 0.5

    def test_default_modality(self):
        claim = Claim(
            id="c1", event_ids=[], claim_type="test",
            subject="s", predicate="p", object="o",
        )
        assert claim.modality == Modality.UNKNOWN

    def test_default_provenance(self):
        claim = Claim(
            id="c1", event_ids=[], claim_type="test",
            subject="s", predicate="p", object="o",
        )
        assert claim.provenance.model_id == "deterministic"
        assert claim.provenance.extraction_method == ExtractionMethod.DETERMINISTIC

    def test_default_sensitivity(self):
        claim = Claim(
            id="c1", event_ids=[], claim_type="test",
            subject="s", predicate="p", object="o",
        )
        assert claim.sensitivity == SensitivityLevel.SENSITIVE

    def test_mutable(self):
        """Claims are mutable (can update confidence, superseded_by)."""
        claim = Claim(
            id="c1", event_ids=[], claim_type="test",
            subject="s", predicate="p", object="o",
        )
        claim.confidence = 0.9
        assert claim.confidence == 0.9
        claim.superseded_by = "c2"
        assert claim.superseded_by == "c2"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Belief tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBeliefDefaults:
    def test_default_confidence(self):
        b = Belief(
            id="b1", belief_type=BeliefType.FACT, subject="s",
            summary="test", data={}, epistemic_level=EpistemicLevel.INFERENCE,
            source_claims=[],
        )
        assert b.confidence == 0.5

    def test_default_status(self):
        b = Belief(
            id="b1", belief_type=BeliefType.FACT, subject="s",
            summary="test", data={}, epistemic_level=EpistemicLevel.INFERENCE,
            source_claims=[],
        )
        assert b.status == BeliefStatus.ACTIVE

    def test_default_source_reliability(self):
        b = Belief(
            id="b1", belief_type=BeliefType.FACT, subject="s",
            summary="test", data={}, epistemic_level=EpistemicLevel.INFERENCE,
            source_claims=[],
        )
        assert b.source_reliability == "C"

    def test_default_info_credibility(self):
        b = Belief(
            id="b1", belief_type=BeliefType.FACT, subject="s",
            summary="test", data={}, epistemic_level=EpistemicLevel.INFERENCE,
            source_claims=[],
        )
        assert b.info_credibility == 4

    def test_optional_fields_none(self):
        b = Belief(
            id="b1", belief_type=BeliefType.FACT, subject="s",
            summary="test", data={}, epistemic_level=EpistemicLevel.INFERENCE,
            source_claims=[],
        )
        assert b.inference_chain is None
        assert b.evidence_log is None
        assert b.supersedes is None
        assert b.superseded_by is None
        assert b.priority is None
        assert b.expires_at is None

    def test_mutable(self):
        b = Belief(
            id="b1", belief_type=BeliefType.FACT, subject="s",
            summary="test", data={}, epistemic_level=EpistemicLevel.INFERENCE,
            source_claims=[],
        )
        b.confidence = 0.95
        assert b.confidence == 0.95
        b.status = BeliefStatus.RESOLVED
        assert b.status == BeliefStatus.RESOLVED


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Extraction Provenance tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExtractionProvenance:
    def test_defaults(self):
        p = ExtractionProvenance()
        assert p.model_id == "deterministic"
        assert p.prompt_version == ""
        assert p.context_hash == ""
        assert p.extraction_method == ExtractionMethod.DETERMINISTIC

    def test_custom_values(self):
        p = ExtractionProvenance(
            model_id="qwen3:30b-a3b",
            prompt_version="v2.1",
            context_hash="abc123",
            extraction_method=ExtractionMethod.LOCAL_MODEL,
        )
        assert p.model_id == "qwen3:30b-a3b"
        assert p.extraction_method == ExtractionMethod.LOCAL_MODEL


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PersonIdentifiers tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPersonIdentifiers:
    def test_empty_person(self):
        p = PersonIdentifiers(person_id="p1")
        assert p.phones == set()
        assert p.emails == set()
        assert p.display_names == set()
        assert p.whatsapp_jids == set()
        assert p.slack_ids == {}
        assert p.linkedin_url is None
        assert p.canonical_name == ""
        assert p.is_user is False
        assert p.sources == set()

    def test_merge_phones_and_emails(self):
        p1 = PersonIdentifiers(
            person_id="p1",
            emails={"a@example.com"},
            phones={"+1111111"},
        )
        p2 = PersonIdentifiers(
            person_id="p2",
            emails={"b@example.com"},
            phones={"+2222222"},
        )
        p1.merge(p2)
        assert p1.emails == {"a@example.com", "b@example.com"}
        assert p1.phones == {"+1111111", "+2222222"}

    def test_merge_preserves_first_canonical_name(self):
        p1 = PersonIdentifiers(person_id="p1", canonical_name="Alice")
        p2 = PersonIdentifiers(person_id="p2", canonical_name="Bob")
        p1.merge(p2)
        assert p1.canonical_name == "Alice"

    def test_merge_takes_other_canonical_name_if_empty(self):
        p1 = PersonIdentifiers(person_id="p1", canonical_name="")
        p2 = PersonIdentifiers(person_id="p2", canonical_name="Bob")
        p1.merge(p2)
        assert p1.canonical_name == "Bob"

    def test_merge_linkedin_url(self):
        p1 = PersonIdentifiers(person_id="p1")
        p2 = PersonIdentifiers(person_id="p2", linkedin_url="https://linkedin.com/in/bob")
        p1.merge(p2)
        assert p1.linkedin_url == "https://linkedin.com/in/bob"

    def test_merge_preserves_existing_linkedin(self):
        p1 = PersonIdentifiers(person_id="p1", linkedin_url="https://linkedin.com/in/alice")
        p2 = PersonIdentifiers(person_id="p2", linkedin_url="https://linkedin.com/in/bob")
        p1.merge(p2)
        assert p1.linkedin_url == "https://linkedin.com/in/alice"

    def test_merge_display_names(self):
        p1 = PersonIdentifiers(person_id="p1", display_names={"Alice"})
        p2 = PersonIdentifiers(person_id="p2", display_names={"A", "Alice A."})
        p1.merge(p2)
        assert p1.display_names == {"Alice", "A", "Alice A."}

    def test_merge_whatsapp_jids(self):
        p1 = PersonIdentifiers(person_id="p1", whatsapp_jids={"111@s.whatsapp.net"})
        p2 = PersonIdentifiers(person_id="p2", whatsapp_jids={"222@s.whatsapp.net"})
        p1.merge(p2)
        assert "111@s.whatsapp.net" in p1.whatsapp_jids
        assert "222@s.whatsapp.net" in p1.whatsapp_jids

    def test_merge_slack_ids(self):
        p1 = PersonIdentifiers(person_id="p1", slack_ids={"T1": "U1"})
        p2 = PersonIdentifiers(person_id="p2", slack_ids={"T2": "U2"})
        p1.merge(p2)
        assert p1.slack_ids == {"T1": "U1", "T2": "U2"}

    def test_merge_sources(self):
        p1 = PersonIdentifiers(person_id="p1", sources={"mail"})
        p2 = PersonIdentifiers(person_id="p2", sources={"whatsapp", "mail"})
        p1.merge(p2)
        assert p1.sources == {"mail", "whatsapp"}

    def test_matches_by_email(self):
        p1 = PersonIdentifiers(person_id="p1", emails={"a@example.com"})
        p2 = PersonIdentifiers(person_id="p2", emails={"a@example.com", "b@example.com"})
        assert p1.matches(p2) is True

    def test_matches_by_phone(self):
        p1 = PersonIdentifiers(person_id="p1", phones={"+1111"})
        p2 = PersonIdentifiers(person_id="p2", phones={"+1111", "+2222"})
        assert p1.matches(p2) is True

    def test_no_match(self):
        p1 = PersonIdentifiers(person_id="p1", emails={"a@example.com"}, phones={"+1111"})
        p2 = PersonIdentifiers(person_id="p2", emails={"b@example.com"}, phones={"+2222"})
        assert p1.matches(p2) is False

    def test_empty_identifiers_no_match(self):
        p1 = PersonIdentifiers(person_id="p1")
        p2 = PersonIdentifiers(person_id="p2")
        assert p1.matches(p2) is False

    def test_matches_is_symmetric(self):
        p1 = PersonIdentifiers(person_id="p1", emails={"shared@example.com"})
        p2 = PersonIdentifiers(person_id="p2", emails={"shared@example.com"})
        assert p1.matches(p2) == p2.matches(p1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Enum tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEnums:
    def test_modality_values(self):
        assert Modality.ASSERTED == "asserted"
        assert Modality.PLANNED == "planned"
        assert Modality.HYPOTHETICAL == "hypothetical"
        assert Modality.QUOTED == "quoted"
        assert Modality.FICTIONAL == "fictional"
        assert Modality.OBSERVED == "observed"
        assert Modality.UNKNOWN == "unknown"

    def test_modality_count(self):
        assert len(Modality) == 7

    def test_extraction_method_values(self):
        assert ExtractionMethod.DETERMINISTIC == "deterministic"
        assert ExtractionMethod.HEURISTIC == "heuristic"
        assert ExtractionMethod.LOCAL_MODEL == "local_model"
        assert ExtractionMethod.CLOUD_MODEL == "cloud_model"
        assert ExtractionMethod.USER_INPUT == "user_input"

    def test_extraction_method_count(self):
        assert len(ExtractionMethod) == 5

    def test_belief_type_values(self):
        assert BeliefType.ENTITY == "entity"
        assert BeliefType.RELATION == "relation"
        assert BeliefType.FACT == "fact"
        assert BeliefType.OBSERVATION == "observation"

    def test_belief_type_count(self):
        assert len(BeliefType) == 4

    def test_epistemic_level_values(self):
        assert EpistemicLevel.OBSERVATION == "observation"
        assert EpistemicLevel.INFERENCE == "inference"
        assert EpistemicLevel.JUDGMENT == "judgment"
        assert EpistemicLevel.COMPUTED == "computed"

    def test_epistemic_level_count(self):
        assert len(EpistemicLevel) == 4

    def test_belief_status_values(self):
        assert BeliefStatus.ACTIVE == "active"
        assert BeliefStatus.RESOLVED == "resolved"
        assert BeliefStatus.STALE == "stale"
        assert BeliefStatus.RETRACTED == "retracted"
        assert BeliefStatus.SUPERSEDED == "superseded"

    def test_belief_status_count(self):
        assert len(BeliefStatus) == 5

    def test_modality_is_str(self):
        """Modality values should be usable as strings."""
        assert Modality.ASSERTED.value == "asserted"
        assert str(Modality.ASSERTED) == "Modality.ASSERTED"

    def test_extraction_method_is_str(self):
        assert ExtractionMethod.DETERMINISTIC.value == "deterministic"
