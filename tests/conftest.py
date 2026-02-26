"""Shared fixtures for all Loom test modules.

Provides:
  - In-memory LayeredGraphStore
  - MockLLMClient with realistic canned responses
  - Sample Events, Claims, Beliefs, PersonIdentifiers
  - Helper functions for creating test data
"""

import hashlib
import json
import time

import pytest

from loom.constants import (
    DEFAULT_CLAIM_CONFIDENCE,
    EVENT_TYPE_CALENDAR,
    EVENT_TYPE_EMAIL,
    EVENT_TYPE_MEETING,
    EVENT_TYPE_MESSAGE,
    HASH_PREFIX_LEN,
)
from loom.llm.mock import (
    MOCK_EXTRACTION_RESPONSE,
    MOCK_SYNTHESIS_RESPONSE,
    MOCK_TRIAGE_RESPONSE,
    MockLLMClient,
)
from loom.models import (
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
from loom.privacy import SensitivityLevel
from loom.store import LayeredGraphStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def store():
    """In-memory LayeredGraphStore for isolated tests."""
    s = LayeredGraphStore(db_path=":memory:")
    # Trigger schema creation
    _ = s.conn
    yield s
    s.close()


@pytest.fixture
def mock_llm():
    """MockLLMClient with sensible default canned responses."""
    return MockLLMClient(
        default_response="Mock LLM response",
        json_responses={
            "triage": MOCK_TRIAGE_RESPONSE,
            "extract": MOCK_EXTRACTION_RESPONSE,
            "commit": MOCK_EXTRACTION_RESPONSE,
            "synthe": MOCK_SYNTHESIS_RESPONSE,
            "belief": MOCK_SYNTHESIS_RESPONSE,
        },
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_COUNTER = 0


def make_event(
    source: str = "mail",
    source_id: str = "",
    event_type: str = EVENT_TYPE_EMAIL,
    timestamp: int = 0,
    participants: tuple[str, ...] = ("sender@example.com", "user@example.com"),
    raw_content: str = "Test email body content.",
    metadata: dict | None = None,
    sensitivity: SensitivityLevel = SensitivityLevel.SENSITIVE,
) -> Event:
    """Create a test Event with deterministic ID."""
    global _COUNTER
    if not source_id:
        _COUNTER += 1
        source_id = f"test_{_COUNTER}"
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
        sensitivity=sensitivity,
    )


def make_claim(
    claim_id: str = "",
    event_ids: list[str] | None = None,
    claim_type: str = "triage",
    subject: str = "person_sam",
    predicate: str = "triage_result",
    object_str: str = "",
    confidence: float = DEFAULT_CLAIM_CONFIDENCE,
    modality: Modality = Modality.ASSERTED,
    extraction_method: ExtractionMethod = ExtractionMethod.LOCAL_MODEL,
    sensitivity: SensitivityLevel = SensitivityLevel.SENSITIVE,
) -> Claim:
    """Create a test Claim."""
    if not claim_id:
        raw = f"test_claim:{subject}:{predicate}:{time.time_ns()}"
        claim_id = f"claim:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"
    if event_ids is None:
        event_ids = []
    if not object_str:
        object_str = json.dumps(MOCK_TRIAGE_RESPONSE)
    return Claim(
        id=claim_id,
        event_ids=event_ids,
        claim_type=claim_type,
        subject=subject,
        predicate=predicate,
        object=object_str,
        confidence=confidence,
        modality=modality,
        provenance=ExtractionProvenance(
            model_id="mock-model",
            extraction_method=extraction_method,
        ),
        sensitivity=sensitivity,
    )


def make_belief(
    belief_id: str = "",
    belief_type: BeliefType = BeliefType.FACT,
    subject: str = "user",
    summary: str = "User has a pending deadline",
    data: dict | None = None,
    epistemic_level: EpistemicLevel = EpistemicLevel.INFERENCE,
    source_claims: list[str] | None = None,
    confidence: float = 0.8,
    status: BeliefStatus = BeliefStatus.ACTIVE,
) -> Belief:
    """Create a test Belief."""
    if not belief_id:
        raw = f"test_belief:{subject}:{summary}:{time.time_ns()}"
        belief_id = f"belief:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"
    return Belief(
        id=belief_id,
        belief_type=belief_type,
        subject=subject,
        summary=summary,
        data=data or {"commitment": "test", "deadline": "2026-02-15"},
        epistemic_level=epistemic_level,
        source_claims=source_claims or [],
        confidence=confidence,
        status=status,
        inference_chain=["Step 1: observed", "Step 2: inferred"],
        evidence_log=[{"timestamp": int(time.time()), "event": "created", "delta": 0.8}],
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sample data fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def sample_events():
    """List of diverse Event objects covering multiple sources."""
    now = int(time.time())
    return [
        make_event(
            source="mail", source_id="email_001", event_type=EVENT_TYPE_EMAIL,
            timestamp=now - 3600,
            participants=("sam@example.com", "user@example.com"),
            raw_content="Please review the pitch deck and send feedback by Friday.",
            metadata={"subject": "Pitch deck review", "thread_id": "thread_a", "is_from_me": False},
        ),
        make_event(
            source="mail", source_id="email_002", event_type=EVENT_TYPE_EMAIL,
            timestamp=now - 1800,
            participants=("user@example.com", "sam@example.com"),
            raw_content="Sure, I will review it today.",
            metadata={"subject": "Re: Pitch deck review", "thread_id": "thread_a", "is_from_me": True},
        ),
        make_event(
            source="imessage", source_id="msg_001", event_type=EVENT_TYPE_MESSAGE,
            timestamp=now - 900,
            participants=("+14155551234",),
            raw_content="Hey, are we still meeting at 3pm?",
            metadata={"is_from_me": False, "chat_id": "chat_1"},
        ),
        make_event(
            source="whatsapp", source_id="wa_001", event_type=EVENT_TYPE_MESSAGE,
            timestamp=now - 600,
            participants=("bob@s.whatsapp.net",),
            raw_content="Can you send me the contract?",
            metadata={"is_from_me": False, "group_jid": None},
        ),
        make_event(
            source="calendar", source_id="cal_001", event_type=EVENT_TYPE_CALENDAR,
            timestamp=now + 86400,
            participants=("sam@example.com", "user@example.com", "bob@example.com"),
            raw_content="Weekly sync meeting",
            metadata={"title": "Weekly Sync", "location": "Zoom", "duration": 3600},
        ),
        make_event(
            source="granola", source_id="gran_001", event_type=EVENT_TYPE_MEETING,
            timestamp=now - 7200,
            participants=("user@example.com", "carol@example.com"),
            raw_content="Discussed Q1 roadmap. Action: Carol to draft proposal by Monday.",
            metadata={"title": "Q1 Planning", "duration": 3600},
        ),
    ]


@pytest.fixture
def sample_claims():
    """List of diverse Claim objects: deterministic, triage, and commitment."""
    now = int(time.time())
    return [
        # Deterministic: communication frequency
        Claim(
            id="claim:freq_sam",
            event_ids=[],
            claim_type="communication_frequency",
            subject="person_sam",
            predicate="communication_frequency",
            object=json.dumps({"event_count": 75, "frequency_label": "daily", "person_name": "Sam"}),
            confidence=0.95,
            modality=Modality.OBSERVED,
            provenance=ExtractionProvenance(model_id="deterministic"),
            sensitivity=SensitivityLevel.PRIVATE,
        ),
        # Deterministic: channel usage
        Claim(
            id="claim:channel_sam",
            event_ids=[],
            claim_type="channel_usage",
            subject="person_sam",
            predicate="channel_usage",
            object=json.dumps({"channels": {"mail": 50, "whatsapp": 25}, "primary": "mail"}),
            confidence=0.9,
            modality=Modality.OBSERVED,
            provenance=ExtractionProvenance(model_id="deterministic"),
        ),
        # Triage claim
        Claim(
            id="claim:triage_email_001",
            event_ids=[Event.make_id("mail", "email_001")],
            claim_type="triage",
            subject="person_sam",
            predicate="triage_result",
            object=json.dumps(MOCK_TRIAGE_RESPONSE),
            confidence=0.75,
            modality=Modality.ASSERTED,
            provenance=ExtractionProvenance(
                model_id="qwen3:30b-a3b",
                extraction_method=ExtractionMethod.LOCAL_MODEL,
            ),
        ),
        # Commitment claim
        Claim(
            id="claim:commit_proposal",
            event_ids=[Event.make_id("mail", "email_001")],
            claim_type="commitment",
            subject="user",
            predicate="promise",
            object=json.dumps({
                "what": "Send the updated proposal to the client",
                "who": "user",
                "whom": "client",
                "deadline": "2026-02-15",
                "commitment_type": "promise",
            }),
            confidence=0.85,
            modality=Modality.ASSERTED,
            provenance=ExtractionProvenance(
                model_id="gemini-flash",
                extraction_method=ExtractionMethod.CLOUD_MODEL,
            ),
        ),
    ]


@pytest.fixture
def sample_beliefs():
    """List of diverse Belief objects covering all belief types."""
    now = int(time.time())
    return [
        Belief(
            id="belief:entity_sam",
            belief_type=BeliefType.ENTITY,
            subject="person_sam",
            summary="Sam is a frequent collaborator via email and WhatsApp",
            data={"role": "colleague", "communication_volume": "high", "primary_channel": "mail"},
            epistemic_level=EpistemicLevel.COMPUTED,
            source_claims=["claim:freq_sam", "claim:channel_sam"],
            confidence=0.9,
            inference_chain=["75 events in 7 days", "Daily frequency", "Multi-channel"],
        ),
        Belief(
            id="belief:relation_sam_user",
            belief_type=BeliefType.RELATION,
            subject="person_sam",
            summary="Sam and user are actively collaborating on Acme Corp proposal",
            data={"relation": "collaborator", "context": "Acme Corp proposal", "strength": "strong"},
            epistemic_level=EpistemicLevel.INFERENCE,
            source_claims=["claim:triage_email_001", "claim:commit_proposal"],
            confidence=0.8,
            inference_chain=["Email thread about pitch deck", "Calendar meeting scheduled"],
        ),
        Belief(
            id="belief:fact_deadline",
            belief_type=BeliefType.FACT,
            subject="user",
            summary="User has a pending proposal deadline for Acme Corp on Feb 15",
            data={"commitment": "send updated proposal", "deadline": "2026-02-15", "counterparty": "Acme Corp"},
            epistemic_level=EpistemicLevel.INFERENCE,
            source_claims=["claim:commit_proposal"],
            confidence=0.85,
            inference_chain=["Email mentions 'will send by Friday'", "Calendar shows meeting next week"],
            expires_at=now + 86400 * 3,
        ),
        Belief(
            id="belief:obs_response_pattern",
            belief_type=BeliefType.OBSERVATION,
            subject="person_sam",
            summary="Sam typically responds to emails within 2 hours during business hours",
            data={"avg_response_time_hours": 2.0, "active_hours": "9am-6pm PST"},
            epistemic_level=EpistemicLevel.COMPUTED,
            source_claims=["claim:freq_sam"],
            confidence=0.7,
        ),
    ]


@pytest.fixture
def sample_persons():
    """List of PersonIdentifiers with various identifier combos."""
    return [
        PersonIdentifiers(
            person_id="person_user",
            emails={"user@example.com", "user@loom.example.com"},
            phones={"+15550100001"},
            display_names={"Alex", "Al"},
            canonical_name="Alex Chen",
            is_user=True,
            sources={"mail", "imessage", "whatsapp", "calendar"},
        ),
        PersonIdentifiers(
            person_id="person_sam",
            emails={"sam@example.com"},
            phones={"+14155551234"},
            display_names={"Sam Park"},
            whatsapp_jids={"14155551234@s.whatsapp.net"},
            canonical_name="Sam Park",
            sources={"mail", "whatsapp", "calendar"},
        ),
        PersonIdentifiers(
            person_id="person_bob",
            emails={"bob@example.com"},
            display_names={"Bob Smith"},
            canonical_name="Bob Smith",
            sources={"mail", "calendar"},
        ),
        PersonIdentifiers(
            person_id="person_carol",
            emails={"carol@example.com"},
            phones={"+16125551234"},
            display_names={"Carol"},
            canonical_name="Carol Williams",
            sources={"mail", "granola"},
        ),
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Populated store fixture
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def populated_store(store, sample_events, sample_claims, sample_beliefs):
    """A store pre-populated with sample events, claims, beliefs, and persons."""
    for event in sample_events:
        store.put_event(event)

    # Add persons
    store.put_person("person_user", canonical_name="Alex", is_user=True, sources=["mail", "imessage"])
    store.put_person("person_sam", canonical_name="Sam Park", sources=["mail", "whatsapp"])
    store.put_person("person_bob", canonical_name="Bob Smith", sources=["mail"])
    store.put_person("person_carol", canonical_name="Carol Williams", sources=["granola"])

    # Add identifiers
    store.add_person_identifier("person_user", "email", "user@example.com", display_name="Alex")
    store.add_person_identifier("person_sam", "email", "sam@example.com", display_name="Sam")
    store.add_person_identifier("person_sam", "phone", "+14155551234")
    store.add_person_identifier("person_bob", "email", "bob@example.com", display_name="Bob")
    store.add_person_identifier("person_carol", "email", "carol@example.com", display_name="Carol")

    # Link events to persons
    store.link_event_person(sample_events[0].id, "person_sam", "sender")
    store.link_event_person(sample_events[0].id, "person_user", "recipient")
    store.link_event_person(sample_events[1].id, "person_user", "sender")
    store.link_event_person(sample_events[1].id, "person_sam", "recipient")
    store.link_event_person(sample_events[4].id, "person_sam", "attendee")
    store.link_event_person(sample_events[4].id, "person_user", "attendee")
    store.link_event_person(sample_events[4].id, "person_bob", "attendee")

    for claim in sample_claims:
        store.put_claim(claim)

    for belief in sample_beliefs:
        store.put_belief(belief)

    return store
