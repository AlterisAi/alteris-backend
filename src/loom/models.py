"""Three-layer data model: Event -> Claim -> Belief.

Event  (immutable) -- something that happened. A message, call, meeting.
Claim  (versioned) -- an interpretation extracted from events.
Belief (mutable)   -- a synthesized conclusion from accumulated claims.

Every Claim traces back to Events (paper trail).
Every Belief traces back to Claims (auditable reasoning).

This module defines the data structures only. Storage is in store.py.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any

from loom.constants import (
    CONTENT_HASH_MIN_LENGTH,
    CONTENT_HASH_PREFIX_LEN,
    DEFAULT_BELIEF_CONFIDENCE,
    DEFAULT_CLAIM_CONFIDENCE,
    HASH_PREFIX_LEN,
)
from loom.privacy import SensitivityLevel


@unique
class Modality(str, Enum):
    """How a claim was expressed in the source event."""
    ASSERTED = "asserted"
    PLANNED = "planned"
    HYPOTHETICAL = "hypothetical"
    QUOTED = "quoted"
    FICTIONAL = "fictional"
    OBSERVED = "observed"
    UNKNOWN = "unknown"


@unique
class ExtractionMethod(str, Enum):
    """How a claim was produced."""
    DETERMINISTIC = "deterministic"
    HEURISTIC = "heuristic"
    LOCAL_MODEL = "local_model"
    CLOUD_MODEL = "cloud_model"
    USER_INPUT = "user_input"


@unique
class BeliefType(str, Enum):
    """What kind of belief this is."""
    ENTITY = "entity"
    RELATION = "relation"
    FACT = "fact"
    OBSERVATION = "observation"


@unique
class EpistemicLevel(str, Enum):
    """How the belief was derived."""
    OBSERVATION = "observation"
    INFERENCE = "inference"
    JUDGMENT = "judgment"
    COMPUTED = "computed"


@unique
class BeliefStatus(str, Enum):
    """Lifecycle state of a belief."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    STALE = "stale"
    RETRACTED = "retracted"
    SUPERSEDED = "superseded"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 1: Events (immutable)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class Event:
    """An immutable record of something that happened.

    The id is a content-addressable hash of (source, source_id).
    If we re-extract the same source artifact, we get the same Event id.
    """
    id: str
    source: str
    source_id: str
    event_type: str
    timestamp: int
    participants: tuple[str, ...] = ()
    raw_content: str | None = None
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    sensitivity: SensitivityLevel = SensitivityLevel.SENSITIVE
    created_at: int = field(default_factory=lambda: int(time.time()))

    @staticmethod
    def make_id(source: str, source_id: str) -> str:
        """Deterministic, content-addressable event ID."""
        raw = f"{source}:{source_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:HASH_PREFIX_LEN]

    @staticmethod
    def content_hash_of(text: str) -> str:
        """SHA-256 prefix for dedup.

        Returns empty string for short messages (< CONTENT_HASH_MIN_LENGTH)
        to avoid false cross-source duplicate matches on common phrases.
        """
        if len(text) < CONTENT_HASH_MIN_LENGTH:
            return ""
        return hashlib.sha256(
            text.encode("utf-8", errors="replace")
        ).hexdigest()[:CONTENT_HASH_PREFIX_LEN]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 2: Claims (versioned, with provenance)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ExtractionProvenance:
    """Full provenance for how a claim was produced."""
    model_id: str = "deterministic"
    prompt_version: str = ""
    context_hash: str = ""
    extraction_method: ExtractionMethod = ExtractionMethod.DETERMINISTIC
    extracted_at: int = field(default_factory=lambda: int(time.time()))


@dataclass
class Claim:
    """An interpretation extracted from one or more events.

    Claims are the bridge between raw data and understanding.
    They carry full provenance so the system knows what events
    produced this claim, how it was extracted, and how confident
    the extractor was.
    """
    id: str
    event_ids: list[str]
    claim_type: str
    subject: str
    predicate: str
    object: str
    confidence: float = DEFAULT_CLAIM_CONFIDENCE
    modality: Modality = Modality.UNKNOWN
    provenance: ExtractionProvenance = field(default_factory=ExtractionProvenance)
    user_verified: bool | None = None
    user_correction: str | None = None
    superseded_by: str | None = None
    sensitivity: SensitivityLevel = SensitivityLevel.SENSITIVE
    created_at: int = field(default_factory=lambda: int(time.time()))

    @property
    def is_active(self) -> bool:
        """True if this claim hasn't been superseded or rejected."""
        return self.superseded_by is None and self.user_verified is not False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Layer 3: Beliefs (mutable, aggregated from claims)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Belief:
    """A synthesized conclusion from accumulated claims.

    Beliefs are what the system "knows". Each belief aggregates
    confidence from its supporting claims. The system can explain
    any belief by walking: Belief -> source Claims -> source Events.
    """
    id: str
    belief_type: BeliefType
    subject: str
    summary: str
    data: dict[str, Any]
    epistemic_level: EpistemicLevel
    source_claims: list[str]
    source_reliability: str = "C"
    info_credibility: int = 4
    confidence: float = DEFAULT_BELIEF_CONFIDENCE
    inference_chain: list[str] | None = None
    evidence_log: list[dict[str, Any]] | None = None
    status: BeliefStatus = BeliefStatus.ACTIVE
    supersedes: str | None = None
    superseded_by: str | None = None
    priority: int | None = None
    created_at: int = field(default_factory=lambda: int(time.time()))
    updated_at: int = field(default_factory=lambda: int(time.time()))
    expires_at: int | None = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Annotations (faceted, lens-independent observations about events)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass(frozen=True)
class Annotation:
    """A faceted observation about an event.

    Annotations are lens-independent: they describe what an event IS,
    not how important it is. An event can have many annotations across
    many facets. Facets are open-ended (life_domain, topic, entity,
    sender_domain, financial_type, emotional_tone, ...).

    Sources include 'structural' (parsed from metadata, no LLM),
    'apple_intelligence' (platform classifier), 'triage_llm_v1', etc.
    """
    event_id: str
    facet: str
    value: str
    confidence: float = 1.0
    source: str = "structural"
    created_at: int = field(default_factory=lambda: int(time.time()))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Projections (lens-scoped scores, disposable read model)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Projection:
    """A lens-scoped score for an event.

    Projections are the CQRS read model: disposable, recomputable from
    annotations + person engagement + lens config. DELETE and recompute
    when lens parameters change.

    The scorer writes projections, not claims. Different lenses produce
    different projections for the same event.
    """
    event_id: str
    lens: str
    score: float
    route: str
    components: dict[str, Any] = field(default_factory=dict)
    computed_at: int = field(default_factory=lambda: int(time.time()))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cross-entity: Person (identity resolution target)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class PersonIdentifiers:
    """All known identifiers for a single person across sources.

    This is the entity resolution target. Each source contributes
    identifiers; the merge layer unifies them into a single person.
    """
    person_id: str
    phones: set[str] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)
    display_names: set[str] = field(default_factory=set)
    whatsapp_jids: set[str] = field(default_factory=set)
    slack_ids: dict[str, str] = field(default_factory=dict)
    linkedin_url: str | None = None
    canonical_name: str = ""
    is_user: bool = False
    sources: set[str] = field(default_factory=set)

    def merge(self, other: PersonIdentifiers) -> None:
        """Merge another person's identifiers into this one."""
        self.phones |= other.phones
        self.emails |= other.emails
        self.display_names |= other.display_names
        self.whatsapp_jids |= other.whatsapp_jids
        self.slack_ids.update(other.slack_ids)
        self.sources |= other.sources
        if other.linkedin_url and not self.linkedin_url:
            self.linkedin_url = other.linkedin_url
        if other.canonical_name and not self.canonical_name:
            self.canonical_name = other.canonical_name

    def matches(self, other: PersonIdentifiers) -> bool:
        """Check if two person records likely refer to the same human."""
        if self.phones & other.phones:
            return True
        if self.emails & other.emails:
            return True
        return False
