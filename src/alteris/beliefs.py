"""Stage 7: Per-thread synthesis + claims-to-beliefs compiler.

Architecture (after Binary Gate + Synthesis refactor):
  1. Read actionable threads from extraction_gate claims
  2. Reconstruct ThreadBundles from stored events
  3. Per-thread: LLM synthesis -> structured commitment claims
  4. Per-thread: logistics extraction -> logistics claims
  5. Per-thread: relational extraction -> relational claims
  6. Dedup commitment claims
  7. Expire stale commitment claims
  8. Entity beliefs (deterministic, from persons table)
  9. Relation beliefs (deterministic, from stage1 claims)
  10. Commitment FACT beliefs (claims -> beliefs with dedup/merge)
  11. Logistics FACT beliefs (claims -> beliefs with dedup/merge)
  12. Expire stale beliefs

The synthesis pass is the single seat of judgment: it has the full person
graph, cross-thread context, and raw thread content. Flash Lite (gate) only
decides "actionable or not"; Flash 3 (synthesis) decides WHO owes WHAT to WHOM.

The fact belief compilation layer (steps 10-11) bridges the gap between
claims and beliefs for commitments and logistics. It uses cluster_by_similarity
— a reusable union-find clustering mechanism with three signals (prefix match,
Jaccard-no-stopwords, SequenceMatcher) — to merge duplicate claims into single
beliefs with merge provenance in the evidence_log.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, field_validator

from alteris.prompts.beliefs import (
    KNOWN_COMMITMENTS_BLOCK,
    LOGISTICS_EXTRACTION_SYSTEM,
    RELATIONAL_EXTRACTION_SYSTEM,
    SYNTHESIS_PROMPT_TEMPLATE,
    SYNTHESIS_SYSTEM,
)
from alteris.constants import (
    BELIEF_COMMITMENT_EXPIRY_DAYS,
    BELIEF_JACCARD_VOTE_THRESHOLD,
    BELIEF_MERGE_MIN_VOTES,
    BELIEF_SEQMATCH_VOTE_THRESHOLD,
    COMMITMENT_DEADLINE_GRACE_DAYS,
    COMMITMENT_NO_DEADLINE_STALE_DAYS,
    DEFAULT_BELIEF_CONFIDENCE,
    DEDUP_WHAT_PREFIX_LEN,
    DEDUP_TOKEN_OVERLAP_THRESHOLD,
    LOGISTICS_EXTRACTION_PROMPT_VERSION,
    RELATIONAL_EXTRACTION_PROMPT_VERSION,
    RELATIONAL_SKIP_GENERIC_NAMES,
    SECONDS_PER_DAY,
    STALENESS_THREAD_AGE_DAYS,
    SYNTHESIS_BATCH_SIZE,
    SYNTHESIS_PROMPT_VERSION,
    WINDOW_SIZE,
)
from alteris.extract import (
    ThreadBundle,
    _format_thread_for_llm,
    build_persons_cache,
    get_actionable_threads,
    get_logistics_threads,
    get_relational_threads,
)
from alteris.llm.base import LLMClient
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
)
from alteris.privacy import SensitivityLevel
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Validation constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALID_COMMITMENT_TYPES = {
    "inbound_request", "user_commitment", "deadline",
    "waiting_on", "payment_due", "follow_up",
}
VALID_STATUSES = {"open", "done", "cancelled"}
VALID_DIRECTIONS = {"direct_ask", "group_ask", "self_directed", "ambiguous"}
VALID_STALENESS_SIGNALS = {
    "none", "overdue_no_followup", "group_broadcast", "old_thread",
}
VALID_PROVENANCES = {
    "assigned_to_user", "user_said", "system_detected", "inferred_from_context",
}

# Action type taxonomy (v5: operational state — whose court is the ball in?)
VALID_ACTION_TYPES = {
    "user_owes_action", "waiting_on_other",
    "scheduling_conflict_or_setup", "passive_tracking_or_reminder",
}

# Speech act classification (structural change #2)
VALID_SPEECH_ACTS = {
    "promise", "request", "decision", "assignment",
    "delegation", "inform",
}

# Counter-party response types (structural change #3)
VALID_RESPONSE_TYPES = {
    "acknowledged", "accepted", "no_response", "continued_discussion",
}

# Minimum boolean confidence fields required for "high confidence" (structural change #6)
CONFIDENCE_MIN_TRUE_FIELDS = 3

# Compatible types for dedup merging (moved from old extract.py)
COMPATIBLE_TYPES = {
    frozenset({"inbound_request", "user_commitment"}),
    frozenset({"deadline", "follow_up"}),
    frozenset({"payment_due", "deadline"}),
}

# Vague verbs that indicate exploration, not commitment
VAGUE_ACTION_VERBS = [
    "explore", "look into", "think about", "consider", "investigate",
    "evaluate", "research", "assess", "brainstorm", "revisit",
    "check out", "noodle on", "mull over", "keep in mind",
]

# Evidence quote validation threshold (fuzzy match ratio)
QUOTE_VALIDATION_THRESHOLD = 0.65


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Evidence quote validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def _quote_exists_in_source(
    quote: str,
    source_text: str,
    threshold: float = QUOTE_VALIDATION_THRESHOLD,
) -> tuple[bool, float]:
    """Check if a quote exists in the source text via fuzzy matching.

    Returns (found, best_score).
    """
    if not quote or not source_text:
        return False, 0.0

    norm_quote = _normalize_text(quote)
    norm_source = _normalize_text(source_text)

    if not norm_quote:
        return False, 0.0

    # Exact substring check (fast path)
    if norm_quote in norm_source:
        return True, 1.0

    quote_words = norm_quote.split()
    source_words = norm_source.split()
    window_size = len(quote_words)

    if window_size == 0 or len(source_words) == 0:
        return False, 0.0

    # For large sources, use word-overlap pre-filter to find candidate regions
    # instead of O(n*m) SequenceMatcher at every position
    quote_word_set = set(quote_words)
    best_score = 0.0

    # Find candidate start positions: windows that share enough words with quote
    # to plausibly match. Skip windows with <30% word overlap.
    min_overlap = max(1, int(len(quote_word_set) * 0.3))

    candidates: list[int] = []
    for size_delta in range(-2, 4):
        ws = max(2, window_size + size_delta)
        if ws > len(source_words):
            continue
        # Sliding count of matching words using a rolling set approach
        for i in range(0, len(source_words) - ws + 1, max(1, ws // 4)):
            window_words = set(source_words[i:i + ws])
            overlap = len(quote_word_set & window_words)
            if overlap >= min_overlap:
                candidates.append((i, ws))

    # Cap candidates to avoid quadratic blowup on huge threads
    if len(candidates) > 200:
        candidates.sort(
            key=lambda x: len(quote_word_set & set(source_words[x[0]:x[0] + x[1]])),
            reverse=True,
        )
        candidates = candidates[:200]

    for start, ws in candidates:
        window = " ".join(source_words[start:start + ws])
        score = SequenceMatcher(None, norm_quote, window).ratio()
        if score > best_score:
            best_score = score
            if best_score >= 0.9:
                return True, best_score

    return best_score >= threshold, best_score


def validate_evidence_quotes(
    results: list[SynthesisCommitmentResult],
    source_text: str,
) -> list[SynthesisCommitmentResult]:
    """Validate evidence quotes against source text.

    Items are kept if:
    - Quote fuzzy-matches the source text, OR
    - Confidence >= 0.7 (anti-timidity: don't drop valid commitments
      just because the LLM paraphrased the quote slightly)
    Items without quotes are kept if confidence >= 0.7.
    """
    validated = []
    for r in results:
        if r.evidence_quote:
            found, score = _quote_exists_in_source(r.evidence_quote, source_text)
            if found:
                validated.append(r)
            elif r.confidence >= 0.7:
                # Anti-timidity: keep high-confidence items even if quote is imperfect
                logger.debug(
                    "Quote validation soft-pass (score=%.2f, conf=%.2f): %s",
                    score, r.confidence, r.what[:50],
                )
                validated.append(r)
            else:
                logger.debug(
                    "Quote validation failed (score=%.2f): %s",
                    score, r.what[:50],
                )
        elif r.confidence >= 0.7:
            validated.append(r)
        else:
            logger.debug("No evidence quote, low confidence: %s", r.what[:50])
    return validated


def filter_vague_actions(
    results: list[SynthesisCommitmentResult],
) -> list[SynthesisCommitmentResult]:
    """Filter out commitments with vague/exploratory action verbs."""
    kept = []
    for r in results:
        what_lower = r.what.lower().strip()
        is_vague = any(what_lower.startswith(v) for v in VAGUE_ACTION_VERBS)
        if is_vague:
            logger.debug("Vague action filtered: %s", r.what[:50])
        else:
            kept.append(r)
    return kept


def filter_by_speech_act(
    results: list[SynthesisCommitmentResult],
) -> list[SynthesisCommitmentResult]:
    """Filter out items whose speech act indicates no real commitment.

    Keep: promise, request, decision, assignment.
    Drop: inform (informational only), delegation (someone else's task).
    Delegation is kept only if the user is the delegator tracking it.
    """
    actionable_acts = {"promise", "request", "decision", "assignment"}
    kept = []
    for r in results:
        if r.speech_act in actionable_acts:
            kept.append(r)
        elif r.speech_act == "delegation" and r.who == "user":
            # User delegated but is tracking it → keep as waiting_on
            kept.append(r)
        else:
            logger.debug(
                "Speech act filtered (%s): %s", r.speech_act, r.what[:50],
            )
    return kept


def filter_by_response_type(
    results: list[SynthesisCommitmentResult],
) -> list[SynthesisCommitmentResult]:
    """Filter out items where the counter-party never actually committed.

    Keep: accepted, acknowledged, no_response (inbound requests may not have reply yet).
    Drop: continued_discussion (topic was discussed but no one committed).
    """
    kept = []
    for r in results:
        if r.response_type == "continued_discussion":
            logger.debug(
                "Response type filtered (continued_discussion): %s",
                r.what[:50],
            )
        else:
            kept.append(r)
    return kept


def filter_by_confidence_fields(
    results: list[SynthesisCommitmentResult],
    min_true: int = 2,
) -> list[SynthesisCommitmentResult]:
    """Filter out items with too few boolean confidence fields set.

    Default minimum: 2 of 4 fields must be true. Items with 0-1 true fields
    are structurally weak (no named actor AND no deliverable AND no deadline
    AND not a response to a request).
    """
    kept = []
    for r in results:
        count = r.confidence_fields_true
        if count >= min_true:
            kept.append(r)
        else:
            logger.debug(
                "Confidence fields filtered (%d/%d): %s",
                count, 4, r.what[:50],
            )
    return kept


def validate_evidence_spans(
    results: list[SynthesisCommitmentResult],
    msg_texts: dict[str, str],
) -> list[SynthesisCommitmentResult]:
    """Validate evidence_start_char/end_char against actual message text.

    If span offsets are provided, check that the extracted span matches the
    evidence_quote. If they don't match, clear the offsets (don't drop the
    item — fall back to fuzzy quote validation).
    """
    for r in results:
        if (
            r.evidence_start_char is not None
            and r.evidence_end_char is not None
            and r.source_message_id
            and r.evidence_quote
        ):
            msg_text = msg_texts.get(r.source_message_id, "")
            if msg_text and r.evidence_end_char <= len(msg_text):
                span = msg_text[r.evidence_start_char:r.evidence_end_char]
                # Normalize for comparison (whitespace/case insensitive)
                if _normalize_text(span) != _normalize_text(r.evidence_quote):
                    logger.debug(
                        "Span mismatch for '%s': expected '%s', got '%s'",
                        r.what[:30], r.evidence_quote[:30], span[:30],
                    )
                    # Clear bad offsets — quote validation still applies
                    r.evidence_start_char = None
                    r.evidence_end_char = None
            else:
                r.evidence_start_char = None
                r.evidence_end_char = None
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SynthesisCommitmentResult validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SynthesisCommitmentResult(BaseModel):
    """Validated commitment from per-thread synthesis."""
    type: str = "inbound_request"
    action_type: str = "user_owes_action"
    who: str = "user"
    what: str
    to_whom: str | None = None
    direction: str = "ambiguous"
    deadline: str | None = None
    status: str = "open"
    priority: int = 2
    confidence: float = 0.8
    staleness_signal: str = "none"
    provenance: str = "system_detected"
    note: str | None = None
    source_message_id: str | None = None
    evidence_quote: str | None = None

    # ── Structural change #1: Source-text anchoring ──
    evidence_start_char: int | None = None
    evidence_end_char: int | None = None

    # ── Structural change #2: Speech act classification ──
    speech_act: str = "request"

    # ── Structural change #3: Counter-party response ──
    proposed_by: str | None = None
    response_from: str | None = None
    response_type: str = "no_response"
    response_quote: str | None = None

    # ── Structural change #4: Temporal grounding ──
    when_committed: str | None = None  # msg_N label of the commitment moment

    # ── Structural change #5: Next physical action ──
    next_action: str | None = None

    # ── Rich extraction fields (v4) ──
    deferred_until: str | None = None
    assignee: str | None = None
    custom_fields: dict | None = None

    # ── Structural change #6: Confidence decomposition ──
    has_named_actor: bool = False
    has_concrete_deliverable: bool = False
    has_temporal_constraint: bool = False
    is_response_to_request: bool = False

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        v = v.lower().strip()
        return v if v in VALID_COMMITMENT_TYPES else "inbound_request"

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        v = v.lower().strip() if v else "user_owes_action"
        return v if v in VALID_ACTION_TYPES else "user_owes_action"

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        v = v.lower().strip()
        return v if v in VALID_STATUSES else "open"

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        v = v.lower().strip()
        return v if v in VALID_DIRECTIONS else "ambiguous"

    @field_validator("staleness_signal")
    @classmethod
    def validate_staleness(cls, v: str) -> str:
        v = v.lower().strip()
        return v if v in VALID_STALENESS_SIGNALS else "none"

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: Any) -> int:
        try:
            v = int(v)
        except (ValueError, TypeError):
            return 2
        return max(1, min(3, v))

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: Any) -> float:
        try:
            v = float(v)
        except (ValueError, TypeError):
            return 0.8
        return max(0.0, min(1.0, v))

    @field_validator("what")
    @classmethod
    def validate_what(cls, v: str) -> str:
        return v[:200].strip() if v else ""

    @field_validator("deadline")
    @classmethod
    def validate_deadline(cls, v: str | None) -> str | None:
        if not v or v.lower() in ("null", "none", "n/a", ""):
            return None
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            return None

    @field_validator("provenance")
    @classmethod
    def validate_provenance(cls, v: str) -> str:
        v = v.lower().strip()
        return v if v in VALID_PROVENANCES else "system_detected"

    @field_validator("note")
    @classmethod
    def validate_note(cls, v: str | None) -> str | None:
        if v:
            return v[:2000].strip()
        return None

    @field_validator("speech_act")
    @classmethod
    def validate_speech_act(cls, v: str) -> str:
        v = v.lower().strip()
        return v if v in VALID_SPEECH_ACTS else "request"

    @field_validator("response_type")
    @classmethod
    def validate_response_type(cls, v: str) -> str:
        v = v.lower().strip()
        return v if v in VALID_RESPONSE_TYPES else "no_response"

    @field_validator("next_action")
    @classmethod
    def validate_next_action(cls, v: str | None) -> str | None:
        if v:
            return v[:200].strip() or None
        return None

    @field_validator("deferred_until")
    @classmethod
    def validate_deferred_until(cls, v: str | None) -> str | None:
        if not v or v.lower() in ("null", "none", "n/a", ""):
            return None
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            return None

    @field_validator("assignee")
    @classmethod
    def validate_assignee(cls, v: str | None) -> str | None:
        if v:
            v = v.strip()[:200]
            return v or None
        return None

    @field_validator("custom_fields")
    @classmethod
    def validate_custom_fields(cls, v: dict | None) -> dict | None:
        if not v or not isinstance(v, dict):
            return None
        # Keep only string key/value pairs
        cleaned = {
            str(k): str(val) for k, val in v.items()
            if isinstance(val, (str, int, float, bool))
        }
        return cleaned or None

    @field_validator("response_quote")
    @classmethod
    def validate_response_quote(cls, v: str | None) -> str | None:
        if v:
            return v[:500].strip() or None
        return None

    @property
    def confidence_fields_true(self) -> int:
        """Count how many boolean confidence decomposition fields are true."""
        return sum([
            self.has_named_actor,
            self.has_concrete_deliverable,
            self.has_temporal_constraint,
            self.is_response_to_request,
        ])

    @property
    def is_structurally_confident(self) -> bool:
        """True if enough boolean confidence fields are set (3 of 4)."""
        return self.confidence_fields_true >= CONFIDENCE_MIN_TRUE_FIELDS


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Synthesis prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logistics extraction prompt (Flash Lite)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Relational extraction prompt (Flash Lite)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Belief ID generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _belief_id(belief_type: str, subject: str, summary: str) -> str:
    """Deterministic belief ID."""
    raw = f"{belief_type}:{subject}:{summary[:50]}"
    return f"belief:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build synthesis prompt context
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_person_context(
    store: LayeredGraphStore,
    bundle: ThreadBundle,
    persons_cache: dict[str, list[dict]] | None = None,
) -> str:
    """Build person context section for the synthesis prompt."""
    parts = []
    seen_persons: set[str] = set()

    for event in bundle.events:
        for p in (persons_cache or {}).get(event.id, []):
            pid = p.get("person_id", "")
            if pid in seen_persons or p.get("is_user"):
                continue
            seen_persons.add(pid)

            name = p.get("name", pid[:12])
            role = p.get("role", "participant")

            # Look up tier from entity beliefs
            entity = store.conn.execute(
                """SELECT data FROM beliefs
                   WHERE belief_type = 'entity' AND subject = ?
                     AND status = 'active'
                   LIMIT 1""",
                (pid,),
            ).fetchone()

            tier = 3
            msg_count = 0
            if entity:
                try:
                    data = json.loads(entity["data"])
                    tier = data.get("tier", 3)
                    msg_count = data.get("message_count", 0)
                except (json.JSONDecodeError, TypeError):
                    pass

            parts.append(
                f"  {name}: tier-{tier}, {msg_count} messages, role={role}"
            )

    if not parts:
        return "PERSON CONTEXT: No known contacts in this thread."
    return "PERSON CONTEXT:\n" + "\n".join(parts)


def _build_triage_context(bundle: ThreadBundle) -> str:
    """Build triage summary + per-message labels from bundle's triage_data.

    The per-message labels feed the label correction task in synthesis.
    """
    if not bundle.triage_data:
        return "TRIAGE SUMMARY: No triage data available."

    scores = [d.get("score", 0) for d in bundle.triage_data]
    avg_score = sum(scores) / max(len(scores), 1)
    domains: set[str] = set()
    topics: set[str] = set()

    for td in bundle.triage_data:
        if td.get("domain"):
            domains.add(td["domain"])
        for t in td.get("specific_topics", td.get("topics", [])):
            topics.add(t)

    parts = [f"TRIAGE SUMMARY: avg_score={avg_score:.2f}"]
    if domains:
        parts.append(f"  domains: {', '.join(sorted(domains))}")
    if topics:
        parts.append(f"  topics: {', '.join(sorted(topics)[:5])}")

    # Per-message labels for the label correction task
    parts.append("")
    parts.append("PER-MESSAGE TRIAGE LABELS (from fast model — review and correct):")
    for i, td in enumerate(bundle.triage_data):
        domain = td.get("domain", "")
        spheres = td.get("universal_spheres", [])
        msg_topics = td.get("specific_topics", td.get("topics", []))
        parts.append(
            f"  msg_{i}: domain={domain}, "
            f"spheres={json.dumps(spheres)}, "
            f"topics={json.dumps(msg_topics)}"
        )

    return "\n".join(parts)


def _build_group_context(bundle: ThreadBundle) -> str:
    """Build group metadata section."""
    is_group = False
    group_name = ""
    member_count = 0

    for event in bundle.events:
        meta = event.metadata or {}
        if meta.get("is_group"):
            is_group = True
            group_name = group_name or meta.get("group_name", "")
        if meta.get("group_jid"):
            is_group = True
        member_count = max(member_count, len(event.participants))

    if is_group:
        name = group_name or "unknown group"
        return (
            f'GROUP METADATA: is_group=true, group_name="{name}", '
            f"members={member_count}"
        )
    return "GROUP METADATA: is_group=false (direct conversation)"


def _build_thread_age_context(bundle: ThreadBundle) -> str:
    """Build thread age and staleness context."""
    now = int(time.time())

    if not bundle.events:
        return "THREAD AGE: No events."

    latest_ts = max(e.timestamp for e in bundle.events)
    earliest_ts = min(e.timestamp for e in bundle.events)
    age_days = (now - latest_ts) / SECONDS_PER_DAY
    span_days = (latest_ts - earliest_ts) / SECONDS_PER_DAY

    parts = [
        f"THREAD AGE: last_activity={age_days:.0f} days ago, "
        f"thread_span={span_days:.0f} days"
    ]

    if age_days > STALENESS_THREAD_AGE_DAYS:
        parts.append(
            f"  WARNING: Thread is {age_days:.0f} days old — likely stale"
        )

    return "\n".join(parts)
def _get_prior_commitments(
    store: LayeredGraphStore,
    thread_id: str,
) -> list[dict]:
    """Fetch active commitment claims for a thread.

    Returns list of dicts with commitment details for prompt inclusion.
    """
    rows = store.conn.execute(
        """SELECT id, object, confidence FROM claims
           WHERE claim_type = 'commitment'
             AND subject = ?
             AND superseded_by IS NULL
           ORDER BY created_at DESC""",
        (thread_id,),
    ).fetchall()

    prior = []
    for row in rows:
        try:
            data = json.loads(row["object"])
            prior.append({
                "claim_id": row["id"],
                "who": data.get("who", "user"),
                "what": data.get("what", ""),
                "deadline": data.get("deadline"),
                "status": data.get("status", "open"),
                "confidence": data.get("confidence", row["confidence"]),
                "type": data.get("type", "inbound_request"),
            })
        except (json.JSONDecodeError, TypeError):
            continue
    return prior


def _format_prior_commitments(prior: list[dict]) -> str:
    """Format prior commitments for the synthesis prompt."""
    if not prior:
        return ""

    lines = []
    for i, c in enumerate(prior, 1):
        deadline = c.get("deadline") or "no deadline"
        lines.append(
            f"{i}. [{c['who']}] committed to [{c['what']}] by [{deadline}] "
            f"-- status: {c['status']}, confidence: {c['confidence']:.1f}"
        )

    commitments_list = "\n".join(lines)
    return KNOWN_COMMITMENTS_BLOCK.format(commitments_list=commitments_list)


def _build_custom_fields_section(field_defs: list[dict]) -> str:
    """Build the CUSTOM FIELDS prompt section from user-defined field definitions."""
    if not field_defs:
        return ""
    lines = ['CUSTOM FIELDS (best-effort — extract into "custom_fields" object if present):']
    for fd in field_defs:
        name = fd.get("name", "")
        desc = fd.get("description", "")
        example = fd.get("example", "")
        entry = f'- "{name}"'
        if desc:
            entry += f": {desc}"
        if example:
            entry += f' (example: "{example}")'
        lines.append(entry)
    return "\n".join(lines)


def _build_synthesis_prompt(
    bundle: ThreadBundle,
    user_email: str = "",
    persons_cache: dict[str, list[dict]] | None = None,
    store: LayeredGraphStore | None = None,
    prior_commitments: list[dict] | None = None,
    custom_field_defs: list[dict] | None = None,
    profile_context: str = "",
    email_to_name: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Build the full synthesis prompt for a thread.

    Args:
        prior_commitments: If provided, adds a KNOWN COMMITMENTS block
            to the prompt for incremental synthesis.
        custom_field_defs: If provided, adds CUSTOM FIELDS section to prompt.

    Returns (prompt_text, msg_id_map).
    """
    thread_text, msg_id_map = _format_thread_for_llm(
        bundle, user_email, persons_cache=persons_cache,
        email_to_name=email_to_name,
    )

    context_parts = []

    # Add known commitments block for incremental synthesis
    if prior_commitments:
        commitments_block = _format_prior_commitments(prior_commitments)
        if commitments_block:
            context_parts.append(commitments_block)

    if profile_context:
        context_parts.append(profile_context)

    if store:
        context_parts.append(
            _build_person_context(store, bundle, persons_cache)
        )
    context_parts.append(_build_triage_context(bundle))
    context_parts.append(_build_group_context(bundle))
    context_parts.append(_build_thread_age_context(bundle))

    context_section = "\n\n".join(context_parts)
    custom_fields_section = _build_custom_fields_section(custom_field_defs or [])

    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        thread_text=thread_text,
        context_section=context_section,
        custom_fields_section=custom_fields_section,
    )

    return prompt, msg_id_map


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sliding-window synthesis for long threads
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WINDOW_SUMMARY_SYSTEM = """\
You are summarizing a segment of a conversation for a downstream commitment extractor.

Your summary MUST capture:
1. All open commitments (who promised what to whom, deadlines)
2. All open questions/requests directed at the user
3. Key people mentioned and their roles
4. Important logistics (dates, locations, plans)
5. The current state of any negotiations or decisions

Be concise but NEVER drop an open commitment or pending request.
Output plain text, not JSON.
"""

WINDOW_SUMMARY_PROMPT = """\
CONVERSATION SEGMENT (messages {start} to {end} of {total}):
{window_text}

{prior_summary}

Summarize this segment. Capture ALL open commitments, pending requests, key people, and logistics.
Be concise (max 500 words). Focus on actionable items.
"""


def _needs_windowed_synthesis(bundle: ThreadBundle) -> bool:
    """Check if a thread is too long for single-pass synthesis."""
    return len(bundle.events) > WINDOW_SIZE


def _synthesize_windowed(
    bundle: ThreadBundle,
    llm_client: LLMClient,
    model: str,
    lite_model: str,
    user_email: str,
    persons_cache: dict[str, list[dict]],
    store: LayeredGraphStore,
    prior_commitments: list[dict] | None = None,
    custom_field_defs: list[dict] | None = None,
    profile_context: str = "",
    email_to_name: dict[str, str] | None = None,
) -> tuple[str, dict[str, str], list[dict]]:
    """Process a long thread using sliding windows with carry-forward summaries.

    Splits the thread into windows of WINDOW_SIZE messages.
    For each window except the last, produces a running summary.
    The final window gets the full synthesis prompt with the accumulated
    summary prepended, producing the actual commitment extraction.

    Returns (final_prompt, msg_id_map, prior_commitments) ready for
    the standard synthesis LLM call.
    """
    events = bundle.events
    n_events = len(events)
    n_windows = (n_events + WINDOW_SIZE - 1) // WINDOW_SIZE

    logger.info(
        "Windowed synthesis for %s: %d msgs in %d windows",
        bundle.thread_id[:16], n_events, n_windows,
    )

    # Use lite model for window summaries (cheaper, just summarization)
    summary_model = lite_model or model
    running_summary = ""

    # Process all windows except the last one as summary windows
    for w in range(n_windows - 1):
        start_idx = w * WINDOW_SIZE
        end_idx = min((w + 1) * WINDOW_SIZE, n_events)
        window_events = events[start_idx:end_idx]

        # Build a mini-bundle for this window
        window_bundle = ThreadBundle(
            thread_id=bundle.thread_id,
            events=window_events,
            triage_data=bundle.triage_data[start_idx:end_idx]
            if bundle.triage_data else [],
        )

        window_text, _ = _format_thread_for_llm(
            window_bundle, user_email, persons_cache=persons_cache,
            email_to_name=email_to_name,
        )

        prior_section = ""
        if running_summary:
            prior_section = (
                f"SUMMARY OF MESSAGES 1-{start_idx} (from prior windows):\n"
                f"{running_summary}\n"
            )

        prompt = WINDOW_SUMMARY_PROMPT.format(
            start=start_idx + 1,
            end=end_idx,
            total=n_events,
            window_text=window_text,
            prior_summary=prior_section,
        )

        raw = llm_client.generate(
            prompt=prompt,
            system=WINDOW_SUMMARY_SYSTEM,
            model=summary_model,
            temperature=0.1,
        )
        running_summary = raw.strip() if raw else running_summary

        logger.debug(
            "Window %d/%d summary: %d chars",
            w + 1, n_windows, len(running_summary),
        )

    # Final window: build a full synthesis prompt with the running summary
    # prepended as context
    final_start = (n_windows - 1) * WINDOW_SIZE
    final_events = events[final_start:]
    final_bundle = ThreadBundle(
        thread_id=bundle.thread_id,
        events=final_events,
        triage_data=bundle.triage_data[final_start:]
        if bundle.triage_data else [],
    )

    # Build the standard synthesis prompt for the final window
    prompt, msg_id_map = _build_synthesis_prompt(
        final_bundle, user_email, persons_cache, store,
        prior_commitments=prior_commitments,
        custom_field_defs=custom_field_defs,
        profile_context=profile_context,
        email_to_name=email_to_name,
    )

    # Prepend the running summary so the synthesis LLM has full context
    if running_summary:
        summary_header = (
            f"CONVERSATION HISTORY (summary of messages 1-{final_start}, "
            f"produced by prior analysis passes):\n"
            f"{running_summary}\n\n"
            f"RECENT MESSAGES (for detailed extraction):\n"
        )
        prompt = summary_header + prompt

    return prompt, msg_id_map, prior_commitments or []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gemini structured output schemas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_synthesis_response_schema():
    """Gemini structured output schema for commitment synthesis."""
    from google.genai import types

    return types.Schema(
        type="OBJECT",
        properties={
            "commitments": types.Schema(
                type="ARRAY",
                items=types.Schema(
                    type="OBJECT",
                    properties={
                        "type": types.Schema(type="STRING", enum=[
                            "inbound_request", "user_commitment", "deadline",
                            "waiting_on", "payment_due", "follow_up",
                        ]),
                        "action_type": types.Schema(type="STRING", enum=[
                            "user_owes_action", "waiting_on_other",
                            "scheduling_conflict_or_setup",
                            "passive_tracking_or_reminder",
                        ]),
                        "who": types.Schema(type="STRING"),
                        "what": types.Schema(type="STRING"),
                        "to_whom": types.Schema(type="STRING", nullable=True),
                        "direction": types.Schema(type="STRING", enum=[
                            "direct_ask", "group_ask", "self_directed", "ambiguous",
                        ]),
                        "deadline": types.Schema(type="STRING", nullable=True),
                        "status": types.Schema(type="STRING", enum=[
                            "open", "done", "cancelled",
                        ]),
                        "priority": types.Schema(type="INTEGER"),
                        "confidence": types.Schema(type="NUMBER"),
                        "staleness_signal": types.Schema(type="STRING", enum=[
                            "none", "overdue_no_followup",
                            "group_broadcast", "old_thread",
                        ]),
                        "provenance": types.Schema(type="STRING", enum=[
                            "assigned_to_user", "user_said",
                            "system_detected", "inferred_from_context",
                        ]),
                        "note": types.Schema(type="STRING", nullable=True),
                        "source_message_id": types.Schema(
                            type="STRING", nullable=True,
                        ),
                        "evidence_quote": types.Schema(
                            type="STRING", nullable=True,
                        ),
                        "evidence_start_char": types.Schema(
                            type="INTEGER", nullable=True,
                        ),
                        "evidence_end_char": types.Schema(
                            type="INTEGER", nullable=True,
                        ),
                        "speech_act": types.Schema(type="STRING", enum=[
                            "promise", "request", "decision",
                            "assignment", "delegation", "inform",
                        ]),
                        "proposed_by": types.Schema(
                            type="STRING", nullable=True,
                        ),
                        "response_from": types.Schema(
                            type="STRING", nullable=True,
                        ),
                        "response_type": types.Schema(type="STRING", enum=[
                            "acknowledged", "accepted",
                            "no_response", "continued_discussion",
                        ]),
                        "response_quote": types.Schema(
                            type="STRING", nullable=True,
                        ),
                        "when_committed": types.Schema(
                            type="STRING", nullable=True,
                        ),
                        "next_action": types.Schema(
                            type="STRING", nullable=True,
                        ),
                        "has_named_actor": types.Schema(type="BOOLEAN"),
                        "has_concrete_deliverable": types.Schema(
                            type="BOOLEAN",
                        ),
                        "has_temporal_constraint": types.Schema(
                            type="BOOLEAN",
                        ),
                        "is_response_to_request": types.Schema(
                            type="BOOLEAN",
                        ),
                    },
                    required=[
                        "type", "action_type", "who", "what", "direction",
                        "status", "priority", "confidence", "staleness_signal",
                        "provenance", "speech_act", "response_type",
                        "has_named_actor", "has_concrete_deliverable",
                        "has_temporal_constraint", "is_response_to_request",
                    ],
                ),
            ),
            "label_corrections": types.Schema(
                type="ARRAY",
                items=types.Schema(
                    type="OBJECT",
                    properties={
                        "message_id": types.Schema(type="STRING"),
                        "domain": types.Schema(type="STRING"),
                        "universal_spheres": types.Schema(
                            type="ARRAY",
                            items=types.Schema(type="STRING"),
                        ),
                        "specific_topics": types.Schema(
                            type="ARRAY",
                            items=types.Schema(type="STRING"),
                        ),
                        "corrected": types.Schema(type="BOOLEAN"),
                    },
                    required=[
                        "message_id", "domain", "universal_spheres",
                        "specific_topics", "corrected",
                    ],
                ),
            ),
        },
        required=["commitments", "label_corrections"],
    )


def _build_logistics_extraction_schema():
    """Gemini structured output schema for logistics fact extraction."""
    from google.genai import types

    return types.Schema(
        type="OBJECT",
        properties={
            "facts": types.Schema(
                type="ARRAY",
                items=types.Schema(
                    type="OBJECT",
                    properties={
                        "type": types.Schema(type="STRING", enum=[
                            "reservation", "travel", "care_provider",
                            "appointment", "activity", "outing",
                            "childcare",
                        ]),
                        "venue": types.Schema(type="STRING", nullable=True),
                        "destination": types.Schema(
                            type="STRING", nullable=True,
                        ),
                        "provider": types.Schema(
                            type="STRING", nullable=True,
                        ),
                        "facility": types.Schema(
                            type="STRING", nullable=True,
                        ),
                        "name": types.Schema(type="STRING", nullable=True),
                        "child": types.Schema(type="STRING", nullable=True),
                        "who": types.Schema(type="STRING", nullable=True),
                        "date": types.Schema(type="STRING", nullable=True),
                        "dates": types.Schema(type="STRING", nullable=True),
                        "time": types.Schema(type="STRING", nullable=True),
                        "hours": types.Schema(type="STRING", nullable=True),
                        "party_size": types.Schema(
                            type="INTEGER", nullable=True,
                        ),
                        "confirmation": types.Schema(
                            type="STRING", nullable=True,
                        ),
                        "airline": types.Schema(
                            type="STRING", nullable=True,
                        ),
                        "rate": types.Schema(type="STRING", nullable=True),
                        "location": types.Schema(
                            type="STRING", nullable=True,
                        ),
                        "notes": types.Schema(type="STRING", nullable=True),
                        "pickup_time": types.Schema(
                            type="STRING", nullable=True,
                        ),
                        "dropoff_time": types.Schema(
                            type="STRING", nullable=True,
                        ),
                    },
                    required=["type"],
                ),
            ),
        },
        required=["facts"],
    )


def _build_relational_extraction_schema():
    """Gemini structured output schema for relational context extraction."""
    from google.genai import types

    return types.Schema(
        type="OBJECT",
        properties={
            "people": types.Schema(
                type="ARRAY",
                items=types.Schema(
                    type="OBJECT",
                    properties={
                        "name": types.Schema(type="STRING"),
                        "relationship_tier": types.Schema(
                            type="STRING",
                            enum=[
                                "core_kinship", "extended_kinship",
                                "intimate_friendship", "vocational_core_team",
                                "vocational_network", "commercial_vendor",
                                "unknown_or_first_contact",
                            ],
                        ),
                        "role": types.Schema(type="STRING"),
                        "organization": types.Schema(
                            type="STRING", nullable=True,
                        ),
                        "context": types.Schema(type="STRING"),
                        "relationship_strength": types.Schema(
                            type="STRING",
                            enum=["strong", "moderate", "weak"],
                        ),
                    },
                    required=[
                        "name", "relationship_tier", "role",
                        "context", "relationship_strength",
                    ],
                ),
            ),
        },
        required=["people"],
    )


def _build_resolution_gate_schema():
    """Gemini structured output schema for commitment resolution gate."""
    from google.genai import types

    return types.Schema(
        type="OBJECT",
        properties={
            "resolved": types.Schema(
                type="BOOLEAN",
                description="Whether the commitment appears resolved.",
            ),
            "evidence": types.Schema(
                type="STRING",
                description="Brief explanation of the resolution evidence.",
            ),
        },
        required=["resolved", "evidence"],
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-thread synthesis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_synthesis_result(
    raw: dict | None,
) -> list[SynthesisCommitmentResult]:
    """Parse LLM output into SynthesisCommitmentResults."""
    if not raw or "commitments" not in raw:
        return []

    results = []
    for item in raw["commitments"]:
        if not isinstance(item, dict):
            continue
        what = item.get("what", "").strip()
        if not what:
            continue
        try:
            result = SynthesisCommitmentResult.model_validate(item)
            results.append(result)
        except Exception:
            continue

    return results


def _parse_label_corrections(
    raw: dict | None,
    msg_id_map: dict[str, str] | None = None,
) -> list[dict]:
    """Parse label corrections from synthesis response.

    Returns list of dicts with resolved event_id, domain, spheres, topics.
    """
    if not raw or not isinstance(raw, dict):
        return []
    corrections = raw.get("label_corrections", [])
    if not isinstance(corrections, list):
        return []

    results = []
    for lc in corrections:
        if not isinstance(lc, dict):
            continue
        if not lc.get("corrected", False):
            continue
        msg_label = lc.get("message_id", "")
        event_id = (msg_id_map or {}).get(msg_label) if msg_label else None
        if not event_id:
            continue
        results.append({
            "event_id": event_id,
            "domain": lc.get("domain", ""),
            "universal_spheres": lc.get("universal_spheres", []),
            "specific_topics": lc.get("specific_topics", []),
        })
    return results


def _apply_label_corrections(
    store: LayeredGraphStore,
    corrections: list[dict],
    model_id: str,
) -> int:
    """Apply synthesis label corrections to triage claims and annotations.

    For each corrected event:
    1. Update the triage claim's domain/topics/spheres in-place
    2. Re-emit topic annotations with the corrected values

    Returns the number of corrections applied.
    """
    from alteris.triage import _emit_topic_annotations, triage_claim_id

    applied = 0
    for corr in corrections:
        event_id = corr["event_id"]
        new_domain = corr.get("domain", "")
        new_topics = corr.get("specific_topics", [])
        new_spheres = corr.get("universal_spheres", [])

        if not new_domain and not new_topics:
            continue

        # Update the triage claim in-place
        claim_id = triage_claim_id(event_id)
        try:
            row = store.conn.execute(
                "SELECT object FROM claims WHERE id = ? AND superseded_by IS NULL",
                (claim_id,),
            ).fetchone()
        except Exception:
            continue

        if not row:
            continue

        try:
            obj = json.loads(row["object"])
        except (json.JSONDecodeError, TypeError):
            continue

        # Apply corrections
        if new_domain:
            obj["domain"] = new_domain
        if new_topics:
            obj["specific_topics"] = new_topics
            obj["topics"] = new_topics  # keep legacy field in sync
        if new_spheres:
            obj["universal_spheres"] = new_spheres
        obj["label_corrected_by"] = f"synthesis:{model_id}"

        now = int(time.time())
        store.conn.execute(
            "UPDATE claims SET object = ?, updated_at = ? WHERE id = ?",
            (json.dumps(obj), now, claim_id),
        )

        # Re-emit topic annotations with corrected topics
        if new_topics:
            _emit_topic_annotations(
                store, event_id, new_topics,
                f"synthesis:{model_id}",
            )

        applied += 1

    if applied:
        store.conn.commit()
        logger.info(
            "Label corrections: %d/%d applied by synthesis",
            applied, len(corrections),
        )

    return applied


def _commitment_claim_id(thread_id: str, what: str, who: str) -> str:
    """Deterministic claim ID for a commitment."""
    key = f"commitment:{thread_id}:{what[:50]}:{who}:{SYNTHESIS_PROMPT_VERSION}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _build_commitment_claim(
    result: SynthesisCommitmentResult,
    thread_id: str,
    event_ids: list[str],
    model_id: str,
    msg_id_map: dict[str, str] | None = None,
) -> Claim:
    """Convert a SynthesisCommitmentResult into a Claim."""
    claim_id = _commitment_claim_id(thread_id, result.what, result.who)

    # Resolve source_message_id to actual event ID
    source_event_id = None
    if result.source_message_id and msg_id_map:
        source_event_id = msg_id_map.get(result.source_message_id)

    obj = {
        "type": result.type,
        "action_type": result.action_type,
        "who": result.who,
        "what": result.what,
        "to_whom": result.to_whom,
        "direction": result.direction,
        "deadline": result.deadline,
        "status": result.status,
        "priority": result.priority,
        "confidence": result.confidence,
        "staleness_signal": result.staleness_signal,
        "provenance": result.provenance,
        "note": result.note,
        "source_message_id": result.source_message_id,
        "source_event_id": source_event_id,
        "evidence_quote": result.evidence_quote,
        # Structural fields
        "evidence_start_char": result.evidence_start_char,
        "evidence_end_char": result.evidence_end_char,
        "speech_act": result.speech_act,
        "proposed_by": result.proposed_by,
        "response_from": result.response_from,
        "response_type": result.response_type,
        "response_quote": result.response_quote,
        "when_committed": result.when_committed,
        "next_action": result.next_action,
        "deferred_until": result.deferred_until,
        "assignee": result.assignee,
        "custom_fields": result.custom_fields,
        "has_named_actor": result.has_named_actor,
        "has_concrete_deliverable": result.has_concrete_deliverable,
        "has_temporal_constraint": result.has_temporal_constraint,
        "is_response_to_request": result.is_response_to_request,
    }

    # Determine extraction method from model name
    if "local" in model_id or "qwen" in model_id or "ollama" in model_id:
        method = ExtractionMethod.LOCAL_MODEL
    else:
        method = ExtractionMethod.CLOUD_MODEL

    return Claim(
        id=claim_id,
        event_ids=event_ids,
        claim_type="commitment",
        subject=thread_id,
        predicate=result.type,
        object=json.dumps(obj),
        confidence=result.confidence,
        modality=Modality.OBSERVED,
        provenance=ExtractionProvenance(
            model_id=model_id,
            prompt_version=SYNTHESIS_PROMPT_VERSION,
            extraction_method=method,
        ),
        sensitivity=SensitivityLevel.SENSITIVE,
    )


def synthesize_thread(
    bundle: ThreadBundle,
    llm_client: LLMClient,
    model: str = "",
    user_email: str = "",
    persons_cache: dict[str, list[dict]] | None = None,
    store: LayeredGraphStore | None = None,
    prior_commitments: list[dict] | None = None,
    custom_field_defs: list[dict] | None = None,
    profile_context: str = "",
    email_to_name: dict[str, str] | None = None,
) -> tuple[list[Claim], dict[str, str], list[dict]]:
    """Run per-thread commitment synthesis.

    Single Flash 3 call per thread. Returns (commitment_claims, msg_id_map, label_corrections).
    Post-processing: evidence quote validation + vague action filtering.

    Args:
        prior_commitments: If provided, includes KNOWN COMMITMENTS in prompt
            for incremental synthesis (thread-aware updates).
        custom_field_defs: User-defined fields to inject into prompt.
    """
    prompt, msg_id_map = _build_synthesis_prompt(
        bundle, user_email, persons_cache, store,
        prior_commitments=prior_commitments,
        custom_field_defs=custom_field_defs,
        profile_context=profile_context,
        email_to_name=email_to_name,
    )

    # Get the thread text for quote validation
    thread_text, _ = _format_thread_for_llm(
        bundle, user_email, persons_cache=persons_cache,
        email_to_name=email_to_name,
    )

    try:
        schema = _build_synthesis_response_schema()
    except ImportError:
        schema = None

    raw_str = llm_client.generate(
        prompt=prompt,
        system=SYNTHESIS_SYSTEM,
        model=model,
        temperature=0.1,
        response_schema=schema,
        format_json=schema is None,
    )

    try:
        raw = json.loads(raw_str) if raw_str else None
    except (json.JSONDecodeError, TypeError):
        raw = None

    results = _parse_synthesis_result(raw)
    label_corrections = _parse_label_corrections(raw, msg_id_map)

    # Build msg_texts map for span validation
    msg_texts: dict[str, str] = {}
    for event in bundle.events:
        text = event.raw_content or ""
        if text and event.id in (msg_id_map or {}).values():
            # Reverse lookup: find msg_N label for this event ID
            for label, eid in (msg_id_map or {}).items():
                if eid == event.id:
                    msg_texts[label] = text
                    break

    # Post-processing pipeline: layered filters
    n_raw = len(results)
    results = validate_evidence_quotes(results, thread_text)
    results = filter_vague_actions(results)
    results = validate_evidence_spans(results, msg_texts)
    results = filter_by_speech_act(results)
    results = filter_by_response_type(results)
    results = filter_by_confidence_fields(results, min_true=2)
    n_filtered = n_raw - len(results)
    if n_filtered > 0:
        logger.info(
            "Thread %s: %d/%d items passed post-processing",
            bundle.thread_id[:8], len(results), n_raw,
        )

    event_ids = [e.id for e in bundle.events]
    claims = []
    for result in results:
        claim = _build_commitment_claim(
            result, bundle.thread_id, event_ids,
            model or "synthesis", msg_id_map,
        )
        claims.append(claim)

    return claims, msg_id_map, label_corrections


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logistics extraction (Flash Lite)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _logistics_claim_id(thread_id: str, fact_summary: str) -> str:
    """Deterministic claim ID for a logistics fact."""
    key = (
        f"logistics:{thread_id}:{fact_summary[:50]}"
        f":{LOGISTICS_EXTRACTION_PROMPT_VERSION}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def extract_logistics(
    bundle: ThreadBundle,
    llm_client: LLMClient,
    model: str = "",
    user_email: str = "",
    persons_cache: dict[str, list[dict]] | None = None,
    profile_context: str = "",
    email_to_name: dict[str, str] | None = None,
) -> list[Claim]:
    """Extract logistics facts from a thread (Flash Lite).

    Returns logistics claims for reservations, travel, care, appointments, etc.
    """
    thread_text, _ = _format_thread_for_llm(
        bundle, user_email, persons_cache=persons_cache,
        email_to_name=email_to_name,
    )

    if profile_context:
        thread_text = profile_context + "\n\n" + thread_text

    try:
        schema = _build_logistics_extraction_schema()
    except ImportError:
        schema = None

    raw_str = llm_client.generate(
        prompt=thread_text,
        system=LOGISTICS_EXTRACTION_SYSTEM,
        model=model,
        temperature=0.1,
        response_schema=schema,
        format_json=schema is None,
    )

    try:
        raw = json.loads(raw_str) if raw_str else None
    except (json.JSONDecodeError, TypeError):
        raw = None

    if not raw or "facts" not in raw:
        return []

    facts = raw["facts"]
    if not isinstance(facts, list):
        return []

    event_ids = [e.id for e in bundle.events]
    claims = []

    for fact in facts:
        if not isinstance(fact, dict):
            continue
        fact_type = fact.get("type", "unknown")

        # Build a summary from the fact
        if fact_type == "reservation":
            summary = (
                f"{fact.get('venue', 'unknown')} on {fact.get('date', '?')} "
                f"at {fact.get('time', '?')}"
            )
        elif fact_type == "travel":
            summary = (
                f"Travel to {fact.get('destination', '?')} on "
                f"{fact.get('date', '?')}"
            )
        elif fact_type == "care_provider":
            summary = (
                f"{fact.get('provider', '?')} on {fact.get('date', '?')} "
                f"{fact.get('hours', '')}"
            )
        elif fact_type == "appointment":
            summary = (
                f"Appt with {fact.get('provider', '?')} on "
                f"{fact.get('date', '?')}"
            )
        elif fact_type == "activity":
            summary = (
                f"{fact.get('name', '?')} on {fact.get('dates', '?')}"
            )
        elif fact_type == "outing":
            parts = [fact.get("name", "?")]
            if fact.get("location"):
                parts.append(f"at {fact['location']}")
            parts.append(f"on {fact.get('date', '?')}")
            summary = " ".join(parts)
        elif fact_type == "childcare":
            summary = (
                f"{fact.get('child', '?')} at {fact.get('facility', '?')} "
                f"on {fact.get('date', '?')}"
            )
        else:
            summary = fact.get("summary", str(fact)[:80])

        claim_id = _logistics_claim_id(bundle.thread_id, summary)

        claim = Claim(
            id=claim_id,
            event_ids=event_ids,
            claim_type="logistics",
            subject=bundle.thread_id,
            predicate=f"logistics_{fact_type}",
            object=json.dumps(fact),
            confidence=0.9,
            modality=Modality.OBSERVED,
            provenance=ExtractionProvenance(
                model_id=model or "logistics_extractor",
                prompt_version=LOGISTICS_EXTRACTION_PROMPT_VERSION,
                extraction_method=ExtractionMethod.CLOUD_MODEL,
            ),
            sensitivity=SensitivityLevel.SENSITIVE,
        )
        claims.append(claim)

    return claims


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Relational extraction (Flash Lite)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _relational_claim_id(thread_id: str, person_name: str) -> str:
    """Deterministic claim ID for a relational context claim."""
    key = (
        f"relational:{thread_id}:{person_name[:50]}"
        f":{RELATIONAL_EXTRACTION_PROMPT_VERSION}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def extract_relational(
    bundle: ThreadBundle,
    llm_client: LLMClient,
    model: str = "",
    user_email: str = "",
    persons_cache: dict[str, list[dict]] | None = None,
    store: LayeredGraphStore | None = None,
    gate_tier: str | None = None,
    profile_context: str = "",
    email_to_name: dict[str, str] | None = None,
) -> list[Claim]:
    """Extract relational context from a thread (Flash Lite).

    Returns relational claims about people, their roles, and relationships.

    Args:
        store: If provided, injects known person context into prompt.
        gate_tier: relationship_tier from the relational gate (Flash Lite
            first guess). Passed to the prompt to guide/validate classification.
    """
    thread_text, _ = _format_thread_for_llm(
        bundle, user_email, persons_cache=persons_cache,
        email_to_name=email_to_name,
    )

    if profile_context:
        thread_text = profile_context + "\n\n" + thread_text

    # Inject known person context so the LLM preserves correct tiers
    context_parts = []
    if store:
        person_ctx = _build_person_context(store, bundle, persons_cache)
        if "No known contacts" not in person_ctx:
            context_parts.append(person_ctx)
    if gate_tier:
        context_parts.append(
            f"GATE HINT: Relational gate classified this thread's primary "
            f"relationship_tier as '{gate_tier}'. Use this as a starting point "
            f"but override if the thread content reveals a different tier."
        )
    if context_parts:
        thread_text = "\n\n".join(context_parts) + "\n\n" + thread_text

    try:
        schema = _build_relational_extraction_schema()
    except ImportError:
        schema = None

    raw_str = llm_client.generate(
        prompt=thread_text,
        system=RELATIONAL_EXTRACTION_SYSTEM,
        model=model,
        temperature=0.1,
        response_schema=schema,
        format_json=schema is None,
    )

    try:
        raw = json.loads(raw_str) if raw_str else None
    except (json.JSONDecodeError, TypeError):
        raw = None

    if not raw or "people" not in raw:
        return []

    people = raw["people"]
    if not isinstance(people, list):
        return []

    event_ids = [e.id for e in bundle.events]
    claims = []

    for person in people:
        if not isinstance(person, dict):
            continue
        name = person.get("name", "").strip()
        if not name:
            continue

        claim_id = _relational_claim_id(bundle.thread_id, name)

        obj = {
            "name": name,
            "relationship_tier": person.get(
                "relationship_tier", "unknown_or_first_contact"
            ),
            "role": person.get("role", "unknown"),
            "organization": person.get("organization"),
            "context": person.get("context", ""),
            "relationship_strength": person.get(
                "relationship_strength", "moderate"
            ),
        }

        claim = Claim(
            id=claim_id,
            event_ids=event_ids,
            claim_type="relational_context",
            subject=bundle.thread_id,
            predicate=f"person_context:{name}",
            object=json.dumps(obj),
            confidence=0.8,
            modality=Modality.OBSERVED,
            provenance=ExtractionProvenance(
                model_id=model or "relational_extractor",
                prompt_version=RELATIONAL_EXTRACTION_PROMPT_VERSION,
                extraction_method=ExtractionMethod.CLOUD_MODEL,
            ),
            sensitivity=SensitivityLevel.SENSITIVE,
        )
        claims.append(claim)

    return claims


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Commitment dedup (moved from old extract.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STOP_WORDS = frozenset({
    "a", "an", "the", "at", "by", "for", "in", "of", "on", "to",
    "and", "or", "is", "are", "was", "were", "with", "from", "as",
    "this", "that", "vs", "via",
})
"""Pure function words removed from Jaccard token sets during belief clustering.
Concentrates similarity on content words so "leave for church by 9" and
"leave for church by 9:00 AM" merge correctly.

IMPORTANT: Do NOT add temporal/directional words like "before", "after",
"until", "during" — these carry meaning ("do X before Y" ≠ "do X after Y")."""


def _normalize_tokens(text: str) -> set[str]:
    """Normalize text to a set of lowercase content tokens (stop words removed)."""
    raw = set(re.sub(r"[^\w\s]", "", text.lower()).split())
    filtered = raw - STOP_WORDS
    # If stop-word removal would empty the set, keep the original tokens
    return filtered if filtered else raw


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


T = TypeVar("T")


MergeLog = list[dict[str, Any]]
"""Per-cluster list of pairwise merge reasons for audit trail."""


def cluster_by_similarity(
    items: list[T],
    key_fn: Callable[[T], str],
    jaccard_threshold: float = BELIEF_JACCARD_VOTE_THRESHOLD,
    seqmatch_threshold: float = BELIEF_SEQMATCH_VOTE_THRESHOLD,
    min_votes: int = BELIEF_MERGE_MIN_VOTES,
) -> tuple[list[list[T]], list[MergeLog]]:
    """Cluster items by fuzzy text similarity using mixture-of-experts.

    Three cheap deterministic expert signals each cast a binary vote:
      1. Prefix match — first 25 chars identical (catches trivial rewording)
      2. Jaccard token overlap (stop words removed) — content-word overlap
      3. SequenceMatcher ratio — character-level similarity (catches
         transcription errors like "Altarus" vs "Alteris")

    Items merge when ``min_votes`` or more experts agree (default 2).
    This prevents any single noisy signal from causing false merges.

    Returns (clusters, merge_logs):
      - clusters: list of item groups
      - merge_logs: parallel list, one per cluster, each containing
        pairwise merge reasons for audit (which experts voted, scores).
        Single-item clusters get an empty merge log.
    """
    if not items:
        return [], []
    if len(items) == 1:
        return [items], [[]]

    texts = [key_fn(item) for item in items]
    token_sets = [_normalize_tokens(t) for t in texts]

    # Union-find with path compression
    parent = list(range(len(items)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Track merge reasons per pair (indexed by representative)
    merge_reasons: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            # Each expert casts a binary vote
            jac_score = _jaccard_similarity(token_sets[i], token_sets[j])
            seq_score = SequenceMatcher(
                None, texts[i].lower(), texts[j].lower(),
            ).ratio()
            prefix_vote = (
                texts[i][:DEDUP_WHAT_PREFIX_LEN].lower().strip()
                == texts[j][:DEDUP_WHAT_PREFIX_LEN].lower().strip()
            )
            jaccard_vote = jac_score >= jaccard_threshold
            seqmatch_vote = seq_score >= seqmatch_threshold

            votes = int(prefix_vote) + int(jaccard_vote) + int(seqmatch_vote)
            if votes >= min_votes:
                union(i, j)
                # Record which experts voted for this merge
                experts = []
                if prefix_vote:
                    experts.append("prefix25")
                if jaccard_vote:
                    experts.append(f"jaccard={jac_score:.2f}")
                if seqmatch_vote:
                    experts.append(f"seqmatch={seq_score:.2f}")
                merge_reasons[find(i)].append({
                    "merged": [texts[i][:60], texts[j][:60]],
                    "experts": experts,
                    "votes": votes,
                })

    clusters: dict[int, list[T]] = defaultdict(list)
    for i, item in enumerate(items):
        clusters[find(i)].append(item)

    cluster_list = list(clusters.values())
    # Build parallel merge_logs: map each cluster's root to its reasons
    logs: list[MergeLog] = []
    for root in clusters:
        logs.append(merge_reasons.get(root, []))

    return cluster_list, logs


def dedup_commitments(store: LayeredGraphStore) -> dict[str, int]:
    """Deduplicate commitment claims by fuzzy matching.

    Match on first 25 chars of `what` + `deadline` + `who`.
    Compatible types merge (keep higher confidence, supersede lower).

    Returns: {"merged": N, "total": N}
    """
    rows = store.conn.execute(
        """SELECT id, subject, object, confidence
           FROM claims
           WHERE claim_type = 'commitment'
             AND superseded_by IS NULL
           ORDER BY confidence DESC"""
    ).fetchall()

    if len(rows) < 2:
        return {"merged": 0, "total": len(rows)}

    parsed = []
    for r in rows:
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue
        parsed.append({
            "id": r["id"],
            "subject": r["subject"],
            "what": obj.get("what", ""),
            "who": obj.get("who", ""),
            "deadline": obj.get("deadline"),
            "type": obj.get("type", ""),
            "confidence": r["confidence"],
        })

    merged = 0
    superseded_ids: set[str] = set()

    for i, a in enumerate(parsed):
        if a["id"] in superseded_ids:
            continue
        for j in range(i + 1, len(parsed)):
            b = parsed[j]
            if b["id"] in superseded_ids:
                continue

            # Check what prefix match
            a_prefix = a["what"][:DEDUP_WHAT_PREFIX_LEN].lower().strip()
            b_prefix = b["what"][:DEDUP_WHAT_PREFIX_LEN].lower().strip()

            prefix_match = a_prefix == b_prefix

            # Token overlap check
            a_tokens = _normalize_tokens(a["what"])
            b_tokens = _normalize_tokens(b["what"])
            token_sim = _jaccard_similarity(a_tokens, b_tokens)

            same_thread = a["subject"] == b["subject"]

            if same_thread:
                # Intra-thread: only dedup near-identical items (re-run artifacts).
                # Genuinely different commitments in the same thread are kept.
                if token_sim < 0.7:
                    continue
            else:
                # Cross-thread dedup: use prefix match or token overlap threshold
                if not prefix_match and token_sim < DEDUP_TOKEN_OVERLAP_THRESHOLD:
                    continue

            # Check deadline + who match
            if a["deadline"] != b["deadline"]:
                continue
            if a["who"] != b["who"]:
                continue

            # Check type compatibility
            types = frozenset({a["type"], b["type"]})
            if a["type"] != b["type"] and types not in COMPATIBLE_TYPES:
                continue

            # Merge: keep higher confidence (a is first = higher), supersede b
            store.conn.execute(
                "UPDATE claims SET superseded_by = ? WHERE id = ?",
                (a["id"], b["id"]),
            )
            superseded_ids.add(b["id"])
            merged += 1

    if merged:
        store.conn.commit()

    return {"merged": merged, "total": len(parsed)}


def expire_stale_commitments(store: LayeredGraphStore) -> int:
    """Mark commitment claims as stale based on deadline + grace period.

    - Deadline passed + grace days -> superseded
    - No deadline + stale days -> superseded
    """
    now = int(time.time())
    today = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")

    rows = store.conn.execute(
        """SELECT id, object, created_at FROM claims
           WHERE claim_type = 'commitment'
             AND superseded_by IS NULL
             AND json_extract(object, '$.status') = 'open'"""
    ).fetchall()

    stale_ids = []
    for r in rows:
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue

        deadline = obj.get("deadline")
        if deadline:
            try:
                deadline_dt = datetime.strptime(deadline, "%Y-%m-%d")
                today_dt = datetime.strptime(today, "%Y-%m-%d")
                days_past = (today_dt - deadline_dt).days
                if days_past > COMMITMENT_DEADLINE_GRACE_DAYS:
                    stale_ids.append(r["id"])
            except ValueError:
                pass
        else:
            age_days = (now - r["created_at"]) / SECONDS_PER_DAY
            if age_days > COMMITMENT_NO_DEADLINE_STALE_DAYS:
                stale_ids.append(r["id"])

    if stale_ids:
        sentinel = f"stale:{now}"
        placeholders = ",".join("?" * len(stale_ids))
        store.conn.execute(
            f"""UPDATE claims SET superseded_by = ?
                WHERE id IN ({placeholders})""",
            [sentinel] + stale_ids,
        )
        store.conn.commit()
        logger.info("Expired %d stale commitment claims", len(stale_ids))

    return len(stale_ids)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Commitment resolution: detect completion evidence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_RESOLUTION_SYSTEM_PROMPT = """\
You are checking whether a commitment/task has been completed.

You will receive:
1. A COMMITMENT: what was promised, by whom, to whom, deadline
2. CANDIDATE EVIDENCE: recent events from related threads/people after the deadline

Decide: does the evidence show this commitment was fulfilled or completed?

Rules:
- "service is complete" or "thank you for your visit" = completed
- A payment transaction matching the commitment = completed
- A calendar event matching the commitment that already occurred = completed
- Evidence must be ABOUT the same thing as the commitment, not just from the same person
- An unrelated email from the same company does NOT count
- Be conservative: if unsure, say not resolved

Respond with JSON:
{"resolved": true/false, "evidence": "one sentence explaining why"}"""

_RESOLUTION_PROMPT_TEMPLATE = """\
COMMITMENT:
- What: {what}
- Who: {who}
- To whom: {to_whom}
- Deadline: {deadline}

CANDIDATE EVIDENCE (events after {deadline}):
{evidence_block}

Is this commitment resolved?"""


def _gather_candidate_evidence(
    store: LayeredGraphStore,
    claim_id: str,
    thread_id: str,
    what: str,
    to_whom: str,
    deadline_ts: int,
    now: int,
) -> list[dict]:
    """Gather candidate follow-up events that might show completion.

    Searches three sources:
    1. Same thread after deadline
    2. Same person/org after deadline
    3. Calendar events matching the commitment
    Returns list of {source, subject, preview, timestamp}.
    """
    candidates: list[dict] = []
    seen_ids: set[str] = set()

    def _add(rows: list) -> None:
        for evt in rows:
            eid = evt["id"] if "id" in evt.keys() else ""
            if eid in seen_ids:
                continue
            seen_ids.add(eid)
            raw = evt["raw_content"] if "raw_content" in evt.keys() else ""
            subj = evt["subj"] if "subj" in evt.keys() else ""
            preview = ((raw or "")[:300] + " " + (subj or ""))
            candidates.append({
                "source": evt["source"],
                "subject": subj or "",
                "preview": preview.strip()[:400],
            })

    # Source 1: Same thread
    _add(store.conn.execute(
        """SELECT id, raw_content, timestamp, source,
                  json_extract(metadata, '$.subject') as subj
           FROM events
           WHERE json_extract(metadata, '$.thread_id') = ?
             AND timestamp > ?
           ORDER BY timestamp ASC LIMIT 10""",
        (thread_id, deadline_ts),
    ).fetchall())

    # Source 2: Same person/org
    claim_event_ids = [
        r["event_id"] for r in store.conn.execute(
            "SELECT event_id FROM claim_events WHERE claim_id = ?",
            (claim_id,),
        ).fetchall()
    ]

    person_ids: list[str] = []
    if claim_event_ids:
        ph = ",".join("?" * len(claim_event_ids))
        person_ids = [
            r["person_id"] for r in store.conn.execute(
                f"""SELECT DISTINCT ep.person_id FROM event_persons ep
                    JOIN persons p ON ep.person_id = p.person_id
                    WHERE ep.event_id IN ({ph}) AND NOT p.is_user""",
                claim_event_ids,
            ).fetchall()
        ]

    # Also match to_whom by keyword against person names
    if to_whom and not person_ids:
        to_words = [
            w for w in to_whom.lower().split()
            if len(w) > 4 and w not in {
                "auto", "group", "service", "services", "care",
                "customer", "inc.", "llc", "corp", "the",
            }
        ]
        for word in to_words[:2]:
            person_ids.extend(
                r["person_id"] for r in store.conn.execute(
                    """SELECT person_id FROM persons
                       WHERE LOWER(canonical_name) LIKE ?
                       AND NOT is_user LIMIT 5""",
                    (f"%{word}%",),
                ).fetchall()
            )
        person_ids = list(set(person_ids))

    for pid in person_ids[:5]:
        _add(store.conn.execute(
            """SELECT e.id, e.raw_content, e.timestamp, e.source,
                      json_extract(e.metadata, '$.subject') as subj
               FROM events e
               JOIN person_events pe ON e.id = pe.event_id
               WHERE pe.person_id = ?
                 AND e.timestamp > ?
               ORDER BY e.timestamp ASC LIMIT 10""",
            (pid, deadline_ts),
        ).fetchall())

    # Source 3: Calendar events that match the commitment by keywords
    what_lower = what.lower()
    what_words = set(what_lower.split()) - {
        "the", "a", "an", "to", "for", "by", "with", "from", "in", "on",
        "at", "of", "and", "or", "user", "attend", "complete", "schedule",
        "pay", "review", "check", "send", "make",
    }
    if what_words:
        for cal in store.conn.execute(
            """SELECT id, json_extract(metadata, '$.title') as subj,
                      '' as raw_content, timestamp, 'calendar' as source
               FROM events
               WHERE source = 'calendar'
                 AND timestamp >= ? AND timestamp <= ?""",
            (deadline_ts - 86400, now),
        ).fetchall():
            cal_words = set((cal["subj"] or "").lower().split()) - {
                "the", "a", "an", "to", "for", "by", "with", "from",
                "in", "on", "at", "of", "and", "or",
            }
            if len(what_words & cal_words) >= 2:
                _add([cal])

    return candidates[:15]


def resolve_completed_commitments(
    store: LayeredGraphStore,
    llm_client: LLMClient | None = None,
    model: str = "",
) -> dict[str, Any]:
    """Detect and resolve commitments with completion evidence.

    Hybrid approach:
    1. Deterministic: gather candidate follow-up events per commitment
    2. LLM gate (Flash Lite): binary resolved/not-resolved + evidence
    3. Mark resolved commitments and beliefs, record checked_until

    If llm_client is None, skips the LLM gate (no-op).
    """
    if not llm_client:
        return {"resolved": 0, "checked": 0, "details": []}

    now = int(time.time())
    today = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")

    rows = store.conn.execute(
        """SELECT id, subject AS thread_id, object FROM claims
           WHERE claim_type = 'commitment'
             AND superseded_by IS NULL
             AND json_extract(object, '$.status') = 'open'
             AND json_extract(object, '$.deadline') IS NOT NULL
             AND json_extract(object, '$.deadline') < ?""",
        (today,),
    ).fetchall()

    if not rows:
        return {"resolved": 0, "checked": 0, "details": []}

    resolved_claims: list[tuple[str, str, str]] = []
    checked = 0

    for r in rows:
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue

        claim_id = r["id"]
        thread_id = r["thread_id"]
        what = obj.get("what", "")
        who = obj.get("who", "user")
        deadline = obj.get("deadline", "")
        to_whom = obj.get("to_whom") or ""

        # Skip if already checked recently (within 24h)
        checked_until = obj.get("resolution_checked_until", 0)
        if now - checked_until < 86400:
            continue

        try:
            deadline_ts = int(
                datetime.strptime(deadline, "%Y-%m-%d")
                .replace(tzinfo=timezone.utc)
                .timestamp()
            )
        except ValueError:
            continue

        # Step 1: Gather candidate evidence
        candidates = _gather_candidate_evidence(
            store, claim_id, thread_id, what, to_whom, deadline_ts, now,
        )

        # Record that we checked, even if no candidates
        obj["resolution_checked_until"] = now
        store.conn.execute(
            "UPDATE claims SET object = ? WHERE id = ?",
            (json.dumps(obj), claim_id),
        )
        checked += 1

        if not candidates:
            continue

        # Step 2: LLM binary gate
        evidence_block = "\n".join(
            f"- [{c['source']}] {c['subject']}: {c['preview'][:200]}"
            for c in candidates
        )
        prompt = _RESOLUTION_PROMPT_TEMPLATE.format(
            what=what, who=who, to_whom=to_whom,
            deadline=deadline, evidence_block=evidence_block,
        )

        try:
            res_schema = _build_resolution_gate_schema()
        except ImportError:
            res_schema = None

        raw_str = llm_client.generate(
            prompt=prompt,
            system=_RESOLUTION_SYSTEM_PROMPT,
            model=model,
            temperature=0.0,
            cache_system=True,
            response_schema=res_schema,
            format_json=res_schema is None,
        )

        try:
            raw = json.loads(raw_str) if raw_str else None
        except (json.JSONDecodeError, TypeError):
            raw = None

        if not raw:
            continue

        resolved = raw.get("resolved", False)
        if isinstance(resolved, str):
            resolved = resolved.lower() in ("true", "yes", "1")
        evidence_text = str(raw.get("evidence", ""))

        if resolved and evidence_text:
            resolved_claims.append((claim_id, what, evidence_text))

    # Apply resolutions
    for claim_id, what, evidence in resolved_claims:
        row = store.conn.execute(
            "SELECT object FROM claims WHERE id = ?", (claim_id,),
        ).fetchone()
        if row:
            obj = json.loads(row["object"])
            obj["status"] = "done"
            obj["resolution_evidence"] = evidence
            store.conn.execute(
                "UPDATE claims SET object = ? WHERE id = ?",
                (json.dumps(obj), claim_id),
            )

        store.conn.execute(
            """UPDATE beliefs SET status = 'resolved', updated_at = ?
               WHERE status = 'active'
                 AND belief_type = 'fact'
                 AND source_claims LIKE ?""",
            (now, f'%{claim_id}%'),
        )

    if resolved_claims:
        store.conn.commit()
        logger.info(
            "Resolved %d commitments with completion evidence",
            len(resolved_claims),
        )
    elif checked:
        store.conn.commit()

    return {
        "resolved": len(resolved_claims),
        "checked": checked,
        "details": [
            {"what": what, "evidence": evidence}
            for _, what, evidence in resolved_claims
        ],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Downstream queries (moved from old extract.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_open_commitments(store: LayeredGraphStore) -> list[dict]:
    """Get all open commitment claims."""
    rows = store.conn.execute(
        """SELECT id, subject, object, confidence
           FROM claims
           WHERE claim_type = 'commitment'
             AND superseded_by IS NULL
             AND json_extract(object, '$.status') = 'open'
           ORDER BY json_extract(object, '$.priority') ASC,
                    confidence DESC"""
    ).fetchall()

    results = []
    for r in rows:
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue
        results.append({
            "id": r["id"],
            "thread_id": r["subject"],
            "what": obj.get("what", ""),
            "who": obj.get("who", ""),
            "to_whom": obj.get("to_whom", ""),
            "deadline": obj.get("deadline"),
            "type": obj.get("type", ""),
            "action_type": obj.get("action_type", "user_owes_action"),
            "priority": obj.get("priority", 3),
            "confidence": r["confidence"],
            "direction": obj.get("direction", "ambiguous"),
            "staleness_signal": obj.get("staleness_signal", "none"),
        })
    return results


def get_overdue_commitments(store: LayeredGraphStore) -> list[dict]:
    """Get open commitments past their deadline."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    all_open = get_open_commitments(store)
    return [
        c for c in all_open
        if c.get("deadline") and c["deadline"] < today
    ]


def get_commitments_for_thread(
    store: LayeredGraphStore, thread_id: str,
) -> list[dict]:
    """Get commitment claims for a specific thread."""
    all_open = get_open_commitments(store)
    return [c for c in all_open if c.get("thread_id") == thread_id]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entity resolution -> entity beliefs (deterministic, no LLM)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_entity_beliefs(
    store: LayeredGraphStore,
) -> list[Belief]:
    """Produce entity beliefs for all known persons.

    Synthesizes from persons table + stage1 claims to create
    entity beliefs without requiring an LLM call.
    """
    now = int(time.time())
    beliefs: list[Belief] = []

    persons = store.get_all_persons()

    for person in persons:
        pid = person["person_id"]
        name = person.get("canonical_name") or pid[:12]
        is_user = bool(person.get("is_user"))

        # Gather identifiers
        identifiers = store.get_person_identifiers(pid)
        emails = [
            i["identifier"] for i in identifiers
            if i["identifier_type"] == "email"
        ]
        phones = [
            i["identifier"] for i in identifiers
            if i["identifier_type"] == "phone"
        ]

        # Gather source claims
        source_claims = []
        freq_claims = store.conn.execute(
            """SELECT id, object FROM claims
               WHERE claim_type = 'communication_frequency'
                 AND predicate LIKE ?
                 AND superseded_by IS NULL""",
            (f"communicates_with:{pid}",),
        ).fetchall()

        msg_count = 0
        for fc in freq_claims:
            source_claims.append(fc["id"])
            try:
                obj = json.loads(fc["object"])
                msg_count = obj.get("event_count", 0)
            except (json.JSONDecodeError, TypeError):
                pass

        # Determine tier
        if msg_count >= 50:
            tier = 1
        elif msg_count >= 10:
            tier = 2
        else:
            tier = 3

        data = {
            "name": name,
            "is_user": is_user,
            "tier": tier,
            "message_count": msg_count,
            "emails": emails[:5],
            "phones": phones[:3],
            "sources": json.loads(person.get("sources", "[]")),
        }

        # Confidence based on evidence volume (Admiralty framework)
        from alteris.confidence import compute_confidence
        confidence = compute_confidence("system_sql", msg_count, evidence_scale=30)
        if is_user:
            confidence = 1.0

        summary = (
            f"{name} is the user" if is_user
            else f"{name}: tier-{tier} contact ({msg_count} messages)"
        )

        bid = _belief_id("entity", pid, summary)
        beliefs.append(Belief(
            id=bid,
            belief_type=BeliefType.ENTITY,
            subject=pid,
            summary=summary,
            data=data,
            epistemic_level=EpistemicLevel.COMPUTED,
            source_reliability="B",
            info_credibility=2,
            confidence=confidence,
            source_claims=source_claims,
            inference_chain=[
                f"Person {name} found in persons table",
                f"Communication frequency: {msg_count} events",
                f"Contact tier: {tier}",
            ],
            evidence_log=[{
                "timestamp": now,
                "event": "entity_resolved",
                "delta": confidence,
            }],
            status=BeliefStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        ))

    return beliefs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Participant inference -> observation beliefs (from call-meeting overlap)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_participant_inference_beliefs(
    store: LayeredGraphStore,
) -> list[Belief]:
    """Produce OBSERVATION beliefs for meeting participants inferred from calls.

    Uses infer_meeting_participants_from_calls() to find temporal overlaps,
    then creates beliefs with Admiralty-grounded confidence scores.
    """
    from alteris.confidence import RELIABILITY, SOURCE_RELIABILITY
    from alteris.linker import infer_meeting_participants_from_calls

    now = int(time.time())
    inferences = infer_meeting_participants_from_calls(store)
    beliefs: list[Belief] = []

    for inf in inferences:
        fraction = inf.overlap_seconds / inf.call_duration if inf.call_duration > 0 else 0

        # Map to Admiralty credibility number (1-6 scale)
        if fraction >= 0.90:
            cred_number = 2  # Probably true
        elif fraction >= 0.50:
            cred_number = 3  # Possibly true
        else:
            cred_number = 4  # Doubtful

        rel_letter = SOURCE_RELIABILITY.get(inf.event_source, "B")

        # Get meeting subject for the summary
        event_row = store.conn.execute(
            "SELECT metadata FROM events WHERE id = ?", (inf.event_id,),
        ).fetchone()
        event_subject = ""
        if event_row:
            try:
                meta = json.loads(event_row["metadata"] or "{}")
                event_subject = meta.get("subject", "") or meta.get("title", "")
            except (json.JSONDecodeError, TypeError):
                pass

        event_label = event_subject or inf.event_id[:12]
        summary = (
            f"{inf.person_name} was likely in '{event_label}' "
            f"(inferred from overlapping {inf.call_source} call)"
        )

        bid = _belief_id("observation", inf.event_id, f"participant:{inf.person_id}")
        beliefs.append(Belief(
            id=bid,
            belief_type=BeliefType.OBSERVATION,
            subject=inf.event_id,
            summary=summary,
            data={
                "inference_type": "call_meeting_overlap",
                "person_id": inf.person_id,
                "person_name": inf.person_name,
                "meeting_event_id": inf.event_id,
                "call_event_id": inf.call_event_id,
                "overlap_seconds": inf.overlap_seconds,
                "overlap_fraction": round(fraction, 3),
                "call_duration_seconds": inf.call_duration,
                "call_source": inf.call_source,
                "event_source": inf.event_source,
            },
            epistemic_level=EpistemicLevel.INFERENCE,
            source_reliability=rel_letter,
            info_credibility=cred_number,
            confidence=inf.confidence,
            source_claims=[],
            inference_chain=[
                f"Call {inf.call_event_id[:12]}: {inf.person_name} for {inf.call_duration:.0f}s",
                f"Event {inf.event_id[:12]}: overlaps by {inf.overlap_seconds:.0f}s ({fraction:.0%})",
                f"Admiralty: {rel_letter}{cred_number} → confidence {inf.confidence}",
                "Temporal overlap implies co-presence",
            ],
            evidence_log=[{
                "timestamp": now,
                "event": "call_meeting_overlap_inferred",
                "delta": inf.confidence,
            }],
            status=BeliefStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        ))

    return beliefs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Relation inference -> relation beliefs (deterministic, no LLM)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_relation_beliefs(
    store: LayeredGraphStore,
) -> list[Belief]:
    """Produce relation beliefs from channel + directionality claims.

    Creates relation beliefs like "user communicates_with X" with
    channel details and directionality info.
    """
    now = int(time.time())
    beliefs: list[Belief] = []

    # Get the user person_id
    user_row = store.conn.execute(
        "SELECT person_id FROM persons WHERE is_user = 1 LIMIT 1"
    ).fetchone()
    if not user_row:
        return []
    user_id = user_row["person_id"]

    # Load channel claims
    channel_rows = store.conn.execute(
        """SELECT id, predicate, object, confidence FROM claims
           WHERE claim_type = 'communication_channel'
             AND superseded_by IS NULL"""
    ).fetchall()

    for r in channel_rows:
        pred = r["predicate"]
        if not pred.startswith("channels_with:"):
            continue
        person_id = pred.replace("channels_with:", "")

        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue

        name = obj.get("person_name", person_id[:12])
        channels = obj.get("channels", [])
        channel_count = obj.get("channel_count", 0)

        # Look up directionality
        dir_row = store.conn.execute(
            """SELECT id, object FROM claims
               WHERE claim_type = 'directionality'
                 AND predicate = ?
                 AND superseded_by IS NULL""",
            (f"direction_with:{person_id}",),
        ).fetchone()

        source_claims = [r["id"]]
        user_initiated_ratio = 0.5
        if dir_row:
            source_claims.append(dir_row["id"])
            try:
                dir_obj = json.loads(dir_row["object"])
                user_initiated_ratio = dir_obj.get(
                    "user_initiated_ratio", 0.5
                )
            except (json.JSONDecodeError, TypeError):
                pass

        relation_type = "communicates_with"
        if channel_count >= 3:
            relation_type = "multi_channel_contact"
        elif user_initiated_ratio > 0.7:
            relation_type = "user_initiates_with"
        elif user_initiated_ratio < 0.3:
            relation_type = "receives_from"

        data = {
            "from": user_id,
            "to": person_id,
            "relation_type": relation_type,
            "channels": channels,
            "channel_count": channel_count,
            "user_initiated_ratio": user_initiated_ratio,
        }

        summary = (
            f"User {relation_type.replace('_', ' ')} {name} "
            f"via {', '.join(channels[:3])}"
        )

        bid = _belief_id("relation", f"{user_id}:{person_id}", summary)
        beliefs.append(Belief(
            id=bid,
            belief_type=BeliefType.RELATION,
            subject=f"{user_id}:{person_id}",
            summary=summary,
            data=data,
            epistemic_level=EpistemicLevel.COMPUTED,
            source_reliability="B",
            info_credibility=3,
            confidence=r["confidence"],
            source_claims=source_claims,
            inference_chain=[
                f"Channel claim shows {channel_count} channels: {channels}",
                f"Directionality: user initiates {user_initiated_ratio:.0%}",
                f"Classified as: {relation_type}",
            ],
            status=BeliefStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        ))

    return beliefs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Commitment claims -> FACT beliefs (the missing compilation layer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _build_commitment_beliefs(
    store: LayeredGraphStore,
) -> list[Belief]:
    """Compile open commitment claims into FACT beliefs with intelligent dedup.

    This is the missing layer: commitment claims currently bypass beliefs
    entirely. This function groups claims by (who, deadline) exact match,
    clusters by fuzzy what-similarity, and produces one FACT belief per
    cluster with full merge provenance.

    Merge decisions are recorded in the evidence_log so the system can
    explain why two claims were considered the same commitment.
    """
    now = int(time.time())

    rows = store.conn.execute(
        """SELECT id, subject, object, confidence, created_at
           FROM claims
           WHERE claim_type = 'commitment'
             AND (superseded_by IS NULL OR superseded_by = '')
             AND json_extract(object, '$.status') = 'open'"""
    ).fetchall()

    if not rows:
        return []

    # Parse into working dicts
    claims_data: list[dict] = []
    for r in rows:
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue
        claims_data.append({
            "claim_id": r["id"],
            "thread_id": r["subject"],
            "what": obj.get("what", ""),
            "who": obj.get("who", "user"),
            "to_whom": obj.get("to_whom"),
            "deadline": obj.get("deadline"),
            "type": obj.get("type", "inbound_request"),
            "action_type": obj.get("action_type", "user_owes_action"),
            "priority": obj.get("priority", 2),
            "confidence": r["confidence"],
            "direction": obj.get("direction", "ambiguous"),
            "note": obj.get("note"),
            "next_action": obj.get("next_action"),
            "evidence_quote": obj.get("evidence_quote"),
            "assignee": obj.get("assignee"),
            "deferred_until": obj.get("deferred_until"),
            "custom_fields": obj.get("custom_fields"),
            "created_at": r["created_at"],
            "source_event_id": obj.get("source_event_id"),
        })

    # Group by (who, deadline) exact match
    exact_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for cd in claims_data:
        key = (cd["who"], cd["deadline"] or "")
        exact_groups[key].append(cd)

    # Within each group, cluster by fuzzy what-similarity
    beliefs: list[Belief] = []

    for _group_key, group_claims in exact_groups.items():
        clusters, merge_logs = cluster_by_similarity(
            group_claims,
            key_fn=lambda c: c["what"],
        )

        for cluster, merge_log in zip(clusters, merge_logs):
            # Canonical = highest confidence claim
            canonical = max(cluster, key=lambda c: c["confidence"])

            all_claim_ids = [c["claim_id"] for c in cluster]
            all_thread_ids = sorted({c["thread_id"] for c in cluster})
            max_confidence = max(c["confidence"] for c in cluster)
            min_priority = min(c["priority"] for c in cluster)

            data: dict[str, Any] = {
                "assertion_type": "commitment",
                "type": canonical["type"],
                "action_type": canonical.get("action_type", "user_owes_action"),
                "who": canonical["who"],
                "what": canonical["what"],
                "to_whom": canonical.get("to_whom"),
                "deadline": canonical["deadline"],
                "priority": min_priority,
                "direction": canonical["direction"],
                "note": canonical.get("note"),
                "next_action": canonical.get("next_action"),
                "evidence_quote": canonical.get("evidence_quote"),
                "assignee": canonical.get("assignee"),
                "deferred_until": canonical.get("deferred_until"),
                "custom_fields": canonical.get("custom_fields"),
                "thread_ids": all_thread_ids,
                "claim_count": len(cluster),
            }

            # Record merge provenance when multiple claims merged
            if len(cluster) > 1:
                data["merged_whats"] = [
                    c["what"] for c in cluster
                    if c["what"] != canonical["what"]
                ]
                data["merge_log"] = merge_log

            who = canonical["who"]
            what_short = canonical["what"][:80]
            deadline_str = (
                f" by {canonical['deadline']}" if canonical["deadline"] else ""
            )
            summary = f"{who}: {what_short}{deadline_str}"

            bid = _belief_id(
                "fact",
                f"commitment:{canonical['who']}",
                canonical["what"][:50],
            )

            chain = [f"Compiled from {len(cluster)} commitment claim(s)"]
            if len(cluster) > 1:
                chain.append(
                    f"Merged across {len(all_thread_ids)} thread(s)"
                )
                for mr in merge_log:
                    chain.append(
                        f"Matched: {', '.join(mr['experts'])}"
                    )
            chain.append(f"Canonical claim: {canonical['claim_id'][:12]}")

            evidence = [{
                "timestamp": now,
                "event": "fact_compiled",
                "delta": max_confidence,
                "sources": len(cluster),
            }]

            # Expiry based on deadline + grace period
            expires_at = None
            if canonical["deadline"]:
                try:
                    dl = datetime.strptime(canonical["deadline"], "%Y-%m-%d")
                    expires_at = (
                        int(dl.replace(tzinfo=timezone.utc).timestamp())
                        + BELIEF_COMMITMENT_EXPIRY_DAYS * SECONDS_PER_DAY
                    )
                except ValueError:
                    pass

            beliefs.append(Belief(
                id=bid,
                belief_type=BeliefType.FACT,
                subject=f"commitment:{canonical['who']}",
                summary=summary,
                data=data,
                epistemic_level=EpistemicLevel.INFERENCE,
                source_reliability="B",
                info_credibility=3,
                confidence=max_confidence,
                source_claims=all_claim_ids,
                inference_chain=chain,
                evidence_log=evidence,
                status=BeliefStatus.ACTIVE,
                priority=min_priority,
                created_at=now,
                updated_at=now,
                expires_at=expires_at,
            ))

    return _cross_group_dedup_beliefs(beliefs)


def _cross_group_dedup_beliefs(beliefs: list[Belief]) -> list[Belief]:
    """Second-pass dedup across commitment beliefs grouped by to_whom.

    The first pass (cluster_by_similarity within each (who, deadline) group)
    catches intra-group dupes. This second pass catches cross-group dupes
    where the same obligation has different deadlines or phrasing, e.g.:
      - "pay BOA minimum payment" vs "pay BOA credit card bill"
      - "send growth strategy deck" vs "send deck covering growth strategy"

    Uses the same 3-expert voting (prefix, Jaccard, SeqMatch) as the
    first pass to avoid false merges.
    """
    if len(beliefs) < 2:
        return beliefs

    # Group by (who, to_whom) — only compare beliefs from the same actor
    by_recipient: dict[str, list[int]] = defaultdict(list)
    for i, b in enumerate(beliefs):
        who = (b.data.get("who") or "user").lower().strip()
        to_whom = b.data.get("to_whom") or ""
        if to_whom in ("unresolved", "null", ""):
            to_whom = ""
        by_recipient[f"{who}|{to_whom.lower().strip()}"].append(i)

    superseded: set[int] = set()

    for _key, indices in by_recipient.items():
        if len(indices) < 2:
            continue

        # Pairwise 3-expert voting on 'what' text
        whats = [beliefs[idx].data.get("what", "") for idx in indices]
        token_sets = [_normalize_tokens(w) for w in whats]

        for i_pos in range(len(indices)):
            idx_a = indices[i_pos]
            if idx_a in superseded:
                continue
            for j_pos in range(i_pos + 1, len(indices)):
                idx_b = indices[j_pos]
                if idx_b in superseded:
                    continue

                what_a, what_b = whats[i_pos], whats[j_pos]

                # Expert 1: prefix match
                prefix_vote = (
                    what_a[:DEDUP_WHAT_PREFIX_LEN].lower().strip()
                    == what_b[:DEDUP_WHAT_PREFIX_LEN].lower().strip()
                )
                # Expert 2: Jaccard (stop words removed)
                jac_score = _jaccard_similarity(
                    token_sets[i_pos], token_sets[j_pos],
                )
                jaccard_vote = jac_score >= BELIEF_JACCARD_VOTE_THRESHOLD
                # Expert 3: SequenceMatcher
                seq_score = SequenceMatcher(
                    None, what_a.lower(), what_b.lower(),
                ).ratio()
                seqmatch_vote = seq_score >= BELIEF_SEQMATCH_VOTE_THRESHOLD

                votes = int(prefix_vote) + int(jaccard_vote) + int(seqmatch_vote)
                if votes >= BELIEF_MERGE_MIN_VOTES:
                    ba, bb = beliefs[idx_a], beliefs[idx_b]

                    # Guard: don't merge if both have distinct deadlines
                    dl_a = ba.data.get("deadline")
                    dl_b = bb.data.get("deadline")
                    if dl_a and dl_b and dl_a != dl_b:
                        continue

                    # Keep higher confidence, supersede other
                    if ba.confidence >= bb.confidence:
                        superseded.add(idx_b)
                        logger.info(
                            "Cross-group dedup: keeping %r (%.2f), "
                            "superseding %r (%.2f) [votes=%d]",
                            ba.data.get("what", "")[:50],
                            ba.confidence,
                            bb.data.get("what", "")[:50],
                            bb.confidence,
                            votes,
                        )
                    else:
                        superseded.add(idx_a)
                        logger.info(
                            "Cross-group dedup: keeping %r (%.2f), "
                            "superseding %r (%.2f) [votes=%d]",
                            bb.data.get("what", "")[:50],
                            bb.confidence,
                            ba.data.get("what", "")[:50],
                            ba.confidence,
                            votes,
                        )
                        break  # idx_a is gone, move to next i_pos

    if superseded:
        logger.info(
            "Cross-group dedup removed %d/%d commitment beliefs",
            len(superseded), len(beliefs),
        )

    return [b for i, b in enumerate(beliefs) if i not in superseded]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logistics claims -> FACT beliefs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _logistics_summary(obj: dict, fact_type: str) -> str:
    """Build a human-readable summary for a logistics fact."""
    if fact_type == "reservation":
        return (
            f"Reservation at {obj.get('venue', '?')} on "
            f"{obj.get('date', '?')} at {obj.get('time', '?')}"
        )
    if fact_type == "travel":
        return (
            f"Travel to {obj.get('destination', '?')} on "
            f"{obj.get('date', '?')}"
        )
    if fact_type == "care_provider":
        return (
            f"{obj.get('provider', '?')} on {obj.get('date', '?')} "
            f"{obj.get('hours', '')}".strip()
        )
    if fact_type == "appointment":
        return (
            f"Appt with {obj.get('provider', '?')} on "
            f"{obj.get('date', '?')}"
        )
    if fact_type == "activity":
        return f"{obj.get('name', '?')} on {obj.get('dates', '?')}"
    if fact_type == "outing":
        parts = [obj.get("name", "?")]
        if obj.get("location"):
            parts.append(f"at {obj['location']}")
        parts.append(f"on {obj.get('date', '?')}")
        return " ".join(parts)
    if fact_type == "childcare":
        return (
            f"{obj.get('child', '?')} at {obj.get('facility', '?')} "
            f"on {obj.get('date', '?')}"
        )
    return obj.get("summary", str(obj)[:80])


def _logistics_expiry(obj: dict) -> int | None:
    """Compute expiry timestamp for a logistics fact from its date field."""
    date_str = obj.get("date")
    if not date_str:
        return None
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        # Expire 3 days after the event date
        return int(dt.replace(tzinfo=timezone.utc).timestamp()) + 3 * SECONDS_PER_DAY
    except ValueError:
        return None


def _build_logistics_beliefs(
    store: LayeredGraphStore,
) -> list[Belief]:
    """Compile logistics claims into FACT beliefs.

    Each logistics claim (reservation, travel, care_provider, etc.) becomes
    a FACT belief. Claims about the same logistics event (discussed in
    multiple threads) are merged via similarity clustering.
    """
    now = int(time.time())

    rows = store.conn.execute(
        """SELECT id, subject, predicate, object, confidence, created_at
           FROM claims
           WHERE claim_type = 'logistics'
             AND (superseded_by IS NULL OR superseded_by = '')"""
    ).fetchall()

    if not rows:
        return []

    # Parse and group by logistics type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue
        fact_type = obj.get("type", "unknown")
        summary = _logistics_summary(obj, fact_type)
        by_type[fact_type].append({
            "claim_id": r["id"],
            "thread_id": r["subject"],
            "obj": obj,
            "fact_type": fact_type,
            "summary": summary,
            "confidence": r["confidence"],
            "created_at": r["created_at"],
        })

    beliefs: list[Belief] = []

    for fact_type, type_claims in by_type.items():
        # Cluster by summary similarity (catches same reservation in 2 threads)
        clusters, merge_logs = cluster_by_similarity(
            type_claims,
            key_fn=lambda c: c["summary"],
        )

        for cluster, merge_log in zip(clusters, merge_logs):
            canonical = max(cluster, key=lambda c: c["confidence"])
            all_claim_ids = [c["claim_id"] for c in cluster]
            max_confidence = max(c["confidence"] for c in cluster)

            data: dict[str, Any] = {
                "assertion_type": f"logistics_{fact_type}",
                **canonical["obj"],
            }
            if len(cluster) > 1:
                data["claim_count"] = len(cluster)
                data["thread_ids"] = sorted(
                    {c["thread_id"] for c in cluster}
                )

            bid = _belief_id(
                "fact",
                f"logistics:{fact_type}",
                canonical["summary"][:50],
            )

            chain = [
                f"Compiled from {len(cluster)} logistics claim(s)",
            ]
            if len(cluster) > 1:
                chain.append(
                    f"Merged {len(cluster)} mentions of same {fact_type}"
                )
                for mr in merge_log:
                    chain.append(
                        f"Matched: {', '.join(mr['experts'])}"
                    )
                data["merge_log"] = merge_log

            beliefs.append(Belief(
                id=bid,
                belief_type=BeliefType.FACT,
                subject=f"logistics:{fact_type}",
                summary=canonical["summary"],
                data=data,
                epistemic_level=EpistemicLevel.INFERENCE,
                source_reliability="B",
                info_credibility=2,
                confidence=max_confidence,
                source_claims=all_claim_ids,
                inference_chain=chain,
                evidence_log=[{
                    "timestamp": now,
                    "event": "fact_compiled",
                    "delta": max_confidence,
                }],
                status=BeliefStatus.ACTIVE,
                created_at=now,
                updated_at=now,
                expires_at=_logistics_expiry(canonical["obj"]),
            ))

    return beliefs


def _resolve_first_name(name: str, profile_lookup: dict[str, list[str]]) -> str:
    """Resolve single-word names to canonical full names via person_profiles.

    Only resolves when there's exactly one tier-1/2 person whose first name
    matches. Ambiguous cases (e.g., "Val" -> Valerie Davis + Valerie Martin)
    are left as-is.
    """
    tokens = name.strip().split()
    if len(tokens) != 1:
        return name
    first = tokens[0].lower()
    candidates = profile_lookup.get(first, [])
    if len(candidates) == 1:
        return candidates[0]
    return name


def _build_relational_context_beliefs(
    store: LayeredGraphStore,
) -> list[Belief]:
    """Compile LLM relational_context claims into relation beliefs.

    Each relational_context claim (from extract_relational) carries FOAF
    relationship_tier, role, organization, and context. Multiple claims
    about the same person across threads are merged.

    Tier resolution uses a specificity hierarchy: when merging claims
    across threads, the most specific (closest) tier wins over
    unknown_or_first_contact, regardless of which claim has higher
    confidence. This prevents calendar-invite-only threads from
    overriding team-call threads where the person's role is established.

    Pre-processing:
      - Single-token first names are resolved to canonical full names
        via person_profiles (Fix 2: name fragmentation)
      - User-self entries and generic relationship words are filtered
        out (Fix 3: user-self leaking)
    """
    now = int(time.time())

    rows = store.conn.execute(
        """SELECT id, subject, predicate, object, confidence, created_at
           FROM claims
           WHERE claim_type = 'relational_context'
             AND (superseded_by IS NULL OR superseded_by = '')"""
    ).fetchall()

    if not rows:
        return []

    # --- Fix 2: Build first-name -> canonical-name lookup from person_profiles ---
    profile_rows = store.conn.execute(
        "SELECT canonical_name FROM person_profiles WHERE tier <= 2 AND canonical_name <> ''"
    ).fetchall()
    profile_lookup: dict[str, list[str]] = defaultdict(list)
    for pr in profile_rows:
        cn = pr["canonical_name"]
        first = cn.split()[0].lower() if cn else ""
        if first:
            profile_lookup[first].append(cn)

    # --- Fix 3: Build user-self name set for filtering ---
    user_names: set[str] = set()
    for ur in store.conn.execute(
        "SELECT LOWER(canonical_name) AS cn FROM persons WHERE is_user = 1 AND canonical_name <> ''"
    ).fetchall():
        user_names.add(ur["cn"])
        # Also add individual tokens (e.g., "Ani" from "Alex Chen")
        for token in ur["cn"].split():
            user_names.add(token)
    # Add display_name variants from person_identifiers
    user_person_ids = store.conn.execute(
        "SELECT person_id FROM persons WHERE is_user = 1"
    ).fetchall()
    for uid_row in user_person_ids:
        for idr in store.conn.execute(
            "SELECT display_name FROM person_identifiers WHERE person_id = ? AND display_name <> ''",
            (uid_row["person_id"],),
        ).fetchall():
            user_names.add(idr["display_name"].lower())
    # Also add reversed forms (e.g., "chen alex" from "alex chen")
    reversed_names = set()
    for un in user_names:
        parts = un.split()
        if len(parts) == 2:
            reversed_names.add(f"{parts[1]} {parts[0]}")
    user_names.update(reversed_names)

    # Group by person name (normalized lowercase), with first-name resolution
    # and user-self/generic filtering
    by_person: dict[str, list[dict]] = defaultdict(list)
    skipped_user = 0
    skipped_generic = 0
    for r in rows:
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue
        name = obj.get("name", "").strip()
        if not name:
            continue

        # Fix 3: Skip generic relationship words
        if name.lower() in RELATIONAL_SKIP_GENERIC_NAMES:
            skipped_generic += 1
            continue

        # Fix 2: Resolve first-name fragments to canonical full names
        resolved_name = _resolve_first_name(name, profile_lookup)
        grouping_key = resolved_name.lower()

        # Fix 3: Skip user-self entries
        if grouping_key in user_names:
            skipped_user += 1
            continue

        by_person[grouping_key].append({
            "claim_id": r["id"],
            "thread_id": r["subject"],
            "obj": obj,
            "name": resolved_name if resolved_name != name else name,
            "confidence": r["confidence"],
        })

    if skipped_user:
        logger.info("Filtered %d user-self relational claims", skipped_user)
    if skipped_generic:
        logger.info("Filtered %d generic-name relational claims", skipped_generic)

    # Tier specificity: lower rank = closer relationship = wins merge
    _TIER_RANK = {
        "core_kinship": 0,
        "extended_kinship": 1,
        "intimate_friendship": 2,
        "vocational_core_team": 3,
        "vocational_network": 4,
        "commercial_vendor": 5,
        "unknown_or_first_contact": 6,
    }

    beliefs: list[Belief] = []

    for _key, person_claims in by_person.items():
        all_claim_ids = [c["claim_id"] for c in person_claims]
        max_confidence = max(c["confidence"] for c in person_claims)

        # Pick canonical by tier specificity first, then confidence
        canonical = min(
            person_claims,
            key=lambda c: (
                _TIER_RANK.get(
                    c["obj"].get("relationship_tier", "unknown_or_first_contact"), 6
                ),
                -c["confidence"],
            ),
        )
        name = canonical["name"]

        # For role/org, prefer the claim with the most specific tier
        # (same canonical) — it comes from a richer context thread
        best_role = canonical["obj"].get("role", "unknown")
        best_org = canonical["obj"].get("organization")
        best_strength = canonical["obj"].get("relationship_strength", "moderate")

        # Merge contexts from multiple threads
        contexts = []
        thread_ids = set()
        for c in person_claims:
            ctx = c["obj"].get("context", "")
            if ctx and ctx not in contexts:
                contexts.append(ctx)
            thread_ids.add(c["thread_id"])

        data: dict[str, Any] = {
            "assertion_type": "relational_context",
            "name": name,
            "relationship_tier": canonical["obj"].get(
                "relationship_tier", "unknown_or_first_contact"
            ),
            "role": best_role,
            "organization": best_org,
            "context": " | ".join(contexts) if len(contexts) > 1 else (contexts[0] if contexts else ""),
            "relationship_strength": best_strength,
            "thread_ids": sorted(thread_ids),
            "claim_count": len(person_claims),
        }

        summary = (
            f"{name}: {data['relationship_tier']}, "
            f"{data['role']}"
            + (f" at {data['organization']}" if data.get("organization") else "")
        )

        bid = _belief_id("relation", f"foaf:{name.lower()}", summary[:50])

        chain = [f"Compiled from {len(person_claims)} relational claim(s)"]
        if len(person_claims) > 1:
            chain.append(
                f"Seen in {len(thread_ids)} threads"
            )

        beliefs.append(Belief(
            id=bid,
            belief_type=BeliefType.RELATION,
            subject=name,
            summary=summary,
            data=data,
            epistemic_level=EpistemicLevel.INFERENCE,
            source_reliability="B",
            info_credibility=2,
            confidence=max_confidence,
            source_claims=all_claim_ids,
            inference_chain=chain,
            evidence_log=[{
                "timestamp": now,
                "event": "relational_compiled",
                "delta": max_confidence,
            }],
            status=BeliefStatus.ACTIVE,
            created_at=now,
            updated_at=now,
        ))

    return beliefs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Supersede stale beliefs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _expire_stale_beliefs(store: LayeredGraphStore) -> int:
    """Mark beliefs past their expires_at as stale."""
    now = int(time.time())
    cursor = store.conn.execute(
        """UPDATE beliefs SET status = 'stale', updated_at = ?
           WHERE status = 'active'
             AND expires_at IS NOT NULL
             AND expires_at < ?""",
        (now, now),
    )
    store.conn.commit()
    count = cursor.rowcount
    if count:
        logger.info("Expired %d stale beliefs", count)
    return count


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reconstruct ThreadBundles from store
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _reconstruct_bundle(
    thread_id: str,
    event_ids: list[str],
    store: LayeredGraphStore,
) -> ThreadBundle | None:
    """Reconstruct a ThreadBundle from stored events."""
    if not event_ids:
        return None

    placeholders = ",".join("?" * len(event_ids))
    rows = store.conn.execute(
        f"""SELECT id, source, source_id, event_type, timestamp,
                   participants, raw_content, metadata, content_hash
            FROM events
            WHERE id IN ({placeholders})
            ORDER BY timestamp ASC""",
        event_ids,
    ).fetchall()

    if not rows:
        return None

    events = []
    for r in rows:
        events.append(Event(
            id=r["id"],
            source=r["source"],
            source_id=r["source_id"],
            event_type=r["event_type"],
            timestamp=r["timestamp"],
            participants=tuple(json.loads(r["participants"] or "[]")),
            raw_content=r["raw_content"] or "",
            metadata=json.loads(r["metadata"] or "{}"),
            content_hash=r["content_hash"] or "",
        ))

    # Load triage data for these events
    triage_data = []
    for eid in event_ids:
        triage_row = store.conn.execute(
            """SELECT object FROM claims
               WHERE claim_type = 'triage' AND subject = ?
                 AND superseded_by IS NULL
               LIMIT 1""",
            (eid,),
        ).fetchone()
        if triage_row:
            try:
                triage_data.append(json.loads(triage_row["object"]))
            except (json.JSONDecodeError, TypeError):
                triage_data.append({})
        else:
            triage_data.append({})

    return ThreadBundle(
        thread_id=thread_id,
        events=events,
        triage_data=triage_data,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main synthesis pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_synthesis(
    store: LayeredGraphStore,
    llm_client: LLMClient,
    model: str = "",
    lite_model: str = "",
    batch_size: int = SYNTHESIS_BATCH_SIZE,
    max_concurrent: int = 10,
    user_email: str = "",
    profile_context: str = "",
) -> dict[str, Any]:
    """Run the complete synthesis pipeline (multi-gate).

    Steps:
    1. Read threads from all 3 gates (actionable, logistics, relational)
    2. Reconstruct ThreadBundles from stored events
    3. Per actionable thread: synthesize_thread() -> commitment claims
    4. Per logistics thread: extract_logistics() -> logistics claims
    5. Per relational thread: extract_relational() -> relational claims
    6. Dedup commitment claims
    7. Expire stale commitment claims
    8. Entity beliefs (deterministic, from persons table)
    9. Relation beliefs (deterministic, from stage1 claims)
    10. Commitment FACT beliefs (claims → beliefs with dedup/merge)
    11. Logistics FACT beliefs (claims → beliefs with dedup/merge)
    12. Expire stale beliefs

    Args:
        store: LayeredGraphStore with gate claims populated
        llm_client: LLM client for synthesis (Flash 3 for commitments)
        model: Model name for commitment synthesis (Flash 3)
        lite_model: Model name for logistics/relational extraction (Flash Lite)
        batch_size: Max threads per synthesis batch
        max_concurrent: Max parallel LLM calls
        user_email: User's email for direction detection

    Returns dict with synthesis statistics.
    """
    start_time = time.time()
    total_commitments = 0
    total_logistics = 0
    total_relational = 0
    total_beliefs = 0
    by_type: dict[str, int] = defaultdict(int)

    # Step 1: Get threads from all 3 gates
    actionable = get_actionable_threads(store)
    logistics_threads = get_logistics_threads(store)
    relational_threads = get_relational_threads(store)

    logger.info(
        "Synthesis: %d actionable, %d logistics, %d relational threads from gates",
        len(actionable), len(logistics_threads), len(relational_threads),
    )
    print(
        f"  Gate threads: {len(actionable)} actionable, "
        f"{len(logistics_threads)} logistics, "
        f"{len(relational_threads)} relational",
        flush=True,
    )

    # Step 2: Reconstruct ThreadBundles for all unique threads
    all_thread_items: dict[str, dict] = {}
    for item in actionable[:batch_size]:
        all_thread_items[item["thread_id"]] = item
    for item in logistics_threads[:batch_size]:
        tid = item["thread_id"]
        if tid in all_thread_items:
            all_thread_items[tid]["event_ids"] = list(set(
                all_thread_items[tid]["event_ids"] + item["event_ids"]
            ))
        else:
            all_thread_items[tid] = item
    for item in relational_threads[:batch_size]:
        tid = item["thread_id"]
        if tid in all_thread_items:
            all_thread_items[tid]["event_ids"] = list(set(
                all_thread_items[tid]["event_ids"] + item["event_ids"]
            ))
        else:
            all_thread_items[tid] = item

    bundles_map: dict[str, ThreadBundle] = {}
    for tid, item in all_thread_items.items():
        bundle = _reconstruct_bundle(tid, item["event_ids"], store)
        if bundle and bundle.events:
            bundles_map[tid] = bundle

    logger.info("Synthesis: %d bundles reconstructed", len(bundles_map))

    # Build persons cache for all bundles
    all_bundles = list(bundles_map.values())
    persons_cache, email_to_name = build_persons_cache(all_bundles, store) if all_bundles else ({}, {})

    # Track which threads pass which gates
    actionable_tids = {item["thread_id"] for item in actionable}
    logistics_tids = {item["thread_id"] for item in logistics_threads}
    relational_tids = {item["thread_id"] for item in relational_threads}

    # Build gate tier lookup for relational extraction context enrichment
    gate_tier_map: dict[str, str] = {}
    for item in relational_threads:
        tier = item.get("relationship_tier")
        if tier:
            gate_tier_map[item["thread_id"]] = tier

    # Load user-defined extractable fields once for all threads
    custom_field_defs = store.get_cq_extractable_fields()

    # Step 3: Per-thread commitment synthesis (Flash 3, parallel)
    # Pre-build prompts on main thread (store access is not thread-safe),
    # then parallelize only the LLM calls.
    synthesis_errors = 0
    actionable_bundles = [
        bundles_map[tid]
        for tid in actionable_tids
        if tid in bundles_map
    ]

    # Pre-build prompts and thread texts on main thread
    # For each thread, check for prior commitments (incremental synthesis)
    # IDEMPOTENCE: skip threads that already have commitments and no new events
    # WINDOWING: threads with > WINDOW_SIZE messages use sliding-window synthesis
    prebuilt: list[tuple[ThreadBundle, str, str, dict[str, str], list[dict]]] = []
    skipped_idempotent = 0
    windowed_count = 0
    for bundle in actionable_bundles:
        prior = _get_prior_commitments(store, bundle.thread_id)

        # Idempotence: if we already have active commitment claims for this
        # thread and no new events arrived since, skip re-synthesis.
        # Re-running the LLM produces slightly different wording each time,
        # which defeats dedup and creates duplicates.
        # Compare by event IDs (not timestamps) because calendar events have
        # future timestamps that would always bypass a timestamp check.
        if prior:
            already_processed_ids = {
                r["event_id"] for r in store.conn.execute(
                    """SELECT DISTINCT ce.event_id
                       FROM claim_events ce
                       JOIN claims c ON c.id = ce.claim_id
                       WHERE c.claim_type = 'commitment'
                         AND c.subject = ?
                         AND c.superseded_by IS NULL""",
                    (bundle.thread_id,),
                ).fetchall()
            }
            bundle_event_ids = {e.id for e in bundle.events}
            new_events = bundle_event_ids - already_processed_ids
            if not new_events:
                skipped_idempotent += 1
                continue

        if _needs_windowed_synthesis(bundle):
            # Long thread: process in windows with carry-forward summaries.
            # The summary windows run synchronously on the main thread
            # (they use the lite model and are fast). Only the final
            # synthesis prompt is parallelized with the other threads.
            prompt, msg_id_map, prior = _synthesize_windowed(
                bundle, llm_client, model, lite_model,
                user_email, persons_cache, store,
                prior_commitments=prior if prior else None,
                custom_field_defs=custom_field_defs,
                profile_context=profile_context,
                email_to_name=email_to_name,
            )
            # For windowed threads, thread_text is just the final window
            thread_text, _ = _format_thread_for_llm(
                bundle, user_email, persons_cache=persons_cache,
                email_to_name=email_to_name,
            )
            windowed_count += 1
        else:
            prompt, msg_id_map = _build_synthesis_prompt(
                bundle, user_email, persons_cache, store,
                prior_commitments=prior if prior else None,
                custom_field_defs=custom_field_defs,
                profile_context=profile_context,
                email_to_name=email_to_name,
            )
            thread_text, _ = _format_thread_for_llm(
                bundle, user_email, persons_cache=persons_cache,
                email_to_name=email_to_name,
            )

        prebuilt.append((bundle, prompt, thread_text, msg_id_map, prior))

    if skipped_idempotent or windowed_count:
        logger.info(
            "Synthesis: %d to process (%d windowed, %d skipped idempotent)",
            len(prebuilt), windowed_count, skipped_idempotent,
        )

    workers = min(max_concurrent, len(prebuilt)) if prebuilt else 0

    try:
        batch_schema = _build_synthesis_response_schema()
    except ImportError:
        batch_schema = None

    def _synth_from_prompt(
        item: tuple[ThreadBundle, str, str, dict[str, str], list[dict]],
    ) -> tuple[ThreadBundle, list[Claim], dict[str, str], list[dict], list[dict]]:
        bundle, prompt, thread_text, msg_id_map, prior = item

        raw_str = llm_client.generate(
            prompt=prompt,
            system=SYNTHESIS_SYSTEM,
            model=model,
            temperature=0.1,
            response_schema=batch_schema,
            format_json=batch_schema is None,
        )

        try:
            raw = json.loads(raw_str) if raw_str else None
        except (json.JSONDecodeError, TypeError):
            raw = None

        results = _parse_synthesis_result(raw)
        label_corrections = _parse_label_corrections(raw, msg_id_map)

        n_raw = len(results)
        results = validate_evidence_quotes(results, thread_text)
        results = filter_vague_actions(results)
        n_filtered = n_raw - len(results)
        if n_filtered > 0:
            logger.info(
                "Thread %s: %d/%d items passed post-processing",
                bundle.thread_id[:8], len(results), n_raw,
            )

        event_ids = [e.id for e in bundle.events]
        claims = [
            _build_commitment_claim(
                r, bundle.thread_id, event_ids,
                model or "synthesis", msg_id_map,
            )
            for r in results
        ]
        return bundle, claims, msg_id_map, prior, label_corrections

    def _write_synthesis_results(
        bundle: ThreadBundle,
        claims: list[Claim],
        prior: list[dict],
        label_corrections: list[dict] | None = None,
    ) -> None:
        """Write synthesis claims, superseding prior commitments when incremental.

        Only supersedes prior claims that have a matching new claim
        (by what-prefix + who). Prior claims with no match are kept.
        Also applies label corrections to triage claims/annotations.
        """
        nonlocal total_commitments

        if prior and claims:
            # Build lookup of new claims by (what_prefix, who)
            new_by_key: dict[tuple[str, str], str] = {}
            for c in claims:
                obj = json.loads(c.object)
                key = (
                    obj.get("what", "")[:DEDUP_WHAT_PREFIX_LEN].lower().strip(),
                    obj.get("who", "user"),
                )
                new_by_key[key] = c.id

            for old in prior:
                old_key = (
                    old.get("what", "")[:DEDUP_WHAT_PREFIX_LEN].lower().strip(),
                    old.get("who", "user"),
                )
                replacement_id = new_by_key.get(old_key)
                if replacement_id:
                    store.supersede_claim(old["claim_id"], replacement_id)

        for claim in claims:
            store.put_claim(claim)
            total_commitments += 1

        # Apply label corrections from synthesis
        if label_corrections:
            _apply_label_corrections(store, label_corrections, model)

    synth_total = len(prebuilt)
    synth_done = 0

    if workers <= 1:
        for item in prebuilt:
            try:
                bundle, claims, _, prior, lc = _synth_from_prompt(item)
                _write_synthesis_results(bundle, claims, prior, lc)
                synth_done += 1
                if synth_done % 50 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"  Commitments: {synth_done}/{synth_total} threads "
                        f"({elapsed:.0f}s)",
                        flush=True,
                    )
            except Exception as e:
                logger.error(
                    "Synthesis failed for %s: %s",
                    item[0].thread_id[:30], e,
                )
                synthesis_errors += 1
                synth_done += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_synth_from_prompt, item): item
                for item in prebuilt
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    bundle, claims, _, prior, lc = future.result()
                    _write_synthesis_results(bundle, claims, prior, lc)
                except Exception as e:
                    logger.error(
                        "Synthesis failed for %s: %s",
                        item[0].thread_id[:30], e,
                    )
                    synthesis_errors += 1
                synth_done += 1
                if synth_done % 50 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"  Commitments: {synth_done}/{synth_total} threads "
                        f"({elapsed:.0f}s)",
                        flush=True,
                    )

    logger.info(
        "Synthesis: %d commitment claims produced", total_commitments,
    )

    # Step 4: Per-thread logistics extraction (Flash Lite)
    logistics_errors = 0
    logistics_model = lite_model or model
    logistics_bundles = [
        bundles_map[tid]
        for tid in logistics_tids
        if tid in bundles_map
    ]

    skipped_logistics = 0
    logistics_done = 0
    logistics_total = len(logistics_bundles)
    for bundle in logistics_bundles:
        # Idempotence: skip if logistics claims already exist with same events
        already_processed_ids = {
            r["event_id"] for r in store.conn.execute(
                """SELECT DISTINCT ce.event_id
                   FROM claim_events ce
                   JOIN claims c ON c.id = ce.claim_id
                   WHERE c.claim_type = 'logistics_fact'
                     AND c.subject = ?
                     AND c.superseded_by IS NULL""",
                (bundle.thread_id,),
            ).fetchall()
        }
        if already_processed_ids:
            bundle_event_ids = {e.id for e in bundle.events}
            if not (bundle_event_ids - already_processed_ids):
                skipped_logistics += 1
                logistics_done += 1
                continue

        try:
            claims = extract_logistics(
                bundle, llm_client, logistics_model,
                user_email, persons_cache,
                profile_context=profile_context,
                email_to_name=email_to_name,
            )
            for claim in claims:
                store.put_claim(claim)
                total_logistics += 1
        except Exception as e:
            logger.error(
                "Logistics extraction failed for %s: %s",
                bundle.thread_id[:30], e,
            )
            logistics_errors += 1
        logistics_done += 1
        if logistics_done % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"  Logistics: {logistics_done}/{logistics_total} threads "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    logger.info(
        "Logistics: %d facts extracted from %d threads (%d skipped, already processed)",
        total_logistics, len(logistics_bundles), skipped_logistics,
    )

    # Step 5: Per-thread relational extraction (Flash Lite)
    relational_errors = 0
    relational_model = lite_model or model
    relational_bundles = [
        bundles_map[tid]
        for tid in relational_tids
        if tid in bundles_map
    ]

    skipped_relational = 0
    relational_done = 0
    relational_total = len(relational_bundles)
    for bundle in relational_bundles:
        # Idempotence: skip if relational claims already exist with same events
        already_processed_ids = {
            r["event_id"] for r in store.conn.execute(
                """SELECT DISTINCT ce.event_id
                   FROM claim_events ce
                   JOIN claims c ON c.id = ce.claim_id
                   WHERE c.claim_type = 'relational_context'
                     AND c.subject = ?
                     AND c.superseded_by IS NULL""",
                (bundle.thread_id,),
            ).fetchall()
        }
        if already_processed_ids:
            bundle_event_ids = {e.id for e in bundle.events}
            if not (bundle_event_ids - already_processed_ids):
                skipped_relational += 1
                relational_done += 1
                continue

        try:
            claims = extract_relational(
                bundle, llm_client, relational_model,
                user_email, persons_cache,
                store=store,
                gate_tier=gate_tier_map.get(bundle.thread_id),
                profile_context=profile_context,
                email_to_name=email_to_name,
            )
            for claim in claims:
                store.put_claim(claim)
                total_relational += 1
        except Exception as e:
            logger.error(
                "Relational extraction failed for %s: %s",
                bundle.thread_id[:30], e,
            )
            relational_errors += 1
        relational_done += 1
        if relational_done % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"  Relational: {relational_done}/{relational_total} threads "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    logger.info(
        "Relational: %d person contexts extracted from %d threads (%d skipped, already processed)",
        total_relational, len(relational_bundles), skipped_relational,
    )

    # Step 6: Dedup commitment claims
    dedup_stats = dedup_commitments(store)
    if dedup_stats["merged"]:
        logger.info(
            "Dedup: merged %d commitment claims", dedup_stats["merged"],
        )

    # Step 7: Expire stale commitment claims
    stale_count = expire_stale_commitments(store)

    # Step 7.5: Resolve commitments with completion evidence (LLM gate)
    resolution_stats = resolve_completed_commitments(
        store, llm_client=llm_client, model=lite_model or model,
    )
    resolved_count = resolution_stats["resolved"]
    if resolved_count:
        print(
            f"  Resolved {resolved_count} commitments with completion evidence",
            flush=True,
        )

    # Step 8: Entity beliefs
    entity_beliefs = _build_entity_beliefs(store)
    for belief in entity_beliefs:
        store.put_belief(belief)
        by_type["entity"] += 1
    total_beliefs += len(entity_beliefs)
    logger.info("Entity beliefs: %d produced", len(entity_beliefs))

    # Step 8.5: Participant inference beliefs (from call-meeting overlap)
    participant_beliefs = _build_participant_inference_beliefs(store)
    for belief in participant_beliefs:
        store.put_belief(belief)
        by_type["observation"] += 1
    total_beliefs += len(participant_beliefs)
    if participant_beliefs:
        logger.info(
            "Participant inference beliefs: %d produced", len(participant_beliefs),
        )

    # Step 9: Relation beliefs
    relation_beliefs = _build_relation_beliefs(store)
    for belief in relation_beliefs:
        store.put_belief(belief)
        by_type["relation"] += 1
    total_beliefs += len(relation_beliefs)
    logger.info("Relation beliefs: %d produced", len(relation_beliefs))

    # Step 9b: Relational context beliefs (FOAF from LLM extraction)
    relational_context_beliefs = _build_relational_context_beliefs(store)
    new_foaf_ids = set()
    for belief in relational_context_beliefs:
        store.put_belief(belief)
        new_foaf_ids.add(belief.id)
        by_type["relation"] += 1
    total_beliefs += len(relational_context_beliefs)

    # Mark stale any old FOAF beliefs no longer produced (user-self, resolved names)
    now = int(time.time())
    if new_foaf_ids:
        placeholders = ",".join("?" for _ in new_foaf_ids)
        stale_foaf = store.conn.execute(
            f"""UPDATE beliefs SET status = 'stale', updated_at = ?
                WHERE belief_type = 'relation'
                  AND status = 'active'
                  AND json_extract(data, '$.assertion_type') = 'relational_context'
                  AND id NOT IN ({placeholders})""",
            (now, *new_foaf_ids),
        ).rowcount
        store.conn.commit()
        if stale_foaf:
            logger.info("Marked %d old FOAF beliefs as stale", stale_foaf)

    logger.info(
        "Relational context beliefs: %d produced (FOAF)",
        len(relational_context_beliefs),
    )

    # Step 10: Commitment FACT beliefs (claims → beliefs compilation)
    commitment_beliefs = _build_commitment_beliefs(store)
    for belief in commitment_beliefs:
        store.put_belief(belief)
        by_type["fact"] += 1
    total_beliefs += len(commitment_beliefs)
    merged_facts = sum(
        1 for b in commitment_beliefs
        if b.data.get("claim_count", 1) > 1
    )
    logger.info(
        "Commitment FACT beliefs: %d produced (%d merged from multiple claims)",
        len(commitment_beliefs), merged_facts,
    )

    # Step 11: Logistics FACT beliefs (claims → beliefs compilation)
    logistics_beliefs = _build_logistics_beliefs(store)
    for belief in logistics_beliefs:
        store.put_belief(belief)
        by_type["fact"] += 1
    total_beliefs += len(logistics_beliefs)
    logger.info("Logistics FACT beliefs: %d produced", len(logistics_beliefs))

    # Step 12: Expire stale beliefs
    expired = _expire_stale_beliefs(store)

    # Checkpoint WAL to reclaim disk space after heavy writes
    store.checkpoint()

    elapsed = time.time() - start_time

    logger.info(
        "Synthesis complete: %d commitments + %d logistics + %d relational "
        "+ %d beliefs in %.1fs",
        total_commitments, total_logistics, total_relational,
        total_beliefs, elapsed,
    )

    return {
        "total_commitments": total_commitments,
        "total_logistics": total_logistics,
        "total_relational": total_relational,
        "total_beliefs": total_beliefs,
        "by_type": dict(by_type),
        "actionable_threads": len(actionable),
        "logistics_threads": len(logistics_threads),
        "relational_threads": len(relational_threads),
        "bundles_processed": len(bundles_map),
        "entity_beliefs": len(entity_beliefs),
        "participant_inference_beliefs": len(participant_beliefs),
        "relation_beliefs": len(relation_beliefs),
        "relational_context_beliefs": len(relational_context_beliefs),
        "commitment_fact_beliefs": len(commitment_beliefs),
        "commitment_fact_merged": merged_facts,
        "logistics_fact_beliefs": len(logistics_beliefs),
        "dedup_merged": dedup_stats["merged"],
        "stale_commitments": stale_count,
        "resolved_commitments": resolved_count,
        "resolution_details": resolution_stats.get("details", []),
        "expired_beliefs": expired,
        "synthesis_errors": synthesis_errors,
        "logistics_errors": logistics_errors,
        "relational_errors": relational_errors,
        "elapsed_seconds": round(elapsed, 2),
    }
