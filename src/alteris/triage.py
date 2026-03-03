"""Stage 4: Thread-based LLM triage.

Operates on THREADS, not individual messages. Three processing strategies:
  THREAD_FULL      — Flash Lite, full thread + rich context
  MSG_BATCH_COMPACT — Flash Lite, 3-5 messages + compact header
  MSG_COMPACT      — Flash Lite, single message + compact header

All strategies use Gemini 2.0 Flash Lite (cheap, no thinking budget).
Strategy selection still determines prompt richness and context level.

Each thread produces:
  1. One thread_triage claim (new, additive)
  2. N per-event triage claims (existing format, backward compatible
     with propagate.py and extract.py)

Pipeline position:
  Events -> Persons -> Stage 1 Claims -> [Embedding]
  -> TRIAGE (this module) -> Message Passing -> Deep Extraction
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum, unique
from typing import Any

from pydantic import BaseModel, Field, field_validator

from alteris.prompts.triage import (
    MSG_COMPACT_SUFFIX,
    MSG_COMPACT_SYSTEM,
    PRIOR_THREAD_CONTEXT_BLOCK,
    THREAD_FULL_SUFFIX,
    THREAD_FULL_SYSTEM,
    THREAD_SUMMARY_SUFFIX,
    THREAD_SUMMARY_SYSTEM,
)
from alteris.constants import (
    BODY_PREVIEW_LENGTH,
    CLOUD_FRONTIER_MODEL,
    CLOUD_LITE_MODEL,
    EVENT_TYPE_CALENDAR,
    EVENT_TYPE_IDENTITY,
    EVENT_TYPE_MEETING,
    INCREMENTAL_CONTEXT_MESSAGES,
    MAX_BODY_CHARS,
    MAX_THREAD_FETCH,
    MSG_BATCH_SIZE,
    MSG_COMPACT_MAX_OUTPUT_TOKENS,
    REACTIVATION_THRESHOLD,
    SCORE_FLOOR_CALENDAR,
    SCORE_FLOOR_COMMITMENT,
    SCORE_FLOOR_MEETING,
    SCORE_FLOOR_TIER1_SENDER,
    SECONDS_PER_DAY,
    THREAD_INPUT_TOKEN_BUDGET,
    THREAD_MAX_SCORED_MSGS,
    TRIAGE_BATCH_SIZE,
    USER_TIMEZONE,
    safe_timezone,
)
from alteris.context import (
    build_compact_header,
    build_full_context,
)
from alteris.llm.base import LLMClient
from alteris.models import (
    Claim,
    ExtractionMethod,
    ExtractionProvenance,
    Modality,
)
from alteris.privacy import SensitivityLevel
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROMPT_VERSION = "triage_v4.0"

# Threads above this size get chunked triage: split into overlapping
# chunks, triage each independently, merge results (keep highest score).
LARGE_THREAD_THRESHOLD = 200

# How many recent messages to keep verbatim in Pass 2 (legacy, used as fallback)
LARGE_THREAD_TAIL_SIZE = 100

# Chunked triage: split large threads into overlapping windows
CHUNK_SIZE = 100        # messages per chunk
CHUNK_OVERLAP = 20      # overlap between consecutive chunks

# For threads with more messages than this, only ask the model to score
# the most recent N messages. Older messages still provide context in
# the prompt but don't need individual scores — saves output tokens.
# Set low for Flash Lite (8K output limit).
MAX_SCORED_MESSAGES = 20

# Flash Lite has 8K output token limit (vs 64K for Flash 3).
# Cap triage output to avoid truncated JSON.
LITE_MAX_OUTPUT_TOKENS = 8192
DEEP_MAX_OUTPUT_TOKENS = 65536
LARGE_THREAD_FRONTIER_THRESHOLD = 250
FRONTIER_TIMEOUT = 600  # seconds — 10 min, large threads with 6mo data need headroom
ASYNC_CALL_TIMEOUT = 300  # seconds — per-API-call asyncio timeout (matches Gemini HTTP timeout)
THREAD_TIMEOUT = 600  # seconds — overall per-thread timeout (kills stalled threads)


def _model_for_thread(thread_id: str, n_messages: int = 0) -> tuple[str, int]:
    """Select model and output budget based on thread source and size.

    All threads use Flash Lite. Granola transcripts that exceed the token
    budget are split into sections by _split_granola_sections().
    """
    return CLOUD_LITE_MODEL, LITE_MAX_OUTPUT_TOKENS

VALID_DOMAINS = {
    "work", "personal", "family", "financial", "health",
    "legal", "travel", "shopping", "automated",
}
VALID_PII = {"financial", "medical", "legal", "credentials", "travel_docs"}
VALID_SENSITIVITY = {
    "health_discussion", "relationship_conflict", "financial_distress",
    "legal_matter", "intimate_content", "child_info", "grief",
}
VALID_COMMITMENT_TYPES = {
    "inbound_request", "user_commitment", "deadline",
    "waiting_on", "payment_due", "follow_up",
}
VALID_THREAD_STATUSES = {
    "active_conversation", "awaiting_user", "awaiting_them",
    "stale", "one_shot",
}

EXCLUDED_EVENT_TYPES = {EVENT_TYPE_IDENTITY, "call"}

BUCKET_LABELS = {
    1: "hot (<=7d)", 2: "recent (7-30d)", 3: "warm (30-90d)",
    4: "aging (90d-1y)", 5: "old (1y+)",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy enum
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@unique
class TriageStrategy(str, Enum):
    """Processing strategy for a thread."""
    THREAD_FULL = "thread_full"           # Gemini, full thread + rich context
    MSG_BATCH_COMPACT = "msg_batch_compact"  # Qwen, 3-5 messages + compact header
    MSG_COMPACT = "msg_compact"           # Qwen, single message + compact header


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pydantic models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MessageScore(BaseModel):
    """Per-event score from thread-level triage."""
    id: str = ""
    score: float = Field(default=0.0)
    reason: str = ""

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(max(0.0, min(1.0, v)), 1)


class ThreadTriageResult(BaseModel):
    """Unified output schema for all triage strategies."""
    thread_id: str = ""
    thread_score: float = Field(default=0.0)
    domain: str = ""
    universal_spheres: list[str] = Field(default_factory=list)
    specific_topics: list[str] = Field(default_factory=list)
    # Legacy field — populated from universal_spheres + specific_topics for backward compat
    topics: list[str] = Field(default_factory=list)
    thread_status: str = ""
    relationship: str = ""
    thread_summary: str = ""
    message_scores: list[MessageScore] = Field(default_factory=list)
    extraction_candidates: list[str] = Field(default_factory=list)
    pii: list[str] = Field(default_factory=list)
    sensitivity: list[str] = Field(default_factory=list)
    commitment_type: str | None = None
    strategy: str = ""
    model_id: str = ""

    @field_validator("thread_score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(max(0.0, min(1.0, v)), 1)

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        v = str(v).lower().strip()
        return v if v in VALID_DOMAINS else ""

    @field_validator("universal_spheres", mode="before")
    @classmethod
    def clean_spheres(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v.strip().lower()[:60]] if v.strip() else []
        if not isinstance(v, list):
            return []
        return [str(t).strip().lower()[:60] for t in v[:3] if t]

    @field_validator("specific_topics", mode="before")
    @classmethod
    def clean_specific_topics(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v.strip().lower()[:60]] if v.strip() else []
        if not isinstance(v, list):
            return []
        return [str(t).strip().lower()[:60] for t in v[:5] if t]

    @field_validator("topics", mode="before")
    @classmethod
    def clean_topics(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v.strip().lower()[:50]] if v.strip() else []
        if not isinstance(v, list):
            return []
        return [str(t).strip().lower()[:50] for t in v[:5] if t]

    @field_validator("pii", mode="before")
    @classmethod
    def validate_pii(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = [v] if v.strip() else []
        if not isinstance(v, list):
            return []
        return [
            str(p).lower().strip() for p in v
            if str(p).lower().strip() in VALID_PII
        ]

    @field_validator("sensitivity", mode="before")
    @classmethod
    def validate_sensitivity(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = [v] if v.strip() else []
        if not isinstance(v, list):
            return []
        return [
            str(s).lower().strip() for s in v
            if str(s).lower().strip() in VALID_SENSITIVITY
        ]

    @field_validator("commitment_type")
    @classmethod
    def validate_commitment(cls, v: str | None) -> str | None:
        if v is None or str(v).lower().strip() in ("", "null", "none"):
            return None
        val = str(v).lower().strip()
        return val if val in VALID_COMMITMENT_TYPES else None


class TriageResult(BaseModel):
    """Per-event triage result (backward compat with MSG_COMPACT parsing)."""
    id: str = ""
    score: float = Field(default=0.0)
    reason: str = ""
    domain: str = ""
    universal_spheres: list[str] = Field(default_factory=list)
    specific_topics: list[str] = Field(default_factory=list)
    # Legacy field — populated from universal_spheres + specific_topics for backward compat
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    pii: list[str] = Field(default_factory=list)
    sensitivity: list[str] = Field(default_factory=list)
    commitment_type: str | None = None

    @field_validator("score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(max(0.0, min(1.0, v)), 1)

    @field_validator("reason")
    @classmethod
    def truncate_reason(cls, v: str) -> str:
        return str(v)[:200]

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        v = str(v).lower().strip()
        return v if v in VALID_DOMAINS else ""

    @field_validator("universal_spheres", mode="before")
    @classmethod
    def clean_spheres(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v.strip().lower()[:60]] if v.strip() else []
        if not isinstance(v, list):
            return []
        return [str(t).strip().lower()[:60] for t in v[:3] if t]

    @field_validator("specific_topics", mode="before")
    @classmethod
    def clean_specific_topics(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v.strip().lower()[:60]] if v.strip() else []
        if not isinstance(v, list):
            return []
        return [str(t).strip().lower()[:60] for t in v[:5] if t]

    @field_validator("topics", mode="before")
    @classmethod
    def clean_topics(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v.strip().lower()[:50]] if v.strip() else []
        if not isinstance(v, list):
            return []
        return [str(t).strip().lower()[:50] for t in v[:5] if t]

    @field_validator("entities", mode="before")
    @classmethod
    def clean_entities(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v.strip()[:100]] if v.strip() else []
        if not isinstance(v, list):
            return []
        return [str(e).strip()[:100] for e in v[:10] if e]

    @field_validator("pii", mode="before")
    @classmethod
    def validate_pii(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = [v] if v.strip() else []
        if not isinstance(v, list):
            return []
        return [
            str(p).lower().strip() for p in v
            if str(p).lower().strip() in VALID_PII
        ]

    @field_validator("sensitivity", mode="before")
    @classmethod
    def validate_sensitivity(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = [v] if v.strip() else []
        if not isinstance(v, list):
            return []
        return [
            str(s).lower().strip() for s in v
            if str(s).lower().strip() in VALID_SENSITIVITY
        ]

    @field_validator("commitment_type")
    @classmethod
    def validate_commitment(cls, v: str | None) -> str | None:
        if v is None or str(v).lower().strip() in ("", "null", "none"):
            return None
        val = str(v).lower().strip()
        return val if val in VALID_COMMITMENT_TYPES else None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thread grouping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def group_events_by_thread(events: list[dict]) -> dict[str, list[dict]]:
    """Group events into threads for triage.

    Grouping rules by source:
      mail/whatsapp/imessage/slack — by metadata.thread_id
      granola — each meeting is its own thread
      calendar — batch all events by day
      standalone — any event without a thread_id
    """
    groups: dict[str, list[dict]] = defaultdict(list)

    for e in events:
        meta = json.loads(e.get("metadata") or "{}")
        thread_id = meta.get("thread_id", "")
        source = e["source"]

        if source == "granola":
            groups[f"granola:{e['id']}"].append(e)
        elif source == "calendar":
            ts = e["timestamp"]
            tz = safe_timezone()
            day = datetime.fromtimestamp(ts, tz=tz).strftime("%Y-%m-%d")
            groups[f"calendar:{day}"].append(e)
        elif thread_id:
            groups[thread_id].append(e)
        else:
            groups[f"standalone:{e['id']}"].append(e)

    # Sort events within each thread chronologically
    for tid in groups:
        groups[tid].sort(key=lambda e: e["timestamp"])

    return dict(groups)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy selection (privacy router)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def select_strategy(
    thread_events: list[dict],
    event_routes: dict[str, str] | None = None,
    cloud_all: bool = False,
) -> TriageStrategy:
    """Select triage strategy based on sensitivity and scoring route.

    Two routing dimensions:
      1. Privacy: CRITICAL events force local model (safety).
      2. Scoring route: low_priority threads → local model (cost).
         If ANY event in the thread is full_triage, the thread goes to
         cloud (the high-value event carries the thread). Only all-low_priority
         threads get routed to local.

    When cloud_all is True, skip all local routing — send everything to cloud.
    When event_routes is None (no projections available), falls back to
    the original sensitivity-only routing.
    """
    # cloud_all mode: everything goes to cloud, no local routing
    if cloud_all:
        return TriageStrategy.THREAD_FULL

    max_sens = max(
        (e.get("sensitivity", SensitivityLevel.SENSITIVE) for e in thread_events),
        default=SensitivityLevel.SENSITIVE,
    )

    # Privacy override — always takes precedence
    if max_sens >= SensitivityLevel.CRITICAL:
        if len(thread_events) == 1:
            return TriageStrategy.MSG_COMPACT
        return TriageStrategy.MSG_BATCH_COMPACT

    # Scoring route: if all events are low_priority, use local model.
    # Events not in event_routes (no projection) are conservatively treated
    # as full_triage — better to over-triage than miss something.
    if event_routes:
        all_low = all(
            event_routes.get(e["id"]) == "low_priority"
            for e in thread_events
        )
        if all_low:
            if len(thread_events) == 1:
                return TriageStrategy.MSG_COMPACT
            return TriageStrategy.MSG_BATCH_COMPACT

    return TriageStrategy.THREAD_FULL


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Context helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_primary_sender_pid(
    events: list[dict], store: LayeredGraphStore,
) -> str | None:
    """Get the primary non-user sender person_id for a thread's events."""
    for e in events:
        meta = json.loads(e.get("metadata") or "{}")
        if meta.get("is_from_me"):
            continue
        row = store.conn.execute(
            """SELECT ep.person_id, p.is_user
               FROM event_persons ep
               JOIN persons p ON ep.person_id = p.person_id
               WHERE ep.event_id = ? AND ep.role = 'sender'
               LIMIT 1""",
            (e["id"],),
        ).fetchone()
        if row and not row["is_user"]:
            return row["person_id"]
    return None


def _get_thread_id_from_events(events: list[dict]) -> str:
    """Extract thread_id from events (all events share it)."""
    for e in events:
        meta = json.loads(e.get("metadata") or "{}")
        tid = meta.get("thread_id", "")
        if tid:
            return tid
    return ""


def build_sender_cache(store: LayeredGraphStore) -> dict[str, dict[str, Any]]:
    """Pre-build sender context from persons + claims.

    Returns {person_id: {name, tier, msg_count, reply_ratio, is_user, email}}.
    """
    cache: dict[str, dict[str, Any]] = {}

    rows = store.conn.execute(
        "SELECT person_id, canonical_name, is_user FROM persons"
    ).fetchall()
    for r in rows:
        cache[r["person_id"]] = {
            "name": r["canonical_name"] or r["person_id"][:12],
            "is_user": bool(r["is_user"]),
            "tier": 3,
            "msg_count": 0,
            "reply_ratio": "0%",
            "email": "",
        }

    # Load primary email identifiers for automated-sender detection
    email_rows = store.conn.execute(
        "SELECT person_id, identifier FROM person_identifiers WHERE identifier_type = 'email'"
    ).fetchall()
    for r in email_rows:
        pid = r["person_id"]
        if pid in cache and not cache[pid]["email"]:
            cache[pid]["email"] = r["identifier"]

    freq_claims = store.conn.execute(
        """SELECT predicate, object, confidence FROM claims
           WHERE claim_type = 'communication_frequency'
             AND superseded_by IS NULL"""
    ).fetchall()
    for r in freq_claims:
        pred = r["predicate"]
        if not pred.startswith("communicates_with:"):
            continue
        pid = pred.replace("communicates_with:", "")
        if pid not in cache:
            continue
        try:
            obj = json.loads(r["object"])
            count = obj.get("event_count", 0)
            cache[pid]["msg_count"] = count
            if count >= 50:
                cache[pid]["tier"] = 1
            elif count >= 10:
                cache[pid]["tier"] = 2
        except (json.JSONDecodeError, TypeError):
            pass

    dir_claims = store.conn.execute(
        """SELECT predicate, object FROM claims
           WHERE claim_type = 'directionality'
             AND superseded_by IS NULL"""
    ).fetchall()
    for r in dir_claims:
        pred = r["predicate"]
        if not pred.startswith("direction_with:"):
            continue
        pid = pred.replace("direction_with:", "")
        if pid not in cache:
            continue
        try:
            obj = json.loads(r["object"])
            ratio = obj.get("user_initiated_ratio", 0.0)
            cache[pid]["reply_ratio"] = f"{ratio:.0%}"
        except (json.JSONDecodeError, TypeError):
            pass

    return cache


def _get_event_sender(
    store: LayeredGraphStore,
    event_id: str,
    sender_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Get sender info for an event from event_persons."""
    row = store.conn.execute(
        """SELECT ep.person_id, p.canonical_name
           FROM event_persons ep
           JOIN persons p ON ep.person_id = p.person_id
           WHERE ep.event_id = ? AND ep.role = 'sender'
           LIMIT 1""",
        (event_id,),
    ).fetchone()

    if not row:
        return {
            "name": "unknown", "tier": 3, "msg_count": 0,
            "reply_ratio": "0%", "is_user": False,
        }

    pid = row["person_id"]
    if pid in sender_cache:
        return sender_cache[pid]

    return {
        "name": row["canonical_name"] or pid[:12],
        "tier": 3, "msg_count": 0, "reply_ratio": "0%", "is_user": False,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt builders
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _safe_body(text: str) -> str:
    """Apply hard safety cap on message body length.

    Prevents malformed events (base64 blobs, injection payloads, adapter bugs)
    from blowing up prompts. Normal messages are never this large.
    """
    if len(text) <= MAX_BODY_CHARS:
        return text
    return text[:MAX_BODY_CHARS] + f"\n[TRUNCATED: body exceeded {MAX_BODY_CHARS:,} char safety limit]"


def _format_granola_body(raw_content: str) -> str:
    """Restructure Granola raw_content into summary + transcript format.

    Granola stores both structured notes and transcript in raw_content as:
        ## Meeting Notes
        [notes]
        ## Transcript
        [transcript]

    We reformat to orient the LLM with Granola's own analysis first:
        MEETING SUMMARY (from Granola):
        [structured notes]

        FULL TRANSCRIPT:
        [transcript]
    """
    if not raw_content:
        return ""

    notes_section = ""
    transcript_section = ""

    # Split on ## headings
    if "## Transcript" in raw_content:
        before_transcript, transcript_section = raw_content.split(
            "## Transcript", 1,
        )
        transcript_section = transcript_section.strip()
        # Extract notes from before the transcript
        if "## Meeting Notes" in before_transcript:
            _, notes_section = before_transcript.split("## Meeting Notes", 1)
            notes_section = notes_section.strip()
        else:
            notes_section = before_transcript.strip()
    elif "## Meeting Notes" in raw_content:
        _, notes_section = raw_content.split("## Meeting Notes", 1)
        notes_section = notes_section.strip()
    else:
        return raw_content

    parts = []
    if notes_section:
        parts.append(f"MEETING SUMMARY (from Granola):\n{notes_section}")
    if transcript_section:
        parts.append(f"FULL TRANSCRIPT:\n{transcript_section}")

    return "\n\n".join(parts) if parts else raw_content


def _compact_thread_for_budget(
    events: list[dict],
    context_chars: int,
) -> list[dict]:
    """When a thread exceeds the input token budget, keep the tail.

    Recent messages are far more valuable for triage than old ones.
    Returns events to include (chronological order, tail-biased).
    """
    # Rough estimate: 4 chars ≈ 1 token
    budget_chars = THREAD_INPUT_TOKEN_BUDGET * 4
    available = budget_chars - context_chars

    if available <= 0:
        # Context alone exceeds budget; keep last 50 messages
        return events[-50:]

    # Estimate total message size
    total_chars = sum(len(e.get("raw_content") or "") + 100 for e in events)

    if total_chars <= available:
        return events  # fits, no compaction needed

    # Compaction needed: keep tail messages that fit
    kept: list[dict] = []
    running = 0
    for e in reversed(events):
        msg_chars = len(e.get("raw_content") or "") + 100
        if running + msg_chars > available:
            break
        kept.append(e)
        running += msg_chars

    kept.reverse()  # back to chronological

    logger.info(
        "Thread compaction: %d -> %d events (kept tail, %d chars)",
        len(events), len(kept), running,
    )
    return kept


def _chunk_thread(events: list[dict]) -> list[list[dict]]:
    """Split a large thread into overlapping chunks for independent triage.

    Each chunk is CHUNK_SIZE messages. Consecutive chunks overlap by
    CHUNK_OVERLAP messages so the LLM has boundary context.
    """
    n = len(events)
    if n <= CHUNK_SIZE:
        return [events]

    chunks: list[list[dict]] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, n, step):
        end = min(start + CHUNK_SIZE, n)
        chunks.append(events[start:end])
        if end >= n:
            break
    return chunks


def _merge_chunk_results(
    chunk_results: list[ThreadTriageResult | None],
    chunks: list[list[dict]],
) -> ThreadTriageResult | None:
    """Merge triage results from overlapping chunks.

    For messages appearing in multiple chunks (the overlap zone),
    keep the result with the highest score. Thread-level fields
    are taken from the chunk with the highest thread_score.
    """
    valid = [
        (r, c) for r, c in zip(chunk_results, chunks) if r is not None
    ]
    if not valid:
        return None

    # Pick the best chunk for thread-level fields
    best_result, _ = max(valid, key=lambda rc: rc[0].thread_score)

    # Merge per-message scores: for overlapping IDs keep highest
    merged_scores: dict[str, MessageScore] = {}
    merged_candidates: set[str] = set()
    all_pii: set[str] = set()
    all_sensitivity: set[str] = set()
    all_spheres: set[str] = set()
    all_specific: set[str] = set()

    for result, chunk_events in valid:
        chunk_ids = {e["id"] for e in chunk_events}
        all_pii.update(result.pii)
        all_sensitivity.update(result.sensitivity)
        all_spheres.update(result.universal_spheres)
        all_specific.update(result.specific_topics)
        merged_candidates.update(result.extraction_candidates)

        for ms in result.message_scores:
            existing = merged_scores.get(ms.id)
            if existing is None or ms.score > existing.score:
                merged_scores[ms.id] = ms

    return ThreadTriageResult(
        thread_id=best_result.thread_id,
        thread_score=best_result.thread_score,
        domain=best_result.domain,
        universal_spheres=list(all_spheres)[:3],
        specific_topics=list(all_specific)[:5],
        thread_status=best_result.thread_status,
        relationship=best_result.relationship,
        thread_summary=best_result.thread_summary,
        message_scores=list(merged_scores.values()),
        extraction_candidates=list(merged_candidates),
        pii=list(all_pii),
        sensitivity=list(all_sensitivity),
        commitment_type=best_result.commitment_type,
        strategy=best_result.strategy,
        model_id=best_result.model_id,
    )


def build_thread_full_prompt(
    thread_id: str,
    events: list[dict],
    store: LayeredGraphStore,
    sender_cache: dict[str, dict[str, Any]],
    now: int,
    profile_context: str = "",
) -> str:
    """Build THREAD_FULL prompt: full thread + rich context for Gemini.

    No body truncation — Gemini has 1M token input.
    When a thread exceeds the token budget, the TAIL (most recent
    messages) is kept and older messages are discarded.

    Structure:
      [context enrichment: contact dossier + thread snapshot + day context]
      === MESSAGES (chronological) ===
      [timestamp] sender: body (full, untruncated)
      ...
      [classification instructions]

    Granola meetings get special handling: structured notes as
    orientation header, then full transcript.
    """
    source = events[0]["source"] if events else "unknown"
    sender_pid = _get_primary_sender_pid(events, store)
    is_granola = source == "granola"

    # Build context enrichment
    context_block = ""
    if sender_pid:
        context_block = build_full_context(
            sender_pid, thread_id, store, target_date=now, now=now,
        )

    header_parts = [f"THREAD: {thread_id}", f"SOURCE: {source}"]
    if context_block:
        header_parts.append("")
        header_parts.append(context_block)

    if profile_context:
        header_parts.append("")
        header_parts.append(profile_context)

    context_chars = sum(len(p) for p in header_parts)

    # Compact if needed (favor recent/tail messages)
    display_events = _compact_thread_for_budget(events, context_chars)

    parts = list(header_parts)
    parts.append("")
    parts.append("=== MESSAGES (chronological) ===")

    if len(display_events) < len(events):
        parts.append(
            f"[... {len(events) - len(display_events)} older messages omitted, "
            f"showing most recent {len(display_events)} ...]"
        )

    tz = safe_timezone()

    for e in display_events:
        ts = e["timestamp"]
        dt = datetime.fromtimestamp(ts, tz=tz)
        time_str = dt.strftime("%m/%d %H:%M")

        meta = json.loads(e.get("metadata") or "{}")
        is_from_me = meta.get("is_from_me", False)

        if is_from_me:
            sender_name = "USER"
        else:
            sender_info = _get_event_sender(store, e["id"], sender_cache)
            sender_name = sender_info["name"]

        body = _safe_body(e.get("raw_content") or "")

        # Granola: restructure into summary + transcript
        if is_granola:
            body = _format_granola_body(body)

        subject = meta.get("subject", "")
        if subject and source == "mail":
            parts.append(
                f"[{time_str}] {sender_name} (id:{e['id'][:12]}): "
                f"[Subject: {subject}] {body}"
            )
        elif is_granola and subject:
            parts.append(
                f"[{time_str}] Meeting: {subject} (id:{e['id'][:12]})\n{body}"
            )
        else:
            parts.append(
                f"[{time_str}] {sender_name} (id:{e['id'][:12]}): {body}"
            )

    # For large threads, only request scores for the most recent N messages
    if len(display_events) > MAX_SCORED_MESSAGES:
        scored_events = display_events[-MAX_SCORED_MESSAGES:]
        suffix = THREAD_FULL_SUFFIX + (
            f"\n\nIMPORTANT: This thread has {len(display_events)} messages. "
            f"Only provide message_scores for the {MAX_SCORED_MESSAGES} most recent messages "
            f"(ids starting from {scored_events[0]['id'][:12]}). "
            f"Older messages are context only — do NOT score them."
        )
        parts.append(suffix)
    else:
        parts.append(THREAD_FULL_SUFFIX)

    return "\n".join(parts)


def _format_messages_for_summary(
    events: list[dict],
    store: LayeredGraphStore,
    sender_cache: dict[str, dict[str, Any]],
) -> str:
    """Format messages for the summary pass (no IDs needed, just content)."""
    tz = safe_timezone()
    lines = []
    for e in events:
        ts = e["timestamp"]
        dt = datetime.fromtimestamp(ts, tz=tz)
        time_str = dt.strftime("%m/%d %H:%M")
        meta = json.loads(e.get("metadata") or "{}")
        is_from_me = meta.get("is_from_me", False)
        if is_from_me:
            sender_name = "USER"
        else:
            sender_info = _get_event_sender(store, e["id"], sender_cache)
            sender_name = sender_info["name"]
        body = _safe_body(e.get("raw_content") or "")[:500]  # Truncate for summary
        lines.append(f"[{time_str}] {sender_name}: {body}")
    return "\n".join(lines)


def build_summary_prompt(
    thread_id: str,
    head_events: list[dict],
    store: LayeredGraphStore,
    sender_cache: dict[str, dict[str, Any]],
) -> str:
    """Build Pass 1 prompt: summarize older messages in a large thread."""
    source = head_events[0]["source"] if head_events else "unknown"
    formatted = _format_messages_for_summary(head_events, store, sender_cache)
    return (
        f"THREAD: {thread_id}\nSOURCE: {source}\n"
        f"MESSAGES ({len(head_events)} messages, oldest first):\n\n"
        f"{formatted}\n\n"
        f"{THREAD_SUMMARY_SUFFIX}"
    )


def build_triage_with_summary_prompt(
    thread_id: str,
    tail_events: list[dict],
    head_summary: str,
    head_count: int,
    store: LayeredGraphStore,
    sender_cache: dict[str, dict[str, Any]],
    now: int,
) -> str:
    """Build Pass 2 prompt: triage using summary of history + recent verbatim messages."""
    source = tail_events[0]["source"] if tail_events else "unknown"
    sender_pid = _get_primary_sender_pid(tail_events, store)

    # Context enrichment (same as full prompt)
    context_block = ""
    if sender_pid:
        context_block = build_full_context(
            sender_pid, thread_id, store, target_date=now, now=now,
        )

    tz = safe_timezone()

    parts = [f"THREAD: {thread_id}", f"SOURCE: {source}"]
    if context_block:
        parts.append("")
        parts.append(context_block)

    parts.append("")
    parts.append(f"CONVERSATION HISTORY SUMMARY ({head_count} earlier messages):")
    parts.append(head_summary)
    parts.append("")
    parts.append(f"=== RECENT MESSAGES (last {len(tail_events)}, verbatim) ===")

    for e in tail_events:
        ts = e["timestamp"]
        dt = datetime.fromtimestamp(ts, tz=tz)
        time_str = dt.strftime("%m/%d %H:%M")
        meta = json.loads(e.get("metadata") or "{}")
        is_from_me = meta.get("is_from_me", False)
        if is_from_me:
            sender_name = "USER"
        else:
            sender_info = _get_event_sender(store, e["id"], sender_cache)
            sender_name = sender_info["name"]
        body = _safe_body(e.get("raw_content") or "")
        subject = meta.get("subject", "")
        if subject and source == "mail":
            parts.append(
                f"[{time_str}] {sender_name} (id:{e['id'][:12]}): "
                f"[Subject: {subject}] {body}"
            )
        else:
            parts.append(
                f"[{time_str}] {sender_name} (id:{e['id'][:12]}): {body}"
            )

    parts.append(THREAD_FULL_SUFFIX)
    return "\n".join(parts)


def build_msg_batch_prompt(
    events: list[dict],
    store: LayeredGraphStore,
    sender_cache: dict[str, dict[str, Any]],
    now: int,
    profile_context: str = "",
) -> str:
    """Build MSG_BATCH_COMPACT prompt: 3-5 messages + compact header for Qwen."""
    sender_pid = _get_primary_sender_pid(events, store)
    thread_id = _get_thread_id_from_events(events)

    header = ""
    if sender_pid:
        header = build_compact_header(sender_pid, thread_id, store, now=now)

    parts = []
    if header:
        parts.append(header)
        parts.append("")

    if profile_context:
        parts.append(profile_context)
        parts.append("")

    for i, e in enumerate(events, 1):
        meta = json.loads(e.get("metadata") or "{}")
        sender_info = _get_event_sender(store, e["id"], sender_cache)
        subject = meta.get("subject", "(no subject)")
        body = _safe_body(e.get("raw_content") or "")[:BODY_PREVIEW_LENGTH]

        dt = datetime.fromtimestamp(e["timestamp"], tz=timezone.utc)
        date_str = dt.strftime("%Y-%m-%d %H:%M")

        age_days = max(0, (now - e["timestamp"]) / SECONDS_PER_DAY)

        parts.append(f"--- ITEM {i} (id: {e['id']}) ---")
        parts.append(f"SOURCE: {e['source']}")
        parts.append(f"SENDER: {sender_info['name']} (tier {sender_info['tier']})")
        parts.append(f"DATE: {date_str} ({age_days:.0f}d ago)")
        parts.append(f"SUBJECT: {subject}")
        parts.append(f"BODY: {body or '(empty)'}")
        parts.append("")

    parts.append(MSG_COMPACT_SUFFIX)

    return "\n".join(parts)


def build_msg_compact_prompt(
    event: dict,
    store: LayeredGraphStore,
    sender_cache: dict[str, dict[str, Any]],
    now: int,
    profile_context: str = "",
) -> str:
    """Build MSG_COMPACT prompt: single message + compact header for Qwen."""
    return build_msg_batch_prompt([event], store, sender_cache, now, profile_context)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM response parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _clean_llm_output(raw: str) -> str:
    """Strip thinking tags and markdown fences from LLM output."""
    cleaned = raw.strip()
    if "<think>" in cleaned:
        parts = cleaned.split("</think>")
        cleaned = parts[-1].strip() if len(parts) > 1 else cleaned
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    return cleaned


def parse_thread_triage_response(
    raw: str,
    thread_id: str,
    event_ids: list[str],
    strategy: str,
    model_id: str,
) -> ThreadTriageResult | None:
    """Parse Gemini response for thread-level triage."""
    if not raw:
        return None

    cleaned = _clean_llm_output(raw)

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract outermost JSON object
        first_brace = cleaned.find("{")
        if first_brace >= 0:
            depth = 0
            end_idx = -1
            in_string = False
            escape_next = False
            for i in range(first_brace, len(cleaned)):
                ch = cleaned[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\":
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            if end_idx > first_brace:
                try:
                    parsed = json.loads(cleaned[first_brace:end_idx + 1])
                except json.JSONDecodeError:
                    pass

    if not isinstance(parsed, dict):
        logger.warning("Failed to parse thread triage response: %s...", raw[:300])
        return None

    # Parse message_scores
    msg_scores_raw = parsed.get("message_scores", [])
    msg_scores = []
    if isinstance(msg_scores_raw, list):
        for ms in msg_scores_raw:
            if isinstance(ms, dict):
                try:
                    msg_scores.append(MessageScore.model_validate(ms))
                except Exception:
                    if "id" in ms and "score" in ms:
                        msg_scores.append(MessageScore(
                            id=str(ms["id"]),
                            score=float(ms.get("score", 0)),
                            reason=str(ms.get("reason", "")),
                        ))

    try:
        result = ThreadTriageResult(
            thread_id=thread_id,
            thread_score=float(parsed.get("thread_score", 0)),
            domain=str(parsed.get("domain", "")),
            universal_spheres=parsed.get("universal_spheres", []),
            specific_topics=parsed.get("specific_topics", parsed.get("topics", [])),
            thread_status=str(parsed.get("thread_status", "")),
            relationship=str(parsed.get("relationship", "")),
            thread_summary=str(parsed.get("thread_summary", "")),
            message_scores=msg_scores,
            extraction_candidates=parsed.get("extraction_candidates", []),
            pii=parsed.get("pii", []),
            sensitivity=parsed.get("sensitivity", []),
            commitment_type=parsed.get("commitment_type"),
            strategy=strategy,
            model_id=model_id,
        )
        return result
    except Exception as exc:
        logger.warning("ThreadTriageResult validation failed: %s", exc)
        return None


def _parse_one_result(obj: dict) -> TriageResult | None:
    """Parse a single dict into a validated TriageResult."""
    try:
        return TriageResult.model_validate(obj)
    except Exception:
        if "score" in obj:
            try:
                return TriageResult(
                    score=float(obj["score"]),
                    reason=str(obj.get("reason", "")),
                )
            except (ValueError, TypeError):
                pass
        if "relevant" in obj:
            try:
                return TriageResult(
                    score=0.7 if obj["relevant"] else 0.1,
                    reason=str(obj.get("reason", "")),
                )
            except Exception:
                pass
    return None


def parse_msg_batch_response(
    raw: str, batch_ids: list[str],
) -> dict[str, TriageResult | None]:
    """Parse Qwen JSON response for a batch of events.

    Multi-strategy: standard JSON -> regex extraction.
    Returns {event_id: TriageResult | None} for each item.
    """
    if not raw:
        return {eid: None for eid in batch_ids}

    cleaned = _clean_llm_output(raw)

    # Try standard JSON parsing
    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    if parsed is None:
        results: dict[str, TriageResult | None] = {}
        found: list[tuple[str, TriageResult]] = []
        for match in re.finditer(r'\{[^{}]+\}', cleaned):
            try:
                obj = json.loads(match.group())
                result = _parse_one_result(obj)
                if result:
                    found.append((obj.get("id", ""), result))
            except (json.JSONDecodeError, ValueError):
                continue

        for i, eid in enumerate(batch_ids):
            matched = None
            for fid, fparsed in found:
                if fid == eid:
                    matched = fparsed
                    break
            if matched is None and i < len(found):
                matched = found[i][1]
            results[eid] = matched

        if not any(v is not None for v in results.values()):
            logger.warning(
                "Failed to parse triage response: %s...", raw[:300]
            )
        return results

    results = {eid: None for eid in batch_ids}

    if isinstance(parsed, list):
        id_map: dict[str, dict] = {}
        for item in parsed:
            if isinstance(item, dict) and "id" in item:
                id_map[str(item["id"])] = item

        for i, eid in enumerate(batch_ids):
            item = id_map.get(eid)
            if not item and i < len(parsed) and isinstance(parsed[i], dict):
                item = parsed[i]
            if item:
                results[eid] = _parse_one_result(item)

    elif isinstance(parsed, dict):
        # Handle {"items": [...]} or {"results": [...]} wrapper
        inner = parsed.get("items") or parsed.get("results") or parsed.get("messages")
        if isinstance(inner, list):
            id_map: dict[str, dict] = {}
            for item in inner:
                if isinstance(item, dict) and "id" in item:
                    id_map[str(item["id"])] = item
            for i, eid in enumerate(batch_ids):
                item = id_map.get(eid)
                if not item and i < len(inner) and isinstance(inner[i], dict):
                    item = inner[i]
                if item:
                    results[eid] = _parse_one_result(item)
        else:
            result = _parse_one_result(parsed)
            if batch_ids:
                obj_id = str(parsed.get("id", ""))
                if obj_id in results:
                    results[obj_id] = result
                else:
                    results[batch_ids[0]] = result

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Score floor rules
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _assign_temporal_bucket(timestamp: int, now: int) -> int:
    """Assign a temporal bucket to an event."""
    age_days = (now - timestamp) / SECONDS_PER_DAY
    if age_days <= 7:
        return 1
    if age_days <= 30:
        return 2
    if age_days <= 90:
        return 3
    if age_days <= 365:
        return 4
    return 5


AUTOMATED_SENDER_PATTERNS = frozenset({
    "noreply", "no-reply", "notifications", "alerts",
    "mailer-daemon", "donotreply", "automated", "marketing",
    "newsletter", "updates@", "info@", "support@", "hello@",
})

GENERIC_SALUTATIONS = (
    "dear user", "dear customer", "dear sir", "dear madam",
    "dear sir/madam", "dear account holder", "dear valued",
    "dear member",
)


def _is_automated_sender(sender_address: str) -> bool:
    """Check if sender looks like an automated/marketing address."""
    addr = sender_address.lower()
    return any(pat in addr for pat in AUTOMATED_SENDER_PATTERNS)


def _has_generic_salutation(body: str) -> bool:
    """Check for phishing/marketing salutations in the body opening."""
    if not body:
        return False
    opening = body[:200].lower()
    return any(sal in opening for sal in GENERIC_SALUTATIONS)


def apply_score_floor(
    score: float,
    event_type: str,
    sender_tier: int,
    commitment_type: str | None,
    sender_address: str = "",
    sender_msg_count: int = 999,
    body: str = "",
) -> float:
    """Post-LLM deterministic floor rules.

    Meetings -> min 0.5
    Calendar events -> min 0.3
    Tier-1 sender -> min 0.3
    commitment_type non-null -> min 0.7 (UNLESS sender is automated/spam)
    """
    if event_type == EVENT_TYPE_MEETING:
        score = max(score, SCORE_FLOOR_MEETING)
    elif event_type == EVENT_TYPE_CALENDAR:
        score = max(score, SCORE_FLOOR_CALENDAR)
    if sender_tier == 1 and score < SCORE_FLOOR_TIER1_SENDER:
        score = SCORE_FLOOR_TIER1_SENDER

    # Skip the commitment_type floor for automated/spam senders
    is_automated = _is_automated_sender(sender_address)
    is_generic_low_history = (
        sender_msg_count <= 2 and _has_generic_salutation(body)
    )
    if commitment_type is not None and not is_automated and not is_generic_low_history:
        score = max(score, SCORE_FLOOR_COMMITMENT)
    return round(score, 1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thread-aware incremental triage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_thread_prior_context(
    thread_id: str,
    store: LayeredGraphStore,
) -> dict[str, Any] | None:
    """Fetch prior thread_triage claim data for incremental triage.

    Returns dict with thread_summary, thread_score, thread_status,
    commitment_type — or None if the thread has never been triaged.
    """
    claim_id = thread_triage_claim_id(thread_id)
    claim = store.get_claim(claim_id)
    if not claim or claim.claim_type != "thread_triage":
        return None

    try:
        data = json.loads(claim.object)
        return {
            "thread_summary": data.get("thread_summary", ""),
            "thread_score": data.get("thread_score", 0),
            "thread_status": data.get("thread_status", ""),
            "commitment_type": data.get("commitment_type"),
            "domain": data.get("domain", ""),
            "universal_spheres": data.get("universal_spheres", []),
            "specific_topics": data.get("specific_topics", data.get("topics", [])),
        }
    except (json.JSONDecodeError, TypeError):
        return None


def _get_triaged_event_ids(
    event_ids: list[str],
    store: LayeredGraphStore,
) -> set[str]:
    """Return the subset of event_ids that already have triage claims."""
    triaged: set[str] = set()
    for eid in event_ids:
        claim = store.get_claim(triage_claim_id(eid))
        if claim and claim.claim_type == "triage":
            try:
                data = json.loads(claim.object)
                if data.get("reason") != "PARSE_FAILED":
                    triaged.add(eid)
            except (json.JSONDecodeError, TypeError):
                pass
    return triaged
def build_incremental_thread_prompt(
    thread_id: str,
    all_events: list[dict],
    triaged_ids: set[str],
    prior_context: dict[str, Any],
    store: LayeredGraphStore,
    sender_cache: dict[str, dict[str, Any]],
    now: int,
    profile_context: str = "",
) -> str:
    """Build an incremental triage prompt for a thread with new messages.

    Shows the N most recent messages with [NEW]/[CONTEXT] markers,
    plus a PRIOR THREAD CONTEXT header from the previous triage.
    """
    source = all_events[0]["source"] if all_events else "unknown"
    sender_pid = _get_primary_sender_pid(all_events, store)

    # Build context enrichment
    context_block = ""
    if sender_pid:
        context_block = build_full_context(
            sender_pid, thread_id, store, target_date=now, now=now,
        )

    header_parts = [f"THREAD: {thread_id}", f"SOURCE: {source}"]

    # Add prior context block
    prior_block = PRIOR_THREAD_CONTEXT_BLOCK.format(
        thread_summary=prior_context.get("thread_summary", ""),
        thread_score=prior_context.get("thread_score", 0),
        thread_status=prior_context.get("thread_status", ""),
        domain=prior_context.get("domain", ""),
        commitment_type=prior_context.get("commitment_type") or "null",
    )
    header_parts.append("")
    header_parts.append(prior_block)

    if context_block:
        header_parts.append("")
        header_parts.append(context_block)

    if profile_context:
        header_parts.append("")
        header_parts.append(profile_context)

    context_chars = sum(len(p) for p in header_parts)

    # Take the most recent N messages
    recent_events = all_events[-INCREMENTAL_CONTEXT_MESSAGES:]
    display_events = _compact_thread_for_budget(recent_events, context_chars)

    parts = list(header_parts)
    parts.append("")
    parts.append("=== MESSAGES (chronological) ===")

    if len(all_events) > len(display_events):
        parts.append(
            f"[... {len(all_events) - len(display_events)} older messages omitted, "
            f"showing most recent {len(display_events)} ...]"
        )

    tz = safe_timezone()

    for e in display_events:
        ts = e["timestamp"]
        dt = datetime.fromtimestamp(ts, tz=tz)
        time_str = dt.strftime("%m/%d %H:%M")

        meta = json.loads(e.get("metadata") or "{}")
        is_from_me = meta.get("is_from_me", False)

        if is_from_me:
            sender_name = "USER"
        else:
            sender_info = _get_event_sender(store, e["id"], sender_cache)
            sender_name = sender_info["name"]

        body = _safe_body(e.get("raw_content") or "")

        # Mark messages as [NEW] or [CONTEXT]
        marker = "[CONTEXT]" if e["id"] in triaged_ids else "[NEW]"

        subject = meta.get("subject", "")
        if subject and source == "mail":
            parts.append(
                f"{marker} [{time_str}] {sender_name} (id:{e['id'][:12]}): "
                f"[Subject: {subject}] {body}"
            )
        else:
            parts.append(
                f"{marker} [{time_str}] {sender_name} (id:{e['id'][:12]}): {body}"
            )

    # Only request scores for NEW messages
    new_events = [e for e in display_events if e["id"] not in triaged_ids]
    if new_events:
        parts.append(THREAD_FULL_SUFFIX + (
            f"\n\nIMPORTANT: Only provide message_scores for the NEW messages "
            f"({len(new_events)} messages). CONTEXT messages already have scores."
        ))
    else:
        parts.append(THREAD_FULL_SUFFIX)

    return "\n".join(parts)


def classify_thread_incremental(
    thread_id: str,
    all_thread_events: list[dict],
    new_event_ids: set[str],
    store: LayeredGraphStore,
) -> str:
    """Classify a thread for incremental processing.

    Returns one of:
      'fresh'       — no prior triage, standard full triage
      'incremental' — known thread with new messages, use incremental prompt
      'reactivated' — dormant thread with new interesting messages, full reprocess
    """
    prior = _get_thread_prior_context(thread_id, store)

    if prior is None:
        return "fresh"

    prior_score = prior.get("thread_score", 0)

    # Path C: Dormant thread reactivation
    if prior_score < REACTIVATION_THRESHOLD:
        # Check if any new event has an extraction gate claim
        has_gate = store.conn.execute(
            """SELECT 1 FROM claims
               WHERE claim_type = 'extraction_gate'
                 AND subject = ?
               LIMIT 1""",
            (thread_id,),
        ).fetchone()

        # Dormant thread (low score, no extraction) gets reactivated
        if not has_gate:
            return "reactivated"

    # Path B: Known thread with new messages
    return "incremental"


def _fetch_all_thread_events(
    thread_id: str,
    store: LayeredGraphStore,
    limit: int = MAX_THREAD_FETCH,
) -> list[dict]:
    """Fetch the most recent events for a thread from the store.

    Returns the *tail* (most recent ``limit`` events, default 500) in
    chronological order.  Older messages beyond the cap are dropped —
    for a 4 000-message WhatsApp group we only need the recent tail for
    triage context, not years of chat history.
    """
    rows = store.conn.execute(
        """SELECT * FROM (
               SELECT e.* FROM events e
               WHERE e.id IN (
                   SELECT ce.event_id FROM claim_events ce
                   JOIN claims c ON ce.claim_id = c.id
                   WHERE c.claim_type = 'thread_triage'
                     AND c.subject = ?
               )
               OR json_extract(e.metadata, '$.thread_id') = ?
               ORDER BY e.timestamp DESC
               LIMIT ?
           ) sub ORDER BY sub.timestamp ASC""",
        (thread_id, thread_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Topic annotation emission
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _emit_topic_annotations(
    store: LayeredGraphStore,
    event_id: str,
    topics: list[str],
    source: str,
) -> None:
    """Write triage-extracted topics as annotations, normalizing via synonym map."""
    from alteris.models import Annotation
    from alteris.topic_normalize import normalize_topic

    now = int(time.time())
    for raw in topics:
        if not raw:
            continue
        canonical = store.get_topic_canonical(raw) or normalize_topic(raw, store)
        store.put_annotation(Annotation(
            event_id=event_id, facet="topic", value=canonical,
            confidence=0.8, source=source, created_at=now,
        ))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Claim construction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def triage_claim_id(event_id: str) -> str:
    """Deterministic claim ID for a per-event triage result."""
    raw = f"triage:{event_id}:triage_result"
    return f"claim:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def thread_triage_claim_id(thread_id: str) -> str:
    """Deterministic claim ID for a thread-level triage result."""
    raw = f"thread_triage:{thread_id}:thread_result"
    return f"claim:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def _per_event_claim_from_thread(
    event_id: str,
    msg_score: MessageScore | None,
    thread_result: ThreadTriageResult,
    event_type: str,
    sender_tier: int,
    model_id: str,
    sender_address: str = "",
    sender_msg_count: int = 999,
    body: str = "",
) -> Claim:
    """Create a per-event triage claim from thread-level results.

    Maintains backward compatibility with propagate.py and extract.py:
      claim_type = "triage"
      subject = event_id
      predicate = "triage_result"
      object = JSON with score, reason, domain, topics, pii, etc.
    """
    score = msg_score.score if msg_score else thread_result.thread_score
    reason = msg_score.reason if msg_score else thread_result.thread_summary

    # Apply score floors
    score = apply_score_floor(
        score, event_type, sender_tier,
        thread_result.commitment_type,
        sender_address=sender_address,
        sender_msg_count=sender_msg_count,
        body=body,
    )

    if thread_result.pii or thread_result.sensitivity:
        sensitivity = SensitivityLevel.CRITICAL
    elif thread_result.domain in ("health", "legal", "financial"):
        sensitivity = SensitivityLevel.SENSITIVE
    else:
        sensitivity = SensitivityLevel.SENSITIVE

    return Claim(
        id=triage_claim_id(event_id),
        event_ids=[event_id],
        claim_type="triage",
        subject=event_id,
        predicate="triage_result",
        object=json.dumps({
            "score": score,
            "reason": reason[:200],
            "domain": thread_result.domain,
            "universal_spheres": thread_result.universal_spheres,
            "specific_topics": thread_result.specific_topics,
            "topics": thread_result.specific_topics,
            "entities": [],
            "pii": thread_result.pii,
            "sensitivity": thread_result.sensitivity,
            "commitment_type": thread_result.commitment_type,
        }),
        confidence=score,
        modality=Modality.OBSERVED,
        provenance=ExtractionProvenance(
            model_id=model_id,
            prompt_version=PROMPT_VERSION,
            extraction_method=(
                ExtractionMethod.CLOUD_MODEL
                if thread_result.strategy == TriageStrategy.THREAD_FULL
                else ExtractionMethod.LOCAL_MODEL
            ),
        ),
        sensitivity=sensitivity,
    )


def _per_event_claim_from_triage_result(
    event_id: str,
    result: TriageResult,
    model_id: str,
) -> Claim:
    """Create a per-event triage claim from MSG_COMPACT/MSG_BATCH result."""
    if result.pii or result.sensitivity:
        sensitivity = SensitivityLevel.CRITICAL
    elif result.domain in ("health", "legal", "financial"):
        sensitivity = SensitivityLevel.SENSITIVE
    else:
        sensitivity = SensitivityLevel.SENSITIVE

    return Claim(
        id=triage_claim_id(event_id),
        event_ids=[event_id],
        claim_type="triage",
        subject=event_id,
        predicate="triage_result",
        object=json.dumps({
            "score": result.score,
            "reason": result.reason,
            "domain": result.domain,
            "universal_spheres": result.universal_spheres,
            "specific_topics": result.specific_topics,
            "topics": result.specific_topics,
            "entities": result.entities,
            "pii": result.pii,
            "sensitivity": result.sensitivity,
            "commitment_type": result.commitment_type,
        }),
        confidence=result.score,
        modality=Modality.OBSERVED,
        provenance=ExtractionProvenance(
            model_id=model_id,
            prompt_version=PROMPT_VERSION,
            extraction_method=ExtractionMethod.CLOUD_MODEL,
        ),
        sensitivity=sensitivity,
    )


def _thread_level_claim(
    thread_id: str,
    event_ids: list[str],
    result: ThreadTriageResult,
    model_id: str,
) -> Claim:
    """Create a thread-level triage claim (additive, new claim type)."""
    return Claim(
        id=thread_triage_claim_id(thread_id),
        event_ids=event_ids,
        claim_type="thread_triage",
        subject=thread_id,
        predicate="thread_triage_result",
        object=json.dumps({
            "thread_score": result.thread_score,
            "domain": result.domain,
            "universal_spheres": result.universal_spheres,
            "specific_topics": result.specific_topics,
            "topics": result.specific_topics,
            "thread_status": result.thread_status,
            "relationship": result.relationship,
            "thread_summary": result.thread_summary,
            "extraction_candidates": result.extraction_candidates,
            "pii": result.pii,
            "sensitivity": result.sensitivity,
            "commitment_type": result.commitment_type,
            "strategy": result.strategy,
            "n_messages": len(event_ids),
        }),
        confidence=result.thread_score,
        modality=Modality.OBSERVED,
        provenance=ExtractionProvenance(
            model_id=model_id,
            prompt_version=PROMPT_VERSION,
            extraction_method=(
                ExtractionMethod.CLOUD_MODEL
                if result.strategy == TriageStrategy.THREAD_FULL
                else ExtractionMethod.LOCAL_MODEL
            ),
        ),
        sensitivity=SensitivityLevel.SENSITIVE,
    )


def _failed_claim(event_id: str, model_id: str) -> Claim:
    """Placeholder claim for events that failed parsing."""
    return Claim(
        id=triage_claim_id(event_id),
        event_ids=[event_id],
        claim_type="triage",
        subject=event_id,
        predicate="triage_result",
        object=json.dumps({"score": 0.1, "reason": "PARSE_FAILED"}),
        confidence=0.0,
        modality=Modality.UNKNOWN,
        provenance=ExtractionProvenance(
            model_id=model_id,
            prompt_version=PROMPT_VERSION,
            extraction_method=ExtractionMethod.LOCAL_MODEL,
        ),
        sensitivity=SensitivityLevel.SENSITIVE,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_triageable_events(
    store: LayeredGraphStore,
    resume: bool = True,
    since_ts: int = 0,
    lens: str = "",
) -> list[dict]:
    """Fetch events eligible for triage.

    Excludes identity/call events and (if resume) events already triaged.
    When a lens is provided, also excludes events with projection route='skip'
    for that lens — these were scored as noise by Stage 2.

    Events without any projection are still included (backward compatible
    with runs where scoring hasn't been done yet).

    Args:
        since_ts: Only include events with timestamp >= this value.
            Pass 0 for all events. Essential for limiting scope when
            the DB contains years of historical data.
        lens: Scoring lens to check projections against. If empty,
            no projection filtering is applied.
    """
    excluded_types_sql = ",".join(f"'{t}'" for t in EXCLUDED_EVENT_TYPES)

    # Skip-route filter: exclude events the scorer identified as noise.
    # LEFT JOIN ensures events without projections are still included.
    skip_filter = ""
    skip_params: list = []
    if lens:
        skip_filter = (
            " AND NOT EXISTS ("
            "   SELECT 1 FROM projections p"
            "   WHERE p.event_id = e.id AND p.lens = ? AND p.route = 'skip'"
            " )"
        )
        skip_params = [lens]

    if resume:
        rows = store.conn.execute(
            f"""SELECT e.* FROM events e
                WHERE e.event_type NOT IN ({excluded_types_sql})
                  AND e.raw_content IS NOT NULL
                  AND e.timestamp >= ?
                  AND NOT EXISTS (
                      SELECT 1 FROM claims c
                      WHERE c.subject = e.id
                        AND c.claim_type = 'triage'
                        AND c.predicate = 'triage_result'
                        AND c.object NOT LIKE '%PARSE_FAILED%'
                  )
                  {skip_filter}
                ORDER BY e.timestamp DESC""",
            [since_ts] + skip_params,
        ).fetchall()
    else:
        rows = store.conn.execute(
            f"""SELECT e.* FROM events e
                WHERE e.event_type NOT IN ({excluded_types_sql})
                  AND e.raw_content IS NOT NULL
                  AND e.timestamp >= ?
                  {skip_filter}
                ORDER BY e.timestamp DESC""",
            [since_ts] + skip_params,
        ).fetchall()

    return [dict(r) for r in rows]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thread triage execution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _summarize_head(
    thread_id: str,
    head_events: list[dict],
    store: LayeredGraphStore,
    sender_cache: dict[str, dict[str, Any]],
    llm_client: LLMClient,
) -> str:
    """Pass 1 of two-pass triage: summarize older messages (sync)."""
    model_id, _ = _model_for_thread(thread_id)
    prompt = build_summary_prompt(thread_id, head_events, store, sender_cache)
    raw = llm_client.generate(
        prompt=prompt,
        system=THREAD_SUMMARY_SYSTEM,
        model=model_id,
        temperature=0.1,
        max_tokens=2048,
        format_json=True,
    )
    if not raw:
        return f"[{len(head_events)} earlier messages — summary unavailable]"
    try:
        data = json.loads(_clean_llm_output(raw))
        parts = [data.get("summary", "")]
        if data.get("open_items"):
            parts.append("Open items: " + "; ".join(data["open_items"]))
        if data.get("participants_summary"):
            parts.append("Participants: " + data["participants_summary"])
        return "\n".join(p for p in parts if p)
    except (json.JSONDecodeError, TypeError):
        return raw.strip()[:2000]


# Granola transcripts: ~4 chars per token, leave room for system prompt + context
GRANOLA_SECTION_CHAR_BUDGET = (THREAD_INPUT_TOKEN_BUDGET - 10_000) * 4


def _split_granola_sections(body: str) -> list[str]:
    """Split a granola transcript into sections by markdown headers.

    Splits on ## or ### headers. Merges small consecutive sections
    to fill the token budget. Returns a list of section strings,
    each within the character budget.
    """
    lines = body.split("\n")
    sections: list[list[str]] = [[]]

    for line in lines:
        if re.match(r"^#{2,3}\s", line) and sections[-1]:
            sections.append([])
        sections[-1].append(line)

    # Filter empty sections
    sections = [s for s in sections if any(l.strip() for l in s)]

    if not sections:
        return [body]

    # Merge small sections to fill budget
    merged: list[str] = []
    current: list[str] = []
    current_len = 0

    for section_lines in sections:
        section_text = "\n".join(section_lines)
        section_len = len(section_text)

        if current_len + section_len > GRANOLA_SECTION_CHAR_BUDGET and current:
            merged.append("\n".join(current))
            current = [section_text]
            current_len = section_len
        else:
            current.append(section_text)
            current_len += section_len

    if current:
        merged.append("\n".join(current))

    return merged if merged else [body]


def _triage_granola_sectioned(
    thread_id: str,
    event: dict,
    store: LayeredGraphStore,
    sender_cache: dict[str, dict[str, Any]],
    llm_client: LLMClient,
    now: int,
    profile_context: str = "",
) -> ThreadTriageResult | None:
    """Triage a granola transcript by splitting into sections.

    Each section is triaged independently with the same event ID,
    then results are merged (keep highest score, union topics).
    """
    body = event.get("raw_content") or ""
    sections = _split_granola_sections(body)
    model_id, max_output = _model_for_thread(thread_id)

    t_start = time.time()
    logger.info(
        "Granola sectioned triage for %s: %d chars -> %d sections [%s]",
        thread_id[:30], len(body), len(sections), model_id,
    )

    section_results: list[ThreadTriageResult | None] = []
    for i, section_body in enumerate(sections):
        # Create a synthetic event with the section body
        section_event = dict(event)
        section_event["raw_content"] = section_body

        prompt = build_thread_full_prompt(
            thread_id, [section_event], store, sender_cache, now,
            profile_context,
        )
        raw = llm_client.generate(
            prompt=prompt,
            system=THREAD_FULL_SYSTEM,
            model=model_id,
            temperature=0.1,
            max_tokens=max_output,
            format_json=True,
            cache_system=True,
        )
        result = parse_thread_triage_response(
            raw or "", thread_id, [event["id"]],
            strategy=TriageStrategy.THREAD_FULL,
            model_id=model_id,
        )
        section_results.append(result)
        logger.info(
            "  section %d/%d (%d chars) -> %.1fs",
            i + 1, len(sections), len(section_body), time.time() - t_start,
        )

    # Merge: take highest score, union spheres/topics
    valid = [r for r in section_results if r is not None]
    if not valid:
        return None

    best = max(valid, key=lambda r: r.thread_score)
    all_spheres: set[str] = set()
    all_specific: set[str] = set()
    all_pii: set[str] = set()
    all_sensitivity: set[str] = set()

    for r in valid:
        all_spheres.update(r.universal_spheres)
        all_specific.update(r.specific_topics)
        all_pii.update(r.pii)
        all_sensitivity.update(r.sensitivity)

    merged = ThreadTriageResult(
        thread_id=best.thread_id,
        thread_score=best.thread_score,
        domain=best.domain,
        universal_spheres=list(all_spheres)[:3],
        specific_topics=list(all_specific)[:5],
        thread_status=best.thread_status,
        relationship=best.relationship,
        thread_summary=best.thread_summary,
        message_scores=best.message_scores,
        extraction_candidates=best.extraction_candidates,
        pii=list(all_pii),
        sensitivity=list(all_sensitivity),
        commitment_type=best.commitment_type,
        strategy=best.strategy,
        model_id=best.model_id,
    )

    logger.info(
        "Granola sectioned triage %s: %d sections, %.1fs total, score=%.1f",
        thread_id[:30], len(sections), time.time() - t_start, merged.thread_score,
    )
    return merged


def _triage_thread_full(
    thread_id: str,
    events: list[dict],
    store: LayeredGraphStore,
    sender_cache: dict[str, dict[str, Any]],
    llm_client: LLMClient,
    now: int,
    profile_context: str = "",
) -> ThreadTriageResult | None:
    """Execute THREAD_FULL triage: full thread via Gemini.

    For threads > LARGE_THREAD_THRESHOLD messages, uses chunked triage:
    split into overlapping chunks of CHUNK_SIZE with CHUNK_OVERLAP,
    triage each independently, merge results (keep highest score per message).

    For granola transcripts that exceed the token budget, splits by
    markdown sections and triages each section independently.
    """
    n_msgs = len(events)
    model_id, max_output = _model_for_thread(thread_id, n_msgs)

    # Granola: single-event threads with large transcripts — split by section
    source = events[0].get("source", "") if events else ""
    if source == "granola" and len(events) == 1:
        body_len = len(events[0].get("raw_content") or "")
        if body_len > GRANOLA_SECTION_CHAR_BUDGET:
            return _triage_granola_sectioned(
                thread_id, events[0], store, sender_cache, llm_client, now,
                profile_context,
            )

    if n_msgs > LARGE_THREAD_THRESHOLD:
        chunks = _chunk_thread(events)
        t_start = time.time()
        logger.info(
            "Chunked triage for %s: %d msgs -> %d chunks of ~%d [%s]",
            thread_id[:30], n_msgs, len(chunks), CHUNK_SIZE, model_id,
        )

        chunk_results: list[ThreadTriageResult | None] = []
        for i, chunk in enumerate(chunks):
            t_chunk = time.time()
            chunk_ids = [e["id"] for e in chunk]
            prompt = build_thread_full_prompt(
                thread_id, chunk, store, sender_cache, now,
                profile_context,
            )
            raw = llm_client.generate(
                prompt=prompt,
                system=THREAD_FULL_SYSTEM,
                model=model_id,
                temperature=0.1,
                max_tokens=max_output,
                format_json=True,
                cache_system=True,
            )
            result = parse_thread_triage_response(
                raw or "", thread_id, chunk_ids,
                strategy=TriageStrategy.THREAD_FULL,
                model_id=model_id,
            )
            chunk_results.append(result)
            logger.info(
                "  chunk %d/%d (%d msgs) -> %.1fs",
                i + 1, len(chunks), len(chunk), time.time() - t_chunk,
            )

        merged = _merge_chunk_results(chunk_results, chunks)
        logger.info(
            "Chunked triage %s: %d msgs, %d chunks, %.1fs total (%.1fs/chunk avg)",
            thread_id[:30], n_msgs, len(chunks),
            time.time() - t_start, (time.time() - t_start) / len(chunks),
        )
        return merged

    t_start = time.time()
    prompt = build_thread_full_prompt(
        thread_id, events, store, sender_cache, now,
        profile_context,
    )
    raw = llm_client.generate(
        prompt=prompt,
        system=THREAD_FULL_SYSTEM,
        model=model_id,
        temperature=0.1,
        max_tokens=max_output,
        format_json=True,
        cache_system=True,
    )
    logger.info(
        "Thread triage %s: %d msgs, %.1fs [%s]",
        thread_id[:30], n_msgs, time.time() - t_start, model_id,
    )

    event_ids = [e["id"] for e in events]
    return parse_thread_triage_response(
        raw or "", thread_id, event_ids,
        strategy=TriageStrategy.THREAD_FULL,
        model_id=model_id,
    )


def _triage_msg_batch(
    thread_id: str,
    events: list[dict],
    store: LayeredGraphStore,
    sender_cache: dict[str, dict[str, Any]],
    llm_client: LLMClient,
    now: int,
    profile_context: str = "",
) -> list[tuple[str, TriageResult | None]]:
    """Execute MSG_BATCH_COMPACT triage: batches of messages via Flash Lite.

    Returns list of (event_id, TriageResult) pairs.
    """
    model_id, _ = _model_for_thread(thread_id, len(events))
    all_results: list[tuple[str, TriageResult | None]] = []

    for i in range(0, len(events), MSG_BATCH_SIZE):
        batch = events[i:i + MSG_BATCH_SIZE]
        batch_ids = [e["id"] for e in batch]

        prompt = build_msg_batch_prompt(batch, store, sender_cache, now, profile_context)

        raw = llm_client.generate(
            prompt=prompt,
            system=MSG_COMPACT_SYSTEM,
            model=model_id,
            temperature=0.1,
            max_tokens=MSG_COMPACT_MAX_OUTPUT_TOKENS * len(batch),
            format_json=True,
        )

        results = parse_msg_batch_response(raw or "", batch_ids)
        for eid in batch_ids:
            all_results.append((eid, results.get(eid)))

    return all_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _write_thread_results(
    store: LayeredGraphStore,
    thread_id: str,
    thread_events: list[dict],
    result: ThreadTriageResult | None,
    event_meta: dict[str, dict],
    stats: dict[str, Any],
    is_incremental: bool = False,
) -> None:
    """Write per-event + thread-level claims from a THREAD_FULL result.

    When is_incremental=True, supersedes the old thread_triage claim
    and creates a new one with an updated ID suffix.
    """
    event_ids = [e["id"] for e in thread_events]
    model_id, _ = _model_for_thread(thread_id, len(thread_events))

    if result:
        msg_score_map: dict[str, MessageScore] = {}
        for ms in result.message_scores:
            for eid in event_ids:
                if ms.id == eid or eid.startswith(ms.id):
                    msg_score_map[eid] = ms
                    break

        for eid in event_ids:
            meta = event_meta.get(eid, {})
            ms = msg_score_map.get(eid)
            claim = _per_event_claim_from_thread(
                eid, ms, result,
                meta.get("event_type", ""),
                meta.get("sender_tier", 3),
                model_id,
                sender_address=meta.get("sender_address", ""),
                sender_msg_count=meta.get("sender_msg_count", 999),
                body=meta.get("body_preview", ""),
            )
            if store.put_claim(claim):
                stats["claims_written"] += 1

            # Emit topic annotations for this event (specific_topics, falling back to legacy topics)
            topics_for_ann = result.specific_topics or result.topics
            _emit_topic_annotations(store, eid, topics_for_ann, f"triage:{model_id}")

            score = claim.confidence
            if score < 0.3:
                stats["ignore"] += 1
            elif score < 0.7:
                stats["lightweight"] += 1
            else:
                stats["deep"] += 1
            stats["triaged"] += 1

        # For incremental triage, supersede the old thread claim
        old_thread_claim_id = thread_triage_claim_id(thread_id)
        if is_incremental:
            # Create new claim with a versioned ID
            ts_suffix = str(int(time.time()))
            new_id_raw = f"thread_triage:{thread_id}:thread_result:{ts_suffix}"
            new_claim_id = f"claim:{hashlib.sha256(new_id_raw.encode()).hexdigest()[:16]}"

            thread_claim = _thread_level_claim(
                thread_id, event_ids, result, model_id,
            )
            # Override the ID with the new versioned one
            thread_claim = Claim(
                id=new_claim_id,
                event_ids=thread_claim.event_ids,
                claim_type=thread_claim.claim_type,
                subject=thread_claim.subject,
                predicate=thread_claim.predicate,
                object=thread_claim.object,
                confidence=thread_claim.confidence,
                modality=thread_claim.modality,
                provenance=thread_claim.provenance,
                sensitivity=thread_claim.sensitivity,
            )

            if store.put_claim(thread_claim):
                stats["claims_written"] += 1
            # Supersede the old claim
            store.supersede_claim(old_thread_claim_id, new_claim_id)
        else:
            thread_claim = _thread_level_claim(
                thread_id, event_ids, result, model_id,
            )
            if store.put_claim(thread_claim):
                stats["claims_written"] += 1

        stats["thread_full"] += 1
    else:
        for eid in event_ids:
            store.put_claim(_failed_claim(eid, model_id))
            stats["failed"] += 1


async def _async_triage_one_thread(
    thread_id: str,
    thread_events: list[dict],
    store: LayeredGraphStore,
    sender_cache: dict[str, dict[str, Any]],
    llm_client: LLMClient,
    now: int,
    semaphore: asyncio.Semaphore,
    prior_context: dict[str, Any] | None = None,
    all_thread_events: list[dict] | None = None,
    profile_context: str = "",
) -> tuple[str, ThreadTriageResult | None]:
    """Triage a single thread via Gemini, with concurrency limit.

    Supports three modes:
      1. Standard: fresh thread, full triage
      2. Incremental: known thread with new messages, augmented prompt
      3. Large thread: chunked (overlapping windows, merge results)

    Args:
        prior_context: Prior thread_triage data for incremental mode.
        all_thread_events: ALL thread events (new + old) for incremental context.
    """
    async with semaphore:
        n_msgs = len(thread_events)
        model_id, max_output = _model_for_thread(thread_id, n_msgs)

        # Incremental mode: known thread with prior context
        if prior_context is not None and all_thread_events is not None:
            new_ids = {e["id"] for e in thread_events}
            triaged_ids = {e["id"] for e in all_thread_events if e["id"] not in new_ids}

            logger.info(
                "Incremental triage for %s: %d all events, %d new [%s]",
                thread_id[:30], len(all_thread_events), len(new_ids), model_id,
            )

            triage_prompt = build_incremental_thread_prompt(
                thread_id, all_thread_events, triaged_ids,
                prior_context, store, sender_cache, now,
                profile_context,
            )
        elif n_msgs > LARGE_THREAD_THRESHOLD:
            # Chunked triage: split into overlapping windows, triage each
            chunks = _chunk_thread(thread_events)
            t_start = time.time()
            logger.info(
                "Chunked triage for %s: %d msgs -> %d chunks of ~%d [%s]",
                thread_id[:30], n_msgs, len(chunks), CHUNK_SIZE, model_id,
            )

            chunk_results: list[ThreadTriageResult | None] = []
            for i, chunk in enumerate(chunks):
                t_chunk = time.time()
                chunk_ids = [e["id"] for e in chunk]
                chunk_prompt = build_thread_full_prompt(
                    thread_id, chunk, store, sender_cache, now,
                    profile_context,
                )
                try:
                    chunk_raw = await asyncio.wait_for(
                        llm_client.agenerate(
                            prompt=chunk_prompt,
                            system=THREAD_FULL_SYSTEM,
                            model=model_id,
                            temperature=0.1,
                            max_tokens=max_output,
                            format_json=True,
                            cache_system=True,
                        ),
                        timeout=ASYNC_CALL_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Chunk %d/%d timed out after %ds for %s", i + 1, len(chunks), ASYNC_CALL_TIMEOUT, thread_id[:30])
                    chunk_raw = None
                chunk_result = parse_thread_triage_response(
                    chunk_raw or "", thread_id, chunk_ids,
                    strategy=TriageStrategy.THREAD_FULL,
                    model_id=model_id,
                )
                chunk_results.append(chunk_result)
                chunk_elapsed = time.time() - t_chunk
                ok = "ok" if chunk_raw else "FAILED"
                logger.info(
                    "  chunk %d/%d (%d msgs) -> %.1fs [%s]",
                    i + 1, len(chunks), len(chunk), chunk_elapsed, ok,
                )
                print(
                    f"    {thread_id[:25]} chunk {i+1}/{len(chunks)} "
                    f"({len(chunk)} msgs) -> {chunk_elapsed:.0f}s [{ok}]",
                    flush=True,
                )

            result = _merge_chunk_results(chunk_results, chunks)
            logger.info(
                "Chunked triage %s: %d msgs, %d chunks, %.1fs total (%.1fs/chunk avg)",
                thread_id[:30], n_msgs, len(chunks),
                time.time() - t_start, (time.time() - t_start) / len(chunks),
            )
            return thread_id, result
        else:
            # Standard single-pass
            triage_prompt = build_thread_full_prompt(
                thread_id, thread_events, store, sender_cache, now,
                profile_context,
            )

        try:
            raw = await asyncio.wait_for(
                llm_client.agenerate(
                    prompt=triage_prompt,
                    system=THREAD_FULL_SYSTEM,
                    model=model_id,
                    temperature=0.1,
                    max_tokens=max_output,
                    format_json=True,
                    cache_system=True,
                ),
                timeout=ASYNC_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Thread triage timed out after %ds for %s", ASYNC_CALL_TIMEOUT, thread_id[:30])
            raw = None

        event_ids = [e["id"] for e in thread_events]
        result = parse_thread_triage_response(
            raw or "", thread_id, event_ids,
            strategy=TriageStrategy.THREAD_FULL,
            model_id=model_id,
        )
        return thread_id, result


def run_triage(
    store: LayeredGraphStore,
    llm_client: LLMClient,
    resume: bool = True,
    max_concurrent: int = 10,
    since_ts: int = 0,
    lens: str = "",
    cloud_all: bool = False,
    profile_context: str = "",
) -> dict[str, Any]:
    """Run thread-based LLM triage on events, output Claims.

    Uses asyncio for concurrent Gemini calls. Large threads (>200 msgs)
    get chunked triage: overlapping windows, each triaged independently, merged.

    Args:
        max_concurrent: Max parallel Gemini API calls.
        since_ts: Only triage events with timestamp >= this value.
        lens: Scoring lens. When provided, skip-routed events are excluded
            and low_priority threads are routed to the local model.
        cloud_all: If True, send everything to cloud (no local routing).

    Returns dict with triage statistics.
    """
    now = int(time.time())

    event_rows = get_triageable_events(
        store, resume=resume, since_ts=since_ts, lens=lens,
    )
    if not event_rows:
        return {
            "triaged": 0, "threads": 0, "thread_full": 0,
            "msg_batch": 0, "msg_compact": 0, "failed": 0,
        }

    sender_cache = build_sender_cache(store)
    thread_groups = group_events_by_thread(event_rows)
    logger.info(
        "Triage: %d events -> %d threads", len(event_rows), len(thread_groups),
    )
    print(
        f"  {len(event_rows)} events in {len(thread_groups)} threads to triage",
        flush=True,
    )

    # Build event metadata cache for score floors (includes spam detection fields)
    event_meta: dict[str, dict] = {}
    for row in event_rows:
        eid = row["id"]
        sender_info = _get_event_sender(store, eid, sender_cache)
        event_meta[eid] = {
            "event_type": row["event_type"],
            "source": row["source"],
            "sender_tier": sender_info["tier"],
            "sender_address": sender_info.get("email", ""),
            "sender_msg_count": sender_info.get("msg_count", 999),
            "body_preview": (row.get("raw_content") or "")[:200],
        }

    # Build event route map from projections (for strategy selection)
    event_routes: dict[str, str] | None = None
    if lens:
        route_rows = store.conn.execute(
            "SELECT event_id, route FROM projections WHERE lens = ?",
            (lens,),
        ).fetchall()
        if route_rows:
            event_routes = {r["event_id"]: r["route"] for r in route_rows}

    stats = {
        "triaged": 0, "threads": 0, "thread_full": 0,
        "msg_batch": 0, "msg_compact": 0, "claims_written": 0,
        "ignore": 0, "lightweight": 0, "deep": 0, "failed": 0,
        "skipped_by_route": 0,
        "incremental": 0, "reactivated": 0,
    }
    start_time = time.time()

    # Separate threads by strategy
    full_threads: list[tuple[str, list[dict]]] = []
    local_threads: list[tuple[str, list[dict]]] = []

    # For incremental triage: detect threads with prior context
    # thread_prior_ctx maps thread_id -> prior context dict (or None)
    thread_prior_ctx: dict[str, dict[str, Any] | None] = {}
    # thread_all_events maps thread_id -> ALL events (new + old) for context
    thread_all_events: dict[str, list[dict]] = {}

    for thread_id, thread_events in thread_groups.items():
        strategy = select_strategy(thread_events, event_routes=event_routes, cloud_all=cloud_all)

        # Check for incremental triage opportunity
        if resume:
            prior = _get_thread_prior_context(thread_id, store)
            if prior is not None:
                thread_prior_ctx[thread_id] = prior

                # Classify: incremental or reactivated
                classification = classify_thread_incremental(
                    thread_id, thread_events,
                    {e["id"] for e in thread_events}, store,
                )

                if classification == "reactivated":
                    # Fetch recent thread events for reprocessing (capped)
                    all_events = _fetch_all_thread_events(
                        thread_id, store,
                    )
                    if all_events:
                        thread_events = all_events
                    stats["reactivated"] += 1
                    logger.info(
                        "Thread %s reactivated (%d events, capped to %d)",
                        thread_id[:30], len(thread_events), MAX_THREAD_FETCH,
                    )
                else:
                    # Fetch recent events for incremental context
                    all_events = _fetch_all_thread_events(
                        thread_id, store,
                    )
                    if all_events:
                        thread_all_events[thread_id] = all_events
                    stats["incremental"] += 1

        if strategy == TriageStrategy.THREAD_FULL:
            full_threads.append((thread_id, thread_events))
        else:
            local_threads.append((thread_id, thread_events))

    # Process THREAD_FULL threads via asyncio
    if full_threads:
        _completed_count = 0
        _failed_count = 0
        _timed_out_count = 0
        _total_threads = len(full_threads)
        _in_flight: set[str] = set()  # thread IDs currently being processed
        _large_threads = {tid for tid, tevts in full_threads if len(tevts) > LARGE_THREAD_THRESHOLD}

        # Log breakdown before starting
        n_large = len(_large_threads)
        n_small = _total_threads - n_large
        total_msgs = sum(len(tevts) for _, tevts in full_threads)
        print(
            f"  Dispatching {_total_threads} cloud threads "
            f"({n_large} chunked, {n_small} single-pass, "
            f"{total_msgs} total msgs, concurrency={max_concurrent})",
            flush=True,
        )

        async def _safe_triage_thread(tid, tevts, sem, prior, all_evts):
            """Wrap thread triage with overall timeout so one hung thread can't block."""
            nonlocal _completed_count, _failed_count, _timed_out_count
            label = tid[:30]
            n = len(tevts)
            _in_flight.add(label)
            try:
                result = await asyncio.wait_for(
                    _async_triage_one_thread(
                        tid, tevts, store, sender_cache, llm_client, now, sem,
                        prior_context=prior,
                        all_thread_events=all_evts,
                        profile_context=profile_context,
                    ),
                    timeout=THREAD_TIMEOUT,
                )
                if result[1] is None:
                    _failed_count += 1
            except asyncio.TimeoutError:
                logger.error("Thread %s (%d msgs) timed out after %ds — skipping", label, n, THREAD_TIMEOUT)
                _timed_out_count += 1
                result = (tid, None)
            _in_flight.discard(label)
            _completed_count += 1
            if _completed_count % 25 == 0 or _completed_count == _total_threads:
                elapsed = time.time() - start_time
                waiting = len(_in_flight)
                status = f"  Triage: {_completed_count}/{_total_threads} done"
                if _timed_out_count:
                    status += f", {_timed_out_count} timed out"
                if _failed_count:
                    status += f", {_failed_count} failed"
                status += f", {waiting} in-flight ({elapsed:.0f}s)"
                print(status, flush=True)
                if waiting > 0 and waiting <= 5:
                    # Show which threads are still running (helps debug hangs)
                    for t in sorted(_in_flight):
                        print(f"    waiting: {t}", flush=True)
            return result

        async def _run_all() -> list[tuple[str, ThreadTriageResult | None]]:
            sem = asyncio.Semaphore(max_concurrent)
            tasks = []
            for tid, tevts in full_threads:
                prior = thread_prior_ctx.get(tid)
                all_evts = thread_all_events.get(tid)
                tasks.append(
                    _safe_triage_thread(tid, tevts, sem, prior, all_evts)
                )
            return await asyncio.gather(*tasks, return_exceptions=False)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context (shouldn't happen in CLI, but safe)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                results = pool.submit(lambda: asyncio.run(_run_all())).result()
        else:
            results = asyncio.run(_run_all())

        # Write results (synchronous — SQLite isn't thread-safe)
        thread_events_map = {tid: tevts for tid, tevts in full_threads}
        for thread_id, result in results:
            thread_events = thread_events_map[thread_id]
            is_incr = thread_id in thread_prior_ctx and thread_id in thread_all_events
            _write_thread_results(
                store, thread_id, thread_events, result, event_meta, stats,
                is_incremental=is_incr,
            )
            stats["threads"] += 1

            # Progress logging
            if stats["threads"] % 50 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    "  Progress: %d/%d threads, %d events, %.0fs",
                    stats["threads"], len(thread_groups), stats["triaged"], elapsed,
                )
                print(
                    f"  {stats['threads']}/{len(thread_groups)} threads "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )

    # Process local model threads sequentially
    for thread_id, thread_events in local_threads:
        model_id, _ = _model_for_thread(thread_id, len(thread_events))
        batch_results = _triage_msg_batch(
            thread_id, thread_events, store, sender_cache,
            llm_client, now, profile_context,
        )

        for eid, result in batch_results:
            meta = event_meta.get(eid, {})
            if result is not None:
                result.score = apply_score_floor(
                    result.score,
                    meta.get("event_type", ""),
                    meta.get("sender_tier", 3),
                    result.commitment_type,
                    sender_address=meta.get("sender_address", ""),
                    sender_msg_count=meta.get("sender_msg_count", 999),
                    body=meta.get("body_preview", ""),
                )
                claim = _per_event_claim_from_triage_result(
                    eid, result, model_id,
                )
                if store.put_claim(claim):
                    stats["claims_written"] += 1

                # Emit topic annotations for this event (specific_topics, falling back to legacy topics)
                topics_for_ann = result.specific_topics or result.topics
                _emit_topic_annotations(store, eid, topics_for_ann, f"triage:{model_id}")

                if result.score < 0.3:
                    stats["ignore"] += 1
                elif result.score < 0.7:
                    stats["lightweight"] += 1
                else:
                    stats["deep"] += 1
                stats["triaged"] += 1
            else:
                store.put_claim(_failed_claim(eid, model_id))
                stats["failed"] += 1

        if len(thread_events) == 1:
            stats["msg_compact"] += 1
        else:
            stats["msg_batch"] += 1
        stats["threads"] += 1

        if stats["threads"] % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"  {stats['threads']}/{len(thread_groups)} threads "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = round(elapsed, 2)

    stats["routed_to_local"] = len(local_threads)
    if lens:
        stats["lens"] = lens

    logger.info(
        "Triage: %d events in %d threads, %.1fs "
        "(thread_full=%d, local=%d, failed=%d)",
        stats["triaged"], stats["threads"], elapsed,
        stats["thread_full"], stats["routed_to_local"],
        stats["failed"],
    )

    return stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Downstream queries (backward compatible)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_triage_result(store: LayeredGraphStore, event_id: str) -> dict | None:
    """Fetch the triage result for a single event."""
    claim = store.get_claim(triage_claim_id(event_id))
    if not claim or claim.claim_type != "triage":
        return None
    try:
        return json.loads(claim.object)
    except (json.JSONDecodeError, TypeError):
        return None


def get_deep_extraction_candidates(
    store: LayeredGraphStore, min_score: float = 0.7,
) -> list[dict]:
    """Fetch events eligible for deep extraction (Stage 6).

    Returns event dicts enriched with their triage result.
    """
    rows = store.conn.execute(
        """SELECT c.subject AS event_id, c.object AS triage_json, c.confidence
           FROM claims c
           WHERE c.claim_type = 'triage'
             AND c.predicate = 'triage_result'
             AND c.superseded_by IS NULL
             AND c.confidence >= ?
           ORDER BY c.confidence DESC""",
        (min_score,),
    ).fetchall()

    candidates = []
    for r in rows:
        try:
            triage = json.loads(r["triage_json"])
        except (json.JSONDecodeError, TypeError):
            continue

        if (triage.get("score", 0) < min_score
                and not triage.get("commitment_type")):
            continue

        event = store.get_event(r["event_id"])
        if event:
            candidates.append({
                "event": event,
                "triage": triage,
                "score": triage.get("score", 0),
            })

    return candidates


def get_sensitive_events(store: LayeredGraphStore) -> list[str]:
    """Get event IDs that have PII or sensitivity flags."""
    rows = store.conn.execute(
        """SELECT c.subject AS event_id, c.object AS triage_json
           FROM claims c
           WHERE c.claim_type = 'triage'
             AND c.predicate = 'triage_result'
             AND c.superseded_by IS NULL"""
    ).fetchall()

    sensitive_ids = []
    for r in rows:
        try:
            triage = json.loads(r["triage_json"])
            if triage.get("pii") or triage.get("sensitivity"):
                sensitive_ids.append(r["event_id"])
        except (json.JSONDecodeError, TypeError):
            continue

    return sensitive_ids
