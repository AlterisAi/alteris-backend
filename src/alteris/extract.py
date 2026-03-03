"""Stage 6: Binary actionable gate for email threads and meetings.

Input:  Events (Layer 1) scored >= 0.7 by triage, plus their triage Claims
Output: extraction_gate Claims (actionable / not_actionable)

Architecture:
  - Consumes get_deep_extraction_candidates() from triage
  - Groups events into threads (via metadata.thread_id)
  - Runs a binary gate per thread: "Is there something actionable here?"
  - Stores gate results as extraction_gate claims
  - Commitment extraction is now handled by beliefs.py (synthesis pass)

The gate is deliberately simple: Flash Lite with few-shot examples,
returning only {actionable: bool, reason: str}. All judgment about
WHO owes WHAT to WHOM is deferred to the synthesis pass which has
the full person graph and cross-thread context.
"""

from __future__ import annotations

import hashlib
import html as html_mod
import json
import logging
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, field_validator

from alteris.prompts.extract import (
    GATE_SYSTEM_PROMPT,
    LOGISTICS_GATE_SYSTEM_PROMPT,
    RELATIONAL_GATE_SYSTEM_PROMPT,
)
from alteris.constants import (
    EXTRACTION_MIN_SCORE,
    EXTRACTION_MIN_THREAD_AVG_SCORE,
    GATE_BATCH_SIZE,
    GATE_PROMPT_VERSION,
    LOGISTICS_GATE_PROMPT_VERSION,
    MAX_THREAD_HISTORY,
    RELATIONAL_GATE_PROMPT_VERSION,
    SECONDS_PER_DAY,
    USER_TIMEZONE,
)
from alteris.llm.base import LLMClient
from alteris.models import (
    Claim,
    Event,
    ExtractionMethod,
    ExtractionProvenance,
    Modality,
)
from alteris.privacy import SensitivityLevel
from alteris.store import LayeredGraphStore
from alteris.triage import get_deep_extraction_candidates

logger = logging.getLogger(__name__)

# PII types that require local processing
SENSITIVE_PII = frozenset({"financial", "credentials", "medical", "legal"})

# Max messages per synthetic thread (iMessage conversations)
MAX_MSGS_PER_THREAD = 50

# Max lookback for thread history enrichment (1 year)
THREAD_HISTORY_LOOKBACK_SECS = 365 * 24 * 3600


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gate result validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class GateResult(BaseModel):
    """Validated result from the binary actionable gate."""
    actionable: bool
    action_type: str | None = None
    reason: str = ""

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        return v[:200].strip() if v else ""


class LogisticsGateResult(BaseModel):
    """Validated result from the logistics gate."""
    logistics: bool
    reason: str = ""

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        return v[:200].strip() if v else ""


class RelationalGateResult(BaseModel):
    """Validated result from the relational gate."""
    relational: bool
    relationship_tier: str | None = None
    reason: str = ""

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        return v[:200].strip() if v else ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gate prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logistics gate prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Relational gate prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thread grouping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ThreadBundle:
    """A group of events in the same thread, ready for processing."""
    thread_id: str
    events: list[Event]
    triage_data: list[dict] = field(default_factory=list)
    sensitive: bool = False


def _is_sensitive(triage: dict) -> bool:
    """Check if triage flags indicate sensitive content."""
    pii = triage.get("pii", [])
    if isinstance(pii, list) and set(pii) & SENSITIVE_PII:
        return True
    return False


def group_into_threads(
    candidates: list[dict],
    store: LayeredGraphStore,
) -> tuple[list[ThreadBundle], list[ThreadBundle]]:
    """Group deep-extraction candidates into threads.

    For threads with recent triaged events, also pulls older events from the
    same thread (up to 1 year back) so the LLM has full conversational context.
    Only threads that have at least one recently triaged event get enriched.

    Args:
        candidates: From get_deep_extraction_candidates()
        store: LayeredGraphStore for lookups

    Returns:
        (threads, standalones) -- ThreadBundle lists
    """
    thread_map: dict[str, list[tuple[Event, dict]]] = defaultdict(list)
    standalone_list: list[tuple[Event, dict]] = []

    for c in candidates:
        event: Event = c["event"]
        triage: dict = c["triage"]
        thread_id = event.metadata.get("thread_id", "")

        if thread_id:
            thread_map[thread_id].append((event, triage))
        elif event.source == "imessage":
            contact = _extract_imessage_contact(event)
            if contact:
                synth_id = f"imessage:{contact}"
                thread_map[synth_id].append((event, triage))
            else:
                standalone_list.append((event, triage))
        else:
            standalone_list.append((event, triage))

    # Enrich threads with historical events for full context.
    # For each thread_id, fetch older events from the store (up to 1 year)
    # so the LLM can see the full conversation, not just recent messages.
    lookback_ts = int(time.time()) - THREAD_HISTORY_LOOKBACK_SECS
    for tid in list(thread_map.keys()):
        triaged_ids = {p[0].id for p in thread_map[tid]}
        # Find older events in the same thread via metadata.thread_id
        # or synthetic thread_id (iMessage contact-based)
        if tid.startswith("imessage:"):
            # Synthetic thread: match by source + contact identifier
            contact = tid.removeprefix("imessage:")
            hist_rows = store.conn.execute(
                """SELECT id, source, source_id, event_type, timestamp,
                          participants, raw_content, metadata, content_hash
                   FROM events
                   WHERE source = 'imessage'
                     AND timestamp >= ?
                     AND id NOT IN ({})
                   ORDER BY timestamp ASC""".format(
                    ",".join("?" * len(triaged_ids))
                ),
                [lookback_ts, *triaged_ids],
            ).fetchall()
            # Filter to events matching this contact
            for r in hist_rows:
                evt = Event(
                    id=r["id"], source=r["source"], source_id=r["source_id"],
                    event_type=r["event_type"], timestamp=r["timestamp"],
                    participants=tuple(json.loads(r["participants"] or "[]")),
                    raw_content=r["raw_content"] or "",
                    metadata=json.loads(r["metadata"] or "{}"),
                    content_hash=r["content_hash"] or "",
                )
                if _extract_imessage_contact(evt) == contact:
                    thread_map[tid].append((evt, {}))
        else:
            # Real thread_id from metadata
            hist_rows = store.conn.execute(
                """SELECT id, source, source_id, event_type, timestamp,
                          participants, raw_content, metadata, content_hash
                   FROM events
                   WHERE json_extract(metadata, '$.thread_id') = ?
                     AND timestamp >= ?
                     AND id NOT IN ({})
                   ORDER BY timestamp ASC""".format(
                    ",".join("?" * len(triaged_ids))
                ),
                [tid, lookback_ts, *triaged_ids],
            ).fetchall()
            for r in hist_rows:
                thread_map[tid].append((
                    Event(
                        id=r["id"], source=r["source"], source_id=r["source_id"],
                        event_type=r["event_type"], timestamp=r["timestamp"],
                        participants=tuple(json.loads(r["participants"] or "[]")),
                        raw_content=r["raw_content"] or "",
                        metadata=json.loads(r["metadata"] or "{}"),
                        content_hash=r["content_hash"] or "",
                    ),
                    {},  # no triage data for historical events
                ))

    # Sort each thread chronologically, cap at MAX_THREAD_HISTORY (500).
    # The gate sees only the tail (MAX_MSGS_PER_THREAD=50) via _format_thread_for_llm.
    # The synthesis pass uses sliding-window processing for threads > 50 msgs.
    threads: list[ThreadBundle] = []
    for tid, pairs in thread_map.items():
        pairs.sort(key=lambda p: p[0].timestamp)
        if len(pairs) > MAX_THREAD_HISTORY:
            pairs = pairs[-MAX_THREAD_HISTORY:]

        events = [p[0] for p in pairs]
        triages = [p[1] for p in pairs]
        sensitive = any(_is_sensitive(t) for t in triages if t)
        threads.append(ThreadBundle(tid, events, triages, sensitive))

    standalones: list[ThreadBundle] = []
    for event, triage in standalone_list:
        standalones.append(ThreadBundle(
            event.id, [event], [triage], _is_sensitive(triage),
        ))

    return threads, standalones


def _extract_imessage_contact(event: Event) -> str | None:
    """Extract the other party from an iMessage event."""
    for p in event.participants:
        if p and p.lower() != "me":
            return p.strip()
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event formatting for LLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def format_event_as_markdown(
    event: Event,
    index: int | None = None,
    persons: list[dict] | None = None,
    email_to_name: dict[str, str] | None = None,
) -> str:
    """Format a single Event as markdown for LLM consumption."""
    lines: list[str] = []
    meta = event.metadata or {}

    if index is not None:
        lines.append(f"## Message {index}")
        lines.append("")

    if event.event_type == "email":
        sender = _get_sender_name(event, persons, email_to_name=email_to_name)
        lines.append(f"**From:** {sender}")
        recipients = _get_recipient_names(event, persons, email_to_name=email_to_name)
        if recipients:
            lines.append(f"**To:** {', '.join(recipients[:5])}")
        cc = meta.get("cc", "")
        if cc and email_to_name:
            if isinstance(cc, list):
                cc_parts = [c.strip() for c in cc if isinstance(c, str) and c.strip()]
            else:
                cc_parts = [c.strip() for c in cc.split(",") if c.strip()]
            cleaned_cc = []
            for c in cc_parts:
                email = _extract_email(c)
                if email and email in email_to_name:
                    cleaned_cc.append(email_to_name[email])
                else:
                    cleaned_cc.append(_clean_participant_name(c))
            cc = ", ".join(cleaned_cc)
        if cc:
            lines.append(f"**Cc:** {cc}")
        lines.append(f"**Subject:** {meta.get('subject', '(no subject)')}")

    elif event.event_type == "meeting":
        lines.append(f"**Meeting:** {meta.get('subject', '(untitled)')}")
        if persons:
            names = [
                p.get("name") or p.get("person_id", "") for p in persons[:10]
            ]
            lines.append(f"**Attendees:** {', '.join(names)}")

    elif event.event_type == "calendar_event":
        lines.append(
            f"**Calendar event:** {meta.get('subject', '(untitled)')}"
        )
        if meta.get("location"):
            lines.append(f"**Location:** {meta['location']}")

    elif event.event_type == "message":
        is_from_me = meta.get("is_from_me", False)
        if is_from_me:
            recipient = None
            for p in event.participants:
                if p and p.lower() != "me":
                    raw = p
                    email = _extract_email(raw)
                    if email and email_to_name and email in email_to_name:
                        recipient = email_to_name[email]
                    else:
                        recipient = _clean_participant_name(raw)
                    break
            if recipient:
                lines.append(
                    f"**Message from:** USER (you) -> {recipient}"
                )
            else:
                lines.append("**Message from:** USER (you, outbound)")
            lines.append("**Direction:** outbound (user sent this)")
        else:
            sender = _get_sender_name(event, persons, email_to_name=email_to_name)
            lines.append(f"**Message from:** {sender}")
            lines.append("**Direction:** inbound (sent to user)")

    else:
        lines.append(
            f"**{event.source}/{event.event_type}:** "
            f"{meta.get('subject', '(untitled)')}"
        )

    if event.timestamp:
        try:
            from zoneinfo import ZoneInfo
            local_tz = ZoneInfo(USER_TIMEZONE)
            dt = datetime.fromtimestamp(event.timestamp, tz=local_tz)
            lines.append(f"**Date:** {dt.strftime('%Y-%m-%d %H:%M %Z')}")
        except (OSError, ValueError, ImportError):
            pass

    lines.append("")
    lines.append("### Body")
    body = event.raw_content or ""

    if body:
        body = html_mod.unescape(body)

    if event.event_type == "calendar_event" and body:
        body = re.sub(
            r"^\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\s*$", "",
            body, flags=re.MULTILINE,
        ).strip()

    lines.append(body if body else "(no content)")

    return "\n".join(lines)


def _extract_email(raw: str) -> str:
    """Extract email address from 'Name <email>' format or return raw if already email."""
    if "<" in raw and ">" in raw:
        start = raw.index("<") + 1
        end = raw.index(">")
        return raw[start:end].strip().lower()
    if "@" in raw:
        return raw.strip().lower()
    return ""


def _clean_participant_name(raw: str) -> str:
    """Extract display name from 'Name <email>' format."""
    if "<" in raw and ">" in raw:
        name = raw.split("<")[0].strip()
        if name:
            return name
    return raw


def _get_sender_name(
    event: Event, persons: list[dict] | None = None,
    email_to_name: dict[str, str] | None = None,
) -> str:
    """Get sender name from persons or participants."""
    if persons:
        for p in persons:
            if p.get("role") == "sender":
                return p.get("name") or p.get("person_id", "unknown")
    if event.participants:
        raw = event.participants[0]
        email = _extract_email(raw)
        if email and email_to_name and email in email_to_name:
            return email_to_name[email]
        return _clean_participant_name(raw)
    return "unknown"


def _get_recipient_names(
    event: Event, persons: list[dict] | None = None,
    email_to_name: dict[str, str] | None = None,
) -> list[str]:
    """Get recipient names."""
    if persons:
        recips = [
            p.get("name") or p.get("person_id")
            for p in persons
            if p.get("role") == "recipient"
        ]
        if recips:
            return recips
    if len(event.participants) > 1:
        result = []
        for raw in event.participants[1:]:
            email = _extract_email(raw)
            if email and email_to_name and email in email_to_name:
                result.append(email_to_name[email])
            else:
                result.append(_clean_participant_name(raw))
        return result
    return []


def _format_thread_for_llm(
    bundle: ThreadBundle,
    user_email: str = "",
    persons_cache: dict[str, list[dict]] | None = None,
    max_messages: int | None = None,
    email_to_name: dict[str, str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Format a full thread as markdown for LLM processing.

    Args:
        max_messages: If set, only format the last N messages (tail).
            Used by the gate to limit context for quick binary decisions.
            The synthesis pass uses None (all messages) and handles
            windowing separately.
        email_to_name: Optional mapping of lowercase email -> canonical name
            for cleaning raw email addresses in participant fields.

    Returns (formatted_text, msg_id_map) where msg_id_map maps
    "msg_0", "msg_1", ... to actual event IDs for source alignment.
    """
    events = bundle.events
    if max_messages and len(events) > max_messages:
        events = events[-max_messages:]

    msg_id_map: dict[str, str] = {}
    parts = [
        f"**User email:** {user_email}",
        "",
    ]

    if max_messages and len(bundle.events) > max_messages:
        parts.append(
            f"*[Thread has {len(bundle.events)} total messages. "
            f"Showing last {len(events)} for analysis.]*"
        )
        parts.append("")

    parts.append("---")

    for i, event in enumerate(events):
        label = f"msg_{i}"
        msg_id_map[label] = event.id
        persons = (persons_cache or {}).get(event.id, [])
        parts.append("")
        parts.append(f"**[{label}]**")
        parts.append(
            format_event_as_markdown(
                event, index=i + 1, persons=persons,
                email_to_name=email_to_name,
            )
        )
        if i < len(events) - 1:
            parts.append("")
            parts.append("---")

    return "\n".join(parts), msg_id_map


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gate response schemas (Gemini structured output)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_actionable_gate_schema():
    """Build Gemini structured output schema for the actionable gate."""
    from google.genai import types

    return types.Schema(
        type="OBJECT",
        properties={
            "actionable": types.Schema(
                type="BOOLEAN",
                description="Whether the thread requires user action.",
            ),
            "action_type": types.Schema(
                type="STRING",
                enum=[
                    "user_owes_action",
                    "waiting_on_other",
                    "scheduling_conflict_or_setup",
                    "financial_obligation",
                    "passive_tracking_or_reminder",
                ],
                description="Type of action required, if actionable.",
            ),
            "reason": types.Schema(
                type="STRING",
                description="5 words max explaining the classification.",
            ),
        },
        required=["actionable", "reason"],
    )


def _build_logistics_gate_schema():
    """Build Gemini structured output schema for the logistics gate."""
    from google.genai import types

    return types.Schema(
        type="OBJECT",
        properties={
            "logistics": types.Schema(
                type="BOOLEAN",
                description="Whether the thread contains logistics information.",
            ),
            "reason": types.Schema(
                type="STRING",
                description="5 words max explaining the classification.",
            ),
        },
        required=["logistics", "reason"],
    )


def _build_relational_gate_schema():
    """Build Gemini structured output schema for the relational gate."""
    from google.genai import types

    return types.Schema(
        type="OBJECT",
        properties={
            "relational": types.Schema(
                type="BOOLEAN",
                description="Whether the thread contains relational information.",
            ),
            "relationship_tier": types.Schema(
                type="STRING",
                enum=[
                    "core_kinship",
                    "extended_kinship",
                    "intimate_friendship",
                    "vocational_core_team",
                    "vocational_network",
                    "commercial_vendor",
                    "unknown_or_first_contact",
                ],
                description="FOAF-inspired relationship tier, if relational.",
            ),
            "reason": types.Schema(
                type="STRING",
                description="5 words max explaining the classification.",
            ),
        },
        required=["relational", "reason"],
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gate claim construction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _gate_claim_id(thread_id: str) -> str:
    """Deterministic claim ID for a gate result."""
    key = f"extraction_gate:{thread_id}:{GATE_PROMPT_VERSION}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _build_gate_claim(
    result: GateResult,
    thread_id: str,
    event_ids: list[str],
    model_id: str,
) -> Claim:
    """Convert a GateResult into a Claim."""
    claim_id = _gate_claim_id(thread_id)

    predicate = "actionable" if result.actionable else "not_actionable"

    obj = {
        "actionable": result.actionable,
        "action_type": result.action_type,
        "reason": result.reason,
    }

    return Claim(
        id=claim_id,
        event_ids=event_ids,
        claim_type="extraction_gate",
        subject=thread_id,
        predicate=predicate,
        object=json.dumps(obj),
        confidence=1.0 if result.actionable else 0.1,
        modality=Modality.OBSERVED,
        provenance=ExtractionProvenance(
            model_id=model_id,
            prompt_version=GATE_PROMPT_VERSION,
            extraction_method=ExtractionMethod.CLOUD_MODEL,
        ),
        sensitivity=SensitivityLevel.SENSITIVE,
    )


def _parse_gate_result(raw: dict | None) -> GateResult:
    """Parse LLM output into a GateResult. Defaults to not_actionable on failure."""
    if not raw:
        return GateResult(actionable=False, reason="no LLM response")

    try:
        return GateResult(**raw)
    except Exception:
        actionable = raw.get("actionable", False)
        if isinstance(actionable, str):
            actionable = actionable.lower() in ("true", "yes", "1")
        reason = str(raw.get("reason", ""))
        action_type = raw.get("action_type")
        if action_type is not None:
            action_type = str(action_type)
        return GateResult(
            actionable=bool(actionable), action_type=action_type, reason=reason,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logistics gate claim construction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _logistics_gate_claim_id(thread_id: str) -> str:
    """Deterministic claim ID for a logistics gate result."""
    key = f"logistics_gate:{thread_id}:{LOGISTICS_GATE_PROMPT_VERSION}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _build_logistics_gate_claim(
    result: LogisticsGateResult,
    thread_id: str,
    event_ids: list[str],
    model_id: str,
) -> Claim:
    """Convert a LogisticsGateResult into a Claim."""
    claim_id = _logistics_gate_claim_id(thread_id)

    predicate = "logistics" if result.logistics else "not_logistics"

    obj = {
        "logistics": result.logistics,
        "reason": result.reason,
    }

    return Claim(
        id=claim_id,
        event_ids=event_ids,
        claim_type="logistics_gate",
        subject=thread_id,
        predicate=predicate,
        object=json.dumps(obj),
        confidence=1.0 if result.logistics else 0.1,
        modality=Modality.OBSERVED,
        provenance=ExtractionProvenance(
            model_id=model_id,
            prompt_version=LOGISTICS_GATE_PROMPT_VERSION,
            extraction_method=ExtractionMethod.CLOUD_MODEL,
        ),
        sensitivity=SensitivityLevel.SENSITIVE,
    )


def _parse_logistics_gate_result(raw: dict | None) -> LogisticsGateResult:
    """Parse LLM output into a LogisticsGateResult."""
    if not raw:
        return LogisticsGateResult(logistics=False, reason="no LLM response")

    try:
        return LogisticsGateResult(**raw)
    except Exception:
        logistics = raw.get("logistics", False)
        if isinstance(logistics, str):
            logistics = logistics.lower() in ("true", "yes", "1")
        reason = str(raw.get("reason", ""))
        return LogisticsGateResult(logistics=bool(logistics), reason=reason)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Relational gate claim construction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _relational_gate_claim_id(thread_id: str) -> str:
    """Deterministic claim ID for a relational gate result."""
    key = f"relational_gate:{thread_id}:{RELATIONAL_GATE_PROMPT_VERSION}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _build_relational_gate_claim(
    result: RelationalGateResult,
    thread_id: str,
    event_ids: list[str],
    model_id: str,
) -> Claim:
    """Convert a RelationalGateResult into a Claim."""
    claim_id = _relational_gate_claim_id(thread_id)

    predicate = "relational" if result.relational else "not_relational"

    obj = {
        "relational": result.relational,
        "relationship_tier": result.relationship_tier,
        "reason": result.reason,
    }

    return Claim(
        id=claim_id,
        event_ids=event_ids,
        claim_type="relational_gate",
        subject=thread_id,
        predicate=predicate,
        object=json.dumps(obj),
        confidence=1.0 if result.relational else 0.1,
        modality=Modality.OBSERVED,
        provenance=ExtractionProvenance(
            model_id=model_id,
            prompt_version=RELATIONAL_GATE_PROMPT_VERSION,
            extraction_method=ExtractionMethod.CLOUD_MODEL,
        ),
        sensitivity=SensitivityLevel.SENSITIVE,
    )


def _parse_relational_gate_result(raw: dict | None) -> RelationalGateResult:
    """Parse LLM output into a RelationalGateResult."""
    if not raw:
        return RelationalGateResult(relational=False, reason="no LLM response")

    try:
        return RelationalGateResult(**raw)
    except Exception:
        relational = raw.get("relational", False)
        if isinstance(relational, str):
            relational = relational.lower() in ("true", "yes", "1")
        reason = str(raw.get("reason", ""))
        relationship_tier = raw.get("relationship_tier")
        if relationship_tier is not None:
            relationship_tier = str(relationship_tier)
        return RelationalGateResult(
            relational=bool(relational), relationship_tier=relationship_tier,
            reason=reason,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Binary gate execution (all 3 gates)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_gate(
    bundle: ThreadBundle,
    llm_client: LLMClient,
    model: str = "",
    user_email: str = "",
    persons_cache: dict[str, list[dict]] | None = None,
    person_profiles: dict[str, dict[str, Any]] | None = None,
    data_span_days: int | None = None,
    profile_context: str = "",
    email_to_name: dict[str, str] | None = None,
) -> GateResult:
    """Run the binary actionable gate on a single thread.

    Single LLM call per thread — returns GateResult.
    Uses tail-only view (MAX_MSGS_PER_THREAD) for long threads.
    """
    thread_text, _ = _format_thread_for_llm(
        bundle, user_email, persons_cache=persons_cache,
        max_messages=MAX_MSGS_PER_THREAD,
        email_to_name=email_to_name,
    )
    person_ctx = _build_gate_person_context(
        bundle, persons_cache, person_profiles, data_span_days,
    )
    prefix_parts = []
    if profile_context:
        prefix_parts.append(profile_context)
    if person_ctx:
        prefix_parts.append(person_ctx)
    if prefix_parts:
        thread_text = "\n\n".join(prefix_parts) + "\n\n" + thread_text

    try:
        schema = _build_actionable_gate_schema()
    except ImportError:
        schema = None

    raw_str = llm_client.generate(
        prompt=thread_text,
        system=GATE_SYSTEM_PROMPT,
        model=model,
        temperature=0.1,
        cache_system=True,
        response_schema=schema,
        format_json=schema is None,
    )

    try:
        raw = json.loads(raw_str) if raw_str else None
    except (json.JSONDecodeError, TypeError):
        raw = None

    return _parse_gate_result(raw)


def run_logistics_gate(
    bundle: ThreadBundle,
    llm_client: LLMClient,
    model: str = "",
    user_email: str = "",
    persons_cache: dict[str, list[dict]] | None = None,
    person_profiles: dict[str, dict[str, Any]] | None = None,
    data_span_days: int | None = None,
    profile_context: str = "",
    email_to_name: dict[str, str] | None = None,
) -> LogisticsGateResult:
    """Run the logistics gate on a single thread.
    Uses tail-only view for long threads."""
    thread_text, _ = _format_thread_for_llm(
        bundle, user_email, persons_cache=persons_cache,
        max_messages=MAX_MSGS_PER_THREAD,
        email_to_name=email_to_name,
    )
    person_ctx = _build_gate_person_context(
        bundle, persons_cache, person_profiles, data_span_days,
    )
    prefix_parts = []
    if profile_context:
        prefix_parts.append(profile_context)
    if person_ctx:
        prefix_parts.append(person_ctx)
    if prefix_parts:
        thread_text = "\n\n".join(prefix_parts) + "\n\n" + thread_text

    try:
        schema = _build_logistics_gate_schema()
    except ImportError:
        schema = None

    raw_str = llm_client.generate(
        prompt=thread_text,
        system=LOGISTICS_GATE_SYSTEM_PROMPT,
        model=model,
        temperature=0.1,
        cache_system=True,
        response_schema=schema,
        format_json=schema is None,
    )

    try:
        raw = json.loads(raw_str) if raw_str else None
    except (json.JSONDecodeError, TypeError):
        raw = None

    return _parse_logistics_gate_result(raw)


def run_relational_gate(
    bundle: ThreadBundle,
    llm_client: LLMClient,
    model: str = "",
    user_email: str = "",
    persons_cache: dict[str, list[dict]] | None = None,
    person_profiles: dict[str, dict[str, Any]] | None = None,
    data_span_days: int | None = None,
    profile_context: str = "",
    email_to_name: dict[str, str] | None = None,
) -> RelationalGateResult:
    """Run the relational gate on a single thread.
    Uses tail-only view for long threads."""
    thread_text, _ = _format_thread_for_llm(
        bundle, user_email, persons_cache=persons_cache,
        max_messages=MAX_MSGS_PER_THREAD,
        email_to_name=email_to_name,
    )
    person_ctx = _build_gate_person_context(
        bundle, persons_cache, person_profiles, data_span_days,
    )
    prefix_parts = []
    if profile_context:
        prefix_parts.append(profile_context)
    if person_ctx:
        prefix_parts.append(person_ctx)
    if prefix_parts:
        thread_text = "\n\n".join(prefix_parts) + "\n\n" + thread_text

    try:
        schema = _build_relational_gate_schema()
    except ImportError:
        schema = None

    raw_str = llm_client.generate(
        prompt=thread_text,
        system=RELATIONAL_GATE_SYSTEM_PROMPT,
        model=model,
        temperature=0.1,
        cache_system=True,
        response_schema=schema,
        format_json=schema is None,
    )

    try:
        raw = json.loads(raw_str) if raw_str else None
    except (json.JSONDecodeError, TypeError):
        raw = None

    return _parse_relational_gate_result(raw)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Persons cache builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_persons_cache(
    bundles: list[ThreadBundle],
    store: LayeredGraphStore,
) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Pre-load event_persons for all events in all bundles.

    Returns (cache, email_to_name) where cache maps event_id -> list of person
    dicts, and email_to_name maps lowercase email -> canonical_name for fallback
    resolution of raw email addresses in LLM prompts.
    """
    cache: dict[str, list[dict]] = {}

    all_event_ids = set()
    for b in bundles:
        for e in b.events:
            all_event_ids.add(e.id)

    if not all_event_ids:
        return cache, {}

    placeholders = ",".join("?" * len(all_event_ids))
    rows = store.conn.execute(
        f"""SELECT ep.event_id, p.person_id, p.canonical_name,
                   p.is_user, ep.role
            FROM event_persons ep
            JOIN persons p ON ep.person_id = p.person_id
            WHERE ep.event_id IN ({placeholders})
            ORDER BY ep.role""",
        list(all_event_ids),
    ).fetchall()

    for r in rows:
        eid = r["event_id"]
        if eid not in cache:
            cache[eid] = []
        cache[eid].append({
            "person_id": r["person_id"],
            "name": r["canonical_name"],
            "is_user": bool(r["is_user"]),
            "role": r["role"],
        })

    # Build email->canonical_name for fallback resolution
    email_to_name: dict[str, str] = {}
    try:
        email_rows = store.conn.execute("""
            SELECT pi.identifier, p.canonical_name
            FROM person_identifiers pi
            JOIN persons p ON pi.person_id = p.person_id
            WHERE pi.identifier_type = 'email' AND p.canonical_name IS NOT NULL
        """).fetchall()
        email_to_name = {r["identifier"].lower(): r["canonical_name"] for r in email_rows}
    except Exception:
        pass

    return cache, email_to_name


def build_person_profiles(
    store: LayeredGraphStore,
) -> dict[str, dict[str, Any]]:
    """Read person profiles from the persistent person_profiles table.

    Returns {person_id: {name, msg_count, tier, user_initiated_ratio,
    channels, days_since_last, relationship_span_days}}.

    Falls back to reconstructing from Stage 1 claims if the table is empty
    (backward compat during migration).
    """
    rows = store.conn.execute(
        "SELECT * FROM person_profiles WHERE message_count > 0"
    ).fetchall()

    if rows:
        profiles: dict[str, dict[str, Any]] = {}
        for r in rows:
            profiles[r["person_id"]] = {
                "name": r["canonical_name"],
                "msg_count": r["message_count"],
                "tier": r["tier"],
                "user_initiated_ratio": r["user_initiated_ratio"],
                "channels": json.loads(r["channels"] or "[]"),
                "days_since_last": r["days_since_last"],
                "relationship_span_days": r["relationship_span_days"],
            }
        return profiles

    # Fallback: reconstruct from claims (pre-migration path)
    return _build_person_profiles_from_claims(store)


def _build_person_profiles_from_claims(
    store: LayeredGraphStore,
) -> dict[str, dict[str, Any]]:
    """Reconstruct person profiles from Stage 1 claims (legacy fallback)."""
    profiles: dict[str, dict[str, Any]] = {}

    def _extract_person_id(predicate: str) -> str | None:
        if ":" not in predicate:
            return None
        parts = predicate.split(":", 1)
        return parts[1] if len(parts) > 1 else None

    freq_rows = store.conn.execute(
        """SELECT predicate, object FROM claims
           WHERE claim_type = 'communication_frequency'
             AND (superseded_by IS NULL OR superseded_by = '')"""
    ).fetchall()
    for r in freq_rows:
        pid = _extract_person_id(r["predicate"])
        if not pid:
            continue
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue
        msg_count = obj.get("event_count", 0)
        if msg_count == 0:
            continue

        if msg_count >= 50:
            tier = 1
        elif msg_count >= 20:
            tier = 2
        elif msg_count >= 5:
            tier = 3
        else:
            tier = 4

        profiles[pid] = {
            "name": obj.get("person_name", pid),
            "msg_count": msg_count,
            "tier": tier,
        }

    dir_rows = store.conn.execute(
        """SELECT predicate, object FROM claims
           WHERE claim_type = 'directionality'
             AND (superseded_by IS NULL OR superseded_by = '')"""
    ).fetchall()
    for r in dir_rows:
        pid = _extract_person_id(r["predicate"])
        if not pid or pid not in profiles:
            continue
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue
        profiles[pid]["user_initiated_ratio"] = obj.get("user_initiated_ratio", 0.5)

    rec_rows = store.conn.execute(
        """SELECT predicate, object FROM claims
           WHERE claim_type = 'recency'
             AND (superseded_by IS NULL OR superseded_by = '')"""
    ).fetchall()
    for r in rec_rows:
        pid = _extract_person_id(r["predicate"])
        if not pid or pid not in profiles:
            continue
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue
        profiles[pid]["days_since_last"] = obj.get("days_since_last")
        profiles[pid]["relationship_span_days"] = obj.get("relationship_span_days")

    chan_rows = store.conn.execute(
        """SELECT predicate, object FROM claims
           WHERE claim_type = 'communication_channel'
             AND (superseded_by IS NULL OR superseded_by = '')"""
    ).fetchall()
    for r in chan_rows:
        pid = _extract_person_id(r["predicate"])
        if not pid or pid not in profiles:
            continue
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            continue
        profiles[pid]["channels"] = obj.get("channels", [])

    return profiles


def _build_gate_person_context(
    bundle: ThreadBundle,
    persons_cache: dict[str, list[dict]] | None,
    person_profiles: dict[str, dict[str, Any]] | None,
    data_span_days: int | None = None,
) -> str:
    """Build a PERSON CONTEXT block for the gate prompt.

    Injects Stage 1 deterministic signals: message count, tier,
    directionality, recency, channels. This gives Flash Lite the
    information it needs to correctly classify relationship tiers
    instead of guessing from raw thread text alone.
    """
    if not person_profiles:
        return ""

    seen: set[str] = set()
    lines: list[str] = []

    for event in bundle.events:
        for p in (persons_cache or {}).get(event.id, []):
            pid = p.get("person_id", "")
            if pid in seen or p.get("is_user"):
                continue
            seen.add(pid)

            profile = person_profiles.get(pid)
            if not profile:
                continue

            parts = [f"{profile['name']}: tier-{profile['tier']}, {profile['msg_count']} messages"]
            ratio = profile.get("user_initiated_ratio")
            if ratio is not None:
                parts.append(f"user initiates {ratio:.0%}")
            channels = profile.get("channels", [])
            if channels:
                parts.append(f"via {', '.join(channels[:3])}")
            days = profile.get("days_since_last")
            if days is not None:
                parts.append(f"last contact {days}d ago")
            span = profile.get("relationship_span_days")
            if span is not None:
                span_label = f"known {span}d"
                if data_span_days:
                    span_label += f" of {data_span_days}d data"
                parts.append(span_label)

            lines.append("  " + ", ".join(parts))

    if not lines:
        return ""
    return "KNOWN CONTACTS:\n" + "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Downstream: query actionable threads for synthesis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_actionable_threads(store: LayeredGraphStore) -> list[dict]:
    """Query extraction_gate claims to get actionable thread IDs.

    Returns list of {thread_id, event_ids, reason} for threads
    marked actionable by the gate.
    """
    rows = store.conn.execute(
        """SELECT c.id, c.subject AS thread_id, c.object
           FROM claims c
           WHERE c.claim_type = 'extraction_gate'
             AND c.predicate = 'actionable'
             AND c.superseded_by IS NULL
           ORDER BY c.created_at DESC"""
    ).fetchall()

    results: list[dict] = []
    for r in rows:
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            obj = {}

        # Fetch event_ids from the join table
        event_rows = store.conn.execute(
            "SELECT event_id FROM claim_events WHERE claim_id = ?",
            (r["id"],),
        ).fetchall()
        event_ids = [er["event_id"] for er in event_rows]

        results.append({
            "thread_id": r["thread_id"],
            "event_ids": event_ids,
            "reason": obj.get("reason", ""),
            "action_type": obj.get("action_type"),
        })

    return results


def get_logistics_threads(store: LayeredGraphStore) -> list[dict]:
    """Query logistics_gate claims to get thread IDs with logistics data.

    Returns list of {thread_id, event_ids, reason} for threads
    marked as containing logistics information.
    """
    rows = store.conn.execute(
        """SELECT c.id, c.subject AS thread_id, c.object
           FROM claims c
           WHERE c.claim_type = 'logistics_gate'
             AND c.predicate = 'logistics'
             AND c.superseded_by IS NULL
           ORDER BY c.created_at DESC"""
    ).fetchall()

    results: list[dict] = []
    for r in rows:
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            obj = {}

        event_rows = store.conn.execute(
            "SELECT event_id FROM claim_events WHERE claim_id = ?",
            (r["id"],),
        ).fetchall()
        event_ids = [er["event_id"] for er in event_rows]

        results.append({
            "thread_id": r["thread_id"],
            "event_ids": event_ids,
            "reason": obj.get("reason", ""),
        })

    return results


def get_relational_threads(store: LayeredGraphStore) -> list[dict]:
    """Query relational_gate claims to get thread IDs with relational context.

    Returns list of {thread_id, event_ids, reason} for threads
    marked as containing relational information.
    """
    rows = store.conn.execute(
        """SELECT c.id, c.subject AS thread_id, c.object
           FROM claims c
           WHERE c.claim_type = 'relational_gate'
             AND c.predicate = 'relational'
             AND c.superseded_by IS NULL
           ORDER BY c.created_at DESC"""
    ).fetchall()

    results: list[dict] = []
    for r in rows:
        try:
            obj = json.loads(r["object"])
        except (json.JSONDecodeError, TypeError):
            obj = {}

        event_rows = store.conn.execute(
            "SELECT event_id FROM claim_events WHERE claim_id = ?",
            (r["id"],),
        ).fetchall()
        event_ids = [er["event_id"] for er in event_rows]

        results.append({
            "thread_id": r["thread_id"],
            "event_ids": event_ids,
            "reason": obj.get("reason", ""),
            "relationship_tier": obj.get("relationship_tier"),
        })

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Resume support
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _is_thread_attempted(store: LayeredGraphStore, thread_id: str) -> bool:
    """Check if any gate was run for this thread."""
    row = store.conn.execute(
        """SELECT 1 FROM claims
           WHERE claim_type IN (
               'extraction_gate', 'logistics_gate',
               'relational_gate', 'extraction_run'
           )
             AND subject = ?
             AND superseded_by IS NULL
           LIMIT 1""",
        (thread_id,),
    ).fetchone()
    return row is not None


def _record_extraction_run(
    store: LayeredGraphStore,
    bundle: ThreadBundle,
    model_id: str,
    gate_result: bool,
    elapsed_ms: int,
    status: str = "success",
    error_msg: str | None = None,
):
    """Record that gate was attempted for a thread (for resume)."""
    event_ids = [e.id for e in bundle.events]
    run_id = hashlib.sha256(
        f"extraction_run:{bundle.thread_id}:{GATE_PROMPT_VERSION}".encode()
    ).hexdigest()[:16]

    obj = {
        "prompt_version": GATE_PROMPT_VERSION,
        "model_id": model_id,
        "gate_result": gate_result,
        "elapsed_ms": elapsed_ms,
        "status": status,
        "error_msg": error_msg,
    }

    claim = Claim(
        id=run_id,
        event_ids=event_ids,
        claim_type="extraction_run",
        subject=bundle.thread_id,
        predicate="gate_attempted",
        object=json.dumps(obj),
        confidence=1.0,
        modality=Modality.OBSERVED,
        provenance=ExtractionProvenance(
            model_id=model_id,
            prompt_version=GATE_PROMPT_VERSION,
            extraction_method=ExtractionMethod.DETERMINISTIC,
        ),
    )
    store.put_claim(claim)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main gate pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_extraction(
    store: LayeredGraphStore,
    local_llm: LLMClient,
    cloud_llm: LLMClient | None = None,
    local_model: str = "",
    cloud_model: str = "",
    user_email: str = "",
    min_score: float = EXTRACTION_MIN_SCORE,
    limit: int | None = None,
    force: bool = False,
    max_concurrent: int = 10,
    profile_context: str = "",
) -> dict[str, Any]:
    """Run Stage 6 binary gate on triaged events.

    Args:
        store: LayeredGraphStore (source + output)
        local_llm: LLM client for local/sensitive processing
        cloud_llm: LLM client for cloud processing (optional)
        local_model: Model name for local gate
        cloud_model: Model name for cloud gate
        user_email: User's email for direction detection
        min_score: Minimum triage score for gate candidates
        limit: Max threads to process (None = all)
        force: Re-run gate even if already processed
        max_concurrent: Max parallel cloud LLM calls (default 10).
        profile_context: Optional user profile context prepended to prompts.

    Returns:
        Stats dict with processing results.
    """
    start_time = time.time()

    # Step 1: Get candidates from triage
    candidates = get_deep_extraction_candidates(store, min_score=min_score)

    if not candidates:
        return {
            "threads_processed": 0, "actionable": 0,
            "not_actionable": 0, "errors": 0, "skipped": 0,
        }

    # Step 2: Group into threads
    threads, standalones = group_into_threads(candidates, store)

    # Step 3: Build persons cache + person profiles from Stage 1 claims
    all_bundles = threads + standalones
    persons_cache, email_to_name = build_persons_cache(all_bundles, store)
    person_profiles = build_person_profiles(store)

    # Compute data span for context ("known 180d of 365d data")
    span_row = store.conn.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM events"
    ).fetchone()
    data_span_days = None
    if span_row and span_row[0] and span_row[1]:
        data_span_days = max(1, (span_row[1] - span_row[0]) // 86400)

    # Step 4: Filter already-processed and low-quality threads
    items_to_process: list[ThreadBundle] = []
    skipped = 0

    all_items = threads + standalones
    if limit:
        all_items = all_items[:limit]

    for bundle in all_items:
        if not force and _is_thread_attempted(store, bundle.thread_id):
            skipped += 1
            continue

        avg_score = sum(
            d.get("score", 0) for d in bundle.triage_data
        ) / max(len(bundle.triage_data), 1)
        if avg_score < min_score:
            skipped += 1
            continue

        items_to_process.append(bundle)

    print(
        f"  {len(items_to_process)} threads to gate ({skipped} skipped)",
        flush=True,
    )

    threads_processed = 0
    actionable_count = 0
    not_actionable_count = 0
    logistics_count = 0
    relational_count = 0
    errors = 0

    # Step 5: Choose LLM client (cloud preferred, local fallback)
    llm = cloud_llm or local_llm
    model = cloud_model or local_model

    # Step 6: Run all 3 gates per thread (parallel where possible)
    def _run_all_gates(
        b: ThreadBundle,
    ) -> tuple[ThreadBundle, GateResult, LogisticsGateResult, RelationalGateResult, int]:
        t0 = time.time()
        actionable_result = run_gate(b, llm, model, user_email, persons_cache, person_profiles, data_span_days, profile_context, email_to_name=email_to_name)
        logistics_result = run_logistics_gate(b, llm, model, user_email, persons_cache, person_profiles, data_span_days, profile_context, email_to_name=email_to_name)
        relational_result = run_relational_gate(b, llm, model, user_email, persons_cache, person_profiles, data_span_days, profile_context, email_to_name=email_to_name)
        elapsed = int((time.time() - t0) * 1000)
        return b, actionable_result, logistics_result, relational_result, elapsed

    def _store_gate_results(
        bundle: ThreadBundle,
        actionable_result: GateResult,
        logistics_result: LogisticsGateResult,
        relational_result: RelationalGateResult,
        elapsed_ms: int,
    ) -> None:
        event_ids = [e.id for e in bundle.events]
        model_tag = model or "gate"

        store.put_claim(_build_gate_claim(
            actionable_result, bundle.thread_id, event_ids, model_tag,
        ))
        store.put_claim(_build_logistics_gate_claim(
            logistics_result, bundle.thread_id, event_ids, model_tag,
        ))
        store.put_claim(_build_relational_gate_claim(
            relational_result, bundle.thread_id, event_ids, model_tag,
        ))

        # Record run sentinel (any gate passing = actionable for resume)
        any_gate = (
            actionable_result.actionable
            or logistics_result.logistics
            or relational_result.relational
        )
        _record_extraction_run(
            store, bundle, model_tag, any_gate, elapsed_ms,
        )

    workers = min(max_concurrent, len(items_to_process)) if items_to_process else 0

    total_to_process = len(items_to_process)

    if workers <= 1:
        for bundle in items_to_process:
            t0 = time.time()
            try:
                a_result = run_gate(bundle, llm, model, user_email, persons_cache, person_profiles, data_span_days, profile_context, email_to_name=email_to_name)
                l_result = run_logistics_gate(bundle, llm, model, user_email, persons_cache, person_profiles, data_span_days, profile_context, email_to_name=email_to_name)
                r_result = run_relational_gate(bundle, llm, model, user_email, persons_cache, person_profiles, data_span_days, profile_context, email_to_name=email_to_name)
                elapsed_ms = int((time.time() - t0) * 1000)

                _store_gate_results(bundle, a_result, l_result, r_result, elapsed_ms)

                if a_result.actionable:
                    actionable_count += 1
                else:
                    not_actionable_count += 1
                if l_result.logistics:
                    logistics_count += 1
                if r_result.relational:
                    relational_count += 1
                threads_processed += 1

                if threads_processed % 50 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"  {threads_processed}/{total_to_process} threads "
                        f"({elapsed:.0f}s)",
                        flush=True,
                    )
            except Exception as e:
                elapsed_ms = int((time.time() - t0) * 1000)
                logger.error(
                    "Gate failed for %s: %s",
                    bundle.thread_id[:30], e,
                )
                _record_extraction_run(
                    store, bundle, "error", False, elapsed_ms,
                    status="error", error_msg=str(e),
                )
                errors += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_all_gates, b): b
                for b in items_to_process
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                bundle_ref = futures[future]
                try:
                    bundle, a_result, l_result, r_result, elapsed_ms = future.result()
                    _store_gate_results(bundle, a_result, l_result, r_result, elapsed_ms)

                    if a_result.actionable:
                        actionable_count += 1
                    else:
                        not_actionable_count += 1
                    if l_result.logistics:
                        logistics_count += 1
                    if r_result.relational:
                        relational_count += 1
                    threads_processed += 1
                except Exception as e:
                    logger.error(
                        "Gate failed for %s: %s",
                        bundle_ref.thread_id[:30], e,
                    )
                    _record_extraction_run(
                        store, bundle_ref, "error", False, 0,
                        status="error", error_msg=str(e),
                    )
                    errors += 1
                if completed % 50 == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        "Gate progress: %d/%d threads",
                        completed, total_to_process,
                    )
                    print(
                        f"  {completed}/{total_to_process} threads "
                        f"({elapsed:.0f}s)",
                        flush=True,
                    )

    elapsed = time.time() - start_time

    logger.info(
        "Gate: %d threads -> %d actionable, %d logistics, %d relational "
        "in %.1fs (%d errors, %d skipped)",
        threads_processed, actionable_count, logistics_count,
        relational_count, elapsed, errors, skipped,
    )

    return {
        "threads_processed": threads_processed,
        "actionable": actionable_count,
        "not_actionable": not_actionable_count,
        "logistics": logistics_count,
        "relational": relational_count,
        "errors": errors,
        "skipped": skipped,
        "elapsed_seconds": round(elapsed, 2),
    }
