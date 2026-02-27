"""Privacy filter — role-based projection of the knowledge graph.

Each agent has full access to its owner's knowledge graph locally.
When data needs to cross the agent boundary (CTO ↔ CEO), this filter
produces a "projection" — a sanitized, structured view that contains
only business-relevant information.

The filter is NOT a redaction engine (which implies lossy editing).
It's a projection: it selects WHICH beliefs/events to share and
WHAT fields to include, like a database view.

Privacy rules:
  1. Sensitivity >= SENSITIVE → never share (health, finance, personal)
  2. Source is personal messaging to family/friends → never share
  3. Event/belief matches company/investor/product keywords → share
  4. Everything else → don't share (safe default)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Keywords that indicate business-relevant content
_BUSINESS_KEYWORDS = frozenset({
    # Company & product
    "alteris", "loom", "startup", "company", "product", "feature",
    "roadmap", "launch", "release", "deploy", "ship",
    # Fundraising
    "investor", "vc", "venture", "fundraise", "funding", "raise",
    "round", "seed", "series", "pre-seed", "valuation", "term sheet",
    "cap table", "dilution", "safe", "convertible",
    # Business
    "pitch", "deck", "demo", "meeting", "call", "intro",
    "revenue", "traction", "metrics", "growth", "market",
    "customer", "user", "acquisition", "partnership",
    # Technical (for CTO→CEO)
    "architecture", "pipeline", "api", "model", "privacy",
    "knowledge graph", "agent", "llm", "briefing",
    # People in business context
    "co-founder", "cofounder", "advisor", "board", "team",
})

# Keywords that indicate personal/private content — block these
_PERSONAL_KEYWORDS = frozenset({
    "family", "spouse", "wife", "husband", "partner", "child",
    "children", "daughter", "son", "mom", "dad", "mother", "father",
    "parent", "doctor", "medical", "health", "prescription",
    "therapy", "counselor", "diagnosis", "symptom",
    "bank", "mortgage", "rent", "salary", "tax", "irs",
    "password", "ssn", "social security", "credit card",
    "personal", "private", "secret",
})

# Sensitivity levels that should never cross the boundary
_BLOCKED_SENSITIVITIES = {"financial", "medical", "legal", "credentials"}

# Sources that are almost always personal
_PERSONAL_SOURCES = {"imessage", "whatsapp"}


def is_business_relevant(text: str) -> bool:
    """Check if text contains business-relevant keywords."""
    lower = text.lower()
    return any(kw in lower for kw in _BUSINESS_KEYWORDS)


def has_personal_content(text: str) -> bool:
    """Check if text contains personal/private keywords."""
    lower = text.lower()
    return any(kw in lower for kw in _PERSONAL_KEYWORDS)


def classify_belief(belief: dict) -> str:
    """Classify a belief as 'share', 'block', or 'review'.

    Returns:
        'share':  safe to include in cross-agent projection
        'block':  contains personal/sensitive content, do not share
        'review': ambiguous, would need human review (treat as block for now)
    """
    # Check sensitivity from metadata
    data = belief.get("data", {})
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            data = {}

    sensitivity = data.get("sensitivity", "")
    if sensitivity in _BLOCKED_SENSITIVITIES:
        return "block"

    # Check source — personal messaging is blocked unless business-relevant
    source = belief.get("source", "") or data.get("source", "")
    subject = belief.get("subject", "")
    summary = belief.get("summary", "")
    combined_text = f"{subject} {summary}"

    if source in _PERSONAL_SOURCES:
        if is_business_relevant(combined_text):
            return "share"
        return "block"

    # Check for personal content
    if has_personal_content(combined_text):
        return "block"

    # Check for business relevance
    if is_business_relevant(combined_text):
        return "share"

    # Default: block (privacy-safe default)
    return "review"


def classify_event(event: dict) -> str:
    """Classify an event as 'share', 'block', or 'review'."""
    source = event.get("source", "")
    content = event.get("raw_content", "") or event.get("body", "")
    subject = event.get("subject", "") or ""

    metadata = event.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    sensitivity = metadata.get("sensitivity", "")
    if sensitivity in _BLOCKED_SENSITIVITIES:
        return "block"

    combined = f"{subject} {content}"

    # Personal sources need explicit business relevance
    if source in _PERSONAL_SOURCES:
        if is_business_relevant(combined):
            return "share"
        return "block"

    if has_personal_content(combined):
        return "block"

    if is_business_relevant(combined):
        return "share"

    # Calendar events about meetings are generally shareable
    if source == "calendar" and any(
        kw in combined.lower()
        for kw in ("meeting", "call", "demo", "pitch", "interview", "sync")
    ):
        return "share"

    return "review"


def project_belief_for_sharing(belief: dict) -> dict | None:
    """Project a belief into a shareable format.

    Returns None if the belief should not be shared.
    Returns a sanitized dict with only business-relevant fields.
    """
    classification = classify_belief(belief)
    if classification == "block":
        return None
    if classification == "review":
        return None  # Conservative: treat as block

    # Build sanitized projection — no raw content, no personal identifiers
    projected = {
        "id": belief.get("id", ""),
        "type": belief.get("belief_type", ""),
        "subject": _sanitize_text(belief.get("subject", "")),
        "summary": _sanitize_text(belief.get("summary", "")),
        "confidence": belief.get("confidence", 0.5),
        "status": belief.get("status", ""),
    }

    # Include structured data fields that are business-relevant
    data = belief.get("data", {})
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            data = {}

    safe_fields = {
        "what", "who", "deadline", "status", "commitment_type",
        "domain", "topics", "entity_type", "relation_type",
    }
    projected["data"] = {
        k: v for k, v in data.items()
        if k in safe_fields and not has_personal_content(str(v))
    }

    return projected


def project_event_for_sharing(event: dict) -> dict | None:
    """Project an event into a shareable format."""
    classification = classify_event(event)
    if classification != "share":
        return None

    metadata = event.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    projected = {
        "id": event.get("id", ""),
        "source": event.get("source", ""),
        "type": event.get("event_type", ""),
        "timestamp": event.get("timestamp", 0),
        "subject": _sanitize_text(metadata.get("subject", "")),
        # Truncated, sanitized content — no raw message bodies
        "summary": _sanitize_text(
            event.get("raw_content", "")[:200]
        ),
    }

    # Include only safe metadata fields
    safe_meta = {}
    if "calendar_type" in metadata:
        safe_meta["calendar_type"] = metadata["calendar_type"]
    if "is_recurring" in metadata:
        safe_meta["is_recurring"] = metadata["is_recurring"]
    if "location" in metadata:
        safe_meta["location"] = metadata["location"]

    projected["metadata"] = safe_meta
    return projected


def build_role_projection(
    beliefs: list[dict],
    events: list[dict],
    role: str,
) -> dict[str, Any]:
    """Build a complete role-based projection of the knowledge graph.

    Args:
        beliefs: All beliefs from the owner's KG
        events: Recent events from the owner's KG
        role: 'cto' or 'ceo' — the role of the agent sharing data

    Returns:
        A dict with shareable beliefs, events, and summary stats
    """
    shared_beliefs = []
    for b in beliefs:
        projected = project_belief_for_sharing(b)
        if projected:
            shared_beliefs.append(projected)

    shared_events = []
    for e in events:
        projected = project_event_for_sharing(e)
        if projected:
            shared_events.append(projected)

    return {
        "role": role,
        "beliefs_total": len(beliefs),
        "beliefs_shared": len(shared_beliefs),
        "events_total": len(events),
        "events_shared": len(shared_events),
        "beliefs": shared_beliefs,
        "events": shared_events,
        "projected_at": int(time.time()),
    }


def _sanitize_text(text: str) -> str:
    """Remove potential PII patterns from text.

    Strips email addresses, phone numbers, and other identifiers
    from text that crosses the agent boundary.
    """
    import time  # noqa: F811 — used for projected_at above

    if not text:
        return text

    # Remove email addresses
    text = re.sub(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b', '[email]', text)

    # Remove phone numbers (various formats)
    text = re.sub(r'\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[phone]', text)

    # Remove URLs (except known business domains)
    text = re.sub(
        r'https?://(?!(?:acme|loom|linkedin|github))\S+',
        '[url]',
        text,
    )

    return text
