"""Message protocol and VC routing logic.

Structured message schemas for cross-agent communication and the
routing algorithm that assigns VCs to CTO or CEO based on their
profile signals.

Routing logic:
  - Tech-forward VCs (former engineers, deep tech portfolio,
    technical blog posts) → CTO agent outreach
  - Business-forward VCs (market-focused, growth metrics,
    operator background) → CEO agent outreach
  - Generalists → CEO agent by default, CTO available for follow-up
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Message types ─────────────────────────────────────────────

MSG_QUESTION = "question"           # One agent asks the other
MSG_ANSWER = "answer"               # Response to a question
MSG_INVESTOR_UPDATE = "investor_update"  # Pipeline status change
MSG_TECH_STATUS = "tech_status"     # CTO posts tech status
MSG_MEETING_REQUEST = "meeting_prep_request"  # Request prep from other agent
MSG_MEETING_BRIEF = "meeting_brief"  # Prep materials delivered
MSG_OUTREACH_REVIEW = "outreach_review"  # Draft for approval
MSG_DECISION = "decision"           # Strategic decision made

VALID_MSG_TYPES = {
    MSG_QUESTION, MSG_ANSWER, MSG_INVESTOR_UPDATE, MSG_TECH_STATUS,
    MSG_MEETING_REQUEST, MSG_MEETING_BRIEF, MSG_OUTREACH_REVIEW,
    MSG_DECISION,
}


# ── Structured message schemas ────────────────────────────────

@dataclass
class InvestorUpdate:
    """CEO → CTO: investor pipeline update."""
    investor_name: str
    firm: str
    stage: str
    sentiment: str = ""            # positive, neutral, negative, unknown
    concerns: list[str] = field(default_factory=list)
    next_step: str = ""
    needs_from_other: str = ""     # What the other agent should prepare

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class TechStatus:
    """CTO → CEO: technical status update."""
    feature: str
    status: str                    # built, in_progress, planned, blocked
    details: str = ""
    demo_ready: bool = False
    blockers: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class MeetingPrepRequest:
    """Request meeting prep from the other agent."""
    investor_name: str
    firm: str
    meeting_date: str
    meeting_type: str = "first_meeting"  # first_meeting, follow_up, dd
    specific_questions: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class OutreachDraft:
    """Draft email for review before sending."""
    investor_name: str
    firm: str
    to_email: str
    subject: str
    body: str
    pitch_angle: str = ""
    personalization_notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# ── VC Routing ────────────────────────────────────────────────

# Signals that a VC prefers technical founders / deep tech
TECH_FORWARD_SIGNALS = {
    # Background keywords
    "backgrounds": [
        "engineer", "engineering", "cto", "technical", "developer",
        "phd", "research", "computer science", "machine learning",
        "ai/ml", "systems", "infrastructure",
    ],
    # Portfolio / thesis keywords
    "thesis": [
        "deep tech", "technical moat", "hard tech", "infrastructure",
        "developer tools", "devtools", "open source", "protocol",
        "ai native", "ml infrastructure", "data infrastructure",
        "systems", "platform", "api-first",
    ],
    # Content signals from blog posts / tweets
    "content": [
        "technical due diligence", "architecture review",
        "code quality", "engineering culture", "technical founder",
        "build vs buy", "scaling challenges", "system design",
    ],
}

# Signals that a VC prefers business/market-oriented pitches
BUSINESS_FORWARD_SIGNALS = {
    "backgrounds": [
        "operator", "business", "mba", "sales", "marketing",
        "growth", "product manager", "pm", "strategy",
        "consulting", "banking", "finance",
    ],
    "thesis": [
        "go to market", "gtm", "product market fit", "pmf",
        "network effects", "marketplace", "consumer", "b2b saas",
        "enterprise", "revenue", "arr", "growth stage",
        "category creation", "market timing",
    ],
    "content": [
        "traction metrics", "unit economics", "tam sam som",
        "market size", "customer acquisition", "churn",
        "business model", "pricing strategy", "sales motion",
    ],
}


@dataclass
class VCProfile:
    """Profile of a VC for routing decisions."""
    name: str
    firm: str
    title: str = ""
    background: str = ""
    thesis: str = ""
    portfolio: list[str] = field(default_factory=list)
    content_signals: list[str] = field(default_factory=list)
    linkedin_summary: str = ""
    # Intro routing: who made the connection and whose network they're in
    introduced_by: str = ""        # Name of the person who introduced
    introducer_agent: str = ""     # 'cto' or 'ceo' — whose contact is the introducer
    # Deep dossier fields (populated when dossier system is used)
    dossier_belief_id: str = ""
    thesis_detail: dict = field(default_factory=dict)
    portfolio_detail: dict = field(default_factory=dict)
    style_detail: dict = field(default_factory=dict)
    interests_detail: dict = field(default_factory=dict)
    fund_status_detail: dict = field(default_factory=dict)
    kg_interactions: dict = field(default_factory=dict)
    admiralty_score: tuple[str, int] = ("F", 6)
    overall_confidence: float = 0.0
    last_researched_at: int = 0
    source_breakdown: dict = field(default_factory=dict)


def score_vc_focus(profile: VCProfile) -> dict[str, Any]:
    """Score a VC profile as tech-forward vs business-forward.

    Returns:
        {
            'focus': 'tech_forward' | 'business_forward' | 'generalist',
            'tech_score': float,
            'business_score': float,
            'signals': [str],  # Which signals matched
            'recommended_owner': 'cto' | 'ceo',
        }
    """
    # Build text corpus — use deep intel when available
    parts = [
        profile.background,
        profile.thesis,
        profile.linkedin_summary,
        " ".join(s for s in profile.portfolio if s),
        " ".join(s for s in profile.content_signals if s),
        profile.title,
    ]

    # Enrich from dossier detail fields if populated
    if profile.interests_detail:
        topics = profile.interests_detail.get("recent_topics", {})
        if isinstance(topics, dict) and topics.get("value"):
            parts.append(" ".join(str(t) for t in topics["value"]))
    if profile.thesis_detail:
        focus = profile.thesis_detail.get("focus_areas", {})
        if isinstance(focus, dict) and focus.get("value"):
            parts.append(" ".join(str(a) for a in focus["value"]))
        summary = profile.thesis_detail.get("thesis_summary", {})
        if isinstance(summary, dict) and summary.get("value"):
            parts.append(str(summary["value"]))
    if profile.style_detail:
        style_sum = profile.style_detail.get("style_summary", {})
        if isinstance(style_sum, dict) and style_sum.get("value"):
            parts.append(str(style_sum["value"]))

    combined = " ".join(filter(None, parts)).lower()

    tech_score = 0.0
    business_score = 0.0
    signals: list[str] = []

    # Score tech signals
    for category, keywords in TECH_FORWARD_SIGNALS.items():
        for kw in keywords:
            if kw in combined:
                tech_score += 1.0
                signals.append(f"tech:{category}:{kw}")

    # Score business signals
    for category, keywords in BUSINESS_FORWARD_SIGNALS.items():
        for kw in keywords:
            if kw in combined:
                business_score += 1.0
                signals.append(f"biz:{category}:{kw}")

    # Title-based boost
    title_lower = (profile.title or "").lower()
    if any(t in title_lower for t in ("cto", "technical", "engineer", "principal")):
        tech_score += 2.0
        signals.append("tech:title_match")
    if any(t in title_lower for t in ("partner", "managing", "general partner")):
        business_score += 0.5  # GPs are often generalists

    # Determine focus from signal balance
    total = tech_score + business_score
    if total == 0:
        focus = "generalist"
    elif tech_score > business_score * 1.5:
        focus = "tech_forward"
    elif business_score > tech_score * 1.5:
        focus = "business_forward"
    else:
        focus = "generalist"

    # ── Routing decision (three tiers of signal) ──
    #
    # Tier 1 — INTRO PATH (strongest signal, overrides everything).
    # Whoever's network produced the introduction owns the outreach.
    # "Sid told me about you" only works from the person Sid knows.
    #
    # Tier 2 — VC FOCUS (fallback for cold outreach).
    # Tech-forward VCs get the CTO, business-forward get the CEO.
    #
    # Tier 3 — DEFAULT (CEO handles unknowns / generalists).

    if profile.introducer_agent:
        recommended_owner = profile.introducer_agent
        signals.append(f"intro:routed_via:{profile.introduced_by or 'warm_path'}")
    elif focus == "tech_forward":
        recommended_owner = "cto"
    elif focus == "business_forward":
        recommended_owner = "ceo"
    else:
        recommended_owner = "ceo"  # Generalist / unknown → CEO

    return {
        "focus": focus,
        "tech_score": round(tech_score, 1),
        "business_score": round(business_score, 1),
        "signals": signals[:10],  # Cap for readability
        "recommended_owner": recommended_owner,
        "intro_override": bool(profile.introducer_agent),
    }


def route_vc(profile: VCProfile) -> str:
    """Route a VC to the appropriate agent.

    Routing priority:
      1. Intro path — whoever's network produced the intro owns outreach
      2. VC focus — tech-forward → CTO, business-forward → CEO
      3. Default — generalists/unknowns → CEO

    Returns: 'cto' or 'ceo'
    """
    result = score_vc_focus(profile)
    owner = result["recommended_owner"]
    if result.get("intro_override"):
        logger.info(
            "Routing %s (%s) → %s [INTRO via %s — overrides focus=%s]",
            profile.name, profile.firm, owner,
            profile.introduced_by, result["focus"],
        )
    else:
        logger.info(
            "Routing %s (%s) → %s [tech=%.1f, biz=%.1f, focus=%s]",
            profile.name, profile.firm, owner,
            result["tech_score"], result["business_score"], result["focus"],
        )
    return owner


def build_pitch_angle(profile: VCProfile, owner: str) -> str:
    """Generate a pitch angle recommendation based on VC profile and owner.

    Returns a brief instruction for the agent on how to frame the pitch.
    """
    result = score_vc_focus(profile)
    signals = result["signals"]

    if owner == "cto":
        # Technical founder pitch
        angles = []
        if any("infrastructure" in s for s in signals):
            angles.append("Lead with the local-first architecture and privacy moat")
        if any("ai" in s or "ml" in s for s in signals):
            angles.append("Emphasize the multi-model pipeline (Gemini + Ollama) and belief synthesis")
        if any("open source" in s for s in signals):
            angles.append("Mention the open agent layer and plugin architecture")
        if any("developer" in s or "devtools" in s for s in signals):
            angles.append("Frame as developer-first platform with CLI + SDK")
        if not angles:
            angles.append("Lead with technical depth: knowledge graph, cross-source reasoning, privacy architecture")
        return " | ".join(angles)

    else:
        # Business founder pitch
        angles = []
        if any("consumer" in s for s in signals):
            angles.append("Frame as personal AI that everyone needs")
        if any("enterprise" in s or "b2b" in s for s in signals):
            angles.append("Position as enterprise personal AI with compliance built in")
        if any("market" in s or "tam" in s for s in signals):
            angles.append("Start with market size: everyone with email + calendar")
        if any("growth" in s or "traction" in s for s in signals):
            angles.append("Lead with user testimonials and engagement metrics")
        if not angles:
            angles.append("Lead with the vision: AI that truly knows you, built on your own data")
        return " | ".join(angles)
