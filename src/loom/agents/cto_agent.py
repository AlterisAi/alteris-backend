"""CTO Agent — Alex's technical co-founder agent.

Capabilities:
  - Deep technical knowledge of the Alteris architecture
  - Can answer investor questions about the technology
  - Reaches out to tech-forward VCs with technical depth
  - Prepares technical briefs for investor meetings
  - Posts tech status updates to the shared workspace
  - Queries the knowledge graph for relationship context
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from loom.agents.base_agent import BaseAgent, ToolDefinition
from loom.agents.kg_tools import KGTools
from loom.agents.privacy import build_role_projection
from loom.agents.workspace import AGENT_CTO, SharedWorkspace
from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the CTO Agent for Alteris.
You represent Alex Chen, the technical co-founder and CTO.

## Your Role
- You are the technical voice of the company in investor outreach
- You reach out to tech-forward VCs who appreciate technical depth
- You prepare technical briefs for investor meetings
- You answer technical questions from the CEO agent
- You post technology status updates to the shared workspace

## About Alteris
Alteris is building a personal AI that truly knows you. The core technology:
- LOCAL-FIRST knowledge graph: runs on the user's Mac, reads their email,
  calendar, messages, meeting notes — 7 data sources
- Cross-source person resolution: maps identities across email, phone,
  Slack handles using union-find with O(α(n)) lookups
- Multi-stage intelligence pipeline: deterministic claims → LLM triage →
  graph propagation → deep extraction → belief synthesis
- Privacy architecture: sensitive content stays local (Ollama), only
  non-sensitive goes to cloud (Gemini). Per-field sensitivity classification.
- Blind spot briefing: the system looks at your upcoming week, traverses
  the knowledge graph, and surfaces things you've missed
- Agent layer (NEW): autonomous agents built on the knowledge graph
  with privacy-preserving cross-agent communication

## Technical Differentiators (for pitching)
1. Local-first = privacy moat. No cloud dependency for core intelligence.
2. Cross-source reasoning. No single app can do this — we connect the dots
   between email, calendar, messages, meeting notes.
3. Belief synthesis with auditable provenance. Every insight traces back
   to source evidence through claims → beliefs → briefing.
4. Mixture-of-experts for deduplication. Cheap deterministic signals
   (Jaccard, SequenceMatcher, prefix) vote to merge related items.
5. Agent architecture with privacy boundaries. Agents can collaborate
   without sharing personal data — role-based projections.

## Communication Style
- Technical but accessible. Don't dumb it down, but don't lecture.
- Concise. Investors are busy. Lead with the insight, not the journey.
- Confident but honest about what's built vs. planned.
- When writing emails: professional, warm, direct. No buzzwords.

## Privacy Rules
- NEVER share personal information from the knowledge graph
- NEVER include health, family, financial details in any output
- When referencing the KG, use ONLY business-relevant facts
- Outreach emails should be genuine, not reveal private information

## Tools Available
You have tools to:
- Query the knowledge graph for relationships, communication patterns
- Search for warm introduction paths to investors
- Read/write the shared workspace (investor pipeline, messages)
- Draft outreach emails in the founder's voice
- Post tech status updates for the CEO agent
"""


def create_cto_agent(
    store: LayeredGraphStore,
    workspace: SharedWorkspace,
    profile: dict | None = None,
) -> BaseAgent:
    """Create a CTO agent with all its tools registered."""

    agent = BaseAgent(role=AGENT_CTO)
    kg = KGTools(store)

    # Load profile for voice calibration
    if profile is None:
        profile_path = Path.home() / ".loom" / "profile.yaml"
        if profile_path.exists():
            try:
                import yaml
                profile = yaml.safe_load(profile_path.read_text()) or {}
            except Exception:
                profile = {}
        else:
            profile = {}

    # ── KG Tools ──────────────────────────────────────────

    agent.register_tool(ToolDefinition(
        name="find_person",
        description="Search the knowledge graph for a person by name or email. Returns matching persons with identifiers and interaction history.",
        parameters={
            "type": "object",
            "properties": {
                "name_or_email": {
                    "type": "string",
                    "description": "Name, email address, or other identifier to search for",
                },
            },
            "required": ["name_or_email"],
        },
        handler=kg.find_person,
    ))

    agent.register_tool(ToolDefinition(
        name="person_context",
        description="Get detailed context about a person: recent interactions, communication patterns, beliefs about them.",
        parameters={
            "type": "object",
            "properties": {
                "person_id": {"type": "string", "description": "Person ID from the knowledge graph"},
                "days": {"type": "integer", "description": "How many days back to look (default 30)", "default": 30},
            },
            "required": ["person_id"],
        },
        handler=kg.person_context,
    ))

    agent.register_tool(ToolDefinition(
        name="find_warm_paths",
        description="Find warm introduction paths to a target VC or person. Searches for direct connections, mutual contacts, and firm-level connections.",
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string", "description": "Name of the target person"},
                "target_firm": {"type": "string", "description": "Name of their firm/company"},
            },
            "required": ["target_name", "target_firm"],
        },
        handler=kg.find_warm_paths,
    ))

    agent.register_tool(ToolDefinition(
        name="voice_samples",
        description="Extract samples of the founder's email writing style for calibrating outreach tone.",
        parameters={
            "type": "object",
            "properties": {
                "n_samples": {"type": "integer", "description": "Number of samples to extract (default 10)", "default": 10},
            },
        },
        handler=kg.extract_voice_samples,
    ))

    agent.register_tool(ToolDefinition(
        name="tech_beliefs",
        description="Get beliefs from the knowledge graph related to technology, product, and architecture.",
        parameters={"type": "object", "properties": {}},
        handler=kg.get_tech_beliefs,
    ))

    agent.register_tool(ToolDefinition(
        name="upcoming_meetings",
        description="Get upcoming calendar events that might be investor meetings.",
        parameters={
            "type": "object",
            "properties": {
                "days": {"type": "integer", "description": "Days ahead to look (default 7)", "default": 7},
            },
        },
        handler=kg.upcoming_meetings,
    ))

    agent.register_tool(ToolDefinition(
        name="open_commitments",
        description="Get open commitments from the knowledge graph, optionally filtered by keyword.",
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Filter keyword (optional)", "default": ""},
            },
        },
        handler=kg.open_commitments,
    ))

    # ── Workspace Tools ───────────────────────────────────

    agent.register_tool(ToolDefinition(
        name="list_investors",
        description="List investors in the pipeline with optional filters.",
        parameters={
            "type": "object",
            "properties": {
                "stage": {"type": "string", "description": "Filter by stage (e.g., 'contacted', 'meeting_scheduled')"},
                "owner": {"type": "string", "description": "Filter by owner ('cto' or 'ceo')"},
                "focus": {"type": "string", "description": "Filter by focus ('tech_forward', 'business_forward', 'generalist')"},
            },
        },
        handler=lambda **kw: workspace.list_investors(**{k: v for k, v in kw.items() if v}),
    ))

    agent.register_tool(ToolDefinition(
        name="add_investor",
        description="Add a new investor to the pipeline.",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Investor's name"},
                "firm": {"type": "string", "description": "Firm/fund name"},
                "title": {"type": "string", "description": "Their title"},
                "email": {"type": "string", "description": "Email address"},
                "tier": {"type": "string", "description": "tier1, tier2, or tier3"},
                "focus": {"type": "string", "description": "tech_forward, business_forward, or generalist"},
                "thesis": {"type": "string", "description": "Their investment thesis or focus areas"},
            },
            "required": ["name", "firm"],
        },
        handler=lambda **kw: {"investor_id": workspace.add_investor(**kw, owner=AGENT_CTO)},
    ))

    agent.register_tool(ToolDefinition(
        name="update_investor",
        description="Update fields on an existing investor record.",
        parameters={
            "type": "object",
            "properties": {
                "inv_id": {"type": "string", "description": "Investor ID"},
                "stage": {"type": "string", "description": "New stage"},
                "notes": {"type": "string", "description": "Additional notes (will be appended)"},
                "next_step": {"type": "string", "description": "What to do next"},
            },
            "required": ["inv_id"],
        },
        handler=lambda **kw: workspace.update_investor(**kw) or {"status": "updated"},
    ))

    agent.register_tool(ToolDefinition(
        name="pipeline_summary",
        description="Get a summary of the investor pipeline: count at each stage.",
        parameters={"type": "object", "properties": {}},
        handler=workspace.pipeline_summary,
    ))

    # ── Messaging Tools ───────────────────────────────────

    agent.register_tool(ToolDefinition(
        name="send_to_ceo",
        description="Send a structured message to the CEO agent. Use this to share tech status, answer questions, or coordinate.",
        parameters={
            "type": "object",
            "properties": {
                "msg_type": {"type": "string", "description": "Message type: 'tech_status', 'answer', 'question', 'meeting_brief'"},
                "subject": {"type": "string", "description": "Brief subject line"},
                "content": {"type": "string", "description": "Message content (structured JSON string preferred)"},
            },
            "required": ["msg_type", "content"],
        },
        handler=lambda **kw: {
            "msg_id": workspace.send_message(AGENT_CTO, "ceo", **kw)
        },
    ))

    agent.register_tool(ToolDefinition(
        name="read_messages",
        description="Read pending messages from the CEO agent.",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "Filter by status: 'pending', 'read', 'answered'", "default": "pending"},
            },
        },
        handler=lambda status="pending": workspace.get_messages(AGENT_CTO, status),
    ))

    # ── Outreach Tools ────────────────────────────────────

    agent.register_tool(ToolDefinition(
        name="draft_outreach",
        description="Draft an outreach email to a VC. The email will be saved as a draft for human approval — it will NOT be sent automatically.",
        parameters={
            "type": "object",
            "properties": {
                "investor_id": {"type": "string", "description": "Investor ID from the pipeline"},
                "to_address": {"type": "string", "description": "Recipient email"},
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body text"},
            },
            "required": ["investor_id", "subject", "body"],
        },
        handler=lambda **kw: {
            "draft_id": workspace.create_outreach(from_agent=AGENT_CTO, **kw),
            "status": "draft — awaiting human approval",
        },
    ))

    agent.register_tool(ToolDefinition(
        name="pending_drafts",
        description="List all outreach drafts awaiting approval.",
        parameters={"type": "object", "properties": {}},
        handler=lambda: workspace.pending_drafts(AGENT_CTO),
    ))

    # Set the system prompt
    enriched_prompt = SYSTEM_PROMPT
    if profile:
        name = profile.get("name", "")
        if name:
            enriched_prompt += f"\n\n## Founder Profile\nName: {name}\n"
        emails = profile.get("emails", [])
        if emails:
            enriched_prompt += f"Email addresses: {', '.join(emails)}\n"

    agent.get_system_prompt = lambda: enriched_prompt

    return agent
