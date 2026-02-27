"""CEO Agent — Sam's business co-founder agent.

Capabilities:
  - Investor relations and pipeline management
  - Pitch strategy and framing for business-forward VCs
  - Meeting prep using Granola notes and email context
  - Tracks who he's contacted, meeting outcomes, follow-ups
  - Coordinates with CTO agent for technical context
  - Writes outreach in Sam's voice
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from alteris.agents.base_agent import BaseAgent, ToolDefinition
from alteris.agents.kg_tools import KGTools
from alteris.agents.privacy import build_role_projection
from alteris.agents.workspace import AGENT_CEO, SharedWorkspace
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the CEO Agent for Alteris.
You represent Sam Park, the CEO and business co-founder.

## Your Role
- You are the business voice of the company in investor outreach
- You reach out to business-forward VCs and generalist investors
- You manage the investor pipeline and track relationships
- You prepare pitch strategy and business framing for meetings
- You coordinate with the CTO agent for technical context
- You learn from meeting notes and past interactions to improve pitches

## About Alteris — The Business Story
The pitch narrative (in order):
1. LLMs are powerful but generic. They don't know YOU.
2. We started by building a personal "DNA" — personality, goals, values
3. Realized that real intelligence needs real data, not self-reports
4. Built a local-first knowledge graph that reads your actual digital life
5. Now we're building agents on top — AI that acts on your behalf with
   full context of your life, relationships, and commitments
6. The moat: privacy-first architecture means users trust us with their
   most sensitive data. Competitors either cloud-only (privacy risk)
   or single-source (can't connect the dots).

## Key Business Metrics & Claims
- 7 data sources integrated (Mail, iMessage, WhatsApp, Calendar,
  Contacts, Granola meeting notes, Slack)
- Cross-source person resolution across all sources
- Blind spot briefing: surfaces 3-5 things per week that no single app shows
- Agent layer in development: first agent is VC outreach (dogfooding)
- Two technical co-founders, one business (Sam)
- Stage: pre-seed / seed

## Communication Style (Sam's Voice)
- Warm, relationship-first. Sam builds rapport before pitching.
- Confident and visionary. Paints the big picture.
- Strategic framing — positions Alteris in the market context.
- Direct about what's built vs. what's the vision.
- Uses phrases like: "Here's what's interesting...", "The way I think about it...",
  "What we've proven is..."

## Privacy Rules
- NEVER share personal information from the knowledge graph
- NEVER include health, family, financial details in any output
- When referencing the KG, use ONLY business-relevant facts
- Outreach emails should be genuine, not reveal private information

## Coordination with CTO Agent
- You can ask the CTO agent technical questions via send_to_cto
- Before investor meetings, request a tech brief
- Share investor feedback so the CTO can address concerns
- The CTO handles tech-forward VCs; you handle business-forward ones
"""


def create_ceo_agent(
    store: LayeredGraphStore,
    workspace: SharedWorkspace,
    profile: dict | None = None,
) -> BaseAgent:
    """Create a CEO agent with all its tools registered."""

    agent = BaseAgent(role=AGENT_CEO)
    kg = KGTools(store)

    if profile is None:
        profile_path = Path.home() / ".alteris" / "profile.yaml"
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
        description="Search the knowledge graph for a person by name or email.",
        parameters={
            "type": "object",
            "properties": {
                "name_or_email": {
                    "type": "string",
                    "description": "Name, email, or identifier to search for",
                },
            },
            "required": ["name_or_email"],
        },
        handler=kg.find_person,
    ))

    agent.register_tool(ToolDefinition(
        name="person_context",
        description="Get detailed context about a person: recent interactions, communication patterns, beliefs.",
        parameters={
            "type": "object",
            "properties": {
                "person_id": {"type": "string"},
                "days": {"type": "integer", "default": 30},
            },
            "required": ["person_id"],
        },
        handler=kg.person_context,
    ))

    agent.register_tool(ToolDefinition(
        name="find_warm_paths",
        description="Find warm introduction paths to a target VC. Searches for direct connections, mutual contacts, firm connections.",
        parameters={
            "type": "object",
            "properties": {
                "target_name": {"type": "string"},
                "target_firm": {"type": "string"},
            },
            "required": ["target_name", "target_firm"],
        },
        handler=kg.find_warm_paths,
    ))

    agent.register_tool(ToolDefinition(
        name="voice_samples",
        description="Extract samples of the founder's email writing style.",
        parameters={
            "type": "object",
            "properties": {
                "n_samples": {"type": "integer", "default": 10},
            },
        },
        handler=kg.extract_voice_samples,
    ))

    agent.register_tool(ToolDefinition(
        name="investor_beliefs",
        description="Get beliefs from the knowledge graph related to investors, fundraising, meetings.",
        parameters={"type": "object", "properties": {}},
        handler=kg.get_investor_beliefs,
    ))

    agent.register_tool(ToolDefinition(
        name="upcoming_meetings",
        description="Get upcoming calendar events that might be investor meetings.",
        parameters={
            "type": "object",
            "properties": {
                "days": {"type": "integer", "default": 7},
            },
        },
        handler=kg.upcoming_meetings,
    ))

    agent.register_tool(ToolDefinition(
        name="open_commitments",
        description="Get open commitments, optionally filtered by keyword.",
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "default": ""},
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
                "stage": {"type": "string"},
                "owner": {"type": "string"},
                "focus": {"type": "string"},
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
                "name": {"type": "string"},
                "firm": {"type": "string"},
                "title": {"type": "string"},
                "email": {"type": "string"},
                "tier": {"type": "string"},
                "focus": {"type": "string"},
                "thesis": {"type": "string"},
            },
            "required": ["name", "firm"],
        },
        handler=lambda **kw: {"investor_id": workspace.add_investor(**kw, owner=AGENT_CEO)},
    ))

    agent.register_tool(ToolDefinition(
        name="update_investor",
        description="Update fields on an existing investor record.",
        parameters={
            "type": "object",
            "properties": {
                "inv_id": {"type": "string"},
                "stage": {"type": "string"},
                "notes": {"type": "string"},
                "next_step": {"type": "string"},
                "sentiment": {"type": "string"},
            },
            "required": ["inv_id"],
        },
        handler=lambda **kw: workspace.update_investor(**kw) or {"status": "updated"},
    ))

    agent.register_tool(ToolDefinition(
        name="pipeline_summary",
        description="Get investor pipeline summary: count at each stage.",
        parameters={"type": "object", "properties": {}},
        handler=workspace.pipeline_summary,
    ))

    # ── Messaging Tools ───────────────────────────────────

    agent.register_tool(ToolDefinition(
        name="send_to_cto",
        description="Send a structured message to the CTO agent. Use for questions, investor feedback, meeting prep requests.",
        parameters={
            "type": "object",
            "properties": {
                "msg_type": {"type": "string", "description": "question, investor_update, meeting_prep_request"},
                "subject": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["msg_type", "content"],
        },
        handler=lambda **kw: {
            "msg_id": workspace.send_message(AGENT_CEO, "cto", **kw)
        },
    ))

    agent.register_tool(ToolDefinition(
        name="read_messages",
        description="Read pending messages from the CTO agent.",
        parameters={
            "type": "object",
            "properties": {
                "status": {"type": "string", "default": "pending"},
            },
        },
        handler=lambda status="pending": workspace.get_messages(AGENT_CEO, status),
    ))

    # ── Outreach Tools ────────────────────────────────────

    agent.register_tool(ToolDefinition(
        name="draft_outreach",
        description="Draft an outreach email to a VC. Saved as draft for human approval — NOT sent automatically.",
        parameters={
            "type": "object",
            "properties": {
                "investor_id": {"type": "string"},
                "to_address": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["investor_id", "subject", "body"],
        },
        handler=lambda **kw: {
            "draft_id": workspace.create_outreach(from_agent=AGENT_CEO, **kw),
            "status": "draft — awaiting human approval",
        },
    ))

    agent.register_tool(ToolDefinition(
        name="pending_drafts",
        description="List outreach drafts awaiting approval.",
        parameters={"type": "object", "properties": {}},
        handler=lambda: workspace.pending_drafts(AGENT_CEO),
    ))

    # ── Meeting Prep Tools ────────────────────────────────

    agent.register_tool(ToolDefinition(
        name="create_meeting_prep",
        description="Create a meeting prep document for an upcoming investor meeting.",
        parameters={
            "type": "object",
            "properties": {
                "investor_id": {"type": "string"},
                "meeting_date": {"type": "string", "description": "ISO date string"},
            },
            "required": ["investor_id", "meeting_date"],
        },
        handler=lambda **kw: {
            "prep_id": workspace.create_meeting_prep(prepared_by=AGENT_CEO, **kw)
        },
    ))

    agent.register_tool(ToolDefinition(
        name="get_meeting_prep",
        description="Get existing meeting prep for an investor.",
        parameters={
            "type": "object",
            "properties": {
                "investor_id": {"type": "string"},
            },
            "required": ["investor_id"],
        },
        handler=workspace.get_meeting_prep,
    ))

    # ── Prospect Discovery ─────────────────────────────────

    def _discover_prospects_handler(
        days: int = 180,
        verticals: str = "",
    ) -> dict:
        from alteris.agents.prospect import discover_prospects
        vert_list = [v.strip() for v in verticals.split(",") if v.strip()] if verticals else None
        result = discover_prospects(
            store, workspace,
            llm_client=None,  # KG-only by default in agent context
            verticals=vert_list,
            days=days,
        )
        # Return a concise version for the agent
        combined = result["combined"][:20]
        return {
            "prospects": [
                {
                    "name": p["name"],
                    "firm": p["firm"],
                    "source": p["source_type"],
                    "confidence": p["confidence"],
                    "signals": p["signals"][:3],
                    "warm_paths": len(p.get("warm_paths", [])),
                    "provenance": [
                        {k: v for k, v in prov.items() if k in ("type", "from", "snippet", "firm", "title")}
                        for prov in p.get("provenance", [])[:2]
                    ],
                }
                for p in combined
            ],
            "summary": result["summary"],
        }

    agent.register_tool(ToolDefinition(
        name="discover_prospects",
        description=(
            "Mine the knowledge graph to discover new VC prospects. "
            "Scans emails for introduction offers ('I can connect you with X'), "
            "finds contacts at VC firms, checks fundraising discussions, "
            "and scans calendar events. Returns prospects with provenance "
            "showing exactly which data produced each lead. "
            "Use this BEFORE suggesting VCs from your own knowledge — "
            "the user's network contains leads they may not have noticed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "default": 180,
                    "description": "How many days back to scan",
                },
                "verticals": {
                    "type": "string",
                    "default": "",
                    "description": "Comma-separated verticals to search (e.g. 'AI infrastructure,personal AI')",
                },
            },
        },
        handler=_discover_prospects_handler,
    ))

    # ── Decision Tracking ─────────────────────────────────

    agent.register_tool(ToolDefinition(
        name="record_decision",
        description="Record a strategic decision with rationale.",
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "decision": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["topic", "decision"],
        },
        handler=lambda **kw: {
            "decision_id": workspace.record_decision(decided_by=AGENT_CEO, **kw)
        },
    ))

    # Set system prompt with profile
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
