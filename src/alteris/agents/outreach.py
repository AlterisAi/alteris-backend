"""VC outreach pipeline — research → qualify → route → draft → approve → send.

This is the end-to-end workflow that both agents use for investor outreach.
The pipeline:
  1. RESEARCH:  Web search for VCs matching criteria
  2. QUALIFY:   Score VC fit, check knowledge graph for warm paths
  3. ROUTE:     Assign to CTO (tech-forward) or CEO (business-forward)
  4. DRAFT:     Agent writes personalized outreach in founder's voice
  5. APPROVE:   Human reviews and approves/edits/rejects
  6. SEND:      Approved emails are sent (via SMTP/Gmail)
  7. TRACK:     Monitor for replies, schedule follow-ups
  8. RESPOND:   Agent drafts contextual follow-up based on reply
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from alteris.agents.kg_tools import KGTools
from alteris.agents.protocol import VCProfile, route_vc, score_vc_focus, build_pitch_angle
from alteris.agents.workspace import SharedWorkspace
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)


def research_vc(
    llm_client: Any,
    name: str,
    firm: str,
    model: str = "",
    store: LayeredGraphStore | None = None,
    deep: bool = True,
) -> VCProfile:
    """Research a VC using web search and LLM synthesis.

    When deep=True and store is provided, uses the dossier system for
    multi-source research with Admiralty scoring. Falls back to the
    single-query approach otherwise.
    """
    # Deep dossier-backed research when store is available
    if deep and store is not None:
        try:
            from alteris.agents.dossier import build_dossier
            result = build_dossier(store, llm_client, name, firm, model=model)
            if result.get("dossier"):
                return _dossier_to_profile(name, firm, result)
        except Exception as exc:
            logger.warning("Deep research failed for %s (%s), falling back: %s", name, firm, exc)

    # Fallback: single web search + LLM synthesis
    search_query = f"{name} {firm} venture capital investor thesis portfolio"
    web_result = None
    if hasattr(llm_client, "web_search"):
        web_result = llm_client.web_search(search_query, model=model)

    prompt = f"""Research this venture capitalist and extract structured information.

Name: {name}
Firm: {firm}

{"Web search results:" if web_result else "No web results available."}
{web_result or ""}

Extract the following as JSON:
{{
    "title": "their title at the firm",
    "background": "brief professional background (former roles, education)",
    "thesis": "their investment thesis or focus areas",
    "portfolio": ["list of notable portfolio companies"],
    "content_signals": ["quotes or themes from their public content"],
    "linkedin_summary": "professional summary if available"
}}

Return ONLY valid JSON."""

    response = llm_client.generate(
        prompt,
        system="You are a research assistant. Extract structured data about venture capitalists.",
        model=model,
        format_json=True,
        temperature=0.1,
    )

    profile_data = {}
    if response:
        try:
            profile_data = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse VC research response")

    return VCProfile(
        name=name,
        firm=firm,
        title=profile_data.get("title") or "",
        background=profile_data.get("background") or "",
        thesis=profile_data.get("thesis") or "",
        portfolio=[p for p in (profile_data.get("portfolio") or []) if p],
        content_signals=[s for s in (profile_data.get("content_signals") or []) if s],
        linkedin_summary=profile_data.get("linkedin_summary") or "",
    )


def _dossier_to_profile(name: str, firm: str, result: dict) -> VCProfile:
    """Map dossier belief data → VCProfile fields."""
    dossier = result.get("dossier", {})

    def _extract_value(section: dict, field: str, default: Any = "") -> Any:
        """Pull value from {field: {value: ..., confidence: ..., from: ...}} structure."""
        entry = section.get(field, {})
        if isinstance(entry, dict) and "value" in entry:
            return entry["value"]
        return default

    # Extract basic profile fields from dossier sections
    thesis_section = dossier.get("thesis", {})
    portfolio_section = dossier.get("portfolio", {})
    style_section = dossier.get("style", {})
    interests_section = dossier.get("interests", {})
    fund_section = dossier.get("fund_status", {})
    interactions_section = dossier.get("interactions", {})

    thesis_text = _extract_value(thesis_section, "thesis_summary", "")
    focus_areas = _extract_value(thesis_section, "focus_areas", [])
    if focus_areas and thesis_text:
        thesis_text = f"{thesis_text} Focus: {', '.join(str(a) for a in focus_areas)}"

    portfolio_list = []
    investments = _extract_value(portfolio_section, "recent_investments", [])
    if isinstance(investments, list):
        for inv in investments:
            if isinstance(inv, dict):
                portfolio_list.append(inv.get("company", ""))
            elif isinstance(inv, str):
                portfolio_list.append(inv)
    portfolio_list = [p for p in portfolio_list if p]

    content_signals = []
    quotes = _extract_value(thesis_section, "quotes", [])
    if isinstance(quotes, list):
        content_signals.extend(str(q) for q in quotes[:3])
    topics = _extract_value(interests_section, "recent_topics", [])
    if isinstance(topics, list):
        content_signals.extend(str(t) for t in topics[:3])

    # Aggregate Admiralty from source breakdown
    breakdown = dossier.get("source_breakdown", {})
    sources_used = dossier.get("sources_used", [])
    best_rel = "F"
    best_cred = 6
    if sources_used:
        from alteris.agents.intel.sources import VCSourceType, admiralty_for_source
        rel_order = "ABCDEF"
        for s in sources_used:
            try:
                st = VCSourceType(s.get("source_type", "vc_web"))
                r, c = admiralty_for_source(st)
                if rel_order.index(r) < rel_order.index(best_rel):
                    best_rel = r
                if c < best_cred:
                    best_cred = c
            except (ValueError, KeyError):
                pass

    return VCProfile(
        name=name,
        firm=firm,
        title="",  # Not directly extracted in dossier
        background="",
        thesis=thesis_text or "",
        portfolio=portfolio_list,
        content_signals=content_signals,
        linkedin_summary="",
        dossier_belief_id=result.get("belief_id", ""),
        thesis_detail=thesis_section,
        portfolio_detail=portfolio_section,
        style_detail=style_section,
        interests_detail=interests_section,
        fund_status_detail=fund_section,
        kg_interactions=interactions_section,
        admiralty_score=(best_rel, best_cred),
        overall_confidence=result.get("dossier", {}).get("overall_confidence", 0.0),
        last_researched_at=int(time.time()),
        source_breakdown=breakdown,
    )


def qualify_and_route(
    profile: VCProfile,
    store: LayeredGraphStore,
    workspace: SharedWorkspace,
) -> dict[str, Any]:
    """Qualify a VC and route to the appropriate agent.

    Returns:
        {
            'investor_id': str,
            'focus': str,
            'owner': str,
            'warm_paths': list,
            'pitch_angle': str,
            'score': dict,
        }
    """
    kg = KGTools(store)

    # If an introducer is specified but no agent assigned, detect whose contact they are
    if profile.introduced_by and not profile.introducer_agent:
        intro_info = kg.whose_contact(profile.introduced_by)
        if intro_info["is_user_contact"]:
            # This KG's owner knows the introducer — they own the outreach
            profile.introducer_agent = "cto"  # Will be overridden per-machine
            logger.info(
                "Intro routing: %s is in this KG (count=%d) → owner gets outreach",
                profile.introduced_by, intro_info["interaction_count"],
            )

    # Score and route
    score = score_vc_focus(profile)
    owner = score["recommended_owner"]

    # Check for warm paths
    warm_paths = kg.find_warm_paths(profile.name, profile.firm)

    # If we have a warm path, that's a strong signal — bump priority
    tier = "tier3"
    if any(p["strength"] == "strong" for p in warm_paths):
        tier = "tier1"
    elif any(p["strength"] == "medium" for p in warm_paths):
        tier = "tier2"

    # Build pitch angle
    pitch_angle = build_pitch_angle(profile, owner)

    # Add to workspace pipeline
    investor_id = workspace.add_investor(
        name=profile.name,
        firm=profile.firm,
        title=profile.title,
        email="",  # Will be filled later
        tier=tier,
        focus=score["focus"],
        owner=owner,
        thesis=profile.thesis,
        introduced_by=profile.introduced_by,
        introducer_agent=profile.introducer_agent,
        warm_path=[
            f"{p['type']}: {p.get('context', p.get('person_name', ''))}"
            for p in warm_paths[:5]
        ],
    )

    # Update stage to qualified
    workspace.update_investor(investor_id, stage="qualified")

    logger.info(
        "Qualified %s (%s): focus=%s, owner=%s, tier=%s, warm_paths=%d",
        profile.name, profile.firm, score["focus"], owner, tier, len(warm_paths),
    )

    return {
        "investor_id": investor_id,
        "focus": score["focus"],
        "owner": owner,
        "warm_paths": warm_paths,
        "pitch_angle": pitch_angle,
        "score": score,
        "tier": tier,
    }


def run_outreach_batch(
    vcs: list[dict[str, str]],
    store: LayeredGraphStore,
    workspace: SharedWorkspace,
    llm_client: Any,
    model: str = "",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the full outreach pipeline on a batch of VCs.

    Args:
        vcs: List of {"name": "...", "firm": "..."} dicts
        store: Knowledge graph store
        workspace: Shared workspace
        llm_client: LLM client for research and drafting
        model: Model to use
        dry_run: If True, research and route but don't draft emails

    Returns:
        Summary of the batch: routed counts, drafts created, etc.
    """
    results = {
        "total": len(vcs),
        "researched": 0,
        "qualified": 0,
        "routed_cto": 0,
        "routed_ceo": 0,
        "warm_paths_found": 0,
        "drafts_created": 0,
        "investors": [],
    }

    for vc in vcs:
        name = vc["name"]
        firm = vc["firm"]

        # Step 1: Research
        logger.info("Researching %s (%s)...", name, firm)
        profile = research_vc(llm_client, name, firm, model=model)
        results["researched"] += 1

        # Step 2: Qualify and route
        qr = qualify_and_route(profile, store, workspace)
        results["qualified"] += 1

        if qr["owner"] == "cto":
            results["routed_cto"] += 1
        else:
            results["routed_ceo"] += 1

        if qr["warm_paths"]:
            results["warm_paths_found"] += 1

        investor_info = {
            "name": name,
            "firm": firm,
            "investor_id": qr["investor_id"],
            "owner": qr["owner"],
            "focus": qr["focus"],
            "tier": qr["tier"],
            "pitch_angle": qr["pitch_angle"],
            "warm_paths": len(qr["warm_paths"]),
        }
        results["investors"].append(investor_info)

        if not dry_run:
            # Step 3: Draft outreach (using the appropriate agent)
            # The agent itself handles drafting — here we just notify
            workspace.send_message(
                from_agent="system",
                to_agent=qr["owner"],
                msg_type="outreach_review",
                subject=f"Draft outreach for {name} ({firm})",
                content=json.dumps({
                    "investor_id": qr["investor_id"],
                    "name": name,
                    "firm": firm,
                    "pitch_angle": qr["pitch_angle"],
                    "warm_paths": qr["warm_paths"][:3],
                    "action": "Please draft an outreach email for this investor.",
                }),
            )
            results["drafts_created"] += 1

    return results


def review_drafts(workspace: SharedWorkspace, agent: str) -> list[dict]:
    """Get all pending drafts for human review.

    Returns drafts with investor context for easy review.
    """
    drafts = workspace.pending_drafts(agent)
    enriched = []
    for d in drafts:
        investor = workspace.get_investor(d["investor_id"]) if d.get("investor_id") else None
        enriched.append({
            "draft_id": d["id"],
            "investor": investor or {},
            "subject": d["subject"],
            "body": d["body"],
            "channel": d["channel"],
            "created_at": d["created_at"],
        })
    return enriched


def approve_draft(workspace: SharedWorkspace, draft_id: str) -> dict:
    """Approve a draft for sending."""
    workspace.approve_outreach(draft_id)
    return {"draft_id": draft_id, "status": "approved"}


def reject_draft(workspace: SharedWorkspace, draft_id: str) -> dict:
    """Reject a draft (deletes it)."""
    conn = workspace._get_conn()
    conn.execute(
        "UPDATE outreach SET status = 'rejected' WHERE id = ?",
        (draft_id,),
    )
    conn.commit()
    return {"draft_id": draft_id, "status": "rejected"}
