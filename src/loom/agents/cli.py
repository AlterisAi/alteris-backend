"""Agent CLI — run CTO/CEO agents, manage outreach pipeline.

Usage:
  python -m loom.agents.cli cto "Find warm paths to Jose at Bunch Capital"
  python -m loom.agents.cli ceo "Prep for tomorrow's meeting with Villi"
  python -m loom.agents.cli research --name "Jose" --firm "Bunch Capital"
  python -m loom.agents.cli pipeline
  python -m loom.agents.cli drafts
  python -m loom.agents.cli outreach --file vcs.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def cmd_agent(args: argparse.Namespace) -> None:
    """Run an agent with a task."""
    from loom.agents.workspace import SharedWorkspace
    from loom.store import LayeredGraphStore

    db_path = Path(args.db_path)
    store = LayeredGraphStore(str(db_path))
    workspace = SharedWorkspace(args.workspace)

    task = " ".join(args.task)
    if not task:
        print("Error: provide a task for the agent")
        sys.exit(1)

    # Build context from workspace state
    pipeline = workspace.pipeline_summary()
    messages = workspace.get_messages(args.role, "pending")
    context_parts = []
    if pipeline:
        context_parts.append(f"Pipeline: {json.dumps(pipeline)}")
    if messages:
        context_parts.append(f"Pending messages ({len(messages)}):")
        for m in messages[:5]:
            context_parts.append(f"  [{m['msg_type']}] {m['subject']}: {m['content'][:200]}")
    context = "\n".join(context_parts)

    if args.role == "cto":
        from loom.agents.cto_agent import create_cto_agent
        agent = create_cto_agent(store, workspace)
    else:
        from loom.agents.ceo_agent import create_ceo_agent
        agent = create_ceo_agent(store, workspace)

    print(f"Running {args.role.upper()} agent...\n")
    result = agent.run(task, context=context)
    print(result)

    store.close()
    workspace.close()


def cmd_research(args: argparse.Namespace) -> None:
    """Research a VC and add to pipeline."""
    from loom.agents.outreach import research_vc, qualify_and_route
    from loom.agents.workspace import SharedWorkspace
    from loom.llm.gemini import GeminiClient
    from loom.store import LayeredGraphStore

    store = LayeredGraphStore(str(args.db_path))
    workspace = SharedWorkspace(args.workspace)
    llm = GeminiClient()

    intro_by = getattr(args, "introduced_by", "") or ""
    intro_agent = getattr(args, "introducer_agent", "") or ""

    json_mode = getattr(args, "json_output", False)

    if not json_mode:
        print(f"Researching {args.name} ({args.firm})...")
        if intro_by:
            print(f"  Introduced by: {intro_by}" + (f" ({intro_agent}'s contact)" if intro_agent else ""))

    profile = research_vc(llm, args.name, args.firm)
    profile.introduced_by = intro_by
    profile.introducer_agent = intro_agent

    if not json_mode:
        print(f"\nProfile:")
        print(f"  Title: {profile.title}")
        print(f"  Background: {profile.background[:200]}")
        print(f"  Thesis: {profile.thesis[:200]}")
        print(f"  Portfolio: {', '.join(profile.portfolio[:5])}")
        print(f"\nQualifying and routing...")

    result = qualify_and_route(profile, store, workspace)

    if json_mode:
        out = {
            "investor_id": result["investor_id"],
            "name": args.name,
            "firm": args.firm,
            "title": profile.title,
            "focus": result["focus"],
            "owner": result["owner"],
            "tier": result["tier"],
            "pitch_angle": result["pitch_angle"],
            "warm_paths": [
                {"type": wp["type"], "strength": wp["strength"],
                 "context": wp.get("context", wp.get("person_name", ""))}
                for wp in result["warm_paths"][:5]
            ],
            "score": result["score"],
        }
        print(json.dumps(out))
    else:
        score = result["score"]
        if score.get("intro_override"):
            routing_reason = f"INTRO OVERRIDE — {profile.introduced_by}'s relationship"
        elif result["focus"] == "tech_forward":
            routing_reason = f"tech-forward VC (tech={score['tech_score']:.0f} > biz={score['business_score']:.0f})"
        elif result["focus"] == "business_forward":
            routing_reason = f"business-forward VC (biz={score['business_score']:.0f} > tech={score['tech_score']:.0f})"
        else:
            routing_reason = f"generalist (tech={score['tech_score']:.0f}, biz={score['business_score']:.0f}) → CEO default"

        print(f"\nResult:")
        print(f"  Investor ID: {result['investor_id']}")
        print(f"  Focus: {result['focus']}")
        print(f"  Assigned to: {result['owner'].upper()} — {routing_reason}")
        print(f"  Tier: {result['tier']}")
        print(f"  Pitch angle: {result['pitch_angle']}")
        print(f"  Warm paths: {len(result['warm_paths'])}")
        for wp in result["warm_paths"][:3]:
            print(f"    - [{wp['strength']}] {wp['type']}: {wp.get('context', wp.get('person_name', ''))}")

    store.close()
    workspace.close()


def cmd_outreach(args: argparse.Namespace) -> None:
    """Run outreach batch from a JSON file."""
    from loom.agents.outreach import run_outreach_batch
    from loom.agents.workspace import SharedWorkspace
    from loom.llm.gemini import GeminiClient
    from loom.store import LayeredGraphStore

    store = LayeredGraphStore(str(args.db_path))
    workspace = SharedWorkspace(args.workspace)
    llm = GeminiClient()

    # Load VC list
    vc_file = Path(args.file)
    if not vc_file.exists():
        print(f"Error: {vc_file} not found")
        sys.exit(1)

    vcs = json.loads(vc_file.read_text())
    if not isinstance(vcs, list):
        print("Error: JSON file must contain a list of {name, firm} objects")
        sys.exit(1)

    print(f"Processing {len(vcs)} VCs...")
    results = run_outreach_batch(
        vcs, store, workspace, llm,
        dry_run=args.dry_run,
    )

    print(f"\nOutreach Batch Results:")
    print(f"  Total: {results['total']}")
    print(f"  Researched: {results['researched']}")
    print(f"  Qualified: {results['qualified']}")
    print(f"  Routed to CTO: {results['routed_cto']}")
    print(f"  Routed to CEO: {results['routed_ceo']}")
    print(f"  Warm paths found: {results['warm_paths_found']}")
    if not args.dry_run:
        print(f"  Drafts created: {results['drafts_created']}")

    print(f"\nInvestor Details:")
    for inv in results["investors"]:
        owner_label = "CTO (tech-forward)" if inv["owner"] == "cto" else "CEO (business)"
        print(f"  {inv['name']} ({inv['firm']})")
        print(f"    → {owner_label} | tier={inv['tier']} | {inv['warm_paths']} warm paths")
        print(f"    Angle: {inv['pitch_angle'][:100]}")

    store.close()
    workspace.close()


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Show investor pipeline status."""
    from loom.agents.workspace import SharedWorkspace

    workspace = SharedWorkspace(args.workspace)

    summary = workspace.pipeline_summary()
    investors = workspace.list_investors()

    if getattr(args, "json_output", False):
        result = {
            "investors": [
                {
                    "id": i["id"],
                    "name": i["name"],
                    "firm": i["firm"],
                    "stage": i["stage"],
                    "tier": i["tier"],
                    "owner": i["owner"],
                    "focus": i["focus"],
                    "introduced_by": i.get("introduced_by", ""),
                    "next_step": i.get("next_step", ""),
                    "updated_at": i["updated_at"],
                }
                for i in investors
            ],
            "summary": summary,
        }
        print(json.dumps(result))
        workspace.close()
        return

    print("Investor Pipeline")
    print("=" * 60)

    if summary:
        for stage, count in sorted(summary.items()):
            bar = "█" * count
            print(f"  {stage:20s} {bar} {count}")
    else:
        print("  (empty)")

    print(f"\nTotal: {len(investors)}")

    # Show by owner
    cto_investors = [i for i in investors if i["owner"] == "cto"]
    ceo_investors = [i for i in investors if i["owner"] == "ceo"]

    if cto_investors:
        print(f"\nCTO owned ({len(cto_investors)}):")
        for i in cto_investors:
            print(f"  {i['name']:20s} {i['firm']:20s} [{i['stage']}]")

    if ceo_investors:
        print(f"\nCEO owned ({len(ceo_investors)}):")
        for i in ceo_investors:
            print(f"  {i['name']:20s} {i['firm']:20s} [{i['stage']}]")

    workspace.close()


def cmd_drafts(args: argparse.Namespace) -> None:
    """Review and approve/reject outreach drafts."""
    from loom.agents.outreach import review_drafts, approve_draft, reject_draft
    from loom.agents.workspace import SharedWorkspace

    workspace = SharedWorkspace(args.workspace)

    agent = getattr(args, "role", "") or ""
    agents = [agent] if agent else ["cto", "ceo"]
    json_mode = getattr(args, "json_output", False)

    all_drafts = []
    for a in agents:
        drafts = review_drafts(workspace, a)
        if not drafts:
            continue

        if json_mode:
            for d in drafts:
                inv = d["investor"]
                all_drafts.append({
                    "draft_id": d["draft_id"],
                    "agent": a,
                    "investor_name": inv.get("name", ""),
                    "investor_firm": inv.get("firm", ""),
                    "subject": d["subject"],
                    "body": d["body"],
                    "channel": d["channel"],
                    "created_at": d["created_at"],
                })
        else:
            print(f"\n{a.upper()} Drafts ({len(drafts)}):")
            print("=" * 60)

            for d in drafts:
                inv = d["investor"]
                print(f"\nDraft ID: {d['draft_id']}")
                print(f"To: {inv.get('name', '?')} ({inv.get('firm', '?')})")
                print(f"Subject: {d['subject']}")
                print(f"---")
                print(d['body'])
                print(f"---")

                if not getattr(args, "auto_approve", False):
                    choice = input("\n[a]pprove / [r]eject / [s]kip? ").strip().lower()
                    if choice == "a":
                        approve_draft(workspace, d["draft_id"])
                        print("Approved")
                    elif choice == "r":
                        reject_draft(workspace, d["draft_id"])
                        print("Rejected")
                    else:
                        print("  Skipped")
                else:
                    approve_draft(workspace, d["draft_id"])
                    print("Auto-approved")

    if json_mode:
        print(json.dumps({"drafts": all_drafts}))

    workspace.close()


def cmd_approve(args: argparse.Namespace) -> None:
    """Approve a draft by ID."""
    from loom.agents.outreach import approve_draft
    from loom.agents.workspace import SharedWorkspace

    workspace = SharedWorkspace(args.workspace)
    result = approve_draft(workspace, args.draft_id)

    if getattr(args, "json_output", False):
        print(json.dumps(result))
    else:
        print(f"Approved draft {args.draft_id}")

    workspace.close()


def cmd_reject(args: argparse.Namespace) -> None:
    """Reject a draft by ID."""
    from loom.agents.outreach import reject_draft
    from loom.agents.workspace import SharedWorkspace

    workspace = SharedWorkspace(args.workspace)
    result = reject_draft(workspace, args.draft_id)

    if getattr(args, "json_output", False):
        print(json.dumps(result))
    else:
        print(f"Rejected draft {args.draft_id}")

    workspace.close()


def cmd_intel(args: argparse.Namespace) -> None:
    """Deep VC intelligence gathering with Admiralty scoring."""
    from loom.agents.dossier import build_dossier
    from loom.llm.gemini import GeminiClient
    from loom.store import LayeredGraphStore

    store = LayeredGraphStore(str(args.db_path))
    llm = GeminiClient()

    json_mode = getattr(args, "json_output", False)

    if not json_mode:
        action = "Refreshing" if args.refresh else "Building"
        print(f"{action} dossier for {args.name} ({args.firm})...")

    result = build_dossier(
        store, llm, args.name, args.firm,
        force_refresh=args.refresh,
    )

    if json_mode:
        print(json.dumps(result, default=str))
    else:
        dossier = result.get("dossier", {})
        if result.get("is_cached"):
            print(f"\nCached dossier (belief: {result['belief_id'][:12]}...)")
        elif result.get("is_refresh"):
            print(f"\nRefreshed dossier (belief: {result['belief_id'][:12]}...)")
        else:
            print(f"\nNew dossier (belief: {result['belief_id'][:12]}...)")

        print(f"  Events stored: {result['events_stored']}")
        print(f"  Claims stored: {result['claims_stored']}")

        # Source breakdown
        breakdown = result.get("source_breakdown", {})
        if breakdown:
            print(f"\nSources:")
            for src, count in sorted(breakdown.items()):
                print(f"  {src}: {count} fragment(s)")

        # Thesis
        thesis = dossier.get("thesis", {})
        if thesis:
            summary = thesis.get("thesis_summary", {})
            if isinstance(summary, dict) and summary.get("value"):
                print(f"\nThesis (conf={summary.get('confidence', 0):.2f}, from={summary.get('from', '?')}):")
                print(f"  {summary['value'][:300]}")
            focus = thesis.get("focus_areas", {})
            if isinstance(focus, dict) and focus.get("value"):
                areas = focus["value"]
                if isinstance(areas, list):
                    print(f"  Focus: {', '.join(str(a) for a in areas[:8])}")

        # Portfolio
        portfolio = dossier.get("portfolio", {})
        if portfolio:
            investments = portfolio.get("recent_investments", {})
            if isinstance(investments, dict) and investments.get("value"):
                inv_list = investments["value"]
                if isinstance(inv_list, list):
                    print(f"\nPortfolio (conf={investments.get('confidence', 0):.2f}):")
                    for inv in inv_list[:5]:
                        if isinstance(inv, dict):
                            print(f"  {inv.get('company', '?')} — {inv.get('round', '?')} ({inv.get('date', '?')})")
                        else:
                            print(f"  {inv}")

        # Style
        style = dossier.get("style", {})
        if style:
            style_sum = style.get("style_summary", {})
            if isinstance(style_sum, dict) and style_sum.get("value"):
                print(f"\nStyle (conf={style_sum.get('confidence', 0):.2f}):")
                print(f"  {style_sum['value'][:300]}")

        # Fund status
        fund = dossier.get("fund_status", {})
        if fund:
            fund_name = fund.get("fund_name", {})
            fund_size = fund.get("fund_size", {})
            if isinstance(fund_name, dict) and fund_name.get("value"):
                print(f"\nFund: {fund_name['value']}")
            if isinstance(fund_size, dict) and fund_size.get("value"):
                print(f"  Size: {fund_size['value']}")

        # Interactions
        interactions = dossier.get("interactions", {})
        if interactions:
            count = interactions.get("interaction_count", {})
            if isinstance(count, dict) and count.get("value"):
                print(f"\nKG Interactions: {count['value']} direct touches")

    store.close()


def cmd_sync(args: argparse.Namespace) -> None:
    """Discover, populate, research active VCs, and recommend next actions."""
    from loom.agents.discover import discover_vcs, add_discovered_to_pipeline
    from loom.agents.recommend import recommend_actions
    from loom.agents.workspace import SharedWorkspace
    from loom.store import LayeredGraphStore

    store = LayeredGraphStore(str(args.db_path))
    workspace = SharedWorkspace(args.workspace)

    json_mode = getattr(args, "json_output", False)
    days = getattr(args, "days", 180)
    do_research = getattr(args, "research", False)
    use_deep = getattr(args, "deep", False)

    # Step 1: Discover VCs from the knowledge graph
    if not json_mode:
        print(f"Step 1: Scanning knowledge graph (last {days} days)...")

    result = discover_vcs(store, workspace, days=days)
    discovered = result["discovered"]
    summary = result["summary"]

    if not json_mode:
        print(f"  Found {summary['total_discovered']} VCs "
              f"({summary['new']} new, {summary['already_tracked']} tracked)")

    # Step 2: Add new VCs to workspace with enriched data
    new_items = [d for d in discovered if not d["in_workspace"]]
    new_pids = [d["person_id"] for d in new_items]

    added = []
    if new_pids:
        if not json_mode:
            print(f"\nStep 2: Adding {len(new_pids)} new VCs to pipeline...")
        added = add_discovered_to_pipeline(
            store, workspace, new_pids,
            discover_results=discovered,
        )
        if not json_mode:
            for a in added:
                action = a.get("action", "added")
                print(f"  {action}: {a['name']} ({a['firm']}) [{a.get('stage', 'discovered')}]")
    elif not json_mode:
        print("\nStep 2: No new VCs to add")

    # Step 3: Auto-research active VCs (optional)
    researched = []
    if do_research:
        from loom.agents.kg_tools import KGTools
        from loom.agents.outreach import research_vc
        from loom.agents.protocol import build_pitch_angle, score_vc_focus
        from loom.llm.gemini import GeminiClient

        llm = GeminiClient()
        kg = KGTools(store)
        active_stages = {"meeting_scheduled", "contacted", "follow_up", "in_progress", "discovered"}
        investors = workspace.list_investors()

        active = [
            inv for inv in investors
            if inv["stage"] in active_stages
            and (not inv.get("focus") or inv.get("focus") == "generalist")
        ]

        if active:
            if not json_mode:
                print(f"\nStep 3: Researching {len(active)} active VCs...")
            for inv in active:
                name = inv["name"]
                firm = inv["firm"]
                if not json_mode:
                    print(f"  Researching {name} ({firm})...")
                try:
                    profile = research_vc(llm, name, firm, store=store if use_deep else None, deep=use_deep)
                    score = score_vc_focus(profile)
                    owner = score["recommended_owner"]
                    warm_paths = kg.find_warm_paths(name, firm)
                    tier = inv["tier"]
                    if any(p["strength"] == "strong" for p in warm_paths):
                        tier = "tier1"
                    elif any(p["strength"] == "medium" for p in warm_paths):
                        tier = "tier2"

                    # Update existing entry — no duplicate created
                    workspace.update_investor(
                        inv["id"],
                        focus=score["focus"],
                        owner=owner,
                        tier=tier,
                        stage="qualified",
                        thesis=profile.thesis,
                        warm_path=[
                            f"{p['type']}: {p.get('context', p.get('person_name', ''))}"
                            for p in warm_paths[:5]
                        ],
                    )
                    researched.append({
                        "investor_id": inv["id"],
                        "name": name,
                        "firm": firm,
                        "focus": score["focus"],
                        "owner": owner,
                        "tier": tier,
                    })
                    if not json_mode:
                        print(f"    → {score['focus']} | {owner} | {tier}")
                except Exception as e:
                    logger.warning("Failed to research %s: %s", name, e)
                    if not json_mode:
                        print(f"    → Error: {e}")
        elif not json_mode:
            print("\nStep 3: No active VCs need research")
    elif not json_mode:
        print("\nStep 3: Skipped (use --research to auto-research)")

    # Step 4: Generate recommendations
    if not json_mode:
        print("\nStep 4: Generating recommendations...")

    recs = recommend_actions(workspace)

    if json_mode:
        output = {
            "discovery": summary,
            "added": added,
            "researched": researched,
            "recommendations": recs,
        }
        print(json.dumps(output))
    else:
        rec_summary = recs["summary"]
        print(f"\n{'=' * 60}")
        print(f"Pipeline: {rec_summary['total']} investors | "
              f"{rec_summary['actionable']} actionable | "
              f"{rec_summary['terminal']} terminal")

        if recs["research_needed"]:
            print(f"\nResearch Needed ({len(recs['research_needed'])}):")
            for r in recs["research_needed"][:5]:
                print(f"  {r['name']:25s} {r['firm']:20s} "
                      f"score={r['priority_score']:.0f} warm={r['warm_path_count']}")

        if recs["ready_to_contact"]:
            print(f"\nReady to Contact ({len(recs['ready_to_contact'])}):")
            for r in recs["ready_to_contact"][:5]:
                print(f"  {r['name']:25s} {r['firm']:20s} "
                      f"→ {r['owner'] or 'unassigned'} | {r['tier']}")

        if recs["follow_up_needed"]:
            print(f"\nFollow Up Needed ({len(recs['follow_up_needed'])}):")
            for r in recs["follow_up_needed"][:5]:
                print(f"  {r['name']:25s} {r['firm']:20s} [{r['stage']}]")

        if recs["re_engage"]:
            print(f"\nRe-engage ({len(recs['re_engage'])}):")
            for r in recs["re_engage"][:5]:
                print(f"  {r['name']:25s} {r['firm']:20s} warm={r['warm_path_count']}")

        if recs["passed"]:
            print(f"\nPassed/Ghosted ({len(recs['passed'])}):")
            for r in recs["passed"][:5]:
                reason = f" — {r['pass_reason'][:60]}" if r.get("pass_reason") else ""
                print(f"  {r['name']:25s} {r['firm']:20s} [{r['stage']}]{reason}")

    store.close()
    workspace.close()


def cmd_prospect(args: argparse.Namespace) -> None:
    """Mine KG and optionally web for new VC prospects."""
    from loom.agents.prospect import discover_prospects
    from loom.agents.workspace import SharedWorkspace
    from loom.store import LayeredGraphStore

    store = LayeredGraphStore(str(args.db_path))
    workspace = SharedWorkspace(args.workspace)

    json_mode = getattr(args, "json_output", False)
    days = getattr(args, "days", 180)
    verticals = getattr(args, "verticals", "")
    use_web = getattr(args, "web", False)

    vert_list = [v.strip() for v in verticals.split(",") if v.strip()] if verticals else [
        "AI infrastructure", "personal AI", "privacy-first", "developer tools",
    ]

    llm = None
    if use_web:
        from loom.llm.gemini import GeminiClient
        llm = GeminiClient()

    if not json_mode:
        print(f"Mining knowledge graph for prospects (last {days} days)...")
        if use_web:
            print(f"Web search enabled for: {', '.join(vert_list)}")

    result = discover_prospects(
        store, workspace,
        llm_client=llm,
        verticals=vert_list,
        days=days,
    )

    if json_mode:
        print(json.dumps(result, default=str))
    else:
        summary = result["summary"]
        print(f"\nFound {summary['total']} prospects "
              f"({summary['kg_found']} from KG, {summary['web_found']} from web)")
        print(f"  With warm paths: {summary['with_warm_paths']}")
        print(f"  With intro offers: {summary['with_intros']}")

        if result["combined"]:
            # Group by source type
            by_source: dict[str, list] = {}
            for p in result["combined"]:
                src = p["source_type"]
                by_source.setdefault(src, []).append(p)

            for src, prospects in by_source.items():
                label = {
                    "kg_intro_offer": "Intro Offers (someone offered to connect you)",
                    "kg_vc_contact": "VC Contacts (people at VC firms in your network)",
                    "kg_fundraising_mention": "Fundraising Mentions (firms discussed in context)",
                    "kg_investor_meeting": "Investor Meetings (calendar with VC keywords)",
                    "web_search": "Web Search (current active investors)",
                }.get(src, src)

                print(f"\n{label} ({len(prospects)}):")
                for p in prospects[:10]:
                    name = p.get("name", "?")
                    firm = p.get("firm", "?")
                    conf = p.get("confidence", 0)
                    warm = len(p.get("warm_paths", []))
                    prov = p.get("provenance", [{}])[0]

                    print(f"  {name:25s} {firm:20s} conf={conf:.0f} warm={warm}")
                    # Show provenance
                    if prov.get("snippet"):
                        print(f"    └ \"{prov['snippet'][:80]}\"")
                    elif prov.get("from"):
                        print(f"    └ from: {prov['from']}")
                    elif prov.get("email"):
                        print(f"    └ {prov['email']}")
        else:
            print("\nNo new prospects found.")

    store.close()
    workspace.close()


def cmd_discover(args: argparse.Namespace) -> None:
    """Scan the knowledge graph to discover VCs."""
    from loom.agents.discover import discover_vcs, add_discovered_to_pipeline
    from loom.agents.workspace import SharedWorkspace
    from loom.store import LayeredGraphStore

    store = LayeredGraphStore(str(args.db_path))
    workspace = SharedWorkspace(args.workspace)

    json_mode = getattr(args, "json_output", False)
    days = getattr(args, "days", 180)
    add_all = getattr(args, "add_all", False)

    if not json_mode:
        print(f"Scanning knowledge graph for VCs (last {days} days)...")

    result = discover_vcs(store, workspace, days=days)
    discovered = result["discovered"]
    summary = result["summary"]

    if json_mode:
        print(json.dumps(result))
    else:
        print(f"\nDiscovered {summary['total_discovered']} potential VCs")
        print(f"  New (not tracked): {summary['new']}")
        print(f"  Already in pipeline: {summary['already_tracked']}")

        if summary["by_stage"]:
            print(f"\n  By stage:")
            for stage, count in sorted(summary["by_stage"].items()):
                print(f"    {stage:20s} {count}")

        if discovered:
            print(f"\nTop VCs:")
            for d in discovered[:15]:
                tracked = " [TRACKED]" if d["in_workspace"] else ""
                warm = f" | {d['warm_paths']} warm" if d["warm_paths"] else ""
                print(f"  {d['name']:25s} {d['firm']:20s} [{d['stage']}]"
                      f" conf={d['confidence']:.1f}{warm}{tracked}")
        else:
            print("\nNo VCs found. Run the pipeline first to populate the knowledge graph.")

    # Optionally add all new to pipeline
    if add_all:
        new_pids = [d["person_id"] for d in discovered if not d["in_workspace"]]
        if new_pids:
            added = add_discovered_to_pipeline(store, workspace, new_pids, discover_results=discovered)
            if json_mode:
                print(json.dumps({"added": added}))
            else:
                print(f"\nAdded {len(added)} VCs to pipeline")

    store.close()
    workspace.close()


def cmd_messages(args: argparse.Namespace) -> None:
    """Show cross-agent messages."""
    from loom.agents.workspace import SharedWorkspace

    workspace = SharedWorkspace(args.workspace)

    for agent in ["cto", "ceo"]:
        msgs = workspace.get_messages(agent, status="")
        if not msgs:
            continue

        print(f"\nMessages for {agent.upper()}:")
        for m in msgs:
            status_icon = {"pending": "●", "read": "○", "answered": "✓"}.get(m["status"], "?")
            print(f"  {status_icon} [{m['msg_type']}] from={m['from_agent']} — {m['subject']}")
            content = m["content"][:200]
            print(f"    {content}")
            if m.get("answer"):
                print(f"    → {m['answer'][:200]}")

    workspace.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Loom Agents — autonomous VC outreach with privacy-preserving coordination",
    )
    parser.add_argument(
        "--db", dest="db_path",
        default=str(Path.home() / ".loom" / "graph.db"),
        help="Path to knowledge graph database",
    )
    parser.add_argument(
        "--workspace", default=str(Path.home() / ".loom" / "workspace.db"),
        help="Path to shared workspace database",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # Agent commands
    p_cto = sub.add_parser("cto", help="Run CTO agent with a task")
    p_cto.add_argument("task", nargs="+", help="Task description")
    p_cto.set_defaults(func=cmd_agent, role="cto")

    p_ceo = sub.add_parser("ceo", help="Run CEO agent with a task")
    p_ceo.add_argument("task", nargs="+", help="Task description")
    p_ceo.set_defaults(func=cmd_agent, role="ceo")

    # Research
    p_res = sub.add_parser("research", help="Research a VC and add to pipeline")
    p_res.add_argument("--name", required=True, help="VC name")
    p_res.add_argument("--firm", required=True, help="Firm name")
    p_res.add_argument("--introduced-by", dest="introduced_by", default="",
                        help="Who made the introduction (e.g., 'Sid')")
    p_res.add_argument("--introducer-agent", dest="introducer_agent", default="",
                        choices=["cto", "ceo"],
                        help="Whose contact is the introducer (auto-detected if omitted)")
    p_res.add_argument("--json", dest="json_output", action="store_true",
                        help="Output as JSON (for app integration)")
    p_res.set_defaults(func=cmd_research)

    # Batch outreach
    p_out = sub.add_parser("outreach", help="Run outreach batch from JSON file")
    p_out.add_argument("--file", required=True, help="JSON file with VC list")
    p_out.add_argument("--dry-run", action="store_true", help="Research and route only")
    p_out.set_defaults(func=cmd_outreach)

    # Pipeline
    p_pipe = sub.add_parser("pipeline", help="Show investor pipeline")
    p_pipe.add_argument("--json", dest="json_output", action="store_true",
                         help="Output as JSON (for app integration)")
    p_pipe.set_defaults(func=cmd_pipeline)

    # Drafts
    p_drafts = sub.add_parser("drafts", help="Review outreach drafts")
    p_drafts.add_argument("--role", choices=["cto", "ceo"], help="Filter by agent")
    p_drafts.add_argument("--auto-approve", action="store_true", help="Approve all drafts")
    p_drafts.add_argument("--json", dest="json_output", action="store_true",
                           help="Output as JSON (for app integration)")
    p_drafts.set_defaults(func=cmd_drafts)

    # Approve a draft
    p_approve = sub.add_parser("approve", help="Approve a draft by ID")
    p_approve.add_argument("draft_id", help="Draft ID to approve")
    p_approve.add_argument("--json", dest="json_output", action="store_true",
                            help="Output as JSON")
    p_approve.set_defaults(func=cmd_approve)

    # Reject a draft
    p_reject = sub.add_parser("reject", help="Reject a draft by ID")
    p_reject.add_argument("draft_id", help="Draft ID to reject")
    p_reject.add_argument("--json", dest="json_output", action="store_true",
                           help="Output as JSON")
    p_reject.set_defaults(func=cmd_reject)

    # Discover
    p_discover = sub.add_parser("discover", help="Scan KG to discover VCs")
    p_discover.add_argument("--days", type=int, default=180,
                             help="How many days back to scan (default: 180)")
    p_discover.add_argument("--add-all", action="store_true",
                             help="Add all discovered VCs to the pipeline")
    p_discover.add_argument("--json", dest="json_output", action="store_true",
                             help="Output as JSON (for app integration)")
    p_discover.set_defaults(func=cmd_discover)

    # Prospect discovery — mine KG + web for new leads
    p_prospect = sub.add_parser("prospect", help="Mine KG and web for new VC prospects")
    p_prospect.add_argument("--days", type=int, default=180,
                             help="How many days back to scan (default: 180)")
    p_prospect.add_argument("--verticals", default="",
                             help="Comma-separated verticals (e.g. 'AI infrastructure,personal AI')")
    p_prospect.add_argument("--web", action="store_true",
                             help="Also search the web via Gemini (costs API credits)")
    p_prospect.add_argument("--json", dest="json_output", action="store_true",
                             help="Output as JSON")
    p_prospect.set_defaults(func=cmd_prospect)

    # Intel — deep VC intelligence gathering
    p_intel = sub.add_parser("intel", help="Deep VC intelligence gathering")
    p_intel.add_argument("--name", required=True, help="VC name")
    p_intel.add_argument("--firm", required=True, help="Firm name")
    p_intel.add_argument("--refresh", action="store_true",
                          help="Force refresh even if cached")
    p_intel.add_argument("--json", dest="json_output", action="store_true",
                          help="Output as JSON")
    p_intel.set_defaults(func=cmd_intel)

    # Sync — full pipeline in one shot
    p_sync = sub.add_parser("sync", help="Discover, populate, research, and recommend")
    p_sync.add_argument("--days", type=int, default=180,
                         help="How many days back to scan (default: 180)")
    p_sync.add_argument("--research", action="store_true",
                         help="Auto-research active VCs via Gemini")
    p_sync.add_argument("--deep", action="store_true",
                         help="Use deep dossier research instead of single-query")
    p_sync.add_argument("--json", dest="json_output", action="store_true",
                         help="Output as JSON (for app integration)")
    p_sync.set_defaults(func=cmd_sync)

    # Messages
    p_msgs = sub.add_parser("messages", help="Show cross-agent messages")
    p_msgs.set_defaults(func=cmd_messages)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
