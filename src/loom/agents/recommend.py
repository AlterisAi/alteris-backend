"""VC pipeline recommendations — rank and prioritize investor outreach."""

from __future__ import annotations

import time
from typing import Any

from loom.agents.workspace import SharedWorkspace

# Terminal stages — no action needed
TERMINAL_STAGES = {"pass", "ghosted", "committed", "term_sheet"}

# Active stages that mean outreach is already in flight
ACTIVE_STAGES = {"meeting_scheduled", "contacted", "replied", "follow_up",
                 "met", "due_diligence", "in_progress"}

# Tier scoring
TIER_SCORES = {"tier1": 3, "tier2": 2, "tier3": 1}

# Staleness thresholds (seconds)
STALE_THRESHOLD = 14 * 86400    # 14 days with no update = stale
RECENT_30D = 30 * 86400
RECENT_60D = 60 * 86400


def recommend_actions(workspace: SharedWorkspace, limit: int = 10) -> dict[str, Any]:
    """Analyze the pipeline and produce ranked recommendations.

    Groups investors by action needed and scores them for prioritization.

    Returns:
        {
            "research_needed": [...],    # In workspace but no research done
            "ready_to_contact": [...],   # Researched + qualified, not contacted
            "follow_up_needed": [...],   # Contacted/replied but gone stale
            "re_engage": [...],          # Inactive/dormant but have warm paths
            "passed": [...],             # Terminal: passed or ghosted
            "summary": {
                "total": int,
                "actionable": int,
                "terminal": int,
                "by_action": {str: int},
            }
        }
    """
    investors = workspace.list_investors()
    now = int(time.time())

    research_needed: list[dict] = []
    ready_to_contact: list[dict] = []
    follow_up_needed: list[dict] = []
    re_engage: list[dict] = []
    passed: list[dict] = []

    for inv in investors:
        stage = inv["stage"]
        entry = _score_investor(inv, now)

        if stage in TERMINAL_STAGES:
            passed.append(entry)
            continue

        if stage in ("discovered", "researched"):
            # Not yet qualified — needs research
            if stage == "discovered":
                research_needed.append(entry)
            else:
                # Researched but not qualified or contacted
                if inv.get("focus") and inv["focus"] != "generalist":
                    ready_to_contact.append(entry)
                else:
                    research_needed.append(entry)
            continue

        if stage in ("qualified", "warm_intro_found"):
            ready_to_contact.append(entry)
            continue

        if stage in ACTIVE_STAGES:
            last_update = inv.get("updated_at") or inv.get("last_contact_at") or 0
            age = now - last_update if last_update else float("inf")
            if age > STALE_THRESHOLD:
                follow_up_needed.append(entry)
            continue

        # Anything else (shouldn't happen with valid pipeline stages)
        # but catch discovered entries that have gone stale
        if entry["warm_path_count"] > 0 or entry["tier_score"] >= 2:
            re_engage.append(entry)

    # Sort each group by priority score descending
    for group in [research_needed, ready_to_contact, follow_up_needed, re_engage]:
        group.sort(key=lambda x: -x["priority_score"])

    # Apply limit
    research_needed = research_needed[:limit]
    ready_to_contact = ready_to_contact[:limit]
    follow_up_needed = follow_up_needed[:limit]
    re_engage = re_engage[:limit]

    actionable = len(research_needed) + len(ready_to_contact) + len(follow_up_needed) + len(re_engage)

    return {
        "research_needed": research_needed,
        "ready_to_contact": ready_to_contact,
        "follow_up_needed": follow_up_needed,
        "re_engage": re_engage,
        "passed": passed,
        "summary": {
            "total": len(investors),
            "actionable": actionable,
            "terminal": len(passed),
            "by_action": {
                "research_needed": len(research_needed),
                "ready_to_contact": len(ready_to_contact),
                "follow_up_needed": len(follow_up_needed),
                "re_engage": len(re_engage),
            },
        },
    }


def _score_investor(inv: dict, now: int) -> dict:
    """Score an investor for prioritization.

    Priority = warm_paths * 3 + tier_score + recency_score + confidence_bonus
    """
    # Parse warm_path to count entries
    warm_path = inv.get("warm_path", "[]")
    if isinstance(warm_path, str):
        try:
            import json
            warm_list = json.loads(warm_path)
        except (ValueError, TypeError):
            warm_list = []
    elif isinstance(warm_path, list):
        warm_list = warm_path
    else:
        warm_list = []

    warm_count = len(warm_list)
    tier_score = TIER_SCORES.get(inv.get("tier", "tier3"), 1)

    # Recency score based on last contact
    last_contact = inv.get("last_contact_at") or inv.get("updated_at") or 0
    recency_age = now - last_contact if last_contact else float("inf")
    if recency_age < RECENT_30D:
        recency_score = 2
    elif recency_age < RECENT_60D:
        recency_score = 1
    else:
        recency_score = 0

    priority_score = warm_count * 3 + tier_score + recency_score

    return {
        "investor_id": inv["id"],
        "name": inv["name"],
        "firm": inv["firm"],
        "stage": inv["stage"],
        "tier": inv.get("tier", "tier3"),
        "owner": inv.get("owner", ""),
        "focus": inv.get("focus", ""),
        "warm_path_count": warm_count,
        "tier_score": tier_score,
        "recency_score": recency_score,
        "priority_score": round(priority_score, 1),
        "pass_reason": inv.get("pass_reason", ""),
        "last_contact_at": last_contact,
        "next_step": inv.get("next_step", ""),
    }
