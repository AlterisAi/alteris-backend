"""CQ Story tools: clustering, feed, pulse, and story management.

Stories group related CQ tasks using LLM-powered clustering (Gemini Flash
Lite) with a person-based fallback. DB caching avoids redundant LLM calls.
Manual overrides (anti-links, splits) persist across re-clusters.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid

from alteris.mcp_tools import ToolDef, ToolParam, register_tool
from alteris.mcp_tools.cq_tools import _beliefs_to_tasks, _commitments_to_tasks
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _stable_story_id(member_ids: list[str]) -> str:
    """Deterministic story ID from sorted member task IDs."""
    key = "|".join(sorted(member_ids))
    return "story_" + hashlib.sha256(key.encode()).hexdigest()[:12]


def _compute_cluster_hash(task_ids: list[str]) -> str:
    """SHA-256 hex of sorted task IDs — used to detect when re-clustering is needed."""
    key = "|".join(sorted(task_ids))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _get_person_context(store: LayeredGraphStore) -> dict[str, dict]:
    """Build compact person context: tier, channels, open commitment count."""
    result: dict[str, dict] = {}
    profiles = store.get_person_profiles()
    for p in profiles:
        if p.get("is_user"):
            continue
        name = p.get("canonical_name", "")
        if not name:
            continue
        # Count open commitments for this person
        open_count = 0
        rows = store.conn.execute(
            """SELECT COUNT(*) FROM claims c
               JOIN claim_events ce ON c.id = ce.claim_id
               JOIN person_events pe ON ce.event_id = pe.event_id
               WHERE pe.person_id = ? AND c.claim_type = 'commitment'
                 AND c.superseded_by IS NULL""",
            (p["person_id"],),
        ).fetchone()
        if rows:
            open_count = rows[0]
        result[name] = {
            "tier": p.get("tier", 4),
            "channels": p.get("channels", []),
            "open_commitments": open_count,
        }
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM Clustering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_CLUSTERING_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "clusters": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "story_title": {
                        "type": "STRING",
                        "description": "3-7 word imperative action title for the story.",
                    },
                    "rationale": {
                        "type": "STRING",
                        "description": "Why these tasks share a single project context.",
                    },
                    "semantic_anchors": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "Core noun/verb pairs defining the project.",
                    },
                    "task_ids": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                },
                "required": ["story_title", "rationale", "semantic_anchors", "task_ids"],
            },
        },
    },
    "required": ["clusters"],
}

_INCREMENTAL_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "assignments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "task_id": {"type": "STRING"},
                    "story_title": {"type": "STRING"},
                    "is_new_story": {"type": "BOOLEAN"},
                    "rationale": {"type": "STRING"},
                },
                "required": ["task_id", "story_title", "is_new_story"],
            },
        },
    },
    "required": ["assignments"],
}

# Phase 1: Structural clustering — groups tasks by semantic identity, not just entity.
_CLUSTERING_SYSTEM_PROMPT = """\
# OBJECTIVE
Group a list of incoming communication tasks into "Stories." A Story is defined \
as a single, unified project or operational goal. Then title each story.

# CLUSTERING CRITERIA
1. SEMANTIC IDENTITY: Only group items that share the same "Reason for Existence."
2. ENTITY NEUTRALITY: The presence of the same person (e.g., "Sam") is NOT a \
sufficient reason to merge. If the topics are disjointed (e.g., "Work API" vs \
"Lunch"), they must remain in separate clusters.
3. GRANULARITY: Prefer two accurate clusters over one "muddy" cluster. If you \
have to explain a connection using the word "and" to bridge two different topics, \
split them.

# PROCESSING STEPS
For each potential cluster, follow this sequence:
1. Extract "Semantic Anchors": Identify the core nouns and verbs that define the intent.
2. Filter Entities: Explicitly check if the link is just a person's name. If yes, reject the merge.
3. Formulate Rationale: Explain why these tasks share a single, non-entity-based project context.
4. Map IDs: Assign the input task IDs to the cluster.

# STORY TITLE RULES
1. STRICT IMPERATIVE: Must start with a strong action verb (Review, Draft, \
Schedule, Finalize, Prepare, Coordinate, Resolve, Complete, etc.).
2. LENGTH: Strictly 3 to 7 words.
3. AGGREGATE: Synthesize the overall goal — don't copy the first task's title.
4. ENTITY INCLUSION: Include key person/org contextually if consistently involved \
(e.g., "Review Alteris specs with Sam" not just "Review specs").
5. FORBIDDEN: Never use "Email thread", "Catch up", "Task", "Discussion", \
"Miscellaneous". Never use a person's name as the entire title.

# FEW-SHOT EXAMPLES

## Clustering — Split vs. Merge signal:

Input tasks: "Review Q4 hiring plan" (Jane Doe, HR), "Interview candidate for \
Eng Lead" (Jane Doe, Recruiting), "Buy Jane Doe a birthday gift" (Jane Doe, Personal)
->
Cluster 1: ["Review Q4 hiring plan", "Interview candidate for Eng Lead"]
  rationale: "Items share the semantic anchor of Talent/Hiring — both contribute \
to team expansion. Birthday gift excluded despite sharing entity Jane Doe."
  title: "Execute Q4 hiring pipeline"
Cluster 2: ["Buy Jane Doe a birthday gift"]
  rationale: "Standalone personal logistics. No semantic overlap with hiring."
  title: "Purchase birthday gift for Jane"

## Title examples:
- Scheduling calendar blocks and calls with Alec -> "Schedule Meta interviews with Alec"
- Monitoring child for HFMD symptoms -> "Monitor child for HFMD symptoms"
- Scheduling demo sessions with Josie -> "Schedule Mac app demo for Josie"
- Paying Bank of America credit card bills -> "Resolve BofA credit card payments"
- Submitting interview forms for Elaine Wu at Meta -> "Complete Meta interview prep for Elaine"
"""


def _llm_cluster(store: LayeredGraphStore, tasks: list[dict]) -> list[dict] | None:
    """Cluster tasks into stories via Gemini Flash 3 (no thinking).

    Uses structural clustering (semantic anchors, not just entity overlap)
    and generates imperative story titles in one pass.

    Returns list of {"story_title": str, "task_ids": [str]} or None on failure.
    """
    from alteris.constants import CQ_CLUSTERING_MAX_TASKS, CQ_CLUSTERING_MODEL

    if len(tasks) > CQ_CLUSTERING_MAX_TASKS:
        tasks = sorted(tasks, key=lambda t: t.get("priority", 99))[:CQ_CLUSTERING_MAX_TASKS]

    # Build JSON task summaries with rich context for semantic clustering
    task_summaries = []
    for t in tasks:
        summary: dict = {
            "task_id": t["id"][:16],
            "title": t.get("title", "")[:80],
        }
        obj_data = t.get("_obj_data", {})
        if t.get("to_whom"):
            summary["to_whom"] = t["to_whom"]
        if t.get("who"):
            summary["who"] = t["who"]
        if t.get("commitment_type"):
            summary["commitment_type"] = t["commitment_type"]
        if t.get("due_date"):
            summary["due_date"] = t["due_date"]
        if obj_data.get("next_action"):
            summary["next_action"] = str(obj_data["next_action"])[:100]
        if obj_data.get("note"):
            summary["note"] = str(obj_data["note"])[:100]
        if t.get("source_channel"):
            summary["source_channel"] = t["source_channel"]
        task_summaries.append(summary)

    prompt = "<tasks>\n" + json.dumps(task_summaries, indent=2) + "\n</tasks>"

    try:
        from alteris.llm.gemini import GeminiClient
        llm = GeminiClient(store=store)
        raw = llm.generate(
            prompt=prompt,
            system=_CLUSTERING_SYSTEM_PROMPT,
            model=CQ_CLUSTERING_MODEL,
            temperature=0.1,
            max_tokens=4096,
            response_schema=_CLUSTERING_RESPONSE_SCHEMA,
            thinking_budget=0,
            cache_system=True,
        )
        if not raw:
            return None
        parsed = json.loads(raw)
        clusters = parsed.get("clusters", []) if isinstance(parsed, dict) else []
        if not clusters:
            return None
        # Expand truncated task IDs back to full IDs
        id_map = {t["id"][:16]: t["id"] for t in tasks}
        result = []
        for c in clusters:
            expanded = []
            for tid in c.get("task_ids", []):
                full = id_map.get(tid[:16], tid)
                expanded.append(full)
            result.append({
                "story_title": c.get("story_title", "Untitled"),
                "task_ids": expanded,
            })
        return result
    except Exception as exc:
        logger.warning("LLM clustering failed: %s", exc)
        return None


def _llm_incremental_cluster(
    store: LayeredGraphStore,
    new_tasks: list[dict],
    existing_stories: list[dict],
) -> list[dict] | None:
    """Assign new tasks to existing stories or create new ones.

    Returns list of {"task_id": str, "story_title": str, "is_new_story": bool}
    or None on failure.
    """
    from alteris.constants import CQ_CLUSTERING_MODEL

    # Build existing story summaries
    story_summaries = []
    for s in existing_stories:
        task_titles = [t.get("title", "")[:40] for t in s.get("tasks", [])[:3]]
        story_summaries.append({
            "story_title": s["title"],
            "sample_tasks": task_titles,
        })

    # Build new task summaries
    task_summaries = []
    for t in new_tasks:
        summary: dict = {
            "task_id": t["id"][:16],
            "title": t.get("title", "")[:80],
        }
        if t.get("to_whom"):
            summary["to_whom"] = t["to_whom"]
        if t.get("commitment_type"):
            summary["commitment_type"] = t["commitment_type"]
        if t.get("due_date"):
            summary["due_date"] = t["due_date"]
        task_summaries.append(summary)

    prompt = json.dumps({
        "existing_stories": story_summaries,
        "new_tasks": task_summaries,
        "instruction": "Assign each new task to an existing story (use exact story_title) or create a new one (is_new_story=true). Apply the same clustering criteria and title rules.",
    }, indent=2)

    try:
        from alteris.llm.gemini import GeminiClient
        llm = GeminiClient(store=store)
        raw = llm.generate(
            prompt=prompt,
            system=_CLUSTERING_SYSTEM_PROMPT,
            model=CQ_CLUSTERING_MODEL,
            temperature=0.1,
            max_tokens=2048,
            response_schema=_INCREMENTAL_RESPONSE_SCHEMA,
            thinking_budget=0,
            cache_system=True,
        )
        if not raw:
            return None
        parsed = json.loads(raw)
        assignments = parsed.get("assignments", []) if isinstance(parsed, dict) else []
        if not assignments:
            return None
        # Expand truncated IDs
        id_map = {t["id"][:16]: t["id"] for t in new_tasks}
        for item in assignments:
            tid = item.get("task_id", "")
            item["task_id"] = id_map.get(tid[:16], tid)
        return assignments
    except Exception as exc:
        logger.warning("LLM incremental clustering failed: %s", exc)
        return None


def _fallback_person_cluster(tasks: list[dict]) -> list[dict]:
    """Group tasks by to_whom person name. Fallback when LLM is unavailable."""
    by_person: dict[str, list[dict]] = {}
    for t in tasks:
        person = (t.get("to_whom") or "").strip()
        if not person:
            person = "__ungrouped__"
        by_person.setdefault(person, []).append(t)

    result = []
    for person, group in by_person.items():
        title = person if person != "__ungrouped__" else "Other items"
        result.append({
            "story_title": title,
            "task_ids": [t["id"] for t in group],
        })
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DB Caching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _save_stories_to_db(
    store: LayeredGraphStore,
    groups: list[dict],
    cluster_hash: str,
):
    """Persist LLM clustering results to cq_stories + cq_story_members.

    Clears old LLM-sourced stories before saving new ones (replace, not accumulate).
    """
    # Delete old LLM stories
    old_stories = store.conn.execute(
        "SELECT id FROM cq_stories WHERE source = 'llm'"
    ).fetchall()
    for row in old_stories:
        store.delete_cq_story(row["id"])

    for group in groups:
        title = group.get("story_title", "Untitled")
        task_ids = group.get("task_ids", [])
        if not task_ids:
            continue
        story_id = _stable_story_id(task_ids)
        store.put_cq_story(
            story_id, title, source="llm", cluster_hash=cluster_hash,
        )
        for i, tid in enumerate(task_ids):
            store.add_story_member(story_id, tid, position=i)


def _rebuild_from_cache(
    store: LayeredGraphStore,
    task_map: dict[str, dict],
) -> list[dict] | None:
    """Try to rebuild stories from cached DB entries. Returns None if no cache."""
    cached = store.conn.execute(
        "SELECT id, title, cluster_hash, priority, priority_override FROM cq_stories WHERE source = 'llm' AND status = 'active'"
    ).fetchall()
    if not cached:
        return None

    stories = []
    for row in cached:
        members = store.get_story_members(row["id"])
        task_ids = [m["task_id"] for m in members]
        # Only keep tasks still in the active task map
        live_ids = [tid for tid in task_ids if tid in task_map]
        if not live_ids:
            continue
        stories.append({
            "story_title": row["title"],
            "task_ids": live_ids,
            "_db_story_id": row["id"],
            "_priority": row["priority"],
            "_priority_override": row["priority_override"],
        })
    return stories if stories else None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Anti-link enforcement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _apply_anti_links(store: LayeredGraphStore, groups: list[dict]) -> list[dict]:
    """Split groups that violate anti-link constraints."""
    result = []
    for group in groups:
        task_ids = group.get("task_ids", [])
        if len(task_ids) <= 1:
            result.append(group)
            continue

        # Find anti-linked tasks
        anti_linked: set[str] = set()
        for i in range(len(task_ids)):
            for j in range(i + 1, len(task_ids)):
                if store.check_story_anti_link(task_ids[i], task_ids[j]):
                    anti_linked.add(task_ids[j])

        if not anti_linked:
            result.append(group)
        else:
            # Keep non-anti-linked in original group
            kept = [tid for tid in task_ids if tid not in anti_linked]
            if kept:
                result.append({
                    "story_title": group["story_title"],
                    "task_ids": kept,
                })
            # Each anti-linked task becomes its own story
            for tid in anti_linked:
                result.append({
                    "story_title": group["story_title"],
                    "task_ids": [tid],
                })
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main clustering entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def cluster_into_stories(
    store: LayeredGraphStore,
    tasks: list[dict] | None = None,
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    """Group active CQ tasks into stories using LLM clustering.

    Clustering is decoupled from triage state (accepted/done/dismissed).
    The cluster hash is computed from ALL KG tasks so that accept/done/dismiss
    never invalidates the cache. Triage filtering happens at the output layer.

    Args:
        exclude_ids: Task IDs to exclude from output (done/accepted/dismissed).
            Callers pass these so the cache stays stable.
    """
    from alteris.constants import CQ_CLUSTERING_CACHE_TTL

    if tasks is not None:
        kg_tasks = list(tasks)
    else:
        kg_tasks = _commitments_to_tasks(store)
        kg_tasks.extend(_beliefs_to_tasks(store))

    if not kg_tasks:
        return []

    # Get user overrides — but only to identify dismissed items (permanently removed)
    # and to include manual tasks. Done/accepted filtering is via exclude_ids.
    user_tasks = store.get_cq_tasks(include_done=True)
    dismissed_ids: set[str] = set()
    for ut in user_tasks:
        uid = ut.get("claim_id") or ut["id"]
        if ut.get("source") == "dismissed":
            dismissed_ids.add(uid)

    # Build full task map — exclude only permanently dismissed items.
    # Done/accepted items stay in the map for stable clustering.
    all_task_map: dict[str, dict] = {}
    for t in kg_tasks:
        if t["id"] in dismissed_ids:
            continue
        obj_data = {}
        if t.get("claim_id"):
            claim = store.get_claim(t["claim_id"])
            if claim:
                try:
                    obj_data = json.loads(claim.object) if isinstance(claim.object, str) else claim.object
                except (json.JSONDecodeError, TypeError):
                    pass
        t["_obj_data"] = obj_data
        all_task_map[t["id"]] = t

    # Also include manual tasks when auto-discovering
    if tasks is None:
        for ut in user_tasks:
            if ut["id"] not in all_task_map and ut.get("source") == "manual" and not ut.get("done"):
                ut["_obj_data"] = {}
                ut["who"] = ""
                ut["to_whom"] = ""
                ut["source_channel"] = ""
                all_task_map[ut["id"]] = ut

    if not all_task_map:
        return []

    # Hash is computed from ALL non-dismissed tasks — stable across accept/done
    current_hash = _compute_cluster_hash(list(all_task_map.keys()))

    # FAST PATH: check DB cache
    cached_groups = _rebuild_from_cache(store, all_task_map)
    if cached_groups is not None:
        cached_hash_row = store.conn.execute(
            "SELECT cluster_hash, updated_at FROM cq_stories WHERE source = 'llm' AND status = 'active' LIMIT 1"
        ).fetchone()
        if cached_hash_row:
            cached_hash = cached_hash_row["cluster_hash"]
            cached_at = cached_hash_row["updated_at"]
            now = int(time.time())
            cache_fresh = (now - cached_at) < CQ_CLUSTERING_CACHE_TTL

            if cached_hash == current_hash and cache_fresh:
                groups = cached_groups
                groups = _apply_anti_links(store, groups)
                return _groups_to_stories(groups, all_task_map, exclude_ids=exclude_ids)

            # Check if incremental clustering is appropriate (< 30% new tasks)
            cached_ids: set[str] = set()
            for g in cached_groups:
                cached_ids.update(g["task_ids"])
            new_ids = set(all_task_map.keys()) - cached_ids
            if new_ids and len(new_ids) < 0.3 * len(all_task_map) and cache_fresh:
                new_tasks = [all_task_map[tid] for tid in new_ids if tid in all_task_map]
                incremental = _llm_incremental_cluster(store, new_tasks, _groups_to_stories(cached_groups, all_task_map))
                if incremental:
                    groups = _merge_incremental(cached_groups, incremental)
                    groups = _apply_anti_links(store, groups)
                    _save_stories_to_db(store, groups, current_hash)
                    return _groups_to_stories(groups, all_task_map, exclude_ids=exclude_ids)

    # SLOW PATH: full LLM clustering
    task_list = list(all_task_map.values())
    groups = _llm_cluster(store, task_list)

    if groups is None:
        groups = _fallback_person_cluster(task_list)

    groups = _apply_anti_links(store, groups)
    _save_stories_to_db(store, groups, current_hash)
    return _groups_to_stories(groups, all_task_map, exclude_ids=exclude_ids)


def _merge_incremental(
    existing_groups: list[dict],
    assignments: list[dict],
) -> list[dict]:
    """Merge incremental assignments into existing groups."""
    # Build title → group mapping
    title_map: dict[str, dict] = {}
    for g in existing_groups:
        title_map[g["story_title"]] = g

    for assign in assignments:
        tid = assign["task_id"]
        title = assign["story_title"]
        is_new = assign.get("is_new_story", False)

        if not is_new and title in title_map:
            # Add to existing group
            title_map[title]["task_ids"].append(tid)
        else:
            # Create new group
            new_group = {"story_title": title, "task_ids": [tid]}
            existing_groups.append(new_group)
            title_map[title] = new_group

    return existing_groups


def _groups_to_stories(
    groups: list[dict],
    task_map: dict[str, dict],
    exclude_ids: set[str] | None = None,
) -> list[dict]:
    """Convert clustering groups into story dicts with full task objects.

    exclude_ids: task IDs to filter out (done/accepted/dismissed).
    Clustering stays stable; filtering happens here at the output layer.
    """
    stories = []
    for group in groups:
        task_ids = group.get("task_ids", [])
        members = [tid for tid in task_ids if tid in task_map]
        if exclude_ids:
            members = [tid for tid in members if tid not in exclude_ids]
        if not members:
            continue

        title = group.get("story_title", "Untitled")
        # Use all cluster members for stable story ID, not just visible subset
        all_valid = [tid for tid in task_ids if tid in task_map]
        story_id = group.get("_db_story_id") or _stable_story_id(all_valid or members)

        # Compute priority from ALL tasks in the cluster (not just visible)
        # so that accepting one low-priority task from a high-priority story
        # doesn't demote the story to "Later" on Desk.
        priorities = []
        for m in all_valid:
            t = task_map.get(m, {})
            p = t.get("priority", 3)
            bucket = t.get("bucket", "background")
            bucket_priority = {"immediate": 1, "review": 2}.get(bucket, 3)
            priorities.append(min(p, bucket_priority))
        effective_priority = min(priorities) if priorities else 99

        # Apply DB overrides
        if group.get("_priority_override") is not None:
            effective_priority = group["_priority_override"]
        elif group.get("_priority") is not None:
            effective_priority = group["_priority"]

        # Collect persons
        persons = set()
        for m in members:
            t = task_map.get(m, {})
            for key in ("who", "to_whom"):
                val = t.get(key, "")
                if val and val.strip():
                    persons.add(val.strip())

        # Check for unseen tasks
        has_unseen = any(
            task_map.get(m, {}).get("seen_at") is None
            and task_map.get(m, {}).get("source") != "manual"
            for m in members
        )

        # Compute latest updated_at
        latest_ts = max(
            (task_map.get(m, {}).get("updated_at", 0) or 0 for m in members),
            default=0,
        )
        if latest_ts == 0:
            latest_ts = int(time.time())

        stories.append({
            "id": story_id,
            "title": title,
            "source": "auto",
            "color": "",
            "icon": "",
            "status": "active",
            "priority": effective_priority,
            "priority_override": None,
            "tasks": [task_map[m] for m in members],
            "persons": [{"person_id": p, "name": p, "role": ""} for p in persons],
            "updated_at": latest_ts,
            "is_new": has_unseen,
        })

    return stories


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool handlers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _clean_task_for_output(task: dict) -> dict:
    """Strip internal fields from task dicts before returning."""
    out = {k: v for k, v in task.items() if not k.startswith("_")}
    return out


def handle_cq_get_stories(store: LayeredGraphStore, **kwargs) -> dict:
    """Desk stories: only accepted tasks, clustered into stories.

    Uses the same stable clustering as Feed — just filters to accepted only.
    No accepted tasks → empty result (frontend shows empty-state).
    """
    user_tasks = store.get_cq_tasks(include_done=True)
    accepted_ids: set[str] = set()
    not_accepted_ids: set[str] = set()
    for ut in user_tasks:
        uid = ut.get("claim_id") or ut["id"]
        if ut.get("accepted"):
            accepted_ids.add(uid)
        else:
            not_accepted_ids.add(uid)

    if not accepted_ids:
        return {"count": 0, "stories": []}

    # Cluster all tasks (cache-stable), then filter to accepted only
    all_stories = cluster_into_stories(store, exclude_ids=not_accepted_ids)
    # Also drop any tasks that weren't explicitly accepted
    filtered: list[dict] = []
    for story in all_stories:
        kept = [t for t in story.get("tasks", []) if t["id"] in accepted_ids]
        if kept:
            story["tasks"] = kept
            filtered.append(story)

    filtered.sort(key=lambda s: (s.get("priority") or 99, -(s.get("updated_at") or 0)))
    for s in filtered:
        s["tasks"] = [_clean_task_for_output(t) for t in s.get("tasks", [])]

    return {"count": len(filtered), "stories": filtered}


def handle_cq_create_story(
    store: LayeredGraphStore,
    title: str = "",
    color: str = "",
    icon: str = "",
    **kwargs,
) -> dict:
    """Create a manual story."""
    if not title:
        return {"error": "title is required"}

    story_id = f"story_{uuid.uuid4().hex[:12]}"
    store.put_cq_story(story_id, title, source="manual", color=color, icon=icon)
    return {"story_id": story_id, "title": title}


def handle_cq_update_story(
    store: LayeredGraphStore,
    story_id: str = "",
    title: str | None = None,
    color: str | None = None,
    icon: str | None = None,
    priority_override: int | None = None,
    **kwargs,
) -> dict:
    """Update story fields."""
    if not story_id:
        return {"error": "story_id is required"}

    fields = {}
    if title is not None:
        fields["title"] = title
    if color is not None:
        fields["color"] = color
    if icon is not None:
        fields["icon"] = icon
    if priority_override is not None:
        fields["priority_override"] = priority_override

    if not fields:
        return {"error": "no fields to update"}

    # Ensure story exists in DB (auto stories may not yet)
    existing = store.get_cq_stories(status="active")
    if not any(s["id"] == story_id for s in existing):
        store.put_cq_story(story_id, title=fields.get("title", ""), source="auto")

    ok = store.update_cq_story(story_id, **fields)
    return {"updated": ok, "story_id": story_id}


def handle_cq_add_to_story(
    store: LayeredGraphStore,
    story_id: str = "",
    task_id: str = "",
    **kwargs,
) -> dict:
    """Add a task to a story."""
    if not story_id or not task_id:
        return {"error": "story_id and task_id are required"}

    # Ensure story exists
    existing = store.get_cq_stories(status="active")
    if not any(s["id"] == story_id for s in existing):
        store.put_cq_story(story_id, title="", source="manual")

    members = store.get_story_members(story_id)
    position = max((m.get("position", 0) for m in members), default=-1) + 1

    store.add_story_member(story_id, task_id, position)
    return {"added": True, "story_id": story_id, "task_id": task_id}


def handle_cq_remove_from_story(
    store: LayeredGraphStore,
    story_id: str = "",
    task_id: str = "",
    **kwargs,
) -> dict:
    """Remove a task from a story."""
    if not story_id or not task_id:
        return {"error": "story_id and task_id are required"}

    store.remove_story_member(story_id, task_id)
    return {"removed": True, "story_id": story_id, "task_id": task_id}


def handle_cq_merge_stories(
    store: LayeredGraphStore,
    source_story_id: str = "",
    target_story_id: str = "",
    **kwargs,
) -> dict:
    """Merge source story into target story. Moves all members."""
    if not source_story_id or not target_story_id:
        return {"error": "source_story_id and target_story_id are required"}

    source_members = store.get_story_members(source_story_id)
    target_members = store.get_story_members(target_story_id)
    next_pos = max((m.get("position", 0) for m in target_members), default=-1) + 1

    moved = 0
    for m in source_members:
        store.remove_story_member(source_story_id, m["task_id"])
        store.add_story_member(target_story_id, m["task_id"], next_pos)
        next_pos += 1
        moved += 1

    # Also move persons
    source_persons = store.get_story_persons(source_story_id)
    for p in source_persons:
        store.add_story_person(target_story_id, p["person_id"], p.get("role", ""))
        store.remove_story_person(source_story_id, p["person_id"])

    # Archive the source story
    store.update_cq_story(source_story_id, status="archived")

    return {
        "merged": True,
        "source_story_id": source_story_id,
        "target_story_id": target_story_id,
        "tasks_moved": moved,
    }


def handle_cq_split_story(
    store: LayeredGraphStore,
    story_id: str = "",
    task_ids: list[str] | None = None,
    new_title: str = "",
    **kwargs,
) -> dict:
    """Split tasks from a story into a new story."""
    if not story_id or not task_ids:
        return {"error": "story_id and task_ids are required"}

    new_story_id = f"story_{uuid.uuid4().hex[:12]}"
    store.put_cq_story(new_story_id, title=new_title or "Split story", source="manual")

    moved = 0
    for i, tid in enumerate(task_ids):
        store.remove_story_member(story_id, tid)
        store.add_story_member(new_story_id, tid, i)
        # Record anti-links between split tasks and remaining members
        remaining = store.get_story_members(story_id)
        for rm in remaining:
            store.add_story_anti_link(tid, rm["task_id"])
        moved += 1

    return {
        "split": True,
        "original_story_id": story_id,
        "new_story_id": new_story_id,
        "tasks_moved": moved,
    }


def handle_cq_add_person_to_story(
    store: LayeredGraphStore,
    story_id: str = "",
    person_id: str = "",
    role: str = "",
    **kwargs,
) -> dict:
    """Add a person to a story."""
    if not story_id or not person_id:
        return {"error": "story_id and person_id are required"}

    store.add_story_person(story_id, person_id, role)
    return {"added": True, "story_id": story_id, "person_id": person_id}


def handle_cq_archive_story(
    store: LayeredGraphStore,
    story_id: str = "",
    **kwargs,
) -> dict:
    """Archive a story (set status=archived)."""
    if not story_id:
        return {"error": "story_id is required"}

    ok = store.update_cq_story(story_id, status="archived")
    return {"archived": ok, "story_id": story_id}


def handle_cq_get_feed(store: LayeredGraphStore, **kwargs) -> dict:
    """Feed: tasks not yet accepted or dismissed, clustered into stories.

    Uses the same stable clustering as Desk — accepts/done/dismiss never
    trigger re-clustering. Only new KG tasks cause incremental clustering.
    """
    # Gather user triage state
    user_tasks = store.get_cq_tasks(include_done=True)
    exclude_ids: set[str] = set()
    seen_map: dict[str, int | None] = {}
    for ut in user_tasks:
        uid = ut.get("claim_id") or ut["id"]
        if ut.get("accepted") or ut.get("done") or ut.get("source") == "dismissed":
            exclude_ids.add(uid)
        seen_map[uid] = ut.get("seen_at")

    # Cluster all tasks (cache-stable), filtering out accepted/done/dismissed
    stories = cluster_into_stories(store, exclude_ids=exclude_ids)

    # Mark unseen tasks
    for s in stories:
        for t in s.get("tasks", []):
            t["is_new"] = t["id"] not in seen_map or seen_map.get(t["id"]) is None

    # Clean internal fields
    for s in stories:
        s["tasks"] = [_clean_task_for_output(t) for t in s.get("tasks", [])]

    # Flat items list for clients
    feed_items = []
    task_story_map: dict[str, str] = {}
    for s in stories:
        for t in s.get("tasks", []):
            task_story_map[t.get("id", "")] = s["id"]
            feed_items.append(t)

    return {
        "count": len(feed_items),
        "items": feed_items,
        "stories": stories,
    }


def handle_cq_get_pulse(store: LayeredGraphStore, **kwargs) -> dict:
    """Return funnel stats, delta feed, and story summary.

    Scoped to a 7-day window for the trust funnel (not lifetime totals).
    """
    now_ts = int(time.time())
    seven_days_ago = now_ts - 7 * 86400

    # Recent-window counts for trust funnel (last 7 days, not lifetime)
    # Use event `timestamp` (when event occurred), not `created_at` (ingestion)
    try:
        row = store.conn.execute(
            "SELECT COUNT(*) FROM events WHERE timestamp >= ?", (seven_days_ago,)
        ).fetchone()
        recent_events = row[0] if row else 0

        row = store.conn.execute(
            "SELECT COUNT(*) FROM claims WHERE created_at >= ?", (seven_days_ago,)
        ).fetchone()
        recent_claims = row[0] if row else 0
    except Exception:
        recent_events = 0
        recent_claims = 0

    # Count stories
    stories_result = handle_cq_get_stories(store)
    all_stories = stories_result.get("stories", [])
    active_stories = [s for s in all_stories if s.get("status") == "active"]

    # Stories needing action: those with P1 or P2 tasks
    stories_need_action = 0
    for s in active_stories:
        for t in s.get("tasks", []):
            p = t.get("priority")
            if p is not None and p <= 2:
                stories_need_action += 1
                break

    # Delta feed: last 5 recently updated tasks as DeltaItem-compatible dicts
    kg_tasks = _commitments_to_tasks(store)
    kg_tasks.extend(_beliefs_to_tasks(store))
    kg_tasks.sort(key=lambda t: t.get("updated_at", 0) or 0, reverse=True)
    deltas = []
    for t in kg_tasks[:5]:
        updated = t.get("updated_at", 0) or 0
        if updated > 0:
            minutes_ago = max(0, (now_ts - updated) // 60)
        else:
            minutes_ago = 0
        # Determine delta type from task bucket/priority
        bucket = t.get("bucket", "background")
        if bucket == "immediate":
            delta_type = "new"
        elif t.get("done"):
            delta_type = "resolved"
        else:
            delta_type = "updated"
        deltas.append({
            "id": t["id"],
            "type": delta_type,
            "text": t.get("title", ""),
            "story_id": None,
            "minutes_ago": minutes_ago,
        })

    # Story summary (compact) — keys match Swift StorySummary CodingKeys
    # Only show stories with immediate-bucket tasks (urgent for today)
    stories_list = []
    for s in active_stories:
        immediate_tasks = [
            t for t in s.get("tasks", [])
            if t.get("bucket") == "immediate"
        ]
        if immediate_tasks:
            stories_list.append({
                "id": s["id"],
                "title": s["title"],
                "item_count": len(immediate_tasks),
                "priority": s.get("priority"),
            })
        if len(stories_list) >= 10:
            break

    # Determine last_updated_at — fall back to now if no task timestamps
    last_updated = 0
    for t in kg_tasks:
        ts = t.get("updated_at", 0) or 0
        if ts > last_updated:
            last_updated = ts
    if last_updated == 0:
        last_updated = now_ts

    # Flat keys matching Swift PulseData CodingKeys
    return {
        "events_scanned": recent_events,
        "claims_extracted": recent_claims,
        "stories_formed": len(active_stories),
        "stories_need_action": stories_need_action,
        "deltas": deltas,
        "stories": stories_list,
        "last_updated_at": last_updated,
    }


def handle_cq_mark_seen(
    store: LayeredGraphStore,
    task_ids: list[str] | None = None,
    **kwargs,
) -> dict:
    """Mark specified tasks as seen (clears 'new' badge)."""
    if not task_ids:
        return {"error": "task_ids is required"}

    marked = 0
    for tid in task_ids:
        store.mark_task_seen(tid)
        marked += 1

    return {"marked": marked}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

register_tool(ToolDef(
    name="cq_get_stories",
    description="Get CQ stories: auto-clustered task groups from KG relationships (thread, person, entity), layered with manual overrides. Each story has nested tasks, persons, priority, and isNew flag.",
    permission="write",
    params=[],
    handler=handle_cq_get_stories,
))

register_tool(ToolDef(
    name="cq_create_story",
    description="Create a manual CQ story for grouping tasks.",
    permission="write",
    params=[
        ToolParam("title", "string", "Story title", required=True),
        ToolParam("color", "string", "Hex color for UI"),
        ToolParam("icon", "string", "SF Symbol name for icon"),
    ],
    handler=handle_cq_create_story,
))

register_tool(ToolDef(
    name="cq_update_story",
    description="Update a CQ story's title, color, icon, or priority override.",
    permission="write",
    params=[
        ToolParam("story_id", "string", "Story ID", required=True),
        ToolParam("title", "string", "New title"),
        ToolParam("color", "string", "New color"),
        ToolParam("icon", "string", "New icon"),
        ToolParam("priority_override", "integer", "Priority override (1=highest)"),
    ],
    handler=handle_cq_update_story,
))

register_tool(ToolDef(
    name="cq_add_to_story",
    description="Add a task to a CQ story.",
    permission="write",
    params=[
        ToolParam("story_id", "string", "Story ID", required=True),
        ToolParam("task_id", "string", "Task ID to add", required=True),
    ],
    handler=handle_cq_add_to_story,
))

register_tool(ToolDef(
    name="cq_remove_from_story",
    description="Remove a task from a CQ story.",
    permission="write",
    params=[
        ToolParam("story_id", "string", "Story ID", required=True),
        ToolParam("task_id", "string", "Task ID to remove", required=True),
    ],
    handler=handle_cq_remove_from_story,
))

register_tool(ToolDef(
    name="cq_merge_stories",
    description="Merge two CQ stories: move all tasks from source into target, archive source.",
    permission="write",
    params=[
        ToolParam("source_story_id", "string", "Story to merge from", required=True),
        ToolParam("target_story_id", "string", "Story to merge into", required=True),
    ],
    handler=handle_cq_merge_stories,
))

register_tool(ToolDef(
    name="cq_split_story",
    description="Split tasks from a story into a new story. Creates anti-links to prevent re-merging.",
    permission="write",
    params=[
        ToolParam("story_id", "string", "Story to split from", required=True),
        ToolParam("task_ids", "array", "Task IDs to move to new story", required=True),
        ToolParam("new_title", "string", "Title for the new story"),
    ],
    handler=handle_cq_split_story,
))

register_tool(ToolDef(
    name="cq_add_person_to_story",
    description="Associate a person with a CQ story.",
    permission="write",
    params=[
        ToolParam("story_id", "string", "Story ID", required=True),
        ToolParam("person_id", "string", "Person ID", required=True),
        ToolParam("role", "string", "Role (e.g. 'stakeholder', 'assignee')"),
    ],
    handler=handle_cq_add_person_to_story,
))

register_tool(ToolDef(
    name="cq_archive_story",
    description="Archive a CQ story (hides from active view).",
    permission="write",
    params=[
        ToolParam("story_id", "string", "Story ID to archive", required=True),
    ],
    handler=handle_cq_archive_story,
))

register_tool(ToolDef(
    name="cq_get_feed",
    description="Get the CQ feed: tasks sorted by recency with 'new' flag and story grouping. For the Feed surface.",
    permission="write",
    params=[],
    handler=handle_cq_get_feed,
))

register_tool(ToolDef(
    name="cq_get_pulse",
    description="Get the CQ pulse: funnel stats (events->claims->stories->action), delta feed (last 5 changes), and compact story summary. For the Pulse surface.",
    permission="read",
    params=[],
    handler=handle_cq_get_pulse,
))

register_tool(ToolDef(
    name="cq_mark_seen",
    description="Mark tasks as seen (clears 'new' badge in feed/stories).",
    permission="write",
    params=[
        ToolParam("task_ids", "array", "List of task IDs to mark as seen", required=True),
    ],
    handler=handle_cq_mark_seen,
))
