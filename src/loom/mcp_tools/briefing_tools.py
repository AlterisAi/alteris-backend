"""Two-phase briefing MCP tools.

Splits the briefing pipeline into two MCP tool calls so the macOS app
can present anticipation-engine questions via its native UI:

  brief_start   — runs phases 1-2.7 (gather, context, anticipation engine),
                   persists intermediate state, returns questions as JSON.
  brief_continue — loads state, injects user answers, runs phases 2.8-3
                   (blind spot generation + final synthesis), returns briefing.

State is persisted to ~/.loom/briefings/session_{id}.pkl using pickle
to avoid dataclass serialization headaches (CalendarEvent and Commitment
use __slots__, not @dataclass).
"""

from __future__ import annotations

import logging
import pickle
import time
import uuid

from loom.constants import CLOUD_DEEP_MODEL, LOOM_DIR
from loom.mcp_tools import ToolDef, ToolParam, register_tool
from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)

BRIEFING_SESSIONS_DIR = LOOM_DIR / "briefings"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool 1: brief_start
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def handle_brief_start(
    store: LayeredGraphStore,
    days: int = 7,
    tz: str = "America/Los_Angeles",
    **kwargs,
) -> dict:
    """Phase 1: Run briefing through anticipation engine, return questions.

    Executes phases 1, 2, 2b, 2.5-2.7 of the briefing pipeline (everything
    up to user questions). Persists intermediate state to disk so
    brief_continue can resume after the app collects answers from the user.
    """
    from loom.briefing import (
        BRIEFING_LOOKBACK_DAYS,
        _build_briefing_prompt,
        _gather_event_context,
        _get_commitment_source_snippet,
        _get_cross_source_links,
        _get_event_facets,
        _get_logistics_facts,
        _get_open_commitments,
        _get_recently_resolved,
        _get_relational_context,
        _get_upcoming_events,
        _load_profile,
        _match_commitments_to_events,
        _run_anticipation_pass,
    )
    from loom.llm.gemini import GeminiClient

    llm_client = GeminiClient()
    model = CLOUD_DEEP_MODEL

    t0 = time.time()
    profile = _load_profile()

    # ── Phase 1: Gather ──
    logger.info("brief_start: Phase 1 — gathering data")

    events = _get_upcoming_events(store, days_ahead=days, user_tz=tz)
    logger.info("Calendar events (next %d days): %d", days, len(events))

    all_commitments = _get_open_commitments(store)
    logger.info("Open commitments: %d", len(all_commitments))

    overdue_count = sum(1 for c in all_commitments if c.is_overdue)
    event_commitments, orphaned = _match_commitments_to_events(
        events, all_commitments,
    )
    matched_count = len(all_commitments) - len(orphaned)

    # ── Phase 2: Context ──
    logger.info("brief_start: Phase 2 — gathering context")

    event_contexts: dict[str, list[dict]] = {}
    for event in events:
        ctx = _gather_event_context(
            store, event, lookback_days=BRIEFING_LOOKBACK_DAYS,
        )
        event_contexts[event.event_id] = ctx

    orphan_snippets: dict[str, str] = {}
    for c in orphaned[:50]:
        snippet = _get_commitment_source_snippet(store, c)
        if snippet:
            orphan_snippets[c.claim_id] = snippet

    recently_resolved = _get_recently_resolved(store, lookback_days=7)
    logistics_facts = _get_logistics_facts(store)
    relational_context = _get_relational_context(store)

    # ── Phase 2b: Cross-source enrichment ──
    logger.info("brief_start: Phase 2b — cross-source enrichment")

    ef: dict[str, dict[str, list[str]]] = {}
    csl: dict[str, list[dict]] = {}
    for event in events:
        facets = _get_event_facets(store, event.event_id)
        if facets:
            ef[event.event_id] = facets
        links = _get_cross_source_links(store, event.event_id)
        if links:
            csl[event.event_id] = links

    # ── Phase 2.5-2.7: Anticipation Engine ──
    anticipation_result: dict = {
        "system_queries": [], "system_results": [],
        "web_searches": [], "web_results": [],
        "user_questions": [], "reassurances": [],
    }

    if events:
        base_prompt = _build_briefing_prompt(
            events=events,
            event_commitments=event_commitments,
            event_contexts=event_contexts,
            orphaned=orphaned,
            orphan_snippets=orphan_snippets,
            user_tz=tz,
            profile=profile,
            event_facets=ef,
            cross_source_links=csl,
            recently_resolved=recently_resolved,
            logistics_facts=logistics_facts or None,
            relational_context=relational_context or None,
        )

        anticipation_result = _run_anticipation_pass(
            prompt=base_prompt,
            llm_client=llm_client,
            store=store,
            model=model,
            user_tz=tz,
        )

    # ── Persist session state ──
    session_id = uuid.uuid4().hex[:12]
    BRIEFING_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    state = {
        "session_id": session_id,
        "t0": t0,
        "days": days,
        "tz": tz,
        "model": model,
        "profile": profile,
        "events": events,
        "all_commitments": all_commitments,
        "event_commitments": event_commitments,
        "orphaned": orphaned,
        "orphan_snippets": orphan_snippets,
        "event_contexts": event_contexts,
        "recently_resolved": recently_resolved,
        "logistics_facts": logistics_facts,
        "relational_context": relational_context,
        "ef": ef,
        "csl": csl,
        "anticipation_result": anticipation_result,
        "matched_count": matched_count,
        "overdue_count": overdue_count,
    }

    state_path = BRIEFING_SESSIONS_DIR / f"session_{session_id}.pkl"
    with open(state_path, "wb") as f:
        pickle.dump(state, f)

    logger.info(
        "brief_start: saved session %s (%d bytes)",
        session_id, state_path.stat().st_size,
    )

    # ── Format questions for the app ──
    questions = []
    for i, q in enumerate(anticipation_result.get("user_questions", []), 1):
        questions.append({
            "index": i,
            "event_subject": q.get("event_subject", ""),
            "category": q.get("category", ""),
            "question": q.get("question", ""),
            "confidence": q.get("confidence", 0.5),
        })

    elapsed = time.time() - t0
    logger.info("brief_start: complete (%.1fs), %d questions", elapsed, len(questions))

    return {
        "session_id": session_id,
        "questions": questions,
        "stats": {
            "events_count": len(events),
            "commitments_open": len(all_commitments),
            "commitments_matched": matched_count,
            "overdue": overdue_count,
        },
        "elapsed_s": round(elapsed, 2),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool 2: brief_continue
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def handle_brief_continue(
    store: LayeredGraphStore,
    session_id: str = "",
    answers: list | None = None,
    **kwargs,
) -> dict:
    """Phase 2: Accept user answers, generate final briefing.

    Loads the session state saved by brief_start, processes user answers,
    runs blind spot candidate generation (phase 2.9) and final briefing
    synthesis (phase 3).
    """
    from loom.briefing import (
        BLIND_SPOT_CANDIDATES,
        BLIND_SPOT_FINAL,
        BRIEFING_SYSTEM_PROMPT,
        _build_briefing_prompt,
        _generate_blind_spot_candidates,
        _rank_and_dedup_candidates,
        _store_user_answer,
    )
    from loom.llm.gemini import GeminiClient

    if not session_id:
        return {"error": "session_id is required"}

    state_path = BRIEFING_SESSIONS_DIR / f"session_{session_id}.pkl"
    if not state_path.exists():
        return {"error": f"Session {session_id} not found"}

    with open(state_path, "rb") as f:
        state = pickle.load(f)

    llm_client = GeminiClient()
    model = state["model"]
    t0 = state["t0"]

    # ── Process user answers ──
    answers = answers or []
    user_answers: list[dict] = []
    questions = state["anticipation_result"].get("user_questions", [])

    for ans in answers:
        idx = ans.get("index", 0) - 1  # 1-indexed from the app
        answer_text = ans.get("answer", "")
        if answer_text and 0 <= idx < len(questions):
            q = questions[idx]
            entry = {
                "event_subject": q.get("event_subject", ""),
                "question": q.get("question", ""),
                "category": q.get("category", ""),
                "answer": answer_text,
                "confidence": q.get("confidence", ""),
            }
            user_answers.append(entry)
            _store_user_answer(store, entry)

    logger.info("brief_continue: %d user answers processed", len(user_answers))

    # ── Unpack state ──
    events = state["events"]
    anticipation_result = state["anticipation_result"]

    # ── Phase 2.9: Blind spot candidate generation + ranking ──
    ranked_blind_spots: list[dict] = []
    all_candidates: list[dict] = []

    if events:
        candidate_prompt = _build_briefing_prompt(
            events=events,
            event_commitments=state["event_commitments"],
            event_contexts=state["event_contexts"],
            orphaned=state["orphaned"],
            orphan_snippets=state["orphan_snippets"],
            user_tz=state["tz"],
            profile=state["profile"],
            event_facets=state["ef"],
            cross_source_links=state["csl"],
            additional_context=(
                anticipation_result["system_results"]
                if anticipation_result["system_results"]
                else None
            ),
            recently_resolved=state["recently_resolved"],
            logistics_facts=state["logistics_facts"] or None,
            relational_context=state["relational_context"] or None,
            web_results=anticipation_result["web_results"] or None,
            user_answers=user_answers or None,
            reassurances=anticipation_result["reassurances"] or None,
        )

        all_candidates = _generate_blind_spot_candidates(
            prompt=candidate_prompt,
            llm_client=llm_client,
            model=model,
            n_candidates=BLIND_SPOT_CANDIDATES,
        )

        if all_candidates:
            ranked_blind_spots = _rank_and_dedup_candidates(
                all_candidates, n_final=BLIND_SPOT_FINAL,
            )

    # ── Phase 3: Final briefing synthesis ──
    logger.info("brief_continue: Phase 3 — synthesizing briefing")

    prompt = ""
    briefing_md = "Briefing generation failed — no events in window."

    if events:
        prompt = _build_briefing_prompt(
            events=events,
            event_commitments=state["event_commitments"],
            event_contexts=state["event_contexts"],
            orphaned=state["orphaned"],
            orphan_snippets=state["orphan_snippets"],
            user_tz=state["tz"],
            profile=state["profile"],
            event_facets=state["ef"],
            cross_source_links=state["csl"],
            additional_context=(
                anticipation_result["system_results"]
                if anticipation_result["system_results"]
                else None
            ),
            recently_resolved=state["recently_resolved"],
            logistics_facts=state["logistics_facts"] or None,
            relational_context=state["relational_context"] or None,
            web_results=anticipation_result["web_results"] or None,
            user_answers=user_answers or None,
            reassurances=anticipation_result["reassurances"] or None,
            blind_spots=ranked_blind_spots or None,
        )

        logger.info("Prompt: %d chars", len(prompt))

        briefing_md = llm_client.generate(
            prompt=prompt,
            system=BRIEFING_SYSTEM_PROMPT,
            model=model,
            temperature=0.3,
            max_tokens=8192,
            google_search=True,
        )

        if not briefing_md:
            briefing_md = "Briefing generation failed — no response from LLM."

    elapsed = time.time() - t0

    # ── Save briefing markdown ──
    ts = int(time.time())
    BRIEFING_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = BRIEFING_SESSIONS_DIR / f"briefing_{ts}.md"
    md_path.write_text(briefing_md)
    logger.info("Briefing saved to %s", md_path)

    # ── Cleanup session file ──
    state_path.unlink(missing_ok=True)

    logger.info("brief_continue: complete (%.1fs total)", elapsed)

    return {
        "briefing": briefing_md,
        "stats": {
            "events_count": len(events),
            "commitments_open": len(state["all_commitments"]),
            "commitments_matched": state["matched_count"],
            "overdue": state["overdue_count"],
            "blind_spots_candidates": len(all_candidates),
            "blind_spots_final": len(ranked_blind_spots),
            "user_answers": len(user_answers),
        },
        "elapsed_s": round(elapsed, 2),
        "prompt_length": len(prompt),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

register_tool(ToolDef(
    name="brief_start",
    description=(
        "Start briefing generation: runs phases 1-2.7 "
        "(gather, context, anticipation engine), "
        "returns questions for the user."
    ),
    permission="write",
    params=[
        ToolParam("days", "integer", "Days ahead to look (default 7)", default=7),
        ToolParam(
            "tz", "string",
            "Timezone (default America/Los_Angeles)",
            default="America/Los_Angeles",
        ),
    ],
    handler=handle_brief_start,
))

register_tool(ToolDef(
    name="brief_continue",
    description=(
        "Continue briefing with user answers: runs phases 2.8-3 "
        "(blind spot generation, synthesis), returns final briefing markdown."
    ),
    permission="write",
    params=[
        ToolParam(
            "session_id", "string",
            "Session ID from brief_start",
            required=True,
        ),
        ToolParam(
            "answers", "array",
            "Array of {index, answer} objects with user answers to questions",
        ),
    ],
    handler=handle_brief_continue,
))
