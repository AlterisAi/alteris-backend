"""MCP tools for Person Model estimation, retrieval, chat onboarding, and updates.

Tools:
  loom_estimate_person_model  — Run full estimation pipeline (scout → surveyor → save)
  loom_get_person_model       — Get the latest person model
  loom_person_model_start     — Sandwich phase 1: estimate + generate targeted questions
  loom_person_model_chat      — Multi-turn onboarding chat to fill in gaps
  loom_update_person_model    — Direct field update (bypass chat)
  loom_person_model_finish    — Sync corrections to profile.yaml and return summary
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from loom.mcp_tools import ToolDef, ToolParam, register_tool
from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)

# Late import to avoid circular dependency
_store_instance: LayeredGraphStore | None = None


def _get_store(**kwargs) -> LayeredGraphStore:
    """Get or create a store instance from kwargs or singleton."""
    global _store_instance
    s = kwargs.get("store")
    if s:
        return s
    if _store_instance is None:
        _store_instance = LayeredGraphStore()
    return _store_instance


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool 1: Estimate Person Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def handle_estimate_person_model(
    store: LayeredGraphStore | None = None,
    force: bool = False,
    **kwargs,
) -> dict:
    """Run the full person model estimation pipeline."""
    store = store or _get_store(**kwargs)

    try:
        from loom.llm.gemini import GeminiClient
        llm = GeminiClient()
    except Exception as exc:
        logger.warning("Gemini not available, running scout-only: %s", exc)
        llm = None

    from loom.person_model import estimate_person_model, get_model_gaps

    model = estimate_person_model(store, llm=llm, force=force)

    gaps = get_model_gaps(model)
    confidences = {}
    for dim_name, dim_data in model.items():
        if isinstance(dim_data, dict) and "confidence" in dim_data:
            confidences[dim_name] = dim_data["confidence"]

    return {
        "status": "ok",
        "model": model,
        "confidences": confidences,
        "gaps": gaps,
        "gap_count": len(gaps),
    }


register_tool(ToolDef(
    name="loom_estimate_person_model",
    description="Estimate the 11-dimension Person Model from all knowledge graph data. "
                "Runs deterministic scouts over events/beliefs/persons, then optionally "
                "synthesizes with a Pro model. Returns the model with per-dimension confidence.",
    permission="write",
    params=[
        ToolParam("force", "boolean", "Force re-estimation even if recent model exists", default=False),
    ],
    handler=handle_estimate_person_model,
))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool 1b: Person Model Start (sandwich phase 1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_QUESTION_GEN_SYSTEM = """\
You are generating intake questions for a Person Model — a computational autobiography.

You will receive:
1. The current estimated Person Model (from automated analysis of the user's email, \
calendar, messages, and contacts)
2. The raw scout data that was used to build it
3. A user profile (if any) from profile.yaml

Your job: Generate 5-8 targeted questions that would fill the HIGHEST-LEVERAGE gaps \
in the model. These are things the system CANNOT infer from data alone.

## Priority Order (ask about these first)
1. **Household/Family** — Who lives with them? Kids, partner, dependents, pets. \
This is the single highest-leverage gap because it affects scheduling, priorities, \
calendar interpretation.
2. **Non-negotiable time blocks** — School pickups, workouts, religious services, \
therapy. The system sees calendar gaps but can't tell what's sacred.
3. **Local ecosystem** — Go-to coffee shop, lunch spots, exercise routes. \
Data shows location patterns but not preferences.
4. **Routines** — Typical weekday/weekend structure. Work-from-home vs office days.
5. **Personal rhythms** — Deep work windows, wake/sleep schedule.
6. **Travel preferences** — Flight class, airline loyalty, seating, hotel chains, \
dietary constraints on flights.

## Rules
- Skip questions for dimensions that already have confidence >= 0.7
- If the scouts found partial data, reference it: "I see you have meetings with X — \
are they part of your core team?"
- Each question must have a clear `category` (one of: family, routines, local, rhythms, \
professional, goals)
- Each question must have a `why` explaining what the system gains from the answer
- Keep questions concrete and specific, not open-ended
- Maximum 8 questions

## Output Format (strict JSON)
```json
{
  "questions": [
    {
      "index": 1,
      "category": "family",
      "question": "The specific question text",
      "why": "Brief explanation of what this unlocks in the model",
      "dimension": "life_architecture",
      "current_evidence": "What the scouts found (or empty string)"
    }
  ],
  "model_summary": "2-3 sentence summary of what the system already knows"
}
```
"""


def _gather_user_kg_context(store: LayeredGraphStore) -> dict:
    """Gather user-specific context from the knowledge graph beyond profile.yaml."""
    conn = store.conn
    now = int(time.time())
    context = {}

    # User person
    user_row = conn.execute(
        "SELECT person_id, canonical_name FROM persons WHERE is_user = 1 LIMIT 1"
    ).fetchone()
    if user_row:
        context["user_name"] = user_row["canonical_name"]
        context["user_person_id"] = user_row["person_id"]

        # All identifiers
        idents = conn.execute(
            "SELECT identifier_type, identifier FROM person_identifiers WHERE person_id = ?",
            (user_row["person_id"],),
        ).fetchall()
        context["user_emails"] = [r["identifier"] for r in idents if r["identifier_type"] == "email"]
        context["user_phones"] = [r["identifier"] for r in idents if r["identifier_type"] == "phone"]

        # User's person profile (communication stats)
        try:
            profile = conn.execute(
                "SELECT * FROM person_profiles WHERE person_id = ?",
                (user_row["person_id"],),
            ).fetchone()
            if profile:
                context["user_profile"] = dict(profile)
        except Exception:
            pass

    # Inner circle (top contacts by interaction volume)
    try:
        top_contacts = conn.execute("""
            SELECT p.canonical_name, pp.message_count, pp.channels
            FROM person_profiles pp
            JOIN persons p ON p.person_id = pp.person_id
            WHERE pp.tier <= 2 AND p.is_user = 0
            ORDER BY pp.message_count DESC
            LIMIT 10
        """).fetchall()
        context["inner_circle"] = [
            {"name": r["canonical_name"], "messages": r["message_count"], "channels": r["channels"]}
            for r in top_contacts
        ]
    except Exception:
        context["inner_circle"] = []

    # Beliefs about the user (relation beliefs, entity beliefs)
    try:
        user_beliefs = conn.execute("""
            SELECT belief_type, subject, summary, confidence
            FROM beliefs
            WHERE status = 'active'
              AND (subject LIKE '%user%' OR subject LIKE '%self%'
                   OR belief_type IN ('relation', 'entity'))
            ORDER BY confidence DESC
            LIMIT 20
        """).fetchall()
        context["user_beliefs"] = [
            {"type": r["belief_type"], "subject": r["subject"],
             "summary": r["summary"], "confidence": r["confidence"]}
            for r in user_beliefs
        ]
    except Exception:
        context["user_beliefs"] = []

    # Open commitments count
    try:
        open_count = conn.execute("""
            SELECT COUNT(*) as cnt FROM cq_tasks WHERE done = 0
        """).fetchone()
        context["open_commitments"] = open_count["cnt"] if open_count else 0
    except Exception:
        context["open_commitments"] = 0

    # KG stats
    try:
        events_count = conn.execute("SELECT COUNT(*) as cnt FROM events").fetchone()
        persons_count = conn.execute("SELECT COUNT(*) as cnt FROM persons").fetchone()
        context["total_events"] = events_count["cnt"] if events_count else 0
        context["total_persons"] = persons_count["cnt"] if persons_count else 0
    except Exception:
        pass

    # Profile.yaml
    from loom.person_model import _load_user_config_safe
    config = _load_user_config_safe()
    if config:
        context["profile_yaml"] = config

    return context


def generate_onboarding_questions(
    model: dict, gaps: list[dict], llm, store: LayeredGraphStore,
    max_questions: int = 8,
) -> list[dict]:
    """Generate targeted onboarding questions from person model gaps.

    Calls the LLM to produce high-leverage questions based on model gaps
    and KG context. Falls back to static questions if LLM unavailable.

    Returns: [{"dimension": str, "field": str, "question": str, "category": str, "why": str}, ...]
    """
    from loom.person_model import phase1_scout_all

    kg_context = _gather_user_kg_context(store)
    scout_data = phase1_scout_all(store)

    confidences = {}
    for dim_name, dim_data in model.items():
        if isinstance(dim_data, dict) and "confidence" in dim_data:
            confidences[dim_name] = dim_data["confidence"]

    if not llm:
        return _fallback_questions(gaps, model)[:max_questions]

    try:
        from loom.constants import PERSON_MODEL_CONSIGLIERE_MODEL

        question_prompt = json.dumps({
            "current_model": model,
            "confidences": confidences,
            "gaps": [g["dimension"] for g in gaps],
            "scout_data_sample": {k: v for k, v in scout_data.items()
                                  if k not in ("apple_mail_categories", "apple_intelligence_signals")},
            "kg_context": kg_context,
        }, indent=2, default=str)

        resp = llm.generate(
            prompt=question_prompt,
            system=_QUESTION_GEN_SYSTEM,
            model=PERSON_MODEL_CONSIGLIERE_MODEL,
            temperature=0.3,
            max_tokens=8192,
            response_schema={
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "category": {"type": "string"},
                                "question": {"type": "string"},
                                "why": {"type": "string"},
                                "dimension": {"type": "string"},
                                "current_evidence": {"type": "string"},
                            },
                            "required": ["index", "category", "question", "why", "dimension"],
                        },
                    },
                    "model_summary": {"type": "string"},
                },
                "required": ["questions", "model_summary"],
            },
        )

        if not resp or not resp.strip():
            return _fallback_questions(gaps, model)[:max_questions]

        text = resp.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        if not text.startswith("{"):
            start = text.find("{")
            if start >= 0:
                text = text[start:]
        if not text.endswith("}"):
            end = text.rfind("}")
            if end >= 0:
                text = text[:end + 1]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
                repaired = repair_json(text, return_objects=True)
                if isinstance(repaired, dict):
                    parsed = repaired
                else:
                    return _fallback_questions(gaps, model)[:max_questions]
            except Exception:
                return _fallback_questions(gaps, model)[:max_questions]

        return parsed.get("questions", [])[:max_questions]

    except Exception as exc:
        logger.warning("Question generation failed: %s", exc)
        return _fallback_questions(gaps, model)[:max_questions]


def handle_person_model_start(
    store: LayeredGraphStore | None = None,
    force: bool = False,
    **kwargs,
) -> dict:
    """Sandwich phase 1: estimate model + generate targeted questions."""
    t0 = time.time()
    store = store or _get_store(**kwargs)

    # Phase 1: Run estimation (scouts + optional surveyor)
    try:
        from loom.llm.gemini import GeminiClient
        llm = GeminiClient()
    except Exception as exc:
        logger.warning("Gemini not available, running scout-only: %s", exc)
        llm = None

    from loom.person_model import (
        estimate_person_model, get_model_gaps,
    )

    model = estimate_person_model(store, llm=llm, force=force)

    # Phase 2: Gather KG context
    kg_context = _gather_user_kg_context(store)

    # Phase 3: Generate questions via reusable helper
    gaps = get_model_gaps(model)
    confidences = {}
    for dim_name, dim_data in model.items():
        if isinstance(dim_data, dict) and "confidence" in dim_data:
            confidences[dim_name] = dim_data["confidence"]

    questions = generate_onboarding_questions(model, gaps, llm, store)
    model_summary = ""  # model_summary is nice-to-have, kept empty if not from LLM
    session_id = f"pm_{uuid.uuid4().hex[:12]}"

    # Save session state
    session_state = {
        "model": model,
        "kg_context": kg_context,
        "questions": questions,
        "model_summary": model_summary,
    }
    store.put_cq_session(
        session_id,
        json.dumps(session_state, default=str),
        title="Person Model Onboarding",
        session_type="person_model_start",
    )

    elapsed = time.time() - t0

    return {
        "session_id": session_id,
        "questions": questions,
        "model_summary": model_summary,
        "confidences": confidences,
        "gaps": [g["dimension"] for g in gaps],
        "gap_count": len(gaps),
        "stats": {
            "total_events": kg_context.get("total_events", 0),
            "total_persons": kg_context.get("total_persons", 0),
            "inner_circle_count": len(kg_context.get("inner_circle", [])),
            "open_commitments": kg_context.get("open_commitments", 0),
        },
        "elapsed_s": round(elapsed, 1),
    }


def _fallback_questions(gaps: list[dict], model: dict) -> list[dict]:
    """Generate static questions when LLM is unavailable."""
    static = [
        {
            "index": 1, "category": "family", "dimension": "life_architecture",
            "question": "Who makes up your immediate household? (Partner, kids, roommates, pets)",
            "why": "Family structure is the single biggest factor in how Loom interprets your calendar and priorities.",
            "current_evidence": "",
        },
        {
            "index": 2, "category": "routines", "dimension": "temporal_patterns",
            "question": "What are your non-negotiable daily time blocks? (School pickups, workouts, meetings that never move)",
            "why": "Knowing your fixed blocks lets Loom identify scheduling conflicts and protect your sacred time.",
            "current_evidence": "",
        },
        {
            "index": 3, "category": "local", "dimension": "life_architecture",
            "question": "What's your go-to coffee shop and usual order? Any favorite lunch spots or restaurants?",
            "why": "Your local ecosystem helps Loom make location-aware suggestions and understand your routines.",
            "current_evidence": "",
        },
        {
            "index": 4, "category": "rhythms", "dimension": "temporal_patterns",
            "question": "When's your optimal deep work window, and what does your typical work day look like start to finish?",
            "why": "Understanding your energy patterns helps Loom time notifications and prioritize information delivery.",
            "current_evidence": "",
        },
        {
            "index": 5, "category": "professional", "dimension": "professional",
            "question": "What's your role, and who are the 2-3 people you interact with most at work?",
            "why": "Role context helps Loom prioritize communications and understand meeting importance.",
            "current_evidence": "",
        },
    ]
    # Filter to only questions for gap dimensions
    gap_dims = {g["dimension"] for g in gaps}
    return [q for q in static if q["dimension"] in gap_dims][:6]


register_tool(ToolDef(
    name="loom_person_model_start",
    description="Sandwich phase 1: Run person model estimation, gather KG context, "
                "and generate highest-leverage intake questions. Returns questions "
                "for the user to answer before starting the chat conversation.",
    permission="write",
    params=[
        ToolParam("force", "boolean", "Force re-estimation even if recent model exists", default=False),
    ],
    handler=handle_person_model_start,
))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool 2: Get Person Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def handle_get_person_model(
    store: LayeredGraphStore | None = None,
    **kwargs,
) -> dict:
    """Return the latest person model."""
    store = store or _get_store(**kwargs)

    from loom.person_model import get_person_model, get_model_gaps

    existing = get_person_model(store)
    if not existing:
        return {"status": "empty", "message": "No person model exists. Run loom_estimate_person_model first."}

    model = existing["model"]
    gaps = get_model_gaps(model)
    confidences = {}
    for dim_name, dim_data in model.items():
        if isinstance(dim_data, dict) and "confidence" in dim_data:
            confidences[dim_name] = dim_data["confidence"]

    return {
        "status": "ok",
        "version": existing.get("version"),
        "estimated_at": existing.get("estimated_at"),
        "estimation_method": existing.get("estimation_method"),
        "model": model,
        "user_corrections": existing.get("user_corrections", {}),
        "confidences": confidences,
        "gaps": gaps,
    }


register_tool(ToolDef(
    name="loom_get_person_model",
    description="Get the latest 11-dimension Person Model including per-dimension confidence "
                "scores, identified gaps, and any user corrections.",
    permission="read",
    params=[],
    handler=handle_get_person_model,
))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool 3: Person Model Chat (multi-turn onboarding)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Chat tool declarations for function calling
def _person_model_chat_tools() -> list[dict]:
    """Function declarations the LLM can call during the onboarding chat."""
    return [
        {
            "name": "update_section",
            "description": "Update a field in the person model based on what the user shared.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dimension": {
                        "type": "string",
                        "description": "Which dimension to update",
                        "enum": [
                            "identity", "professional", "communication_fingerprint",
                            "relationship_map", "active_workstreams", "goals_and_values",
                            "life_architecture", "temporal_patterns", "domain_expertise",
                            "financial_landscape", "visibility_gaps",
                        ],
                    },
                    "field": {
                        "type": "string",
                        "description": "The specific field within the dimension to update",
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to set (JSON-encoded if array/object)",
                    },
                },
                "required": ["dimension", "field", "value"],
            },
        },
        {
            "name": "confirm_section",
            "description": "Mark a dimension as confirmed/accurate by the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dimension": {
                        "type": "string",
                        "description": "The dimension the user confirms is accurate",
                    },
                },
                "required": ["dimension"],
            },
        },
        {
            "name": "skip_section",
            "description": "The user doesn't want to discuss this dimension right now.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dimension": {
                        "type": "string",
                        "description": "The dimension to skip",
                    },
                },
                "required": ["dimension"],
            },
        },
    ]


def _person_model_tool_executor(store: LayeredGraphStore):
    """Returns a tool executor closure bound to the store."""
    from loom.person_model import update_person_model_field

    actions = []

    def execute(name: str, args: dict) -> dict:
        try:
            if name == "update_section":
                value = args["value"]
                # Parse JSON values for arrays/objects
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass
                result = update_person_model_field(
                    store, args["dimension"], args["field"], value,
                )
                actions.append(f"Updated {args['dimension']}.{args['field']}")
                return {"status": "ok", "action": f"Updated {args['dimension']}.{args['field']}"}
            elif name == "confirm_section":
                actions.append(f"Confirmed {args['dimension']}")
                return {"status": "ok", "action": f"Confirmed {args['dimension']}"}
            elif name == "skip_section":
                actions.append(f"Skipped {args['dimension']}")
                return {"status": "ok", "action": f"Skipped {args['dimension']}"}
            else:
                return {"error": f"Unknown tool: {name}"}
        except Exception as exc:
            return {"error": str(exc)}

    execute.actions = actions  # type: ignore[attr-defined]
    return execute


_PERSON_MODEL_CHAT_SYSTEM = """\
You are the Loom Person Model onboarding assistant. Your job is to map the user's \
"Computational Autobiography" — a structured, holistic representation of their life \
architecture, relationships, and routines.

You are conducting a direct, conversational intake. Your goal is to fill in the gaps \
of their Person Model so the system understands not just how they work, but how they \
*live*. Get this right, and you become a deeply contextual partner. Get it wrong, and \
you're just another generic chatbot.

This is an intake conversation, not a therapy or coaching session. You are gathering \
concrete lifestyle infrastructure, not facilitating personal reflection.

## The Tools You Must Use
- `update_section(dimension, field, value)`: Call this IMMEDIATELY and silently the \
moment a user provides a piece of information. Do not wait.
- `confirm_section(dimension)`: Call this when a user confirms a summarized dimension \
looks correct.
- `skip_section(dimension)`: Call this if the user explicitly wants to skip or move on \
from a topic.

## What You Must Learn (The Dimensions)

You must explicitly cover ALL of these areas if they are not already known with high \
confidence.

### 1. Family & Relationships (The Core Orbit)
- Immediate household (spouse/partner, kids, roommates)
- Key relationships and their nature (names, ages of kids if applicable)
- Relationship status (single, partnered, married)
- Caregiving responsibilities (elderly parents, family members)
- Pets (names and types — these are critical to daily routines)

### 2. Routines & Typical Week (The Rhythms)
- Typical weekday structure (start to finish)
- Typical weekend structure
- Non-negotiable time blocks (gym, school drop-offs, date nights, religious services)
- Fixed morning and evening rituals

### 3. Local Life & Habits (The Ecosystem)
- Living area/neighborhood vibe (not exact address)
- Go-to coffee shop and specific order
- Go-to takeout, lunch spots, or favorite cuisines
- Decompression zones (where they walk, run, or exercise)
- Regular local infrastructure (barber, grocery store, clubs/memberships)

### 4. Personal Rhythms (Energy & Environment)
- Optimal deep work windows
- Wake up and sleep schedules
- Work environment (home, office, hybrid)
- Dietary preferences or restrictions

## Critical Communication Rules

### 1. Pyramid Communication (Mandatory)
Every single question MUST follow this structure:
1. **Value first:** State exactly why knowing this helps you build their model.
2. **Request second:** Ask the specific, concrete question.

Good: "To make sure I map your time accurately, I need to know your hard constraints. \
What are your non-negotiable daily blocks, like school pickups or workouts?"
Good: "Knowing your local ecosystem helps me tailor recommendations. What is your \
absolute go-to coffee shop, and what's your usual order?"
Good: "I want to understand your household dynamics so I have the right context for \
your weekend planning. Who makes up your immediate family or household?"

Bad: "Tell me about your family." (No value stated, too broad)
Bad: "What do you like to do in your neighborhood?" (Vague)
Bad: "How do you feel about your morning routine?" (Coaching mode)

### 2. Adaptation and Boundaries
- Never go into coaching mode. If they say they struggle with sleep, note it and ask \
about their ideal wake time. Do not ask *why* they struggle or how it makes them feel.
- One question at a time. Never stack multiple questions in a single response.
- Trust what you hear. Accept answers at face value and immediately call \
`update_section`. Do not over-explore.
- No bullet points in chat. Speak in short, clean paragraphs.

### 5. Travel Preferences (New)
- Flight class preference (economy, premium economy, business, first)
- Airline loyalty programs and preferred airlines
- Hotel chain preferences
- Seating preferences (window, aisle)
- Any travel constraints (dietary on flights, mobility, connecting flight tolerance)
- Passport / visa situations if relevant

## Conversation Flow & Strategy

1. Review the current model state (provided in the first user message context).
2. Review which questions were already answered in the intake phase — NEVER re-ask \
something the user has already told you, either in the intake answers or in prior messages.
3. Target the gaps — start with the highest-gap dimensions.
4. Show, then ask: if you have partial data, show it first. "I see you're based in \
the city, but I don't have your local spots mapped. What's your go-to place for takeout?"
5. Target 10-14 exchanges. Be warm but highly efficient.

## Rules
- When the user provides information, ALWAYS call update_section to save it.
- For array fields (family, routines, etc.), pass a JSON array as the value.
- Don't ask about things the model already has with high confidence unless the user \
wants to correct something.
- NEVER repeat a question that was already answered in the intake phase (see "User's \
Answers to Intake Questions" in the context) or in a previous message in this conversation.
- When the user says they're done or wants to stop, respect that immediately. Do NOT \
continue asking questions.
"""

_PERSON_MODEL_CONTEXT_TEMPLATE = """\
[PERSON MODEL CONTEXT — current state of the user's model]

## Current Person Model State
{model_json}

## Confidence Scores by Dimension
{confidence_summary}

## Identified Gaps (Focus Here)
{gaps_summary}
{extra_context}
[END CONTEXT]

{user_message}"""


def handle_person_model_chat(
    store: LayeredGraphStore | None = None,
    message: str = "",
    session_id: str | None = None,
    start_session_id: str | None = None,
    answers: list | None = None,
    **kwargs,
) -> dict:
    """Multi-turn onboarding chat to fill in Person Model gaps."""
    if not message:
        return {"error": "message is required"}

    store = store or _get_store(**kwargs)
    now = int(time.time())

    # Load start session context (from loom_person_model_start)
    start_context = None
    if start_session_id:
        start_session = store.get_cq_session(start_session_id)
        if start_session:
            raw_msgs = start_session.get("messages", "")
            if isinstance(raw_msgs, str):
                try:
                    start_context = json.loads(raw_msgs)
                except (json.JSONDecodeError, TypeError):
                    pass

    # Load or create chat session
    if session_id:
        session = store.get_cq_session(session_id)
        if session:
            messages = session.get("messages", [])
            if isinstance(messages, str):
                messages = json.loads(messages)
        else:
            messages = []
    else:
        session_id = f"pm_{uuid.uuid4().hex[:12]}"
        messages = []

    messages.append({"role": "user", "content": message, "timestamp": now})

    # Build context from current person model
    from loom.person_model import get_person_model, get_model_gaps

    existing = get_person_model(store)
    if existing:
        model = existing["model"]
        gaps = get_model_gaps(model)
    else:
        from loom.person_model import _empty_model
        model = _empty_model()
        gaps = [{"dimension": d, "confidence": 0.0} for d in model]

    confidences = {}
    for dim_name, dim_data in model.items():
        if isinstance(dim_data, dict) and "confidence" in dim_data:
            confidences[dim_name] = dim_data["confidence"]

    confidence_summary = "\n".join(
        f"- {dim}: {conf:.1f}" for dim, conf in sorted(confidences.items(), key=lambda x: x[1])
    )
    gaps_summary = "\n".join(
        f"- {g['dimension']}: confidence {g['confidence']:.2f}" for g in gaps
    ) or "No major gaps identified."

    # Build enriched context with start session data + user answers
    extra_context = ""
    if start_context:
        # Include KG context from the start phase
        kg_ctx = start_context.get("kg_context", {})
        model_summary = start_context.get("model_summary", "")
        if model_summary:
            extra_context += f"\n## System Summary\n{model_summary}\n"
        if kg_ctx.get("inner_circle"):
            extra_context += "\n## Inner Circle (Tier 1-2 contacts from KG)\n"
            for c in kg_ctx["inner_circle"][:10]:
                name = c.get("name", "?")
                msgs = c.get("messages", 0)
                channels = c.get("channels", "")
                extra_context += f"- {name} ({msgs} messages, via {channels})\n"
        if kg_ctx.get("user_beliefs"):
            belief_lines = [f"- {b['summary']}" for b in kg_ctx["user_beliefs"][:8]]
            extra_context += f"\n## Beliefs About User\n" + "\n".join(belief_lines) + "\n"

        # If question generation failed, inject the partial response as context
        partial = start_context.get("partial_question_gen_response", "")
        if partial:
            extra_context += f"\n## Partial Analysis (from failed question generation)\n{partial}\n"

    if answers:
        answered = [a for a in answers if a.get("answer", "").strip()]
        if answered:
            extra_context += "\n## User's Answers to Intake Questions (DO NOT re-ask these)\n"
            for a in answered:
                q_text = a.get("question", "")
                a_text = a.get("answer", "")
                dim = a.get("dimension", "")
                extra_context += f"- [{dim}] Q: {q_text}\n  A: {a_text}\n"
            extra_context += "\nIMPORTANT: The above questions have been answered. Do NOT repeat them.\n"

    # System prompt is static (cacheable); dynamic context goes in first user message
    system_prompt = _PERSON_MODEL_CHAT_SYSTEM

    actions_taken = []

    try:
        from loom.llm.gemini import GeminiClient
        from loom.constants import PERSON_MODEL_CHAT_MODEL

        llm = GeminiClient()
        chat_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

        # Inject model context + start context into the first user message only
        if len(chat_messages) >= 1 and chat_messages[0]["role"] == "user":
            chat_messages[0] = {
                "role": "user",
                "content": _PERSON_MODEL_CONTEXT_TEMPLATE.format(
                    model_json=json.dumps(model, indent=2, default=str),
                    confidence_summary=confidence_summary,
                    gaps_summary=gaps_summary,
                    extra_context=extra_context,
                    user_message=chat_messages[0]["content"],
                ),
            }

        executor = _person_model_tool_executor(store)

        def tracking_executor(name: str, args: dict) -> dict:
            result = executor(name, args)
            if result.get("status") == "ok":
                actions_taken.append(result.get("action", name))
            return result

        response = llm.chat_with_tools(
            messages=chat_messages,
            tools=_person_model_chat_tools(),
            system=system_prompt,
            model=PERSON_MODEL_CHAT_MODEL,
            temperature=0.4,
            max_tokens=2048,
            tool_executor=tracking_executor,
            max_tool_rounds=5,
        )
        if not response:
            response = "I'm having trouble connecting. Please try again."

    except Exception as exc:
        logger.warning("Person model chat LLM error: %s", exc)
        response = f"Chat is temporarily unavailable: {exc}"

    messages.append({"role": "assistant", "content": response, "timestamp": int(time.time())})

    title = "Person Model Onboarding" if len(messages) <= 2 else ""
    store.put_cq_session(session_id, messages, title=title, session_type="person_model")

    result = {
        "session_id": session_id,
        "response": response,
        "message_count": len(messages),
    }
    if actions_taken:
        result["actions"] = actions_taken

    return result


register_tool(ToolDef(
    name="loom_person_model_chat",
    description="Multi-turn onboarding chat to fill in Person Model gaps. The AI walks through "
                "low-confidence dimensions, asking about family, routines, local habits, work patterns, "
                "and other life details. Answers are saved to user_corrections immediately.",
    permission="write",
    params=[
        ToolParam("message", "string", "User's message", required=True),
        ToolParam("session_id", "string", "Session ID for multi-turn (omit to start new)"),
        ToolParam("start_session_id", "string", "Session ID from loom_person_model_start (first message only)"),
        ToolParam("answers", "array", "User's answers from the start phase questions [{index, answer}]"),
    ],
    handler=handle_person_model_chat,
))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool 4: Direct field update
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def handle_update_person_model(
    store: LayeredGraphStore | None = None,
    dimension: str = "",
    field: str = "",
    value: str = "",
    **kwargs,
) -> dict:
    """Directly update a field in the person model (bypass chat)."""
    if not dimension or not field:
        return {"error": "dimension and field are required"}

    store = store or _get_store(**kwargs)

    from loom.person_model import update_person_model_field

    # Parse JSON values
    parsed_value = value
    try:
        parsed_value = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        pass

    return update_person_model_field(store, dimension, field, parsed_value)


register_tool(ToolDef(
    name="loom_update_person_model",
    description="Directly update a field in the Person Model. Writes to user_corrections "
                "which are preserved across re-estimation.",
    permission="write",
    params=[
        ToolParam("dimension", "string", "Dimension to update", required=True,
                  enum=["identity", "professional", "communication_fingerprint",
                        "relationship_map", "active_workstreams", "goals_and_values",
                        "life_architecture", "temporal_patterns", "domain_expertise",
                        "financial_landscape", "visibility_gaps"]),
        ToolParam("field", "string", "Field within the dimension", required=True),
        ToolParam("value", "string", "Value to set (JSON-encoded for arrays/objects)", required=True),
    ],
    handler=handle_update_person_model,
))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool 5: Finish — sync corrections to profile.yaml and return summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_PROFILE_YAML_SYNTHESIZER_SYSTEM = """\
You are the Loom Person Model Compiler. Your sole function is to take a raw JSON object \
containing a user's accumulated personal data and transform it into a structured, \
well-formatted profile.yaml document representing their "Computational Autobiography."

You do not converse with the user. You do not explain your work. You only output the \
final YAML content (no code fences, no commentary).

## Instructions
1. Analyze the Input: You receive a JSON object with the user's Person Model state and \
user corrections, plus their existing profile.yaml.
2. Map to Schema: Map all data points to the profile.yaml structure below.
3. Synthesize and Clean:
   - Resolve conflicting data by favoring user_corrections over estimated model data, \
     and existing profile.yaml over both for fields like emails/phones.
   - Group related entities logically (e.g., family members under immediate_family).
   - Omit fields where no data exists. No null values, empty arrays, or placeholders.
4. Formatting: Perfect YAML syntax, 2-space indentation, lists, key-value pairs.
5. PRESERVE all existing profile.yaml fields that have data (especially emails, phones, \
   timezone, sensitivity_mode).
6. NAMES: Always use FULL names exactly as they appear in the input data.
   Do NOT abbreviate last names to initials. "Sam Park" stays "Sam Park",
   not "Sam P." The user needs full names for cross-referencing with their messages.

## Schema Guidelines (profile.yaml)
Organize into these top-level keys if data is available:

* identity: name, preferred_name, emails (list), phones (list), timezone, location (city, neighborhood)
* professional: company, role, context, key_colleagues
* family_and_relationships: immediate_family (spouse, children with names/ages), \
  extended_network (frequent guests, caregivers), pets
* work_patterns: environment (home/office/hybrid), non_negotiable_blocks (list of \
  daily fixed commitments with times), deep_work_windows, wake_sleep_schedule
* routines: weekday_structure (list), weekend_structure (list)
* local_life_and_habits: dietary_preferences, staple_locations (coffee with order, \
  dining list), decompression (list of activities/places), regular_infrastructure \
  (barber, grocery, gym, etc.)
* hobbies_and_interests: media (shows, books, podcasts), communities_and_clubs, \
  active_pursuits
* travel_preferences: flight_class, airline_loyalty, seating, hotel_preferences, \
  constraints
* sensitivity_mode: cloud_all or local_sensitive
"""


def handle_person_model_finish(
    store: LayeredGraphStore | None = None,
    **kwargs,
) -> dict:
    """Sync person model user_corrections back to profile.yaml via LLM synthesis."""
    store = store or _get_store(**kwargs)

    from loom.person_model import get_person_model

    existing = get_person_model(store)
    if not existing:
        return {"status": "empty", "message": "No person model to sync."}

    model = existing.get("model", {})
    corrections = existing.get("user_corrections", {})

    # Read existing profile.yaml
    profile_path = Path.home() / ".loom" / "profile.yaml"
    existing_yaml = ""
    if profile_path.exists():
        existing_yaml = profile_path.read_text()

    # Build input for the synthesizer
    synth_input = json.dumps({
        "existing_profile_yaml": existing_yaml,
        "person_model": model,
        "user_corrections": corrections,
    }, indent=2, default=str)

    # Try LLM synthesis
    new_yaml = None
    try:
        from loom.llm.gemini import GeminiClient
        from loom.constants import PERSON_MODEL_CONSIGLIERE_MODEL

        llm = GeminiClient()
        new_yaml = llm.generate(
            prompt=synth_input,
            system=_PROFILE_YAML_SYNTHESIZER_SYSTEM,
            model=PERSON_MODEL_CONSIGLIERE_MODEL,
            temperature=0.1,
            max_tokens=8192,
        )
        if new_yaml:
            # Strip code fences if present
            new_yaml = new_yaml.strip()
            if new_yaml.startswith("```"):
                lines = new_yaml.split("\n")
                # Remove first line (```yaml) and last line (```)
                if lines[-1].strip() == "```":
                    lines = lines[1:-1]
                else:
                    lines = lines[1:]
                new_yaml = "\n".join(lines)
    except Exception as exc:
        logger.warning("Profile YAML synthesis failed, using fallback: %s", exc)

    if not new_yaml:
        # Fallback: simple merge
        new_yaml = _fallback_profile_yaml(model, corrections, existing_yaml)

    # Write to profile.yaml
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(new_yaml.rstrip() + "\n")

    # Parse back for display
    display_profile: dict = {}
    try:
        import yaml
        display_profile = yaml.safe_load(new_yaml) or {}
    except Exception:
        display_profile = {"raw": new_yaml}

    return {
        "status": "ok",
        "profile": display_profile,
        "profile_yaml": new_yaml,
        "message": "Profile updated successfully.",
    }


def _fallback_profile_yaml(model: dict, corrections: dict, existing_yaml: str) -> str:
    """Simple fallback if LLM synthesis is unavailable."""
    import yaml

    existing: dict = {}
    if existing_yaml:
        try:
            existing = yaml.safe_load(existing_yaml) or {}
        except Exception:
            existing = {}

    # Identity
    identity = model.get("identity", {})
    if identity.get("name"):
        existing.setdefault("name", identity["name"])
    if identity.get("emails"):
        existing.setdefault("emails", identity["emails"])
    if identity.get("phones"):
        existing.setdefault("phones", identity["phones"])
    if identity.get("timezone"):
        existing.setdefault("timezone", identity["timezone"])
    if identity.get("location"):
        existing.setdefault("city", identity["location"])

    # Family from corrections
    family_corr = corrections.get("relationship_map", {}).get("family", [])
    if family_corr:
        family_list = []
        for m in family_corr:
            if isinstance(m, dict):
                parts = [m.get("name", "")]
                role = m.get("role", "")
                age = m.get("age")
                if role:
                    detail = f"({role}"
                    if age:
                        detail += f", age {age}"
                    detail += ")"
                    parts.append(detail)
                family_list.append(" ".join(parts))
            else:
                family_list.append(str(m))
        existing["family"] = family_list

    # Routines
    la = corrections.get("life_architecture", {})
    if la.get("dietary_preferences"):
        existing["dietary_preferences"] = la["dietary_preferences"]

    lines = ["# Loom user profile", "# Updated by Person Model onboarding", ""]
    for key in sorted(existing.keys()):
        val = existing[key]
        if isinstance(val, list):
            lines.append(f"{key}:")
            for item in val:
                lines.append(f'  - "{item}"')
        elif isinstance(val, dict):
            lines.append(yaml.dump({key: val}, default_flow_style=False).rstrip())
        else:
            lines.append(f'{key}: "{val}"')
    lines.append("")
    return "\n".join(lines)


register_tool(ToolDef(
    name="loom_person_model_finish",
    description="Finish the Person Model onboarding: syncs user corrections back to profile.yaml "
                "and returns the merged profile for display.",
    permission="write",
    params=[],
    handler=handle_person_model_finish,
))
