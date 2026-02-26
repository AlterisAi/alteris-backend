"""Pro-Lite-Pro sandwich pipeline for blind spot discovery.

Three-phase architecture:
  Phase 1 (Surveyor): Pro model surveys graph_ls, identifies inquiry vectors
  Phase 2 (Scout): Deterministic SQL queries against graph.db
  Phase 3 (Consigliere): Pro model synthesizes evidence into user questions

The Surveyor outputs structured scout_queries (tool name + args), so Phase 2
requires no LLM — just SQL execution. This keeps costs to 2 Pro calls total.

Usage:
    from loom.sandwich import run_sandwich
    result = run_sandwich(store, llm)
    print(result["output"])
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from loom.constants import (
    SANDWICH_CONSIGLIERE_MODEL,
    SANDWICH_MAX_INQUIRY_VECTORS,
    SANDWICH_SCOUT_CONTEXT_MAX_CHARS,
    SANDWICH_SCOUT_MAX_RESULTS,
    SANDWICH_SURVEYOR_MODEL,
)
from loom.graph_ls import generate_graph_ls
from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)

_7D = 7 * 86400
_30D = 30 * 86400


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Scout tools — deterministic database queries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def scout_search_events(
    conn, source: str = "", keyword: str = "",
    days_back: int = 30, limit: int = 0,
) -> list[dict]:
    """Search events by source and keyword."""
    limit = limit or SANDWICH_SCOUT_MAX_RESULTS
    now = int(time.time())
    since = now - days_back * 86400
    kw = f"%{keyword}%" if keyword else "%"

    params: list = []
    clauses = ["e.timestamp > ?"]
    params.append(since)

    if source:
        clauses.append("e.source = ?")
        params.append(source)

    clauses.append(
        "(e.raw_content LIKE ? OR e.metadata LIKE ? OR e.participants LIKE ?)"
    )
    params.extend([kw, kw, kw])

    sql = f"""
        SELECT e.id, e.source, e.event_type, e.timestamp,
               SUBSTR(e.raw_content, 1, 300) as body_preview,
               e.metadata->>'subject' as subject,
               e.metadata->>'thread_id' as thread_id,
               e.participants
        FROM events e
        WHERE {' AND '.join(clauses)}
        ORDER BY e.timestamp DESC
        LIMIT ?
    """
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "source": r["source"],
            "type": r["event_type"],
            "time": datetime.fromtimestamp(
                r["timestamp"], tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M"),
            "subject": r.get("subject"),
            "body": r.get("body_preview"),
            "thread_id": r.get("thread_id"),
            "participants": r.get("participants"),
        })
    return results


def scout_search_commitments(conn, keyword: str = "") -> list[dict]:
    """Search open commitments by keyword."""
    kw = f"%{keyword}%"
    rows = conn.execute("""
        SELECT json_extract(object, '$.what') as what,
               json_extract(object, '$.who') as who,
               json_extract(object, '$.to_whom') as to_whom,
               json_extract(object, '$.deadline') as deadline,
               json_extract(object, '$.type') as ctype,
               json_extract(object, '$.direction') as direction,
               json_extract(object, '$.status') as status,
               confidence, subject as source_event_id
        FROM claims
        WHERE claim_type = 'commitment'
          AND (superseded_by IS NULL OR superseded_by = '')
          AND json_extract(object, '$.status') = 'open'
          AND (json_extract(object, '$.what') LIKE ?
               OR json_extract(object, '$.who') LIKE ?
               OR json_extract(object, '$.to_whom') LIKE ?)
        ORDER BY confidence DESC
        LIMIT ?
    """, (kw, kw, kw, SANDWICH_SCOUT_MAX_RESULTS)).fetchall()

    return [dict(r) for r in rows]


def scout_get_person_context(
    conn, name: str, days_back: int = 30,
) -> dict:
    """Get all known context for a person."""
    now = int(time.time())
    since = now - days_back * 86400
    name_like = f"%{name}%"

    # Find person
    person = conn.execute(
        "SELECT person_id, canonical_name FROM persons WHERE canonical_name LIKE ? LIMIT 1",
        (name_like,),
    ).fetchone()

    if not person:
        return {"error": f"Person '{name}' not found"}

    pid = person["person_id"]
    result: dict = {
        "person_id": pid,
        "name": person["canonical_name"],
    }

    # Recent events
    events = conn.execute("""
        SELECT e.source, e.event_type, e.timestamp,
               SUBSTR(e.raw_content, 1, 200) as body,
               e.metadata->>'subject' as subject
        FROM events e
        JOIN event_persons ep ON e.id = ep.event_id
        WHERE ep.person_id = ? AND e.timestamp > ?
        ORDER BY e.timestamp DESC
        LIMIT 15
    """, (pid, since)).fetchall()
    result["recent_events"] = [
        {
            "source": e["source"],
            "type": e["event_type"],
            "time": datetime.fromtimestamp(
                e["timestamp"], tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M"),
            "subject": e.get("subject"),
            "body": e.get("body"),
        }
        for e in events
    ]

    # Commitments
    commits = conn.execute("""
        SELECT json_extract(object, '$.what') as what,
               json_extract(object, '$.deadline') as deadline,
               json_extract(object, '$.direction') as direction,
               confidence
        FROM claims
        WHERE claim_type = 'commitment'
          AND (superseded_by IS NULL OR superseded_by = '')
          AND json_extract(object, '$.status') = 'open'
          AND (LOWER(json_extract(object, '$.who')) LIKE ?
               OR LOWER(json_extract(object, '$.to_whom')) LIKE ?)
        LIMIT 10
    """, (name_like.lower(), name_like.lower())).fetchall()
    result["open_commitments"] = [dict(c) for c in commits]

    # Beliefs
    beliefs = conn.execute("""
        SELECT belief_type, summary, confidence,
               json_extract(data, '$.context') as context
        FROM beliefs
        WHERE status = 'active' AND subject LIKE ?
        ORDER BY confidence DESC LIMIT 5
    """, (f"%{pid}%",)).fetchall()
    result["beliefs"] = [dict(b) for b in beliefs]

    # Profile
    profile = conn.execute(
        "SELECT tier, message_count, channels, user_initiated_ratio "
        "FROM person_profiles WHERE person_id = ?",
        (pid,),
    ).fetchone()
    if profile:
        result["profile"] = dict(profile)

    return result


def scout_temporal_xref(
    conn, date_str: str, source: str,
    keyword: str = "", window_hours: int = 24,
) -> list[dict]:
    """Find events near a date from a specific source."""
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        center_ts = int(dt.timestamp())
    except ValueError:
        return [{"error": f"Invalid date: {date_str}"}]

    window = window_hours * 3600
    params: list = [source, center_ts - window, center_ts + window]
    kw_clause = ""
    if keyword:
        kw_clause = "AND (e.raw_content LIKE ? OR e.metadata LIKE ?)"
        params.extend([f"%{keyword}%", f"%{keyword}%"])

    rows = conn.execute(f"""
        SELECT e.id, e.source, e.event_type, e.timestamp,
               SUBSTR(e.raw_content, 1, 300) as body,
               e.metadata->>'subject' as subject
        FROM events e
        WHERE e.source = ?
          AND e.timestamp >= ? AND e.timestamp <= ?
          {kw_clause}
        ORDER BY e.timestamp
        LIMIT ?
    """, params + [SANDWICH_SCOUT_MAX_RESULTS]).fetchall()

    return [
        {
            "source": r["source"],
            "type": r["event_type"],
            "time": datetime.fromtimestamp(
                r["timestamp"], tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M"),
            "subject": r.get("subject"),
            "body": r.get("body"),
        }
        for r in rows
    ]


def scout_get_thread(conn, thread_id: str, limit: int = 20) -> list[dict]:
    """Get messages from a thread."""
    rows = conn.execute("""
        SELECT e.id, e.source, e.timestamp,
               SUBSTR(e.raw_content, 1, 500) as body,
               e.participants,
               e.metadata->>'subject' as subject
        FROM events e
        WHERE e.metadata->>'thread_id' = ?
        ORDER BY e.timestamp DESC
        LIMIT ?
    """, (thread_id, limit)).fetchall()

    return [
        {
            "source": r["source"],
            "time": datetime.fromtimestamp(
                r["timestamp"], tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M"),
            "subject": r.get("subject"),
            "body": r.get("body"),
            "participants": r.get("participants"),
        }
        for r in rows
    ]


def compute_visibility_index(conn) -> dict:
    """Compute per-domain visibility coefficients.

    V(domain) = commitments_with_ambient_trace / total_commitments_in_domain

    Low V means the system is structurally blind to that domain.
    Used by the Surveyor's Doubt Filter and Consigliere's Observability Doubt.
    """
    now = int(time.time())

    # Get all resolved/stale commitments grouped by domain-like keywords
    domains = {
        "finance": ["payment", "bill", "bank", "invoice", "tax", "insurance",
                     "tuition", "rent", "mortgage"],
        "family": ["child", "kid", "school", "camp", "preschool", "family",
                    "tuition", "doctor", "pediatr"],
        "career": ["interview", "resume", "job", "career", "recruiter",
                    "offer", "salary", "Meta", "LinkedIn"],
        "health": ["doctor", "appointment", "prescription", "pharmacy",
                    "health", "dentist", "gym"],
        "work": ["proposal", "review", "deploy", "release", "sprint",
                 "presentation", "deck", "report"],
    }

    result = {}
    for domain, keywords in domains.items():
        # Count total commitments in this domain
        keyword_clauses = " OR ".join(
            f"LOWER(json_extract(object, '$.what')) LIKE '%{kw.lower()}%'"
            for kw in keywords
        )
        total = conn.execute(f"""
            SELECT COUNT(*) as c FROM claims
            WHERE claim_type = 'commitment'
              AND ({keyword_clauses})
        """).fetchone()["c"]

        if total == 0:
            continue

        # Count commitments that had ANY ambient event near their deadline
        traced = 0
        commit_rows = conn.execute(f"""
            SELECT json_extract(object, '$.deadline') as deadline,
                   json_extract(object, '$.what') as what
            FROM claims
            WHERE claim_type = 'commitment'
              AND ({keyword_clauses})
              AND json_extract(object, '$.deadline') IS NOT NULL
            LIMIT 50
        """).fetchall()

        for cr in commit_rows:
            deadline = cr["deadline"]
            what = cr["what"] or ""
            if not deadline:
                continue
            # Check for ambient traces near the deadline
            try:
                from datetime import datetime as dt_cls
                dl = dt_cls.fromisoformat(deadline)
                dl_ts = int(dl.replace(tzinfo=timezone.utc).timestamp())
            except (ValueError, TypeError):
                continue

            # Look for any ambient event within 48h of deadline
            trace = conn.execute("""
                SELECT 1 FROM events
                WHERE source IN ('knowledgec','safari','chrome','notes','shell_history')
                  AND timestamp >= ? AND timestamp <= ?
                LIMIT 1
            """, (dl_ts - 2 * 86400, dl_ts + 2 * 86400)).fetchone()
            if trace:
                traced += 1

        v = traced / total if total > 0 else 0.0
        result[domain] = {
            "total_commitments": total,
            "traced": traced,
            "visibility": round(v, 2),
            "assessment": (
                "HIGH" if v > 0.5
                else "MEDIUM" if v > 0.2
                else "LOW — system is likely blind to this domain"
            ),
        }

    return result


# Scout tool dispatch
SCOUT_TOOLS = {
    "search_events": scout_search_events,
    "search_commitments": scout_search_commitments,
    "get_person_context": scout_get_person_context,
    "temporal_cross_reference": scout_temporal_xref,
    "get_thread_messages": scout_get_thread,
}


def execute_scout_query(conn, query: dict) -> dict:
    """Execute a single scout query and return results."""
    tool_name = query.get("tool", "")
    args = dict(query.get("args", {}))

    tool_fn = SCOUT_TOOLS.get(tool_name)
    if not tool_fn:
        return {"error": f"Unknown scout tool: {tool_name}"}

    # Normalize LLM arg name variants to match function signatures
    if "date" in args and "date_str" not in args:
        args["date_str"] = args.pop("date")
    if "thread_id" not in args and "thread" in args:
        args["thread_id"] = args.pop("thread")

    try:
        result = tool_fn(conn, **args)
        return {"tool": tool_name, "args": args, "results": result}
    except Exception as exc:
        logger.warning("Scout query %s failed: %s", tool_name, exc)
        return {"tool": tool_name, "args": args, "error": str(exc)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SURVEYOR_SYSTEM = """\
You are the Surveyor for a Personal Operating System.

Objective: Analyze the provided 7-tier graph_ls to identify up to 20 \
"Information Foraging" directions that will reveal the user's blind spots. \
Generate as many distinct, non-overlapping vectors as the data supports.

Strategy:
1. Focus on Tier 7 (Epistemic Gaps): These are your primary targets. \
Each gap has scout_instructions that tell you what to look for.
2. Cross-Reference Tier 3 (Action Matrix) with Tier 6 (Anomalies): \
Look for "Orphan" commitments that lack behavioral evidence.
3. Identify "Rising Signals" (Tier 2): Determine if a burst contact \
is a potential unstated dependency for a Tier 3 goal.
4. Check work_rhythm in meta: If the user's first/last interaction times \
suggest unusual patterns (late nights, early mornings), flag it.

DOUBT FILTER (Critical):
Identify nodes where Commitment Density is high but Ambient Trace Density \
is low. These are "High-Doubt Zones" — the system may be structurally blind \
to progress in these areas. For High-Doubt Zones, instruct the Scout to \
look for the CHANNEL itself (was there ANY activity from this source?) \
rather than the specific task. If ambient_sources_present shows "NONE", \
every commitment is automatically a High-Doubt Zone.

Available scout query tools:
- search_events: {"source": "mail|imessage|whatsapp|calendar|granola|slack|knowledgec|safari|chrome|notes|shell_history", "keyword": "...", "days_back": N, "limit": N}
- search_commitments: {"keyword": "..."}
- get_person_context: {"name": "...", "days_back": N}
- temporal_cross_reference: {"date": "YYYY-MM-DD", "source": "...", "keyword": "...", "window_hours": N}
- get_thread_messages: {"thread_id": "...", "limit": N}

Output Format:
Return a JSON object with key "inquiry_vectors" containing up to 20 items:
{
  "inquiry_vectors": [
    {
      "target": "Short title of what you're investigating",
      "logic": "Why this matters (1-2 sentences)",
      "severity": "CRITICAL|HIGH|MEDIUM",
      "doubt_zone": true/false,
      "scout_queries": [
        {"tool": "search_events", "args": {"source": "safari", "keyword": "bofa.com", "days_back": 7}},
        {"tool": "temporal_cross_reference", "args": {"date": "2026-02-17", "source": "knowledgec"}}
      ]
    }
  ]
}"""

CONSIGLIERE_SYSTEM = """\
You are the Consigliere for a Personal Operating System. \
Your role is to bridge the "Intention-Action Gap" for the user.

Objective: Synthesize the gathered evidence and produce actionable intelligence.

ABSENCE-OF-EVIDENCE REASONING (Critical):
- Implicit Resolution: If a commitment is past-due but ambient traces show \
high activity near the deadline (bank app usage, browser visits, git pushes), \
treat as "potentially_resolved" and ask a validation question instead of nagging.
- Procrastination Trigger: If a Priority-1 commitment exists but ambient \
traces show distraction-heavy activity (YouTube, social media), flag as \
"Focus Without Intention."
- Conflict Promotion: If ambient traces show the user at Location X but \
calendar says Location Y, promote to structural anomaly.

BLINDSPOT AUDIT:
If the Scout found ZERO traces for a resolved/overdue commitment, generate \
a "Blindspot Hypothesis" — WHY the system was blind to it:
- "Shadow Work": Done in a non-instrumented environment (paper, local terminal, \
private browser, partner's device).
- "Delegation": Completed by asking someone else via an unmonitored channel \
(verbal sync, phone call).
- "Auto-Resolution": The problem went away on its own (auto-pay, self-healing).
- "NLP Failure": The data WAS there but the system's keyword matching missed it \
(e.g., "Thanks for the help!" not linked to "Package system for testing").

OBSERVABILITY DOUBT:
If this is a "doubt_zone" vector, state your uncertainty explicitly: \
"I am historically blind to this type of task. I will rely on user-polling \
rather than ambient-sensing for this domain."

Rules:
1. If the Scout found evidence that a task was resolved, DO NOT ASK. \
Mark as "resolved" with evidence.
2. If evidence is missing, craft a micro-query: "I found X. Does that mean Y?"
3. Look for the "Missing 20%" — activities without matching commitments.
4. If work_rhythm shows unusual patterns, note it as context.

Output Format:
{
  "findings": [
    {
      "vector": "Title from Surveyor",
      "status": "resolved|potentially_resolved|unresolved|ambiguous",
      "evidence_summary": "What the Scout found (or didn't)",
      "user_question": "Micro-query for the user (null if resolved)",
      "blindspot_hypothesis": "Why the system might be blind (null if evidence found)",
      "belief_updates": [
        {"subject": "...", "action": "resolve|flag|create|adjust_visibility", "detail": "..."}
      ]
    }
  ],
  "blind_spots": [
    "Things the user likely hasn't thought about, based on evidence gaps"
  ],
  "decision_dag": [
    {
      "state": "Q1",
      "question": "Did you finish [Task]?",
      "yes_action": "Close commitment",
      "no_next": "Q2"
    }
  ],
  "visibility_assessment": {
    "high_visibility_domains": ["email", "messaging"],
    "low_visibility_domains": ["finance", "family logistics"],
    "recommendation": "Consider instrumenting [X] to improve coverage"
  },
  "work_rhythm_note": "Optional observation about work patterns"
}"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase execution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def phase1_surveyor(
    llm, graph_ls_data: dict, model: str = "",
    thinking_level: str | None = None,
) -> list[dict]:
    """Phase 1: Survey graph_ls and produce inquiry vectors.

    Returns a list of inquiry_vectors, each with target, logic, scout_queries.
    """
    use_model = model or SANDWICH_SURVEYOR_MODEL
    prompt = (
        "Here is the complete graph_ls of the user's knowledge graph:\n\n"
        + json.dumps(graph_ls_data, indent=2, default=str)
    )

    result = llm.generate_json(
        prompt=prompt,
        system=SURVEYOR_SYSTEM,
        model=use_model,
        temperature=0.3,
        max_tokens=16384,
        thinking_level=thinking_level,
    )

    if not result:
        logger.error("Surveyor returned no result")
        return []

    vectors = result.get("inquiry_vectors", [])
    if not isinstance(vectors, list):
        logger.error("Surveyor returned non-list inquiry_vectors: %s", type(vectors))
        return []

    # Cap at max
    return vectors[:SANDWICH_MAX_INQUIRY_VECTORS]


def phase2_scout(conn, inquiry_vectors: list[dict]) -> list[dict]:
    """Phase 2: Execute scout queries deterministically.

    For each inquiry vector, runs all scout_queries and collects raw results.
    Returns a list parallel to inquiry_vectors with results attached.
    """
    scout_results = []
    total_chars = 0

    for vector in inquiry_vectors:
        queries = vector.get("scout_queries", [])
        query_results = []

        for q in queries:
            if total_chars >= SANDWICH_SCOUT_CONTEXT_MAX_CHARS:
                break
            result = execute_scout_query(conn, q)
            result_str = json.dumps(result, default=str)
            total_chars += len(result_str)
            query_results.append(result)

        scout_results.append({
            "target": vector.get("target", ""),
            "logic": vector.get("logic", ""),
            "severity": vector.get("severity", "MEDIUM"),
            "evidence": query_results,
        })

    return scout_results


def phase3_consigliere(
    llm, scout_results: list[dict], graph_ls_meta: dict,
    model: str = "", thinking_level: str | None = None,
) -> dict:
    """Phase 3: Synthesize evidence into findings and user questions.

    Returns the Consigliere's output dict.
    """
    use_model = model or SANDWICH_CONSIGLIERE_MODEL

    # Build the context packet
    prompt_parts = [
        "## Graph Summary\n",
        json.dumps(graph_ls_meta, indent=2, default=str),
        "\n\n## Scout Evidence\n\n",
    ]
    for i, sr in enumerate(scout_results, 1):
        prompt_parts.append(f"### Vector {i}: {sr['target']}\n")
        prompt_parts.append(f"Logic: {sr['logic']}\n")
        prompt_parts.append(f"Severity: {sr['severity']}\n")
        prompt_parts.append("Evidence:\n")
        prompt_parts.append(json.dumps(sr["evidence"], indent=2, default=str))
        prompt_parts.append("\n\n")

    prompt = "".join(prompt_parts)

    # Truncate if needed
    if len(prompt) > SANDWICH_SCOUT_CONTEXT_MAX_CHARS:
        prompt = prompt[:SANDWICH_SCOUT_CONTEXT_MAX_CHARS] + "\n\n[... truncated]"

    result = llm.generate_json(
        prompt=prompt,
        system=CONSIGLIERE_SYSTEM,
        model=use_model,
        temperature=0.3,
        max_tokens=32768,
        thinking_level=thinking_level,
    )

    if not result:
        return {"findings": [], "blind_spots": [], "error": "Consigliere returned no result"}

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Output formatting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def format_sandwich_output(result: dict) -> str:
    """Format the sandwich result for terminal display."""
    lines = []

    # Phase 1 summary
    vectors = result.get("inquiry_vectors", [])
    p1_elapsed = result.get("phase1_elapsed", 0)
    lines.append(f"Phase 1: Surveyor ({result.get('surveyor_model', '?')})")
    lines.append(f"  {len(vectors)} inquiry vectors identified in {p1_elapsed:.1f}s")
    for i, v in enumerate(vectors, 1):
        lines.append(f"  Vector {i}: {v.get('target', '?')}")
        lines.append(f"    Logic: {v.get('logic', '?')}")
        q_count = len(v.get("scout_queries", []))
        lines.append(f"    Scout queries: {q_count}")
    lines.append("")

    # Phase 2 summary
    scout = result.get("scout_results", [])
    p2_elapsed = result.get("phase2_elapsed", 0)
    total_results = sum(
        len(e.get("results", []) if isinstance(e.get("results"), list) else [])
        for s in scout
        for e in s.get("evidence", [])
    )
    lines.append(f"Phase 2: Scout (deterministic SQL)")
    total_queries = sum(len(s.get("evidence", [])) for s in scout)
    lines.append(f"  {total_queries} queries executed in {p2_elapsed:.1f}s")
    lines.append(f"  {total_results} results retrieved")
    lines.append("")

    # Phase 3 findings
    consigliere = result.get("consigliere", {})
    p3_elapsed = result.get("phase3_elapsed", 0)
    findings = consigliere.get("findings", [])
    lines.append(f"Phase 3: Consigliere ({result.get('consigliere_model', '?')})")
    lines.append(f"  {len(findings)} findings synthesized in {p3_elapsed:.1f}s")
    lines.append("")

    for i, f in enumerate(findings, 1):
        status = f.get("status", "?").upper()
        status_marker = {
            "RESOLVED": "[RESOLVED]",
            "UNRESOLVED": "[UNRESOLVED]",
            "AMBIGUOUS": "[AMBIGUOUS]",
        }.get(status, f"[{status}]")

        lines.append(f"  Finding {i}: {f.get('vector', '?')} {status_marker}")
        lines.append(f"    Evidence: {f.get('evidence_summary', 'n/a')}")

        question = f.get("user_question")
        if question:
            lines.append(f"    Question: {question}")

        updates = f.get("belief_updates", [])
        for u in updates:
            lines.append(
                f"    -> {u.get('action', '?')}: {u.get('subject', '?')} "
                f"— {u.get('detail', '')}"
            )
        lines.append("")

    # Blind spots
    blind_spots = consigliere.get("blind_spots", [])
    if blind_spots:
        lines.append("  Blind Spots:")
        for bs in blind_spots:
            lines.append(f"    - {bs}")
        lines.append("")

    # Decision DAG
    dag = consigliere.get("decision_dag", [])
    if dag:
        lines.append("  Decision DAG:")
        for node in dag:
            q = node.get("question", "?")
            yes = node.get("yes_action", "?")
            no = node.get("no_next", "?")
            lines.append(f"    {node.get('state', '?')}: {q}")
            lines.append(f"      Yes -> {yes}")
            lines.append(f"      No  -> {no}")
        lines.append("")

    # Visibility assessment
    vis = consigliere.get("visibility_assessment", {})
    if vis:
        low = vis.get("low_visibility_domains", [])
        if low:
            lines.append(f"  Low Visibility Domains: {', '.join(low)}")
        rec = vis.get("recommendation")
        if rec:
            lines.append(f"  Recommendation: {rec}")
        lines.append("")

    # Work rhythm note
    rhythm_note = consigliere.get("work_rhythm_note")
    if rhythm_note:
        lines.append(f"  Work Rhythm: {rhythm_note}")
        lines.append("")

    # Total
    total = result.get("total_elapsed", 0)
    lines.append(f"Total elapsed: {total:.1f}s")

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main entry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_sandwich(
    store: LayeredGraphStore,
    llm,
    model: str = "",
    thinking_level: str | None = None,
) -> dict:
    """Run the full Pro-Lite-Pro sandwich pipeline.

    Returns a dict with all phase results plus formatted output.
    """
    t0 = time.time()
    conn = store.conn

    # Ensure dict row factory
    def dict_factory(cursor, row):
        return {col[0]: row[i] for i, col in enumerate(cursor.description)}
    conn.row_factory = dict_factory

    # Phase 1: Generate graph_ls, compute visibility, and survey it
    graph_ls_data = generate_graph_ls(store)
    graph_ls_data["visibility_index"] = compute_visibility_index(conn)

    t1 = time.time()
    inquiry_vectors = phase1_surveyor(
        llm, graph_ls_data, model=model, thinking_level=thinking_level,
    )
    phase1_elapsed = time.time() - t1

    if not inquiry_vectors:
        return {
            "error": "Surveyor produced no inquiry vectors",
            "graph_ls": graph_ls_data,
            "total_elapsed": time.time() - t0,
            "output": "Surveyor produced no inquiry vectors. Check LLM connectivity.",
        }

    # Phase 2: Execute scout queries
    t2 = time.time()
    scout_results = phase2_scout(conn, inquiry_vectors)
    phase2_elapsed = time.time() - t2

    # Phase 3: Synthesize
    t3 = time.time()
    consigliere = phase3_consigliere(
        llm, scout_results,
        graph_ls_meta=graph_ls_data.get("meta", {}),
        model=model, thinking_level=thinking_level,
    )
    phase3_elapsed = time.time() - t3

    result = {
        "inquiry_vectors": inquiry_vectors,
        "scout_results": scout_results,
        "consigliere": consigliere,
        "surveyor_model": model or SANDWICH_SURVEYOR_MODEL,
        "consigliere_model": model or SANDWICH_CONSIGLIERE_MODEL,
        "phase1_elapsed": phase1_elapsed,
        "phase2_elapsed": phase2_elapsed,
        "phase3_elapsed": phase3_elapsed,
        "total_elapsed": time.time() - t0,
        "graph_ls": graph_ls_data,
    }

    result["output"] = format_sandwich_output(result)
    return result
