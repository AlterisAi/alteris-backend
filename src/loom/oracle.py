"""Oracle — interactive retrieval mode for user-initiated questions.

When the user asks "Where did I say this?" or "How does X relate to Y?",
the Oracle:
  1. Uses a Pro model to translate the question into structured queries
  2. Executes queries deterministically against graph.db (same scout tools)
  3. Synthesizes results into a concise answer with provenance

Usage:
    from loom.oracle import ask_oracle
    result = ask_oracle(store, llm, "Where did I promise to send the proposal?")
    print(result["answer"])
"""

from __future__ import annotations

import json
import logging
import time

from loom.constants import SANDWICH_ORACLE_MODEL, SANDWICH_SCOUT_MAX_RESULTS
from loom.sandwich import (
    SCOUT_TOOLS,
    execute_scout_query,
)
from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)


ORACLE_QUERY_EXPANSION_SYSTEM = """\
You are the Oracle for a Personal Operating System backed by a knowledge graph.

The user is asking a question about their digital life. Your job is to translate \
their natural language question into structured database queries.

Available query tools:
- search_events: {"source": "mail|imessage|whatsapp|calendar|granola|slack|knowledgec|safari|chrome|notes|shell_history", "keyword": "...", "days_back": N, "limit": N}
  Source is optional (omit to search all). Keyword searches body, metadata, participants.
- search_commitments: {"keyword": "..."}
  Searches open commitments by keyword in what/who/to_whom fields.
- get_person_context: {"name": "...", "days_back": N}
  Gets full context for a person: events, commitments, beliefs, profile.
- temporal_cross_reference: {"date": "YYYY-MM-DD", "source": "...", "keyword": "...", "window_hours": N}
  Finds events near a date from a specific source.
- get_thread_messages: {"thread_id": "...", "limit": N}
  Gets messages from a specific thread.

Strategy:
- For "Where did I say X?" → search_events with keyword
- For "Who was at X?" → search_events for the event, then get_person_context
- For "How does X relate to Y?" → get_person_context for both, search_commitments
- For "What do I need to do for X?" → search_commitments + get_person_context
- For "When did X happen?" → search_events with keyword + temporal_cross_reference

Output Format:
{
  "interpretation": "What you understand the user is asking (1 sentence)",
  "queries": [
    {"tool": "search_events", "args": {"keyword": "proposal", "source": "mail", "days_back": 30}},
    {"tool": "get_person_context", "args": {"name": "Kai"}}
  ]
}"""

ORACLE_SYNTHESIS_SYSTEM = """\
You are the Oracle for a Personal Operating System. You have retrieved evidence \
from the user's knowledge graph to answer their question.

Rules:
1. Answer concisely but with full provenance — cite sources, dates, and people.
2. If the evidence is ambiguous, say so and explain what's unclear.
3. If no evidence was found, say "I couldn't find anything matching that" \
and suggest what the user might search for instead.
4. For relational queries ("How does X relate to Y?"), trace the path \
through common events, people, or beliefs.
5. Always include timestamps and source types so the user can verify.

Output Format:
{
  "answer": "Your concise answer with provenance",
  "confidence": 0.0-1.0,
  "sources": [
    {"type": "event|commitment|belief|person", "summary": "...", "date": "...", "source": "..."}
  ],
  "follow_up": "Optional: a suggested follow-up question"
}"""


def ask_oracle(
    store: LayeredGraphStore,
    llm,
    question: str,
    model: str = "",
    thinking_level: str | None = None,
) -> dict:
    """Answer a user question by querying the knowledge graph.

    Args:
        store: The graph store to query.
        llm: LLM client for query expansion and synthesis.
        question: The user's natural language question.
        model: Optional model override.
        thinking_level: Optional thinking level for Gemini 3.

    Returns:
        Dict with answer, confidence, sources, and metadata.
    """
    t0 = time.time()
    use_model = model or SANDWICH_ORACLE_MODEL
    conn = store.conn

    # Ensure dict row factory
    def dict_factory(cursor, row):
        return {col[0]: row[i] for i, col in enumerate(cursor.description)}
    conn.row_factory = dict_factory

    # Phase 1: Query expansion
    expansion = llm.generate_json(
        prompt=f"User question: {question}",
        system=ORACLE_QUERY_EXPANSION_SYSTEM,
        model=use_model,
        temperature=0.1,
        max_tokens=2048,
        thinking_level=thinking_level,
    )

    if not expansion:
        return {
            "answer": "I couldn't understand that question. Try rephrasing it.",
            "confidence": 0.0,
            "sources": [],
            "elapsed": time.time() - t0,
        }

    interpretation = expansion.get("interpretation", "")
    queries = expansion.get("queries", [])

    # Phase 2: Execute queries
    evidence = []
    for q in queries:
        result = execute_scout_query(conn, q)
        evidence.append(result)

    # Phase 3: Synthesize answer
    prompt_parts = [
        f"## User Question\n{question}\n\n",
        f"## Interpretation\n{interpretation}\n\n",
        "## Evidence\n\n",
        json.dumps(evidence, indent=2, default=str),
    ]

    synthesis = llm.generate_json(
        prompt="".join(prompt_parts),
        system=ORACLE_SYNTHESIS_SYSTEM,
        model=use_model,
        temperature=0.2,
        max_tokens=4096,
        thinking_level=thinking_level,
    )

    if not synthesis:
        synthesis = {
            "answer": "I found evidence but couldn't synthesize an answer.",
            "confidence": 0.0,
            "sources": [],
        }

    result = {
        "question": question,
        "interpretation": interpretation,
        "queries_executed": len(queries),
        "evidence_items": sum(
            len(e.get("results", []) if isinstance(e.get("results"), list) else [])
            for e in evidence
        ),
        "answer": synthesis.get("answer", "No answer"),
        "confidence": synthesis.get("confidence", 0.0),
        "sources": synthesis.get("sources", []),
        "follow_up": synthesis.get("follow_up"),
        "elapsed": time.time() - t0,
        "model": use_model,
        "raw_evidence": evidence,
    }

    return result


def format_oracle_output(result: dict) -> str:
    """Format the oracle result for terminal display."""
    lines = []

    lines.append(f"Q: {result.get('question', '?')}")
    lines.append(f"Interpretation: {result.get('interpretation', '?')}")
    lines.append(f"Queries: {result.get('queries_executed', 0)}, "
                 f"Evidence: {result.get('evidence_items', 0)}")
    lines.append("")

    lines.append(f"A: {result.get('answer', 'No answer')}")
    lines.append(f"Confidence: {result.get('confidence', 0):.0%}")

    sources = result.get("sources", [])
    if sources:
        lines.append("")
        lines.append("Sources:")
        for s in sources[:5]:
            src_type = s.get("type", "?")
            summary = s.get("summary", "")
            date = s.get("date", "")
            source = s.get("source", "")
            lines.append(f"  [{src_type}] {summary}")
            if date or source:
                lines.append(f"    {source} | {date}")

    follow_up = result.get("follow_up")
    if follow_up:
        lines.append("")
        lines.append(f"Follow-up: {follow_up}")

    lines.append("")
    lines.append(f"Elapsed: {result.get('elapsed', 0):.1f}s ({result.get('model', '?')})")

    return "\n".join(lines)
