"""Mock LLM client for testing.

Returns deterministic, reproducible responses. Every test file
imports from this module, never from ollama or gemini.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from alteris.constants import EMBEDDING_DIM
from alteris.llm.base import LLMClient


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Canned response fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MOCK_TRIAGE_RESPONSE: dict[str, Any] = {
    "score": 0.75,
    "reason": "Contains an action item with a deadline",
    "domain": "work",
    "universal_spheres": ["primary_vocational_execution"],
    "specific_topics": ["product_roadmap_planning"],
    "topics": ["product_roadmap_planning"],
    "entities": ["Example Corp"],
    "pii": [],
    "sensitivity": "sensitive",
    "commitment_type": "deadline",
}

MOCK_EXTRACTION_RESPONSE: dict[str, Any] = {
    "commitments": [
        {
            "what": "Send the updated proposal to the client",
            "who": "user",
            "whom": "client",
            "deadline": "2026-02-15",
            "commitment_type": "promise",
            "confidence": 0.85,
            "direction": "outbound",
        }
    ]
}

MOCK_SYNTHESIS_RESPONSE: dict[str, Any] = {
    "beliefs": [
        {
            "belief_type": "fact",
            "subject": "user",
            "summary": "User has a pending proposal deadline for Example Corp on Feb 15",
            "data": {
                "commitment": "send updated proposal",
                "deadline": "2026-02-15",
                "counterparty": "Example Corp",
            },
            "epistemic_level": "inference",
            "confidence": 0.8,
            "inference_chain": [
                "Email from user to client mentions 'will send by Friday'",
                "Calendar shows meeting with Example Corp next week",
            ],
        }
    ]
}

MOCK_GATE_RESPONSE: dict[str, Any] = {
    "actionable": True,
    "action_type": "user_owes_action",
    "reason": "Contains a direct request for the user to take action",
}

MOCK_GATE_NOT_ACTIONABLE_RESPONSE: dict[str, Any] = {
    "actionable": False,
    "reason": "Newsletter / promotional content with no user action required",
}

MOCK_SYNTHESIS_COMMITMENT_RESPONSE: dict[str, Any] = {
    "commitments": [
        {
            "type": "inbound_request",
            "action_type": "user_owes_action",
            "who": "user",
            "what": "send the updated proposal",
            "to_whom": "Kai",
            "direction": "direct_ask",
            "deadline": "2026-02-15",
            "status": "open",
            "priority": 2,
            "confidence": 0.85,
            "staleness_signal": "none",
            "provenance": "assigned_to_user",
            "note": "Kai asked in email thread",
            "source_message_id": "msg_0",
            "evidence_quote": "Please send the updated proposal by Friday",
            "speech_act": "request",
            "proposed_by": "Kai",
            "response_from": None,
            "response_type": "no_response",
            "when_committed": "msg_0",
            "next_action": "Open docs folder, export proposal as PDF, email to Kai",
            "has_named_actor": True,
            "has_concrete_deliverable": True,
            "has_temporal_constraint": True,
            "is_response_to_request": True,
        }
    ]
}

MOCK_LOGISTICS_GATE_RESPONSE: dict[str, Any] = {
    "logistics": True,
    "reason": "Contains reservation details",
}

MOCK_LOGISTICS_GATE_NOT_LOGISTICS_RESPONSE: dict[str, Any] = {
    "logistics": False,
    "reason": "No scheduling information",
}

MOCK_RELATIONAL_GATE_RESPONSE: dict[str, Any] = {
    "relational": True,
    "relationship_tier": "vocational_core_team",
    "reason": "Contains person role info",
}

MOCK_RELATIONAL_GATE_NOT_RELATIONAL_RESPONSE: dict[str, Any] = {
    "relational": False,
    "reason": "No person context",
}

MOCK_LOGISTICS_EXTRACTION_RESPONSE: dict[str, Any] = {
    "facts": [
        {
            "type": "reservation",
            "venue": "The Garden Table",
            "date": "2026-02-14",
            "time": "16:00",
            "party_size": 2,
            "confirmation": None,
        }
    ]
}

MOCK_RELATIONAL_EXTRACTION_RESPONSE: dict[str, Any] = {
    "people": [
        {
            "name": "Kai Tanaka",
            "relationship_tier": "vocational_core_team",
            "role": "co-founder",
            "organization": "Example Corp",
            "context": "Concerned about velocity, wants mass-market approach",
            "relationship_strength": "strong",
        }
    ]
}

MOCK_SURVEYOR_RESPONSE: dict[str, Any] = {
    "inquiry_vectors": [
        {
            "target": "Intention-Action Gap — Overdue commitments",
            "logic": "3 overdue commitments with no behavioral evidence. "
                     "Cannot distinguish dropped balls from quietly resolved tasks.",
            "severity": "CRITICAL",
            "doubt_zone": True,
            "scout_queries": [
                {"tool": "search_commitments", "args": {"keyword": "payment"}},
                {"tool": "search_events", "args": {"source": "safari", "keyword": "bank", "days_back": 7}},
            ],
        },
        {
            "target": "Rising Signal — New contact burst",
            "logic": "New contact with 12 events in 7 days, may be unstated dependency.",
            "severity": "HIGH",
            "doubt_zone": False,
            "scout_queries": [
                {"tool": "get_person_context", "args": {"name": "Kai", "days_back": 14}},
            ],
        },
        {
            "target": "Career Trajectory Ambiguity",
            "logic": "Career-related emails but no browser/app evidence of active pursuit.",
            "severity": "HIGH",
            "doubt_zone": True,
            "scout_queries": [
                {"tool": "search_events", "args": {"source": "mail", "keyword": "interview", "days_back": 14}},
            ],
        },
    ]
}

MOCK_CONSIGLIERE_RESPONSE: dict[str, Any] = {
    "findings": [
        {
            "vector": "Intention-Action Gap — Overdue commitments",
            "status": "ambiguous",
            "evidence_summary": "No ambient traces found for overdue payments.",
            "user_question": "I see 3 overdue payments but no bank activity. "
                             "Did you handle these through auto-pay or another device?",
            "blindspot_hypothesis": "Shadow Work — payment may have been made on "
                                    "a partner's device or through auto-pay.",
            "belief_updates": [
                {"subject": "payment_visibility", "action": "adjust_visibility",
                 "detail": "Finance domain visibility < 10%"},
            ],
        },
        {
            "vector": "Rising Signal — New contact burst",
            "status": "unresolved",
            "evidence_summary": "Kai Tanaka active in 12 WhatsApp messages, no commitments linked.",
            "user_question": "Kai has been very active. Is there something you need to prepare?",
            "blindspot_hypothesis": None,
            "belief_updates": [],
        },
        {
            "vector": "Career Trajectory Ambiguity",
            "status": "unresolved",
            "evidence_summary": "5 career emails but no browser prep activity.",
            "user_question": "Are you actively pursuing the interview, or should I deprioritize?",
            "blindspot_hypothesis": None,
            "belief_updates": [],
        },
    ],
    "blind_spots": [
        "Finance domain has < 10% visibility — system cannot track payments",
        "Family logistics may be handled on partner's device",
    ],
    "decision_dag": [
        {"state": "Q1", "question": "Did you pay the BofA bill?",
         "yes_action": "Close commitment + log auto-pay channel",
         "no_next": "Q2"},
        {"state": "Q2", "question": "Is it on auto-pay?",
         "yes_action": "Mark as auto-resolved, reduce future alerts",
         "no_next": "Flag as genuinely overdue"},
    ],
    "visibility_assessment": {
        "high_visibility_domains": ["email", "messaging", "meetings"],
        "low_visibility_domains": ["finance", "family logistics"],
        "recommendation": "Consider instrumenting banking apps via knowledgeC to improve coverage",
    },
    "work_rhythm_note": None,
}

MOCK_ORACLE_EXPANSION_RESPONSE: dict[str, Any] = {
    "interpretation": "User wants to find where they promised to send a proposal",
    "queries": [
        {"tool": "search_events", "args": {"keyword": "proposal", "source": "mail", "days_back": 30}},
        {"tool": "search_commitments", "args": {"keyword": "proposal"}},
    ],
}

MOCK_ORACLE_SYNTHESIS_RESPONSE: dict[str, Any] = {
    "answer": "You promised to send the updated proposal to Kai on Feb 12 via email. "
              "The commitment is tracked with deadline Feb 15.",
    "confidence": 0.85,
    "sources": [
        {"type": "event", "summary": "Email from user to Kai about proposal",
         "date": "2026-02-12", "source": "mail"},
        {"type": "commitment", "summary": "Send updated proposal by Feb 15",
         "date": "2026-02-15", "source": "claims"},
    ],
    "follow_up": "Would you like me to check if Kai has responded to the proposal?",
}

MOCK_ANTICIPATION_RESPONSE: dict[str, Any] = {
    "system_queries": [
        {"query_type": "recent_from_person", "params": {"name": "Sam"}, "reason": "Check recent messages"}
    ],
    "web_searches": [
        {"query": "Seattle weather forecast", "reason": "Outdoor meeting planned"}
    ],
    "user_questions": [
        {"event_subject": "Team Meeting", "question": "Do you have the updated deck?",
         "category": "MATERIALS", "confidence": "medium",
         "context_if_answered": "Prevents arriving unprepared"}
    ],
    "reassurances": [
        {"event_subject": "School Pickup", "note": "Confirmed pickup today."}
    ],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mock client
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MockLLMClient(LLMClient):
    """Mock LLM client that returns canned, deterministic responses.

    Args:
        embed_dim: Dimension of mock embedding vectors.
        default_response: Fallback text response for generate/chat.
        responses: Dict mapping prompt substrings to text responses.
        json_responses: Dict mapping prompt substrings to dict responses.
    """

    def __init__(
        self,
        embed_dim: int = EMBEDDING_DIM,
        default_response: str = "",
        responses: dict[str, str] | None = None,
        json_responses: dict[str, dict] | None = None,
    ):
        self.embed_dim = embed_dim
        self.default_response = default_response
        self.responses = responses or {}
        self.json_responses = json_responses or {}

        # Call tracking for test assertions
        self.embed_calls: list[list[str]] = []
        self.generate_calls: list[dict[str, Any]] = []
        self.chat_calls: list[dict[str, Any]] = []

    def embed(
        self,
        texts: list[str],
        model: str = "",
    ) -> list[list[float] | None]:
        """Return deterministic embeddings seeded by text hash."""
        self.embed_calls.append(texts)
        results: list[list[float] | None] = []
        for text in texts:
            results.append(self._deterministic_embedding(text))
        return results

    def web_search(
        self,
        query: str,
        model: str = "",
    ) -> str | None:
        """Return a canned web search result."""
        return f"[Mock web search result for: {query}] No results found."

    def generate(
        self,
        prompt: str,
        system: str = "",
        model: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        format_json: bool = False,
        thinking_budget: int | None = None,
        thinking_level: str | None = None,
        google_search: bool = False,
        cache_system: bool = False,
        response_schema: object | None = None,
    ) -> str | None:
        """Return a canned response based on prompt substring matching."""
        self.generate_calls.append({
            "prompt": prompt,
            "system": system,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "format_json": format_json,
        })

        if format_json or response_schema is not None:
            for substring, response in self.json_responses.items():
                if substring in prompt:
                    return json.dumps(response)

            # Route sandwich/oracle prompts to appropriate mocks
            if "inquiry_vectors" in (system or "") or "graph_ls" in prompt[:200]:
                return json.dumps(MOCK_SURVEYOR_RESPONSE)
            if "Scout Evidence" in prompt or "Blindspot" in (system or ""):
                return json.dumps(MOCK_CONSIGLIERE_RESPONSE)
            if "User question:" in prompt and "query tools" in (system or ""):
                return json.dumps(MOCK_ORACLE_EXPANSION_RESPONSE)
            if "Evidence" in prompt and "Oracle" in (system or ""):
                return json.dumps(MOCK_ORACLE_SYNTHESIS_RESPONSE)

            if "Anticipation Engine" in (system or ""):
                return json.dumps(MOCK_ANTICIPATION_RESPONSE)

            return json.dumps(MOCK_TRIAGE_RESPONSE)

        for substring, response in self.responses.items():
            if substring in prompt:
                return response

        return self.default_response

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        format_json: bool = False,
    ) -> str | None:
        """Extract last user message and delegate to generate."""
        self.chat_calls.append({
            "messages": messages,
            "system": system,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "format_json": format_json,
        })

        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        return self.generate(
            prompt=last_user_msg,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            format_json=format_json,
        )

    def _deterministic_embedding(self, text: str) -> list[float]:
        """Generate a reproducible embedding vector from text hash.

        Uses SHA-256 bytes to seed pseudo-random values, then
        L2-normalizes. Same text always produces the same vector.
        """
        h = hashlib.sha256(text.encode("utf-8", errors="replace")).digest()
        # Expand hash bytes to fill the embedding dimension
        raw: list[float] = []
        for i in range(self.embed_dim):
            byte_val = h[i % len(h)]
            # XOR with position to add variation beyond hash length
            mixed = byte_val ^ (i & 0xFF)
            raw.append((mixed / 127.5) - 1.0)

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in raw))
        if norm > 1e-8:
            raw = [x / norm for x in raw]
        return raw
