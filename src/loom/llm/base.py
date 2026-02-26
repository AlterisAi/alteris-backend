"""Abstract base class for LLM clients.

All LLM clients (Ollama, Gemini, Mock) implement this interface
so that triage, extraction, beliefs, and briefing modules can
swap between local, cloud, and mock models transparently.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        model: str = "",
    ) -> list[list[float] | None]:
        """Generate embeddings for a list of texts.

        Returns a list parallel to input. None for items that failed.
        """

    @abstractmethod
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
        """Generate text from a prompt. Returns None on failure.

        thinking_budget: explicit reasoning token budget (Gemini 2.x).
            None = model decides. 0 = disable. N = cap at N tokens.
        thinking_level: reasoning level for Gemini 3 models.
            None = model decides (defaults to high).
            One of: "minimal", "low", "medium", "high".
            Preferred over thinking_budget for Gemini 3.
        google_search: enable Gemini's google_search grounding tool.
            Model can fetch live web results during generation.
        cache_system: cache the system instruction for cheaper repeated use.
            Only effective with Gemini clients; ignored by others.
        response_schema: Gemini structured output schema (types.Schema).
            When provided, forces format_json=True and mechanically
            constrains output to match the schema (enum enforcement).
            Only effective with Gemini clients; ignored by others.
        """

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        format_json: bool = False,
    ) -> str | None:
        """Multi-turn chat. Returns None on failure.

        messages format: [{"role": "system"|"user"|"assistant", "content": "..."}]
        """

    async def agenerate(
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
        """Async generate text. Default: wraps sync generate in executor."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.generate(
                prompt=prompt, system=system, model=model,
                temperature=temperature, max_tokens=max_tokens,
                format_json=format_json, thinking_budget=thinking_budget,
                thinking_level=thinking_level, google_search=google_search,
                cache_system=cache_system, response_schema=response_schema,
            ),
        )

    def web_search(
        self,
        query: str,
        model: str = "",
    ) -> str | None:
        """Execute a grounded web search. Returns summary text or None.

        Default: not supported. Override in clients that support search
        (e.g. GeminiClient with google_search tool).
        """
        return None

    def generate_json(
        self,
        prompt: str,
        system: str = "",
        model: str = "",
        temperature: float = 0.1,
        max_tokens: int = 8192,
        thinking_budget: int | None = None,
        thinking_level: str | None = None,
        cache_system: bool = False,
    ) -> dict | None:
        """Generate and parse JSON output from a prompt.

        Multi-strategy parsing: direct -> strip fences -> extract braces.
        """
        raw = self.generate(
            prompt=prompt,
            system=system,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            format_json=True,
            thinking_budget=thinking_budget,
            thinking_level=thinking_level,
            cache_system=cache_system,
        )
        if not raw:
            return None

        cleaned = raw.strip()

        # Strip markdown fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        # Strip thinking tags
        if "<think>" in cleaned:
            parts = cleaned.split("</think>")
            cleaned = parts[-1].strip() if len(parts) > 1 else cleaned

        # Strategy 1: Direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find the outermost { ... } or [ ... ]
        first_brace = cleaned.find("{")
        first_bracket = cleaned.find("[")
        start_char = None
        end_char = None

        if first_brace >= 0 and (first_bracket < 0 or first_brace <= first_bracket):
            start_char, end_char = "{", "}"
            start_idx = first_brace
        elif first_bracket >= 0:
            start_char, end_char = "[", "]"
            start_idx = first_bracket
        else:
            logger.warning("No JSON structure found in LLM response: %s...", raw[:200])
            return None

        # Find matching close by counting nesting
        depth = 0
        end_idx = -1
        in_string = False
        escape_next = False
        for i in range(start_idx, len(cleaned)):
            ch = cleaned[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break

        if end_idx > start_idx:
            try:
                return json.loads(cleaned[start_idx:end_idx + 1])
            except json.JSONDecodeError:
                pass

        # Strategy 3: json_repair — handles truncated, malformed, or trailing-comma JSON
        try:
            from json_repair import repair_json
            repaired = repair_json(cleaned, return_objects=True)
            if isinstance(repaired, (dict, list)):
                logger.info("json_repair recovered %d-char response", len(cleaned))
                return repaired
        except Exception:
            pass

        logger.warning("Failed to parse LLM JSON response: %s...", raw[:200])
        return None
