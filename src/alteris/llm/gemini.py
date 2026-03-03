"""Gemini client for cloud synthesis and extraction.

Provides the same LLMClient interface as OllamaClient but routes
to Google's Gemini API. Used for non-sensitive content only.

The API key is loaded from the GEMINI_API_KEY environment variable
or from ~/.alteris/config.json.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from alteris.llm.base import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"


class SpendLimitError(Exception):
    """Raised when the daily spend limit is exceeded for the bundled key."""

    def __init__(self, daily_total: float, limit: float):
        self.daily_total = daily_total
        self.limit = limit
        super().__init__(
            f"Daily Gemini spend limit reached (${daily_total:.4f} / ${limit:.2f}). "
            f"To continue, add your own Gemini API key in Settings, or wait until tomorrow."
        )


def _parse_retry_delay(exc_str: str) -> float | None:
    """Extract retryDelay seconds from a Gemini 429 error message."""
    m = re.search(r"retryDelay.*?(\d+(?:\.\d+)?)\s*s", exc_str, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", exc_str, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


class DailyQuotaExhaustedError(Exception):
    """Raised when the Gemini daily API quota is exhausted. Don't retry."""
    pass


def _is_daily_quota_error(exc_str: str) -> bool:
    """Check if a 429 error is a daily quota exhaustion (not per-minute)."""
    return "perday" in exc_str.replace(" ", "").lower()


def _resolve_model(requested: str, default: str) -> str:
    """Return the requested model if it's a Gemini model, else fall back."""
    if not requested:
        return default
    if requested.startswith("gemini-") or requested.startswith("models/"):
        return requested
    return default


_REQUEST_TIMEOUT_MS = 300_000  # 300 seconds — Gemini 3 Flash needs headroom for long threads


class GeminiClient(LLMClient):
    """LLM client for Google's Gemini API.

    Supports embedding, generation, and chat via the google-genai SDK.
    All methods are synchronous wrappers suitable for thread-pool execution.
    """

    def __init__(self, model: str = DEFAULT_MODEL, store: Any = None, has_own_key: bool = False):
        self.model = model
        self._api_key: str | None = None
        self._client: Any = None
        self._async_client: Any = None
        self._store = store
        self._has_own_key = has_own_key
        self._cache_map: dict[str, tuple[str, float]] = {}  # hash → (cache_name, expiry_ts)

    def _get_key(self) -> str:
        if self._api_key:
            return self._api_key

        key = os.environ.get("GEMINI_API_KEY")

        if not key:
            from alteris.constants import load_config
            key = load_config().get("gemini_api_key", "")

        if not key:
            # Try macOS Keychain (app stores under ai.alteris.app)
            for service in ("ai.alteris.app", "alteris-listener", "loom"):
                try:
                    result = subprocess.run(
                        ["security", "find-generic-password",
                         "-a", "gemini", "-s", service, "-w"],
                        capture_output=True, text=True,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        key = result.stdout.strip()
                        break
                except FileNotFoundError:
                    pass

        if not key:
            raise ValueError(
                "GEMINI_API_KEY not set. Place config.json in ~/Downloads/ "
                "or ~/.alteris/, add gemini_api_key to it, or set the "
                "GEMINI_API_KEY environment variable."
            )

        self._api_key = key
        return key

    def _get_client(self) -> Any:
        """Return a cached genai Client with request timeout."""
        if self._client is not None:
            return self._client
        from google import genai
        from google.genai import types as gtypes
        key = self._get_key()
        self._client = genai.Client(
            api_key=key,
            http_options=gtypes.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
        )
        return self._client

    def _get_async_client(self) -> Any:
        """Return a cached async genai Client."""
        if self._async_client is not None:
            return self._async_client
        from google import genai
        from google.genai import types as gtypes
        key = self._get_key()
        self._async_client = genai.Client(
            api_key=key,
            http_options=gtypes.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
        )
        return self._async_client

    # Models that don't support context caching (400 errors, slow timeout)
    _CACHE_BLOCKLIST = {"gemini-2.5-flash-lite", "gemini-2.0-flash-lite"}

    def _get_or_create_cache(self, model: str, system_text: str, ttl_seconds: int = 3600) -> str | None:
        """Get or create a cached system instruction. Returns cache name or None.

        Uses Gemini's context caching API: cached input tokens cost 90% less.
        Caches are keyed by hash of (model + system_text) and reused within TTL.
        """
        import hashlib
        import time as _time

        # Skip models that don't support caching (avoids slow 400 timeouts)
        if any(blocked in model for blocked in self._CACHE_BLOCKLIST):
            return None

        cache_key = hashlib.sha256(f"{model}:{system_text}".encode()).hexdigest()[:16]
        now = _time.time()

        # Check in-memory map first
        if cache_key in self._cache_map:
            cache_name, expiry = self._cache_map[cache_key]
            if now < expiry:
                return cache_name
            # Expired — remove from map
            del self._cache_map[cache_key]

        try:
            from google.genai import types
            client = self._get_client()

            cache = client.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_text,
                    display_name=f"alteris_{cache_key}",
                    ttl=f"{ttl_seconds}s",
                ),
            )
            cache_name = cache.name
            self._cache_map[cache_key] = (cache_name, now + ttl_seconds - 60)
            logger.debug("Created Gemini cache %s for model %s (TTL %ds)", cache_name, model, ttl_seconds)
            return cache_name

        except Exception as exc:
            logger.debug("Gemini cache creation failed (will use uncached): %s", exc)
            # Add model to blocklist so we don't retry on every call
            self._CACHE_BLOCKLIST = self._CACHE_BLOCKLIST | {model}
            return None

    def is_available(self) -> bool:
        """Check if the Gemini API key is configured."""
        try:
            self._get_key()
            return True
        except ValueError:
            return False

    def _check_spend_limit(self) -> None:
        """Raise SpendLimitError if daily limit exceeded (bundled key only)."""
        if not self._store:
            return
        from alteris.spend import check_limit
        result = check_limit(self._store, "gemini", has_own_key=self._has_own_key)
        if not result["within_limit"]:
            raise SpendLimitError(result["daily_total"], result["limit"])

    def _record_spend(self, model_used: str, response: Any, source: str = "") -> None:
        """Extract token counts from a Gemini response and record spend."""
        if not self._store:
            return
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            return
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0
        thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
        try:
            from alteris.spend import record_usage
            record_usage(
                self._store, "gemini", model_used,
                input_tokens, output_tokens, source,
                cached_input_tokens=cached_tokens,
                thinking_tokens=thinking_tokens,
            )
        except Exception:
            pass  # Spend tracking is best-effort; don't break generation

    # ── Web search ─────────────────────────────────────────────

    def web_search(
        self,
        query: str,
        model: str = "",
    ) -> str | None:
        """Execute a grounded web search via Gemini's google_search tool.

        Makes a generate call with google_search grounding enabled.
        Returns the model's summarized answer with search citations.
        """
        self._check_spend_limit()
        try:
            from google.genai import types

            client = self._get_client()
            use_model = _resolve_model(model, self.model)

            config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1,
                max_output_tokens=2048,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            )

            response = client.models.generate_content(
                model=use_model,
                contents=(
                    f"Search for the following and provide a concise factual "
                    f"summary. Include specific details (dates, times, addresses, "
                    f"prices) when available.\n\nQuery: {query}"
                ),
                config=config,
            )
            self._record_spend(use_model, response, "web_search")
            return response.text

        except ImportError:
            logger.error("google-genai package not installed")
            return None
        except SpendLimitError:
            raise
        except Exception as exc:
            logger.warning("Gemini web_search failed for '%s': %s", query[:80], exc)
            return None

    # ── Embedding ────────────────────────────────────────────────

    def embed(
        self,
        texts: list[str],
        model: str = "",
    ) -> list[list[float] | None]:
        """Generate embeddings via Gemini's embedding API.

        Note: Gemini's embedding model is different from the generation model.
        Uses text-embedding-004 by default.
        """
        self._check_spend_limit()
        try:
            client = self._get_client()
            embed_model = model or "text-embedding-004"

            results: list[list[float] | None] = []
            total_input_tokens = 0
            for text in texts:
                try:
                    response = client.models.embed_content(
                        model=embed_model,
                        content=text,
                    )
                    if response and response.embedding:
                        results.append(list(response.embedding))
                    else:
                        results.append(None)
                    # Embed responses may have usage metadata
                    usage = getattr(response, "usage_metadata", None)
                    if usage:
                        total_input_tokens += getattr(usage, "prompt_token_count", 0) or 0
                except Exception as exc:
                    logger.error("Gemini embed failed for text: %s", exc)
                    results.append(None)

            if self._store and total_input_tokens > 0:
                from alteris.spend import record_usage
                record_usage(self._store, "gemini", embed_model, total_input_tokens, 0, "embed")

            return results

        except ImportError:
            logger.error(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )
            return [None] * len(texts)
        except SpendLimitError:
            raise
        except ValueError as exc:
            logger.error("Gemini embed: %s", exc)
            return [None] * len(texts)

    # ── Text generation ──────────────────────────────────────────

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
        """Generate text via Gemini API.

        thinking_budget: explicit thinking token budget (Gemini 2.x).
            None = model decides. 0 = disable. N > 0 = cap at N tokens.
        thinking_level: reasoning level for Gemini 3 models.
            One of: "minimal", "low", "medium", "high".
            Preferred over thinking_budget for Gemini 3.
        google_search: enable google_search grounding tool.
        cache_system: if True, cache the system instruction via Gemini's
            context caching API. Cached input tokens cost 90% less.
            Best for stable system prompts reused across many calls.
        response_schema: Gemini structured output schema (types.Schema).
            When provided, forces JSON output and mechanically constrains
            the response to match the schema (enum enforcement, etc.).
        """
        self._check_spend_limit()
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                from google.genai import types

                client = self._get_client()
                use_model = _resolve_model(model, self.model)

                # Try context caching for the system instruction
                cached_model = None
                if cache_system and system:
                    cache_name = self._get_or_create_cache(use_model, system)
                    if cache_name:
                        cached_model = cache_name

                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system if (system and not cached_model) else None,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                )
                if google_search:
                    config.tools = [types.Tool(google_search=types.GoogleSearch())]
                if response_schema is not None:
                    config.response_mime_type = "application/json"
                    config.response_schema = response_schema
                elif format_json:
                    config.response_mime_type = "application/json"
                if thinking_level is not None:
                    config.thinking_config = types.ThinkingConfig(
                        thinking_level=thinking_level,
                    )
                elif thinking_budget is not None:
                    config.thinking_config = types.ThinkingConfig(
                        thinking_budget=thinking_budget,
                    )

                response = client.models.generate_content(
                    model=cached_model or use_model,
                    contents=prompt,
                    config=config,
                )
                self._record_spend(use_model, response, "generate")
                return response.text

            except ImportError:
                logger.error(
                    "google-genai package not installed. "
                    "Install with: pip install google-genai"
                )
                return None
            except SpendLimitError:
                raise
            except Exception as exc:
                exc_str = str(exc).lower()
                if _is_daily_quota_error(exc_str):
                    logger.error("Gemini daily quota exhausted — not retrying: %s", exc)
                    raise DailyQuotaExhaustedError(str(exc)) from exc
                is_retryable = (
                    "timeout" in exc_str or "timed out" in exc_str
                    or "503" in exc_str or "unavailable" in exc_str
                    or "429" in exc_str or "resource_exhausted" in exc_str
                )
                if attempt < max_retries and is_retryable:
                    server_delay = _parse_retry_delay(str(exc))
                    wait = min(server_delay or (2 ** attempt), 60)
                    logger.warning("Gemini generate retryable error (attempt %d/%d), waiting %.0fs: %s", attempt + 1, max_retries + 1, wait, exc)
                    import time
                    time.sleep(wait)
                    self._client = None
                    continue
                logger.error("Gemini generate failed: %s", exc)
                return None
        return None

    # ── Async generation ────────────────────────────────────────

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
        """Async generate text via Gemini API.

        Uses a cached async client for concurrent requests. Suitable for
        asyncio.gather() parallelism in triage.
        """
        self._check_spend_limit()
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                from google.genai import types

                async_client = self._get_async_client()
                use_model = _resolve_model(model, self.model)

                # Try context caching for the system instruction
                cached_model = None
                if cache_system and system:
                    cache_name = self._get_or_create_cache(use_model, system)
                    if cache_name:
                        cached_model = cache_name

                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system if (system and not cached_model) else None,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                )
                if google_search:
                    config.tools = [types.Tool(google_search=types.GoogleSearch())]
                if response_schema is not None:
                    config.response_mime_type = "application/json"
                    config.response_schema = response_schema
                elif format_json:
                    config.response_mime_type = "application/json"
                if thinking_level is not None:
                    config.thinking_config = types.ThinkingConfig(
                        thinking_level=thinking_level,
                    )
                elif thinking_budget is not None:
                    config.thinking_config = types.ThinkingConfig(
                        thinking_budget=thinking_budget,
                    )

                response = await async_client.aio.models.generate_content(
                    model=cached_model or use_model,
                    contents=prompt,
                    config=config,
                )
                self._record_spend(use_model, response, "agenerate")
                return response.text

            except ImportError:
                logger.error(
                    "google-genai package not installed. "
                    "Install with: pip install google-genai"
                )
                return None
            except SpendLimitError:
                raise
            except Exception as exc:
                exc_str = str(exc).lower()
                if _is_daily_quota_error(exc_str):
                    logger.error("Gemini daily quota exhausted — not retrying: %s", exc)
                    raise DailyQuotaExhaustedError(str(exc)) from exc
                is_retryable = (
                    "timeout" in exc_str or "timed out" in exc_str
                    or "503" in exc_str or "unavailable" in exc_str
                    or "429" in exc_str or "resource_exhausted" in exc_str
                )
                if attempt < max_retries and is_retryable:
                    server_delay = _parse_retry_delay(str(exc))
                    wait = min(server_delay or (2 ** attempt), 60)
                    logger.warning("Gemini async retryable error (attempt %d/%d), waiting %.0fs: %s", attempt + 1, max_retries + 1, wait, exc)
                    import asyncio
                    await asyncio.sleep(wait)
                    self._async_client = None
                    continue
                logger.error("Gemini async generate failed: %s", exc)
                return None
        return None

    # ── Chat (multi-turn) ────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        format_json: bool = False,
    ) -> str | None:
        """Multi-turn chat via Gemini API.

        Converts the standard messages format to Gemini's content format.
        """
        self._check_spend_limit()
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                from google.genai import types

                client = self._get_client()
                use_model = _resolve_model(model, self.model)

                # Extract system message
                system_text = system
                contents = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        system_text = content
                    elif role == "assistant":
                        contents.append(
                            types.Content(
                                role="model",
                                parts=[types.Part(text=content)],
                            )
                        )
                    else:
                        contents.append(
                            types.Content(
                                role="user",
                                parts=[types.Part(text=content)],
                            )
                        )

                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system_text if system_text else None,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                )
                if format_json:
                    config.response_mime_type = "application/json"

                response = client.models.generate_content(
                    model=use_model,
                    contents=contents,
                    config=config,
                )
                self._record_spend(use_model, response, "chat")
                return response.text

            except ImportError:
                logger.error(
                    "google-genai package not installed. "
                    "Install with: pip install google-genai"
                )
                return None
            except SpendLimitError:
                raise
            except Exception as exc:
                exc_str = str(exc).lower()
                if _is_daily_quota_error(exc_str):
                    logger.error("Gemini daily quota exhausted — not retrying: %s", exc)
                    raise DailyQuotaExhaustedError(str(exc)) from exc
                is_retryable = (
                    "timeout" in exc_str or "timed out" in exc_str
                    or "503" in exc_str or "unavailable" in exc_str
                    or "429" in exc_str or "resource_exhausted" in exc_str
                )
                if attempt < max_retries and is_retryable:
                    server_delay = _parse_retry_delay(str(exc))
                    wait = min(server_delay or (2 ** attempt), 60)
                    logger.warning("Gemini chat retryable error (attempt %d/%d), waiting %.0fs: %s", attempt + 1, max_retries + 1, wait, exc)
                    import time
                    time.sleep(wait)
                    self._client = None
                    continue
                logger.error("Gemini chat failed: %s", exc)
                return None
        return None

    def chat_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        system: str = "",
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        tool_executor: "callable | None" = None,
        max_tool_rounds: int = 3,
    ) -> str | None:
        """Chat with Gemini function calling. Executes tool calls in a loop.

        tools: list of {"name": str, "description": str, "parameters": dict}
            where parameters is a JSON Schema object.
        tool_executor: callable(name, args) -> dict that runs the tool.
        max_tool_rounds: max number of tool-call round-trips before returning.

        Returns the final text response after all tool calls are resolved.
        """
        self._check_spend_limit()
        try:
            from google.genai import types

            client = self._get_client()
            use_model = _resolve_model(model, self.model)

            # Build function declarations
            func_decls = []
            for tool in tools:
                params = tool.get("parameters", {})
                schema = types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(
                            type=v.get("type", "STRING").upper(),
                            description=v.get("description", ""),
                            enum=v.get("enum"),
                        )
                        for k, v in params.get("properties", {}).items()
                    },
                    required=params.get("required", []),
                )
                func_decls.append(types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool["description"],
                    parameters=schema,
                ))

            gemini_tools = [types.Tool(function_declarations=func_decls)]

            # Build contents from messages
            system_text = system
            contents = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_text = content
                elif role == "assistant":
                    contents.append(types.Content(
                        role="model",
                        parts=[types.Part(text=content)],
                    ))
                else:
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part(text=content)],
                    ))

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_text if system_text else None,
                tools=gemini_tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
            )

            for _round in range(max_tool_rounds + 1):
                response = client.models.generate_content(
                    model=use_model,
                    contents=contents,
                    config=config,
                )
                self._record_spend(use_model, response, "chat_tools")

                if not response.candidates:
                    return None

                parts = response.candidates[0].content.parts
                fn_calls = [p for p in parts if p.function_call]

                if not fn_calls:
                    # No tool calls — return text
                    text_parts = [p.text for p in parts if p.text]
                    return " ".join(text_parts) if text_parts else None

                if not tool_executor:
                    # Tool calls requested but no executor — return text if any
                    text_parts = [p.text for p in parts if p.text]
                    return " ".join(text_parts) if text_parts else None

                # Append model response to conversation
                contents.append(response.candidates[0].content)

                # Execute each tool call and build function response
                fn_response_parts = []
                for fc in fn_calls:
                    try:
                        result = tool_executor(fc.function_call.name, dict(fc.function_call.args))
                    except Exception as exc:
                        logger.warning("Tool %s execution error: %s", fc.function_call.name, exc)
                        result = {"error": str(exc)}

                    fn_response_parts.append(types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.function_call.name,
                            response=result,
                        )
                    ))

                contents.append(types.Content(role="user", parts=fn_response_parts))

            # Exhausted tool rounds — return whatever text we have
            return response.text if response else None

        except ImportError:
            logger.error("google-genai package not installed.")
            return None
        except SpendLimitError:
            raise
        except Exception as exc:
            logger.error("Gemini chat_with_tools failed: %s", exc)
            return None
