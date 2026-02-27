"""Local LLM client via Ollama HTTP API.

All calls go to localhost. Nothing leaves the machine.
Supports embedding, text generation, and multi-turn chat.

Setup:
    brew install ollama
    ollama serve &
    ollama pull nomic-embed-text       # 137M param embedding model
    ollama pull qwen3:8b               # 8B param, fast structured extraction
    ollama pull qwen3:30b-a3b          # 30B MoE (3B active), deep reasoning
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from alteris.constants import (
    EMBEDDING_DIM,
    LOCAL_EMBED_MODEL,
    LOCAL_FAST_MODEL,
    LOCAL_REASONING_MODEL,
)
from alteris.llm.base import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"

EMBEDDING_DIMS = {
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "mxbai-embed-large": 1024,
}


class OllamaClient(LLMClient):
    """Client for Ollama's local inference API."""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL):
        self.base_url = base_url.rstrip("/")
        self._available_models: set[str] | None = None

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags", timeout=2,
            )
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def list_models(self) -> list[str]:
        """List locally available models."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags", timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                self._available_models = set(models)
                return models
        except requests.RequestException:
            pass
        return []

    def has_model(self, model: str) -> bool:
        """Check if a specific model is available."""
        if self._available_models is None:
            self.list_models()
        return model in (self._available_models or set())

    # ── Embedding ────────────────────────────────────────────────

    def embed(
        self,
        texts: list[str],
        model: str = "",
    ) -> list[list[float] | None]:
        """Generate embeddings for a list of texts.

        Uses Ollama's /api/embed endpoint which supports batch input.
        """
        use_model = model or LOCAL_EMBED_MODEL
        results: list[list[float] | None] = []

        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": use_model, "input": texts},
                timeout=60,
            )

            if resp.status_code != 200:
                logger.error(
                    "Ollama embed failed: %s %s",
                    resp.status_code, resp.text[:200],
                )
                return [None] * len(texts)

            data = resp.json()
            embeddings = data.get("embeddings", [])

            for emb in embeddings:
                results.append(emb)

            # Pad if fewer results than inputs
            while len(results) < len(texts):
                results.append(None)

        except requests.RequestException as exc:
            logger.error("Ollama embed request failed: %s", exc)
            return [None] * len(texts)

        return results

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
        """Generate text from a prompt."""
        use_model = model or LOCAL_FAST_MODEL

        payload: dict[str, Any] = {
            "model": use_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system:
            payload["system"] = system

        if format_json:
            payload["format"] = "json"
            # Disable thinking mode for JSON -- it conflicts with
            # structured output and produces empty responses.
            payload["think"] = False

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )

            if resp.status_code != 200:
                logger.error(
                    "Ollama generate failed: %s %s",
                    resp.status_code, resp.text[:200],
                )
                return None

            data = resp.json()
            return data.get("response", "")

        except requests.RequestException as exc:
            logger.error("Ollama generate request failed: %s", exc)
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
        """Multi-turn chat with a local model."""
        use_model = model or LOCAL_FAST_MODEL

        # Inject system message if provided and not already present
        chat_messages = list(messages)
        if system and (
            not chat_messages
            or chat_messages[0].get("role") != "system"
        ):
            chat_messages.insert(0, {"role": "system", "content": system})

        payload: dict[str, Any] = {
            "model": use_model,
            "messages": chat_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if format_json:
            payload["format"] = "json"

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )

            if resp.status_code != 200:
                logger.error(
                    "Ollama chat failed: %s", resp.status_code,
                )
                return None

            data = resp.json()
            return data.get("message", {}).get("content", "")

        except requests.RequestException as exc:
            logger.error("Ollama chat request failed: %s", exc)
            return None
