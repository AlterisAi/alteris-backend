"""Tests for loom.llm: base interface, mock client, canned responses.

Tests cover:
  - MockLLMClient embedding determinism and dimensionality
  - generate() prompt substring matching
  - generate_json() parsing and fallback
  - chat() delegation to generate()
  - Call tracking for assertions
  - Canned response fixtures
"""

import json
import math

import pytest

from loom.constants import EMBEDDING_DIM
from loom.llm.base import LLMClient
from loom.llm.mock import (
    MOCK_EXTRACTION_RESPONSE,
    MOCK_SYNTHESIS_RESPONSE,
    MOCK_TRIAGE_RESPONSE,
    MockLLMClient,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Interface compliance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestInterface:
    def test_mock_is_llm_client(self):
        client = MockLLMClient()
        assert isinstance(client, LLMClient)

    def test_abstract_methods_exist(self):
        """LLMClient defines required abstract methods."""
        assert hasattr(LLMClient, "embed")
        assert hasattr(LLMClient, "generate")
        assert hasattr(LLMClient, "chat")

    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            LLMClient()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Embeddings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMockEmbeddings:
    def test_returns_correct_dimension(self):
        client = MockLLMClient()
        result = client.embed(["hello"])
        assert len(result) == 1
        assert len(result[0]) == EMBEDDING_DIM

    def test_custom_dimension(self):
        client = MockLLMClient(embed_dim=128)
        result = client.embed(["hello"])
        assert len(result[0]) == 128

    def test_deterministic(self):
        client = MockLLMClient()
        r1 = client.embed(["hello"])[0]
        r2 = client.embed(["hello"])[0]
        assert r1 == r2

    def test_different_texts_different_embeddings(self):
        client = MockLLMClient()
        r1 = client.embed(["hello"])[0]
        r2 = client.embed(["world"])[0]
        assert r1 != r2

    def test_batch_embeddings(self):
        client = MockLLMClient()
        result = client.embed(["hello", "world", "test"])
        assert len(result) == 3
        assert all(len(v) == EMBEDDING_DIM for v in result)

    def test_l2_normalized(self):
        client = MockLLMClient()
        vec = client.embed(["normalize me"])[0]
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-6

    def test_tracks_embed_calls(self):
        client = MockLLMClient()
        client.embed(["a", "b"])
        client.embed(["c"])
        assert len(client.embed_calls) == 2
        assert client.embed_calls[0] == ["a", "b"]
        assert client.embed_calls[1] == ["c"]

    def test_empty_text(self):
        client = MockLLMClient()
        result = client.embed([""])
        assert len(result[0]) == EMBEDDING_DIM


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Generate
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMockGenerate:
    def test_default_response(self):
        client = MockLLMClient(default_response="fallback")
        result = client.generate("any prompt")
        assert result == "fallback"

    def test_substring_match(self):
        client = MockLLMClient(responses={"hello": "matched hello"})
        result = client.generate("Say hello world")
        assert result == "matched hello"

    def test_no_match_returns_default(self):
        client = MockLLMClient(
            default_response="default",
            responses={"xyz": "found xyz"},
        )
        result = client.generate("abc")
        assert result == "default"

    def test_json_format_returns_triage(self):
        """When format_json=True and no match, returns default triage response."""
        client = MockLLMClient()
        result = client.generate("some prompt", format_json=True)
        parsed = json.loads(result)
        assert parsed["score"] == 0.75
        assert "domain" in parsed

    def test_json_format_substring_match(self):
        client = MockLLMClient(
            json_responses={"extract": {"commitments": []}}
        )
        result = client.generate("Please extract commitments", format_json=True)
        parsed = json.loads(result)
        assert parsed == {"commitments": []}

    def test_tracks_generate_calls(self):
        client = MockLLMClient()
        client.generate("first prompt", system="sys1")
        client.generate("second prompt", model="model1")
        assert len(client.generate_calls) == 2
        assert client.generate_calls[0]["prompt"] == "first prompt"
        assert client.generate_calls[0]["system"] == "sys1"
        assert client.generate_calls[1]["model"] == "model1"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chat
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMockChat:
    def test_delegates_to_generate(self):
        client = MockLLMClient(responses={"test query": "chat response"})
        result = client.chat([
            {"role": "user", "content": "test query"},
        ])
        assert result == "chat response"

    def test_uses_last_user_message(self):
        client = MockLLMClient(responses={"second": "got second"})
        result = client.chat([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ])
        assert result == "got second"

    def test_no_user_message_empty_prompt(self):
        client = MockLLMClient(default_response="empty")
        result = client.chat([
            {"role": "assistant", "content": "no user message"},
        ])
        assert result == "empty"

    def test_tracks_chat_calls(self):
        client = MockLLMClient()
        messages = [{"role": "user", "content": "hello"}]
        client.chat(messages, system="sys")
        assert len(client.chat_calls) == 1
        assert client.chat_calls[0]["messages"] == messages
        assert client.chat_calls[0]["system"] == "sys"

    def test_chat_json_format(self):
        client = MockLLMClient()
        result = client.chat(
            [{"role": "user", "content": "triage this event"}],
            format_json=True,
        )
        parsed = json.loads(result)
        assert "score" in parsed


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# generate_json (base class method)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGenerateJson:
    def test_parses_valid_json(self):
        client = MockLLMClient(
            json_responses={"test": {"key": "value"}}
        )
        result = client.generate_json("test prompt")
        assert result == {"key": "value"}

    def test_returns_none_on_empty_response(self):
        client = MockLLMClient(default_response="")
        # Override generate to return empty
        orig = client.generate

        def empty_generate(*args, **kwargs):
            return ""

        client.generate = empty_generate
        result = client.generate_json("test")
        assert result is None

    def test_strips_markdown_fences(self):
        client = MockLLMClient()
        orig = client.generate

        def fenced_generate(*args, **kwargs):
            return '```json\n{"key": "value"}\n```'

        client.generate = fenced_generate
        result = client.generate_json("test")
        assert result == {"key": "value"}

    def test_returns_none_on_invalid_json(self):
        client = MockLLMClient()
        orig = client.generate

        def bad_generate(*args, **kwargs):
            return "not json at all"

        client.generate = bad_generate
        result = client.generate_json("test")
        assert result is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Canned response fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCannedResponses:
    def test_triage_response_schema(self):
        assert "score" in MOCK_TRIAGE_RESPONSE
        assert "reason" in MOCK_TRIAGE_RESPONSE
        assert "domain" in MOCK_TRIAGE_RESPONSE
        assert "topics" in MOCK_TRIAGE_RESPONSE
        assert "entities" in MOCK_TRIAGE_RESPONSE
        assert "pii" in MOCK_TRIAGE_RESPONSE
        assert "sensitivity" in MOCK_TRIAGE_RESPONSE
        assert "commitment_type" in MOCK_TRIAGE_RESPONSE

    def test_triage_response_types(self):
        assert isinstance(MOCK_TRIAGE_RESPONSE["score"], float)
        assert isinstance(MOCK_TRIAGE_RESPONSE["topics"], list)
        assert isinstance(MOCK_TRIAGE_RESPONSE["entities"], list)
        assert isinstance(MOCK_TRIAGE_RESPONSE["pii"], list)

    def test_extraction_response_schema(self):
        assert "commitments" in MOCK_EXTRACTION_RESPONSE
        assert len(MOCK_EXTRACTION_RESPONSE["commitments"]) > 0
        commit = MOCK_EXTRACTION_RESPONSE["commitments"][0]
        assert "what" in commit
        assert "who" in commit
        assert "deadline" in commit
        assert "commitment_type" in commit
        assert "confidence" in commit

    def test_synthesis_response_schema(self):
        assert "beliefs" in MOCK_SYNTHESIS_RESPONSE
        assert len(MOCK_SYNTHESIS_RESPONSE["beliefs"]) > 0
        belief = MOCK_SYNTHESIS_RESPONSE["beliefs"][0]
        assert "belief_type" in belief
        assert "subject" in belief
        assert "summary" in belief
        assert "data" in belief
        assert "epistemic_level" in belief
        assert "confidence" in belief
        assert "inference_chain" in belief
