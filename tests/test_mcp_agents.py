"""Integration tests for MCP tools, YAML agent storage, and CQ merge logic.

Tests use temp directories and in-memory SQLite — no real KG or filesystem side effects.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def tmp_agents_dir(tmp_path):
    """Create a temp agents directory and patch constants to use it."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    with patch("alteris.agent_store.AGENTS_DIR", agents_dir), \
         patch("alteris.agent_store.BUILTIN_AGENTS_DIR", tmp_path / "builtins"):
        yield agents_dir


@pytest.fixture
def tmp_agents_with_builtins(tmp_path):
    """Temp agents dir with a builtins dir containing a sample spec."""
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    builtins_dir = tmp_path / "builtins"
    builtins_dir.mkdir()

    # Create a builtin spec
    spec = {
        "name": "VC Research",
        "description": "Researches VCs",
        "system_prompt": "You are a VC research assistant.",
        "tool_permissions": ["alteris_search", "alteris_query_persons"],
        "status": "active",
        "llm_backend": "anthropic",
        "trigger": "manual",
    }
    (builtins_dir / "vc-research.yaml").write_text(yaml.dump(spec))

    with patch("alteris.agent_store.AGENTS_DIR", agents_dir), \
         patch("alteris.agent_store.BUILTIN_AGENTS_DIR", builtins_dir):
        yield agents_dir, builtins_dir


@pytest.fixture
def store():
    """Create an in-memory LayeredGraphStore."""
    from alteris.store import LayeredGraphStore
    s = LayeredGraphStore(":memory:")
    return s


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent Store: YAML CRUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAgentStore:
    """Test YAML-based agent storage."""

    def test_create_agent(self, tmp_agents_dir):
        from alteris.agent_store import create_agent, get_agent

        agent_id = create_agent(
            name="Test Agent",
            system_prompt="You are a test agent.",
            tool_permissions=["alteris_search"],
            description="A test agent",
        )

        assert agent_id == "test-agent"
        assert (tmp_agents_dir / "test-agent.yaml").exists()

        agent = get_agent("test-agent")
        assert agent is not None
        assert agent["name"] == "Test Agent"
        assert agent["system_prompt"] == "You are a test agent."
        assert agent["tool_permissions"] == ["alteris_search"]
        assert agent["id"] == "test-agent"

    def test_create_agent_custom_id(self, tmp_agents_dir):
        from alteris.agent_store import create_agent, get_agent

        agent_id = create_agent(
            name="My Agent",
            system_prompt="Test",
            agent_id="custom-id",
        )

        assert agent_id == "custom-id"
        assert (tmp_agents_dir / "custom-id.yaml").exists()

    def test_create_agent_dedup_id(self, tmp_agents_dir):
        from alteris.agent_store import create_agent

        # Create first agent
        id1 = create_agent(name="Dupe", system_prompt="First")
        assert id1 == "dupe"

        # Create second with same name — should get a suffix
        id2 = create_agent(name="Dupe", system_prompt="Second")
        assert id2 != "dupe"
        assert id2.startswith("dupe-")

    def test_list_agents_empty(self, tmp_agents_dir):
        from alteris.agent_store import list_agents
        assert list_agents() == []

    def test_list_agents(self, tmp_agents_dir):
        from alteris.agent_store import create_agent, list_agents

        create_agent(name="Agent A", system_prompt="Prompt A", status="active")
        create_agent(name="Agent B", system_prompt="Prompt B", status="draft")

        all_agents = list_agents()
        assert len(all_agents) == 2

        active_only = list_agents(status="active")
        assert len(active_only) == 1
        assert active_only[0]["name"] == "Agent A"

    def test_update_agent(self, tmp_agents_dir):
        from alteris.agent_store import create_agent, get_agent, update_agent

        create_agent(name="Old Name", system_prompt="Old prompt")
        ok = update_agent("old-name", name="New Name", status="active")
        assert ok is True

        agent = get_agent("old-name")
        assert agent["name"] == "New Name"
        assert agent["status"] == "active"

    def test_update_nonexistent(self, tmp_agents_dir):
        from alteris.agent_store import update_agent
        assert update_agent("nonexistent", name="X") is False

    def test_delete_agent(self, tmp_agents_dir):
        from alteris.agent_store import create_agent, delete_agent, get_agent

        create_agent(name="Doomed", system_prompt="Bye")
        assert get_agent("doomed") is not None

        ok = delete_agent("doomed")
        assert ok is True
        assert get_agent("doomed") is None

    def test_delete_nonexistent(self, tmp_agents_dir):
        from alteris.agent_store import delete_agent
        assert delete_agent("nonexistent") is False

    def test_get_agent_not_found(self, tmp_agents_dir):
        from alteris.agent_store import get_agent
        assert get_agent("nonexistent") is None

    def test_defaults_applied(self, tmp_agents_dir):
        from alteris.agent_store import create_agent, get_agent

        create_agent(name="Minimal", system_prompt="Hello")
        agent = get_agent("minimal")

        assert agent["llm_backend"] == "anthropic"
        assert agent["model"] == ""
        assert agent["trigger"] == "manual"
        assert agent["status"] == "draft"

    def test_builtin_seeding(self, tmp_agents_with_builtins):
        from alteris.agent_store import ensure_agents_dir, list_agents

        agents_dir, _ = tmp_agents_with_builtins
        ensure_agents_dir()

        # Builtin should have been copied
        assert (agents_dir / "vc-research.yaml").exists()

        agents = list_agents()
        assert len(agents) == 1
        assert agents[0]["name"] == "VC Research"

    def test_builtin_not_overwritten(self, tmp_agents_with_builtins):
        from alteris.agent_store import ensure_agents_dir

        agents_dir, _ = tmp_agents_with_builtins

        # User modifies the file
        custom = {"name": "My Custom VC", "system_prompt": "Custom", "tool_permissions": []}
        (agents_dir / "vc-research.yaml").write_text(yaml.dump(custom))

        # Re-seed should NOT overwrite
        ensure_agents_dir()
        with open(agents_dir / "vc-research.yaml") as f:
            data = yaml.safe_load(f)
        assert data["name"] == "My Custom VC"

    def test_malformed_yaml_skipped(self, tmp_agents_dir):
        from alteris.agent_store import list_agents

        # Write invalid YAML
        (tmp_agents_dir / "bad.yaml").write_text("not: valid: yaml: [")
        # Write valid but missing required fields
        (tmp_agents_dir / "empty.yaml").write_text(yaml.dump({"name": "X"}))

        agents = list_agents()
        assert len(agents) == 0

    def test_yaml_roundtrip_preserves_multiline(self, tmp_agents_dir):
        from alteris.agent_store import create_agent, get_agent

        prompt = "Line 1\nLine 2\nLine 3\n\nParagraph 2"
        create_agent(name="Multi", system_prompt=prompt)

        agent = get_agent("multi")
        assert agent["system_prompt"] == prompt

    def test_id_derived_from_name(self, tmp_agents_dir):
        from alteris.agent_store import create_agent

        # Special chars should be stripped
        agent_id = create_agent(name="My Agent! (v2)", system_prompt="Test")
        assert agent_id == "my-agent-v2"

    def test_tool_permissions_roundtrip(self, tmp_agents_dir):
        from alteris.agent_store import create_agent, get_agent

        perms = ["alteris_search", "alteris_query_beliefs", "alteris_person_detail"]
        create_agent(name="Perms", system_prompt="Test", tool_permissions=perms)

        agent = get_agent("perms")
        assert agent["tool_permissions"] == perms


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MCP Tool Registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMCPToolRegistration:
    """Verify all MCP tools register correctly."""

    def test_all_tools_registered(self):
        from alteris.mcp_tools import ensure_tools_loaded, get_all_tools
        ensure_tools_loaded()
        tools = get_all_tools()
        names = {t.name for t in tools}

        # KG read tools
        assert "alteris_stats" in names
        assert "alteris_query_events" in names
        assert "alteris_query_beliefs" in names
        assert "alteris_query_persons" in names
        assert "alteris_query_commitments" in names
        assert "alteris_person_detail" in names
        assert "alteris_search" in names

        # CQ tools
        assert "cq_get_state" in names
        assert "cq_add_task" in names
        assert "cq_update_task" in names
        assert "cq_move_task" in names
        assert "cq_dismiss" in names
        assert "cq_chat" in names

        # Agent tools
        assert "agent_list" in names
        assert "agent_create" in names
        assert "agent_update" in names
        assert "agent_delete" in names
        assert "agent_run" in names
        assert "agent_test" in names
        assert "agent_get_builder_prompt" in names

    def test_read_tools_safe_for_agents(self):
        from alteris.mcp_tools import ensure_tools_loaded, get_read_tools
        ensure_tools_loaded()
        read_tools = get_read_tools()
        names = {t.name for t in read_tools}

        # These should be read-only
        for name in ["alteris_stats", "alteris_query_events", "alteris_query_beliefs",
                      "alteris_query_persons", "alteris_query_commitments",
                      "alteris_person_detail", "alteris_search", "agent_list"]:
            assert name in names, f"{name} should be a read tool"

        # These should NOT be read-only
        for name in ["cq_add_task", "agent_create", "agent_run",
                      "alteris_run_pipeline"]:
            assert name not in names, f"{name} should NOT be a read tool"

    def test_every_tool_has_handler(self):
        from alteris.mcp_tools import ensure_tools_loaded, get_all_tools
        ensure_tools_loaded()
        for tool in get_all_tools():
            assert tool.handler is not None, f"Tool {tool.name} has no handler"

    def test_tool_lookup(self):
        from alteris.mcp_tools import ensure_tools_loaded, get_tool
        ensure_tools_loaded()

        t = get_tool("alteris_stats")
        assert t is not None
        assert t.permission == "read"

        t = get_tool("nonexistent_tool")
        assert t is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent MCP Tools (end-to-end via handlers)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAgentMCPTools:
    """Test agent tools through their MCP handler functions."""

    def test_agent_create_via_tool(self, tmp_agents_dir, store):
        from alteris.mcp_tools.agent_tools import handle_agent_create

        result = handle_agent_create(
            store,
            name="Test Agent",
            system_prompt="You are helpful.",
            tool_permissions=["alteris_search"],
        )

        assert "error" not in result
        assert result["name"] == "Test Agent"
        assert "agent_id" in result

    def test_agent_create_missing_name(self, tmp_agents_dir, store):
        from alteris.mcp_tools.agent_tools import handle_agent_create

        result = handle_agent_create(store, system_prompt="Hi")
        assert "error" in result

    def test_agent_create_invalid_perms(self, tmp_agents_dir, store):
        from alteris.mcp_tools.agent_tools import handle_agent_create

        result = handle_agent_create(
            store,
            name="Bad",
            system_prompt="Hi",
            tool_permissions=["alteris_run_pipeline"],
        )
        assert "error" in result
        assert "Invalid" in result["error"]

    def test_agent_list_via_tool(self, tmp_agents_dir, store):
        from alteris.mcp_tools.agent_tools import handle_agent_create, handle_agent_list

        handle_agent_create(store, name="Agent 1", system_prompt="P1", status="active")
        handle_agent_create(store, name="Agent 2", system_prompt="P2", status="draft")

        result = handle_agent_list(store)
        assert result["count"] == 2

        # List summaries should not include system_prompt
        for agent in result["agents"]:
            assert "system_prompt" not in agent

    def test_agent_update_via_tool(self, tmp_agents_dir, store):
        from alteris.mcp_tools.agent_tools import handle_agent_create, handle_agent_update

        result = handle_agent_create(store, name="Original", system_prompt="Old")
        agent_id = result["agent_id"]

        update_result = handle_agent_update(store, agent_id=agent_id, name="Updated")
        assert update_result["updated"] is True

    def test_agent_delete_via_tool(self, tmp_agents_dir, store):
        from alteris.mcp_tools.agent_tools import handle_agent_create, handle_agent_delete, handle_agent_list

        result = handle_agent_create(store, name="Temp", system_prompt="Hi")
        agent_id = result["agent_id"]

        delete_result = handle_agent_delete(store, agent_id=agent_id)
        assert delete_result["deleted"] is True

        list_result = handle_agent_list(store)
        assert list_result["count"] == 0

    def test_agent_run_not_found(self, tmp_agents_dir, store):
        from alteris.mcp_tools.agent_tools import handle_agent_run

        result = handle_agent_run(store, agent_id="nonexistent", message="Hi")
        assert "error" in result

    def test_agent_get_builder_prompt(self, tmp_agents_dir, store):
        from alteris.mcp_tools.agent_tools import handle_agent_get_builder_prompt

        result = handle_agent_get_builder_prompt(store)
        assert "system_prompt" in result
        assert "available_tools" in result
        assert "alteris_search" in result["available_tools"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CQ Merge Logic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestCQAutoBucketing:
    """Test CQ auto-bucketing logic."""

    def test_no_deadline_goes_to_background(self):
        from alteris.mcp_tools.cq_tools import _auto_bucket

        now = int(time.time())
        assert _auto_bucket(None, now) == "background"
        assert _auto_bucket("", now) == "background"

    def test_overdue_goes_to_critical(self):
        from alteris.mcp_tools.cq_tools import _auto_bucket

        now = int(time.time())
        yesterday = "2020-01-01"
        assert _auto_bucket(yesterday, now) == "immediate"

    def test_today_goes_to_critical(self):
        from alteris.mcp_tools.cq_tools import _auto_bucket
        from datetime import datetime, timezone

        now = int(time.time())
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert _auto_bucket(today, now) == "immediate"

    def test_this_week_bucket(self):
        from alteris.mcp_tools.cq_tools import _auto_bucket
        from datetime import datetime, timedelta, timezone

        now = int(time.time())
        in_3_days = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%d")
        assert _auto_bucket(in_3_days, now) == "review"

    def test_far_future_goes_to_background(self):
        from alteris.mcp_tools.cq_tools import _auto_bucket
        from datetime import datetime, timedelta, timezone

        now = int(time.time())
        in_30_days = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%d")
        assert _auto_bucket(in_30_days, now) == "background"


class TestCQGetState:
    """Test CQ state merging of KG data + user overrides."""

    def test_empty_state(self, store):
        from alteris.mcp_tools.cq_tools import handle_cq_get_state

        result = handle_cq_get_state(store)
        assert "immediate" in result
        assert "review" in result
        assert "background" in result

    def test_manual_task_appears(self, store):
        from alteris.mcp_tools.cq_tools import handle_cq_add_task, handle_cq_get_state

        handle_cq_add_task(store, bucket="immediate", title="Manual task")
        result = handle_cq_get_state(store)

        tasks = result["immediate"]
        assert len(tasks) >= 1
        manual = [t for t in tasks if t.get("title") == "Manual task"]
        assert len(manual) == 1
        assert manual[0]["source"] == "manual"

    def test_dismiss_removes_from_state(self, store):
        from alteris.mcp_tools.cq_tools import handle_cq_add_task, handle_cq_dismiss, handle_cq_get_state

        handle_cq_add_task(store, bucket="review", title="Delete me")
        state1 = handle_cq_get_state(store)
        tasks = state1["review"]
        task_id = [t for t in tasks if t.get("title") == "Delete me"][0]["id"]

        handle_cq_dismiss(store, task_id=task_id)
        state2 = handle_cq_get_state(store)
        remaining = [t for t in state2["review"] if t.get("title") == "Delete me"]
        assert len(remaining) == 0


class TestCQTaskCRUD:
    """Test CQ task add/update/move/dismiss."""

    def test_add_task(self, store):
        from alteris.mcp_tools.cq_tools import handle_cq_add_task

        result = handle_cq_add_task(
            store, bucket="immediate", title="Important thing",
            note="Details here",
        )
        assert "task_id" in result
        assert result["bucket"] == "immediate"

    def test_update_task(self, store):
        from alteris.mcp_tools.cq_tools import handle_cq_add_task, handle_cq_update_task

        add_result = handle_cq_add_task(store, bucket="review", title="Old title")
        task_id = add_result["task_id"]

        update_result = handle_cq_update_task(store, task_id=task_id, title="New title")
        assert update_result.get("updated") is True

    def test_move_task(self, store):
        from alteris.mcp_tools.cq_tools import handle_cq_add_task, handle_cq_get_state, handle_cq_move_task

        add_result = handle_cq_add_task(store, bucket="review", title="Moveable")
        task_id = add_result["task_id"]

        handle_cq_move_task(store, task_id=task_id, target_bucket="immediate")
        state = handle_cq_get_state(store)

        moved = [t for t in state["immediate"] if t.get("title") == "Moveable"]
        assert len(moved) == 1

    def test_toggle_done(self, store):
        from alteris.mcp_tools.cq_tools import handle_cq_add_task, handle_cq_update_task

        add_result = handle_cq_add_task(store, bucket="review", title="Do this")
        task_id = add_result["task_id"]

        handle_cq_update_task(store, task_id=task_id, done=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent Runtime (scoped tools)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAgentRuntime:
    """Test agent runtime tool scoping."""

    def test_build_scoped_tools(self, store):
        from alteris.agent_runtime import _build_scoped_tools

        tools = _build_scoped_tools(store, ["alteris_stats", "alteris_search"])
        assert "alteris_stats" in tools
        assert "alteris_search" in tools
        assert "alteris_run_pipeline" not in tools
        assert "agent_create" not in tools

    def test_scoped_tools_empty_perms(self, store):
        from alteris.agent_runtime import _build_scoped_tools

        tools = _build_scoped_tools(store, [])
        assert len(tools) == 0

    def test_scoped_tools_invalid_name_skipped(self, store):
        from alteris.agent_runtime import _build_scoped_tools

        tools = _build_scoped_tools(store, ["alteris_stats", "nonexistent_tool"])
        assert "alteris_stats" in tools
        assert "nonexistent_tool" not in tools


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KG Tool Handlers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestKGTools:
    """Test KG read tool handlers with in-memory store."""

    def test_alteris_stats(self, store):
        from alteris.mcp_tools.kg_tools import handle_alteris_stats

        result = handle_alteris_stats(store)
        assert "events_count" in result
        assert "claims_count" in result
        assert "beliefs_count" in result
        assert "persons_count" in result

    def test_alteris_query_events_empty(self, store):
        from alteris.mcp_tools.kg_tools import handle_alteris_query_events

        result = handle_alteris_query_events(store)
        assert "events" in result
        assert result["count"] == 0

    def test_alteris_query_beliefs_empty(self, store):
        from alteris.mcp_tools.kg_tools import handle_alteris_query_beliefs

        result = handle_alteris_query_beliefs(store)
        assert "beliefs" in result
        assert result["count"] == 0

    def test_alteris_query_persons_empty(self, store):
        from alteris.mcp_tools.kg_tools import handle_alteris_query_persons

        result = handle_alteris_query_persons(store)
        assert "persons" in result
        assert result["count"] == 0

    def test_alteris_search_empty(self, store):
        from alteris.mcp_tools.kg_tools import handle_alteris_search

        result = handle_alteris_search(store, query="test")
        assert "events" in result
        assert result["count"] == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Built-in VC Research Agent Spec
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestVCResearchAgent:
    """Verify the built-in VC Research agent spec is valid."""

    def test_spec_file_exists(self):
        from alteris.constants import BUILTIN_AGENTS_DIR
        spec_path = BUILTIN_AGENTS_DIR / "vc-research.yaml"
        assert spec_path.exists(), f"VC Research spec not found at {spec_path}"

    def test_spec_valid_yaml(self):
        from alteris.constants import BUILTIN_AGENTS_DIR
        spec_path = BUILTIN_AGENTS_DIR / "vc-research.yaml"
        with open(spec_path) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "VC Research"
        assert data["status"] == "active"
        assert "system_prompt" in data
        assert len(data["system_prompt"]) > 100

    def test_spec_tool_permissions_valid(self):
        from alteris.constants import AGENT_READ_TOOLS, BUILTIN_AGENTS_DIR
        spec_path = BUILTIN_AGENTS_DIR / "vc-research.yaml"
        with open(spec_path) as f:
            data = yaml.safe_load(f)

        for tool in data["tool_permissions"]:
            assert tool in AGENT_READ_TOOLS, f"VC spec has invalid tool: {tool}"

    def test_spec_seeded_on_ensure(self, tmp_agents_with_builtins):
        """Verify builtin specs are copied to user's agents dir."""
        from alteris.agent_store import ensure_agents_dir, get_agent

        agents_dir, _ = tmp_agents_with_builtins
        ensure_agents_dir()

        agent = get_agent("vc-research")
        assert agent is not None
        assert agent["name"] == "VC Research"
