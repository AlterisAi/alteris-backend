"""Agent management MCP tools.

CRUD for user-created agents, plus agent execution (run/test).
Agents are stored as YAML files in ~/.loom/agents/.
Most tools are 'write' — agent_list is 'read'.
"""

from __future__ import annotations

import json
import logging

from loom import agent_store
from loom.mcp_tools import ToolDef, ToolParam, register_tool
from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool implementations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def handle_agent_list(
    store: LayeredGraphStore,
    status: str | None = None,
    **kwargs,
) -> dict:
    """List user's agents from YAML files."""
    agents = agent_store.list_agents(status=status)
    # Return lightweight summaries (no system_prompt in list)
    summaries = []
    for a in agents:
        summaries.append({
            "id": a["id"],
            "name": a["name"],
            "description": a.get("description", ""),
            "status": a.get("status", "draft"),
            "llm_backend": a.get("llm_backend", "anthropic"),
            "trigger": a.get("trigger", "manual"),
            "tool_count": len(a.get("tool_permissions", [])),
            "updated_at": a.get("updated_at", 0),
        })
    return {"count": len(summaries), "agents": summaries}


def handle_agent_create(
    store: LayeredGraphStore,
    name: str = "",
    description: str = "",
    system_prompt: str = "",
    tool_permissions: list[str] | None = None,
    llm_backend: str = "anthropic",
    model: str = "",
    trigger: str = "manual",
    trigger_config: dict | None = None,
    **kwargs,
) -> dict:
    """Create a new agent YAML file."""
    if not name:
        return {"error": "name is required"}
    if not system_prompt:
        return {"error": "system_prompt is required"}

    from loom.constants import AGENT_READ_TOOLS

    # Validate tool permissions — only allow read tools
    if tool_permissions is not None:
        invalid = [t for t in tool_permissions if t not in AGENT_READ_TOOLS]
        if invalid:
            return {"error": f"Invalid tool permissions (not in read tools): {invalid}"}

    agent_id = agent_store.create_agent(
        name=name,
        system_prompt=system_prompt,
        tool_permissions=tool_permissions,
        description=description,
        llm_backend=llm_backend,
        model=model,
        trigger=trigger,
        trigger_config=trigger_config,
        status="draft",
    )
    return {"agent_id": agent_id, "name": name}


def handle_agent_update(
    store: LayeredGraphStore,
    agent_id: str = "",
    name: str | None = None,
    description: str | None = None,
    system_prompt: str | None = None,
    tool_permissions: list[str] | None = None,
    llm_backend: str | None = None,
    model: str | None = None,
    trigger: str | None = None,
    trigger_config: dict | None = None,
    status: str | None = None,
    **kwargs,
) -> dict:
    """Update an existing agent's YAML file."""
    if not agent_id:
        return {"error": "agent_id is required"}

    if tool_permissions is not None:
        from loom.constants import AGENT_READ_TOOLS
        invalid = [t for t in tool_permissions if t not in AGENT_READ_TOOLS]
        if invalid:
            return {"error": f"Invalid tool permissions: {invalid}"}

    fields = {}
    if name is not None:
        fields["name"] = name
    if description is not None:
        fields["description"] = description
    if system_prompt is not None:
        fields["system_prompt"] = system_prompt
    if tool_permissions is not None:
        fields["tool_permissions"] = tool_permissions
    if llm_backend is not None:
        fields["llm_backend"] = llm_backend
    if model is not None:
        fields["model"] = model
    if trigger is not None:
        fields["trigger"] = trigger
    if trigger_config is not None:
        fields["trigger_config"] = trigger_config
    if status is not None:
        fields["status"] = status

    if not fields:
        return {"error": "no fields to update"}

    ok = agent_store.update_agent(agent_id, **fields)
    return {"updated": ok, "agent_id": agent_id}


def handle_agent_delete(
    store: LayeredGraphStore,
    agent_id: str = "",
    **kwargs,
) -> dict:
    """Delete an agent YAML file."""
    if not agent_id:
        return {"error": "agent_id is required"}

    ok = agent_store.delete_agent(agent_id)
    return {"deleted": ok, "agent_id": agent_id}


def handle_agent_run(
    store: LayeredGraphStore,
    agent_id: str = "",
    message: str = "",
    **kwargs,
) -> dict:
    """Run an agent with a user message. Returns the agent's response."""
    if not agent_id or not message:
        return {"error": "agent_id and message are required"}

    agent = agent_store.get_agent(agent_id)
    if not agent:
        return {"error": f"Agent {agent_id} not found"}

    if agent.get("status") not in ("active", "draft"):
        return {"error": f"Agent is {agent.get('status')}, not runnable"}

    try:
        from loom.agent_runtime import run_agent
        result = run_agent(store, agent, message)
        return result
    except ImportError:
        return {"error": "agent_runtime module not yet available"}
    except Exception as exc:
        logger.error("Agent run failed: %s", exc)
        return {"error": str(exc)}


def handle_agent_test(
    store: LayeredGraphStore,
    agent_id: str = "",
    test_message: str = "",
    **kwargs,
) -> dict:
    """Test an agent in sandbox mode."""
    if not agent_id:
        return {"error": "agent_id is required"}
    if not test_message:
        test_message = "Hello! What can you tell me about my recent activity?"

    agent = agent_store.get_agent(agent_id)
    if not agent:
        return {"error": f"Agent {agent_id} not found"}

    try:
        from loom.agent_runtime import run_agent
        result = run_agent(store, agent, test_message, sandbox=True)
        return {"test_result": result, "agent_id": agent_id}
    except ImportError:
        return {"error": "agent_runtime module not yet available"}
    except Exception as exc:
        logger.error("Agent test failed: %s", exc)
        return {"error": str(exc)}


def handle_agent_get_builder_prompt(
    store: LayeredGraphStore,
    **kwargs,
) -> dict:
    """Get the system prompt for the agent builder agent."""
    from loom.constants import AGENT_READ_TOOLS

    return {
        "system_prompt": _BUILDER_SYSTEM_PROMPT,
        "available_tools": AGENT_READ_TOOLS,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent builder system prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_BUILDER_SYSTEM_PROMPT = """You are an agent builder assistant. You help users create personalized AI agents that operate on their knowledge graph.

When a user describes what agent they want, you:
1. Ask clarifying questions about the agent's purpose, trigger, and scope
2. Generate a complete agent specification:
   - Name: A short, descriptive name
   - Description: What the agent does in 1-2 sentences
   - System prompt: The full system prompt for the agent
   - Tool permissions: Which KG tools the agent needs (from the available set)
   - Trigger: When the agent should run (manual, scheduled, or event-based)

Available KG tools the agent can use:
- loom_stats: Get database statistics
- loom_query_events: Search events (email, messages, calendar, etc.)
- loom_query_beliefs: Query synthesized beliefs about people, topics, commitments
- loom_query_persons: List resolved person identities
- loom_query_commitments: Query active commitments and deadlines
- loom_person_detail: Deep info about a specific person
- loom_search: Full-text search over all event content

Agents can ONLY read from the knowledge graph — they cannot modify it, run the pipeline, or manage other agents.

Built-in example agents:
- "VC Research": Researches investors and firms using the KG for prior interactions, warm intro paths, and communication patterns

When generating the system prompt, make it specific and actionable. Include instructions for how to use the tools effectively."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

register_tool(ToolDef(
    name="agent_list",
    description="List all user-created agents, optionally filtered by status.",
    permission="read",
    params=[
        ToolParam("status", "string", "Filter by status", enum=["draft", "active", "paused"]),
    ],
    handler=handle_agent_list,
))

register_tool(ToolDef(
    name="agent_create",
    description="Create a new agent with a system prompt and tool permissions.",
    permission="write",
    params=[
        ToolParam("name", "string", "Agent name", required=True),
        ToolParam("description", "string", "What the agent does"),
        ToolParam("system_prompt", "string", "The agent's system prompt", required=True),
        ToolParam("tool_permissions", "array", "List of allowed tool names (default: all read tools)"),
        ToolParam("llm_backend", "string", "LLM provider", default="anthropic", enum=["anthropic", "gemini"]),
        ToolParam("model", "string", "Specific model ID"),
        ToolParam("trigger", "string", "When to run", default="manual", enum=["manual", "scheduled", "event"]),
        ToolParam("trigger_config", "object", "Trigger configuration (cron, event types, etc.)"),
    ],
    handler=handle_agent_create,
))

register_tool(ToolDef(
    name="agent_update",
    description="Update an existing agent's configuration.",
    permission="write",
    params=[
        ToolParam("agent_id", "string", "Agent ID (filename without .yaml)", required=True),
        ToolParam("name", "string", "New name"),
        ToolParam("description", "string", "New description"),
        ToolParam("system_prompt", "string", "New system prompt"),
        ToolParam("tool_permissions", "array", "New tool permissions"),
        ToolParam("llm_backend", "string", "New LLM backend"),
        ToolParam("model", "string", "New model"),
        ToolParam("trigger", "string", "New trigger"),
        ToolParam("trigger_config", "object", "New trigger config"),
        ToolParam("status", "string", "New status", enum=["draft", "active", "paused"]),
    ],
    handler=handle_agent_update,
))

register_tool(ToolDef(
    name="agent_delete",
    description="Delete an agent.",
    permission="write",
    params=[
        ToolParam("agent_id", "string", "Agent ID to delete", required=True),
    ],
    handler=handle_agent_delete,
))

register_tool(ToolDef(
    name="agent_run",
    description="Run an agent with a user message. The agent uses its allowed KG tools to formulate a response.",
    permission="write",
    params=[
        ToolParam("agent_id", "string", "Agent ID", required=True),
        ToolParam("message", "string", "User message to the agent", required=True),
    ],
    handler=handle_agent_run,
))

register_tool(ToolDef(
    name="agent_test",
    description="Test an agent in sandbox mode. Results are saved but not persisted to the KG.",
    permission="write",
    params=[
        ToolParam("agent_id", "string", "Agent ID", required=True),
        ToolParam("test_message", "string", "Test message (default: generic greeting)"),
    ],
    handler=handle_agent_test,
))

register_tool(ToolDef(
    name="agent_get_builder_prompt",
    description="Get the system prompt and available tools for the agent builder assistant.",
    permission="write",
    handler=handle_agent_get_builder_prompt,
))
