"""MCP tool registration and permission tagging.

Tools are tagged with permission levels:
  - 'read': Safe for all consumers including user agents
  - 'write': Restricted to the app and admin (pipeline, CQ, agent management)

Usage:
    from alteris.mcp_tools import get_all_tools, get_read_tools

    all_tools = get_all_tools(store)
    safe_tools = get_read_tools(store)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Tool registry populated by register_tool()
_TOOL_REGISTRY: list[ToolDef] = []
_ALL_MODULES_LOADED = False


@dataclass
class ToolParam:
    """One parameter of an MCP tool."""
    name: str
    type: str  # "string", "integer", "number", "boolean", "array", "object"
    description: str = ""
    required: bool = False
    enum: list[str] | None = None
    default: Any = None


@dataclass
class ToolDef:
    """Full definition of an MCP tool."""
    name: str
    description: str
    permission: str  # "read" or "write"
    params: list[ToolParam] = field(default_factory=list)
    handler: Callable[..., Any] | None = None


def register_tool(tool_def: ToolDef) -> ToolDef:
    """Register a tool definition. Returns the def for chaining."""
    _TOOL_REGISTRY.append(tool_def)
    return tool_def


def get_all_tools() -> list[ToolDef]:
    """Return all registered tool definitions."""
    return list(_TOOL_REGISTRY)


def get_read_tools() -> list[ToolDef]:
    """Return only 'read' permission tools (safe for user agents)."""
    return [t for t in _TOOL_REGISTRY if t.permission == "read"]


def get_write_tools() -> list[ToolDef]:
    """Return only 'write' permission tools."""
    return [t for t in _TOOL_REGISTRY if t.permission == "write"]


def get_tool(name: str) -> ToolDef | None:
    """Look up a tool by name."""
    for t in _TOOL_REGISTRY:
        if t.name == name:
            return t
    return None


def ensure_tools_loaded():
    """Import all tool modules to trigger registration."""
    global _ALL_MODULES_LOADED
    if _ALL_MODULES_LOADED:
        return
    # Importing the modules triggers their register_tool() calls
    import alteris.mcp_tools.kg_tools  # noqa: F401
    import alteris.mcp_tools.cq_tools  # noqa: F401
    import alteris.mcp_tools.pipeline_tools  # noqa: F401
    import alteris.mcp_tools.agent_tools  # noqa: F401
    import alteris.mcp_tools.onboarding_tools  # noqa: F401
    import alteris.mcp_tools.story_tools  # noqa: F401
    import alteris.mcp_tools.briefing_tools  # noqa: F401
    import alteris.mcp_tools.person_model_tools  # noqa: F401
    _ALL_MODULES_LOADED = True
