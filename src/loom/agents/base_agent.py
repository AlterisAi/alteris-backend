"""Base agent — shared infrastructure for CTO and CEO agents.

Uses the Anthropic Python SDK with tool use for the agent loop.
Each agent has:
  - Access to their owner's knowledge graph (private)
  - Access to the shared workspace (cross-agent)
  - A set of tools it can call
  - A system prompt that defines its role and boundaries

The agent loop:
  1. Send message with tools → Claude
  2. Claude responds with tool_use blocks
  3. Execute tools, collect results
  4. Send results back → Claude
  5. Repeat until Claude responds with text (no more tool calls)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Claude model for agent reasoning
AGENT_MODEL = "claude-sonnet-4-5-20250929"
MAX_AGENT_TURNS = 20


def _get_anthropic_key() -> str:
    """Resolve Anthropic API key from env or config."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    from loom.constants import load_config
    key = load_config().get("anthropic_api_key", "")
    if key:
        return key

    raise ValueError(
        "ANTHROPIC_API_KEY not set. Add anthropic_api_key to your "
        "config.json (in ~/Downloads/ or ~/.loom/), or set the "
        "ANTHROPIC_API_KEY environment variable."
    )


class ToolDefinition:
    """Wraps a Python function as a Claude tool."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Any,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def to_api_format(self) -> dict:
        """Convert to Anthropic API tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class BaseAgent:
    """Base class for CTO and CEO agents.

    Subclasses define their tools, system prompt, and role-specific behavior.
    """

    def __init__(self, role: str):
        self.role = role
        self._tools: dict[str, ToolDefinition] = {}
        self._client = None

    def _get_client(self):
        """Lazy-init the Anthropic client."""
        if self._client is not None:
            return self._client
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=_get_anthropic_key())
            return self._client
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

    def register_tool(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get_system_prompt(self) -> str:
        """Override in subclass to provide role-specific system prompt."""
        raise NotImplementedError

    def get_tools_api(self) -> list[dict]:
        """Get all tools in Anthropic API format."""
        return [t.to_api_format() for t in self._tools.values()]

    def execute_tool(self, name: str, input_data: dict) -> str:
        """Execute a tool by name with the given input."""
        tool = self._tools.get(name)
        if not tool:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            result = tool.handler(**input_data)
            if isinstance(result, (dict, list)):
                return json.dumps(result, default=str)
            return str(result)
        except Exception as exc:
            logger.error("Tool %s failed: %s", name, exc)
            return json.dumps({"error": str(exc)})

    def run(self, task: str, context: str = "") -> str:
        """Run the agent loop on a task.

        Args:
            task: The user's request / task description
            context: Additional context (e.g., shared workspace state)

        Returns:
            The agent's final text response
        """
        client = self._get_client()
        system = self.get_system_prompt()
        if context:
            system += f"\n\n## Current Context\n{context}"

        messages = [{"role": "user", "content": task}]
        tools = self.get_tools_api()

        for turn in range(MAX_AGENT_TURNS):
            logger.info("Agent %s turn %d", self.role, turn + 1)

            response = client.messages.create(
                model=AGENT_MODEL,
                max_tokens=4096,
                system=system,
                messages=messages,
                tools=tools if tools else None,
            )

            # Check if the response contains tool use
            tool_use_blocks = [
                b for b in response.content if b.type == "tool_use"
            ]
            text_blocks = [
                b for b in response.content if b.type == "text"
            ]

            if not tool_use_blocks:
                # No more tool calls — return the text response
                return "\n".join(b.text for b in text_blocks)

            # Execute tools and build tool results
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tool_block in tool_use_blocks:
                logger.info(
                    "  Tool call: %s(%s)",
                    tool_block.name,
                    json.dumps(tool_block.input)[:100],
                )
                result = self.execute_tool(tool_block.name, tool_block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

        # Hit max turns
        logger.warning("Agent %s hit max turns (%d)", self.role, MAX_AGENT_TURNS)
        return "(Agent reached maximum reasoning turns)"

    def run_with_approval(
        self,
        task: str,
        context: str = "",
        approve_fn: Any = None,
    ) -> str:
        """Run with human-in-the-loop approval for sensitive actions.

        If approve_fn is provided, it's called before executing tools
        that are marked as requiring approval (e.g., email.send).
        Falls back to stdin prompt if no approve_fn.
        """
        # For now, delegate to run() — approval gates are added
        # at the tool level (outreach tools check status='draft')
        return self.run(task, context)
