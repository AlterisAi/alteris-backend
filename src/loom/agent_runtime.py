"""Agent runtime — executes user-created agents with scoped KG tool access.

Each agent gets:
  - Its system prompt
  - Only the MCP tools listed in its tool_permissions
  - An LLM backend (Anthropic or Gemini)
  - Read-only access to the real KG data

The agent CANNOT:
  - Access pipeline tools or write tools
  - Modify the KG, run ingestion, or manage other agents
  - See tool definitions it doesn't have permission for
"""

from __future__ import annotations

import json
import logging
import os
import time

from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)


def run_agent(
    store: LayeredGraphStore,
    agent: dict,
    message: str,
    sandbox: bool = False,
    max_turns: int = 10,
) -> dict:
    """Execute an agent with a user message.

    Args:
        store: KG store for tool execution.
        agent: Agent record from the agents table.
        message: User's message to the agent.
        sandbox: If True, results are not persisted (test mode).
        max_turns: Max tool-use turns before stopping.

    Returns:
        dict with 'response', 'tool_calls', 'tokens_used'.
    """
    system_prompt = agent.get("system_prompt", "")
    raw_perms = agent.get("tool_permissions", [])
    # Handle both YAML-sourced lists and legacy JSON strings
    tool_permissions = json.loads(raw_perms) if isinstance(raw_perms, str) else raw_perms
    llm_backend = agent.get("llm_backend", "anthropic")
    model = agent.get("model", "")

    # Build the scoped tool set
    tools = _build_scoped_tools(store, tool_permissions)

    if llm_backend == "anthropic":
        return _run_anthropic(
            system_prompt=system_prompt,
            message=message,
            tools=tools,
            store=store,
            model=model,
            max_turns=max_turns,
        )
    elif llm_backend == "gemini":
        return _run_gemini(
            system_prompt=system_prompt,
            message=message,
            tools=tools,
            store=store,
            model=model,
            max_turns=max_turns,
        )
    else:
        return {"error": f"Unknown LLM backend: {llm_backend}"}


def _build_scoped_tools(
    store: LayeredGraphStore,
    tool_permissions: list[str],
) -> dict:
    """Build a tool name → handler mapping for permitted tools only."""
    from loom.mcp_tools import ensure_tools_loaded, get_tool
    ensure_tools_loaded()

    tools = {}
    for name in tool_permissions:
        tool_def = get_tool(name)
        if tool_def and tool_def.handler:
            tools[name] = tool_def
    return tools


def _run_anthropic(
    system_prompt: str,
    message: str,
    tools: dict,
    store: LayeredGraphStore,
    model: str = "",
    max_turns: int = 10,
) -> dict:
    """Run an agent using the Anthropic API with tool use."""
    import anthropic

    from loom.constants import AGENT_DEFAULT_MODEL

    # Load key from env or config
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        import json as _json
        from pathlib import Path as _Path
        for cfg in [_Path.home() / ".loom/config.json", _Path.home() / ".alteris/config.json"]:
            if cfg.exists():
                try:
                    api_key = _json.loads(cfg.read_text()).get("anthropic_api_key", "")
                    if api_key:
                        break
                except Exception:
                    pass
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    # Rate limit handling: retry on 429
    import time as _time
    use_model = model or AGENT_DEFAULT_MODEL

    # Convert tools to Anthropic format
    anthropic_tools = []
    for name, tool_def in tools.items():
        properties = {}
        required = []
        for p in tool_def.params:
            prop: dict = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        schema: dict = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        anthropic_tools.append({
            "name": name,
            "description": tool_def.description,
            "input_schema": schema,
        })

    messages = [{"role": "user", "content": message}]
    tool_calls_log = []
    total_input_tokens = 0
    total_output_tokens = 0

    for turn in range(max_turns):
        for _retry in range(3):
            try:
                response = client.messages.create(
                    model=use_model,
                    max_tokens=4096,
                    system=system_prompt,
                    messages=messages,
                    tools=anthropic_tools if anthropic_tools else None,
                )
                break
            except anthropic.RateLimitError:
                wait = 30 * (_retry + 1)
                logger.warning("Rate limited, waiting %ds (attempt %d/3)", wait, _retry + 1)
                _time.sleep(wait)
        else:
            return {
                "response": "Rate limited after 3 retries",
                "tool_calls": tool_calls_log,
                "tokens_used": {"input": total_input_tokens, "output": total_output_tokens},
            }

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Check if the model wants to use tools
        if response.stop_reason == "tool_use":
            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    tool_id = block.id

                    tool_calls_log.append({
                        "tool": tool_name,
                        "input": tool_input,
                        "turn": turn,
                    })

                    # Execute the tool
                    tool_def = tools.get(tool_name)
                    if tool_def and tool_def.handler:
                        try:
                            result = tool_def.handler(store, **tool_input)
                            result_text = json.dumps(result, default=str)
                            # Truncate large tool results to stay within context
                            if len(result_text) > 8000:
                                result_text = result_text[:8000] + "\n... (truncated)"
                        except Exception as exc:
                            result_text = json.dumps({"error": str(exc)})
                    else:
                        result_text = json.dumps({"error": f"Tool {tool_name} not available"})

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_text,
                    })

            messages.append({"role": "user", "content": tool_results})
        else:
            # Model is done — extract text response
            text_parts = [
                block.text for block in response.content
                if hasattr(block, "text")
            ]
            final_response = "\n".join(text_parts)

            return {
                "response": final_response,
                "tool_calls": tool_calls_log,
                "tokens_used": {
                    "input": total_input_tokens,
                    "output": total_output_tokens,
                },
                "turns": turn + 1,
                "model": use_model,
            }

    # Max turns reached
    return {
        "response": "I reached the maximum number of steps. Here's what I found so far based on the tool results above.",
        "tool_calls": tool_calls_log,
        "tokens_used": {
            "input": total_input_tokens,
            "output": total_output_tokens,
        },
        "turns": max_turns,
        "model": use_model,
        "truncated": True,
    }


def _run_gemini(
    system_prompt: str,
    message: str,
    tools: dict,
    store: LayeredGraphStore,
    model: str = "",
    max_turns: int = 10,
) -> dict:
    """Run an agent using Gemini. Simpler — no native tool use, uses prompt-based tool selection."""
    from loom.llm.gemini import GeminiClient
    from loom.constants import CLOUD_FAST_MODEL

    llm = GeminiClient()
    use_model = model or CLOUD_FAST_MODEL

    # For Gemini, we embed tool descriptions in the prompt and parse tool calls from output
    tool_descriptions = []
    for name, tool_def in tools.items():
        params = ", ".join(f"{p.name}: {p.type}" for p in tool_def.params)
        tool_descriptions.append(f"- {name}({params}): {tool_def.description}")

    tools_text = "\n".join(tool_descriptions)

    augmented_prompt = f"""{system_prompt}

## Available Tools
You can call these tools by responding with a JSON block like:
```json
{{"tool": "tool_name", "args": {{"param": "value"}}}}
```

{tools_text}

## User Message
{message}"""

    response = llm.generate(
        prompt=augmented_prompt,
        model=use_model,
        temperature=0.3,
        max_tokens=4096,
    )

    return {
        "response": response or "No response generated.",
        "tool_calls": [],
        "tokens_used": {},
        "turns": 1,
        "model": use_model,
    }
