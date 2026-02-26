"""File-based agent storage.

Each agent is a YAML file in ~/.loom/agents/. The filename (without extension)
is the agent ID. Built-in example agents are shipped in src/loom/agents/specs/
and copied to the user's directory on first use.

YAML was chosen over SQLite for agents because:
  - Human-editable (users can tweak system prompts in their editor)
  - Version-controllable (easy to share, back up, or diff)
  - Transparent (ls ~/.loom/agents/ shows everything)
  - Each agent is self-contained in one file
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import yaml

from loom.constants import AGENTS_DIR, BUILTIN_AGENTS_DIR

logger = logging.getLogger(__name__)

# Fields that are valid in an agent YAML spec
_AGENT_FIELDS = {
    "name", "description", "system_prompt", "tool_permissions",
    "llm_backend", "model", "trigger", "trigger_config",
    "status",
}

_DEFAULTS = {
    "description": "",
    "llm_backend": "anthropic",
    "model": "",
    "trigger": "manual",
    "trigger_config": None,
    "status": "draft",
    "tool_permissions": [],
}


def ensure_agents_dir() -> Path:
    """Create ~/.loom/agents/ if it doesn't exist. Seed with built-in specs."""
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)

    if BUILTIN_AGENTS_DIR.exists():
        for spec_file in BUILTIN_AGENTS_DIR.glob("*.yaml"):
            target = AGENTS_DIR / spec_file.name
            if not target.exists():
                shutil.copy2(spec_file, target)
                logger.info("Seeded built-in agent: %s", spec_file.stem)

    return AGENTS_DIR


def _load_yaml(path: Path) -> dict | None:
    """Load and validate a single agent YAML file."""
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return None
        # Apply defaults
        for key, default in _DEFAULTS.items():
            data.setdefault(key, default)
        # Ensure required fields
        if not data.get("name") or not data.get("system_prompt"):
            logger.warning("Agent %s missing name or system_prompt", path.stem)
            return None
        # ID comes from filename
        data["id"] = path.stem
        # File timestamps
        stat = path.stat()
        data["created_at"] = int(stat.st_birthtime) if hasattr(stat, "st_birthtime") else int(stat.st_ctime)
        data["updated_at"] = int(stat.st_mtime)
        return data
    except Exception as exc:
        logger.warning("Failed to load agent %s: %s", path, exc)
        return None


def _save_yaml(agent_id: str, data: dict) -> Path:
    """Write agent data to YAML file. Returns the path."""
    ensure_agents_dir()
    path = AGENTS_DIR / f"{agent_id}.yaml"

    # Only write spec fields, not computed ones
    out = {}
    for key in _AGENT_FIELDS:
        if key in data:
            out[key] = data[key]

    with open(path, "w") as f:
        yaml.dump(out, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return path


def list_agents(status: str | None = None) -> list[dict]:
    """List all agents from ~/.loom/agents/. Optionally filter by status."""
    ensure_agents_dir()
    agents = []
    for path in sorted(AGENTS_DIR.glob("*.yaml")):
        agent = _load_yaml(path)
        if agent is None:
            continue
        if status and agent.get("status") != status:
            continue
        agents.append(agent)
    return agents


def get_agent(agent_id: str) -> dict | None:
    """Load a single agent by ID (filename stem)."""
    ensure_agents_dir()
    path = AGENTS_DIR / f"{agent_id}.yaml"
    if not path.exists():
        return None
    return _load_yaml(path)


def create_agent(
    name: str,
    system_prompt: str,
    tool_permissions: list[str] | None = None,
    description: str = "",
    llm_backend: str = "anthropic",
    model: str = "",
    trigger: str = "manual",
    trigger_config: dict | None = None,
    status: str = "draft",
    agent_id: str | None = None,
) -> str:
    """Create a new agent YAML file. Returns the agent ID."""
    if agent_id is None:
        # Derive ID from name: lowercase, hyphens, no special chars
        agent_id = name.lower().replace(" ", "-")
        agent_id = "".join(c for c in agent_id if c.isalnum() or c == "-")
        agent_id = agent_id.strip("-")

    # Avoid clobbering existing agents
    ensure_agents_dir()
    path = AGENTS_DIR / f"{agent_id}.yaml"
    if path.exists():
        # Append a short suffix
        agent_id = f"{agent_id}-{int(time.time()) % 10000}"

    from loom.constants import AGENT_READ_TOOLS
    if tool_permissions is None:
        tool_permissions = list(AGENT_READ_TOOLS)

    data = {
        "name": name,
        "description": description,
        "system_prompt": system_prompt,
        "tool_permissions": tool_permissions,
        "llm_backend": llm_backend,
        "model": model,
        "trigger": trigger,
        "trigger_config": trigger_config,
        "status": status,
    }

    _save_yaml(agent_id, data)
    return agent_id


def update_agent(agent_id: str, **fields) -> bool:
    """Update specific fields on an agent. Returns True if agent existed."""
    path = AGENTS_DIR / f"{agent_id}.yaml"
    if not path.exists():
        return False

    agent = _load_yaml(path)
    if agent is None:
        return False

    for key, val in fields.items():
        if key in _AGENT_FIELDS:
            agent[key] = val

    _save_yaml(agent_id, agent)
    return True


def delete_agent(agent_id: str) -> bool:
    """Delete an agent YAML file. Returns True if it existed."""
    path = AGENTS_DIR / f"{agent_id}.yaml"
    if not path.exists():
        return False
    path.unlink()
    return True
