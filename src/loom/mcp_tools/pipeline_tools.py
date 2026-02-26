"""Pipeline and admin MCP tools.

These are 'write' tools — only available to the app, not user agents.
"""

from __future__ import annotations

import json
import logging
import time

from loom.mcp_tools import ToolDef, ToolParam, register_tool
from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)


def handle_loom_run_pipeline(
    store: LayeredGraphStore,
    stages: list[str] | None = None,
    hours: int = 168,
    sources: list[str] | None = None,
    llm: str = "gemini",
    **kwargs,
) -> dict:
    """Run the pipeline (or specific stages)."""
    import argparse

    args = argparse.Namespace(
        db_path=store._db_path if store._db_path != ":memory:" else None,
        hours=hours,
        since=None,
        sources=sources,
        limit=None,
        dry_run=False,
        llm=llm,
        verbose=False,
        lens="chief_of_staff",
        brief=False,
        days=7,
        lookback=30,
        tz="America/Los_Angeles",
        save=None,
    )

    from loom.cli import run_pipeline_stages

    # Close the shared store so the pipeline can open its own connection
    db_path = store._db_path
    store.close()

    try:
        result = run_pipeline_stages(args)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        result = {"error": str(exc)}

    # Reopen
    store.__init__(db_path)

    return result


def handle_loom_run_ingest(
    store: LayeredGraphStore,
    sources: list[str] | None = None,
    hours: int = 168,
    **kwargs,
) -> dict:
    """Run only the ingest stage."""
    import argparse
    from loom.cli import cmd_ingest

    args = argparse.Namespace(
        db_path=store._db_path if store._db_path != ":memory:" else None,
        hours=hours,
        since=None,
        sources=sources,
        limit=None,
        dry_run=False,
        llm="gemini",
        verbose=False,
    )

    db_path = store._db_path
    store.close()

    try:
        cmd_ingest(args)
        result = {"status": "complete"}
    except Exception as exc:
        result = {"error": str(exc)}

    store.__init__(db_path)
    return result


def handle_loom_update_profile(
    store: LayeredGraphStore,
    profile_data: dict | None = None,
    **kwargs,
) -> dict:
    """Update the user's profile.yaml."""
    if not profile_data:
        return {"error": "profile_data is required"}

    from pathlib import Path
    from loom.constants import LOOM_DIR

    profile_path = LOOM_DIR / "profile.yaml"
    LOOM_DIR.mkdir(parents=True, exist_ok=True)

    # Read existing profile
    existing = {}
    if profile_path.exists():
        try:
            import yaml
            existing = yaml.safe_load(profile_path.read_text()) or {}
        except ImportError:
            # Manual YAML parsing fallback
            pass
        except Exception:
            pass

    # Merge
    existing.update(profile_data)

    # Write YAML
    lines = []
    for key, value in existing.items():
        if isinstance(value, list):
            lines.append(f"{key}: {json.dumps(value)}")
        elif isinstance(value, str):
            lines.append(f"{key}: '{value}'")
        else:
            lines.append(f"{key}: {value}")

    profile_path.write_text("\n".join(lines) + "\n")

    return {"updated": True, "path": str(profile_path)}


def handle_loom_reset_spend(
    store: LayeredGraphStore,
    bypass_code: str = "",
    date: str | None = None,
    **kwargs,
) -> dict:
    """Reset daily spend using a bypass code."""
    if not bypass_code:
        return {"error": "bypass_code is required"}

    try:
        from loom.spend import verify_bypass_code, reset_daily_spend
    except ImportError:
        return {"error": "spend module not available"}

    if not verify_bypass_code(bypass_code, date):
        return {"error": "invalid bypass code"}

    deleted = reset_daily_spend(store, date)
    return {"reset": True, "rows_deleted": deleted}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

register_tool(ToolDef(
    name="loom_run_pipeline",
    description="Run the full processing pipeline (ingest, resolve, score, triage, extract, synthesize).",
    permission="write",
    params=[
        ToolParam("stages", "array", "Specific stages to run (default: all)"),
        ToolParam("hours", "integer", "Lookback hours (default: 168)", default=168),
        ToolParam("sources", "array", "Sources to ingest"),
        ToolParam("llm", "string", "LLM provider", default="gemini", enum=["gemini", "ollama", "mock"]),
    ],
    handler=handle_loom_run_pipeline,
))

register_tool(ToolDef(
    name="loom_run_ingest",
    description="Run only the data ingestion stage (no LLM required).",
    permission="write",
    params=[
        ToolParam("sources", "array", "Sources to ingest (default: all)"),
        ToolParam("hours", "integer", "Lookback hours (default: 168)", default=168),
    ],
    handler=handle_loom_run_ingest,
))

register_tool(ToolDef(
    name="loom_update_profile",
    description="Update the user's profile (name, emails, timezone, etc.).",
    permission="write",
    params=[
        ToolParam("profile_data", "object", "Profile fields to update", required=True),
    ],
    handler=handle_loom_update_profile,
))

register_tool(ToolDef(
    name="loom_reset_spend",
    description="Reset daily API spend using a bypass code.",
    permission="write",
    params=[
        ToolParam("bypass_code", "string", "Bypass code for the current date", required=True),
        ToolParam("date", "string", "Date to reset (default: today)"),
    ],
    handler=handle_loom_reset_spend,
))
