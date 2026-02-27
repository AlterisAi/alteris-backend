"""Shell history adapter — reads command history from zsh or bash.

zsh extended history format: `: TIMESTAMP:0;COMMAND`
bash history format: plain commands (one per line, no timestamps)

Includes secret sanitization to strip tokens, passwords, and API keys
from commands before storing them as events.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path

from alteris.adapters import (
    AvailabilityResult,
    IngestResult,
    SchemaResult,
    SourceAdapter,
    check_file_readable,
    make_source_id_hash,
)
from alteris.constants import (
    BASH_HISTORY_FILE,
    EVENT_TYPE_SHELL_COMMAND,
    SHELL_HISTORY_MAX_COMMAND_LEN,
    ZSH_HISTORY_FILE,
)
from alteris.models import Event
from alteris.privacy import SensitivityLevel

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Secret sanitization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Patterns to sanitize (order matters — more specific first)
_SECRET_PATTERNS = [
    # export KEY=value or KEY=value at start of command
    (re.compile(
        r'\b(\w*(?:KEY|TOKEN|SECRET|PASSWORD|PASS|API_KEY|AUTH|CREDENTIAL)\w*)'
        r'\s*=\s*\S+',
        re.IGNORECASE,
    ), r'\1=[REDACTED]'),
    # --token=value, --password=value, --api-key=value
    (re.compile(
        r'(--\w*(?:token|password|pass|secret|key|auth)\w*)[=\s]+\S+',
        re.IGNORECASE,
    ), r'\1=[REDACTED]'),
    # -p password (short flag for mysql, etc)
    (re.compile(r'\s-p\s+\S+'), ' -p [REDACTED]'),
    # curl -H "Authorization: ..."
    (re.compile(
        r'(-H\s+["\']?Authorization:\s*)\S+(?:\s+\S+)?["\']?',
        re.IGNORECASE,
    ), r'\1[REDACTED]"'),
    # curl -u user:password
    (re.compile(r'(-u\s+\S+?:)\S+'), r'\1[REDACTED]'),
    # Bearer tokens
    (re.compile(r'(Bearer\s+)\S+', re.IGNORECASE), r'\1[REDACTED]'),
]


def _sanitize_command(cmd: str) -> str:
    """Strip secrets from a shell command, preserving structure."""
    for pattern, replacement in _SECRET_PATTERNS:
        cmd = pattern.sub(replacement, cmd)
    return cmd[:SHELL_HISTORY_MAX_COMMAND_LEN]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# History file parsers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_zsh_history(path: Path, since_ts: int) -> list[tuple[int, str]]:
    """Parse zsh extended history format: `: TIMESTAMP:0;COMMAND`"""
    entries: list[tuple[int, str]] = []
    try:
        raw = path.read_bytes()
        # zsh history can have mixed encodings
        text = raw.decode("utf-8", errors="replace")
    except OSError:
        return []

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Match zsh extended format
        match = re.match(r'^:\s*(\d+):\d+;(.*)$', line)
        if match:
            ts = int(match.group(1))
            cmd = match.group(2)
            # Handle backslash continuation lines
            while cmd.endswith('\\') and i + 1 < len(lines):
                i += 1
                cmd = cmd[:-1] + '\n' + lines[i]
            if ts >= since_ts:
                entries.append((ts, cmd))
        i += 1
    return entries


def _parse_bash_history(path: Path, since_ts: int) -> list[tuple[int, str]]:
    """Parse bash history (no timestamps — use file mtime)."""
    try:
        mtime = int(path.stat().st_mtime)
        if mtime < since_ts:
            return []
        text = path.read_text(errors="replace")
    except OSError:
        return []
    return [(mtime, line.strip()) for line in text.splitlines() if line.strip()]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Adapter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ShellHistoryAdapter(SourceAdapter):
    """Reads command history from zsh or bash history files."""

    @property
    def source_name(self) -> str:
        return "shell_history"

    def check_availability(self) -> AvailabilityResult:
        result = check_file_readable(ZSH_HISTORY_FILE, "shell_history")
        if result.available:
            return result
        return check_file_readable(BASH_HISTORY_FILE, "shell_history")

    def check_schema(self) -> SchemaResult:
        return SchemaResult(compatible=True, source="shell_history")

    def ingest(self, since_ts: int = 0, limit: int = 0) -> IngestResult:
        t0 = time.time()

        # Try zsh first, fall back to bash
        shell_type = "zsh"
        entries = _parse_zsh_history(ZSH_HISTORY_FILE, since_ts)
        if not entries:
            shell_type = "bash"
            entries = _parse_bash_history(BASH_HISTORY_FILE, since_ts)

        if not entries:
            return IngestResult(
                source="shell_history",
                duration_seconds=time.time() - t0,
            )

        if limit > 0:
            entries = entries[:limit]

        events: list[Event] = []
        for ts, cmd in entries:
            cmd = cmd.strip()
            if not cmd:
                continue

            sanitized_cmd = _sanitize_command(cmd)

            # Deterministic ID: timestamp + command hash
            cmd_hash = hashlib.sha256(cmd.encode()).hexdigest()[:40]
            source_id = make_source_id_hash("shell_history", f"{ts}|{cmd_hash}")
            event_id = Event.make_id("shell_history", source_id)

            events.append(Event(
                id=event_id,
                source="shell_history",
                source_id=source_id,
                event_type=EVENT_TYPE_SHELL_COMMAND,
                timestamp=ts,
                participants=(),
                raw_content=sanitized_cmd,
                content_hash="",
                metadata={
                    "subject": sanitized_cmd[:80],
                    "command": sanitized_cmd,
                    "shell_type": shell_type,
                    "is_from_me": True,
                    "thread_id": "",
                },
                sensitivity=SensitivityLevel.SENSITIVE,
            ))

        return IngestResult(
            source="shell_history",
            events=events,
            duration_seconds=time.time() - t0,
        )
