"""Source adapter framework for Alteris ingestion.

Each adapter reads from a Mac-native data source (SQLite DB, API, filesystem)
and normalizes records into the Event schema. Adapters are idempotent:
re-ingesting the same source produces zero new events thanks to deterministic
Event IDs (content-addressable hashes of source + source_id).

Every adapter implements three methods:
    check_availability()  -- can we reach the data source?
    check_schema()        -- does the DB schema match expectations?
    ingest()              -- read source, return list of Events

Availability checks return actionable error messages so the user knows
exactly which System Settings permission to grant.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from alteris.constants import (
    BODY_PREVIEW_LEN,
    SOURCE_ID_HASH_LEN,
)
from alteris.models import Event
from alteris.privacy import SensitivityLevel

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Result types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AvailabilityResult:
    """Result of check_availability()."""
    available: bool
    source: str
    reason: str = ""
    user_action: str = ""


@dataclass
class SchemaResult:
    """Result of check_schema()."""
    compatible: bool
    source: str
    missing_tables: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class IngestResult:
    """Result of ingest()."""
    source: str
    events: list[Event] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Base adapter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SourceAdapter(ABC):
    """Abstract base for all source adapters."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Short identifier for this source (e.g. 'mail', 'imessage')."""

    @abstractmethod
    def check_availability(self) -> AvailabilityResult:
        """Check if the data source is accessible."""

    @abstractmethod
    def check_schema(self) -> SchemaResult:
        """Check if the data source schema is compatible."""

    @abstractmethod
    def ingest(self, since_ts: int = 0, limit: int = 0) -> IngestResult:
        """Read events from the source.

        Args:
            since_ts: Unix timestamp. Only return events after this time.
                      0 means full bootstrap (all available data).
            limit: Maximum events to return. 0 means no limit.

        Returns:
            IngestResult with events and any errors.
        """


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_sqlite_readable(path: Path, source: str) -> AvailabilityResult:
    """Check that a SQLite database exists and is readable."""
    if not path.exists():
        return AvailabilityResult(
            available=False,
            source=source,
            reason="database_not_found",
            user_action=f"Database not found at {path}. Ensure the app is installed.",
        )
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.execute("SELECT 1")
        conn.close()
        return AvailabilityResult(available=True, source=source)
    except sqlite3.OperationalError:
        return AvailabilityResult(
            available=False,
            source=source,
            reason="permission_denied",
            user_action=(
                "Grant Full Disk Access in System Settings > "
                "Privacy & Security > Full Disk Access"
            ),
        )


def check_sqlite_tables(
    path: Path, source: str, required: list[str], optional: list[str] | None = None,
) -> SchemaResult:
    """Check that expected tables exist in a SQLite database."""
    optional = optional or []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        tables = {
            r[0] for r in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()
    except sqlite3.OperationalError:
        return SchemaResult(
            compatible=False, source=source,
            warnings=["Cannot open database to check schema"],
        )

    missing = [t for t in required if t not in tables]
    if missing:
        return SchemaResult(
            compatible=False, source=source, missing_tables=missing,
        )
    return SchemaResult(compatible=True, source=source)


def make_source_id_hash(*parts: str) -> str:
    """Create a deterministic source_id from arbitrary parts."""
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:SOURCE_ID_HASH_LEN]


def check_file_readable(path: Path, source: str) -> AvailabilityResult:
    """Check that a plain file exists and is readable (for non-SQLite sources)."""
    if not path.exists():
        return AvailabilityResult(
            available=False,
            source=source,
            reason="file_not_found",
            user_action=f"File not found at {path}.",
        )
    try:
        with open(path, "rb") as f:
            f.read(1)
        return AvailabilityResult(available=True, source=source)
    except PermissionError:
        return AvailabilityResult(
            available=False,
            source=source,
            reason="permission_denied",
            user_action=(
                "Grant Full Disk Access in System Settings > "
                "Privacy & Security > Full Disk Access"
            ),
        )


def row_val(row: sqlite3.Row, key: str, default: object = "") -> object:
    """Safe extraction from sqlite3.Row with default."""
    try:
        val = row[key]
        return val if val is not None else default
    except (IndexError, KeyError):
        return default


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_all_adapters() -> list[SourceAdapter]:
    """Get instances of all registered adapters (declarative sources only)."""
    from alteris.adapters.calendar import CalendarAdapter
    from alteris.adapters.calls import MacOSCallAdapter, WhatsAppCallAdapter
    from alteris.adapters.contacts import ContactsAdapter
    from alteris.adapters.granola import GranolaAdapter
    from alteris.adapters.imessage import IMessageAdapter
    from alteris.adapters.mail import MailAdapter
    from alteris.adapters.slack import SlackAdapter
    from alteris.adapters.whatsapp import WhatsAppAdapter

    return [
        MailAdapter(),
        IMessageAdapter(),
        WhatsAppAdapter(),
        CalendarAdapter(),
        ContactsAdapter(),
        GranolaAdapter(),
        SlackAdapter(),
        MacOSCallAdapter(),
        WhatsAppCallAdapter(),
    ]


def get_ambient_adapters() -> list[SourceAdapter]:
    """Get instances of all ambient data source adapters."""
    from alteris.adapters.chrome import get_chromium_adapters
    from alteris.adapters.knowledgec import KnowledgeCAdapter
    from alteris.adapters.notes import NotesAdapter
    from alteris.adapters.safari import SafariAdapter
    from alteris.adapters.shell_history import ShellHistoryAdapter

    adapters: list[SourceAdapter] = [
        KnowledgeCAdapter(),
        SafariAdapter(),
        NotesAdapter(),
        ShellHistoryAdapter(),
    ]
    adapters.extend(get_chromium_adapters())
    return adapters


def get_adapter(source_name: str) -> SourceAdapter | None:
    """Get a single adapter by source name (searches declarative and ambient)."""
    for adapter in get_all_adapters():
        if adapter.source_name == source_name:
            return adapter
    for adapter in get_ambient_adapters():
        if adapter.source_name == source_name:
            return adapter
    return None
