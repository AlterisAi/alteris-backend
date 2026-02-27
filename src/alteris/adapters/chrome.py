"""Chrome/Chromium adapter — reads browser history from Chromium-based browsers.

Supports Chrome, Arc, Brave, Edge, and Vivaldi via a single ChromiumAdapter class.
Chrome holds an exclusive lock on its History database while running, so we must
copy it to a temp file before reading.

Chrome timestamps are microseconds since 1601-01-01 (Windows/Chrome epoch).
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

from alteris.adapters import (
    AvailabilityResult,
    IngestResult,
    SchemaResult,
    SourceAdapter,
    check_sqlite_tables,
    row_val,
)
from alteris.constants import (
    CHROME_EPOCH_OFFSET,
    EVENT_TYPE_BROWSER_VISIT,
)
from alteris.models import Event
from alteris.privacy import SensitivityLevel

logger = logging.getLogger(__name__)

# Internal browser URLs to skip — these are not real page visits
_SKIP_URL_PREFIXES = (
    "about:",
    "chrome://",
    "chrome-extension://",
    "edge://",
    "brave://",
    "vivaldi://",
)

# Core transition types (lower 8 bits of the transition bitmask)
TRANSITION_TYPES = {
    0: "link",
    1: "typed",
    2: "auto_bookmark",
    3: "auto_subframe",
    4: "manual_subframe",
    5: "generated",
    6: "auto_toplevel",
    7: "form_submit",
    8: "reload",
    9: "keyword",
    10: "keyword_generated",
}


class ChromiumAdapter(SourceAdapter):
    """Reads browser history from a Chromium-based browser.

    Works with Chrome, Arc, Brave, Edge, and Vivaldi. Each browser stores
    its History SQLite database in a browser-specific Application Support
    subdirectory but uses the same schema.
    """

    def __init__(self, browser_name: str, db_path: Path):
        self._browser_name = browser_name
        self._db_path = db_path

    @property
    def source_name(self) -> str:
        return self._browser_name

    def _copy_db(self) -> Path:
        """Copy Chrome DB to temp location to avoid lock conflicts.

        Chromium holds an exclusive lock on the History file while running.
        We copy the database (plus WAL/SHM journals if present) to a temp
        directory so we can read without conflicting with the browser process.
        """
        tmp = Path(tempfile.mkdtemp()) / "History"
        shutil.copy2(self._db_path, tmp)
        for suffix in ["-wal", "-shm"]:
            src = Path(str(self._db_path) + suffix)
            if src.exists():
                shutil.copy2(src, Path(str(tmp) + suffix))
        return tmp

    def check_availability(self) -> AvailabilityResult:
        if not self._db_path.exists():
            return AvailabilityResult(
                available=False,
                source=self._browser_name,
                reason="database_not_found",
                user_action=f"{self._browser_name.capitalize()} History database not found at {self._db_path}. "
                            f"Ensure {self._browser_name.capitalize()} is installed.",
            )
        return AvailabilityResult(available=True, source=self._browser_name)

    def check_schema(self) -> SchemaResult:
        if not self._db_path.exists():
            return SchemaResult(
                compatible=False,
                source=self._browser_name,
                warnings=["Database not found"],
            )
        # Copy to temp to avoid lock, then check tables
        tmp_path = self._copy_db()
        try:
            return check_sqlite_tables(
                tmp_path, self._browser_name,
                required=["visits", "urls"],
                optional=["content_annotations", "context_annotations"],
            )
        finally:
            shutil.rmtree(tmp_path.parent, ignore_errors=True)

    def _has_table(self, conn: sqlite3.Connection, table_name: str) -> bool:
        """Check if a table exists in the database."""
        row = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row[0] > 0

    def ingest(self, since_ts: int = 0, limit: int = 0) -> IngestResult:
        t0 = time.time()

        if not self._db_path.exists():
            return IngestResult(
                source=self._browser_name,
                errors=[f"{self._browser_name.capitalize()} History database not found"],
            )

        tmp_path = self._copy_db()
        try:
            return self._ingest_from_copy(tmp_path, since_ts, limit, t0)
        finally:
            shutil.rmtree(tmp_path.parent, ignore_errors=True)

    def _ingest_from_copy(
        self, db_path: Path, since_ts: int, limit: int, t0: float,
    ) -> IngestResult:
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.OperationalError:
            return IngestResult(
                source=self._browser_name,
                errors=[f"Cannot open {self._browser_name} History database copy."],
            )

        # Convert Unix seconds to Chrome microseconds for the WHERE clause
        if since_ts > 0:
            cutoff_chrome = (since_ts * 1_000_000) + CHROME_EPOCH_OFFSET
        else:
            cutoff_chrome = 0

        # Check for optional annotation tables
        has_content_annotations = self._has_table(conn, "content_annotations")
        has_context_annotations = self._has_table(conn, "context_annotations")

        # Build query with optional LEFT JOINs
        select_cols = [
            "v.id AS visit_id",
            "v.visit_time",
            "v.visit_duration",
            "v.transition",
            "u.url",
            "u.title",
        ]
        joins = "JOIN urls u ON v.url = u.id"

        if has_content_annotations:
            select_cols.append("ca.page_language")
            joins += "\nLEFT JOIN content_annotations ca ON v.id = ca.visit_id"
        if has_context_annotations:
            select_cols.append("cxa.duration_since_last_visit")
            joins += "\nLEFT JOIN context_annotations cxa ON v.id = cxa.visit_id"

        query = f"""
            SELECT {', '.join(select_cols)}
            FROM visits v
            {joins}
            WHERE v.visit_time > ?
            ORDER BY v.visit_time DESC
        """

        if limit > 0:
            query += f" LIMIT {limit}"

        events: list[Event] = []
        errors: list[str] = []

        try:
            for row in conn.execute(query, (cutoff_chrome,)):
                visit_id = row["visit_id"]
                url = str(row_val(row, "url", ""))
                title = str(row_val(row, "title", ""))
                visit_time = row_val(row, "visit_time", 0)
                visit_duration = row_val(row, "visit_duration", 0) or 0
                transition = row_val(row, "transition", 0)

                if not url:
                    continue

                # Skip internal browser URLs
                if any(url.startswith(prefix) for prefix in _SKIP_URL_PREFIXES):
                    continue

                # Convert Chrome timestamp to Unix seconds
                unix_ts = (int(visit_time) - CHROME_EPOCH_OFFSET) // 1_000_000

                # Decode transition type from lower 8 bits
                transition_core = (int(transition) if transition else 0) & 0xFF
                transition_type = TRANSITION_TYPES.get(transition_core, "other")

                # Extract optional annotation fields
                page_language = ""
                if has_content_annotations:
                    page_language = str(row_val(row, "page_language", ""))

                domain = urlparse(url).netloc

                source_id = str(visit_id)
                event_id = Event.make_id(self._browser_name, source_id)
                content_hash = Event.content_hash_of(url)

                events.append(Event(
                    id=event_id,
                    source=self._browser_name,
                    source_id=source_id,
                    event_type=EVENT_TYPE_BROWSER_VISIT,
                    timestamp=unix_ts,
                    participants=(),
                    raw_content=url,
                    content_hash=content_hash,
                    metadata={
                        "subject": title or url[:100],
                        "url": url,
                        "domain": domain,
                        "title": title,
                        "is_from_me": True,
                        "thread_id": "",
                        "visit_duration_us": int(visit_duration),
                        "transition_type": transition_type,
                        "page_language": page_language,
                    },
                    sensitivity=SensitivityLevel.SENSITIVE,
                ))

        except sqlite3.OperationalError as exc:
            errors.append(f"{self._browser_name} history query failed: {exc}")
        finally:
            conn.close()

        return IngestResult(
            source=self._browser_name,
            events=events,
            errors=errors,
            duration_seconds=time.time() - t0,
        )


def get_chromium_adapters() -> list[ChromiumAdapter]:
    """Discover installed Chromium browsers and return adapters for each."""
    from alteris.constants import CHROMIUM_BROWSER_PATHS

    adapters = []
    for name, path in CHROMIUM_BROWSER_PATHS.items():
        if path.exists():
            adapters.append(ChromiumAdapter(name, path))
    return adapters
