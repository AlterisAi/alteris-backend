"""Safari adapter — reads browser history from macOS Safari.

Safari stores history in History.db with visit timestamps as CFAbsoluteTime
(seconds since Apple epoch 2001-01-01). history_tags provides WikiData IDs
from Safari's auto-tagging system.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from urllib.parse import urlparse

from loom.adapters import (
    AvailabilityResult,
    IngestResult,
    SchemaResult,
    SourceAdapter,
    check_sqlite_readable,
    check_sqlite_tables,
    row_val,
)
from loom.constants import (
    APPLE_EPOCH_OFFSET,
    EVENT_TYPE_BROWSER_VISIT,
    SAFARI_HISTORY_DB,
)
from loom.models import Event
from loom.privacy import SensitivityLevel

logger = logging.getLogger(__name__)

# URL schemes that are internal/non-navigational — skip these
_SKIP_SCHEMES = ("about:", "blob:", "data:")


class SafariAdapter(SourceAdapter):
    """Reads Safari browser history from History.db."""

    @property
    def source_name(self) -> str:
        return "safari"

    def check_availability(self) -> AvailabilityResult:
        return check_sqlite_readable(SAFARI_HISTORY_DB, "safari")

    def check_schema(self) -> SchemaResult:
        if not SAFARI_HISTORY_DB.exists():
            return SchemaResult(compatible=False, source="safari", warnings=["Database not found"])
        return check_sqlite_tables(
            SAFARI_HISTORY_DB, "safari",
            required=["history_visits", "history_items"],
            optional=["history_tags"],
        )

    def ingest(self, since_ts: int = 0, limit: int = 0) -> IngestResult:
        t0 = time.time()

        if not SAFARI_HISTORY_DB.exists():
            return IngestResult(source="safari", errors=["Safari history database not found"])

        try:
            conn = sqlite3.connect(f"file:{SAFARI_HISTORY_DB}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.OperationalError:
            return IngestResult(
                source="safari",
                errors=["Cannot read Safari history. Grant Full Disk Access in System Settings."],
            )

        # Detect whether history_tags table exists
        has_tags = False
        try:
            tables = {
                r[0] for r in
                conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            has_tags = "history_tags" in tables
        except sqlite3.OperationalError:
            pass

        # Detect whether history_items has a title column
        has_title = False
        try:
            col_info = conn.execute("PRAGMA table_info(history_items)").fetchall()
            has_title = any(c["name"] == "title" for c in col_info)
        except sqlite3.OperationalError:
            pass

        # Convert since_ts (Unix epoch) to CFAbsoluteTime (Apple epoch)
        if since_ts > 0:
            cutoff_cf = since_ts - APPLE_EPOCH_OFFSET
        else:
            cutoff_cf = 0

        # Build query with optional title column and tags join
        title_col = ", hi.title" if has_title else ""
        tags_join = ""
        tags_col = ""
        if has_tags:
            tags_join = "LEFT JOIN history_tags ht ON hv.history_item = ht.history_item"
            tags_col = ", ht.tag"

        fetch_limit = limit if limit > 0 else 500_000

        query = f"""
            SELECT hv.id AS visit_id, hv.visit_time,
                   hi.url, hi.domain_expansion AS domain
                   {title_col}{tags_col}
            FROM history_visits hv
            JOIN history_items hi ON hv.history_item = hi.id
            {tags_join}
            WHERE hv.visit_time > ?
            ORDER BY hv.visit_time DESC
            LIMIT ?
        """

        # When history_tags is joined, multiple rows share the same visit_id.
        # Accumulate tags per visit, emit one event per visit.
        events: list[Event] = []
        errors: list[str] = []

        # Track visits we've already processed (for tag aggregation)
        seen_visits: dict[int, int] = {}  # visit_id -> index in events list

        try:
            for row in conn.execute(query, (cutoff_cf, fetch_limit)):
                visit_id = row_val(row, "visit_id", 0)
                url = str(row_val(row, "url", ""))

                if not url or any(url.startswith(s) for s in _SKIP_SCHEMES):
                    continue

                # If we've seen this visit before (multiple tags), just append the tag
                if has_tags and visit_id in seen_visits:
                    tag = str(row_val(row, "tag", ""))
                    if tag:
                        idx = seen_visits[visit_id]
                        events[idx].metadata["tags"].append(tag)
                    continue

                visit_time = row_val(row, "visit_time", 0)
                ts = int(visit_time) + APPLE_EPOCH_OFFSET

                domain = str(row_val(row, "domain", ""))
                if not domain:
                    domain = urlparse(url).netloc

                title = str(row_val(row, "title", "")) if has_title else ""

                tags_list: list[str] = []
                if has_tags:
                    tag = str(row_val(row, "tag", ""))
                    if tag:
                        tags_list.append(tag)

                source_id = str(visit_id)
                event_id = Event.make_id("safari", source_id)
                content_hash = Event.content_hash_of(url)

                event = Event(
                    id=event_id,
                    source="safari",
                    source_id=source_id,
                    event_type=EVENT_TYPE_BROWSER_VISIT,
                    timestamp=ts,
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
                        "tags": tags_list,
                    },
                    sensitivity=SensitivityLevel.SENSITIVE,
                )
                events.append(event)

                if has_tags:
                    seen_visits[visit_id] = len(events) - 1

        except sqlite3.OperationalError as exc:
            errors.append(f"Safari query failed: {exc}")
        finally:
            conn.close()

        return IngestResult(
            source="safari",
            events=events,
            errors=errors,
            duration_seconds=time.time() - t0,
        )
