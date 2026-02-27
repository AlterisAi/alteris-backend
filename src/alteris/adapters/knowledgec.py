"""KnowledgeC adapter — reads app focus/usage events from macOS Knowledge database.

knowledgeC.db tracks app usage via ZOBJECT table with ZSTREAMNAME='/app/usage'.
Provides behavioral signals: which apps the user actually opened, when, for how long.

Timestamps are stored as seconds since Apple epoch (2001-01-01).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime

from alteris.adapters import (
    AvailabilityResult,
    IngestResult,
    SchemaResult,
    SourceAdapter,
    check_sqlite_readable,
    check_sqlite_tables,
    row_val,
)
from alteris.constants import (
    APPLE_EPOCH_OFFSET,
    EVENT_TYPE_APP_FOCUS,
    KNOWLEDGEC_DB,
)
from alteris.models import Event
from alteris.privacy import SensitivityLevel

logger = logging.getLogger(__name__)


class KnowledgeCAdapter(SourceAdapter):
    """Reads app focus/usage events from macOS knowledgeC.db."""

    @property
    def source_name(self) -> str:
        return "knowledgec"

    def check_availability(self) -> AvailabilityResult:
        return check_sqlite_readable(KNOWLEDGEC_DB, "knowledgec")

    def check_schema(self) -> SchemaResult:
        if not KNOWLEDGEC_DB.exists():
            return SchemaResult(compatible=False, source="knowledgec", warnings=["Database not found"])
        return check_sqlite_tables(
            KNOWLEDGEC_DB, "knowledgec",
            required=["ZOBJECT"],
        )

    def ingest(self, since_ts: int = 0, limit: int = 0) -> IngestResult:
        t0 = time.time()

        if not KNOWLEDGEC_DB.exists():
            return IngestResult(source="knowledgec", errors=["knowledgeC database not found"])

        try:
            conn = sqlite3.connect(f"file:{KNOWLEDGEC_DB}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.OperationalError:
            return IngestResult(
                source="knowledgec",
                errors=["Cannot read knowledgeC database. Grant Full Disk Access in System Settings."],
            )

        # Convert since_ts (Unix) to Apple epoch for the WHERE clause.
        # ZSTARTDATE is seconds since Apple epoch (2001-01-01).
        # Apple ts = Unix ts - APPLE_EPOCH_OFFSET
        if since_ts > 0:
            cutoff_apple = since_ts - APPLE_EPOCH_OFFSET
        else:
            cutoff_apple = 0

        query = """
            SELECT Z_PK, ZSTREAMNAME, ZVALUESTRING, ZSTARTDATE, ZENDDATE, ZSECONDSFROMGMT
            FROM ZOBJECT
            WHERE ZSTREAMNAME = '/app/usage'
              AND ZSTARTDATE > ?
            ORDER BY ZSTARTDATE DESC
        """

        if limit > 0:
            query += f" LIMIT {limit}"

        events: list[Event] = []
        errors: list[str] = []

        try:
            for row in conn.execute(query, (cutoff_apple,)):
                bundle_id = row_val(row, "ZVALUESTRING", None)
                if not bundle_id:
                    continue

                bundle_id = str(bundle_id)

                z_pk = row["Z_PK"]
                start_date = row_val(row, "ZSTARTDATE", None)
                end_date = row_val(row, "ZENDDATE", None)

                if start_date is None:
                    continue

                unix_ts = int(start_date) + APPLE_EPOCH_OFFSET

                # Duration: difference between end and start if both present
                if start_date is not None and end_date is not None:
                    duration = int(end_date - start_date)
                    if duration < 0:
                        duration = 0
                else:
                    duration = 0

                day_of_week = datetime.fromtimestamp(unix_ts).strftime("%A")

                source_id = str(z_pk)
                event_id = Event.make_id("knowledgec", source_id)

                events.append(Event(
                    id=event_id,
                    source="knowledgec",
                    source_id=source_id,
                    event_type=EVENT_TYPE_APP_FOCUS,
                    timestamp=unix_ts,
                    participants=(),
                    raw_content=bundle_id,
                    content_hash="",
                    metadata={
                        "subject": bundle_id,
                        "bundle_id": bundle_id,
                        "is_from_me": True,
                        "thread_id": "",
                        "duration_seconds": duration,
                        "day_of_week": day_of_week,
                    },
                    sensitivity=SensitivityLevel.SENSITIVE,
                ))

        except sqlite3.OperationalError as exc:
            errors.append(f"knowledgeC query failed: {exc}")
        finally:
            conn.close()

        return IngestResult(
            source="knowledgec",
            events=events,
            errors=errors,
            duration_seconds=time.time() - t0,
        )
