"""Apple Notes adapter — reads notes from macOS Notes.app database.

Notes are stored in NoteStore.sqlite. The content is in ZICCLOUDSYNCINGOBJECT
joined with ZICNOTEDATA. Note content is gzip-compressed protobuf/plist data
in ZICNOTEDATA.ZDATA — we do best-effort text extraction.

Sensitivity is CRITICAL — notes may contain deeply personal content and must
be routed through local model only.
"""

from __future__ import annotations

import gzip
import logging
import sqlite3
import time

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
    EVENT_TYPE_NOTE,
    NOTES_CONTENT_MAX_LEN,
    NOTES_DB,
)
from alteris.models import Event
from alteris.privacy import SensitivityLevel

logger = logging.getLogger(__name__)


def _extract_note_text(data: bytes | None) -> str:
    """Best-effort text extraction from Notes gzip-compressed data.

    The decompressed data is protobuf/plist binary. We scan for runs of
    printable ASCII (plus tab/newline/CR) of at least 4 characters and
    concatenate them. This captures the user-visible text without needing
    a full protobuf parser.
    """
    if not data:
        return ""
    try:
        decompressed = gzip.decompress(data)
    except (gzip.BadGzipFile, OSError):
        return ""

    text_parts: list[str] = []
    current: list[int] = []
    for byte in decompressed:
        if 32 <= byte < 127 or byte in (9, 10, 13):
            current.append(byte)
        else:
            if len(current) >= 4:
                try:
                    text_parts.append(bytes(current).decode("utf-8", errors="replace"))
                except Exception:
                    pass
            current = []
    if len(current) >= 4:
        try:
            text_parts.append(bytes(current).decode("utf-8", errors="replace"))
        except Exception:
            pass

    return "\n".join(text_parts)[:NOTES_CONTENT_MAX_LEN]


class NotesAdapter(SourceAdapter):
    """Reads Apple Notes from NoteStore.sqlite."""

    @property
    def source_name(self) -> str:
        return "notes"

    def check_availability(self) -> AvailabilityResult:
        return check_sqlite_readable(NOTES_DB, "notes")

    def check_schema(self) -> SchemaResult:
        if not NOTES_DB.exists():
            return SchemaResult(compatible=False, source="notes", warnings=["Database not found"])
        return check_sqlite_tables(
            NOTES_DB, "notes",
            required=["ZICCLOUDSYNCINGOBJECT", "ZICNOTEDATA"],
        )

    def ingest(self, since_ts: int = 0, limit: int = 0) -> IngestResult:
        t0 = time.time()

        if not NOTES_DB.exists():
            return IngestResult(source="notes", errors=["Apple Notes database not found"])

        try:
            conn = sqlite3.connect(f"file:{NOTES_DB}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.OperationalError:
            return IngestResult(
                source="notes",
                errors=["Cannot read Notes database. Grant Full Disk Access in System Settings."],
            )

        fetch_limit = limit if limit > 0 else 500_000

        # Notes timestamps are seconds since Apple epoch (2001-01-01).
        # Filter: ZMODIFICATIONDATE1 > since_ts - APPLE_EPOCH_OFFSET
        query = """
            SELECT
                z.Z_PK,
                z.ZIDENTIFIER,
                z.ZTITLE1 AS title,
                z.ZMODIFICATIONDATE1,
                z.ZCREATIONDATE3,
                z.ZISPASSWORDPROTECTED,
                z.ZFOLDER,
                nd.ZDATA AS note_data,
                folder.ZTITLE2 AS folder_name
            FROM ZICCLOUDSYNCINGOBJECT z
            LEFT JOIN ZICNOTEDATA nd ON z.ZNOTEDATA = nd.Z_PK
            LEFT JOIN ZICCLOUDSYNCINGOBJECT folder ON z.ZFOLDER = folder.Z_PK
            WHERE z.ZTITLE1 IS NOT NULL
              AND z.ZMODIFICATIONDATE1 IS NOT NULL
              AND z.ZMODIFICATIONDATE1 + ? > ?
            ORDER BY z.ZMODIFICATIONDATE1 DESC
            LIMIT ?
        """

        events: list[Event] = []
        errors: list[str] = []

        try:
            for row in conn.execute(query, (APPLE_EPOCH_OFFSET, since_ts, fetch_limit)):
                is_password_protected = bool(row_val(row, "ZISPASSWORDPROTECTED", 0))
                if is_password_protected:
                    continue

                identifier = str(row_val(row, "ZIDENTIFIER", ""))
                if not identifier:
                    continue

                mod_date = row_val(row, "ZMODIFICATIONDATE1", 0)
                ts = int(mod_date) + APPLE_EPOCH_OFFSET

                title = str(row_val(row, "title", ""))
                folder_name = str(row_val(row, "folder_name", ""))
                note_data = row["note_data"]

                raw_content = _extract_note_text(note_data)
                if not raw_content:
                    raw_content = title

                content_hash = Event.content_hash_of(raw_content) if raw_content else ""

                event_id = Event.make_id("notes", identifier)

                events.append(Event(
                    id=event_id,
                    source="notes",
                    source_id=identifier,
                    event_type=EVENT_TYPE_NOTE,
                    timestamp=ts,
                    participants=(),
                    raw_content=raw_content,
                    content_hash=content_hash,
                    metadata={
                        "subject": title,
                        "is_from_me": True,
                        "thread_id": "",
                        "folder_name": folder_name or "",
                        "is_password_protected": is_password_protected,
                        "note_identifier": identifier,
                    },
                    sensitivity=SensitivityLevel.CRITICAL,
                ))

        except sqlite3.OperationalError as exc:
            errors.append(f"Notes query failed: {exc}")
        finally:
            conn.close()

        return IngestResult(
            source="notes",
            events=events,
            errors=errors,
            duration_seconds=time.time() - t0,
        )
