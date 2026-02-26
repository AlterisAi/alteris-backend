"""iMessage adapter — reads from macOS Messages chat.db.

Handles:
- attributedBody decoding (typedstream format) for messages where text is NULL
- Spam filtering (is_spam column)
- Group chat participant resolution
- Service type tracking (iMessage, RCS, SMS)

Timestamps are stored as nanoseconds since Apple epoch (2001-01-01).
"""

from __future__ import annotations

import logging
import sqlite3
import time

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
    EVENT_TYPE_MESSAGE,
    IMESSAGE_DB,
)
from loom.models import Event
from loom.privacy import SensitivityLevel

logger = logging.getLogger(__name__)

# iMessage timestamps are nanoseconds since Apple epoch (2001-01-01)
_NS_PER_SECOND = 1_000_000_000

try:
    from typedstream.stream import TypedStreamReader
    _HAS_TYPEDSTREAM = True
except ImportError:
    _HAS_TYPEDSTREAM = False


def _decode_attributed_body(blob: bytes) -> str:
    """Extract plain text from attributedBody typedstream blob."""
    if not blob or not _HAS_TYPEDSTREAM:
        return ""
    try:
        for event in TypedStreamReader.from_data(blob):
            if isinstance(event, bytes):
                return event.decode("utf-8", errors="replace")
    except Exception:
        pass
    return ""


class IMessageAdapter(SourceAdapter):
    """Reads iMessage chat.db database."""

    @property
    def source_name(self) -> str:
        return "imessage"

    def check_availability(self) -> AvailabilityResult:
        return check_sqlite_readable(IMESSAGE_DB, "imessage")

    def check_schema(self) -> SchemaResult:
        if not IMESSAGE_DB.exists():
            return SchemaResult(compatible=False, source="imessage", warnings=["Database not found"])
        return check_sqlite_tables(
            IMESSAGE_DB, "imessage",
            required=["message", "handle", "chat", "chat_message_join"],
            optional=["chat_handle_join"],
        )

    def ingest(self, since_ts: int = 0, limit: int = 0) -> IngestResult:
        t0 = time.time()

        if not IMESSAGE_DB.exists():
            return IngestResult(source="imessage", errors=["iMessage database not found"])

        try:
            conn = sqlite3.connect(f"file:{IMESSAGE_DB}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.OperationalError:
            return IngestResult(
                source="imessage",
                errors=["Cannot read iMessage database. Grant Full Disk Access in System Settings."],
            )

        # Convert since_ts to nanoseconds since Apple epoch
        if since_ts > 0:
            cutoff_ns = int((since_ts - APPLE_EPOCH_OFFSET) * _NS_PER_SECOND)
        else:
            cutoff_ns = 0

        fetch_limit = limit if limit > 0 else 500_000

        query = """
            SELECT
                m.ROWID,
                m.text,
                m.attributedBody,
                m.date AS date_ns,
                m.is_from_me,
                m.is_spam,
                m.cache_roomnames,
                m.service,
                m.associated_message_type,
                m.associated_message_guid,
                m.associated_message_emoji,
                m.is_auto_reply,
                m.was_delivered_quietly,
                h.id AS handle_id,
                c.ROWID AS chat_rowid,
                c.display_name AS chat_display_name,
                c.is_filtered AS chat_is_filtered
            FROM message m
            LEFT JOIN handle h ON m.handle_id = h.ROWID
            LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
            LEFT JOIN chat c ON c.ROWID = cmj.chat_id
            WHERE m.date > ?
              AND m.is_spam = 0
              AND m.is_system_message = 0
              AND m.is_service_message = 0
            ORDER BY m.date DESC
            LIMIT ?
        """

        # Cache chat participants
        participant_cache: dict[int, list[str]] = {}

        def _get_participants(chat_rowid: int) -> list[str]:
            if chat_rowid in participant_cache:
                return participant_cache[chat_rowid]
            try:
                parts = [
                    r["id"] for r in conn.execute(
                        "SELECT h.id FROM chat_handle_join chj "
                        "JOIN handle h ON h.ROWID = chj.handle_id "
                        "WHERE chj.chat_id = ?",
                        (chat_rowid,),
                    )
                ]
            except sqlite3.OperationalError:
                parts = []
            participant_cache[chat_rowid] = parts
            return parts

        # Tapback type labels for metadata
        _TAPBACK_TYPES = {
            2000: "liked", 2001: "loved", 2002: "disliked",
            2003: "laughed", 2004: "emphasized", 2005: "questioned",
            2006: "removed_reaction",
        }

        events: list[Event] = []
        errors: list[str] = []

        try:
            for row in conn.execute(query, (cutoff_ns, fetch_limit)):
                date_ns = row_val(row, "date_ns", 0)
                if date_ns:
                    ts = int(date_ns / _NS_PER_SECOND + APPLE_EPOCH_OFFSET)
                else:
                    ts = int(time.time())

                handle = str(row_val(row, "handle_id", "unknown"))
                is_from_me = bool(row_val(row, "is_from_me", 0))
                service = str(row_val(row, "service", ""))
                chat_rowid = row_val(row, "chat_rowid", None)
                chat_name = str(row_val(row, "chat_display_name", ""))
                is_auto_reply = bool(row_val(row, "is_auto_reply", 0))
                delivered_quietly = bool(row_val(row, "was_delivered_quietly", 0))
                chat_is_filtered = bool(row_val(row, "chat_is_filtered", 0))
                assoc_type = row_val(row, "associated_message_type", 0) or 0

                # Tapbacks/reactions: store as lightweight events, not full content
                if assoc_type != 0:
                    tapback_label = _TAPBACK_TYPES.get(assoc_type, f"type_{assoc_type}")
                    # "removed_reaction" is noise — skip entirely
                    if tapback_label == "removed_reaction":
                        continue
                    assoc_guid = str(row_val(row, "associated_message_guid", ""))
                    assoc_emoji = str(row_val(row, "associated_message_emoji", ""))
                    source_id = str(row["ROWID"])
                    event_id = Event.make_id("imessage", source_id)
                    # Minimal participants for the reaction
                    react_participants = [handle] if not is_from_me else []

                    events.append(Event(
                        id=event_id,
                        source="imessage",
                        source_id=source_id,
                        event_type="reaction",
                        timestamp=ts,
                        participants=tuple(react_participants),
                        raw_content=None,
                        content_hash="",
                        metadata={
                            "subject": "",
                            "is_from_me": is_from_me,
                            "thread_id": "",
                            "service": service,
                            "reaction_type": tapback_label,
                            "reaction_emoji": assoc_emoji,
                            "parent_message_guid": assoc_guid,
                            "is_reaction": True,
                        },
                        sensitivity=SensitivityLevel.SENSITIVE,
                    ))
                    continue

                # Get body: text column first, then attributedBody
                text = str(row_val(row, "text", ""))
                if not text:
                    text = _decode_attributed_body(row["attributedBody"])
                if not text:
                    continue  # skip attachment-only messages

                # Build participants list
                participants = []
                if is_from_me:
                    if chat_rowid:
                        chat_parts = _get_participants(chat_rowid)
                        participants.extend(chat_parts)
                    elif handle != "unknown":
                        participants.append(handle)
                else:
                    participants.append(handle)
                    if chat_rowid:
                        chat_parts = _get_participants(chat_rowid)
                        for p in chat_parts:
                            if p != handle:
                                participants.append(p)

                # Thread ID: use native cache_roomnames for group chats,
                # synthetic "imessage:<handle>" for 1:1 conversations.
                # For outbound messages, handle may be missing — fall back
                # to first participant (the recipient).
                native_thread = str(row_val(row, "cache_roomnames", "")) or ""
                if native_thread:
                    thread_id = native_thread
                elif handle and handle != "unknown":
                    thread_id = f"imessage:{handle}"
                elif participants:
                    thread_id = f"imessage:{participants[0]}"
                else:
                    thread_id = ""

                source_id = str(row["ROWID"])
                event_id = Event.make_id("imessage", source_id)
                content_hash = Event.content_hash_of(text)

                events.append(Event(
                    id=event_id,
                    source="imessage",
                    source_id=source_id,
                    event_type=EVENT_TYPE_MESSAGE,
                    timestamp=ts,
                    participants=tuple(participants),
                    raw_content=text,
                    content_hash=content_hash,
                    metadata={
                        "subject": "",
                        "is_from_me": is_from_me,
                        "thread_id": thread_id,
                        "service": service,
                        "chat_display_name": chat_name,
                        "is_auto_reply": is_auto_reply,
                        "delivered_quietly": delivered_quietly,
                        "is_filtered": chat_is_filtered,
                    },
                    sensitivity=SensitivityLevel.SENSITIVE,
                ))

        except sqlite3.OperationalError as exc:
            errors.append(f"iMessage query failed: {exc}")
        finally:
            conn.close()

        return IngestResult(
            source="imessage",
            events=events,
            errors=errors,
            duration_seconds=time.time() - t0,
        )
