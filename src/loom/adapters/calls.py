"""Call history adapters — macOS FaceTime/Phone + WhatsApp calls.

Reads system-recorded call logs as immutable facts (confidence 1.0).
No text content — calls carry duration, direction, and participant only.

macOS: ~/Library/Application Support/CallHistoryDB/CallHistory.storedata
WhatsApp: ~/Library/Group Containers/group.net.whatsapp.WhatsApp.shared/CallHistory.sqlite

Requires Full Disk Access in System Settings.
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
    make_source_id_hash,
)
from loom.constants import (
    APPLE_EPOCH_OFFSET,
    CALL_MIN_DURATION_SECONDS,
    EVENT_TYPE_CALL,
    MACOS_CALL_HISTORY_DB,
    WHATSAPP_CALL_DB,
    WHATSAPP_LID_DB,
)
from loom.models import Event
from loom.privacy import SensitivityLevel
from loom.resolver import normalize_phone

logger = logging.getLogger(__name__)

# macOS call type codes
_CALL_TYPE_LABELS = {
    1: "voice",
    8: "facetime_video",
    16: "facetime_audio",
}


def _duration_label(seconds: float) -> str:
    """Human-readable duration like '2m 30s' or '45s'."""
    if seconds < 60:
        return f"{int(seconds)}s"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}m {secs}s" if secs else f"{mins}m"


class MacOSCallAdapter(SourceAdapter):
    """Reads macOS call history (FaceTime + Phone)."""

    @property
    def source_name(self) -> str:
        return "calls_macos"

    def check_availability(self) -> AvailabilityResult:
        return check_sqlite_readable(MACOS_CALL_HISTORY_DB, self.source_name)

    def check_schema(self) -> SchemaResult:
        return check_sqlite_tables(
            MACOS_CALL_HISTORY_DB, self.source_name,
            required=["ZCALLRECORD"],
        )

    def ingest(self, since_ts: int = 0, limit: int = 0) -> IngestResult:
        t0 = time.time()
        avail = self.check_availability()
        if not avail.available:
            return IngestResult(
                source=self.source_name,
                errors=[avail.user_action or avail.reason],
            )

        conn = sqlite3.connect(
            f"file:{MACOS_CALL_HISTORY_DB}?mode=ro", uri=True,
        )
        conn.row_factory = sqlite3.Row

        conditions = ["ZDURATION >= ?"]
        params: list = [CALL_MIN_DURATION_SECONDS]

        if since_ts:
            apple_since = since_ts - APPLE_EPOCH_OFFSET
            conditions.append("ZDATE >= ?")
            params.append(apple_since)

        where = " AND ".join(conditions)
        order = "ORDER BY ZDATE DESC"
        limit_clause = f"LIMIT {limit}" if limit else ""

        rows = conn.execute(
            f"SELECT Z_PK, ZDATE, ZDURATION, ZADDRESS, ZORIGINATED, "  # noqa: S608
            f"ZANSWERED, ZCALLTYPE FROM ZCALLRECORD "
            f"WHERE {where} {order} {limit_clause}",
            params,
        ).fetchall()

        events: list[Event] = []
        errors: list[str] = []

        for row in rows:
            try:
                pk = str(row["Z_PK"])
                ts_apple = row["ZDATE"] or 0
                timestamp = int(ts_apple + APPLE_EPOCH_OFFSET)
                duration = float(row["ZDURATION"] or 0)
                address = (row["ZADDRESS"] or "").strip()
                originated = bool(row["ZORIGINATED"])
                answered = bool(row["ZANSWERED"])
                call_type_code = row["ZCALLTYPE"] or 1
                call_type = _CALL_TYPE_LABELS.get(call_type_code, "voice")

                if not address:
                    continue

                phone = normalize_phone(address)
                if not phone:
                    phone = address

                direction = "Outgoing" if originated else "Incoming"
                source_id = make_source_id_hash("macos_call", pk)
                event_id = Event.make_id(self.source_name, source_id)

                events.append(Event(
                    id=event_id,
                    source=self.source_name,
                    source_id=source_id,
                    event_type=EVENT_TYPE_CALL,
                    timestamp=timestamp,
                    participants=(phone,),
                    raw_content="",
                    metadata={
                        "subject": f"{direction} call ({_duration_label(duration)})",
                        "duration_seconds": round(duration, 1),
                        "originated": originated,
                        "answered": answered,
                        "call_type": call_type,
                    },
                    sensitivity=SensitivityLevel.SENSITIVE,
                ))
            except Exception as e:
                errors.append(f"Row {row['Z_PK']}: {e}")

        conn.close()
        return IngestResult(
            source=self.source_name,
            events=events,
            errors=errors,
            duration_seconds=time.time() - t0,
        )


class WhatsAppCallAdapter(SourceAdapter):
    """Reads WhatsApp call history."""

    @property
    def source_name(self) -> str:
        return "calls_whatsapp"

    def check_availability(self) -> AvailabilityResult:
        return check_sqlite_readable(WHATSAPP_CALL_DB, self.source_name)

    def check_schema(self) -> SchemaResult:
        return check_sqlite_tables(
            WHATSAPP_CALL_DB, self.source_name,
            required=["ZWAAGGREGATECALLEVENT"],
            optional=["ZWACDCALLEVENT", "ZWACDCALLEVENTPARTICIPANT"],
        )

    def _build_lid_map(self) -> dict[str, str]:
        """Build LID -> phone number mapping from LID.sqlite."""
        lid_map: dict[str, str] = {}
        if not WHATSAPP_LID_DB.exists():
            return lid_map
        try:
            conn = sqlite3.connect(
                f"file:{WHATSAPP_LID_DB}?mode=ro", uri=True,
            )
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ZIDENTIFIER, ZPHONENUMBER FROM ZWAZACCOUNT "
                "WHERE ZIDENTIFIER IS NOT NULL AND ZPHONENUMBER IS NOT NULL"
            ).fetchall()
            for r in rows:
                lid_raw = r["ZIDENTIFIER"] or ""
                phone_raw = (r["ZPHONENUMBER"] or "").strip()
                if lid_raw and phone_raw:
                    lid_map[lid_raw] = phone_raw
                    # Also store with @lid suffix
                    if "@lid" not in lid_raw:
                        lid_map[f"{lid_raw}@lid"] = phone_raw
            conn.close()
        except sqlite3.OperationalError:
            pass
        return lid_map

    def _resolve_jid(self, jid: str, lid_map: dict[str, str]) -> str:
        """Resolve a WhatsApp JID to a phone number."""
        if not jid:
            return ""
        # Standard JID: phone@s.whatsapp.net
        if "@s.whatsapp.net" in jid:
            return jid.replace("@s.whatsapp.net", "")
        # LID: numeric@lid or just numeric
        phone = lid_map.get(jid)
        if phone:
            return phone
        # Try stripping @lid
        bare = jid.replace("@lid", "")
        phone = lid_map.get(bare)
        if phone:
            return phone
        return bare

    def ingest(self, since_ts: int = 0, limit: int = 0) -> IngestResult:
        t0 = time.time()
        avail = self.check_availability()
        if not avail.available:
            return IngestResult(
                source=self.source_name,
                errors=[avail.user_action or avail.reason],
            )

        conn = sqlite3.connect(
            f"file:{WHATSAPP_CALL_DB}?mode=ro", uri=True,
        )
        conn.row_factory = sqlite3.Row

        lid_map = self._build_lid_map()

        # Join aggregate -> cd call events -> participants
        # Schema: agg -< cd (via cd.Z1CALLEVENTS = agg.Z_PK)
        #          cd -< participant (via p.Z1PARTICIPANTS = cd.Z_PK)
        conditions = []
        params: list = []

        if since_ts:
            apple_since = since_ts - APPLE_EPOCH_OFFSET
            conditions.append("agg.ZFIRSTDATE >= ?")
            params.append(apple_since)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
            SELECT agg.Z_PK AS agg_pk,
                   agg.ZFIRSTDATE,
                   agg.ZINCOMING,
                   agg.ZVIDEO,
                   cd.ZDURATION,
                   p.ZJIDSTRING
            FROM ZWAAGGREGATECALLEVENT agg
            LEFT JOIN ZWACDCALLEVENT cd ON cd.Z1CALLEVENTS = agg.Z_PK
            LEFT JOIN ZWACDCALLEVENTPARTICIPANT p ON p.Z1PARTICIPANTS = cd.Z_PK
            {where}
            ORDER BY agg.ZFIRSTDATE DESC
            {limit_clause}
        """  # noqa: S608

        try:
            rows = conn.execute(query, params).fetchall()
        except sqlite3.OperationalError:
            rows = conn.execute(
                f"SELECT Z_PK AS agg_pk, ZFIRSTDATE, ZINCOMING, ZVIDEO, "  # noqa: S608
                f"0 AS ZDURATION, '' AS ZJIDSTRING "
                f"FROM ZWAAGGREGATECALLEVENT {where} "
                f"ORDER BY ZFIRSTDATE DESC {limit_clause}",
                params,
            ).fetchall()

        # Group by aggregate event (one call can have multiple participants)
        agg_events: dict[int, dict] = {}
        for row in rows:
            pk = row["agg_pk"]
            if pk not in agg_events:
                ts_apple = row["ZFIRSTDATE"] or 0
                agg_events[pk] = {
                    "timestamp": int(ts_apple + APPLE_EPOCH_OFFSET),
                    "duration": float(row["ZDURATION"] or 0),
                    "outgoing": not bool(row["ZINCOMING"]),
                    "participants": [],
                }
            jid = row["ZJIDSTRING"] or ""
            if jid:
                phone = self._resolve_jid(jid, lid_map)
                if phone:
                    normalized = normalize_phone(phone)
                    agg_events[pk]["participants"].append(normalized or phone)

        events: list[Event] = []
        errors: list[str] = []

        for pk, data in agg_events.items():
            if data["duration"] < CALL_MIN_DURATION_SECONDS:
                continue
            if not data["participants"]:
                continue

            # Deduplicate participants
            participants = list(dict.fromkeys(data["participants"]))

            direction = "Outgoing" if data["outgoing"] else "Incoming"
            source_id = make_source_id_hash("whatsapp_call", str(pk))
            event_id = Event.make_id(self.source_name, source_id)

            events.append(Event(
                id=event_id,
                source=self.source_name,
                source_id=source_id,
                event_type=EVENT_TYPE_CALL,
                timestamp=data["timestamp"],
                participants=tuple(participants),
                raw_content="",
                metadata={
                    "subject": f"{direction} WhatsApp call ({_duration_label(data['duration'])})",
                    "duration_seconds": round(data["duration"], 1),
                    "originated": data["outgoing"],
                    "answered": True,
                    "call_type": "whatsapp",
                },
                sensitivity=SensitivityLevel.SENSITIVE,
            ))

        conn.close()
        return IngestResult(
            source=self.source_name,
            events=events,
            errors=errors,
            duration_seconds=time.time() - t0,
        )
