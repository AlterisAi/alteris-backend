"""WhatsApp adapter — reads from macOS WhatsApp local SQLite database.

Path: ~/Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite

Handles:
- Core Data epoch timestamps (seconds since 2001-01-01)
- LID-to-phone identity resolution via multiple cross-referencing strategies
- Group chat member resolution
- User phone auto-detection

Requires Full Disk Access in System Settings.
"""

from __future__ import annotations

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
    EVENT_TYPE_MESSAGE,
    EVENT_TYPE_REACTION,
    WHATSAPP_CHAT_DB,
    WHATSAPP_LID_DB,
)
from alteris.models import Event
from alteris.privacy import SensitivityLevel

logger = logging.getLogger(__name__)


def _normalize_jid(jid: str) -> str:
    """Extract phone number or group ID from WhatsApp JID."""
    if not jid:
        return ""
    return (
        jid.replace("@s.whatsapp.net", "")
        .replace("@g.us", "")
        .replace("@lid", "")
    )


def _build_identity_map(conn: sqlite3.Connection) -> dict[str, dict]:
    """Build a mapping from any JID/LID to {phone, name, jid}.

    Cross-references:
    1. LID.sqlite ZWAZACCOUNT for LID->phone
    2. ZWACHATSESSION for phone-based contacts
    3. ZWAPROFILEPUSHNAME for push names
    4. ZWAGROUPMEMBER for group members
    5. ZWAMESSAGE for LID->phone correlation via chat sessions
    """
    identity: dict[str, dict] = {}

    # Source 0: LID.sqlite for LID->phone resolution
    try:
        lid_conn = sqlite3.connect(f"file:{WHATSAPP_LID_DB}?mode=ro", uri=True)
        lid_conn.row_factory = sqlite3.Row
        accts = lid_conn.execute("""
            SELECT ZIDENTIFIER, ZPHONENUMBER, ZDISPLAYNAME
            FROM ZWAZACCOUNT WHERE ZIDENTIFIER IS NOT NULL
        """).fetchall()
        for a in accts:
            lid_raw = a["ZIDENTIFIER"] or ""
            phone = (a["ZPHONENUMBER"] or "").lstrip("+").replace(" ", "").replace("-", "")
            name = a["ZDISPLAYNAME"] or ""
            if not lid_raw:
                continue
            lid = lid_raw if "@lid" in lid_raw else f"{lid_raw}@lid"
            if phone or name:
                jid = f"{phone}@s.whatsapp.net" if phone else ""
                entry = {"phone": phone, "name": name, "jid": jid}
                identity[lid] = entry
                bare = _normalize_jid(lid)
                if bare:
                    identity[bare] = entry
        lid_conn.close()
    except (sqlite3.OperationalError, FileNotFoundError):
        pass

    # Source 1: Chat sessions for phone-based contacts
    try:
        rows = conn.execute("""
            SELECT
                S.ZCONTACTJID,
                COALESCE(S.ZPARTNERNAME, '') AS partner_name,
                COALESCE(P.ZPUSHNAME, '') AS push_name
            FROM ZWACHATSESSION S
            LEFT JOIN ZWAPROFILEPUSHNAME P ON S.ZCONTACTJID = P.ZJID
            WHERE S.ZCONTACTJID LIKE '%@s.whatsapp.net'
        """).fetchall()
        for row in rows:
            jid = str(row_val(row, "ZCONTACTJID", ""))
            phone = _normalize_jid(jid)
            partner = str(row_val(row, "partner_name", ""))
            push = str(row_val(row, "push_name", ""))
            name = partner or push
            entry = {"phone": phone, "name": name, "jid": jid,
                     "partner_name": partner, "push_name": push}
            identity[jid] = entry
            identity[phone] = entry
    except sqlite3.OperationalError:
        pass

    # Source 2: Push names — fill in names for entries created by Source 0/1
    # with empty names, and create new entries for unknown JIDs.
    # Push names are secondary to contact names (partner_name) in authority.
    try:
        pn_rows = conn.execute(
            "SELECT ZJID, ZPUSHNAME FROM ZWAPROFILEPUSHNAME"
        ).fetchall()
        for row in pn_rows:
            jid = str(row_val(row, "ZJID", ""))
            name = str(row_val(row, "ZPUSHNAME", ""))
            if not jid or not name:
                continue
            if jid in identity:
                # Update name if existing entry has no name
                if not identity[jid].get("name"):
                    identity[jid]["name"] = name
                # Store push_name for alias tracking
                identity[jid]["push_name"] = name
                # Also update bare key
                bare = _normalize_jid(jid)
                if bare and bare in identity and not identity[bare].get("name"):
                    identity[bare]["name"] = name
                if bare and bare in identity:
                    identity[bare]["push_name"] = name
            else:
                phone = _normalize_jid(jid)
                entry = {"phone": phone, "name": name, "jid": jid, "push_name": name}
                identity[jid] = entry
                if phone:
                    identity[phone] = entry
    except sqlite3.OperationalError:
        pass

    # Source 3: Group members
    try:
        gm_rows = conn.execute("""
            SELECT ZMEMBERJID, ZCONTACTNAME, ZFIRSTNAME
            FROM ZWAGROUPMEMBER
        """).fetchall()
        for row in gm_rows:
            member_jid = str(row_val(row, "ZMEMBERJID", ""))
            name = str(row_val(row, "ZCONTACTNAME", "")) or str(row_val(row, "ZFIRSTNAME", ""))
            if not member_jid:
                continue
            bare = _normalize_jid(member_jid)
            if member_jid in identity:
                if name and not identity[member_jid].get("name"):
                    identity[member_jid]["name"] = name
            else:
                entry = {"phone": bare, "name": name, "jid": member_jid}
                identity[member_jid] = entry
                if bare:
                    identity[bare] = entry
    except sqlite3.OperationalError:
        pass

    # Source 4: Correlate LIDs to phones via message/chat session pairs
    try:
        lid_rows = conn.execute("""
            SELECT DISTINCT M.ZFROMJID, S.ZCONTACTJID
            FROM ZWAMESSAGE M
            JOIN ZWACHATSESSION S ON M.ZCHATSESSION = S.Z_PK
            WHERE M.ZFROMJID LIKE '%@lid'
              AND S.ZCONTACTJID LIKE '%@s.whatsapp.net'
              AND M.ZISFROMME = 0
            LIMIT 500
        """).fetchall()
        for row in lid_rows:
            lid = row[0]
            phone_jid = row[1]
            if lid and phone_jid and phone_jid in identity:
                phone_entry = identity[phone_jid]
                if lid not in identity or not identity[lid].get("phone"):
                    identity[lid] = phone_entry
                    bare_lid = _normalize_jid(lid)
                    if bare_lid:
                        identity[bare_lid] = phone_entry
    except sqlite3.OperationalError:
        pass

    # Cross-reference: LID entries inherit name from phone-based identity
    for key, entry in list(identity.items()):
        if "@lid" not in key:
            continue
        phone = entry.get("phone", "")
        if not phone:
            continue
        phone_jid = f"{phone}@s.whatsapp.net"
        phone_entry = identity.get(phone_jid) or identity.get(phone)
        if phone_entry and phone_entry.get("name"):
            entry["name"] = phone_entry["name"]
            entry["jid"] = phone_entry.get("jid", entry.get("jid", ""))

    return identity


def _detect_user_phone(conn: sqlite3.Connection) -> str:
    """Detect the user's own phone number by group membership frequency."""
    try:
        rows = conn.execute("""
            SELECT ZMEMBERJID, COUNT(DISTINCT ZCHATSESSION) AS group_count
            FROM ZWAGROUPMEMBER
            WHERE ZMEMBERJID IS NOT NULL
            GROUP BY ZMEMBERJID
            ORDER BY group_count DESC
            LIMIT 5
        """).fetchall()
        if rows:
            top_jid = rows[0][0]
            top_count = rows[0][1]
            if top_count >= 3:
                return _normalize_jid(top_jid)
    except sqlite3.OperationalError:
        pass
    return ""


def _try_decode_reaction_emoji(blob: bytes | None) -> str | None:
    """Best-effort extraction of reaction emoji from ZWAMEDIAITEM.ZMETADATA.

    WhatsApp stores the emoji in a binary blob (likely protobuf).
    We scan for valid UTF-8 emoji codepoints in the common ranges.
    Returns the emoji string if found, None otherwise.
    """
    if not blob or not isinstance(blob, (bytes, bytearray)):
        return None
    # Scan the blob for UTF-8 emoji sequences.
    # Most emoji are in U+1F600..U+1FAFF (4-byte UTF-8: 0xF0 0x9F ..)
    # or basic emoji like hearts U+2764 (3-byte UTF-8: 0xE2 ..)
    i = 0
    while i < len(blob):
        b = blob[i]
        # 4-byte UTF-8 sequence starting with 0xF0 (emoji plane)
        if b == 0xF0 and i + 3 < len(blob):
            try:
                candidate = bytes(blob[i:i + 4]).decode("utf-8")
                cp = ord(candidate)
                if 0x1F300 <= cp <= 0x1FAFF or 0x1F900 <= cp <= 0x1F9FF:
                    return candidate
            except (UnicodeDecodeError, ValueError):
                pass
            i += 1
            continue
        # 3-byte UTF-8 (some emoji like hearts, arrows)
        if (b & 0xF0) == 0xE0 and i + 2 < len(blob):
            try:
                candidate = bytes(blob[i:i + 3]).decode("utf-8")
                cp = ord(candidate)
                if 0x2600 <= cp <= 0x27BF or 0x2700 <= cp <= 0x27BF:
                    return candidate
            except (UnicodeDecodeError, ValueError):
                pass
            i += 1
            continue
        i += 1
    return None


def _format_sender(name: str, phone: str) -> str:
    """Format sender/recipient as 'Name <phone>'."""
    if name and phone and name != "me":
        return f"{name} <{phone}>"
    if name == "me" and phone:
        return f"USER <{phone}>"
    return phone or name or "unknown"


class WhatsAppAdapter(SourceAdapter):
    """Reads WhatsApp ChatStorage.sqlite database."""

    @property
    def source_name(self) -> str:
        return "whatsapp"

    def check_availability(self) -> AvailabilityResult:
        return check_sqlite_readable(WHATSAPP_CHAT_DB, "whatsapp")

    def check_schema(self) -> SchemaResult:
        if not WHATSAPP_CHAT_DB.exists():
            return SchemaResult(compatible=False, source="whatsapp", warnings=["Database not found"])
        return check_sqlite_tables(
            WHATSAPP_CHAT_DB, "whatsapp",
            required=["ZWAMESSAGE", "ZWACHATSESSION"],
            optional=["ZWAGROUPMEMBER", "ZWAPROFILEPUSHNAME"],
        )

    def ingest(self, since_ts: int = 0, limit: int = 0) -> IngestResult:
        t0 = time.time()

        if not WHATSAPP_CHAT_DB.exists():
            return IngestResult(source="whatsapp", errors=["WhatsApp database not found"])

        try:
            conn = sqlite3.connect(f"file:{WHATSAPP_CHAT_DB}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.OperationalError:
            return IngestResult(
                source="whatsapp",
                errors=["Cannot read WhatsApp database. Grant Full Disk Access in System Settings."],
            )

        # Convert to Core Data epoch
        cutoff_cd = (since_ts - APPLE_EPOCH_OFFSET) if since_ts > 0 else 0

        id_map = _build_identity_map(conn)
        user_phone = _detect_user_phone(conn)

        # Build blocked contacts set for filtering
        blocked_jids: set[str] = set()
        try:
            for brow in conn.execute("SELECT ZJID FROM ZWABLACKLISTITEM WHERE ZJID IS NOT NULL"):
                blocked_jids.add(str(brow[0]))
        except sqlite3.OperationalError:
            pass

        fetch_limit = limit if limit > 0 else 500_000

        # System/noise message types to exclude:
        # 10=missed calls, 11=encryption notifications, 14=deleted/revoked,
        # 15=live location, 23=stickers
        # Note: type 59 (reactions) are now handled as reaction events
        query = """
            SELECT
                M.Z_PK,
                M.ZTEXT,
                M.ZMESSAGEDATE,
                M.ZISFROMME,
                M.ZMESSAGETYPE,
                M.ZFROMJID,
                M.ZTOJID,
                M.ZCHATSESSION,
                M.ZGROUPMEMBER,
                M.ZSTANZAID,
                S.ZCONTACTJID,
                S.ZPARTNERNAME,
                S.ZSESSIONTYPE,
                GM.ZMEMBERJID AS group_sender_jid,
                COALESCE(GM.ZCONTACTNAME, GM.ZFIRSTNAME, '') AS group_sender_name,
                MI.ZMETADATA AS media_metadata
            FROM ZWAMESSAGE M
            LEFT JOIN ZWACHATSESSION S ON M.ZCHATSESSION = S.Z_PK
            LEFT JOIN ZWAGROUPMEMBER GM ON M.ZGROUPMEMBER = GM.Z_PK
            LEFT JOIN ZWAMEDIAITEM MI ON M.ZMEDIAITEM = MI.Z_PK
            WHERE M.ZMESSAGEDATE > ?
              AND M.ZMESSAGETYPE NOT IN (10, 11, 14, 15, 23)
              AND (M.ZMESSAGETYPE = 59 OR (M.ZTEXT IS NOT NULL AND LENGTH(M.ZTEXT) > 0))
            ORDER BY M.ZMESSAGEDATE DESC
            LIMIT ?
        """

        # Pre-cache group members by session PK
        group_member_cache: dict[int, list[str]] = {}

        def _get_group_members(session_pk: int) -> list[str]:
            """Return formatted participant list for a group session.

            Resolves member names via (in priority order):
            1. ZWAGROUPMEMBER.ZCONTACTNAME (WhatsApp contact name)
            2. Identity map (partner_name from chat sessions)
            3. ZWAPROFILEPUSHNAME (push name, prefixed with ~ in WhatsApp UI)
            """
            if session_pk in group_member_cache:
                return group_member_cache[session_pk]
            members: list[str] = []
            seen: set[str] = set()
            user_jid = f"{user_phone}@s.whatsapp.net" if user_phone else ""
            try:
                # Join with push names directly to catch LID-based members
                member_rows = conn.execute("""
                    SELECT GM.ZMEMBERJID,
                           COALESCE(GM.ZCONTACTNAME, GM.ZFIRSTNAME, '') AS member_name,
                           COALESCE(PN.ZPUSHNAME, '') AS push_name
                    FROM ZWAGROUPMEMBER GM
                    LEFT JOIN ZWAPROFILEPUSHNAME PN ON GM.ZMEMBERJID = PN.ZJID
                    WHERE GM.ZCHATSESSION = ?
                    ORDER BY GM.ZCONTACTNAME
                """, (session_pk,)).fetchall()
                for mr in member_rows:
                    member_jid = str(row_val(mr, "ZMEMBERJID", ""))
                    if not member_jid:
                        continue
                    raw_name = str(row_val(mr, "member_name", ""))
                    push_name = str(row_val(mr, "push_name", ""))
                    # Skip garbage WhatsApp encryption artifacts
                    if raw_name and "+EAA" in raw_name:
                        raw_name = ""
                    # Resolve via identity map (has partner_name, push_name, LID->phone)
                    resolved = id_map.get(member_jid) or {}
                    phone = resolved.get("phone", "") or _normalize_jid(member_jid)
                    # Name priority: contact_name > identity map name > push_name
                    resolved_name = resolved.get("name", "")
                    if resolved_name and "+EAA" in resolved_name:
                        resolved_name = ""
                    if push_name and "+EAA" in push_name:
                        push_name = ""
                    name = raw_name or resolved_name or push_name
                    # Skip the user
                    if phone == user_phone or member_jid == user_jid:
                        continue
                    # Dedup by phone
                    dedup_key = phone if (phone and len(phone) >= 10 and phone.isdigit()) else member_jid
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    members.append(_format_sender(name, phone))
            except sqlite3.OperationalError:
                pass
            group_member_cache[session_pk] = members
            return members

        events: list[Event] = []
        errors: list[str] = []

        try:
            rows = conn.execute(query, (cutoff_cd, fetch_limit)).fetchall()

            for row in rows:
                msg_date = row_val(row, "ZMESSAGEDATE", 0)
                if msg_date:
                    try:
                        ts = int(msg_date + APPLE_EPOCH_OFFSET)
                    except (TypeError, ValueError):
                        ts = int(time.time())
                else:
                    ts = int(time.time())

                message_type = row_val(row, "ZMESSAGETYPE", 0) or 0

                # Reaction events (type 59) — lightweight, no text content
                if message_type == 59:
                    is_from_me = bool(row_val(row, "ZISFROMME", 0))
                    stanza_id = str(row_val(row, "ZSTANZAID", "") or "")
                    contact_jid = str(row_val(row, "ZCONTACTJID", ""))
                    from_jid = str(row_val(row, "ZFROMJID", ""))

                    # Skip reactions from blocked contacts
                    if contact_jid in blocked_jids or from_jid in blocked_jids:
                        continue

                    # Resolve reactor identity
                    reactor_info = id_map.get(from_jid) or id_map.get(contact_jid) or {}
                    reactor_phone = reactor_info.get("phone", "") or _normalize_jid(from_jid or contact_jid)
                    reactor_name = reactor_info.get("name", "")

                    if is_from_me:
                        participants = [_format_sender("me", user_phone)]
                    else:
                        participants = [_format_sender(reactor_name, reactor_phone)]

                    # Try to extract emoji from ZWAMEDIAITEM.ZMETADATA blob
                    media_blob = row_val(row, "media_metadata", None)
                    reaction_emoji = _try_decode_reaction_emoji(media_blob)

                    source_id = stanza_id or f"wa-react-{row['Z_PK']}"
                    event_id = Event.make_id("whatsapp", source_id)

                    events.append(Event(
                        id=event_id,
                        source="whatsapp",
                        source_id=source_id,
                        event_type=EVENT_TYPE_REACTION,
                        timestamp=ts,
                        participants=tuple(participants),
                        raw_content=None,
                        content_hash="",
                        metadata={
                            "subject": "",
                            "is_from_me": is_from_me,
                            "thread_id": contact_jid or "",
                            "message_type": 59,
                            "reaction_type": "emoji",
                            "reaction_emoji": reaction_emoji,
                            "parent_stanza_id": stanza_id,
                            "is_reaction": True,
                        },
                        sensitivity=SensitivityLevel.SENSITIVE,
                    ))
                    continue

                text = str(row_val(row, "ZTEXT", ""))
                if not text:
                    continue

                is_from_me = bool(row_val(row, "ZISFROMME", 0))
                contact_jid = str(row_val(row, "ZCONTACTJID", ""))

                # Skip messages from blocked contacts
                from_jid_raw = str(row_val(row, "ZFROMJID", ""))
                if contact_jid in blocked_jids or from_jid_raw in blocked_jids:
                    continue
                partner_name = str(row_val(row, "ZPARTNERNAME", ""))
                from_jid = str(row_val(row, "ZFROMJID", ""))
                session_type = row_val(row, "ZSESSIONTYPE", 0)

                is_group = session_type == 1 or "@g.us" in contact_jid

                # Resolve identity
                contact_info = id_map.get(contact_jid) or id_map.get(from_jid) or {}
                contact_phone = contact_info.get("phone", "") or _normalize_jid(contact_jid)
                contact_name = contact_info.get("name", "") or partner_name

                # Resolve sender/recipient
                if is_group and not is_from_me:
                    group_sender_jid = str(row_val(row, "group_sender_jid", ""))
                    group_sender_name = str(row_val(row, "group_sender_name", ""))
                    if group_sender_jid:
                        sender_info = id_map.get(group_sender_jid) or {}
                        sender_phone = sender_info.get("phone", "") or _normalize_jid(group_sender_jid)
                        sender_name = group_sender_name or sender_info.get("name", "")
                    elif from_jid and "@g.us" not in from_jid:
                        sender_info = id_map.get(from_jid) or {}
                        sender_phone = sender_info.get("phone", "") or _normalize_jid(from_jid)
                        sender_name = sender_info.get("name", "")
                    else:
                        sender_phone = ""
                        sender_name = ""
                    sender = _format_sender(sender_name, sender_phone)
                    group_name = partner_name or contact_phone
                    recipient = _format_sender(group_name, contact_phone)
                elif is_from_me:
                    sender = _format_sender("me", user_phone)
                    recipient = _format_sender(contact_name, contact_phone)
                else:
                    sender = _format_sender(contact_name, contact_phone)
                    recipient = _format_sender("me", user_phone)

                # Build participants — for groups, include all members
                participants = [sender]
                if recipient and recipient not in participants:
                    participants.append(recipient)

                if is_group:
                    session_pk = row_val(row, "ZCHATSESSION", None)
                    if session_pk:
                        group_members = _get_group_members(session_pk)
                        for member in group_members:
                            if member not in participants:
                                participants.append(member)

                source_id = str(row["Z_PK"])
                event_id = Event.make_id("whatsapp", source_id)
                content_hash = Event.content_hash_of(text)

                # Collect alias metadata
                wa_partner = (id_map.get(contact_jid) or id_map.get(from_jid) or {}).get("partner_name", "") or partner_name
                wa_push = (id_map.get(contact_jid) or id_map.get(from_jid) or {}).get("push_name", "")

                events.append(Event(
                    id=event_id,
                    source="whatsapp",
                    source_id=source_id,
                    event_type=EVENT_TYPE_MESSAGE,
                    timestamp=ts,
                    participants=tuple(participants),
                    raw_content=text,
                    content_hash=content_hash,
                    metadata={
                        "subject": "",
                        "is_from_me": is_from_me,
                        "thread_id": contact_jid or "",
                        "phone": contact_phone,
                        "is_group": is_group,
                        "partner_name": wa_partner,
                        "push_name": wa_push,
                        "group_name": partner_name if is_group else "",
                        "message_type": row_val(row, "ZMESSAGETYPE", 0),
                        # Content type tags — URL shares (7) and document shares (8)
                        # are potentially high-signal (booking links, shared docs)
                        "shared_url": message_type == 7,
                        "shared_document": message_type == 8,
                    },
                    sensitivity=SensitivityLevel.SENSITIVE,
                ))

        except sqlite3.OperationalError as exc:
            errors.append(f"WhatsApp query failed: {exc}")
        finally:
            conn.close()

        return IngestResult(
            source="whatsapp",
            events=events,
            errors=errors,
            duration_seconds=time.time() - t0,
        )
