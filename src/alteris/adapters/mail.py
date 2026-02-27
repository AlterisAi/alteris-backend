"""Mail.app adapter — reads from macOS Mail's Envelope Index SQLite database.

Body retrieval priority:
  1. summaries table (Apple's search index text)
  2. EMLX file on disk (full RFC822 message)

Handles epoch auto-detection (Unix vs Apple epoch) and cross-account
deduplication via (date_sent, sender_addr, subject) tuples.
"""

from __future__ import annotations

import email as email_lib
import logging
import os
import re
import sqlite3
import time
from pathlib import Path

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
    BODY_PREVIEW_LEN,
    EVENT_TYPE_EMAIL,
    MAIL_DB,
    SOURCE_ID_HASH_LEN,
)
from alteris.models import Event
from alteris.privacy import SensitivityLevel

logger = logging.getLogger(__name__)

# Mail.app V10+ stores date_sent as Unix epoch (NOT Apple epoch).
# We auto-detect based on value range.
_UNIX_EPOCH_THRESHOLD = 9.5e8

# Maximum body length to store
_MAIL_BODY_MAX = 10_000

# Mailbox folder patterns to skip during ingestion.
# These folders add noise without useful signal for the pipeline.
# Only skip folders the user has explicitly rejected or that are noise.
# DO NOT skip All Mail, Archive, or Sent — Gmail stores real inbox messages
# there and Sent contains user-authored content needed for commitment tracking.
_SKIP_MAILBOX_PATTERNS = (
    "/Spam",
    "/Junk",
    "/Junk%20Email",
    "/Trash",
    "/Deleted",
    "/Deleted%20Items",
    "/Deleted%20Messages",
    "/Recovered%20Messages",
    "/Drafts",
)

# Sender patterns that indicate automated/system-generated email.
# Used to distinguish genuine automated senders from personal corporate
# emails that Apple Intelligence misclassifies as automated (e.g.,
# miranda@anthropic.com sent via Greenhouse ATS).
_NOREPLY_PREFIXES = (
    "noreply@", "no-reply@", "no_reply@",
    "donotreply@", "do-not-reply@", "do_not_reply@",
    "notifications@", "notification@", "notify@",
    "mailer-daemon@", "postmaster@",
    "auto@", "automated@", "system@",
    "bounce@", "bounces@",
    "info@", "support@", "updates@",
)
_ATS_DOMAINS = (
    "greenhouse-mail.io", "greenhouse.io",
    "codesignal.com", "lever.co", "ashbyhq.com",
    "workday.com", "icims.com", "smartrecruiters.com",
    "jobvite.com", "myworkday.com",
)


def _is_noreply_sender(addr: str) -> bool:
    """Check if a sender address matches known automated/ATS patterns."""
    if not addr:
        return False
    addr_lower = addr.lower()
    if any(addr_lower.startswith(p) for p in _NOREPLY_PREFIXES):
        return True
    # Check if the domain is a known ATS/transactional sender
    at_idx = addr_lower.find("@")
    if at_idx >= 0:
        domain = addr_lower[at_idx + 1:]
        if any(domain == d or domain.endswith("." + d) for d in _ATS_DOMAINS):
            return True
    return False


def _find_mail_db() -> Path | None:
    """Find the newest Mail.app Envelope Index."""
    mail_dir = Path.home() / "Library" / "Mail"
    if not mail_dir.exists():
        return None
    for vdir in sorted(mail_dir.glob("V*"), reverse=True):
        candidate = vdir / "MailData" / "Envelope Index"
        if candidate.exists():
            return candidate
    return None


def _find_mail_vdir() -> Path | None:
    """Locate the Mail V* directory (parent of MailData)."""
    mail_dir = Path.home() / "Library" / "Mail"
    if not mail_dir.exists():
        return None
    for vdir in sorted(mail_dir.glob("V*"), reverse=True):
        if (vdir / "MailData" / "Envelope Index").exists():
            return vdir
    return None


def _to_unix_timestamp(raw_date: float) -> float:
    """Convert a Mail.app date_sent value to Unix timestamp."""
    if raw_date <= 0:
        return 0.0
    if raw_date < _UNIX_EPOCH_THRESHOLD:
        return raw_date + APPLE_EPOCH_OFFSET
    return raw_date


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EMLX body extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_emlx_index: dict[int, Path] | None = None


def _build_emlx_index(mail_vdir: Path) -> dict[int, Path]:
    """Walk the Mail directory once and build {ROWID: Path} index."""
    index: dict[int, Path] = {}
    for dirpath, _dirs, filenames in os.walk(str(mail_vdir)):
        if "/Messages" not in dirpath:
            continue
        for fn in filenames:
            if not fn.endswith(".emlx"):
                continue
            base = fn.replace(".partial.emlx", "").replace(".emlx", "")
            try:
                rowid = int(base)
            except ValueError:
                continue
            is_partial = ".partial." in fn
            full_path = Path(dirpath) / fn
            if rowid not in index or (not is_partial):
                index[rowid] = full_path
    return index


def _read_emlx_body(emlx_path: Path) -> str:
    """Read an EMLX file and return the text body."""
    try:
        raw = emlx_path.read_bytes()
        first_nl = raw.index(b"\n")
        byte_count = int(raw[:first_nl])
        rfc822_bytes = raw[first_nl + 1 : first_nl + 1 + byte_count]
        msg = email_lib.message_from_bytes(rfc822_bytes)
        return _extract_text_from_email(msg)
    except Exception:
        return ""


def _extract_text_from_email(msg: email_lib.message.Message) -> str:
    """Extract text body from a parsed email.message.Message."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return _html_to_text(payload.decode(charset, errors="replace"))
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/html":
                return _html_to_text(text)
            return text
    return ""


def _html_to_text(html: str) -> str:
    """Convert HTML to clean text using BeautifulSoup if available, regex fallback."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        # Fallback: regex-based stripping
        text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"<p[^>]*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)

    from html import unescape
    text = unescape(text)

    # Strip invisible Unicode (format chars, line/paragraph separators)
    # but preserve normal whitespace
    import unicodedata
    text = "".join(
        c for c in text
        if unicodedata.category(c) not in ("Cf", "Zl", "Zp")
        or c in ("\n", "\t")
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_email_body(body: str) -> str:
    """Remove quoted reply chains and boilerplate from email body.

    Strips: quoted lines (>), On...wrote: attributions, forwarded headers,
    Outlook From:/Sent:/To: blocks, device boilerplate.
    Preserves: sign-offs (tone signal), inline content.
    """
    # If the body looks like HTML, parse it first
    if "<html" in body.lower() or "<body" in body.lower() or "<div" in body.lower():
        body = _html_to_text(body)

    lines = body.split("\n")
    cleaned: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Quoted reply lines
        if stripped.startswith(">"):
            break

        # "On Mon, Jan 13, 2026 at 3:45 PM John wrote:"
        if re.match(r"^On .+ wrote:\s*$", stripped):
            break
        # Multi-line variant: blank line then "On ... wrote:" on next
        if (stripped == "" and i + 1 < len(lines)
                and re.match(r"^On .+ wrote:\s*$", lines[i + 1].strip())):
            break

        # Forwarded message markers
        if re.match(r"^-{3,}\s*(Forwarded|Original)", stripped):
            break
        if stripped.lower().startswith("begin forwarded message"):
            break

        # Outlook-style header block: "From: ...\nSent: ...\nTo: ..."
        if (re.match(r"^From:\s+.+", stripped) and i + 2 < len(lines)
                and re.match(r"^Sent:\s+", lines[i + 1].strip())):
            break

        # Separator lines (underscores)
        if re.match(r"^_{3,}$", stripped):
            break

        # Device boilerplate
        stripped_lower = stripped.lower()
        if re.match(r"^sent from my (iphone|ipad|galaxy|android|samsung)", stripped_lower):
            break
        if stripped_lower.startswith("get outlook for"):
            break

        # Legal disclaimers
        if re.match(r"^this (email|message|communication) (is|was|may be) (confidential|intended|privileged)", stripped_lower):
            break

        cleaned.append(line)

    result = "\n".join(cleaned).strip()

    # Final pass: strip invisible Unicode from summary text too
    import unicodedata
    result = "".join(
        c for c in result
        if unicodedata.category(c) not in ("Cf", "Zl", "Zp")
        or c in ("\n", "\t")
    )
    result = re.sub(r"[ \t]+", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def _get_body(row: sqlite3.Row, mail_vdir: Path | None) -> str:
    """Get body text: summary -> EMLX fallback."""
    global _emlx_index

    body = str(row_val(row, "summary_text", ""))
    if len(body.strip()) > 5:
        if len(body) > _MAIL_BODY_MAX:
            body = body[:_MAIL_BODY_MAX]
        return _clean_email_body(body)

    if mail_vdir is not None:
        rowid = row["ROWID"]
        if _emlx_index is None:
            _emlx_index = _build_emlx_index(mail_vdir)
        emlx_path = _emlx_index.get(rowid)
        if emlx_path is not None:
            body = _read_emlx_body(emlx_path)
            if body and len(body.strip()) > 5:
                if len(body) > _MAIL_BODY_MAX:
                    body = body[:_MAIL_BODY_MAX]
                return _clean_email_body(body)

    return ""


def _parse_sender(row: sqlite3.Row) -> str:
    """Format sender as 'Name <email>' or just email."""
    addr = str(row_val(row, "sender_address", ""))
    name = str(row_val(row, "sender_name", ""))
    if name and addr:
        return f"{name} <{addr}>"
    return addr or name or "unknown"


def _fetch_recipients(
    conn: sqlite3.Connection, message_rowids: list[int],
) -> dict[int, dict[str, list[str]]]:
    """Batch fetch recipients for messages. Returns {rowid: {to, cc, bcc}}."""
    if not message_rowids:
        return {}
    result: dict[int, dict[str, list[str]]] = {}
    chunk_size = 500
    for i in range(0, len(message_rowids), chunk_size):
        chunk = message_rowids[i:i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        try:
            rows = conn.execute(
                f"""SELECT r.message, a.address, a.comment, r.type
                    FROM recipients r
                    LEFT JOIN addresses a ON r.address = a.ROWID
                    WHERE r.message IN ({placeholders})
                    ORDER BY r.message, r.type, r.position""",
                chunk,
            ).fetchall()
        except sqlite3.OperationalError:
            continue

        for row in rows:
            msg_id = row["message"]
            if msg_id not in result:
                result[msg_id] = {"to": [], "cc": [], "bcc": []}
            addr = str(row_val(row, "address", ""))
            name = str(row_val(row, "comment", ""))
            if not addr:
                continue
            formatted = f"{name} <{addr}>" if name else addr
            rtype = row["type"] or 0
            if rtype == 0:
                result[msg_id]["to"].append(formatted)
            elif rtype == 1:
                result[msg_id]["cc"].append(formatted)
            elif rtype == 2:
                result[msg_id]["bcc"].append(formatted)
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Adapter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MailAdapter(SourceAdapter):
    """Reads Mail.app Envelope Index SQLite database."""

    @property
    def source_name(self) -> str:
        return "mail"

    def check_availability(self) -> AvailabilityResult:
        db = _find_mail_db()
        if not db:
            return AvailabilityResult(
                available=False,
                source="mail",
                reason="database_not_found",
                user_action="Mail.app Envelope Index not found. Open Mail.app at least once.",
            )
        return check_sqlite_readable(db, "mail")

    def check_schema(self) -> SchemaResult:
        db = _find_mail_db()
        if not db:
            return SchemaResult(compatible=False, source="mail", warnings=["Database not found"])
        return check_sqlite_tables(
            db, "mail",
            required=["messages", "addresses", "subjects", "summaries", "mailboxes"],
            optional=["recipients"],
        )

    def ingest(self, since_ts: int = 0, limit: int = 0) -> IngestResult:
        t0 = time.time()
        db = _find_mail_db()
        if not db:
            return IngestResult(source="mail", errors=["Mail.app database not found"])

        mail_vdir = _find_mail_vdir()

        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except sqlite3.OperationalError:
            return IngestResult(
                source="mail",
                errors=["Cannot read Mail.app database. Grant Full Disk Access in System Settings."],
            )

        # Detect epoch format
        cutoff = since_ts
        if since_ts > 0:
            sample = conn.execute(
                "SELECT date_sent FROM messages WHERE date_sent > 0 LIMIT 1"
            ).fetchone()
            if sample and sample["date_sent"] < _UNIX_EPOCH_THRESHOLD:
                cutoff = since_ts - APPLE_EPOCH_OFFSET

        fetch_limit = (limit * 3) if limit > 0 else 500_000

        query = """
            SELECT
                m.ROWID,
                a.address AS sender_address,
                a.comment AS sender_name,
                s.subject AS subject_text,
                sm.summary AS summary_text,
                m.date_sent,
                m.date_received,
                m.read,
                m.flagged,
                m.conversation_id,
                m.automated_conversation,
                m.list_id_hash,
                m.is_urgent,
                mb.url AS mailbox_url,
                mgd.model_category,
                mgd.model_high_impact,
                mgd.urgent AS mgd_urgent,
                srv.replied,
                srv.junk_level
            FROM messages m
            LEFT JOIN addresses a ON m.sender = a.ROWID
            LEFT JOIN subjects s ON m.subject = s.ROWID
            LEFT JOIN summaries sm ON m.summary = sm.ROWID
            LEFT JOIN mailboxes mb ON m.mailbox = mb.ROWID
            LEFT JOIN message_global_data mgd ON m.message_id = mgd.message_id
            LEFT JOIN server_messages srv ON m.ROWID = srv.message
            WHERE m.date_sent > ?
            ORDER BY m.date_sent DESC
            LIMIT ?
        """

        events: list[Event] = []
        errors: list[str] = []
        seen_keys: set[tuple] = set()

        try:
            rows = conn.execute(query, (cutoff, fetch_limit)).fetchall()
            rowids = [row["ROWID"] for row in rows]
            all_recipients = _fetch_recipients(conn, rowids)

            for row in rows:
                # Skip junk mailboxes (spam, trash, deleted, drafts, recovered)
                mailbox_url = str(row_val(row, "mailbox_url", ""))
                if any(pat in mailbox_url for pat in _SKIP_MAILBOX_PATTERNS):
                    continue

                sender_addr = str(row_val(row, "sender_address", ""))
                date_sent = row_val(row, "date_sent", 0)
                subject = str(row_val(row, "subject_text", ""))
                dedup_key = (date_sent, sender_addr.lower(), subject.lower().strip())

                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                # Timestamp
                raw_date = row_val(row, "date_sent", 0) or row_val(row, "date_received", 0)
                ts = int(_to_unix_timestamp(raw_date)) if raw_date else int(time.time())

                body = _get_body(row, mail_vdir)
                sender = _parse_sender(row)
                is_from_me = False  # will be resolved by linker using user config
                conv_id = row_val(row, "conversation_id", None)

                msg_recips = all_recipients.get(row["ROWID"], {"to": [], "cc": [], "bcc": []})
                to_list = msg_recips["to"]
                cc_list = msg_recips["cc"]
                recip_str = ", ".join(to_list) if to_list else ""
                if cc_list:
                    recip_str += "; CC: " + ", ".join(cc_list)

                # Participants: sender + all recipients
                participants = []
                if sender and sender != "unknown":
                    participants.append(sender)
                for r in to_list + cc_list:
                    if r:
                        participants.append(r)

                source_id = str(row["ROWID"])
                event_id = Event.make_id("mail", source_id)
                content_hash = Event.content_hash_of(body) if body else ""

                # Apple Intelligence + server-side signals → metadata
                automated = bool(row_val(row, "automated_conversation", 0))
                list_id_raw = row_val(row, "list_id_hash", 0) or 0
                model_cat = row_val(row, "model_category", None)
                high_impact = bool(row_val(row, "model_high_impact", 0))
                is_urgent = bool(row_val(row, "is_urgent", 0))
                mgd_urgent = bool(row_val(row, "mgd_urgent", 0))
                replied = bool(row_val(row, "replied", 0))
                flagged = bool(row_val(row, "flagged", 0))
                junk_level = row_val(row, "junk_level", 0) or 0

                sender_is_noreply = _is_noreply_sender(sender_addr)

                events.append(Event(
                    id=event_id,
                    source="mail",
                    source_id=source_id,
                    event_type=EVENT_TYPE_EMAIL,
                    timestamp=ts,
                    participants=tuple(participants),
                    raw_content=body if body else None,
                    content_hash=content_hash,
                    metadata={
                        "subject": subject,
                        "is_from_me": is_from_me,
                        "thread_id": str(conv_id) if conv_id else "",
                        "mailbox": mailbox_url,
                        "cc": cc_list,
                        "rowid": row["ROWID"],
                        # Apple Intelligence + server signals for Stage 2
                        "automated": automated,
                        "list_id": list_id_raw != 0,
                        "model_category": model_cat,
                        "high_impact": high_impact,
                        "urgent": is_urgent or mgd_urgent,
                        "replied": replied,
                        "flagged": flagged,
                        "junk_level": junk_level,
                        # Sender pattern signal: True = confirmed automated sender,
                        # False with automated=True = possible false positive
                        # (e.g., miranda@anthropic.com via Greenhouse ATS)
                        "sender_is_noreply": sender_is_noreply,
                    },
                    sensitivity=SensitivityLevel.SENSITIVE,
                ))

                if limit > 0 and len(events) >= limit:
                    break

        except sqlite3.OperationalError as exc:
            errors.append(f"Mail query failed: {exc}")
        finally:
            conn.close()

        return IngestResult(
            source="mail",
            events=events,
            errors=errors,
            duration_seconds=time.time() - t0,
        )
