"""Contacts adapter — reads from macOS AddressBook SQLite databases.

Unlike other adapters, this doesn't produce message events.
It produces identity events — records of who exists in the user's
address book with their phones and emails. These are consumed by
the resolver to create phone/email bridges for cross-source merging.

Database locations:
  ~/Library/Application Support/AddressBook/AddressBook-v22.abcddb
  ~/Library/Application Support/AddressBook/Sources/*/AddressBook-v22.abcddb
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from pathlib import Path

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
    ADDRESSBOOK_DIR,
    EVENT_TYPE_IDENTITY,
    SOURCE_ID_HASH_LEN,
)
from loom.models import Event
from loom.privacy import SensitivityLevel

logger = logging.getLogger(__name__)


def _find_addressbook_dbs() -> list[Path]:
    """Find all AddressBook SQLite databases (root + per-source)."""
    dbs = []
    # Root DB
    root = ADDRESSBOOK_DIR / "AddressBook-v22.abcddb"
    if root.exists():
        dbs.append(root)
    # Per-source DBs (iCloud, Exchange, etc.)
    sources_dir = ADDRESSBOOK_DIR / "Sources"
    if sources_dir.exists():
        for src_dir in sources_dir.iterdir():
            candidate = src_dir / "AddressBook-v22.abcddb"
            if candidate.exists():
                dbs.append(candidate)
    return dbs


def _read_contacts_from_db(path: Path) -> list[dict]:
    """Read contacts from a single AddressBook database."""
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.OperationalError:
        return []

    contacts_by_pk: dict[int, dict] = {}

    try:
        for row in conn.execute(
            """SELECT Z_PK, ZFIRSTNAME, ZLASTNAME, ZORGANIZATION, ZNICKNAME, ZJOBTITLE
               FROM ZABCDRECORD
               WHERE ZFIRSTNAME IS NOT NULL OR ZLASTNAME IS NOT NULL OR ZORGANIZATION IS NOT NULL"""
        ):
            first = row["ZFIRSTNAME"] or ""
            last = row["ZLASTNAME"] or ""
            org = row["ZORGANIZATION"] or ""
            nick = row["ZNICKNAME"] or ""
            job_title = row["ZJOBTITLE"] or ""
            name = f"{first} {last}".strip()
            if not name:
                name = org or nick or ""
            if not name:
                continue
            contacts_by_pk[row["Z_PK"]] = {
                "name": name,
                "emails": [],
                "phones": [],
                "social_profiles": [],
                "organization": org,
                "job_title": job_title,
            }
    except sqlite3.OperationalError:
        conn.close()
        return []

    # Email addresses
    try:
        for row in conn.execute(
            "SELECT ZOWNER, ZADDRESSNORMALIZED, ZADDRESS FROM ZABCDEMAILADDRESS WHERE ZOWNER IS NOT NULL"
        ):
            pk = row["ZOWNER"]
            addr = row["ZADDRESSNORMALIZED"] or row["ZADDRESS"] or ""
            if pk in contacts_by_pk and addr:
                contacts_by_pk[pk]["emails"].append(addr.lower().strip())
    except sqlite3.OperationalError:
        pass

    # Phone numbers
    try:
        for row in conn.execute(
            "SELECT ZOWNER, ZFULLNUMBER FROM ZABCDPHONENUMBER WHERE ZOWNER IS NOT NULL"
        ):
            pk = row["ZOWNER"]
            phone = row["ZFULLNUMBER"] or ""
            if pk in contacts_by_pk and phone:
                contacts_by_pk[pk]["phones"].append(phone.strip())
    except sqlite3.OperationalError:
        pass

    # Social profiles (Slack, Facebook, WhatsApp handles — cross-source identity bridges)
    try:
        for row in conn.execute(
            """SELECT ZOWNER, ZSERVICENAME, ZUSERNAME, ZURLSTRING
               FROM ZABCDSOCIALPROFILE WHERE ZOWNER IS NOT NULL"""
        ):
            pk = row["ZOWNER"]
            service = (row["ZSERVICENAME"] or "").strip()
            username = (row["ZUSERNAME"] or "").strip()
            url = (row["ZURLSTRING"] or "").strip()
            if pk in contacts_by_pk and (service or username):
                contacts_by_pk[pk]["social_profiles"].append({
                    "service": service,
                    "username": username,
                    "url": url,
                })
    except sqlite3.OperationalError:
        pass

    conn.close()
    return list(contacts_by_pk.values())


class ContactsAdapter(SourceAdapter):
    """Reads macOS Contacts.app AddressBook databases."""

    @property
    def source_name(self) -> str:
        return "contacts"

    def check_availability(self) -> AvailabilityResult:
        if not ADDRESSBOOK_DIR.exists():
            return AvailabilityResult(
                available=False,
                source="contacts",
                reason="directory_not_found",
                user_action="AddressBook directory not found. Open Contacts.app at least once.",
            )
        dbs = _find_addressbook_dbs()
        if not dbs:
            return AvailabilityResult(
                available=False,
                source="contacts",
                reason="no_databases",
                user_action="No AddressBook databases found.",
            )
        return check_sqlite_readable(dbs[0], "contacts")

    def check_schema(self) -> SchemaResult:
        dbs = _find_addressbook_dbs()
        if not dbs:
            return SchemaResult(compatible=False, source="contacts", warnings=["No databases found"])
        return check_sqlite_tables(
            dbs[0], "contacts",
            required=["ZABCDRECORD"],
            optional=["ZABCDEMAILADDRESS", "ZABCDPHONENUMBER"],
        )

    def ingest(self, since_ts: int = 0, limit: int = 0) -> IngestResult:
        """Ingest contacts as identity events.

        Contacts don't have timestamps, so since_ts is ignored.
        Every ingest is a full scan (idempotent via deterministic IDs).
        """
        t0 = time.time()

        dbs = _find_addressbook_dbs()
        if not dbs:
            return IngestResult(
                source="contacts",
                errors=["No AddressBook databases found. Open Contacts.app at least once."],
            )

        all_contacts: list[dict] = []
        for path in dbs:
            all_contacts.extend(_read_contacts_from_db(path))

        events: list[Event] = []
        seen_ids: set[str] = set()

        for c in all_contacts:
            # Build stable source_id from sorted identifiers
            id_parts = sorted(set(
                e.lower().strip()
                for e in c["emails"] + c["phones"]
                if e.strip()
            ))
            if not id_parts:
                continue

            source_id = hashlib.sha256(
                "|".join(id_parts).encode()
            ).hexdigest()[:SOURCE_ID_HASH_LEN]

            if source_id in seen_ids:
                continue
            seen_ids.add(source_id)

            event_id = Event.make_id("contacts", source_id)

            events.append(Event(
                id=event_id,
                source="contacts",
                source_id=source_id,
                event_type=EVENT_TYPE_IDENTITY,
                timestamp=int(time.time()),
                participants=tuple(c["emails"] + c["phones"]),
                raw_content=None,
                metadata={
                    "name": c["name"],
                    "emails": c["emails"],
                    "phones": c["phones"],
                    "social_profiles": c.get("social_profiles", []),
                    "organization": c.get("organization", ""),
                    "job_title": c.get("job_title", ""),
                },
                sensitivity=SensitivityLevel.PRIVATE,
            ))

            if limit > 0 and len(events) >= limit:
                break

        return IngestResult(
            source="contacts",
            events=events,
            duration_seconds=time.time() - t0,
        )
