"""Person resolution: cross-source identity merging via union-find.

Parses participant strings from all sources, normalizes identifiers
(emails, phones), and merges them into Person entities.

Resolution strategy:
1. Parse every participant string into (name, identifier_type, identifier)
2. Normalize: lowercase emails, digits-only phones with country code
3. Use Contacts.app as ground truth -- each contact with both email and phone
   creates a bridge between those namespaces
4. Merge: if two participant strings share any normalized identifier, they're
   the same person (union-find with path compression + union by rank)
5. The user is identified by known emails/phones and marked is_user=True

This runs as a deterministic pass (no LLM).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Pattern: "Name <identifier>" or bare identifier
_NAME_ANGLE_RE = re.compile(r"^(.+?)\s*<([^>]+)>$")

# WhatsApp group IDs: contain a hyphen in the numeric part
_WA_GROUP_RE = re.compile(r"^\d+-\d+$")

# Slack channels
_SLACK_CHANNEL_RE = re.compile(r"^#")

# Email pattern (loose)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Calendar names that aren't people
_CALENDAR_NAMES = {"US Holidays", "Other", "Birthdays", "Siri Suggestions"}


@dataclass
class ParsedIdentity:
    """A single parsed identity from a participant string."""
    raw: str
    display_name: str = ""
    email: str = ""
    phone: str = ""
    source: str = ""
    is_group: bool = False
    is_noise: bool = False


def normalize_phone(raw: str) -> str:
    """Normalize any phone format to +<digits>.

    Handles:
        +19196274709        -> +19196274709
        15550100001         -> +15550100001
        (612) 207-2839      -> +16122072839
        +91 98445 67566     -> +919844567566
        5550100002          -> +15550100002
    """
    normalized = unicodedata.normalize("NFKD", raw)
    digits = "".join(c for c in normalized if c.isdigit())
    if not digits:
        return ""
    if len(digits) == 10:
        digits = "1" + digits
    return f"+{digits}"


def normalize_email(raw: str) -> str:
    """Normalize email: lowercase, strip whitespace, Gmail dot normalization.

    Gmail (and Google Workspace) treats dots in the local part as meaningless:
    foo.bar@gmail.com == foobar@gmail.com. Apply this to all @gmail.com and
    @googlemail.com addresses.
    """
    email = raw.lower().strip()
    if "@" in email:
        local, domain = email.rsplit("@", 1)
        if domain in ("gmail.com", "googlemail.com"):
            local = local.replace(".", "")
        email = f"{local}@{domain}"
    return email


def parse_participant(raw: str, source: str) -> ParsedIdentity:
    """Parse a single participant string into a structured identity.

    Handles all observed formats:
        Mail:     "Name <email>" or "email" or "email; CC: other@email"
        iMessage: "+19196274709"
        WhatsApp: "Name <phone>" or "phone" or "GroupName <groupid>"
        Slack:    "Display Name" or "#channel"
        Calendar: "Calendar Name" or "email"
        Contacts: phone (various formats)
    """
    raw = raw.strip()
    if not raw:
        return ParsedIdentity(raw=raw, is_noise=True)

    result = ParsedIdentity(raw=raw, source=source)

    # Slack channels
    if _SLACK_CHANNEL_RE.match(raw):
        result.is_group = True
        result.display_name = raw
        return result

    # Calendar noise
    if source == "calendar" and raw in _CALENDAR_NAMES:
        result.is_noise = True
        result.display_name = raw
        return result

    # Granola: always "meeting"
    if source == "granola" and raw == "meeting":
        result.is_noise = True
        return result

    # Strip CC notation
    if "; CC:" in raw:
        raw = raw.split("; CC:")[0].strip()
        result.raw = raw

    # Try "Name <identifier>" pattern
    m = _NAME_ANGLE_RE.match(raw)
    if m:
        result.display_name = m.group(1).strip()
        identifier = m.group(2).strip()
        if _EMAIL_RE.match(identifier):
            result.email = normalize_email(identifier)
        elif _WA_GROUP_RE.match(identifier):
            result.is_group = True
        else:
            phone = normalize_phone(identifier)
            if phone and len(phone) >= 8:
                result.phone = phone
            elif len(identifier) > 15:
                result.is_group = True
        return result

    # Bare email
    if _EMAIL_RE.match(raw):
        result.email = normalize_email(raw)
        return result

    # Bare phone
    phone = normalize_phone(raw)
    if phone and len(phone) >= 8:
        result.phone = phone
        return result

    # Display name (Slack names, calendar names, etc.)
    result.display_name = raw
    return result


def parse_mail_participants(participants: list[str], meta: dict) -> list[ParsedIdentity]:
    """Parse mail participants, expanding CC notation."""
    results = []
    for p in participants:
        parts = re.split(r";\s*CC:\s*", p)
        for part in parts:
            part = part.strip()
            if part:
                results.append(parse_participant(part, "mail"))
    return results


def parse_calendar_participants(participants: list[str], meta: dict) -> list[ParsedIdentity]:
    """Parse calendar participants including attendees from metadata."""
    results = []
    for p in participants:
        parsed = parse_participant(p, "calendar")
        if not parsed.is_noise:
            results.append(parsed)

    for att in meta.get("attendees", []):
        email = (att.get("email") or "").strip()
        name = (att.get("name") or "").strip()
        if email:
            pid = ParsedIdentity(
                raw=f"{name} <{email}>" if name else email,
                display_name=name if name != email else "",
                email=normalize_email(email),
                source="calendar",
            )
            results.append(pid)

    org_email = (meta.get("organizer_email") or "").strip()
    if org_email:
        org_name = (meta.get("organizer_name") or "").strip()
        results.append(ParsedIdentity(
            raw=org_email,
            display_name=org_name,
            email=normalize_email(org_email),
            source="calendar",
        ))

    return results


def _parse_contact_event(meta: dict) -> list[ParsedIdentity]:
    """Parse a contacts identity event into ParsedIdentity objects."""
    name = meta.get("name", "")
    emails = meta.get("emails", [])
    phones = meta.get("phones", [])

    results = []
    for email in emails:
        results.append(ParsedIdentity(
            raw=email, display_name=name,
            email=normalize_email(email), source="contacts",
        ))
    for phone in phones:
        normalized = normalize_phone(phone)
        if normalized:
            results.append(ParsedIdentity(
                raw=phone, display_name=name,
                phone=normalized, source="contacts",
            ))
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Union-Find
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UnionFind:
    """Disjoint set with path compression and union by rank.

    O(alpha(n)) amortized per find/union, where alpha is the inverse
    Ackermann function (effectively constant for all practical sizes).
    """

    def __init__(self):
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def find(self, x: str) -> str:
        """Find the root representative for x, with path compression."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        """Merge the sets containing x and y."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def groups(self) -> dict[str, list[str]]:
        """Return {root: [members]} for all elements."""
        result: dict[str, list[str]] = defaultdict(list)
        for x in self._parent:
            result[self.find(x)].append(x)
        return dict(result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ResolvedPerson
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ResolvedPerson:
    """A merged person with all known identifiers."""
    person_id: str = ""
    canonical_name: str = ""
    emails: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)
    display_names: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    is_user: bool = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Resolver
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def resolve_persons(
    store: LayeredGraphStore,
    user_config: dict | None = None,
) -> list[ResolvedPerson]:
    """Resolve all event participants into merged Person entities.

    Args:
        store: LayeredGraphStore with ingested events
        user_config: Optional dict with user's known identifiers:
            {
                "emails": ["user@example.com", ...],
                "phones": ["+15550100001", ...],
                "name": "User Name",
            }

    Returns:
        List of ResolvedPerson with merged identifiers.
    """
    user_config = user_config or {}
    user_emails = {normalize_email(e) for e in user_config.get("emails", [])}
    user_phones = {normalize_phone(p) for p in user_config.get("phones", [])}
    user_name = user_config.get("name", "")

    uf = UnionFind()

    # Identifier -> display names observed
    id_names: dict[str, set[str]] = defaultdict(set)
    # Identifier -> sources seen in
    id_sources: dict[str, set[str]] = defaultdict(set)
    # Identifier -> name from Contacts.app (highest priority for canonical name)
    contacts_name: dict[str, str] = {}

    # ── Phase 1: Parse all events and collect identifiers ─────

    rows = store.conn.execute(
        "SELECT source, participants, metadata FROM events"
    ).fetchall()

    for row in rows:
        source = row["source"]
        participants = json.loads(row["participants"] or "[]")
        meta = json.loads(row["metadata"] or "{}")

        if source == "mail":
            identities = parse_mail_participants(participants, meta)
        elif source == "calendar":
            identities = parse_calendar_participants(participants, meta)
        elif source == "contacts":
            identities = _parse_contact_event(meta)
        else:
            identities = [parse_participant(p, source) for p in participants]

        for pid in identities:
            if pid.is_noise or pid.is_group:
                continue

            ids_for_this = []
            if pid.email:
                key = f"email:{pid.email}"
                ids_for_this.append(key)
                id_sources[key].add(source)
                if pid.display_name:
                    id_names[key].add(pid.display_name)

            if pid.phone:
                key = f"phone:{pid.phone}"
                ids_for_this.append(key)
                id_sources[key].add(source)
                if pid.display_name:
                    id_names[key].add(pid.display_name)

            # Register every identifier in the union-find (even singletons).
            # Without this, phone-only participants are invisible to uf.groups()
            # and never become persons — causing massive under-linking.
            for key in ids_for_this:
                uf.find(key)

            # If this identity has both email and phone, merge them
            if len(ids_for_this) >= 2:
                for i in range(1, len(ids_for_this)):
                    uf.union(ids_for_this[0], ids_for_this[i])

    # ── Phase 2: Contact bridges (phone <-> email) ──────────

    contact_rows = store.conn.execute(
        "SELECT metadata FROM events WHERE source = 'contacts'"
    ).fetchall()

    for row in contact_rows:
        meta = json.loads(row["metadata"] or "{}")
        emails = meta.get("emails", [])
        phones = meta.get("phones", [])
        name = meta.get("name", "")

        normalized_emails = [f"email:{normalize_email(e)}" for e in emails if e.strip()]
        normalized_phones = [f"phone:{normalize_phone(p)}" for p in phones if normalize_phone(p)]
        all_ids = normalized_emails + normalized_phones

        for key in all_ids:
            if name:
                id_names[key].add(name)
                contacts_name[key] = name
            id_sources[key].add("contacts")

        if len(all_ids) >= 2:
            for i in range(1, len(all_ids)):
                uf.union(all_ids[0], all_ids[i])

    # ── Phase 3: User identity merging ────────────────────────

    user_ids = []
    for email in user_emails:
        key = f"email:{email}"
        user_ids.append(key)
        id_names[key].add(user_name or "me")
        id_sources[key].add("user_config")
    for phone in user_phones:
        key = f"phone:{phone}"
        user_ids.append(key)
        id_names[key].add(user_name or "me")
        id_sources[key].add("user_config")

    if len(user_ids) >= 2:
        for i in range(1, len(user_ids)):
            uf.union(user_ids[0], user_ids[i])

    # ── Phase 4: Build merged persons ─────────────────────────

    groups = uf.groups()
    persons: list[ResolvedPerson] = []

    for root, members in groups.items():
        person = ResolvedPerson()

        for member in members:
            id_type, identifier = member.split(":", 1)
            if id_type == "email":
                person.emails.add(identifier)
            elif id_type == "phone":
                person.phones.add(identifier)
            person.display_names.update(id_names.get(member, set()))
            person.sources.update(id_sources.get(member, set()))

        # Check if this is the user
        if person.emails & user_emails or person.phones & user_phones:
            person.is_user = True
            person.canonical_name = user_name or "me"

        # Pick best canonical name.
        # Priority: Contacts.app name > longest real name > any display name.
        # Contacts is ground truth — if a person exists there, use that name
        # regardless of what email headers or other sources say.
        if not person.canonical_name:
            # Check if any identifier has a Contacts.app name
            contact_name_for_person = ""
            for member in members:
                if member in contacts_name:
                    contact_name_for_person = contacts_name[member]
                    break

            if contact_name_for_person:
                person.canonical_name = contact_name_for_person
            elif person.display_names:
                real_names = [n for n in person.display_names
                              if "@" not in n and n != "me" and len(n) > 1]
                if real_names:
                    person.canonical_name = max(real_names, key=len)
                else:
                    person.canonical_name = next(iter(person.display_names))

        # Generate stable person_id from sorted identifiers
        all_ids = sorted(person.emails) + sorted(person.phones)
        if all_ids:
            person.person_id = f"person:{hashlib.sha256('|'.join(all_ids).encode()).hexdigest()[:16]}"
        else:
            person.person_id = f"person:name:{hashlib.sha256(person.canonical_name.encode()).hexdigest()[:12]}"

        persons.append(person)

    # Sort: user first, then by number of identifiers
    persons.sort(key=lambda p: (not p.is_user, -(len(p.emails) + len(p.phones))))

    return persons


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Persistence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def persist_persons(
    store: LayeredGraphStore,
    persons: list[ResolvedPerson],
) -> dict[str, int]:
    """Write resolved persons and their identifiers to the graph store.

    Returns: {persons_written, identifiers_written, user_found}
    """
    persons_written = 0
    identifiers_written = 0
    user_found = False

    for person in persons:
        store.put_person(
            person_id=person.person_id,
            canonical_name=person.canonical_name,
            is_user=person.is_user,
            sources=sorted(person.sources),
        )
        persons_written += 1

        if person.is_user:
            user_found = True

        for email in person.emails:
            store.add_person_identifier(
                person_id=person.person_id,
                id_type="email",
                identifier=email,
                display_name=person.canonical_name,
                source="resolved",
            )
            identifiers_written += 1

        for phone in person.phones:
            store.add_person_identifier(
                person_id=person.person_id,
                id_type="phone",
                identifier=phone,
                display_name=person.canonical_name,
                source="resolved",
            )
            identifiers_written += 1

        for name in person.display_names:
            if name and name != person.canonical_name:
                store.add_person_identifier(
                    person_id=person.person_id,
                    id_type="display_name",
                    identifier=name,
                    display_name=person.canonical_name,
                    source="resolved",
                )
                identifiers_written += 1

    return {
        "persons_written": persons_written,
        "identifiers_written": identifiers_written,
        "user_found": user_found,
    }
