"""Link events to persons via participant resolution.

For each event, parses its participant strings, normalizes identifiers,
looks up person_ids from the person_identifiers table, and writes
event_persons edges with appropriate roles.

Roles:
    sender     -- who sent/wrote the message
    recipient  -- who received it (To:)
    cc         -- carbon-copied
    attendee   -- calendar event attendee
    organizer  -- calendar event organizer
    self       -- the user's own participation
    mentioned  -- named but not a direct participant
    identity   -- contacts: this event defines the person

This is a deterministic pass (no LLM).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum

from loom.constants import (
    CALL_MEETING_OVERLAP_WINDOW_SECONDS,
    CALL_MIN_DURATION_SECONDS,
    OVERLAP_FRACTION_CONTAINED,
    OVERLAP_FRACTION_PARTIAL,
    SPEECH_RATE_WPM,
)
from loom.resolver import (
    ParsedIdentity,
    _parse_contact_event,
    normalize_email,
    normalize_phone,
    parse_calendar_participants,
    parse_mail_participants,
    parse_participant,
)
from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)


class ParticipantRole(str, Enum):
    """Role of a person in an event."""
    SENDER = "sender"
    RECIPIENT = "recipient"
    CC = "cc"
    ATTENDEE = "attendee"
    ORGANIZER = "organizer"
    SELF = "self"
    MENTIONED = "mentioned"
    IDENTITY = "identity"
    MEMBER = "member"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Identifier -> person_id lookup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_lookup(store: LayeredGraphStore) -> dict[str, str]:
    """Build a fast identifier -> person_id lookup from person_identifiers.

    Returns dict like:
        "email:user@example.com" -> "person:1b300a08..."
        "phone:+15550100002"     -> "person:1784d038..."
    """
    rows = store.conn.execute(
        "SELECT person_id, identifier_type, identifier FROM person_identifiers"
    ).fetchall()
    lookup: dict[str, str] = {}
    for r in rows:
        key = f"{r['identifier_type']}:{r['identifier']}"
        lookup[key] = r["person_id"]
    return lookup


def _resolve_identity(pid: ParsedIdentity, lookup: dict[str, str]) -> str | None:
    """Resolve a ParsedIdentity to a person_id using the lookup cache."""
    if pid.email:
        person = lookup.get(f"email:{pid.email}")
        if person:
            return person
    if pid.phone:
        person = lookup.get(f"phone:{pid.phone}")
        if person:
            return person
    if pid.display_name:
        person = lookup.get(f"display_name:{pid.display_name}")
        if person:
            return person
    return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Role assignment per source type
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _assign_roles_mail(
    participants: list[str], meta: dict, user_person_id: str | None,
    lookup: dict[str, str], *, recent: bool = False,
) -> list[tuple[str, ParticipantRole]]:
    """Assign roles for a mail event.

    Thread-based linking: only sender per message. Recipients get thread
    membership instead. For recent mail (last 30 days), keep full
    recipient linking for richer context.
    """
    results: list[tuple[str, ParticipantRole]] = []
    is_from_me = meta.get("is_from_me", False)
    identities = parse_mail_participants(participants, meta)

    for i, pid in enumerate(identities):
        if pid.is_noise or pid.is_group:
            continue
        person_id = _resolve_identity(pid, lookup)
        if not person_id:
            continue

        raw_for_this = participants[min(i, len(participants) - 1)] if participants else ""
        is_cc = "; CC:" in raw_for_this and pid.raw != raw_for_this.split("; CC:")[0].strip()

        if person_id == user_person_id:
            # No per-message self edge; user participation via thread membership
            continue
        elif i == 0 and not is_from_me:
            results.append((person_id, ParticipantRole.SENDER))
        elif recent:
            # Recent mail: keep full recipient/cc linking
            if is_cc:
                results.append((person_id, ParticipantRole.CC))
            else:
                results.append((person_id, ParticipantRole.RECIPIENT))
        # Older mail: recipients handled by thread membership

    return results


def _assign_roles_message(
    participants: list[str], meta: dict, source: str,
    user_person_id: str | None, lookup: dict[str, str],
) -> list[tuple[str, ParticipantRole]]:
    """Assign roles for iMessage/WhatsApp/Slack message events.

    For WhatsApp/iMessage: only sender + self per message.
    Thread membership and mention edges are handled by separate passes
    (_link_thread_memberships, _link_group_mentions).

    For Slack: all participants linked (legacy behavior).
    """
    results: list[tuple[str, ParticipantRole]] = []
    is_from_me = meta.get("is_from_me", False)
    thread_mode = source in ("whatsapp", "imessage")

    for i, raw in enumerate(participants):
        pid = parse_participant(raw, source)
        if pid.is_noise or pid.is_group:
            continue
        person_id = _resolve_identity(pid, lookup)
        if not person_id:
            continue

        if person_id == user_person_id:
            # Thread mode: user participation implied by thread membership.
            # No per-message self edge needed.
            if not thread_mode:
                results.append((person_id, ParticipantRole.SELF))
        elif thread_mode:
            # Thread mode: only tag the actual sender per message.
            # Recipients get thread membership instead of per-message edges.
            if not is_from_me and i == 0:
                results.append((person_id, ParticipantRole.SENDER))
        else:
            # Slack: legacy behavior
            if is_from_me:
                results.append((person_id, ParticipantRole.RECIPIENT))
            elif i == 0:
                results.append((person_id, ParticipantRole.SENDER))
            else:
                results.append((person_id, ParticipantRole.RECIPIENT))

    return results


def _assign_roles_calendar(
    participants: list[str], meta: dict, user_person_id: str | None,
    lookup: dict[str, str],
) -> list[tuple[str, ParticipantRole]]:
    """Assign roles for calendar events."""
    results: list[tuple[str, ParticipantRole]] = []
    identities = parse_calendar_participants(participants, meta)

    for pid in identities:
        if pid.is_noise or pid.is_group:
            continue
        person_id = _resolve_identity(pid, lookup)
        if not person_id:
            continue

        if person_id == user_person_id:
            results.append((person_id, ParticipantRole.SELF))
        else:
            org_email = normalize_email(meta.get("organizer_email", ""))
            if org_email and pid.email == org_email:
                results.append((person_id, ParticipantRole.ORGANIZER))
            else:
                results.append((person_id, ParticipantRole.ATTENDEE))

    return results


def _assign_roles_granola(
    participants: list[str], user_person_id: str | None,
    lookup: dict[str, str],
) -> list[tuple[str, ParticipantRole]]:
    """Assign roles for Granola meeting events.

    Participants are email addresses from meeting attendees.
    """
    results: list[tuple[str, ParticipantRole]] = []
    for raw in participants:
        pid = parse_participant(raw, "granola")
        if pid.is_noise or pid.is_group:
            continue
        person_id = _resolve_identity(pid, lookup)
        if not person_id:
            continue
        if person_id == user_person_id:
            results.append((person_id, ParticipantRole.SELF))
        else:
            results.append((person_id, ParticipantRole.ATTENDEE))
    return results


def _assign_roles_call(
    participants: list[str], meta: dict, user_person_id: str | None,
    lookup: dict[str, str],
) -> list[tuple[str, ParticipantRole]]:
    """Assign roles for call events (macOS + WhatsApp)."""
    results: list[tuple[str, ParticipantRole]] = []
    originated = meta.get("originated", False)

    for raw in participants:
        pid = parse_participant(raw, "calls_macos")
        if pid.is_noise or pid.is_group:
            continue
        person_id = _resolve_identity(pid, lookup)
        if not person_id:
            continue

        if person_id == user_person_id:
            results.append((person_id, ParticipantRole.SELF))
        elif originated:
            results.append((person_id, ParticipantRole.RECIPIENT))
        else:
            results.append((person_id, ParticipantRole.SENDER))

    return results


def _assign_roles_contacts(
    meta: dict, lookup: dict[str, str],
) -> list[tuple[str, ParticipantRole]]:
    """Assign roles for contact identity events."""
    results: list[tuple[str, ParticipantRole]] = []
    identities = _parse_contact_event(meta)
    for pid in identities:
        person_id = _resolve_identity(pid, lookup)
        if person_id:
            results.append((person_id, ParticipantRole.IDENTITY))
            break  # One link per contact event is sufficient
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main linker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def link_events_to_persons(
    store: LayeredGraphStore,
    user_person_id: str | None = None,
    since_ts: int = 0,
) -> dict[str, int]:
    """Link events to their resolved persons.

    Args:
        store: Graph store with events and persons already populated
        user_person_id: The person_id for the user (for role assignment)
        since_ts: Only link events with timestamp >= this value (0 = all)

    Returns:
        {edges_created, events_linked, events_unlinked, persons_referenced}
    """
    lookup = _build_lookup(store)
    logger.info("Built person lookup: %d identifiers", len(lookup))

    # Auto-detect user person_id if not provided
    if not user_person_id:
        row = store.conn.execute(
            "SELECT person_id FROM persons WHERE is_user = 1 LIMIT 1"
        ).fetchone()
        if row:
            user_person_id = row["person_id"]

    if since_ts:
        rows = store.conn.execute(
            "SELECT id, source, participants, metadata, timestamp FROM events WHERE timestamp >= ?",
            (since_ts,),
        ).fetchall()
        logger.info("Linking %d events (since_ts=%d)", len(rows), since_ts)
    else:
        rows = store.conn.execute(
            "SELECT id, source, participants, metadata, timestamp FROM events"
        ).fetchall()

    edges_created = 0
    events_linked = 0
    events_unlinked = 0
    persons_seen: set[str] = set()

    import time as _time
    recency_cutoff = int(_time.time()) - 30 * 86400  # 30 days ago

    for row in rows:
        event_id = row["id"]
        source = row["source"]
        participants = json.loads(row["participants"] or "[]")
        meta = json.loads(row["metadata"] or "{}")

        if source == "mail":
            recent = row["timestamp"] >= recency_cutoff
            roles = _assign_roles_mail(
                participants, meta, user_person_id, lookup, recent=recent,
            )
        elif source in ("imessage", "whatsapp", "slack"):
            roles = _assign_roles_message(participants, meta, source, user_person_id, lookup)
        elif source == "calendar":
            roles = _assign_roles_calendar(participants, meta, user_person_id, lookup)
        elif source == "contacts":
            roles = _assign_roles_contacts(meta, lookup)
        elif source == "granola":
            roles = _assign_roles_granola(participants, user_person_id, lookup)
        elif source in ("calls_macos", "calls_whatsapp"):
            roles = _assign_roles_call(participants, meta, user_person_id, lookup)
        else:
            roles = []

        # Implicit user participation: the user is always involved in their
        # own communication events even if not listed in participants.
        # For WhatsApp/iMessage/mail, user participation is captured by
        # thread membership — no per-message self edge needed.
        if user_person_id and source not in ("whatsapp", "imessage", "mail") and source in (
            "slack", "calendar", "granola",
            "calls_macos", "calls_whatsapp",
        ):
            user_already_tagged = any(pid == user_person_id for pid, _ in roles)
            if not user_already_tagged:
                roles.append((user_person_id, ParticipantRole.SELF))

        if roles:
            events_linked += 1
            seen_pairs: set[tuple[str, str]] = set()
            for person_id, role in roles:
                pair = (person_id, role.value)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                store.link_event_person(event_id, person_id, role.value)
                edges_created += 1
                persons_seen.add(person_id)
        else:
            events_unlinked += 1

    # Thread membership pass: one "member" edge per (person, thread)
    membership_edges = _link_thread_memberships(store, lookup, user_person_id)
    edges_created += membership_edges

    # Mention pass: scan group messages for name mentions
    mention_edges = _link_group_mentions(store, lookup, user_person_id)
    edges_created += mention_edges

    # Materialize person_events table (thread-expanded participation)
    person_events_count = rebuild_person_events(store)

    return {
        "edges_created": edges_created,
        "events_linked": events_linked,
        "events_unlinked": events_unlinked,
        "persons_referenced": len(persons_seen),
        "membership_edges": membership_edges,
        "mention_edges": mention_edges,
        "person_events": person_events_count,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thread membership + mention linking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MAIL_THREAD_MEMBER_CAP = 10  # Don't create member edges for mass mail


def _link_thread_memberships(
    store: LayeredGraphStore,
    lookup: dict[str, str],
    user_person_id: str | None,
) -> int:
    """Create one 'member' edge per (person, thread).

    Replaces per-message recipient edges with per-thread membership.
    Anchors to the earliest event in each thread.

    Thread identification:
    - WhatsApp groups: metadata.group_name
    - WhatsApp 1:1 / orphaned groups: sorted participant person_ids
    - iMessage: metadata.thread_id
    - Mail: metadata.thread_id

    For mail threads with >10 unique people (mass mail, newsletters),
    member edges are skipped — recipients are still 2-hop reachable
    via the sender edge on the thread anchor.
    """
    edges_created = 0

    # threads: thread_key -> {anchor_event_id, person_ids}
    threads: dict[str, dict] = {}

    rows = store.conn.execute(
        """SELECT id, source, participants, metadata
           FROM events
           WHERE source IN ('whatsapp', 'imessage', 'mail')
           ORDER BY timestamp ASC"""
    ).fetchall()

    for row in rows:
        source = row["source"]
        participants = json.loads(row["participants"] or "[]")
        meta = json.loads(row["metadata"] or "{}")

        if source == "whatsapp":
            group_name = meta.get("group_name", "")
            if group_name:
                thread_key = f"wa_group:{group_name}"
            else:
                pids = _resolve_participant_set(participants, source, lookup)
                if not pids:
                    continue
                thread_key = f"wa_dm:{'|'.join(sorted(pids))}"
        elif source == "imessage":
            thread_id = meta.get("thread_id", "")
            if thread_id:
                thread_key = f"im:{thread_id}"
            else:
                pids = _resolve_participant_set(participants, source, lookup)
                if not pids:
                    continue
                thread_key = f"im_dm:{'|'.join(sorted(pids))}"
        elif source == "mail":
            thread_id = meta.get("thread_id", "")
            if thread_id:
                thread_key = f"mail:{thread_id}"
            else:
                continue  # No thread_id = can't group
        else:
            continue

        # First event in thread = anchor (rows ordered by timestamp ASC)
        if thread_key not in threads:
            threads[thread_key] = {"anchor": row["id"], "persons": set()}

        # Collect all participant person_ids for this thread
        if source == "mail":
            identities = parse_mail_participants(participants, meta)
            for pid in identities:
                if pid.is_noise or pid.is_group:
                    continue
                person_id = _resolve_identity(pid, lookup)
                if person_id:
                    threads[thread_key]["persons"].add(person_id)
        else:
            for raw in participants:
                pid = parse_participant(raw, source)
                if pid.is_noise or pid.is_group:
                    continue
                person_id = _resolve_identity(pid, lookup)
                if person_id:
                    threads[thread_key]["persons"].add(person_id)

    # Create member edges (skip mass mail threads)
    skipped_mass = 0
    for thread_key, data in threads.items():
        if thread_key.startswith("mail:") and len(data["persons"]) > _MAIL_THREAD_MEMBER_CAP:
            skipped_mass += 1
            continue
        for person_id in data["persons"]:
            store.link_event_person(
                data["anchor"], person_id, ParticipantRole.MEMBER.value,
            )
            edges_created += 1

    logger.info(
        "Thread membership: %d threads, %d member edges (%d mass mail threads skipped)",
        len(threads), edges_created, skipped_mass,
    )
    return edges_created


def _resolve_participant_set(
    participants: list[str], source: str, lookup: dict[str, str],
) -> set[str]:
    """Resolve a participant list to a set of person_ids."""
    pids: set[str] = set()
    for raw in participants:
        pid = parse_participant(raw, source)
        if pid.is_noise or pid.is_group:
            continue
        person_id = _resolve_identity(pid, lookup)
        if person_id:
            pids.add(person_id)
    return pids


def _link_group_mentions(
    store: LayeredGraphStore,
    lookup: dict[str, str],
    user_person_id: str | None,
) -> int:
    """Scan group message content for name mentions of thread members.

    For each group message, checks if the content mentions any participant
    by first name or last name. Creates 'mentioned' edges.
    Only applies to group messages (>2 participants or has group_name).
    Skips the message sender (self-mentions aren't useful).
    """
    import re

    edges_created = 0

    # Build person name index: person_id -> set of searchable name tokens
    name_index: dict[str, set[str]] = {}
    person_rows = store.conn.execute(
        "SELECT person_id, canonical_name FROM persons WHERE canonical_name != ''"
    ).fetchall()
    for pr in person_rows:
        name = pr["canonical_name"]
        if not name or name.startswith("person:") or name.startswith("+"):
            continue
        tokens: set[str] = set()
        for part in name.split():
            cleaned = part.strip("(),.'\"").lower()
            if len(cleaned) >= 3:  # Skip short tokens to avoid false positives
                tokens.add(cleaned)
        if tokens:
            name_index[pr["person_id"]] = tokens

    # Process group messages
    rows = store.conn.execute(
        """SELECT id, source, participants, metadata, raw_content
           FROM events
           WHERE source IN ('whatsapp', 'imessage')
             AND event_type = 'message'
             AND raw_content != ''
             AND (json_array_length(participants) > 2
                  OR (json_extract(metadata, '$.group_name') IS NOT NULL
                      AND json_extract(metadata, '$.group_name') != ''))"""
    ).fetchall()

    for row in rows:
        content = row["raw_content"]
        if not content:
            continue

        # Extract word tokens from content
        content_words = set(re.findall(r'\b\w+\b', content.lower()))

        # Identify the sender to skip self-mentions
        participants = json.loads(row["participants"] or "[]")
        meta = json.loads(row["metadata"] or "{}")
        is_from_me = meta.get("is_from_me", False)
        sender_pid = None
        if is_from_me:
            sender_pid = user_person_id
        elif participants:
            first_parsed = parse_participant(participants[0], row["source"])
            if not first_parsed.is_noise and not first_parsed.is_group:
                sender_pid = _resolve_identity(first_parsed, lookup)

        # Resolve thread members
        member_pids = _resolve_participant_set(
            participants, row["source"], lookup,
        )

        # Check for mentions
        for person_id in member_pids:
            if person_id == sender_pid or person_id == user_person_id:
                continue
            names = name_index.get(person_id, set())
            if names and names & content_words:
                store.link_event_person(
                    row["id"], person_id, ParticipantRole.MENTIONED.value,
                )
                edges_created += 1

    logger.info("Group mentions: %d mention edges created", edges_created)
    return edges_created


_THREAD_SOURCES = ("whatsapp", "imessage", "mail")


def rebuild_person_events(store: LayeredGraphStore) -> int:
    """Rebuild the materialized person_events + thread_members tables.

    person_events maps (person_id, event_id) for all participation:
      - Direct edges from event_persons (sender, recipient, etc.)
      - Expanded thread membership (member of thread → all events in thread)

    Returns the total number of participation edges.
    """
    t0 = __import__("time").time()

    # 1. Rebuild thread_members from member edges
    store.conn.execute("DELETE FROM thread_members")
    placeholders = ",".join("?" for _ in _THREAD_SOURCES)
    store.conn.execute(
        f"""INSERT OR IGNORE INTO thread_members (source, thread_id, person_id)
            SELECT e.source, json_extract(e.metadata, '$.thread_id'), ep.person_id
            FROM event_persons ep
            JOIN events e ON ep.event_id = e.id
            WHERE ep.role = 'member'
              AND e.source IN ({placeholders})
              AND json_extract(e.metadata, '$.thread_id') IS NOT NULL
              AND json_extract(e.metadata, '$.thread_id') != ''""",
        _THREAD_SOURCES,
    )
    tm_count = store.conn.execute("SELECT COUNT(*) FROM thread_members").fetchone()[0]

    # 2. Rebuild person_events
    store.conn.execute("DELETE FROM person_events")

    # 2a. Direct edges (sender, recipient, cc, attendee, mentioned, self)
    store.conn.execute("""
        INSERT OR IGNORE INTO person_events (person_id, event_id)
        SELECT person_id, event_id FROM event_persons
        WHERE role NOT IN ('identity', 'member')
    """)

    # 2b. Expand thread membership via thread_members.
    # Uses the expression index on events(source, json_extract(metadata, '$.thread_id'))
    store.conn.execute("""
        INSERT OR IGNORE INTO person_events (person_id, event_id)
        SELECT tm.person_id, e.id
        FROM thread_members tm
        JOIN events e ON e.source = tm.source
          AND json_extract(e.metadata, '$.thread_id') = tm.thread_id
    """)

    store.conn.commit()
    pe_count = store.conn.execute("SELECT COUNT(*) FROM person_events").fetchone()[0]
    elapsed = __import__("time").time() - t0
    logger.info(
        "Person events: %d edges (%d thread_members) in %.1fs",
        pe_count, tm_count, elapsed,
    )
    return pe_count


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Query helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_person_events(
    store: LayeredGraphStore, person_id: str,
    role: str | None = None, source: str | None = None,
    limit: int = 100,
) -> list[dict]:
    """Get all events involving a person, optionally filtered."""
    conditions = ["ep.person_id = ?"]
    params: list = [person_id]

    if role:
        conditions.append("ep.role = ?")
        params.append(role)
    if source:
        conditions.append("e.source = ?")
        params.append(source)
    params.append(limit)

    rows = store.conn.execute(
        f"""SELECT e.id, e.source, e.event_type, e.timestamp, e.metadata, ep.role
            FROM events e
            JOIN event_persons ep ON e.id = ep.event_id
            WHERE {' AND '.join(conditions)}
            ORDER BY e.timestamp DESC
            LIMIT ?""",
        params,
    ).fetchall()

    return [
        {
            "event_id": r["id"],
            "source": r["source"],
            "event_type": r["event_type"],
            "timestamp": r["timestamp"],
            "role": r["role"],
            "metadata": json.loads(r["metadata"] or "{}"),
        }
        for r in rows
    ]


def get_event_persons(store: LayeredGraphStore, event_id: str) -> list[dict]:
    """Get all persons involved in an event with their roles."""
    rows = store.conn.execute(
        """SELECT p.person_id, p.canonical_name, p.is_user, ep.role
           FROM persons p
           JOIN event_persons ep ON p.person_id = ep.person_id
           WHERE ep.event_id = ?
           ORDER BY ep.role""",
        (event_id,),
    ).fetchall()

    return [
        {
            "person_id": r["person_id"],
            "name": r["canonical_name"],
            "is_user": bool(r["is_user"]),
            "role": r["role"],
        }
        for r in rows
    ]


def get_communication_partners(
    store: LayeredGraphStore, person_id: str,
) -> list[dict]:
    """Get all persons who share events with a given person."""
    rows = store.conn.execute(
        """SELECT p.person_id, p.canonical_name, p.is_user,
                  COUNT(DISTINCT ep2.event_id) as shared_events,
                  GROUP_CONCAT(DISTINCT e.source) as sources,
                  GROUP_CONCAT(DISTINCT ep2.role) as roles
           FROM event_persons ep1
           JOIN event_persons ep2 ON ep1.event_id = ep2.event_id
              AND ep1.person_id != ep2.person_id
           JOIN persons p ON ep2.person_id = p.person_id
           JOIN events e ON ep1.event_id = e.id
           WHERE ep1.person_id = ?
           GROUP BY p.person_id
           ORDER BY shared_events DESC""",
        (person_id,),
    ).fetchall()

    return [
        {
            "person_id": r["person_id"],
            "name": r["canonical_name"],
            "is_user": bool(r["is_user"]),
            "shared_events": r["shared_events"],
            "sources": r["sources"].split(",") if r["sources"] else [],
            "roles": r["roles"].split(",") if r["roles"] else [],
        }
        for r in rows
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Cross-source participant inference (calls × meetings)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class InferredParticipant:
    """A meeting participant inferred from temporal overlap with a call."""
    event_id: str        # The meeting/Granola event
    person_id: str       # The inferred participant
    person_name: str
    call_event_id: str   # The call that overlaps
    call_source: str     # calls_macos or calls_whatsapp
    overlap_seconds: float
    call_duration: float
    event_source: str    # granola, calendar, etc.
    confidence: float


def infer_meeting_participants_from_calls(
    store: LayeredGraphStore,
) -> list[InferredParticipant]:
    """Infer meeting participants from temporal overlap between calls and events.

    For each call event, find concurrent meetings/calendar events. If the call's
    non-user participant isn't already linked to the meeting, create an inference.

    This is a deterministic pass: pure SQL + timestamp math, no LLM.
    """
    window = CALL_MEETING_OVERLAP_WINDOW_SECONDS

    # Get all call events with their duration and linked persons
    call_rows = store.conn.execute(
        """SELECT e.id, e.source, e.timestamp, e.metadata,
                  ep.person_id, p.canonical_name
           FROM events e
           JOIN event_persons ep ON e.id = ep.event_id
           JOIN persons p ON ep.person_id = p.person_id
           WHERE e.event_type = 'call'
             AND ep.role != 'self'
             AND p.is_user = 0"""
    ).fetchall()

    if not call_rows:
        return []

    # Get meetings and timed calendar events (not all-day events, not emails)
    # Include raw_content for Granola so we can estimate duration from word count
    meeting_rows = store.conn.execute(
        """SELECT id, source, event_type, timestamp, metadata, raw_content
           FROM events
           WHERE event_type IN ('meeting', 'calendar_event')
             AND source IN ('granola', 'calendar')"""
    ).fetchall()

    if not meeting_rows:
        return []

    # Pre-build calendar event durations for Granola fallback.
    # When a Granola meeting has no explicit duration, we use
    # max(word_count_estimate, overlapping_calendar_duration).
    calendar_windows: list[tuple[int, int, int]] = []  # (start, end, duration)
    for m in meeting_rows:
        if m["source"] != "calendar":
            continue
        m_meta = json.loads(m["metadata"] or "{}")
        if m_meta.get("is_all_day"):
            continue
        end_str = m_meta.get("end", "")
        if end_str:
            from datetime import datetime
            try:
                end_dt = datetime.fromisoformat(end_str)
                dur = int(end_dt.timestamp()) - m["timestamp"]
                if dur > 0:
                    calendar_windows.append((m["timestamp"], m["timestamp"] + dur, dur))
            except (ValueError, TypeError):
                pass

    # Build existing event_persons edges for fast lookup
    existing_links: set[tuple[str, str]] = set()
    link_rows = store.conn.execute(
        "SELECT event_id, person_id FROM event_persons"
    ).fetchall()
    for lr in link_rows:
        existing_links.add((lr["event_id"], lr["person_id"]))

    # Only infer for meetings that already have other attendees.
    # Personal calendar events (reminders, errands, holidays) have no
    # attendees and shouldn't get participant inferences from calls.
    # Granola meetings always qualify (they're recordings of real meetings).
    meetings_with_attendees: set[str] = set()
    for m in meeting_rows:
        if m["source"] == "granola":
            meetings_with_attendees.add(m["id"])
        elif m["source"] == "calendar":
            has_others = store.conn.execute(
                """SELECT 1 FROM event_persons
                   WHERE event_id = ? AND role NOT IN ('self', 'identity')
                   LIMIT 1""",
                (m["id"],),
            ).fetchone()
            if has_others:
                meetings_with_attendees.add(m["id"])

    inferences: list[InferredParticipant] = []

    for call in call_rows:
        call_meta = json.loads(call["metadata"] or "{}")
        call_duration = call_meta.get("duration_seconds", 0)
        if call_duration < CALL_MIN_DURATION_SECONDS:
            continue

        call_ts = call["timestamp"]
        call_end = call_ts + int(call_duration)
        call_person_id = call["person_id"]
        call_person_name = call["canonical_name"]

        for meeting in meeting_rows:
            # Only infer for real meetings with other people
            if meeting["id"] not in meetings_with_attendees:
                continue

            meeting_ts = meeting["timestamp"]
            meeting_meta = json.loads(meeting["metadata"] or "{}")

            # Skip all-day events: they span 24h so every call
            # would match — temporal correlation is meaningless
            if meeting_meta.get("is_all_day"):
                continue

            meeting_duration = meeting_meta.get("duration_seconds", 0)
            if not meeting_duration:
                # Calendar adapter stores end as ISO string
                end_str = meeting_meta.get("end", "")
                if end_str:
                    from datetime import datetime, timezone
                    try:
                        end_dt = datetime.fromisoformat(end_str)
                        meeting_duration = int(end_dt.timestamp()) - meeting_ts
                    except (ValueError, TypeError):
                        pass
            if not meeting_duration and meeting["source"] == "granola":
                # Infer lower-bound duration from transcript word count.
                # Someone spoke all those words at ~150 WPM, so the meeting
                # was at least word_count/150 minutes long.
                word_estimate = 0
                transcript = meeting["raw_content"] or ""
                if transcript:
                    word_count = len(transcript.split())
                    word_estimate = int((word_count / SPEECH_RATE_WPM) * 60)
                # If linked to a calendar event, use max of both estimates
                cal_duration = 0
                for cal_start, cal_end, cal_dur in calendar_windows:
                    if abs(meeting_ts - cal_start) < CALL_MEETING_OVERLAP_WINDOW_SECONDS:
                        cal_duration = cal_dur
                        break
                meeting_duration = max(word_estimate, cal_duration)
            # Skip events with unknown duration
            if meeting_duration <= 0:
                continue

            meeting_end = meeting_ts + int(meeting_duration)

            # Check temporal overlap with tolerance window
            overlap_start = max(call_ts, meeting_ts - window)
            overlap_end = min(call_end, meeting_end + window)
            overlap = max(0, overlap_end - overlap_start)

            if overlap <= 0:
                continue

            # Already linked?
            if (meeting["id"], call_person_id) in existing_links:
                continue

            # Compute confidence via Admiralty framework
            from loom.confidence import RELIABILITY, SOURCE_RELIABILITY

            fraction = overlap / call_duration if call_duration > 0 else 0
            if fraction >= OVERLAP_FRACTION_CONTAINED:
                credibility = 0.85
            elif fraction >= OVERLAP_FRACTION_PARTIAL:
                credibility = 0.70
            else:
                credibility = 0.50

            event_source = meeting["source"]
            rel_grade = SOURCE_RELIABILITY.get(event_source, "B")
            reliability = RELIABILITY[rel_grade]
            confidence = round(reliability * credibility, 4)

            # Insert the event_persons edge
            store.link_event_person(
                meeting["id"], call_person_id, ParticipantRole.ATTENDEE.value,
            )
            existing_links.add((meeting["id"], call_person_id))

            inferences.append(InferredParticipant(
                event_id=meeting["id"],
                person_id=call_person_id,
                person_name=call_person_name,
                call_event_id=call["id"],
                call_source=call["source"],
                overlap_seconds=overlap,
                call_duration=call_duration,
                event_source=event_source,
                confidence=confidence,
            ))

    if inferences:
        logger.info(
            "Inferred %d meeting participants from call overlap",
            len(inferences),
        )

    return inferences
