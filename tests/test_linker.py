"""Tests for alteris.linker: Event-Person linking with role assignment.

Tests cover:
  - ParticipantRole enum
  - _build_lookup (identifier -> person_id cache)
  - _resolve_identity from lookup
  - Role assignment per source type (mail, message, calendar, contacts)
  - Implicit user participation
  - link_events_to_persons full pipeline
  - Auto-detection of user person_id
  - Dedup of same person+role pairs
  - get_person_events query helper
  - get_event_persons query helper
  - get_communication_partners query helper
  - Edge cases (empty store, no persons, unresolvable participants)
"""

import json
import time

import pytest

from alteris.linker import (
    ParticipantRole,
    _assign_roles_calendar,
    _assign_roles_contacts,
    _assign_roles_mail,
    _assign_roles_message,
    _build_lookup,
    _resolve_identity,
    get_communication_partners,
    get_event_persons,
    get_person_events,
    link_events_to_persons,
)
from alteris.models import Event
from alteris.resolver import ParsedIdentity
from alteris.store import LayeredGraphStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ParticipantRole
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestParticipantRole:
    def test_values(self):
        assert ParticipantRole.SENDER.value == "sender"
        assert ParticipantRole.RECIPIENT.value == "recipient"
        assert ParticipantRole.CC.value == "cc"
        assert ParticipantRole.ATTENDEE.value == "attendee"
        assert ParticipantRole.ORGANIZER.value == "organizer"
        assert ParticipantRole.SELF.value == "self"
        assert ParticipantRole.IDENTITY.value == "identity"
        assert ParticipantRole.MEMBER.value == "member"

    def test_is_string_enum(self):
        assert isinstance(ParticipantRole.SENDER, str)
        assert ParticipantRole.SENDER == "sender"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Lookup and resolution
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLookup:
    def _seed_lookup(self, store):
        store.put_person("person_alice", canonical_name="Alice")
        store.add_person_identifier("person_alice", "email", "alice@test.com")
        store.add_person_identifier("person_alice", "phone", "+1234567890")
        store.put_person("person_bob", canonical_name="Bob")
        store.add_person_identifier("person_bob", "email", "bob@test.com")

    def test_build_lookup(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_lookup(store)
        lookup = _build_lookup(store)
        assert lookup["email:alice@test.com"] == "person_alice"
        assert lookup["phone:+1234567890"] == "person_alice"
        assert lookup["email:bob@test.com"] == "person_bob"

    def test_empty_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        lookup = _build_lookup(store)
        assert lookup == {}

    def test_resolve_by_email(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_lookup(store)
        lookup = _build_lookup(store)
        pid = ParsedIdentity(raw="alice@test.com", email="alice@test.com")
        assert _resolve_identity(pid, lookup) == "person_alice"

    def test_resolve_by_phone(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_lookup(store)
        lookup = _build_lookup(store)
        pid = ParsedIdentity(raw="+1234567890", phone="+1234567890")
        assert _resolve_identity(pid, lookup) == "person_alice"

    def test_resolve_not_found(self):
        store = LayeredGraphStore(db_path=":memory:")
        lookup = _build_lookup(store)
        pid = ParsedIdentity(raw="unknown@test.com", email="unknown@test.com")
        assert _resolve_identity(pid, lookup) is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Role assignment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRoleAssignment:
    def _make_lookup(self):
        return {
            "email:alice@test.com": "person_alice",
            "email:bob@test.com": "person_bob",
            "email:user@test.com": "person_user",
            "phone:+15550100002": "person_alice",
        }

    def test_mail_sender_inbound(self):
        """Inbound mail: sender gets per-message edge, user skipped (thread membership)."""
        lookup = self._make_lookup()
        roles = _assign_roles_mail(
            ["alice@test.com", "user@test.com"],
            {"is_from_me": False},
            "person_user", lookup,
        )
        role_dict = {pid: role for pid, role in roles}
        assert role_dict.get("person_alice") == ParticipantRole.SENDER
        # User participation is via thread membership, not per-message self edge
        assert "person_user" not in role_dict

    def test_mail_sender_inbound_recent(self):
        """Recent inbound mail: sender edge only (user still via thread membership)."""
        lookup = self._make_lookup()
        roles = _assign_roles_mail(
            ["alice@test.com", "user@test.com"],
            {"is_from_me": False},
            "person_user", lookup, recent=True,
        )
        role_dict = {pid: role for pid, role in roles}
        assert role_dict.get("person_alice") == ParticipantRole.SENDER
        assert "person_user" not in role_dict

    def test_mail_sender_outbound(self):
        """Outbound old mail: no per-message edges (recipients via thread membership)."""
        lookup = self._make_lookup()
        roles = _assign_roles_mail(
            ["user@test.com", "alice@test.com"],
            {"is_from_me": True},
            "person_user", lookup,
        )
        role_dict = {pid: role for pid, role in roles}
        # Old outbound mail: user skipped, recipients handled by thread membership
        assert "person_user" not in role_dict
        assert "person_alice" not in role_dict

    def test_mail_sender_outbound_recent(self):
        """Recent outbound mail: recipients get per-message edges."""
        lookup = self._make_lookup()
        roles = _assign_roles_mail(
            ["user@test.com", "alice@test.com"],
            {"is_from_me": True},
            "person_user", lookup, recent=True,
        )
        role_dict = {pid: role for pid, role in roles}
        assert "person_user" not in role_dict
        assert role_dict.get("person_alice") == ParticipantRole.RECIPIENT

    def test_message_inbound(self):
        lookup = self._make_lookup()
        roles = _assign_roles_message(
            ["+15550100002"], {"is_from_me": False},
            "imessage", "person_user", lookup,
        )
        role_dict = {pid: role for pid, role in roles}
        assert role_dict.get("person_alice") == ParticipantRole.SENDER

    def test_message_outbound(self):
        """WhatsApp/iMessage outbound: no per-message edges (thread membership)."""
        lookup = self._make_lookup()
        roles = _assign_roles_message(
            ["+15550100002"], {"is_from_me": True},
            "whatsapp", "person_user", lookup,
        )
        role_dict = {pid: role for pid, role in roles}
        # Thread mode: outbound messages don't create per-message recipient edges
        assert "person_alice" not in role_dict

    def test_message_outbound_slack(self):
        """Slack keeps legacy per-message recipient edges."""
        lookup = self._make_lookup()
        roles = _assign_roles_message(
            ["+15550100002"], {"is_from_me": True},
            "slack", "person_user", lookup,
        )
        role_dict = {pid: role for pid, role in roles}
        assert role_dict.get("person_alice") == ParticipantRole.RECIPIENT

    def test_calendar_attendees(self):
        lookup = self._make_lookup()
        roles = _assign_roles_calendar(
            [], {"attendees": [{"email": "alice@test.com", "name": "Alice"}]},
            "person_user", lookup,
        )
        role_dict = {pid: role for pid, role in roles}
        assert role_dict.get("person_alice") == ParticipantRole.ATTENDEE

    def test_calendar_organizer(self):
        lookup = self._make_lookup()
        roles = _assign_roles_calendar(
            [],
            {
                "attendees": [{"email": "alice@test.com", "name": "Alice"}],
                "organizer_email": "alice@test.com",
            },
            "person_user", lookup,
        )
        role_dict = {pid: role for pid, role in roles}
        assert role_dict.get("person_alice") == ParticipantRole.ORGANIZER

    def test_contacts_identity(self):
        lookup = self._make_lookup()
        roles = _assign_roles_contacts(
            {"name": "Alice", "emails": ["alice@test.com"], "phones": []},
            lookup,
        )
        assert len(roles) == 1
        assert roles[0][1] == ParticipantRole.IDENTITY


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# link_events_to_persons
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLinkEventsToPersons:
    def _seed_store(self, store):
        now = int(time.time())

        store.put_person("person_user", canonical_name="User", is_user=True)
        store.add_person_identifier("person_user", "email", "user@test.com")

        store.put_person("person_alice", canonical_name="Alice")
        store.add_person_identifier("person_alice", "email", "alice@test.com")

        store.put_person("person_bob", canonical_name="Bob")
        store.add_person_identifier("person_bob", "email", "bob@test.com")

        eid1 = Event.make_id("mail", "email_1")
        store.put_event(Event(
            id=eid1, source="mail", source_id="email_1",
            event_type="email", timestamp=now,
            participants=["alice@test.com", "user@test.com"],
            raw_content="Hello from Alice",
            metadata={"subject": "Test", "is_from_me": False,
                       "thread_id": "thread_1"},
        ))

        eid2 = Event.make_id("mail", "email_2")
        store.put_event(Event(
            id=eid2, source="mail", source_id="email_2",
            event_type="email", timestamp=now - 3600,
            participants=["user@test.com", "bob@test.com"],
            raw_content="Hello to Bob",
            metadata={"subject": "Test 2", "is_from_me": True,
                       "thread_id": "thread_2"},
        ))

        return eid1, eid2

    @pytest.fixture
    def linker_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_store(store)
        return store

    def test_links_events(self, linker_store):
        result = link_events_to_persons(linker_store)
        assert result["edges_created"] > 0
        assert result["events_linked"] > 0

    def test_auto_detects_user(self, linker_store):
        result = link_events_to_persons(linker_store)
        assert result["persons_referenced"] > 0

    def test_implicit_user_member(self, linker_store):
        """User gets 'member' role via thread membership for mail events."""
        result = link_events_to_persons(linker_store)
        assert result["membership_edges"] > 0
        # User should have member edge on the thread anchor event
        eid1 = Event.make_id("mail", "email_1")
        persons = linker_store.get_persons_for_event(eid1)
        user_roles = [(pid, role) for pid, role in persons if pid == "person_user"]
        assert any(role == "member" for _, role in user_roles)

    def test_empty_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        result = link_events_to_persons(store)
        assert result["edges_created"] == 0
        assert result["events_linked"] == 0

    def test_idempotent(self, linker_store):
        r1 = link_events_to_persons(linker_store)
        r2 = link_events_to_persons(linker_store)
        # Second run should create same edges (INSERT OR IGNORE)
        assert r2["events_linked"] == r1["events_linked"]

    def test_unresolvable_participants(self):
        store = LayeredGraphStore(db_path=":memory:")
        now = int(time.time())
        eid = Event.make_id("mail", "unknown")
        store.put_event(Event(
            id=eid, source="mail", source_id="unknown",
            event_type="email", timestamp=now,
            participants=["unknown@nowhere.com"],
            raw_content="Mystery email",
        ))
        result = link_events_to_persons(store)
        assert result["events_unlinked"] >= 1

    def test_explicit_user_id(self, linker_store):
        result = link_events_to_persons(
            linker_store, user_person_id="person_user",
        )
        assert result["edges_created"] > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Query helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestQueryHelpers:
    @pytest.fixture
    def linked_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        now = int(time.time())

        store.put_person("person_user", canonical_name="User", is_user=True)
        store.add_person_identifier("person_user", "email", "user@test.com")
        store.put_person("person_alice", canonical_name="Alice")
        store.add_person_identifier("person_alice", "email", "alice@test.com")

        eid = Event.make_id("mail", "email_q")
        store.put_event(Event(
            id=eid, source="mail", source_id="email_q",
            event_type="email", timestamp=now,
            participants=["alice@test.com", "user@test.com"],
            raw_content="Test",
            metadata={"subject": "Query test", "is_from_me": False,
                       "thread_id": "thread_q"},
        ))
        link_events_to_persons(store)
        return store, eid

    def test_get_person_events(self, linked_store):
        store, _ = linked_store
        events = get_person_events(store, "person_alice")
        assert len(events) >= 1
        assert events[0]["source"] == "mail"

    def test_get_person_events_with_role_filter(self, linked_store):
        """User gets 'member' role via thread membership."""
        store, _ = linked_store
        events = get_person_events(store, "person_user", role="member")
        assert len(events) >= 1

    def test_get_event_persons(self, linked_store):
        store, eid = linked_store
        persons = get_event_persons(store, eid)
        assert len(persons) >= 1
        names = [p["name"] for p in persons]
        assert "Alice" in names or "User" in names

    def test_get_communication_partners(self, linked_store):
        """User and Alice share events via thread membership edges."""
        store, _ = linked_store
        partners = get_communication_partners(store, "person_user")
        assert len(partners) >= 1
        assert partners[0]["shared_events"] >= 1

    def test_get_person_events_empty(self):
        store = LayeredGraphStore(db_path=":memory:")
        events = get_person_events(store, "nonexistent")
        assert events == []

    def test_get_event_persons_empty(self):
        store = LayeredGraphStore(db_path=":memory:")
        persons = get_event_persons(store, "nonexistent")
        assert persons == []
