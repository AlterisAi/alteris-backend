"""Tests for alteris.resolver: Union-find person resolution.

Tests cover:
  - Phone normalization (various formats)
  - Email normalization
  - parse_participant for all source types
  - parse_mail_participants with CC notation
  - parse_calendar_participants with attendees metadata
  - WhatsApp group detection
  - Slack channel detection
  - Calendar noise filtering
  - Granola noise filtering
  - UnionFind operations (find, union, groups, path compression)
  - ResolvedPerson construction
  - resolve_persons full pipeline
  - Contact bridges (phone <-> email merging)
  - User identity merging
  - persist_persons
  - Idempotency
  - Edge cases (empty store, no user config)
"""

import json
import time

import pytest

from alteris.models import Event
from alteris.resolver import (
    ParsedIdentity,
    ResolvedPerson,
    UnionFind,
    normalize_email,
    normalize_phone,
    parse_calendar_participants,
    parse_mail_participants,
    parse_participant,
    persist_persons,
    resolve_persons,
)
from alteris.store import LayeredGraphStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phone normalization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestNormalizePhone:
    def test_already_normalized(self):
        assert normalize_phone("+19196274709") == "+19196274709"

    def test_strip_plus(self):
        assert normalize_phone("15550100001") == "+15550100001"

    def test_ten_digit_us(self):
        assert normalize_phone("5550100002") == "+15550100002"

    def test_parenthetical_format(self):
        assert normalize_phone("(612) 207-2839") == "+16122072839"

    def test_international_with_spaces(self):
        assert normalize_phone("+91 98445 67566") == "+919844567566"

    def test_empty_returns_empty(self):
        assert normalize_phone("") == ""

    def test_no_digits(self):
        assert normalize_phone("abc") == ""

    def test_short_number(self):
        # Very short numbers still get processed but won't be useful
        result = normalize_phone("12")
        assert result.startswith("+")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Email normalization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestNormalizeEmail:
    def test_lowercase(self):
        assert normalize_email("User@Example.COM") == "user@example.com"

    def test_strip_whitespace(self):
        assert normalize_email("  user@test.com  ") == "user@test.com"

    def test_already_normalized(self):
        assert normalize_email("user@test.com") == "user@test.com"

    def test_gmail_dot_removal(self):
        assert normalize_email("foo.bar@gmail.com") == "foobar@gmail.com"

    def test_gmail_multiple_dots(self):
        assert normalize_email("f.o.o.b.a.r@gmail.com") == "foobar@gmail.com"

    def test_gmail_case_insensitive_domain(self):
        assert normalize_email("Foo.Bar@Gmail.COM") == "foobar@gmail.com"

    def test_googlemail_dot_removal(self):
        assert normalize_email("foo.bar@googlemail.com") == "foobar@googlemail.com"

    def test_non_gmail_dots_preserved(self):
        assert normalize_email("foo.bar@outlook.com") == "foo.bar@outlook.com"

    def test_non_gmail_dots_preserved_corporate(self):
        assert normalize_email("first.last@company.com") == "first.last@company.com"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# parse_participant
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestParseParticipant:
    def test_name_angle_email(self):
        p = parse_participant("Sam Park <sam@example.com>", "mail")
        assert p.display_name == "Sam Park"
        assert p.email == "sam@example.com"
        assert not p.is_noise

    def test_bare_email(self):
        p = parse_participant("user@test.com", "mail")
        assert p.email == "user@test.com"
        assert p.display_name == ""

    def test_name_angle_phone(self):
        p = parse_participant("Sam <+919844567566>", "whatsapp")
        assert p.display_name == "Sam"
        assert p.phone == "+919844567566"

    def test_bare_phone(self):
        p = parse_participant("+15550100002", "imessage")
        assert p.phone == "+15550100002"

    def test_slack_channel(self):
        p = parse_participant("#general", "slack")
        assert p.is_group is True
        assert p.display_name == "#general"

    def test_calendar_noise(self):
        p = parse_participant("US Holidays", "calendar")
        assert p.is_noise is True

    def test_granola_noise(self):
        p = parse_participant("meeting", "granola")
        assert p.is_noise is True

    def test_empty_string(self):
        p = parse_participant("", "mail")
        assert p.is_noise is True

    def test_display_name_only(self):
        p = parse_participant("John Smith", "slack")
        assert p.display_name == "John Smith"
        assert not p.email
        assert not p.phone

    def test_whatsapp_group_id(self):
        p = parse_participant("Family <120363123456789-1234567890>", "whatsapp")
        assert p.is_group is True
        assert p.display_name == "Family"

    def test_cc_notation_stripped(self):
        p = parse_participant("user@test.com; CC: other@test.com", "mail")
        assert p.email == "user@test.com"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# parse_mail_participants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestParseMailParticipants:
    def test_simple_list(self):
        results = parse_mail_participants(
            ["user@test.com", "other@test.com"], {},
        )
        assert len(results) == 2
        assert results[0].email == "user@test.com"

    def test_cc_expansion(self):
        results = parse_mail_participants(
            ["user@test.com; CC: cc1@test.com"], {},
        )
        assert len(results) == 2

    def test_empty_participants(self):
        results = parse_mail_participants([], {})
        assert results == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# parse_calendar_participants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestParseCalendarParticipants:
    def test_with_attendees_meta(self):
        meta = {
            "attendees": [
                {"email": "alice@test.com", "name": "Alice"},
                {"email": "bob@test.com", "name": "Bob"},
            ],
        }
        results = parse_calendar_participants([], meta)
        assert len(results) == 2
        assert results[0].email == "alice@test.com"
        assert results[0].display_name == "Alice"

    def test_with_organizer(self):
        meta = {
            "organizer_email": "org@test.com",
            "organizer_name": "Organizer",
        }
        results = parse_calendar_participants([], meta)
        assert len(results) == 1
        assert results[0].email == "org@test.com"

    def test_filters_noise(self):
        results = parse_calendar_participants(
            ["US Holidays", "Real Person"], {},
        )
        # US Holidays should be filtered
        assert len(results) == 1
        assert results[0].display_name == "Real Person"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UnionFind
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestUnionFind:
    def test_find_creates_element(self):
        uf = UnionFind()
        assert uf.find("a") == "a"

    def test_union_same_root(self):
        uf = UnionFind()
        uf.find("a")
        uf.union("a", "a")
        assert uf.find("a") == "a"

    def test_union_two_elements(self):
        uf = UnionFind()
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")

    def test_transitive_union(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.find("a") == uf.find("c")

    def test_groups(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("c", "d")
        groups = uf.groups()
        assert len(groups) == 2
        group_sizes = sorted(len(v) for v in groups.values())
        assert group_sizes == [2, 2]

    def test_three_way_merge(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("c", "d")
        groups = uf.groups()
        assert len(groups) == 1
        assert len(list(groups.values())[0]) == 4

    def test_path_compression(self):
        uf = UnionFind()
        # Build a chain: a->b->c->d
        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("c", "d")
        # After find, path should be compressed
        root = uf.find("a")
        assert uf._parent["a"] == root

    def test_empty_groups(self):
        uf = UnionFind()
        assert uf.groups() == {}

    def test_singleton_groups(self):
        uf = UnionFind()
        uf.find("x")
        uf.find("y")
        groups = uf.groups()
        assert len(groups) == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ResolvedPerson
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestResolvedPerson:
    def test_defaults(self):
        rp = ResolvedPerson()
        assert rp.person_id == ""
        assert rp.canonical_name == ""
        assert rp.emails == set()
        assert rp.phones == set()
        assert rp.is_user is False

    def test_fields_populated(self):
        rp = ResolvedPerson(
            person_id="person:abc",
            canonical_name="Alice",
            emails={"alice@test.com"},
            phones={"+1234567890"},
            sources={"mail", "contacts"},
            is_user=False,
        )
        assert rp.canonical_name == "Alice"
        assert len(rp.emails) == 1
        assert len(rp.sources) == 2


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# resolve_persons
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestResolvePersons:
    def _seed_store(self, store):
        now = int(time.time())
        # Email event
        eid1 = Event.make_id("mail", "email_1")
        store.put_event(Event(
            id=eid1, source="mail", source_id="email_1",
            event_type="email", timestamp=now,
            participants=["Sam Park <sam@test.com>", "user@example.com"],
            raw_content="Hello",
            metadata={"subject": "Test"},
        ))
        # iMessage event with phone
        eid2 = Event.make_id("imessage", "im_1")
        store.put_event(Event(
            id=eid2, source="imessage", source_id="im_1",
            event_type="message", timestamp=now,
            participants=["+15550100002"],
            raw_content="Hey",
        ))
        # Contact event that bridges phone and email
        eid3 = Event.make_id("contacts", "contact_sam")
        store.put_event(Event(
            id=eid3, source="contacts", source_id="contact_sam",
            event_type="identity", timestamp=now,
            metadata={
                "name": "Sam Park",
                "emails": ["sam@test.com"],
                "phones": ["+15550100002"],
            },
        ))

    @pytest.fixture
    def resolver_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        self._seed_store(store)
        return store

    def test_resolves_persons(self, resolver_store):
        persons = resolve_persons(resolver_store)
        assert len(persons) > 0

    def test_user_config_marks_user(self, resolver_store):
        # The resolver only creates persons for identifiers that participate
        # in a union (via contacts bridge or multi-identifier).
        # The user config with both email AND phone triggers a union.
        persons = resolve_persons(resolver_store, user_config={
            "emails": ["user@example.com"],
            "phones": ["+19196274709"],
            "name": "Test User",
        })
        user = [p for p in persons if p.is_user]
        assert len(user) >= 1
        assert user[0].canonical_name == "Test User"

    def test_contact_bridge_merges(self, resolver_store):
        """Contacts event bridges email and phone into one person."""
        persons = resolve_persons(resolver_store)
        # Sam's email and phone should merge via contacts bridge
        sam_person = [
            p for p in persons
            if "sam@test.com" in p.emails
        ]
        assert len(sam_person) == 1
        assert "+15550100002" in sam_person[0].phones

    def test_person_id_deterministic(self, resolver_store):
        p1 = resolve_persons(resolver_store)
        p2 = resolve_persons(resolver_store)
        ids1 = {p.person_id for p in p1}
        ids2 = {p.person_id for p in p2}
        assert ids1 == ids2

    def test_person_id_starts_with_person(self, resolver_store):
        persons = resolve_persons(resolver_store)
        for p in persons:
            assert p.person_id.startswith("person:")

    def test_empty_store(self):
        store = LayeredGraphStore(db_path=":memory:")
        persons = resolve_persons(store)
        assert persons == []

    def test_user_sorted_first(self, resolver_store):
        persons = resolve_persons(resolver_store, user_config={
            "emails": ["user@example.com"],
            "phones": ["+19196274709"],
            "name": "Test User",
        })
        if persons:
            assert persons[0].is_user is True

    def test_sources_tracked(self, resolver_store):
        persons = resolve_persons(resolver_store)
        sam_person = [p for p in persons if "sam@test.com" in p.emails]
        if sam_person:
            assert "mail" in sam_person[0].sources or "contacts" in sam_person[0].sources

    def test_no_user_config_no_user_flag(self, resolver_store):
        persons = resolve_persons(resolver_store)
        users = [p for p in persons if p.is_user]
        assert len(users) == 0  # No user without config


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# persist_persons
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPersistPersons:
    def test_writes_persons(self):
        store = LayeredGraphStore(db_path=":memory:")
        persons = [
            ResolvedPerson(
                person_id="person:abc",
                canonical_name="Alice",
                emails={"alice@test.com"},
                sources={"mail"},
            ),
        ]
        result = persist_persons(store, persons)
        assert result["persons_written"] == 1
        assert result["identifiers_written"] >= 1

    def test_writes_user_flag(self):
        store = LayeredGraphStore(db_path=":memory:")
        persons = [
            ResolvedPerson(
                person_id="person:user",
                canonical_name="Me",
                emails={"me@test.com"},
                is_user=True,
                sources={"mail"},
            ),
        ]
        result = persist_persons(store, persons)
        assert result["user_found"] is True

        db_person = store.get_person("person:user")
        assert db_person is not None
        assert db_person["is_user"] == 1

    def test_identifiers_queryable(self):
        store = LayeredGraphStore(db_path=":memory:")
        persons = [
            ResolvedPerson(
                person_id="person:test",
                canonical_name="Test",
                emails={"test@test.com"},
                phones={"+1234567890"},
                sources={"mail"},
            ),
        ]
        persist_persons(store, persons)

        identifiers = store.get_person_identifiers("person:test")
        id_types = {i["identifier_type"] for i in identifiers}
        assert "email" in id_types
        assert "phone" in id_types

    def test_idempotent(self):
        store = LayeredGraphStore(db_path=":memory:")
        persons = [
            ResolvedPerson(
                person_id="person:test",
                canonical_name="Test",
                emails={"test@test.com"},
                sources={"mail"},
            ),
        ]
        persist_persons(store, persons)
        persist_persons(store, persons)
        # Should not duplicate
        all_persons = store.get_all_persons()
        assert len(all_persons) == 1

    def test_display_names_stored(self):
        store = LayeredGraphStore(db_path=":memory:")
        persons = [
            ResolvedPerson(
                person_id="person:test",
                canonical_name="Canonical Name",
                emails={"test@test.com"},
                display_names={"Canonical Name", "Other Name"},
                sources={"mail"},
            ),
        ]
        persist_persons(store, persons)
        identifiers = store.get_person_identifiers("person:test")
        id_types = {i["identifier_type"] for i in identifiers}
        assert "display_name" in id_types

    def test_empty_list(self):
        store = LayeredGraphStore(db_path=":memory:")
        result = persist_persons(store, [])
        assert result["persons_written"] == 0
        assert result["user_found"] is False
