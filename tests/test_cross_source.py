"""Tests for alteris.cross_source: Cross-source content linking.

Tests cover:
  - Dollar amount regex extraction
  - Name index building from persons + profile
  - Name-in-text matching with word boundaries
  - Annotation generation (dollar_amount, person_mention facets)
  - Cross-source dollar cluster claims
  - Cross-source temporal burst claims
  - Cross-source entity bridge claims
  - Cross-source name bridge claims
  - Calendar corroboration claims
  - find_related_events context retrieval
  - Idempotency (re-run produces zero new claims)
  - Edge cases (single source, no matches, empty content)
"""

import json
import time

import pytest

from alteris.cross_source import (
    DOLLAR_CLUSTER_WINDOW_S,
    TEMPORAL_BURST_WINDOW_S,
    annotate_events,
    build_name_index,
    extract_dollar_amounts,
    find_names_in_text,
    find_related_events,
    run_cross_source_linking,
    _extract_calendar_corroboration,
    _extract_dollar_clusters,
    _extract_entity_bridges,
    _extract_name_bridges,
    _extract_temporal_bursts,
)
from alteris.models import Annotation, Event
from alteris.store import LayeredGraphStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_event(source, source_id, event_type, timestamp, raw_content="",
                metadata=None, participants=()):
    """Create and return an Event with proper ID."""
    eid = Event.make_id(source, source_id)
    return Event(
        id=eid,
        source=source,
        source_id=source_id,
        event_type=event_type,
        timestamp=timestamp,
        participants=tuple(participants),
        raw_content=raw_content,
        metadata=metadata or {},
    )


def _store_with_events_and_persons(*events, persons=None):
    """Create an in-memory store, insert events and optionally persons."""
    store = LayeredGraphStore(db_path=":memory:")
    for e in events:
        store.put_event(e)
    for p in (persons or []):
        store.put_person(p["id"], canonical_name=p["name"],
                         is_user=p.get("is_user", False),
                         sources=p.get("sources", ["mail"]))
    return store


def _store_with_events(*events):
    """Create an in-memory store and insert events."""
    return _store_with_events_and_persons(*events)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unit tests: dollar amount extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDollarAmounts:
    def test_simple_amount(self):
        assert extract_dollar_amounts("You paid $472.50") == ["472.50"]

    def test_multiple_amounts(self):
        result = extract_dollar_amounts("$100.00 deposit, $1,200.00 total")
        assert result == ["100.00", "1200.00"]

    def test_no_amounts(self):
        assert extract_dollar_amounts("No money here") == []

    def test_comma_separated(self):
        assert extract_dollar_amounts("$12,345.67 charge") == ["12345.67"]

    def test_empty_string(self):
        assert extract_dollar_amounts("") == []

    def test_none_input(self):
        assert extract_dollar_amounts(None) == []

    def test_deduplicates(self):
        result = extract_dollar_amounts("$50.00 and again $50.00")
        assert result == ["50.00"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unit tests: name index and matching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestNameIndex:
    def test_builds_from_persons(self):
        store = LayeredGraphStore(db_path=":memory:")
        store.put_person("p1", canonical_name="Sam Park", sources=["mail"])
        store.put_person("p2", canonical_name="Dana Kim", sources=["mail"])
        store.put_person("p_user", canonical_name="Me", is_user=True, sources=["mail"])

        index = build_name_index(store)
        assert "sam park" in index
        assert "dana kim" in index
        # First names shorter than FIRST_NAME_MIN_LEN (5) are excluded
        # to avoid common word collisions (Sam=3, Dana=4)
        assert "sam" not in index
        assert "dana" not in index
        # User is excluded
        assert index.get("me") is None or index["me"] == "user"  # may come from profile

    def test_skips_short_names(self):
        store = LayeredGraphStore(db_path=":memory:")
        store.put_person("p1", canonical_name="Al", sources=["mail"])
        store.put_person("p2", canonical_name="Ed", sources=["mail"])

        index = build_name_index(store)
        assert "al" not in index
        assert "ed" not in index

    def test_full_name_before_first_name(self):
        """Full name match should suppress first name match."""
        index = {"sam park": "p1", "sam": "p1"}
        found = find_names_in_text("Message from Sam Park about the project", index)
        # Should match "sam park" but NOT separately match "sam"
        assert len(found) == 1
        assert found[0][0] == "sam park"


class TestNameMatching:
    def test_word_boundary(self):
        """Name must be at word boundary, not partial match."""
        index = {"mark": "p1"}
        # Should match "Mark" as a standalone word
        assert len(find_names_in_text("Mark sent a message", index)) == 1
        # Should NOT match "bookmark" or "marking"
        assert len(find_names_in_text("Check the bookmark", index)) == 0
        assert len(find_names_in_text("Marking the page", index)) == 0

    def test_case_insensitive(self):
        index = {"sam": "p1"}
        assert len(find_names_in_text("SAM sent a message", index)) == 1
        assert len(find_names_in_text("sam sent a message", index)) == 1

    def test_multiple_names(self):
        index = {"sam": "p1", "dana": "p2", "maya": "p3"}
        found = find_names_in_text("Sam and Dana talked about Maya", index)
        assert len(found) == 3

    def test_empty_text(self):
        index = {"sam": "p1"}
        assert find_names_in_text("", index) == []
        assert find_names_in_text(None, index) == []

    def test_name_in_email_subject(self):
        """Names in email subjects should match."""
        index = {"maya": "p1", "aquarium": "p2"}
        # "aquarium" won't be in the name index — only real names
        # but if it somehow were, it would match
        found = find_names_in_text("Maya's aquarium camp registration", index)
        names = {f[0] for f in found}
        assert "maya" in names


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration tests: annotation generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAnnotateEvents:
    def test_dollar_annotations(self):
        now = int(time.time())
        e = _make_event("mail", "email_1", "email", now,
                        raw_content="Your charge of $472.50 has been processed")
        store = _store_with_events(e)

        stats = annotate_events(store)
        assert stats["dollar_amounts"] == 1

        anns = store.get_annotations(facet="dollar_amount")
        assert len(anns) == 1
        assert anns[0].value == "472.50"

    def test_name_mention_annotations(self):
        now = int(time.time())
        e = _make_event("mail", "email_1", "email", now,
                        raw_content="Sam Park confirmed the meeting")
        store = _store_with_events(e)
        store.put_person("p_sam", canonical_name="Sam Park", sources=["mail"])

        stats = annotate_events(store)
        assert stats["name_mentions"] >= 1

        anns = store.get_annotations(facet="person_mention")
        names = {a.value for a in anns}
        assert "sam park" in names

    def test_idempotent(self):
        now = int(time.time())
        e = _make_event("mail", "email_1", "email", now,
                        raw_content="Charge: $100.00")
        store = _store_with_events(e)

        stats1 = annotate_events(store)
        stats2 = annotate_events(store)
        assert stats2["dollar_amounts"] == 0
        assert stats2["name_mentions"] == 0

    def test_since_ts_filter(self):
        now = int(time.time())
        old = _make_event("mail", "old", "email", now - 86400,
                          raw_content="$50.00 old charge")
        new = _make_event("mail", "new", "email", now,
                          raw_content="$75.00 new charge")
        store = _store_with_events(old, new)

        stats = annotate_events(store, since_ts=now - 3600)
        assert stats["events_scanned"] == 1
        assert stats["dollar_amounts"] == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration tests: dollar cluster claims
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDollarClusters:
    def test_cross_source_dollar_cluster(self):
        """Same $ amount from mail + iMessage within 1hr -> claim."""
        now = int(time.time())
        e1 = _make_event("mail", "charge_email", "email", now,
                         raw_content="Payment of $472.50 confirmed")
        e2 = _make_event("imessage", "confirm_msg", "message", now + 60,
                         raw_content="Got the $472.50 charge notification")
        store = _store_with_events(e1, e2)
        annotate_events(store)

        claims = _extract_dollar_clusters(store)
        assert len(claims) == 1
        claim = claims[0]
        assert claim.claim_type == "cross_source_dollar_cluster"
        assert claim.subject == "$472.50"
        obj = json.loads(claim.object)
        assert set(obj["sources"]) == {"mail", "imessage"}

    def test_same_source_no_claim(self):
        """Same $ amount from same source -> no cross-source claim."""
        now = int(time.time())
        e1 = _make_event("mail", "email_1", "email", now,
                         raw_content="$100.00 charge")
        e2 = _make_event("mail", "email_2", "email", now + 30,
                         raw_content="$100.00 confirmed")
        store = _store_with_events(e1, e2)
        annotate_events(store)

        claims = _extract_dollar_clusters(store)
        assert len(claims) == 0

    def test_outside_window_no_claim(self):
        """Same $ from different sources but >1hr apart -> no claim."""
        now = int(time.time())
        e1 = _make_event("mail", "email_1", "email", now,
                         raw_content="$200.00 charge")
        e2 = _make_event("imessage", "msg_1", "message",
                         now + DOLLAR_CLUSTER_WINDOW_S + 100,
                         raw_content="$200.00 notification")
        store = _store_with_events(e1, e2)
        annotate_events(store)

        claims = _extract_dollar_clusters(store)
        assert len(claims) == 0

    def test_multiple_clusters(self):
        """Two different amounts each from 2 sources -> 2 claims."""
        now = int(time.time())
        events = [
            _make_event("mail", "e1", "email", now,
                        raw_content="$100.00 for item A"),
            _make_event("imessage", "e2", "message", now + 30,
                        raw_content="$100.00 confirmed"),
            _make_event("mail", "e3", "email", now + 300,
                        raw_content="$500.00 for item B"),
            _make_event("whatsapp", "e4", "message", now + 400,
                        raw_content="$500.00 sent"),
        ]
        store = _store_with_events(*events)
        annotate_events(store)

        claims = _extract_dollar_clusters(store)
        assert len(claims) == 2
        amounts = {c.subject for c in claims}
        assert amounts == {"$100.00", "$500.00"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration tests: temporal burst claims
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTemporalBursts:
    def test_multi_source_burst(self):
        """3+ events from 2+ sources within 5min -> burst claim."""
        now = int(time.time())
        events = [
            _make_event("mail", "e1", "email", now),
            _make_event("imessage", "e2", "message", now + 30),
            _make_event("calendar", "e3", "calendar_event", now + 60),
        ]
        store = _store_with_events(*events)

        claims = _extract_temporal_bursts(store)
        assert len(claims) == 1
        obj = json.loads(claims[0].object)
        assert obj["event_count"] >= 3
        assert obj["source_count"] >= 2

    def test_single_source_no_burst(self):
        """3 events from same source -> no burst claim."""
        now = int(time.time())
        events = [
            _make_event("mail", f"e{i}", "email", now + i * 10)
            for i in range(5)
        ]
        store = _store_with_events(*events)

        claims = _extract_temporal_bursts(store)
        assert len(claims) == 0

    def test_too_few_events(self):
        """2 events from 2 sources -> no burst (need 3+)."""
        now = int(time.time())
        events = [
            _make_event("mail", "e1", "email", now),
            _make_event("imessage", "e2", "message", now + 30),
        ]
        store = _store_with_events(*events)

        claims = _extract_temporal_bursts(store)
        assert len(claims) == 0

    def test_identity_events_excluded(self):
        """Identity events don't count toward bursts."""
        now = int(time.time())
        events = [
            _make_event("contacts", "c1", "identity", now),
            _make_event("contacts", "c2", "identity", now + 10),
            _make_event("mail", "e1", "email", now + 20),
        ]
        store = _store_with_events(*events)

        claims = _extract_temporal_bursts(store)
        assert len(claims) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration tests: entity bridge claims
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestEntityBridges:
    def test_person_in_two_sources(self):
        """Person appearing in mail + whatsapp -> bridge claim."""
        now = int(time.time())
        e1 = _make_event("mail", "email_1", "email", now)
        e2 = _make_event("whatsapp", "wa_1", "message", now + 100)
        store = _store_with_events(e1, e2)

        store.put_person("p_sam", canonical_name="Sam", sources=["mail", "whatsapp"])
        store.put_person("p_user", canonical_name="User", is_user=True, sources=["mail"])
        store.conn.execute(
            "INSERT INTO event_persons (event_id, person_id, role) VALUES (?, ?, ?)",
            (e1.id, "p_sam", "sender"),
        )
        store.conn.execute(
            "INSERT INTO event_persons (event_id, person_id, role) VALUES (?, ?, ?)",
            (e2.id, "p_sam", "sender"),
        )
        store.conn.commit()

        claims = _extract_entity_bridges(store)
        assert len(claims) == 1
        obj = json.loads(claims[0].object)
        assert obj["person_name"] == "Sam"
        assert set(obj["sources"]) == {"mail", "whatsapp"}

    def test_user_not_a_bridge(self):
        """User person appearing everywhere should NOT create bridge claims."""
        now = int(time.time())
        e1 = _make_event("mail", "e1", "email", now)
        e2 = _make_event("whatsapp", "e2", "message", now)
        store = _store_with_events(e1, e2)

        store.put_person("p_user", canonical_name="Me", is_user=True, sources=["mail"])
        store.conn.execute(
            "INSERT INTO event_persons (event_id, person_id, role) VALUES (?, ?, ?)",
            (e1.id, "p_user", "sender"),
        )
        store.conn.execute(
            "INSERT INTO event_persons (event_id, person_id, role) VALUES (?, ?, ?)",
            (e2.id, "p_user", "sender"),
        )
        store.conn.commit()

        claims = _extract_entity_bridges(store)
        assert len(claims) == 0

    def test_single_source_person_no_bridge(self):
        """Person only in one source -> no bridge."""
        now = int(time.time())
        e1 = _make_event("mail", "e1", "email", now)
        e2 = _make_event("mail", "e2", "email", now + 100)
        store = _store_with_events(e1, e2)

        store.put_person("p_bob", canonical_name="Bob", sources=["mail"])
        store.conn.execute(
            "INSERT INTO event_persons (event_id, person_id, role) VALUES (?, ?, ?)",
            (e1.id, "p_bob", "sender"),
        )
        store.conn.execute(
            "INSERT INTO event_persons (event_id, person_id, role) VALUES (?, ?, ?)",
            (e2.id, "p_bob", "sender"),
        )
        store.conn.commit()

        claims = _extract_entity_bridges(store)
        assert len(claims) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration tests: name bridge claims
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestNameBridges:
    def test_name_mentioned_in_two_sources(self):
        """Same name in mail body + whatsapp body -> name bridge claim."""
        now = int(time.time())
        e1 = _make_event("mail", "email_1", "email", now,
                         raw_content="Sam Park confirmed the venue")
        e2 = _make_event("whatsapp", "wa_1", "message", now + 100,
                         raw_content="Can you ask Sam Park about the time?")
        store = _store_with_events(e1, e2)
        store.put_person("p_sam", canonical_name="Sam Park", sources=["mail"])
        annotate_events(store)

        claims = _extract_name_bridges(store)
        sam_claims = [c for c in claims if "sam" in c.subject]
        assert len(sam_claims) >= 1
        obj = json.loads(sam_claims[0].object)
        assert set(obj["sources"]) == {"mail", "whatsapp"}

    def test_name_single_source_no_bridge(self):
        """Name only in one source -> no bridge."""
        now = int(time.time())
        e1 = _make_event("mail", "email_1", "email", now,
                         raw_content="Sam confirmed")
        e2 = _make_event("mail", "email_2", "email", now + 100,
                         raw_content="Sam sent the agenda")
        store = _store_with_events(e1, e2)
        store.put_person("p_sam", canonical_name="Sam", sources=["mail"])
        annotate_events(store)

        claims = _extract_name_bridges(store)
        assert len(claims) == 0

    def test_family_name_from_profile(self):
        """Names from profile.yaml (family members) create bridges."""
        now = int(time.time())
        e1 = _make_event("mail", "email_1", "email", now,
                         raw_content="Maya Chen's school event is next week")
        e2 = _make_event("imessage", "msg_1", "message", now + 300,
                         raw_content="Pick up Maya Chen at 3pm")
        store = _store_with_events(e1, e2)
        store.put_person("p_maya", canonical_name="Maya Chen",
                         sources=["contacts"])
        annotate_events(store)

        claims = _extract_name_bridges(store)
        maya_claims = [c for c in claims if "maya" in c.subject]
        assert len(maya_claims) >= 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration tests: calendar corroboration claims
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCalendarCorroboration:
    def test_calendar_with_matching_name(self):
        """Calendar event mentions name + nearby email mentions same name."""
        now = int(time.time())
        cal = _make_event("calendar", "cal_1", "calendar_event", now,
                          metadata={"subject": "Meeting with Sam Park"})
        email = _make_event("mail", "email_1", "email", now + 300,
                            raw_content="Sam Park sent the agenda for the meeting")
        store = _store_with_events(cal, email)
        store.put_person("p_sam", canonical_name="Sam Park", sources=["mail"])
        annotate_events(store)

        claims = _extract_calendar_corroboration(store)
        assert len(claims) == 1
        obj = json.loads(claims[0].object)
        assert "sam park" in obj["shared_signals"]
        assert "mail" in obj["corroborating_sources"]

    def test_no_shared_annotations(self):
        """Calendar event + email with no shared names/amounts -> no claim."""
        now = int(time.time())
        cal = _make_event("calendar", "cal_1", "calendar_event", now,
                          metadata={"subject": "Team standup"})
        email = _make_event("mail", "email_1", "email", now + 100,
                            raw_content="Please review the quarterly report")
        store = _store_with_events(cal, email)
        annotate_events(store)

        claims = _extract_calendar_corroboration(store)
        assert len(claims) == 0

    def test_outside_window(self):
        """Calendar + email with shared name but >30min apart -> no claim."""
        now = int(time.time())
        cal = _make_event("calendar", "cal_1", "calendar_event", now,
                          metadata={"subject": "Meeting with Sam"})
        email = _make_event("mail", "email_1", "email", now + 7200,
                            raw_content="Sam mentioned the project")
        store = _store_with_events(cal, email)
        store.put_person("p_sam", canonical_name="Sam", sources=["mail"])
        annotate_events(store)

        claims = _extract_calendar_corroboration(store)
        assert len(claims) == 0

    def test_dollar_amount_corroboration(self):
        """Calendar event + email share a dollar amount -> corroboration."""
        now = int(time.time())
        cal = _make_event("calendar", "cal_1", "calendar_event", now,
                          raw_content="Budget: $500.00",
                          metadata={"subject": "Budget review"})
        email = _make_event("mail", "email_1", "email", now + 300,
                            raw_content="The $500.00 expense was approved")
        store = _store_with_events(cal, email)
        annotate_events(store)

        claims = _extract_calendar_corroboration(store)
        assert len(claims) == 1
        obj = json.loads(claims[0].object)
        assert "500.00" in obj["shared_signals"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration tests: find_related_events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFindRelatedEvents:
    def test_finds_by_dollar_amount(self):
        """Event with $472.50 finds matching event from another source."""
        now = int(time.time())
        e1 = _make_event("mail", "email_1", "email", now,
                         raw_content="Your charge of $472.50")
        e2 = _make_event("imessage", "msg_1", "message", now + 60,
                         raw_content="$472.50 notification received")
        store = _store_with_events(e1, e2)
        annotate_events(store)

        related = find_related_events(store, e1.id)
        assert len(related) >= 1
        assert any(r["event_id"] == e2.id for r in related)
        dollar_match = [r for r in related if r["relationship"] == "same_dollar_amount"]
        assert len(dollar_match) == 1
        assert dollar_match[0]["shared_signal"] == "$472.50"

    def test_finds_by_name_mention(self):
        """Event mentioning Sam Park finds Sam Park mentions in other sources."""
        now = int(time.time())
        e1 = _make_event("mail", "email_1", "email", now,
                         raw_content="Sam Park confirmed the venue")
        e2 = _make_event("whatsapp", "wa_1", "message", now + 100,
                         raw_content="Sam Park is running late")
        store = _store_with_events(e1, e2)
        store.put_person("p_sam", canonical_name="Sam Park", sources=["mail"])
        annotate_events(store)

        related = find_related_events(store, e1.id)
        name_matches = [r for r in related if r["relationship"] == "same_person_mentioned"]
        assert len(name_matches) >= 1
        assert name_matches[0]["shared_signal"] == "sam park"

    def test_finds_by_shared_person(self):
        """Event linked to Sam in event_persons finds Sam's events."""
        now = int(time.time())
        e1 = _make_event("mail", "email_1", "email", now)
        e2 = _make_event("whatsapp", "wa_1", "message", now + 100)
        store = _store_with_events(e1, e2)

        store.put_person("p_sam", canonical_name="Sam",
                         sources=["mail", "whatsapp"])
        store.conn.execute(
            "INSERT INTO event_persons (event_id, person_id, role) VALUES (?, ?, ?)",
            (e1.id, "p_sam", "sender"),
        )
        store.conn.execute(
            "INSERT INTO event_persons (event_id, person_id, role) VALUES (?, ?, ?)",
            (e2.id, "p_sam", "sender"),
        )
        store.conn.commit()

        related = find_related_events(store, e1.id)
        person_matches = [r for r in related if r["relationship"] == "shared_person"]
        assert len(person_matches) == 1
        assert person_matches[0]["shared_signal"] == "Sam"

    def test_nonexistent_event(self):
        """Querying a nonexistent event returns empty."""
        store = LayeredGraphStore(db_path=":memory:")
        related = find_related_events(store, "nonexistent")
        assert related == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Integration tests: full pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFullPipeline:
    def test_aquarium_scenario(self):
        """Realistic: aquarium camp booking with mail + iMessage + calendar."""
        now = int(time.time())
        events = [
            _make_event("mail", "campdoc_confirm", "email", now,
                        metadata={"subject": "CampDoc Registration Confirmed"},
                        raw_content="Your registration for Aquarium Summer Camp "
                                    "for Maya Chen is confirmed. Amount: $472.50"),
            _make_event("mail", "chase_charge", "email", now + 30,
                        metadata={"subject": "Transaction Alert"},
                        raw_content="A charge of $472.50 from DOCNETWORK"),
            _make_event("imessage", "apple_notif", "message", now + 60,
                        raw_content="Maya Chen's camp registration is done"),
            _make_event("calendar", "camp_event", "calendar_event",
                        now + 86400 * 30,
                        metadata={"subject": "Maya Chen Aquarium Camp"}),
        ]
        store = _store_with_events(*events)
        # Add Maya to persons so name matching works
        store.put_person("p_maya", canonical_name="Maya Chen",
                         sources=["contacts"])

        stats = run_cross_source_linking(store)

        # Should have name mentions for Maya across sources
        name_anns = store.get_annotations(facet="person_mention")
        ananya_anns = [a for a in name_anns if "maya" in a.value]
        assert len(ananya_anns) >= 2  # mail + imessage + calendar

        # Should have temporal burst (3 events from 2+ sources in 90s)
        burst_claims = _get_claims_by_type(store, "cross_source_temporal_burst")
        assert len(burst_claims) >= 1

        # Should have name bridge for Maya across sources
        name_claims = _get_claims_by_type(store, "cross_source_name_bridge")
        ananya_bridges = [c for c in name_claims
                          if "maya" in c["subject"]]
        assert len(ananya_bridges) >= 1

    def test_idempotent(self):
        """Running twice produces zero new claims."""
        now = int(time.time())
        events = [
            _make_event("mail", "e1", "email", now,
                        raw_content="$100.00 charge"),
            _make_event("imessage", "e2", "message", now + 30,
                        raw_content="$100.00 confirmed"),
        ]
        store = _store_with_events(*events)

        stats1 = run_cross_source_linking(store)
        stats2 = run_cross_source_linking(store)
        assert stats2["new_claims"] == 0

    def test_empty_store(self):
        """Empty store produces zero claims, no errors."""
        store = LayeredGraphStore(db_path=":memory:")
        stats = run_cross_source_linking(store)
        assert stats["total_claims"] == 0


def _get_claims_by_type(store, claim_type):
    """Helper to get claims of a given type from the store."""
    return store.conn.execute(
        "SELECT * FROM claims WHERE claim_type = ?",
        (claim_type,),
    ).fetchall()
