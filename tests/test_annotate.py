"""Tests for loom.annotate: structural annotation extraction."""

import time

import pytest

from loom.annotate import annotate_event, annotate_structural
from loom.models import Annotation, Event
from loom.store import LayeredGraphStore


@pytest.fixture
def store():
    s = LayeredGraphStore(db_path=":memory:")
    yield s
    s.close()


def make_event(source="mail", source_id="test_1", event_type="email",
               participants=(), metadata=None, **kwargs):
    eid = Event.make_id(source, source_id)
    return Event(
        id=eid, source=source, source_id=source_id,
        event_type=event_type, timestamp=int(time.time()),
        participants=participants, raw_content="test",
        metadata=metadata or {}, **kwargs,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mail annotations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMailAnnotations:
    def test_sender_domain(self):
        ev = make_event(participants=("user@chase.com",))
        anns = annotate_event(ev)
        domains = [a for a in anns if a.facet == "sender_domain"]
        assert len(domains) == 1
        assert domains[0].value == "chase.com"

    def test_automated(self):
        ev = make_event(metadata={"automated": True})
        anns = annotate_event(ev)
        assert any(a.facet == "is_automated" for a in anns)

    def test_list_id(self):
        ev = make_event(metadata={"list_id": True})
        anns = annotate_event(ev)
        assert any(a.facet == "has_list_id" for a in anns)

    def test_email_category(self):
        ev = make_event(metadata={"model_category": 3})
        anns = annotate_event(ev)
        cats = [a for a in anns if a.facet == "email_category"]
        assert len(cats) == 1
        assert cats[0].value == "3"
        assert cats[0].source == "apple_intelligence"

    def test_junk_level(self):
        ev = make_event(metadata={"junk_level": 2})
        anns = annotate_event(ev)
        assert any(a.facet == "junk_level" and a.value == "2" for a in anns)

    def test_high_impact(self):
        ev = make_event(metadata={"high_impact": True})
        anns = annotate_event(ev)
        ts = [a for a in anns if a.facet == "time_sensitive"]
        assert len(ts) == 1
        assert ts[0].source == "apple_intelligence"

    def test_noreply(self):
        ev = make_event(participants=("noreply@example.com",))
        anns = annotate_event(ev)
        assert any(a.facet == "is_noreply" for a in anns)

    def test_noreply_with_angle_brackets(self):
        ev = make_event(participants=("Example <no-reply@example.com>",))
        anns = annotate_event(ev)
        assert any(a.facet == "is_noreply" for a in anns)

    def test_replied(self):
        ev = make_event(metadata={"replied": True})
        anns = annotate_event(ev)
        assert any(a.facet == "has_reply" for a in anns)

    def test_from_me(self):
        ev = make_event(metadata={"is_from_me": True})
        anns = annotate_event(ev)
        assert any(a.facet == "is_from_me" for a in anns)

    def test_clean_mail_minimal_annotations(self):
        """Mail with no metadata flags produces minimal annotations."""
        ev = make_event(metadata={})
        anns = annotate_event(ev)
        # No participants → no sender_domain, no flags → empty or very few
        assert len(anns) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# iMessage annotations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestIMessageAnnotations:
    def test_filtered(self):
        ev = make_event(source="imessage", event_type="message",
                        metadata={"is_filtered": True})
        anns = annotate_event(ev)
        assert any(a.facet == "is_filtered" for a in anns)

    def test_quietly_delivered(self):
        ev = make_event(source="imessage", event_type="message",
                        metadata={"delivered_quietly": True})
        anns = annotate_event(ev)
        assert any(a.facet == "delivered_quietly" for a in anns)

    def test_group(self):
        ev = make_event(source="imessage", event_type="message",
                        metadata={"is_group": True})
        anns = annotate_event(ev)
        assert any(a.facet == "is_group" for a in anns)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WhatsApp annotations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestWhatsAppAnnotations:
    def test_group_with_size(self):
        ev = make_event(source="whatsapp", event_type="message",
                        metadata={"is_group": True, "group_size": 15})
        anns = annotate_event(ev)
        assert any(a.facet == "is_group" for a in anns)
        assert any(a.facet == "group_size" and a.value == "15" for a in anns)

    def test_content_type(self):
        ev = make_event(source="whatsapp", event_type="message",
                        metadata={"content_type": "image"})
        anns = annotate_event(ev)
        assert any(a.facet == "content_type" and a.value == "image" for a in anns)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Calendar annotations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCalendarAnnotations:
    def test_holiday(self):
        ev = make_event(source="calendar", event_type="calendar_event",
                        metadata={"is_holiday": True})
        anns = annotate_event(ev)
        assert any(a.facet == "calendar_noise" and a.value == "holiday" for a in anns)

    def test_birthday(self):
        ev = make_event(source="calendar", event_type="calendar_event",
                        metadata={"is_birthday": True})
        anns = annotate_event(ev)
        assert any(a.facet == "calendar_noise" and a.value == "birthday" for a in anns)

    def test_declined_stripped(self):
        """Calendar acceptance status is stripped — unreliable defaults."""
        ev = make_event(source="calendar", event_type="calendar_event",
                        metadata={"is_declined": True})
        anns = annotate_event(ev)
        assert not any(a.facet == "is_declined" for a in anns)

    def test_accepted_stripped(self):
        """Calendar acceptance status is stripped — unreliable defaults."""
        ev = make_event(source="calendar", event_type="calendar_event",
                        metadata={"is_accepted": True, "has_attendees": True})
        anns = annotate_event(ev)
        assert not any(a.facet == "is_accepted" for a in anns)
        assert any(a.facet == "has_attendees" for a in anns)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Granola annotations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGranolaAnnotations:
    def test_has_transcript(self):
        ev = make_event(source="granola", event_type="meeting",
                        metadata={"has_transcript": True})
        anns = annotate_event(ev)
        assert any(a.facet == "has_transcript" for a in anns)

    def test_participant_count(self):
        ev = make_event(source="granola", event_type="meeting",
                        metadata={"participant_count": 5})
        anns = annotate_event(ev)
        assert any(a.facet == "participant_count" and a.value == "5" for a in anns)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Slack annotations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSlackAnnotations:
    def test_channel(self):
        ev = make_event(source="slack", event_type="message",
                        metadata={"channel_name": "general"})
        anns = annotate_event(ev)
        assert any(a.facet == "channel" and a.value == "general" for a in anns)

    def test_threaded(self):
        ev = make_event(source="slack", event_type="message",
                        metadata={"has_thread": True, "reply_count": 5})
        anns = annotate_event(ev)
        assert any(a.facet == "has_thread" for a in anns)
        assert any(a.facet == "reply_count" and a.value == "5" for a in anns)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# End-to-end: annotate_structural
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAnnotateStructural:
    def test_end_to_end(self, store):
        events = [
            make_event(source="mail", source_id="a1",
                       participants=("sender@chase.com",),
                       metadata={"automated": True, "model_category": 3}),
            make_event(source="imessage", source_id="a2",
                       event_type="message",
                       metadata={"is_filtered": True}),
        ]
        for ev in events:
            store.put_event(ev)

        result = annotate_structural(store, events)
        assert result["events_processed"] == 2
        assert result["annotations_written"] > 0

        # Check annotations are queryable
        anns = store.get_annotations(facet="sender_domain", value="chase.com")
        assert len(anns) == 1

    def test_idempotent(self, store):
        """Running annotation twice writes zero new on second run."""
        ev = make_event(source="mail", source_id="idem",
                        participants=("x@example.com",),
                        metadata={"automated": True})
        store.put_event(ev)

        r1 = annotate_structural(store, [ev])
        r2 = annotate_structural(store, [ev])
        assert r1["annotations_written"] > 0
        assert r2["annotations_written"] == 0

    def test_contacts_no_annotations(self, store):
        """Contacts source has no annotator → zero annotations."""
        ev = make_event(source="contacts", source_id="c1",
                        event_type="identity", metadata={})
        store.put_event(ev)
        result = annotate_structural(store, [ev])
        assert result["annotations_total"] == 0

    def test_empty_events(self, store):
        result = annotate_structural(store, [])
        assert result["events_processed"] == 0
        assert result["annotations_written"] == 0
