"""Tests for loom.briefing: Weekly briefing generator (Stage 8).

Tests cover:
  - CalendarEvent and Commitment data classes
  - Calendar event retrieval with attendee resolution
  - Open commitment retrieval from claims table
  - Commitment-to-calendar matching via person graph overlap
  - Per-event context gathering (recent comms with attendees)
  - Source snippet retrieval for orphaned commitments
  - Prompt construction with pre-grouped orphaned commitments
  - Full pipeline with MockLLMClient
  - Edge cases: no calendar events, no commitments, empty LLM response
"""

import json
import time

import pytest

from loom.briefing import (
    BRIEFING_DAYS_AHEAD,
    CalendarEvent,
    Commitment,
    _build_briefing_prompt,
    _execute_graph_query,
    _gather_event_context,
    _get_commitment_source_snippet,
    _get_cross_source_links,
    _get_event_facets,
    _get_open_commitments,
    _get_upcoming_events,
    _match_commitments_to_events,
    _run_anticipation_pass,
    run_briefing,
)
from loom.llm.mock import MockLLMClient
from loom.models import (
    Annotation,
    Claim,
    Event,
    ExtractionProvenance,
    Modality,
)
from loom.privacy import SensitivityLevel
from loom.store import LayeredGraphStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CalendarEvent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCalendarEvent:
    def test_duration_minutes(self):
        e = CalendarEvent(
            event_id="e1", title="Meeting",
            start_ts=1000, end_ts=4600,
        )
        assert e.duration_minutes == 60

    def test_zero_duration(self):
        e = CalendarEvent(event_id="e1", title="X", start_ts=1000, end_ts=1000)
        assert e.duration_minutes == 0

    def test_attendee_person_ids(self):
        e = CalendarEvent(
            event_id="e1", title="X", start_ts=0, end_ts=0,
            attendees=[
                {"person_id": "p1", "name": "Alice", "role": "attendee"},
                {"person_id": "p2", "name": "Bob", "role": "attendee"},
                {"name": "Unknown", "role": "attendee"},  # no person_id
            ],
        )
        assert e.attendee_person_ids == {"p1", "p2"}

    def test_format_time(self):
        # 2026-02-14 10:00 AM PST
        e = CalendarEvent(event_id="e1", title="X", start_ts=1771350000, end_ts=1771353600)
        result = e.format_time("America/Los_Angeles")
        assert "Feb" in result

    def test_defaults(self):
        e = CalendarEvent(event_id="e1", title="X", start_ts=0, end_ts=0)
        assert e.location == ""
        assert e.description == ""
        assert e.attendees == []
        assert e.metadata == {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Commitment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCommitment:
    def test_is_overdue_true(self):
        c = Commitment(
            claim_id="c1", type="promise", who="user",
            what="send report", deadline="2020-01-01",
        )
        assert c.is_overdue is True

    def test_is_overdue_false_future(self):
        c = Commitment(
            claim_id="c1", type="promise", who="user",
            what="send report", deadline="2099-12-31",
        )
        assert c.is_overdue is False

    def test_is_overdue_no_deadline(self):
        c = Commitment(claim_id="c1", type="promise", who="user", what="X")
        assert c.is_overdue is False

    def test_deadline_in_days_overdue(self):
        c = Commitment(
            claim_id="c1", type="promise", who="user",
            what="X", deadline="2020-01-01",
        )
        assert c.deadline_in_days is not None
        assert c.deadline_in_days < 0

    def test_deadline_in_days_future(self):
        c = Commitment(
            claim_id="c1", type="promise", who="user",
            what="X", deadline="2099-12-31",
        )
        assert c.deadline_in_days is not None
        assert c.deadline_in_days > 0

    def test_deadline_in_days_none(self):
        c = Commitment(claim_id="c1", type="promise", who="user", what="X")
        assert c.deadline_in_days is None

    def test_defaults(self):
        c = Commitment(claim_id="c1", type="task", who="user", what="do thing")
        assert c.to_whom == ""
        assert c.deadline is None
        assert c.status == "open"
        assert c.priority == 3
        assert c.source_person_ids == set()
        assert c.confidence == 0.8
        assert c.direction == "ambiguous"
        assert c.staleness_signal == "none"

    def test_direction_explicit(self):
        c = Commitment(
            claim_id="c1", type="task", who="user", what="X",
            direction="group_ask",
        )
        assert c.direction == "group_ask"

    def test_staleness_signal_explicit(self):
        c = Commitment(
            claim_id="c1", type="task", who="user", what="X",
            staleness_signal="overdue_no_followup",
        )
        assert c.staleness_signal == "overdue_no_followup"

    def test_direction_direct_ask(self):
        c = Commitment(
            claim_id="c1", type="task", who="user", what="X",
            direction="direct_ask",
        )
        assert c.direction == "direct_ask"

    def test_all_staleness_signals(self):
        for signal in ("none", "overdue_no_followup", "group_broadcast", "old_thread"):
            c = Commitment(
                claim_id="c1", type="task", who="user", what="X",
                staleness_signal=signal,
            )
            assert c.staleness_signal == signal


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Commitment-to-event matching
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMatchCommitmentsToEvents:
    def test_match_via_shared_person(self):
        events = [
            CalendarEvent(
                event_id="cal1", title="Meeting",
                start_ts=0, end_ts=3600,
                attendees=[{"person_id": "p1", "name": "Alice", "role": "attendee"}],
            ),
        ]
        commitments = [
            Commitment(
                claim_id="c1", type="promise", who="user",
                what="send proposal", source_person_ids={"p1"},
            ),
        ]
        matched, orphaned = _match_commitments_to_events(events, commitments)
        assert len(matched["cal1"]) == 1
        assert matched["cal1"][0].claim_id == "c1"
        assert len(orphaned) == 0

    def test_orphaned_when_no_overlap(self):
        events = [
            CalendarEvent(
                event_id="cal1", title="Meeting",
                start_ts=0, end_ts=3600,
                attendees=[{"person_id": "p1", "name": "Alice", "role": "attendee"}],
            ),
        ]
        commitments = [
            Commitment(
                claim_id="c1", type="promise", who="user",
                what="unrelated task", source_person_ids={"p999"},
            ),
        ]
        matched, orphaned = _match_commitments_to_events(events, commitments)
        assert len(matched["cal1"]) == 0
        assert len(orphaned) == 1
        assert orphaned[0].claim_id == "c1"

    def test_no_events(self):
        commitments = [
            Commitment(
                claim_id="c1", type="task", who="user",
                what="X", source_person_ids={"p1"},
            ),
        ]
        matched, orphaned = _match_commitments_to_events([], commitments)
        assert matched == {}
        assert len(orphaned) == 1

    def test_no_commitments(self):
        events = [
            CalendarEvent(
                event_id="cal1", title="Meeting",
                start_ts=0, end_ts=3600,
            ),
        ]
        matched, orphaned = _match_commitments_to_events(events, [])
        assert len(matched["cal1"]) == 0
        assert len(orphaned) == 0

    def test_commitment_matched_to_multiple_events(self):
        """Same person in two events → commitment matched to both."""
        events = [
            CalendarEvent(
                event_id="cal1", title="Morning sync",
                start_ts=0, end_ts=3600,
                attendees=[{"person_id": "p1", "name": "Alice", "role": "attendee"}],
            ),
            CalendarEvent(
                event_id="cal2", title="Afternoon review",
                start_ts=3600, end_ts=7200,
                attendees=[{"person_id": "p1", "name": "Alice", "role": "attendee"}],
            ),
        ]
        commitments = [
            Commitment(
                claim_id="c1", type="promise", who="user",
                what="send doc", source_person_ids={"p1"},
            ),
        ]
        matched, orphaned = _match_commitments_to_events(events, commitments)
        assert len(matched["cal1"]) == 1
        assert len(matched["cal2"]) == 1
        assert len(orphaned) == 0

    def test_no_duplicate_match(self):
        """Commitment with two persons pointing to same event → only matched once."""
        events = [
            CalendarEvent(
                event_id="cal1", title="Meeting",
                start_ts=0, end_ts=3600,
                attendees=[
                    {"person_id": "p1", "name": "Alice", "role": "attendee"},
                    {"person_id": "p2", "name": "Bob", "role": "attendee"},
                ],
            ),
        ]
        commitments = [
            Commitment(
                claim_id="c1", type="promise", who="user",
                what="X", source_person_ids={"p1", "p2"},
            ),
        ]
        matched, orphaned = _match_commitments_to_events(events, commitments)
        assert len(matched["cal1"]) == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Open commitments from claims table
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetOpenCommitments:
    def test_returns_open_commitments(self, store):
        now = int(time.time())
        eid = Event.make_id("mail", "commit_src")
        store.put_event(Event(
            id=eid, source="mail", source_id="commit_src",
            event_type="email", timestamp=now,
        ))
        store.put_person("person_bob", canonical_name="Bob")
        store.link_event_person(eid, "person_bob", "sender")

        claim = Claim(
            id="claim:commit1", event_ids=[eid],
            claim_type="commitment", subject="thread:123",
            predicate="commitment",
            object=json.dumps({
                "type": "promise", "who": "user", "what": "send report",
                "to_whom": "Bob", "status": "open", "priority": 2,
            }),
            confidence=0.9,
            provenance=ExtractionProvenance(model_id="test"),
        )
        store.put_claim(claim)

        commitments = _get_open_commitments(store)
        assert len(commitments) >= 1
        c = next(c for c in commitments if c.claim_id == "claim:commit1")
        assert c.what == "send report"
        assert c.who == "user"
        assert c.priority == 2
        assert "person_bob" in c.source_person_ids

    def test_excludes_superseded(self, store):
        now = int(time.time())
        eid = Event.make_id("mail", "commit_sup")
        store.put_event(Event(
            id=eid, source="mail", source_id="commit_sup",
            event_type="email", timestamp=now,
        ))
        claim = Claim(
            id="claim:old", event_ids=[eid],
            claim_type="commitment", subject="thread:456",
            predicate="commitment",
            object=json.dumps({"type": "task", "who": "user", "what": "old task", "status": "open"}),
            provenance=ExtractionProvenance(model_id="test"),
        )
        store.put_claim(claim)
        # Supersede it
        store.conn.execute(
            "UPDATE claims SET superseded_by = 'claim:new' WHERE id = 'claim:old'"
        )
        store.conn.commit()

        commitments = _get_open_commitments(store)
        assert not any(c.claim_id == "claim:old" for c in commitments)

    def test_empty_store(self, store):
        assert _get_open_commitments(store) == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event context gathering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGatherEventContext:
    def test_returns_recent_comms(self, store):
        now = int(time.time())
        store.put_person("person_alice", canonical_name="Alice")

        eid = Event.make_id("mail", "ctx_mail")
        store.put_event(Event(
            id=eid, source="mail", source_id="ctx_mail",
            event_type="email", timestamp=now - 3600,
            raw_content="Let's discuss the proposal",
            metadata={"subject": "Proposal", "is_from_me": False},
        ))
        store.link_event_person(eid, "person_alice", "sender")

        cal_event = CalendarEvent(
            event_id="cal1", title="Sync",
            start_ts=now + 86400, end_ts=now + 86400 + 3600,
            attendees=[{"person_id": "person_alice", "name": "Alice", "role": "attendee"}],
        )

        ctx = _gather_event_context(store, cal_event, lookback_days=30)
        assert len(ctx) >= 1
        assert ctx[0]["person_name"] == "Alice"
        assert ctx[0]["source"] == "mail"
        assert "proposal" in ctx[0]["snippet"].lower()

    def test_no_attendees(self, store):
        cal_event = CalendarEvent(
            event_id="cal1", title="Solo",
            start_ts=0, end_ts=3600,
        )
        ctx = _gather_event_context(store, cal_event)
        assert ctx == []

    def test_deduplicates_events(self, store):
        """Same event linked to two attendees should appear once."""
        now = int(time.time())
        store.put_person("person_alice", canonical_name="Alice")
        store.put_person("person_bob", canonical_name="Bob")

        eid = Event.make_id("mail", "shared_mail")
        store.put_event(Event(
            id=eid, source="mail", source_id="shared_mail",
            event_type="email", timestamp=now - 3600,
            raw_content="Group thread",
        ))
        store.link_event_person(eid, "person_alice", "sender")
        store.link_event_person(eid, "person_bob", "recipient")

        cal_event = CalendarEvent(
            event_id="cal1", title="Group",
            start_ts=now + 86400, end_ts=now + 86400 + 3600,
            attendees=[
                {"person_id": "person_alice", "name": "Alice", "role": "attendee"},
                {"person_id": "person_bob", "name": "Bob", "role": "attendee"},
            ],
        )
        ctx = _gather_event_context(store, cal_event)
        event_ids = [c.get("_event_id") for c in ctx]
        # Should only appear once despite two attendees
        assert len(ctx) == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Source snippet for orphaned commitments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetCommitmentSourceSnippet:
    def test_returns_snippet(self, store):
        now = int(time.time())
        eid = Event.make_id("mail", "snippet_src")
        store.put_event(Event(
            id=eid, source="mail", source_id="snippet_src",
            event_type="email", timestamp=now,
            raw_content="Please pay the invoice by Friday",
            metadata={"subject": "Invoice"},
        ))
        claim = Claim(
            id="claim:snippet", event_ids=[eid],
            claim_type="commitment", subject="thread:789",
            predicate="commitment",
            object=json.dumps({"type": "task", "who": "user", "what": "pay invoice", "status": "open"}),
            provenance=ExtractionProvenance(model_id="test"),
        )
        store.put_claim(claim)

        c = Commitment(
            claim_id="claim:snippet", type="task", who="user", what="pay invoice",
        )
        snippet = _get_commitment_source_snippet(store, c)
        assert "[mail]" in snippet
        assert "Invoice" in snippet
        assert "invoice" in snippet.lower()

    def test_no_source_event(self, store):
        c = Commitment(
            claim_id="claim:nonexistent", type="task", who="user", what="X",
        )
        snippet = _get_commitment_source_snippet(store, c)
        assert snippet == ""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt building
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBuildBriefingPrompt:
    def test_includes_profile(self):
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=[], orphan_snippets={},
            user_tz="America/Los_Angeles",
            profile={"name": "Alex", "role": "CTO"},
        )
        assert "Alex" in prompt
        assert "CTO" in prompt

    def test_includes_calendar_events(self):
        events = [
            CalendarEvent(
                event_id="cal1", title="Team Sync",
                start_ts=int(time.time()) + 86400,
                end_ts=int(time.time()) + 86400 + 3600,
                location="Zoom",
                attendees=[{"person_id": "p1", "name": "Alice", "role": "attendee"}],
            ),
        ]
        prompt = _build_briefing_prompt(
            events=events,
            event_commitments={"cal1": []},
            event_contexts={"cal1": []},
            orphaned=[], orphan_snippets={},
            user_tz="America/Los_Angeles",
            profile={},
        )
        assert "Team Sync" in prompt
        assert "Zoom" in prompt
        assert "Alice" in prompt

    def test_includes_matched_commitments(self):
        events = [
            CalendarEvent(
                event_id="cal1", title="Sync",
                start_ts=int(time.time()), end_ts=int(time.time()) + 3600,
            ),
        ]
        matched = [
            Commitment(
                claim_id="c1", type="promise", who="user",
                what="deliver the report", to_whom="Alice",
                deadline="2026-02-15",
            ),
        ]
        prompt = _build_briefing_prompt(
            events=events,
            event_commitments={"cal1": matched},
            event_contexts={"cal1": []},
            orphaned=[], orphan_snippets={},
            user_tz="America/Los_Angeles",
            profile={},
        )
        assert "deliver the report" in prompt
        assert "2026-02-15" in prompt

    def test_groups_orphaned_by_urgency(self):
        orphaned = [
            Commitment(
                claim_id="c1", type="promise", who="user",
                what="overdue task", deadline="2020-01-01",
            ),
            Commitment(
                claim_id="c2", type="task", who="user",
                what="no deadline task",
            ),
            Commitment(
                claim_id="c3", type="promise", who="user",
                what="future task", deadline="2099-12-31",
            ),
        ]
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=orphaned, orphan_snippets={},
            user_tz="America/Los_Angeles",
            profile={},
        )
        assert "OVERDUE" in prompt
        assert "overdue task" in prompt
        assert "NO DEADLINE" in prompt
        assert "no deadline task" in prompt
        assert "FUTURE DEADLINES" in prompt
        assert "future task" in prompt

    def test_includes_source_snippets(self):
        orphaned = [
            Commitment(
                claim_id="c1", type="task", who="user",
                what="pay bill", deadline="2020-01-01",
            ),
        ]
        snippets = {"c1": "[mail] Invoice — Please pay the $200 bill"}
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=orphaned, orphan_snippets=snippets,
            user_tz="America/Los_Angeles",
            profile={},
        )
        assert "Source:" in prompt
        assert "[mail] Invoice" in prompt

    def test_no_events_no_commitments(self):
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=[], orphan_snippets={},
            user_tz="America/Los_Angeles",
            profile={},
        )
        assert "No upcoming calendar events" in prompt
        assert "No unscheduled commitments" in prompt

    def test_includes_recent_comms(self):
        now = int(time.time())
        events = [
            CalendarEvent(
                event_id="cal1", title="Meeting",
                start_ts=now + 86400, end_ts=now + 86400 + 3600,
            ),
        ]
        comms = [{
            "person_name": "Alice",
            "source": "mail",
            "subject": "RE: Project update",
            "snippet": "Can we discuss the timeline?",
            "timestamp": now - 3600,
            "is_from_me": False,
        }]
        prompt = _build_briefing_prompt(
            events=events,
            event_commitments={"cal1": []},
            event_contexts={"cal1": comms},
            orphaned=[], orphan_snippets={},
            user_tz="America/Los_Angeles",
            profile={},
        )
        assert "Alice" in prompt
        assert "RE: Project update" in prompt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunBriefing:
    def _seed_store(self, store):
        """Seed store with calendar events, persons, claims, and comms."""
        now = int(time.time())
        tomorrow = now + 86400

        store.put_person("person_user", canonical_name="User", is_user=True)
        store.put_person("person_sam", canonical_name="Sam")

        # Calendar event tomorrow
        cal_eid = Event.make_id("calendar", "cal_tomorrow")
        store.put_event(Event(
            id=cal_eid, source="calendar", source_id="cal_tomorrow",
            event_type="calendar_event", timestamp=tomorrow,
            raw_content="Weekly sync",
            metadata={
                "title": "Weekly Sync", "location": "Zoom",
                "duration_minutes": 60,
            },
        ))
        store.link_event_person(cal_eid, "person_sam", "attendee")

        # Recent email from Sam
        mail_eid = Event.make_id("mail", "recent_mail")
        store.put_event(Event(
            id=mail_eid, source="mail", source_id="recent_mail",
            event_type="email", timestamp=now - 3600,
            raw_content="Can you send the proposal?",
            metadata={"subject": "Proposal", "is_from_me": False},
        ))
        store.link_event_person(mail_eid, "person_sam", "sender")
        store.link_event_person(mail_eid, "person_user", "recipient")

        # Commitment claim from that email
        claim = Claim(
            id="claim:commit1", event_ids=[mail_eid],
            claim_type="commitment", subject="thread:proposal",
            predicate="commitment",
            object=json.dumps({
                "type": "promise", "who": "user",
                "what": "send proposal to Sam",
                "to_whom": "Sam", "status": "open",
                "priority": 2, "deadline": "2026-02-14",
            }),
            confidence=0.9,
            provenance=ExtractionProvenance(model_id="test"),
        )
        store.put_claim(claim)

        # Orphaned commitment (no calendar match)
        orphan_eid = Event.make_id("mail", "orphan_mail")
        store.put_event(Event(
            id=orphan_eid, source="mail", source_id="orphan_mail",
            event_type="email", timestamp=now - 7200,
            raw_content="Please pay the Xfinity bill",
            metadata={"subject": "Xfinity bill"},
        ))
        store.put_person("person_xfinity", canonical_name="Xfinity")
        store.link_event_person(orphan_eid, "person_xfinity", "sender")

        orphan_claim = Claim(
            id="claim:orphan1", event_ids=[orphan_eid],
            claim_type="commitment", subject="thread:xfinity",
            predicate="commitment",
            object=json.dumps({
                "type": "task", "who": "user",
                "what": "pay Xfinity bill",
                "to_whom": "Xfinity", "status": "open",
                "priority": 3, "deadline": "2020-01-01",
            }),
            confidence=0.85,
            provenance=ExtractionProvenance(model_id="test"),
        )
        store.put_claim(orphan_claim)

    def test_full_pipeline(self, store):
        self._seed_store(store)
        mock_llm = MockLLMClient(
            responses={
                "SECTION 1": "# Weekly Briefing\n\nTest briefing content."
            },
        )
        result = run_briefing(
            store=store,
            llm_client=mock_llm,
            profile={"name": "Alex", "role": "CTO"},
            interactive=False,
        )
        assert "briefing" in result
        assert result["events_count"] >= 1
        assert result["commitments"]["total_open"] >= 2
        assert result["commitments"]["matched_to_calendar"] >= 1
        assert result["commitments"]["unscheduled"] >= 1
        assert result["commitments"]["overdue"] >= 1
        assert result["elapsed_s"] >= 0

    def test_no_calendar_events(self, store):
        store.put_person("person_user", is_user=True)
        mock_llm = MockLLMClient(
            responses={"SECTION 1": "# Weekly Briefing\nQuiet week."},
        )
        result = run_briefing(store=store, llm_client=mock_llm, profile={},
                              interactive=False)
        assert result["events_count"] == 0
        assert result["commitments"]["total_open"] == 0

    def test_empty_llm_response(self, store):
        self._seed_store(store)
        mock_llm = MockLLMClient()  # returns None from generate
        result = run_briefing(store=store, llm_client=mock_llm, profile={},
                              interactive=False)
        assert "failed" in result["briefing"].lower()

    def test_llm_receives_prompt(self, store):
        self._seed_store(store)
        mock_llm = MockLLMClient(
            responses={"SECTION 1": "Test briefing"},
        )
        run_briefing(store=store, llm_client=mock_llm, profile={},
                     interactive=False)
        # Verify LLM was called
        assert len(mock_llm.generate_calls) > 0
        # Prompt should contain calendar and commitment data
        prompt = mock_llm.generate_calls[0]["prompt"]
        assert "Weekly Sync" in prompt
        assert "send proposal" in prompt
        assert "Xfinity" in prompt

    def test_commitment_matched_to_event(self, store):
        """Commitment about Sam should match the meeting with Sam."""
        self._seed_store(store)
        mock_llm = MockLLMClient(
            responses={"SECTION 1": "Test"},
        )
        run_briefing(store=store, llm_client=mock_llm, profile={},
                     interactive=False)
        prompt = mock_llm.generate_calls[0]["prompt"]
        # The proposal commitment should appear under the calendar event section
        # (matched via person_sam)
        assert "Open commitments with these people" in prompt
        assert "send proposal" in prompt

    def test_orphaned_has_source_snippet(self, store):
        """Orphaned commitments should include source thread snippets."""
        self._seed_store(store)
        mock_llm = MockLLMClient(
            responses={"SECTION 1": "Test"},
        )
        run_briefing(store=store, llm_client=mock_llm, profile={},
                     interactive=False)
        prompt = mock_llm.generate_calls[0]["prompt"]
        # The Xfinity orphan should have source context
        assert "Source:" in prompt
        assert "Xfinity bill" in prompt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event facets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetEventFacets:
    def test_returns_facets(self, store):
        eid = Event.make_id("mail", "facet_src")
        store.put_event(Event(
            id=eid, source="mail", source_id="facet_src",
            event_type="email", timestamp=int(time.time()),
        ))
        store.put_annotation(Annotation(event_id=eid, facet="dollar_amount", value="472.50"))
        store.put_annotation(Annotation(event_id=eid, facet="person_mention", value="Alice"))
        store.put_annotation(Annotation(event_id=eid, facet="person_mention", value="Bob"))

        facets = _get_event_facets(store, eid)
        assert facets["dollar_amount"] == ["472.50"]
        assert set(facets["person_mention"]) == {"Alice", "Bob"}

    def test_no_annotations(self, store):
        eid = Event.make_id("mail", "no_facets")
        store.put_event(Event(
            id=eid, source="mail", source_id="no_facets",
            event_type="email", timestamp=int(time.time()),
        ))
        facets = _get_event_facets(store, eid)
        assert facets == {}

    def test_multiple_facet_types(self, store):
        eid = Event.make_id("mail", "multi_facet")
        store.put_event(Event(
            id=eid, source="mail", source_id="multi_facet",
            event_type="email", timestamp=int(time.time()),
        ))
        store.put_annotation(Annotation(event_id=eid, facet="email_category", value="financial"))
        store.put_annotation(Annotation(event_id=eid, facet="time_sensitive", value="true"))
        store.put_annotation(Annotation(event_id=eid, facet="dollar_amount", value="100.00"))

        facets = _get_event_facets(store, eid)
        assert len(facets) == 3
        assert "email_category" in facets
        assert "time_sensitive" in facets


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Execute graph queries
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestExecuteGraphQuery:
    def _seed_data(self, store):
        """Seed store with events, persons, annotations, and claims for query tests."""
        now = int(time.time())
        store.put_person("person_alice", canonical_name="Alice Smith")
        store.put_person("person_bob", canonical_name="Bob Jones")

        # Mail event from Alice
        eid1 = Event.make_id("mail", "query_mail1")
        store.put_event(Event(
            id=eid1, source="mail", source_id="query_mail1",
            event_type="email", timestamp=now - 3600,
            raw_content="Let's discuss the budget proposal for Q2",
            metadata={"subject": "Budget Proposal Q2"},
        ))
        store.link_event_person(eid1, "person_alice", "sender")

        # Slack event with "budget" keyword
        eid2 = Event.make_id("slack", "query_slack1")
        store.put_event(Event(
            id=eid2, source="slack", source_id="query_slack1",
            event_type="message", timestamp=now - 1800,
            raw_content="Budget numbers look good for the quarter",
            metadata={"subject": ""},
        ))
        store.link_event_person(eid2, "person_bob", "sender")

        # Annotations
        store.put_annotation(Annotation(
            event_id=eid1, facet="dollar_amount", value="5000.00",
        ))
        store.put_annotation(Annotation(
            event_id=eid1, facet="person_mention", value="Alice Smith",
        ))

        # Commitment claim
        claim = Claim(
            id="claim:budget_task", event_ids=[eid1],
            claim_type="commitment", subject="thread:budget",
            predicate="commitment",
            object=json.dumps({
                "type": "task", "who": "user",
                "what": "finalize budget numbers",
                "to_whom": "Alice", "status": "open", "priority": 2,
            }),
            confidence=0.9,
            provenance=ExtractionProvenance(model_id="test"),
        )
        store.put_claim(claim)

        # Thread events
        eid3 = Event.make_id("mail", "thread_msg1")
        store.put_event(Event(
            id=eid3, source="mail", source_id="thread_msg1",
            event_type="email", timestamp=now - 7200,
            raw_content="Starting the budget discussion",
            metadata={"subject": "Budget thread start", "thread_id": "thread:budget_discussion"},
        ))
        eid4 = Event.make_id("mail", "thread_msg2")
        store.put_event(Event(
            id=eid4, source="mail", source_id="thread_msg2",
            event_type="email", timestamp=now - 3600,
            raw_content="Updated budget figures attached",
            metadata={"subject": "Re: Budget thread start", "thread_id": "thread:budget_discussion"},
        ))

        return {"eid1": eid1, "eid2": eid2, "eid3": eid3, "eid4": eid4, "now": now}

    def test_person_events(self, store):
        self._seed_data(store)
        results = _execute_graph_query(
            {"query_type": "person_events", "params": {"person_name": "Alice", "days": 14}},
            store,
        )
        assert len(results) >= 1
        assert any("alice" in r.get("query_source", "").lower() for r in results)

    def test_person_events_empty_name(self, store):
        results = _execute_graph_query(
            {"query_type": "person_events", "params": {"person_name": "", "days": 14}},
            store,
        )
        assert results == []

    def test_topic_search(self, store):
        self._seed_data(store)
        results = _execute_graph_query(
            {"query_type": "topic_search", "params": {"keywords": ["budget"]}},
            store,
        )
        assert len(results) >= 2  # mail + slack events mention budget
        sources = {r["source"] for r in results}
        assert "mail" in sources

    def test_topic_search_no_keywords(self, store):
        results = _execute_graph_query(
            {"query_type": "topic_search", "params": {"keywords": []}},
            store,
        )
        assert results == []

    def test_commitment_search(self, store):
        self._seed_data(store)
        results = _execute_graph_query(
            {"query_type": "commitment_search", "params": {"keywords": ["budget"]}},
            store,
        )
        assert len(results) >= 1
        assert results[0]["what"] == "finalize budget numbers"

    def test_commitment_search_no_match(self, store):
        self._seed_data(store)
        results = _execute_graph_query(
            {"query_type": "commitment_search", "params": {"keywords": ["nonexistent"]}},
            store,
        )
        assert results == []

    def test_facet_query_by_type(self, store):
        self._seed_data(store)
        results = _execute_graph_query(
            {"query_type": "facet_query", "params": {"facet": "dollar_amount"}},
            store,
        )
        assert len(results) >= 1
        assert results[0]["value"] == "5000.00"
        assert results[0]["facet"] == "dollar_amount"

    def test_facet_query_by_value(self, store):
        self._seed_data(store)
        results = _execute_graph_query(
            {"query_type": "facet_query", "params": {"facet": "dollar_amount", "value": "5000.00"}},
            store,
        )
        assert len(results) >= 1
        assert results[0]["value"] == "5000.00"

    def test_facet_query_by_person(self, store):
        self._seed_data(store)
        results = _execute_graph_query(
            {"query_type": "facet_query", "params": {"facet": "person_mention", "person_name": "Alice"}},
            store,
        )
        assert len(results) >= 1

    def test_facet_query_empty_facet(self, store):
        results = _execute_graph_query(
            {"query_type": "facet_query", "params": {"facet": ""}},
            store,
        )
        assert results == []

    def test_thread_events(self, store):
        self._seed_data(store)
        results = _execute_graph_query(
            {"query_type": "thread_events", "params": {"thread_id": "thread:budget_discussion"}},
            store,
        )
        assert len(results) == 2
        # Should be ordered by timestamp ASC
        assert results[0]["snippet"].startswith("Starting")
        assert results[1]["snippet"].startswith("Updated")

    def test_thread_events_empty_id(self, store):
        results = _execute_graph_query(
            {"query_type": "thread_events", "params": {"thread_id": ""}},
            store,
        )
        assert results == []

    def test_unknown_query_type(self, store):
        results = _execute_graph_query(
            {"query_type": "nonexistent_type", "params": {}},
            store,
        )
        assert results == []

    def test_deduplicates_results(self, store):
        """topic_search with overlapping keywords should deduplicate."""
        self._seed_data(store)
        results = _execute_graph_query(
            {"query_type": "topic_search", "params": {"keywords": ["budget", "Budget"]}},
            store,
        )
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Anticipation pass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunAnticipationPass:
    def test_no_queries_needed(self, store):
        """When LLM returns empty channels, pass returns empties."""
        mock_llm = MockLLMClient(
            json_responses={
                "SECTION 1": {
                    "system_queries": [],
                    "web_searches": [],
                    "user_questions": [],
                    "reassurances": [{"event_subject": "Team sync", "note": "All good"}],
                },
            },
        )
        result = _run_anticipation_pass(
            prompt="SECTION 1: CALENDAR EVENTS\nSome events here",
            llm_client=mock_llm,
            store=store,
        )
        assert result["system_queries"] == []
        assert result["system_results"] == []
        assert result["web_searches"] == []
        assert result["web_results"] == []
        assert result["user_questions"] == []
        assert len(result["reassurances"]) == 1

    def test_executes_system_queries(self, store):
        """When LLM requests system queries, they should be executed."""
        now = int(time.time())
        store.put_person("person_jane", canonical_name="Jane Doe")
        eid = Event.make_id("mail", "jane_mail")
        store.put_event(Event(
            id=eid, source="mail", source_id="jane_mail",
            event_type="email", timestamp=now - 3600,
            raw_content="Jane's recent message about the project",
            metadata={"subject": "Project Update"},
        ))
        store.link_event_person(eid, "person_jane", "sender")

        mock_llm = MockLLMClient(
            json_responses={
                "SECTION 1": {
                    "system_queries": [
                        {
                            "query_type": "person_events",
                            "params": {"person_name": "Jane", "days": 14},
                            "reason": "Need recent messages with Jane",
                        },
                    ],
                    "web_searches": [],
                    "user_questions": [],
                    "reassurances": [],
                },
            },
        )
        result = _run_anticipation_pass(
            prompt="SECTION 1: CALENDAR EVENTS\nMeeting with Jane",
            llm_client=mock_llm,
            store=store,
        )
        assert len(result["system_queries"]) == 1
        assert len(result["system_results"]) >= 1

    def test_max_queries_enforced(self, store):
        """Should cap at TRIAGE_MAX_QUERIES even if LLM requests more."""
        mock_llm = MockLLMClient(
            json_responses={
                "SECTION 1": {
                    "system_queries": [
                        {"query_type": "topic_search", "params": {"keywords": [f"kw{i}"]}}
                        for i in range(10)
                    ],
                    "web_searches": [],
                    "user_questions": [],
                    "reassurances": [],
                },
            },
        )
        result = _run_anticipation_pass(
            prompt="SECTION 1: test",
            llm_client=mock_llm,
            store=store,
        )
        # Capped to TRIAGE_MAX_QUERIES (5)
        assert len(result["system_queries"]) <= 5
        assert len(mock_llm.generate_calls) == 1

    def test_non_dict_response_graceful(self, store):
        """Default mock now routes anticipation pass to MOCK_ANTICIPATION_RESPONSE."""
        mock_llm = MockLLMClient()
        result = _run_anticipation_pass(
            prompt="SECTION 1: data",
            llm_client=mock_llm,
            store=store,
        )
        # Mock routes "Anticipation Engine" system prompt → MOCK_ANTICIPATION_RESPONSE
        assert len(result["system_queries"]) == 1
        assert len(result["user_questions"]) == 1

    def test_list_response_unwrapped(self, store):
        """If LLM returns a list with one dict, it should be unwrapped."""
        mock_llm = MockLLMClient(
            json_responses={
                "SECTION 1": [
                    {
                        "system_queries": [],
                        "web_searches": [{"query": "test", "reason": "check"}],
                        "user_questions": [],
                        "reassurances": [],
                    }
                ],
            },
        )
        result = _run_anticipation_pass(
            prompt="SECTION 1: test",
            llm_client=mock_llm,
            store=store,
        )
        assert len(result["web_searches"]) == 1

    def test_multiple_query_types(self, store):
        """Anticipation can request different query types."""
        now = int(time.time())
        store.put_person("person_alice", canonical_name="Alice Smith")
        eid = Event.make_id("mail", "alice_triage")
        store.put_event(Event(
            id=eid, source="mail", source_id="alice_triage",
            event_type="email", timestamp=now - 3600,
            raw_content="Discussing the quarterly budget review",
            metadata={"subject": "Q2 Budget"},
        ))
        store.link_event_person(eid, "person_alice", "sender")
        store.put_annotation(Annotation(
            event_id=eid, facet="dollar_amount", value="15000",
        ))

        mock_llm = MockLLMClient(
            json_responses={
                "SECTION 1": {
                    "system_queries": [
                        {
                            "query_type": "person_events",
                            "params": {"person_name": "Alice", "days": 7},
                            "reason": "Recent messages with Alice",
                        },
                        {
                            "query_type": "facet_query",
                            "params": {"facet": "dollar_amount"},
                            "reason": "Check dollar amounts across events",
                        },
                    ],
                    "web_searches": [],
                    "user_questions": [
                        {
                            "event_subject": "Budget Review",
                            "question": "Do you have the Q2 numbers ready?",
                            "category": "preparation",
                        },
                    ],
                    "reassurances": [],
                },
            },
        )
        result = _run_anticipation_pass(
            prompt="SECTION 1: CALENDAR EVENTS\nMeeting data",
            llm_client=mock_llm,
            store=store,
        )
        assert len(result["system_results"]) >= 2
        assert len(result["user_questions"]) == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt building with enrichment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBuildBriefingPromptEnriched:
    def test_includes_annotations(self):
        events = [
            CalendarEvent(
                event_id="cal1", title="Finance Review",
                start_ts=int(time.time()) + 86400,
                end_ts=int(time.time()) + 86400 + 3600,
            ),
        ]
        event_facets = {
            "cal1": {
                "dollar_amount": ["472.50", "1200.00"],
                "person_mention": ["Alice", "Bob"],
                "time_sensitive": ["true"],
            },
        }
        prompt = _build_briefing_prompt(
            events=events,
            event_commitments={"cal1": []},
            event_contexts={"cal1": []},
            orphaned=[], orphan_snippets={},
            user_tz="America/Los_Angeles",
            profile={},
            event_facets=event_facets,
        )
        assert "$472.50" in prompt
        assert "$1200.00" in prompt
        assert "Alice" in prompt
        assert "Bob" in prompt
        assert "Time-sensitive: yes" in prompt

    def test_includes_cross_source_links(self):
        events = [
            CalendarEvent(
                event_id="cal1", title="Team Sync",
                start_ts=int(time.time()) + 86400,
                end_ts=int(time.time()) + 86400 + 3600,
            ),
        ]
        cross_source_links = {
            "cal1": [
                {
                    "event_id": "mail_123",
                    "source": "mail",
                    "timestamp": int(time.time()) - 3600,
                    "relationship": "shared_dollar_amount",
                    "shared_signal": "$472.50",
                    "subject": "Invoice from vendor",
                },
            ],
        }
        prompt = _build_briefing_prompt(
            events=events,
            event_commitments={"cal1": []},
            event_contexts={"cal1": []},
            orphaned=[], orphan_snippets={},
            user_tz="America/Los_Angeles",
            profile={},
            cross_source_links=cross_source_links,
        )
        assert "Cross-source links" in prompt
        assert "[mail]" in prompt
        assert "shared_dollar_amount" in prompt
        assert "$472.50" in prompt

    def test_includes_triage_assessment(self):
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=[], orphan_snippets={},
            user_tz="America/Los_Angeles",
            profile={},
            triage_assessment="Context is strong for Monday meetings, weak for Wednesday.",
        )
        assert "SECTION 3" in prompt
        assert "Context is strong for Monday meetings" in prompt

    def test_includes_additional_context(self):
        additional = [
            {
                "id": "evt_1",
                "source": "mail",
                "date": "2026-02-12 10:00",
                "subject": "Budget follow-up",
                "snippet": "Please review the updated numbers",
                "query_source": "person_events:alice",
            },
            {
                "id": "claim_1",
                "type": "task",
                "what": "review quarterly budget",
                "who": "user",
                "to_whom": "Finance Team",
                "deadline": "2026-02-20",
                "priority": 2,
                "query_source": "commitment_search:budget",
            },
        ]
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=[], orphan_snippets={},
            user_tz="America/Los_Angeles",
            profile={},
            triage_assessment="Need budget context",
            additional_context=additional,
        )
        assert "Additional retrieved items (2)" in prompt
        assert "person_events:alice" in prompt
        assert "commitment_search:budget" in prompt
        assert "review quarterly budget" in prompt

    def test_no_section3_without_enrichment(self):
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=[], orphan_snippets={},
            user_tz="America/Los_Angeles",
            profile={},
        )
        assert "SECTION 3" not in prompt

    def test_facet_query_formatting(self):
        """Facet query results format correctly in SECTION 3."""
        additional = [
            {
                "event_id": "evt_1",
                "facet": "dollar_amount",
                "value": "750.00",
                "source": "mail",
                "date": "2026-02-10 14:30",
                "subject": "Wire transfer",
                "query_source": "facet_query:dollar_amount",
            },
        ]
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=[], orphan_snippets={},
            user_tz="America/Los_Angeles",
            profile={},
            triage_assessment="Checking financials",
            additional_context=additional,
        )
        assert "facet_query:dollar_amount" in prompt
        assert "750.00" in prompt
        assert "Wire transfer" in prompt

    def test_cross_source_links_formatting(self):
        """Cross-source claim results format correctly in SECTION 3."""
        additional = [
            {
                "claim_id": "claim_cs1",
                "claim_type": "cross_source_dollar_cluster",
                "detail": {"amount": "472.50", "sources": ["mail", "calendar"]},
                "confidence": 0.85,
                "query_source": "cross_source_links:alice",
            },
        ]
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=[], orphan_snippets={},
            user_tz="America/Los_Angeles",
            profile={},
            triage_assessment="Cross-source check",
            additional_context=additional,
        )
        assert "cross_source_links:alice" in prompt
        assert "cross_source_dollar_cluster" in prompt
        assert "0.85" in prompt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Direction and staleness tags in prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDirectionStalenessInPrompt:
    def test_group_ask_tag(self):
        """group_ask direction adds [GROUP ASK] tag to orphaned commitment."""
        orphaned = [
            Commitment(
                claim_id="c1", type="inbound_request", who="user",
                what="pick up supplies", deadline="2020-01-01",
                to_whom="group:Magnolia Indians",
                direction="group_ask", staleness_signal="none",
            ),
        ]
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=orphaned, orphan_snippets={},
            user_tz="America/Los_Angeles", profile={},
        )
        assert "GROUP ASK" in prompt
        assert "may not be user's responsibility" in prompt

    def test_direct_ask_no_group_tag(self):
        """direct_ask direction does NOT add group tag."""
        orphaned = [
            Commitment(
                claim_id="c1", type="inbound_request", who="user",
                what="send report", deadline="2020-01-01",
                direction="direct_ask", staleness_signal="none",
            ),
        ]
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=orphaned, orphan_snippets={},
            user_tz="America/Los_Angeles", profile={},
        )
        assert "GROUP ASK" not in prompt

    def test_overdue_no_followup_tag(self):
        """staleness_signal=overdue_no_followup adds [POSSIBLY RESOLVED] tag."""
        orphaned = [
            Commitment(
                claim_id="c1", type="deadline", who="user",
                what="pay bill", deadline="2020-01-01",
                direction="direct_ask", staleness_signal="overdue_no_followup",
            ),
        ]
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=orphaned, orphan_snippets={},
            user_tz="America/Los_Angeles", profile={},
        )
        assert "POSSIBLY RESOLVED" in prompt
        assert "overdue with no follow-up" in prompt

    def test_group_broadcast_tag(self):
        """staleness_signal=group_broadcast adds [GROUP BROADCAST] tag."""
        orphaned = [
            Commitment(
                claim_id="c1", type="inbound_request", who="user",
                what="help with event", deadline="2020-01-01",
                direction="group_ask", staleness_signal="group_broadcast",
            ),
        ]
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=orphaned, orphan_snippets={},
            user_tz="America/Los_Angeles", profile={},
        )
        assert "GROUP BROADCAST" in prompt

    def test_old_thread_tag(self):
        """staleness_signal=old_thread adds [OLD THREAD] tag."""
        orphaned = [
            Commitment(
                claim_id="c1", type="follow_up", who="user",
                what="check status", deadline="2020-01-01",
                direction="ambiguous", staleness_signal="old_thread",
            ),
        ]
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=orphaned, orphan_snippets={},
            user_tz="America/Los_Angeles", profile={},
        )
        assert "OLD THREAD" in prompt

    def test_no_tags_for_defaults(self):
        """Default direction=ambiguous and staleness_signal=none produce no tags."""
        orphaned = [
            Commitment(
                claim_id="c1", type="inbound_request", who="user",
                what="send proposal", deadline="2020-01-01",
            ),
        ]
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=orphaned, orphan_snippets={},
            user_tz="America/Los_Angeles", profile={},
        )
        assert "GROUP ASK" not in prompt
        assert "POSSIBLY RESOLVED" not in prompt
        assert "GROUP BROADCAST" not in prompt
        assert "OLD THREAD" not in prompt

    def test_self_directed_tag(self):
        """direction=self_directed adds [SELF-DIRECTED] tag."""
        orphaned = [
            Commitment(
                claim_id="c1", type="user_commitment", who="user",
                what="clean up code", deadline="2020-01-01",
                direction="self_directed",
            ),
        ]
        prompt = _build_briefing_prompt(
            events=[], event_commitments={}, event_contexts={},
            orphaned=orphaned, orphan_snippets={},
            user_tz="America/Los_Angeles", profile={},
        )
        assert "SELF-DIRECTED" in prompt

    def test_open_commitments_reads_direction(self, store):
        """_get_open_commitments reads direction from claims.object JSON."""
        now = int(time.time())
        eid = Event.make_id("mail", "dir_src")
        store.put_event(Event(
            id=eid, source="mail", source_id="dir_src",
            event_type="email", timestamp=now,
        ))
        claim = Claim(
            id="claim:dir1", event_ids=[eid],
            claim_type="commitment", subject="thread:test",
            predicate="commitment",
            object=json.dumps({
                "type": "inbound_request", "who": "user",
                "what": "test task", "status": "open",
                "direction": "group_ask",
                "staleness_signal": "group_broadcast",
            }),
            confidence=0.9,
            provenance=ExtractionProvenance(model_id="test"),
        )
        store.put_claim(claim)
        commitments = _get_open_commitments(store)
        c = next(c for c in commitments if c.claim_id == "claim:dir1")
        assert c.direction == "group_ask"
        assert c.staleness_signal == "group_broadcast"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full pipeline with anticipation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunBriefingWithAnticipation:
    def _seed_store(self, store):
        """Seed store with calendar + comms + annotations."""
        now = int(time.time())
        tomorrow = now + 86400

        store.put_person("person_user", canonical_name="User", is_user=True)
        store.put_person("person_kai", canonical_name="Kai")

        cal_eid = Event.make_id("calendar", "cal_anticipation")
        store.put_event(Event(
            id=cal_eid, source="calendar", source_id="cal_anticipation",
            event_type="calendar_event", timestamp=tomorrow,
            raw_content="Budget review",
            metadata={"title": "Budget Review", "duration_minutes": 60},
        ))
        store.link_event_person(cal_eid, "person_kai", "attendee")

        # Annotate the calendar event
        store.put_annotation(Annotation(
            event_id=cal_eid, facet="dollar_amount", value="5000.00",
        ))

        # Mail from Kai
        mail_eid = Event.make_id("mail", "anticipation_mail")
        store.put_event(Event(
            id=mail_eid, source="mail", source_id="anticipation_mail",
            event_type="email", timestamp=now - 3600,
            raw_content="Here are the budget numbers for review",
            metadata={"subject": "Budget Numbers", "is_from_me": False},
        ))
        store.link_event_person(mail_eid, "person_kai", "sender")

        claim = Claim(
            id="claim:anticipation_commit", event_ids=[mail_eid],
            claim_type="commitment", subject="thread:budget",
            predicate="commitment",
            object=json.dumps({
                "type": "task", "who": "user",
                "what": "review budget numbers",
                "to_whom": "Kai", "status": "open", "priority": 2,
            }),
            confidence=0.9,
            provenance=ExtractionProvenance(model_id="test"),
        )
        store.put_claim(claim)

    def test_anticipation_runs_and_returns_info(self, store):
        """run_briefing with anticipation enabled returns anticipation info."""
        self._seed_store(store)
        mock_llm = MockLLMClient(
            responses={"SECTION 1": "# Briefing\nTest with anticipation"},
            json_responses={
                "SECTION 1": {
                    "system_queries": [],
                    "web_searches": [],
                    "user_questions": [],
                    "reassurances": [{"event_subject": "Budget", "note": "Good context"}],
                },
            },
        )
        result = run_briefing(
            store=store,
            llm_client=mock_llm,
            profile={"name": "Test"},
            interactive=False,
        )
        assert "anticipation" in result
        assert result["anticipation"]["system_queries"] == 0
        assert result["anticipation"]["reassurances"] == 1

    def test_skip_anticipation(self, store):
        """skip_anticipation=True should skip the anticipation pass."""
        self._seed_store(store)
        mock_llm = MockLLMClient(
            responses={"SECTION 1": "# Briefing\nNo anticipation"},
        )
        result = run_briefing(
            store=store,
            llm_client=mock_llm,
            profile={},
            skip_anticipation=True,
        )
        assert result["anticipation"]["system_queries"] == 0
        assert result["anticipation"]["web_searches"] == 0
        # No anticipation JSON call
        json_calls = [c for c in mock_llm.generate_calls if c["format_json"]]
        assert len(json_calls) == 0

    def test_anticipation_enriches_prompt(self, store):
        """When anticipation returns system queries, results feed into briefing prompt."""
        self._seed_store(store)
        mock_llm = MockLLMClient(
            responses={"SECTION 1": "# Briefing\nEnriched"},
            json_responses={
                "SECTION 1": {
                    "system_queries": [
                        {
                            "query_type": "commitment_search",
                            "params": {"keywords": ["budget"]},
                            "reason": "Check for budget-related commitments",
                        },
                    ],
                    "web_searches": [],
                    "user_questions": [],
                    "reassurances": [],
                },
            },
        )
        result = run_briefing(
            store=store,
            llm_client=mock_llm,
            profile={},
            interactive=False,
        )
        assert result["anticipation"]["system_results"] >= 1
        # The briefing generate call (non-JSON) should contain SECTION 3
        non_json_calls = [c for c in mock_llm.generate_calls if not c["format_json"]]
        briefing_prompt = non_json_calls[-1]["prompt"]
        assert "SECTION 3" in briefing_prompt
        assert "review budget numbers" in briefing_prompt

    def test_facets_in_prompt(self, store):
        """Calendar event annotations appear in the final prompt."""
        self._seed_store(store)
        mock_llm = MockLLMClient(
            responses={"SECTION 1": "# Briefing\nTest"},
        )
        result = run_briefing(
            store=store,
            llm_client=mock_llm,
            profile={},
            skip_anticipation=True,
        )
        # The briefing generate call (non-JSON) should contain facets
        non_json_calls = [c for c in mock_llm.generate_calls if not c["format_json"]]
        briefing_prompt = non_json_calls[-1]["prompt"]
        assert "$5000.00" in briefing_prompt

    def test_no_anticipation_without_events(self, store):
        """Anticipation should be skipped when there are no calendar events."""
        store.put_person("person_user", is_user=True)
        mock_llm = MockLLMClient(
            responses={"SECTION 1": "# Briefing\nQuiet week"},
        )
        result = run_briefing(store=store, llm_client=mock_llm, profile={})
        assert result["anticipation"]["system_queries"] == 0
        # No json generate calls (anticipation skipped because no events)
        json_calls = [c for c in mock_llm.generate_calls if c["format_json"]]
        assert len(json_calls) == 0
