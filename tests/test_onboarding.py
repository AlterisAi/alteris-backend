"""Tests for the onboarding summary MCP tool and KG tool registration.

Covers: onboarding summary structure, top contacts, commitments,
beliefs, source coverage, sender rule suggestions, category suggestions,
cross-source discoveries, and tool registration.
"""

import json
import time

import pytest

from alteris.store import LayeredGraphStore
from alteris.models import Event, Claim, Belief, Modality, ExtractionProvenance, ExtractionMethod
from alteris.models import BeliefType, EpistemicLevel, BeliefStatus
from alteris.privacy import SensitivityLevel


@pytest.fixture
def store():
    s = LayeredGraphStore(":memory:")
    _ = s.conn  # Initialize schema
    return s


def _make_event(id, source, source_id, timestamp, participants=(), raw_content="",
                metadata=None, event_type="message"):
    return Event(
        id=id, source=source, source_id=source_id,
        event_type=event_type, timestamp=timestamp,
        participants=tuple(participants),
        raw_content=raw_content,
        metadata=metadata or {},
        sensitivity=SensitivityLevel.PUBLIC,
        created_at=timestamp,
    )


def _make_commitment_claim(id, what, who="user", to_whom="", deadline=None,
                           status="open", priority=3, direction="outbound",
                           confidence=0.8):
    now = int(time.time())
    obj = json.dumps({
        "what": what, "who": who, "to_whom": to_whom,
        "deadline": deadline, "status": status,
        "priority": priority, "direction": direction,
        "type": "task",
    })
    return Claim(
        id=id, event_ids=["evt_1"], claim_type="commitment",
        subject=who, predicate="has_commitment", object=obj,
        confidence=confidence, modality=Modality.ASSERTED,
        provenance=ExtractionProvenance(
            model_id="test", prompt_version="v1",
            extraction_method=ExtractionMethod.CLOUD_MODEL,
            extracted_at=now,
        ),
        created_at=now,
    )


def _make_belief(id, subject, summary, belief_type=BeliefType.ENTITY,
                 confidence=0.8, data=None):
    now = int(time.time())
    return Belief(
        id=id, belief_type=belief_type, subject=subject,
        summary=summary,
        data=data or {},
        epistemic_level=EpistemicLevel.INFERENCE,
        confidence=confidence,
        source_claims=["claim_1"],
        status=BeliefStatus.ACTIVE,
        created_at=now, updated_at=now,
    )


def _seed_basic_kg(store):
    """Seed a KG with realistic data for onboarding tests."""
    now = int(time.time())

    # Events from multiple sources
    events = [
        _make_event("e1", "mail", "m1", now - 3600, ["alice@test.com"],
                     "Meeting notes from today", {"sender": "alice@test.com", "subject": "Notes"}),
        _make_event("e2", "mail", "m2", now - 7200, ["bob@work.com"],
                     "Quarterly report", {"sender": "bob@work.com", "subject": "Q4 Report"}),
        _make_event("e3", "imessage", "im1", now - 1800, ["alice@test.com"],
                     "Hey, did you see the doc?"),
        _make_event("e4", "calendar", "c1", now + 86400, ["alice@test.com", "bob@work.com"],
                     "Team standup", {"title": "Team Standup", "location": "Zoom"},
                     event_type="calendar_event"),
        _make_event("e5", "whatsapp", "w1", now - 600, ["carol@home.com"],
                     "Picking up groceries"),
        _make_event("e6", "mail", "m3", now - 500, ["dave@vendor.com"],
                     "Invoice attached", {"sender": "dave@vendor.com", "subject": "Invoice #123"}),
        _make_event("e7", "mail", "m4", now - 400, ["alice@test.com"],
                     "Follow up on deck", {"sender": "alice@test.com", "subject": "Re: Deck"}),
        _make_event("e8", "slack", "s1", now - 300, ["bob@work.com"],
                     "PR ready for review"),
    ]
    for e in events:
        store.put_event(e)

    # Persons
    store.put_person("p_alice", "Alice Chen", sources=["mail", "imessage", "calendar"])
    store.put_person("p_bob", "Bob Smith", sources=["mail", "calendar", "slack"])
    store.put_person("p_carol", "Carol Davis", sources=["whatsapp"])
    store.put_person("p_dave", "Dave Vendor", sources=["mail"])
    store.put_person("p_user", "User", is_user=True, sources=["mail", "imessage"])

    # Event-person links
    store.link_event_person("e1", "p_alice", "sender")
    store.link_event_person("e2", "p_bob", "sender")
    store.link_event_person("e3", "p_alice", "sender")
    store.link_event_person("e4", "p_alice", "attendee")
    store.link_event_person("e4", "p_bob", "attendee")
    store.link_event_person("e5", "p_carol", "sender")
    store.link_event_person("e6", "p_dave", "sender")
    store.link_event_person("e7", "p_alice", "sender")
    store.link_event_person("e8", "p_bob", "sender")

    # Commitments
    commitments = [
        _make_commitment_claim("cm1", "Send quarterly deck", deadline="2026-02-15",
                                priority=1, direction="outbound"),
        _make_commitment_claim("cm2", "Review PR", who="Bob", deadline="2026-02-18",
                                priority=2),
        _make_commitment_claim("cm3", "Pick up groceries", deadline=None, priority=3),
    ]
    for c in commitments:
        store.put_claim(c)

    # Beliefs
    beliefs = [
        _make_belief("b1", "Alice Chen", "Product manager, primary collaborator",
                      data={"domain": "work"}),
        _make_belief("b2", "Bob Smith", "Engineering lead, reviews PRs",
                      belief_type=BeliefType.ENTITY, data={"domain": "work"}),
        _make_belief("b3", "Q4 Report", "Quarterly report due for board review",
                      belief_type=BeliefType.FACT, confidence=0.9,
                      data={"domain": "finance"}),
        _make_belief("b4", "home maintenance", "Gutters need cleaning before spring",
                      belief_type=BeliefType.OBSERVATION, confidence=0.5,
                      data={"domain": "home"}),
    ]
    for b in beliefs:
        store.put_belief(b)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Onboarding Summary Tool
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestOnboardingSummaryStructure:
    """Test the overall shape of the onboarding summary response."""

    def test_empty_kg(self, store):
        from alteris.mcp_tools.onboarding_tools import handle_alteris_onboarding_summary
        result = handle_alteris_onboarding_summary(store)

        assert "stats" in result
        assert "top_contacts" in result
        assert "commitments" in result
        assert "notable_beliefs" in result
        assert "upcoming_events" in result
        assert "source_coverage" in result
        assert "suggested_sender_rules" in result
        assert "suggested_categories" in result
        assert "cross_source_discoveries" in result

    def test_empty_stats_zeroes(self, store):
        from alteris.mcp_tools.onboarding_tools import handle_alteris_onboarding_summary
        result = handle_alteris_onboarding_summary(store)

        stats = result["stats"]
        assert stats["total_events"] == 0
        assert stats["total_persons"] == 0
        assert stats["total_beliefs"] == 0

    def test_populated_kg(self, store):
        from alteris.mcp_tools.onboarding_tools import handle_alteris_onboarding_summary
        _seed_basic_kg(store)
        result = handle_alteris_onboarding_summary(store)

        stats = result["stats"]
        assert stats["total_events"] == 8
        assert stats["total_persons"] == 5
        assert stats["total_beliefs"] == 4
        assert "mail" in stats["events_by_source"]


class TestTopContacts:
    """Test top contacts extraction from event-person links."""

    def test_no_contacts_empty_kg(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_top_contacts
        assert _get_top_contacts(store) == []

    def test_contacts_ordered_by_event_count(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_top_contacts
        _seed_basic_kg(store)
        contacts = _get_top_contacts(store)

        assert len(contacts) >= 3
        # Alice has 3 events (e1, e3, e7 + e4 attendee = 4), Bob has 3 (e2, e4, e8)
        assert contacts[0]["name"] == "Alice Chen"
        assert contacts[0]["event_count"] >= 3

    def test_cross_source_flag(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_top_contacts
        _seed_basic_kg(store)
        contacts = _get_top_contacts(store)

        alice = next(c for c in contacts if c["name"] == "Alice Chen")
        assert alice["cross_source"] is True
        assert len(alice["sources"]) >= 2

    def test_user_excluded(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_top_contacts
        _seed_basic_kg(store)
        contacts = _get_top_contacts(store)

        names = [c["name"] for c in contacts]
        assert "User" not in names

    def test_limit_respected(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_top_contacts
        _seed_basic_kg(store)
        contacts = _get_top_contacts(store, limit=2)
        assert len(contacts) <= 2


class TestCommitmentSummary:
    """Test commitment summary aggregation."""

    def test_empty_commitments(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_commitment_summary
        result = _get_commitment_summary(store, "2026-02-17")

        assert result["total_open"] == 0
        assert result["overdue_count"] == 0
        assert result["sample"] == []

    def test_overdue_detected(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_commitment_summary
        _seed_basic_kg(store)
        result = _get_commitment_summary(store, "2026-02-17")

        assert result["total_open"] >= 2
        assert result["overdue_count"] >= 1  # cm1 deadline 2026-02-15 is overdue

    def test_high_priority_filtered(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_commitment_summary
        _seed_basic_kg(store)
        result = _get_commitment_summary(store, "2026-02-17")

        assert len(result["high_priority"]) >= 1
        for item in result["high_priority"]:
            assert item["priority"] <= 2

    def test_sample_sorted_by_priority(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_commitment_summary
        _seed_basic_kg(store)
        result = _get_commitment_summary(store, "2026-02-17")

        priorities = [i["priority"] for i in result["sample"]]
        assert priorities == sorted(priorities)


class TestNotableBeliefs:
    """Test notable belief selection."""

    def test_empty_beliefs(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_notable_beliefs
        assert _get_notable_beliefs(store) == []

    def test_beliefs_filtered_by_confidence(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_notable_beliefs
        _seed_basic_kg(store)
        beliefs = _get_notable_beliefs(store)

        # b4 has confidence 0.5, below 0.6 threshold
        summaries = [b["summary"] for b in beliefs]
        assert "Gutters need cleaning before spring" not in summaries

    def test_entities_before_observations(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_notable_beliefs
        _seed_basic_kg(store)
        beliefs = _get_notable_beliefs(store)

        types = [b["type"] for b in beliefs]
        # Entities should come before facts/observations
        if "entity" in types and "observation" in types:
            assert types.index("entity") < types.index("observation")


class TestUpcomingEvents:
    """Test upcoming calendar event extraction."""

    def test_no_calendar_events(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_upcoming_events
        assert _get_upcoming_events(store, int(time.time())) == []

    def test_finds_upcoming_events(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_upcoming_events
        _seed_basic_kg(store)
        now = int(time.time())
        events = _get_upcoming_events(store, now)

        assert len(events) >= 1
        assert events[0]["title"] == "Team Standup"


class TestSourceCoverage:
    """Test source coverage description."""

    def test_empty_sources(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_source_coverage
        assert _get_source_coverage({}) == []

    def test_sources_ordered_by_count(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_source_coverage
        _seed_basic_kg(store)
        stats = store.stats()
        coverage = _get_source_coverage(stats)

        counts = [c["event_count"] for c in coverage]
        assert counts == sorted(counts, reverse=True)

    def test_source_has_icon(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_source_coverage
        stats = {"events_by_source": {"mail": 100, "imessage": 50}}
        coverage = _get_source_coverage(stats)

        mail = next(c for c in coverage if c["source"] == "mail")
        assert mail["icon"] == "envelope.fill"
        assert mail["label"] == "Mail"


class TestSuggestSenderRules:
    """Test sender rule suggestions."""

    def test_no_events(self, store):
        from alteris.mcp_tools.onboarding_tools import _suggest_sender_rules
        assert _suggest_sender_rules(store) == []

    def test_suggests_from_frequent_senders(self, store):
        from alteris.mcp_tools.onboarding_tools import _suggest_sender_rules
        _seed_basic_kg(store)
        rules = _suggest_sender_rules(store)

        # alice@test.com appears 3 times as sender (e1, e7 from mail)
        patterns = [r["pattern"] for r in rules]
        assert "alice@test.com" in patterns

    def test_top_senders_get_p1(self, store):
        from alteris.mcp_tools.onboarding_tools import _suggest_sender_rules
        _seed_basic_kg(store)
        rules = _suggest_sender_rules(store)

        if rules:
            assert rules[0]["priority"] == "P1"

    def test_max_10_suggestions(self, store):
        from alteris.mcp_tools.onboarding_tools import _suggest_sender_rules
        _seed_basic_kg(store)
        rules = _suggest_sender_rules(store)
        assert len(rules) <= 10


class TestSuggestCategories:
    """Test category suggestions from beliefs."""

    def test_default_categories_always_present(self, store):
        from alteris.mcp_tools.onboarding_tools import _suggest_categories
        categories = _suggest_categories(store)

        names = [c["name"] for c in categories]
        assert "work" in names
        assert "personal" in names
        assert "finance" in names

    def test_categories_with_evidence_ranked_first(self, store):
        from alteris.mcp_tools.onboarding_tools import _suggest_categories
        _seed_basic_kg(store)
        categories = _suggest_categories(store)

        # "work" should rank high (2 beliefs about work domain)
        assert categories[0]["kg_evidence_count"] > 0

    def test_each_has_icon(self, store):
        from alteris.mcp_tools.onboarding_tools import _suggest_categories
        categories = _suggest_categories(store)

        for cat in categories:
            assert "icon" in cat
            assert cat["icon"]  # non-empty


class TestCrossSourceDiscoveries:
    """Test cross-source discovery extraction."""

    def test_no_cross_source(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_cross_source_discoveries
        assert _get_cross_source_discoveries(store) == []

    def test_finds_cross_source_people(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_cross_source_discoveries
        _seed_basic_kg(store)
        discoveries = _get_cross_source_discoveries(store)

        # Alice appears in mail + imessage + calendar, Bob in mail + calendar + slack
        assert len(discoveries) >= 2
        names = [d["person"] for d in discoveries]
        assert "Alice Chen" in names

    def test_discovery_has_description(self, store):
        from alteris.mcp_tools.onboarding_tools import _get_cross_source_discoveries
        _seed_basic_kg(store)
        discoveries = _get_cross_source_discoveries(store)

        for d in discoveries:
            assert "description" in d
            assert "Appears across" in d["description"]
            assert len(d["sources"]) > 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tool Registration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestOnboardingToolRegistration:
    """Verify the onboarding tool registers correctly."""

    def test_tool_registered(self):
        from alteris.mcp_tools import ensure_tools_loaded, get_tool
        ensure_tools_loaded()

        t = get_tool("alteris_onboarding_summary")
        assert t is not None
        assert t.permission == "read"

    def test_tool_in_all_tools(self):
        from alteris.mcp_tools import ensure_tools_loaded, get_all_tools
        ensure_tools_loaded()

        names = {t.name for t in get_all_tools()}
        assert "alteris_onboarding_summary" in names

    def test_handler_callable(self):
        from alteris.mcp_tools import ensure_tools_loaded, get_tool
        ensure_tools_loaded()

        t = get_tool("alteris_onboarding_summary")
        assert callable(t.handler)

    def test_handler_returns_dict(self, store):
        from alteris.mcp_tools.onboarding_tools import handle_alteris_onboarding_summary
        result = handle_alteris_onboarding_summary(store)
        assert isinstance(result, dict)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# End-to-end integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestOnboardingEndToEnd:
    """Full integration test: seed KG, call summary, validate all sections."""

    def test_full_summary_with_data(self, store):
        from alteris.mcp_tools.onboarding_tools import handle_alteris_onboarding_summary
        _seed_basic_kg(store)
        result = handle_alteris_onboarding_summary(store)

        # Stats populated
        assert result["stats"]["total_events"] == 8
        assert result["stats"]["total_persons"] == 5

        # Contacts found
        assert len(result["top_contacts"]) >= 3

        # Commitments found
        assert result["commitments"]["total_open"] >= 2

        # Beliefs found (only >=0.6 confidence, so b4 excluded)
        assert len(result["notable_beliefs"]) >= 2

        # Source coverage populated
        sources = [c["source"] for c in result["source_coverage"]]
        assert "mail" in sources

        # Categories always present
        assert len(result["suggested_categories"]) >= 4

        # Cross-source found
        assert len(result["cross_source_discoveries"]) >= 1

    def test_result_json_serializable(self, store):
        from alteris.mcp_tools.onboarding_tools import handle_alteris_onboarding_summary
        _seed_basic_kg(store)
        result = handle_alteris_onboarding_summary(store)

        # Must be JSON serializable for MCP transport
        serialized = json.dumps(result)
        assert len(serialized) > 100
        deserialized = json.loads(serialized)
        assert deserialized["stats"]["total_events"] == 8
