"""Tests for the Pro-Lite-Pro sandwich pipeline and Oracle.

Tests cover:
- Scout tool implementations (deterministic SQL queries)
- Phase 1 Surveyor (mock LLM)
- Phase 2 Scout execution
- Phase 3 Consigliere (mock LLM)
- Full sandwich pipeline
- Oracle query expansion and synthesis
- Visibility index computation
- Graph ls refinements (work rhythm, domain bleed, shadow activity)
- Output formatting
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from loom.llm.mock import (
    MOCK_CONSIGLIERE_RESPONSE,
    MOCK_ORACLE_EXPANSION_RESPONSE,
    MOCK_ORACLE_SYNTHESIS_RESPONSE,
    MOCK_SURVEYOR_RESPONSE,
    MockLLMClient,
)
from loom.store import LayeredGraphStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _dict_factory(cursor, row):
    """Row factory that returns dicts — matches what scout tools expect."""
    return {col[0]: row[i] for i, col in enumerate(cursor.description)}


@pytest.fixture
def store(tmp_path):
    """Create an in-memory store with test data."""
    db_path = str(tmp_path / "test.db")
    s = LayeredGraphStore(db_path)
    _seed_test_data(s)
    # Scout tools and graph_ls expect dict rows
    s.conn.row_factory = _dict_factory
    return s


@pytest.fixture
def mock_llm():
    """Return a mock LLM client."""
    return MockLLMClient()


def _seed_test_data(store: LayeredGraphStore):
    """Seed the store with realistic test data for sandwich testing."""
    now = int(time.time())
    conn = store.conn

    # Events — sensitivity is INTEGER (1=PRIVATE, 2=SENSITIVE, 3=CRITICAL)
    events = [
        ("evt_mail_1", "mail", "mail_1", "email", now - 86400,
         "alice@test.com", "Please send the proposal by Friday",
         "", json.dumps({"subject": "Proposal deadline", "thread_id": "thread_1"}),
         2, now),
        ("evt_mail_2", "mail", "mail_2", "email", now - 172800,
         "bob@test.com", "Payment confirmation for BofA",
         "", json.dumps({"subject": "BofA Payment", "thread_id": "thread_2"}),
         2, now),
        ("evt_wa_1", "whatsapp", "wa_1", "message", now - 3600,
         "kai@test.com", "Hey, when can we sync on the project?",
         "", json.dumps({"thread_id": "thread_3"}),
         1, now),
        ("evt_cal_1", "calendar", "cal_1", "calendar_event", now + 86400,
         "alice@test.com,kai@test.com", "",
         "", json.dumps({"subject": "Team sync", "calendar": "Work", "is_recurring": "0", "is_all_day": "0"}),
         1, now),
        ("evt_safari_1", "safari", "saf_1", "browser_visit", now - 7200,
         "", "Checking bank balance",
         "", json.dumps({"url": "https://bofa.com/login", "domain": "bofa.com"}),
         2, now),
        ("evt_kc_1", "knowledgec", "kc_1", "app_focus", now - 3600,
         "", "",
         "", json.dumps({"bundle_id": "com.apple.Safari", "duration_seconds": 300}),
         1, now),
        ("evt_shell_1", "shell_history", "sh_1", "shell_command", now - 1800,
         "", "cd ~/Code/loom-v6 && git status",
         "", json.dumps({"command": "cd ~/Code/loom-v6 && git status"}),
         1, now),
    ]
    for evt in events:
        conn.execute(
            "INSERT OR IGNORE INTO events "
            "(id, source, source_id, event_type, timestamp, participants, "
            "raw_content, content_hash, metadata, sensitivity, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            evt,
        )

    # Persons — schema: person_id, canonical_name, is_user, sources, created_at, updated_at
    conn.execute(
        "INSERT OR IGNORE INTO persons "
        "(person_id, canonical_name, is_user, sources, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("person_alice", "Alice Smith", 0, "[]", now, now),
    )
    conn.execute(
        "INSERT OR IGNORE INTO persons "
        "(person_id, canonical_name, is_user, sources, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("person_kai", "Kai Tanaka", 0, "[]", now, now),
    )
    conn.execute(
        "INSERT OR IGNORE INTO persons "
        "(person_id, canonical_name, is_user, sources, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("person_user", "User", 1, "[]", now, now),
    )

    # Person profiles — schema: person_id, canonical_name, message_count, tier, ...
    try:
        conn.execute(
            "INSERT OR IGNORE INTO person_profiles "
            "(person_id, canonical_name, tier, message_count, channels, "
            "user_initiated_ratio, computed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("person_alice", "Alice Smith", 1, 50, '["mail","whatsapp"]', 0.4, now),
        )
        conn.execute(
            "INSERT OR IGNORE INTO person_profiles "
            "(person_id, canonical_name, tier, message_count, channels, "
            "user_initiated_ratio, computed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("person_kai", "Kai Tanaka", 2, 20, '["whatsapp"]', 0.6, now),
        )
    except sqlite3.OperationalError:
        pass  # Table might not exist in minimal store

    # Event-person edges
    conn.execute(
        "INSERT OR IGNORE INTO event_persons (event_id, person_id, role) "
        "VALUES (?, ?, ?)",
        ("evt_mail_1", "person_alice", "sender"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO event_persons (event_id, person_id, role) "
        "VALUES (?, ?, ?)",
        ("evt_wa_1", "person_kai", "sender"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO event_persons (event_id, person_id, role) "
        "VALUES (?, ?, ?)",
        ("evt_cal_1", "person_alice", "attendee"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO event_persons (event_id, person_id, role) "
        "VALUES (?, ?, ?)",
        ("evt_cal_1", "person_kai", "attendee"),
    )

    # Claims (commitments) — schema: id, claim_type, subject, predicate, object,
    #   confidence, created_at, prompt_version (NO source_event_id column)
    conn.execute(
        "INSERT OR IGNORE INTO claims "
        "(id, claim_type, subject, predicate, object, confidence, "
        "created_at, prompt_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "claim_1", "commitment", "evt_mail_1", "commitment_of",
            json.dumps({
                "what": "Send the updated proposal",
                "who": "user",
                "to_whom": "Alice",
                "deadline": "2026-02-15",
                "type": "inbound_request",
                "direction": "direct_ask",
                "status": "open",
                "priority": 2,
            }),
            0.85, now, "test_v1",
        ),
    )
    # Link claim to source event via claim_events junction table
    conn.execute(
        "INSERT OR IGNORE INTO claim_events (claim_id, event_id) VALUES (?, ?)",
        ("claim_1", "evt_mail_1"),
    )

    conn.execute(
        "INSERT OR IGNORE INTO claims "
        "(id, claim_type, subject, predicate, object, confidence, "
        "created_at, prompt_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "claim_2", "commitment", "evt_mail_2", "commitment_of",
            json.dumps({
                "what": "Pay BofA bill",
                "who": "user",
                "to_whom": "BofA",
                "deadline": "2026-02-10",
                "type": "payment_due",
                "direction": "outbound",
                "status": "open",
                "priority": 1,
            }),
            0.9, now, "test_v1",
        ),
    )
    conn.execute(
        "INSERT OR IGNORE INTO claim_events (claim_id, event_id) VALUES (?, ?)",
        ("claim_2", "evt_mail_2"),
    )

    # Beliefs
    conn.execute(
        "INSERT OR IGNORE INTO beliefs "
        "(id, belief_type, subject, summary, data, epistemic_level, "
        "confidence, source_claims, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "belief_1", "relation", "person_alice",
            "User multi channel contact Alice Smith via mail, whatsapp",
            json.dumps({
                "name": "Alice Smith",
                "relationship_type": "colleague",
                "relationship_tier": "vocational_core_team",
                "context": "Works on proposals together",
            }),
            "inference", 0.8, json.dumps(["claim_1"]),
            "active", now, now,
        ),
    )

    conn.commit()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Scout tool tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestScoutSearchEvents:
    """Tests for scout_search_events."""

    def test_search_by_keyword(self, store):
        from loom.sandwich import scout_search_events
        results = scout_search_events(store.conn, keyword="proposal")
        assert len(results) >= 1
        assert any("proposal" in (r.get("body") or "").lower() for r in results)

    def test_search_by_source(self, store):
        from loom.sandwich import scout_search_events
        results = scout_search_events(store.conn, source="mail", keyword="")
        assert all(r["source"] == "mail" for r in results)

    def test_search_with_limit(self, store):
        from loom.sandwich import scout_search_events
        results = scout_search_events(store.conn, keyword="", limit=2)
        assert len(results) <= 2

    def test_search_returns_timestamps(self, store):
        from loom.sandwich import scout_search_events
        results = scout_search_events(store.conn, keyword="proposal")
        for r in results:
            assert "time" in r
            assert r["time"]  # Not empty


class TestScoutSearchCommitments:
    """Tests for scout_search_commitments."""

    def test_search_by_keyword(self, store):
        from loom.sandwich import scout_search_commitments
        results = scout_search_commitments(store.conn, keyword="proposal")
        assert len(results) >= 1
        assert any("proposal" in (r.get("what") or "").lower() for r in results)

    def test_search_payment(self, store):
        from loom.sandwich import scout_search_commitments
        results = scout_search_commitments(store.conn, keyword="BofA")
        assert len(results) >= 1

    def test_empty_search(self, store):
        from loom.sandwich import scout_search_commitments
        results = scout_search_commitments(store.conn, keyword="nonexistent_xyz")
        assert len(results) == 0


class TestScoutGetPersonContext:
    """Tests for scout_get_person_context."""

    def test_find_known_person(self, store):
        from loom.sandwich import scout_get_person_context
        result = scout_get_person_context(store.conn, name="Alice")
        assert "error" not in result
        assert result["name"] == "Alice Smith"
        assert "recent_events" in result

    def test_person_not_found(self, store):
        from loom.sandwich import scout_get_person_context
        result = scout_get_person_context(store.conn, name="Nonexistent Person XYZ")
        assert "error" in result

    def test_includes_commitments(self, store):
        from loom.sandwich import scout_get_person_context
        result = scout_get_person_context(store.conn, name="Alice")
        assert "open_commitments" in result

    def test_includes_beliefs(self, store):
        from loom.sandwich import scout_get_person_context
        result = scout_get_person_context(store.conn, name="Alice")
        assert "beliefs" in result


class TestScoutTemporalXref:
    """Tests for scout_temporal_xref."""

    def test_find_events_near_date(self, store):
        from loom.sandwich import scout_temporal_xref
        from datetime import datetime, timezone
        now = int(time.time())
        date_str = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d")
        results = scout_temporal_xref(store.conn, date_str, source="safari", window_hours=48)
        assert len(results) >= 1

    def test_invalid_date(self, store):
        from loom.sandwich import scout_temporal_xref
        results = scout_temporal_xref(store.conn, "not-a-date", source="mail")
        assert len(results) == 1
        assert "error" in results[0]


class TestScoutGetThread:
    """Tests for scout_get_thread."""

    def test_find_thread(self, store):
        from loom.sandwich import scout_get_thread
        results = scout_get_thread(store.conn, thread_id="thread_1")
        assert len(results) >= 1

    def test_empty_thread(self, store):
        from loom.sandwich import scout_get_thread
        results = scout_get_thread(store.conn, thread_id="nonexistent_thread")
        assert len(results) == 0


class TestExecuteScoutQuery:
    """Tests for execute_scout_query."""

    def test_valid_query(self, store):
        from loom.sandwich import execute_scout_query
        result = execute_scout_query(store.conn, {
            "tool": "search_commitments",
            "args": {"keyword": "proposal"},
        })
        assert "results" in result
        assert "error" not in result

    def test_unknown_tool(self, store):
        from loom.sandwich import execute_scout_query
        result = execute_scout_query(store.conn, {
            "tool": "nonexistent_tool",
            "args": {},
        })
        assert "error" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Visibility index tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVisibilityIndex:
    """Tests for compute_visibility_index."""

    def test_computes_visibility(self, store):
        from loom.sandwich import compute_visibility_index
        result = compute_visibility_index(store.conn)
        # Should detect "finance" domain from BofA payment commitment
        assert "finance" in result
        assert "visibility" in result["finance"]
        assert 0.0 <= result["finance"]["visibility"] <= 1.0

    def test_assessment_labels(self, store):
        from loom.sandwich import compute_visibility_index
        result = compute_visibility_index(store.conn)
        for domain, data in result.items():
            assert data["assessment"] in (
                "HIGH",
                "MEDIUM",
                "LOW — system is likely blind to this domain",
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPhase1Surveyor:
    """Tests for phase1_surveyor."""

    def test_returns_vectors(self, mock_llm):
        from loom.sandwich import phase1_surveyor
        vectors = phase1_surveyor(mock_llm, {"meta": {}, "schema": {}})
        assert len(vectors) == 3

    def test_vectors_have_structure(self, mock_llm):
        from loom.sandwich import phase1_surveyor
        vectors = phase1_surveyor(mock_llm, {"meta": {}})
        for v in vectors:
            assert "target" in v
            assert "logic" in v
            assert "scout_queries" in v

    def test_caps_at_max(self, mock_llm):
        from loom.sandwich import phase1_surveyor
        vectors = phase1_surveyor(mock_llm, {"meta": {}})
        assert len(vectors) <= 3


class TestPhase2Scout:
    """Tests for phase2_scout."""

    def test_executes_queries(self, store):
        from loom.sandwich import phase2_scout
        vectors = [
            {
                "target": "Test vector",
                "logic": "Testing",
                "scout_queries": [
                    {"tool": "search_commitments", "args": {"keyword": "proposal"}},
                ],
            },
        ]
        results = phase2_scout(store.conn, vectors)
        assert len(results) == 1
        assert results[0]["target"] == "Test vector"
        assert len(results[0]["evidence"]) == 1

    def test_handles_empty_vectors(self, store):
        from loom.sandwich import phase2_scout
        results = phase2_scout(store.conn, [])
        assert results == []


class TestPhase3Consigliere:
    """Tests for phase3_consigliere."""

    def test_returns_findings(self, mock_llm):
        from loom.sandwich import phase3_consigliere
        scout_results = [{"target": "Test", "logic": "Testing", "severity": "HIGH", "evidence": []}]
        result = phase3_consigliere(mock_llm, scout_results, graph_ls_meta={})
        assert "findings" in result
        assert len(result["findings"]) > 0

    def test_includes_blind_spots(self, mock_llm):
        from loom.sandwich import phase3_consigliere
        scout_results = [{"target": "T", "logic": "L", "severity": "H", "evidence": []}]
        result = phase3_consigliere(mock_llm, scout_results, graph_ls_meta={})
        assert "blind_spots" in result

    def test_includes_decision_dag(self, mock_llm):
        from loom.sandwich import phase3_consigliere
        scout_results = [{"target": "T", "logic": "L", "severity": "H", "evidence": []}]
        result = phase3_consigliere(mock_llm, scout_results, graph_ls_meta={})
        assert "decision_dag" in result

    def test_includes_visibility_assessment(self, mock_llm):
        from loom.sandwich import phase3_consigliere
        scout_results = [{"target": "T", "logic": "L", "severity": "H", "evidence": []}]
        result = phase3_consigliere(mock_llm, scout_results, graph_ls_meta={})
        assert "visibility_assessment" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full pipeline tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunSandwich:
    """Tests for run_sandwich end-to-end."""

    def test_full_pipeline(self, store, mock_llm):
        from loom.sandwich import run_sandwich
        result = run_sandwich(store, mock_llm)
        assert "inquiry_vectors" in result
        assert "scout_results" in result
        assert "consigliere" in result
        assert "output" in result
        assert result["total_elapsed"] >= 0

    def test_pipeline_produces_output(self, store, mock_llm):
        from loom.sandwich import run_sandwich
        result = run_sandwich(store, mock_llm)
        output = result["output"]
        assert "Phase 1: Surveyor" in output
        assert "Phase 2: Scout" in output
        assert "Phase 3: Consigliere" in output

    def test_pipeline_includes_findings(self, store, mock_llm):
        from loom.sandwich import run_sandwich
        result = run_sandwich(store, mock_llm)
        output = result["output"]
        assert "Finding" in output

    def test_pipeline_includes_dag(self, store, mock_llm):
        from loom.sandwich import run_sandwich
        result = run_sandwich(store, mock_llm)
        output = result["output"]
        assert "Decision DAG" in output

    def test_pipeline_includes_visibility(self, store, mock_llm):
        from loom.sandwich import run_sandwich
        result = run_sandwich(store, mock_llm)
        output = result["output"]
        assert "Low Visibility Domains" in output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Output formatting tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestFormatSandwichOutput:
    """Tests for format_sandwich_output."""

    def test_formats_all_sections(self):
        from loom.sandwich import format_sandwich_output
        result = {
            "inquiry_vectors": MOCK_SURVEYOR_RESPONSE["inquiry_vectors"],
            "scout_results": [
                {"target": "T1", "logic": "L1", "severity": "HIGH",
                 "evidence": [{"tool": "search_commitments", "args": {}, "results": []}]},
            ],
            "consigliere": MOCK_CONSIGLIERE_RESPONSE,
            "surveyor_model": "test-model",
            "consigliere_model": "test-model",
            "phase1_elapsed": 1.0,
            "phase2_elapsed": 0.5,
            "phase3_elapsed": 2.0,
            "total_elapsed": 3.5,
        }
        output = format_sandwich_output(result)
        assert "Phase 1" in output
        assert "Phase 2" in output
        assert "Phase 3" in output
        assert "3.5s" in output

    def test_handles_empty_consigliere(self):
        from loom.sandwich import format_sandwich_output
        result = {
            "inquiry_vectors": [],
            "scout_results": [],
            "consigliere": {"findings": [], "blind_spots": []},
            "surveyor_model": "m",
            "consigliere_model": "m",
            "phase1_elapsed": 0,
            "phase2_elapsed": 0,
            "phase3_elapsed": 0,
            "total_elapsed": 0,
        }
        output = format_sandwich_output(result)
        assert "0 inquiry vectors" in output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Oracle tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestOracleAsk:
    """Tests for ask_oracle."""

    def test_returns_answer(self, store, mock_llm):
        from loom.oracle import ask_oracle
        result = ask_oracle(store, mock_llm, "Where did I promise to send the proposal?")
        assert "answer" in result
        assert result["answer"]

    def test_includes_confidence(self, store, mock_llm):
        from loom.oracle import ask_oracle
        result = ask_oracle(store, mock_llm, "Where did I promise?")
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1

    def test_includes_sources(self, store, mock_llm):
        from loom.oracle import ask_oracle
        result = ask_oracle(store, mock_llm, "Where did I promise?")
        assert "sources" in result

    def test_includes_interpretation(self, store, mock_llm):
        from loom.oracle import ask_oracle
        result = ask_oracle(store, mock_llm, "How does Alice relate to Kai?")
        assert "interpretation" in result

    def test_includes_elapsed(self, store, mock_llm):
        from loom.oracle import ask_oracle
        result = ask_oracle(store, mock_llm, "test question")
        assert "elapsed" in result
        assert result["elapsed"] >= 0


class TestOracleFormat:
    """Tests for format_oracle_output."""

    def test_formats_answer(self):
        from loom.oracle import format_oracle_output
        result = {
            "question": "Where did I promise?",
            "interpretation": "Looking for promise location",
            "queries_executed": 2,
            "evidence_items": 5,
            "answer": "You promised in an email on Feb 12.",
            "confidence": 0.85,
            "sources": [
                {"type": "event", "summary": "Email thread", "date": "Feb 12", "source": "mail"},
            ],
            "follow_up": "Want to check for responses?",
            "elapsed": 1.5,
            "model": "test-model",
        }
        output = format_oracle_output(result)
        assert "Q: Where did I promise?" in output
        assert "A: You promised" in output
        assert "85%" in output
        assert "Sources:" in output
        assert "Follow-up:" in output

    def test_handles_no_sources(self):
        from loom.oracle import format_oracle_output
        result = {
            "question": "test",
            "interpretation": "test",
            "queries_executed": 0,
            "evidence_items": 0,
            "answer": "No results",
            "confidence": 0.0,
            "sources": [],
            "follow_up": None,
            "elapsed": 0.1,
            "model": "test",
        }
        output = format_oracle_output(result)
        assert "No results" in output
        assert "Sources:" not in output


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph ls refinement tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGraphLsRefinements:
    """Tests for graph_ls work rhythm and domain bleed."""

    def test_work_rhythm_no_data(self, store):
        from loom.graph_ls import _detect_work_rhythm
        result = _detect_work_rhythm(store.conn, int(time.time()))
        # Should have data since we seeded knowledgec events
        assert isinstance(result, dict)

    def test_work_rhythm_with_data(self, store):
        from loom.graph_ls import _detect_work_rhythm
        # We seeded one knowledgec event, so should have data
        result = _detect_work_rhythm(store.conn, int(time.time()))
        if "days" in result:
            assert len(result["days"]) >= 1

    def test_shadow_activity_detection(self, store):
        from loom.graph_ls import _detect_shadow_activity
        result = _detect_shadow_activity(store.conn, int(time.time()))
        # Should be a list (may be empty if not enough data)
        assert isinstance(result, list)

    def test_graph_ls_includes_work_rhythm(self, store):
        from loom.graph_ls import generate_graph_ls
        result = generate_graph_ls(store)
        assert "work_rhythm" in result["meta"]

    def test_tier4_includes_domain_bleed(self, store):
        from loom.graph_ls import generate_graph_ls
        result = generate_graph_ls(store)
        tier4 = result["TIER_4_RELATIONSHIPS"]
        assert "domain bleed" in tier4["description"].lower()

    def test_graph_ls_full_output(self, store):
        from loom.graph_ls import generate_graph_ls
        result = generate_graph_ls(store)
        assert "meta" in result
        assert "TIER_1_INNER_CIRCLE" in result
        assert "TIER_7_EPISTEMIC_GAPS" in result
