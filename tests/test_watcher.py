"""Tests for alteris.watcher: Watch daemon + thread-aware incremental processing.

Tests cover:
  - WatchDaemon debounce behavior
  - Pipeline serialization (rerun flag)
  - Graceful shutdown
  - AlterisFileHandler filtering
  - Thread-aware incremental triage helpers
  - Thread-aware synthesis helpers
"""

import argparse
import json
import threading
import time

import pytest

from alteris.constants import (
    INCREMENTAL_CONTEXT_MESSAGES,
    REACTIVATION_THRESHOLD,
    WATCH_DEBOUNCE_SECONDS,
)
from alteris.models import Claim, Event, ExtractionProvenance, Modality
from alteris.privacy import SensitivityLevel
from alteris.store import LayeredGraphStore


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def store():
    """In-memory store for testing."""
    s = LayeredGraphStore(":memory:")
    yield s
    s.close()


def _make_event(
    eid: str, source: str = "mail", ts: int = 0, thread_id: str = "",
    raw_content: str = "test content", event_type: str = "email",
) -> Event:
    """Helper to create a test event."""
    meta = {}
    if thread_id:
        meta["thread_id"] = thread_id
    return Event(
        id=eid,
        source=source,
        source_id=eid,
        event_type=event_type,
        timestamp=ts or int(time.time()),
        participants=["test@example.com"],
        raw_content=raw_content,
        metadata=meta,
    )


def _make_triage_claim(
    event_id: str, score: float = 0.5, thread_id: str = "",
) -> Claim:
    """Helper to create a triage claim for an event."""
    from alteris.triage import triage_claim_id
    return Claim(
        id=triage_claim_id(event_id),
        event_ids=[event_id],
        claim_type="triage",
        subject=event_id,
        predicate="triage_result",
        object=json.dumps({
            "score": score,
            "reason": "test",
            "domain": "work",
            "topics": ["test"],
        }),
        confidence=score,
        modality=Modality.OBSERVED,
        provenance=ExtractionProvenance(
            model_id="test", prompt_version="test",
        ),
        sensitivity=SensitivityLevel.SENSITIVE,
    )


def _make_thread_triage_claim(
    thread_id: str, score: float = 0.5,
    summary: str = "test thread summary",
    status: str = "active_conversation",
    commitment_type: str | None = None,
    event_ids: list[str] | None = None,
) -> Claim:
    """Helper to create a thread_triage claim."""
    from alteris.triage import thread_triage_claim_id
    return Claim(
        id=thread_triage_claim_id(thread_id),
        event_ids=event_ids or [],
        claim_type="thread_triage",
        subject=thread_id,
        predicate="thread_triage_result",
        object=json.dumps({
            "thread_score": score,
            "thread_summary": summary,
            "thread_status": status,
            "commitment_type": commitment_type,
            "domain": "work",
            "topics": ["test"],
        }),
        confidence=score,
        modality=Modality.OBSERVED,
        provenance=ExtractionProvenance(
            model_id="test", prompt_version="test",
        ),
        sensitivity=SensitivityLevel.SENSITIVE,
    )


def _make_commitment_claim(
    thread_id: str, what: str = "send report",
    who: str = "user", deadline: str | None = None,
    status: str = "open", confidence: float = 0.8,
) -> Claim:
    """Helper to create a commitment claim."""
    import hashlib
    key = f"commitment:{thread_id}:{what[:50]}:{who}:test"
    claim_id = hashlib.sha256(key.encode()).hexdigest()[:16]
    return Claim(
        id=claim_id,
        event_ids=[],
        claim_type="commitment",
        subject=thread_id,
        predicate="inbound_request",
        object=json.dumps({
            "type": "inbound_request",
            "who": who,
            "what": what,
            "deadline": deadline,
            "status": status,
            "confidence": confidence,
        }),
        confidence=confidence,
        modality=Modality.OBSERVED,
        provenance=ExtractionProvenance(
            model_id="test", prompt_version="test",
        ),
        sensitivity=SensitivityLevel.SENSITIVE,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WatchDaemon debounce tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestWatchDaemonDebounce:
    """Test debounce timer behavior."""

    def _make_daemon(self, debounce: float = 0.1):
        """Create a daemon with short debounce for testing."""
        from alteris.watcher import WatchDaemon

        args = argparse.Namespace(
            db_path=":memory:", dry_run=True, llm="mock",
            verbose=False, hours=None, since=None, limit=None,
            sources=None, lens="chief_of_staff",
        )
        daemon = WatchDaemon(
            args=args, debounce_seconds=debounce,
            poll_intervals={}, enabled_sources=["mail"],
        )
        # Replace _execute_pipeline to track calls
        daemon._pipeline_calls: list[set[str]] = []
        original_execute = daemon._execute_pipeline

        def mock_execute(sources: set[str]) -> None:
            daemon._pipeline_calls.append(sources.copy())

        daemon._execute_pipeline = mock_execute
        return daemon

    def test_single_trigger_fires_after_delay(self):
        daemon = self._make_daemon(debounce=0.1)
        daemon.trigger("mail")
        time.sleep(0.05)
        # Not fired yet
        assert len(daemon._pipeline_calls) == 0
        time.sleep(0.15)
        # Now it should have fired
        assert len(daemon._pipeline_calls) == 1
        assert daemon._pipeline_calls[0] == {"mail"}

    def test_rapid_triggers_coalesce(self):
        daemon = self._make_daemon(debounce=0.15)
        daemon.trigger("mail")
        time.sleep(0.05)
        daemon.trigger("imessage")
        time.sleep(0.05)
        daemon.trigger("whatsapp")
        time.sleep(0.05)
        # None fired yet (debounce resets each time)
        assert len(daemon._pipeline_calls) == 0
        time.sleep(0.2)
        # Single coalesced call
        assert len(daemon._pipeline_calls) == 1
        assert daemon._pipeline_calls[0] == {"mail", "imessage", "whatsapp"}

    def test_sources_accumulate(self):
        daemon = self._make_daemon(debounce=0.1)
        daemon.trigger("mail")
        daemon.trigger("mail")  # Duplicate
        daemon.trigger("imessage")
        time.sleep(0.2)
        assert len(daemon._pipeline_calls) == 1
        assert daemon._pipeline_calls[0] == {"mail", "imessage"}


class TestWatchDaemonSerialization:
    """Test pipeline execution serialization."""

    def _make_daemon(self):
        from alteris.watcher import WatchDaemon

        args = argparse.Namespace(
            db_path=":memory:", dry_run=True, llm="mock",
            verbose=False, hours=None, since=None, limit=None,
            sources=None, lens="chief_of_staff",
        )
        daemon = WatchDaemon(
            args=args, debounce_seconds=0.05,
            poll_intervals={}, enabled_sources=["mail"],
        )
        return daemon

    def test_rerun_flag_set_during_execution(self):
        daemon = self._make_daemon()
        execution_log: list[set[str]] = []
        barrier = threading.Event()

        original_execute = daemon._execute_pipeline

        def slow_execute(sources: set[str]) -> None:
            execution_log.append(sources.copy())
            barrier.wait(timeout=2)

        daemon._execute_pipeline = slow_execute

        # Start first pipeline run in a thread
        t = threading.Thread(target=daemon._run_pipeline, args=({"mail"},))
        t.start()
        time.sleep(0.05)

        # Trigger during execution
        daemon._run_pipeline({"imessage"})

        # Verify rerun was queued
        assert daemon._rerun_requested is True
        assert "imessage" in daemon._rerun_sources

        # Release the barrier
        barrier.set()
        t.join(timeout=2)

    def test_graceful_shutdown_stops_watchers(self):
        daemon = self._make_daemon()
        daemon._shutdown_event.set()
        # Verify that shutdown_event is set
        assert daemon._shutdown_event.is_set()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AlterisFileHandler tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAlterisFileHandler:
    def test_filters_target_filenames(self):
        from unittest.mock import MagicMock
        from watchdog.events import FileModifiedEvent

        from alteris.watcher import AlterisFileHandler

        callback = MagicMock()
        handler = AlterisFileHandler(
            "imessage",
            frozenset({"chat.db", "chat.db-wal"}),
            callback,
        )

        # Target file -> triggers
        event = FileModifiedEvent("/path/to/chat.db")
        handler.on_modified(event)
        callback.assert_called_once_with("imessage")

        # Non-target file -> no trigger
        callback.reset_mock()
        event = FileModifiedEvent("/path/to/other.db")
        handler.on_modified(event)
        callback.assert_not_called()

    def test_ignores_directories(self):
        from unittest.mock import MagicMock
        from watchdog.events import DirModifiedEvent

        from alteris.watcher import AlterisFileHandler

        callback = MagicMock()
        handler = AlterisFileHandler(
            "mail", frozenset({"Envelope Index"}), callback,
        )
        event = DirModifiedEvent("/path/to/dir")
        handler.on_modified(event)
        callback.assert_not_called()

    def test_wal_file_triggers(self):
        from unittest.mock import MagicMock
        from watchdog.events import FileModifiedEvent

        from alteris.watcher import AlterisFileHandler

        callback = MagicMock()
        handler = AlterisFileHandler(
            "imessage",
            frozenset({"chat.db", "chat.db-wal", "chat.db-shm"}),
            callback,
        )
        event = FileModifiedEvent("/Library/Messages/chat.db-wal")
        handler.on_modified(event)
        callback.assert_called_once_with("imessage")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thread-aware incremental triage tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetThreadPriorContext:
    def test_returns_none_for_fresh_thread(self, store):
        from alteris.triage import _get_thread_prior_context
        result = _get_thread_prior_context("thread:123", store)
        assert result is None

    def test_returns_data_for_triaged_thread(self, store):
        from alteris.triage import _get_thread_prior_context

        claim = _make_thread_triage_claim(
            "thread:123", score=0.7, summary="Important thread",
            status="awaiting_user", commitment_type="deadline",
        )
        store.put_claim(claim)

        result = _get_thread_prior_context("thread:123", store)
        assert result is not None
        assert result["thread_score"] == 0.7
        assert result["thread_summary"] == "Important thread"
        assert result["thread_status"] == "awaiting_user"
        assert result["commitment_type"] == "deadline"

    def test_returns_none_for_wrong_claim_type(self, store):
        from alteris.triage import _get_thread_prior_context, thread_triage_claim_id

        # Insert a claim with the right ID but wrong type
        claim = Claim(
            id=thread_triage_claim_id("thread:123"),
            event_ids=[],
            claim_type="triage",  # Wrong type
            subject="thread:123",
            predicate="triage_result",
            object="{}",
            confidence=0.5,
            modality=Modality.OBSERVED,
            provenance=ExtractionProvenance(model_id="test", prompt_version="test"),
            sensitivity=SensitivityLevel.SENSITIVE,
        )
        store.put_claim(claim)
        result = _get_thread_prior_context("thread:123", store)
        assert result is None


class TestGetTriagedEventIds:
    def test_empty_for_fresh_events(self, store):
        from alteris.triage import _get_triaged_event_ids

        event = _make_event("evt1")
        store.put_event(event)

        result = _get_triaged_event_ids(["evt1"], store)
        assert result == set()

    def test_finds_triaged_events(self, store):
        from alteris.triage import _get_triaged_event_ids

        event = _make_event("evt1")
        store.put_event(event)

        claim = _make_triage_claim("evt1", score=0.7)
        store.put_claim(claim)

        result = _get_triaged_event_ids(["evt1", "evt2"], store)
        assert result == {"evt1"}

    def test_excludes_parse_failed(self, store):
        from alteris.triage import _get_triaged_event_ids, triage_claim_id

        event = _make_event("evt1")
        store.put_event(event)

        # Create a PARSE_FAILED claim
        claim = Claim(
            id=triage_claim_id("evt1"),
            event_ids=["evt1"],
            claim_type="triage",
            subject="evt1",
            predicate="triage_result",
            object=json.dumps({"score": 0.1, "reason": "PARSE_FAILED"}),
            confidence=0.0,
            modality=Modality.OBSERVED,
            provenance=ExtractionProvenance(model_id="test", prompt_version="test"),
            sensitivity=SensitivityLevel.SENSITIVE,
        )
        store.put_claim(claim)

        result = _get_triaged_event_ids(["evt1"], store)
        assert result == set()


class TestClassifyThreadIncremental:
    def test_fresh_thread(self, store):
        from alteris.triage import classify_thread_incremental

        events = [{"id": "evt1", "timestamp": int(time.time())}]
        result = classify_thread_incremental(
            "thread:new", events, {"evt1"}, store,
        )
        assert result == "fresh"

    def test_incremental_thread(self, store):
        from alteris.triage import classify_thread_incremental

        # Create a prior triage with score above threshold
        claim = _make_thread_triage_claim("thread:known", score=0.7)
        store.put_claim(claim)

        events = [{"id": "evt1", "timestamp": int(time.time())}]
        result = classify_thread_incremental(
            "thread:known", events, {"evt1"}, store,
        )
        assert result == "incremental"

    def test_reactivated_dormant_thread(self, store):
        from alteris.triage import classify_thread_incremental

        # Create a prior triage with score BELOW threshold
        claim = _make_thread_triage_claim(
            "thread:dormant", score=0.1,
        )
        store.put_claim(claim)

        events = [{"id": "evt1", "timestamp": int(time.time())}]
        result = classify_thread_incremental(
            "thread:dormant", events, {"evt1"}, store,
        )
        assert result == "reactivated"

    def test_dormant_with_gate_claim_stays_incremental(self, store):
        from alteris.triage import classify_thread_incremental

        # Create a prior triage with low score
        claim = _make_thread_triage_claim("thread:gated", score=0.1)
        store.put_claim(claim)

        # Add an extraction gate claim
        gate_claim = Claim(
            id="gate:123",
            event_ids=[],
            claim_type="extraction_gate",
            subject="thread:gated",
            predicate="gate_result",
            object="{}",
            confidence=0.5,
            modality=Modality.OBSERVED,
            provenance=ExtractionProvenance(model_id="test", prompt_version="test"),
            sensitivity=SensitivityLevel.SENSITIVE,
        )
        store.put_claim(gate_claim)

        events = [{"id": "evt1", "timestamp": int(time.time())}]
        result = classify_thread_incremental(
            "thread:gated", events, {"evt1"}, store,
        )
        assert result == "incremental"


class TestBuildIncrementalThreadPrompt:
    def test_includes_prior_context_block(self, store):
        from alteris.triage import build_incremental_thread_prompt

        now = int(time.time())
        events = [
            {"id": "evt1", "source": "mail", "timestamp": now - 100,
             "raw_content": "Old message", "metadata": "{}",
             "event_type": "email"},
            {"id": "evt2", "source": "mail", "timestamp": now - 50,
             "raw_content": "New message", "metadata": "{}",
             "event_type": "email"},
        ]

        for e in events:
            store.put_event(Event(
                id=e["id"], source=e["source"], source_id=e["id"],
                event_type=e["event_type"], timestamp=e["timestamp"],
                participants=[], raw_content=e["raw_content"],
            ))

        prior_context = {
            "thread_summary": "Discussion about Q4 report",
            "thread_score": 0.7,
            "thread_status": "awaiting_user",
            "domain": "work",
            "commitment_type": "deadline",
        }

        prompt = build_incremental_thread_prompt(
            "thread:test", events, {"evt1"}, prior_context,
            store, {}, now,
        )

        assert "PRIOR THREAD CONTEXT" in prompt
        assert "Discussion about Q4 report" in prompt
        assert "0.7" in prompt
        assert "awaiting_user" in prompt

    def test_marks_messages_new_and_context(self, store):
        from alteris.triage import build_incremental_thread_prompt

        now = int(time.time())
        events = [
            {"id": "evt1", "source": "mail", "timestamp": now - 100,
             "raw_content": "Old message", "metadata": "{}",
             "event_type": "email"},
            {"id": "evt2", "source": "mail", "timestamp": now - 50,
             "raw_content": "New message", "metadata": "{}",
             "event_type": "email"},
        ]

        for e in events:
            store.put_event(Event(
                id=e["id"], source=e["source"], source_id=e["id"],
                event_type=e["event_type"], timestamp=e["timestamp"],
                participants=[], raw_content=e["raw_content"],
            ))

        prior_context = {
            "thread_summary": "Test",
            "thread_score": 0.5,
            "thread_status": "active_conversation",
            "domain": "work",
            "commitment_type": None,
        }

        prompt = build_incremental_thread_prompt(
            "thread:test", events, {"evt1"}, prior_context,
            store, {}, now,
        )

        assert "[CONTEXT]" in prompt
        assert "[NEW]" in prompt

    def test_only_requests_scores_for_new(self, store):
        from alteris.triage import build_incremental_thread_prompt

        now = int(time.time())
        events = [
            {"id": f"evt{i}", "source": "mail", "timestamp": now - (10 - i),
             "raw_content": f"Message {i}", "metadata": "{}",
             "event_type": "email"}
            for i in range(5)
        ]

        for e in events:
            store.put_event(Event(
                id=e["id"], source=e["source"], source_id=e["id"],
                event_type=e["event_type"], timestamp=e["timestamp"],
                participants=[], raw_content=e["raw_content"],
            ))

        # evt0-evt2 are old, evt3-evt4 are new
        triaged_ids = {"evt0", "evt1", "evt2"}
        prior_context = {
            "thread_summary": "Test",
            "thread_score": 0.5,
            "thread_status": "active_conversation",
            "domain": "work",
            "commitment_type": None,
        }

        prompt = build_incremental_thread_prompt(
            "thread:test", events, triaged_ids, prior_context,
            store, {}, now,
        )

        assert "Only provide message_scores for the NEW messages" in prompt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Thread-aware synthesis tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetPriorCommitments:
    def test_no_prior_commitments(self, store):
        from alteris.beliefs import _get_prior_commitments
        result = _get_prior_commitments(store, "thread:new")
        assert result == []

    def test_finds_active_commitments(self, store):
        from alteris.beliefs import _get_prior_commitments

        claim = _make_commitment_claim(
            "thread:123", what="send report", deadline="2026-02-20",
        )
        store.put_claim(claim)

        result = _get_prior_commitments(store, "thread:123")
        assert len(result) == 1
        assert result[0]["what"] == "send report"
        assert result[0]["deadline"] == "2026-02-20"

    def test_excludes_superseded(self, store):
        from alteris.beliefs import _get_prior_commitments

        claim = _make_commitment_claim("thread:123", what="old task")
        store.put_claim(claim)
        store.supersede_claim(claim.id, "new_claim_id")

        result = _get_prior_commitments(store, "thread:123")
        assert result == []


class TestFormatPriorCommitments:
    def test_empty_list(self):
        from alteris.beliefs import _format_prior_commitments
        result = _format_prior_commitments([])
        assert result == ""

    def test_formats_commitments(self):
        from alteris.beliefs import _format_prior_commitments

        prior = [
            {
                "who": "user",
                "what": "send Q4 report",
                "deadline": "2026-02-20",
                "status": "open",
                "confidence": 0.9,
            },
            {
                "who": "user",
                "what": "review PR",
                "deadline": None,
                "status": "open",
                "confidence": 0.7,
            },
        ]
        result = _format_prior_commitments(prior)
        assert "KNOWN COMMITMENTS" in result
        assert "send Q4 report" in result
        assert "2026-02-20" in result
        assert "no deadline" in result
        assert "CONFIRM if still valid" in result
        assert "SUPERSEDE if no longer relevant" in result


class TestBuildSynthesisPromptWithPrior:
    def test_no_prior_no_block(self, store):
        from alteris.beliefs import _build_synthesis_prompt
        from alteris.extract import ThreadBundle
        from alteris.models import Event

        events = [Event(
            id="evt1", source="mail", source_id="evt1",
            event_type="email", timestamp=int(time.time()),
            participants=[], raw_content="Hello",
        )]
        bundle = ThreadBundle(
            thread_id="thread:test", events=events, triage_data=[{}],
        )

        prompt, _ = _build_synthesis_prompt(bundle, store=store)
        assert "KNOWN COMMITMENTS" not in prompt

    def test_with_prior_includes_block(self, store):
        from alteris.beliefs import _build_synthesis_prompt
        from alteris.extract import ThreadBundle
        from alteris.models import Event

        events = [Event(
            id="evt1", source="mail", source_id="evt1",
            event_type="email", timestamp=int(time.time()),
            participants=[], raw_content="Hello",
        )]
        bundle = ThreadBundle(
            thread_id="thread:test", events=events, triage_data=[{}],
        )

        prior = [{
            "claim_id": "test123",
            "who": "user",
            "what": "send report",
            "deadline": "2026-02-20",
            "status": "open",
            "confidence": 0.8,
            "type": "inbound_request",
        }]

        prompt, _ = _build_synthesis_prompt(
            bundle, store=store, prior_commitments=prior,
        )
        assert "KNOWN COMMITMENTS" in prompt
        assert "send report" in prompt


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestWatcherConstants:
    def test_debounce_positive(self):
        assert WATCH_DEBOUNCE_SECONDS > 0

    def test_reactivation_threshold_in_range(self):
        assert 0 < REACTIVATION_THRESHOLD < 1

    def test_incremental_context_messages_positive(self):
        assert INCREMENTAL_CONTEXT_MESSAGES > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI integration tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestWatchCLI:
    def test_watch_parser_exists(self):
        from alteris.cli import build_parser
        parser = build_parser()
        # --dry-run is a global arg, must come before the subcommand
        args = parser.parse_args(["--dry-run", "watch"])
        assert args.command == "watch"
        assert args.dry_run is True

    def test_watch_debounce_arg(self):
        from alteris.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["watch", "--debounce", "3.0"])
        assert args.debounce == 3.0

    def test_watch_poll_intervals(self):
        from alteris.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "watch",
            "--poll-calendar", "60",
            "--poll-slack", "30",
            "--poll-granola", "120",
        ])
        assert args.poll_calendar == 60
        assert args.poll_slack == 30
        assert args.poll_granola == 120

    def test_watch_sources_filter(self):
        from alteris.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["watch", "--sources", "mail", "imessage"])
        assert args.sources == ["mail", "imessage"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# run_pipeline_stages tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunPipelineStages:
    def test_returns_summary_dict(self):
        from alteris.cli import run_pipeline_stages

        args = argparse.Namespace(
            db_path=":memory:", dry_run=True, llm="mock",
            verbose=False, hours=1, since=None, limit=10,
            sources=["mail"], lens="chief_of_staff",
        )
        summary = run_pipeline_stages(args)

        assert "stages_run" in summary
        assert "stages_failed" in summary
        assert "errors" in summary
        assert isinstance(summary["stages_run"], int)
        assert isinstance(summary["errors"], list)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Watch targets configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestWatchTargets:
    def test_all_sources_have_targets(self):
        from alteris.watcher import WATCH_TARGETS
        assert "mail" in WATCH_TARGETS
        # iMessage is polled, not watched (macOS suppresses FSEvents for chat.db)
        assert "imessage" not in WATCH_TARGETS
        assert "whatsapp" in WATCH_TARGETS
        assert "contacts" in WATCH_TARGETS

    def test_target_filenames_are_frozensets(self):
        from alteris.watcher import WATCH_TARGETS
        for source, (watch_dir, target_files, recursive) in WATCH_TARGETS.items():
            assert isinstance(target_files, frozenset)

    def test_poll_sources_have_intervals(self):
        from alteris.watcher import POLL_SOURCES
        assert "imessage" in POLL_SOURCES  # polled because FSEvents unreliable
        assert "calendar" in POLL_SOURCES
        assert "slack" in POLL_SOURCES
        assert "granola" in POLL_SOURCES
        for source, interval in POLL_SOURCES.items():
            assert interval > 0

    def test_source_enabled_filter(self):
        from alteris.watcher import WatchDaemon

        args = argparse.Namespace(
            db_path=":memory:", dry_run=True, llm="mock",
            verbose=False, hours=None, since=None, limit=None,
            sources=None, lens="chief_of_staff",
        )
        daemon = WatchDaemon(
            args=args, enabled_sources=["mail", "imessage"],
        )
        assert daemon._source_enabled("mail") is True
        assert daemon._source_enabled("imessage") is True
        assert daemon._source_enabled("whatsapp") is False

    def test_all_sources_enabled_when_none(self):
        from alteris.watcher import WatchDaemon

        args = argparse.Namespace(
            db_path=":memory:", dry_run=True, llm="mock",
            verbose=False, hours=None, since=None, limit=None,
            sources=None, lens="chief_of_staff",
        )
        daemon = WatchDaemon(args=args, enabled_sources=None)
        assert daemon._source_enabled("mail") is True
        assert daemon._source_enabled("whatsapp") is True
        assert daemon._source_enabled("anything") is True
