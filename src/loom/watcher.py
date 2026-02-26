"""Continuous watch daemon for Loom.

Monitors Mac-native source databases for changes via FSEvents (watchdog)
and polls API-based sources on intervals. When changes are detected,
runs the pipeline (ingest through synthesis) with debouncing.

Usage:
    python -m loom.cli watch --llm gemini -v
    python -m loom.cli watch --dry-run --debounce 3
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from loom.constants import (
    ADDRESSBOOK_DIR,
    IMESSAGE_DB,
    MAIL_DB,
    WATCH_DEBOUNCE_SECONDS,
    WATCH_POLL_CALENDAR_SECONDS,
    WATCH_POLL_FILE_SOURCES_SECONDS,
    WATCH_POLL_GRANOLA_SECONDS,
    WATCH_POLL_SLACK_SECONDS,
    WHATSAPP_BASE,
)

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# File watch targets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Map source name -> (watch_dir, target_filenames, recursive)
WATCH_TARGETS: dict[str, tuple[Path, frozenset[str], bool]] = {
    "mail": (
        MAIL_DB.parent,
        frozenset({"Envelope Index", "Envelope Index-wal", "Envelope Index-shm"}),
        False,
    ),
    # NOTE: iMessage is NOT in WATCH_TARGETS because macOS suppresses FSEvents
    # for ~/Library/Messages/chat.db (written by imagent under TCC protection).
    # It's polled via POLL_SOURCES instead.
    "whatsapp": (
        WHATSAPP_BASE,
        frozenset({
            "ChatStorage.sqlite", "ChatStorage.sqlite-wal", "ChatStorage.sqlite-shm",
            "LID.sqlite", "LID.sqlite-wal", "LID.sqlite-shm",
            "CallHistory.sqlite", "CallHistory.sqlite-wal", "CallHistory.sqlite-shm",
        }),
        False,
    ),
    "contacts": (
        ADDRESSBOOK_DIR,
        frozenset({"AddressBook-v22.abcddb", "AddressBook-v22.abcddb-wal",
                   "AddressBook-v22.abcddb-shm"}),
        True,
    ),
}

# Polled sources — either API-based or where FSEvents is unreliable
POLL_SOURCES: dict[str, float] = {
    "imessage": WATCH_POLL_FILE_SOURCES_SECONDS,  # macOS suppresses FSEvents for chat.db
    "calendar": WATCH_POLL_CALENDAR_SECONDS,
    "slack": WATCH_POLL_SLACK_SECONDS,
    "granola": WATCH_POLL_GRANOLA_SECONDS,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FSEvents handler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LoomFileHandler(FileSystemEventHandler):
    """Watchdog handler that filters for target filenames and triggers callback."""

    def __init__(
        self,
        source_name: str,
        target_filenames: frozenset[str],
        trigger_callback: Any,
    ) -> None:
        super().__init__()
        self.source_name = source_name
        self.target_filenames = target_filenames
        self.trigger_callback = trigger_callback

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        filename = Path(event.src_path).name
        if filename in self.target_filenames:
            logger.debug("FSEvent: %s modified (%s)", filename, self.source_name)
            self.trigger_callback(self.source_name)

    def on_created(self, event: FileSystemEvent) -> None:
        self.on_modified(event)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Watch daemon
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class WatchDaemon:
    """Orchestrator: watches files, polls APIs, debounces, triggers pipeline.

    Architecture:
      - watchdog Observer for file-based sources (mail, imessage, whatsapp, contacts)
      - Polling threads for API sources (calendar, slack, granola)
      - Shared debounce: each trigger resets the timer. When timer fires,
        pipeline runs with accumulated triggered_sources.
      - Serialized execution: Lock ensures one pipeline run at a time.
        Triggers during a run set _rerun_requested with merged sources.
      - Graceful shutdown via SIGINT/SIGTERM -> threading.Event.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        debounce_seconds: float = WATCH_DEBOUNCE_SECONDS,
        poll_intervals: dict[str, float] | None = None,
        enabled_sources: list[str] | None = None,
    ) -> None:
        self.args = args
        self.debounce_seconds = debounce_seconds
        self.poll_intervals = poll_intervals or dict(POLL_SOURCES)
        self.enabled_sources = set(enabled_sources) if enabled_sources else None

        # Shutdown coordination
        self._shutdown_event = threading.Event()

        # Debounce state
        self._trigger_lock = threading.Lock()
        self._triggered_sources: set[str] = set()
        self._debounce_timer: threading.Timer | None = None

        # Pipeline serialization
        self._pipeline_lock = threading.Lock()
        self._rerun_requested = False
        self._rerun_sources: set[str] = set()

        # Stats
        self._pipeline_runs = 0
        self._last_run_time: float = 0

    def _source_enabled(self, source: str) -> bool:
        """Check if a source is enabled for watching."""
        if self.enabled_sources is None:
            return True
        return source in self.enabled_sources

    def trigger(self, source_name: str) -> None:
        """Called when a source changes. Resets the debounce timer."""
        with self._trigger_lock:
            self._triggered_sources.add(source_name)

            # Cancel existing timer and start a new one
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()

            self._debounce_timer = threading.Timer(
                self.debounce_seconds, self._debounce_fired,
            )
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def _debounce_fired(self) -> None:
        """Debounce timer expired — collect sources and run pipeline."""
        with self._trigger_lock:
            sources = self._triggered_sources.copy()
            self._triggered_sources.clear()
            self._debounce_timer = None

        if not sources:
            return

        self._run_pipeline(sources)

    def _run_pipeline(self, sources: set[str]) -> None:
        """Run the pipeline, serialized. Queue re-run if already running."""
        acquired = self._pipeline_lock.acquire(blocking=False)
        if not acquired:
            # Pipeline already running — queue a re-run
            with self._trigger_lock:
                self._rerun_requested = True
                self._rerun_sources.update(sources)
            logger.info(
                "Pipeline busy, queued re-run for: %s",
                ", ".join(sorted(sources)),
            )
            return

        try:
            self._execute_pipeline(sources)
        finally:
            self._pipeline_lock.release()

        # Check for queued re-run
        rerun_sources: set[str] = set()
        with self._trigger_lock:
            if self._rerun_requested:
                rerun_sources = self._rerun_sources.copy()
                self._rerun_requested = False
                self._rerun_sources.clear()

        if rerun_sources:
            logger.info(
                "Processing queued re-run for: %s",
                ", ".join(sorted(rerun_sources)),
            )
            self._run_pipeline(rerun_sources)

    def _execute_pipeline(self, sources: set[str]) -> None:
        """Actually run the pipeline stages."""
        from loom.cli import run_pipeline_stages

        self._pipeline_runs += 1
        logger.info(
            "Pipeline run #%d triggered by: %s",
            self._pipeline_runs, ", ".join(sorted(sources)),
        )

        # Narrow ingest to triggered sources only
        pipeline_args = argparse.Namespace(**vars(self.args))
        pipeline_args.sources = list(sources)

        t0 = time.time()
        try:
            summary = run_pipeline_stages(pipeline_args)
            elapsed = time.time() - t0
            self._last_run_time = elapsed
            logger.info(
                "Pipeline run #%d complete: %d stages, %d failed, %.1fs",
                self._pipeline_runs,
                summary["stages_run"],
                summary["stages_failed"],
                elapsed,
            )
        except Exception as exc:
            logger.error("Pipeline run #%d failed: %s", self._pipeline_runs, exc)

    def _start_file_watchers(self) -> Observer:
        """Start watchdog observers for file-based sources."""
        observer = Observer()

        for source_name, (watch_dir, target_files, recursive) in WATCH_TARGETS.items():
            if not self._source_enabled(source_name):
                continue
            if not watch_dir.exists():
                logger.warning(
                    "Watch dir for %s not found: %s (skipping)",
                    source_name, watch_dir,
                )
                continue

            handler = LoomFileHandler(source_name, target_files, self.trigger)
            observer.schedule(handler, str(watch_dir), recursive=recursive)
            logger.info("Watching %s: %s", source_name, watch_dir)

        observer.start()
        return observer

    def _poll_source(self, source_name: str, interval: float) -> None:
        """Polling thread for an API-based source."""
        logger.info("Polling %s every %.0fs", source_name, interval)
        while not self._shutdown_event.wait(timeout=interval):
            logger.debug("Poll trigger: %s", source_name)
            self.trigger(source_name)

    def _start_pollers(self) -> list[threading.Thread]:
        """Start polling threads for API-based and FSEvents-unreliable sources."""
        threads = []
        for source_name, interval in self.poll_intervals.items():
            if not self._source_enabled(source_name):
                continue

            t = threading.Thread(
                target=self._poll_source,
                args=(source_name, interval),
                name=f"poll-{source_name}",
                daemon=True,
            )
            t.start()
            threads.append(t)

        return threads

    def _setup_signals(self) -> None:
        """Register signal handlers for graceful shutdown."""
        def _handler(signum: int, frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            logger.info("Received %s, shutting down...", sig_name)
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def run(self) -> None:
        """Main entry point. Blocks until shutdown signal."""
        self._setup_signals()

        print("\n" + "=" * 60)
        print("  Loom Watch Daemon")
        print("=" * 60)
        print(f"  Debounce: {self.debounce_seconds}s")
        if self.enabled_sources:
            print(f"  Sources: {', '.join(sorted(self.enabled_sources))}")
        else:
            print("  Sources: all available")
        print(f"  Poll intervals: {self.poll_intervals}")
        print("  Press Ctrl-C to stop\n")

        # Initial catchup run with all enabled file sources
        initial_sources = set()
        for source_name in WATCH_TARGETS:
            if self._source_enabled(source_name):
                initial_sources.add(source_name)
        for source_name in self.poll_intervals:
            if self._source_enabled(source_name):
                initial_sources.add(source_name)

        if initial_sources:
            logger.info("Initial catchup run...")
            self._execute_pipeline(initial_sources)

        # Start watchers and pollers
        observer = self._start_file_watchers()
        poll_threads = self._start_pollers()

        # Block until shutdown
        try:
            while not self._shutdown_event.wait(timeout=1.0):
                pass
        finally:
            logger.info("Stopping file watchers...")
            observer.stop()
            observer.join(timeout=5)

            # Cancel any pending debounce timer
            with self._trigger_lock:
                if self._debounce_timer is not None:
                    self._debounce_timer.cancel()

            # Poll threads are daemons — they'll exit with the process
            logger.info(
                "Shutdown complete. %d pipeline runs total.", self._pipeline_runs,
            )
