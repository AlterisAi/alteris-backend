"""Tests for Loom ambient adapters: KnowledgeC, Safari, Chrome, Notes, Shell History.

All tests use synthetic in-memory SQLite databases or temp files. No real databases.
Monkeypatching overrides path constants so adapters read from controlled test data.
"""

import gzip
import sqlite3
import time
from pathlib import Path

import pytest

from loom.adapters.knowledgec import KnowledgeCAdapter
from loom.adapters.safari import SafariAdapter
from loom.adapters.chrome import ChromiumAdapter, get_chromium_adapters, TRANSITION_TYPES
from loom.adapters.notes import NotesAdapter, _extract_note_text
from loom.adapters.shell_history import (
    ShellHistoryAdapter,
    _sanitize_command,
    _parse_zsh_history,
)
from loom.constants import APPLE_EPOCH_OFFSET, CHROME_EPOCH_OFFSET
from loom.models import Event


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _create_knowledgec_db(path: Path, rows: list[dict] | None = None) -> None:
    """Create a knowledgeC.db with ZOBJECT table and optional rows."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE ZOBJECT (
            Z_PK INTEGER PRIMARY KEY,
            ZSTREAMNAME TEXT,
            ZVALUESTRING TEXT,
            ZSTARTDATE REAL,
            ZENDDATE REAL,
            ZSECONDSFROMGMT INTEGER
        )
    """)
    if rows:
        for r in rows:
            conn.execute(
                "INSERT INTO ZOBJECT (Z_PK, ZSTREAMNAME, ZVALUESTRING, ZSTARTDATE, ZENDDATE, ZSECONDSFROMGMT) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (r["z_pk"], r.get("stream", "/app/usage"), r.get("bundle", "com.apple.Safari"),
                 r.get("start"), r.get("end"), r.get("gmt", -28800)),
            )
    conn.commit()
    conn.close()


def _create_safari_db(path: Path, rows: list[dict] | None = None, with_title: bool = True) -> None:
    """Create a Safari History.db with history_visits + history_items tables."""
    conn = sqlite3.connect(str(path))
    if with_title:
        conn.execute("CREATE TABLE history_items (id INTEGER PRIMARY KEY, url TEXT, domain_expansion TEXT, title TEXT)")
    else:
        conn.execute("CREATE TABLE history_items (id INTEGER PRIMARY KEY, url TEXT, domain_expansion TEXT)")
    conn.execute("CREATE TABLE history_visits (id INTEGER PRIMARY KEY, history_item INTEGER, visit_time REAL)")
    if rows:
        for r in rows:
            item_id = r["item_id"]
            if with_title:
                conn.execute(
                    "INSERT OR IGNORE INTO history_items (id, url, domain_expansion, title) VALUES (?, ?, ?, ?)",
                    (item_id, r["url"], r.get("domain", ""), r.get("title", "")),
                )
            else:
                conn.execute(
                    "INSERT OR IGNORE INTO history_items (id, url, domain_expansion) VALUES (?, ?, ?)",
                    (item_id, r["url"], r.get("domain", "")),
                )
            conn.execute(
                "INSERT INTO history_visits (id, history_item, visit_time) VALUES (?, ?, ?)",
                (r["visit_id"], item_id, r["visit_time"]),
            )
    conn.commit()
    conn.close()


def _create_chrome_db(path: Path, rows: list[dict] | None = None) -> None:
    """Create a Chrome History db with visits + urls tables."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE urls (id INTEGER PRIMARY KEY, url TEXT, title TEXT)")
    conn.execute("""
        CREATE TABLE visits (
            id INTEGER PRIMARY KEY,
            url INTEGER,
            visit_time INTEGER,
            visit_duration INTEGER DEFAULT 0,
            transition INTEGER DEFAULT 0
        )
    """)
    if rows:
        for r in rows:
            url_id = r["url_id"]
            conn.execute(
                "INSERT OR IGNORE INTO urls (id, url, title) VALUES (?, ?, ?)",
                (url_id, r["url"], r.get("title", "")),
            )
            conn.execute(
                "INSERT INTO visits (id, url, visit_time, visit_duration, transition) VALUES (?, ?, ?, ?, ?)",
                (r["visit_id"], url_id, r["visit_time"], r.get("duration", 0), r.get("transition", 0)),
            )
    conn.commit()
    conn.close()


def _create_notes_db(path: Path, rows: list[dict] | None = None) -> None:
    """Create a NoteStore.sqlite with the required tables and optional rows."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE ZICCLOUDSYNCINGOBJECT (
            Z_PK INTEGER PRIMARY KEY,
            ZIDENTIFIER TEXT,
            ZTITLE1 TEXT,
            ZMODIFICATIONDATE1 REAL,
            ZCREATIONDATE3 REAL,
            ZISPASSWORDPROTECTED INTEGER DEFAULT 0,
            ZFOLDER INTEGER,
            ZNOTEDATA INTEGER,
            ZTITLE2 TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE ZICNOTEDATA (
            Z_PK INTEGER PRIMARY KEY,
            ZDATA BLOB
        )
    """)
    if rows:
        for r in rows:
            # Insert note data blob if provided
            note_data_pk = r.get("notedata_pk")
            if note_data_pk is not None:
                conn.execute(
                    "INSERT OR IGNORE INTO ZICNOTEDATA (Z_PK, ZDATA) VALUES (?, ?)",
                    (note_data_pk, r.get("data_blob")),
                )
            # Insert the folder row if provided (for the LEFT JOIN in the query)
            folder_pk = r.get("folder_pk")
            if folder_pk is not None:
                conn.execute(
                    "INSERT OR IGNORE INTO ZICCLOUDSYNCINGOBJECT "
                    "(Z_PK, ZIDENTIFIER, ZTITLE1, ZMODIFICATIONDATE1, ZCREATIONDATE3, "
                    " ZISPASSWORDPROTECTED, ZFOLDER, ZNOTEDATA, ZTITLE2) "
                    "VALUES (?, NULL, NULL, NULL, NULL, 0, NULL, NULL, ?)",
                    (folder_pk, r.get("folder_name", "")),
                )
            conn.execute(
                "INSERT OR IGNORE INTO ZICCLOUDSYNCINGOBJECT "
                "(Z_PK, ZIDENTIFIER, ZTITLE1, ZMODIFICATIONDATE1, ZCREATIONDATE3, "
                " ZISPASSWORDPROTECTED, ZFOLDER, ZNOTEDATA, ZTITLE2) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
                (r["pk"], r["identifier"], r.get("title"), r.get("mod_date"),
                 r.get("create_date"), r.get("password_protected", 0),
                 folder_pk, note_data_pk),
            )
    conn.commit()
    conn.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KnowledgeC Adapter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestKnowledgeCAdapter:

    def test_check_availability_present(self, tmp_path, monkeypatch):
        db_path = tmp_path / "knowledgeC.db"
        _create_knowledgec_db(db_path)
        monkeypatch.setattr("loom.adapters.knowledgec.KNOWLEDGEC_DB", db_path)
        adapter = KnowledgeCAdapter()
        result = adapter.check_availability()
        assert result.available is True

    def test_check_availability_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("loom.adapters.knowledgec.KNOWLEDGEC_DB", tmp_path / "nonexistent.db")
        adapter = KnowledgeCAdapter()
        result = adapter.check_availability()
        assert result.available is False
        assert result.reason == "database_not_found"

    def test_check_schema_missing_table(self, tmp_path, monkeypatch):
        """DB exists but lacks the ZOBJECT table."""
        db_path = tmp_path / "knowledgeC.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE OTHER (id INTEGER)")
        conn.close()
        monkeypatch.setattr("loom.adapters.knowledgec.KNOWLEDGEC_DB", db_path)
        adapter = KnowledgeCAdapter()
        result = adapter.check_schema()
        assert result.compatible is False
        assert "ZOBJECT" in result.missing_tables

    def test_ingest_basic(self, tmp_path, monkeypatch):
        """Verify correct Event field mapping from ZOBJECT rows."""
        now_apple = int(time.time()) - APPLE_EPOCH_OFFSET
        db_path = tmp_path / "knowledgeC.db"
        _create_knowledgec_db(db_path, rows=[
            {"z_pk": 1, "bundle": "com.apple.Safari", "start": now_apple, "end": now_apple + 120},
            {"z_pk": 2, "bundle": "com.apple.Mail", "start": now_apple + 200, "end": now_apple + 400},
        ])
        monkeypatch.setattr("loom.adapters.knowledgec.KNOWLEDGEC_DB", db_path)
        adapter = KnowledgeCAdapter()
        result = adapter.ingest()
        assert len(result.events) == 2
        assert result.errors == []
        ev = result.events[0]
        assert ev.source == "knowledgec"
        assert ev.event_type == "app_focus"
        assert ev.metadata["bundle_id"] in ("com.apple.Safari", "com.apple.Mail")
        assert ev.metadata["is_from_me"] is True

    def test_timestamp_conversion(self, tmp_path, monkeypatch):
        """Apple epoch start + APPLE_EPOCH_OFFSET should equal the correct unix ts."""
        apple_start = 700_000_000  # arbitrary Apple epoch seconds
        expected_unix = apple_start + APPLE_EPOCH_OFFSET
        db_path = tmp_path / "knowledgeC.db"
        _create_knowledgec_db(db_path, rows=[
            {"z_pk": 1, "bundle": "com.test.app", "start": apple_start, "end": apple_start + 60},
        ])
        monkeypatch.setattr("loom.adapters.knowledgec.KNOWLEDGEC_DB", db_path)
        adapter = KnowledgeCAdapter()
        result = adapter.ingest()
        assert len(result.events) == 1
        assert result.events[0].timestamp == expected_unix

    def test_since_ts_filtering(self, tmp_path, monkeypatch):
        """Only events after since_ts should be returned."""
        # Event 1: old (apple ts = 600_000_000, unix = 600M + offset)
        # Event 2: recent (apple ts = 800_000_000, unix = 800M + offset)
        db_path = tmp_path / "knowledgeC.db"
        _create_knowledgec_db(db_path, rows=[
            {"z_pk": 1, "bundle": "com.old.app", "start": 600_000_000, "end": 600_000_100},
            {"z_pk": 2, "bundle": "com.new.app", "start": 800_000_000, "end": 800_000_100},
        ])
        monkeypatch.setattr("loom.adapters.knowledgec.KNOWLEDGEC_DB", db_path)
        adapter = KnowledgeCAdapter()
        # since_ts in unix: only events where apple_start > since_ts - APPLE_EPOCH_OFFSET
        # We want to filter out event 1 (apple 600M) but keep event 2 (apple 800M)
        cutoff_unix = 700_000_000 + APPLE_EPOCH_OFFSET
        result = adapter.ingest(since_ts=cutoff_unix)
        assert len(result.events) == 1
        assert result.events[0].raw_content == "com.new.app"

    def test_limit(self, tmp_path, monkeypatch):
        """Limiting to N rows should return at most N events."""
        now_apple = int(time.time()) - APPLE_EPOCH_OFFSET
        db_path = tmp_path / "knowledgeC.db"
        _create_knowledgec_db(db_path, rows=[
            {"z_pk": i, "bundle": f"com.app{i}.test", "start": now_apple + i * 10, "end": now_apple + i * 10 + 5}
            for i in range(1, 11)
        ])
        monkeypatch.setattr("loom.adapters.knowledgec.KNOWLEDGEC_DB", db_path)
        adapter = KnowledgeCAdapter()
        result = adapter.ingest(limit=3)
        assert len(result.events) == 3

    def test_deterministic_ids(self, tmp_path, monkeypatch):
        """Ingesting twice should produce identical Event IDs."""
        now_apple = int(time.time()) - APPLE_EPOCH_OFFSET
        db_path = tmp_path / "knowledgeC.db"
        _create_knowledgec_db(db_path, rows=[
            {"z_pk": 42, "bundle": "com.test.deterministic", "start": now_apple, "end": now_apple + 30},
        ])
        monkeypatch.setattr("loom.adapters.knowledgec.KNOWLEDGEC_DB", db_path)
        adapter = KnowledgeCAdapter()
        r1 = adapter.ingest()
        r2 = adapter.ingest()
        assert r1.events[0].id == r2.events[0].id
        assert r1.events[0].id == Event.make_id("knowledgec", "42")

    def test_empty_db(self, tmp_path, monkeypatch):
        """ZOBJECT exists but has no rows: get empty IngestResult."""
        db_path = tmp_path / "knowledgeC.db"
        _create_knowledgec_db(db_path)
        monkeypatch.setattr("loom.adapters.knowledgec.KNOWLEDGEC_DB", db_path)
        adapter = KnowledgeCAdapter()
        result = adapter.ingest()
        assert len(result.events) == 0
        assert result.errors == []

    def test_skip_null_bundle(self, tmp_path, monkeypatch):
        """Rows with NULL ZVALUESTRING are skipped."""
        now_apple = int(time.time()) - APPLE_EPOCH_OFFSET
        db_path = tmp_path / "knowledgeC.db"
        _create_knowledgec_db(db_path, rows=[
            {"z_pk": 1, "bundle": None, "start": now_apple, "end": now_apple + 10},
            {"z_pk": 2, "bundle": "com.valid.app", "start": now_apple + 20, "end": now_apple + 30},
        ])
        monkeypatch.setattr("loom.adapters.knowledgec.KNOWLEDGEC_DB", db_path)
        adapter = KnowledgeCAdapter()
        result = adapter.ingest()
        assert len(result.events) == 1
        assert result.events[0].raw_content == "com.valid.app"

    def test_duration_calculation(self, tmp_path, monkeypatch):
        """Duration should be end - start in seconds."""
        now_apple = int(time.time()) - APPLE_EPOCH_OFFSET
        db_path = tmp_path / "knowledgeC.db"
        _create_knowledgec_db(db_path, rows=[
            {"z_pk": 1, "bundle": "com.app.test", "start": now_apple, "end": now_apple + 300},
        ])
        monkeypatch.setattr("loom.adapters.knowledgec.KNOWLEDGEC_DB", db_path)
        adapter = KnowledgeCAdapter()
        result = adapter.ingest()
        assert result.events[0].metadata["duration_seconds"] == 300


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Safari Adapter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSafariAdapter:

    def test_check_availability_present(self, tmp_path, monkeypatch):
        db_path = tmp_path / "History.db"
        _create_safari_db(db_path)
        monkeypatch.setattr("loom.adapters.safari.SAFARI_HISTORY_DB", db_path)
        adapter = SafariAdapter()
        result = adapter.check_availability()
        assert result.available is True

    def test_check_availability_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("loom.adapters.safari.SAFARI_HISTORY_DB", tmp_path / "no.db")
        adapter = SafariAdapter()
        result = adapter.check_availability()
        assert result.available is False

    def test_check_schema_missing_table(self, tmp_path, monkeypatch):
        """DB exists but lacks history_visits table."""
        db_path = tmp_path / "History.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE history_items (id INTEGER)")
        conn.close()
        monkeypatch.setattr("loom.adapters.safari.SAFARI_HISTORY_DB", db_path)
        adapter = SafariAdapter()
        result = adapter.check_schema()
        assert result.compatible is False
        assert "history_visits" in result.missing_tables

    def test_ingest_basic(self, tmp_path, monkeypatch):
        """Verify correct Event field mapping from Safari history rows."""
        now_cf = int(time.time()) - APPLE_EPOCH_OFFSET
        db_path = tmp_path / "History.db"
        _create_safari_db(db_path, rows=[
            {
                "visit_id": 1, "item_id": 100, "visit_time": now_cf,
                "url": "https://example.com/page", "domain": "example.com",
                "title": "Example Page",
            },
        ])
        monkeypatch.setattr("loom.adapters.safari.SAFARI_HISTORY_DB", db_path)
        adapter = SafariAdapter()
        result = adapter.ingest()
        assert len(result.events) == 1
        ev = result.events[0]
        assert ev.source == "safari"
        assert ev.event_type == "browser_visit"
        assert ev.metadata["url"] == "https://example.com/page"
        assert ev.metadata["domain"] == "example.com"
        assert ev.metadata["title"] == "Example Page"
        assert ev.metadata["is_from_me"] is True

    def test_timestamp_conversion(self, tmp_path, monkeypatch):
        """CFAbsoluteTime + APPLE_EPOCH_OFFSET should produce correct Unix ts."""
        cf_time = 700_000_000
        expected_unix = cf_time + APPLE_EPOCH_OFFSET
        db_path = tmp_path / "History.db"
        _create_safari_db(db_path, rows=[
            {"visit_id": 1, "item_id": 1, "visit_time": cf_time, "url": "https://test.com", "domain": "test.com"},
        ])
        monkeypatch.setattr("loom.adapters.safari.SAFARI_HISTORY_DB", db_path)
        adapter = SafariAdapter()
        result = adapter.ingest()
        assert result.events[0].timestamp == expected_unix

    def test_since_ts_filtering(self, tmp_path, monkeypatch):
        """Only visits after since_ts should be returned."""
        db_path = tmp_path / "History.db"
        _create_safari_db(db_path, rows=[
            {"visit_id": 1, "item_id": 1, "visit_time": 600_000_000, "url": "https://old.com", "domain": "old.com"},
            {"visit_id": 2, "item_id": 2, "visit_time": 800_000_000, "url": "https://new.com", "domain": "new.com"},
        ])
        monkeypatch.setattr("loom.adapters.safari.SAFARI_HISTORY_DB", db_path)
        adapter = SafariAdapter()
        cutoff = 700_000_000 + APPLE_EPOCH_OFFSET
        result = adapter.ingest(since_ts=cutoff)
        assert len(result.events) == 1
        assert result.events[0].metadata["url"] == "https://new.com"

    def test_skip_internal_urls(self, tmp_path, monkeypatch):
        """about:, blob:, data: URLs should be skipped."""
        now_cf = int(time.time()) - APPLE_EPOCH_OFFSET
        db_path = tmp_path / "History.db"
        _create_safari_db(db_path, rows=[
            {"visit_id": 1, "item_id": 1, "visit_time": now_cf, "url": "about:blank", "domain": ""},
            {"visit_id": 2, "item_id": 2, "visit_time": now_cf + 1, "url": "blob:https://x.com/abc", "domain": ""},
            {"visit_id": 3, "item_id": 3, "visit_time": now_cf + 2, "url": "data:text/html,<h1>Hi</h1>", "domain": ""},
            {"visit_id": 4, "item_id": 4, "visit_time": now_cf + 3, "url": "https://valid.com", "domain": "valid.com"},
        ])
        monkeypatch.setattr("loom.adapters.safari.SAFARI_HISTORY_DB", db_path)
        adapter = SafariAdapter()
        result = adapter.ingest()
        assert len(result.events) == 1
        assert result.events[0].metadata["url"] == "https://valid.com"

    def test_deterministic_ids(self, tmp_path, monkeypatch):
        """Ingesting twice produces identical Event IDs."""
        now_cf = int(time.time()) - APPLE_EPOCH_OFFSET
        db_path = tmp_path / "History.db"
        _create_safari_db(db_path, rows=[
            {"visit_id": 99, "item_id": 1, "visit_time": now_cf, "url": "https://stable.com", "domain": "stable.com"},
        ])
        monkeypatch.setattr("loom.adapters.safari.SAFARI_HISTORY_DB", db_path)
        adapter = SafariAdapter()
        r1 = adapter.ingest()
        r2 = adapter.ingest()
        assert r1.events[0].id == r2.events[0].id
        assert r1.events[0].id == Event.make_id("safari", "99")

    def test_empty_db(self, tmp_path, monkeypatch):
        """Tables exist but no rows: empty IngestResult."""
        db_path = tmp_path / "History.db"
        _create_safari_db(db_path)
        monkeypatch.setattr("loom.adapters.safari.SAFARI_HISTORY_DB", db_path)
        adapter = SafariAdapter()
        result = adapter.ingest()
        assert len(result.events) == 0
        assert result.errors == []

    def test_domain_fallback_from_url(self, tmp_path, monkeypatch):
        """When domain_expansion is empty, domain should be parsed from URL."""
        now_cf = int(time.time()) - APPLE_EPOCH_OFFSET
        db_path = tmp_path / "History.db"
        _create_safari_db(db_path, rows=[
            {"visit_id": 1, "item_id": 1, "visit_time": now_cf,
             "url": "https://fallback-domain.com/path", "domain": ""},
        ])
        monkeypatch.setattr("loom.adapters.safari.SAFARI_HISTORY_DB", db_path)
        adapter = SafariAdapter()
        result = adapter.ingest()
        assert result.events[0].metadata["domain"] == "fallback-domain.com"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chrome/Chromium Adapter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestChromeAdapter:

    def test_check_availability_present(self, tmp_path):
        db_path = tmp_path / "History"
        _create_chrome_db(db_path)
        adapter = ChromiumAdapter("test_chrome", db_path)
        result = adapter.check_availability()
        assert result.available is True

    def test_check_availability_missing(self, tmp_path):
        adapter = ChromiumAdapter("test_chrome", tmp_path / "nonexistent")
        result = adapter.check_availability()
        assert result.available is False
        assert result.reason == "database_not_found"

    def test_ingest_basic(self, tmp_path):
        """Verify correct Event field mapping from Chrome history."""
        # Chrome timestamp: Unix seconds -> chrome microseconds
        unix_now = int(time.time())
        chrome_ts = (unix_now * 1_000_000) + CHROME_EPOCH_OFFSET
        db_path = tmp_path / "History"
        _create_chrome_db(db_path, rows=[
            {
                "visit_id": 1, "url_id": 10, "visit_time": chrome_ts,
                "url": "https://example.com/chrome", "title": "Chrome Page",
                "duration": 5_000_000, "transition": 1,
            },
        ])
        adapter = ChromiumAdapter("test_chrome", db_path)
        result = adapter.ingest()
        assert len(result.events) == 1
        ev = result.events[0]
        assert ev.source == "test_chrome"
        assert ev.event_type == "browser_visit"
        assert ev.metadata["url"] == "https://example.com/chrome"
        assert ev.metadata["title"] == "Chrome Page"
        assert ev.metadata["domain"] == "example.com"
        assert ev.metadata["is_from_me"] is True

    def test_timestamp_conversion(self, tmp_path):
        """Chrome epoch microseconds should convert to correct Unix seconds."""
        known_unix = 1_700_000_000
        chrome_ts = (known_unix * 1_000_000) + CHROME_EPOCH_OFFSET
        db_path = tmp_path / "History"
        _create_chrome_db(db_path, rows=[
            {"visit_id": 1, "url_id": 1, "visit_time": chrome_ts, "url": "https://ts-test.com", "title": ""},
        ])
        adapter = ChromiumAdapter("test_chrome", db_path)
        result = adapter.ingest()
        assert result.events[0].timestamp == known_unix

    def test_since_ts_filtering(self, tmp_path):
        """Only events after since_ts should be returned."""
        old_unix = 1_600_000_000
        new_unix = 1_700_000_000
        old_chrome = (old_unix * 1_000_000) + CHROME_EPOCH_OFFSET
        new_chrome = (new_unix * 1_000_000) + CHROME_EPOCH_OFFSET
        db_path = tmp_path / "History"
        _create_chrome_db(db_path, rows=[
            {"visit_id": 1, "url_id": 1, "visit_time": old_chrome, "url": "https://old.com"},
            {"visit_id": 2, "url_id": 2, "visit_time": new_chrome, "url": "https://new.com"},
        ])
        adapter = ChromiumAdapter("test_chrome", db_path)
        cutoff = 1_650_000_000
        result = adapter.ingest(since_ts=cutoff)
        assert len(result.events) == 1
        assert result.events[0].metadata["url"] == "https://new.com"

    def test_skip_internal_urls(self, tmp_path):
        """chrome://, about:, etc. should be skipped."""
        unix_now = int(time.time())
        chrome_ts = (unix_now * 1_000_000) + CHROME_EPOCH_OFFSET
        db_path = tmp_path / "History"
        _create_chrome_db(db_path, rows=[
            {"visit_id": 1, "url_id": 1, "visit_time": chrome_ts, "url": "chrome://settings"},
            {"visit_id": 2, "url_id": 2, "visit_time": chrome_ts + 1_000_000, "url": "about:blank"},
            {"visit_id": 3, "url_id": 3, "visit_time": chrome_ts + 2_000_000, "url": "chrome-extension://abc/popup.html"},
            {"visit_id": 4, "url_id": 4, "visit_time": chrome_ts + 3_000_000, "url": "brave://settings"},
            {"visit_id": 5, "url_id": 5, "visit_time": chrome_ts + 4_000_000, "url": "https://valid.com"},
        ])
        adapter = ChromiumAdapter("test_chrome", db_path)
        result = adapter.ingest()
        assert len(result.events) == 1
        assert result.events[0].metadata["url"] == "https://valid.com"

    def test_db_copy_mechanism(self, tmp_path):
        """Adapter copies the DB to a temp location before reading.

        Verify that the original DB file is intact after ingestion and
        that the adapter produces correct results from the copy.
        """
        unix_now = int(time.time())
        chrome_ts = (unix_now * 1_000_000) + CHROME_EPOCH_OFFSET
        db_path = tmp_path / "History"
        _create_chrome_db(db_path, rows=[
            {"visit_id": 1, "url_id": 1, "visit_time": chrome_ts, "url": "https://copy-test.com", "title": "Copy"},
        ])
        original_size = db_path.stat().st_size
        adapter = ChromiumAdapter("test_chrome", db_path)
        result = adapter.ingest()
        # Original file should be untouched
        assert db_path.exists()
        assert db_path.stat().st_size == original_size
        # Data should have been read correctly from the copy
        assert len(result.events) == 1
        assert result.events[0].metadata["url"] == "https://copy-test.com"

    def test_transition_type_decoding(self, tmp_path):
        """Transition bitmask lower 8 bits should decode to named types."""
        unix_now = int(time.time())
        chrome_ts = (unix_now * 1_000_000) + CHROME_EPOCH_OFFSET
        db_path = tmp_path / "History"
        # transition=1 (typed), transition=0x02000007 (form_submit with redirect flag)
        _create_chrome_db(db_path, rows=[
            {"visit_id": 1, "url_id": 1, "visit_time": chrome_ts, "url": "https://typed.com",
             "transition": 1},
            {"visit_id": 2, "url_id": 2, "visit_time": chrome_ts + 1_000_000, "url": "https://form.com",
             "transition": 0x02000007},
        ])
        adapter = ChromiumAdapter("test_chrome", db_path)
        result = adapter.ingest()
        events_by_url = {e.metadata["url"]: e for e in result.events}
        assert events_by_url["https://typed.com"].metadata["transition_type"] == "typed"
        # 0x02000007 & 0xFF = 7 -> form_submit
        assert events_by_url["https://form.com"].metadata["transition_type"] == "form_submit"

    def test_deterministic_ids(self, tmp_path):
        """Ingesting twice produces identical Event IDs."""
        unix_now = int(time.time())
        chrome_ts = (unix_now * 1_000_000) + CHROME_EPOCH_OFFSET
        db_path = tmp_path / "History"
        _create_chrome_db(db_path, rows=[
            {"visit_id": 77, "url_id": 1, "visit_time": chrome_ts, "url": "https://stable.com"},
        ])
        adapter = ChromiumAdapter("test_chrome", db_path)
        r1 = adapter.ingest()
        r2 = adapter.ingest()
        assert r1.events[0].id == r2.events[0].id
        assert r1.events[0].id == Event.make_id("test_chrome", "77")

    def test_multi_browser_discovery(self, tmp_path, monkeypatch):
        """get_chromium_adapters() should discover browsers whose History DB exists."""
        # Create two browser paths
        chrome_path = tmp_path / "Chrome" / "Default" / "History"
        arc_path = tmp_path / "Arc" / "Default" / "History"
        chrome_path.parent.mkdir(parents=True)
        arc_path.parent.mkdir(parents=True)
        _create_chrome_db(chrome_path)
        _create_chrome_db(arc_path)

        fake_paths = {
            "chrome": chrome_path,
            "arc": arc_path,
            "brave": tmp_path / "Brave" / "nonexistent",
            "edge": tmp_path / "Edge" / "nonexistent",
        }
        # get_chromium_adapters() does `from loom.constants import CHROMIUM_BROWSER_PATHS`
        # inside the function body, so we patch the constants module directly.
        import loom.constants
        monkeypatch.setattr(loom.constants, "CHROMIUM_BROWSER_PATHS", fake_paths)

        adapters = get_chromium_adapters()
        names = {a.source_name for a in adapters}
        assert "chrome" in names
        assert "arc" in names
        assert "brave" not in names
        assert "edge" not in names


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Notes Adapter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNotesAdapter:

    def test_check_availability_present(self, tmp_path, monkeypatch):
        db_path = tmp_path / "NoteStore.sqlite"
        _create_notes_db(db_path)
        monkeypatch.setattr("loom.adapters.notes.NOTES_DB", db_path)
        adapter = NotesAdapter()
        result = adapter.check_availability()
        assert result.available is True

    def test_check_availability_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("loom.adapters.notes.NOTES_DB", tmp_path / "missing.sqlite")
        adapter = NotesAdapter()
        result = adapter.check_availability()
        assert result.available is False

    def test_check_schema_missing_table(self, tmp_path, monkeypatch):
        """DB exists but missing ZICNOTEDATA table."""
        db_path = tmp_path / "NoteStore.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE ZICCLOUDSYNCINGOBJECT (Z_PK INTEGER)")
        conn.close()
        monkeypatch.setattr("loom.adapters.notes.NOTES_DB", db_path)
        adapter = NotesAdapter()
        result = adapter.check_schema()
        assert result.compatible is False
        assert "ZICNOTEDATA" in result.missing_tables

    def test_ingest_basic(self, tmp_path, monkeypatch):
        """Verify correct Event field mapping from Notes rows."""
        now_apple = int(time.time()) - APPLE_EPOCH_OFFSET
        test_text = b"This is a test note with some important content for extraction"
        compressed = gzip.compress(test_text)
        db_path = tmp_path / "NoteStore.sqlite"
        _create_notes_db(db_path, rows=[
            {
                "pk": 10, "identifier": "note-uuid-123", "title": "My Test Note",
                "mod_date": now_apple, "create_date": now_apple - 1000,
                "notedata_pk": 20, "data_blob": compressed,
            },
        ])
        monkeypatch.setattr("loom.adapters.notes.NOTES_DB", db_path)
        adapter = NotesAdapter()
        result = adapter.ingest()
        assert len(result.events) == 1
        ev = result.events[0]
        assert ev.source == "notes"
        assert ev.event_type == "note"
        assert ev.metadata["subject"] == "My Test Note"
        assert ev.metadata["is_from_me"] is True
        assert ev.source_id == "note-uuid-123"

    def test_timestamp_conversion(self, tmp_path, monkeypatch):
        """Apple epoch + APPLE_EPOCH_OFFSET should produce correct Unix ts."""
        apple_mod = 700_000_000
        expected_unix = apple_mod + APPLE_EPOCH_OFFSET
        db_path = tmp_path / "NoteStore.sqlite"
        _create_notes_db(db_path, rows=[
            {
                "pk": 1, "identifier": "ts-test", "title": "Timestamp Note",
                "mod_date": apple_mod, "create_date": apple_mod - 100,
                "notedata_pk": 2, "data_blob": gzip.compress(b"Some content for the note body"),
            },
        ])
        monkeypatch.setattr("loom.adapters.notes.NOTES_DB", db_path)
        adapter = NotesAdapter()
        result = adapter.ingest()
        assert result.events[0].timestamp == expected_unix

    def test_skip_password_protected(self, tmp_path, monkeypatch):
        """Notes with ZISPASSWORDPROTECTED=1 should be skipped."""
        now_apple = int(time.time()) - APPLE_EPOCH_OFFSET
        db_path = tmp_path / "NoteStore.sqlite"
        _create_notes_db(db_path, rows=[
            {
                "pk": 1, "identifier": "locked-note", "title": "Secret Note",
                "mod_date": now_apple, "create_date": now_apple,
                "password_protected": 1, "notedata_pk": 10,
                "data_blob": gzip.compress(b"Secret content hidden"),
            },
            {
                "pk": 2, "identifier": "open-note", "title": "Open Note",
                "mod_date": now_apple + 1, "create_date": now_apple,
                "password_protected": 0, "notedata_pk": 11,
                "data_blob": gzip.compress(b"Visible content in this open note"),
            },
        ])
        monkeypatch.setattr("loom.adapters.notes.NOTES_DB", db_path)
        adapter = NotesAdapter()
        result = adapter.ingest()
        assert len(result.events) == 1
        assert result.events[0].source_id == "open-note"

    def test_gzip_content_extraction(self):
        """Gzip-compressed blob with known text should be extractable."""
        test_text = b"This is a test note with some content that should be extracted"
        compressed = gzip.compress(test_text)
        extracted = _extract_note_text(compressed)
        # The text extraction looks for runs of printable ASCII >= 4 chars
        assert "test note" in extracted.lower()
        assert "extracted" in extracted.lower()

    def test_gzip_content_extraction_empty(self):
        """Null/empty data should return empty string."""
        assert _extract_note_text(None) == ""
        assert _extract_note_text(b"") == ""

    def test_deterministic_ids(self, tmp_path, monkeypatch):
        """Ingesting twice produces identical Event IDs."""
        now_apple = int(time.time()) - APPLE_EPOCH_OFFSET
        db_path = tmp_path / "NoteStore.sqlite"
        _create_notes_db(db_path, rows=[
            {
                "pk": 5, "identifier": "stable-note", "title": "Stable",
                "mod_date": now_apple, "create_date": now_apple,
                "notedata_pk": 6, "data_blob": gzip.compress(b"Content for stable note"),
            },
        ])
        monkeypatch.setattr("loom.adapters.notes.NOTES_DB", db_path)
        adapter = NotesAdapter()
        r1 = adapter.ingest()
        r2 = adapter.ingest()
        assert r1.events[0].id == r2.events[0].id
        assert r1.events[0].id == Event.make_id("notes", "stable-note")

    def test_empty_db(self, tmp_path, monkeypatch):
        """Tables exist but no rows: empty IngestResult."""
        db_path = tmp_path / "NoteStore.sqlite"
        _create_notes_db(db_path)
        monkeypatch.setattr("loom.adapters.notes.NOTES_DB", db_path)
        adapter = NotesAdapter()
        result = adapter.ingest()
        assert len(result.events) == 0
        assert result.errors == []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shell History Adapter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestShellHistoryAdapter:

    def test_check_availability_present(self, tmp_path, monkeypatch):
        hist_path = tmp_path / ".zsh_history"
        hist_path.write_text(": 1700000000:0;echo hello\n")
        monkeypatch.setattr("loom.adapters.shell_history.ZSH_HISTORY_FILE", hist_path)
        adapter = ShellHistoryAdapter()
        result = adapter.check_availability()
        assert result.available is True

    def test_check_availability_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("loom.adapters.shell_history.ZSH_HISTORY_FILE", tmp_path / "no_zsh")
        monkeypatch.setattr("loom.adapters.shell_history.BASH_HISTORY_FILE", tmp_path / "no_bash")
        adapter = ShellHistoryAdapter()
        result = adapter.check_availability()
        assert result.available is False

    def test_parse_zsh_format(self, tmp_path, monkeypatch):
        """Verify `: TIMESTAMP:0;COMMAND` parsing."""
        hist_path = tmp_path / ".zsh_history"
        hist_path.write_text(
            ": 1700000000:0;ls -la\n"
            ": 1700000100:0;git status\n"
            ": 1700000200:0;python main.py\n"
        )
        monkeypatch.setattr("loom.adapters.shell_history.ZSH_HISTORY_FILE", hist_path)
        monkeypatch.setattr("loom.adapters.shell_history.BASH_HISTORY_FILE", tmp_path / "no_bash")
        adapter = ShellHistoryAdapter()
        result = adapter.ingest()
        assert len(result.events) == 3
        commands = [e.metadata["command"] for e in result.events]
        assert "ls -la" in commands
        assert "git status" in commands
        assert "python main.py" in commands

    def test_multiline_commands(self, tmp_path, monkeypatch):
        r"""Backslash continuation should be joined into one command."""
        hist_path = tmp_path / ".zsh_history"
        hist_path.write_text(
            ": 1700000000:0;echo hello \\\n"
            "world\n"
            ": 1700000100:0;ls\n"
        )
        monkeypatch.setattr("loom.adapters.shell_history.ZSH_HISTORY_FILE", hist_path)
        monkeypatch.setattr("loom.adapters.shell_history.BASH_HISTORY_FILE", tmp_path / "no_bash")
        adapter = ShellHistoryAdapter()
        result = adapter.ingest()
        assert len(result.events) == 2
        # The multiline command should be joined
        multiline_ev = [e for e in result.events if "hello" in e.metadata["command"]][0]
        assert "world" in multiline_ev.metadata["command"]

    def test_secret_sanitization(self):
        """export TOKEN=secret should become TOKEN=[REDACTED]."""
        sanitized = _sanitize_command("export API_KEY=sk-abc123def456")
        assert "sk-abc123def456" not in sanitized
        assert "[REDACTED]" in sanitized
        assert "API_KEY" in sanitized

    def test_sanitize_curl_auth(self):
        """curl -H 'Authorization: Bearer xxx' should be redacted."""
        sanitized = _sanitize_command('curl -H "Authorization: Bearer my_secret_token" https://api.example.com')
        assert "my_secret_token" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_since_ts_filtering(self, tmp_path, monkeypatch):
        """Commands before since_ts should be excluded."""
        hist_path = tmp_path / ".zsh_history"
        hist_path.write_text(
            ": 1600000000:0;old command\n"
            ": 1700000000:0;new command\n"
        )
        monkeypatch.setattr("loom.adapters.shell_history.ZSH_HISTORY_FILE", hist_path)
        monkeypatch.setattr("loom.adapters.shell_history.BASH_HISTORY_FILE", tmp_path / "no_bash")
        adapter = ShellHistoryAdapter()
        result = adapter.ingest(since_ts=1650000000)
        assert len(result.events) == 1
        assert "new command" in result.events[0].metadata["command"]

    def test_deterministic_ids(self, tmp_path, monkeypatch):
        """Ingesting twice produces identical Event IDs."""
        hist_path = tmp_path / ".zsh_history"
        hist_path.write_text(": 1700000000:0;echo stable\n")
        monkeypatch.setattr("loom.adapters.shell_history.ZSH_HISTORY_FILE", hist_path)
        monkeypatch.setattr("loom.adapters.shell_history.BASH_HISTORY_FILE", tmp_path / "no_bash")
        adapter = ShellHistoryAdapter()
        r1 = adapter.ingest()
        r2 = adapter.ingest()
        assert len(r1.events) == 1
        assert r1.events[0].id == r2.events[0].id

    def test_empty_file(self, tmp_path, monkeypatch):
        """Empty history file should produce empty result (no errors)."""
        hist_path = tmp_path / ".zsh_history"
        hist_path.write_text("")
        monkeypatch.setattr("loom.adapters.shell_history.ZSH_HISTORY_FILE", hist_path)
        monkeypatch.setattr("loom.adapters.shell_history.BASH_HISTORY_FILE", tmp_path / "no_bash")
        adapter = ShellHistoryAdapter()
        result = adapter.ingest()
        assert len(result.events) == 0

    def test_sanitize_password_flag(self):
        """mysql -p password should be redacted."""
        sanitized = _sanitize_command("mysql -u root -p mysecretpass dbname")
        assert "mysecretpass" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_sanitize_bearer_token(self):
        """Bare Bearer token should be redacted."""
        sanitized = _sanitize_command("curl -H 'Authorization: Bearer eyJhbGciOiJIUzI1NiJ9'")
        assert "eyJhbGciOiJIUzI1NiJ9" not in sanitized
        assert "[REDACTED]" in sanitized
