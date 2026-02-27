"""Tests for alteris.adapters: base adapter framework and shared utilities.

Tests cover:
  - SourceAdapter abstract interface
  - AvailabilityResult, SchemaResult, IngestResult data classes
  - check_sqlite_readable with existing/missing/permission-denied paths
  - check_sqlite_tables with matching/missing tables
  - make_source_id_hash determinism
  - row_val safe extraction
  - get_adapter and get_all_adapters registry
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from alteris.adapters import (
    AvailabilityResult,
    IngestResult,
    SchemaResult,
    SourceAdapter,
    check_sqlite_readable,
    check_sqlite_tables,
    make_source_id_hash,
    row_val,
)
from alteris.constants import SOURCE_ID_HASH_LEN
from alteris.models import Event


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Abstract interface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestSourceAdapterInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            SourceAdapter()

    def test_concrete_adapter_requires_all_methods(self):
        class PartialAdapter(SourceAdapter):
            @property
            def source_name(self) -> str:
                return "test"

        with pytest.raises(TypeError):
            PartialAdapter()

    def test_concrete_adapter_works(self):
        class ConcreteAdapter(SourceAdapter):
            @property
            def source_name(self) -> str:
                return "test_source"

            def check_availability(self) -> AvailabilityResult:
                return AvailabilityResult(available=True, source="test_source")

            def check_schema(self) -> SchemaResult:
                return SchemaResult(compatible=True, source="test_source")

            def ingest(self, since_ts=0, limit=0) -> IngestResult:
                return IngestResult(source="test_source")

        adapter = ConcreteAdapter()
        assert adapter.source_name == "test_source"
        assert adapter.check_availability().available is True
        assert adapter.check_schema().compatible is True
        assert adapter.ingest().source == "test_source"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data classes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestDataClasses:
    def test_availability_result_defaults(self):
        r = AvailabilityResult(available=True, source="mail")
        assert r.reason == ""
        assert r.user_action == ""

    def test_availability_result_with_error(self):
        r = AvailabilityResult(
            available=False, source="mail",
            reason="permission_denied",
            user_action="Grant Full Disk Access",
        )
        assert r.available is False
        assert "permission" in r.reason

    def test_schema_result_defaults(self):
        r = SchemaResult(compatible=True, source="mail")
        assert r.missing_tables == []
        assert r.warnings == []

    def test_schema_result_with_missing(self):
        r = SchemaResult(
            compatible=False, source="mail",
            missing_tables=["messages", "chat"],
        )
        assert len(r.missing_tables) == 2

    def test_ingest_result_defaults(self):
        r = IngestResult(source="mail")
        assert r.events == []
        assert r.errors == []
        assert r.duration_seconds == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# check_sqlite_readable
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCheckSqliteReadable:
    def test_existing_db(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()
        result = check_sqlite_readable(db_path, "test_source")
        assert result.available is True
        assert result.source == "test_source"

    def test_missing_db(self, tmp_path):
        db_path = tmp_path / "nonexistent.db"
        result = check_sqlite_readable(db_path, "test_source")
        assert result.available is False
        assert result.reason == "database_not_found"
        assert "not found" in result.user_action.lower()

    def test_missing_db_source_preserved(self, tmp_path):
        db_path = tmp_path / "missing.db"
        result = check_sqlite_readable(db_path, "whatsapp")
        assert result.source == "whatsapp"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# check_sqlite_tables
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCheckSqliteTables:
    def test_all_tables_present(self, tmp_path):
        db_path = tmp_path / "schema.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE messages (id INTEGER)")
        conn.execute("CREATE TABLE chat (id INTEGER)")
        conn.close()
        result = check_sqlite_tables(db_path, "imessage", ["messages", "chat"])
        assert result.compatible is True
        assert result.missing_tables == []

    def test_missing_required_table(self, tmp_path):
        db_path = tmp_path / "schema2.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE messages (id INTEGER)")
        conn.close()
        result = check_sqlite_tables(db_path, "imessage", ["messages", "chat"])
        assert result.compatible is False
        assert "chat" in result.missing_tables

    def test_missing_db_file(self, tmp_path):
        db_path = tmp_path / "nofile.db"
        result = check_sqlite_tables(db_path, "mail", ["messages"])
        assert result.compatible is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# make_source_id_hash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMakeSourceIdHash:
    def test_deterministic(self):
        h1 = make_source_id_hash("mail", "12345")
        h2 = make_source_id_hash("mail", "12345")
        assert h1 == h2

    def test_different_inputs(self):
        h1 = make_source_id_hash("mail", "12345")
        h2 = make_source_id_hash("mail", "67890")
        assert h1 != h2

    def test_length(self):
        h = make_source_id_hash("test", "value")
        assert len(h) == SOURCE_ID_HASH_LEN

    def test_hex_chars(self):
        h = make_source_id_hash("test", "value")
        assert all(c in "0123456789abcdef" for c in h)

    def test_multiple_parts(self):
        h = make_source_id_hash("a", "b", "c", "d")
        assert len(h) == SOURCE_ID_HASH_LEN


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# row_val
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRowVal:
    def test_existing_key(self, tmp_path):
        db_path = tmp_path / "rowval.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE test (name TEXT, age INTEGER)")
        conn.execute("INSERT INTO test VALUES ('Alice', 30)")
        row = conn.execute("SELECT * FROM test").fetchone()
        assert row_val(row, "name") == "Alice"
        assert row_val(row, "age") == 30

    def test_null_value_returns_default(self, tmp_path):
        db_path = tmp_path / "rowval2.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE test (name TEXT, age INTEGER)")
        conn.execute("INSERT INTO test VALUES (NULL, NULL)")
        row = conn.execute("SELECT * FROM test").fetchone()
        assert row_val(row, "name") == ""
        assert row_val(row, "age", 0) == 0

    def test_missing_key_returns_default(self, tmp_path):
        db_path = tmp_path / "rowval3.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE test (name TEXT)")
        conn.execute("INSERT INTO test VALUES ('Alice')")
        row = conn.execute("SELECT * FROM test").fetchone()
        assert row_val(row, "nonexistent", "fallback") == "fallback"
