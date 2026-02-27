"""Three-layer graph storage: Events -> Claims -> Beliefs.

Each layer is a separate set of tables in the same SQLite database.
Cross-layer edges connect Claims to Events.
Beliefs store source claim IDs as a JSON array column.

All writes are transactional. Reads are non-blocking (WAL mode).

Usage:
    store = LayeredGraphStore()
    store.put_event(event)
    store.put_claim(claim)
    store.put_belief(belief)

    # Paper trail: Belief -> Claims -> Events
    belief = store.get_belief(belief_id)
    claims = [store.get_claim(cid) for cid in belief.source_claims]
    events = store.get_events_for_claim(claim_id)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from alteris.constants import DEFAULT_EVENT_QUERY_LIMIT, DEFAULT_QUERY_LIMIT
from alteris.models import (
    Annotation,
    Belief,
    BeliefStatus,
    BeliefType,
    Claim,
    EpistemicLevel,
    Event,
    ExtractionMethod,
    ExtractionProvenance,
    Modality,
    Projection,
)
from alteris.privacy import SensitivityLevel

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Schema
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCHEMA_SQL = """
-- Layer 1: Events (immutable, append-only)

CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    source_id       TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    timestamp       INTEGER NOT NULL,
    participants    TEXT DEFAULT '[]',
    raw_content     TEXT,
    content_hash    TEXT DEFAULT '',
    metadata        TEXT DEFAULT '{}',
    sensitivity     INTEGER DEFAULT 2,
    created_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_source ON events(source, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_hash ON events(content_hash) WHERE content_hash != '';
CREATE INDEX IF NOT EXISTS idx_events_source_threadid ON events(source, json_extract(metadata, '$.thread_id'));
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_source_id ON events(source, source_id);


-- Layer 2: Claims (versioned, with full provenance)

CREATE TABLE IF NOT EXISTS claims (
    id              TEXT PRIMARY KEY,
    claim_type      TEXT NOT NULL,
    subject         TEXT NOT NULL,
    predicate       TEXT NOT NULL,
    object          TEXT NOT NULL,
    confidence      REAL DEFAULT 0.5,
    modality        TEXT DEFAULT 'unknown',
    sensitivity     INTEGER DEFAULT 2,

    model_id        TEXT DEFAULT 'deterministic',
    prompt_version  TEXT DEFAULT '',
    context_hash    TEXT DEFAULT '',
    extraction_method TEXT DEFAULT 'deterministic',

    user_verified   INTEGER,
    user_correction TEXT,

    superseded_by   TEXT,
    created_at      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject);
CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);
CREATE INDEX IF NOT EXISTS idx_claims_predicate ON claims(subject, predicate);
CREATE INDEX IF NOT EXISTS idx_claims_active ON claims(superseded_by) WHERE superseded_by IS NULL;
CREATE INDEX IF NOT EXISTS idx_claims_model ON claims(model_id);
CREATE INDEX IF NOT EXISTS idx_claims_confidence ON claims(confidence);

CREATE TABLE IF NOT EXISTS claim_events (
    claim_id        TEXT NOT NULL,
    event_id        TEXT NOT NULL,
    PRIMARY KEY (claim_id, event_id)
);

CREATE INDEX IF NOT EXISTS idx_claim_events_event ON claim_events(event_id);


-- Layer 3: Beliefs (mutable, aggregated from claims)

CREATE TABLE IF NOT EXISTS beliefs (
    id              TEXT PRIMARY KEY,
    belief_type     TEXT NOT NULL,
    subject         TEXT NOT NULL,
    summary         TEXT NOT NULL,
    data            TEXT NOT NULL,
    epistemic_level TEXT NOT NULL,
    source_reliability TEXT DEFAULT 'F',
    info_credibility INTEGER DEFAULT 6,
    confidence      REAL DEFAULT 0.5,
    source_claims   TEXT NOT NULL,
    inference_chain TEXT,
    evidence_log    TEXT,
    status          TEXT DEFAULT 'active',
    supersedes      TEXT,
    superseded_by   TEXT,
    priority        INTEGER,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    expires_at      INTEGER
);

CREATE INDEX IF NOT EXISTS idx_beliefs_type_status ON beliefs(belief_type, status);
CREATE INDEX IF NOT EXISTS idx_beliefs_subject ON beliefs(subject);
CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON beliefs(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_beliefs_updated ON beliefs(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_beliefs_supersedes ON beliefs(supersedes);
CREATE INDEX IF NOT EXISTS idx_beliefs_expires ON beliefs(expires_at) WHERE expires_at IS NOT NULL;


-- Cross-source identity: Person registry

CREATE TABLE IF NOT EXISTS persons (
    person_id       TEXT PRIMARY KEY,
    canonical_name  TEXT DEFAULT '',
    is_user         INTEGER DEFAULT 0,
    sources         TEXT DEFAULT '[]',
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS person_identifiers (
    person_id       TEXT NOT NULL,
    identifier_type TEXT NOT NULL,
    identifier      TEXT NOT NULL,
    display_name    TEXT DEFAULT '',
    source          TEXT DEFAULT '',
    PRIMARY KEY (identifier_type, identifier)
);

CREATE INDEX IF NOT EXISTS idx_person_ids_person ON person_identifiers(person_id);


-- Event-Person links

CREATE TABLE IF NOT EXISTS event_persons (
    event_id        TEXT NOT NULL,
    person_id       TEXT NOT NULL,
    role            TEXT NOT NULL,
    PRIMARY KEY (event_id, person_id, role)
);

CREATE INDEX IF NOT EXISTS idx_event_persons_person ON event_persons(person_id);
CREATE INDEX IF NOT EXISTS idx_event_persons_event ON event_persons(event_id);


-- Person-Event participation (materialized expansion of thread membership).
-- Thread-based sources (WhatsApp, iMessage, mail) only store per-thread
-- membership edges in event_persons. This table expands those to per-event
-- rows so "find all events involving person X" is a single indexed lookup.
-- Rebuilt by the linker after each ingest.

CREATE TABLE IF NOT EXISTS person_events (
    person_id   TEXT NOT NULL,
    event_id    TEXT NOT NULL,
    PRIMARY KEY (person_id, event_id)
) WITHOUT ROWID;


-- Thread membership mapping (source, thread_id, person_id).
-- Small lookup table used to rebuild person_events efficiently.

CREATE TABLE IF NOT EXISTS thread_members (
    source      TEXT NOT NULL,
    thread_id   TEXT NOT NULL,
    person_id   TEXT NOT NULL,
    PRIMARY KEY (source, thread_id, person_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_tm_person ON thread_members(person_id);


-- Annotations: faceted observations about events (lens-independent, additive)

CREATE TABLE IF NOT EXISTS annotations (
    event_id    TEXT NOT NULL,
    facet       TEXT NOT NULL,
    value       TEXT NOT NULL,
    confidence  REAL DEFAULT 1.0,
    source      TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (event_id, facet, value, source)
);

CREATE INDEX IF NOT EXISTS idx_annotations_facet ON annotations(facet, value);
CREATE INDEX IF NOT EXISTS idx_annotations_event ON annotations(event_id);


-- Projections: lens-scoped scores (disposable read model, recomputable)

CREATE TABLE IF NOT EXISTS projections (
    event_id    TEXT NOT NULL,
    lens        TEXT NOT NULL,
    score       REAL NOT NULL,
    route       TEXT NOT NULL,
    components  TEXT DEFAULT '{}',
    computed_at INTEGER NOT NULL,
    PRIMARY KEY (event_id, lens)
);

CREATE INDEX IF NOT EXISTS idx_projections_lens_route ON projections(lens, route);
CREATE INDEX IF NOT EXISTS idx_projections_lens_score ON projections(lens, score DESC);


-- Sync state tracking

CREATE TABLE IF NOT EXISTS sync_state (
    source          TEXT PRIMARY KEY,
    last_sync       INTEGER,
    last_event_ts   INTEGER,
    event_count     INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'idle',
    error_message   TEXT DEFAULT '',
    cursor          TEXT DEFAULT '{}',
    updated_at      INTEGER NOT NULL
);


-- Clarity Queue tasks
CREATE TABLE IF NOT EXISTS cq_tasks (
    id          TEXT PRIMARY KEY,
    bucket      TEXT NOT NULL,
    title       TEXT NOT NULL,
    note        TEXT DEFAULT '',
    done        INTEGER DEFAULT 0,
    source      TEXT DEFAULT 'manual',
    due_date    TEXT,
    labels      TEXT,
    position    INTEGER DEFAULT 0,
    claim_id    TEXT,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cq_bucket ON cq_tasks(bucket, position);

-- CQ undo log (append-only, prunable)
CREATE TABLE IF NOT EXISTS cq_undo_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    action      TEXT NOT NULL,
    task_id     TEXT NOT NULL,
    prev_state  TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_undo_created ON cq_undo_log(created_at DESC);

-- Sender rules (priority tiers or block)
CREATE TABLE IF NOT EXISTS sender_rules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern     TEXT NOT NULL,
    priority    TEXT NOT NULL,
    source      TEXT DEFAULT '',
    note        TEXT DEFAULT '',
    created_at  INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sender_pattern ON sender_rules(pattern, source);

-- User-defined categories
CREATE TABLE IF NOT EXISTS cq_categories (
    name        TEXT PRIMARY KEY,
    color       TEXT DEFAULT '',
    icon        TEXT DEFAULT '',
    position    INTEGER DEFAULT 0,
    created_at  INTEGER NOT NULL
);

-- CQ chat sessions
CREATE TABLE IF NOT EXISTS cq_sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT DEFAULT '',
    session_type TEXT DEFAULT 'clarity',
    messages    TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);

-- API spend tracking
CREATE TABLE IF NOT EXISTS api_spend (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,
    provider    TEXT NOT NULL,
    model       TEXT NOT NULL,
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cost_usd    REAL DEFAULT 0.0,
    source      TEXT DEFAULT '',
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spend_date_provider ON api_spend(date, provider);

-- CQ Stories: group related tasks
CREATE TABLE IF NOT EXISTS cq_stories (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    source      TEXT DEFAULT 'auto',
    color       TEXT DEFAULT '',
    icon        TEXT DEFAULT '',
    status      TEXT DEFAULT 'active',
    priority    INTEGER,
    priority_override INTEGER,
    cluster_hash TEXT,
    updated_at  INTEGER NOT NULL,
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_stories_status ON cq_stories(status, updated_at DESC);

-- Story members: tasks belonging to stories
CREATE TABLE IF NOT EXISTS cq_story_members (
    story_id    TEXT NOT NULL,
    task_id     TEXT NOT NULL,
    position    INTEGER DEFAULT 0,
    added_at    INTEGER NOT NULL,
    PRIMARY KEY (story_id, task_id)
);
CREATE INDEX IF NOT EXISTS idx_story_members_task ON cq_story_members(task_id);

-- Story persons: people associated with stories
CREATE TABLE IF NOT EXISTS cq_story_persons (
    story_id    TEXT NOT NULL,
    person_id   TEXT NOT NULL,
    role        TEXT DEFAULT '',
    added_at    INTEGER NOT NULL,
    PRIMARY KEY (story_id, person_id)
);

-- Anti-links: user said "these don't belong together"
CREATE TABLE IF NOT EXISTS cq_story_anti_links (
    task_id_a   TEXT NOT NULL,
    task_id_b   TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (task_id_a, task_id_b)
);

-- Topic normalization: raw topic string -> canonical form
CREATE TABLE IF NOT EXISTS topic_synonyms (
    raw_topic   TEXT PRIMARY KEY,
    canonical   TEXT NOT NULL,
    method      TEXT NOT NULL DEFAULT 'deterministic',
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_topic_syn_canonical ON topic_synonyms(canonical);


-- Person profiles: pre-computed per-person engagement data from Stage 1 claims.
-- Populated after Stage 1 extraction, queryable by any downstream module.

CREATE TABLE IF NOT EXISTS person_profiles (
    person_id              TEXT PRIMARY KEY,
    canonical_name         TEXT NOT NULL,
    message_count          INTEGER DEFAULT 0,
    direct_count           INTEGER DEFAULT 0,
    group_count            INTEGER DEFAULT 0,
    tier                   INTEGER DEFAULT 4,
    user_initiated_ratio   REAL,
    channels               TEXT DEFAULT '[]',
    channel_count          INTEGER DEFAULT 0,
    days_since_last        REAL,
    first_contact_ts       INTEGER,
    last_contact_ts        INTEGER,
    relationship_span_days REAL,
    is_user                INTEGER DEFAULT 0,
    previous_tier          INTEGER,
    previous_message_count INTEGER,
    computed_at            INTEGER NOT NULL,
    FOREIGN KEY (person_id) REFERENCES persons(person_id)
);
CREATE INDEX IF NOT EXISTS idx_pp_tier ON person_profiles(tier);
CREATE INDEX IF NOT EXISTS idx_pp_msg ON person_profiles(message_count DESC);
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Store implementation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LayeredGraphStore:
    """Three-layer graph backed by SQLite."""

    def __init__(self, db_path: Path | str | None = None):
        from alteris.constants import DEFAULT_DB_PATH
        if db_path == ":memory:":
            self._db_path = ":memory:"
        elif db_path is not None:
            self._db_path = os.path.expanduser(str(db_path))
        else:
            self._db_path = os.path.expanduser(str(DEFAULT_DB_PATH))
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if self._db_path != ":memory:":
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA_SQL)
            self._migrate_cq_schema()
            self._migrate_person_profiles()
            self._conn.commit()
        return self._conn

    def close(self):
        if self._conn:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.OperationalError:
                pass  # DB may be locked by other readers
            self._conn.close()
            self._conn = None

    def checkpoint(self):
        """Truncate the WAL file to reclaim disk space.

        Safe to call periodically — if other readers hold the DB open,
        a PASSIVE checkpoint runs instead (no-op on the WAL size but
        still flushes pages to the main DB).
        """
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

    def _migrate_cq_schema(self):
        """Add new columns to cq_tasks if missing. Safe for existing DBs."""
        cols = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(cq_tasks)").fetchall()
        }
        migrations = [
            ("defer_until", "TEXT"),
            ("recurrence", "TEXT"),
            ("custom_fields", "TEXT"),
            ("category", "TEXT"),
            ("story_id", "TEXT"),
            ("seen_at", "INTEGER"),
            ("priority_override", "INTEGER"),
            ("accepted", "INTEGER"),
        ]
        for col_name, col_type in migrations:
            if col_name not in cols:
                self._conn.execute(
                    f"ALTER TABLE cq_tasks ADD COLUMN {col_name} {col_type}"
                )

        # Migrate cq_stories: add cluster_hash if missing
        story_cols = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(cq_stories)").fetchall()
        }
        if "cluster_hash" not in story_cols:
            self._conn.execute(
                "ALTER TABLE cq_stories ADD COLUMN cluster_hash TEXT"
            )

        # cq_extractable_fields: user-defined fields injected into synthesis prompt
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cq_extractable_fields (
                name        TEXT PRIMARY KEY,
                description TEXT DEFAULT '',
                example     TEXT DEFAULT '',
                position    INTEGER DEFAULT 0,
                created_at  INTEGER NOT NULL
            )
        """)

        # Person Model: 11-dimension estimated user profile
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS person_model (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                version         INTEGER NOT NULL DEFAULT 1,
                model_json      TEXT NOT NULL,
                user_corrections TEXT DEFAULT '{}',
                estimated_at    INTEGER NOT NULL,
                confirmed_at    INTEGER,
                estimation_method TEXT DEFAULT 'scout'
            )
        """)

    def _migrate_person_profiles(self):
        """Add direct_count/group_count columns to person_profiles if missing."""
        cols = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(person_profiles)").fetchall()
        }
        for col_name in ("direct_count", "group_count"):
            if col_name not in cols:
                self._conn.execute(
                    f"ALTER TABLE person_profiles ADD COLUMN {col_name} INTEGER DEFAULT 0"
                )

    # ── Layer 1: Events ──────────────────────────────────────────

    def put_event(self, event: Event) -> bool:
        """Insert an event. Returns True if new, False if duplicate."""
        cursor = self.conn.execute(
            """INSERT OR IGNORE INTO events
               (id, source, source_id, event_type, timestamp, participants,
                raw_content, content_hash, metadata, sensitivity, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.id,
                event.source,
                event.source_id,
                event.event_type,
                event.timestamp,
                json.dumps(list(event.participants)),
                event.raw_content,
                event.content_hash,
                json.dumps(event.metadata),
                event.sensitivity.value,
                event.created_at,
            ),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def put_events_batch(self, events: list[Event]) -> int:
        """Insert a batch of events in a single transaction. Returns count inserted."""
        inserted = 0
        with self.conn:
            for event in events:
                cursor = self.conn.execute(
                    """INSERT OR IGNORE INTO events
                       (id, source, source_id, event_type, timestamp, participants,
                        raw_content, content_hash, metadata, sensitivity, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event.id,
                        event.source,
                        event.source_id,
                        event.event_type,
                        event.timestamp,
                        json.dumps(list(event.participants)),
                        event.raw_content,
                        event.content_hash,
                        json.dumps(event.metadata),
                        event.sensitivity.value,
                        event.created_at,
                    ),
                )
                if cursor.rowcount > 0:
                    inserted += 1
        return inserted

    def get_event(self, event_id: str) -> Event | None:
        """Fetch a single event by ID."""
        row = self.conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        return self._row_to_event(row) if row else None

    def get_events(
        self,
        since: int = 0,
        until: int = 0,
        source: str | None = None,
        event_type: str | None = None,
        limit: int = DEFAULT_EVENT_QUERY_LIMIT,
    ) -> list[Event]:
        """Fetch events with flexible filters."""
        conditions: list[str] = []
        params: list[Any] = []
        if since > 0:
            conditions.append("timestamp >= ?")
            params.append(since)
        if until > 0:
            conditions.append("timestamp <= ?")
            params.append(until)
        if source:
            conditions.append("source = ?")
            params.append(source)
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        params.append(limit)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self.conn.execute(
            f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_events_by_source(
        self, source: str, since: int = 0, limit: int = DEFAULT_EVENT_QUERY_LIMIT,
    ) -> list[Event]:
        """Fetch events from a source, optionally since a timestamp."""
        rows = self.conn.execute(
            """SELECT * FROM events
               WHERE source = ? AND timestamp >= ?
               ORDER BY timestamp DESC LIMIT ?""",
            (source, since, limit),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def count_events(self, source: str | None = None) -> int:
        """Count events, optionally filtered by source."""
        if source:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM events WHERE source = ?", (source,)
            ).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return row[0]

    def get_events_for_claim(self, claim_id: str) -> list[Event]:
        """Paper trail: which events support a claim?"""
        rows = self.conn.execute(
            """SELECT e.* FROM events e
               JOIN claim_events ce ON e.id = ce.event_id
               WHERE ce.claim_id = ?
               ORDER BY e.timestamp""",
            (claim_id,),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        return Event(
            id=row["id"],
            source=row["source"],
            source_id=row["source_id"],
            event_type=row["event_type"],
            timestamp=row["timestamp"],
            participants=tuple(json.loads(row["participants"] or "[]")),
            raw_content=row["raw_content"],
            content_hash=row["content_hash"] or "",
            metadata=json.loads(row["metadata"] or "{}"),
            sensitivity=SensitivityLevel(row["sensitivity"]),
            created_at=row["created_at"],
        )

    # ── Layer 2: Claims ──────────────────────────────────────────

    def put_claim(self, claim: Claim, *, commit: bool = True) -> bool:
        """Insert a claim with its event links. Returns True if new.

        Set commit=False for batch inserts, then call conn.commit() once.
        """
        cursor = self.conn.execute(
            """INSERT OR IGNORE INTO claims
               (id, claim_type, subject, predicate, object, confidence,
                modality, sensitivity, model_id, prompt_version, context_hash,
                extraction_method, user_verified, user_correction,
                superseded_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                claim.id,
                claim.claim_type,
                claim.subject,
                claim.predicate,
                claim.object,
                claim.confidence,
                claim.modality.value,
                claim.sensitivity.value,
                claim.provenance.model_id,
                claim.provenance.prompt_version,
                claim.provenance.context_hash,
                claim.provenance.extraction_method.value,
                claim.user_verified if claim.user_verified is not None else None,
                claim.user_correction,
                claim.superseded_by,
                claim.created_at,
            ),
        )
        is_new = cursor.rowcount > 0

        for event_id in claim.event_ids:
            self.conn.execute(
                "INSERT OR IGNORE INTO claim_events (claim_id, event_id) VALUES (?, ?)",
                (claim.id, event_id),
            )
        if commit:
            self.conn.commit()
        return is_new

    def get_claim(self, claim_id: str) -> Claim | None:
        """Fetch a single claim by ID."""
        row = self.conn.execute(
            "SELECT * FROM claims WHERE id = ?", (claim_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_claim(row)

    def get_active_claims(
        self,
        subject: str | None = None,
        claim_type: str | None = None,
        min_confidence: float = 0.0,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> list[Claim]:
        """Fetch active (non-superseded) claims, optionally filtered."""
        conditions = ["superseded_by IS NULL"]
        params: list[Any] = []
        if subject:
            conditions.append("subject = ?")
            params.append(subject)
        if claim_type:
            conditions.append("claim_type = ?")
            params.append(claim_type)
        if min_confidence > 0:
            conditions.append("confidence >= ?")
            params.append(min_confidence)
        params.append(limit)

        rows = self.conn.execute(
            f"""SELECT * FROM claims
                WHERE {' AND '.join(conditions)}
                ORDER BY confidence DESC LIMIT ?""",
            params,
        ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def get_claims_for_event(self, event_id: str) -> list[Claim]:
        """Reverse lookup: which claims cite this event?"""
        rows = self.conn.execute(
            """SELECT c.* FROM claims c
               JOIN claim_events ce ON c.id = ce.claim_id
               WHERE ce.event_id = ?
               ORDER BY c.confidence DESC""",
            (event_id,),
        ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def supersede_claim(self, old_id: str, new_id: str):
        """Mark a claim as superseded by a newer version."""
        self.conn.execute(
            "UPDATE claims SET superseded_by = ? WHERE id = ?",
            (new_id, old_id),
        )
        self.conn.commit()

    def update_claim_confidence(self, claim_id: str, confidence: float):
        """Update the confidence score of a claim (used by propagation)."""
        self.conn.execute(
            "UPDATE claims SET confidence = ? WHERE id = ?",
            (confidence, claim_id),
        )
        self.conn.commit()

    def _row_to_claim(self, row: sqlite3.Row) -> Claim:
        event_rows = self.conn.execute(
            "SELECT event_id FROM claim_events WHERE claim_id = ?",
            (row["id"],),
        ).fetchall()
        event_ids = [r["event_id"] for r in event_rows]

        uv = row["user_verified"]
        user_verified = None if uv is None else bool(uv)

        return Claim(
            id=row["id"],
            event_ids=event_ids,
            claim_type=row["claim_type"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            confidence=row["confidence"],
            modality=Modality(row["modality"]),
            provenance=ExtractionProvenance(
                model_id=row["model_id"],
                prompt_version=row["prompt_version"],
                context_hash=row["context_hash"],
                extraction_method=ExtractionMethod(row["extraction_method"]),
                extracted_at=row["created_at"],
            ),
            user_verified=user_verified,
            user_correction=row["user_correction"],
            superseded_by=row["superseded_by"],
            sensitivity=SensitivityLevel(row["sensitivity"]),
            created_at=row["created_at"],
        )

    # ── Layer 3: Beliefs ─────────────────────────────────────────

    def put_belief(self, belief: Belief) -> bool:
        """Insert or update a belief. Returns True on success."""
        now = int(time.time())
        self.conn.execute(
            """INSERT INTO beliefs
               (id, belief_type, subject, summary, data, epistemic_level,
                source_reliability, info_credibility, confidence,
                source_claims, inference_chain, evidence_log,
                status, supersedes, superseded_by, priority,
                created_at, updated_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 summary = excluded.summary,
                 data = excluded.data,
                 confidence = excluded.confidence,
                 source_claims = excluded.source_claims,
                 inference_chain = excluded.inference_chain,
                 evidence_log = excluded.evidence_log,
                 status = excluded.status,
                 superseded_by = excluded.superseded_by,
                 priority = excluded.priority,
                 updated_at = ?""",
            (
                belief.id,
                belief.belief_type.value,
                belief.subject,
                belief.summary,
                json.dumps(belief.data),
                belief.epistemic_level.value,
                belief.source_reliability,
                belief.info_credibility,
                belief.confidence,
                json.dumps(belief.source_claims),
                json.dumps(belief.inference_chain) if belief.inference_chain is not None else None,
                json.dumps(belief.evidence_log) if belief.evidence_log is not None else None,
                belief.status.value,
                belief.supersedes,
                belief.superseded_by,
                belief.priority,
                belief.created_at,
                belief.updated_at,
                belief.expires_at,
                now,
            ),
        )
        self.conn.commit()
        return True

    def get_belief(self, belief_id: str) -> Belief | None:
        """Fetch a single belief by ID."""
        row = self.conn.execute(
            "SELECT * FROM beliefs WHERE id = ?", (belief_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_belief(row)

    def get_beliefs(
        self,
        subject: str | None = None,
        belief_type: str | None = None,
        status: str = "active",
        min_confidence: float = 0.0,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> list[Belief]:
        """Fetch beliefs, optionally filtered."""
        conditions: list[str] = []
        params: list[Any] = []
        if subject:
            conditions.append("subject = ?")
            params.append(subject)
        if belief_type:
            conditions.append("belief_type = ?")
            params.append(belief_type)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if min_confidence > 0:
            conditions.append("confidence >= ?")
            params.append(min_confidence)
        params.append(limit)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self.conn.execute(
            f"""SELECT * FROM beliefs {where}
                ORDER BY confidence DESC LIMIT ?""",
            params,
        ).fetchall()
        return [self._row_to_belief(r) for r in rows]

    def get_beliefs_by_claims(self, claim_ids: list[str]) -> list[Belief]:
        """Find beliefs whose source_claims contain any of the given claim IDs."""
        if not claim_ids:
            return []
        results: list[Belief] = []
        seen: set[str] = set()
        for cid in claim_ids:
            rows = self.conn.execute(
                """SELECT * FROM beliefs
                   WHERE source_claims LIKE ?""",
                (f'%"{cid}"%',),
            ).fetchall()
            for row in rows:
                if row["id"] not in seen:
                    seen.add(row["id"])
                    results.append(self._row_to_belief(row))
        return results

    def update_belief_status(self, belief_id: str, status: str):
        """Update the status of a belief."""
        now = int(time.time())
        self.conn.execute(
            "UPDATE beliefs SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, belief_id),
        )
        self.conn.commit()

    def supersede_belief(self, old_id: str, new_id: str):
        """Mark a belief as superseded by a newer version."""
        now = int(time.time())
        self.conn.execute(
            "UPDATE beliefs SET superseded_by = ?, status = 'superseded', updated_at = ? WHERE id = ?",
            (new_id, now, old_id),
        )
        self.conn.execute(
            "UPDATE beliefs SET supersedes = ?, updated_at = ? WHERE id = ?",
            (old_id, now, new_id),
        )
        self.conn.commit()

    def _row_to_belief(self, row: sqlite3.Row) -> Belief:
        inference_chain_raw = row["inference_chain"]
        evidence_log_raw = row["evidence_log"]

        return Belief(
            id=row["id"],
            belief_type=BeliefType(row["belief_type"]),
            subject=row["subject"],
            summary=row["summary"],
            data=json.loads(row["data"]),
            epistemic_level=EpistemicLevel(row["epistemic_level"]),
            source_reliability=row["source_reliability"] or "F",
            info_credibility=row["info_credibility"] or 6,
            confidence=row["confidence"],
            source_claims=json.loads(row["source_claims"]),
            inference_chain=json.loads(inference_chain_raw) if inference_chain_raw else None,
            evidence_log=json.loads(evidence_log_raw) if evidence_log_raw else None,
            status=BeliefStatus(row["status"]),
            supersedes=row["supersedes"],
            superseded_by=row["superseded_by"],
            priority=row["priority"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
        )

    # ── Annotations (faceted observations) ──────────────────────

    def put_annotation(self, ann: Annotation) -> bool:
        """Insert an annotation. Returns True if new, False if duplicate."""
        cursor = self.conn.execute(
            """INSERT OR IGNORE INTO annotations
               (event_id, facet, value, confidence, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ann.event_id, ann.facet, ann.value, ann.confidence,
             ann.source, ann.created_at),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def put_annotations_batch(self, annotations: list[Annotation]) -> int:
        """Insert a batch of annotations. Returns count inserted."""
        inserted = 0
        with self.conn:
            for ann in annotations:
                cursor = self.conn.execute(
                    """INSERT OR IGNORE INTO annotations
                       (event_id, facet, value, confidence, source, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (ann.event_id, ann.facet, ann.value, ann.confidence,
                     ann.source, ann.created_at),
                )
                if cursor.rowcount > 0:
                    inserted += 1
        return inserted

    def get_annotations(
        self,
        event_id: str | None = None,
        facet: str | None = None,
        value: str | None = None,
    ) -> list[Annotation]:
        """Fetch annotations with flexible filters."""
        conditions: list[str] = []
        params: list[Any] = []
        if event_id:
            conditions.append("event_id = ?")
            params.append(event_id)
        if facet:
            conditions.append("facet = ?")
            params.append(facet)
        if value:
            conditions.append("value = ?")
            params.append(value)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self.conn.execute(
            f"SELECT * FROM annotations {where}",  # noqa: S608
            params,
        ).fetchall()
        return [
            Annotation(
                event_id=r["event_id"], facet=r["facet"], value=r["value"],
                confidence=r["confidence"], source=r["source"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ── Topic synonyms ────────────────────────────────────────

    def get_topic_canonical(self, raw_topic: str) -> str | None:
        """Look up the canonical form of a topic. Returns None if no mapping."""
        row = self.conn.execute(
            "SELECT canonical FROM topic_synonyms WHERE raw_topic = ?",
            (raw_topic,),
        ).fetchone()
        return row["canonical"] if row else None

    def put_topic_synonym(self, raw_topic: str, canonical: str, method: str = "deterministic"):
        """Insert or update a topic synonym mapping."""
        now = int(time.time())
        self.conn.execute(
            """INSERT INTO topic_synonyms (raw_topic, canonical, method, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(raw_topic) DO UPDATE SET
                 canonical = excluded.canonical,
                 method = excluded.method""",
            (raw_topic, canonical, method, now),
        )
        self.conn.commit()

    def put_topic_synonyms_batch(self, mappings: list[tuple[str, str, str]]) -> int:
        """Batch insert topic synonyms. Each tuple: (raw, canonical, method).

        Returns count inserted/updated.
        """
        now = int(time.time())
        count = 0
        with self.conn:
            for raw, canonical, method in mappings:
                self.conn.execute(
                    """INSERT INTO topic_synonyms (raw_topic, canonical, method, created_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(raw_topic) DO UPDATE SET
                         canonical = excluded.canonical,
                         method = excluded.method""",
                    (raw, canonical, method, now),
                )
                count += 1
        return count

    def get_all_topic_synonyms(self) -> dict[str, str]:
        """Return {raw_topic: canonical} for all synonym mappings."""
        rows = self.conn.execute(
            "SELECT raw_topic, canonical FROM topic_synonyms"
        ).fetchall()
        return {r["raw_topic"]: r["canonical"] for r in rows}

    # ── Projections (lens-scoped scores) ──────────────────────

    def put_projection(self, proj: Projection) -> bool:
        """Insert or update a projection. Returns True on success."""
        self.conn.execute(
            """INSERT INTO projections
               (event_id, lens, score, route, components, computed_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(event_id, lens) DO UPDATE SET
                 score = excluded.score,
                 route = excluded.route,
                 components = excluded.components,
                 computed_at = excluded.computed_at""",
            (proj.event_id, proj.lens, proj.score, proj.route,
             json.dumps(proj.components), proj.computed_at),
        )
        self.conn.commit()
        return True

    def put_projections_batch(self, projections: list[Projection]) -> int:
        """Insert/update a batch of projections. Returns count written."""
        with self.conn:
            for proj in projections:
                self.conn.execute(
                    """INSERT INTO projections
                       (event_id, lens, score, route, components, computed_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(event_id, lens) DO UPDATE SET
                         score = excluded.score,
                         route = excluded.route,
                         components = excluded.components,
                         computed_at = excluded.computed_at""",
                    (proj.event_id, proj.lens, proj.score, proj.route,
                     json.dumps(proj.components), proj.computed_at),
                )
        return len(projections)

    def get_projection(self, event_id: str, lens: str) -> Projection | None:
        """Fetch a single projection by event + lens."""
        row = self.conn.execute(
            "SELECT * FROM projections WHERE event_id = ? AND lens = ?",
            (event_id, lens),
        ).fetchone()
        if not row:
            return None
        return self._row_to_projection(row)

    def get_projections(
        self,
        lens: str,
        route: str | None = None,
        min_score: float = 0.0,
        limit: int = DEFAULT_EVENT_QUERY_LIMIT,
    ) -> list[Projection]:
        """Fetch projections for a lens, optionally filtered by route/score."""
        conditions = ["lens = ?"]
        params: list[Any] = [lens]
        if route:
            conditions.append("route = ?")
            params.append(route)
        if min_score > 0:
            conditions.append("score >= ?")
            params.append(min_score)
        params.append(limit)
        rows = self.conn.execute(
            f"""SELECT * FROM projections
                WHERE {' AND '.join(conditions)}
                ORDER BY score DESC LIMIT ?""",
            params,
        ).fetchall()
        return [self._row_to_projection(r) for r in rows]

    def delete_projections(self, lens: str) -> int:
        """Delete all projections for a lens. Returns count deleted."""
        cursor = self.conn.execute(
            "DELETE FROM projections WHERE lens = ?", (lens,),
        )
        self.conn.commit()
        return cursor.rowcount

    def _row_to_projection(self, row: sqlite3.Row) -> Projection:
        return Projection(
            event_id=row["event_id"],
            lens=row["lens"],
            score=row["score"],
            route=row["route"],
            components=json.loads(row["components"] or "{}"),
            computed_at=row["computed_at"],
        )

    # ── Person registry ──────────────────────────────────────────

    def put_person(
        self,
        person_id: str,
        canonical_name: str = "",
        is_user: bool = False,
        sources: list[str] | None = None,
    ):
        """Create or update a person record."""
        now = int(time.time())
        self.conn.execute(
            """INSERT INTO persons (person_id, canonical_name, is_user, sources, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(person_id) DO UPDATE SET
                 canonical_name = COALESCE(NULLIF(excluded.canonical_name, ''), persons.canonical_name),
                 sources = excluded.sources,
                 updated_at = ?""",
            (person_id, canonical_name, int(is_user),
             json.dumps(sources or []), now, now, now),
        )
        self.conn.commit()

    def add_person_identifier(
        self,
        person_id: str,
        id_type: str,
        identifier: str,
        display_name: str = "",
        source: str = "",
    ):
        """Add an identifier to a person. Idempotent."""
        self.conn.execute(
            """INSERT OR REPLACE INTO person_identifiers
               (person_id, identifier_type, identifier, display_name, source)
               VALUES (?, ?, ?, ?, ?)""",
            (person_id, id_type, identifier, display_name, source),
        )
        self.conn.commit()

    def resolve_person(self, id_type: str, identifier: str) -> str | None:
        """Look up a person_id by identifier. Returns None if unknown."""
        row = self.conn.execute(
            """SELECT person_id FROM person_identifiers
               WHERE identifier_type = ? AND identifier = ?""",
            (id_type, identifier),
        ).fetchone()
        return row["person_id"] if row else None

    def get_person(self, person_id: str) -> dict | None:
        """Fetch a person record as a dict."""
        row = self.conn.execute(
            "SELECT * FROM persons WHERE person_id = ?", (person_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_persons(self) -> list[dict]:
        """Fetch all person records."""
        rows = self.conn.execute(
            "SELECT * FROM persons ORDER BY canonical_name"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_person_identifiers(self, person_id: str) -> list[dict]:
        """Fetch all identifiers for a person."""
        rows = self.conn.execute(
            "SELECT * FROM person_identifiers WHERE person_id = ?",
            (person_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Person profiles ────────────────────────────────────────────

    def upsert_person_profile(self, profile: dict) -> bool:
        """Insert or update a person profile.

        Before replacing, reads the current row to capture previous_tier
        and previous_message_count for delta detection.
        """
        person_id = profile["person_id"]
        now = int(time.time())

        # Read existing row for delta tracking
        existing = self.conn.execute(
            "SELECT tier, message_count FROM person_profiles WHERE person_id = ?",
            (person_id,),
        ).fetchone()

        previous_tier = existing["tier"] if existing else None
        previous_message_count = existing["message_count"] if existing else None

        self.conn.execute(
            """INSERT INTO person_profiles
               (person_id, canonical_name, message_count, direct_count, group_count,
                tier, user_initiated_ratio, channels, channel_count,
                days_since_last, first_contact_ts, last_contact_ts,
                relationship_span_days, is_user,
                previous_tier, previous_message_count, computed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(person_id) DO UPDATE SET
                 canonical_name = excluded.canonical_name,
                 message_count = excluded.message_count,
                 direct_count = excluded.direct_count,
                 group_count = excluded.group_count,
                 tier = excluded.tier,
                 user_initiated_ratio = excluded.user_initiated_ratio,
                 channels = excluded.channels,
                 channel_count = excluded.channel_count,
                 days_since_last = excluded.days_since_last,
                 first_contact_ts = excluded.first_contact_ts,
                 last_contact_ts = excluded.last_contact_ts,
                 relationship_span_days = excluded.relationship_span_days,
                 is_user = excluded.is_user,
                 previous_tier = excluded.previous_tier,
                 previous_message_count = excluded.previous_message_count,
                 computed_at = excluded.computed_at""",
            (
                person_id,
                profile.get("canonical_name", ""),
                profile.get("message_count", 0),
                profile.get("direct_count", 0),
                profile.get("group_count", 0),
                profile.get("tier", 4),
                profile.get("user_initiated_ratio"),
                json.dumps(profile.get("channels", [])),
                profile.get("channel_count", 0),
                profile.get("days_since_last"),
                profile.get("first_contact_ts"),
                profile.get("last_contact_ts"),
                profile.get("relationship_span_days"),
                int(profile.get("is_user", False)),
                previous_tier,
                previous_message_count,
                now,
            ),
        )
        self.conn.commit()
        tier_changed = previous_tier is not None and previous_tier != profile.get("tier", 4)
        return tier_changed

    def upsert_person_profiles_batch(self, profiles: list[dict]) -> int:
        """Batch upsert person profiles. Returns count written."""
        now = int(time.time())

        # Pre-fetch existing tiers/counts for delta tracking
        existing: dict[str, tuple[int, int]] = {}
        rows = self.conn.execute(
            "SELECT person_id, tier, message_count FROM person_profiles"
        ).fetchall()
        for r in rows:
            existing[r["person_id"]] = (r["tier"], r["message_count"])

        with self.conn:
            for profile in profiles:
                person_id = profile["person_id"]
                prev = existing.get(person_id)
                previous_tier = prev[0] if prev else None
                previous_message_count = prev[1] if prev else None

                self.conn.execute(
                    """INSERT INTO person_profiles
                       (person_id, canonical_name, message_count, direct_count, group_count,
                        tier, user_initiated_ratio, channels, channel_count,
                        days_since_last, first_contact_ts, last_contact_ts,
                        relationship_span_days, is_user,
                        previous_tier, previous_message_count, computed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(person_id) DO UPDATE SET
                         canonical_name = excluded.canonical_name,
                         message_count = excluded.message_count,
                         direct_count = excluded.direct_count,
                         group_count = excluded.group_count,
                         tier = excluded.tier,
                         user_initiated_ratio = excluded.user_initiated_ratio,
                         channels = excluded.channels,
                         channel_count = excluded.channel_count,
                         days_since_last = excluded.days_since_last,
                         first_contact_ts = excluded.first_contact_ts,
                         last_contact_ts = excluded.last_contact_ts,
                         relationship_span_days = excluded.relationship_span_days,
                         is_user = excluded.is_user,
                         previous_tier = excluded.previous_tier,
                         previous_message_count = excluded.previous_message_count,
                         computed_at = excluded.computed_at""",
                    (
                        person_id,
                        profile.get("canonical_name", ""),
                        profile.get("message_count", 0),
                        profile.get("direct_count", 0),
                        profile.get("group_count", 0),
                        profile.get("tier", 4),
                        profile.get("user_initiated_ratio"),
                        json.dumps(profile.get("channels", [])),
                        profile.get("channel_count", 0),
                        profile.get("days_since_last"),
                        profile.get("first_contact_ts"),
                        profile.get("last_contact_ts"),
                        profile.get("relationship_span_days"),
                        int(profile.get("is_user", False)),
                        previous_tier,
                        previous_message_count,
                        now,
                    ),
                )
        return len(profiles)

    def get_person_profile(self, person_id: str) -> dict | None:
        """Fetch a single person profile."""
        row = self.conn.execute(
            "SELECT * FROM person_profiles WHERE person_id = ?",
            (person_id,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["channels"] = json.loads(d["channels"] or "[]")
        return d

    def get_person_profiles(self, min_messages: int = 0) -> list[dict]:
        """Fetch all person profiles, optionally filtered by min message count."""
        if min_messages > 0:
            rows = self.conn.execute(
                "SELECT * FROM person_profiles WHERE message_count >= ? ORDER BY message_count DESC",
                (min_messages,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM person_profiles ORDER BY message_count DESC"
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["channels"] = json.loads(d["channels"] or "[]")
            result.append(d)
        return result

    # ── Event-Person links ───────────────────────────────────────

    def link_event_person(self, event_id: str, person_id: str, role: str):
        """Link an event to a person with a role. Idempotent.

        Also populates person_events (materialized reverse index) so
        person-first lookups work without a full linker rebuild.
        """
        self.conn.execute(
            "INSERT OR IGNORE INTO event_persons (event_id, person_id, role) VALUES (?, ?, ?)",
            (event_id, person_id, role),
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO person_events (person_id, event_id) VALUES (?, ?)",
            (person_id, event_id),
        )
        self.conn.commit()

    def get_persons_for_event(self, event_id: str) -> list[tuple[str, str]]:
        """Get (person_id, role) pairs for an event."""
        rows = self.conn.execute(
            "SELECT person_id, role FROM event_persons WHERE event_id = ?",
            (event_id,),
        ).fetchall()
        return [(r["person_id"], r["role"]) for r in rows]

    def get_events_for_person(
        self,
        person_id: str,
        role: str | None = None,
        since: int = 0,
        limit: int = DEFAULT_EVENT_QUERY_LIMIT,
    ) -> list[Event]:
        """Get events involving a person, optionally filtered by role.

        Uses the materialized person_events table for thread-aware lookups.
        Falls back to event_persons if person_events isn't populated yet,
        or when filtering by a specific role.
        """
        if role:
            # Role-specific queries use event_persons directly
            rows = self.conn.execute(
                """SELECT e.* FROM events e
                   JOIN event_persons ep ON e.id = ep.event_id
                   WHERE ep.person_id = ? AND ep.role = ? AND e.timestamp >= ?
                   ORDER BY e.timestamp DESC LIMIT ?""",
                (person_id, role, since, limit),
            ).fetchall()
        else:
            # Use person_events for full thread-expanded participation
            rows = self.conn.execute(
                """SELECT e.* FROM events e
                   JOIN person_events pe ON e.id = pe.event_id
                   WHERE pe.person_id = ? AND e.timestamp >= ?
                   ORDER BY e.timestamp DESC LIMIT ?""",
                (person_id, since, limit),
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    # ── Sync state ───────────────────────────────────────────────

    def get_sync_state(self, source: str) -> dict | None:
        """Get sync state for a source."""
        row = self.conn.execute(
            "SELECT * FROM sync_state WHERE source = ?", (source,)
        ).fetchone()
        return dict(row) if row else None

    def update_sync_state(
        self,
        source: str,
        last_event_ts: int = 0,
        event_count: int = 0,
        status: str = "idle",
        error_message: str = "",
        cursor: dict | None = None,
    ):
        """Update or create sync state for a source."""
        now = int(time.time())
        self.conn.execute(
            """INSERT INTO sync_state
               (source, last_sync, last_event_ts, event_count, status,
                error_message, cursor, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source) DO UPDATE SET
                 last_sync = ?,
                 last_event_ts = MAX(sync_state.last_event_ts, excluded.last_event_ts),
                 event_count = sync_state.event_count + excluded.event_count,
                 status = excluded.status,
                 error_message = excluded.error_message,
                 cursor = excluded.cursor,
                 updated_at = ?""",
            (source, now, last_event_ts, event_count, status,
             error_message, json.dumps(cursor or {}), now,
             now, now),
        )
        self.conn.commit()

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Get counts across all layers."""
        result: dict[str, Any] = {}
        for table in ("events", "claims", "beliefs", "persons", "annotations", "projections"):
            row = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608
            result[f"{table}_count"] = row[0]

        rows = self.conn.execute(
            "SELECT source, COUNT(*) as cnt FROM events GROUP BY source"
        ).fetchall()
        result["events_by_source"] = {r["source"]: r["cnt"] for r in rows}

        rows = self.conn.execute(
            "SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type"
        ).fetchall()
        result["events_by_type"] = {r["event_type"]: r["cnt"] for r in rows}

        row = self.conn.execute(
            "SELECT COUNT(*) FROM claims WHERE superseded_by IS NULL"
        ).fetchone()
        result["active_claims"] = row[0]

        rows = self.conn.execute("SELECT * FROM sync_state").fetchall()
        result["sync_state"] = {r["source"]: dict(r) for r in rows}

        return result

    # ── Clarity Queue: Tasks ──────────────────────────────────────

    def put_cq_task(
        self,
        task_id: str,
        bucket: str,
        title: str,
        note: str = "",
        source: str = "manual",
        due_date: str | None = None,
        labels: list[str] | None = None,
        position: int = 0,
        claim_id: str | None = None,
        defer_until: str | None = None,
        recurrence: dict | None = None,
        custom_fields: dict | None = None,
        category: str | None = None,
    ) -> bool:
        """Insert a CQ task. Returns True on success."""
        now = int(time.time())
        self.conn.execute(
            """INSERT INTO cq_tasks
               (id, bucket, title, note, done, source, due_date, labels,
                position, claim_id, defer_until, recurrence, custom_fields,
                category, created_at, updated_at)
               VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 bucket = excluded.bucket,
                 title = excluded.title,
                 note = excluded.note,
                 source = excluded.source,
                 due_date = excluded.due_date,
                 labels = excluded.labels,
                 position = excluded.position,
                 claim_id = excluded.claim_id,
                 defer_until = excluded.defer_until,
                 recurrence = excluded.recurrence,
                 custom_fields = excluded.custom_fields,
                 category = excluded.category,
                 updated_at = excluded.updated_at""",
            (task_id, bucket, title, note, source, due_date,
             json.dumps(labels) if labels else None,
             position, claim_id, defer_until,
             json.dumps(recurrence) if recurrence else None,
             json.dumps(custom_fields) if custom_fields else None,
             category, now, now),
        )
        self.conn.commit()
        return True

    def get_cq_tasks(
        self,
        bucket: str | None = None,
        include_done: bool = False,
    ) -> list[dict]:
        """Fetch CQ tasks, optionally filtered by bucket."""
        conditions: list[str] = []
        params: list[Any] = []
        if bucket:
            conditions.append("bucket = ?")
            params.append(bucket)
        if not include_done:
            conditions.append("done = 0")
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self.conn.execute(
            f"SELECT * FROM cq_tasks {where} ORDER BY bucket, position",  # noqa: S608
            params,
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("labels"):
                d["labels"] = json.loads(d["labels"])
            if d.get("recurrence"):
                d["recurrence"] = json.loads(d["recurrence"])
            if d.get("custom_fields"):
                d["custom_fields"] = json.loads(d["custom_fields"])
            result.append(d)
        return result

    def update_cq_task(self, task_id: str, **fields) -> bool:
        """Update specific fields on a CQ task. Returns True if row existed."""
        if not fields:
            return False
        allowed = {"bucket", "title", "note", "done", "source", "due_date",
                    "labels", "position", "claim_id", "defer_until",
                    "recurrence", "custom_fields", "category",
                    "story_id", "seen_at", "priority_override", "accepted"}
        sets: list[str] = []
        params: list[Any] = []
        for key, val in fields.items():
            if key not in allowed:
                continue
            if key == "labels" and isinstance(val, list):
                val = json.dumps(val)
            if key == "recurrence" and isinstance(val, dict):
                val = json.dumps(val)
            if key == "custom_fields" and isinstance(val, dict):
                val = json.dumps(val)
            sets.append(f"{key} = ?")
            params.append(val)
        if not sets:
            return False
        sets.append("updated_at = ?")
        params.append(int(time.time()))
        params.append(task_id)
        cursor = self.conn.execute(
            f"UPDATE cq_tasks SET {', '.join(sets)} WHERE id = ?",  # noqa: S608
            params,
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_cq_task(self, task_id: str) -> bool:
        """Delete a CQ task. Returns True if row existed."""
        cursor = self.conn.execute(
            "DELETE FROM cq_tasks WHERE id = ?", (task_id,),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def move_cq_task(self, task_id: str, target_bucket: str, position: int = 0) -> bool:
        """Move a CQ task to a different bucket at a given position."""
        now = int(time.time())
        cursor = self.conn.execute(
            "UPDATE cq_tasks SET bucket = ?, position = ?, updated_at = ? WHERE id = ?",
            (target_bucket, position, now, task_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def reorder_cq_tasks(self, bucket: str, task_ids: list[str]) -> bool:
        """Set position ordering for tasks in a bucket."""
        now = int(time.time())
        with self.conn:
            for idx, tid in enumerate(task_ids):
                self.conn.execute(
                    "UPDATE cq_tasks SET position = ?, updated_at = ? WHERE id = ? AND bucket = ?",
                    (idx, now, tid, bucket),
                )
        return True

    # ── CQ: Undo log ─────────────────────────────────────────────

    def snapshot_cq_task(self, task_id: str) -> dict | None:
        """Read a single cq_tasks row as a dict (for undo snapshots)."""
        row = self.conn.execute(
            "SELECT * FROM cq_tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        # Keep JSON columns as raw strings for faithful restore
        return d

    def push_cq_undo(self, action: str, task_id: str, prev_state: dict):
        """Push an entry onto the undo log."""
        from alteris.constants import CQ_UNDO_MAX_ENTRIES
        now = int(time.time())
        self.conn.execute(
            """INSERT INTO cq_undo_log (action, task_id, prev_state, created_at)
               VALUES (?, ?, ?, ?)""",
            (action, task_id, json.dumps(prev_state), now),
        )
        # Prune if over max entries
        self.conn.execute(
            """DELETE FROM cq_undo_log WHERE id NOT IN (
                 SELECT id FROM cq_undo_log ORDER BY created_at DESC LIMIT ?
               )""",
            (CQ_UNDO_MAX_ENTRIES,),
        )
        self.conn.commit()

    def pop_cq_undo(self) -> dict | None:
        """Pop the most recent undo entry (read + delete). Returns None if empty."""
        row = self.conn.execute(
            "SELECT * FROM cq_undo_log ORDER BY created_at DESC, id DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        entry = dict(row)
        entry["prev_state"] = json.loads(entry["prev_state"])
        self.conn.execute("DELETE FROM cq_undo_log WHERE id = ?", (entry["id"],))
        self.conn.commit()
        return entry

    def prune_cq_undo(self, max_age_seconds: int | None = None):
        """Delete undo entries older than max_age."""
        from alteris.constants import CQ_UNDO_MAX_AGE_SECONDS
        age = max_age_seconds if max_age_seconds is not None else CQ_UNDO_MAX_AGE_SECONDS
        cutoff = int(time.time()) - age
        self.conn.execute(
            "DELETE FROM cq_undo_log WHERE created_at < ?", (cutoff,),
        )
        self.conn.commit()

    # ── CQ: Sender rules ──────────────────────────────────────────

    def put_sender_rule(
        self, pattern: str, priority: str, source: str = "", note: str = "",
    ):
        """Upsert a sender rule. priority: 'P1'|'P2'|'P3'|'block'."""
        now = int(time.time())
        self.conn.execute(
            """INSERT INTO sender_rules (pattern, priority, source, note, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(pattern, source) DO UPDATE SET
                 priority = excluded.priority,
                 note = excluded.note""",
            (pattern, priority, source, note, now),
        )
        self.conn.commit()

    def get_sender_rules(self) -> list[dict]:
        """Return all sender rules."""
        rows = self.conn.execute(
            "SELECT * FROM sender_rules ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_sender_rule(self, rule_id: int):
        """Delete a sender rule by ID."""
        self.conn.execute("DELETE FROM sender_rules WHERE id = ?", (rule_id,))
        self.conn.commit()

    def check_sender_rules(self, identifier: str) -> str | None:
        """Check if an identifier matches any sender rule.

        Checks exact match first, then domain match (e.g. '@example.com').
        Returns 'P1', 'P2', 'P3', 'block', or None.
        """
        # Exact match
        row = self.conn.execute(
            "SELECT priority FROM sender_rules WHERE pattern = ?", (identifier,),
        ).fetchone()
        if row:
            return row["priority"]

        # Domain match: extract domain from email
        if "@" in identifier:
            domain = "@" + identifier.split("@", 1)[1]
            row = self.conn.execute(
                "SELECT priority FROM sender_rules WHERE pattern = ?", (domain,),
            ).fetchone()
            if row:
                return row["priority"]

        return None

    # ── CQ: Categories ─────────────────────────────────────────────

    def put_cq_category(self, name: str, color: str = "", icon: str = ""):
        """Upsert a user-defined category."""
        now = int(time.time())
        # Get next position
        row = self.conn.execute(
            "SELECT MAX(position) FROM cq_categories"
        ).fetchone()
        next_pos = (row[0] or 0) + 1

        self.conn.execute(
            """INSERT INTO cq_categories (name, color, icon, position, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                 color = excluded.color,
                 icon = excluded.icon""",
            (name, color, icon, next_pos, now),
        )
        self.conn.commit()

    def get_cq_categories(self) -> list[dict]:
        """Return all categories ordered by position."""
        rows = self.conn.execute(
            "SELECT * FROM cq_categories ORDER BY position"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_cq_category(self, name: str):
        """Delete a category by name."""
        self.conn.execute("DELETE FROM cq_categories WHERE name = ?", (name,))
        self.conn.commit()

    # ── CQ: Extractable fields ─────────────────────────────────────

    def put_cq_extractable_field(
        self, name: str, description: str = "", example: str = "",
    ):
        """Upsert a user-defined extractable field definition."""
        now = int(time.time())
        # Get next position
        row = self.conn.execute(
            "SELECT MAX(position) FROM cq_extractable_fields"
        ).fetchone()
        next_pos = (row[0] or 0) + 1

        self.conn.execute(
            """INSERT INTO cq_extractable_fields (name, description, example, position, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(name) DO UPDATE SET
                 description = excluded.description,
                 example = excluded.example""",
            (name, description, example, next_pos, now),
        )
        self.conn.commit()

    def get_cq_extractable_fields(self) -> list[dict]:
        """Return all extractable field definitions ordered by position."""
        rows = self.conn.execute(
            "SELECT * FROM cq_extractable_fields ORDER BY position"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_cq_extractable_field(self, name: str):
        """Delete an extractable field definition by name."""
        self.conn.execute(
            "DELETE FROM cq_extractable_fields WHERE name = ?", (name,)
        )
        self.conn.commit()

    # ── CQ: Stories ──────────────────────────────────────────────

    def put_cq_story(
        self,
        story_id: str,
        title: str,
        source: str = "auto",
        color: str = "",
        icon: str = "",
        status: str = "active",
        priority: int | None = None,
        priority_override: int | None = None,
        cluster_hash: str | None = None,
    ) -> bool:
        """Insert or update a CQ story."""
        now = int(time.time())
        self.conn.execute(
            """INSERT INTO cq_stories
               (id, title, source, color, icon, status, priority,
                priority_override, cluster_hash, updated_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 title = excluded.title,
                 color = excluded.color,
                 icon = excluded.icon,
                 status = excluded.status,
                 priority = excluded.priority,
                 priority_override = excluded.priority_override,
                 cluster_hash = excluded.cluster_hash,
                 updated_at = excluded.updated_at""",
            (story_id, title, source, color, icon, status,
             priority, priority_override, cluster_hash, now, now),
        )
        self.conn.commit()
        return True

    def get_cq_stories(self, status: str = "active") -> list[dict]:
        """Fetch stories filtered by status, ordered by priority then updated_at."""
        rows = self.conn.execute(
            """SELECT * FROM cq_stories
               WHERE status = ?
               ORDER BY COALESCE(priority_override, priority, 99),
                        updated_at DESC""",
            (status,),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_cq_story(self, story_id: str, **fields) -> bool:
        """Update specific fields on a story."""
        if not fields:
            return False
        allowed = {"title", "color", "icon", "status", "priority",
                   "priority_override", "source", "cluster_hash"}
        sets: list[str] = []
        params: list = []
        for key, val in fields.items():
            if key not in allowed:
                continue
            sets.append(f"{key} = ?")
            params.append(val)
        if not sets:
            return False
        sets.append("updated_at = ?")
        params.append(int(time.time()))
        params.append(story_id)
        cursor = self.conn.execute(
            f"UPDATE cq_stories SET {', '.join(sets)} WHERE id = ?",  # noqa: S608
            params,
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_cq_story(self, story_id: str) -> bool:
        """Delete a story and its member/person links."""
        self.conn.execute("DELETE FROM cq_story_members WHERE story_id = ?", (story_id,))
        self.conn.execute("DELETE FROM cq_story_persons WHERE story_id = ?", (story_id,))
        cursor = self.conn.execute("DELETE FROM cq_stories WHERE id = ?", (story_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def add_story_member(self, story_id: str, task_id: str, position: int = 0):
        """Add a task to a story."""
        now = int(time.time())
        self.conn.execute(
            """INSERT OR REPLACE INTO cq_story_members
               (story_id, task_id, position, added_at)
               VALUES (?, ?, ?, ?)""",
            (story_id, task_id, position, now),
        )
        # Update task's story_id column
        self.conn.execute(
            "UPDATE cq_tasks SET story_id = ? WHERE id = ?",
            (story_id, task_id),
        )
        # Touch story updated_at
        self.conn.execute(
            "UPDATE cq_stories SET updated_at = ? WHERE id = ?",
            (now, story_id),
        )
        self.conn.commit()

    def remove_story_member(self, story_id: str, task_id: str):
        """Remove a task from a story."""
        self.conn.execute(
            "DELETE FROM cq_story_members WHERE story_id = ? AND task_id = ?",
            (story_id, task_id),
        )
        self.conn.execute(
            "UPDATE cq_tasks SET story_id = NULL WHERE id = ? AND story_id = ?",
            (task_id, story_id),
        )
        self.conn.commit()

    def get_story_members(self, story_id: str) -> list[dict]:
        """Get tasks in a story, ordered by position."""
        rows = self.conn.execute(
            """SELECT task_id, position, added_at
               FROM cq_story_members
               WHERE story_id = ?
               ORDER BY position""",
            (story_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def add_story_person(self, story_id: str, person_id: str, role: str = ""):
        """Associate a person with a story."""
        now = int(time.time())
        self.conn.execute(
            """INSERT OR REPLACE INTO cq_story_persons
               (story_id, person_id, role, added_at)
               VALUES (?, ?, ?, ?)""",
            (story_id, person_id, role, now),
        )
        self.conn.commit()

    def remove_story_person(self, story_id: str, person_id: str):
        """Remove a person from a story."""
        self.conn.execute(
            "DELETE FROM cq_story_persons WHERE story_id = ? AND person_id = ?",
            (story_id, person_id),
        )
        self.conn.commit()

    def get_story_persons(self, story_id: str) -> list[dict]:
        """Get persons associated with a story."""
        rows = self.conn.execute(
            """SELECT sp.person_id, sp.role, sp.added_at,
                      p.canonical_name
               FROM cq_story_persons sp
               LEFT JOIN persons p ON sp.person_id = p.person_id
               WHERE sp.story_id = ?""",
            (story_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def add_story_anti_link(self, task_id_a: str, task_id_b: str):
        """Record that two tasks should NOT be in the same story."""
        now = int(time.time())
        a, b = sorted([task_id_a, task_id_b])
        self.conn.execute(
            "INSERT OR IGNORE INTO cq_story_anti_links (task_id_a, task_id_b, created_at) VALUES (?, ?, ?)",
            (a, b, now),
        )
        self.conn.commit()

    def check_story_anti_link(self, task_id_a: str, task_id_b: str) -> bool:
        """Check if two tasks have an anti-link (should not be grouped)."""
        a, b = sorted([task_id_a, task_id_b])
        row = self.conn.execute(
            "SELECT 1 FROM cq_story_anti_links WHERE task_id_a = ? AND task_id_b = ?",
            (a, b),
        ).fetchone()
        return row is not None

    def mark_task_seen(self, task_id: str):
        """Set seen_at timestamp on a task (clears 'new' badge)."""
        now = int(time.time())
        self.conn.execute(
            "UPDATE cq_tasks SET seen_at = ? WHERE id = ? AND seen_at IS NULL",
            (now, task_id),
        )
        self.conn.commit()

    # ── Thread retrieval ───────────────────────────────────────────

    def get_events_by_thread(self, thread_id: str, source: str | None = None) -> list[Event]:
        """Query events by thread_id in metadata JSON."""
        if source:
            rows = self.conn.execute(
                """SELECT * FROM events
                   WHERE json_extract(metadata, '$.thread_id') = ? AND source = ?
                   ORDER BY timestamp""",
                (thread_id, source),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """SELECT * FROM events
                   WHERE json_extract(metadata, '$.thread_id') = ?
                   ORDER BY timestamp""",
                (thread_id,),
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    # ── Clarity Queue: Sessions ───────────────────────────────────

    def put_cq_session(
        self,
        session_id: str,
        messages: list[dict] | str,
        title: str = "",
        session_type: str = "clarity",
    ) -> bool:
        """Insert or update a CQ chat session."""
        now = int(time.time())
        msgs = json.dumps(messages) if isinstance(messages, list) else messages
        self.conn.execute(
            """INSERT INTO cq_sessions (id, title, session_type, messages, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 title = excluded.title,
                 session_type = excluded.session_type,
                 messages = excluded.messages,
                 updated_at = excluded.updated_at""",
            (session_id, title, session_type, msgs, now, now),
        )
        self.conn.commit()
        return True

    def get_cq_session(self, session_id: str) -> dict | None:
        """Fetch a single CQ session by ID."""
        row = self.conn.execute(
            "SELECT * FROM cq_sessions WHERE id = ?", (session_id,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["messages"] = json.loads(d["messages"])
        return d

    def get_cq_sessions(self, limit: int = 20) -> list[dict]:
        """Fetch recent CQ sessions."""
        rows = self.conn.execute(
            "SELECT * FROM cq_sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["messages"] = json.loads(d["messages"])
            result.append(d)
        return result

    # ── API Spend ─────────────────────────────────────────────────

    def insert_spend(
        self,
        date: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        source: str = "",
    ) -> int:
        """Insert a spend record. Returns the row ID."""
        now = int(time.time())
        cursor = self.conn.execute(
            """INSERT INTO api_spend
               (date, provider, model, input_tokens, output_tokens, cost_usd, source, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (date, provider, model, input_tokens, output_tokens, cost_usd, source, now),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_daily_spend(self, date: str, provider: str | None = None) -> dict:
        """Get spend summary for a single date.

        Returns dict with total_usd, by_provider, by_source breakdowns.
        """
        if provider:
            rows = self.conn.execute(
                "SELECT * FROM api_spend WHERE date = ? AND provider = ?",
                (date, provider),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM api_spend WHERE date = ?", (date,),
            ).fetchall()
        if not rows:
            return {"date": date, "total_usd": 0.0, "by_provider": {}, "by_source": {}}
        total = 0.0
        by_provider: dict[str, float] = {}
        by_source: dict[str, float] = {}
        for r in rows:
            cost = r["cost_usd"]
            total += cost
            prov = r["provider"]
            by_provider[prov] = by_provider.get(prov, 0.0) + cost
            src = r["source"] or ""
            by_source[src] = by_source.get(src, 0.0) + cost
        return {
            "date": date,
            "total_usd": round(total, 8),
            "by_provider": {k: round(v, 8) for k, v in by_provider.items()},
            "by_source": {k: round(v, 8) for k, v in by_source.items()},
        }

    def get_spend_range(self, start_date: str, end_date: str) -> list[dict]:
        """Get daily spend summaries for a date range (inclusive)."""
        rows = self.conn.execute(
            """SELECT date, SUM(cost_usd) as total, provider, source
               FROM api_spend
               WHERE date >= ? AND date <= ?
               GROUP BY date, provider, source
               ORDER BY date DESC""",
            (start_date, end_date),
        ).fetchall()
        # Aggregate into per-day summaries
        days: dict[str, dict] = {}
        for r in rows:
            d = r["date"]
            if d not in days:
                days[d] = {"date": d, "total_usd": 0.0, "by_provider": {}, "by_source": {}}
            cost = r["total"]
            days[d]["total_usd"] = round(days[d]["total_usd"] + cost, 8)
            prov = r["provider"]
            days[d]["by_provider"][prov] = round(
                days[d]["by_provider"].get(prov, 0.0) + cost, 8
            )
            src = r["source"] or ""
            days[d]["by_source"][src] = round(
                days[d]["by_source"].get(src, 0.0) + cost, 8
            )
        return sorted(days.values(), key=lambda x: x["date"], reverse=True)

    def delete_spend_for_date(self, date: str) -> int:
        """Delete all spend records for a date. Returns rows deleted."""
        cursor = self.conn.execute(
            "DELETE FROM api_spend WHERE date = ?", (date,),
        )
        self.conn.commit()
        return cursor.rowcount
