"""Shared workspace — the communication layer between CEO and CTO agents.

This is the "board room" that both agents can read/write to. Each agent's
private knowledge graph stays on their own machine. Only structured,
business-relevant data crosses this boundary.

The workspace is a SQLite database that can be synced between machines
via git, iCloud, or a simple relay server.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE_PATH = Path.home() / ".loom" / "workspace.db"

_SCHEMA = """
-- Investor pipeline: tracks every VC from research to commitment/pass
CREATE TABLE IF NOT EXISTS investors (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    firm            TEXT NOT NULL,
    title           TEXT DEFAULT '',
    email           TEXT DEFAULT '',
    linkedin        TEXT DEFAULT '',
    stage           TEXT DEFAULT 'researched',
    tier            TEXT DEFAULT 'tier3',
    focus           TEXT DEFAULT 'generalist',
    owner           TEXT DEFAULT '',
    introduced_by   TEXT DEFAULT '',
    introducer_agent TEXT DEFAULT '',
    warm_path       TEXT DEFAULT '[]',
    thesis          TEXT DEFAULT '',
    portfolio_fit   TEXT DEFAULT '',
    last_contact_at INTEGER,
    next_step       TEXT DEFAULT '',
    notes           TEXT DEFAULT '[]',
    pass_reason     TEXT DEFAULT '',
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_investors_stage ON investors(stage);
CREATE INDEX IF NOT EXISTS idx_investors_owner ON investors(owner);
CREATE INDEX IF NOT EXISTS idx_investors_focus ON investors(focus);

-- Cross-agent messages: structured, auditable communication
CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    from_agent      TEXT NOT NULL,
    to_agent        TEXT,
    msg_type        TEXT NOT NULL,
    subject         TEXT DEFAULT '',
    content         TEXT NOT NULL,
    status          TEXT DEFAULT 'pending',
    created_at      INTEGER NOT NULL,
    read_at         INTEGER,
    answered_at     INTEGER,
    answer          TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_to ON messages(to_agent, status);
CREATE INDEX IF NOT EXISTS idx_messages_type ON messages(msg_type);

-- Meeting prep: coordinated preparation for investor meetings
CREATE TABLE IF NOT EXISTS meeting_prep (
    id              TEXT PRIMARY KEY,
    investor_id     TEXT REFERENCES investors(id),
    meeting_date    TEXT,
    prepared_by     TEXT DEFAULT '',
    pitch_angle     TEXT DEFAULT '',
    talking_points  TEXT DEFAULT '[]',
    anticipated_qs  TEXT DEFAULT '[]',
    tech_brief      TEXT DEFAULT '',
    business_brief  TEXT DEFAULT '',
    vc_research     TEXT DEFAULT '',
    status          TEXT DEFAULT 'draft',
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_prep_investor ON meeting_prep(investor_id);
CREATE INDEX IF NOT EXISTS idx_prep_date ON meeting_prep(meeting_date);

-- Outreach: every email/message sent or drafted
CREATE TABLE IF NOT EXISTS outreach (
    id              TEXT PRIMARY KEY,
    investor_id     TEXT REFERENCES investors(id),
    channel         TEXT DEFAULT 'email',
    direction       TEXT DEFAULT 'outbound',
    from_agent      TEXT NOT NULL,
    to_address      TEXT DEFAULT '',
    subject         TEXT DEFAULT '',
    body            TEXT DEFAULT '',
    status          TEXT DEFAULT 'draft',
    sent_at         INTEGER,
    reply_at        INTEGER,
    reply_body      TEXT,
    thread_id       TEXT DEFAULT '',
    created_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outreach_investor ON outreach(investor_id);
CREATE INDEX IF NOT EXISTS idx_outreach_status ON outreach(status);

-- Decisions: shared strategic decisions with rationale
CREATE TABLE IF NOT EXISTS decisions (
    id              TEXT PRIMARY KEY,
    topic           TEXT NOT NULL,
    decision        TEXT NOT NULL,
    rationale       TEXT DEFAULT '',
    decided_by      TEXT NOT NULL,
    status          TEXT DEFAULT 'active',
    created_at      INTEGER NOT NULL
);
"""

# Valid stages for investor pipeline
INVESTOR_STAGES = (
    "discovered", "researched", "qualified", "warm_intro_found", "contacted",
    "replied", "meeting_scheduled", "met", "follow_up",
    "due_diligence", "term_sheet", "committed", "pass", "ghosted",
)

# VC focus classification
VC_FOCUS = ("tech_forward", "business_forward", "generalist")

# Agent roles
AGENT_CTO = "cto"
AGENT_CEO = "ceo"


class SharedWorkspace:
    """SQLite-backed shared workspace for cross-agent coordination."""

    def __init__(self, db_path: str | Path = DEFAULT_WORKSPACE_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Investors ─────────────────────────────────────────────

    def add_investor(
        self,
        name: str,
        firm: str,
        *,
        title: str = "",
        email: str = "",
        linkedin: str = "",
        tier: str = "tier3",
        focus: str = "generalist",
        owner: str = "",
        thesis: str = "",
        warm_path: list[str] | None = None,
        introduced_by: str = "",
        introducer_agent: str = "",
    ) -> str:
        """Add a VC to the pipeline. Returns the investor ID."""
        inv_id = str(uuid.uuid4())[:12]
        now = int(time.time())
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO investors
               (id, name, firm, title, email, linkedin, stage, tier, focus,
                owner, introduced_by, introducer_agent, warm_path, thesis,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 'researched', ?, ?, ?,
                       ?, ?, ?, ?, ?, ?)""",
            (inv_id, name, firm, title, email, linkedin, tier, focus,
             owner, introduced_by, introducer_agent,
             json.dumps(warm_path or []), thesis, now, now),
        )
        conn.commit()
        logger.info("Added investor: %s (%s) → %s", name, firm, inv_id)
        return inv_id

    def update_investor(self, inv_id: str, **fields: Any) -> None:
        """Update investor fields. Accepts any column name as kwarg."""
        if not fields:
            return
        fields["updated_at"] = int(time.time())
        # Serialize lists/dicts to JSON
        for k, v in fields.items():
            if isinstance(v, (list, dict)):
                fields[k] = json.dumps(v)
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        conn = self._get_conn()
        conn.execute(
            f"UPDATE investors SET {set_clause} WHERE id = ?",
            (*fields.values(), inv_id),
        )
        conn.commit()

    def get_investor(self, inv_id: str) -> dict | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM investors WHERE id = ?", (inv_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_investors(
        self,
        stage: str | None = None,
        owner: str | None = None,
        focus: str | None = None,
    ) -> list[dict]:
        """List investors with optional filters."""
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if stage:
            clauses.append("stage = ?")
            params.append(stage)
        if owner:
            clauses.append("owner = ?")
            params.append(owner)
        if focus:
            clauses.append("focus = ?")
            params.append(focus)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM investors {where} ORDER BY updated_at DESC",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def pipeline_summary(self) -> dict[str, int]:
        """Return count of investors at each stage."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT stage, COUNT(*) as cnt FROM investors GROUP BY stage"
        ).fetchall()
        return {r["stage"]: r["cnt"] for r in rows}

    # ── Messages ──────────────────────────────────────────────

    def send_message(
        self,
        from_agent: str,
        to_agent: str | None,
        msg_type: str,
        content: dict | str,
        subject: str = "",
    ) -> str:
        """Send a structured message to another agent (or broadcast if to=None)."""
        msg_id = str(uuid.uuid4())[:12]
        now = int(time.time())
        content_str = json.dumps(content) if isinstance(content, dict) else content
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO messages
               (id, from_agent, to_agent, msg_type, subject, content, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
            (msg_id, from_agent, to_agent, msg_type, subject, content_str, now),
        )
        conn.commit()
        return msg_id

    def get_messages(
        self,
        for_agent: str,
        status: str = "pending",
        msg_type: str | None = None,
    ) -> list[dict]:
        """Get messages addressed to this agent."""
        conn = self._get_conn()
        clauses = ["(to_agent = ? OR to_agent IS NULL)"]
        params: list[Any] = [for_agent]
        if status:
            clauses.append("status = ?")
            params.append(status)
        if msg_type:
            clauses.append("msg_type = ?")
            params.append(msg_type)
        where = f"WHERE {' AND '.join(clauses)}"
        rows = conn.execute(
            f"SELECT * FROM messages {where} ORDER BY created_at ASC",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_read(self, msg_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE messages SET status = 'read', read_at = ? WHERE id = ?",
            (int(time.time()), msg_id),
        )
        conn.commit()

    def answer_message(self, msg_id: str, answer: dict | str) -> None:
        answer_str = json.dumps(answer) if isinstance(answer, dict) else answer
        conn = self._get_conn()
        conn.execute(
            "UPDATE messages SET status = 'answered', answered_at = ?, answer = ? WHERE id = ?",
            (int(time.time()), answer_str, msg_id),
        )
        conn.commit()

    # ── Meeting Prep ──────────────────────────────────────────

    def create_meeting_prep(
        self,
        investor_id: str,
        meeting_date: str,
        prepared_by: str,
    ) -> str:
        prep_id = str(uuid.uuid4())[:12]
        now = int(time.time())
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO meeting_prep
               (id, investor_id, meeting_date, prepared_by, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (prep_id, investor_id, meeting_date, prepared_by, now, now),
        )
        conn.commit()
        return prep_id

    def update_meeting_prep(self, prep_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = int(time.time())
        for k, v in fields.items():
            if isinstance(v, (list, dict)):
                fields[k] = json.dumps(v)
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        conn = self._get_conn()
        conn.execute(
            f"UPDATE meeting_prep SET {set_clause} WHERE id = ?",
            (*fields.values(), prep_id),
        )
        conn.commit()

    def get_meeting_prep(self, investor_id: str) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM meeting_prep WHERE investor_id = ? ORDER BY meeting_date DESC",
            (investor_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Outreach ──────────────────────────────────────────────

    def create_outreach(
        self,
        investor_id: str,
        from_agent: str,
        *,
        channel: str = "email",
        to_address: str = "",
        subject: str = "",
        body: str = "",
    ) -> str:
        out_id = str(uuid.uuid4())[:12]
        now = int(time.time())
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO outreach
               (id, investor_id, channel, direction, from_agent,
                to_address, subject, body, status, created_at)
               VALUES (?, ?, ?, 'outbound', ?, ?, ?, ?, 'draft', ?)""",
            (out_id, investor_id, channel, from_agent, to_address, subject, body, now),
        )
        conn.commit()
        return out_id

    def approve_outreach(self, out_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE outreach SET status = 'approved' WHERE id = ?",
            (out_id,),
        )
        conn.commit()

    def mark_sent(self, out_id: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE outreach SET status = 'sent', sent_at = ? WHERE id = ?",
            (int(time.time()), out_id),
        )
        conn.commit()

    def record_reply(self, out_id: str, reply_body: str) -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE outreach SET status = 'replied', reply_at = ?, reply_body = ? WHERE id = ?",
            (int(time.time()), reply_body, out_id),
        )
        conn.commit()

    def get_outreach(
        self,
        investor_id: str | None = None,
        status: str | None = None,
        from_agent: str | None = None,
    ) -> list[dict]:
        conn = self._get_conn()
        clauses: list[str] = []
        params: list[Any] = []
        if investor_id:
            clauses.append("investor_id = ?")
            params.append(investor_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if from_agent:
            clauses.append("from_agent = ?")
            params.append(from_agent)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"SELECT * FROM outreach {where} ORDER BY created_at DESC",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def pending_drafts(self, agent: str) -> list[dict]:
        """Get all drafts awaiting approval for this agent."""
        return self.get_outreach(from_agent=agent, status="draft")

    # ── Decisions ─────────────────────────────────────────────

    def record_decision(
        self,
        topic: str,
        decision: str,
        rationale: str,
        decided_by: str,
    ) -> str:
        dec_id = str(uuid.uuid4())[:12]
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO decisions (id, topic, decision, rationale, decided_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (dec_id, topic, decision, rationale, decided_by, int(time.time())),
        )
        conn.commit()
        return dec_id

    def get_decisions(self, topic: str | None = None) -> list[dict]:
        conn = self._get_conn()
        if topic:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE topic LIKE ? AND status = 'active' ORDER BY created_at DESC",
                (f"%{topic}%",),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE status = 'active' ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]
