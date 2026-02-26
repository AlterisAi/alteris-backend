"""Stratified sampling from production tables for eval review.

Pulls items from the database with configurable stratification:
  - by_source: proportional to source distribution
  - by_tier: based on confidence tiers (ignore/lightweight/deep)
  - by_domain: based on triage domain field
  - random: uniform random sample

Each sample returns enough context for a reviewer to judge correctness.
"""

from __future__ import annotations

import json
import random
import sqlite3
from typing import Any

from loom.eval.golden import (
    STAGE_CLAIMS,
    STAGE_EXTRACT,
    STAGE_INGEST,
    STAGE_LINK,
    STAGE_PROPAGATE,
    STAGE_RESOLVE,
    STAGE_SYNTHESIZE,
    STAGE_TRIAGE,
)


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _dict_row(row: sqlite3.Row) -> dict[str, Any]:
    return dict(row)


def sample_events(
    db_path: str,
    n: int = 50,
    stratify_by: str = "source",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Sample events for Stage 0 review.

    stratify_by: 'source' | 'random'
    """
    if seed is not None:
        random.seed(seed)

    conn = _connect(db_path)
    try:
        if stratify_by == "source":
            return _stratified_sample_events_by_source(conn, n)
        return _random_sample(conn, "events", n)
    finally:
        conn.close()


def _stratified_sample_events_by_source(conn: sqlite3.Connection, n: int) -> list[dict[str, Any]]:
    """Sample proportional to source distribution."""
    counts = conn.execute(
        "SELECT source, COUNT(*) as cnt FROM events GROUP BY source"
    ).fetchall()
    total = sum(r["cnt"] for r in counts)
    if total == 0:
        return []

    samples: list[dict[str, Any]] = []
    for r in counts:
        source = r["source"]
        # Proportional allocation, at least 1 per source
        k = max(1, round(n * r["cnt"] / total))
        rows = conn.execute(
            "SELECT * FROM events WHERE source = ? ORDER BY RANDOM() LIMIT ?",
            (source, k),
        ).fetchall()
        for row in rows:
            samples.append(_dict_row(row))

    random.shuffle(samples)
    return samples[:n]


def sample_triage(
    db_path: str,
    n: int = 50,
    stratify_by: str = "tier",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Sample triage claims for Stage 4 review.

    stratify_by: 'tier' | 'source' | 'random'
    Returns claims enriched with their source event context.
    """
    if seed is not None:
        random.seed(seed)

    conn = _connect(db_path)
    try:
        if stratify_by == "tier":
            return _stratified_sample_triage_by_tier(conn, n)
        if stratify_by == "source":
            return _stratified_sample_triage_by_source(conn, n)
        return _random_sample_claims(conn, "triage", n)
    finally:
        conn.close()


def _stratified_sample_triage_by_tier(conn: sqlite3.Connection, n: int) -> list[dict[str, Any]]:
    """Sample triage claims across confidence tiers."""
    tiers = [
        ("ignore", "confidence < 0.3"),
        ("lightweight", "confidence >= 0.3 AND confidence < 0.7"),
        ("deep", "confidence >= 0.7"),
    ]
    per_tier = max(1, n // len(tiers))
    samples: list[dict[str, Any]] = []

    for tier_name, condition in tiers:
        rows = conn.execute(
            f"""SELECT c.*, ce.event_id
                FROM claims c
                LEFT JOIN claim_events ce ON c.id = ce.claim_id
                WHERE c.claim_type = 'triage'
                  AND c.superseded_by IS NULL
                  AND {condition}
                ORDER BY RANDOM()
                LIMIT ?""",
            (per_tier,),
        ).fetchall()

        for row in rows:
            item = _dict_row(row)
            item["_tier"] = tier_name
            # Enrich with event context
            event_id = item.pop("event_id", None)
            if event_id:
                event_row = conn.execute(
                    "SELECT * FROM events WHERE id = ?", (event_id,)
                ).fetchone()
                if event_row:
                    item["_event"] = _dict_row(event_row)
            samples.append(item)

    random.shuffle(samples)
    return samples[:n]


def _stratified_sample_triage_by_source(conn: sqlite3.Connection, n: int) -> list[dict[str, Any]]:
    """Sample triage claims proportional to event source distribution."""
    rows = conn.execute(
        """SELECT e.source, COUNT(*) as cnt
           FROM claims c
           JOIN claim_events ce ON c.id = ce.claim_id
           JOIN events e ON ce.event_id = e.id
           WHERE c.claim_type = 'triage' AND c.superseded_by IS NULL
           GROUP BY e.source"""
    ).fetchall()
    total = sum(r["cnt"] for r in rows)
    if total == 0:
        return []

    samples: list[dict[str, Any]] = []
    for r in rows:
        source = r["source"]
        k = max(1, round(n * r["cnt"] / total))
        claim_rows = conn.execute(
            """SELECT c.*, ce.event_id
               FROM claims c
               JOIN claim_events ce ON c.id = ce.claim_id
               JOIN events e ON ce.event_id = e.id
               WHERE c.claim_type = 'triage'
                 AND c.superseded_by IS NULL
                 AND e.source = ?
               ORDER BY RANDOM()
               LIMIT ?""",
            (source, k),
        ).fetchall()
        for crow in claim_rows:
            item = _dict_row(crow)
            event_id = item.pop("event_id", None)
            if event_id:
                ev = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
                if ev:
                    item["_event"] = _dict_row(ev)
            samples.append(item)

    random.shuffle(samples)
    return samples[:n]


def sample_persons(
    db_path: str,
    n: int = 50,
    stratify_by: str = "tier",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Sample persons for Stage 1 (resolve) review.

    Each person is enriched with their identifiers and cross-source merge info.
    stratify_by: 'tier' | 'random'
    """
    if seed is not None:
        random.seed(seed)

    conn = _connect(db_path)
    try:
        if stratify_by == "tier":
            return _stratified_sample_persons_by_tier(conn, n)

        rows = conn.execute(
            "SELECT * FROM persons ORDER BY RANDOM() LIMIT ?", (n,),
        ).fetchall()
        return [_enrich_person(conn, _dict_row(r)) for r in rows]
    finally:
        conn.close()


def _stratified_sample_persons_by_tier(conn: sqlite3.Connection, n: int) -> list[dict[str, Any]]:
    """Sample persons across tiers: user, multi-source, single-source, contacts-only."""
    samples: list[dict[str, Any]] = []

    # Always include the user
    user_rows = conn.execute(
        "SELECT * FROM persons WHERE is_user = 1"
    ).fetchall()
    for r in user_rows:
        item = _enrich_person(conn, _dict_row(r))
        item["_tier"] = "user"
        samples.append(item)

    # Multi-source persons (cross-source merges)
    multi_rows = conn.execute(
        """SELECT p.* FROM persons p
           WHERE p.is_user = 0 AND json_array_length(p.sources) > 1
           ORDER BY RANDOM() LIMIT ?""",
        (max(1, n // 3),),
    ).fetchall()
    for r in multi_rows:
        item = _enrich_person(conn, _dict_row(r))
        item["_tier"] = "multi_source"
        samples.append(item)

    # Single communication source (not contacts-only)
    single_rows = conn.execute(
        """SELECT p.* FROM persons p
           WHERE p.is_user = 0
             AND json_array_length(p.sources) = 1
             AND p.sources != '["contacts"]'
           ORDER BY RANDOM() LIMIT ?""",
        (max(1, n // 3),),
    ).fetchall()
    for r in single_rows:
        item = _enrich_person(conn, _dict_row(r))
        item["_tier"] = "single_source"
        samples.append(item)

    # Contacts-only
    contacts_rows = conn.execute(
        """SELECT p.* FROM persons p
           WHERE p.is_user = 0 AND p.sources = '["contacts"]'
           ORDER BY RANDOM() LIMIT ?""",
        (max(1, n // 3),),
    ).fetchall()
    for r in contacts_rows:
        item = _enrich_person(conn, _dict_row(r))
        item["_tier"] = "contacts_only"
        samples.append(item)

    random.shuffle(samples)
    return samples[:n]


def _enrich_person(conn: sqlite3.Connection, person: dict[str, Any]) -> dict[str, Any]:
    """Add identifiers and event counts to a person record."""
    pid = person["person_id"]

    # Fetch identifiers
    id_rows = conn.execute(
        "SELECT * FROM person_identifiers WHERE person_id = ?", (pid,)
    ).fetchall()
    person["_identifiers"] = [_dict_row(r) for r in id_rows]

    # Count linked events
    ev_count = conn.execute(
        "SELECT COUNT(*) FROM event_persons WHERE person_id = ?", (pid,)
    ).fetchone()[0]
    person["_event_count"] = ev_count

    # Count distinct sources from linked events
    src_rows = conn.execute(
        """SELECT DISTINCT e.source FROM events e
           JOIN event_persons ep ON e.id = ep.event_id
           WHERE ep.person_id = ?""",
        (pid,),
    ).fetchall()
    person["_linked_sources"] = [r["source"] for r in src_rows]

    return person


def sample_links(
    db_path: str,
    n: int = 50,
    stratify_by: str = "role",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Sample event-person links for Stage 2 (link) review.

    Each link is enriched with event and person context.
    stratify_by: 'role' | 'source' | 'random'
    """
    if seed is not None:
        random.seed(seed)

    conn = _connect(db_path)
    try:
        if stratify_by == "role":
            return _stratified_sample_links_by_role(conn, n)
        if stratify_by == "source":
            return _stratified_sample_links_by_source(conn, n)

        rows = conn.execute(
            """SELECT ep.*, e.source, e.event_type, e.timestamp, e.raw_content,
                      p.canonical_name, p.is_user
               FROM event_persons ep
               JOIN events e ON ep.event_id = e.id
               JOIN persons p ON ep.person_id = p.person_id
               ORDER BY RANDOM() LIMIT ?""",
            (n,),
        ).fetchall()
        return [_dict_row(r) for r in rows]
    finally:
        conn.close()


def _stratified_sample_links_by_role(conn: sqlite3.Connection, n: int) -> list[dict[str, Any]]:
    """Sample links proportional to role distribution."""
    roles = conn.execute(
        "SELECT role, COUNT(*) as cnt FROM event_persons GROUP BY role ORDER BY cnt DESC"
    ).fetchall()
    total = sum(r["cnt"] for r in roles)
    if total == 0:
        return []

    samples: list[dict[str, Any]] = []
    for r in roles:
        role = r["role"]
        k = max(1, round(n * r["cnt"] / total))
        rows = conn.execute(
            """SELECT ep.*, e.source, e.event_type, e.timestamp,
                      SUBSTR(e.raw_content, 1, 200) as content_preview,
                      e.participants, e.metadata,
                      p.canonical_name, p.is_user
               FROM event_persons ep
               JOIN events e ON ep.event_id = e.id
               JOIN persons p ON ep.person_id = p.person_id
               WHERE ep.role = ?
               ORDER BY RANDOM() LIMIT ?""",
            (role, k),
        ).fetchall()
        for row in rows:
            item = _dict_row(row)
            item["_role_label"] = role
            samples.append(item)

    random.shuffle(samples)
    return samples[:n]


def _stratified_sample_links_by_source(conn: sqlite3.Connection, n: int) -> list[dict[str, Any]]:
    """Sample links proportional to event source distribution."""
    sources = conn.execute(
        """SELECT e.source, COUNT(*) as cnt
           FROM event_persons ep
           JOIN events e ON ep.event_id = e.id
           GROUP BY e.source"""
    ).fetchall()
    total = sum(r["cnt"] for r in sources)
    if total == 0:
        return []

    samples: list[dict[str, Any]] = []
    for r in sources:
        source = r["source"]
        k = max(1, round(n * r["cnt"] / total))
        rows = conn.execute(
            """SELECT ep.*, e.source, e.event_type, e.timestamp,
                      SUBSTR(e.raw_content, 1, 200) as content_preview,
                      e.participants, e.metadata,
                      p.canonical_name, p.is_user
               FROM event_persons ep
               JOIN events e ON ep.event_id = e.id
               JOIN persons p ON ep.person_id = p.person_id
               WHERE e.source = ?
               ORDER BY RANDOM() LIMIT ?""",
            (source, k),
        ).fetchall()
        samples.extend(_dict_row(row) for row in rows)

    random.shuffle(samples)
    return samples[:n]


def sample_deterministic_claims(
    db_path: str,
    n: int = 50,
    stratify_by: str = "type",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Sample deterministic (Stage 1) claims for Stage 3 review.

    stratify_by: 'type' | 'random'
    """
    if seed is not None:
        random.seed(seed)

    conn = _connect(db_path)
    try:
        # Stage 1 claim types: communication_frequency, communication_channel,
        # directionality, recency, timing_pattern, thread_activity
        stage1_types = conn.execute(
            """SELECT claim_type, COUNT(*) as cnt FROM claims
               WHERE claim_type NOT IN ('triage', 'commitment', 'extraction_run')
                 AND superseded_by IS NULL
               GROUP BY claim_type ORDER BY cnt DESC"""
        ).fetchall()

        if not stage1_types:
            return []

        if stratify_by == "type":
            total = sum(r["cnt"] for r in stage1_types)
            samples: list[dict[str, Any]] = []
            for r in stage1_types:
                ct = r["claim_type"]
                k = max(1, round(n * r["cnt"] / total))
                rows = conn.execute(
                    """SELECT c.* FROM claims c
                       WHERE c.claim_type = ? AND c.superseded_by IS NULL
                       ORDER BY RANDOM() LIMIT ?""",
                    (ct, k),
                ).fetchall()
                for row in rows:
                    item = _dict_row(row)
                    item["_claim_category"] = ct
                    # Enrich with linked events
                    ev_rows = conn.execute(
                        """SELECT e.source, e.event_type, e.timestamp,
                                  SUBSTR(e.raw_content, 1, 150) as content_preview
                           FROM events e
                           JOIN claim_events ce ON e.id = ce.event_id
                           WHERE ce.claim_id = ?
                           LIMIT 3""",
                        (item["id"],),
                    ).fetchall()
                    item["_source_events"] = [_dict_row(er) for er in ev_rows]
                    # Enrich with person name if subject is a person_id
                    subj = item.get("subject", "")
                    if subj:
                        prow = conn.execute(
                            "SELECT canonical_name FROM persons WHERE person_id = ?",
                            (subj,),
                        ).fetchone()
                        if prow:
                            item["_person_name"] = prow["canonical_name"]
                    samples.append(item)

            random.shuffle(samples)
            return samples[:n]

        rows = conn.execute(
            """SELECT * FROM claims
               WHERE claim_type NOT IN ('triage', 'commitment', 'extraction_run')
                 AND superseded_by IS NULL
               ORDER BY RANDOM() LIMIT ?""",
            (n,),
        ).fetchall()
        return [_dict_row(r) for r in rows]
    finally:
        conn.close()


def sample_propagated(
    db_path: str,
    n: int = 50,
    stratify_by: str = "tier",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Sample propagated triage claims for Stage 5 review.

    Shows triage claims with context about what propagation rules applied.
    stratify_by: 'tier' | 'random'
    """
    if seed is not None:
        random.seed(seed)

    conn = _connect(db_path)
    try:
        if stratify_by == "tier":
            return _stratified_sample_propagated_by_tier(conn, n)

        rows = conn.execute(
            """SELECT c.*, ce.event_id FROM claims c
               LEFT JOIN claim_events ce ON c.id = ce.claim_id
               WHERE c.claim_type = 'triage' AND c.superseded_by IS NULL
               ORDER BY RANDOM() LIMIT ?""",
            (n,),
        ).fetchall()
        return [_enrich_propagated(conn, _dict_row(r)) for r in rows]
    finally:
        conn.close()


def _stratified_sample_propagated_by_tier(conn: sqlite3.Connection, n: int) -> list[dict[str, Any]]:
    """Sample propagated claims across tiers, enriched with propagation context."""
    tiers = [
        ("ignore", "c.confidence < 0.3"),
        ("lightweight", "c.confidence >= 0.3 AND c.confidence < 0.7"),
        ("deep", "c.confidence >= 0.7"),
    ]
    per_tier = max(1, n // len(tiers))
    samples: list[dict[str, Any]] = []

    for tier_name, condition in tiers:
        rows = conn.execute(
            f"""SELECT c.*, ce.event_id FROM claims c
                LEFT JOIN claim_events ce ON c.id = ce.claim_id
                WHERE c.claim_type = 'triage'
                  AND c.superseded_by IS NULL
                  AND {condition}
                ORDER BY RANDOM() LIMIT ?""",
            (per_tier,),
        ).fetchall()
        for row in rows:
            item = _enrich_propagated(conn, _dict_row(row))
            item["_tier"] = tier_name
            samples.append(item)

    random.shuffle(samples)
    return samples[:n]


def _enrich_propagated(conn: sqlite3.Connection, item: dict[str, Any]) -> dict[str, Any]:
    """Add propagation context: sender person, event source, person tier."""
    event_id = item.pop("event_id", None)
    if event_id:
        ev = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if ev:
            item["_event"] = _dict_row(ev)
            # Find sender person
            sender_rows = conn.execute(
                """SELECT ep.person_id, p.canonical_name, p.is_user
                   FROM event_persons ep
                   JOIN persons p ON ep.person_id = p.person_id
                   WHERE ep.event_id = ? AND ep.role = 'sender'""",
                (event_id,),
            ).fetchall()
            if sender_rows:
                item["_sender"] = _dict_row(sender_rows[0])
            # Count how many events this sender has (proxy for tier)
            if sender_rows:
                sender_count = conn.execute(
                    "SELECT COUNT(*) FROM event_persons WHERE person_id = ?",
                    (sender_rows[0]["person_id"],),
                ).fetchone()[0]
                item["_sender_event_count"] = sender_count
    return item


def sample_commitments(
    db_path: str,
    n: int = 50,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Sample commitment claims for Stage 6 review."""
    if seed is not None:
        random.seed(seed)

    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """SELECT c.*, ce.event_id
               FROM claims c
               LEFT JOIN claim_events ce ON c.id = ce.claim_id
               WHERE c.claim_type = 'commitment'
                 AND c.superseded_by IS NULL
               ORDER BY RANDOM()
               LIMIT ?""",
            (n,),
        ).fetchall()

        samples: list[dict[str, Any]] = []
        for row in rows:
            item = _dict_row(row)
            event_id = item.pop("event_id", None)
            if event_id:
                ev = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
                if ev:
                    item["_event"] = _dict_row(ev)
            samples.append(item)
        return samples
    finally:
        conn.close()


def sample_beliefs(
    db_path: str,
    n: int = 50,
    stratify_by: str = "type",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Sample beliefs for Stage 7 review."""
    if seed is not None:
        random.seed(seed)

    conn = _connect(db_path)
    try:
        if stratify_by == "type":
            types = conn.execute(
                "SELECT belief_type, COUNT(*) as cnt FROM beliefs WHERE status = 'active' GROUP BY belief_type"
            ).fetchall()
            total = sum(r["cnt"] for r in types)
            if total == 0:
                return []

            samples: list[dict[str, Any]] = []
            for r in types:
                k = max(1, round(n * r["cnt"] / total))
                rows = conn.execute(
                    "SELECT * FROM beliefs WHERE status = 'active' AND belief_type = ? ORDER BY RANDOM() LIMIT ?",
                    (r["belief_type"], k),
                ).fetchall()
                samples.extend(_dict_row(row) for row in rows)
            random.shuffle(samples)
            return samples[:n]

        rows = conn.execute(
            "SELECT * FROM beliefs WHERE status = 'active' ORDER BY RANDOM() LIMIT ?",
            (n,),
        ).fetchall()
        return [_dict_row(r) for r in rows]
    finally:
        conn.close()


def _random_sample(conn: sqlite3.Connection, table: str, n: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT ?",  # noqa: S608
        (n,),
    ).fetchall()
    return [_dict_row(r) for r in rows]


def _random_sample_claims(conn: sqlite3.Connection, claim_type: str, n: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """SELECT * FROM claims
           WHERE claim_type = ? AND superseded_by IS NULL
           ORDER BY RANDOM() LIMIT ?""",
        (claim_type, n),
    ).fetchall()
    return [_dict_row(r) for r in rows]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unified sample dispatcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sample(
    db_path: str,
    stage: str,
    n: int = 50,
    stratify_by: str = "default",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Unified sampling entry point.

    stage: one of the STAGE_* constants (e.g. '0_ingest', '4_triage')
    """
    dispatch = {
        STAGE_INGEST: lambda: sample_events(db_path, n, "source" if stratify_by == "default" else stratify_by, seed),
        STAGE_RESOLVE: lambda: sample_persons(db_path, n, "tier" if stratify_by == "default" else stratify_by, seed),
        STAGE_LINK: lambda: sample_links(db_path, n, "role" if stratify_by == "default" else stratify_by, seed),
        STAGE_CLAIMS: lambda: sample_deterministic_claims(db_path, n, "type" if stratify_by == "default" else stratify_by, seed),
        STAGE_TRIAGE: lambda: sample_triage(db_path, n, "tier" if stratify_by == "default" else stratify_by, seed),
        STAGE_PROPAGATE: lambda: sample_propagated(db_path, n, "tier" if stratify_by == "default" else stratify_by, seed),
        STAGE_EXTRACT: lambda: sample_commitments(db_path, n, seed),
        STAGE_SYNTHESIZE: lambda: sample_beliefs(db_path, n, "type" if stratify_by == "default" else stratify_by, seed),
    }

    sampler = dispatch.get(stage)
    if sampler is None:
        raise ValueError(f"Sampling not implemented for stage: {stage}. Available: {list(dispatch.keys())}")
    return sampler()
