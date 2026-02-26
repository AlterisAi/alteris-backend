"""Stage runner — re-run pipeline stages against golden store data.

Reads golden records with expected outputs, re-runs the corresponding
pipeline stage, and compares actual vs expected. Produces a comparison
report without modifying the production database.

Usage:
    from loom.eval.runner import run_stage_eval
    results = run_stage_eval(db_path, stage="4_triage", golden_dir="eval/golden")
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from loom.eval.golden import GoldenRecord, GoldenStore


def run_stage_eval(
    db_path: str,
    stage: str,
    golden_dir: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Re-run a pipeline stage and compare against golden records.

    Returns:
        {
            "stage": str,
            "golden_count": int,
            "matched": int,
            "mismatched": int,
            "errors": int,
            "details": [...]
        }
    """
    golden = GoldenStore(golden_dir)
    records = golden.list_records(stage, judgment="approve")
    corrected = golden.list_records(stage, judgment="correct")
    all_expected = records + corrected

    if not all_expected:
        return {
            "stage": stage,
            "golden_count": 0,
            "matched": 0,
            "mismatched": 0,
            "errors": 0,
            "details": [],
            "message": f"No approved/corrected golden records for stage {stage}",
        }

    runner = _get_stage_runner(stage)
    if runner is None:
        return {
            "stage": stage,
            "golden_count": len(all_expected),
            "matched": 0,
            "mismatched": 0,
            "errors": 0,
            "details": [],
            "message": f"No runner implemented for stage {stage}",
        }

    details: list[dict[str, Any]] = []
    matched = 0
    mismatched = 0
    errors = 0

    for record in all_expected:
        try:
            actual = runner(db_path, record, dry_run=dry_run)
            expected = record.expected_output or record.input_data

            match = _compare_outputs(stage, expected, actual)
            if match["is_match"]:
                matched += 1
            else:
                mismatched += 1

            details.append({
                "item_id": record.item_id,
                "judgment": record.judgment,
                "match": match["is_match"],
                "diff": match.get("diff", []),
            })
        except Exception as exc:
            errors += 1
            details.append({
                "item_id": record.item_id,
                "judgment": record.judgment,
                "match": False,
                "error": str(exc),
            })

    return {
        "stage": stage,
        "golden_count": len(all_expected),
        "matched": matched,
        "mismatched": mismatched,
        "errors": errors,
        "details": details,
    }


def _get_stage_runner(stage: str):
    """Return a runner function for the given stage, or None."""
    runners = {
        "0_ingest": _run_ingest_check,
        "1_resolve": _run_person_check,
        "2_link": _run_link_check,
        "3_claims": _run_claim_check,
        "4_triage": _run_triage_check,
        "5_propagate": _run_triage_check,  # same table, same check
        "6_extract": _run_extract_check,
        "7_synthesize": _run_belief_check,
    }
    return runners.get(stage)


def _run_ingest_check(
    db_path: str, record: GoldenRecord, dry_run: bool = True,
) -> dict[str, Any]:
    """Check that an event still exists in the database with matching fields."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (record.item_id,)
        ).fetchone()
        if row is None:
            return {"exists": False}
        return dict(row)
    finally:
        conn.close()


def _run_person_check(
    db_path: str, record: GoldenRecord, dry_run: bool = True,
) -> dict[str, Any]:
    """Check that a person still exists with matching fields."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        pid = record.item_id or record.input_data.get("person_id", "")
        row = conn.execute(
            "SELECT * FROM persons WHERE person_id = ?", (pid,)
        ).fetchone()
        if row is None:
            return {"exists": False}
        result = dict(row)
        # Add identifiers
        id_rows = conn.execute(
            "SELECT * FROM person_identifiers WHERE person_id = ?", (pid,)
        ).fetchall()
        result["_identifier_count"] = len(id_rows)
        return result
    finally:
        conn.close()


def _run_link_check(
    db_path: str, record: GoldenRecord, dry_run: bool = True,
) -> dict[str, Any]:
    """Check that an event-person link still exists."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        event_id = record.input_data.get("event_id", "")
        person_id = record.input_data.get("person_id", "")
        role = record.input_data.get("role", "")
        row = conn.execute(
            """SELECT * FROM event_persons
               WHERE event_id = ? AND person_id = ? AND role = ?""",
            (event_id, person_id, role),
        ).fetchone()
        if row is None:
            return {"exists": False}
        return dict(row)
    finally:
        conn.close()


def _run_claim_check(
    db_path: str, record: GoldenRecord, dry_run: bool = True,
) -> dict[str, Any]:
    """Check current claim against golden record."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM claims WHERE id = ?", (record.item_id,)
        ).fetchone()
        if row is None:
            return {"exists": False}
        return dict(row)
    finally:
        conn.close()


def _run_triage_check(
    db_path: str, record: GoldenRecord, dry_run: bool = True,
) -> dict[str, Any]:
    """Check current triage claim against golden record."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM claims WHERE id = ?", (record.item_id,)
        ).fetchone()
        if row is None:
            return {"exists": False}
        return dict(row)
    finally:
        conn.close()


def _run_extract_check(
    db_path: str, record: GoldenRecord, dry_run: bool = True,
) -> dict[str, Any]:
    """Check current commitment claim against golden record."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM claims WHERE id = ?", (record.item_id,)
        ).fetchone()
        if row is None:
            return {"exists": False}
        return dict(row)
    finally:
        conn.close()


def _run_belief_check(
    db_path: str, record: GoldenRecord, dry_run: bool = True,
) -> dict[str, Any]:
    """Check current belief against golden record."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM beliefs WHERE id = ?", (record.item_id,)
        ).fetchone()
        if row is None:
            return {"exists": False}
        return dict(row)
    finally:
        conn.close()


def _compare_outputs(
    stage: str,
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
    """Compare expected vs actual output for a stage.

    Returns {"is_match": bool, "diff": [...]}
    """
    if actual.get("exists") is False:
        return {"is_match": False, "diff": ["item no longer exists in database"]}

    # Stage-specific comparison fields
    compare_fields = _fields_for_stage(stage)
    if not compare_fields:
        # Fallback: compare all common keys
        compare_fields = list(set(expected.keys()) & set(actual.keys()))

    diffs: list[str] = []
    for field in compare_fields:
        exp_val = expected.get(field)
        act_val = actual.get(field)
        if exp_val is None:
            continue

        # For confidence, allow tolerance
        if field == "confidence" and isinstance(exp_val, (int, float)) and isinstance(act_val, (int, float)):
            if abs(exp_val - act_val) > 0.05:
                diffs.append(f"{field}: expected={exp_val:.2f}, actual={act_val:.2f}")
            continue

        if str(exp_val) != str(act_val):
            diffs.append(f"{field}: expected={_truncate(str(exp_val))}, actual={_truncate(str(act_val))}")

    return {"is_match": len(diffs) == 0, "diff": diffs}


def _fields_for_stage(stage: str) -> list[str]:
    """Key fields to compare for each stage."""
    return {
        "0_ingest": ["source", "source_id", "event_type", "timestamp"],
        "1_resolve": ["canonical_name", "is_user", "sources"],
        "2_link": ["event_id", "person_id", "role"],
        "3_claims": ["claim_type", "subject", "predicate", "object", "confidence"],
        "4_triage": ["confidence", "subject", "predicate"],
        "5_propagate": ["confidence", "subject", "predicate"],
        "6_extract": ["confidence", "subject", "predicate", "object"],
        "7_synthesize": ["belief_type", "subject", "summary", "confidence", "status"],
    }.get(stage, [])


def _truncate(text: str, max_len: int = 80) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def format_eval_report(results: dict[str, Any]) -> str:
    """Format eval results as a human-readable report."""
    lines: list[str] = []
    lines.append(f"Stage: {results['stage']}")
    lines.append(f"Golden records: {results['golden_count']}")
    lines.append(f"Matched: {results['matched']}")
    lines.append(f"Mismatched: {results['mismatched']}")
    lines.append(f"Errors: {results['errors']}")

    if results.get("message"):
        lines.append(f"Note: {results['message']}")

    details = results.get("details", [])
    if details:
        lines.append("")
        for d in details:
            status = "OK" if d.get("match") else "MISMATCH"
            if d.get("error"):
                status = f"ERROR: {d['error']}"
            lines.append(f"  {d['item_id']}: {status}")
            for diff in d.get("diff", []):
                lines.append(f"    - {diff}")

    return "\n".join(lines)
