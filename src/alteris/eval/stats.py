"""Statistics and comparison reporting for eval golden store.

Computes precision, recall, agreement, and distribution metrics
from reviewed golden records.

Usage:
    from alteris.eval.stats import compute_stats, format_stats
    stats = compute_stats(golden_dir, stage="4_triage")
    print(format_stats(stats))
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from alteris.eval.golden import ALL_STAGES, GoldenStore


def compute_stats(
    golden_dir: str | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    """Compute evaluation statistics from golden store.

    If stage is None, computes across all stages.
    """
    golden = GoldenStore(golden_dir)

    stages = [stage] if stage else ALL_STAGES
    all_records = []
    for s in stages:
        all_records.extend(golden.list_records(s))

    if not all_records:
        return {
            "total": 0,
            "message": "No golden records found",
        }

    # Judgment distribution
    judgments = Counter(r.judgment for r in all_records)

    # Per-stage breakdown
    by_stage: dict[str, dict[str, int]] = {}
    for r in all_records:
        if r.stage not in by_stage:
            by_stage[r.stage] = Counter()
        by_stage[r.stage][r.judgment] += 1

    # Approval rate (approve / (approve + reject))
    approved = judgments.get("approve", 0)
    rejected = judgments.get("reject", 0)
    corrected = judgments.get("correct", 0)
    flagged = judgments.get("flag", 0)
    reviewed = approved + rejected + corrected + flagged
    approval_rate = approved / reviewed if reviewed > 0 else 0.0
    accuracy_rate = (approved + corrected) / reviewed if reviewed > 0 else 0.0

    # Confidence distribution for triage (if we have triage data)
    confidence_buckets = _confidence_distribution(all_records)

    # Common issues (from notes)
    issues = _extract_issues(all_records)

    return {
        "total": len(all_records),
        "reviewed": reviewed,
        "judgments": dict(judgments),
        "approval_rate": round(approval_rate, 3),
        "accuracy_rate": round(accuracy_rate, 3),
        "by_stage": {s: dict(c) for s, c in by_stage.items()},
        "confidence_distribution": confidence_buckets,
        "common_issues": issues,
    }


def _confidence_distribution(records: list) -> dict[str, int]:
    """Bucket confidence scores for items that have them."""
    buckets: dict[str, int] = {
        "0.0-0.3": 0,
        "0.3-0.5": 0,
        "0.5-0.7": 0,
        "0.7-0.9": 0,
        "0.9-1.0": 0,
    }
    for r in records:
        conf = r.input_data.get("confidence")
        if conf is None:
            continue
        try:
            c = float(conf)
        except (ValueError, TypeError):
            continue
        if c < 0.3:
            buckets["0.0-0.3"] += 1
        elif c < 0.5:
            buckets["0.3-0.5"] += 1
        elif c < 0.7:
            buckets["0.5-0.7"] += 1
        elif c < 0.9:
            buckets["0.7-0.9"] += 1
        else:
            buckets["0.9-1.0"] += 1
    return buckets


def _extract_issues(records: list) -> list[dict[str, Any]]:
    """Extract common issues from reviewer notes."""
    note_counts: Counter = Counter()
    for r in records:
        if r.notes and r.judgment in ("reject", "flag", "correct"):
            # Normalize notes for grouping
            normalized = r.notes.strip().lower()
            note_counts[normalized] += 1

    return [
        {"issue": issue, "count": count}
        for issue, count in note_counts.most_common(10)
    ]


def compute_stage_agreement(
    golden_dir: str | None = None,
    stage: str = "4_triage",
) -> dict[str, Any]:
    """Compute inter-rater agreement stats if multiple reviewers exist."""
    golden = GoldenStore(golden_dir)

    # Get raw JSONL (all records including superseded)
    path = golden._path_for_stage(stage)
    if not path.exists():
        return {"message": "No data for stage"}

    import json as _json
    all_rows: list[dict] = []
    for line in path.read_text().splitlines():
        if line.strip():
            all_rows.append(_json.loads(line))

    # Group by item_id
    by_item: dict[str, list[dict]] = {}
    for row in all_rows:
        iid = row.get("item_id", "")
        if iid not in by_item:
            by_item[iid] = []
        by_item[iid].append(row)

    # Find items reviewed by multiple reviewers
    multi_reviewed = {
        iid: rows for iid, rows in by_item.items()
        if len(set(r.get("reviewer", "") for r in rows if r.get("reviewer"))) > 1
    }

    if not multi_reviewed:
        return {
            "stage": stage,
            "items_multi_reviewed": 0,
            "message": "No items reviewed by multiple reviewers",
        }

    agreements = 0
    disagreements = 0
    for iid, rows in multi_reviewed.items():
        judgments = [r["judgment"] for r in rows if r.get("reviewer")]
        unique_judgments = set(judgments)
        if len(unique_judgments) == 1:
            agreements += 1
        else:
            disagreements += 1

    total = agreements + disagreements
    return {
        "stage": stage,
        "items_multi_reviewed": total,
        "agreements": agreements,
        "disagreements": disagreements,
        "agreement_rate": round(agreements / total, 3) if total > 0 else 0.0,
    }


def format_stats(stats: dict[str, Any]) -> str:
    """Format statistics as a human-readable report."""
    lines: list[str] = []

    if stats.get("message") and stats.get("total", 0) == 0:
        return stats["message"]

    lines.append("=" * 50)
    lines.append("  Evaluation Statistics")
    lines.append("=" * 50)
    lines.append(f"  Total records: {stats['total']}")
    lines.append(f"  Reviewed:      {stats['reviewed']}")
    lines.append(f"  Approval rate: {stats['approval_rate']:.1%}")
    lines.append(f"  Accuracy rate: {stats['accuracy_rate']:.1%}")

    lines.append("")
    lines.append("  Judgments:")
    for j, count in sorted(stats.get("judgments", {}).items()):
        lines.append(f"    {j:>10s}: {count}")

    by_stage = stats.get("by_stage", {})
    if by_stage:
        lines.append("")
        lines.append("  By Stage:")
        for s, counts in sorted(by_stage.items()):
            total = sum(counts.values())
            approved = counts.get("approve", 0)
            rate = approved / total if total > 0 else 0.0
            lines.append(f"    {s}: {total} reviewed, {rate:.0%} approved")

    conf = stats.get("confidence_distribution", {})
    if any(v > 0 for v in conf.values()):
        lines.append("")
        lines.append("  Confidence Distribution (reviewed items):")
        for bucket, count in conf.items():
            if count > 0:
                lines.append(f"    {bucket}: {count}")

    issues = stats.get("common_issues", [])
    if issues:
        lines.append("")
        lines.append("  Common Issues:")
        for issue in issues[:5]:
            lines.append(f"    ({issue['count']}x) {issue['issue']}")

    return "\n".join(lines)
