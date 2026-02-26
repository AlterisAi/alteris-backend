"""Stage 0a: Automated ingestion completeness checks.

Codifies the SQL queries from the Stage 0a analysis into a reusable
function that produces a structured report against any Loom database.

Usage:
    from loom.eval.checks import run_stage0_checks
    report = run_stage0_checks("~/.loom/graph.db")
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _ts_to_str(ts: int) -> str:
    """Unix timestamp to human-readable string."""
    if ts <= 0:
        return "N/A"
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%b %d %H:%M")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Check functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_event_counts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Per-source event counts and date ranges."""
    rows = conn.execute(
        """SELECT source,
                  COUNT(*) as count,
                  MIN(timestamp) as earliest,
                  MAX(timestamp) as latest
           FROM events
           GROUP BY source
           ORDER BY count DESC"""
    ).fetchall()
    results = []
    for r in rows:
        results.append({
            "source": r["source"],
            "count": r["count"],
            "earliest": r["earliest"],
            "earliest_str": _ts_to_str(r["earliest"]),
            "latest": r["latest"],
            "latest_str": _ts_to_str(r["latest"]),
        })
    total = sum(r["count"] for r in results)
    results.append({"source": "TOTAL", "count": total, "earliest": None, "latest": None})
    return results


def check_gap_detection(conn: sqlite3.Connection, days: int = 30) -> dict[str, Any]:
    """Detect days with zero events per source over the last N days."""
    cutoff = int(time.time()) - days * 86400
    rows = conn.execute(
        """SELECT source,
                  DATE(timestamp, 'unixepoch') as day,
                  COUNT(*) as cnt
           FROM events
           WHERE timestamp >= ?
           GROUP BY source, day
           ORDER BY source, day""",
        (cutoff,),
    ).fetchall()

    by_source: dict[str, dict[str, int]] = {}
    for r in rows:
        src = r["source"]
        if src not in by_source:
            by_source[src] = {}
        by_source[src][r["day"]] = r["cnt"]

    # Generate all days in range
    from datetime import timedelta
    start = datetime.fromtimestamp(cutoff, tz=timezone.utc).date()
    end = datetime.fromtimestamp(int(time.time()), tz=timezone.utc).date()
    all_days = []
    d = start
    while d <= end:
        all_days.append(d.isoformat())
        d += timedelta(days=1)

    gaps: dict[str, list[str]] = {}
    daily_counts: dict[str, dict[str, int]] = {}
    for src, day_counts in by_source.items():
        missing = [day for day in all_days if day not in day_counts]
        if missing:
            gaps[src] = missing
        daily_counts[src] = day_counts

    return {
        "period_days": days,
        "sources": list(by_source.keys()),
        "gaps": gaps,
        "daily_counts": daily_counts,
    }


def check_duplicates(conn: sqlite3.Connection) -> dict[str, Any]:
    """Duplicate analysis: content_hash groups and source-ID collisions."""
    # Content-hash duplicates (groups with >1 event)
    hash_groups = conn.execute(
        """SELECT content_hash, COUNT(*) as cnt, GROUP_CONCAT(source) as sources
           FROM events
           WHERE content_hash != '' AND content_hash IS NOT NULL
           GROUP BY content_hash
           HAVING cnt > 1
           ORDER BY cnt DESC
           LIMIT 20"""
    ).fetchall()

    dup_groups = []
    excess_total = 0
    for r in hash_groups:
        excess = r["cnt"] - 1
        excess_total += excess
        dup_groups.append({
            "content_hash": r["content_hash"],
            "count": r["cnt"],
            "excess": excess,
            "sources": r["sources"],
        })

    total_dup_groups = conn.execute(
        """SELECT COUNT(*) FROM (
             SELECT content_hash FROM events
             WHERE content_hash != '' AND content_hash IS NOT NULL
             GROUP BY content_hash HAVING COUNT(*) > 1
           )"""
    ).fetchone()[0]

    total_excess = conn.execute(
        """SELECT SUM(cnt - 1) FROM (
             SELECT COUNT(*) as cnt FROM events
             WHERE content_hash != '' AND content_hash IS NOT NULL
             GROUP BY content_hash HAVING cnt > 1
           )"""
    ).fetchone()[0] or 0

    # Source-ID duplicates (should be zero with INSERT OR IGNORE)
    source_id_dups = conn.execute(
        """SELECT source, source_id, COUNT(*) as cnt
           FROM events
           GROUP BY source, source_id
           HAVING cnt > 1"""
    ).fetchall()

    # Cross-source duplicates (same content_hash in different sources)
    cross_source = conn.execute(
        """SELECT content_hash, COUNT(DISTINCT source) as src_count,
                  GROUP_CONCAT(DISTINCT source) as sources, COUNT(*) as total
           FROM events
           WHERE content_hash != '' AND content_hash IS NOT NULL
           GROUP BY content_hash
           HAVING src_count > 1
           ORDER BY total DESC
           LIMIT 10"""
    ).fetchall()

    total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    return {
        "total_events": total_events,
        "duplicate_hash_groups": total_dup_groups,
        "excess_duplicate_events": total_excess,
        "duplicate_pct": round(total_excess / total_events * 100, 1) if total_events else 0,
        "source_id_duplicates": len(source_id_dups),
        "top_duplicates": dup_groups[:10],
        "cross_source_duplicates": [
            {
                "content_hash": r["content_hash"],
                "sources": r["sources"],
                "total_events": r["total"],
            }
            for r in cross_source
        ],
    }


def check_schema_coverage(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Per-source fill rates for content, participants, metadata."""
    rows = conn.execute(
        """SELECT source,
                  COUNT(*) as total,
                  SUM(CASE WHEN raw_content IS NOT NULL AND raw_content != '' THEN 1 ELSE 0 END) as has_content,
                  SUM(CASE WHEN participants != '[]' AND participants IS NOT NULL THEN 1 ELSE 0 END) as has_participants,
                  SUM(CASE WHEN metadata != '{}' AND metadata IS NOT NULL THEN 1 ELSE 0 END) as has_metadata,
                  SUM(CASE WHEN content_hash != '' AND content_hash IS NOT NULL THEN 1 ELSE 0 END) as has_hash
           FROM events
           GROUP BY source
           ORDER BY source"""
    ).fetchall()

    results = []
    for r in rows:
        total = r["total"]
        results.append({
            "source": r["source"],
            "total": total,
            "content_pct": round(r["has_content"] / total * 100, 1) if total else 0,
            "participants_pct": round(r["has_participants"] / total * 100, 1) if total else 0,
            "metadata_pct": round(r["has_metadata"] / total * 100, 1) if total else 0,
            "content_hash_pct": round(r["has_hash"] / total * 100, 1) if total else 0,
        })
    return results


def check_encoding_issues(conn: sqlite3.Connection) -> dict[str, Any]:
    """Detect HTML tags, quoted-printable artifacts, and other encoding problems."""
    # We scan in batches to avoid loading all content into memory
    issues: dict[str, list[dict[str, Any]]] = {
        "html_tags": [],
        "qp_artifacts": [],
        "very_short": [],
        "empty_content": [],
        "very_long": [],
    }

    rows = conn.execute(
        """SELECT id, source, LENGTH(raw_content) as content_len, raw_content
           FROM events
           WHERE raw_content IS NOT NULL"""
    ).fetchall()

    for r in rows:
        content = r["raw_content"] or ""
        content_len = r["content_len"] or 0
        source = r["source"]

        if content_len == 0:
            issues["empty_content"].append({"id": r["id"], "source": source})
            continue

        if content_len < 10:
            issues["very_short"].append({
                "id": r["id"], "source": source,
                "content": content[:50],
            })

        if content_len > 50000:
            issues["very_long"].append({
                "id": r["id"], "source": source,
                "length": content_len,
            })

        # Only scan first 2000 chars for patterns (performance)
        sample = content[:2000]
        if re.search(r"<[a-zA-Z][^>]*>", sample):
            issues["html_tags"].append({"id": r["id"], "source": source})

        if re.search(r"=[0-9A-F]{2}", sample):
            issues["qp_artifacts"].append({"id": r["id"], "source": source})

    # Also count events with NULL content
    null_count = conn.execute(
        """SELECT COUNT(*) FROM events WHERE raw_content IS NULL"""
    ).fetchone()[0]

    # Aggregate by source
    def _by_source(items: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            src = item["source"]
            counts[src] = counts.get(src, 0) + 1
        return counts

    return {
        "html_tags": {"count": len(issues["html_tags"]), "by_source": _by_source(issues["html_tags"])},
        "qp_artifacts": {"count": len(issues["qp_artifacts"]), "by_source": _by_source(issues["qp_artifacts"])},
        "very_short": {"count": len(issues["very_short"]), "by_source": _by_source(issues["very_short"])},
        "empty_content": {"count": len(issues["empty_content"]) + null_count, "by_source": _by_source(issues["empty_content"])},
        "very_long": {"count": len(issues["very_long"]), "by_source": _by_source(issues["very_long"])},
    }


def check_email_body_quality(conn: sqlite3.Connection) -> dict[str, Any]:
    """Check email content for leaked quoted text, signatures, and HTML artifacts.

    This catches cases where the quote-stripping in the mail adapter failed:
    - Quoted reply chains (lines starting with ">")
    - "On ... wrote:" attribution lines
    - Outlook-style "From: / Sent: / To: / Subject:" header blocks
    - "---------- Forwarded message ---------" markers
    - Signature blocks ("Best regards", "Sent from my iPhone", etc.)
    - Raw HTML that wasn't converted to text
    - Invisible characters / zero-width spaces
    """
    rows = conn.execute(
        """SELECT id, source_id, raw_content, metadata, timestamp
           FROM events
           WHERE source = 'mail' AND raw_content IS NOT NULL AND raw_content != ''"""
    ).fetchall()

    total_emails = len(rows)
    issues: dict[str, list[dict[str, Any]]] = {
        "quoted_reply": [],
        "on_wrote_attribution": [],
        "outlook_headers": [],
        "forwarded_markers": [],
        "signature_leak": [],
        "raw_html": [],
        "invisible_chars": [],
    }

    for r in rows:
        content = r["raw_content"]
        eid = r["id"]
        meta = r["metadata"] or "{}"
        try:
            meta_d = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            meta_d = {}
        subject = meta_d.get("subject", "")

        lines = content.split("\n")
        content_lower = content.lower()

        # Quoted reply lines ("> " at start)
        quoted_lines = [l for l in lines if l.startswith(">")]
        if len(quoted_lines) >= 2:
            issues["quoted_reply"].append({
                "id": eid, "subject": subject,
                "quoted_lines": len(quoted_lines),
                "sample": quoted_lines[0][:100],
            })

        # "On ... wrote:" attribution
        for line in lines:
            if re.match(r"^On .+ wrote:\s*$", line):
                issues["on_wrote_attribution"].append({
                    "id": eid, "subject": subject,
                    "line": line[:120],
                })
                break

        # Outlook-style header blocks
        if re.search(r"^From:\s+.+\nSent:\s+.+\nTo:\s+", content, re.MULTILINE):
            issues["outlook_headers"].append({
                "id": eid, "subject": subject,
            })

        # Forwarded message markers
        if "---------- Forwarded message" in content or "Begin forwarded message" in content_lower:
            issues["forwarded_markers"].append({
                "id": eid, "subject": subject,
            })

        # Signature noise — only flag actual boilerplate, not sign-offs
        # Sign-offs like "Thanks!" or "Best regards," carry tone/relationship
        # signal, so we leave those alone. Only flag device boilerplate and
        # legal disclaimers that add zero information.
        tail_lines = lines[-15:] if len(lines) > 15 else lines
        noise_patterns = [
            r"^sent from my (iphone|ipad|galaxy|android)",
            r"^get outlook for",
            r"^this (email|message) (is|was|may be) (confidential|intended)",
        ]
        for tl in tail_lines:
            tl_stripped = tl.strip().lower()
            for pat in noise_patterns:
                if re.match(pat, tl_stripped):
                    issues["signature_leak"].append({
                        "id": eid, "subject": subject,
                        "line": tl.strip()[:80],
                    })
                    break
            else:
                continue
            break

        # Raw HTML tags remaining (not just <https://...> links)
        html_matches = re.findall(r"<(?!https?://)[a-zA-Z][^>]{0,50}>", content[:3000])
        if len(html_matches) >= 3:
            issues["raw_html"].append({
                "id": eid, "subject": subject,
                "tag_count": len(html_matches),
                "sample": html_matches[0],
            })

        # Invisible / zero-width characters
        invisible = re.findall(r"[\u200b\u200c\u200d\u00ad\ufeff]", content[:3000])
        if len(invisible) >= 5:
            issues["invisible_chars"].append({
                "id": eid, "subject": subject,
                "count": len(invisible),
            })

    return {
        "total_emails": total_emails,
        "quoted_reply": {
            "count": len(issues["quoted_reply"]),
            "samples": issues["quoted_reply"][:10],
        },
        "on_wrote_attribution": {
            "count": len(issues["on_wrote_attribution"]),
            "samples": issues["on_wrote_attribution"][:10],
        },
        "outlook_headers": {
            "count": len(issues["outlook_headers"]),
            "samples": issues["outlook_headers"][:5],
        },
        "forwarded_markers": {
            "count": len(issues["forwarded_markers"]),
            "samples": issues["forwarded_markers"][:5],
        },
        "signature_leak": {
            "count": len(issues["signature_leak"]),
            "samples": issues["signature_leak"][:10],
        },
        "raw_html": {
            "count": len(issues["raw_html"]),
            "samples": issues["raw_html"][:10],
        },
        "invisible_chars": {
            "count": len(issues["invisible_chars"]),
            "samples": issues["invisible_chars"][:5],
        },
    }


def check_sensitivity_distribution(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Sensitivity level distribution per source."""
    rows = conn.execute(
        """SELECT source, sensitivity, COUNT(*) as cnt
           FROM events
           GROUP BY source, sensitivity
           ORDER BY source, sensitivity"""
    ).fetchall()
    return [{"source": r["source"], "sensitivity": r["sensitivity"], "count": r["cnt"]} for r in rows]


def check_downstream_state(conn: sqlite3.Connection) -> dict[str, Any]:
    """Counts for all downstream tables: persons, edges, claims, beliefs."""
    result: dict[str, Any] = {}

    # Table counts
    for table in ("persons", "event_persons", "claims", "beliefs"):
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()  # noqa: S608
            result[f"{table}_count"] = row[0]
        except sqlite3.OperationalError:
            result[f"{table}_count"] = 0

    # Claims by type
    try:
        rows = conn.execute(
            """SELECT claim_type, COUNT(*) as cnt
               FROM claims
               GROUP BY claim_type
               ORDER BY cnt DESC"""
        ).fetchall()
        result["claims_by_type"] = {r["claim_type"]: r["cnt"] for r in rows}
    except sqlite3.OperationalError:
        result["claims_by_type"] = {}

    # Beliefs by type
    try:
        rows = conn.execute(
            """SELECT belief_type, COUNT(*) as cnt
               FROM beliefs
               GROUP BY belief_type
               ORDER BY cnt DESC"""
        ).fetchall()
        result["beliefs_by_type"] = {r["belief_type"]: r["cnt"] for r in rows}
    except sqlite3.OperationalError:
        result["beliefs_by_type"] = {}

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main check runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_stage0_checks(db_path: str | Path) -> dict[str, Any]:
    """Run all Stage 0a completeness checks. Returns structured results."""
    db_path = Path(db_path).expanduser()
    if not db_path.exists():
        return {"error": f"Database not found: {db_path}"}

    conn = _connect(db_path)
    try:
        report: dict[str, Any] = {
            "db_path": str(db_path),
            "checked_at": int(time.time()),
            "event_counts": check_event_counts(conn),
            "gap_detection": check_gap_detection(conn),
            "duplicates": check_duplicates(conn),
            "schema_coverage": check_schema_coverage(conn),
            "encoding_issues": check_encoding_issues(conn),
            "email_body_quality": check_email_body_quality(conn),
            "sensitivity": check_sensitivity_distribution(conn),
            "downstream": check_downstream_state(conn),
        }
        return report
    finally:
        conn.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Report formatting
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def format_report(report: dict[str, Any]) -> str:
    """Format a Stage 0a report as a human-readable string."""
    if "error" in report:
        return f"Error: {report['error']}"

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  Stage 0a: Ingestion Completeness Report")
    lines.append("=" * 60)
    lines.append("")

    # 1. Event counts
    lines.append("1. Per-Source Event Counts")
    lines.append("-" * 50)
    lines.append(f"  {'Source':<12} {'Count':>7}  {'Earliest':<14} {'Latest':<14}")
    for r in report["event_counts"]:
        if r["source"] == "TOTAL":
            lines.append(f"  {'─' * 46}")
            lines.append(f"  {'TOTAL':<12} {r['count']:>7}")
        else:
            lines.append(
                f"  {r['source']:<12} {r['count']:>7}  "
                f"{r.get('earliest_str', 'N/A'):<14} {r.get('latest_str', 'N/A'):<14}"
            )
    lines.append("")

    # 2. Gap detection
    gaps = report["gap_detection"]
    lines.append(f"2. Gap Detection (last {gaps['period_days']} days)")
    lines.append("-" * 50)
    if gaps["gaps"]:
        for src, missing_days in gaps["gaps"].items():
            lines.append(f"  {src}: {len(missing_days)} missing day(s)")
            if len(missing_days) <= 5:
                lines.append(f"    days: {', '.join(missing_days)}")
    else:
        lines.append("  No fully missing days detected.")
    lines.append("")

    # 3. Duplicates
    dups = report["duplicates"]
    lines.append("3. Duplicate Analysis")
    lines.append("-" * 50)
    lines.append(f"  Content-hash duplicate groups: {dups['duplicate_hash_groups']}")
    lines.append(f"  Excess duplicate events: {dups['excess_duplicate_events']} ({dups['duplicate_pct']}%)")
    lines.append(f"  Source-ID duplicates: {dups['source_id_duplicates']}")
    if dups["top_duplicates"]:
        lines.append("  Top duplicates:")
        for d in dups["top_duplicates"][:5]:
            lines.append(f"    {d['content_hash']}: {d['count']}x ({d['sources']})")
    if dups["cross_source_duplicates"]:
        lines.append(f"  Cross-source duplicates: {len(dups['cross_source_duplicates'])} hashes")
    lines.append("")

    # 4. Schema coverage
    lines.append("4. Schema Coverage")
    lines.append("-" * 50)
    lines.append(f"  {'Source':<12} {'Content%':>9} {'Particip%':>10} {'Metadata%':>10}")
    for r in report["schema_coverage"]:
        lines.append(
            f"  {r['source']:<12} {r['content_pct']:>8.1f}% "
            f"{r['participants_pct']:>9.1f}% {r['metadata_pct']:>9.1f}%"
        )
    lines.append("")

    # 5. Encoding issues
    enc = report["encoding_issues"]
    lines.append("5. Encoding / Sanitization Issues")
    lines.append("-" * 50)
    for issue_type in ("html_tags", "qp_artifacts", "very_short", "empty_content", "very_long"):
        info = enc[issue_type]
        label = issue_type.replace("_", " ").title()
        by_src = info.get("by_source", {})
        src_str = ", ".join(f"{s}={c}" for s, c in sorted(by_src.items())) if by_src else ""
        lines.append(f"  {label:<25} {info['count']:>5}  {src_str}")
    lines.append("")

    # 6. Email body quality
    ebq = report.get("email_body_quality", {})
    if ebq:
        lines.append(f"6. Email Body Quality ({ebq.get('total_emails', 0)} emails)")
        lines.append("-" * 50)
        for issue_key in ("quoted_reply", "on_wrote_attribution", "outlook_headers",
                          "forwarded_markers", "signature_leak", "raw_html", "invisible_chars"):
            info = ebq.get(issue_key, {})
            label = issue_key.replace("_", " ").title()
            count = info.get("count", 0)
            lines.append(f"  {label:<30} {count:>5}")
            if count > 0:
                for s in info.get("samples", [])[:3]:
                    subj = s.get("subject", "")[:50]
                    extra = ""
                    if "quoted_lines" in s:
                        extra = f" ({s['quoted_lines']} quoted lines)"
                    elif "line" in s:
                        extra = f" [{s['line'][:60]}]"
                    elif "tag_count" in s:
                        extra = f" ({s['tag_count']} tags)"
                    lines.append(f"    - {subj}{extra}")
        lines.append("")

    # 7. Sensitivity distribution
    lines.append("7. Sensitivity Distribution")
    lines.append("-" * 50)
    for r in report["sensitivity"]:
        lines.append(f"  {r['source']:<12} level={r['sensitivity']}: {r['count']}")
    lines.append("")

    # 8. Downstream state
    ds = report["downstream"]
    lines.append("8. Downstream Pipeline State")
    lines.append("-" * 50)
    lines.append(f"  Persons:       {ds.get('persons_count', 0)}")
    lines.append(f"  Event-Person:  {ds.get('event_persons_count', 0)}")
    lines.append(f"  Claims:        {ds.get('claims_count', 0)}")
    if ds.get("claims_by_type"):
        for ct, cnt in ds["claims_by_type"].items():
            lines.append(f"    {ct}: {cnt}")
    lines.append(f"  Beliefs:       {ds.get('beliefs_count', 0)}")
    if ds.get("beliefs_by_type"):
        for bt, cnt in ds["beliefs_by_type"].items():
            lines.append(f"    {bt}: {cnt}")

    return "\n".join(lines)


def write_markdown_report(
    report: dict[str, Any],
    samples: dict[str, list[dict[str, Any]]] | None = None,
    output_path: str | Path = "eval_report.md",
) -> str:
    """Write a full eval report as Markdown. Returns the output path."""
    from datetime import datetime, timezone

    checked_at = report.get("checked_at", 0)
    ts_str = datetime.fromtimestamp(checked_at, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if checked_at else "N/A"

    md: list[str] = []
    md.append(f"# Loom Eval Report")
    md.append(f"")
    md.append(f"**Database:** `{report.get('db_path', 'N/A')}`")
    md.append(f"**Generated:** {ts_str}")
    md.append(f"")

    # ── Section 1: Event Counts ──
    md.append("## 1. Per-Source Event Counts")
    md.append("")
    md.append("| Source | Count | Earliest | Latest |")
    md.append("|--------|------:|----------|--------|")
    for r in report.get("event_counts", []):
        if r["source"] == "TOTAL":
            md.append(f"| **TOTAL** | **{r['count']}** | | |")
        else:
            md.append(f"| {r['source']} | {r['count']} | {r.get('earliest_str', 'N/A')} | {r.get('latest_str', 'N/A')} |")
    md.append("")

    # ── Section 2: Gap Detection ──
    gaps = report.get("gap_detection", {})
    md.append(f"## 2. Gap Detection (last {gaps.get('period_days', 30)} days)")
    md.append("")
    if gaps.get("gaps"):
        md.append("| Source | Missing Days | Dates |")
        md.append("|--------|------------:|-------|")
        for src, missing in gaps["gaps"].items():
            dates = ", ".join(missing[:5])
            if len(missing) > 5:
                dates += f" ... +{len(missing)-5} more"
            md.append(f"| {src} | {len(missing)} | {dates} |")
    else:
        md.append("No fully missing days detected.")
    md.append("")

    # ── Section 3: Duplicates ──
    dups = report.get("duplicates", {})
    md.append("## 3. Duplicate Analysis")
    md.append("")
    md.append(f"- **Content-hash duplicate groups:** {dups.get('duplicate_hash_groups', 0)}")
    md.append(f"- **Excess duplicate events:** {dups.get('excess_duplicate_events', 0)} ({dups.get('duplicate_pct', 0)}%)")
    md.append(f"- **Source-ID duplicates:** {dups.get('source_id_duplicates', 0)}")
    md.append(f"- **Cross-source duplicates:** {len(dups.get('cross_source_duplicates', []))} hashes")
    top_dups = dups.get("top_duplicates", [])
    if top_dups:
        md.append("")
        md.append("**Top duplicates:**")
        md.append("")
        md.append("| Hash | Count | Sources |")
        md.append("|------|------:|---------|")
        for d in top_dups[:5]:
            # Deduplicate the sources string
            sources = d.get("sources", "")
            unique_src = sorted(set(sources.split(",")))
            md.append(f"| `{d['content_hash']}` | {d['count']} | {', '.join(unique_src)} |")
    md.append("")

    # ── Section 4: Schema Coverage ──
    md.append("## 4. Schema Coverage")
    md.append("")
    md.append("| Source | Content % | Participants % | Metadata % |")
    md.append("|--------|----------:|---------------:|-----------:|")
    for r in report.get("schema_coverage", []):
        md.append(f"| {r['source']} | {r['content_pct']}% | {r['participants_pct']}% | {r['metadata_pct']}% |")
    md.append("")

    # ── Section 5: Encoding Issues ──
    enc = report.get("encoding_issues", {})
    md.append("## 5. Encoding / Sanitization Issues")
    md.append("")
    md.append("| Issue | Count | Sources |")
    md.append("|-------|------:|---------|")
    for issue_type in ("html_tags", "qp_artifacts", "very_short", "empty_content", "very_long"):
        info = enc.get(issue_type, {})
        label = issue_type.replace("_", " ").title()
        by_src = info.get("by_source", {})
        src_str = ", ".join(f"{s}={c}" for s, c in sorted(by_src.items())) if by_src else ""
        md.append(f"| {label} | {info.get('count', 0)} | {src_str} |")
    md.append("")

    # ── Section 6: Email Body Quality ──
    ebq = report.get("email_body_quality", {})
    if ebq:
        md.append(f"## 6. Email Body Quality ({ebq.get('total_emails', 0)} emails scanned)")
        md.append("")
        md.append("| Issue | Count |")
        md.append("|-------|------:|")
        for issue_key in ("quoted_reply", "on_wrote_attribution", "outlook_headers",
                          "forwarded_markers", "signature_leak", "raw_html", "invisible_chars"):
            info = ebq.get(issue_key, {})
            label = issue_key.replace("_", " ").title()
            md.append(f"| {label} | {info.get('count', 0)} |")

        md.append("")

        # Samples for each non-zero issue
        for issue_key in ("quoted_reply", "on_wrote_attribution", "outlook_headers",
                          "forwarded_markers", "signature_leak", "raw_html", "invisible_chars"):
            info = ebq.get(issue_key, {})
            if info.get("count", 0) > 0:
                label = issue_key.replace("_", " ").title()
                md.append(f"### {label} — samples")
                md.append("")
                for s in info.get("samples", [])[:5]:
                    eid = s.get("id", "?")[:16]
                    subj = s.get("subject", "(no subject)")
                    detail = ""
                    if "quoted_lines" in s:
                        detail = f" — {s['quoted_lines']} quoted lines, e.g. `{s.get('sample', '')[:60]}`"
                    elif "line" in s:
                        detail = f" — `{s['line'][:80]}`"
                    elif "tag_count" in s:
                        detail = f" — {s['tag_count']} tags, e.g. `{s.get('sample', '')}`"
                    elif "count" in s:
                        detail = f" — {s['count']} occurrences"
                    md.append(f"- `{eid}` **{subj}**{detail}")
                md.append("")

    # ── Section 7: Sensitivity Distribution ──
    md.append("## 7. Sensitivity Distribution")
    md.append("")
    md.append("| Source | Level | Count |")
    md.append("|--------|------:|------:|")
    for r in report.get("sensitivity", []):
        md.append(f"| {r['source']} | {r['sensitivity']} | {r['count']} |")
    md.append("")

    # ── Section 8: Downstream State ──
    ds = report.get("downstream", {})
    md.append("## 8. Downstream Pipeline State")
    md.append("")
    md.append("| Table | Count |")
    md.append("|-------|------:|")
    md.append(f"| Persons | {ds.get('persons_count', 0)} |")
    md.append(f"| Event-Person edges | {ds.get('event_persons_count', 0)} |")
    md.append(f"| Claims | {ds.get('claims_count', 0)} |")
    md.append(f"| Beliefs | {ds.get('beliefs_count', 0)} |")

    if ds.get("claims_by_type"):
        md.append("")
        md.append("**Claims by type:**")
        md.append("")
        md.append("| Type | Count |")
        md.append("|------|------:|")
        for ct, cnt in ds["claims_by_type"].items():
            md.append(f"| {ct} | {cnt} |")

    if ds.get("beliefs_by_type"):
        md.append("")
        md.append("**Beliefs by type:**")
        md.append("")
        md.append("| Type | Count |")
        md.append("|------|------:|")
        for bt, cnt in ds["beliefs_by_type"].items():
            md.append(f"| {bt} | {cnt} |")
    md.append("")

    # ── Samples (if provided) ──
    if samples:
        md.append("---")
        md.append("")
        md.append("# Stage Samples")
        md.append("")

        for stage_name, items in samples.items():
            if not items:
                continue
            md.append(f"## {stage_name}")
            md.append("")

            for i, item in enumerate(items, 1):
                md.append(f"### Sample {i}")
                md.append("")
                md.append("```json")
                # Clean for display — truncate long content
                display_item = {}
                for k, v in item.items():
                    if k == "raw_content" and isinstance(v, str) and len(v) > 300:
                        display_item[k] = v[:300] + "..."
                    elif k == "content_preview" and isinstance(v, str) and len(v) > 200:
                        display_item[k] = v[:200] + "..."
                    elif k == "_event" and isinstance(v, dict):
                        ev = dict(v)
                        if "raw_content" in ev and isinstance(ev["raw_content"], str) and len(ev["raw_content"]) > 300:
                            ev["raw_content"] = ev["raw_content"][:300] + "..."
                        display_item[k] = ev
                    else:
                        display_item[k] = v
                md.append(json.dumps(display_item, indent=2, ensure_ascii=False, default=str))
                md.append("```")
                md.append("")

    content = "\n".join(md)
    Path(output_path).write_text(content)
    return str(output_path)
