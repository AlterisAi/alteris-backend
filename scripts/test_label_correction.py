#!/usr/bin/env python3
"""Test whether thread-level context improves triage label quality.

Emulates what the synthesis stage sees: full thread text + person context +
triage summary, then asks a model to review/correct the per-message labels.

Picks a mix of singleton threads and multi-message threads from graph.db.
For the 30 comparison events (where we have Pro gold labels), measures
improvement via Jaccard overlap. For all threads, reports what changed.

Usage:
    python scripts/test_label_correction.py
    python scripts/test_label_correction.py --model gemini-2.5-flash-lite
    python scripts/test_label_correction.py --model gemini-3-flash-preview
    python scripts/test_label_correction.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types as genai_types

sys.path.insert(0, "src")

from alteris.prompts.triage import SPECIFIC_TOPICS, UNIVERSAL_SPHERES

# ─── Gemini client setup ────────────────────────────────────────────

def _load_api_key() -> str:
    cfg_path = Path.home() / ".alteris" / "config.json"
    with open(cfg_path) as f:
        return json.load(f)["gemini_api_key"]

_GENAI_CLIENT: genai.Client | None = None

def _get_client() -> genai.Client:
    global _GENAI_CLIENT
    if _GENAI_CLIENT is None:
        _GENAI_CLIENT = genai.Client(api_key=_load_api_key())
    return _GENAI_CLIENT

# ─── Valid taxonomy ──────────────────────────────────────────────────

VALID_DOMAINS = [
    "work", "personal", "family", "financial", "health",
    "legal", "travel", "shopping", "automated",
]

# ─── Thread selection ────────────────────────────────────────────────

# Singleton event IDs from our comparison set (have Pro gold labels).
# Pick ones with interesting Lite-vs-Pro mismatches.
SINGLETON_EVENT_IDS = [
    "fbbd4013d5f4665dcf6d5cbd",   # domain: automated(lite) vs work(pro)
    "6855bc58891f3b2e23cdf980",   # domain: work(lite) vs automated(pro)
    "a496b95d0b5f01c5ce68ca80",   # domain: shopping(lite) vs financial(pro)
    "6632652c2f904d58495679a1",   # domain: personal(lite) vs work(pro)
    "7602ec2deb7ac3eba8c5f228",   # domain: work(lite) vs automated(pro)
    "40bbc3f99ec6b8e490a614a1",   # topics: recruiting(lite) vs interviewing(pro)
    "6e1dad202462f3a58703a00a",   # topics: hobby(lite) vs friend+hiking(pro)
    "30bc382c55fab997c3119ef0",   # topics: primary_care(lite) vs specialist(pro)
]

# Multi-message thread IDs from graph.db (mail + whatsapp).
THREAD_IDS = [
    "219947",                          # 5 msgs — Anthropic interview
    "220433",                          # 4 msgs — Alteris <> GPV meeting
    "219021",                          # 8 msgs — Mauna Lani reservation
    "120363422826118805@g.us",         # 5 msgs — Alteris Core (WhatsApp)
    "219581",                          # 5 msgs — Meta recruiter call
]

# ─── Correction prompt ──────────────────────────────────────────────

CORRECTION_SYSTEM = """\
You are a triage quality reviewer for a personal knowledge management system.
You receive a THREAD of messages (possibly just one) along with the initial
per-message triage labels from a fast model. Your job is to review and CORRECT
the labels using the full thread context.

VALID DOMAINS: {domains}

UNIVERSAL SPHERES (pick 1-2 per message): {spheres}

SPECIFIC TOPICS (pick 0-3 per message, ONLY from this list): {topics}

Rules:
- Review ALL messages in the thread together — context from earlier/later
  messages should inform label corrections.
- Common mistakes by the fast model:
  * Labeling automated notifications as "work" or "personal" — newsletters,
    transaction alerts, webhook notifications, and system-generated emails
    should be domain="automated".
  * Using vague topics when specific ones exist (e.g. "software_backend_development"
    for a recruiting email should be "talent_sourcing_and_recruiting").
  * Missing topics that become clear from thread context (e.g. a reply
    mentioning a deadline reveals "event_planning_and_conference_ops").
  * Assigning "personal" when the thread is clearly work-related or vice versa.
- Only change what's clearly wrong. Keep corrections conservative.
- Output ONLY valid JSON.\
""".format(
    domains=", ".join(VALID_DOMAINS),
    spheres=", ".join(UNIVERSAL_SPHERES),
    topics=", ".join(SPECIFIC_TOPICS),
)

CORRECTION_PROMPT = """\
{thread_text}

CONTEXT:
{context_section}

---

PER-MESSAGE TRIAGE LABELS (from fast model — review and correct):
{per_message_labels}

---

Review each message's labels in the context of the full thread. Return JSON:
{{
  "messages": [
    {{
      "id": "event_id",
      "domain": "...",
      "universal_spheres": ["..."],
      "specific_topics": ["..."],
      "corrected": true/false,
      "correction_note": "brief explanation or 'no change'"
    }}
  ],
  "thread_level_notes": "what the thread context revealed that per-message triage missed"
}}
"""

# ─── Data loading ────────────────────────────────────────────────────

def open_graph_db() -> sqlite3.Connection:
    conn = sqlite3.connect(
        "file:///Users/aniruddha/.alteris/graph.db?mode=ro", uri=True,
    )
    conn.row_factory = sqlite3.Row
    return conn


def load_comparison_gold() -> tuple[dict, dict]:
    """Load Flash Lite and Pro labels from the comparison run."""
    try:
        with open("scripts/model_comparison_results.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}, {}

    lite_items: dict[str, dict] = {}
    pro_items: dict[str, dict] = {}
    for batch in data.get("per_batch", {}).get("gemini-2.5-flash-lite", []):
        if batch.get("response"):
            parsed = json.loads(batch["response"])
            items = parsed if isinstance(parsed, list) else parsed.get("items", [parsed])
            for it in items:
                if isinstance(it, dict) and "id" in it:
                    lite_items[str(it["id"])] = it
    for batch in data.get("per_batch", {}).get("gemini-3.1-pro-preview", []):
        if batch.get("response"):
            parsed = json.loads(batch["response"])
            items = parsed if isinstance(parsed, list) else parsed.get("items", [parsed])
            for it in items:
                if isinstance(it, dict) and "id" in it:
                    pro_items[str(it["id"])] = it
    return lite_items, pro_items


def fetch_thread_events(conn: sqlite3.Connection, thread_id: str) -> list[dict]:
    """Fetch all events in a thread, ordered chronologically."""
    rows = conn.execute("""
        SELECT id, source, timestamp, raw_content, metadata, participants
        FROM events
        WHERE json_extract(metadata, '$.thread_id') = ?
        ORDER BY timestamp ASC
    """, (thread_id,)).fetchall()
    return [dict(r) for r in rows]


def fetch_singleton_event(conn: sqlite3.Connection, event_id: str) -> dict | None:
    """Fetch a single event by ID."""
    row = conn.execute("""
        SELECT id, source, timestamp, raw_content, metadata, participants
        FROM events WHERE id = ?
    """, (event_id,)).fetchone()
    return dict(row) if row else None


def fetch_triage_data(conn: sqlite3.Connection, event_id: str) -> dict | None:
    """Get the existing triage claim for an event."""
    row = conn.execute("""
        SELECT object FROM claims
        WHERE claim_type = 'triage' AND subject = ?
          AND superseded_by IS NULL
        LIMIT 1
    """, (event_id,)).fetchone()
    if row:
        return json.loads(row["object"])
    return None


def fetch_persons_for_events(
    conn: sqlite3.Connection, event_ids: list[str],
) -> dict[str, list[dict]]:
    """Get person info for events."""
    if not event_ids:
        return {}
    placeholders = ",".join("?" * len(event_ids))
    rows = conn.execute(f"""
        SELECT ep.event_id, ep.person_id, ep.role,
               p.canonical_name, p.is_user
        FROM event_persons ep
        JOIN persons p ON p.person_id = ep.person_id
        WHERE ep.event_id IN ({placeholders})
        ORDER BY ep.event_id, ep.role
    """, event_ids).fetchall()

    result: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        result[r["event_id"]].append({
            "person_id": r["person_id"],
            "name": r["canonical_name"] or r["person_id"][:12],
            "role": r["role"],
            "is_user": bool(r["is_user"]),
        })
    return dict(result)


# ─── Context builders (emulate beliefs.py) ───────────────────────────

def format_event_text(event: dict, index: int, persons: list[dict]) -> str:
    """Format a single event like _format_thread_for_llm does."""
    meta = json.loads(event["metadata"]) if event["metadata"] else {}
    source = event["source"]
    content = (event["raw_content"] or "")[:3000]
    ts = event["timestamp"]
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sender_persons = [p for p in persons if p["role"] == "sender"]
    recip_persons = [p for p in persons if p["role"] in ("recipient", "member") and not p["is_user"]]
    sender_name = sender_persons[0]["name"] if sender_persons else meta.get("sender", "Unknown")
    is_from_me = meta.get("is_from_me", False)

    lines = [f"## Message {index}"]

    if source == "mail":
        if is_from_me:
            lines.append("**From:** USER (you)")
        else:
            lines.append(f"**From:** {sender_name}")
        if recip_persons:
            names = ", ".join(p["name"] for p in recip_persons[:5])
            lines.append(f"**To:** {names}")
        subject = meta.get("subject", "")
        if subject:
            lines.append(f"**Subject:** {subject}")
        lines.append(f"**Date:** {dt}")
        lines.append("")
        lines.append("### Body")
        lines.append(content)

    elif source in ("imessage", "whatsapp"):
        if is_from_me:
            recip = recip_persons[0]["name"] if recip_persons else "contact"
            lines.append(f"**Message from:** USER (you) -> {recip}")
            lines.append("**Direction:** outbound (user sent this)")
        else:
            lines.append(f"**Message from:** {sender_name}")
            lines.append("**Direction:** inbound (sent to user)")
        group_name = meta.get("group_name")
        if group_name:
            lines.append(f"**Group:** {group_name}")
        lines.append(f"**Date:** {dt}")
        lines.append("")
        lines.append("### Body")
        lines.append(content)

    else:
        lines.append(f"**Source:** {source}")
        lines.append(f"**Date:** {dt}")
        lines.append("")
        lines.append(content)

    return "\n".join(lines)


def build_thread_text(
    events: list[dict], persons_cache: dict[str, list[dict]],
) -> str:
    """Build the full thread text like _format_thread_for_llm."""
    parts = ["THREAD CONTENT:"]
    for i, ev in enumerate(events):
        persons = persons_cache.get(ev["id"], [])
        parts.append("---")
        parts.append(format_event_text(ev, i + 1, persons))
    return "\n\n".join(parts)


def build_person_context(
    events: list[dict], persons_cache: dict[str, list[dict]],
) -> str:
    """Build person context like _build_person_context."""
    seen = set()
    lines = ["PERSON CONTEXT:"]
    for ev in events:
        for p in persons_cache.get(ev["id"], []):
            if p["is_user"] or p["person_id"] in seen:
                continue
            seen.add(p["person_id"])
            lines.append(f"  {p['name']}: role={p['role']}")
    if len(lines) == 1:
        lines.append("  No known contacts in this thread.")
    return "\n".join(lines)


def build_triage_context(triage_data: list[dict]) -> str:
    """Build triage summary like _build_triage_context."""
    if not triage_data:
        return "TRIAGE SUMMARY: No triage data available."

    scores = [d.get("score", 0) for d in triage_data]
    domains = sorted(set(d.get("domain", "") for d in triage_data if d.get("domain")))
    topics = set()
    for d in triage_data:
        for t in (d.get("specific_topics") or d.get("topics") or []):
            topics.add(t)
    topics_sorted = sorted(topics)[:5]

    avg = sum(scores) / len(scores) if scores else 0
    lines = [f"TRIAGE SUMMARY: avg_score={avg:.2f}"]
    if domains:
        lines.append(f"  domains: {', '.join(domains)}")
    if topics_sorted:
        lines.append(f"  topics: {', '.join(topics_sorted)}")
    return "\n".join(lines)


def build_group_context(events: list[dict]) -> str:
    """Build group metadata like _build_group_context."""
    for ev in events:
        meta = json.loads(ev["metadata"]) if ev["metadata"] else {}
        if meta.get("is_group") or meta.get("group_name"):
            gname = meta.get("group_name", "unknown")
            return f"GROUP METADATA: is_group=true, group_name=\"{gname}\""
    return "GROUP METADATA: is_group=false (direct conversation)"


def build_thread_age_context(events: list[dict]) -> str:
    """Build thread age context like _build_thread_age_context."""
    if not events:
        return ""
    now = int(time.time())
    timestamps = [e["timestamp"] for e in events]
    latest = max(timestamps)
    earliest = min(timestamps)
    age_days = (now - latest) / 86400
    span_days = (latest - earliest) / 86400
    lines = [f"THREAD AGE: last_activity={age_days:.0f} days ago, thread_span={span_days:.0f} days"]
    if age_days > 30:
        lines.append(f"  WARNING: Thread is {age_days:.0f} days old — likely stale")
    return "\n".join(lines)


def build_per_message_labels(events: list[dict], triage_data: list[dict]) -> str:
    """Format per-message triage labels for the correction prompt."""
    lines = []
    for ev, td in zip(events, triage_data):
        meta = json.loads(ev["metadata"]) if ev["metadata"] else {}
        subject = meta.get("subject", "")[:40]
        content_preview = (ev["raw_content"] or "")[:60].replace("\n", " ")
        lines.append(f"Message {ev['id']}:")
        if subject:
            lines.append(f"  subject: {subject}")
        lines.append(f"  preview: {content_preview}...")
        lines.append(f"  score: {td.get('score', 0)}")
        lines.append(f"  domain: {td.get('domain', '')}")
        lines.append(f"  universal_spheres: {json.dumps(td.get('universal_spheres', []))}")
        lines.append(f"  specific_topics: {json.dumps(td.get('specific_topics', td.get('topics', [])))}")
        lines.append(f"  commitment_type: {td.get('commitment_type') or 'null'}")
        lines.append("")
    return "\n".join(lines)


# ─── LLM call ───────────────────────────────────────────────────────

def call_gemini(model: str, prompt: str, system: str) -> tuple[str | None, float]:
    client = _get_client()

    # Model-specific thinking config
    thinking_cfg = None
    if "3.1-pro" in model:
        thinking_cfg = genai_types.ThinkingConfig(thinking_budget=-1)  # auto
    else:
        thinking_cfg = genai_types.ThinkingConfig(thinking_budget=0)

    config = genai_types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.1,
        max_output_tokens=2048,
        response_mime_type="application/json",
        thinking_config=thinking_cfg,
    )

    t0 = time.time()
    try:
        resp = client.models.generate_content(
            model=model, contents=prompt, config=config,
        )
        raw = resp.text
    except Exception as e:
        print(f"    GEMINI ERROR: {e}")
        raw = None
    return raw, time.time() - t0


# ─── Metrics ─────────────────────────────────────────────────────────

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ─── Main ────────────────────────────────────────────────────────────

def run_test(model: str, verbose: bool):
    is_ollama = not model.startswith("gemini-")
    call_fn = call_ollama if is_ollama else call_gemini

    conn = open_graph_db()
    lite_gold, pro_gold = load_comparison_gold()

    # Build thread bundles
    threads: list[dict] = []  # each: {thread_id, events, triage_data, has_pro_gold}

    # 1. Singleton events from comparison set
    for eid in SINGLETON_EVENT_IDS:
        ev = fetch_singleton_event(conn, eid)
        if not ev:
            print(f"  WARN: event {eid} not found in graph.db, skipping")
            continue
        # Use comparison data triage if available, else fetch from DB
        td = None
        if eid in lite_gold:
            td = lite_gold[eid]
        else:
            td = fetch_triage_data(conn, eid)
        if not td:
            print(f"  WARN: no triage data for {eid}, skipping")
            continue
        threads.append({
            "thread_id": f"singleton:{eid}",
            "events": [ev],
            "triage_data": [td],
            "has_pro_gold": eid in pro_gold,
            "label": "singleton",
        })

    # 2. Multi-message threads
    for tid in THREAD_IDS:
        evts = fetch_thread_events(conn, tid)
        if not evts:
            print(f"  WARN: thread {tid} empty, skipping")
            continue
        # Limit to last 10 messages for very long threads
        if len(evts) > 10:
            evts = evts[-10:]
        triage = []
        for ev in evts:
            td = fetch_triage_data(conn, ev["id"])
            triage.append(td or {"score": 0, "domain": "", "specific_topics": [], "universal_spheres": []})
        threads.append({
            "thread_id": tid,
            "events": evts,
            "triage_data": triage,
            "has_pro_gold": any(ev["id"] in pro_gold for ev in evts),
            "label": f"thread ({len(evts)} msgs)",
        })

    print(f"\n{'='*90}")
    print(f"  Thread-Level Label Correction Test")
    print(f"{'='*90}")
    print(f"  Correction model: {model}")
    print(f"  Threads: {len(threads)} ({sum(1 for t in threads if t['label'] == 'singleton')} singletons, "
          f"{sum(1 for t in threads if t['label'] != 'singleton')} multi-msg)")
    print(f"  Total messages: {sum(len(t['events']) for t in threads)}")
    print(f"  Threads with Pro gold: {sum(1 for t in threads if t['has_pro_gold'])}")
    print()

    # Metrics accumulators (for events with Pro gold)
    before_topic_j = []
    before_sphere_j = []
    before_domain_match = 0
    after_topic_j = []
    after_sphere_j = []
    after_domain_match = 0
    gold_total = 0
    corrections_made = 0
    total_time = 0.0

    for tidx, thread in enumerate(threads):
        events = thread["events"]
        triage_data = thread["triage_data"]
        event_ids = [ev["id"] for ev in events]

        # Fetch person data
        persons_cache = fetch_persons_for_events(conn, event_ids)

        # Build synthesis-style context
        thread_text = build_thread_text(events, persons_cache)
        person_ctx = build_person_context(events, persons_cache)
        triage_ctx = build_triage_context(triage_data)
        group_ctx = build_group_context(events)
        age_ctx = build_thread_age_context(events)

        context_section = "\n\n".join(
            p for p in [person_ctx, triage_ctx, group_ctx, age_ctx] if p
        )

        per_msg_labels = build_per_message_labels(events, triage_data)

        prompt = CORRECTION_PROMPT.format(
            thread_text=thread_text,
            context_section=context_section,
            per_message_labels=per_msg_labels,
        )

        # Call correction model
        raw, elapsed = call_fn(model, prompt, CORRECTION_SYSTEM)
        total_time += elapsed

        # Parse response
        corrected_msgs = {}
        thread_notes = ""
        if raw:
            try:
                result = json.loads(raw)
                # Handle both {"messages": [...]} and bare [...]
                if isinstance(result, list):
                    msgs_list = result
                elif isinstance(result, dict):
                    msgs_list = result.get("messages", [])
                    thread_notes = result.get("thread_level_notes", "")
                else:
                    msgs_list = []
                for m in msgs_list:
                    if isinstance(m, dict):
                        mid = str(m.get("id", ""))
                        corrected_msgs[mid] = m
            except json.JSONDecodeError:
                pass

        # Print thread header
        meta0 = json.loads(events[0]["metadata"]) if events[0]["metadata"] else {}
        subj = meta0.get("subject", meta0.get("group_name", "")) or "(no subject)"
        print(f"  Thread {tidx+1}/{len(threads)}: {thread['label']:15s}  "
              f"tid={thread['thread_id'][:30]:30s}  ({elapsed:.1f}s)")
        print(f"    subject: {subj[:60]}")

        # Compare per-message results
        thread_corrections = 0
        for ev, td in zip(events, triage_data):
            eid = ev["id"]
            corr = corrected_msgs.get(eid)

            # Lite (before) labels
            lite_domain = (td.get("domain") or "").lower()
            lite_topics = set(t.lower().strip() for t in (td.get("specific_topics") or td.get("topics") or []))
            lite_spheres = set(s.lower().strip() for s in (td.get("universal_spheres") or []))

            # Corrected (after) labels
            if corr:
                corr_domain = (corr.get("domain") or "").lower()
                corr_topics = set(t.lower().strip() for t in (corr.get("specific_topics") or []))
                corr_spheres = set(t.lower().strip() for t in (corr.get("universal_spheres") or []))
                did_correct = corr.get("corrected", False)
            else:
                corr_domain = lite_domain
                corr_topics = lite_topics
                corr_spheres = lite_spheres
                did_correct = False

            if did_correct:
                thread_corrections += 1
                corrections_made += 1

            # Compare against Pro gold if available
            if eid in pro_gold:
                pro = pro_gold[eid]
                pro_domain = (pro.get("domain") or "").lower()
                pro_topics = set(t.lower().strip() for t in (pro.get("specific_topics") or []))
                pro_spheres = set(s.lower().strip() for s in (pro.get("universal_spheres") or []))
                gold_total += 1

                # Before
                before_topic_j.append(jaccard(pro_topics, lite_topics))
                before_sphere_j.append(jaccard(pro_spheres, lite_spheres))
                if lite_domain == pro_domain:
                    before_domain_match += 1

                # After
                after_topic_j.append(jaccard(pro_topics, corr_topics))
                after_sphere_j.append(jaccard(pro_spheres, corr_spheres))
                if corr_domain == pro_domain:
                    after_domain_match += 1

                if verbose or did_correct:
                    note = (corr.get("correction_note") or "") if corr else ""
                    print(f"    msg {eid[:16]}  [HAS PRO GOLD]")
                    print(f"      PRO:  domain={pro_domain:<12s} topics={sorted(pro_topics)}")
                    print(f"      LITE: domain={lite_domain:<12s} topics={sorted(lite_topics)}")
                    print(f"      CORR: domain={corr_domain:<12s} topics={sorted(corr_topics)}")
                    bj = jaccard(pro_topics, lite_topics)
                    aj = jaccard(pro_topics, corr_topics)
                    delta = aj - bj
                    marker = ">>>" if delta > 0 else ("<<<" if delta < 0 else "   ")
                    print(f"      topic_J: {bj:.0%} -> {aj:.0%}  {marker}  {note[:60]}")
            elif verbose or did_correct:
                note = (corr.get("correction_note") or "") if corr else ""
                if did_correct:
                    print(f"    msg {eid[:16]}  corrected:")
                    print(f"      LITE: domain={lite_domain:<12s} topics={sorted(lite_topics)}")
                    print(f"      CORR: domain={corr_domain:<12s} topics={sorted(corr_topics)}")
                    print(f"      note: {note[:80]}")

        if thread_notes and (verbose or thread_corrections > 0):
            print(f"    thread_notes: {thread_notes[:100]}")

        status = f"{thread_corrections} corrections" if thread_corrections else "no changes"
        print(f"    -> {status}")
        print()

    # ─── Summary ─────────────────────────────────────────────────────
    print(f"{'='*90}")
    print(f"  Results: Before vs After Correction (vs Pro gold)")
    print(f"{'='*90}")
    print()

    if gold_total > 0:
        b_tj = statistics.mean(before_topic_j) if before_topic_j else 0
        a_tj = statistics.mean(after_topic_j) if after_topic_j else 0
        b_sj = statistics.mean(before_sphere_j) if before_sphere_j else 0
        a_sj = statistics.mean(after_sphere_j) if after_sphere_j else 0
        b_dm = before_domain_match / gold_total
        a_dm = after_domain_match / gold_total

        def fmt_delta(before: float, after: float) -> str:
            d = after - before
            return f"{'+' if d >= 0 else ''}{d:.0%}"

        print(f"  {'Metric':<30} {'Before (Lite)':>15} {'After (corrected)':>20} {'Delta':>10}")
        print(f"  {'-'*30} {'-'*15} {'-'*20} {'-'*10}")
        print(f"  {'Topic Jaccard':<30} {b_tj:>15.0%} {a_tj:>20.0%} {fmt_delta(b_tj, a_tj):>10}")
        print(f"  {'Sphere Jaccard':<30} {b_sj:>15.0%} {a_sj:>20.0%} {fmt_delta(b_sj, a_sj):>10}")
        print(f"  {'Domain agreement':<30} {b_dm:>15.0%} {a_dm:>20.0%} {fmt_delta(b_dm, a_dm):>10}")
        print()
        print(f"  Gold comparisons: {gold_total} messages")
    else:
        print("  No Pro gold labels available for comparison.")
        print()

    total_msgs = sum(len(t["events"]) for t in threads)
    print(f"  Total corrections: {corrections_made}/{total_msgs} messages")
    print(f"  Total time: {total_time:.0f}s ({total_time/len(threads):.1f}s/thread)")
    print()

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test thread-level label correction")
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    run_test(args.model, args.verbose)
