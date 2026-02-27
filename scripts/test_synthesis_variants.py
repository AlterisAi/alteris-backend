#!/usr/bin/env python3
"""3-way comparison: current synthesis prompt vs thinking-off vs augmented+thinking-off.

Tests whether adding label correction to the synthesis prompt degrades
commitment extraction quality. Uses the actual synthesis system prompt
and schema from beliefs.py.

Variants:
  A: Current prompt, thinking=low  (production baseline)
  B: Current prompt, thinking=off  (speed test)
  C: Augmented prompt with label correction block, thinking=off

Usage:
    python scripts/test_synthesis_variants.py
    python scripts/test_synthesis_variants.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types as genai_types

sys.path.insert(0, "src")

from alteris.prompts.beliefs import SYNTHESIS_SYSTEM, SYNTHESIS_PROMPT_TEMPLATE
from alteris.prompts.triage import SPECIFIC_TOPICS, UNIVERSAL_SPHERES
from alteris.constants import CLOUD_DEEP_MODEL

# ─── Config ──────────────────────────────────────────────────────────

VALID_DOMAINS = [
    "work", "personal", "family", "financial", "health",
    "legal", "travel", "shopping", "automated",
]

# Fresh threads — not used in previous label correction test
TEST_THREADS = [
    # Mail threads with actionable content
    "220191",       # 2 msgs — "Time next week?"
    "219803",       # 4 msgs — Alteris AI <> Gautam Krishnamurthi (GPV)
    "219725",       # 2 msgs — networking intro
    "219520",       # 2 msgs — Meta recruiter chat scheduling
    "220079",       # 3 msgs — FIRESIDE appointment confirmation
    "220505",       # 9 msgs — Follow-Up From Anthropic
    "219448",       # 2 msgs — Puffin Pediatrics message
    "219920",       # 3 msgs — OpenClaw video recommendation
]

MODEL = CLOUD_DEEP_MODEL  # gemini-3-flash-preview

# ─── Augmented prompt (adds label correction to synthesis) ───────────

LABEL_CORRECTION_BLOCK = """
LABEL CORRECTION TASK (in addition to commitment extraction):
The triage labels below were assigned by a fast model processing messages individually.
Now that you see the full thread context, review and correct them.

VALID DOMAINS: {domains}
UNIVERSAL SPHERES (1-2 per message): {spheres}
SPECIFIC TOPICS (0-3 per message, ONLY from this list): {topics}

Common fast-model mistakes:
- Labeling automated notifications as "work" or "personal" (should be "automated")
- Using vague topics when specific ones exist
- Missing topics that become clear from thread context
- Not distinguishing phatic messages (short "thanks", "ok") from substantive ones
""".format(
    domains=", ".join(VALID_DOMAINS),
    spheres=", ".join(UNIVERSAL_SPHERES),
    topics=", ".join(SPECIFIC_TOPICS),
)

AUGMENTED_PROMPT_SUFFIX = """
In addition to commitments, also return corrected triage labels for each message:
"label_corrections": [
  {
    "message_id": "msg_0",
    "domain": "...",
    "universal_spheres": ["..."],
    "specific_topics": ["..."],
    "corrected": true/false
  }
]
"""

# ─── Gemini client ───────────────────────────────────────────────────

def _load_api_key() -> str:
    cfg_path = Path.home() / ".alteris" / "config.json"
    with open(cfg_path) as f:
        return json.load(f)["gemini_api_key"]

_CLIENT: genai.Client | None = None

def _get_client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(api_key=_load_api_key())
    return _CLIENT


def call_gemini(
    prompt: str, system: str,
    thinking_budget: int = 0,
    schema=None,
    max_retries: int = 3,
) -> tuple[str | None, float]:
    """Call Gemini with retry/backoff for 429s and 503s."""
    client = _get_client()

    thinking_cfg = genai_types.ThinkingConfig(thinking_budget=thinking_budget)

    config = genai_types.GenerateContentConfig(
        system_instruction=system,
        temperature=0.1,
        max_output_tokens=4096,
        response_mime_type="application/json",
        thinking_config=thinking_cfg,
        response_schema=schema,
    )

    t0 = time.time()
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=prompt, config=config,
            )
            return resp.text, time.time() - t0
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = 30 * (attempt + 1)
                print(f"    *** 429 rate limited — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            elif "503" in err_str or "UNAVAILABLE" in err_str:
                wait = 15 * (attempt + 1)
                print(f"    *** 503 unavailable — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                print(f"    *** ERROR: {err_str[:120]}")
                return None, time.time() - t0
    print(f"    *** FAILED after {max_retries} retries")
    return None, time.time() - t0


# ─── Data loading ────────────────────────────────────────────────────

def open_graph_db() -> sqlite3.Connection:
    conn = sqlite3.connect(
        "file:///Users/aniruddha/.alteris/graph.db?mode=ro", uri=True,
    )
    conn.row_factory = sqlite3.Row
    return conn


def fetch_thread(conn, thread_id) -> list[dict]:
    rows = conn.execute("""
        SELECT id, source, timestamp, raw_content, metadata, participants
        FROM events
        WHERE json_extract(metadata, '$.thread_id') = ?
        ORDER BY timestamp ASC
    """, (thread_id,)).fetchall()
    return [dict(r) for r in rows]


def fetch_triage(conn, event_id) -> dict:
    row = conn.execute("""
        SELECT object FROM claims
        WHERE claim_type = 'triage' AND subject = ? AND superseded_by IS NULL
        LIMIT 1
    """, (event_id,)).fetchone()
    return json.loads(row["object"]) if row else {}


def fetch_persons(conn, event_ids) -> dict[str, list[dict]]:
    if not event_ids:
        return {}
    ph = ",".join("?" * len(event_ids))
    rows = conn.execute(f"""
        SELECT ep.event_id, ep.person_id, ep.role, p.canonical_name, p.is_user
        FROM event_persons ep
        JOIN persons p ON p.person_id = ep.person_id
        WHERE ep.event_id IN ({ph})
        ORDER BY ep.event_id, ep.role
    """, event_ids).fetchall()
    result = defaultdict(list)
    for r in rows:
        result[r["event_id"]].append({
            "person_id": r["person_id"],
            "name": r["canonical_name"] or r["person_id"][:12],
            "role": r["role"],
            "is_user": bool(r["is_user"]),
        })
    return dict(result)


# ─── Context builders (mirror beliefs.py) ────────────────────────────

def format_event(event, index, persons):
    meta = json.loads(event["metadata"]) if event["metadata"] else {}
    content = (event["raw_content"] or "")[:3000]
    dt = datetime.fromtimestamp(event["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    source = event["source"]
    is_from_me = meta.get("is_from_me", False)
    senders = [p for p in persons if p["role"] == "sender"]
    recips = [p for p in persons if p["role"] in ("recipient", "member") and not p["is_user"]]
    sender = senders[0]["name"] if senders else meta.get("sender", "Unknown")

    lines = [f"**[msg_{index}]**", f"## Message {index + 1}"]
    if source == "mail":
        lines.append(f"**From:** {'USER (you)' if is_from_me else sender}")
        if recips:
            lines.append(f"**To:** {', '.join(p['name'] for p in recips[:5])}")
        subj = meta.get("subject", "")
        if subj:
            lines.append(f"**Subject:** {subj}")
        lines.append(f"**Date:** {dt}")
        lines += ["", "### Body", content]
    elif source in ("imessage", "whatsapp"):
        if is_from_me:
            recip = recips[0]["name"] if recips else "contact"
            lines.append(f"**Message from:** USER (you) -> {recip}")
            lines.append("**Direction:** outbound")
        else:
            lines.append(f"**Message from:** {sender}")
            lines.append("**Direction:** inbound")
        gname = meta.get("group_name")
        if gname:
            lines.append(f"**Group:** {gname}")
        lines.append(f"**Date:** {dt}")
        lines += ["", "### Body", content]
    else:
        lines += [f"**Source:** {source}", f"**Date:** {dt}", "", content]
    return "\n".join(lines)


def build_thread_text(events, persons_cache):
    parts = []
    for i, ev in enumerate(events):
        persons = persons_cache.get(ev["id"], [])
        parts.append("---")
        parts.append(format_event(ev, i, persons))
    return "\n\n".join(parts)


def build_person_context(events, persons_cache):
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


def build_triage_context(triage_data):
    if not triage_data:
        return "TRIAGE SUMMARY: No triage data available."
    scores = [d.get("score", 0) for d in triage_data]
    domains = sorted(set(d.get("domain", "") for d in triage_data if d.get("domain")))
    topics = set()
    for d in triage_data:
        for t in (d.get("specific_topics") or d.get("topics") or []):
            topics.add(t)
    avg = sum(scores) / len(scores) if scores else 0
    lines = [f"TRIAGE SUMMARY: avg_score={avg:.2f}"]
    if domains:
        lines.append(f"  domains: {', '.join(domains)}")
    if topics:
        lines.append(f"  topics: {', '.join(sorted(topics)[:5])}")
    return "\n".join(lines)


def build_group_context(events):
    for ev in events:
        meta = json.loads(ev["metadata"]) if ev["metadata"] else {}
        if meta.get("is_group") or meta.get("group_name"):
            return f"GROUP METADATA: is_group=true, group_name=\"{meta.get('group_name', '')}\""
    return "GROUP METADATA: is_group=false (direct conversation)"


def build_thread_age(events):
    if not events:
        return ""
    now = int(time.time())
    timestamps = [e["timestamp"] for e in events]
    age = (now - max(timestamps)) / 86400
    span = (max(timestamps) - min(timestamps)) / 86400
    return f"THREAD AGE: last_activity={age:.0f} days ago, thread_span={span:.0f} days"


def build_per_message_triage(events, triage_data):
    """Per-message triage labels — used by augmented prompt only."""
    lines = ["PER-MESSAGE TRIAGE LABELS (from fast model):"]
    for i, (ev, td) in enumerate(zip(events, triage_data)):
        lines.append(f"  msg_{i}: domain={td.get('domain', '')}, "
                     f"spheres={json.dumps(td.get('universal_spheres', []))}, "
                     f"topics={json.dumps(td.get('specific_topics', td.get('topics', [])))}")
    return "\n".join(lines)


# ─── Schema builders ────────────────────────────────────────────────

def build_commitment_schema():
    """Current production schema for commitments."""
    return genai_types.Schema(
        type="OBJECT",
        properties={
            "commitments": genai_types.Schema(
                type="ARRAY",
                items=genai_types.Schema(
                    type="OBJECT",
                    properties={
                        "type": genai_types.Schema(type="STRING", enum=[
                            "inbound_request", "user_commitment", "deadline",
                            "waiting_on", "payment_due", "follow_up",
                        ]),
                        "action_type": genai_types.Schema(type="STRING", enum=[
                            "user_owes_action", "waiting_on_other",
                            "scheduling_conflict_or_setup", "passive_tracking_or_reminder",
                        ]),
                        "who": genai_types.Schema(type="STRING"),
                        "what": genai_types.Schema(type="STRING"),
                        "to_whom": genai_types.Schema(type="STRING", nullable=True),
                        "direction": genai_types.Schema(type="STRING", enum=[
                            "direct_ask", "group_ask", "self_directed", "ambiguous",
                        ]),
                        "deadline": genai_types.Schema(type="STRING", nullable=True),
                        "status": genai_types.Schema(type="STRING", enum=["open", "done", "cancelled"]),
                        "priority": genai_types.Schema(type="INTEGER"),
                        "confidence": genai_types.Schema(type="NUMBER"),
                        "staleness_signal": genai_types.Schema(type="STRING", enum=[
                            "none", "overdue_no_followup", "group_broadcast", "old_thread",
                        ]),
                        "provenance": genai_types.Schema(type="STRING", enum=[
                            "assigned_to_user", "user_said", "system_detected", "inferred_from_context",
                        ]),
                        "note": genai_types.Schema(type="STRING", nullable=True),
                        "source_message_id": genai_types.Schema(type="STRING", nullable=True),
                        "evidence_quote": genai_types.Schema(type="STRING", nullable=True),
                        "speech_act": genai_types.Schema(type="STRING", enum=[
                            "promise", "request", "decision", "assignment", "delegation", "inform",
                        ]),
                        "response_type": genai_types.Schema(type="STRING", enum=[
                            "acknowledged", "accepted", "no_response", "continued_discussion",
                        ]),
                        "has_named_actor": genai_types.Schema(type="BOOLEAN"),
                        "has_concrete_deliverable": genai_types.Schema(type="BOOLEAN"),
                        "has_temporal_constraint": genai_types.Schema(type="BOOLEAN"),
                        "is_response_to_request": genai_types.Schema(type="BOOLEAN"),
                    },
                    required=[
                        "type", "action_type", "who", "what", "direction",
                        "status", "priority", "confidence", "staleness_signal",
                        "provenance", "speech_act", "response_type",
                        "has_named_actor", "has_concrete_deliverable",
                        "has_temporal_constraint", "is_response_to_request",
                    ],
                ),
            ),
        },
        required=["commitments"],
    )


def build_augmented_schema():
    """Commitment schema + label_corrections array."""
    base = build_commitment_schema()
    base.properties["label_corrections"] = genai_types.Schema(
        type="ARRAY",
        items=genai_types.Schema(
            type="OBJECT",
            properties={
                "message_id": genai_types.Schema(type="STRING"),
                "domain": genai_types.Schema(type="STRING"),
                "universal_spheres": genai_types.Schema(
                    type="ARRAY", items=genai_types.Schema(type="STRING"),
                ),
                "specific_topics": genai_types.Schema(
                    type="ARRAY", items=genai_types.Schema(type="STRING"),
                ),
                "corrected": genai_types.Schema(type="BOOLEAN"),
            },
            required=["message_id", "domain", "universal_spheres", "specific_topics", "corrected"],
        ),
    )
    base.required.append("label_corrections")
    return base


# ─── Main ────────────────────────────────────────────────────────────

def run_comparison(verbose: bool):
    conn = open_graph_db()

    # Build thread data
    threads = []
    for tid in TEST_THREADS:
        events = fetch_thread(conn, tid)
        if not events:
            print(f"  WARN: thread {tid} empty, skipping")
            continue
        if len(events) > 12:
            events = events[-12:]
        triage_data = [fetch_triage(conn, ev["id"]) for ev in events]
        event_ids = [ev["id"] for ev in events]
        persons_cache = fetch_persons(conn, event_ids)

        # Build base prompt (matches _build_synthesis_prompt)
        thread_text = build_thread_text(events, persons_cache)
        person_ctx = build_person_context(events, persons_cache)
        triage_ctx = build_triage_context(triage_data)
        group_ctx = build_group_context(events)
        age_ctx = build_thread_age(events)
        context_section = "\n\n".join(p for p in [person_ctx, triage_ctx, group_ctx, age_ctx] if p)

        base_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            thread_text=thread_text,
            context_section=context_section,
            custom_fields_section="",
        )

        # Build augmented prompt (adds label correction task + per-message labels)
        per_msg = build_per_message_triage(events, triage_data)
        aug_context = "\n\n".join(p for p in [person_ctx, triage_ctx, per_msg, group_ctx, age_ctx] if p)
        aug_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            thread_text=thread_text,
            context_section=aug_context,
            custom_fields_section=AUGMENTED_PROMPT_SUFFIX,
        )

        meta0 = json.loads(events[0]["metadata"]) if events[0]["metadata"] else {}
        subject = meta0.get("subject", meta0.get("group_name", "")) or "(no subject)"

        threads.append({
            "thread_id": tid,
            "events": events,
            "triage_data": triage_data,
            "subject": subject[:55],
            "msg_count": len(events),
            "base_prompt": base_prompt,
            "aug_prompt": aug_prompt,
        })

    print(f"\n{'='*90}")
    print(f"  Synthesis Prompt Variant Comparison")
    print(f"{'='*90}")
    print(f"  Model: {MODEL}")
    print(f"  Threads: {len(threads)}")
    print(f"  Total messages: {sum(t['msg_count'] for t in threads)}")
    print()
    print(f"  Variants:")
    print(f"    A: Current prompt + thinking=low   (production baseline)")
    print(f"    B: Current prompt + thinking=off   (speed test)")
    print(f"    C: Augmented prompt + thinking=off (label correction added)")
    print()

    commitment_schema = build_commitment_schema()
    augmented_schema = build_augmented_schema()

    # Augmented system prompt
    aug_system = SYNTHESIS_SYSTEM + "\n\n" + LABEL_CORRECTION_BLOCK

    results = []
    for tidx, thread in enumerate(threads):
        print(f"  Thread {tidx+1}/{len(threads)}: {thread['subject'][:50]:50s} ({thread['msg_count']} msgs)")

        # Variant A: current + thinking=low (thinking_budget=-1 for "low" equivalent)
        # Use thinking_budget=1024 as an approximation of "low"
        raw_a, time_a = call_gemini(
            thread["base_prompt"], SYNTHESIS_SYSTEM,
            thinking_budget=1024, schema=commitment_schema,
        )

        # Variant B: current + thinking=off
        raw_b, time_b = call_gemini(
            thread["base_prompt"], SYNTHESIS_SYSTEM,
            thinking_budget=0, schema=commitment_schema,
        )

        # Variant C: augmented + thinking=off
        raw_c, time_c = call_gemini(
            thread["aug_prompt"], aug_system,
            thinking_budget=0, schema=augmented_schema,
        )

        # Parse results
        def parse_commitments(raw):
            if not raw:
                return [], raw
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    return obj.get("commitments", []), obj
                return [], obj
            except json.JSONDecodeError:
                return [], None

        commits_a, full_a = parse_commitments(raw_a)
        commits_b, full_b = parse_commitments(raw_b)
        commits_c, full_c = parse_commitments(raw_c)

        label_corrections = []
        if isinstance(full_c, dict):
            label_corrections = full_c.get("label_corrections", [])

        result = {
            "thread_id": thread["thread_id"],
            "subject": thread["subject"],
            "msg_count": thread["msg_count"],
            "A": {"commits": commits_a, "count": len(commits_a), "time": time_a},
            "B": {"commits": commits_b, "count": len(commits_b), "time": time_b},
            "C": {"commits": commits_c, "count": len(commits_c), "time": time_c,
                  "label_corrections": label_corrections},
        }
        results.append(result)

        # Print per-thread summary
        print(f"    A (think=low):  {len(commits_a):2d} commitments  {time_a:5.1f}s")
        print(f"    B (think=off):  {len(commits_b):2d} commitments  {time_b:5.1f}s")
        print(f"    C (augmented):  {len(commits_c):2d} commitments  {time_c:5.1f}s  "
              f"+ {sum(1 for lc in label_corrections if lc.get('corrected'))} label corrections")

        # Compare commitment content A vs C
        def commit_key(c):
            return (c.get("what", "")[:40].lower(), c.get("who", "").lower())

        keys_a = set(commit_key(c) for c in commits_a)
        keys_b = set(commit_key(c) for c in commits_b)
        keys_c = set(commit_key(c) for c in commits_c)

        # Check for dropped commitments (in A but not in C)
        dropped = keys_a - keys_c
        added = keys_c - keys_a
        if dropped:
            print(f"    !! C DROPPED vs A: {[d[0] for d in dropped]}")
        if added:
            print(f"    ++ C ADDED vs A: {[a[0] for a in added]}")

        if verbose:
            print(f"\n    --- A commitments ---")
            for c in commits_a:
                print(f"      [{c.get('type','?'):18s}] {c.get('what','')[:60]}  (conf={c.get('confidence',0):.1f})")
            print(f"    --- B commitments ---")
            for c in commits_b:
                print(f"      [{c.get('type','?'):18s}] {c.get('what','')[:60]}  (conf={c.get('confidence',0):.1f})")
            print(f"    --- C commitments ---")
            for c in commits_c:
                print(f"      [{c.get('type','?'):18s}] {c.get('what','')[:60]}  (conf={c.get('confidence',0):.1f})")
            if label_corrections:
                print(f"    --- C label corrections ---")
                for lc in label_corrections:
                    if lc.get("corrected"):
                        print(f"      {lc.get('message_id','?')}: domain={lc.get('domain','')} "
                              f"topics={lc.get('specific_topics',[])} spheres={lc.get('universal_spheres',[])}")

        print()

    # ─── Summary ─────────────────────────────────────────────────────
    print(f"{'='*90}")
    print(f"  Summary")
    print(f"{'='*90}")
    print()

    total_a = sum(r["A"]["count"] for r in results)
    total_b = sum(r["B"]["count"] for r in results)
    total_c = sum(r["C"]["count"] for r in results)
    time_a = sum(r["A"]["time"] for r in results)
    time_b = sum(r["B"]["time"] for r in results)
    time_c = sum(r["C"]["time"] for r in results)
    lc_total = sum(len(r["C"].get("label_corrections", [])) for r in results)
    lc_corrected = sum(
        sum(1 for lc in r["C"].get("label_corrections", []) if lc.get("corrected"))
        for r in results
    )

    print(f"  {'Variant':<35} {'Commits':>10} {'Time':>10} {'Avg/thread':>12}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*12}")
    print(f"  {'A: Current + thinking=low':<35} {total_a:>10d} {time_a:>9.0f}s {time_a/len(results):>11.1f}s")
    print(f"  {'B: Current + thinking=off':<35} {total_b:>10d} {time_b:>9.0f}s {time_b/len(results):>11.1f}s")
    print(f"  {'C: Augmented + thinking=off':<35} {total_c:>10d} {time_c:>9.0f}s {time_c/len(results):>11.1f}s")
    print()

    # Agreement analysis
    all_match = 0
    a_vs_b_match = 0
    a_vs_c_match = 0
    for r in results:
        def keys(commits):
            return set((c.get("what", "")[:40].lower(), c.get("who", "").lower()) for c in commits)
        ka, kb, kc = keys(r["A"]["commits"]), keys(r["B"]["commits"]), keys(r["C"]["commits"])
        if ka == kb:
            a_vs_b_match += 1
        if ka == kc:
            a_vs_c_match += 1
        if ka == kb == kc:
            all_match += 1

    n = len(results)
    print(f"  Commitment agreement (by thread):")
    print(f"    A == B (thinking effect):     {a_vs_b_match}/{n}")
    print(f"    A == C (augmentation effect):  {a_vs_c_match}/{n}")
    print(f"    All three agree:              {all_match}/{n}")
    print()
    print(f"  Label corrections (variant C): {lc_corrected}/{lc_total} messages corrected")
    print()

    # Save results
    output_path = "scripts/synthesis_variant_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")
    print()

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare synthesis prompt variants")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    run_comparison(args.verbose)
