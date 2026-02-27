"""Autonomous VC discovery — scan the knowledge graph to find investors.

Instead of manually entering VC names, this module scans the user's
email, calendar, beliefs, and contacts to:
  1. Identify people who are likely VCs/investors
  2. Classify the relationship stage (active, cold, passed, etc.)
  3. Score fit for outreach
  4. Cross-reference with the workspace pipeline

This is the "scan my network" feature that powers the Outreach tab.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from alteris.agents.kg_tools import KGTools
from alteris.agents.workspace import SharedWorkspace
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)

# Email domain patterns that suggest VC/investor firms
# Substring matches against the full email address
VC_DOMAIN_KEYWORDS = [
    "capital", "ventures", "partners", "invest",
    "fund", "angel", "seed",
]

# TLDs and domain suffixes that are strong VC signals
VC_DOMAIN_SUFFIXES = [".vc", ".ventures"]

# Keywords in event content / beliefs that suggest investor interactions
INVESTOR_CONTENT_KEYWORDS = [
    "investor", "vc ", "venture", "fundrais", "pitch",
    "term sheet", "due diligence", "portfolio",
    "series a", "series b", "pre-seed", "seed round",
    "raise", "valuation", "cap table",
]

# Calendar event titles that suggest investor meetings
INVESTOR_MEETING_KEYWORDS = [
    "pitch", "investor", "vc ", "fund", "partner",
    "demo day", "office hours", "coffee chat",
    "intro call", "follow up",
]

# Top VC firms — comprehensive list for matching against email domains,
# person names, and event content. Covers top global firms + notable
# seed/early-stage. Searched against the DB to find any interactions.
KNOWN_VC_FIRMS = {
    # Mega / multi-stage
    "a16z", "andreessen horowitz", "sequoia", "benchmark",
    "greylock", "accel", "kleiner perkins", "lightspeed",
    "general catalyst", "index ventures", "bessemer",
    "first round", "founders fund", "khosla ventures",
    "union square", "spark capital", "redpoint",
    "matrix partners", "battery ventures", "insight partners",
    "nea", "ivp", "tiger global", "coatue",
    "thrive capital", "ribbit capital", "addition",
    "iconiq", "altimeter", "d1 capital", "dragoneer",
    "viking global", "lone pine", "durable capital",
    # Growth
    "general atlantic", "kkr", "warburg pincus", "silver lake",
    "permira", "vista equity", "hellman friedman",
    "francisco partners", "thoma bravo", "summit partners",
    # Early stage / seed
    "ycombinator", "y combinator", "techstars",
    "500 startups", "500 global", "bunch capital",
    "initialized", "floodgate", "true ventures",
    "lux capital", "craft ventures", "felicis",
    "antler", "mayfield", "golden ventures",
    "svquad", "sv quad", "precursor", "hustle fund",
    "root ventures", "bloom venture", "unshackled",
    "pioneer fund", "lemnos", "the fund",
    # Notable sector-specific
    "andreessen bio", "arch venture", "flagship pioneering",
    "lux bio", "playground global", "eclipse ventures",
    "congruent ventures", "energize ventures",
    "obvious ventures", "breakthrough energy",
    # International
    "softbank", "dst global", "naspers", "prosus",
    "tencent", "alibaba", "temasek", "gic",
    # Notable mid-size
    "upfront ventures", "greycroft", "menlo ventures",
    "norwest venture", "canaan partners", "shasta ventures",
    "emergence capital", "bain capital ventures",
    "sapphire ventures", "scale venture", "meritech",
    "institutional venture", "8vc", "social capital",
    "slow ventures", "cowboy ventures", "forerunner",
    "boldstart", "work-bench", "notation capital",
    "boxgroup", "homebrew", "lowercase capital",
    "sv angel", "naval ravikant",
    # Crossover / hedge
    "d1 capital", "lone pine", "viking", "marshall wace",
    # Specific to user's network
    "category vc", "categoryvc", "transform vc",
    "fuse vc", "elevata", "highwater",
    "maven partnership", "raju reddy",
}

# Known VC firm email domains (domain → firm display name)
# This catches firms that don't match keyword patterns
KNOWN_VC_DOMAINS = {
    "antler.co": "Antler",
    "mayfield.com": "Mayfield",
    "khoslaventures.com": "Khosla Ventures",
    "insightpartners.com": "Insight Partners",
    "mavenpartnership.com": "Maven Partnership",
    "svquad.com": "SV Quad",
    "bunchcapital.com": "Bunch Capital",
    "a16z.com": "a16z",
    "sequoiacap.com": "Sequoia",
    "benchmark.com": "Benchmark",
    "greylock.com": "Greylock",
    "accel.com": "Accel",
    "kpcb.com": "Kleiner Perkins",
    "lsvp.com": "Lightspeed",
    "generalcatalyst.com": "General Catalyst",
    "indexventures.com": "Index Ventures",
    "bvp.com": "Bessemer",
    "firstround.com": "First Round",
    "foundersfund.com": "Founders Fund",
    "floodgate.com": "Floodgate",
    "trueventures.com": "True Ventures",
    "luxcapital.com": "Lux Capital",
    "craftventures.com": "Craft Ventures",
    "felicis.com": "Felicis",
    "bitsaa.org": "BITSAA/SV Quad",
    "fuse.vc": "Fuse VC",
    "categoryvc.com": "Category VC",
    "transform.vc": "Transform VC",
    "elevata.vc": "Elevata VC",
    "highwater.vc": "Highwater VC",
    "golden.ventures": "Golden Ventures",
    "mahmoud.vc": "Mahmoud VC",
    "vreddy.com": "Vijay Reddy",
}

# Relationship stage thresholds (seconds)
ACTIVE_THRESHOLD = 14 * 86400       # 14 days
WARM_THRESHOLD = 30 * 86400         # 30 days
COOLING_THRESHOLD = 60 * 86400      # 60 days
COLD_THRESHOLD = 90 * 86400         # 90 days


def _extract_firm_from_email(email: str) -> str:
    """Extract a likely firm name from an email domain."""
    if "@" not in email:
        return ""
    domain = email.split("@")[1].lower()
    # Skip generic domains
    generic = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
               "icloud.com", "me.com", "aol.com", "proton.me", "protonmail.com",
               "fastmail.com", "googlegroups.com", "substack.com",
               "google.com", "apple.com"}
    if domain in generic:
        return ""
    # Check known VC domains first
    if domain in KNOWN_VC_DOMAINS:
        return KNOWN_VC_DOMAINS[domain]
    # Strip subdomains and TLDs
    parts = domain.split(".")
    name = parts[0] if len(parts) <= 2 else parts[-2]
    name = name.replace("-", " ").replace("_", " ")
    return name.title()


def _classify_stage(last_seen: int, first_seen: int, event_count: int,
                    has_calendar: bool, has_outbound: bool) -> str:
    """Classify relationship stage from interaction patterns."""
    now = int(time.time())
    recency = now - last_seen if last_seen else float("inf")
    tenure = now - first_seen if first_seen else 0

    if has_calendar and recency < ACTIVE_THRESHOLD:
        return "meeting_scheduled"
    if recency < ACTIVE_THRESHOLD:
        if event_count >= 3:
            return "in_progress"
        return "contacted"
    if recency < WARM_THRESHOLD:
        return "follow_up"
    if recency < COOLING_THRESHOLD:
        return "inactive"
    if recency < COLD_THRESHOLD:
        return "cold"
    return "dormant"


def _detect_stage_from_events(
    conn, person_id: str, name: str, firm: str, heuristic_stage: str,
) -> tuple[str, str]:
    """Override heuristic stage using event content signals.

    Scans raw events for pass/rejection language, ghosting patterns,
    and upcoming meetings. Returns (stage, reason) — reason is empty
    if no override.
    """
    now = int(time.time())
    name_lower = name.lower()
    firm_lower = (firm or "").lower()

    # --- Pass detection: scan inbound messages for rejection language ---
    PASS_KEYWORDS = [
        "pass on this", "not a fit", "not the right fit",
        "won't be pursuing", "not moving forward", "decided not to",
        "not the right time", "passing on", "we'll pass",
        "decline", "unfortunately we", "not able to move forward",
    ]

    inbound_rows = conn.execute("""
        SELECT e.raw_content, e.metadata, e.timestamp
        FROM events e
        JOIN person_events pe ON pe.event_id = e.id
        WHERE pe.person_id = ?
          AND json_extract(e.metadata, '$.is_from_me') != 1
          AND e.source IN ('mail', 'imessage', 'whatsapp')
        ORDER BY e.timestamp DESC
        LIMIT 50
    """, (person_id,)).fetchall()

    for row in inbound_rows:
        content = (row[0] or "").lower()
        for kw in PASS_KEYWORDS:
            if kw in content:
                idx = content.index(kw)
                snippet = content[max(0, idx - 40):idx + 60]
                return ("pass", f"'{kw}' found: ...{snippet.strip()}...")

    # --- Ghosted detection: outbound with no response ---
    outbound_count = conn.execute("""
        SELECT COUNT(*) FROM events e
        JOIN person_events pe ON pe.event_id = e.id
        WHERE pe.person_id = ?
          AND json_extract(e.metadata, '$.is_from_me') = 1
          AND e.source IN ('mail', 'imessage', 'whatsapp')
    """, (person_id,)).fetchone()[0]

    inbound_count = len(inbound_rows)

    if outbound_count >= 2 and inbound_count == 0:
        last_out = conn.execute("""
            SELECT MAX(e.timestamp) FROM events e
            JOIN person_events pe ON pe.event_id = e.id
            WHERE pe.person_id = ?
              AND json_extract(e.metadata, '$.is_from_me') = 1
        """, (person_id,)).fetchone()
        if last_out and last_out[0] and (now - last_out[0]) > 30 * 86400:
            days_ago = (now - last_out[0]) // 86400
            return ("ghosted", f"{outbound_count} outbound, 0 inbound, last sent {days_ago}d ago")

    # --- Future meeting detection ---
    search_terms = [t for t in [name_lower, firm_lower] if t]
    if search_terms:
        for term in search_terms:
            future_cal = conn.execute("""
                SELECT e.timestamp, json_extract(e.metadata, '$.subject')
                FROM events e
                WHERE e.source = 'calendar'
                  AND e.timestamp > ?
                  AND LOWER(e.raw_content) LIKE ?
                LIMIT 1
            """, (now, f"%{term}%")).fetchone()

            if future_cal:
                return ("meeting_scheduled", f"Upcoming: {future_cal[1] or 'calendar event'}")

    return (heuristic_stage, "")


def _vc_signal_score(signals: list[str]) -> float:
    """Score how likely someone is a VC based on accumulated signals."""
    score = 0.0
    for s in signals:
        if s.startswith("known_domain:"):
            score += 5.0
        elif s.startswith("email_domain:"):
            score += 3.0
        elif s.startswith("known_firm:"):
            score += 5.0
        elif s.startswith("belief:"):
            score += 2.0
        elif s.startswith("calendar:"):
            score += 2.5
        elif s.startswith("content:"):
            score += 1.0
        elif s.startswith("title:"):
            score += 4.0
    return min(score, 10.0)


def discover_vcs(
    store: LayeredGraphStore,
    workspace: SharedWorkspace,
    days: int = 180,
) -> dict[str, Any]:
    """Scan the knowledge graph to discover VCs and classify relationships.

    Returns:
        {
            "discovered": [
                {
                    "person_id": str,
                    "name": str,
                    "email": str,
                    "firm": str,
                    "stage": str,
                    "confidence": float,
                    "signals": [str],
                    "event_count": int,
                    "first_seen": int,
                    "last_seen": int,
                    "warm_paths": int,
                    "in_workspace": bool,
                    "workspace_stage": str,
                },
                ...
            ],
            "summary": {
                "total_discovered": int,
                "new": int,
                "already_tracked": int,
                "by_stage": {str: int},
            }
        }
    """
    conn = store.conn
    kg = KGTools(store)
    now = int(time.time())
    cutoff = now - days * 86400

    candidates: dict[str, dict] = {}

    # ── Step 1: Find VC candidates from multiple signals ──────────

    # 1a. Email domain keyword scan
    for kw in VC_DOMAIN_KEYWORDS:
        rows = conn.execute("""
            SELECT DISTINCT p.person_id, p.canonical_name, i.identifier
            FROM persons p
            JOIN person_identifiers i ON i.person_id = p.person_id
            WHERE i.identifier_type = 'email'
              AND LOWER(i.identifier) LIKE ?
              AND p.is_user = 0
        """, (f"%{kw}%",)).fetchall()

        for r in rows:
            pid = r[0]
            if pid not in candidates:
                candidates[pid] = {
                    "person_id": pid,
                    "name": r[1] or "",
                    "emails": [],
                    "firm": "",
                    "signals": [],
                }
            email = r[2]
            if email not in candidates[pid]["emails"]:
                candidates[pid]["emails"].append(email)
            candidates[pid]["signals"].append(f"email_domain:{kw}")
            firm = _extract_firm_from_email(email)
            if firm and not candidates[pid]["firm"]:
                candidates[pid]["firm"] = firm

    # 1a-ii. TLD scan — catch .vc and .ventures domains
    for suffix in VC_DOMAIN_SUFFIXES:
        rows = conn.execute("""
            SELECT DISTINCT p.person_id, p.canonical_name, i.identifier
            FROM persons p
            JOIN person_identifiers i ON i.person_id = p.person_id
            WHERE i.identifier_type = 'email'
              AND LOWER(i.identifier) LIKE ?
              AND p.is_user = 0
        """, (f"%{suffix}",)).fetchall()

        for r in rows:
            pid = r[0]
            if pid not in candidates:
                candidates[pid] = {
                    "person_id": pid,
                    "name": r[1] or "",
                    "emails": [],
                    "firm": "",
                    "signals": [],
                }
            email = r[2]
            if email not in candidates[pid]["emails"]:
                candidates[pid]["emails"].append(email)
            candidates[pid]["signals"].append(f"email_domain:{suffix}")
            firm = _extract_firm_from_email(email)
            if firm and not candidates[pid]["firm"]:
                candidates[pid]["firm"] = firm

    # 1a-iii. Known VC domain scan — catch specific firm domains
    for domain, firm_name in KNOWN_VC_DOMAINS.items():
        rows = conn.execute("""
            SELECT DISTINCT p.person_id, p.canonical_name, i.identifier
            FROM persons p
            JOIN person_identifiers i ON i.person_id = p.person_id
            WHERE i.identifier_type = 'email'
              AND LOWER(i.identifier) LIKE ?
              AND p.is_user = 0
        """, (f"%@{domain}",)).fetchall()

        for r in rows:
            pid = r[0]
            if pid not in candidates:
                candidates[pid] = {
                    "person_id": pid,
                    "name": r[1] or "",
                    "emails": [],
                    "firm": firm_name,
                    "signals": [],
                }
            email = r[2]
            if email not in candidates[pid]["emails"]:
                candidates[pid]["emails"].append(email)
            candidates[pid]["signals"].append(f"known_domain:{domain}")
            if not candidates[pid]["firm"]:
                candidates[pid]["firm"] = firm_name

    # 1b. Check for known VC firm names in person identifiers
    for firm_name in KNOWN_VC_FIRMS:
        rows = conn.execute("""
            SELECT DISTINCT p.person_id, p.canonical_name, i.identifier
            FROM persons p
            JOIN person_identifiers i ON i.person_id = p.person_id
            WHERE (LOWER(i.identifier) LIKE ? OR LOWER(p.canonical_name) LIKE ?)
              AND p.is_user = 0
            LIMIT 10
        """, (f"%{firm_name}%", f"%{firm_name}%")).fetchall()

        for r in rows:
            pid = r[0]
            if pid not in candidates:
                candidates[pid] = {
                    "person_id": pid,
                    "name": r[1] or "",
                    "emails": [],
                    "firm": firm_name.title(),
                    "signals": [],
                }
            candidates[pid]["signals"].append(f"known_firm:{firm_name}")

    # 1c. Belief scan — investor/VC keywords
    for kw in ["investor", "vc", "venture", "fund", "pitch", "raise", "portfolio"]:
        beliefs = conn.execute("""
            SELECT id, subject, summary, data, confidence
            FROM beliefs
            WHERE (LOWER(subject) LIKE ? OR LOWER(summary) LIKE ?)
              AND status = 'active'
            LIMIT 20
        """, (f"%{kw}%", f"%{kw}%")).fetchall()

        for b in beliefs:
            # Try to find a person referenced in the belief
            subject = b[1] or ""
            # Look for person IDs in the belief subject
            person_rows = conn.execute("""
                SELECT p.person_id, p.canonical_name
                FROM persons p
                WHERE LOWER(p.canonical_name) LIKE ?
                  AND p.is_user = 0
                LIMIT 5
            """, (f"%{subject.lower().split(':')[0].strip()[:30]}%",)).fetchall()

            for pr in person_rows:
                pid = pr[0]
                if pid not in candidates:
                    candidates[pid] = {
                        "person_id": pid,
                        "name": pr[1] or "",
                        "emails": [],
                        "firm": "",
                        "signals": [],
                    }
                candidates[pid]["signals"].append(f"belief:{kw}:{b[0][:8]}")

    # 1d. Calendar event scan — meetings with VC-related titles
    calendar_rows = conn.execute("""
        SELECT e.id, e.metadata, e.timestamp, e.participants
        FROM events e
        WHERE e.source = 'calendar'
          AND e.timestamp > ?
        ORDER BY e.timestamp DESC
    """, (cutoff,)).fetchall()

    for cr in calendar_rows:
        meta = {}
        try:
            meta = json.loads(cr[1]) if cr[1] else {}
        except (json.JSONDecodeError, TypeError):
            pass
        title = (meta.get("subject", "") or "").lower()
        # Check if title contains investor-related keywords
        if not any(kw in title for kw in INVESTOR_MEETING_KEYWORDS):
            continue

        # Find attendees
        attendees = meta.get("attendees", [])
        participants = []
        try:
            participants = json.loads(cr[3]) if cr[3] else []
        except (json.JSONDecodeError, TypeError):
            pass

        all_people = attendees + participants
        for person_ref in all_people:
            if isinstance(person_ref, str):
                # Try to find this person
                p_rows = conn.execute("""
                    SELECT DISTINCT p.person_id, p.canonical_name
                    FROM persons p
                    LEFT JOIN person_identifiers i ON i.person_id = p.person_id
                    WHERE (LOWER(p.canonical_name) LIKE ?
                           OR LOWER(i.identifier) LIKE ?)
                      AND p.is_user = 0
                    LIMIT 3
                """, (f"%{person_ref.lower()}%", f"%{person_ref.lower()}%")).fetchall()

                for pr in p_rows:
                    pid = pr[0]
                    if pid not in candidates:
                        candidates[pid] = {
                            "person_id": pid,
                            "name": pr[1] or "",
                            "emails": [],
                            "firm": "",
                            "signals": [],
                        }
                    candidates[pid]["signals"].append(f"calendar:{title[:40]}")

    # 1e. Scan event content for investor keywords in email subjects
    # Only match on subjects (not full body) to reduce false positives
    for kw in INVESTOR_CONTENT_KEYWORDS:
        content_rows = conn.execute("""
            SELECT DISTINCT pe.person_id, p.canonical_name
            FROM person_events pe
            JOIN persons p ON p.person_id = pe.person_id
            JOIN events e ON e.id = pe.event_id
            WHERE e.timestamp > ?
              AND p.is_user = 0
              AND e.source IN ('mail', 'imessage', 'whatsapp')
              AND LOWER(json_extract(e.metadata, '$.subject')) LIKE ?
            LIMIT 50
        """, (cutoff, f"%{kw}%")).fetchall()

        for r in content_rows:
            pid = r[0]
            if pid not in candidates:
                candidates[pid] = {
                    "person_id": pid,
                    "name": r[1] or "",
                    "emails": [],
                    "firm": "",
                    "signals": [],
                }
            candidates[pid]["signals"].append(f"content:{kw}")

    # ── Step 2: Enrich each candidate with interaction history ────

    for pid, c in candidates.items():
        # Get emails if not already found
        if not c["emails"]:
            email_rows = conn.execute("""
                SELECT identifier FROM person_identifiers
                WHERE person_id = ? AND identifier_type = 'email'
                LIMIT 3
            """, (pid,)).fetchall()
            c["emails"] = [r[0] for r in email_rows]
            if c["emails"] and not c["firm"]:
                c["firm"] = _extract_firm_from_email(c["emails"][0])

        # Interaction stats
        stats = conn.execute("""
            SELECT COUNT(*) as total,
                   MIN(e.timestamp) as first_seen,
                   MAX(e.timestamp) as last_seen,
                   COUNT(DISTINCT e.source) as sources,
                   SUM(CASE WHEN e.source = 'calendar' THEN 1 ELSE 0 END) as cal_count,
                   SUM(CASE WHEN json_extract(e.metadata, '$.is_from_me') = 1 THEN 1 ELSE 0 END) as outbound
            FROM events e
            JOIN person_events pe ON pe.event_id = e.id
            WHERE pe.person_id = ?
        """, (pid,)).fetchone()

        c["event_count"] = stats[0] if stats else 0
        c["first_seen"] = stats[1] if stats else 0
        c["last_seen"] = stats[2] if stats else 0
        c["sources"] = stats[3] if stats else 0
        has_calendar = (stats[4] or 0) > 0 if stats else False
        has_outbound = (stats[5] or 0) > 0 if stats else False

        # Classify stage
        c["stage"] = _classify_stage(
            c["last_seen"], c["first_seen"], c["event_count"],
            has_calendar, has_outbound,
        )
        c["stage_reason"] = ""

        # Override with event-based signals (pass, ghosted, meeting)
        override_stage, override_reason = _detect_stage_from_events(
            conn, pid, c["name"], c.get("firm", ""), c["stage"],
        )
        if override_stage != c["stage"]:
            c["stage"] = override_stage
            c["stage_reason"] = override_reason

        # VC confidence score
        c["confidence"] = _vc_signal_score(c["signals"])

        # Check warm paths
        warm = kg.find_warm_paths(c["name"], c["firm"])
        c["warm_path_count"] = len(warm)

    # ── Step 3: Cross-reference with workspace ────────────────────

    existing = {}
    for inv in workspace.list_investors():
        key = inv["name"].lower().strip()
        existing[key] = inv
        # Also index by firm
        firm_key = f"{inv['name'].lower().strip()}@{inv['firm'].lower().strip()}"
        existing[firm_key] = inv

    for pid, c in candidates.items():
        name_key = c["name"].lower().strip()
        firm_key = f"{name_key}@{c['firm'].lower().strip()}"

        matched = existing.get(firm_key) or existing.get(name_key)
        if matched:
            c["in_workspace"] = True
            c["workspace_stage"] = matched["stage"]
            c["workspace_id"] = matched["id"]
        else:
            c["in_workspace"] = False
            c["workspace_stage"] = ""
            c["workspace_id"] = ""

    # ── Step 4: Filter and rank ───────────────────────────────────

    # Filter out false positives
    # Domains that are clearly not VC firms
    noise_domains = {
        "substack", "googlegroups", "google", "apple", "amazon",
        "microsoft", "fastmail", "protonmail", "fidelity", "mail",
        "shareholderdocs", "linkedin", "facebook", "twitter",
        "customink", "gv",
    }

    # Auto-detect user's own company from is_from_me email senders.
    # Cofounders/team share a domain — exclude them from VC results.
    _generic_email_providers = {
        "gmail", "yahoo", "hotmail", "outlook", "icloud",
        "me", "mac", "live", "aol", "protonmail", "fastmail",
    }
    from_me_rows = conn.execute(
        """SELECT DISTINCT json_extract(e.metadata, '$.sender_email') as email
           FROM events e
           WHERE json_extract(e.metadata, '$.is_from_me') = 1
           AND json_extract(e.metadata, '$.sender_email') IS NOT NULL
           LIMIT 100""",
    ).fetchall()
    for r in from_me_rows:
        email = (r[0] or "").lower()
        if "@" in email:
            domain = email.split("@")[1].split(".")[0]
            if domain and len(domain) > 2 and domain not in _generic_email_providers:
                noise_domains.add(domain)

    # Also check calendar account names (user@loom.example.com -> loom)
    cal_rows = conn.execute(
        """SELECT DISTINCT json_extract(metadata, '$.calendar') as cal
           FROM events WHERE source = 'calendar'""",
    ).fetchall()
    for r in cal_rows:
        cal_name = (r[0] or "").lower()
        if "@" in cal_name:
            domain = cal_name.split("@")[1].split(".")[0]
            if domain and len(domain) > 2 and domain not in _generic_email_providers:
                noise_domains.add(domain)

    def _is_noise(c: dict) -> bool:
        name = (c.get("name") or "").lower()
        firm = (c.get("firm") or "").lower()
        # Skip unnamed entries
        if not name:
            return True
        # Skip if firm is a noise domain
        if firm in noise_domains:
            return True
        # Skip newsletters, mailing lists, and generic addresses
        emails = c.get("emails", [])
        if any("noreply" in e or "newsletter" in e or "substack" in e
               or "googlegroups" in e or "meet@" in e or "info@" in e
               or "events@" in e or "service@" in e
               or "customercare@" in e for e in emails):
            return True
        # Skip if only content signals (too noisy without domain/firm evidence)
        signals = c.get("signals", [])
        has_structural = any(
            s.startswith("email_domain:") or s.startswith("known_firm:")
            or s.startswith("calendar:") or s.startswith("title:")
            for s in signals
        )
        if not has_structural and c.get("confidence", 0) < 4.0:
            return True
        return False

    # Check for user person IDs to exclude (including unmerged duplicates)
    user_pids = set()
    user_names = set()
    user_rows = conn.execute(
        "SELECT person_id, canonical_name FROM persons WHERE is_user = 1"
    ).fetchall()
    for r in user_rows:
        user_pids.add(r[0])
        if r[1]:
            user_names.add(r[1].lower().strip())
    # Also exclude any person with the same canonical name as the user
    if user_names:
        for uname in list(user_names):
            dup_rows = conn.execute(
                "SELECT person_id FROM persons WHERE LOWER(canonical_name) = ?",
                (uname,),
            ).fetchall()
            for dr in dup_rows:
                user_pids.add(dr[0])

    # Require minimum confidence (at least 2 signals), exclude noise and user.
    # No-firm entries need strong domain/firm signals (confidence >= 5) to avoid
    # picking up friends/advisors who discuss VC topics.
    results = [
        c for c in candidates.values()
        if c["confidence"] >= 2.0
        and c["person_id"] not in user_pids
        and not _is_noise(c)
        and (c["firm"] or c["confidence"] >= 5.0)
    ]

    # Sort: higher confidence first, then by recency
    results.sort(key=lambda x: (-x["confidence"], -(x["last_seen"] or 0)))

    # Build summary
    by_stage: dict[str, int] = {}
    new_count = 0
    tracked_count = 0
    for r in results:
        stage = r["workspace_stage"] if r["in_workspace"] else r["stage"]
        by_stage[stage] = by_stage.get(stage, 0) + 1
        if r["in_workspace"]:
            tracked_count += 1
        else:
            new_count += 1

    # Format output
    discovered = []
    for r in results:
        discovered.append({
            "person_id": r["person_id"],
            "name": r["name"],
            "email": r["emails"][0] if r["emails"] else "",
            "firm": r["firm"],
            "stage": r["workspace_stage"] if r["in_workspace"] else r["stage"],
            "stage_reason": r.get("stage_reason", ""),
            "confidence": round(r["confidence"], 1),
            "signals": list(set(r["signals"]))[:5],
            "event_count": r["event_count"],
            "first_seen": r["first_seen"],
            "last_seen": r["last_seen"],
            "warm_paths": r["warm_path_count"],
            "in_workspace": r["in_workspace"],
            "workspace_id": r.get("workspace_id", ""),
        })

    return {
        "discovered": discovered,
        "summary": {
            "total_discovered": len(discovered),
            "new": new_count,
            "already_tracked": tracked_count,
            "by_stage": by_stage,
        },
    }


def add_discovered_to_pipeline(
    store: LayeredGraphStore,
    workspace: SharedWorkspace,
    person_ids: list[str],
    discover_results: list[dict] | None = None,
) -> list[dict]:
    """Add discovered VCs to the workspace pipeline for tracking.

    Takes a list of person_ids from discovery results and adds them
    to the investors table. Uses enriched data from discover_results
    when available (stage, firm, confidence, signals).

    If a VC already exists in the workspace, updates their stage if
    discovery detected a terminal state (pass/ghosted).
    """
    conn = store.conn
    kg = KGTools(store)
    added = []

    # Index discover results by person_id for quick lookup
    disc_by_pid: dict[str, dict] = {}
    if discover_results:
        for d in discover_results:
            disc_by_pid[d["person_id"]] = d

    # Cache existing investors (check once, not per iteration)
    existing_investors = workspace.list_investors()
    existing_by_name: dict[str, dict] = {}
    for inv in existing_investors:
        existing_by_name[inv["name"].lower().strip()] = inv

    for pid in person_ids:
        disc = disc_by_pid.get(pid, {})

        # Get person info
        person = conn.execute("""
            SELECT p.person_id, p.canonical_name
            FROM persons p WHERE p.person_id = ?
        """, (pid,)).fetchone()
        if not person:
            continue

        name = disc.get("name") or person[1] or ""
        name_key = name.lower().strip()

        # Use discover result data when available, fall back to DB lookup
        email = disc.get("email", "")
        if not email:
            email_row = conn.execute("""
                SELECT identifier FROM person_identifiers
                WHERE person_id = ? AND identifier_type = 'email'
                LIMIT 1
            """, (pid,)).fetchone()
            email = email_row[0] if email_row else ""

        firm = disc.get("firm", "")
        if not firm:
            firm = _extract_firm_from_email(email) if email else ""

        # Map discovery stages to pipeline stages
        raw_stage = disc.get("stage", "discovered")
        _STAGE_MAP = {
            "in_progress": "contacted",
            "inactive": "discovered",
            "cold": "discovered",
            "dormant": "discovered",
        }
        stage = _STAGE_MAP.get(raw_stage, raw_stage)
        stage_reason = disc.get("stage_reason", "")

        # Check if already in workspace
        existing = existing_by_name.get(name_key)
        if existing:
            # Update stage if discovery detected a terminal or active override
            if stage in ("pass", "ghosted") and existing["stage"] not in ("pass", "ghosted"):
                updates: dict[str, Any] = {"stage": stage}
                if stage == "pass" and stage_reason:
                    updates["pass_reason"] = stage_reason
                workspace.update_investor(existing["id"], **updates)
                added.append({
                    "investor_id": existing["id"],
                    "name": name,
                    "firm": firm,
                    "email": email,
                    "tier": existing["tier"],
                    "stage": stage,
                    "action": "updated",
                })
            continue

        # Determine tier from warm paths
        warm_count = disc.get("warm_paths", 0)
        if warm_count == 0:
            warm = kg.find_warm_paths(name, firm)
            warm_count = len(warm)
            tier = "tier3"
            if any(p["strength"] == "strong" for p in warm):
                tier = "tier1"
            elif any(p["strength"] == "medium" for p in warm):
                tier = "tier2"
        else:
            # Estimate tier from warm path count
            tier = "tier1" if warm_count >= 2 else ("tier2" if warm_count >= 1 else "tier3")

        inv_id = workspace.add_investor(
            name=name,
            firm=firm,
            email=email,
            tier=tier,
            owner="",
        )

        updates: dict[str, Any] = {"stage": stage}
        if stage == "pass" and stage_reason:
            updates["pass_reason"] = stage_reason
        if disc.get("last_seen"):
            updates["last_contact_at"] = disc["last_seen"]
        workspace.update_investor(inv_id, **updates)

        # Track in existing map to prevent duplicates within this batch
        existing_by_name[name_key] = {"id": inv_id, "name": name, "stage": stage, "tier": tier}

        added.append({
            "investor_id": inv_id,
            "name": name,
            "firm": firm,
            "email": email,
            "tier": tier,
            "stage": stage,
            "action": "added",
        })

    return added
