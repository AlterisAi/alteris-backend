"""Prospect discovery — mine the KG and web for new investor leads.

Two discovery modes:
  1. KG mining — scan the user's own data for investor signals they missed:
     - Intro offers in emails ("I can connect you with X at Y")
     - Contacts at VC firms (email domain matching)
     - Fundraising discussions mentioning specific firms/people
     - Calendar events with investor-adjacent keywords
  2. Web search — use Gemini grounding to find current, active investors:
     - Pre-seed/seed funds in relevant verticals
     - Recent fund announcements and thesis updates
     - Portfolio overlap with similar companies

Results are cross-referenced with KG warm paths and deduplicated
against the existing pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from alteris.agents.discover import (
    KNOWN_VC_DOMAINS,
    KNOWN_VC_FIRMS,
    VC_DOMAIN_KEYWORDS,
    VC_DOMAIN_SUFFIXES,
    _extract_firm_from_email,
)
from alteris.agents.kg_tools import KGTools
from alteris.agents.workspace import SharedWorkspace
from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)

# Patterns that indicate someone is offering an introduction
INTRO_PATTERNS = [
    r"(?:i can |i'll |let me |happy to )(?:introduce|connect|put) you (?:to|with) (.{5,80})",
    r"you should (?:talk to|meet|reach out to|connect with) (.{5,80})",
    r"have you (?:talked to|met|spoken with|reached out to) (.{5,80})",
    r"(?:i know |there's )(?:someone|a partner|a vc|an investor) at (.{5,60})",
    r"(?:intro to|introduction to|connecting you with) (.{5,80})",
    r"(?:cc'ing|copying|looping in) (.{5,60}) (?:from|at|who)",
]
_INTRO_RES = [re.compile(p, re.IGNORECASE) for p in INTRO_PATTERNS]

# Keywords that signal fundraising context
FUNDRAISING_CONTEXT = [
    "fundrais", "raising", "pre-seed", "seed round", "series a",
    "pitch", "investor", "term sheet", "valuation",
    "deck", "cap table", "lead investor",
]


def mine_kg_for_prospects(
    store: LayeredGraphStore,
    workspace: SharedWorkspace,
    days: int = 180,
) -> list[dict]:
    """Deep-mine the knowledge graph for investor leads.

    Returns a list of prospects with provenance showing exactly which
    KG data produced each lead.
    """
    conn = store.conn
    kg = KGTools(store)
    now = int(time.time())
    cutoff = now - days * 86400

    # Get existing pipeline names for dedup
    existing_names = {
        inv["name"].lower().strip()
        for inv in workspace.list_investors()
    }
    existing_firms = {
        inv["firm"].lower().strip()
        for inv in workspace.list_investors()
        if inv.get("firm")
    }

    prospects: dict[str, dict] = {}  # keyed by name_lower

    # ── 1. Intro offers: scan emails for introduction language ──

    intro_rows = conn.execute("""
        SELECT e.id, e.raw_content, e.metadata, e.timestamp, e.source
        FROM events e
        WHERE e.timestamp > ?
          AND e.source IN ('mail', 'imessage', 'whatsapp')
          AND e.raw_content IS NOT NULL
          AND LENGTH(e.raw_content) > 20
        ORDER BY e.timestamp DESC
        LIMIT 2000
    """, (cutoff,)).fetchall()

    for row in intro_rows:
        content = row[1] or ""
        meta = {}
        try:
            meta = json.loads(row[2]) if row[2] else {}
        except (json.JSONDecodeError, TypeError):
            pass

        for regex in _INTRO_RES:
            matches = regex.findall(content)
            for match in matches:
                # Clean up the match — extract name and firm
                match = match.strip().rstrip(".,;:!?)")
                # Skip if too short or too long
                if len(match) < 3 or len(match) > 80:
                    continue
                # Skip URLs — regex sometimes matches link text
                if "http" in match or "://" in match or ".com/" in match:
                    continue

                # Try to extract "Name at Firm" or "Name from Firm"
                name, firm = _parse_name_firm(match)
                if not name:
                    continue

                key = name.lower().strip()
                if key in existing_names:
                    continue

                if key not in prospects:
                    prospects[key] = {
                        "name": name,
                        "firm": firm,
                        "source_type": "kg_intro_offer",
                        "signals": [],
                        "provenance": [],
                        "warm_paths": [],
                        "confidence": 0.0,
                    }

                subject = meta.get("subject", "")
                sender = meta.get("sender_name", meta.get("sender_email", ""))
                prospects[key]["signals"].append(f"intro_offer:{sender}")
                prospects[key]["provenance"].append({
                    "type": "intro_offer",
                    "event_id": row[0],
                    "source": row[4],
                    "timestamp": row[3],
                    "subject": subject,
                    "from": sender,
                    "snippet": match[:100],
                })

    # ── 2. VC contacts not yet in pipeline ──
    # Find people with VC-firm email domains who aren't tracked

    for domain, firm_name in KNOWN_VC_DOMAINS.items():
        if firm_name.lower() in existing_firms:
            continue

        rows = conn.execute("""
            SELECT DISTINCT p.person_id, p.canonical_name, i.identifier
            FROM persons p
            JOIN person_identifiers i ON i.person_id = p.person_id
            WHERE i.identifier_type = 'email'
              AND LOWER(i.identifier) LIKE ?
              AND p.is_user = 0
        """, (f"%@{domain}",)).fetchall()

        for r in rows:
            name = r[1] or ""
            key = name.lower().strip()
            if not name or key in existing_names:
                continue

            if key not in prospects:
                prospects[key] = {
                    "name": name,
                    "firm": firm_name,
                    "email": r[2],
                    "person_id": r[0],
                    "source_type": "kg_vc_contact",
                    "signals": [],
                    "provenance": [],
                    "warm_paths": [],
                    "confidence": 0.0,
                }
            prospects[key]["signals"].append(f"vc_domain:{domain}")
            prospects[key]["provenance"].append({
                "type": "vc_contact",
                "person_id": r[0],
                "email": r[2],
                "firm": firm_name,
            })

    # ── 3. Fundraising discussions mentioning specific people/firms ──

    # Firm names that are common English words — require 2+ mentions to count
    AMBIGUOUS_FIRMS = {
        "addition", "the fund", "first round", "precursor",
        "accel", "nea", "gic", "craft", "lux", "pioneer fund",
        "root", "seed", "bloom", "emergence", "bold",
        "initialized", "obvious", "forerunner",
    }

    for kw in FUNDRAISING_CONTEXT:
        fund_rows = conn.execute("""
            SELECT e.id, e.raw_content, e.metadata, e.timestamp, e.source
            FROM events e
            WHERE e.timestamp > ?
              AND e.source IN ('mail', 'imessage', 'whatsapp', 'granola')
              AND LOWER(e.raw_content) LIKE ?
            ORDER BY e.timestamp DESC
            LIMIT 100
        """, (cutoff, f"%{kw}%")).fetchall()

        for row in fund_rows:
            content = row[1] or ""
            # Look for firm names mentioned in fundraising context
            content_lower = content.lower()
            for firm_name in KNOWN_VC_FIRMS:
                # Always require word boundaries to prevent substring matches
                if not re.search(r'\b' + re.escape(firm_name) + r'\b', content_lower):
                    continue

                # For ambiguous firms (common English words), require
                # the name appears capitalized as a proper noun
                if firm_name in AMBIGUOUS_FIRMS:
                    # Build title-case version to search in original text
                    title_form = firm_name.title()
                    if not re.search(r'\b' + re.escape(title_form) + r'\b', content):
                        continue

                # Check if this firm is already tracked
                if firm_name in existing_firms:
                    continue

                key = f"_firm_{firm_name}"
                if key not in prospects:
                    prospects[key] = {
                        "name": "",
                        "firm": firm_name.title(),
                        "source_type": "kg_fundraising_mention",
                        "signals": [],
                        "provenance": [],
                        "warm_paths": [],
                        "confidence": 0.0,
                    }
                prospects[key]["signals"].append(f"fundraising_mention:{kw}")
                # Track provenance (keep all for distinct-event counting)
                if len(prospects[key]["provenance"]) < 10:
                    meta = {}
                    try:
                        meta = json.loads(row[2]) if row[2] else {}
                    except (json.JSONDecodeError, TypeError):
                        pass
                    idx = content_lower.index(firm_name)
                    snippet = content[max(0, idx - 30):idx + 50]
                    prospects[key]["provenance"].append({
                        "type": "fundraising_mention",
                        "event_id": row[0],
                        "source": row[4],
                        "timestamp": row[3],
                        "subject": meta.get("subject", ""),
                        "snippet": snippet.strip(),
                    })

    # Filter out ambiguous firm names unless they appear in multiple distinct events
    # (common English words like "addition", "the fund", "precursor")
    for key in list(prospects.keys()):
        if not key.startswith("_firm_"):
            continue
        firm_name = key[6:]  # strip "_firm_" prefix
        if firm_name in AMBIGUOUS_FIRMS:
            distinct_events = {
                p.get("event_id") for p in prospects[key].get("provenance", [])
                if p.get("event_id")
            }
            if len(distinct_events) < 3:
                del prospects[key]

    # ── 4. Calendar events with investor keywords + untracked people ──

    # Build set of user emails to exclude
    user_emails = set()
    user_rows = conn.execute(
        "SELECT person_id FROM persons WHERE is_user = 1"
    ).fetchall()
    for ur in user_rows:
        id_rows = conn.execute(
            "SELECT identifier FROM person_identifiers WHERE person_id = ? AND identifier_type = 'email'",
            (ur[0],),
        ).fetchall()
        for ir in id_rows:
            user_emails.add(ir[0].lower())
    # Also get from_me sender emails
    from_me_rows = conn.execute("""
        SELECT DISTINCT json_extract(e.metadata, '$.sender_email')
        FROM events e WHERE json_extract(e.metadata, '$.is_from_me') = 1
        AND json_extract(e.metadata, '$.sender_email') IS NOT NULL LIMIT 20
    """).fetchall()
    for r in from_me_rows:
        if r[0]:
            user_emails.add(r[0].lower())

    cal_rows = conn.execute("""
        SELECT e.id, e.raw_content, e.metadata, e.timestamp, e.participants
        FROM events e
        WHERE e.source = 'calendar'
          AND e.timestamp > ?
        ORDER BY e.timestamp DESC
    """, (cutoff,)).fetchall()

    investor_cal_kw = [
        "investor", "vc", "pitch", "fund", "partner meeting",
        "intro call", "coffee chat", "office hours",
    ]
    for row in cal_rows:
        meta = {}
        try:
            meta = json.loads(row[2]) if row[2] else {}
        except (json.JSONDecodeError, TypeError):
            pass
        title = (meta.get("subject", "") or "").lower()
        if not any(kw in title for kw in investor_cal_kw):
            continue

        # Extract attendee names/emails
        attendees = meta.get("attendees", [])
        participants = []
        try:
            participants = json.loads(row[4]) if row[4] else []
        except (json.JSONDecodeError, TypeError):
            pass

        for person in attendees + participants:
            if not isinstance(person, str):
                continue
            person_clean = person.strip()

            # Skip user's own emails
            if person_clean.lower() in user_emails:
                continue
            # Skip generic email-looking entries — resolve to names
            if "@" in person_clean:
                # Try to resolve email to a person name
                name_row = conn.execute("""
                    SELECT p.canonical_name FROM persons p
                    JOIN person_identifiers i ON i.person_id = p.person_id
                    WHERE LOWER(i.identifier) = ? AND p.is_user = 0
                    LIMIT 1
                """, (person_clean.lower(),)).fetchone()
                if name_row and name_row[0]:
                    person_clean = name_row[0]
                else:
                    continue  # Skip unresolvable emails

            key = person_clean.lower()
            if not key or key in existing_names or len(key) < 3:
                continue

            if key not in prospects:
                prospects[key] = {
                    "name": person_clean,
                    "firm": "",
                    "source_type": "kg_investor_meeting",
                    "signals": [],
                    "provenance": [],
                    "warm_paths": [],
                    "confidence": 0.0,
                }
            prospects[key]["signals"].append(f"investor_calendar:{title[:40]}")
            if len(prospects[key]["provenance"]) < 2:
                prospects[key]["provenance"].append({
                    "type": "investor_meeting",
                    "event_id": row[0],
                    "timestamp": row[3],
                    "title": title,
                })

    # ── 5. Score and enrich each prospect ──

    scored = []
    for key, p in prospects.items():
        # Deduplicate signals
        p["signals"] = list(set(p["signals"]))

        # Score by signal strength
        score = 0.0
        for s in p["signals"]:
            if s.startswith("intro_offer:"):
                score += 5.0  # Someone offered to introduce — gold
            elif s.startswith("vc_domain:"):
                score += 3.0  # Known VC firm contact
            elif s.startswith("fundraising_mention:"):
                score += 1.0  # Mentioned in fundraising context
            elif s.startswith("investor_calendar:"):
                score += 2.0  # Calendar event with investor keywords

        p["confidence"] = min(score, 10.0)

        # Skip low-confidence or unnamed entries
        if score < 2.0:
            continue
        if not p["name"] and not p["firm"]:
            continue

        # Find warm paths if we have enough info
        name = p.get("name", "")
        firm = p.get("firm", "")
        if name or firm:
            warm = kg.find_warm_paths(name or firm, firm or name)
            p["warm_paths"] = warm[:5]

        scored.append(p)

    # Sort by confidence
    scored.sort(key=lambda x: -x["confidence"])

    return scored


def web_search_prospects(
    llm_client: Any,
    company_description: str,
    verticals: list[str],
    stage: str = "pre-seed",
    existing_firms: set[str] | None = None,
    model: str = "",
) -> list[dict]:
    """Search the web for current, active investors.

    Uses Gemini's web search grounding to find up-to-date VC info
    that Claude's training data may not have.

    Args:
        llm_client: Gemini client with web_search capability
        company_description: What the company does (for thesis matching)
        verticals: Target verticals (e.g. ["AI infrastructure", "personal AI"])
        stage: Investment stage to target
        existing_firms: Firms already in pipeline (for dedup)
        model: Gemini model to use

    Returns:
        List of prospect dicts with web provenance
    """
    existing = existing_firms or set()
    prospects = []

    # Build search queries from verticals
    queries = []
    for vertical in verticals[:3]:
        queries.append(f"{stage} {vertical} investors 2025 2026 active")
    queries.append(f"new VC funds {stage} seed 2025 2026 launched")
    queries.append(f"investors in {' '.join(verticals[:2])} startups recently funded")

    for query in queries:
        web_result = None
        if hasattr(llm_client, "web_search"):
            web_result = llm_client.web_search(query, model=model)

        if not web_result:
            continue

        # Ask Gemini to extract structured investor data from search results
        extract_prompt = f"""From these web search results, extract active VCs/investors.
I'm building: {company_description}
Stage: {stage}

Search results:
{web_result}

Already in pipeline (SKIP these): {', '.join(list(existing)[:20])}

Extract as JSON array. For each investor found:
{{
    "name": "partner/investor name",
    "firm": "fund name",
    "thesis": "their investment focus in 1-2 sentences",
    "why_relevant": "why they'd be interested in our company",
    "source_url": "URL where you found this info",
    "recent_activity": "any recent investments or fund announcements"
}}

Return ONLY valid JSON array. If no relevant investors found, return [].
"""

        response = llm_client.generate(
            extract_prompt,
            system="Extract structured investor data from web search results. Be precise about names and firms.",
            model=model,
            format_json=True,
            temperature=0.1,
        )

        if not response:
            continue

        try:
            results = json.loads(response)
            if not isinstance(results, list):
                continue
        except json.JSONDecodeError:
            continue

        for r in results:
            name = (r.get("name") or "").strip()
            firm = (r.get("firm") or "").strip()
            if not name or not firm:
                continue
            if firm.lower() in existing:
                continue

            prospects.append({
                "name": name,
                "firm": firm,
                "source_type": "web_search",
                "signals": [f"web:{query[:40]}"],
                "provenance": [{
                    "type": "web_search",
                    "query": query,
                    "thesis": r.get("thesis", ""),
                    "why_relevant": r.get("why_relevant", ""),
                    "source_url": r.get("source_url", ""),
                    "recent_activity": r.get("recent_activity", ""),
                }],
                "warm_paths": [],
                "confidence": 3.0,  # Base score for web-found prospects
            })
            existing.add(firm.lower())

    return prospects


def discover_prospects(
    store: LayeredGraphStore,
    workspace: SharedWorkspace,
    llm_client: Any = None,
    company_description: str = "",
    verticals: list[str] | None = None,
    stage: str = "pre-seed",
    days: int = 180,
) -> dict[str, Any]:
    """Combined prospect discovery: KG mining + optional web search.

    Returns:
        {
            "kg_prospects": [...],       # From knowledge graph mining
            "web_prospects": [...],      # From web search (if llm_client provided)
            "combined": [...],           # Merged and ranked
            "summary": {
                "kg_found": int,
                "web_found": int,
                "total": int,
                "with_warm_paths": int,
                "with_intros": int,
            }
        }
    """
    # Mine the KG
    kg_prospects = mine_kg_for_prospects(store, workspace, days=days)

    # Web search (optional — requires Gemini client)
    web_prospects = []
    if llm_client and verticals:
        existing_firms = {
            inv["firm"].lower().strip()
            for inv in workspace.list_investors()
            if inv.get("firm")
        }
        # Also exclude KG-found firms
        for p in kg_prospects:
            if p.get("firm"):
                existing_firms.add(p["firm"].lower())

        web_prospects = web_search_prospects(
            llm_client,
            company_description=company_description or "Local-first AI that builds a knowledge graph from your digital life",
            verticals=verticals or ["AI infrastructure", "personal AI", "privacy-first"],
            stage=stage,
            existing_firms=existing_firms,
        )

    # Merge: KG prospects first (higher trust), then web
    combined = []
    seen = set()
    for p in kg_prospects + web_prospects:
        key = (p.get("name", "").lower(), p.get("firm", "").lower())
        if key in seen:
            continue
        seen.add(key)
        combined.append(p)

    combined.sort(key=lambda x: -x["confidence"])

    # Summary stats
    with_warm = sum(1 for p in combined if p.get("warm_paths"))
    with_intros = sum(
        1 for p in combined
        if any(s.startswith("intro_offer:") for s in p.get("signals", []))
    )

    return {
        "kg_prospects": kg_prospects,
        "web_prospects": web_prospects,
        "combined": combined,
        "summary": {
            "kg_found": len(kg_prospects),
            "web_found": len(web_prospects),
            "total": len(combined),
            "with_warm_paths": with_warm,
            "with_intros": with_intros,
        },
    }


def _parse_name_firm(text: str) -> tuple[str, str]:
    """Try to extract a person name and firm from a matched intro snippet.

    Handles patterns like:
        "Sarah at Benchmark"
        "John Smith from Sequoia"
        "the team at a16z"
    """
    # "X at/from Y" pattern
    for sep in [" at ", " from ", " who's at ", " who is at ", " over at "]:
        if sep in text.lower():
            parts = text.split(sep, 1) if sep == " at " else re.split(sep, text, 1, re.IGNORECASE)
            if len(parts) == 2:
                name = parts[0].strip().strip('"\'')
                firm = parts[1].strip().strip('"\'.,;:!?)')
                # Skip generic subjects
                if name.lower() in ("someone", "a friend", "people", "them", "him", "her"):
                    return ("", firm)
                return (name, firm)

    # No separator — check if it's a known firm name
    text_lower = text.lower().strip()
    for firm in KNOWN_VC_FIRMS:
        if firm in text_lower:
            # The text might be just the firm name or "person at firm"
            return ("", firm.title())

    # Might just be a person name
    # Only accept if it looks like a name (2+ words, title case)
    words = text.strip().split()
    if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w[0].isalpha()):
        return (text.strip(), "")

    return ("", "")
