"""Knowledge graph query tools for agents.

These are the tools that agents use to query their owner's private
knowledge graph. Each tool returns structured data suitable for
the agent's reasoning loop.

The tools operate on the existing Alteris store (graph.db) — they
read beliefs, events, persons, and claims to build context for
investor outreach, meeting prep, and technical briefings.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from alteris.store import LayeredGraphStore

logger = logging.getLogger(__name__)


class KGTools:
    """Knowledge graph query tools available to both agents.

    Each method is designed to be callable as a Claude tool.
    Methods return dicts that can be serialized to tool results.
    """

    def __init__(self, store: LayeredGraphStore):
        self.store = store

    # ── Relationship queries ──────────────────────────────────

    def find_person(self, name_or_email: str) -> list[dict]:
        """Find a person in the knowledge graph by name or email.

        Returns matching persons with their identifiers and interaction stats.
        """
        conn = self.store.conn
        query = name_or_email.lower()

        # Search by identifier (email, phone, etc.)
        rows = conn.execute("""
            SELECT DISTINCT p.person_id, p.canonical_name, p.is_user,
                   i.identifier, i.identifier_type
            FROM persons p
            JOIN person_identifiers i ON i.person_id = p.person_id
            WHERE LOWER(i.identifier) LIKE ?
               OR LOWER(p.canonical_name) LIKE ?
            LIMIT 20
        """, (f"%{query}%", f"%{query}%")).fetchall()

        persons = {}
        for r in rows:
            pid = r[0]
            if pid not in persons:
                persons[pid] = {
                    "person_id": pid,
                    "name": r[1],
                    "is_user": bool(r[2]),
                    "identifiers": [],
                }
            persons[pid]["identifiers"].append({
                "type": r[4],
                "value": r[3],
            })

        return list(persons.values())

    def person_context(self, person_id: str, days: int = 30) -> dict:
        """Get rich context about a person: recent interactions,
        shared events, communication patterns.
        """
        conn = self.store.conn
        cutoff = int(time.time()) - days * 86400

        # Recent events involving this person
        event_rows = conn.execute("""
            SELECT e.id, e.source, e.event_type, e.timestamp,
                   e.raw_content, e.metadata
            FROM events e
            JOIN person_events pe ON pe.event_id = e.id
            WHERE pe.person_id = ? AND e.timestamp > ?
            ORDER BY e.timestamp DESC
            LIMIT 20
        """, (person_id, cutoff)).fetchall()

        events = []
        for r in event_rows:
            meta = {}
            try:
                meta = json.loads(r[5]) if r[5] else {}
            except (json.JSONDecodeError, TypeError):
                pass
            events.append({
                "source": r[1],
                "type": r[2],
                "timestamp": r[3],
                "subject": meta.get("subject", ""),
                "snippet": (r[4] or "")[:200],
            })

        # Communication stats
        stats_row = conn.execute("""
            SELECT COUNT(*) as total,
                   COUNT(DISTINCT e.source) as sources,
                   MIN(e.timestamp) as first_seen,
                   MAX(e.timestamp) as last_seen
            FROM events e
            JOIN person_events pe ON pe.event_id = e.id
            WHERE pe.person_id = ?
        """, (person_id,)).fetchone()

        # Beliefs about this person
        beliefs = conn.execute("""
            SELECT id, belief_type, subject, summary, confidence
            FROM beliefs
            WHERE subject LIKE ? AND status = 'active'
            ORDER BY confidence DESC
            LIMIT 10
        """, (f"%{person_id}%",)).fetchall()

        return {
            "person_id": person_id,
            "recent_events": events,
            "total_interactions": stats_row[0] if stats_row else 0,
            "sources": stats_row[1] if stats_row else 0,
            "first_seen": stats_row[2] if stats_row else None,
            "last_seen": stats_row[3] if stats_row else None,
            "beliefs": [
                {"type": b[1], "subject": b[2], "summary": b[3], "confidence": b[4]}
                for b in (beliefs or [])
            ],
        }

    def find_warm_paths(self, target_name: str, target_firm: str) -> list[dict]:
        """Find warm introduction paths to a VC.

        Searches the knowledge graph for:
        1. Direct connections (have we emailed/messaged this person?)
        2. Mutual connections (who do we both know?)
        3. Firm connections (do we know anyone at the firm?)
        """
        conn = self.store.conn
        paths = []

        # 1. Direct connection — search by name or firm in events
        direct = conn.execute("""
            SELECT DISTINCT e.source, e.timestamp,
                   json_extract(e.metadata, '$.subject') as subject
            FROM events e
            WHERE (e.raw_content LIKE ? OR e.raw_content LIKE ?
                   OR json_extract(e.metadata, '$.subject') LIKE ?
                   OR json_extract(e.metadata, '$.subject') LIKE ?)
            ORDER BY e.timestamp DESC
            LIMIT 5
        """, (
            f"%{target_name}%", f"%{target_firm}%",
            f"%{target_name}%", f"%{target_firm}%",
        )).fetchall()

        for r in direct:
            paths.append({
                "type": "direct",
                "source": r[0],
                "timestamp": r[1],
                "context": r[2] or "Direct interaction found",
                "strength": "strong",
            })

        # 2. Firm connections — search for anyone at the firm
        firm_lower = target_firm.lower()
        firm_hits = conn.execute("""
            SELECT DISTINCT p.person_id, p.canonical_name, i.identifier
            FROM persons p
            JOIN person_identifiers i ON i.person_id = p.person_id
            WHERE LOWER(i.identifier) LIKE ?
            LIMIT 10
        """, (f"%{firm_lower}%",)).fetchall()

        for r in firm_hits:
            paths.append({
                "type": "firm_connection",
                "person_id": r[0],
                "person_name": r[1],
                "identifier": r[2],
                "strength": "medium",
            })

        # 3. Search beliefs for mentions
        belief_hits = conn.execute("""
            SELECT id, subject, summary, confidence
            FROM beliefs
            WHERE (subject LIKE ? OR summary LIKE ?
                   OR subject LIKE ? OR summary LIKE ?)
              AND status = 'active'
            LIMIT 5
        """, (
            f"%{target_name}%", f"%{target_name}%",
            f"%{target_firm}%", f"%{target_firm}%",
        )).fetchall()

        for r in belief_hits:
            paths.append({
                "type": "belief_mention",
                "belief_id": r[0],
                "subject": r[1],
                "summary": r[2],
                "confidence": r[3],
                "strength": "weak",
            })

        return paths

    def whose_contact(self, name_or_email: str) -> dict:
        """Determine whose contact this person is (CTO or CEO).

        Looks at interaction patterns in the KG: who sent more emails
        to/from this person, whose calendar has more meetings with them.
        The co-founder with more interaction history "owns" this relationship.

        This is critical for intro-based routing: if Sid is the CTO's contact,
        then VCs introduced by Sid should be routed to the CTO regardless
        of the VC's focus preference.

        Returns:
            {
                'name': str,
                'person_id': str or None,
                'interaction_count': int,
                'sources': list[str],
                'is_user_contact': bool,  # True = in the graph owner's network
                'confidence': float,      # 0-1, how sure we are
            }
        """
        conn = self.store.conn
        query = name_or_email.lower()

        # Find the person
        rows = conn.execute("""
            SELECT DISTINCT p.person_id, p.canonical_name
            FROM persons p
            LEFT JOIN person_identifiers i ON i.person_id = p.person_id
            WHERE LOWER(p.canonical_name) LIKE ?
               OR LOWER(i.identifier) LIKE ?
            LIMIT 5
        """, (f"%{query}%", f"%{query}%")).fetchall()

        if not rows:
            return {
                "name": name_or_email,
                "person_id": None,
                "interaction_count": 0,
                "sources": [],
                "is_user_contact": False,
                "confidence": 0.0,
            }

        # Take the best match
        person_id = rows[0][0]
        person_name = rows[0][1]

        # Count interactions — events involving this person
        stats = conn.execute("""
            SELECT COUNT(*) as total,
                   COUNT(DISTINCT e.source) as sources,
                   GROUP_CONCAT(DISTINCT e.source) as source_list
            FROM events e
            JOIN person_events pe ON pe.event_id = e.id
            WHERE pe.person_id = ?
        """, (person_id,)).fetchone()

        total = stats[0] if stats else 0
        source_list = (stats[2] or "").split(",") if stats else []

        # Check for direct messages from user to this person
        user_sent = conn.execute("""
            SELECT COUNT(*)
            FROM events e
            JOIN person_events pe ON pe.event_id = e.id
            WHERE pe.person_id = ?
              AND json_extract(e.metadata, '$.is_from_me') = 1
        """, (person_id,)).fetchone()
        user_sent_count = user_sent[0] if user_sent else 0

        # Confidence based on interaction volume
        if total >= 10:
            confidence = 0.9
        elif total >= 5:
            confidence = 0.7
        elif total >= 1:
            confidence = 0.5
        else:
            confidence = 0.1

        return {
            "name": person_name,
            "person_id": person_id,
            "interaction_count": total,
            "user_sent_count": user_sent_count,
            "sources": source_list,
            "is_user_contact": total > 0,
            "confidence": confidence,
        }

    # ── Communication style ───────────────────────────────────

    def extract_voice_samples(self, n_samples: int = 10) -> list[dict]:
        """Extract samples of the user's email writing style.

        Finds sent emails to professional contacts and extracts
        opening lines, closing lines, and overall tone markers.
        """
        conn = self.store.conn

        # Get events where user is sender (is_from_me = true) from email
        rows = conn.execute("""
            SELECT e.raw_content, e.metadata, e.timestamp
            FROM events e
            WHERE e.source = 'mail'
              AND json_extract(e.metadata, '$.is_from_me') = 1
              AND LENGTH(e.raw_content) > 50
              AND LENGTH(e.raw_content) < 3000
            ORDER BY e.timestamp DESC
            LIMIT ?
        """, (n_samples * 3,)).fetchall()

        samples = []
        for r in rows:
            content = r[0] or ""
            meta = {}
            try:
                meta = json.loads(r[1]) if r[1] else {}
            except (json.JSONDecodeError, TypeError):
                pass

            # Extract structural elements
            lines = content.strip().split("\n")
            opening = lines[0] if lines else ""
            closing = lines[-1] if len(lines) > 1 else ""

            samples.append({
                "subject": meta.get("subject", ""),
                "opening": opening[:200],
                "closing": closing[:200],
                "length": len(content),
                "timestamp": r[2],
            })

            if len(samples) >= n_samples:
                break

        return samples

    # ── Technical status (CTO-specific) ───────────────────────

    def get_tech_beliefs(self) -> list[dict]:
        """Get beliefs about the product/technology.

        Searches for beliefs that mention technical concepts,
        architecture decisions, product features.
        """
        conn = self.store.conn

        tech_keywords = [
            "architecture", "pipeline", "model", "api", "deploy",
            "feature", "bug", "release", "test", "build", "code",
            "privacy", "security", "performance", "scale",
            "knowledge graph", "agent", "briefing", "loom", "alteris",
        ]

        all_results = []
        seen_ids: set[str] = set()

        for kw in tech_keywords:
            rows = conn.execute("""
                SELECT id, belief_type, subject, summary, confidence, data
                FROM beliefs
                WHERE (subject LIKE ? OR summary LIKE ?)
                  AND status = 'active'
                ORDER BY confidence DESC
                LIMIT 5
            """, (f"%{kw}%", f"%{kw}%")).fetchall()

            for r in rows:
                if r[0] not in seen_ids:
                    seen_ids.add(r[0])
                    all_results.append({
                        "id": r[0],
                        "type": r[1],
                        "subject": r[2],
                        "summary": r[3],
                        "confidence": r[4],
                    })

        return sorted(all_results, key=lambda x: x["confidence"], reverse=True)[:20]

    # ── Investor-related context (CEO-specific) ───────────────

    def get_investor_beliefs(self) -> list[dict]:
        """Get beliefs related to investors, fundraising, pitch."""
        conn = self.store.conn

        investor_keywords = [
            "investor", "vc", "funding", "pitch", "raise",
            "meeting", "term sheet", "valuation", "demo",
        ]

        all_results = []
        seen_ids: set[str] = set()

        for kw in investor_keywords:
            rows = conn.execute("""
                SELECT id, belief_type, subject, summary, confidence, data
                FROM beliefs
                WHERE (subject LIKE ? OR summary LIKE ?)
                  AND status = 'active'
                ORDER BY confidence DESC
                LIMIT 5
            """, (f"%{kw}%", f"%{kw}%")).fetchall()

            for r in rows:
                if r[0] not in seen_ids:
                    seen_ids.add(r[0])
                    all_results.append({
                        "id": r[0],
                        "type": r[1],
                        "subject": r[2],
                        "summary": r[3],
                        "confidence": r[4],
                    })

        return sorted(all_results, key=lambda x: x["confidence"], reverse=True)[:20]

    # ── Calendar context ──────────────────────────────────────

    def upcoming_meetings(self, days: int = 7) -> list[dict]:
        """Get upcoming calendar events that look like investor meetings."""
        conn = self.store.conn
        now = int(time.time())
        cutoff = now + days * 86400

        rows = conn.execute("""
            SELECT e.id, e.timestamp, e.raw_content, e.metadata, e.participants
            FROM events e
            WHERE e.source = 'calendar'
              AND e.timestamp BETWEEN ? AND ?
            ORDER BY e.timestamp ASC
        """, (now, cutoff)).fetchall()

        meetings = []
        for r in rows:
            meta = {}
            try:
                meta = json.loads(r[3]) if r[3] else {}
            except (json.JSONDecodeError, TypeError):
                pass

            meetings.append({
                "event_id": r[0],
                "timestamp": r[1],
                "title": meta.get("subject", ""),
                "location": meta.get("location", ""),
                "attendees": meta.get("attendees", []),
                "organizer": meta.get("organizer_name", ""),
            })

        return meetings

    # ── Commitment context ────────────────────────────────────

    def open_commitments(self, keyword: str = "") -> list[dict]:
        """Get open commitments, optionally filtered by keyword."""
        conn = self.store.conn

        if keyword:
            rows = conn.execute("""
                SELECT id, belief_type, subject, summary, confidence, data
                FROM beliefs
                WHERE belief_type = 'fact'
                  AND status = 'active'
                  AND (subject LIKE ? OR summary LIKE ?)
                ORDER BY confidence DESC
                LIMIT 20
            """, (f"%{keyword}%", f"%{keyword}%")).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, belief_type, subject, summary, confidence, data
                FROM beliefs
                WHERE belief_type = 'fact'
                  AND status = 'active'
                ORDER BY confidence DESC
                LIMIT 30
            """).fetchall()

        results = []
        for r in rows:
            data = {}
            try:
                data = json.loads(r[5]) if r[5] else {}
            except (json.JSONDecodeError, TypeError):
                pass

            results.append({
                "id": r[0],
                "subject": r[2],
                "summary": r[3],
                "confidence": r[4],
                "deadline": data.get("deadline", ""),
                "who": data.get("who", ""),
                "status": data.get("status", ""),
            })

        return results
