"""Dossier builder — end-to-end VC intelligence pipeline.

Orchestrates multi-source research, stores Events + Claims with Admiralty
scoring, synthesizes into a FACT belief, and manages dossier lifecycle
(caching, staleness, supersession).

Paper trail: dossier belief → source_claims → events with raw web search text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from loom.agents.intel.collector import (
    IntelFragment,
    fragment_to_claim,
    fragment_to_event,
)
from loom.agents.intel.sources import (
    VCSourceType,
    admiralty_for_source,
    aggregate_confidence,
    compute_confidence,
)
from loom.agents.intel.web_research import research_vc_deep
from loom.constants import HASH_PREFIX_LEN, SECONDS_PER_DAY
from loom.models import Belief, BeliefStatus, BeliefType, EpistemicLevel
from loom.store import LayeredGraphStore

logger = logging.getLogger(__name__)

# Dossier cache TTL
VC_DOSSIER_EXPIRY_DAYS = 30
VC_KG_LOOKBACK_DAYS = 365


def _make_subject(name: str, firm: str) -> str:
    """Canonical subject key for VC dossier beliefs."""
    return f"vc:{name.lower().replace(' ', '_')}:{firm.lower().replace(' ', '_')}"


def _make_belief_id(subject: str, timestamp: int, claim_ids: list[str] | None = None) -> str:
    """Deterministic, content-addressable belief ID.

    Includes claim IDs so refreshes with different content produce different IDs.
    """
    claims_hash = hashlib.sha256(
        ",".join(sorted(claim_ids or [])).encode()
    ).hexdigest()[:8]
    raw = f"vc_dossier:{subject}:{timestamp}:{claims_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:HASH_PREFIX_LEN]


def build_dossier(
    store: LayeredGraphStore,
    llm_client: Any,
    name: str,
    firm: str,
    model: str = "",
    force_refresh: bool = False,
    include_kg: bool = True,
    max_workers: int = 3,
) -> dict:
    """Build or refresh a VC dossier through the full pipeline.

    Returns:
        {
            "belief_id": str,
            "dossier": dict,           # The dossier data structure
            "events_stored": int,
            "claims_stored": int,
            "is_refresh": bool,
            "is_cached": bool,
            "source_breakdown": dict,  # count per source type
        }
    """
    subject = _make_subject(name, firm)
    now = int(time.time())

    # Check for existing active dossier
    existing = _find_existing_dossier(store, subject)
    if existing and not force_refresh:
        age_days = (now - existing.updated_at) / SECONDS_PER_DAY
        if age_days < VC_DOSSIER_EXPIRY_DAYS:
            logger.info("Returning cached dossier for %s (%.1f days old)", subject, age_days)
            return {
                "belief_id": existing.id,
                "dossier": existing.data,
                "events_stored": 0,
                "claims_stored": 0,
                "is_refresh": False,
                "is_cached": True,
                "source_breakdown": existing.data.get("source_breakdown", {}),
            }

    # Gather intel from all sources
    fragments: list[IntelFragment] = []

    # Web research (6 parallel queries)
    web_fragments = research_vc_deep(
        llm_client, name, firm, model=model, max_workers=max_workers,
    )
    fragments.extend(web_fragments)

    # KG intel (direct interactions from user's data)
    if include_kg:
        kg_fragments = _collect_kg_intel(store, name, firm, subject, now)
        fragments.extend(kg_fragments)

    if not fragments:
        logger.warning("No intel gathered for %s (%s)", name, firm)
        return {
            "belief_id": "",
            "dossier": {},
            "events_stored": 0,
            "claims_stored": 0,
            "is_refresh": False,
            "is_cached": False,
            "source_breakdown": {},
        }

    # Store events and claims
    events_stored = 0
    claims_stored = 0
    claim_ids: list[str] = []
    claim_records: list[dict] = []

    for frag in fragments:
        event = fragment_to_event(frag)
        if store.put_event(event):
            events_stored += 1

        claim = fragment_to_claim(frag, event.id)
        if store.put_claim(claim):
            claims_stored += 1

        claim_ids.append(claim.id)
        claim_records.append({
            "claim_id": claim.id,
            "claim_type": frag.claim_type,
            "source_type": frag.source_type.value,
            "confidence": claim.confidence,
        })

    # Synthesize dossier from fragments
    dossier_data = _synthesize_dossier(name, firm, fragments, claim_records)

    # Compute aggregate Admiralty scores
    rel, cred, overall_conf = _aggregate_admiralty(fragments)

    # Create belief — use nanosecond-precision time for uniqueness across rapid refreshes
    belief_id = _make_belief_id(subject, time.time_ns(), claim_ids)
    belief = Belief(
        id=belief_id,
        belief_type=BeliefType.FACT,
        subject=subject,
        summary=f"VC dossier for {name} ({firm})",
        data=dossier_data,
        epistemic_level=EpistemicLevel.INFERENCE,
        source_reliability=rel,
        info_credibility=cred,
        confidence=overall_conf,
        source_claims=claim_ids,
        inference_chain=[
            f"Gathered {len(fragments)} fragments from {len(set(f.source_type for f in fragments))} source types",
            f"Stored {events_stored} events and {claims_stored} claims",
            f"Synthesized dossier with aggregate confidence {overall_conf:.3f}",
        ],
        status=BeliefStatus.ACTIVE,
        created_at=now,
        updated_at=now,
        expires_at=now + (VC_DOSSIER_EXPIRY_DAYS * SECONDS_PER_DAY),
    )
    store.put_belief(belief)

    # Handle supersession
    is_refresh = False
    if existing:
        store.supersede_belief(existing.id, belief_id)
        is_refresh = True
        logger.info("Superseded old dossier %s with %s", existing.id, belief_id)

    source_breakdown = {}
    for frag in fragments:
        key = frag.source_type.value
        source_breakdown[key] = source_breakdown.get(key, 0) + 1

    logger.info(
        "Built dossier for %s: %d fragments, %d events, %d claims, confidence=%.3f",
        subject, len(fragments), events_stored, claims_stored, overall_conf,
    )

    return {
        "belief_id": belief_id,
        "dossier": dossier_data,
        "events_stored": events_stored,
        "claims_stored": claims_stored,
        "is_refresh": is_refresh,
        "is_cached": False,
        "source_breakdown": source_breakdown,
    }


# ── Internal helpers ──────────────────────────────────────────

def _find_existing_dossier(store: LayeredGraphStore, subject: str) -> Belief | None:
    """Find the most recent active dossier belief for a VC."""
    beliefs = store.get_beliefs(subject=subject, belief_type="fact", status="active")
    for b in beliefs:
        if b.data.get("assertion_type") == "vc_dossier":
            return b
    return None


def _collect_kg_intel(
    store: LayeredGraphStore,
    name: str,
    firm: str,
    subject: str,
    now: int,
) -> list[IntelFragment]:
    """Scan the knowledge graph for direct interactions with this VC."""
    fragments: list[IntelFragment] = []
    lookback = now - (VC_KG_LOOKBACK_DAYS * SECONDS_PER_DAY)

    # Search for events mentioning the VC's name or firm
    name_lower = name.lower()
    firm_lower = firm.lower()

    # Search across all event sources for mentions
    events = store.get_events(since=lookback, limit=5000)
    matching = []
    for ev in events:
        content = (ev.raw_content or "").lower()
        meta_str = json.dumps(ev.metadata).lower()
        combined = f"{content} {meta_str} {' '.join(ev.participants).lower()}"
        if name_lower in combined or firm_lower in combined:
            matching.append(ev)

    if not matching:
        return fragments

    # Build a summary of interactions
    interactions = []
    for ev in matching[:20]:
        interactions.append({
            "source": ev.source,
            "type": ev.event_type,
            "timestamp": ev.timestamp,
            "snippet": (ev.raw_content or "")[:200],
        })

    fragments.append(IntelFragment(
        source_type=VCSourceType.KG,
        query=f"kg_scan:{name}:{firm}",
        raw_text=json.dumps(interactions),
        claim_type="vc_interaction",
        subject=subject,
        predicate="interacted_with_user",
        object_data={
            "interaction_count": len(matching),
            "sources": list(set(ev.source for ev in matching)),
            "earliest": min(ev.timestamp for ev in matching),
            "latest": max(ev.timestamp for ev in matching),
            "interactions": interactions,
        },
        content_quality=min(1.0, len(matching) / 5),
        extracted_at=now,
    ))

    return fragments


def _synthesize_dossier(
    name: str,
    firm: str,
    fragments: list[IntelFragment],
    claim_records: list[dict],
) -> dict:
    """Merge fragments into a structured dossier with per-field source attribution."""
    dossier: dict[str, Any] = {
        "assertion_type": "vc_dossier",
        "name": name,
        "firm": firm,
        "thesis": {},
        "portfolio": {},
        "style": {},
        "interests": {},
        "fund_status": {},
        "interactions": {},
        "sources_used": claim_records,
    }

    # Group fragments by claim_type, pick highest-quality for each field
    by_type: dict[str, list[IntelFragment]] = {}
    for frag in fragments:
        by_type.setdefault(frag.claim_type, []).append(frag)

    # Thesis (from vc_thesis claims)
    thesis_frags = by_type.get("vc_thesis", [])
    if thesis_frags:
        best = max(thesis_frags, key=lambda f: f.content_quality)
        data = best.object_data
        conf = compute_confidence(best.source_type, best.extracted_at, best.content_quality)
        dossier["thesis"] = _with_attribution(data, conf, best.source_type.value)

    # Portfolio (from vc_portfolio claims)
    portfolio_frags = by_type.get("vc_portfolio", [])
    if portfolio_frags:
        best = max(portfolio_frags, key=lambda f: f.content_quality)
        data = best.object_data
        conf = compute_confidence(best.source_type, best.extracted_at, best.content_quality)
        dossier["portfolio"] = _with_attribution(data, conf, best.source_type.value)

    # Style (from vc_style claims)
    style_frags = by_type.get("vc_style", [])
    if style_frags:
        best = max(style_frags, key=lambda f: f.content_quality)
        data = best.object_data
        conf = compute_confidence(best.source_type, best.extracted_at, best.content_quality)
        dossier["style"] = _with_attribution(data, conf, best.source_type.value)

    # Interests (from vc_interest claims)
    interest_frags = by_type.get("vc_interest", [])
    if interest_frags:
        best = max(interest_frags, key=lambda f: f.content_quality)
        data = best.object_data
        conf = compute_confidence(best.source_type, best.extracted_at, best.content_quality)
        dossier["interests"] = _with_attribution(data, conf, best.source_type.value)

    # Fund status (from vc_fund_status claims)
    fund_frags = by_type.get("vc_fund_status", [])
    if fund_frags:
        best = max(fund_frags, key=lambda f: f.content_quality)
        data = best.object_data
        conf = compute_confidence(best.source_type, best.extracted_at, best.content_quality)
        dossier["fund_status"] = _with_attribution(data, conf, best.source_type.value)

    # Interactions (from vc_interaction claims — KG data)
    interaction_frags = by_type.get("vc_interaction", [])
    if interaction_frags:
        best = max(interaction_frags, key=lambda f: f.content_quality)
        data = best.object_data
        conf = compute_confidence(best.source_type, best.extracted_at, best.content_quality)
        dossier["interactions"] = _with_attribution(data, conf, best.source_type.value)

    # Source breakdown for display
    source_breakdown: dict[str, int] = {}
    for frag in fragments:
        key = frag.source_type.value
        source_breakdown[key] = source_breakdown.get(key, 0) + 1
    dossier["source_breakdown"] = source_breakdown

    return dossier


def _with_attribution(data: dict, confidence: float, source: str) -> dict:
    """Wrap each field value with confidence and source attribution."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key == "raw_summary":
            result[key] = {"value": value, "confidence": confidence, "from": source}
        elif isinstance(value, (list, dict, str, int, float, bool)) or value is None:
            result[key] = {"value": value, "confidence": confidence, "from": source}
        else:
            result[key] = {"value": str(value), "confidence": confidence, "from": source}
    return result


def _aggregate_admiralty(
    fragments: list[IntelFragment],
) -> tuple[str, int, float]:
    """Compute aggregate Admiralty scores across all fragments.

    Returns: (best_reliability, best_credibility, weighted_confidence)
    """
    if not fragments:
        return ("F", 6, 0.0)

    # Best reliability and credibility across sources
    rel_order = "ABCDEF"
    best_rel = "F"
    best_cred = 6

    confidences: list[float] = []
    for frag in fragments:
        rel, cred = admiralty_for_source(frag.source_type)
        conf = compute_confidence(frag.source_type, frag.extracted_at, frag.content_quality)
        confidences.append(conf)

        if rel_order.index(rel) < rel_order.index(best_rel):
            best_rel = rel
        if cred < best_cred:
            best_cred = cred

    overall = aggregate_confidence(confidences)
    return (best_rel, best_cred, overall)
