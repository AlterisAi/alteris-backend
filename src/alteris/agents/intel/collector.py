"""Fragment model and query orchestration for VC intelligence.

IntelFragment is the intermediate format between raw web search results
and the Event/Claim data model. QuerySpec defines the 6 targeted queries
that cover all intelligence tiers.
"""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from alteris.agents.intel.sources import VCSourceType, admiralty_for_source, compute_confidence
from alteris.constants import HASH_PREFIX_LEN
from alteris.models import Claim, Event, ExtractionMethod, ExtractionProvenance, Modality
from alteris.privacy import SensitivityLevel

logger = logging.getLogger(__name__)


@dataclass
class IntelFragment:
    """Intermediate format between raw web search and Event/Claim."""
    source_type: VCSourceType
    query: str
    raw_text: str
    claim_type: str            # vc_thesis, vc_portfolio, vc_style, ...
    subject: str               # "vc:{name_lower}:{firm_lower}"
    predicate: str             # has_thesis, invested_in, has_style, ...
    object_data: dict          # Structured extraction (JSON-serializable)
    content_quality: float     # 0-1 — how confident was the extraction
    extracted_at: int = field(default_factory=lambda: int(time.time()))


@dataclass(frozen=True)
class QuerySpec:
    """Defines a targeted search query and how to classify results."""
    template: str              # Query template with {name} and {firm} placeholders
    source_type: VCSourceType
    claim_type: str
    predicate: str = ""        # Defaults to claim_type if empty

    def build_query(self, name: str, firm: str) -> str:
        return self.template.format(name=name, firm=firm)

    @property
    def effective_predicate(self) -> str:
        return self.predicate or self.claim_type


# ── The 6 targeted queries ────────────────────────────────────

QUERY_SPECS = [
    QuerySpec(
        '"{name}" "{firm}" investment thesis focus areas',
        VCSourceType.PROFILE,
        "vc_thesis",
        "has_thesis",
    ),
    QuerySpec(
        '"{name}" site:x.com OR site:twitter.com',
        VCSourceType.TWITTER,
        "vc_interest",
        "interested_in",
    ),
    QuerySpec(
        '"{name}" podcast interview 2025 2026',
        VCSourceType.PODCAST,
        "vc_style",
        "has_style",
    ),
    QuerySpec(
        '"{firm}" recent investments portfolio 2025 2026',
        VCSourceType.PORTFOLIO,
        "vc_portfolio",
        "invested_in",
    ),
    QuerySpec(
        '"{name}" blog substack writing',
        VCSourceType.BLOG,
        "vc_thesis",
        "has_thesis",
    ),
    QuerySpec(
        '"{name}" "{firm}" fund size new fund',
        VCSourceType.FUND_STATUS,
        "vc_fund_status",
        "fund_status",
    ),
]


# ── Conversion: Fragment → Event ──────────────────────────────

def _make_event_id(source_type: str, query: str, extracted_at: int) -> str:
    """Deterministic event ID from source, query, and timestamp."""
    raw = f"{source_type}:{query}:{extracted_at}"
    return hashlib.sha256(raw.encode()).hexdigest()[:HASH_PREFIX_LEN]


def fragment_to_event(frag: IntelFragment) -> Event:
    """Convert an IntelFragment to an Event for storage."""
    eid = _make_event_id(frag.source_type.value, frag.query, frag.extracted_at)
    return Event(
        id=eid,
        source=frag.source_type.value,
        source_id=f"{frag.subject}:{frag.claim_type}:{frag.extracted_at}",
        event_type="vc_research",
        timestamp=frag.extracted_at,
        participants=(),
        raw_content=frag.raw_text[:50_000] if frag.raw_text else "",
        content_hash=Event.content_hash_of(frag.raw_text) if frag.raw_text else "",
        metadata={
            "query": frag.query,
            "claim_type": frag.claim_type,
            "content_quality": frag.content_quality,
        },
        sensitivity=SensitivityLevel.SENSITIVE,
        created_at=frag.extracted_at,
    )


# ── Conversion: Fragment → Claim ──────────────────────────────

def _make_claim_id(subject: str, predicate: str, source_type: str, extracted_at: int) -> str:
    """Deterministic claim ID."""
    raw = f"{subject}:{predicate}:{source_type}:{extracted_at}"
    return hashlib.sha256(raw.encode()).hexdigest()[:HASH_PREFIX_LEN]


def fragment_to_claim(frag: IntelFragment, event_id: str) -> Claim:
    """Convert an IntelFragment to a Claim with Admiralty-derived confidence."""
    import json

    cid = _make_claim_id(frag.subject, frag.predicate, frag.source_type.value, frag.extracted_at)
    confidence = compute_confidence(
        frag.source_type,
        frag.extracted_at,
        content_quality=frag.content_quality,
    )
    rel, cred = admiralty_for_source(frag.source_type)

    return Claim(
        id=cid,
        event_ids=[event_id],
        claim_type=frag.claim_type,
        subject=frag.subject,
        predicate=frag.predicate,
        object=json.dumps(frag.object_data),
        confidence=confidence,
        modality=Modality.OBSERVED,
        provenance=ExtractionProvenance(
            model_id="gemini-web-search",
            prompt_version="vc_intel_v1",
            extraction_method=ExtractionMethod.CLOUD_MODEL,
            extracted_at=frag.extracted_at,
        ),
        sensitivity=SensitivityLevel.SENSITIVE,
        created_at=frag.extracted_at,
    )


# ── Parallel query execution ─────────────────────────────────

def run_queries_parallel(
    specs: list[QuerySpec],
    name: str,
    firm: str,
    execute_fn: Callable[[QuerySpec, str, str], list[IntelFragment]],
    max_workers: int = 3,
) -> list[IntelFragment]:
    """Run queries in parallel using a thread pool.

    Args:
        specs: Query specifications to execute
        name: VC name
        firm: VC firm
        execute_fn: Function that takes (spec, name, firm) and returns fragments
        max_workers: Max parallel threads

    Returns:
        All fragments from all queries, flattened
    """
    all_fragments: list[IntelFragment] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(execute_fn, spec, name, firm): spec
            for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                fragments = future.result()
                all_fragments.extend(fragments)
                logger.info(
                    "Query [%s] → %d fragments",
                    spec.claim_type, len(fragments),
                )
            except Exception as exc:
                logger.warning(
                    "Query [%s] failed: %s",
                    spec.claim_type, exc,
                )

    return all_fragments
