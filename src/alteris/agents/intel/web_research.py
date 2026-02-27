"""Gemini multi-query research for VC intelligence.

Runs 6 targeted web searches in parallel, each with a claim-type-specific
extraction prompt that produces structured JSON. Each query's source type
is known from the query itself (not post-hoc classification).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from alteris.agents.intel.collector import (
    IntelFragment,
    QuerySpec,
    QUERY_SPECS,
    run_queries_parallel,
)
from alteris.agents.intel.sources import VCSourceType

logger = logging.getLogger(__name__)


# ── Per-claim-type extraction prompts ─────────────────────────

EXTRACTION_PROMPTS: dict[str, str] = {
    "vc_thesis": """Extract structured information about this VC's investment thesis.

Return JSON with these fields (use null for unknown):
{{
    "thesis_summary": "1-2 sentence summary of their investment thesis",
    "focus_areas": ["list of focus areas/sectors"],
    "stage_preference": "seed | series_a | series_b | growth | multi_stage",
    "sector_keywords": ["specific keywords they use repeatedly"],
    "anti_portfolio": ["things they explicitly avoid"],
    "quotes": ["direct quotes about their thesis, max 3"]
}}

Source text:
{raw_text}

Return ONLY valid JSON.""",

    "vc_interest": """Extract this VC's recent interests and opinions from social media.

Return JSON with these fields (use null for unknown):
{{
    "recent_topics": ["topics they've discussed recently"],
    "opinions": [
        {{"topic": "...", "stance": "bullish|bearish|neutral", "quote": "..."}}
    ],
    "engaged_with": ["people/companies they interact with"],
    "tone": "technical | casual | provocative | thoughtful | promotional"
}}

Source text:
{raw_text}

Return ONLY valid JSON.""",

    "vc_style": """Extract this VC's decision-making style from podcast/interview content.

Return JSON with these fields (use null for unknown):
{{
    "style_summary": "1-2 sentence summary of their investing style",
    "deal_breakers": ["things that make them pass"],
    "what_excites_them": ["what gets them excited about a deal"],
    "meeting_format": "how they prefer to take meetings",
    "founder_preferences": ["what they look for in founders"],
    "quotes": ["direct quotes about their style, max 3"]
}}

Source text:
{raw_text}

Return ONLY valid JSON.""",

    "vc_portfolio": """Extract recent portfolio and investment activity.

Return JSON with these fields (use null for unknown):
{{
    "recent_investments": [
        {{"company": "...", "round": "seed|A|B|...", "date": "YYYY-MM", "sector": "..."}}
    ],
    "notable_exits": ["list of notable exits/IPOs"],
    "portfolio_themes": ["recurring themes across portfolio"],
    "investment_pace": "estimated deals per year",
    "co_investors": ["firms they frequently co-invest with"]
}}

Source text:
{raw_text}

Return ONLY valid JSON.""",

    "vc_fund_status": """Extract fund information and deployment status.

Return JSON with these fields (use null for unknown):
{{
    "fund_name": "name of the current/most recent fund",
    "fund_size": "size in dollars (e.g. '$150M')",
    "fund_vintage": "year the fund was raised",
    "deployment_stage": "early | mid | late | fully_deployed",
    "recent_announcements": ["recent fund-related news"],
    "check_size_range": "typical check size range (e.g. '$500K-$5M')"
}}

Source text:
{raw_text}

Return ONLY valid JSON.""",
}

# Map claim_type → predicate
CLAIM_PREDICATES: dict[str, str] = {
    "vc_thesis": "has_thesis",
    "vc_interest": "interested_in",
    "vc_style": "has_style",
    "vc_portfolio": "invested_in",
    "vc_fund_status": "fund_status",
}


def _compute_content_quality(extracted: dict | None, claim_type: str) -> float:
    """Score extraction completeness (0-1) based on how many fields are populated."""
    if not extracted:
        return 0.1

    # Count non-null, non-empty fields
    total_fields = len(extracted)
    if total_fields == 0:
        return 0.1

    populated = 0
    for v in extracted.values():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if isinstance(v, list) and not v:
            continue
        populated += 1

    return max(0.1, round(populated / total_fields, 2))


def _execute_single_query(
    llm_client: Any,
    spec: QuerySpec,
    name: str,
    firm: str,
    model: str = "",
) -> list[IntelFragment]:
    """Execute a single query: web_search → extract → IntelFragment."""
    query = spec.build_query(name, firm)
    now = int(time.time())
    subject = f"vc:{name.lower().replace(' ', '_')}:{firm.lower().replace(' ', '_')}"

    # Step 1: Web search
    raw_text = None
    if hasattr(llm_client, "web_search"):
        raw_text = llm_client.web_search(query, model=model)

    if not raw_text:
        logger.info("No web results for query: %s", query[:80])
        return []

    # Step 2: Structured extraction via LLM
    extraction_prompt = EXTRACTION_PROMPTS.get(spec.claim_type)
    if not extraction_prompt:
        # Fallback: store raw text as generic web intel
        return [IntelFragment(
            source_type=spec.source_type,
            query=query,
            raw_text=raw_text,
            claim_type=spec.claim_type,
            subject=subject,
            predicate=spec.effective_predicate,
            object_data={"raw_summary": raw_text[:2000]},
            content_quality=0.3,
            extracted_at=now,
        )]

    prompt = extraction_prompt.format(raw_text=raw_text[:8000])
    response = llm_client.generate(
        prompt,
        system="You are a structured data extractor. Extract only factual information from the provided text. Return valid JSON.",
        model=model,
        format_json=True,
        temperature=0.1,
    )

    extracted = None
    if response:
        try:
            extracted = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse extraction for %s", spec.claim_type)

    quality = _compute_content_quality(extracted, spec.claim_type)

    return [IntelFragment(
        source_type=spec.source_type,
        query=query,
        raw_text=raw_text,
        claim_type=spec.claim_type,
        subject=subject,
        predicate=spec.effective_predicate,
        object_data=extracted or {"raw_summary": raw_text[:2000]},
        content_quality=quality,
        extracted_at=now,
    )]


def research_vc_deep(
    llm_client: Any,
    name: str,
    firm: str,
    model: str = "",
    max_workers: int = 3,
    specs: list[QuerySpec] | None = None,
) -> list[IntelFragment]:
    """Run 6 targeted queries in parallel, each producing structured IntelFragments.

    Args:
        llm_client: LLM client with web_search() and generate() methods
        name: VC name (e.g., "Sarah Guo")
        firm: VC firm (e.g., "Conviction")
        model: Model to use for extraction
        max_workers: Max parallel query threads
        specs: Override default QUERY_SPECS (for testing)

    Returns:
        All IntelFragments from all successful queries
    """
    use_specs = specs or QUERY_SPECS

    def execute_fn(spec: QuerySpec, n: str, f: str) -> list[IntelFragment]:
        return _execute_single_query(llm_client, spec, n, f, model=model)

    logger.info("Starting deep VC research: %s (%s) — %d queries", name, firm, len(use_specs))
    fragments = run_queries_parallel(use_specs, name, firm, execute_fn, max_workers=max_workers)
    logger.info("Deep research complete: %d fragments", len(fragments))

    return fragments
