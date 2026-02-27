"""Admiralty scoring engine for VC intelligence sources.

Each source type has a defined Admiralty Code (source reliability A-F,
information credibility 1-6) that produces a confidence score via:

    confidence = reliability_base * credibility_mult * recency_decay * content_quality

Recency decay: 2^(-age_days / half_life_days)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum, unique


@unique
class VCSourceType(str, Enum):
    """Intelligence source types for VC research."""
    PORTFOLIO = "vc_portfolio"
    PODCAST = "vc_podcast"
    BLOG = "vc_blog"
    TWITTER = "vc_twitter"
    PROFILE = "vc_profile"
    WEB = "vc_web"
    KG = "vc_kg"
    FUND_STATUS = "vc_fund_status"


@dataclass(frozen=True)
class AdmiraltyScore:
    """Admiralty Code for a source type."""
    source_reliability: str    # A-F
    info_credibility: int      # 1-6
    half_life_days: int


# ── Source → Admiralty mapping ────────────────────────────────

SOURCE_ADMIRALTY: dict[VCSourceType, AdmiraltyScore] = {
    VCSourceType.KG:          AdmiraltyScore("A", 1, 365),
    VCSourceType.PORTFOLIO:   AdmiraltyScore("A", 2, 180),
    VCSourceType.FUND_STATUS: AdmiraltyScore("A", 2, 180),
    VCSourceType.PODCAST:     AdmiraltyScore("B", 2, 365),
    VCSourceType.BLOG:        AdmiraltyScore("B", 3, 365),
    VCSourceType.PROFILE:     AdmiraltyScore("B", 3, 730),
    VCSourceType.TWITTER:     AdmiraltyScore("C", 3, 90),
    VCSourceType.WEB:         AdmiraltyScore("D", 4, 90),
}

# Reliability letter → numeric multiplier (NATO standard: A=most reliable)
RELIABILITY_SCORES: dict[str, float] = {
    "A": 1.0,
    "B": 0.8,
    "C": 0.6,
    "D": 0.4,
    "E": 0.2,
    "F": 0.0,
}

# Credibility number → numeric multiplier (1=confirmed, 6=truth cannot be judged)
CREDIBILITY_SCORES: dict[int, float] = {
    1: 1.0,
    2: 0.85,
    3: 0.7,
    4: 0.5,
    5: 0.3,
    6: 0.1,
}


def compute_confidence(
    source_type: VCSourceType,
    extracted_at: int,
    content_quality: float = 1.0,
    now: int | None = None,
) -> float:
    """Compute confidence from source Admiralty code, recency, and content quality.

    confidence = reliability_base * credibility_mult * recency_decay * content_quality

    Recency decay uses exponential half-life: 2^(-age_days / half_life_days).
    """
    admiralty = SOURCE_ADMIRALTY[source_type]
    reliability_base = RELIABILITY_SCORES[admiralty.source_reliability]
    credibility_mult = CREDIBILITY_SCORES[admiralty.info_credibility]

    if now is None:
        now = int(time.time())
    age_days = max(0, (now - extracted_at) / 86_400)
    recency_decay = math.pow(2, -age_days / admiralty.half_life_days)

    content_quality = max(0.0, min(1.0, content_quality))

    return round(reliability_base * credibility_mult * recency_decay * content_quality, 4)


def admiralty_for_source(source_type: VCSourceType) -> tuple[str, int]:
    """Return (reliability_letter, credibility_number) for a source type."""
    admiralty = SOURCE_ADMIRALTY[source_type]
    return (admiralty.source_reliability, admiralty.info_credibility)


def aggregate_confidence(confidences: list[float]) -> float:
    """Aggregate multiple source confidences into overall dossier confidence.

    Formula: best_source + 0.05 * second_best + 0.05 * third_best (capped at 0.99).
    """
    if not confidences:
        return 0.0
    sorted_conf = sorted(confidences, reverse=True)
    result = sorted_conf[0]
    for c in sorted_conf[1:3]:
        result += 0.05 * c
    return min(0.99, round(result, 4))
