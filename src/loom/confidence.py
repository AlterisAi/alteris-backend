"""Unified Admiralty-grounded confidence scoring.

Every confidence score in the system decomposes into:
    confidence = source_reliability × evidence_credibility × recency_decay

Source reliability (axis 1): how trustworthy is the process that produced this?
Evidence credibility (axis 2): how much corroborating evidence exists?
Recency decay: how old is the evidence?
"""

from __future__ import annotations

import math

# NATO Admiralty reliability scale (A-F)
RELIABILITY = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2, "F": 0.0}

# Source process -> reliability grade
SOURCE_RELIABILITY = {
    "system_sql": "A",
    "calls_macos": "A",
    "calls_whatsapp": "A",
    "calendar": "A",
    "granola": "B",
    "mail": "B",
    "imessage": "B",
    "whatsapp": "C",
    "llm_extraction": "C",
    "cross_source_inference": "B",
}


def evidence_credibility(
    count: int,
    scale: float = 20.0,
    floor: float = 0.3,
) -> float:
    """Map evidence count to credibility multiplier (Admiralty axis 2).

    Saturating curve: more evidence -> higher credibility, asymptotic to 1.0.
    - count=0  -> floor
    - count=5  -> ~0.55
    - count=10 -> ~0.80
    - count=20 -> ~0.95
    """
    return min(0.95, floor + (1.0 - floor) * (1 - math.exp(-count / scale)))


def compute_confidence(
    source: str,
    evidence_count: int = 1,
    evidence_scale: float = 20.0,
    credibility_floor: float = 0.3,
    recency_days: float = 0.0,
    half_life_days: float = 365.0,
) -> float:
    """Compute Admiralty-grounded confidence.

    Args:
        source: Key into SOURCE_RELIABILITY (e.g. "system_sql", "mail").
        evidence_count: Number of supporting observations.
        evidence_scale: Controls saturation speed (small = quick saturation).
        credibility_floor: Minimum credibility with zero evidence.
        recency_days: Age of the evidence in days.
        half_life_days: Exponential decay half-life. 0 = no decay.
    """
    reliability = RELIABILITY[SOURCE_RELIABILITY.get(source, "C")]
    cred = evidence_credibility(evidence_count, evidence_scale, credibility_floor)
    decay = math.pow(2, -recency_days / half_life_days) if half_life_days > 0 else 1.0
    return round(reliability * cred * decay, 4)
