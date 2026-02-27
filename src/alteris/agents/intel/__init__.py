"""VC intelligence gathering — multi-source research with Admiralty scoring."""

from alteris.agents.intel.sources import (
    AdmiraltyScore,
    VCSourceType,
    admiralty_for_source,
    compute_confidence,
)
from alteris.agents.intel.collector import IntelFragment, QuerySpec

__all__ = [
    "AdmiraltyScore",
    "VCSourceType",
    "admiralty_for_source",
    "compute_confidence",
    "IntelFragment",
    "QuerySpec",
]
