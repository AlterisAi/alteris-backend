"""VC intelligence gathering — multi-source research with Admiralty scoring."""

from loom.agents.intel.sources import (
    AdmiraltyScore,
    VCSourceType,
    admiralty_for_source,
    compute_confidence,
)
from loom.agents.intel.collector import IntelFragment, QuerySpec

__all__ = [
    "AdmiraltyScore",
    "VCSourceType",
    "admiralty_for_source",
    "compute_confidence",
    "IntelFragment",
    "QuerySpec",
]
