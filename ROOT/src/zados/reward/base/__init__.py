"""
Reward system base types, ABCs, and structural primitives.
"""

from .types import (
    RewardContext,
    RewardSubscore,
    RewardDomainResult,
    RewardWeights,
    RewardMetaDirective,
)
from .interfaces import RewardSubmodule, RewardDomain
from .structure import ThresholdSpec, RewardFlag, RewardFlagSet, ProvenanceRecord

__all__ = [
    # Core types
    "RewardContext",
    "RewardSubscore",
    "RewardDomainResult",
    "RewardWeights",
    "RewardMetaDirective",
    # ABCs
    "RewardSubmodule",
    "RewardDomain",
    # Structural primitives
    "ThresholdSpec",
    "RewardFlag",
    "RewardFlagSet",
    "ProvenanceRecord",
]
