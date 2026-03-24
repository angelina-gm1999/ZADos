from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RewardProfile:
    """
    Static reward profile.

    Pure configuration object.
    No logic, no mutation.
    """

    name: str

    # Domain weight scaling (0.0 – 1.0)
    domain_weights: Dict[str, float]

    # Per-domain tolerance thresholds (0.0 – 1.0)
    threshold_tolerances: Dict[str, float]

    # Global bias terms
    suppression_bias: float
    abstention_bias: float
