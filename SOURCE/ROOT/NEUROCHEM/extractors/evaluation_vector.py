"""
Evaluation Vector Assembler (Stochastic Extractor 1).

Maps reward domain subscores into a unified evaluation vector E(t) ∈ [0,1]^n
with optional Gaussian noise injection per axis.

Usage
-----
>>> from zados.neurochem.extractors.evaluation_vector import (
...     assemble_evaluation_vector, DEFAULT_EVALUATION_CONFIG,
... )
>>> E = assemble_evaluation_vector(domain_results, rng=engine.rng)
>>> E["novelty"]  # float in [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from zados.reward.base.types import RewardDomainResult


# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True)
class EvaluationAxisConfig:
    """
    Configuration for a single evaluation axis.

    Attributes
    ----------
    name : str
        Axis name (e.g., "novelty", "urgency").
    domain : str
        Source reward domain (e.g., "innovation", "logic").
    subscore_key : str
        Key into RewardDomainResult.subscores dict.
    transform : str
        "identity" → raw score, "invert" → 1 - score,
        "general_score" → use domain.general_score instead of subscore.
    sigma : float
        Gaussian noise std dev (0.0 = no noise).
    weight : float
        Multiplicative scaling applied after extraction.
    """
    name: str
    domain: str
    subscore_key: str
    transform: str = "identity"
    sigma: float = 0.0
    weight: float = 1.0


@dataclass(frozen=True)
class EvaluationVectorConfig:
    """
    Full evaluation vector specification.

    Attributes
    ----------
    axes : tuple of EvaluationAxisConfig
        Ordered list of evaluation axes.
    """
    axes: Tuple[EvaluationAxisConfig, ...]


# Default 8-axis evaluation vector mapping to existing reward domains.
DEFAULT_EVALUATION_CONFIG = EvaluationVectorConfig(axes=(
    EvaluationAxisConfig(
        "novelty", "innovation", "novelty_generation",
    ),
    EvaluationAxisConfig(
        "emotional_valence", "human_attunement", "empathetic_inference",
    ),
    EvaluationAxisConfig(
        "urgency", "ethics", "failure_mode_awareness",
    ),
    EvaluationAxisConfig(
        "logical_conflict", "logic", "internal_consistency",
        transform="invert",
    ),
    EvaluationAxisConfig(
        "coherence", "logic", "semantic_continuity",
    ),
    EvaluationAxisConfig(
        "social_salience", "human_attunement", "cognitive_reading",
    ),
    EvaluationAxisConfig(
        "reward_alignment", "innovation", "general_score",
        transform="general_score",
    ),
    EvaluationAxisConfig(
        "identity_resonance", "ethics", "intent_clarity",
    ),
))


# =====================================================================
# Pure functions
# =====================================================================

def extract_axis_value(
    domain_results: Dict[str, RewardDomainResult],
    axis: EvaluationAxisConfig,
) -> float:
    """
    Extract a single evaluation axis value from domain results.

    Parameters
    ----------
    domain_results : dict
        Maps domain name → RewardDomainResult.
    axis : EvaluationAxisConfig
        Axis specification.

    Returns
    -------
    float
        Raw value in [0, 1]. Returns 0.0 if domain or subscore missing.
    """
    result = domain_results.get(axis.domain)
    if result is None:
        return 0.0

    if axis.transform == "general_score":
        raw = result.general_score
    else:
        subscore = result.subscores.get(axis.subscore_key)
        if subscore is None:
            return 0.0
        raw = subscore.score

    if axis.transform == "invert":
        raw = 1.0 - raw

    # Apply weight and clamp
    value = raw * axis.weight
    return max(0.0, min(1.0, value))


def inject_noise(
    value: float,
    sigma: float,
    rng: np.random.Generator,
) -> float:
    """
    Add Gaussian noise to a value and clamp to [0, 1].

    Parameters
    ----------
    value : float
        Input value.
    sigma : float
        Noise standard deviation. If 0.0, returns value unchanged.
    rng : np.random.Generator
        Numpy RNG for reproducibility.

    Returns
    -------
    float
        Noisy value clamped to [0, 1].
    """
    if sigma <= 0.0:
        return value
    noisy = value + float(rng.normal(0.0, sigma))
    return max(0.0, min(1.0, noisy))


def assemble_evaluation_vector(
    domain_results: Dict[str, RewardDomainResult],
    config: EvaluationVectorConfig = DEFAULT_EVALUATION_CONFIG,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Assemble unified evaluation vector E(t) from reward domain results.

    Maps reward domain subscores to a dict of named evaluation axes,
    each in [0, 1], with optional per-axis Gaussian noise.

    Parameters
    ----------
    domain_results : dict
        Maps domain name → RewardDomainResult.
    config : EvaluationVectorConfig
        Evaluation vector specification. Defaults to 8-axis config.
    rng : np.random.Generator, optional
        Numpy RNG for noise injection. If None, no noise is added
        (sigma is ignored).

    Returns
    -------
    dict
        Maps axis name → float in [0, 1].
        e.g., {"novelty": 0.73, "urgency": 0.12, ...}
    """
    vector: Dict[str, float] = {}

    for axis in config.axes:
        value = extract_axis_value(domain_results, axis)

        if rng is not None and axis.sigma > 0.0:
            value = inject_noise(value, axis.sigma, rng)

        vector[axis.name] = value

    return vector
