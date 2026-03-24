"""
Oscillation-dependent CTMC transition rate and threshold modulation.

Provides dataclasses and pure functions for multi-band modulation of
receptor state transition rates and saturation thresholds.

From PDF Appendix H.7 (transition rates) and E.2 (threshold shifts).

Usage
-----
>>> from zados.neurochem.oscillations.transition_modulation import (
...     TransitionBandSpec,
...     compute_transition_multiplier,
...     ThresholdBandSpec,
...     modulate_threshold,
... )
>>> specs = [TransitionBandSpec("beta", 0.3), TransitionBandSpec("gamma", -0.2)]
>>> m = compute_transition_multiplier({"beta": 0.8, "gamma": 0.5}, specs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TransitionBandSpec:
    """
    One oscillation band's contribution to a CTMC transition rate multiplier.

    The total multiplier is: m(t) = clip(1 + Σ λ_k * φ_k, m_min, m_max)

    Attributes
    ----------
    band : str
        Oscillation band name ("delta", "theta", "alpha", "beta", "gamma",
        "theta_gamma", "alpha_beta")
    lambda_coeff : float
        Coefficient in the multiplier formula. Positive values speed up
        the transition when the band is active; negative values slow it.
    """
    band: str
    lambda_coeff: float


def compute_transition_multiplier(
    osc_amplitudes: Dict[str, float],
    specs: List[TransitionBandSpec],
    m_min: float = 0.1,
    m_max: float = 3.0,
) -> float:
    """
    Compute oscillation-dependent multiplier for a CTMC transition.

    m(t) = clip(1 + Σ λ_k * φ_k(t), m_min, m_max)

    Applied to timing thresholds: t_eff = t_base / m(t).
    Higher multiplier → faster transition (shorter effective wait).

    Parameters
    ----------
    osc_amplitudes : dict
        Map of band name -> amplitude φ_k(t) ∈ [0, 1]
    specs : list of TransitionBandSpec
        Band contributions to the multiplier
    m_min : float, default=0.1
        Minimum multiplier (prevents division by near-zero)
    m_max : float, default=3.0
        Maximum multiplier (caps acceleration)

    Returns
    -------
    float
        Transition rate multiplier in [m_min, m_max]
    """
    if not specs:
        return 1.0
    total = sum(
        spec.lambda_coeff * osc_amplitudes.get(spec.band, 0.0)
        for spec in specs
    )
    return max(m_min, min(m_max, 1.0 + total))


@dataclass(frozen=True)
class ThresholdBandSpec:
    """
    One oscillation band's contribution to a saturation threshold shift.

    threshold_eff = clip(base + Σ shift_k * φ_k, 0, 1)

    Attributes
    ----------
    band : str
        Oscillation band name
    shift_coefficient : float
        How much φ_k shifts the threshold. Positive values raise
        the threshold (harder to trigger); negative values lower it.
    """
    band: str
    shift_coefficient: float


def modulate_threshold(
    base_threshold: float,
    osc_amplitudes: Dict[str, float],
    specs: List[ThresholdBandSpec],
) -> float:
    """
    Compute oscillation-modulated saturation threshold.

    threshold_eff = clip(base + Σ shift_k * φ_k(t), 0, 1)

    Parameters
    ----------
    base_threshold : float
        Static threshold from config
    osc_amplitudes : dict
        Map of band name -> amplitude φ_k(t) ∈ [0, 1]
    specs : list of ThresholdBandSpec
        Band contributions to the threshold shift

    Returns
    -------
    float
        Effective threshold in [0, 1]
    """
    if not specs:
        return base_threshold
    total_shift = sum(
        spec.shift_coefficient * osc_amplitudes.get(spec.band, 0.0)
        for spec in specs
    )
    return max(0.0, min(1.0, base_threshold + total_shift))
