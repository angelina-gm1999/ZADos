"""
Reactivity Matrix and Phasic Burst Computation (Extractor 2 — routing).

Defines the reactivity matrix B ∈ R^{m×n} mapping evaluation axes to
per-NT stochastic burst deltas via threshold-gated impulse generation:

    ΔC(t) = B · E(t) ⊙ ξ(t) ⊙ I_{E>θ}

Usage
-----
>>> from zados.neurochem.extractors.reactivity_matrix import (
...     compute_stochastic_burst_deltas, burst_deltas_to_modulation_signals,
... )
>>> deltas = compute_stochastic_burst_deltas(eval_vector, prev_vector, dt, rng=rng)
>>> signals = burst_deltas_to_modulation_signals(deltas, existing_signals)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from zados.neurochem.extractors.stochastic_impulse import sample_impulse


# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True)
class ReactivityEntry:
    """
    Single entry in reactivity matrix: one NT–axis coupling.

    Attributes
    ----------
    nt_name : str
        Target neurotransmitter (e.g., "DA").
    axis_name : str
        Source evaluation axis (e.g., "novelty").
    weight : float
        Coupling strength β_k^(i).
    threshold : float
        Gating threshold θ_k. Axis must exceed this for burst to fire.
    distribution : str
        Stochastic impulse distribution ("gamma", "poisson", "lognormal").
    """
    nt_name: str
    axis_name: str
    weight: float
    threshold: float = 0.3
    distribution: str = "gamma"


@dataclass(frozen=True)
class ReactivityMatrixConfig:
    """
    Full reactivity matrix specification.

    Attributes
    ----------
    entries : tuple of ReactivityEntry
        All NT–axis couplings.
    """
    entries: Tuple[ReactivityEntry, ...]


# Default reactivity matrix covering all 12 NTs × relevant evaluation axes.
DEFAULT_REACTIVITY_ENTRIES: Tuple[ReactivityEntry, ...] = (
    # DA — novelty-driven exploration
    ReactivityEntry("DA",  "novelty",            0.8, 0.3, "gamma"),
    ReactivityEntry("DA",  "reward_alignment",   0.5, 0.2, "gamma"),
    # NE — urgency and conflict detection
    ReactivityEntry("NE",  "urgency",            0.7, 0.4, "poisson"),
    ReactivityEntry("NE",  "logical_conflict",   0.6, 0.3, "poisson"),
    # 5HT — emotional stability / disinhibition
    ReactivityEntry("5HT", "emotional_valence",  0.6, 0.2, "lognormal"),
    ReactivityEntry("5HT", "coherence",          0.4, 0.3, "lognormal"),
    # OXT — social bonding
    ReactivityEntry("OXT", "social_salience",    0.8, 0.2, "gamma"),
    ReactivityEntry("OXT", "emotional_valence",  0.5, 0.3, "gamma"),
    # ACh — precision / attention
    ReactivityEntry("ACh", "coherence",          0.6, 0.3, "gamma"),
    ReactivityEntry("ACh", "urgency",            0.4, 0.4, "poisson"),
    # GABA — inhibitory control under conflict (NT-level, phasic bursts).
    # Distinct from GABA_B in regulatory_modulator.py, which targets the
    # GABA_B *receptor* K_d via slow τ-smoothed integrator.
    ReactivityEntry("GABA", "logical_conflict",  0.5, 0.4, "lognormal"),
    # cortisol — stress
    ReactivityEntry("cortisol", "urgency",       0.7, 0.5, "poisson"),
    # CRH — stress cascade
    ReactivityEntry("CRH", "urgency",            0.6, 0.5, "poisson"),
    # CB1 — identity / flexibility
    ReactivityEntry("CB1", "identity_resonance", 0.5, 0.3, "gamma"),
    ReactivityEntry("CB1", "novelty",            0.4, 0.3, "gamma"),
    # GLU — integration / conflict binding
    ReactivityEntry("GLU", "coherence",          0.5, 0.3, "gamma"),
    ReactivityEntry("GLU", "logical_conflict",   0.4, 0.4, "gamma"),
    # MOR — hedonic / emotional
    ReactivityEntry("MOR", "emotional_valence",  0.4, 0.3, "lognormal"),
    # histamine — arousal / alertness
    ReactivityEntry("histamine", "urgency",      0.5, 0.3, "poisson"),
    ReactivityEntry("histamine", "novelty",      0.3, 0.3, "poisson"),
)

DEFAULT_REACTIVITY_CONFIG = ReactivityMatrixConfig(entries=DEFAULT_REACTIVITY_ENTRIES)


# =====================================================================
# Pure functions
# =====================================================================

def apply_threshold_gating(value: float, threshold: float) -> float:
    """
    Threshold gating indicator: returns value if > threshold, else 0.0.

    Parameters
    ----------
    value : float
        Evaluation axis value.
    threshold : float
        Gating threshold.

    Returns
    -------
    float
        Gated value or 0.0.
    """
    return value if value > threshold else 0.0


def compute_stochastic_burst_deltas(
    evaluation_vector: Dict[str, float],
    prev_evaluation_vector: Optional[Dict[str, float]],
    dt: float,
    config: ReactivityMatrixConfig = DEFAULT_REACTIVITY_CONFIG,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Compute per-NT stochastic burst deltas.

    ΔC_i = Σ_k β_k^(i) · e_k · ξ_k · I[e_k > θ_k]

    Parameters
    ----------
    evaluation_vector : dict
        Current E(t), maps axis name → float.
    prev_evaluation_vector : dict, optional
        Previous E(t-dt) for computing de/dt. None → zero volatility.
    dt : float
        Simulation timestep (for computing de/dt).
    config : ReactivityMatrixConfig
        Reactivity matrix entries.
    rng : np.random.Generator, optional
        Numpy RNG.

    Returns
    -------
    dict
        Maps NT name → burst delta (non-negative float).
        e.g., {"DA": 0.23, "NE": 0.15, ...}
    """
    deltas: Dict[str, float] = {}

    for entry in config.entries:
        e_k = evaluation_vector.get(entry.axis_name, 0.0)

        # Threshold gating
        gated = apply_threshold_gating(e_k, entry.threshold)
        if gated <= 0.0:
            continue

        # Compute volatility de/dt
        d_eval_dt = 0.0
        if prev_evaluation_vector is not None and dt > 0.0:
            prev_val = prev_evaluation_vector.get(entry.axis_name, 0.0)
            d_eval_dt = abs(e_k - prev_val) / dt

        # Sample stochastic impulse
        xi = sample_impulse(
            gated,
            d_eval_dt=d_eval_dt,
            distribution=entry.distribution,
            rng=rng,
        )

        # Weighted contribution
        contribution = entry.weight * xi

        # Accumulate per NT
        if entry.nt_name in deltas:
            deltas[entry.nt_name] += contribution
        else:
            deltas[entry.nt_name] = contribution

    return deltas


def burst_deltas_to_modulation_signals(
    burst_deltas: Dict[str, float],
    existing_signals: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Convert per-NT burst deltas into modulation_signals format
    compatible with ``engine.step(modulation_signals)``.

    Injects each burst delta as a ``"stochastic_burst"`` signal key.

    Parameters
    ----------
    burst_deltas : dict
        Maps NT name → burst delta float.
    existing_signals : dict, optional
        Existing modulation_signals dict to merge into.
        If None, starts from empty dict.

    Returns
    -------
    dict
        Modulation signals dict: {nt_name: {signal_key: value}}.
    """
    if existing_signals is None:
        signals: Dict[str, Dict[str, float]] = {}
    else:
        # Deep copy to avoid mutating caller's dict
        signals = {
            nt: dict(sigs) for nt, sigs in existing_signals.items()
        }

    for nt_name, delta in burst_deltas.items():
        if nt_name not in signals:
            signals[nt_name] = {}
        signals[nt_name]["stochastic_burst"] = delta

    return signals
