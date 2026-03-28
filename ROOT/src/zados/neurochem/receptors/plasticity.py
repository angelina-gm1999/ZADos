"""
Emotion-driven receptor plasticity.

Pure functions for computing and applying emotion → receptor state changes
using the emotion_plasticity_rules defined in each ReceptorSpec.

Usage
-----
>>> from zados.neurochem.receptors.plasticity import (
...     compute_plasticity_deltas, apply_plasticity_delta,
... )
>>> deltas = compute_plasticity_deltas("joy", receptor_modules)
>>> new_state = apply_plasticity_delta(old_state, deltas["DA_D1"], intensity=0.8)
"""

from __future__ import annotations

from typing import Dict

from zados.neurochem.receptors.base import ReceptorFamilyModule
from zados.neurochem.state.receptor_state import ReceptorState


def compute_plasticity_deltas(
    emotion_id: str,
    receptor_modules: Dict[str, ReceptorFamilyModule],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-receptor parameter deltas for an emotion event.

    Iterates all registered receptor family modules and collects
    emotion_plasticity_rules matching the given emotion_id.

    Parameters
    ----------
    emotion_id : str
        Emotion identifier (e.g., "joy", "fear", "curiosity")
    receptor_modules : dict
        Map of parent_nt -> ReceptorFamilyModule

    Returns
    -------
    dict
        Map of receptor_id -> {"sigma_delta": float, "rho_delta": float, ...}
        for all receptors affected by this emotion. Empty dict if no matches.
    """
    result: Dict[str, Dict[str, float]] = {}
    for module in receptor_modules.values():
        for receptor_id in module.get_receptor_ids():
            rules = module.get_emotion_plasticity(receptor_id, emotion_id)
            if rules:
                result[receptor_id] = dict(rules)
    return result


def apply_plasticity_delta(
    receptor_state: ReceptorState,
    deltas: Dict[str, float],
    intensity: float = 1.0,
) -> ReceptorState:
    """
    Apply emotion plasticity deltas to a receptor state.

    Creates a copy of the state and applies sigma_delta and rho_delta
    scaled by intensity.

    Parameters
    ----------
    receptor_state : ReceptorState
        Current receptor state (not mutated)
    deltas : dict
        Parameter deltas, e.g. {"sigma_delta": 0.1, "rho_delta": 0.05}
    intensity : float, default=1.0
        Scaling factor for the deltas (e.g., emotion strength)

    Returns
    -------
    ReceptorState
        New state with deltas applied
    """
    state = receptor_state.copy()
    if "sigma_delta" in deltas:
        state.update_sensitivity(deltas["sigma_delta"] * intensity)
    if "rho_delta" in deltas:
        state.update_density(deltas["rho_delta"] * intensity)
    return state
