"""
Receptor subtype switching mechanism.

Under sustained activation of one receptor subtype, expression shifts
toward a complementary subtype (homeostatic compensation). Density is
conserved: what the source loses, the target gains.

Usage
-----
>>> from zados.neurochem.receptors.subtype_switching import (
...     SubtypeSwitchRule, compute_subtype_switch_deltas,
...     apply_subtype_switch_deltas,
... )
>>> rules = [SubtypeSwitchRule("DA_D1", "DA_D2", 20.0, 0.005, 0.02)]
>>> deltas = compute_subtype_switch_deltas(receptor_states, rules, dt=1.0)
>>> new_states = apply_subtype_switch_deltas(receptor_states, deltas)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from zados.neurochem.state.receptor_state import ReceptorState


@dataclass(frozen=True)
class SubtypeSwitchRule:
    """
    Defines a conditional density transfer between receptor subtypes.

    When the source receptor's exposure_trace exceeds the threshold,
    density (rho) is transferred from source to target at the specified
    rate. Conservation: source loses what target gains.

    Attributes
    ----------
    source_receptor_id : str
        Receptor subtype that loses density (e.g., "DA_D1")
    target_receptor_id : str
        Receptor subtype that gains density (e.g., "DA_D2")
    exposure_threshold : float
        exposure_trace threshold to trigger switching
    rho_transfer_rate : float
        Density transfer rate per unit time
    max_transfer_per_step : float
        Cap on density transfer per step
    """
    source_receptor_id: str
    target_receptor_id: str
    exposure_threshold: float
    rho_transfer_rate: float
    max_transfer_per_step: float


def compute_subtype_switch_deltas(
    receptor_states: Dict[str, ReceptorState],
    rules: List[SubtypeSwitchRule],
    dt: float,
) -> Dict[str, float]:
    """
    Compute density transfer deltas from subtype switching rules.

    For each rule, if the source receptor's exposure_trace exceeds
    the threshold, compute a density transfer. Conservation is enforced:
    source_delta + target_delta = 0.

    Parameters
    ----------
    receptor_states : dict
        Map of receptor_id -> ReceptorState
    rules : list
        SubtypeSwitchRule instances
    dt : float
        Time step

    Returns
    -------
    dict
        Map of receptor_id -> rho_delta (cumulative across all rules)
    """
    deltas: Dict[str, float] = {}
    for rule in rules:
        source_state = receptor_states.get(rule.source_receptor_id)
        if source_state is None:
            continue
        if source_state.exposure_trace <= rule.exposure_threshold:
            continue
        # Compute transfer amount
        transfer = rule.rho_transfer_rate * dt
        transfer = min(transfer, rule.max_transfer_per_step)
        # Don't transfer more than the source has
        transfer = min(transfer, source_state.rho)
        if transfer <= 0.0:
            continue
        # Accumulate deltas
        deltas[rule.source_receptor_id] = deltas.get(rule.source_receptor_id, 0.0) - transfer
        deltas[rule.target_receptor_id] = deltas.get(rule.target_receptor_id, 0.0) + transfer
    return deltas


def apply_subtype_switch_deltas(
    receptor_states: Dict[str, ReceptorState],
    deltas: Dict[str, float],
) -> Dict[str, ReceptorState]:
    """
    Apply density transfer deltas to receptor states.

    Returns new dict with updated ReceptorState copies for affected
    receptors. Unaffected receptors are returned as-is (not copied).

    Parameters
    ----------
    receptor_states : dict
        Map of receptor_id -> ReceptorState
    deltas : dict
        Map of receptor_id -> rho_delta from compute_subtype_switch_deltas

    Returns
    -------
    dict
        Map of receptor_id -> ReceptorState (new copies for affected receptors)
    """
    result = dict(receptor_states)
    for receptor_id, rho_delta in deltas.items():
        state = receptor_states.get(receptor_id)
        if state is None:
            continue
        new_state = state.copy()
        new_state.update_density(rho_delta)
        result[receptor_id] = new_state
    return result
