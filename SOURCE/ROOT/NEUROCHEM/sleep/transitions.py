"""
Pharmacodynamic transition logic for sleep phase changes.

Pure functions implementing the smooth exponential approach used
for all NT and oscillatory transitions during sleep entry, phase
changes, and waking return (Spec §5).

Update rule:
    C_i(t + dt) = C_i(t) + (C_target - C_i(t)) * k * dt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TransitionConfig:
    """Rate constants for pharmacodynamic transitions.

    Attributes
    ----------
    k_enter : float
        Waking -> triage transition rate.
    k_phase : float
        Inter-phase transition rate (triage -> rem, rem -> dream).
    k_exit : float
        Sleep -> waking return rate (slower than entry).
    k_fast : float
        Fast monoamine collapse rate (NE, 5-HT for REM -> dream).
    """
    k_enter: float = 0.2
    k_phase: float = 0.25
    k_exit: float = 0.1
    k_fast: float = 0.4


DEFAULT_TRANSITION_CONFIG = TransitionConfig()

# NTs that use fast monoamine collapse rate for REM -> dream transition
FAST_COLLAPSE_NTS = frozenset({"NE", "5HT"})


def compute_transition_step(
    current: float,
    target: float,
    k: float,
    dt: float,
) -> float:
    """Compute one step of exponential approach toward target.

    C(t + dt) = C(t) + (C_target - C(t)) * k * dt

    Parameters
    ----------
    current : float
        Current value.
    target : float
        Target value.
    k : float
        Transition rate constant.
    dt : float
        Time step.

    Returns
    -------
    float
        Updated value, clamped to [0, 1].
    """
    new_val = current + (target - current) * k * dt
    return max(0.0, min(1.0, new_val))


def transition_nt_baselines(
    current_baselines: Dict[str, float],
    target_baselines: Dict[str, float],
    k: float,
    dt: float,
    fast_nts: frozenset = FAST_COLLAPSE_NTS,
    k_fast: float = 0.4,
) -> Dict[str, float]:
    """Transition all NT baselines one step toward targets.

    NTs in ``fast_nts`` use ``k_fast`` instead of ``k`` (for rapid
    monoamine collapse during REM -> dream transition).

    Parameters
    ----------
    current_baselines : dict
        Current NT name -> tonic baseline.
    target_baselines : dict
        Target NT name -> tonic baseline.
    k : float
        Standard transition rate.
    dt : float
        Time step.
    fast_nts : frozenset
        NTs that use fast rate.
    k_fast : float
        Fast transition rate for monoamine collapse.

    Returns
    -------
    dict
        Updated NT baselines.
    """
    result = {}
    for nt_name, current_val in current_baselines.items():
        target_val = target_baselines.get(nt_name, current_val)
        rate = k_fast if nt_name in fast_nts else k
        result[nt_name] = compute_transition_step(current_val, target_val, rate, dt)
    return result


def transition_osc_config(
    current_config: Dict[str, float],
    target_config: Dict[str, float],
    k: float,
    dt: float,
) -> Dict[str, float]:
    """Transition all oscillatory band amplitudes toward targets.

    Parameters
    ----------
    current_config : dict
        Current band name -> amplitude.
    target_config : dict
        Target band name -> amplitude.
    k : float
        Transition rate.
    dt : float
        Time step.

    Returns
    -------
    dict
        Updated oscillatory config.
    """
    result = {}
    for band_name, current_val in current_config.items():
        target_val = target_config.get(band_name, current_val)
        result[band_name] = compute_transition_step(current_val, target_val, k, dt)
    return result


def check_triage_to_rem_conditions(
    nt_baselines: Dict[str, float],
    osc_config: Dict[str, float],
) -> bool:
    """Check if conditions are met for triage -> REM processing transition.

    Condition (Spec §5.2):
        5-HT > 0.50 AND ACh < 0.30 AND phi_delta > 0.60 AND phi_sigma > 0.55

    Parameters
    ----------
    nt_baselines : dict
        Current NT baselines.
    osc_config : dict
        Current oscillatory config.

    Returns
    -------
    bool
        True if transition conditions met.
    """
    sht = nt_baselines.get("5HT", 0.0)
    ach = nt_baselines.get("ACh", 0.0)
    phi_delta = osc_config.get("delta", 0.0)
    phi_sigma = osc_config.get("sigma", 0.0)

    return sht > 0.50 and ach < 0.30 and phi_delta > 0.60 and phi_sigma > 0.55


def check_rem_to_dream_conditions(
    nt_baselines: Dict[str, float],
    osc_config: Dict[str, float],
    desensitization_flag: bool = False,
    stagnated_queue_nonempty: bool = False,
) -> bool:
    """Check if conditions are met for REM processing -> dream transition.

    Condition (Spec §5.3):
        5-HT1A desensitization flag
        OR (stagnated concept queue non-empty AND 5-HT < 0.40)

    Parameters
    ----------
    nt_baselines : dict
        Current NT baselines.
    osc_config : dict
        Current oscillatory config (unused but kept for API consistency).
    desensitization_flag : bool
        True if 5-HT1A has desensitized.
    stagnated_queue_nonempty : bool
        True if stagnated concept queue has items.

    Returns
    -------
    bool
        True if transition conditions met.
    """
    if desensitization_flag:
        return True

    sht = nt_baselines.get("5HT", 0.0)
    return stagnated_queue_nonempty and sht < 0.40
