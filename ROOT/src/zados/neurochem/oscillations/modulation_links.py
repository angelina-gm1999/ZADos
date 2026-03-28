"""
LEGACY MODULE — retained for backward compatibility with the batch
SimulationRunner and old Dopamine class only.

New code should use ``oscillations.oscillation_modulation`` instead,
which provides per-parameter pure functions aligned with PDF Appendix I.
"""
from __future__ import annotations
from typing import Dict


def modulate_parameters(base_params: Dict[str, float], oscillations: Dict[str, float]) -> Dict[str, float]:
    """
    Modulates neurotransmitter parameters based on oscillatory band activity.

    .. deprecated::
        Use the pure functions in ``oscillation_modulation.py`` instead
        (``modulate_release``, ``modulate_noise``, ``modulate_K_d``, etc.).
        This function is retained only for the legacy batch SimulationRunner.

    Args:
        base_params (dict): Dictionary of baseline parameters (e.g., βrew, βnov).
        oscillations (dict): Oscillatory power levels for EEG bands.

    Returns:
        dict: Modulated parameter values.
    """
    beta = oscillations.get("beta", 0.0)
    gamma = oscillations.get("gamma", 0.0)
    alpha = oscillations.get("alpha", 0.0)

    modulated = dict(base_params)

    # Example modulations (can expand per domain):
    if "βrew" in modulated:
        modulated["βrew"] *= 1.0 + 0.5 * gamma  # reward sensitivity ↑ with gamma

    if "βnov" in modulated:
        modulated["βnov"] *= 1.0 + 0.3 * beta   # novelty sensitivity ↑ with beta

    if "R0" in modulated:
        modulated["R0"] *= 1.0 - 0.2 * alpha    # baseline release ↓ with alpha

    return modulated
