from __future__ import annotations
from typing import Optional


def reuptake(
    C: float,
    F: float,
    ku0: float,
    gamma: float,
    modulator: Optional[str] = "DA"
) -> float:
    """
    Computes the reuptake rate for a given neurotransmitter.

    Args:
        C (float): Current concentration.
        F (float): Fatigue factor (affects uptake efficiency).
        ku0 (float): Baseline reuptake constant.
        gamma (float): Fatigue sensitivity (0–1).
        modulator (str): Identifier (for future customization).

    Returns:
        float: Reuptake rate at current time step.
    """
    fatigue_scaling = 1.0 - gamma * F
    ku = ku0 * max(fatigue_scaling, 0.0)

    return ku * C
