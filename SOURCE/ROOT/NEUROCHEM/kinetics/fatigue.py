from __future__ import annotations


def fatigue(
    F: float,
    C: float,
    epsilon: float = 0.01,
    decay: float = 0.001,
) -> float:
    """
    Updates fatigue level F based on neurotransmitter concentration C.

    Args:
        F (float): Current fatigue level.
        C (float): Current concentration.
        epsilon (float): Fatigue accumulation rate.
        decay (float): Fatigue decay rate per timestep.

    Returns:
        float: Updated fatigue level.
    """
    dF = epsilon * C - decay * F
    F_next = F + dF
    return max(F_next, 0.0)
