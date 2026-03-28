"""
GABA-A dreambox containment monitoring (Spec §3.4).

Pure functions for verifying dream state integrity:
- GABA-A containment gate (atonia analog)
- NE/5-HT floor enforcement
- NE receptor upregulation cap
"""

from __future__ import annotations


def check_containment(
    gaba_a_saturation: float,
    threshold: float = 0.55,
) -> bool:
    """Check if GABA-A dreambox containment gate is closed (safe).

    Containment condition: S_GABA-A >= threshold => gate CLOSED (safe).
    If gate OPEN (below threshold), dream outputs could escape quarantine.

    Parameters
    ----------
    gaba_a_saturation : float
        Current GABA-A receptor saturation.
    threshold : float
        Containment threshold. Default 0.55 per spec.

    Returns
    -------
    bool
        True if containment is intact (gate CLOSED, safe).
    """
    return gaba_a_saturation >= threshold


def check_dream_state_validity(
    ne_baseline: float,
    sht_baseline: float,
    ceiling: float = 0.10,
) -> bool:
    """Check if NE and 5-HT are at valid dream-mode floor levels.

    During dream mode, NE and 5-HT must stay near-zero. If either
    rises above ceiling, the system is not in a valid dream state
    (Spec §5.3).

    Parameters
    ----------
    ne_baseline : float
        Current NE tonic baseline.
    sht_baseline : float
        Current 5-HT tonic baseline.
    ceiling : float
        Maximum allowed value. Default 0.10 per spec.

    Returns
    -------
    bool
        True if dream state is valid (both NE and 5-HT at floor).
    """
    return ne_baseline <= ceiling and sht_baseline <= ceiling


def check_ne_upregulation_cap(
    receptor_rho: float,
    max_multiplier: float = 1.5,
) -> float:
    """Cap NE receptor upregulation during sleep.

    NE receptors upregulate during sleep (low NE exposure), but
    must be capped at max_multiplier to prevent pathological
    hypersensitivity on waking return (Spec §3.2).

    Parameters
    ----------
    receptor_rho : float
        Current receptor density multiplier.
    max_multiplier : float
        Maximum allowed density multiplier. Default 1.5 per spec.

    Returns
    -------
    float
        Capped receptor density multiplier.
    """
    return min(receptor_rho, max_multiplier)
