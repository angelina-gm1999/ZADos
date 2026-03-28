"""
Plasticity event operators for neurosymbolic encoding (Appendix K.4).

Discrete plasticity events emitted as symbolic tokens:
- INT(R)          : internalization of receptor R
- UPR(R)          : upregulation of receptor R
- SWITCH(Ra->Rb)  : receptor subtype switching from Ra to Rb
- DSN(R)          : desensitization of receptor R
- REC(R)          : recovery of receptor R

Each operator has:
1. A condition checker (pure function) that decides whether the event fires.
2. An application function (pure function) that returns a new ReceptorState.
3. A PlasticityEvent frozen dataclass capturing the event log entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from zados.neurochem.state.receptor_state import ReceptorFunctionalState, ReceptorState


# ---------------------------------------------------------------------------
# Event data structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlasticityEvent:
    """Immutable record of a plasticity event (K.4.3)."""
    time: float
    operator: str           # "INT", "UPR", "SWITCH", "DSN", "REC"
    target: str             # receptor_id
    target_b: Optional[str] = None  # second receptor for SWITCH
    trigger_window: Optional[float] = None
    evidence: Optional[Dict[str, float]] = field(default=None, hash=False)
    pre_state: Optional[str] = None   # chi before
    post_state: Optional[str] = None  # chi after


# ---------------------------------------------------------------------------
# Condition checkers (pure functions)
# ---------------------------------------------------------------------------

def check_internalization_condition(
    saturation: float,
    time_in_state: float,
    theta_int: float = 0.8,
    t_int: float = 15.0,
) -> bool:
    """
    Check whether internalization should fire (K.4.1).

    Fires when saturation exceeds theta_int for at least t_int time units.

    Parameters
    ----------
    saturation : float
        Current receptor saturation S_ij in [0, 1].
    time_in_state : float
        Duration receptor has been in the ACTIVE state.
    theta_int : float
        Saturation threshold for internalization.
    t_int : float
        Minimum time in active state under high saturation.
    """
    return saturation >= theta_int and time_in_state >= t_int


def check_upregulation_condition(
    saturation: float,
    time_in_state: float,
    epsilon_upr: float = 0.1,
    t_upr: float = 20.0,
) -> bool:
    """
    Check whether upregulation should fire (K.4.2).

    Fires when saturation is below epsilon_upr for at least t_upr time units.

    Parameters
    ----------
    saturation : float
        Current receptor saturation S_ij in [0, 1].
    time_in_state : float
        Duration receptor has been in the ACTIVE state.
    epsilon_upr : float
        Low-saturation threshold for upregulation.
    t_upr : float
        Minimum time in active state under low saturation.
    """
    return saturation <= epsilon_upr and time_in_state >= t_upr


def check_desensitization_condition(
    saturation: float,
    time_in_state: float,
    theta_dsn: float = 0.7,
    t_dsn: float = 10.0,
) -> bool:
    """
    Check whether desensitization should fire (K.4).

    Fires when saturation exceeds theta_dsn for at least t_dsn time units
    (shorter/milder threshold than full internalization).
    """
    return saturation >= theta_dsn and time_in_state >= t_dsn


# ---------------------------------------------------------------------------
# Application functions (pure — return new state copies)
# ---------------------------------------------------------------------------

def apply_internalization(
    state: ReceptorState,
    kappa_int: float = 0.3,
    kappa_gamma: float = 0.1,
) -> ReceptorState:
    """
    Apply INT(R): reduce density, reduce G-protein coupling, set chi=INTERNALIZED (K.4.1).

    Parameters
    ----------
    state : ReceptorState
        Current receptor state (not mutated).
    kappa_int : float
        Fraction of rho to remove.
    kappa_gamma : float
        Reduction in gamma_gprotein.

    Returns
    -------
    ReceptorState
        New state with reduced rho, reduced gamma, chi=INTERNALIZED, time_in_state reset.
    """
    new = state.copy()
    new.rho = max(0.0, min(1.0, state.rho * (1.0 - kappa_int)))
    new.gamma_gprotein = max(0.0, min(1.0, state.gamma_gprotein - kappa_gamma))
    new.chi = ReceptorFunctionalState.INTERNALIZED
    new.time_in_state = 0.0
    return new


def apply_upregulation(
    state: ReceptorState,
    delta_rho: float = 0.1,
    kappa_upr: float = 0.1,
) -> ReceptorState:
    """
    Apply UPR(R): increase density and sensitivity, set chi=UPREGULATED (K.4.2).

    Parameters
    ----------
    state : ReceptorState
        Current receptor state (not mutated).
    delta_rho : float
        Additive increase to rho.
    kappa_upr : float
        Additive increase to sigma.

    Returns
    -------
    ReceptorState
        New state with increased rho, increased sigma, chi=UPREGULATED.
    """
    new = state.copy()
    new.rho = max(0.0, min(1.0, state.rho + delta_rho))
    new.sigma = max(0.0, min(1.0, state.sigma + kappa_upr))
    new.chi = ReceptorFunctionalState.UPREGULATED
    new.time_in_state = 0.0
    return new


def apply_desensitization(
    state: ReceptorState,
    kappa_dsn: float = 0.3,
) -> ReceptorState:
    """
    Apply DSN(R): reduce sensitivity, set chi=DESENSITIZED.

    Parameters
    ----------
    state : ReceptorState
        Current receptor state (not mutated).
    kappa_dsn : float
        Fraction of sigma to remove.

    Returns
    -------
    ReceptorState
        New state with reduced sigma, chi=DESENSITIZED.
    """
    new = state.copy()
    new.sigma = max(0.0, min(1.0, state.sigma * (1.0 - kappa_dsn)))
    new.chi = ReceptorFunctionalState.DESENSITIZED
    new.time_in_state = 0.0
    return new


def apply_recovery(
    state: ReceptorState,
    target_sigma: float = 1.0,
    recovery_rate: float = 0.5,
) -> ReceptorState:
    """
    Apply REC(R): recover sensitivity toward target, set chi=ACTIVE.

    Parameters
    ----------
    state : ReceptorState
        Current receptor state (not mutated).
    target_sigma : float
        Target sensitivity to recover toward.
    recovery_rate : float
        Fraction of gap to close in one step.

    Returns
    -------
    ReceptorState
        New state with recovered sigma, chi=ACTIVE.
    """
    new = state.copy()
    gap = target_sigma - state.sigma
    new.sigma = max(0.0, min(1.0, state.sigma + gap * recovery_rate))
    new.chi = ReceptorFunctionalState.ACTIVE
    new.time_in_state = 0.0
    return new


def apply_switch(
    state_a: ReceptorState,
    state_b: ReceptorState,
    kappa_sw: float = 0.3,
) -> Tuple[ReceptorState, ReceptorState]:
    """
    Apply SWITCH(Ra->Rb): reallocate density from Ra to Rb (K.4.3).

    Transfers kappa_sw fraction of Ra's density to Rb.
    Also boosts Rb's sensitivity slightly.

    Parameters
    ----------
    state_a : ReceptorState
        Source receptor state (not mutated).
    state_b : ReceptorState
        Destination receptor state (not mutated).
    kappa_sw : float
        Fraction of density to transfer from A to B.

    Returns
    -------
    tuple[ReceptorState, ReceptorState]
        (new_state_a, new_state_b)
    """
    transfer = state_a.rho * kappa_sw

    new_a = state_a.copy()
    new_a.rho = max(0.0, min(1.0, state_a.rho - transfer))

    new_b = state_b.copy()
    new_b.rho = max(0.0, min(1.0, state_b.rho + transfer))
    # Sensitivity boost: small fraction of transfer
    new_b.sigma = max(0.0, min(1.0, state_b.sigma + transfer * 0.1))

    return new_a, new_b


# ---------------------------------------------------------------------------
# Encoding (roundtrip support)
# ---------------------------------------------------------------------------

def encode_plasticity_event(event: PlasticityEvent) -> str:
    """
    Encode a PlasticityEvent to mastergrid operator string.

    Returns e.g. "INT(D2)", "UPR(OXTR)", "SWITCH(D1->D3)".
    """
    if event.operator == "SWITCH" and event.target_b:
        return f"SWITCH({event.target}->{event.target_b})"
    return f"{event.operator}({event.target})"
