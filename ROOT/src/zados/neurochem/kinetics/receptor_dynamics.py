from __future__ import annotations

from typing import Dict, List, Optional
import math

from zados.neurochem.state.receptor_state import ReceptorState, ReceptorFunctionalState


# ---------------------------------------------------------------------------
# Default CTMC transition thresholds (all overridable via config dicts)
# ---------------------------------------------------------------------------

DEFAULT_RECEPTOR_THRESHOLDS: Dict[str, float] = {
    # ACTIVE -> DESENSITIZED
    "theta_desens": 0.7,          # saturation must exceed this
    "t0_desens": 5.0,             # minimum time in ACTIVE with high saturation

    # DESENSITIZED -> INTERNALIZED
    "theta_intern": 15.0,         # exposure_trace must exceed this

    # DESENSITIZED -> ACTIVE (recovery)
    "epsilon_recovery": 0.3,      # saturation must drop below this
    "t_recovery": 10.0,           # minimum time in DESENSITIZED

    # ACTIVE -> UPREGULATED
    "epsilon_upreg": 0.1,         # saturation must stay below this
    "t0_upreg": 20.0,             # minimum time in ACTIVE with low saturation

    # UPREGULATED -> ACTIVE
    "theta_upreg_exit": 0.4,      # saturation must exceed this
    "t_upreg_exit": 5.0,          # minimum time in UPREGULATED

    # INTERNALIZED -> ACTIVE (recycling)
    "t_recycle": 50.0,            # minimum time in INTERNALIZED

    # State-effect multipliers
    "sigma_desens_factor": 0.5,   # sigma *= this on entering DESENSITIZED
    "rho_intern_factor": 0.7,     # rho *= this on entering INTERNALIZED
    "sigma_intern_factor": 0.3,   # sigma *= this on entering INTERNALIZED
    "sigma_upreg_factor": 1.3,    # sigma *= this on entering UPREGULATED
    "rho_upreg_factor": 1.2,      # rho *= this on entering UPREGULATED
    "sigma_recovery_rate": 0.05,  # rate of sigma recovery toward 1.0 per unit time

    # Oscillation modulation
    "beta_desens_scaling": 0.3,   # how much beta accelerates desensitization

    # G-protein coupling dynamics
    "gamma_degrade_rate": 0.02,       # rate of gamma degradation per unit time
    "gamma_recover_rate": 0.01,       # rate of gamma recovery per unit time
    "gamma_degrade_threshold": 0.5,   # saturation above which gamma degrades
    "gamma_min": 0.05,                # floor for gamma (never fully decouples)
}


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def compute_effective_concentration(
    C_tonic: float,
    C_phasic: float,
    lambda_loc: float,
) -> float:
    """
    Compute the effective concentration seen by a receptor based on
    its synaptic localization.

    C_effective = C_tonic + (1 - lambda_loc) * C_phasic

    - lambda_loc=0 (presynaptic): sees full C_tonic + C_phasic
    - lambda_loc=0.5 (synaptic): sees C_tonic + 0.5 * C_phasic
    - lambda_loc=1.0 (extrasynaptic): sees only C_tonic (volume transmission)

    From PDF Appendix D.2.

    Parameters
    ----------
    C_tonic : float
        Tonic (volume) concentration component
    C_phasic : float
        Phasic (synaptic burst) concentration component
    lambda_loc : float
        Localization: 0=presynaptic, 0.5=synaptic, 1=extrasynaptic

    Returns
    -------
    float
        Effective concentration (non-negative)
    """
    return max(0.0, C_tonic + (1.0 - lambda_loc) * C_phasic)


def compute_saturation(
    concentration: float,
    K_d: float,
) -> float:
    """
    Compute receptor saturation using Michaelis-Menten kinetics.

    S = C / (C + K_d)

    Parameters
    ----------
    concentration : float
        Ligand concentration C_i(t)
    K_d : float
        Dissociation constant (half-saturation concentration)

    Returns
    -------
    float
        Saturation in [0, 1]
    """
    if concentration <= 0.0 or K_d <= 0.0:
        return 0.0
    return concentration / (concentration + K_d)


def update_exposure_trace(
    exposure_trace: float,
    saturation: float,
    dt: float,
    tau: float = 10.0,
) -> float:
    """
    Update exposure trace with exponential decay.

    E(t+dt) = E(t) * exp(-dt/tau) + S(t) * dt

    Parameters
    ----------
    exposure_trace : float
        Current exposure trace value
    saturation : float
        Current receptor saturation S_ij(t)
    dt : float
        Time step
    tau : float, default=10.0
        Decay time constant

    Returns
    -------
    float
        Updated exposure trace (non-negative)
    """
    decay_factor = math.exp(-dt / tau)
    return max(0.0, exposure_trace * decay_factor + saturation * dt)


def _merge_thresholds(overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Merge caller-supplied thresholds with defaults.

    Parameters
    ----------
    overrides : dict, optional
        Threshold overrides

    Returns
    -------
    dict
        Complete threshold dict
    """
    merged = dict(DEFAULT_RECEPTOR_THRESHOLDS)
    if overrides:
        merged.update(overrides)
    return merged


def compute_transition_rates(
    current_state: ReceptorFunctionalState,
    saturation: float,
    exposure_trace: float,
    time_in_state: float,
    thresholds: Optional[Dict[str, float]] = None,
    beta_amplitude: float = 0.0,
    osc_amplitudes: Optional[Dict[str, float]] = None,
    transition_specs: Optional[Dict[str, list]] = None,
    threshold_modulation_specs: Optional[Dict[str, list]] = None,
) -> Dict[ReceptorFunctionalState, float]:
    """
    Compute transition rates for all possible next states.

    Uses deterministic threshold-based logic: a transition is eligible
    (rate = 1.0) when all conditions are met, otherwise rate = 0.0.

    Parameters
    ----------
    current_state : ReceptorFunctionalState
        Current CTMC state chi
    saturation : float
        Current receptor saturation S in [0, 1]
    exposure_trace : float
        Cumulative exposure trace E
    time_in_state : float
        Duration spent in current state T
    thresholds : dict, optional
        Override default thresholds
    beta_amplitude : float, default=0.0
        Beta oscillation amplitude; higher beta accelerates desensitization
        (legacy single-band path)
    osc_amplitudes : dict, optional
        Map of band name -> amplitude for multi-band modulation
    transition_specs : dict, optional
        Map of transition name -> list of TransitionBandSpec.
        Keys: "desensitization", "internalization", "recovery",
        "upregulation", "upreg_exit", "recycling".
        When present, timing thresholds are divided by the computed
        multiplier: t_eff = t_base / m(t).
    threshold_modulation_specs : dict, optional
        Map of threshold name -> list of ThresholdBandSpec.
        Keys: "theta_desens", "epsilon_upreg", "epsilon_recovery",
        "theta_upreg_exit".
        When present, saturation thresholds are shifted by oscillations.

    Returns
    -------
    dict
        Map of ReceptorFunctionalState -> rate (0.0 or 1.0)
    """
    from zados.neurochem.oscillations.transition_modulation import (
        compute_transition_multiplier,
        modulate_threshold,
    )

    th = _merge_thresholds(thresholds)
    rates: Dict[ReceptorFunctionalState, float] = {}

    # Helper: get timing multiplier for a transition
    def _timing_multiplier(transition_name: str) -> float:
        if osc_amplitudes and transition_specs and transition_name in transition_specs:
            return compute_transition_multiplier(
                osc_amplitudes, transition_specs[transition_name],
            )
        return 1.0

    # Helper: get modulated threshold
    def _modulated_threshold(threshold_key: str, base_val: float) -> float:
        if osc_amplitudes and threshold_modulation_specs and threshold_key in threshold_modulation_specs:
            return modulate_threshold(
                base_val, osc_amplitudes, threshold_modulation_specs[threshold_key],
            )
        return base_val

    if current_state == ReceptorFunctionalState.ACTIVE:
        # ACTIVE -> DESENSITIZED (overstimulation)
        # Legacy: beta scales t0_desens directly
        t0_desens_eff = th["t0_desens"] * (1.0 - th["beta_desens_scaling"] * beta_amplitude)
        t0_desens_eff = max(0.0, t0_desens_eff)
        # Multi-band: divide by transition multiplier
        desens_mult = _timing_multiplier("desensitization")
        if desens_mult > 0:
            t0_desens_eff = t0_desens_eff / desens_mult
        theta_desens_eff = _modulated_threshold("theta_desens", th["theta_desens"])
        if saturation > theta_desens_eff and time_in_state > t0_desens_eff:
            rates[ReceptorFunctionalState.DESENSITIZED] = 1.0

        # ACTIVE -> UPREGULATED (understimulation)
        upreg_mult = _timing_multiplier("upregulation")
        t0_upreg_eff = th["t0_upreg"] / upreg_mult if upreg_mult > 0 else th["t0_upreg"]
        epsilon_upreg_eff = _modulated_threshold("epsilon_upreg", th["epsilon_upreg"])
        if saturation < epsilon_upreg_eff and time_in_state > t0_upreg_eff:
            rates[ReceptorFunctionalState.UPREGULATED] = 1.0

    elif current_state == ReceptorFunctionalState.DESENSITIZED:
        # DESENSITIZED -> INTERNALIZED (prolonged exposure)
        intern_mult = _timing_multiplier("internalization")
        # For internalization, higher multiplier lowers effective threshold
        theta_intern_eff = th["theta_intern"] / intern_mult if intern_mult > 0 else th["theta_intern"]
        if exposure_trace > theta_intern_eff:
            rates[ReceptorFunctionalState.INTERNALIZED] = 1.0

        # DESENSITIZED -> ACTIVE (recovery)
        recovery_mult = _timing_multiplier("recovery")
        t_recovery_eff = th["t_recovery"] / recovery_mult if recovery_mult > 0 else th["t_recovery"]
        epsilon_recovery_eff = _modulated_threshold("epsilon_recovery", th["epsilon_recovery"])
        if saturation < epsilon_recovery_eff and time_in_state > t_recovery_eff:
            rates[ReceptorFunctionalState.ACTIVE] = 1.0

    elif current_state == ReceptorFunctionalState.UPREGULATED:
        # UPREGULATED -> ACTIVE (sufficient stimulation returns)
        upreg_exit_mult = _timing_multiplier("upreg_exit")
        t_upreg_exit_eff = th["t_upreg_exit"] / upreg_exit_mult if upreg_exit_mult > 0 else th["t_upreg_exit"]
        theta_upreg_exit_eff = _modulated_threshold("theta_upreg_exit", th["theta_upreg_exit"])
        if saturation > theta_upreg_exit_eff and time_in_state > t_upreg_exit_eff:
            rates[ReceptorFunctionalState.ACTIVE] = 1.0

    elif current_state == ReceptorFunctionalState.INTERNALIZED:
        # INTERNALIZED -> ACTIVE (slow recycling)
        recycle_mult = _timing_multiplier("recycling")
        t_recycle_eff = th["t_recycle"] / recycle_mult if recycle_mult > 0 else th["t_recycle"]
        if time_in_state > t_recycle_eff:
            rates[ReceptorFunctionalState.ACTIVE] = 1.0

    return rates


# Priority ordering for transition selection when multiple are eligible.
# Protective responses (desensitization, internalization) take priority.
_TRANSITION_PRIORITY = [
    ReceptorFunctionalState.DESENSITIZED,
    ReceptorFunctionalState.INTERNALIZED,
    ReceptorFunctionalState.UPREGULATED,
    ReceptorFunctionalState.ACTIVE,
]


def select_transition(
    transition_rates: Dict[ReceptorFunctionalState, float],
) -> Optional[ReceptorFunctionalState]:
    """
    Select which transition to execute based on rates.

    In deterministic mode, selects the highest-priority transition
    with rate > 0. Returns None if no transition is eligible.

    Parameters
    ----------
    transition_rates : dict
        Map of target state -> rate from compute_transition_rates

    Returns
    -------
    ReceptorFunctionalState or None
        The selected next state, or None if no transition occurs
    """
    for state in _TRANSITION_PRIORITY:
        if transition_rates.get(state, 0.0) > 0.0:
            return state
    return None


def apply_state_transition_effects(
    new_state: ReceptorFunctionalState,
    current_sigma: float,
    current_rho: float,
    thresholds: Optional[Dict[str, float]] = None,
) -> tuple:
    """
    Compute updated sigma and rho after a state transition.

    Parameters
    ----------
    new_state : ReceptorFunctionalState
        The state being transitioned INTO
    current_sigma : float
        Current sensitivity
    current_rho : float
        Current density
    thresholds : dict, optional
        Override default factors

    Returns
    -------
    tuple[float, float]
        (new_sigma, new_rho) both clamped to [0, 1]
    """
    th = _merge_thresholds(thresholds)
    sigma = current_sigma
    rho = current_rho

    if new_state == ReceptorFunctionalState.DESENSITIZED:
        sigma *= th["sigma_desens_factor"]
    elif new_state == ReceptorFunctionalState.INTERNALIZED:
        sigma *= th["sigma_intern_factor"]
        rho *= th["rho_intern_factor"]
    elif new_state == ReceptorFunctionalState.UPREGULATED:
        sigma *= th["sigma_upreg_factor"]
        rho *= th["rho_upreg_factor"]
    # ACTIVE recovery: no immediate multiplier (gradual via apply_sensitivity_recovery)

    sigma = max(0.0, min(1.0, sigma))
    rho = max(0.0, min(1.0, rho))
    return sigma, rho


def apply_sensitivity_recovery(
    sigma: float,
    dt: float,
    recovery_rate: float = 0.05,
) -> float:
    """
    Gradually recover sensitivity toward 1.0 while in ACTIVE state.

    sigma(t+dt) = sigma(t) + recovery_rate * (1.0 - sigma(t)) * dt

    Parameters
    ----------
    sigma : float
        Current sensitivity
    dt : float
        Time step
    recovery_rate : float, default=0.05
        Rate of recovery toward 1.0

    Returns
    -------
    float
        Updated sensitivity, clamped to [0, 1]
    """
    new_sigma = sigma + recovery_rate * (1.0 - sigma) * dt
    return max(0.0, min(1.0, new_sigma))


def update_gamma_gprotein(
    gamma: float,
    saturation: float,
    dt: float,
    k_degrade: float = 0.02,
    k_recover: float = 0.01,
    threshold: float = 0.5,
    gamma_min: float = 0.05,
) -> float:
    """
    Update G-protein coupling efficacy based on receptor saturation.

    Under sustained high saturation, G-protein coupling fatigues
    (gamma decreases). When saturation drops, gamma recovers toward 1.0.

    From PDF Appendix D.1.4.

    Parameters
    ----------
    gamma : float
        Current G-protein coupling efficacy in [gamma_min, 1.0]
    saturation : float
        Current receptor saturation S in [0, 1]
    dt : float
        Time step
    k_degrade : float, default=0.02
        Degradation rate per unit time
    k_recover : float, default=0.01
        Recovery rate per unit time
    threshold : float, default=0.5
        Saturation above which gamma degrades
    gamma_min : float, default=0.05
        Minimum gamma (never fully decouples)

    Returns
    -------
    float
        Updated gamma, clamped to [gamma_min, 1.0]
    """
    if saturation > threshold:
        gamma -= k_degrade * saturation * dt
    else:
        gamma += k_recover * (1.0 - gamma) * dt
    return max(gamma_min, min(1.0, gamma))


def step_receptor_dynamics(
    receptor_state: ReceptorState,
    concentration: float,
    K_d: float,
    dt: float,
    thresholds: Optional[Dict[str, float]] = None,
    beta_amplitude: float = 0.0,
    exposure_tau: float = 10.0,
    osc_amplitudes: Optional[Dict[str, float]] = None,
    transition_specs: Optional[Dict[str, list]] = None,
    threshold_modulation_specs: Optional[Dict[str, list]] = None,
) -> ReceptorState:
    """
    Execute one complete receptor dynamics step.

    Orchestrates: saturation -> exposure trace update -> transition check ->
    state effects -> sensitivity recovery -> time increment.

    Returns a NEW ReceptorState (does not mutate the input).

    Parameters
    ----------
    receptor_state : ReceptorState
        Current receptor state (will be copied, not mutated)
    concentration : float
        Parent neurotransmitter total concentration C_i(t)
    K_d : float
        Dissociation constant for this receptor
    dt : float
        Time step
    thresholds : dict, optional
        CTMC transition thresholds (defaults used if not provided)
    beta_amplitude : float, default=0.0
        Beta oscillation band amplitude for modulating desensitization
        (legacy single-band path)
    exposure_tau : float, default=10.0
        Time constant for exposure trace decay
    osc_amplitudes : dict, optional
        Band amplitudes for multi-band CTMC modulation
    transition_specs : dict, optional
        Per-transition TransitionBandSpec lists
    threshold_modulation_specs : dict, optional
        Per-threshold ThresholdBandSpec lists

    Returns
    -------
    ReceptorState
        Updated receptor state (new object)
    """
    # Work on a copy to avoid mutating the input
    state = receptor_state.copy()

    # 1. Compute saturation
    sat = compute_saturation(concentration, K_d)

    # 2. Update exposure trace
    state.exposure_trace = update_exposure_trace(
        state.exposure_trace, sat, dt, tau=exposure_tau,
    )

    # 2b. Update G-protein coupling dynamics
    th = _merge_thresholds(thresholds)
    state.gamma_gprotein = update_gamma_gprotein(
        state.gamma_gprotein, sat, dt,
        k_degrade=th["gamma_degrade_rate"],
        k_recover=th["gamma_recover_rate"],
        threshold=th["gamma_degrade_threshold"],
        gamma_min=th["gamma_min"],
    )

    # 3. Compute transition rates
    rates = compute_transition_rates(
        current_state=state.chi,
        saturation=sat,
        exposure_trace=state.exposure_trace,
        time_in_state=state.time_in_state,
        thresholds=thresholds,
        beta_amplitude=beta_amplitude,
        osc_amplitudes=osc_amplitudes,
        transition_specs=transition_specs,
        threshold_modulation_specs=threshold_modulation_specs,
    )

    # 4. Select transition
    next_state = select_transition(rates)

    if next_state is not None:
        # 5. Apply state effects (sigma, rho changes)
        new_sigma, new_rho = apply_state_transition_effects(
            next_state, state.sigma, state.rho, thresholds,
        )
        state.sigma = new_sigma
        state.rho = new_rho
        state.chi = next_state
        state.time_in_state = 0.0
    else:
        # 6. No transition: if ACTIVE, apply gradual sensitivity recovery
        if state.chi == ReceptorFunctionalState.ACTIVE:
            th = _merge_thresholds(thresholds)
            state.sigma = apply_sensitivity_recovery(
                state.sigma, dt, th["sigma_recovery_rate"],
            )

    # 7. Increment time in state
    state.time_in_state += dt

    return state
