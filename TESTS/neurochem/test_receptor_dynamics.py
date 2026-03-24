import pytest
import math
from zados.neurochem.state.receptor_state import ReceptorState, ReceptorFunctionalState
from zados.neurochem.kinetics.receptor_dynamics import (
    compute_saturation,
    update_exposure_trace,
    compute_transition_rates,
    select_transition,
    apply_state_transition_effects,
    apply_sensitivity_recovery,
    step_receptor_dynamics,
    DEFAULT_RECEPTOR_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# compute_saturation
# ---------------------------------------------------------------------------

def test_compute_saturation_half_saturation():
    """When C == K_d, saturation is 0.5."""
    assert compute_saturation(0.5, 0.5) == pytest.approx(0.5)


def test_compute_saturation_zero_concentration():
    """Zero concentration gives zero saturation."""
    assert compute_saturation(0.0, 0.5) == 0.0


def test_compute_saturation_negative_concentration():
    """Negative concentration gives zero saturation."""
    assert compute_saturation(-0.1, 0.5) == 0.0


def test_compute_saturation_high_concentration():
    """High concentration approaches 1.0."""
    sat = compute_saturation(100.0, 1.0)
    assert sat == pytest.approx(100.0 / 101.0)
    assert sat > 0.99


# ---------------------------------------------------------------------------
# update_exposure_trace
# ---------------------------------------------------------------------------

def test_update_exposure_trace_accumulates():
    """Exposure trace increases with positive saturation."""
    result = update_exposure_trace(0.0, 0.8, dt=1.0, tau=10.0)
    assert result == pytest.approx(0.8)


def test_update_exposure_trace_decays():
    """Exposure trace decays when saturation is zero."""
    result = update_exposure_trace(10.0, 0.0, dt=1.0, tau=10.0)
    expected = 10.0 * math.exp(-1.0 / 10.0)
    assert result == pytest.approx(expected)


def test_update_exposure_trace_combined():
    """Exposure trace combines decay and accumulation."""
    result = update_exposure_trace(5.0, 0.6, dt=0.5, tau=10.0)
    expected = 5.0 * math.exp(-0.5 / 10.0) + 0.6 * 0.5
    assert result == pytest.approx(expected)


def test_update_exposure_trace_non_negative():
    """Exposure trace never goes negative."""
    result = update_exposure_trace(0.0, 0.0, dt=1.0, tau=10.0)
    assert result >= 0.0


# ---------------------------------------------------------------------------
# compute_transition_rates
# ---------------------------------------------------------------------------

def test_active_to_desensitized_eligible():
    """Transition ACTIVE->DESENSITIZED when S > theta and T > t0."""
    rates = compute_transition_rates(
        current_state=ReceptorFunctionalState.ACTIVE,
        saturation=0.8,  # > 0.7 default theta_desens
        exposure_trace=0.0,
        time_in_state=6.0,  # > 5.0 default t0_desens
    )
    assert ReceptorFunctionalState.DESENSITIZED in rates
    assert rates[ReceptorFunctionalState.DESENSITIZED] == 1.0


def test_active_to_desensitized_not_eligible_low_saturation():
    """No desensitization when saturation below threshold."""
    rates = compute_transition_rates(
        current_state=ReceptorFunctionalState.ACTIVE,
        saturation=0.5,  # < 0.7
        exposure_trace=0.0,
        time_in_state=10.0,
    )
    assert ReceptorFunctionalState.DESENSITIZED not in rates


def test_active_to_desensitized_not_eligible_insufficient_time():
    """No desensitization when time in state too short."""
    rates = compute_transition_rates(
        current_state=ReceptorFunctionalState.ACTIVE,
        saturation=0.9,
        exposure_trace=0.0,
        time_in_state=2.0,  # < 5.0
    )
    assert ReceptorFunctionalState.DESENSITIZED not in rates


def test_active_to_upregulated_eligible():
    """Transition ACTIVE->UPREGULATED when S < epsilon and T > t0."""
    rates = compute_transition_rates(
        current_state=ReceptorFunctionalState.ACTIVE,
        saturation=0.05,  # < 0.1 default epsilon_upreg
        exposure_trace=0.0,
        time_in_state=25.0,  # > 20.0 default t0_upreg
    )
    assert ReceptorFunctionalState.UPREGULATED in rates
    assert rates[ReceptorFunctionalState.UPREGULATED] == 1.0


def test_desensitized_to_internalized_eligible():
    """Transition DESENSITIZED->INTERNALIZED when E > theta."""
    rates = compute_transition_rates(
        current_state=ReceptorFunctionalState.DESENSITIZED,
        saturation=0.5,
        exposure_trace=20.0,  # > 15.0 default theta_intern
        time_in_state=1.0,
    )
    assert ReceptorFunctionalState.INTERNALIZED in rates


def test_desensitized_to_active_recovery_eligible():
    """Recovery DESENSITIZED->ACTIVE when S < epsilon and T > t_recovery."""
    rates = compute_transition_rates(
        current_state=ReceptorFunctionalState.DESENSITIZED,
        saturation=0.2,  # < 0.3 default epsilon_recovery
        exposure_trace=5.0,  # < 15.0 so no internalization
        time_in_state=12.0,  # > 10.0 default t_recovery
    )
    assert ReceptorFunctionalState.ACTIVE in rates


def test_upregulated_to_active_eligible():
    """Transition UPREGULATED->ACTIVE when S > theta and T > t."""
    rates = compute_transition_rates(
        current_state=ReceptorFunctionalState.UPREGULATED,
        saturation=0.5,  # > 0.4 default theta_upreg_exit
        exposure_trace=0.0,
        time_in_state=6.0,  # > 5.0 default t_upreg_exit
    )
    assert ReceptorFunctionalState.ACTIVE in rates


def test_internalized_to_active_recycling():
    """INTERNALIZED->ACTIVE after t_recycle."""
    rates = compute_transition_rates(
        current_state=ReceptorFunctionalState.INTERNALIZED,
        saturation=0.0,
        exposure_trace=0.0,
        time_in_state=55.0,  # > 50.0 default t_recycle
    )
    assert ReceptorFunctionalState.ACTIVE in rates


def test_beta_accelerates_desensitization():
    """Higher beta reduces t0_desens, enabling earlier desensitization."""
    # Without beta: t0_desens=5.0, time=4.0 -> not eligible
    rates_no_beta = compute_transition_rates(
        current_state=ReceptorFunctionalState.ACTIVE,
        saturation=0.9,
        exposure_trace=0.0,
        time_in_state=4.0,
        beta_amplitude=0.0,
    )
    assert ReceptorFunctionalState.DESENSITIZED not in rates_no_beta

    # With beta=1.0: t0_desens_eff = 5.0 * (1 - 0.3*1.0) = 3.5, time=4.0 -> eligible
    rates_beta = compute_transition_rates(
        current_state=ReceptorFunctionalState.ACTIVE,
        saturation=0.9,
        exposure_trace=0.0,
        time_in_state=4.0,
        beta_amplitude=1.0,
    )
    assert ReceptorFunctionalState.DESENSITIZED in rates_beta


def test_no_transitions_moderate_saturation():
    """No transitions when saturation is in the middle range."""
    rates = compute_transition_rates(
        current_state=ReceptorFunctionalState.ACTIVE,
        saturation=0.4,  # Between 0.1 and 0.7
        exposure_trace=0.0,
        time_in_state=30.0,
    )
    assert len(rates) == 0


def test_custom_thresholds():
    """Can override thresholds via dict."""
    rates = compute_transition_rates(
        current_state=ReceptorFunctionalState.ACTIVE,
        saturation=0.5,
        exposure_trace=0.0,
        time_in_state=2.0,
        thresholds={"theta_desens": 0.3, "t0_desens": 1.0},
    )
    assert ReceptorFunctionalState.DESENSITIZED in rates


# ---------------------------------------------------------------------------
# select_transition
# ---------------------------------------------------------------------------

def test_select_transition_single_eligible():
    """Selects the one eligible transition."""
    result = select_transition({ReceptorFunctionalState.DESENSITIZED: 1.0})
    assert result == ReceptorFunctionalState.DESENSITIZED


def test_select_transition_none_eligible():
    """Returns None when no transitions eligible."""
    result = select_transition({})
    assert result is None


def test_select_transition_priority():
    """Desensitization takes priority over upregulation."""
    result = select_transition({
        ReceptorFunctionalState.DESENSITIZED: 1.0,
        ReceptorFunctionalState.UPREGULATED: 1.0,
    })
    assert result == ReceptorFunctionalState.DESENSITIZED


# ---------------------------------------------------------------------------
# apply_state_transition_effects
# ---------------------------------------------------------------------------

def test_effects_desensitized():
    """Entering DESENSITIZED halves sigma, rho unchanged."""
    sigma, rho = apply_state_transition_effects(
        ReceptorFunctionalState.DESENSITIZED, 1.0, 1.0,
    )
    assert sigma == pytest.approx(0.5)
    assert rho == pytest.approx(1.0)


def test_effects_internalized():
    """Entering INTERNALIZED reduces both sigma and rho."""
    sigma, rho = apply_state_transition_effects(
        ReceptorFunctionalState.INTERNALIZED, 1.0, 1.0,
    )
    assert sigma == pytest.approx(0.3)
    assert rho == pytest.approx(0.7)


def test_effects_upregulated():
    """Entering UPREGULATED increases sigma and rho."""
    sigma, rho = apply_state_transition_effects(
        ReceptorFunctionalState.UPREGULATED, 0.6, 0.6,
    )
    assert sigma == pytest.approx(0.6 * 1.3)
    assert rho == pytest.approx(0.6 * 1.2)


def test_effects_upregulated_clamped():
    """Upregulation effects are clamped to 1.0."""
    sigma, rho = apply_state_transition_effects(
        ReceptorFunctionalState.UPREGULATED, 0.9, 0.9,
    )
    assert sigma == 1.0  # 0.9 * 1.3 = 1.17 -> clamped
    assert rho == 1.0    # 0.9 * 1.2 = 1.08 -> clamped


def test_effects_active_no_change():
    """Entering ACTIVE does not change sigma or rho."""
    sigma, rho = apply_state_transition_effects(
        ReceptorFunctionalState.ACTIVE, 0.5, 0.7,
    )
    assert sigma == pytest.approx(0.5)
    assert rho == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# apply_sensitivity_recovery
# ---------------------------------------------------------------------------

def test_sensitivity_recovery_increases():
    """Sensitivity recovers toward 1.0 from below."""
    new_sigma = apply_sensitivity_recovery(0.5, dt=1.0, recovery_rate=0.05)
    expected = 0.5 + 0.05 * (1.0 - 0.5) * 1.0
    assert new_sigma == pytest.approx(expected)
    assert new_sigma > 0.5


def test_sensitivity_recovery_at_max():
    """Sensitivity stays at 1.0 when already there."""
    new_sigma = apply_sensitivity_recovery(1.0, dt=1.0, recovery_rate=0.05)
    assert new_sigma == pytest.approx(1.0)


def test_sensitivity_recovery_rate():
    """Higher recovery rate converges faster."""
    slow = apply_sensitivity_recovery(0.5, dt=1.0, recovery_rate=0.01)
    fast = apply_sensitivity_recovery(0.5, dt=1.0, recovery_rate=0.1)
    assert fast > slow


# ---------------------------------------------------------------------------
# step_receptor_dynamics (integration tests)
# ---------------------------------------------------------------------------

def test_step_does_not_mutate_input():
    """step_receptor_dynamics returns new state, does not mutate input."""
    state = ReceptorState(receptor_id="DA_D1", sigma=0.8)
    result = step_receptor_dynamics(state, concentration=0.5, K_d=0.5, dt=0.1)
    assert state.sigma == 0.8  # Original unchanged
    assert result is not state


def test_step_active_stable():
    """Receptor stays ACTIVE with moderate saturation."""
    state = ReceptorState(receptor_id="DA_D1")
    result = step_receptor_dynamics(state, concentration=0.3, K_d=0.5, dt=0.1)
    assert result.chi == ReceptorFunctionalState.ACTIVE


def test_step_active_to_desensitized():
    """After enough time with high saturation, transitions to DESENSITIZED."""
    state = ReceptorState(
        receptor_id="DA_D1",
        time_in_state=6.0,  # > t0_desens=5.0
    )
    # High saturation: C=0.9, K_d=0.2 -> S = 0.9/1.1 ≈ 0.818 > 0.7
    result = step_receptor_dynamics(state, concentration=0.9, K_d=0.2, dt=0.1)
    assert result.chi == ReceptorFunctionalState.DESENSITIZED
    assert result.sigma < 1.0  # sigma reduced


def test_step_desensitized_recovery():
    """DESENSITIZED receptor recovers to ACTIVE when saturation drops."""
    state = ReceptorState(
        receptor_id="DA_D1",
        chi=ReceptorFunctionalState.DESENSITIZED,
        sigma=0.5,
        time_in_state=12.0,  # > t_recovery=10.0
        exposure_trace=5.0,  # < theta_intern=15.0
    )
    # Low saturation: C=0.1, K_d=0.5 -> S = 0.1/0.6 ≈ 0.167 < 0.3
    result = step_receptor_dynamics(state, concentration=0.1, K_d=0.5, dt=0.1)
    assert result.chi == ReceptorFunctionalState.ACTIVE


def test_step_desensitized_to_internalized():
    """High exposure trace drives DESENSITIZED->INTERNALIZED."""
    state = ReceptorState(
        receptor_id="DA_D1",
        chi=ReceptorFunctionalState.DESENSITIZED,
        sigma=0.5,
        exposure_trace=14.5,
        time_in_state=2.0,
    )
    # High saturation to push exposure_trace over 15.0
    # C=0.9, K_d=0.2 -> S ≈ 0.818; exposure += 0.818 * 1.0 = 0.818
    result = step_receptor_dynamics(state, concentration=0.9, K_d=0.2, dt=1.0)
    # exposure_trace = 14.5 * exp(-0.1) + 0.818 * 1.0 ≈ 13.12 + 0.818 = 13.94
    # Hmm, not quite > 15. Use larger dt or higher starting trace.
    # Let's check: with these params, trace after update:
    # 14.5 * exp(-1/10) + 0.818 * 1.0 = 14.5 * 0.9048 + 0.818 = 13.12 + 0.818 = 13.94
    # Still < 15.0. So no transition yet. Let's use a starting trace of 14.8.
    state2 = ReceptorState(
        receptor_id="DA_D1",
        chi=ReceptorFunctionalState.DESENSITIZED,
        sigma=0.5,
        exposure_trace=15.5,  # Already above threshold
        time_in_state=2.0,
    )
    result2 = step_receptor_dynamics(state2, concentration=0.9, K_d=0.2, dt=0.1)
    assert result2.chi == ReceptorFunctionalState.INTERNALIZED
    assert result2.rho < 1.0  # rho reduced


def test_step_upregulation_with_low_saturation():
    """Prolonged low saturation triggers UPREGULATED."""
    state = ReceptorState(
        receptor_id="DA_D1",
        time_in_state=25.0,  # > t0_upreg=20.0
    )
    # Very low saturation: C=0.01, K_d=0.5 -> S = 0.01/0.51 ≈ 0.0196 < 0.1
    result = step_receptor_dynamics(state, concentration=0.01, K_d=0.5, dt=0.1)
    assert result.chi == ReceptorFunctionalState.UPREGULATED
    # sigma starts at 1.0, * 1.3 = 1.3 -> clamped to 1.0
    assert result.sigma == 1.0
    assert result.rho == 1.0  # 1.0 * 1.2 = 1.2 -> clamped to 1.0


def test_step_internalized_recycling():
    """INTERNALIZED receptor recycles to ACTIVE after t_recycle."""
    state = ReceptorState(
        receptor_id="DA_D1",
        chi=ReceptorFunctionalState.INTERNALIZED,
        sigma=0.3,
        rho=0.7,
        time_in_state=55.0,  # > t_recycle=50.0
    )
    result = step_receptor_dynamics(state, concentration=0.0, K_d=0.5, dt=0.1)
    assert result.chi == ReceptorFunctionalState.ACTIVE


def test_step_sigma_recovery_in_active():
    """Sigma gradually recovers toward 1.0 in ACTIVE state."""
    state = ReceptorState(receptor_id="DA_D1", sigma=0.5, time_in_state=0.0)
    # Moderate saturation so no transitions
    result = step_receptor_dynamics(state, concentration=0.3, K_d=0.5, dt=1.0)
    assert result.chi == ReceptorFunctionalState.ACTIVE
    assert result.sigma > 0.5  # Recovery happened


def test_step_exposure_trace_updates():
    """Exposure trace is updated each step."""
    state = ReceptorState(receptor_id="DA_D1", exposure_trace=0.0)
    result = step_receptor_dynamics(state, concentration=0.5, K_d=0.5, dt=1.0)
    assert result.exposure_trace > 0.0


def test_step_time_in_state_increments():
    """time_in_state increases by dt each step when no transition."""
    state = ReceptorState(receptor_id="DA_D1", time_in_state=5.0)
    result = step_receptor_dynamics(state, concentration=0.3, K_d=0.5, dt=0.5)
    assert result.time_in_state == pytest.approx(5.5)


def test_step_time_resets_on_transition():
    """time_in_state resets to dt after transition (reset to 0 + increment)."""
    state = ReceptorState(
        receptor_id="DA_D1",
        time_in_state=6.0,
    )
    result = step_receptor_dynamics(state, concentration=0.9, K_d=0.2, dt=0.1)
    assert result.chi == ReceptorFunctionalState.DESENSITIZED
    assert result.time_in_state == pytest.approx(0.1)  # reset to 0.0 then +dt
