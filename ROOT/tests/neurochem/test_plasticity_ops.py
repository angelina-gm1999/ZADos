"""
Tests for plasticity event operators (Appendix K.4).

Phase 31: Condition checkers, application functions, and event encoding.
"""

import pytest

from zados.neurochem.state.receptor_state import ReceptorFunctionalState, ReceptorState
from zados.neurochem.neurosymbolic.plasticity_ops import (
    PlasticityEvent,
    check_internalization_condition,
    check_upregulation_condition,
    check_desensitization_condition,
    apply_internalization,
    apply_upregulation,
    apply_desensitization,
    apply_recovery,
    apply_switch,
    encode_plasticity_event,
)


# ---------------------------------------------------------------------------
# PlasticityEvent dataclass
# ---------------------------------------------------------------------------

class TestPlasticityEvent:
    """Tests for PlasticityEvent frozen dataclass."""

    def test_event_frozen(self):
        ev = PlasticityEvent(time=1.0, operator="INT", target="D2")
        with pytest.raises(AttributeError):
            ev.time = 2.0

    def test_event_fields(self):
        ev = PlasticityEvent(
            time=5.0, operator="SWITCH", target="D1", target_b="D3",
            trigger_window=15.0, pre_state="active", post_state="internalized",
        )
        assert ev.time == 5.0
        assert ev.operator == "SWITCH"
        assert ev.target == "D1"
        assert ev.target_b == "D3"
        assert ev.trigger_window == 15.0
        assert ev.pre_state == "active"
        assert ev.post_state == "internalized"

    def test_event_defaults(self):
        ev = PlasticityEvent(time=0.0, operator="UPR", target="OXTR")
        assert ev.target_b is None
        assert ev.trigger_window is None
        assert ev.evidence is None
        assert ev.pre_state is None
        assert ev.post_state is None


# ---------------------------------------------------------------------------
# Condition checkers — internalization
# ---------------------------------------------------------------------------

class TestCheckInternalization:
    """Tests for check_internalization_condition."""

    def test_fires_above_threshold_and_time(self):
        assert check_internalization_condition(0.9, 20.0) is True

    def test_below_saturation_threshold(self):
        assert check_internalization_condition(0.5, 20.0) is False

    def test_below_time_threshold(self):
        assert check_internalization_condition(0.9, 10.0) is False

    def test_exact_boundary(self):
        assert check_internalization_condition(0.8, 15.0) is True

    def test_custom_thresholds(self):
        assert check_internalization_condition(0.6, 5.0, theta_int=0.5, t_int=3.0) is True
        assert check_internalization_condition(0.6, 5.0, theta_int=0.7, t_int=3.0) is False


# ---------------------------------------------------------------------------
# Condition checkers — upregulation
# ---------------------------------------------------------------------------

class TestCheckUpregulation:
    """Tests for check_upregulation_condition."""

    def test_fires_below_threshold_and_time(self):
        assert check_upregulation_condition(0.05, 25.0) is True

    def test_above_saturation_threshold(self):
        assert check_upregulation_condition(0.3, 25.0) is False

    def test_below_time_threshold(self):
        assert check_upregulation_condition(0.05, 10.0) is False

    def test_exact_boundary(self):
        assert check_upregulation_condition(0.1, 20.0) is True

    def test_custom_thresholds(self):
        assert check_upregulation_condition(0.2, 10.0, epsilon_upr=0.25, t_upr=5.0) is True


# ---------------------------------------------------------------------------
# Condition checkers — desensitization
# ---------------------------------------------------------------------------

class TestCheckDesensitization:
    """Tests for check_desensitization_condition."""

    def test_fires(self):
        assert check_desensitization_condition(0.8, 12.0) is True

    def test_below_threshold(self):
        assert check_desensitization_condition(0.5, 12.0) is False


# ---------------------------------------------------------------------------
# Apply internalization
# ---------------------------------------------------------------------------

class TestApplyInternalization:
    """Tests for apply_internalization."""

    def test_reduces_rho(self):
        state = ReceptorState(receptor_id="D2", rho=1.0, sigma=1.0)
        new = apply_internalization(state)
        assert new.rho == pytest.approx(0.7)  # 1.0 * (1 - 0.3)

    def test_reduces_gamma(self):
        state = ReceptorState(receptor_id="D2", rho=1.0, gamma_gprotein=0.8)
        new = apply_internalization(state)
        assert new.gamma_gprotein == pytest.approx(0.7)  # 0.8 - 0.1

    def test_sets_chi_internalized(self):
        state = ReceptorState(receptor_id="D2", rho=1.0)
        new = apply_internalization(state)
        assert new.chi == ReceptorFunctionalState.INTERNALIZED

    def test_resets_time_in_state(self):
        state = ReceptorState(receptor_id="D2", rho=1.0, time_in_state=20.0)
        new = apply_internalization(state)
        assert new.time_in_state == 0.0

    def test_does_not_mutate_original(self):
        state = ReceptorState(receptor_id="D2", rho=1.0)
        _ = apply_internalization(state)
        assert state.rho == 1.0
        assert state.chi == ReceptorFunctionalState.ACTIVE

    def test_clamps_rho_floor(self):
        state = ReceptorState(receptor_id="D2", rho=0.05)
        new = apply_internalization(state, kappa_int=0.99)
        assert new.rho >= 0.0

    def test_clamps_gamma_floor(self):
        state = ReceptorState(receptor_id="D2", rho=1.0, gamma_gprotein=0.05)
        new = apply_internalization(state, kappa_gamma=0.5)
        assert new.gamma_gprotein == 0.0


# ---------------------------------------------------------------------------
# Apply upregulation
# ---------------------------------------------------------------------------

class TestApplyUpregulation:
    """Tests for apply_upregulation."""

    def test_increases_rho(self):
        state = ReceptorState(receptor_id="OXTR", rho=0.5)
        new = apply_upregulation(state)
        assert new.rho == pytest.approx(0.6)  # 0.5 + 0.1

    def test_increases_sigma(self):
        state = ReceptorState(receptor_id="OXTR", rho=0.5, sigma=0.8)
        new = apply_upregulation(state)
        assert new.sigma == pytest.approx(0.9)  # 0.8 + 0.1

    def test_sets_chi_upregulated(self):
        state = ReceptorState(receptor_id="OXTR", rho=0.5)
        new = apply_upregulation(state)
        assert new.chi == ReceptorFunctionalState.UPREGULATED

    def test_clamps_rho_ceiling(self):
        state = ReceptorState(receptor_id="OXTR", rho=0.95)
        new = apply_upregulation(state)
        assert new.rho == 1.0

    def test_does_not_mutate_original(self):
        state = ReceptorState(receptor_id="OXTR", rho=0.5)
        _ = apply_upregulation(state)
        assert state.rho == 0.5


# ---------------------------------------------------------------------------
# Apply desensitization
# ---------------------------------------------------------------------------

class TestApplyDesensitization:
    """Tests for apply_desensitization."""

    def test_reduces_sigma(self):
        state = ReceptorState(receptor_id="D1", sigma=1.0)
        new = apply_desensitization(state)
        assert new.sigma == pytest.approx(0.7)  # 1.0 * (1 - 0.3)

    def test_sets_chi_desensitized(self):
        state = ReceptorState(receptor_id="D1", sigma=1.0)
        new = apply_desensitization(state)
        assert new.chi == ReceptorFunctionalState.DESENSITIZED


# ---------------------------------------------------------------------------
# Apply recovery
# ---------------------------------------------------------------------------

class TestApplyRecovery:
    """Tests for apply_recovery."""

    def test_recovers_sigma(self):
        state = ReceptorState(receptor_id="D1", sigma=0.4)
        new = apply_recovery(state)
        # gap = 1.0 - 0.4 = 0.6, recovery = 0.6 * 0.5 = 0.3
        assert new.sigma == pytest.approx(0.7)

    def test_sets_chi_active(self):
        state = ReceptorState(
            receptor_id="D1", sigma=0.5,
            chi=ReceptorFunctionalState.DESENSITIZED,
        )
        new = apply_recovery(state)
        assert new.chi == ReceptorFunctionalState.ACTIVE

    def test_clamps_sigma_ceiling(self):
        state = ReceptorState(receptor_id="D1", sigma=0.9)
        new = apply_recovery(state, target_sigma=1.0, recovery_rate=1.0)
        assert new.sigma == 1.0


# ---------------------------------------------------------------------------
# Apply switch
# ---------------------------------------------------------------------------

class TestApplySwitch:
    """Tests for apply_switch."""

    def test_reallocation(self):
        a = ReceptorState(receptor_id="D1", rho=1.0)
        b = ReceptorState(receptor_id="D3", rho=0.5)
        new_a, new_b = apply_switch(a, b)
        # transfer = 1.0 * 0.3 = 0.3
        assert new_a.rho == pytest.approx(0.7)
        assert new_b.rho == pytest.approx(0.8)

    def test_sensitivity_boost(self):
        a = ReceptorState(receptor_id="D1", rho=1.0)
        b = ReceptorState(receptor_id="D3", rho=0.5, sigma=0.5)
        _, new_b = apply_switch(a, b)
        # transfer = 0.3, boost = 0.3 * 0.1 = 0.03
        assert new_b.sigma == pytest.approx(0.53)

    def test_does_not_mutate_originals(self):
        a = ReceptorState(receptor_id="D1", rho=1.0)
        b = ReceptorState(receptor_id="D3", rho=0.5)
        _ = apply_switch(a, b)
        assert a.rho == 1.0
        assert b.rho == 0.5

    def test_custom_kappa_sw(self):
        a = ReceptorState(receptor_id="D1", rho=0.8)
        b = ReceptorState(receptor_id="D3", rho=0.2)
        new_a, new_b = apply_switch(a, b, kappa_sw=0.5)
        # transfer = 0.8 * 0.5 = 0.4
        assert new_a.rho == pytest.approx(0.4)
        assert new_b.rho == pytest.approx(0.6)

    def test_clamps_rho(self):
        a = ReceptorState(receptor_id="D1", rho=0.1)
        b = ReceptorState(receptor_id="D3", rho=0.99)
        new_a, new_b = apply_switch(a, b, kappa_sw=0.5)
        assert new_a.rho >= 0.0
        assert new_b.rho <= 1.0


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

class TestEncodePlasticityEvent:
    """Tests for encode_plasticity_event."""

    def test_encode_int(self):
        ev = PlasticityEvent(time=1.0, operator="INT", target="D2")
        assert encode_plasticity_event(ev) == "INT(D2)"

    def test_encode_upr(self):
        ev = PlasticityEvent(time=2.0, operator="UPR", target="OXTR")
        assert encode_plasticity_event(ev) == "UPR(OXTR)"

    def test_encode_switch(self):
        ev = PlasticityEvent(time=3.0, operator="SWITCH", target="D1", target_b="D3")
        assert encode_plasticity_event(ev) == "SWITCH(D1->D3)"

    def test_encode_dsn(self):
        ev = PlasticityEvent(time=1.0, operator="DSN", target="D1")
        assert encode_plasticity_event(ev) == "DSN(D1)"

    def test_encode_rec(self):
        ev = PlasticityEvent(time=1.0, operator="REC", target="D1")
        assert encode_plasticity_event(ev) == "REC(D1)"
