"""
Tests for G-protein coupling (gamma_gprotein) dynamics.

Phase 20: gamma_gprotein degrades under sustained high saturation and
recovers toward 1.0 when saturation drops. The pure function
update_gamma_gprotein is wired into step_receptor_dynamics.
"""

import pytest

from zados.neurochem.kinetics.receptor_dynamics import (
    update_gamma_gprotein,
    step_receptor_dynamics,
)
from zados.neurochem.state.receptor_state import ReceptorState, ReceptorFunctionalState


# ---------------------------------------------------------------------------
# Pure function: update_gamma_gprotein
# ---------------------------------------------------------------------------

class TestUpdateGammaGprotein:
    """Tests for the pure update_gamma_gprotein function."""

    def test_degrades_under_high_saturation(self):
        """gamma should decrease when saturation > threshold."""
        gamma = update_gamma_gprotein(
            gamma=1.0, saturation=0.8, dt=1.0,
            k_degrade=0.02, threshold=0.5,
        )
        # Expected: 1.0 - 0.02 * 0.8 * 1.0 = 0.984
        assert gamma == pytest.approx(0.984)
        assert gamma < 1.0

    def test_recovers_under_low_saturation(self):
        """gamma should increase toward 1.0 when saturation < threshold."""
        gamma = update_gamma_gprotein(
            gamma=0.5, saturation=0.2, dt=1.0,
            k_recover=0.01, threshold=0.5,
        )
        # Expected: 0.5 + 0.01 * (1.0 - 0.5) * 1.0 = 0.505
        assert gamma == pytest.approx(0.505)
        assert gamma > 0.5

    def test_clamped_min(self):
        """gamma should never go below gamma_min."""
        gamma = update_gamma_gprotein(
            gamma=0.06, saturation=1.0, dt=100.0,
            k_degrade=0.5, gamma_min=0.05,
        )
        assert gamma == pytest.approx(0.05)

    def test_clamped_max(self):
        """gamma should never exceed 1.0."""
        gamma = update_gamma_gprotein(
            gamma=0.99, saturation=0.0, dt=100.0,
            k_recover=0.5,
        )
        assert gamma == pytest.approx(1.0)

    def test_stable_at_threshold(self):
        """At exactly the threshold, saturation is not > threshold, so recovery occurs."""
        gamma = update_gamma_gprotein(
            gamma=0.8, saturation=0.5, dt=1.0,
            threshold=0.5,
        )
        # saturation == threshold → not > → recovery path
        # 0.8 + 0.01 * (1.0 - 0.8) * 1.0 = 0.802
        assert gamma == pytest.approx(0.802)

    def test_no_change_at_equilibrium(self):
        """gamma=1.0 with low saturation: recovery adds nothing."""
        gamma = update_gamma_gprotein(
            gamma=1.0, saturation=0.0, dt=1.0,
        )
        # 1.0 + 0.01 * (1.0 - 1.0) * 1.0 = 1.0
        assert gamma == pytest.approx(1.0)

    def test_custom_rates(self):
        """Custom k_degrade and k_recover."""
        gamma = update_gamma_gprotein(
            gamma=1.0, saturation=0.9, dt=1.0,
            k_degrade=0.1, threshold=0.3,
        )
        # 1.0 - 0.1 * 0.9 * 1.0 = 0.91
        assert gamma == pytest.approx(0.91)

    def test_cycle_degrade_then_recover(self):
        """Degradation followed by recovery cycle."""
        # Degrade: high saturation
        gamma = 1.0
        for _ in range(50):
            gamma = update_gamma_gprotein(
                gamma, saturation=0.9, dt=1.0,
                k_degrade=0.02, threshold=0.5,
            )
        assert gamma < 0.5  # Significant degradation

        # Recover: low saturation
        for _ in range(200):
            gamma = update_gamma_gprotein(
                gamma, saturation=0.1, dt=1.0,
                k_recover=0.01, threshold=0.5,
            )
        assert gamma > 0.8  # Significant recovery


# ---------------------------------------------------------------------------
# Integration: step_receptor_dynamics updates gamma
# ---------------------------------------------------------------------------

class TestStepReceptorDynamicsGamma:
    """Tests that step_receptor_dynamics updates gamma_gprotein."""

    def test_gamma_changes_over_multiple_steps(self):
        """Running multiple steps with high concentration should degrade gamma."""
        state = ReceptorState(receptor_id="DA_D1", gamma_gprotein=1.0)
        for _ in range(100):
            state = step_receptor_dynamics(
                receptor_state=state,
                concentration=0.9,
                K_d=0.5,
                dt=1.0,
            )
        # High saturation (0.9/(0.9+0.5)≈0.643 > 0.5) should degrade gamma
        assert state.gamma_gprotein < 1.0

    def test_gamma_stays_above_min(self):
        """Even with extreme degradation, gamma stays above gamma_min."""
        state = ReceptorState(receptor_id="DA_D1", gamma_gprotein=1.0)
        for _ in range(1000):
            state = step_receptor_dynamics(
                receptor_state=state,
                concentration=2.0,
                K_d=0.5,
                dt=1.0,
            )
        assert state.gamma_gprotein >= 0.05

    def test_custom_gamma_rates_via_thresholds(self):
        """Custom gamma rates via thresholds dict override defaults."""
        state = ReceptorState(receptor_id="DA_D1", gamma_gprotein=1.0)
        # Very fast degradation rate
        state = step_receptor_dynamics(
            receptor_state=state,
            concentration=2.0,
            K_d=0.5,
            dt=1.0,
            thresholds={"gamma_degrade_rate": 0.5, "gamma_degrade_threshold": 0.3},
        )
        # sat = 2.0/(2.0+0.5) = 0.8, degrade = 0.5 * 0.8 * 1.0 = 0.4
        # gamma = 1.0 - 0.4 = 0.6
        assert state.gamma_gprotein == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# ReceptorState.update_gamma_gprotein method
# ---------------------------------------------------------------------------

class TestReceptorStateGammaMethod:
    """Tests for the update_gamma_gprotein method on ReceptorState."""

    def test_increase(self):
        state = ReceptorState(receptor_id="DA_D1", gamma_gprotein=0.5)
        state.update_gamma_gprotein(0.2)
        assert state.gamma_gprotein == pytest.approx(0.7)

    def test_decrease(self):
        state = ReceptorState(receptor_id="DA_D1", gamma_gprotein=0.5)
        state.update_gamma_gprotein(-0.3)
        assert state.gamma_gprotein == pytest.approx(0.2)

    def test_clamped_upper(self):
        state = ReceptorState(receptor_id="DA_D1", gamma_gprotein=0.9)
        state.update_gamma_gprotein(0.5)
        assert state.gamma_gprotein == pytest.approx(1.0)

    def test_clamped_lower(self):
        state = ReceptorState(receptor_id="DA_D1", gamma_gprotein=0.1)
        state.update_gamma_gprotein(-0.5)
        assert state.gamma_gprotein == pytest.approx(0.0)
