"""Tests for leaky integrator primitives (Extractor 3 temporal smoothing)."""

import math
import pytest

from zados.neurochem.extractors.leaky_integrator import (
    LeakyIntegratorState,
    leaky_integrator_step,
    exponential_moving_average_step,
    batch_leaky_integrator_step,
)


# =====================================================================
# LeakyIntegratorState
# =====================================================================

class TestLeakyIntegratorState:
    def test_defaults(self):
        s = LeakyIntegratorState()
        assert s.value == 0.0
        assert s.baseline == 0.0

    def test_as_dict(self):
        s = LeakyIntegratorState(value=0.5, baseline=0.1)
        d = s.as_dict()
        assert d == {"value": 0.5, "baseline": 0.1}

    def test_from_dict(self):
        s = LeakyIntegratorState.from_dict({"value": 0.7, "baseline": 0.2})
        assert s.value == 0.7
        assert s.baseline == 0.2

    def test_from_dict_defaults(self):
        s = LeakyIntegratorState.from_dict({})
        assert s.value == 0.0
        assert s.baseline == 0.0

    def test_roundtrip(self):
        s = LeakyIntegratorState(value=0.42, baseline=0.1)
        s2 = LeakyIntegratorState.from_dict(s.as_dict())
        assert s2.value == s.value
        assert s2.baseline == s.baseline


# =====================================================================
# leaky_integrator_step
# =====================================================================

class TestLeakyIntegratorStep:
    def test_zero_input_decays_to_baseline(self):
        """With zero input, value should decay toward baseline."""
        state = LeakyIntegratorState(value=1.0, baseline=0.0)
        new = leaky_integrator_step(state, 0.0, dt=0.1, tau=1.0)
        # dR/dt = 0 - (1.0 - 0.0)/1.0 = -1.0
        # new = 1.0 + 0.1 * (-1.0) = 0.9
        assert new.value == pytest.approx(0.9)

    def test_constant_input_drives_up(self):
        """Constant positive input should drive value upward."""
        state = LeakyIntegratorState(value=0.0, baseline=0.0)
        new = leaky_integrator_step(state, 1.0, dt=0.1, tau=1.0, gain=1.0)
        # dR/dt = 1*1 - (0-0)/1 = 1.0
        # new = 0.0 + 0.1 * 1.0 = 0.1
        assert new.value == pytest.approx(0.1)

    def test_steady_state(self):
        """After many steps with constant input, should approach gain*input*tau + baseline."""
        state = LeakyIntegratorState(value=0.0, baseline=0.0)
        for _ in range(10000):
            state = leaky_integrator_step(state, 1.0, dt=0.01, tau=1.0, gain=1.0)
        # Steady state: gain*input*tau = 1.0*1.0*1.0 = 1.0
        assert state.value == pytest.approx(1.0, abs=0.01)

    def test_nonzero_baseline(self):
        """Should decay toward baseline, not zero."""
        state = LeakyIntegratorState(value=0.0, baseline=0.5)
        for _ in range(10000):
            state = leaky_integrator_step(state, 0.0, dt=0.01, tau=1.0)
        assert state.value == pytest.approx(0.5, abs=0.01)

    def test_gain_scales_input(self):
        """Gain should scale the input effect."""
        s1 = leaky_integrator_step(
            LeakyIntegratorState(0.0, 0.0), 1.0, dt=0.1, tau=1.0, gain=1.0,
        )
        s2 = leaky_integrator_step(
            LeakyIntegratorState(0.0, 0.0), 1.0, dt=0.1, tau=1.0, gain=0.5,
        )
        assert s1.value > s2.value

    def test_large_tau_slow_decay(self):
        """Larger tau → slower decay back to baseline."""
        # Start displaced from baseline with zero input → pure decay
        s_fast = leaky_integrator_step(
            LeakyIntegratorState(1.0, 0.0), 0.0, dt=0.1, tau=1.0,
        )
        s_slow = leaky_integrator_step(
            LeakyIntegratorState(1.0, 0.0), 0.0, dt=0.1, tau=100.0,
        )
        # Both should decay, but fast τ decays more
        assert s_fast.value < s_slow.value

    def test_does_not_mutate_original(self):
        state = LeakyIntegratorState(value=0.5, baseline=0.0)
        new = leaky_integrator_step(state, 1.0, dt=0.1, tau=1.0)
        assert state.value == 0.5  # unchanged
        assert new.value != 0.5

    def test_zero_tau_raises(self):
        state = LeakyIntegratorState()
        with pytest.raises(ValueError, match="tau must be positive"):
            leaky_integrator_step(state, 1.0, dt=0.1, tau=0.0)

    def test_negative_tau_raises(self):
        state = LeakyIntegratorState()
        with pytest.raises(ValueError, match="tau must be positive"):
            leaky_integrator_step(state, 1.0, dt=0.1, tau=-1.0)

    def test_preserves_baseline(self):
        state = LeakyIntegratorState(value=0.5, baseline=0.3)
        new = leaky_integrator_step(state, 1.0, dt=0.1, tau=1.0)
        assert new.baseline == 0.3


# =====================================================================
# exponential_moving_average_step
# =====================================================================

class TestExponentialMovingAverageStep:
    def test_converges_to_constant(self):
        """EMA of constant signal should converge to that constant."""
        val = 0.0
        for _ in range(10000):
            val = exponential_moving_average_step(val, 1.0, dt=0.01, tau=1.0)
        assert val == pytest.approx(1.0, abs=0.01)

    def test_small_tau_fast_tracking(self):
        """Small tau should track the new sample closely."""
        val = exponential_moving_average_step(0.0, 1.0, dt=10.0, tau=1.0)
        # alpha = 1 - exp(-10) ≈ 1.0
        assert val == pytest.approx(1.0, abs=0.001)

    def test_large_tau_slow_tracking(self):
        """Large tau should barely move."""
        val = exponential_moving_average_step(0.0, 1.0, dt=0.001, tau=100.0)
        # alpha = 1 - exp(-0.001/100) ≈ 0.00001
        assert val < 0.001

    def test_zero_dt_no_change(self):
        """Zero dt → alpha = 0 → no change."""
        val = exponential_moving_average_step(0.5, 1.0, dt=0.0, tau=1.0)
        assert val == pytest.approx(0.5)

    def test_zero_tau_raises(self):
        with pytest.raises(ValueError, match="tau must be positive"):
            exponential_moving_average_step(0.0, 1.0, dt=0.1, tau=0.0)

    def test_known_value(self):
        """EMA with known alpha."""
        # dt=1, tau=1 → alpha = 1 - exp(-1) ≈ 0.6321
        alpha = 1.0 - math.exp(-1.0)
        result = exponential_moving_average_step(0.0, 1.0, dt=1.0, tau=1.0)
        expected = 0.0 * (1.0 - alpha) + 1.0 * alpha
        assert result == pytest.approx(expected)


# =====================================================================
# batch_leaky_integrator_step
# =====================================================================

class TestBatchLeakyIntegratorStep:
    def test_multiple_integrators(self):
        states = {
            "a": LeakyIntegratorState(0.0, 0.0),
            "b": LeakyIntegratorState(1.0, 0.0),
        }
        inputs = {"a": 1.0, "b": 0.0}
        taus = {"a": 1.0, "b": 1.0}
        result = batch_leaky_integrator_step(states, inputs, dt=0.1, taus=taus)
        assert "a" in result and "b" in result
        assert result["a"].value > 0.0  # driven up
        assert result["b"].value < 1.0  # decaying

    def test_missing_input_defaults_zero(self):
        states = {"a": LeakyIntegratorState(0.5, 0.0)}
        result = batch_leaky_integrator_step(
            states, {}, dt=0.1, taus={"a": 1.0},
        )
        # Zero input → decay
        assert result["a"].value < 0.5

    def test_custom_gains(self):
        states = {"a": LeakyIntegratorState(0.0, 0.0)}
        inputs = {"a": 1.0}
        r1 = batch_leaky_integrator_step(
            states, inputs, dt=0.1, taus={"a": 1.0}, gains={"a": 1.0},
        )
        r2 = batch_leaky_integrator_step(
            states, inputs, dt=0.1, taus={"a": 1.0}, gains={"a": 0.1},
        )
        assert r1["a"].value > r2["a"].value

    def test_empty_states(self):
        result = batch_leaky_integrator_step({}, {"a": 1.0}, dt=0.1, taus={})
        assert result == {}
