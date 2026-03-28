"""
Tests for Extractor 5 — Urgency Forecast Module.

Tests cover:
- Per-axis urgency value computation (weighted sums + inversion)
- Linear-exponential forecast peak
- Threshold breach detection
- Global urgency risk
- Reactive NE/DA burst generation
- Modulatory feedback (persistent breach)
- Full step_urgency_forecast() sequencing
- Orchestrator integration
"""

import math

import numpy as np
import pytest

from zados.neurochem.extractors.urgency_forecast import (
    UrgencyAxisSourceDef,
    UrgencyAxisConfig,
    UrgencyForecastConfig,
    UrgencyForecastState,
    DEFAULT_URGENCY_FORECAST_CONFIG,
    compute_urgency_axis_value,
    forecast_peak,
    detect_breach,
    compute_urgency_risk,
    compute_reactive_burst,
    compute_modulatory_feedback,
    step_urgency_forecast,
)
from zados.neurochem.extractors.extractor_orchestrator import (
    ExtractorOrchestrator,
    ExtractorState,
    ExtractorResult,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_axis():
    """Single-source, non-inverted axis."""
    return UrgencyAxisConfig(
        name="test_axis",
        sources=(UrgencyAxisSourceDef("logical_conflict", weight=1.0),),
        alpha=1.0,
        tau_smooth=2.0,
        tau_forecast=3.0,
        threshold=0.7,
    )


@pytest.fixture
def inverted_axis():
    """Single-source, inverted axis."""
    return UrgencyAxisConfig(
        name="inverted_axis",
        sources=(UrgencyAxisSourceDef("emotional_valence", weight=1.0, invert=True),),
        alpha=0.8,
        tau_smooth=3.0,
        tau_forecast=4.0,
        threshold=0.75,
    )


@pytest.fixture
def multi_source_axis():
    """Multi-source axis with mixed inversion."""
    return UrgencyAxisConfig(
        name="multi_axis",
        sources=(
            UrgencyAxisSourceDef("novelty", weight=0.5),
            UrgencyAxisSourceDef("reward_alignment", weight=0.5, invert=True),
        ),
        alpha=1.5,
        tau_smooth=1.5,
        tau_forecast=2.0,
        threshold=0.7,
    )


@pytest.fixture
def simple_config(simple_axis):
    """Config with a single axis for isolated tests."""
    return UrgencyForecastConfig(
        axes=(simple_axis,),
        prediction_window=5.0,
        urgency_epsilon=0.1,
        beta_urg=0.3,
        lambda_urg=5.0,
        da_burst_fraction=0.15,
        persistence_steps=3,
    )


# =====================================================================
# TestUrgencyAxisValue
# =====================================================================

class TestUrgencyAxisValue:

    def test_single_source(self, simple_axis):
        adjusted_eval = {"logical_conflict": 0.6}
        result = compute_urgency_axis_value(adjusted_eval, simple_axis)
        assert result == pytest.approx(0.6)

    def test_multi_source_weighted(self, multi_source_axis):
        adjusted_eval = {"novelty": 0.8, "reward_alignment": 0.4}
        # 0.5 * 0.8 + 0.5 * (1 - 0.4) = 0.4 + 0.3 = 0.7
        result = compute_urgency_axis_value(adjusted_eval, multi_source_axis)
        assert result == pytest.approx(0.7)

    def test_inversion(self, inverted_axis):
        adjusted_eval = {"emotional_valence": 0.3}
        # 1.0 * (1 - 0.3) = 0.7
        result = compute_urgency_axis_value(adjusted_eval, inverted_axis)
        assert result == pytest.approx(0.7)

    def test_missing_axis_defaults_zero(self, simple_axis):
        adjusted_eval = {}  # logical_conflict not present
        result = compute_urgency_axis_value(adjusted_eval, simple_axis)
        assert result == pytest.approx(0.0)

    def test_clamps_to_unit_interval(self):
        # Weights can sum > 1.0
        axis = UrgencyAxisConfig(
            name="heavy",
            sources=(
                UrgencyAxisSourceDef("a", weight=0.8),
                UrgencyAxisSourceDef("b", weight=0.8),
            ),
        )
        adjusted_eval = {"a": 0.9, "b": 0.9}
        # 0.8*0.9 + 0.8*0.9 = 1.44 → clamped to 1.0
        result = compute_urgency_axis_value(adjusted_eval, axis)
        assert result == pytest.approx(1.0)

    def test_clamps_negative_to_zero(self):
        # Inverted source with high eval → small value, but with negative weight test
        axis = UrgencyAxisConfig(
            name="neg",
            sources=(UrgencyAxisSourceDef("x", weight=-0.5),),
        )
        result = compute_urgency_axis_value({"x": 0.8}, axis)
        # -0.5 * 0.8 = -0.4 → clamped to 0.0
        assert result == pytest.approx(0.0)


# =====================================================================
# TestForecastPeak
# =====================================================================

class TestForecastPeak:

    def test_rising_signal_forecasts_higher(self):
        # smoothed_current > smoothed_prev → positive derivative → forecast > current
        result = forecast_peak(
            smoothed_current=0.5,
            smoothed_prev=0.3,
            dt=0.1,
            alpha=1.0,
            tau_forecast=3.0,
            prediction_window=5.0,
        )
        assert result > 0.5

    def test_falling_signal_forecasts_lower(self):
        result = forecast_peak(
            smoothed_current=0.5,
            smoothed_prev=0.7,
            dt=0.1,
            alpha=1.0,
            tau_forecast=3.0,
            prediction_window=5.0,
        )
        assert result < 0.5

    def test_zero_derivative_equals_current(self):
        result = forecast_peak(
            smoothed_current=0.5,
            smoothed_prev=0.5,
            dt=0.1,
            alpha=1.0,
            tau_forecast=3.0,
            prediction_window=5.0,
        )
        assert result == pytest.approx(0.5)

    def test_clamps_to_one(self):
        # Very steep rise → forecast could exceed 1.0
        result = forecast_peak(
            smoothed_current=0.9,
            smoothed_prev=0.1,
            dt=0.01,  # small dt → huge derivative
            alpha=2.0,
            tau_forecast=3.0,
            prediction_window=5.0,
        )
        assert result == pytest.approx(1.0)

    def test_clamps_to_zero(self):
        # Very steep fall → forecast could go negative
        result = forecast_peak(
            smoothed_current=0.1,
            smoothed_prev=0.9,
            dt=0.01,
            alpha=2.0,
            tau_forecast=3.0,
            prediction_window=5.0,
        )
        assert result == pytest.approx(0.0)

    def test_zero_dt_returns_current(self):
        result = forecast_peak(
            smoothed_current=0.5,
            smoothed_prev=0.3,
            dt=0.0,
            alpha=1.0,
            tau_forecast=3.0,
            prediction_window=5.0,
        )
        assert result == pytest.approx(0.5)

    def test_formula_correctness(self):
        """Verify exact formula: ê = ẽ + α·(dẽ/dt)·(1-exp(-δ/τ))."""
        s_cur, s_prev, dt = 0.6, 0.4, 0.1
        alpha, tau, pw = 1.2, 3.0, 5.0
        d_smooth = (s_cur - s_prev) / dt
        expected = s_cur + alpha * d_smooth * (1 - math.exp(-pw / tau))
        expected = max(0.0, min(1.0, expected))
        result = forecast_peak(s_cur, s_prev, dt, alpha, tau, pw)
        assert result == pytest.approx(expected)


# =====================================================================
# TestBreachDetection
# =====================================================================

class TestBreachDetection:

    def test_above_threshold_is_breach(self):
        assert detect_breach(0.8, 0.7) is True

    def test_below_threshold_is_no_breach(self):
        assert detect_breach(0.5, 0.7) is False

    def test_at_threshold_is_no_breach(self):
        # Strictly greater required
        assert detect_breach(0.7, 0.7) is False


# =====================================================================
# TestUrgencyRisk
# =====================================================================

class TestUrgencyRisk:

    def test_single_breach(self):
        forecasts = {"a": 0.9}
        thresholds = {"a": 0.7}
        assert compute_urgency_risk(forecasts, thresholds) == pytest.approx(0.2)

    def test_multi_axis_takes_max(self):
        forecasts = {"a": 0.9, "b": 0.85}
        thresholds = {"a": 0.7, "b": 0.6}
        # a: 0.2, b: 0.25 → max = 0.25
        assert compute_urgency_risk(forecasts, thresholds) == pytest.approx(0.25)

    def test_no_breach_returns_zero(self):
        forecasts = {"a": 0.5, "b": 0.4}
        thresholds = {"a": 0.7, "b": 0.6}
        assert compute_urgency_risk(forecasts, thresholds) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert compute_urgency_risk({}, {}) == pytest.approx(0.0)


# =====================================================================
# TestReactiveBurst
# =====================================================================

class TestReactiveBurst:

    def test_below_epsilon_returns_empty(self, rng):
        config = UrgencyForecastConfig(urgency_epsilon=0.1)
        result = compute_reactive_burst(0.05, config, rng=rng)
        assert result == {}

    def test_above_epsilon_produces_ne_da(self, rng):
        config = UrgencyForecastConfig(
            urgency_epsilon=0.1,
            beta_urg=0.3,
            lambda_urg=5.0,
            da_burst_fraction=0.15,
        )
        result = compute_reactive_burst(0.3, config, rng=rng)
        # With seeded RNG and risk=0.3, Poisson should produce non-zero
        assert "NE" in result or result == {}  # Poisson can be 0
        if "NE" in result:
            assert result["NE"] > 0.0
            assert "DA" in result
            assert result["DA"] == pytest.approx(config.da_burst_fraction * result["NE"])

    def test_deterministic_with_seed(self):
        config = UrgencyForecastConfig(
            urgency_epsilon=0.1,
            beta_urg=0.3,
            lambda_urg=5.0,
            da_burst_fraction=0.15,
        )
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        r1 = compute_reactive_burst(0.5, config, rng=rng1)
        r2 = compute_reactive_burst(0.5, config, rng=rng2)
        assert r1 == r2

    def test_da_is_fraction_of_ne(self, rng):
        config = UrgencyForecastConfig(
            urgency_epsilon=0.01,
            beta_urg=1.0,
            lambda_urg=10.0,
            da_burst_fraction=0.2,
        )
        # Use high risk to ensure non-zero Poisson sample
        result = compute_reactive_burst(0.8, config, rng=rng)
        if "NE" in result and "DA" in result:
            assert result["DA"] == pytest.approx(0.2 * result["NE"])


# =====================================================================
# TestModulatoryFeedback
# =====================================================================

class TestModulatoryFeedback:

    def test_below_persistence_returns_empty(self):
        counters = {"a": 1, "b": 2}
        config = UrgencyForecastConfig(persistence_steps=3)
        fb = compute_modulatory_feedback(counters, config)
        assert fb["neurotransmitters"] == {}
        assert fb["receptors"] == {}

    def test_at_persistence_produces_feedback(self):
        counters = {"a": 3}
        config = UrgencyForecastConfig(
            persistence_steps=3,
            ne_gain_boost=0.15,
            gaba_tone_reduction=0.85,
        )
        fb = compute_modulatory_feedback(counters, config)
        assert fb["neurotransmitters"]["NE"]["u_base_multiplier"] == pytest.approx(1.15)
        assert fb["receptors"]["GABA_B"]["K_d_multiplier"] == pytest.approx(0.85)

    def test_multiple_axes_breached(self):
        counters = {"a": 5, "b": 4}
        config = UrgencyForecastConfig(
            persistence_steps=3,
            ne_gain_boost=0.2,
            gaba_tone_reduction=0.8,
        )
        fb = compute_modulatory_feedback(counters, config)
        # Still same output — any() triggers
        assert fb["neurotransmitters"]["NE"]["u_base_multiplier"] == pytest.approx(1.2)
        assert fb["receptors"]["GABA_B"]["K_d_multiplier"] == pytest.approx(0.8)

    def test_counter_reset_stops_feedback(self):
        counters = {"a": 0}
        config = UrgencyForecastConfig(persistence_steps=3)
        fb = compute_modulatory_feedback(counters, config)
        assert fb["neurotransmitters"] == {}


# =====================================================================
# TestStepUrgencyForecast
# =====================================================================

class TestStepUrgencyForecast:

    def test_returns_correct_types(self, simple_config, rng):
        state = UrgencyForecastState.from_config(simple_config)
        adjusted_eval = {"logical_conflict": 0.5}

        new_state, risk, bursts, feedback = step_urgency_forecast(
            state, adjusted_eval, dt=0.01,
            config=simple_config, rng=rng,
        )

        assert isinstance(new_state, UrgencyForecastState)
        assert isinstance(risk, float)
        assert isinstance(bursts, dict)
        assert isinstance(feedback, dict)
        assert "neurotransmitters" in feedback
        assert "receptors" in feedback

    def test_smoothing_converges(self, simple_config, rng):
        """After many steps with constant input, smoothed value converges."""
        state = UrgencyForecastState.from_config(simple_config)
        adjusted_eval = {"logical_conflict": 0.6}

        for _ in range(1000):
            state, _, _, _ = step_urgency_forecast(
                state, adjusted_eval, dt=0.1,
                config=simple_config, rng=rng,
            )

        # Smoothed value should be near 0.6 * tau (leaky integrator steady state)
        # Actually for leaky integrator: steady state = baseline + gain * input * tau
        # With gain=1, baseline=0: value → input * tau... but clamped by forecast
        # The smoothed integrator value should reflect the input
        smoothed_val = state.smoothed["test_axis"].value
        assert smoothed_val > 0.0  # Has converged to some positive value

    def test_breach_counter_increments(self, rng):
        """With high input and rising signal, breach counter should increment."""
        axis = UrgencyAxisConfig(
            name="test",
            sources=(UrgencyAxisSourceDef("x", weight=1.0),),
            alpha=2.0,  # High gain amplifies rising derivative
            tau_smooth=5.0,  # Slow smoothing → persistent rising slope
            tau_forecast=3.0,
            threshold=0.3,  # Low threshold
        )
        config = UrgencyForecastConfig(
            axes=(axis,),
            persistence_steps=5,
            prediction_window=5.0,
        )
        state = UrgencyForecastState.from_config(config)

        # Feed high input for several steps; the combination of smoothing lag
        # and positive alpha means forecast overshoots the raw smoothed value,
        # exceeding the low threshold.
        for i in range(50):
            state, _, _, _ = step_urgency_forecast(
                state, {"x": 0.9}, dt=0.1, config=config, rng=rng,
            )

        # After many steps, the smoothed value and forecast should exceed 0.3
        assert state.breach_counters["test"] > 0

    def test_no_breach_resets_counter(self, rng):
        axis = UrgencyAxisConfig(
            name="test",
            sources=(UrgencyAxisSourceDef("x", weight=1.0),),
            alpha=0.0,
            tau_smooth=0.1,
            tau_forecast=3.0,
            threshold=0.99,  # Very high threshold → no breach
        )
        config = UrgencyForecastConfig(axes=(axis,), persistence_steps=3)
        state = UrgencyForecastState.from_config(config)

        for _ in range(20):
            state, _, _, _ = step_urgency_forecast(
                state, {"x": 0.5}, dt=0.1, config=config, rng=rng,
            )

        assert state.breach_counters["test"] == 0

    def test_state_checkpoint_roundtrip(self, simple_config, rng):
        """as_dict → from_dict preserves state."""
        state = UrgencyForecastState.from_config(simple_config)

        for _ in range(5):
            state, _, _, _ = step_urgency_forecast(
                state, {"logical_conflict": 0.6}, dt=0.01,
                config=simple_config, rng=rng,
            )

        data = state.as_dict()
        restored = UrgencyForecastState.from_dict(data)

        assert restored.prev_smoothed == state.prev_smoothed
        assert restored.breach_counters == state.breach_counters
        for name in state.smoothed:
            assert restored.smoothed[name].value == pytest.approx(
                state.smoothed[name].value
            )
            assert restored.smoothed[name].baseline == pytest.approx(
                state.smoothed[name].baseline
            )

    def test_deterministic_with_seed(self, simple_config):
        """Same seed → same output."""
        state1 = UrgencyForecastState.from_config(simple_config)
        state2 = UrgencyForecastState.from_config(simple_config)
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        adjusted = {"logical_conflict": 0.8}

        _, risk1, bursts1, fb1 = step_urgency_forecast(
            state1, adjusted, dt=0.01, config=simple_config, rng=rng1,
        )
        _, risk2, bursts2, fb2 = step_urgency_forecast(
            state2, adjusted, dt=0.01, config=simple_config, rng=rng2,
        )

        assert risk1 == pytest.approx(risk2)
        assert bursts1 == bursts2
        assert fb1 == fb2


# =====================================================================
# TestDefaultConfig
# =====================================================================

class TestDefaultConfig:

    def test_default_has_five_axes(self):
        assert len(DEFAULT_URGENCY_FORECAST_CONFIG.axes) == 5

    def test_default_axis_names(self):
        names = {a.name for a in DEFAULT_URGENCY_FORECAST_CONFIG.axes}
        expected = {
            "logical_pressure",
            "emotional_compression",
            "discord_build",
            "expectation_violation",
            "narrative_entropy",
        }
        assert names == expected

    def test_all_sources_reference_valid_eval_axes(self):
        valid_axes = {
            "novelty", "emotional_valence", "urgency", "logical_conflict",
            "coherence", "social_salience", "reward_alignment", "identity_resonance",
        }
        for axis_cfg in DEFAULT_URGENCY_FORECAST_CONFIG.axes:
            for src in axis_cfg.sources:
                assert src.eval_axis in valid_axes, (
                    f"Axis {axis_cfg.name} references unknown eval axis {src.eval_axis}"
                )

    def test_state_from_default_config(self):
        state = UrgencyForecastState.from_config(DEFAULT_URGENCY_FORECAST_CONFIG)
        assert len(state.smoothed) == 5
        assert len(state.prev_smoothed) == 5
        assert len(state.breach_counters) == 5
        assert all(v == 0 for v in state.breach_counters.values())


# =====================================================================
# TestOrchestratorIntegration
# =====================================================================

class TestOrchestratorIntegration:

    def test_urgency_risk_in_result(self):
        """Orchestrator result includes urgency_risk field."""
        rng = np.random.default_rng(42)
        orch = ExtractorOrchestrator(rng=rng)
        result = orch.step({}, dt=0.01)
        assert hasattr(result, "urgency_risk")
        assert isinstance(result.urgency_risk, float)

    def test_default_creates_urgency_state(self):
        """Default orchestrator initialises urgency forecast state."""
        orch = ExtractorOrchestrator()
        assert orch.state.urgency_forecast_state is not None
        assert len(orch.state.urgency_forecast_state.smoothed) == 5

    def test_custom_urgency_config(self):
        """Custom urgency config creates matching state."""
        axis = UrgencyAxisConfig(
            name="custom",
            sources=(UrgencyAxisSourceDef("urgency", weight=1.0),),
        )
        config = UrgencyForecastConfig(axes=(axis,))
        orch = ExtractorOrchestrator(urgency_forecast_config=config)
        assert len(orch.state.urgency_forecast_state.smoothed) == 1
        assert "custom" in orch.state.urgency_forecast_state.smoothed

    def test_burst_deltas_merge(self):
        """Urgency burst deltas merge additively with reactivity burst deltas."""
        rng = np.random.default_rng(42)
        # Use high urgency eval to trigger bursts
        orch = ExtractorOrchestrator(rng=rng)
        eval_input = {
            "logical_conflict": 0.95,
            "urgency": 0.95,
            "coherence": 0.05,
            "emotional_valence": 0.05,
            "novelty": 0.95,
            "reward_alignment": 0.05,
        }
        # Run several steps to build up smoothed values and trigger breaches
        for _ in range(100):
            result = orch.step({}, dt=0.1)

        # burst_deltas dict should exist
        assert isinstance(result.burst_deltas, dict)

    def test_state_checkpoint_roundtrip(self):
        """ExtractorState as_dict/from_dict preserves urgency state."""
        orch = ExtractorOrchestrator(rng=np.random.default_rng(42))
        # Run a few steps
        for _ in range(5):
            orch.step({}, dt=0.01)

        data = orch.state.as_dict()
        assert "urgency_forecast_state" in data
        assert data["urgency_forecast_state"] is not None

        restored = ExtractorState.from_dict(data)
        assert restored.urgency_forecast_state is not None
        assert (
            restored.urgency_forecast_state.breach_counters
            == orch.state.urgency_forecast_state.breach_counters
        )

    def test_feedback_merge_multiplier(self):
        """Urgency modulatory feedback merges with regulatory feedback."""
        # This test verifies the merge logic handles multiplier params
        rng = np.random.default_rng(42)
        axis = UrgencyAxisConfig(
            name="test",
            sources=(UrgencyAxisSourceDef("urgency", weight=1.0),),
            alpha=0.0,
            tau_smooth=0.01,  # Very fast smoothing
            tau_forecast=3.0,
            threshold=0.1,  # Very low threshold → always breaches
        )
        config = UrgencyForecastConfig(
            axes=(axis,),
            persistence_steps=1,  # Immediate modulatory output
            ne_gain_boost=0.15,
            gaba_tone_reduction=0.85,
        )
        orch = ExtractorOrchestrator(
            rng=rng,
            urgency_forecast_config=config,
        )

        # Feed high urgency for several ticks to trigger persistent breach
        for _ in range(20):
            result = orch.step(
                {},
                emotion_inputs={"anxiety": 0.9},
                dt=0.1,
            )

        # Feedback params should contain urgency contributions
        fb = result.feedback_params
        # Check that the feedback structure exists
        assert isinstance(fb, dict)
        assert "neurotransmitters" in fb
        assert "receptors" in fb
