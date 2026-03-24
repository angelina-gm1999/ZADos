"""Tests for emotion saturation tracker (Extractor 4 state tracking)."""

import pytest

from zados.neurochem.extractors.leaky_integrator import LeakyIntegratorState
from zados.neurochem.extractors.emotion_tracker import (
    EmotionTrackerConfig,
    EmotionTrackerState,
    DEFAULT_EMOTION_TRACKER_CONFIGS,
    step_emotion_tracker,
    get_dominant_emotion,
    get_saturation,
    get_emotion_saturations,
)


# =====================================================================
# EmotionTrackerState
# =====================================================================

class TestEmotionTrackerState:
    def test_from_emotion_ids_default(self):
        state = EmotionTrackerState.from_emotion_ids()
        assert len(state.integrators) == 12

    def test_from_emotion_ids_custom(self):
        state = EmotionTrackerState.from_emotion_ids(["joy", "fear"])
        assert len(state.integrators) == 2
        assert "joy" in state.integrators
        assert "fear" in state.integrators

    def test_initial_values_zero(self):
        state = EmotionTrackerState.from_emotion_ids(["joy"])
        assert state.integrators["joy"].value == 0.0

    def test_roundtrip_dict(self):
        state = EmotionTrackerState.from_emotion_ids(["joy", "curiosity"])
        state.integrators["joy"].value = 0.5
        d = state.as_dict()
        restored = EmotionTrackerState.from_dict(d)
        assert restored.integrators["joy"].value == 0.5
        assert restored.integrators["curiosity"].value == 0.0


# =====================================================================
# step_emotion_tracker
# =====================================================================

class TestStepEmotionTracker:
    def test_input_increases_saturation(self):
        state = EmotionTrackerState.from_emotion_ids(["joy"])
        state = step_emotion_tracker(state, {"joy": 1.0}, dt=0.1)
        assert state.integrators["joy"].value > 0.0

    def test_no_input_decays(self):
        state = EmotionTrackerState.from_emotion_ids(["joy"])
        state.integrators["joy"] = LeakyIntegratorState(value=0.5, baseline=0.0)
        state = step_emotion_tracker(state, {}, dt=0.1)
        assert state.integrators["joy"].value < 0.5

    def test_clamped_non_negative(self):
        state = EmotionTrackerState.from_emotion_ids(["joy"])
        # Force a very large negative drive
        state = step_emotion_tracker(state, {"joy": -100.0}, dt=1.0)
        assert state.integrators["joy"].value >= 0.0

    def test_clamped_at_cap(self):
        state = EmotionTrackerState.from_emotion_ids(["joy"])
        for _ in range(10000):
            state = step_emotion_tracker(state, {"joy": 1.0}, dt=0.1)
        # Should not exceed saturation_cap (default 1.0)
        assert state.integrators["joy"].value <= 1.0

    def test_multi_emotion_independent(self):
        state = EmotionTrackerState.from_emotion_ids(["joy", "fear"])
        state = step_emotion_tracker(
            state, {"joy": 1.0, "fear": 0.0}, dt=0.1,
        )
        assert state.integrators["joy"].value > 0.0
        assert state.integrators["fear"].value == 0.0

    def test_custom_tau_affects_dynamics(self):
        """Different tau → different decay rates."""
        fast_cfg = {"joy": EmotionTrackerConfig("joy", tau=1.0, gain=1.0)}
        slow_cfg = {"joy": EmotionTrackerConfig("joy", tau=100.0, gain=1.0)}

        state_fast = EmotionTrackerState.from_emotion_ids(["joy"])
        state_fast.integrators["joy"] = LeakyIntegratorState(value=0.5, baseline=0.0)
        state_fast = step_emotion_tracker(state_fast, {}, dt=0.1, configs=fast_cfg)

        state_slow = EmotionTrackerState.from_emotion_ids(["joy"])
        state_slow.integrators["joy"] = LeakyIntegratorState(value=0.5, baseline=0.0)
        state_slow = step_emotion_tracker(state_slow, {}, dt=0.1, configs=slow_cfg)

        # Fast decays more
        assert state_fast.integrators["joy"].value < state_slow.integrators["joy"].value

    def test_default_configs_12_emotions(self):
        assert len(DEFAULT_EMOTION_TRACKER_CONFIGS) == 12

    def test_surprise_has_short_tau(self):
        """Surprise should have short tau (fast decay) — high gain, quick burst."""
        cfg = DEFAULT_EMOTION_TRACKER_CONFIGS["surprise"]
        assert cfg.tau < 8.0  # shorter than most others

    def test_sadness_has_long_tau(self):
        """Sadness should have long tau (slow decay)."""
        cfg = DEFAULT_EMOTION_TRACKER_CONFIGS["sadness"]
        assert cfg.tau > 15.0


# =====================================================================
# get_dominant_emotion
# =====================================================================

class TestGetDominantEmotion:
    def test_single_emotion(self):
        state = EmotionTrackerState.from_emotion_ids(["joy"])
        state.integrators["joy"] = LeakyIntegratorState(value=0.7, baseline=0.0)
        eid, val = get_dominant_emotion(state)
        assert eid == "joy"
        assert val == 0.7

    def test_multiple_emotions(self):
        state = EmotionTrackerState.from_emotion_ids(["joy", "fear", "calm"])
        state.integrators["joy"] = LeakyIntegratorState(value=0.3, baseline=0.0)
        state.integrators["fear"] = LeakyIntegratorState(value=0.9, baseline=0.0)
        state.integrators["calm"] = LeakyIntegratorState(value=0.1, baseline=0.0)
        eid, val = get_dominant_emotion(state)
        assert eid == "fear"
        assert val == 0.9

    def test_empty_state(self):
        state = EmotionTrackerState()
        eid, val = get_dominant_emotion(state)
        assert eid == "none"
        assert val == 0.0


# =====================================================================
# get_saturation
# =====================================================================

class TestGetSaturation:
    def test_zero_state(self):
        state = EmotionTrackerState.from_emotion_ids(["joy", "fear"])
        assert get_saturation(state) == 0.0

    def test_known_values(self):
        state = EmotionTrackerState.from_emotion_ids(["joy", "fear"])
        state.integrators["joy"] = LeakyIntegratorState(value=0.6, baseline=0.0)
        state.integrators["fear"] = LeakyIntegratorState(value=0.4, baseline=0.0)
        assert get_saturation(state) == pytest.approx(0.5)

    def test_empty_state(self):
        state = EmotionTrackerState()
        assert get_saturation(state) == 0.0


# =====================================================================
# get_emotion_saturations
# =====================================================================

class TestGetEmotionSaturations:
    def test_returns_all(self):
        state = EmotionTrackerState.from_emotion_ids(["joy", "fear"])
        state.integrators["joy"] = LeakyIntegratorState(value=0.3, baseline=0.0)
        sats = get_emotion_saturations(state)
        assert sats["joy"] == 0.3
        assert sats["fear"] == 0.0

    def test_empty(self):
        state = EmotionTrackerState()
        assert get_emotion_saturations(state) == {}
