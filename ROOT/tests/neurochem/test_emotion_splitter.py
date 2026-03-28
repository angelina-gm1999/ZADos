"""Tests for emotion 4M/4R splitter (Extractor 4M/4R routing)."""

import pytest

from zados.neurochem.extractors.leaky_integrator import LeakyIntegratorState
from zados.neurochem.extractors.emotion_tracker import EmotionTrackerState
from zados.neurochem.extractors.emotion_splitter import (
    EmotionSplitConfig,
    DEFAULT_EMOTION_SPLIT_CONFIGS,
    compute_modulatory_adjustments,
    compute_reactive_signals,
    split_emotion_effects,
)


# =====================================================================
# Helpers
# =====================================================================

def _make_tracker_state(saturations: dict) -> EmotionTrackerState:
    """Build a tracker state with given saturation values."""
    integrators = {
        eid: LeakyIntegratorState(value=val, baseline=0.0)
        for eid, val in saturations.items()
    }
    return EmotionTrackerState(integrators=integrators)


# =====================================================================
# DEFAULT_EMOTION_SPLIT_CONFIGS
# =====================================================================

class TestDefaultSplitConfigs:
    def test_covers_12_emotions(self):
        assert len(DEFAULT_EMOTION_SPLIT_CONFIGS) == 12

    def test_fractions_sum_to_one(self):
        for eid, cfg in DEFAULT_EMOTION_SPLIT_CONFIGS.items():
            total = cfg.modulatory_fraction + cfg.reactive_fraction
            assert total == pytest.approx(1.0), (
                f"{eid}: modulatory={cfg.modulatory_fraction} + "
                f"reactive={cfg.reactive_fraction} = {total}"
            )

    def test_all_fractions_in_range(self):
        for eid, cfg in DEFAULT_EMOTION_SPLIT_CONFIGS.items():
            assert 0.0 <= cfg.modulatory_fraction <= 1.0
            assert 0.0 <= cfg.reactive_fraction <= 1.0


# =====================================================================
# compute_modulatory_adjustments
# =====================================================================

class TestComputeModulatoryAdjustments:
    def test_single_emotion(self):
        """Joy with saturation 1.0 → modulatory contribution to reward_alignment."""
        configs = {
            "joy": EmotionSplitConfig(
                "joy", modulatory_fraction=0.6, reactive_fraction=0.4,
                modulatory_target_axes={"reward_alignment": 0.5},
            ),
        }
        adj = compute_modulatory_adjustments({"joy": 1.0}, configs)
        # 1.0 * 0.6 * 0.5 = 0.3
        assert adj["reward_alignment"] == pytest.approx(0.3)

    def test_multi_emotion_same_axis(self):
        """Two emotions contributing to same axis should sum."""
        configs = {
            "joy": EmotionSplitConfig(
                "joy", modulatory_fraction=0.5, reactive_fraction=0.5,
                modulatory_target_axes={"novelty": 0.4},
            ),
            "curiosity": EmotionSplitConfig(
                "curiosity", modulatory_fraction=0.5, reactive_fraction=0.5,
                modulatory_target_axes={"novelty": 0.6},
            ),
        }
        adj = compute_modulatory_adjustments(
            {"joy": 1.0, "curiosity": 1.0}, configs,
        )
        # (1.0*0.5*0.4) + (1.0*0.5*0.6) = 0.2 + 0.3 = 0.5
        assert adj["novelty"] == pytest.approx(0.5)

    def test_zero_saturation_no_contribution(self):
        configs = {
            "joy": EmotionSplitConfig(
                "joy", modulatory_fraction=0.6, reactive_fraction=0.4,
                modulatory_target_axes={"novelty": 0.5},
            ),
        }
        adj = compute_modulatory_adjustments({"joy": 0.0}, configs)
        assert len(adj) == 0

    def test_missing_config_skipped(self):
        adj = compute_modulatory_adjustments({"unknown": 0.8}, {})
        assert len(adj) == 0

    def test_negative_weight_decreases_axis(self):
        """Sadness with negative weight should produce negative adjustment."""
        configs = {
            "sadness": EmotionSplitConfig(
                "sadness", modulatory_fraction=0.7, reactive_fraction=0.3,
                modulatory_target_axes={"emotional_valence": -0.3},
            ),
        }
        adj = compute_modulatory_adjustments({"sadness": 0.8}, configs)
        # 0.8 * 0.7 * (-0.3) = -0.168
        assert adj["emotional_valence"] < 0.0
        assert adj["emotional_valence"] == pytest.approx(-0.168)

    def test_multi_axis_per_emotion(self):
        configs = {
            "anxiety": EmotionSplitConfig(
                "anxiety", modulatory_fraction=0.7, reactive_fraction=0.3,
                modulatory_target_axes={"urgency": 0.4, "logical_conflict": 0.2},
            ),
        }
        adj = compute_modulatory_adjustments({"anxiety": 1.0}, configs)
        assert "urgency" in adj
        assert "logical_conflict" in adj
        assert adj["urgency"] == pytest.approx(0.28)
        assert adj["logical_conflict"] == pytest.approx(0.14)


# =====================================================================
# compute_reactive_signals
# =====================================================================

class TestComputeReactiveSignals:
    def test_single_emotion(self):
        configs = {
            "fear": EmotionSplitConfig(
                "fear", modulatory_fraction=0.4, reactive_fraction=0.6,
                reactive_boost_gain=1.5,
            ),
        }
        profile = compute_reactive_signals({"fear": 0.8}, configs)
        # 0.8 * 0.6 * 1.5 = 0.72
        assert profile["fear"] == pytest.approx(0.72)

    def test_zero_saturation_excluded(self):
        configs = {
            "joy": EmotionSplitConfig("joy", reactive_fraction=0.4),
        }
        profile = compute_reactive_signals({"joy": 0.0}, configs)
        assert len(profile) == 0

    def test_default_boost_gain(self):
        configs = {
            "calm": EmotionSplitConfig(
                "calm", modulatory_fraction=0.8, reactive_fraction=0.2,
                reactive_boost_gain=1.0,
            ),
        }
        profile = compute_reactive_signals({"calm": 1.0}, configs)
        # 1.0 * 0.2 * 1.0 = 0.2
        assert profile["calm"] == pytest.approx(0.2)

    def test_missing_config_skipped(self):
        profile = compute_reactive_signals({"unknown": 0.8}, {})
        assert len(profile) == 0


# =====================================================================
# split_emotion_effects (integration test)
# =====================================================================

class TestSplitEmotionEffects:
    def test_with_defaults(self):
        state = _make_tracker_state({"joy": 0.8, "curiosity": 0.5})
        modulatory, reactive = split_emotion_effects(state)

        # joy has modulatory_target_axes: reward_alignment=0.4, emotional_valence=0.3
        # curiosity has modulatory_target_axes: novelty=0.5, coherence=0.2
        assert "reward_alignment" in modulatory
        assert "novelty" in modulatory

        # reactive should have both emotions
        assert "joy" in reactive
        assert "curiosity" in reactive

    def test_zero_state_empty_outputs(self):
        state = _make_tracker_state({"joy": 0.0, "fear": 0.0})
        modulatory, reactive = split_emotion_effects(state)
        assert len(modulatory) == 0
        assert len(reactive) == 0

    def test_custom_configs(self):
        custom = {
            "test_emo": EmotionSplitConfig(
                "test_emo",
                modulatory_fraction=0.3,
                reactive_fraction=0.7,
                modulatory_target_axes={"urgency": 1.0},
                reactive_boost_gain=2.0,
            ),
        }
        state = _make_tracker_state({"test_emo": 1.0})
        modulatory, reactive = split_emotion_effects(state, custom)
        assert modulatory["urgency"] == pytest.approx(0.3)  # 1.0 * 0.3 * 1.0
        assert reactive["test_emo"] == pytest.approx(1.4)   # 1.0 * 0.7 * 2.0

    def test_modulatory_additive_to_eval(self):
        """Modulatory adjustments should be suitable for adding to E(t)."""
        state = _make_tracker_state({"joy": 0.5})
        modulatory, _ = split_emotion_effects(state)
        # All adjustments should be floats
        for axis, val in modulatory.items():
            assert isinstance(val, float)

    def test_reactive_suitable_for_emotion_profile(self):
        """Reactive output should be suitable for emotion_profile_to_signals()."""
        state = _make_tracker_state({"joy": 0.5, "fear": 0.3})
        _, reactive = split_emotion_effects(state)
        # Should be {emotion_id: float}
        for eid, val in reactive.items():
            assert isinstance(eid, str)
            assert isinstance(val, float)
            assert val > 0.0
