"""Tests for Evaluation Vector assembler (Stochastic Extractor 1)."""

import numpy as np
import pytest

from zados.reward.base.types import RewardDomainResult, RewardSubscore
from zados.neurochem.extractors.evaluation_vector import (
    EvaluationAxisConfig,
    EvaluationVectorConfig,
    DEFAULT_EVALUATION_CONFIG,
    extract_axis_value,
    inject_noise,
    assemble_evaluation_vector,
)


# =====================================================================
# Helpers
# =====================================================================

def _make_domain_result(domain, general_score=0.5, subscores=None):
    """Build a RewardDomainResult with specified subscores."""
    subs = {}
    if subscores:
        for name, score in subscores.items():
            subs[name] = RewardSubscore(name=name, score=score)
    return RewardDomainResult(domain=domain, general_score=general_score, subscores=subs)


def _make_all_domains():
    """Build a full set of 4 domain results with known subscores."""
    return {
        "innovation": _make_domain_result("innovation", 0.7, {
            "novelty_generation": 0.8,
            "conceptual_novelty": 0.6,
            "pattern_divergence": 0.5,
        }),
        "logic": _make_domain_result("logic", 0.6, {
            "internal_consistency": 0.9,
            "semantic_continuity": 0.75,
            "epistemic_calibration": 0.65,
        }),
        "human_attunement": _make_domain_result("human_attunement", 0.65, {
            "empathetic_inference": 0.7,
            "cognitive_reading": 0.6,
            "intention_calibration": 0.55,
        }),
        "ethics": _make_domain_result("ethics", 0.8, {
            "failure_mode_awareness": 0.85,
            "intent_clarity": 0.9,
            "timeline_reflection": 0.7,
        }),
    }


# =====================================================================
# EvaluationAxisConfig tests
# =====================================================================

class TestEvaluationAxisConfig:
    def test_frozen(self):
        cfg = EvaluationAxisConfig("novelty", "innovation", "novelty_generation")
        with pytest.raises(AttributeError):
            cfg.name = "other"

    def test_defaults(self):
        cfg = EvaluationAxisConfig("x", "y", "z")
        assert cfg.transform == "identity"
        assert cfg.sigma == 0.0
        assert cfg.weight == 1.0


# =====================================================================
# extract_axis_value tests
# =====================================================================

class TestExtractAxisValue:
    def test_identity_transform(self):
        results = _make_all_domains()
        axis = EvaluationAxisConfig("novelty", "innovation", "novelty_generation")
        assert extract_axis_value(results, axis) == pytest.approx(0.8)

    def test_invert_transform(self):
        results = _make_all_domains()
        axis = EvaluationAxisConfig(
            "logical_conflict", "logic", "internal_consistency", transform="invert",
        )
        # 1 - 0.9 = 0.1
        assert extract_axis_value(results, axis) == pytest.approx(0.1)

    def test_general_score_transform(self):
        results = _make_all_domains()
        axis = EvaluationAxisConfig(
            "reward_alignment", "innovation", "general_score", transform="general_score",
        )
        assert extract_axis_value(results, axis) == pytest.approx(0.7)

    def test_missing_domain_returns_zero(self):
        results = _make_all_domains()
        axis = EvaluationAxisConfig("x", "nonexistent_domain", "some_key")
        assert extract_axis_value(results, axis) == 0.0

    def test_missing_subscore_returns_zero(self):
        results = _make_all_domains()
        axis = EvaluationAxisConfig("x", "innovation", "nonexistent_subscore")
        assert extract_axis_value(results, axis) == 0.0

    def test_weight_scales_value(self):
        results = _make_all_domains()
        axis = EvaluationAxisConfig(
            "novelty", "innovation", "novelty_generation", weight=0.5,
        )
        # 0.8 * 0.5 = 0.4
        assert extract_axis_value(results, axis) == pytest.approx(0.4)

    def test_clamped_to_unit(self):
        results = _make_all_domains()
        # weight > 1 could push value above 1.0
        axis = EvaluationAxisConfig(
            "novelty", "innovation", "novelty_generation", weight=2.0,
        )
        # 0.8 * 2.0 = 1.6, clamped to 1.0
        assert extract_axis_value(results, axis) == pytest.approx(1.0)

    def test_invert_high_score_gives_low_conflict(self):
        """High internal_consistency (0.9) → low logical_conflict (0.1)."""
        results = _make_all_domains()
        axis = EvaluationAxisConfig(
            "conflict", "logic", "internal_consistency", transform="invert",
        )
        assert extract_axis_value(results, axis) == pytest.approx(0.1)


# =====================================================================
# inject_noise tests
# =====================================================================

class TestInjectNoise:
    def test_zero_sigma_no_change(self):
        rng = np.random.default_rng(42)
        assert inject_noise(0.5, 0.0, rng) == 0.5

    def test_noise_reproducible(self):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        v1 = inject_noise(0.5, 0.1, rng1)
        v2 = inject_noise(0.5, 0.1, rng2)
        assert v1 == pytest.approx(v2)

    def test_noise_changes_value(self):
        rng = np.random.default_rng(42)
        noisy = inject_noise(0.5, 0.2, rng)
        # Very unlikely to be exactly 0.5 with sigma=0.2
        assert noisy != 0.5

    def test_clamped_lower(self):
        rng = np.random.default_rng(42)
        # Large sigma on value near 0 should clamp
        result = inject_noise(0.01, 10.0, rng)
        assert 0.0 <= result <= 1.0

    def test_clamped_upper(self):
        rng = np.random.default_rng(42)
        result = inject_noise(0.99, 10.0, rng)
        assert 0.0 <= result <= 1.0


# =====================================================================
# assemble_evaluation_vector tests
# =====================================================================

class TestAssembleEvaluationVector:
    def test_default_config_produces_8_axes(self):
        results = _make_all_domains()
        E = assemble_evaluation_vector(results)
        assert len(E) == 8
        expected_keys = {
            "novelty", "emotional_valence", "urgency", "logical_conflict",
            "coherence", "social_salience", "reward_alignment", "identity_resonance",
        }
        assert set(E.keys()) == expected_keys

    def test_all_values_in_unit(self):
        results = _make_all_domains()
        E = assemble_evaluation_vector(results)
        for name, val in E.items():
            assert 0.0 <= val <= 1.0, f"{name} = {val} out of [0,1]"

    def test_novelty_axis(self):
        results = _make_all_domains()
        E = assemble_evaluation_vector(results)
        assert E["novelty"] == pytest.approx(0.8)

    def test_logical_conflict_inverted(self):
        results = _make_all_domains()
        E = assemble_evaluation_vector(results)
        # internal_consistency = 0.9, inverted = 0.1
        assert E["logical_conflict"] == pytest.approx(0.1)

    def test_reward_alignment_uses_general_score(self):
        results = _make_all_domains()
        E = assemble_evaluation_vector(results)
        # innovation general_score = 0.7
        assert E["reward_alignment"] == pytest.approx(0.7)

    def test_social_salience(self):
        results = _make_all_domains()
        E = assemble_evaluation_vector(results)
        assert E["social_salience"] == pytest.approx(0.6)

    def test_with_noise(self):
        results = _make_all_domains()
        noisy_config = EvaluationVectorConfig(axes=(
            EvaluationAxisConfig("novelty", "innovation", "novelty_generation", sigma=0.1),
        ))
        rng = np.random.default_rng(42)
        E = assemble_evaluation_vector(results, config=noisy_config, rng=rng)
        # Should be close to 0.8 but not exact
        assert 0.0 <= E["novelty"] <= 1.0

    def test_no_rng_ignores_sigma(self):
        results = _make_all_domains()
        noisy_config = EvaluationVectorConfig(axes=(
            EvaluationAxisConfig("novelty", "innovation", "novelty_generation", sigma=0.5),
        ))
        E = assemble_evaluation_vector(results, config=noisy_config, rng=None)
        # No RNG → no noise → exact value
        assert E["novelty"] == pytest.approx(0.8)

    def test_empty_domain_results(self):
        E = assemble_evaluation_vector({})
        assert len(E) == 8
        for val in E.values():
            assert val == 0.0

    def test_custom_config(self):
        results = _make_all_domains()
        custom = EvaluationVectorConfig(axes=(
            EvaluationAxisConfig("my_axis", "logic", "epistemic_calibration"),
        ))
        E = assemble_evaluation_vector(results, config=custom)
        assert len(E) == 1
        assert E["my_axis"] == pytest.approx(0.65)
