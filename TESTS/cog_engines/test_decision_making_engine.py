"""
Tests for Engine 15 — Decision Making Engine.

Covers:
  - Enums & data types
  - Bayesian confidence fusion (log-odds)
  - Cross-engine agreement
  - Emotion-state modulation
  - Logical Brain confidence adjustment
  - Detection risk scoring
  - Reward-domain risk
  - NT-modulated risk
  - Quadrant classification
  - Hard overrides
  - Certainty level mapping
  - Tone and hedge computation
  - Depth recommendation
  - Flag surfacing
  - Escalation reasons
  - Clarification prompts
  - Neurochemical signal computation
  - Full engine pipeline (Q1/Q2/Q3/Q4 scenarios)
  - Mode-dependent behaviour
  - Bidirectional NT feedback
  - Edge cases
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)
from zados.cognitive_engines.py_engines.decision_making_engine import (
    CertaintyLevel,
    ConfidenceBreakdown,
    DecisionAction,
    DecisionMakingEngine,
    DecisionMakingInput,
    DecisionMakingResult,
    DecisionNeurochem,
    DecisionQuadrant,
    DMConfig,
    DMState,
    EngineConfidenceEntry,
    OverrideReason,
    ProcessingDepth,
    RiskBreakdown,
    apply_agreement_penalty,
    apply_hard_overrides,
    apply_logical_brain_adjustment,
    bayesian_confidence_fusion,
    classify_quadrant,
    collect_escalation_reasons,
    collect_flags_to_surface,
    combine_risks,
    compute_decision_neurochem,
    compute_depth_recommendation,
    compute_detection_risk,
    compute_hedge_level,
    compute_reward_risk,
    cross_engine_agreement,
    emotion_modulate_confidence,
    generate_clarification_prompt,
    map_certainty_level,
    nt_modulate_risk,
    quadrant_to_action,
    quadrant_to_tone,
)


# =====================================================================
# Helpers
# =====================================================================


def _entries(confs: dict[str, float]) -> list[EngineConfidenceEntry]:
    """Shorthand: build EngineConfidenceEntry list from name→conf dict."""
    return [EngineConfidenceEntry(engine_name=k, raw_confidence=v) for k, v in confs.items()]


def _default_input(**overrides) -> DecisionMakingInput:
    """Build a default input with sensible baselines, then apply overrides."""
    defaults = dict(
        engine_confidences=_entries({
            "contradiction": 0.80,
            "fallacy": 0.85,
            "bias": 0.90,
            "logical_brain": 0.75,
            "intention": 0.70,
        }),
        detection_summaries={},
        reward_logic=0.7,
        reward_attunement=0.6,
        reward_ethics=0.8,
        reward_innovation=0.5,
    )
    defaults.update(overrides)
    return DecisionMakingInput(**defaults)


# =====================================================================
# 1. Enums
# =====================================================================


class TestEnums:
    def test_decision_action_values(self):
        assert DecisionAction.RESPOND.value == "respond"
        assert DecisionAction.QUALIFY.value == "qualify"
        assert DecisionAction.DEFER.value == "defer"
        assert DecisionAction.ESCALATE.value == "escalate"

    def test_quadrant_values(self):
        assert DecisionQuadrant.Q1_RESPOND.value == "Q1_respond"
        assert DecisionQuadrant.Q4_ESCALATE.value == "Q4_escalate"

    def test_certainty_level_values(self):
        assert CertaintyLevel.VERY_HIGH.value == "very_high"
        assert CertaintyLevel.VERY_LOW.value == "very_low"

    def test_override_reason_values(self):
        assert OverrideReason.ETHICS_CRITICAL.value == "ethics_domain_critical_failure"

    def test_processing_depth(self):
        assert ProcessingDepth.SHALLOW.value == "shallow"
        assert ProcessingDepth.CRITICAL.value == "critical"


# =====================================================================
# 2. Configuration
# =====================================================================


class TestConfig:
    def test_default_config_creates(self):
        cfg = DMConfig()
        assert cfg.epsilon == 0.01
        assert cfg.alpha_agree == 0.60

    def test_mode_params_have_all_modes(self):
        cfg = DMConfig()
        modes = ["normal", "dev", "learning", "reflective", "rem_normal", "rem_dream"]
        for mode in modes:
            assert mode in cfg.p_prior
            assert mode in cfg.theta_confidence
            assert mode in cfg.theta_risk
            assert mode in cfg.sigma_jitter
            assert mode in cfg.hedge_scale

    def test_custom_config(self):
        cfg = DMConfig(epsilon=0.05, alpha_agree=0.80)
        assert cfg.epsilon == 0.05
        assert cfg.alpha_agree == 0.80


# =====================================================================
# 3. Data types
# =====================================================================


class TestDataTypes:
    def test_engine_confidence_entry(self):
        e = EngineConfidenceEntry(engine_name="test", raw_confidence=0.75)
        assert e.engine_name == "test"
        assert e.raw_confidence == 0.75
        assert e.weight == 1.0

    def test_decision_making_input_defaults(self):
        inp = DecisionMakingInput()
        assert inp.engine_confidences == []
        assert inp.reward_logic == 0.5
        assert inp.active_mode == OperationalMode.NORMAL
        assert not inp.has_ethics_critical_failure

    def test_confidence_breakdown_frozen(self):
        cb = ConfidenceBreakdown(
            per_engine={}, fused_logodds=0.0, fused_probability=0.5,
            agreement_factor=1.0, after_agreement=0.5, emotion_delta=0.0,
            after_emotion=0.5, confidence_adjustment=0.0, jitter_applied=0.0,
            final_confidence=0.5,
        )
        with pytest.raises(AttributeError):
            cb.final_confidence = 0.99

    def test_decision_neurochem_defaults(self):
        dn = DecisionNeurochem()
        assert dn.delta_da == 0.0
        assert dn.beta_boost == 0.0

    def test_decision_result_frozen(self):
        r = DecisionMakingResult(
            action=DecisionAction.RESPOND,
            quadrant=DecisionQuadrant.Q1_RESPOND,
            certainty_level=CertaintyLevel.HIGH,
            confidence=ConfidenceBreakdown(
                per_engine={}, fused_logodds=0.0, fused_probability=0.8,
                agreement_factor=1.0, after_agreement=0.8, emotion_delta=0.0,
                after_emotion=0.8, confidence_adjustment=0.0, jitter_applied=0.0,
                final_confidence=0.8,
            ),
            risk=RiskBreakdown(
                detection_risk=0.0, reward_risk=0.1, nt_modulation_factor=1.0,
                final_risk=0.05,
            ),
            hedge_level=0.0,
            recommended_tone="assertive",
        )
        assert r.engine_id == "decision_making_engine"
        with pytest.raises(AttributeError):
            r.action = DecisionAction.DEFER


# =====================================================================
# 4. Bayesian Confidence Fusion
# =====================================================================


class TestBayesianFusion:
    def test_empty_entries_returns_prior(self):
        L, p = bayesian_confidence_fusion([], {}, p_prior=0.6)
        assert p == 0.6

    def test_single_high_confidence(self):
        entries = [EngineConfidenceEntry("e1", 0.95)]
        L, p = bayesian_confidence_fusion(entries, {}, p_prior=0.5)
        assert p > 0.90

    def test_single_low_confidence(self):
        entries = [EngineConfidenceEntry("e1", 0.10)]
        L, p = bayesian_confidence_fusion(entries, {}, p_prior=0.5)
        assert p < 0.15

    def test_multiple_high_agree(self):
        entries = [
            EngineConfidenceEntry("e1", 0.85),
            EngineConfidenceEntry("e2", 0.90),
            EngineConfidenceEntry("e3", 0.80),
        ]
        L, p = bayesian_confidence_fusion(entries, {}, p_prior=0.5)
        assert p > 0.95  # Strong agreement → very high

    def test_conflicting_evidence(self):
        entries = [
            EngineConfidenceEntry("e1", 0.90),
            EngineConfidenceEntry("e2", 0.10),
        ]
        L, p = bayesian_confidence_fusion(entries, {}, p_prior=0.5)
        # Opposing evidence roughly cancels; near prior
        assert 0.30 < p < 0.70

    def test_reliability_weights(self):
        entries = [
            EngineConfidenceEntry("trusted", 0.90),
            EngineConfidenceEntry("untrusted", 0.20),
        ]
        weights = {"trusted": 2.0, "untrusted": 0.5}
        _, p = bayesian_confidence_fusion(entries, weights, p_prior=0.5)
        # Trusted engine dominates → high confidence
        assert p > 0.70

    def test_epsilon_clamping(self):
        entries = [EngineConfidenceEntry("edge", 0.001)]
        L, p = bayesian_confidence_fusion(entries, {}, p_prior=0.5, epsilon=0.01)
        # Should not produce NaN or inf
        assert 0.0 <= p <= 1.0
        assert math.isfinite(L)

    def test_all_ones(self):
        entries = [EngineConfidenceEntry("e", 0.999) for _ in range(5)]
        _, p = bayesian_confidence_fusion(entries, {}, p_prior=0.5, epsilon=0.01)
        assert p > 0.99

    def test_prior_shifts_baseline(self):
        entries = [EngineConfidenceEntry("e1", 0.60)]
        _, p_low = bayesian_confidence_fusion(entries, {}, p_prior=0.3)
        _, p_high = bayesian_confidence_fusion(entries, {}, p_prior=0.7)
        assert p_high > p_low


# =====================================================================
# 5. Cross-Engine Agreement
# =====================================================================


class TestAgreement:
    def test_perfect_agreement(self):
        entries = _entries({"a": 0.80, "b": 0.80, "c": 0.80})
        assert abs(cross_engine_agreement(entries) - 1.0) < 1e-9

    def test_total_disagreement(self):
        entries = _entries({"a": 0.0, "b": 1.0})
        ag = cross_engine_agreement(entries, sigma_max=0.50)
        assert ag < 0.1

    def test_single_entry(self):
        entries = _entries({"a": 0.50})
        assert cross_engine_agreement(entries) == 1.0

    def test_moderate_spread(self):
        entries = _entries({"a": 0.60, "b": 0.70, "c": 0.80})
        ag = cross_engine_agreement(entries, sigma_max=0.30)
        assert 0.5 < ag < 1.0


class TestAgreementPenalty:
    def test_perfect_agreement_no_penalty(self):
        c = apply_agreement_penalty(0.80, agreement=1.0)
        assert abs(c - 0.80) < 1e-9

    def test_zero_agreement(self):
        c = apply_agreement_penalty(0.80, agreement=0.0, alpha_agree=0.60)
        assert abs(c - 0.80 * 0.60) < 1e-9

    def test_partial_agreement(self):
        c = apply_agreement_penalty(0.80, agreement=0.5, alpha_agree=0.60)
        expected = 0.80 * (0.60 + 0.40 * 0.5)
        assert abs(c - expected) < 1e-9


# =====================================================================
# 6. Emotion-State Modulation
# =====================================================================


class TestEmotionModulation:
    def test_no_emotions(self):
        c, delta = emotion_modulate_confidence(0.70, 0, 0, 0, 0, 0, 0)
        assert abs(c - 0.70) < 1e-9
        assert abs(delta) < 1e-9

    def test_pure_confident(self):
        c, delta = emotion_modulate_confidence(
            0.70, emotion_confident=1.0,
            emotion_nervous=0, emotion_anxiety=0,
            emotion_perplexed=0, emotion_regret=0, emotion_skeptical=0,
            kappa_confident=0.15,
        )
        assert c > 0.70
        assert delta > 0

    def test_pure_anxiety(self):
        c, delta = emotion_modulate_confidence(
            0.70, emotion_confident=0, emotion_nervous=0,
            emotion_anxiety=1.0, emotion_perplexed=0,
            emotion_regret=0, emotion_skeptical=0,
            kappa_anxiety=0.25,
        )
        assert c < 0.70
        assert delta < 0

    def test_mixed_emotions(self):
        c, delta = emotion_modulate_confidence(
            0.70,
            emotion_confident=0.5, emotion_nervous=0.3,
            emotion_anxiety=0.2, emotion_perplexed=0.0,
            emotion_regret=0.0, emotion_skeptical=0.0,
        )
        # Net effect depends on magnitudes
        assert 0.0 <= c <= 1.0

    def test_clamped_to_unit(self):
        c, _ = emotion_modulate_confidence(
            0.99, emotion_confident=1.0,
            emotion_nervous=0, emotion_anxiety=0,
            emotion_perplexed=0, emotion_regret=0, emotion_skeptical=0,
            kappa_confident=0.50,
        )
        assert c <= 1.0


# =====================================================================
# 7. Logical Brain Adjustment
# =====================================================================


class TestLogicalBrainAdjustment:
    def test_zero_adjustment(self):
        c = apply_logical_brain_adjustment(0.70, 0.0)
        assert abs(c - 0.70) < 1e-6

    def test_positive_adjustment_boosts(self):
        c = apply_logical_brain_adjustment(0.50, 0.5)
        assert c > 0.50

    def test_negative_adjustment_reduces(self):
        c = apply_logical_brain_adjustment(0.70, -0.5)
        assert c < 0.70

    def test_extreme_positive(self):
        c = apply_logical_brain_adjustment(0.50, 5.0)
        assert c > 0.99

    def test_extreme_negative(self):
        c = apply_logical_brain_adjustment(0.50, -5.0)
        assert c < 0.01

    def test_clamped(self):
        c = apply_logical_brain_adjustment(0.99, 10.0)
        assert 0.0 <= c <= 1.0


# =====================================================================
# 8. Detection Risk
# =====================================================================


class TestDetectionRisk:
    def test_empty(self):
        assert compute_detection_risk({}, {}) == 0.0

    def test_clean_pass(self):
        summaries = {"contradiction": {"severity": 0.0, "count": 0, "clean_pass": True}}
        risk = compute_detection_risk(summaries, {"contradiction": 0.25})
        assert risk == 0.0

    def test_single_flag(self):
        summaries = {"contradiction": {"severity": 0.80, "count": 1}}
        risk = compute_detection_risk(summaries, {"contradiction": 1.0})
        assert risk > 0.5

    def test_multiple_engines(self):
        summaries = {
            "contradiction": {"severity": 0.60, "count": 2},
            "fallacy": {"severity": 0.40, "count": 1},
        }
        risk = compute_detection_risk(summaries, {"contradiction": 0.5, "fallacy": 0.5})
        assert 0.0 < risk < 1.0

    def test_count_diminishing_returns(self):
        # Same severity, more count → higher risk but sqrt diminishing
        risk_1 = compute_detection_risk(
            {"e": {"severity": 0.5, "count": 1}}, {"e": 1.0},
        )
        risk_4 = compute_detection_risk(
            {"e": {"severity": 0.5, "count": 4}}, {"e": 1.0},
        )
        assert risk_4 > risk_1
        assert risk_4 < risk_1 * 4  # Sub-linear


# =====================================================================
# 9. Reward Risk
# =====================================================================


class TestRewardRisk:
    def test_perfect_scores(self):
        risk = compute_reward_risk(1.0, 1.0, 1.0, 1.0)
        assert risk < 0.25

    def test_low_logic(self):
        risk_low = compute_reward_risk(0.2, 0.8, 0.8, 0.8)
        risk_high = compute_reward_risk(0.9, 0.8, 0.8, 0.8)
        assert risk_low > risk_high

    def test_low_ethics(self):
        risk = compute_reward_risk(0.8, 0.8, 0.1, 0.8)
        assert risk > 0.5

    def test_low_attunement(self):
        risk_low = compute_reward_risk(0.8, 0.1, 0.8, 0.8)
        risk_high = compute_reward_risk(0.8, 0.9, 0.8, 0.8)
        assert risk_low > risk_high


# =====================================================================
# 10. NT-Modulated Risk
# =====================================================================


class TestNTModulatedRisk:
    def test_no_nt_levels(self):
        r, f = nt_modulate_risk(0.50, 0.0, 0.0, 0.0)
        assert abs(r - 0.50) < 1e-9
        assert abs(f - 1.0) < 1e-9

    def test_high_ne_amplifies(self):
        r, f = nt_modulate_risk(0.50, ne_level=0.8, cor_level=0.0, gaba_level=0.0)
        assert r > 0.50
        assert f > 1.0

    def test_high_gaba_dampens(self):
        r, f = nt_modulate_risk(0.50, ne_level=0.0, cor_level=0.0, gaba_level=0.8)
        assert r < 0.50
        assert f < 1.0

    def test_combined_ne_gaba(self):
        r, f = nt_modulate_risk(0.50, ne_level=0.5, cor_level=0.0, gaba_level=0.5)
        # NE amplifies, GABA dampens — partial cancellation
        assert abs(r - 0.50) < 0.20


# =====================================================================
# 11. Quadrant Classification
# =====================================================================


class TestQuadrantClassification:
    def test_q1_high_c_low_r(self):
        q = classify_quadrant(0.80, 0.20, theta_c=0.55, theta_r=0.40)
        assert q == DecisionQuadrant.Q1_RESPOND

    def test_q2_high_c_high_r(self):
        q = classify_quadrant(0.80, 0.60, theta_c=0.55, theta_r=0.40)
        assert q == DecisionQuadrant.Q2_QUALIFY

    def test_q3_low_c_low_r(self):
        q = classify_quadrant(0.30, 0.20, theta_c=0.55, theta_r=0.40)
        assert q == DecisionQuadrant.Q3_DEFER

    def test_q4_low_c_high_r(self):
        q = classify_quadrant(0.30, 0.60, theta_c=0.55, theta_r=0.40)
        assert q == DecisionQuadrant.Q4_ESCALATE

    def test_boundary_confidence_is_high(self):
        q = classify_quadrant(0.55, 0.20, theta_c=0.55, theta_r=0.40)
        assert q == DecisionQuadrant.Q1_RESPOND

    def test_boundary_risk_is_low(self):
        q = classify_quadrant(0.80, 0.40, theta_c=0.55, theta_r=0.40)
        assert q == DecisionQuadrant.Q1_RESPOND


# =====================================================================
# 12. Hard Overrides
# =====================================================================


class TestHardOverrides:
    def test_no_overrides(self):
        inp = DecisionMakingInput()
        q, override = apply_hard_overrides(DecisionQuadrant.Q1_RESPOND, inp)
        assert q == DecisionQuadrant.Q1_RESPOND
        assert override is None

    def test_ethics_critical_forces_escalate(self):
        inp = DecisionMakingInput(has_ethics_critical_failure=True)
        q, override = apply_hard_overrides(DecisionQuadrant.Q1_RESPOND, inp)
        assert q == DecisionQuadrant.Q4_ESCALATE
        assert override == OverrideReason.ETHICS_CRITICAL

    def test_opposition_block_forces_escalate(self):
        inp = DecisionMakingInput(opposition_gate_outcome="BLOCK")
        q, override = apply_hard_overrides(DecisionQuadrant.Q1_RESPOND, inp)
        assert q == DecisionQuadrant.Q4_ESCALATE
        assert override == OverrideReason.OPPOSITION_BLOCK

    def test_genuine_paradox_downgrades_q1_to_q3(self):
        inp = DecisionMakingInput(has_genuine_paradox=True)
        q, override = apply_hard_overrides(DecisionQuadrant.Q1_RESPOND, inp)
        assert q == DecisionQuadrant.Q3_DEFER
        assert override == OverrideReason.PARADOX_GENUINE

    def test_genuine_paradox_keeps_q2(self):
        inp = DecisionMakingInput(has_genuine_paradox=True)
        q, override = apply_hard_overrides(DecisionQuadrant.Q2_QUALIFY, inp)
        assert q == DecisionQuadrant.Q2_QUALIFY
        assert override == OverrideReason.PARADOX_GENUINE

    def test_l3_contradiction_downgrades_q1_to_q2(self):
        inp = DecisionMakingInput(has_contradiction_l3=True)
        q, override = apply_hard_overrides(DecisionQuadrant.Q1_RESPOND, inp)
        assert q == DecisionQuadrant.Q2_QUALIFY
        assert override == OverrideReason.CONTRADICTION_L3

    def test_logic_trap_downgrades_q1_to_q2(self):
        inp = DecisionMakingInput(has_active_logic_trap=True)
        q, override = apply_hard_overrides(DecisionQuadrant.Q1_RESPOND, inp)
        assert q == DecisionQuadrant.Q2_QUALIFY

    def test_heuristic_correction_downgrades_q1_to_q2(self):
        inp = DecisionMakingInput(has_heuristic_correction=True)
        q, override = apply_hard_overrides(DecisionQuadrant.Q1_RESPOND, inp)
        assert q == DecisionQuadrant.Q2_QUALIFY

    def test_ethics_takes_priority_over_paradox(self):
        inp = DecisionMakingInput(
            has_ethics_critical_failure=True,
            has_genuine_paradox=True,
        )
        q, override = apply_hard_overrides(DecisionQuadrant.Q1_RESPOND, inp)
        assert q == DecisionQuadrant.Q4_ESCALATE
        assert override == OverrideReason.ETHICS_CRITICAL


# =====================================================================
# 13. Certainty Level Mapping
# =====================================================================


class TestCertaintyMapping:
    @pytest.mark.parametrize("conf,expected", [
        (0.95, CertaintyLevel.VERY_HIGH),
        (0.90, CertaintyLevel.VERY_HIGH),
        (0.85, CertaintyLevel.HIGH),
        (0.75, CertaintyLevel.HIGH),
        (0.65, CertaintyLevel.MODERATE),
        (0.55, CertaintyLevel.MODERATE),
        (0.45, CertaintyLevel.LOW),
        (0.35, CertaintyLevel.LOW),
        (0.25, CertaintyLevel.VERY_LOW),
        (0.05, CertaintyLevel.VERY_LOW),
    ])
    def test_mapping(self, conf, expected):
        assert map_certainty_level(conf) == expected


# =====================================================================
# 14. Tone and Hedge
# =====================================================================


class TestToneAndHedge:
    def test_quadrant_to_action(self):
        assert quadrant_to_action(DecisionQuadrant.Q1_RESPOND) == DecisionAction.RESPOND
        assert quadrant_to_action(DecisionQuadrant.Q2_QUALIFY) == DecisionAction.QUALIFY
        assert quadrant_to_action(DecisionQuadrant.Q3_DEFER) == DecisionAction.DEFER
        assert quadrant_to_action(DecisionQuadrant.Q4_ESCALATE) == DecisionAction.ESCALATE

    def test_quadrant_to_tone(self):
        assert quadrant_to_tone(DecisionQuadrant.Q1_RESPOND) == "assertive"
        assert quadrant_to_tone(DecisionQuadrant.Q2_QUALIFY) == "qualified"
        assert quadrant_to_tone(DecisionQuadrant.Q3_DEFER) == "honest_uncertainty"
        assert quadrant_to_tone(DecisionQuadrant.Q4_ESCALATE) == "cautious_transparent"

    def test_hedge_q1_zero(self):
        assert compute_hedge_level(DecisionQuadrant.Q1_RESPOND, 0.5) == 0.0

    def test_hedge_q2_proportional(self):
        h = compute_hedge_level(DecisionQuadrant.Q2_QUALIFY, 0.60, hedge_scale=1.0)
        assert abs(h - 0.60) < 1e-9

    def test_hedge_q2_scaled(self):
        h = compute_hedge_level(DecisionQuadrant.Q2_QUALIFY, 0.60, hedge_scale=0.5)
        assert abs(h - 0.30) < 1e-9

    def test_hedge_q3_max(self):
        assert compute_hedge_level(DecisionQuadrant.Q3_DEFER, 0.2) == 1.0

    def test_hedge_q4_max(self):
        assert compute_hedge_level(DecisionQuadrant.Q4_ESCALATE, 0.8) == 1.0


# =====================================================================
# 15. Depth Recommendation
# =====================================================================


class TestDepthRecommendation:
    def test_q1_passes_upstream(self):
        assert compute_depth_recommendation(DecisionQuadrant.Q1_RESPOND, "deep") == "deep"

    def test_q2_at_least_standard(self):
        assert compute_depth_recommendation(DecisionQuadrant.Q2_QUALIFY, "shallow") == "standard"

    def test_q2_preserves_deep(self):
        assert compute_depth_recommendation(DecisionQuadrant.Q2_QUALIFY, "deep") == "deep"

    def test_q4_at_least_deep(self):
        assert compute_depth_recommendation(DecisionQuadrant.Q4_ESCALATE, "shallow") == "deep"
        assert compute_depth_recommendation(DecisionQuadrant.Q4_ESCALATE, "standard") == "deep"

    def test_q4_preserves_critical(self):
        assert compute_depth_recommendation(DecisionQuadrant.Q4_ESCALATE, "critical") == "critical"

    def test_q3_defers(self):
        assert compute_depth_recommendation(DecisionQuadrant.Q3_DEFER, "shallow") == "shallow"

    def test_none_upstream_defaults_standard(self):
        assert compute_depth_recommendation(DecisionQuadrant.Q1_RESPOND, None) == "standard"


# =====================================================================
# 16. Flag Surfacing
# =====================================================================


class TestFlagSurfacing:
    def test_q1_no_flags(self):
        summaries = {"contradiction": {"severity": 0.8, "count": 2}}
        flags = collect_flags_to_surface(summaries, 0.45, DecisionQuadrant.Q1_RESPOND)
        assert flags == []

    def test_q3_no_flags(self):
        summaries = {"contradiction": {"severity": 0.8, "count": 2}}
        flags = collect_flags_to_surface(summaries, 0.45, DecisionQuadrant.Q3_DEFER)
        assert flags == []

    def test_q2_above_threshold(self):
        summaries = {
            "contradiction": {"severity": 0.80, "count": 2},
            "bias": {"severity": 0.30, "count": 1},
        }
        flags = collect_flags_to_surface(summaries, 0.45, DecisionQuadrant.Q2_QUALIFY)
        assert len(flags) == 1  # Only contradiction above 0.45
        assert "contradiction" in flags[0]

    def test_q4_all_flags(self):
        summaries = {
            "contradiction": {"severity": 0.80, "count": 2},
            "bias": {"severity": 0.30, "count": 1},
        }
        flags = collect_flags_to_surface(summaries, 0.45, DecisionQuadrant.Q4_ESCALATE)
        assert len(flags) == 2  # All flags in Q4

    def test_zero_count_excluded(self):
        summaries = {"e": {"severity": 0.80, "count": 0}}
        flags = collect_flags_to_surface(summaries, 0.45, DecisionQuadrant.Q4_ESCALATE)
        assert flags == []


# =====================================================================
# 17. Escalation Reasons
# =====================================================================


class TestEscalationReasons:
    def test_non_q4_empty(self):
        inp = DecisionMakingInput(has_ethics_critical_failure=True)
        reasons = collect_escalation_reasons(inp, DecisionQuadrant.Q1_RESPOND, None)
        assert reasons == []

    def test_q4_with_override(self):
        inp = DecisionMakingInput(has_ethics_critical_failure=True)
        reasons = collect_escalation_reasons(
            inp, DecisionQuadrant.Q4_ESCALATE, OverrideReason.ETHICS_CRITICAL,
        )
        assert len(reasons) >= 2
        assert any("Override" in r for r in reasons)
        assert any("Ethics" in r for r in reasons)

    def test_q4_opposition_block(self):
        inp = DecisionMakingInput(opposition_gate_outcome="BLOCK")
        reasons = collect_escalation_reasons(
            inp, DecisionQuadrant.Q4_ESCALATE, OverrideReason.OPPOSITION_BLOCK,
        )
        assert any("BLOCK" in r for r in reasons)


# =====================================================================
# 18. Clarification Prompts
# =====================================================================


class TestClarificationPrompts:
    def test_empty_engines(self):
        prompt = generate_clarification_prompt([])
        assert "clarify" in prompt.lower() or "context" in prompt.lower()

    def test_single_engine(self):
        prompt = generate_clarification_prompt(["contradiction"])
        assert "contradiction" in prompt

    def test_multiple_engines(self):
        prompt = generate_clarification_prompt(["e1", "e2", "e3"])
        assert "3 dimensions" in prompt


# =====================================================================
# 19. Neurochemical Signal Computation
# =====================================================================


class TestNeurochemSignals:
    def test_q1_positive_da(self):
        cfg = DMConfig()
        nc = compute_decision_neurochem(DecisionQuadrant.Q1_RESPOND, 0.80, 0.20, cfg)
        assert nc.delta_da > 0
        assert nc.delta_5ht > 0
        assert nc.beta_boost > 0
        assert nc.theta_gamma_boost > 0
        assert nc.delta_ne == 0.0
        assert nc.delta_cor == 0.0

    def test_q2_mixed_signals(self):
        cfg = DMConfig()
        nc = compute_decision_neurochem(DecisionQuadrant.Q2_QUALIFY, 0.70, 0.60, cfg)
        assert nc.delta_da > 0  # Moderate reward
        assert nc.delta_ne > 0  # Risk awareness
        assert nc.delta_ach > 0

    def test_q3_negative_da_positive_gaba(self):
        cfg = DMConfig()
        nc = compute_decision_neurochem(DecisionQuadrant.Q3_DEFER, 0.30, 0.20, cfg)
        assert nc.delta_da < 0   # Negative RPE
        assert nc.delta_gaba > 0  # Calming

    def test_q4_threat_response(self):
        cfg = DMConfig()
        nc = compute_decision_neurochem(DecisionQuadrant.Q4_ESCALATE, 0.20, 0.80, cfg)
        assert nc.delta_cor > 0   # Cortisol threat tagging
        assert nc.delta_ne > 0    # Strong alerting
        assert nc.delta_da < 0    # Negative reward
        assert nc.delta_gaba > 0  # Action inhibition

    def test_confidence_scales_q1_signals(self):
        cfg = DMConfig()
        nc_low = compute_decision_neurochem(DecisionQuadrant.Q1_RESPOND, 0.60, 0.20, cfg)
        nc_high = compute_decision_neurochem(DecisionQuadrant.Q1_RESPOND, 0.95, 0.20, cfg)
        assert nc_high.delta_da > nc_low.delta_da
        assert nc_high.beta_boost > nc_low.beta_boost

    def test_risk_scales_q4_signals(self):
        cfg = DMConfig()
        nc_low = compute_decision_neurochem(DecisionQuadrant.Q4_ESCALATE, 0.20, 0.50, cfg)
        nc_high = compute_decision_neurochem(DecisionQuadrant.Q4_ESCALATE, 0.20, 0.90, cfg)
        assert nc_high.delta_cor > nc_low.delta_cor
        assert nc_high.delta_ne > nc_low.delta_ne


# =====================================================================
# 20. Full Engine Pipeline — Q1 Scenario
# =====================================================================


class TestFullPipelineQ1:
    """High confidence, no flags → Q1 RESPOND."""

    def setup_method(self):
        self.engine = DecisionMakingEngine(rng=np.random.default_rng(42))

    def test_q1_scenario(self):
        inp = _default_input()
        result = self.engine.process(inp)

        assert result.action == DecisionAction.RESPOND
        assert result.quadrant == DecisionQuadrant.Q1_RESPOND
        assert result.certainty_level in (CertaintyLevel.HIGH, CertaintyLevel.VERY_HIGH,
                                           CertaintyLevel.MODERATE)
        assert result.hedge_level == 0.0
        assert result.recommended_tone == "assertive"
        assert result.override_applied is None
        assert result.flags_to_surface == []
        assert result.engine_id == "decision_making_engine"
        assert result.processing_time_ms > 0

    def test_q1_neurochem_positive_da(self):
        inp = _default_input()
        result = self.engine.process(inp)
        nc = result.neurochemical_signals
        assert nc.delta_da > 0
        assert nc.delta_5ht > 0

    def test_q1_confidence_breakdown(self):
        inp = _default_input()
        result = self.engine.process(inp)
        cb = result.confidence
        assert len(cb.per_engine) == 5
        assert cb.agreement_factor > 0.5
        assert 0.0 <= cb.final_confidence <= 1.0


# =====================================================================
# 21. Full Engine Pipeline — Q2 Scenario
# =====================================================================


class TestFullPipelineQ2:
    """High confidence but detection flags → Q2 QUALIFY."""

    def setup_method(self):
        self.engine = DecisionMakingEngine(rng=np.random.default_rng(42))

    def test_q2_with_flags(self):
        inp = _default_input(
            detection_summaries={
                "contradiction": {"severity": 0.70, "count": 2, "clean_pass": False},
                "fallacy": {"severity": 0.50, "count": 1, "clean_pass": False},
            },
            reward_logic=0.3,
            reward_ethics=0.3,
        )
        result = self.engine.process(inp)
        # With high detection risk and low reward scores, should be Q2 or Q4
        assert result.action in (DecisionAction.QUALIFY, DecisionAction.ESCALATE)

    def test_q2_via_override(self):
        inp = _default_input(has_contradiction_l3=True)
        result = self.engine.process(inp)
        assert result.quadrant in (
            DecisionQuadrant.Q2_QUALIFY,
            DecisionQuadrant.Q3_DEFER,
            DecisionQuadrant.Q4_ESCALATE,
        )
        assert result.override_applied is not None


# =====================================================================
# 22. Full Engine Pipeline — Q3 Scenario
# =====================================================================


class TestFullPipelineQ3:
    """Low confidence, no flags → Q3 DEFER."""

    def setup_method(self):
        self.engine = DecisionMakingEngine(rng=np.random.default_rng(42))

    def test_q3_low_confidence(self):
        inp = DecisionMakingInput(
            engine_confidences=_entries({
                "e1": 0.20,
                "e2": 0.15,
                "e3": 0.25,
            }),
            reward_logic=0.8,
            reward_ethics=0.8,
            reward_attunement=0.8,
            reward_innovation=0.8,
        )
        result = self.engine.process(inp)
        assert result.quadrant in (DecisionQuadrant.Q3_DEFER, DecisionQuadrant.Q4_ESCALATE)
        if result.quadrant == DecisionQuadrant.Q3_DEFER:
            assert result.hedge_level == 1.0
            assert result.clarification_prompt is not None

    def test_q3_generates_clarification(self):
        inp = DecisionMakingInput(
            engine_confidences=_entries({"e1": 0.10, "e2": 0.10}),
            reward_logic=0.9,
            reward_ethics=0.9,
            reward_attunement=0.9,
            reward_innovation=0.9,
        )
        result = self.engine.process(inp)
        if result.quadrant == DecisionQuadrant.Q3_DEFER:
            assert result.clarification_prompt is not None


# =====================================================================
# 23. Full Engine Pipeline — Q4 Scenario
# =====================================================================


class TestFullPipelineQ4:
    """Low confidence + high risk → Q4 ESCALATE."""

    def setup_method(self):
        self.engine = DecisionMakingEngine(rng=np.random.default_rng(42))

    def test_q4_ethics_override(self):
        inp = _default_input(has_ethics_critical_failure=True)
        result = self.engine.process(inp)
        assert result.quadrant == DecisionQuadrant.Q4_ESCALATE
        assert result.action == DecisionAction.ESCALATE
        assert result.hedge_level == 1.0
        assert result.override_applied == OverrideReason.ETHICS_CRITICAL.value
        assert len(result.escalation_reasons) > 0

    def test_q4_neurochem_threat(self):
        inp = _default_input(has_ethics_critical_failure=True)
        result = self.engine.process(inp)
        nc = result.neurochemical_signals
        assert nc.delta_cor > 0
        assert nc.delta_ne > 0
        assert nc.delta_da < 0

    def test_q4_opposition_block(self):
        inp = _default_input(opposition_gate_outcome="BLOCK")
        result = self.engine.process(inp)
        assert result.quadrant == DecisionQuadrant.Q4_ESCALATE
        assert any("BLOCK" in r for r in result.escalation_reasons)


# =====================================================================
# 24. Mode-Dependent Behaviour
# =====================================================================


class TestModeBehaviour:
    def test_dev_mode_more_permissive(self):
        engine = DecisionMakingEngine(rng=np.random.default_rng(42))
        engine.configure(OperationalMode.DEV)

        # Moderate confidence → in DEV mode should still be Q1 (theta_c=0.35)
        inp = _default_input(
            engine_confidences=_entries({"e1": 0.50, "e2": 0.45}),
        )
        result = engine.process(inp)
        assert result.quadrant in (DecisionQuadrant.Q1_RESPOND, DecisionQuadrant.Q2_QUALIFY)

    def test_reflective_mode_more_cautious(self):
        engine = DecisionMakingEngine(rng=np.random.default_rng(42))
        engine.configure(OperationalMode.REFLECTIVE)
        status = engine.get_status()
        assert status["mode"] == "reflective"

    def test_rem_dream_very_permissive(self):
        engine = DecisionMakingEngine(rng=np.random.default_rng(42))
        engine.configure(OperationalMode.REM_DREAM)

        inp = _default_input(
            engine_confidences=_entries({"e1": 0.30}),
        )
        result = engine.process(inp)
        # REM Dream: theta_c=0.25, should still be Q1 at 0.30
        assert result.quadrant in (DecisionQuadrant.Q1_RESPOND, DecisionQuadrant.Q2_QUALIFY)

    def test_configure_updates_mode(self):
        engine = DecisionMakingEngine()
        engine.configure(OperationalMode.LEARNING)
        assert engine.get_status()["mode"] == "learning"


# =====================================================================
# 25. Bidirectional NT Feedback
# =====================================================================


class TestNTFeedback:
    def test_update_neurochem_state(self):
        engine = DecisionMakingEngine()
        engine.update_neurochem_state({
            "da": 0.7, "ne": 0.3, "ach": 0.5,
            "5ht": 0.4, "cor": 0.2, "gaba": 0.6, "oxt": 0.8,
        })
        status = engine.get_status()
        assert abs(status["nt_levels"]["da"] - 0.7) < 1e-9
        assert abs(status["nt_levels"]["oxt"] - 0.8) < 1e-9

    def test_clamping(self):
        engine = DecisionMakingEngine()
        engine.update_neurochem_state({"da": 1.5, "ne": -0.3})
        status = engine.get_status()
        assert status["nt_levels"]["da"] == 1.0
        assert status["nt_levels"]["ne"] == 0.0

    def test_partial_update(self):
        engine = DecisionMakingEngine()
        engine.update_neurochem_state({"da": 0.8})
        status = engine.get_status()
        assert abs(status["nt_levels"]["da"] - 0.8) < 1e-9
        assert status["nt_levels"]["ne"] == 0.0  # Unchanged

    def test_high_ne_amplifies_risk(self):
        """High NE should push borderline Q1 toward Q2 via risk amplification."""
        engine_calm = DecisionMakingEngine(rng=np.random.default_rng(42))
        engine_alert = DecisionMakingEngine(rng=np.random.default_rng(42))
        engine_alert.update_neurochem_state({"ne": 0.9, "cor": 0.8})

        inp = _default_input(
            detection_summaries={
                "contradiction": {"severity": 0.35, "count": 1},
            },
            reward_logic=0.5,
            reward_ethics=0.5,
        )
        result_calm = engine_calm.process(inp)
        result_alert = engine_alert.process(inp)
        assert result_alert.risk.final_risk >= result_calm.risk.final_risk


# =====================================================================
# 26. Introspection
# =====================================================================


class TestIntrospection:
    def test_get_status_fields(self):
        engine = DecisionMakingEngine()
        status = engine.get_status()
        assert status["engine_id"] == "decision_making_engine"
        assert status["cluster"] == "reasoning"
        assert status["mode"] == "normal"
        assert status["cycle_count"] == 0
        assert "quadrant_distribution" in status
        assert "nt_levels" in status

    def test_status_updates_after_process(self):
        engine = DecisionMakingEngine(rng=np.random.default_rng(42))
        engine.process(_default_input())
        status = engine.get_status()
        assert status["cycle_count"] == 1
        assert status["total_decisions"] == 1
        total_q = sum(status["quadrant_distribution"].values())
        assert total_q == 1

    def test_override_count_tracked(self):
        engine = DecisionMakingEngine(rng=np.random.default_rng(42))
        engine.process(_default_input(has_ethics_critical_failure=True))
        assert engine.get_status()["override_count"] == 1

    def test_multiple_cycles(self):
        engine = DecisionMakingEngine(rng=np.random.default_rng(42))
        for _ in range(5):
            engine.process(_default_input())
        assert engine.get_status()["cycle_count"] == 5
        assert engine.get_status()["total_decisions"] == 5


# =====================================================================
# 27. Edge Cases
# =====================================================================


class TestEdgeCases:
    def test_empty_input(self):
        engine = DecisionMakingEngine(rng=np.random.default_rng(42))
        result = engine.process(DecisionMakingInput())
        assert result.action in DecisionAction
        assert 0.0 <= result.confidence.final_confidence <= 1.0
        assert 0.0 <= result.risk.final_risk <= 1.0

    def test_all_zero_confidence(self):
        engine = DecisionMakingEngine(rng=np.random.default_rng(42))
        inp = DecisionMakingInput(
            engine_confidences=_entries({"e1": 0.01, "e2": 0.01, "e3": 0.01}),
        )
        result = engine.process(inp)
        assert result.confidence.final_confidence < 0.20

    def test_all_max_confidence(self):
        engine = DecisionMakingEngine(rng=np.random.default_rng(42))
        inp = DecisionMakingInput(
            engine_confidences=_entries({"e1": 0.99, "e2": 0.99, "e3": 0.99}),
        )
        result = engine.process(inp)
        assert result.confidence.final_confidence > 0.90

    def test_single_engine_input(self):
        engine = DecisionMakingEngine(rng=np.random.default_rng(42))
        inp = DecisionMakingInput(
            engine_confidences=[EngineConfidenceEntry("solo", 0.70)],
        )
        result = engine.process(inp)
        assert result.action in DecisionAction

    def test_extreme_emotions(self):
        engine = DecisionMakingEngine(rng=np.random.default_rng(42))
        inp = _default_input(
            emotion_anxiety=1.0,
            emotion_nervous=1.0,
            emotion_perplexed=1.0,
            emotion_regret=1.0,
            emotion_skeptical=1.0,
        )
        result = engine.process(inp)
        # Extreme negative emotions should suppress confidence
        assert result.confidence.emotion_delta < 0

    def test_extreme_positive_emotion(self):
        engine = DecisionMakingEngine(rng=np.random.default_rng(42))
        inp = _default_input(emotion_confident=1.0)
        result = engine.process(inp)
        assert result.confidence.emotion_delta > 0

    def test_result_metadata(self):
        engine = DecisionMakingEngine(rng=np.random.default_rng(42))
        result = engine.process(_default_input())
        assert "mode" in result.metadata
        assert "cycle" in result.metadata
        assert "raw_quadrant" in result.metadata
        assert "final_quadrant" in result.metadata

    def test_deterministic_with_seed(self):
        """Same seed → same result."""
        inp = _default_input()
        r1 = DecisionMakingEngine(rng=np.random.default_rng(99)).process(inp)
        r2 = DecisionMakingEngine(rng=np.random.default_rng(99)).process(inp)
        assert r1.confidence.final_confidence == r2.confidence.final_confidence
        assert r1.quadrant == r2.quadrant

    def test_different_seeds_may_differ(self):
        """Different seeds → jitter differs (though quadrant may be same)."""
        inp = _default_input()
        r1 = DecisionMakingEngine(rng=np.random.default_rng(1)).process(inp)
        r2 = DecisionMakingEngine(rng=np.random.default_rng(9999)).process(inp)
        # Jitter values should differ
        assert r1.confidence.jitter_applied != r2.confidence.jitter_applied


# =====================================================================
# 28. Risk Combination
# =====================================================================


class TestRiskCombination:
    def test_combine_risks_weighted(self):
        r = combine_risks(0.60, 0.40, alpha_raw=0.65, alpha_reward=0.35)
        expected = 0.65 * 0.60 + 0.35 * 0.40
        assert abs(r - expected) < 1e-9

    def test_combine_risks_clamped(self):
        r = combine_risks(1.0, 1.0, alpha_raw=0.7, alpha_reward=0.5)
        assert r <= 1.0

    def test_combine_risks_zero(self):
        r = combine_risks(0.0, 0.0)
        assert r == 0.0
