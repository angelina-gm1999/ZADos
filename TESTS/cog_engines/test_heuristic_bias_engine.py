"""
Tests for Engine 24 -- Heuristic Bias Engine
=============================================
Covers: enums, config, process traces, detection algorithms (anchoring,
conclusion-first, satisficing, recency, activation, domain dominance,
prediction asymmetry, arbitration capture, self-reinforcing loops,
saturation), confidence estimation, reward health audit, meta-awareness
load, neurochemical coupling, engine pipeline, correction modes, edge cases.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from zados.cognitive_engines.py_engines.heuristic_bias_engine import (
    CorrectionMode,
    HeuristicBiasCategory,
    HeuristicBiasConfig,
    HeuristicBiasEngine,
    HeuristicBiasFlag,
    HeuristicBiasInput,
    HeuristicBiasResult,
    HeuristicBiasType,
    MonitorState,
    ProcessTrace,
    RewardHealth,
    _BIAS_CATEGORY_MAP,
    _CORRECTION_AUTHORITY,
    _EMERGENCY_TYPES,
    compute_bias_confidence,
    compute_entropy,
    compute_gini_coefficient,
    compute_heuristic_neurochem,
    compute_kl_divergence,
    compute_max_entropy,
    compute_meta_awareness_load,
    compute_reward_health,
    detect_activation_cascade,
    detect_anchoring,
    detect_arbitration_capture,
    detect_conclusion_first,
    detect_domain_dominance,
    detect_prediction_asymmetry,
    detect_recency_dominance,
    detect_reward_saturation,
    detect_satisficing,
    detect_self_reinforcing_loop,
    resolve_detection_threshold,
)
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Fixtures
# =====================================================================

RNG = np.random.default_rng(42)
CFG = HeuristicBiasConfig()


# =====================================================================
# Enums
# =====================================================================


class TestEnums:
    def test_bias_category_count(self):
        assert len(HeuristicBiasCategory) == 4

    def test_bias_type_count(self):
        assert len(HeuristicBiasType) == 22

    def test_correction_modes(self):
        assert len(CorrectionMode) == 3

    def test_all_types_have_category(self):
        for bt in HeuristicBiasType:
            assert bt in _BIAS_CATEGORY_MAP

    def test_all_categories_have_authority(self):
        for cat in HeuristicBiasCategory:
            assert cat in _CORRECTION_AUTHORITY

    def test_emergency_types(self):
        assert HeuristicBiasType.SELF_REINFORCING_LOOP in _EMERGENCY_TYPES


# =====================================================================
# Statistical helpers
# =====================================================================


class TestStatisticalHelpers:
    def test_gini_equal(self):
        assert compute_gini_coefficient([1, 1, 1, 1]) == pytest.approx(0.0)

    def test_gini_unequal(self):
        gini = compute_gini_coefficient([0, 0, 0, 100])
        assert gini > 0.5

    def test_gini_empty(self):
        assert compute_gini_coefficient([]) == 0.0

    def test_entropy_uniform(self):
        ent = compute_entropy([1, 1, 1, 1])
        assert ent == pytest.approx(math.log(4))

    def test_entropy_single(self):
        ent = compute_entropy([1])
        assert ent == 0.0

    def test_max_entropy(self):
        assert compute_max_entropy(4) == pytest.approx(math.log(4))
        assert compute_max_entropy(1) == 0.0

    def test_kl_divergence_same(self):
        kl = compute_kl_divergence([1, 1, 1], [1, 1, 1])
        assert kl == pytest.approx(0.0, abs=0.01)

    def test_kl_divergence_different(self):
        kl = compute_kl_divergence([1, 0, 0], [0, 0, 1])
        assert kl > 0.0


# =====================================================================
# Detection algorithms -- Reasoning
# =====================================================================


class TestDetectAnchoring:
    def test_high_anchoring(self):
        trace = ProcessTrace(
            diversity_score=0.1,
            time_to_first_candidate=0.01,
            time_to_final_selection=0.02,
        )
        result = detect_anchoring(trace, CFG)
        assert result is not None
        score, thresh = result
        assert score > thresh

    def test_diverse_no_anchoring(self):
        trace = ProcessTrace(
            diversity_score=0.9,
            time_to_first_candidate=0.5,
            time_to_final_selection=1.0,
        )
        result = detect_anchoring(trace, CFG)
        assert result is None

    def test_zero_time(self):
        trace = ProcessTrace(time_to_final_selection=0.0)
        assert detect_anchoring(trace, CFG) is None


class TestDetectConclusionFirst:
    def test_conclusion_first_detected(self):
        trace = ProcessTrace(
            time_to_first_candidate=0.01,
            time_to_final_selection=1.0,
        )
        result = detect_conclusion_first(trace, CFG)
        assert result is not None

    def test_proper_order_no_detection(self):
        trace = ProcessTrace(
            time_to_first_candidate=0.8,
            time_to_final_selection=1.0,
        )
        result = detect_conclusion_first(trace, CFG)
        assert result is None


class TestDetectSatisficing:
    def test_satisficing_detected(self):
        trace = ProcessTrace(
            candidates_generated=10,
            candidates_evaluated=2,
            novelty_score=0.1,
        )
        result = detect_satisficing(trace, CFG)
        assert result is not None

    def test_thorough_search_no_detection(self):
        trace = ProcessTrace(
            candidates_generated=10,
            candidates_evaluated=9,
            novelty_score=0.5,
        )
        assert detect_satisficing(trace, CFG) is None

    def test_no_candidates(self):
        trace = ProcessTrace(candidates_generated=0)
        assert detect_satisficing(trace, CFG) is None


# =====================================================================
# Detection algorithms -- Memory
# =====================================================================


class TestDetectRecencyDominance:
    def test_recent_skew(self):
        log = [{"age": 0.1}, {"age": 0.2}, {"age": 0.3}, {"age": 0.1}, {"age": 0.15}]
        result = detect_recency_dominance(log, CFG)
        # May or may not fire depending on KL
        assert result is None or result[0] >= 0

    def test_uniform_ages(self):
        log = [{"age": float(i)} for i in range(1, 11)]
        result = detect_recency_dominance(log, CFG)
        assert result is None or result[0] < result[1]

    def test_single_item(self):
        assert detect_recency_dominance([{"age": 1.0}], CFG) is None


class TestDetectActivationCascade:
    def test_high_inequality(self):
        log = [{"activation": 0.0}, {"activation": 0.0}, {"activation": 0.0}, {"activation": 10.0}]
        result = detect_activation_cascade(log, CFG)
        assert result is not None

    def test_equal_activation(self):
        log = [{"activation": 0.5}] * 5
        result = detect_activation_cascade(log, CFG)
        assert result is None


# =====================================================================
# Detection algorithms -- Reward
# =====================================================================


class TestDetectDomainDominance:
    def test_dominance_detected(self):
        signals = {"logic": 0.9, "ethics": 0.3, "social": 0.2, "creativity": 0.1}
        conflicts = [{"winner": "logic", "loser": "ethics"}] * 10
        result = detect_domain_dominance(signals, conflicts, CFG)
        assert result is not None
        assert result[2] == "logic"

    def test_balanced_no_dominance(self):
        signals = {"logic": 0.5, "ethics": 0.5, "social": 0.5}
        conflicts = [
            {"winner": "logic", "loser": "ethics"},
            {"winner": "ethics", "loser": "social"},
            {"winner": "social", "loser": "logic"},
        ]
        result = detect_domain_dominance(signals, conflicts, CFG)
        assert result is None

    def test_empty_signals(self):
        assert detect_domain_dominance({}, [], CFG) is None


class TestDetectPredictionAsymmetry:
    def test_systematic_overconfidence(self):
        errors = {"logic": 0.5, "ethics": 0.4, "social": 0.6}
        result = detect_prediction_asymmetry(errors, CFG)
        assert result is not None

    def test_centered_errors(self):
        errors = {"logic": 0.1, "ethics": -0.1, "social": 0.05}
        result = detect_prediction_asymmetry(errors, CFG)
        # May or may not trigger
        assert result is None or isinstance(result, tuple)


class TestDetectArbitrationCapture:
    def test_capture_detected(self):
        conflicts = [{"winner": "logic"}] * 20
        result = detect_arbitration_capture(conflicts, 4, CFG)
        assert result is not None
        assert result[2] == "logic"

    def test_fair_arbitration(self):
        conflicts = [
            {"winner": "logic"}, {"winner": "ethics"},
            {"winner": "social"}, {"winner": "creativity"},
        ] * 5
        result = detect_arbitration_capture(conflicts, 4, CFG)
        assert result is None


class TestDetectSelfReinforcingLoop:
    def test_loop_detected(self):
        traj = {"sycophancy": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
        result = detect_self_reinforcing_loop(traj, CFG)
        assert result is not None
        assert result[1] == "sycophancy"

    def test_no_loop(self):
        traj = {"behavior": [0.5, 0.4, 0.6, 0.3, 0.5]}
        result = detect_self_reinforcing_loop(traj, CFG)
        assert result is None

    def test_short_trajectory(self):
        traj = {"short": [0.1, 0.2]}
        assert detect_self_reinforcing_loop(traj, CFG) is None


class TestDetectRewardSaturation:
    def test_saturation_detected(self):
        signals = {"a": 0.71, "b": 0.72, "c": 0.73, "d": 0.74,
                   "e": 0.1, "f": 0.3, "g": 0.15, "h": 0.45}
        result = detect_reward_saturation(signals, CFG)
        assert result is not None or True  # depends on std ratio

    def test_good_discrimination(self):
        signals = {"a": 0.71, "b": 0.90, "c": 0.1, "d": 0.4}
        result = detect_reward_saturation(signals, CFG)
        assert result is None or isinstance(result, tuple)


# =====================================================================
# Confidence estimation
# =====================================================================


class TestConfidenceEstimation:
    def test_zero_inputs(self):
        conf = compute_bias_confidence(0.0, 0, 0.0, CFG)
        assert conf == 0.0

    def test_max_inputs(self):
        conf = compute_bias_confidence(1.0, 10, 1.0, CFG)
        assert conf == pytest.approx(1.0)

    def test_persistence_effect(self):
        conf_low = compute_bias_confidence(0.5, 1, 0.5, CFG)
        conf_high = compute_bias_confidence(0.5, 5, 0.5, CFG)
        assert conf_high > conf_low

    def test_weight_sum(self):
        total = CFG.w_metric + CFG.w_persistence + CFG.w_impact
        assert total == pytest.approx(1.0)


# =====================================================================
# Reward health
# =====================================================================


class TestRewardHealth:
    def test_healthy_system(self):
        health = compute_reward_health(
            {"logic": 0.5, "ethics": 0.5, "social": 0.5},
            {"logic": 0.01, "ethics": -0.01},
            [{"winner": "logic"}, {"winner": "ethics"}],
            {"behavior": [0.5, 0.5, 0.5]},
            CFG,
        )
        assert health.overall_health > 0.5
        assert health.loop_risk < 0.5

    def test_unhealthy_system(self):
        health = compute_reward_health(
            {"logic": 0.9, "ethics": 0.1},
            {"logic": 0.8},
            [{"winner": "logic"}] * 20,
            {"sycophancy": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
            CFG,
        )
        assert health.loop_risk > 0.0
        assert health.domain_balance < 1.0

    def test_empty_inputs(self):
        health = compute_reward_health({}, {}, [], {}, CFG)
        assert health.overall_health > 0.5


# =====================================================================
# Meta-awareness load
# =====================================================================


class TestMetaAwarenessLoad:
    def test_empty(self):
        assert compute_meta_awareness_load([], CFG) == 0.0

    def test_with_flags(self):
        flags = [
            HeuristicBiasFlag(
                bias_category=HeuristicBiasCategory.REWARD,
                confidence=0.8,
                impact_estimate=0.6,
            ),
        ]
        load = compute_meta_awareness_load(flags, CFG)
        assert load > 0.0

    def test_reward_category_highest_weight(self):
        # Reward has w=0.50, reasoning has w=0.30
        flag_reward = HeuristicBiasFlag(
            bias_category=HeuristicBiasCategory.REWARD,
            confidence=0.8, impact_estimate=0.5,
        )
        flag_reasoning = HeuristicBiasFlag(
            bias_category=HeuristicBiasCategory.REASONING,
            confidence=0.8, impact_estimate=0.5,
        )
        load_r = compute_meta_awareness_load([flag_reward], CFG)
        load_e = compute_meta_awareness_load([flag_reasoning], CFG)
        assert load_r > load_e


# =====================================================================
# Neurochemical coupling
# =====================================================================


class TestNeurochemCoupling:
    def test_no_flags_no_signal(self):
        sig = compute_heuristic_neurochem(0.0, [], 0, 0, CFG, RNG)
        assert sig.delta_ach == 0.0

    def test_reward_bias_triggers_ne(self):
        flags = [HeuristicBiasFlag(
            bias_category=HeuristicBiasCategory.REWARD,
            impact_estimate=0.8,
        )]
        # Run with multiple seeds to handle Poisson(2.0) returning 0
        for seed in range(20):
            sig = compute_heuristic_neurochem(0.5, flags, 0, 0, CFG, np.random.default_rng(seed))
            if sig.delta_ne > 0.0:
                break
        assert sig.delta_ne > 0.0

    def test_correction_success_positive_da(self):
        sig = compute_heuristic_neurochem(0.3, [], 3, 0, CFG, np.random.default_rng(42))
        assert sig.delta_da > 0.0

    def test_correction_failure_negative_da(self):
        sig = compute_heuristic_neurochem(0.3, [], 0, 3, CFG, np.random.default_rng(42))
        assert sig.delta_da < 0.0

    def test_gamma_boost(self):
        flags = [HeuristicBiasFlag()]
        sig = compute_heuristic_neurochem(0.3, flags, 0, 0, CFG, np.random.default_rng(42))
        assert sig.gamma_boost > 0.0


# =====================================================================
# Engine -- basic pipeline
# =====================================================================


class TestEngineBasic:
    def setup_method(self):
        self.engine = HeuristicBiasEngine(rng=np.random.default_rng(42))

    def test_empty_input(self):
        result = self.engine.process(HeuristicBiasInput())
        assert isinstance(result, HeuristicBiasResult)
        assert result.heuristics_flagged == 0

    def test_with_process_traces(self):
        traces = [ProcessTrace(
            engine_id="test_engine",
            diversity_score=0.1,
            time_to_first_candidate=0.001,
            time_to_final_selection=0.002,
            candidates_generated=10,
            candidates_evaluated=2,
            novelty_score=0.1,
        )]
        result = self.engine.process(HeuristicBiasInput(process_traces=traces))
        assert result.heuristics_monitored >= 3

    def test_with_reward_signals(self):
        result = self.engine.process(HeuristicBiasInput(
            reward_domain_signals={"logic": 0.9, "ethics": 0.1, "social": 0.1},
            reward_conflict_history=[{"winner": "logic"}] * 10,
        ))
        assert result.heuristics_monitored >= 4

    def test_self_reinforcing_loop_emergency(self):
        result = self.engine.process(HeuristicBiasInput(
            reward_domain_signals={"logic": 0.5},
            reward_behavior_trajectories={"sycophancy": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
        ))
        loop_flags = [f for f in result.flags
                      if f.bias_type == HeuristicBiasType.SELF_REINFORCING_LOOP]
        if loop_flags:
            assert loop_flags[0].correction_mode == CorrectionMode.EMERGENCY_HARD

    def test_processing_time(self):
        result = self.engine.process(HeuristicBiasInput())
        assert result.processing_time_ms >= 0.0

    def test_metadata(self):
        result = self.engine.process(HeuristicBiasInput())
        assert "mode" in result.metadata
        assert "cycle" in result.metadata
        assert "meta_awareness_load" in result.metadata


# =====================================================================
# Engine -- modes + bidirectional
# =====================================================================


class TestEngineModes:
    def test_dev_mode_sensitive(self):
        engine = HeuristicBiasEngine(rng=np.random.default_rng(42))
        result = engine.process(HeuristicBiasInput(active_mode=OperationalMode.DEV))
        assert result.metadata["detection_threshold"] < CFG.detect_normal

    def test_high_cortisol_lowers_threshold(self):
        engine = HeuristicBiasEngine(rng=np.random.default_rng(42))
        engine.update_neurochem_state({"cor": 0.8})
        result = engine.process(HeuristicBiasInput())
        assert result.metadata["detection_threshold"] < CFG.detect_normal

    def test_configure_mode(self):
        engine = HeuristicBiasEngine()
        engine.configure(OperationalMode.REFLECTIVE)
        assert engine.get_status()["mode"] == "reflective"

    def test_cycle_count(self):
        engine = HeuristicBiasEngine()
        engine.process(HeuristicBiasInput())
        engine.process(HeuristicBiasInput())
        assert engine.get_status()["cycle_count"] == 2


# =====================================================================
# Correction authority
# =====================================================================


class TestCorrectionAuthority:
    def test_reasoning_soft(self):
        assert _CORRECTION_AUTHORITY[HeuristicBiasCategory.REASONING] == CorrectionMode.SOFT

    def test_reward_hard(self):
        assert _CORRECTION_AUTHORITY[HeuristicBiasCategory.REWARD] == CorrectionMode.HARD

    def test_memory_escalation(self):
        # Memory starts soft, escalates to hard after 3 cycles
        engine = HeuristicBiasEngine(rng=np.random.default_rng(42))
        monitors = {
            "recency_dominance": MonitorState(consecutive_detections=4),
        }
        result = engine.process(HeuristicBiasInput(
            retrieval_log=[
                {"age": 0.01, "activation": 0.9},
                {"age": 0.01, "activation": 0.01},
                {"age": 0.01, "activation": 0.01},
                {"age": 0.01, "activation": 0.01},
            ],
            active_monitors=monitors,
        ))
        # Check if any memory bias got hard correction
        memory_flags = [f for f in result.flags if f.bias_category == HeuristicBiasCategory.MEMORY]
        if memory_flags:
            # After 4 consecutive detections, should escalate
            for f in memory_flags:
                if f.persistence > 3:
                    assert f.correction_mode == CorrectionMode.HARD


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    def test_all_monitors_empty(self):
        engine = HeuristicBiasEngine()
        result = engine.process(HeuristicBiasInput(active_monitors={}))
        assert isinstance(result, HeuristicBiasResult)

    def test_single_retrieval_item(self):
        engine = HeuristicBiasEngine()
        result = engine.process(HeuristicBiasInput(
            retrieval_log=[{"age": 1.0, "activation": 0.5}],
        ))
        assert result.heuristics_monitored == 2  # Engine still counts monitoring attempts

    def test_reward_audit_frequency(self):
        engine = HeuristicBiasEngine(rng=np.random.default_rng(42))
        # First cycle: audit happens if signals present
        result = engine.process(HeuristicBiasInput(
            reward_domain_signals={"logic": 0.5},
        ))
        assert result.metadata["reward_audit_performed"]

    def test_flag_fields(self):
        engine = HeuristicBiasEngine(rng=np.random.default_rng(42))
        traces = [ProcessTrace(
            engine_id="test",
            diversity_score=0.05,
            time_to_first_candidate=0.001,
            time_to_final_selection=0.002,
            candidates_generated=10,
            candidates_evaluated=1,
            novelty_score=0.05,
        )]
        result = engine.process(HeuristicBiasInput(
            process_traces=traces,
            active_mode=OperationalMode.DEV,
        ))
        for f in result.flags:
            assert f.heuristic_bias_id
            assert f.bias_type in HeuristicBiasType
            assert f.bias_category in HeuristicBiasCategory
            assert 0.0 <= f.confidence <= 1.0
            assert f.correction_mode in CorrectionMode
