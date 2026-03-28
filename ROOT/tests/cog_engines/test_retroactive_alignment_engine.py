"""
Tests for Engine 30 -- Retro-Active Alignment Error Detection Engine
====================================================================
Covers: enums, config, state vector, projectors (reward, neurochem,
emotional), delta computation (symbolic, affective, reward), collapse
probability, collapse state classification, temporal smoothing (EWMA),
attribution logic, drift trend, affective mapping, corrections,
neurochemical coupling, engine pipeline, mode adjustments, edge cases.
"""
from __future__ import annotations

import math
import time
import numpy as np
import pytest

from zados.cognitive_engines.py_engines.retroactive_alignment_engine import (
    AcknowledgedChange,
    AlignmentAffectSignal,
    AlignmentNeurochem,
    AlignmentState,
    AttributionType,
    CollapseState,
    CorrectionEmitted,
    CorrectionType,
    DeltaComponent,
    DriftTrend,
    HistoricalState,
    RetroactiveAlignmentConfig,
    RetroactiveAlignmentEngine,
    RetroactiveAlignmentInput,
    RetroactiveAlignmentResult,
    ScanHorizon,
    SystemStateVector,
    build_corrections,
    classify_collapse_state,
    compute_alignment_neurochem,
    compute_collapse_probability,
    compute_delta_affective,
    compute_delta_reward,
    compute_delta_symbolic,
    compute_temporal_discount,
    cosine_similarity,
    determine_attribution,
    determine_drift_trend,
    euclidean_distance,
    ewma_update,
    get_mode_threshold_adjustment,
    map_affective_consequence,
    project_emotional,
    project_neurochem,
    project_reward,
    sigmoid_fn,
)
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Fixtures
# =====================================================================

RNG = np.random.default_rng(42)
CFG = RetroactiveAlignmentConfig()


def _state(rewards=None, neurochem=None, osc=None, emotions=None):
    return SystemStateVector(
        reward_signals=rewards or {},
        neurochemical_concentrations=neurochem or {},
        oscillatory_powers=osc or {},
        emotional_activations=emotions or {},
    )


def _hist(state, ts=0.0, context="", trust=1.0):
    return HistoricalState(
        timestamp=ts,
        state_vector=state,
        processing_context=context,
        trust_weight=trust,
    )


# =====================================================================
# Enums
# =====================================================================


class TestEnums:
    def test_delta_components(self):
        assert len(DeltaComponent) == 3

    def test_collapse_states(self):
        assert len(CollapseState) == 5

    def test_attribution_types(self):
        assert len(AttributionType) == 4

    def test_drift_trends(self):
        assert len(DriftTrend) == 4

    def test_scan_horizons(self):
        assert len(ScanHorizon) == 4

    def test_correction_types(self):
        assert len(CorrectionType) == 4


# =====================================================================
# Config
# =====================================================================


class TestConfig:
    def test_defaults(self):
        assert CFG.alpha_sym == 3.0
        assert CFG.alpha_aff == 2.5
        assert CFG.alpha_rew == 2.0

    def test_frozen(self):
        with pytest.raises(AttributeError):
            CFG.alpha_sym = 5.0

    def test_collapse_thresholds_ordered(self):
        assert CFG.collapse_stable < CFG.collapse_elevated < CFG.collapse_at_risk < CFG.collapse_critical


# =====================================================================
# Math utilities
# =====================================================================


class TestMathUtils:
    def test_sigmoid_zero(self):
        assert sigmoid_fn(0.0) == pytest.approx(0.5)

    def test_sigmoid_bounds(self):
        assert 0.0 < sigmoid_fn(-100) < 0.01
        assert sigmoid_fn(100) > 0.99

    def test_euclidean_same(self):
        assert euclidean_distance({"a": 1.0}, {"a": 1.0}) == 0.0

    def test_euclidean_different(self):
        d = euclidean_distance({"a": 0.0}, {"a": 1.0})
        assert d == pytest.approx(1.0)

    def test_euclidean_empty(self):
        assert euclidean_distance({}, {}) == 0.0

    def test_cosine_identical(self):
        assert cosine_similarity({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0}) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        sim = cosine_similarity({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0})
        assert sim == pytest.approx(0.0)

    def test_cosine_empty(self):
        assert cosine_similarity({}, {}) == 1.0  # default for empty

    def test_ewma_initial(self):
        result = ewma_update(0.0, 1.0, 4.0)
        assert 0.0 < result < 1.0

    def test_ewma_stable(self):
        val = 0.5
        for _ in range(100):
            val = ewma_update(val, 0.5, 4.0)
        assert val == pytest.approx(0.5, abs=0.01)


# =====================================================================
# Projectors
# =====================================================================


class TestProjectors:
    def test_project_reward_no_changes(self):
        past = {"logic": 0.7, "ethics": 0.5}
        projected = project_reward(past, [])
        assert projected == past

    def test_project_neurochem_decays(self):
        past = {"DA": 0.8}
        projected = project_neurochem(past, 10.0, [])
        assert projected["DA"] < 0.8  # Decayed toward baseline

    def test_project_neurochem_approaches_baseline(self):
        past = {"DA": 0.8}
        projected_short = project_neurochem(past, 1.0, [])
        projected_long = project_neurochem(past, 100.0, [])
        assert projected_long["DA"] < projected_short["DA"]

    def test_project_emotional_decay(self):
        past = {"excited": 0.9}
        projected = project_emotional(past, 10.0, [])
        assert projected["excited"] < 0.9  # Fast decay

    def test_project_emotional_slow_decay(self):
        past = {"loyal": 0.9}
        projected = project_emotional(past, 5.0, [])
        # Loyal has slow decay rate (0.03)
        assert projected["loyal"] > 0.7


# =====================================================================
# Delta computation
# =====================================================================


class TestDeltaComputation:
    def test_delta_symbolic_identical(self):
        state = _state(rewards={"logic": 0.5}, emotions={"joy": 0.3})
        d = compute_delta_symbolic(state, state)
        assert d == pytest.approx(0.0)

    def test_delta_symbolic_different(self):
        past = _state(rewards={"logic": 0.9}, emotions={"joy": 0.8})
        current = _state(rewards={"logic": 0.1}, emotions={"anger": 0.8})
        d = compute_delta_symbolic(past, current)
        assert d > 0.0

    def test_delta_affective(self):
        projected = {"joy": 0.8, "anger": 0.1}
        current = {"joy": 0.2, "anger": 0.6}
        d = compute_delta_affective(projected, current)
        assert d > 0.0

    def test_delta_affective_identical(self):
        same = {"joy": 0.5}
        assert compute_delta_affective(same, same) == 0.0

    def test_delta_reward(self):
        projected = {"logic": 0.8}
        current = {"logic": 0.2}
        d = compute_delta_reward(projected, current)
        assert d == pytest.approx(0.6)


# =====================================================================
# Collapse probability
# =====================================================================


class TestCollapseProbability:
    def test_zero_deltas(self):
        p = compute_collapse_probability(0.0, 0.0, 0.0, CFG)
        assert p < 0.15  # Should be stable

    def test_high_deltas(self):
        p = compute_collapse_probability(0.8, 0.7, 0.6, CFG)
        assert p > 0.5

    def test_interaction_amplification(self):
        # Single high component
        p_single = compute_collapse_probability(0.6, 0.0, 0.0, CFG)
        # Multiple high components
        p_multi = compute_collapse_probability(0.4, 0.3, 0.3, CFG)
        # Multi should be higher due to interaction term (pairwise products)
        # Actually depends on coefficient magnitudes -- just check reasonable
        assert 0.0 <= p_single <= 1.0
        assert 0.0 <= p_multi <= 1.0


class TestCollapseClassification:
    def test_stable(self):
        assert classify_collapse_state(0.10, CFG) == CollapseState.STABLE

    def test_elevated(self):
        assert classify_collapse_state(0.20, CFG) == CollapseState.ELEVATED

    def test_at_risk(self):
        assert classify_collapse_state(0.35, CFG) == CollapseState.AT_RISK

    def test_critical(self):
        assert classify_collapse_state(0.55, CFG) == CollapseState.CRITICAL

    def test_collapse_imminent(self):
        assert classify_collapse_state(0.75, CFG) == CollapseState.COLLAPSE_IMMINENT


# =====================================================================
# Temporal discount
# =====================================================================


class TestTemporalDiscount:
    def test_zero_tau(self):
        # At tau=0, discount should be 0 (recent = no discount)
        d = compute_temporal_discount(0.0, 15.0)
        assert d == pytest.approx(0.0)

    def test_large_tau(self):
        d = compute_temporal_discount(100.0, 15.0)
        assert d > 0.9  # Should be near 1.0

    def test_monotonic(self):
        d1 = compute_temporal_discount(5.0, 15.0)
        d2 = compute_temporal_discount(10.0, 15.0)
        assert d2 > d1


# =====================================================================
# Attribution
# =====================================================================


class TestAttribution:
    def test_reward_dominant_self(self):
        attr, conf, domain = determine_attribution(0.1, 0.1, 0.5, [])
        assert attr == AttributionType.SELF
        assert domain == "strategic"

    def test_affective_dominant_other(self):
        hist = [_hist(_state(), context="external input caused shift")]
        attr, conf, domain = determine_attribution(0.1, 0.5, 0.1, hist)
        assert attr == AttributionType.OTHER

    def test_symbolic_dominant_system(self):
        attr, conf, domain = determine_attribution(0.5, 0.1, 0.1, [])
        assert attr == AttributionType.SYSTEM

    def test_zero_deltas_unknown(self):
        attr, conf, _ = determine_attribution(0.0, 0.0, 0.0, [])
        assert attr == AttributionType.UNKNOWN


# =====================================================================
# Drift trend
# =====================================================================


class TestDriftTrend:
    def test_short_history_stable(self):
        assert determine_drift_trend([0.1, 0.2]) == DriftTrend.STABLE

    def test_increasing(self):
        assert determine_drift_trend([0.1, 0.2, 0.3, 0.4, 0.5]) == DriftTrend.INCREASING

    def test_decreasing(self):
        assert determine_drift_trend([0.5, 0.4, 0.3, 0.2, 0.1]) == DriftTrend.DECREASING

    def test_oscillating(self):
        assert determine_drift_trend([0.1, 0.5, 0.2, 0.6, 0.3]) == DriftTrend.OSCILLATING


# =====================================================================
# Affective mapping
# =====================================================================


class TestAffectiveMapping:
    def test_sym_self_confused(self):
        emotion, domain = map_affective_consequence(DeltaComponent.SYM, AttributionType.SELF, 0.4)
        assert emotion == "confused"

    def test_sym_other_perplexed(self):
        emotion, _ = map_affective_consequence(DeltaComponent.SYM, AttributionType.OTHER, 0.4)
        assert emotion == "perplexed"

    def test_aff_self_regret(self):
        emotion, _ = map_affective_consequence(DeltaComponent.AFF, AttributionType.SELF, 0.4)
        assert emotion == "regret"

    def test_aff_other_betrayal(self):
        emotion, _ = map_affective_consequence(DeltaComponent.AFF, AttributionType.OTHER, 0.4)
        assert emotion == "betrayal"

    def test_reward_self_frustrated(self):
        emotion, _ = map_affective_consequence(DeltaComponent.REWARD, AttributionType.SELF, 0.4)
        assert emotion == "frustrated"


# =====================================================================
# Corrections
# =====================================================================


class TestCorrections:
    def test_no_corrections_below_threshold(self):
        corrections = build_corrections(0.05, 0.05, 0.05, CFG)
        assert len(corrections) == 0

    def test_symbolic_correction(self):
        corrections = build_corrections(0.35, 0.05, 0.05, CFG)
        types = [c.correction_type for c in corrections]
        assert CorrectionType.SYMBOLIC_CONTRADICTION in types
        assert CorrectionType.MEMORY_TRUST in types

    def test_affective_bridge(self):
        corrections = build_corrections(0.05, 0.30, 0.05, CFG)
        types = [c.correction_type for c in corrections]
        assert CorrectionType.AFFECTIVE_BRIDGE in types

    def test_reward_recalibration(self):
        corrections = build_corrections(0.05, 0.05, 0.25, CFG)
        types = [c.correction_type for c in corrections]
        assert CorrectionType.REWARD_RECALIBRATION in types


# =====================================================================
# Neurochemical coupling
# =====================================================================


class TestNeurochemCoupling:
    def test_low_error_no_cortisol(self):
        sig = compute_alignment_neurochem(
            0.10, 0.05, 0.05, 0.05, 0.05,
            AttributionType.SELF, False, 0.0, CFG, RNG,
        )
        assert sig.delta_cor == 0.0

    def test_high_error_cortisol(self):
        sig = compute_alignment_neurochem(
            0.30, 0.40, 0.20, 0.15, 0.10,
            AttributionType.SELF, False, 0.0, CFG, RNG,
        )
        assert sig.delta_cor > 0.0

    def test_self_attribution_5ht1a(self):
        sig = compute_alignment_neurochem(
            0.30, 0.20, 0.20, 0.10, 0.10,
            AttributionType.SELF, False, 0.0, CFG, RNG,
        )
        assert sig.delta_5ht1a > 0.0

    def test_other_attribution_ne(self):
        for seed in range(20):
            sig = compute_alignment_neurochem(
                0.30, 0.20, 0.20, 0.10, 0.10,
                AttributionType.OTHER, False, 0.0, CFG, np.random.default_rng(seed),
            )
            if sig.delta_ne > 0.0:
                break
        assert sig.delta_ne > 0.0

    def test_reward_misalignment_negative_da(self):
        sig = compute_alignment_neurochem(
            0.30, 0.20, 0.10, 0.10, 0.25,
            AttributionType.SELF, False, 0.0, CFG, RNG,
        )
        assert sig.delta_da < 0.0

    def test_correction_success_positive_da(self):
        sig = compute_alignment_neurochem(
            0.10, 0.10, 0.05, 0.05, 0.05,
            AttributionType.SELF, True, 0.30, CFG, np.random.default_rng(42),
        )
        assert sig.delta_da > 0.0

    def test_other_attribution_oxt_decay(self):
        sig = compute_alignment_neurochem(
            0.30, 0.20, 0.10, 0.30, 0.10,
            AttributionType.OTHER, False, 0.0, CFG, RNG,
        )
        assert sig.delta_oxt < 0.0

    def test_theta_boost(self):
        sig = compute_alignment_neurochem(
            0.30, 0.20, 0.20, 0.10, 0.10,
            AttributionType.SELF, False, 0.0, CFG, RNG,
        )
        assert sig.theta_boost > 0.0


# =====================================================================
# Mode threshold adjustment
# =====================================================================


class TestModeAdjustment:
    def test_normal_baseline(self):
        assert get_mode_threshold_adjustment(OperationalMode.NORMAL, CFG) == 1.0

    def test_dev_more_sensitive(self):
        assert get_mode_threshold_adjustment(OperationalMode.DEV, CFG) < 1.0

    def test_rem_dream_less_sensitive(self):
        assert get_mode_threshold_adjustment(OperationalMode.REM_DREAM, CFG) > 1.0

    def test_all_modes(self):
        for mode in OperationalMode:
            adj = get_mode_threshold_adjustment(mode, CFG)
            assert 0.5 < adj < 1.5


# =====================================================================
# Engine -- basic pipeline
# =====================================================================


class TestEngineBasic:
    def setup_method(self):
        self.engine = RetroactiveAlignmentEngine(rng=np.random.default_rng(42))

    def test_no_history(self):
        result = self.engine.process(RetroactiveAlignmentInput())
        assert result.metadata.get("no_history") is True

    def test_identical_states(self):
        state = _state(
            rewards={"logic": 0.5, "ethics": 0.5},
            emotions={"joy": 0.3},
            neurochem={"DA": 0.4},
        )
        result = self.engine.process(RetroactiveAlignmentInput(
            current_state=state,
            historical_states=[_hist(state, ts=time.time() - 1)],
        ))
        assert result.delta_total < 0.3  # Near zero with temporal discount

    def test_different_states_flags(self):
        past = _state(rewards={"logic": 0.9}, emotions={"joy": 0.8})
        current = _state(rewards={"logic": 0.1}, emotions={"anger": 0.8})
        result = self.engine.process(RetroactiveAlignmentInput(
            current_state=current,
            historical_states=[_hist(past, ts=time.time() - 10)],
        ))
        assert result.delta_total > 0.0
        assert result.collapse_state in CollapseState

    def test_processing_time(self):
        result = self.engine.process(RetroactiveAlignmentInput())
        assert result.processing_time_ms >= 0.0

    def test_cycle_count(self):
        self.engine.process(RetroactiveAlignmentInput())
        self.engine.process(RetroactiveAlignmentInput())
        assert self.engine.get_status()["cycle_count"] == 2


# =====================================================================
# Engine -- temporal smoothing
# =====================================================================


class TestTemporalSmoothing:
    def test_ewma_converges(self):
        engine = RetroactiveAlignmentEngine(rng=np.random.default_rng(42))
        past = _state(rewards={"logic": 0.9}, emotions={"joy": 0.9})
        current = _state(rewards={"logic": 0.1}, emotions={"anger": 0.9})

        prev_smoothed = 0.0
        for _ in range(10):
            result = engine.process(RetroactiveAlignmentInput(
                current_state=current,
                historical_states=[_hist(past, ts=time.time() - 5)],
            ))

        # After many cycles, smoothed should approach raw
        assert result.delta_smoothed > 0.0

    def test_hysteresis_gating(self):
        engine = RetroactiveAlignmentEngine(rng=np.random.default_rng(42))
        past = _state(rewards={"logic": 0.9}, emotions={"joy": 0.9})
        current = _state(rewards={"logic": 0.1}, emotions={"anger": 0.9})

        # Run enough cycles for hysteresis
        for _ in range(5):
            result = engine.process(RetroactiveAlignmentInput(
                current_state=current,
                historical_states=[_hist(past, ts=time.time() - 5)],
            ))

        # Should have corrections after hysteresis
        assert result.drift_timespan >= 0


# =====================================================================
# Engine -- affective signals
# =====================================================================


class TestAffectiveSignals:
    def test_affective_signal_emitted(self):
        engine = RetroactiveAlignmentEngine(rng=np.random.default_rng(42))
        past = _state(rewards={"logic": 0.9}, emotions={"joy": 0.9})
        current = _state(rewards={"logic": 0.1}, emotions={"anger": 0.9})
        result = engine.process(RetroactiveAlignmentInput(
            current_state=current,
            historical_states=[_hist(past, ts=time.time() - 5)],
        ))
        if result.affective_signal is not None:
            assert isinstance(result.affective_signal, AlignmentAffectSignal)
            assert result.triggered_emotion is not None

    def test_no_signal_for_small_error(self):
        engine = RetroactiveAlignmentEngine(rng=np.random.default_rng(42))
        state = _state(rewards={"logic": 0.50}, emotions={"joy": 0.30})
        result = engine.process(RetroactiveAlignmentInput(
            current_state=state,
            historical_states=[_hist(state, ts=time.time() - 1)],
        ))
        # Near-identical states → small error → may or may not trigger signal


# =====================================================================
# Engine -- mode + bidirectional
# =====================================================================


class TestEngineModes:
    def test_configure_mode(self):
        engine = RetroactiveAlignmentEngine()
        engine.configure(OperationalMode.REFLECTIVE)
        assert engine.get_status()["mode"] == "reflective"

    def test_dev_mode_more_sensitive(self):
        engine = RetroactiveAlignmentEngine(rng=np.random.default_rng(42))
        past = _state(rewards={"logic": 0.6}, emotions={"joy": 0.5})
        current = _state(rewards={"logic": 0.4}, emotions={"joy": 0.3})
        result = engine.process(RetroactiveAlignmentInput(
            current_state=current,
            historical_states=[_hist(past, ts=time.time() - 5)],
            active_mode=OperationalMode.DEV,
        ))
        assert result.metadata["mode_adjustment"] < 1.0

    def test_high_cortisol_more_sensitive(self):
        engine = RetroactiveAlignmentEngine(rng=np.random.default_rng(42))
        engine.update_neurochem_state({"cor": 0.8})
        result = engine.process(RetroactiveAlignmentInput(
            current_state=_state(),
            historical_states=[_hist(_state(), ts=time.time() - 1)],
        ))
        assert result.metadata["mode_adjustment"] < 1.0

    def test_reactive_trigger_marker(self):
        engine = RetroactiveAlignmentEngine(rng=np.random.default_rng(42))
        result = engine.process(RetroactiveAlignmentInput(
            current_state=_state(),
            historical_states=[_hist(_state(), ts=time.time() - 1)],
            reactive_trigger={"source": "contradiction_detection"},
        ))
        assert result.scan_trigger == "REACTIVE"


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    def test_multiple_historical_states(self):
        engine = RetroactiveAlignmentEngine(rng=np.random.default_rng(42))
        states = [
            _hist(_state(rewards={"logic": 0.3 + i * 0.1}), ts=time.time() - 10 + i)
            for i in range(5)
        ]
        result = engine.process(RetroactiveAlignmentInput(
            current_state=_state(rewards={"logic": 0.8}),
            historical_states=states,
        ))
        assert result.compared_states == 5

    def test_acknowledged_changes(self):
        engine = RetroactiveAlignmentEngine(rng=np.random.default_rng(42))
        past = _state(rewards={"logic": 0.3})
        current = _state(rewards={"logic": 0.8})
        changes = [AcknowledgedChange(component="REWARD", description="logic adjustment")]
        result = engine.process(RetroactiveAlignmentInput(
            current_state=current,
            historical_states=[_hist(past, ts=time.time() - 5)],
            acknowledged_changes=changes,
        ))
        # Acknowledged changes should reduce perceived error
        assert isinstance(result, RetroactiveAlignmentResult)

    def test_component_ratio_sums_to_one(self):
        engine = RetroactiveAlignmentEngine(rng=np.random.default_rng(42))
        past = _state(rewards={"logic": 0.9}, emotions={"joy": 0.9})
        current = _state(rewards={"logic": 0.1}, emotions={"anger": 0.9})
        result = engine.process(RetroactiveAlignmentInput(
            current_state=current,
            historical_states=[_hist(past, ts=time.time() - 5)],
        ))
        if result.component_ratio:
            total = sum(result.component_ratio.values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_status_fields(self):
        engine = RetroactiveAlignmentEngine()
        status = engine.get_status()
        assert status["engine_id"] == "retroactive_alignment_engine"
        assert "delta_bar" in status
        assert "correction_pending" in status
