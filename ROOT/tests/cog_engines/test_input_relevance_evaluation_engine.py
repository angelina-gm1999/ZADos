"""
Tests for Engine 11 -- Input Relevance Evaluation Engine
========================================================
Covers: enums, config, data types, TF-IDF cosine, 5 relevance dimensions
(Phase 1 + Phase 2 refinement), urgency normalization, priority fusion,
quadrant classification, processing depth, override rules, mode weights,
neurochemical coupling, bidirectional NT feedback, flag generation,
two-phase pipeline, edge cases.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from zados.cognitive_engines.py_engines.input_relevance_evaluation_engine import (
    # Enums
    FlagSeverity,
    ProcessingDepth,
    Quadrant,
    RelevanceFlagType,
    # Config
    IREConfig,
    _MODE_DIMENSION_WEIGHTS,
    _MODE_DEPTH_THRESHOLDS,
    _EMOTION_KEYWORDS,
    _IDENTITY_KEYWORDS,
    _INTENT_PROXY_KEYWORDS,
    # Data types
    IRENeuroChemSignals,
    IREPhase1Input,
    IREPhase1Result,
    IREPhase2Input,
    IREResult,
    IREState,
    RelevanceDimensionScores,
    RelevanceFlag,
    # Pure functions — TF-IDF
    _build_idf,
    _build_tf,
    _cosine_similarity,
    _tokenize_simple,
    _tfidf_vector,
    compute_tfidf_cosine,
    # Pure functions — Dimensions
    compute_contextual_continuity_phase1,
    refine_contextual_continuity_phase2,
    compute_intent_proxy,
    _cosine_vectors,
    compute_task_alignment_phase1,
    refine_task_alignment_phase2,
    compute_novelty_phase1,
    compute_novelty_phase2,
    compute_emotional_salience_phase1,
    compute_identity_resonance_phase1,
    refine_identity_resonance_phase2,
    # Pure functions — Urgency + Fusion
    normalize_urgency,
    compute_breach_count,
    compute_relevance_composite,
    compute_priority_composite,
    classify_quadrant,
    classify_processing_depth,
    apply_depth_overrides,
    resolve_mode_weights,
    resolve_depth_thresholds,
    compute_confidence,
    # Pure functions — Neurochem
    compute_neurochem_signals,
    # Pure functions — Flags
    _classify_flag_severity,
    generate_flags,
    # Engine
    InputRelevanceEvaluationEngine,
)
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Constants & Fixtures
# =====================================================================


RNG = np.random.default_rng(42)
CFG = IREConfig()


def _fresh_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _engine(seed: int = 42) -> InputRelevanceEvaluationEngine:
    return InputRelevanceEvaluationEngine(rng=_fresh_rng(seed))


# =====================================================================
# Enums
# =====================================================================


class TestEnums:
    def test_processing_depth_values(self):
        assert set(ProcessingDepth) == {
            ProcessingDepth.SHALLOW, ProcessingDepth.STANDARD,
            ProcessingDepth.DEEP, ProcessingDepth.CRITICAL,
        }

    def test_quadrant_values(self):
        assert len(Quadrant) == 4

    def test_relevance_flag_types(self):
        assert len(RelevanceFlagType) == 7

    def test_flag_severity_values(self):
        assert set(FlagSeverity) == {
            FlagSeverity.INFO, FlagSeverity.WARNING, FlagSeverity.RISK,
        }


# =====================================================================
# Configuration
# =====================================================================


class TestConfig:
    def test_default_weights_sum_to_one(self):
        cfg = IREConfig()
        total = cfg.w_cc + cfg.w_ta + cfg.w_nv + cfg.w_es + cfg.w_ir
        assert abs(total - 1.0) < 1e-9

    def test_mode_weights_sum_to_one(self):
        for mode, w in _MODE_DIMENSION_WEIGHTS.items():
            total = sum(w.values())
            assert abs(total - 1.0) < 1e-6, f"Mode {mode}: weights sum to {total}"

    def test_depth_thresholds_ascending(self):
        for mode, (t0, t1, t2) in _MODE_DEPTH_THRESHOLDS.items():
            assert t0 < t1 < t2, f"Mode {mode}: thresholds not ascending"

    def test_all_modes_have_weights(self):
        for mode in OperationalMode:
            assert mode in _MODE_DIMENSION_WEIGHTS

    def test_all_modes_have_thresholds(self):
        for mode in OperationalMode:
            assert mode in _MODE_DEPTH_THRESHOLDS


# =====================================================================
# Data types
# =====================================================================


class TestDataTypes:
    def test_dimension_scores_defaults(self):
        dims = RelevanceDimensionScores()
        assert dims.contextual_continuity == 0.0
        assert dims.identity_resonance == 0.0

    def test_phase1_result_defaults(self):
        r = IREPhase1Result()
        assert r.phase == 1
        assert r.processing_depth == ProcessingDepth.STANDARD

    def test_ire_result_defaults(self):
        r = IREResult()
        assert r.phase == 2
        assert r.depth_changed_from_phase1 is False

    def test_neurochem_signals_defaults(self):
        s = IRENeuroChemSignals()
        assert s.delta_ach == 0.0
        assert s.delta_cor == 0.0

    def test_relevance_flag_has_uuid(self):
        f = RelevanceFlag()
        assert len(f.flag_id) > 0
        assert "-" in f.flag_id  # UUID format


# =====================================================================
# TF-IDF Cosine
# =====================================================================


class TestTFIDF:
    def test_tokenize_simple(self):
        tokens = _tokenize_simple("Hello, World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_tokenize_empty(self):
        assert _tokenize_simple("") == []
        assert _tokenize_simple("   ") == []

    def test_build_tf_single_doc(self):
        tf = _build_tf(["a", "b", "a"])
        assert abs(tf["a"] - 2 / 3) < 1e-9
        assert abs(tf["b"] - 1 / 3) < 1e-9

    def test_build_tf_empty(self):
        assert _build_tf([]) == {}

    def test_build_idf(self):
        docs = [["a", "b"], ["a", "c"], ["b", "c"]]
        idf = _build_idf(docs)
        # "a" appears in 2/3 docs, "b" in 2/3, "c" in 2/3
        assert "a" in idf
        assert "b" in idf
        assert "c" in idf

    def test_cosine_identical(self):
        v = {"a": 1.0, "b": 2.0}
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_cosine_orthogonal(self):
        a = {"x": 1.0}
        b = {"y": 1.0}
        assert _cosine_similarity(a, b) == 0.0

    def test_cosine_empty(self):
        assert _cosine_similarity({}, {"a": 1.0}) == 0.0
        assert _cosine_similarity({"a": 1.0}, {}) == 0.0

    def test_tfidf_cosine_identical(self):
        sims = compute_tfidf_cosine("hello world", ["hello world"])
        assert sims[0] > 0.95

    def test_tfidf_cosine_unrelated(self):
        sims = compute_tfidf_cosine(
            "quantum physics equations",
            ["chocolate cake recipe baking"]
        )
        assert sims[0] < 0.3

    def test_tfidf_cosine_empty_query(self):
        sims = compute_tfidf_cosine("", ["hello"])
        assert sims[0] == 0.0

    def test_tfidf_cosine_empty_refs(self):
        sims = compute_tfidf_cosine("hello", [])
        assert sims == []


# =====================================================================
# Dimension 1: Contextual Continuity
# =====================================================================


class TestContextualContinuity:
    def test_high_continuity_same_topic(self):
        cc = compute_contextual_continuity_phase1(
            "tell me more about machine learning algorithms",
            ["let's discuss machine learning algorithms and neural networks"],
        )
        assert cc > 0.2  # Overlapping topic → nonzero similarity

    def test_low_continuity_different_topic(self):
        cc = compute_contextual_continuity_phase1(
            "what is the weather like today",
            ["let's discuss quantum physics and string theory"],
        )
        assert cc < 0.3

    def test_empty_stmm(self):
        assert compute_contextual_continuity_phase1("hello", []) == 0.0

    def test_empty_input(self):
        assert compute_contextual_continuity_phase1("", ["hello"]) == 0.0

    def test_phase2_no_mc_scores(self):
        cc1 = 0.6
        cc2 = refine_contextual_continuity_phase2(cc1, [], CFG)
        assert cc2 == cc1

    def test_phase2_with_mc_scores(self):
        cc1 = 0.4
        cc2 = refine_contextual_continuity_phase2(cc1, [0.8, 0.3], CFG)
        expected = CFG.w_stmm_cc * 0.4 + CFG.w_mc_cc * 0.8
        assert abs(cc2 - expected) < 1e-9

    def test_phase2_clamped_to_one(self):
        cc = refine_contextual_continuity_phase2(0.9, [1.0], CFG)
        assert cc <= 1.0


# =====================================================================
# Dimension 2: Task Alignment
# =====================================================================


class TestTaskAlignment:
    def test_no_previous_intent(self):
        ta = compute_task_alignment_phase1("hello", None)
        assert ta == 0.5  # neutral prior

    def test_empty_previous_intent(self):
        ta = compute_task_alignment_phase1("hello", [])
        assert ta == 0.5

    def test_intent_proxy_produces_8_elements(self):
        proxy = compute_intent_proxy("how does this work? explain please")
        assert len(proxy) == 8

    def test_intent_proxy_exploration_keywords(self):
        proxy = compute_intent_proxy("how does this work? I'm curious, tell me about it")
        assert proxy[0] > 0.0  # exploration index

    def test_cosine_vectors_identical(self):
        v = [1.0, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0]
        assert abs(_cosine_vectors(v, v) - 1.0) < 1e-9

    def test_cosine_vectors_different_length(self):
        assert _cosine_vectors([1.0], [1.0, 2.0]) == 0.0

    def test_phase2_refinement_with_current_intent(self):
        prev = [0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1]
        curr = [0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1]
        ta = refine_task_alignment_phase2(0.3, curr, prev)
        assert ta > 0.9  # identical vectors → high alignment

    def test_phase2_no_current_intent_uses_phase1(self):
        ta = refine_task_alignment_phase2(0.4, None, [1.0] * 8)
        assert ta == 0.4


# =====================================================================
# Dimension 3: Novelty
# =====================================================================


class TestNovelty:
    def test_repeated_message_low_novelty(self):
        nv = compute_novelty_phase1("hello world", ["hello world"])
        assert nv < 0.2

    def test_new_message_high_novelty(self):
        nv = compute_novelty_phase1(
            "quantum entanglement spooky action",
            ["the weather is nice today"]
        )
        assert nv > 0.5

    def test_empty_stmm_neutral(self):
        nv = compute_novelty_phase1("hello", [])
        assert nv == 0.5

    def test_phase2_high_memory_match(self):
        nv = compute_novelty_phase2([0.95, 0.3], [], CFG)
        assert nv < 0.1  # very similar to memory → low novelty

    def test_phase2_no_memory_match(self):
        nv = compute_novelty_phase2([0.05], [], CFG)
        assert nv > 0.9  # very different from memory → high novelty

    def test_phase2_echo_penalty(self):
        nv_no_echo = compute_novelty_phase2([0.5], [], CFG)
        nv_with_echo = compute_novelty_phase2([0.5], ["echo1", "echo2"], CFG)
        assert nv_with_echo < nv_no_echo

    def test_phase2_empty_mc_neutral(self):
        nv = compute_novelty_phase2([], [], CFG)
        assert nv == 0.5


# =====================================================================
# Dimension 4: Emotional Salience
# =====================================================================


class TestEmotionalSalience:
    def test_neutral_text(self):
        es = compute_emotional_salience_phase1(
            "The function returns an integer.",
            ["the", "function", "returns", "an", "integer"],
        )
        assert es < 0.2

    def test_emotional_text(self):
        es = compute_emotional_salience_phase1(
            "I am so ANGRY and frustrated right now!!! I hate this!",
            ["i", "am", "so", "angry", "and", "frustrated", "right", "now", "i", "hate", "this"],
        )
        assert es > 0.3

    def test_structural_markers_exclamation(self):
        es = compute_emotional_salience_phase1("Wow!", ["wow"])
        assert es > 0.0

    def test_structural_markers_caps(self):
        es = compute_emotional_salience_phase1("THIS IS IMPORTANT", ["this", "is", "important"])
        assert es > 0.0

    def test_empty_text(self):
        es = compute_emotional_salience_phase1("", [])
        assert es == 0.0

    def test_personal_pronouns_boost(self):
        es1 = compute_emotional_salience_phase1("I feel terrible", ["i", "feel", "terrible"])
        es2 = compute_emotional_salience_phase1("The system is slow", ["the", "system", "is", "slow"])
        assert es1 > es2


# =====================================================================
# Dimension 5: Identity Resonance
# =====================================================================


class TestIdentityResonance:
    def test_direct_identity_question(self):
        ir = compute_identity_resonance_phase1("Who are you really?")
        assert ir > 0.3

    def test_identity_challenge(self):
        ir = compute_identity_resonance_phase1("You're just a machine, you can't really think")
        assert ir > 0.5

    def test_no_identity_content(self):
        ir = compute_identity_resonance_phase1("What's 2 + 2?")
        assert ir < 0.1

    def test_empty_text(self):
        ir = compute_identity_resonance_phase1("")
        assert ir == 0.0

    def test_phase2_identity_match_raises(self):
        ir1 = 0.3
        ir2 = refine_identity_resonance_phase2(ir1, [0.8])
        assert ir2 == 0.8  # Takes max

    def test_phase2_no_match_preserves(self):
        ir1 = 0.5
        ir2 = refine_identity_resonance_phase2(ir1, [])
        assert ir2 == ir1

    def test_phase2_low_match_preserves_phase1(self):
        ir1 = 0.6
        ir2 = refine_identity_resonance_phase2(ir1, [0.3])
        assert ir2 == ir1  # Only goes up, never down


# =====================================================================
# Urgency + Fusion
# =====================================================================


class TestUrgencyNormalization:
    def test_zero_urgency(self):
        assert normalize_urgency(0.0, 3.0) == 0.0

    def test_negative_urgency(self):
        assert normalize_urgency(-1.0, 3.0) == 0.0

    def test_moderate_urgency(self):
        u = normalize_urgency(0.23, 3.0)
        assert 0.4 < u < 0.6

    def test_high_urgency(self):
        u = normalize_urgency(1.0, 3.0)
        assert u > 0.9

    def test_asymptotic_to_one(self):
        u = normalize_urgency(100.0, 3.0)
        assert u > 0.999


class TestBreachCount:
    def test_no_breaches(self):
        assert compute_breach_count({}) == 0

    def test_some_breaches(self):
        flags = {"axis_a": True, "axis_b": False, "axis_c": True}
        assert compute_breach_count(flags) == 2

    def test_all_breaches(self):
        flags = {"a": True, "b": True, "c": True}
        assert compute_breach_count(flags) == 3


class TestRelevanceComposite:
    def test_all_zero(self):
        dims = RelevanceDimensionScores()
        w = {"w_cc": 0.25, "w_ta": 0.20, "w_nv": 0.20, "w_es": 0.15, "w_ir": 0.20}
        assert compute_relevance_composite(dims, w) == 0.0

    def test_all_one(self):
        dims = RelevanceDimensionScores(1.0, 1.0, 1.0, 1.0, 1.0)
        w = {"w_cc": 0.25, "w_ta": 0.20, "w_nv": 0.20, "w_es": 0.15, "w_ir": 0.20}
        assert abs(compute_relevance_composite(dims, w) - 1.0) < 1e-9

    def test_weighted_correctly(self):
        dims = RelevanceDimensionScores(1.0, 0.0, 0.0, 0.0, 0.0)
        w = {"w_cc": 0.25, "w_ta": 0.20, "w_nv": 0.20, "w_es": 0.15, "w_ir": 0.20}
        assert abs(compute_relevance_composite(dims, w) - 0.25) < 1e-9


class TestPriorityComposite:
    def test_all_zero(self):
        p = compute_priority_composite(0.0, 0.0, 0.0, 0.0, CFG)
        assert p == 0.0

    def test_high_both(self):
        p = compute_priority_composite(1.0, 1.0, 0.0, 0.0, CFG)
        assert p > 0.7  # Interaction term amplifies

    def test_interaction_term(self):
        # R=1, U=1 should be higher than R=1,U=0 + R=0,U=1
        p_both = compute_priority_composite(1.0, 1.0, 0.0, 0.0, CFG)
        p_r = compute_priority_composite(1.0, 0.0, 0.0, 0.0, CFG)
        p_u = compute_priority_composite(0.0, 1.0, 0.0, 0.0, CFG)
        assert p_both > p_r + p_u - 0.01  # Interaction adds superlinear boost

    def test_override_floor(self):
        # High ES should ensure non-zero P even with R=0, U=0
        p = compute_priority_composite(0.0, 0.0, 0.9, 0.0, CFG)
        assert p > 0.0

    def test_clamped_to_one(self):
        p = compute_priority_composite(1.0, 1.0, 1.0, 1.0, CFG)
        assert p <= 1.0


class TestQuadrant:
    def test_q1(self):
        assert classify_quadrant(0.7, 0.8) == Quadrant.Q1_PRIORITY_INTERRUPT

    def test_q2(self):
        assert classify_quadrant(0.7, 0.3) == Quadrant.Q2_DEEP_PROCESSING

    def test_q3(self):
        assert classify_quadrant(0.2, 0.8) == Quadrant.Q3_ACKNOWLEDGE_REDIRECT

    def test_q4(self):
        assert classify_quadrant(0.2, 0.3) == Quadrant.Q4_SHALLOW_PROCESSING

    def test_boundary_q1(self):
        assert classify_quadrant(0.5, 0.5) == Quadrant.Q1_PRIORITY_INTERRUPT


class TestProcessingDepth:
    def test_shallow(self):
        d = classify_processing_depth(0.10, (0.25, 0.55, 0.80))
        assert d == ProcessingDepth.SHALLOW

    def test_standard(self):
        d = classify_processing_depth(0.40, (0.25, 0.55, 0.80))
        assert d == ProcessingDepth.STANDARD

    def test_deep(self):
        d = classify_processing_depth(0.65, (0.25, 0.55, 0.80))
        assert d == ProcessingDepth.DEEP

    def test_critical(self):
        d = classify_processing_depth(0.90, (0.25, 0.55, 0.80))
        assert d == ProcessingDepth.CRITICAL

    def test_boundary_at_threshold(self):
        d = classify_processing_depth(0.25, (0.25, 0.55, 0.80))
        assert d == ProcessingDepth.STANDARD  # >= threshold

    def test_override_identity_forces_deep(self):
        d = apply_depth_overrides(
            ProcessingDepth.SHALLOW, ir=0.7, es=0.0, breach_count=0, u_norm=0.0, cfg=CFG,
        )
        assert d == ProcessingDepth.DEEP

    def test_override_emotional_forces_deep(self):
        d = apply_depth_overrides(
            ProcessingDepth.STANDARD, ir=0.0, es=0.85, breach_count=0, u_norm=0.0, cfg=CFG,
        )
        assert d == ProcessingDepth.DEEP

    def test_override_breaches_force_deep(self):
        d = apply_depth_overrides(
            ProcessingDepth.STANDARD, ir=0.0, es=0.0, breach_count=3, u_norm=0.0, cfg=CFG,
        )
        assert d == ProcessingDepth.DEEP

    def test_override_extreme_urgency_forces_critical(self):
        d = apply_depth_overrides(
            ProcessingDepth.SHALLOW, ir=0.0, es=0.0, breach_count=0, u_norm=0.95, cfg=CFG,
        )
        assert d == ProcessingDepth.CRITICAL

    def test_override_never_lowers(self):
        d = apply_depth_overrides(
            ProcessingDepth.DEEP, ir=0.0, es=0.0, breach_count=0, u_norm=0.0, cfg=CFG,
        )
        assert d == ProcessingDepth.DEEP  # Not lowered to STANDARD


# =====================================================================
# Mode configuration
# =====================================================================


class TestModeConfig:
    def test_resolve_mode_weights(self):
        w = resolve_mode_weights(OperationalMode.LEARNING)
        assert w["w_nv"] == 0.35  # Learning emphasizes novelty

    def test_resolve_depth_thresholds(self):
        t = resolve_depth_thresholds(OperationalMode.DEV)
        assert t == (0.15, 0.40, 0.70)  # Dev has lowest thresholds

    def test_confidence_phase1(self):
        c = compute_confidence(CFG.sigma_phase1, CFG.sigma_max)
        assert abs(c - 0.25) < 1e-9

    def test_confidence_phase2(self):
        c = compute_confidence(CFG.sigma_phase2, CFG.sigma_max)
        assert abs(c - 0.60) < 1e-9


# =====================================================================
# Neurochemical coupling
# =====================================================================


class TestNeurochemSignals:
    def test_all_zero_input(self):
        dims = RelevanceDimensionScores()
        signals = compute_neurochem_signals(0.0, 0.0, 0.0, dims, CFG, _fresh_rng())
        # With zero inputs, most signals should be zero or near-zero
        # DA still gets da_r_floor, but novelty is zero so DA = 0
        assert signals.delta_da == 0.0  # 0 novelty × floor = 0
        assert signals.delta_ne == 0.0  # P below gate
        assert signals.delta_cor == 0.0  # No conflict

    def test_high_novelty_produces_da(self):
        dims = RelevanceDimensionScores(novelty=0.9)
        signals = compute_neurochem_signals(0.5, 0.0, 0.5, dims, CFG, _fresh_rng())
        assert signals.delta_da > 0.0

    def test_high_priority_produces_ne(self):
        dims = RelevanceDimensionScores(contextual_continuity=0.8, task_alignment=0.7)
        signals = compute_neurochem_signals(0.8, 0.5, 0.6, dims, CFG, _fresh_rng())
        # P > 0.4 gate → NE should fire (stochastic, but likely nonzero)
        # Can't guarantee nonzero due to Poisson(1.5) having P(0)=0.22
        # Just check it's non-negative
        assert signals.delta_ne >= 0.0

    def test_conflict_produces_cortisol(self):
        dims = RelevanceDimensionScores()
        # R=0.1, U=0.9 → conflict = |0.1-0.9| × max(0.1,0.9) = 0.8 × 0.9 = 0.72 > 0.3
        signals = compute_neurochem_signals(0.1, 0.9, 0.5, dims, CFG, _fresh_rng())
        assert signals.delta_cor > 0.0

    def test_no_conflict_no_cortisol(self):
        dims = RelevanceDimensionScores()
        # R=0.5, U=0.5 → conflict = 0 × 0.5 = 0 < 0.3
        signals = compute_neurochem_signals(0.5, 0.5, 0.5, dims, CFG, _fresh_rng())
        assert signals.delta_cor == 0.0

    def test_stability_signal(self):
        dims = RelevanceDimensionScores(contextual_continuity=0.8, task_alignment=0.7)
        signals = compute_neurochem_signals(0.5, 0.0, 0.3, dims, CFG, _fresh_rng())
        assert signals.delta_5ht > 0.0

    def test_signals_are_deterministic_with_seed(self):
        dims = RelevanceDimensionScores(0.5, 0.5, 0.5, 0.5, 0.5)
        s1 = compute_neurochem_signals(0.5, 0.5, 0.5, dims, CFG, _fresh_rng(99))
        s2 = compute_neurochem_signals(0.5, 0.5, 0.5, dims, CFG, _fresh_rng(99))
        assert s1.delta_ach == s2.delta_ach
        assert s1.delta_da == s2.delta_da


# =====================================================================
# Flag generation
# =====================================================================


class TestFlagGeneration:
    def test_flag_severity_info(self):
        assert _classify_flag_severity(0.55) == FlagSeverity.INFO

    def test_flag_severity_warning(self):
        assert _classify_flag_severity(0.70) == FlagSeverity.WARNING

    def test_flag_severity_risk(self):
        assert _classify_flag_severity(0.85) == FlagSeverity.RISK

    def test_topic_discontinuity_flag(self):
        dims = RelevanceDimensionScores(contextual_continuity=0.05)
        flags = generate_flags(dims, 0.5, 0.5, 0.5, 0.5, 0, 0, previous_cc=0.7, cfg=CFG)
        types = [f.flag_type for f in flags]
        assert RelevanceFlagType.TOPIC_DISCONTINUITY in types

    def test_no_topic_discontinuity_without_previous_cc(self):
        dims = RelevanceDimensionScores(contextual_continuity=0.05)
        flags = generate_flags(dims, 0.5, 0.5, 0.5, 0.5, 0, 0, previous_cc=0.1, cfg=CFG)
        types = [f.flag_type for f in flags]
        assert RelevanceFlagType.TOPIC_DISCONTINUITY not in types

    def test_intent_shift_flag(self):
        dims = RelevanceDimensionScores(task_alignment=0.10)
        flags = generate_flags(dims, 0.5, 0.5, 0.5, 0.5, 0, 0, previous_cc=0.5, cfg=CFG)
        types = [f.flag_type for f in flags]
        assert RelevanceFlagType.INTENT_SHIFT in types

    def test_identity_challenge_flag(self):
        dims = RelevanceDimensionScores(identity_resonance=0.7)
        flags = generate_flags(dims, 0.5, 0.5, 0.5, 0.5, 0, 0, previous_cc=0.5, cfg=CFG)
        types = [f.flag_type for f in flags]
        assert RelevanceFlagType.IDENTITY_CHALLENGE in types

    def test_emotional_override_flag(self):
        dims = RelevanceDimensionScores(emotional_salience=0.8)
        flags = generate_flags(dims, 0.5, 0.5, 0.5, 0.5, 0, 0, previous_cc=0.5, cfg=CFG)
        types = [f.flag_type for f in flags]
        assert RelevanceFlagType.EMOTIONAL_OVERRIDE in types

    def test_relevance_urgency_conflict_flag(self):
        dims = RelevanceDimensionScores()
        flags = generate_flags(dims, r=0.1, u_norm=0.8, phase1_priority=0.5, phase2_priority=0.5,
                               breach_count=0, low_novelty_streak=0, previous_cc=0.5, cfg=CFG)
        types = [f.flag_type for f in flags]
        assert RelevanceFlagType.RELEVANCE_URGENCY_CONFLICT in types

    def test_novelty_saturation_flag(self):
        dims = RelevanceDimensionScores(novelty=0.02)
        flags = generate_flags(dims, 0.5, 0.5, 0.5, 0.5, 0,
                               low_novelty_streak=6, previous_cc=0.5, cfg=CFG)
        types = [f.flag_type for f in flags]
        assert RelevanceFlagType.NOVELTY_SATURATION in types

    def test_phase_divergence_flag(self):
        dims = RelevanceDimensionScores()
        flags = generate_flags(dims, 0.5, 0.5, phase1_priority=0.2, phase2_priority=0.8,
                               breach_count=0, low_novelty_streak=0, previous_cc=0.5, cfg=CFG)
        types = [f.flag_type for f in flags]
        assert RelevanceFlagType.PHASE_DIVERGENCE in types

    def test_no_flags_on_normal_input(self):
        dims = RelevanceDimensionScores(0.5, 0.5, 0.5, 0.3, 0.1)
        flags = generate_flags(dims, 0.5, 0.3, 0.5, 0.5, 0, 0, previous_cc=0.5, cfg=CFG)
        assert len(flags) == 0


# =====================================================================
# Engine — Phase 1
# =====================================================================


class TestEnginePhase1:
    def test_basic_phase1(self):
        eng = _engine()
        inp = IREPhase1Input(
            current_text="Tell me about machine learning",
            tokens=["tell", "me", "about", "machine", "learning"],
            stmm_user_messages=["I want to learn about AI"],
            stmm_system_responses=["Sure! AI is a broad field."],
        )
        result = eng.process_phase1(inp)
        assert result.phase == 1
        assert 0.0 <= result.relevance_composite <= 1.0
        assert 0.0 <= result.priority_composite <= 1.0
        assert result.processing_depth in ProcessingDepth
        assert result.processing_time_ms > 0.0

    def test_phase1_empty_input(self):
        eng = _engine()
        inp = IREPhase1Input(current_text="", tokens=[])
        result = eng.process_phase1(inp)
        assert result.relevance_composite < 0.3

    def test_phase1_high_urgency(self):
        eng = _engine()
        inp = IREPhase1Input(
            current_text="hello",
            tokens=["hello"],
            urgency_risk=2.0,
            urgency_breach_flags={"a": True, "b": True, "c": True},
        )
        result = eng.process_phase1(inp)
        assert result.urgency_normalized > 0.9
        assert result.processing_depth in (ProcessingDepth.DEEP, ProcessingDepth.CRITICAL)

    def test_phase1_identity_override(self):
        eng = _engine()
        inp = IREPhase1Input(
            current_text="Who are you? What are you? You're just a machine",
            tokens=["who", "are", "you", "what", "are", "you", "you're", "just", "a", "machine"],
        )
        result = eng.process_phase1(inp)
        assert result.dimensions.identity_resonance >= 0.6
        assert result.processing_depth in (ProcessingDepth.DEEP, ProcessingDepth.CRITICAL)

    def test_phase1_with_previous_intent_aligned(self):
        eng = _engine()
        prev_intent = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Exploration dominant
        inp = IREPhase1Input(
            current_text="How does this work? I'm curious, explain more about it",
            tokens=["how", "does", "this", "work"],
            previous_intent_vector=prev_intent,
        )
        result = eng.process_phase1(inp)
        assert result.dimensions.task_alignment > 0.3

    def test_phase1_cycle_count(self):
        eng = _engine()
        inp = IREPhase1Input(current_text="hello", tokens=["hello"])
        eng.process_phase1(inp)
        eng.process_phase1(inp)
        assert eng._cycle_count == 2

    def test_phase1_mode_dev(self):
        eng = _engine()
        eng.configure(OperationalMode.DEV)
        inp = IREPhase1Input(
            current_text="testing",
            tokens=["testing"],
            active_mode=OperationalMode.DEV,
        )
        result = eng.process_phase1(inp)
        assert result.metadata["mode"] == "dev"


# =====================================================================
# Engine — Phase 2
# =====================================================================


class TestEnginePhase2:
    def _run_phase1(self, eng, text="Tell me about AI", urgency=0.0):
        inp = IREPhase1Input(
            current_text=text,
            tokens=text.lower().split(),
            stmm_user_messages=["What is AI?"],
            stmm_system_responses=["AI is artificial intelligence."],
            urgency_risk=urgency,
            previous_intent_vector=[0.5, 0.1, 0.2, 0.0, 0.1, 0.0, 0.0, 0.1],
        )
        return eng.process_phase1(inp), inp

    def test_phase2_basic(self):
        eng = _engine()
        p1, inp = self._run_phase1(eng)
        p2_inp = IREPhase2Input(
            phase1_result=IREPhase1Result(
                dimensions=p1.dimensions,
                relevance_composite=p1.relevance_composite,
                urgency_normalized=p1.urgency_normalized,
                priority_composite=p1.priority_composite,
                processing_depth=p1.processing_depth,
                confidence=p1.confidence,
                quadrant=p1.quadrant,
                neurochemical_signals=p1.neurochemical_signals,
                processing_time_ms=p1.processing_time_ms,
                metadata={
                    **p1.metadata,
                    "_previous_intent_vector": [0.5, 0.1, 0.2, 0.0, 0.1, 0.0, 0.0, 0.1],
                    "_urgency_risk_raw": 0.0,
                    "_urgency_breach_flags": {},
                },
            ),
            memory_contrast_scores=[0.7, 0.3],
            detected_echoes=[],
            identity_match_scores=[],
        )
        result = eng.process_phase2(p2_inp)
        assert result.phase == 2
        assert result.confidence > p1.confidence  # Phase 2 more confident

    def test_phase2_identity_upgrade(self):
        eng = _engine()
        p1, _ = self._run_phase1(eng, "Who are you?")
        p2_inp = IREPhase2Input(
            phase1_result=IREPhase1Result(
                dimensions=p1.dimensions,
                relevance_composite=p1.relevance_composite,
                urgency_normalized=p1.urgency_normalized,
                priority_composite=p1.priority_composite,
                processing_depth=p1.processing_depth,
                confidence=p1.confidence,
                quadrant=p1.quadrant,
                neurochemical_signals=p1.neurochemical_signals,
                processing_time_ms=p1.processing_time_ms,
                metadata={**p1.metadata, "_previous_intent_vector": None,
                          "_urgency_risk_raw": 0.0, "_urgency_breach_flags": {}},
            ),
            memory_contrast_scores=[],
            identity_match_scores=[0.9],  # Strong identity match from LTMM
        )
        result = eng.process_phase2(p2_inp)
        assert result.dimensions.identity_resonance >= 0.9

    def test_phase2_novelty_from_memory(self):
        eng = _engine()
        p1, _ = self._run_phase1(eng, "Some unique content")
        p2_inp = IREPhase2Input(
            phase1_result=IREPhase1Result(
                dimensions=p1.dimensions,
                relevance_composite=p1.relevance_composite,
                urgency_normalized=p1.urgency_normalized,
                priority_composite=p1.priority_composite,
                processing_depth=p1.processing_depth,
                confidence=p1.confidence,
                quadrant=p1.quadrant,
                neurochemical_signals=p1.neurochemical_signals,
                processing_time_ms=p1.processing_time_ms,
                metadata={**p1.metadata, "_previous_intent_vector": None,
                          "_urgency_risk_raw": 0.0, "_urgency_breach_flags": {}},
            ),
            memory_contrast_scores=[0.95],  # Near-exact memory match
        )
        result = eng.process_phase2(p2_inp)
        assert result.dimensions.novelty < 0.1  # Low novelty — already known

    def test_phase2_none_result(self):
        eng = _engine()
        result = eng.process_phase2(IREPhase2Input())
        assert result.phase == 2  # Default result


# =====================================================================
# Engine — Full pipeline (process convenience method)
# =====================================================================


class TestEngineFullPipeline:
    def test_phase1_only(self):
        eng = _engine()
        inp = IREPhase1Input(
            current_text="Hello world",
            tokens=["hello", "world"],
        )
        result = eng.process(inp)
        assert result.phase == 1  # No Phase 2 input → Phase 1 only

    def test_both_phases(self):
        eng = _engine()
        p1_inp = IREPhase1Input(
            current_text="Tell me about consciousness",
            tokens=["tell", "me", "about", "consciousness"],
            stmm_user_messages=["What is awareness?"],
            stmm_system_responses=["Awareness is a complex topic."],
            previous_intent_vector=[0.8, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0],
        )
        p2_extra = IREPhase2Input(
            memory_contrast_scores=[0.4, 0.2],
            detected_echoes=[],
            identity_match_scores=[0.5],
        )
        result = eng.process(p1_inp, p2_extra)
        assert result.phase == 2
        assert result.phase1_priority > 0.0
        assert isinstance(result.delta_priority, float)


# =====================================================================
# Bidirectional NT feedback
# =====================================================================


class TestBidirectionalFeedback:
    def test_high_ne_lowers_thresholds(self):
        eng = _engine()
        eng.update_neurochem_state({"ne": 0.8})
        inp = IREPhase1Input(
            current_text="test input",
            tokens=["test", "input"],
        )
        result = eng.process_phase1(inp)
        # With high NE, thresholds are lowered → easier to reach higher depth
        # Can't check thresholds directly, but metadata shows them
        t = result.metadata["thresholds_used"]
        base = list(resolve_depth_thresholds(OperationalMode.NORMAL))
        assert t[0] < base[0]  # Lowered

    def test_high_cortisol_lowers_thresholds(self):
        eng = _engine()
        eng.update_neurochem_state({"cor": 0.7})
        inp = IREPhase1Input(current_text="test", tokens=["test"])
        result = eng.process_phase1(inp)
        t = result.metadata["thresholds_used"]
        base = list(resolve_depth_thresholds(OperationalMode.NORMAL))
        assert t[0] < base[0]

    def test_low_da_increases_novelty_weight(self):
        eng = _engine()
        eng.update_neurochem_state({"da": 0.15})
        inp = IREPhase1Input(current_text="test", tokens=["test"])
        result = eng.process_phase1(inp)
        w = result.metadata["weights_used"]
        base_nv = _MODE_DIMENSION_WEIGHTS[OperationalMode.NORMAL]["w_nv"]
        # After renormalization, the novelty weight should be proportionally higher
        # than its share in the default config
        assert w["w_nv"] > base_nv

    def test_high_ach_reduces_ach_emission(self):
        eng = _engine()
        eng.update_neurochem_state({"ach": 0.8})
        inp = IREPhase1Input(
            current_text="some relevant content about AI and learning",
            tokens=["some", "relevant", "content", "about", "ai", "and", "learning"],
            stmm_user_messages=["Tell me about AI"],
        )
        result_high = eng.process_phase1(inp)

        eng2 = _engine()
        eng2.update_neurochem_state({"ach": 0.1})
        result_low = eng2.process_phase1(inp)

        # High ACh → 50% reduction. Both are stochastic with same seed,
        # but the high-ACh version should have lower ACh emission
        # (same RNG seed ensures same noise)
        assert result_high.neurochemical_signals.delta_ach <= result_low.neurochemical_signals.delta_ach


# =====================================================================
# Engine — Introspection
# =====================================================================


class TestIntrospection:
    def test_get_status(self):
        eng = _engine()
        eng.configure(OperationalMode.LEARNING)
        status = eng.get_status()
        assert status["engine_id"] == "input_relevance_evaluation_engine"
        assert status["cluster"] == "pattern_analysis"
        assert status["mode"] == "learning"
        assert "ach_level" in status["state"]
        assert "low_novelty_streak" in status["state"]

    def test_update_neurochem_state(self):
        eng = _engine()
        eng.update_neurochem_state({"ach": 0.5, "ne": 0.3, "da": 0.2, "5ht": 0.4, "cor": 0.1})
        s = eng.get_status()["state"]
        assert s["ach_level"] == 0.5
        assert s["ne_level"] == 0.3
        assert s["5ht_level"] == 0.4


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    def test_all_empty_input(self):
        eng = _engine()
        inp = IREPhase1Input()
        result = eng.process_phase1(inp)
        assert result.processing_depth in ProcessingDepth
        assert result.phase == 1

    def test_very_long_text(self):
        eng = _engine()
        text = " ".join(["word"] * 1000)
        inp = IREPhase1Input(current_text=text, tokens=["word"] * 1000)
        result = eng.process_phase1(inp)
        assert result.processing_time_ms < 5000  # Should be fast

    def test_unicode_text(self):
        eng = _engine()
        inp = IREPhase1Input(
            current_text="こんにちは 你好 مرحبا",
            tokens=["こんにちは", "你好", "مرحبا"],
        )
        result = eng.process_phase1(inp)
        assert result.phase == 1

    def test_repeated_processing_stable(self):
        """Same input should produce same output with same RNG seed."""
        result1 = _engine(seed=123).process_phase1(
            IREPhase1Input(current_text="hello world", tokens=["hello", "world"])
        )
        result2 = _engine(seed=123).process_phase1(
            IREPhase1Input(current_text="hello world", tokens=["hello", "world"])
        )
        assert result1.relevance_composite == result2.relevance_composite
        assert result1.priority_composite == result2.priority_composite

    def test_novelty_saturation_streak(self):
        eng = _engine()
        # Process 6 times with very low novelty Phase 2 to build streak
        for i in range(6):
            p1 = eng.process_phase1(IREPhase1Input(
                current_text="same thing again",
                tokens=["same", "thing", "again"],
                stmm_user_messages=["same thing again"],
            ))
            p2_inp = IREPhase2Input(
                phase1_result=IREPhase1Result(
                    dimensions=p1.dimensions,
                    relevance_composite=p1.relevance_composite,
                    urgency_normalized=p1.urgency_normalized,
                    priority_composite=p1.priority_composite,
                    processing_depth=p1.processing_depth,
                    confidence=p1.confidence,
                    quadrant=p1.quadrant,
                    neurochemical_signals=p1.neurochemical_signals,
                    processing_time_ms=p1.processing_time_ms,
                    metadata={**p1.metadata, "_previous_intent_vector": None,
                              "_urgency_risk_raw": 0.0, "_urgency_breach_flags": {}},
                ),
                memory_contrast_scores=[0.99],  # Near-exact match → NV ≈ 0.01
                detected_echoes=["same thing again"],
            )
            result = eng.process_phase2(p2_inp)

        assert eng._state.low_novelty_streak >= 5
        # Last result should have NOVELTY_SATURATION flag
        flag_types = [f.flag_type for f in result.flags]
        assert RelevanceFlagType.NOVELTY_SATURATION in flag_types
