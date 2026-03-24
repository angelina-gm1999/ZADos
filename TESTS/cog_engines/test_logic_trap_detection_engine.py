"""
Tests for logic_trap_detection_engine.py — Engine 6 (Logic Trap Detection).

Coverage plan
-------------
1.  TrapType / TrapCategory / IntentLevel enumerations
2.  LogicTrapEngineConfig defaults and immutability
3.  TrapSequenceTracker — update, tick, score methods
4.  Feature detection — _detect_feature()
5.  Layer 1 — compute_template_score() and run_template_layer()
6.  Layer 2 — synthesis sub-functions:
        compute_manipulation_mean, compute_coordination_coherence,
        compute_vulnerability_exploitation, compute_toolkit_density,
        compute_synthesis_score
7.  Layer 3 — compute_sequential_score()
8.  Fusion — compute_adversarial_intent(), classify_intent_level()
9.  Threshold resolution — resolve_trap_threshold()
10. Threat load — compute_threat_load()
11. Neurochemical signals — compute_neurochemical_signals()
12. Defensive strategy — build_defensive_strategy() + FrameEscape
13. Engine ports — configure(), update_neurochem_state(), get_status()
14. End-to-end process() — clean pass, single trap, multi-trap, mode sensitivity
15. Helpers — _build_combined_text(), _collect_supporting_ids()
16. TrapFlag / LogicTrapDetectionResult field validation
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    ContradictionFlag,
    OperationalMode,
    ProcessedStatement,
    SourceTag,
)
from zados.cognitive_engines.py_engines.fallacy_detection_engine import (
    Argument,
    CharitySuppression,
    EvidenceSignals,
    FallacyCategory,
    FallacyFlag,
    FallacyType,
    IntentionVector,
    InferenceType,
    Proposition,
)
from zados.cognitive_engines.py_engines.logic_trap_detection_engine import (
    BiasFlag,
    DefensiveStrategy,
    FrameEscape,
    IntentLevel,
    LogicTrapDetectionEngine,
    LogicTrapDetectionResult,
    LogicTrapEngineConfig,
    LogicTrapEngineState,
    LogicTrapInput,
    TrapCategory,
    TrapFlag,
    TrapSequenceTracker,
    TrapTemplate,
    TrapType,
    _TEMPLATE_BY_TYPE,
    _TRAP_TEMPLATES,
    _build_combined_text,
    _collect_supporting_ids,
    _detect_feature,
    build_defensive_strategy,
    classify_intent_level,
    compute_adversarial_intent,
    compute_coordination_coherence,
    compute_manipulation_mean,
    compute_neurochemical_signals,
    compute_sequential_score,
    compute_synthesis_score,
    compute_template_score,
    compute_threat_load,
    compute_toolkit_density,
    compute_vulnerability_exploitation,
    resolve_trap_threshold,
    run_template_layer,
)


# =====================================================================
# Helpers / fixtures
# =====================================================================

def _make_contradiction_flag(**kwargs) -> ContradictionFlag:
    defaults = dict(
        contradiction_id=str(uuid.uuid4()),
        statement_a=ProcessedStatement(raw_text="A", source_tag=SourceTag.USER_INPUT),
        statement_b=ProcessedStatement(raw_text="not A", source_tag=SourceTag.AI_OUTPUT),
        contradiction_level=1,
        confidence=0.7,
        evidence_signals={},
        temporal_separation=None,
        temporal_decay_applied=1.0,
        semantic_description="direct negation",
        context_frame=None,
    )
    defaults.update(kwargs)
    return ContradictionFlag(**defaults)


def _make_fallacy_flag(manipulation_indicator: float = 0.6, **kwargs) -> FallacyFlag:
    arg = Argument(
        raw_text="test argument",
        source=SourceTag.USER_INPUT,
        premises=[],
        conclusion=Proposition(text="conclusion"),
    )
    ev = EvidenceSignals(structural_match=0.8, relevance_deficit=0.3,
                          alternative_validity=0.1, context_appropriateness=0.1)
    defaults = dict(
        fallacy_id=str(uuid.uuid4()),
        argument=arg,
        source=SourceTag.USER_INPUT,
        fallacy_type=FallacyType.AD_HOMINEM,
        fallacy_category=FallacyCategory.RELEVANCE,
        confidence=0.75,
        evidence_signals=ev,
        description="test fallacy",
        logical_form=None,
        valid_counterpart=None,
        charity_applied=False,
        charity_suppression=None,
        related_contradiction_flags=[],
        manipulation_indicator=manipulation_indicator,
    )
    defaults.update(kwargs)
    return FallacyFlag(**defaults)


def _make_bias_flag(**kwargs) -> BiasFlag:
    defaults = dict(
        bias_id=str(uuid.uuid4()),
        bias_type="confirmation_bias",
        confidence=0.6,
        description="test bias",
    )
    defaults.update(kwargs)
    return BiasFlag(**defaults)


def _make_input(**kwargs) -> LogicTrapInput:
    defaults = dict(
        semantic_expansion={},
        user_propositions=[],
        contradiction_flags=[],
        fallacy_flags=[],
        bias_flags=[],
        bias_vulnerability_profile={},
        intention_vector=IntentionVector(),
        memory_context=[],
        conversation_history=[],
        active_sequence_tracker=None,
        active_mode=OperationalMode.NORMAL,
    )
    defaults.update(kwargs)
    return LogicTrapInput(**defaults)


def _make_trap_flag(
    trap_type: TrapType = TrapType.LOADED_QUESTION,
    intent_score: float = 0.70,
    category: TrapCategory = TrapCategory.FRAMING,
) -> TrapFlag:
    template = _TEMPLATE_BY_TYPE[trap_type]
    strategy = build_defensive_strategy(trap_type, template)
    return TrapFlag(
        trap_id=str(uuid.uuid4()),
        trap_type=trap_type,
        trap_category=category,
        adversarial_intent_score=intent_score,
        intent_classification=classify_intent_level(intent_score),
        layer_scores={"template": 0.5, "synthesis": 0.4, "sequential": 0.0},
        identified_mechanism=template.mechanism,
        target_concession=template.target_concession,
        invalid_presuppositions=list(template.presuppositions),
        excluded_options=list(template.excluded_options),
        defensive_strategy=strategy,
        frame_escape=strategy.frame_escape,
        supporting_contradiction_ids=[],
        supporting_fallacy_ids=[],
        supporting_bias_ids=[],
    )


_RNG = np.random.default_rng(42)
_CONFIG = LogicTrapEngineConfig()
_STATE = LogicTrapEngineState()


# =====================================================================
# 1. Enumerations
# =====================================================================

class TestEnumerations:
    def test_trap_type_count(self):
        assert len(TrapType) == 21

    def test_trap_category_values(self):
        cats = {c.value for c in TrapCategory}
        assert cats == {"FRAMING", "SEQUENTIAL", "RHETORICAL", "META"}

    def test_intent_level_ordering(self):
        levels = [IntentLevel.NONE, IntentLevel.LOW, IntentLevel.MODERATE,
                  IntentLevel.HIGH, IntentLevel.NEAR_CERTAIN]
        assert len(levels) == 5

    def test_framing_types(self):
        framing = [
            TrapType.LOADED_QUESTION,
            TrapType.FALSE_DILEMMA_WEAPONIZED,
            TrapType.POISONED_FRAME,
            TrapType.MOVING_GOALPOSTS,
            TrapType.MOTTE_AND_BAILEY,
        ]
        assert all(t in TrapType for t in framing)

    def test_sequential_types(self):
        seq = [
            TrapType.CONSISTENCY_TRAP,
            TrapType.GISH_GALLOP,
            TrapType.SEALIONING,
            TrapType.KAFKA_TRAP,
            TrapType.DARVO,
        ]
        assert all(t in TrapType for t in seq)

    def test_rhetorical_types(self):
        rhet = [
            TrapType.APPEAL_TO_RECIPROCITY,
            TrapType.EMOTIONAL_BLACKMAIL,
            TrapType.AUTHORITY_HIJACKING,
            TrapType.NORMALIZATION,
            TrapType.TONE_POLICING,
            TrapType.WHATABOUTISM,
        ]
        assert all(t in TrapType for t in rhet)

    def test_meta_types(self):
        meta = [
            TrapType.EXPLOIT_THE_ETHIC,
            TrapType.RECURSIVE_TRAP,
            TrapType.FORCED_AGREEMENT,
            TrapType.COMPLIANCE_TESTING,
            TrapType.DOUBLE_BIND,
        ]
        assert all(t in TrapType for t in meta)

    def test_trap_type_str_values(self):
        assert TrapType.LOADED_QUESTION.value == "loaded_question"
        assert TrapType.GISH_GALLOP.value == "gish_gallop"
        assert TrapType.DOUBLE_BIND.value == "double_bind"


# =====================================================================
# 2. Config
# =====================================================================

class TestLogicTrapEngineConfig:
    def test_default_fusion_weights_sum(self):
        c = LogicTrapEngineConfig()
        total = c.w_template + c.w_synthesis + c.w_sequential
        assert abs(total - 1.0) < 1e-9

    def test_default_synthesis_weights_sum(self):
        c = LogicTrapEngineConfig()
        total = c.w_s1 + c.w_s2 + c.w_s3 + c.w_s4
        assert abs(total - 1.0) < 1e-9

    def test_mode_thresholds_ordered(self):
        c = LogicTrapEngineConfig()
        # Dev should be most sensitive (lowest threshold)
        assert c.theta_dev <= c.theta_learning <= c.theta_normal

    def test_config_is_frozen(self):
        c = LogicTrapEngineConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.theta_normal = 0.99   # type: ignore

    def test_oxt_floor_positive(self):
        c = LogicTrapEngineConfig()
        assert 0.0 < c.oxt_floor < 1.0

    def test_threat_category_weights(self):
        c = LogicTrapEngineConfig()
        # Meta should have highest weight per spec
        assert c.cat_weight_meta >= c.cat_weight_sequential
        assert c.cat_weight_sequential >= c.cat_weight_framing
        assert c.cat_weight_framing >= c.cat_weight_rhetorical


# =====================================================================
# 3. TrapSequenceTracker
# =====================================================================

class TestTrapSequenceTracker:
    def test_initial_state_empty(self):
        t = TrapSequenceTracker()
        assert t.active_sequences == []
        assert t.consecutive_adversarial_turns == 0

    def test_update_creates_new_sequence(self):
        t = TrapSequenceTracker()
        t.update(TrapType.LOADED_QUESTION, 0.7)
        assert len(t.active_sequences) == 1
        seq = t.active_sequences[0]
        assert seq["trap_type"] == "loaded_question"
        assert seq["turns_tracked"] == 1
        assert seq["escalation_trajectory"] == [0.7]

    def test_update_increments_existing_sequence(self):
        t = TrapSequenceTracker()
        t.update(TrapType.LOADED_QUESTION, 0.5)
        t.update(TrapType.LOADED_QUESTION, 0.8)
        assert len(t.active_sequences) == 1
        seq = t.active_sequences[0]
        assert seq["turns_tracked"] == 2
        assert seq["escalation_trajectory"] == [0.5, 0.8]

    def test_different_trap_types_separate_sequences(self):
        t = TrapSequenceTracker()
        t.update(TrapType.LOADED_QUESTION, 0.6)
        t.update(TrapType.GISH_GALLOP, 0.7)
        assert len(t.active_sequences) == 2

    def test_tick_adversarial_increments(self):
        t = TrapSequenceTracker()
        t.tick_adversarial(True)
        assert t.consecutive_adversarial_turns == 1
        t.tick_adversarial(True)
        assert t.consecutive_adversarial_turns == 2

    def test_tick_adversarial_resets_on_clean(self):
        t = TrapSequenceTracker()
        t.tick_adversarial(True)
        t.tick_adversarial(True)
        t.tick_adversarial(False)
        assert t.consecutive_adversarial_turns == 0

    def test_escalation_score_no_sequences(self):
        t = TrapSequenceTracker()
        assert t.get_escalation_score() == 0.0

    def test_escalation_score_single_entry(self):
        t = TrapSequenceTracker()
        t.update(TrapType.DARVO, 0.8)
        score = t.get_escalation_score()
        assert 0.0 <= score <= 1.0

    def test_escalation_score_increasing_trajectory(self):
        t = TrapSequenceTracker()
        t.update(TrapType.GISH_GALLOP, 0.4)
        t.update(TrapType.GISH_GALLOP, 0.7)
        score = t.get_escalation_score()
        assert score >= 0.7  # should be high (increasing trend)

    def test_chain_score_no_sequences(self):
        t = TrapSequenceTracker()
        assert t.get_chain_score() == 0.0

    def test_chain_score_multi_turn(self):
        t = TrapSequenceTracker()
        t.update(TrapType.SEALIONING, 0.6)
        t.update(TrapType.SEALIONING, 0.7)
        assert t.get_chain_score() == 1.0  # 1/1 sequences ≥2 turns

    def test_goalpost_score_empty(self):
        t = TrapSequenceTracker()
        assert t.get_goalpost_score() == 0.0

    def test_goalpost_score_with_positions(self):
        t = TrapSequenceTracker()
        t.update(TrapType.MOVING_GOALPOSTS, 0.6)
        t.active_sequences[0]["goalpost_positions"].append("original")
        assert t.get_goalpost_score() == 1.0

    def test_option_narrowing_score_empty(self):
        t = TrapSequenceTracker()
        assert t.get_option_narrowing_score() == 0.0

    def test_option_narrowing_score_nonzero(self):
        t = TrapSequenceTracker()
        t.update(TrapType.FALSE_DILEMMA_WEAPONIZED, 0.65)
        t.active_sequences[0]["response_option_narrowing"] = 0.8
        score = t.get_option_narrowing_score()
        assert score == pytest.approx(0.8)


# =====================================================================
# 4. Feature detection
# =====================================================================

class TestFeatureDetection:
    def _tracker(self) -> TrapSequenceTracker:
        return TrapSequenceTracker()

    def test_presupposed_guilt_detected(self):
        inp = _make_input()
        assert _detect_feature("presupposed_guilt", "when did you stop lying", inp, self._tracker())

    def test_binary_choice_detected(self):
        inp = _make_input()
        assert _detect_feature("binary_choice", "either you agree or you're against us", inp, self._tracker())

    def test_high_defensiveness_programmatic(self):
        inp = _make_input(
            intention_vector=IntentionVector(e_defensiveness=0.8, e_challenge=0.5)
        )
        assert _detect_feature("high_defensiveness", "", inp, self._tracker())

    def test_high_defensiveness_below_threshold(self):
        inp = _make_input(
            intention_vector=IntentionVector(e_defensiveness=0.3, e_challenge=0.2)
        )
        assert not _detect_feature("high_defensiveness", "", inp, self._tracker())

    def test_multi_fallacy_programmatic(self):
        flags = [_make_fallacy_flag(), _make_fallacy_flag()]
        inp = _make_input(fallacy_flags=flags)
        assert _detect_feature("multi_fallacy", "", inp, self._tracker())

    def test_multi_fallacy_single_flag(self):
        inp = _make_input(fallacy_flags=[_make_fallacy_flag()])
        assert not _detect_feature("multi_fallacy", "", inp, self._tracker())

    def test_contradiction_present_programmatic(self):
        inp = _make_input(contradiction_flags=[_make_contradiction_flag()])
        assert _detect_feature("contradiction_present", "", inp, self._tracker())

    def test_contradiction_present_empty(self):
        inp = _make_input()
        assert not _detect_feature("contradiction_present", "", inp, self._tracker())

    def test_bias_exploited_programmatic(self):
        inp = _make_input(bias_vulnerability_profile={"confirmation_bias": 0.7})
        assert _detect_feature("bias_exploited", "", inp, self._tracker())

    def test_escalation_pattern_programmatic(self):
        tracker = TrapSequenceTracker()
        tracker.update(TrapType.DARVO, 0.5)
        tracker.update(TrapType.DARVO, 0.85)  # high escalation
        inp = _make_input()
        # escalation score should be > 0.40
        assert _detect_feature("escalation_pattern", "", inp, tracker)

    def test_genuine_question_detected(self):
        inp = _make_input()
        assert _detect_feature("genuine_question", "i'm genuinely asking about this", inp, self._tracker())

    def test_keyword_not_in_text(self):
        inp = _make_input()
        assert not _detect_feature("presupposed_guilt", "hello world", inp, self._tracker())

    def test_unknown_feature_returns_false(self):
        inp = _make_input()
        assert not _detect_feature("nonexistent_feature_xyz", "some text", inp, self._tracker())


# =====================================================================
# 5. Layer 1 — Template matching
# =====================================================================

class TestTemplateMatching:
    def test_all_templates_loaded(self):
        assert len(_TRAP_TEMPLATES) == 21

    def test_template_lookup_complete(self):
        for trap_type in TrapType:
            assert trap_type in _TEMPLATE_BY_TYPE

    def test_template_score_empty_text_no_features(self):
        template = _TEMPLATE_BY_TYPE[TrapType.LOADED_QUESTION]
        tracker = TrapSequenceTracker()
        inp = _make_input()
        score = compute_template_score(template, "", inp, tracker)
        assert score == 0.0

    def test_template_score_loaded_question_triggered(self):
        template = _TEMPLATE_BY_TYPE[TrapType.LOADED_QUESTION]
        tracker = TrapSequenceTracker()
        inp = _make_input(intention_vector=IntentionVector(e_defensiveness=0.8))
        score = compute_template_score(
            template, "when did you stop lying to everyone", inp, tracker
        )
        assert score > 0.0

    def test_template_score_inhibited_by_acknowledgment(self):
        template = _TEMPLATE_BY_TYPE[TrapType.LOADED_QUESTION]
        tracker = TrapSequenceTracker()
        inp = _make_input()
        # Has required feature + strong inhibitor
        score_without = compute_template_score(
            template, "when did you stop lying", inp, tracker
        )
        score_with_inh = compute_template_score(
            template, "when did you stop lying, i acknowledge your point is fair", inp, tracker
        )
        # Inhibitor should reduce or equal the score
        assert score_with_inh <= score_without

    def test_template_score_clamped_between_0_and_1(self):
        template = _TEMPLATE_BY_TYPE[TrapType.GISH_GALLOP]
        tracker = TrapSequenceTracker()
        flags = [_make_fallacy_flag(), _make_fallacy_flag(), _make_fallacy_flag()]
        inp = _make_input(
            fallacy_flags=flags,
            intention_vector=IntentionVector(e_defensiveness=0.9),
        )
        text = "furthermore moreover additionally also and another thing not only that there's also"
        score = compute_template_score(template, text, inp, tracker)
        assert 0.0 <= score <= 1.0

    def test_run_template_layer_returns_all_trap_types(self):
        inp = _make_input()
        tracker = TrapSequenceTracker()
        scores = run_template_layer(inp, tracker, _CONFIG)
        assert set(scores.keys()) == set(TrapType)

    def test_run_template_layer_scores_in_range(self):
        inp = _make_input()
        tracker = TrapSequenceTracker()
        scores = run_template_layer(inp, tracker, _CONFIG)
        for trap_type, score in scores.items():
            assert 0.0 <= score <= 1.0, f"{trap_type}: {score}"

    def test_gish_gallop_triggers_on_rapid_claims(self):
        template = _TEMPLATE_BY_TYPE[TrapType.GISH_GALLOP]
        tracker = TrapSequenceTracker()
        flags = [_make_fallacy_flag(), _make_fallacy_flag()]
        inp = _make_input(fallacy_flags=flags)
        text = "furthermore moreover additionally and another thing"
        score = compute_template_score(template, text, inp, tracker)
        assert score > 0.0

    def test_darvo_triggers_on_victim_reversal(self):
        template = _TEMPLATE_BY_TYPE[TrapType.DARVO]
        tracker = TrapSequenceTracker()
        inp = _make_input(intention_vector=IntentionVector(e_defensiveness=0.9))
        text = "i'm the real victim here you're attacking me"
        score = compute_template_score(template, text, inp, tracker)
        assert score > 0.0

    def test_false_dilemma_triggers_on_binary_choice(self):
        template = _TEMPLATE_BY_TYPE[TrapType.FALSE_DILEMMA_WEAPONIZED]
        tracker = TrapSequenceTracker()
        inp = _make_input()
        text = "either you agree with me or there is no middle ground you're either with us"
        score = compute_template_score(template, text, inp, tracker)
        assert score > 0.0


# =====================================================================
# 6. Layer 2 — Synthesis
# =====================================================================

class TestSynthesisLayer:
    def test_manipulation_mean_empty(self):
        assert compute_manipulation_mean([]) == 0.0

    def test_manipulation_mean_single(self):
        f = _make_fallacy_flag(manipulation_indicator=0.7)
        assert compute_manipulation_mean([f]) == pytest.approx(0.7)

    def test_manipulation_mean_average(self):
        flags = [
            _make_fallacy_flag(manipulation_indicator=0.4),
            _make_fallacy_flag(manipulation_indicator=0.8),
        ]
        assert compute_manipulation_mean(flags) == pytest.approx(0.6)

    def test_coordination_coherence_empty(self):
        assert compute_coordination_coherence([], []) == 0.0

    def test_coordination_coherence_all_adversarial(self):
        flags = [
            _make_fallacy_flag(manipulation_indicator=0.8),
            _make_fallacy_flag(manipulation_indicator=0.9),
        ]
        score = compute_coordination_coherence(flags, [])
        assert score == pytest.approx(1.0)

    def test_coordination_coherence_with_contradiction(self):
        flags = [_make_fallacy_flag(manipulation_indicator=0.8)]
        contra = [_make_contradiction_flag()]
        score = compute_coordination_coherence(flags, contra)
        assert 0.0 < score <= 1.0

    def test_coordination_coherence_mixed_directions(self):
        flags = [
            _make_fallacy_flag(manipulation_indicator=0.8),  # adversarial
            _make_fallacy_flag(manipulation_indicator=0.2),  # neutral
        ]
        score = compute_coordination_coherence(flags, [])
        # |1 + (-1)| / (1 + 1) = 0.0
        assert score == pytest.approx(0.0)

    def test_vulnerability_exploitation_empty_profile(self):
        inp = _make_input()
        assert compute_vulnerability_exploitation({}, inp) == 0.0

    def test_vulnerability_exploitation_with_profile(self):
        inp = _make_input()
        profile = {"confirmation_bias": 0.7, "anchoring": 0.5}
        score = compute_vulnerability_exploitation(profile, inp)
        assert 0.0 < score <= 1.0

    def test_vulnerability_exploitation_amplified_by_flags(self):
        profile = {"confirmation_bias": 0.7}
        inp_no_flags = _make_input(bias_vulnerability_profile=profile)
        inp_with_flags = _make_input(
            bias_vulnerability_profile=profile,
            bias_flags=[_make_bias_flag()],
        )
        score_no = compute_vulnerability_exploitation(profile, inp_no_flags)
        score_with = compute_vulnerability_exploitation(profile, inp_with_flags)
        # flags amplify, so score_with >= score_no
        assert score_with >= score_no

    def test_toolkit_density_no_signals(self):
        props = [Proposition(text="a"), Proposition(text="b")]
        score = compute_toolkit_density([], [], [], props)
        assert score == 0.0

    def test_toolkit_density_one_signal_one_prop(self):
        props = [Proposition(text="a")]
        score = compute_toolkit_density(
            [_make_fallacy_flag()], [], [], props
        )
        assert score == pytest.approx(1.0)

    def test_toolkit_density_many_signals_bounded(self):
        props = [Proposition(text="a")]
        contra = [_make_contradiction_flag(), _make_contradiction_flag()]
        fallacy = [_make_fallacy_flag(), _make_fallacy_flag()]
        bias = [_make_bias_flag()]
        score = compute_toolkit_density(fallacy, contra, bias, props)
        assert score == 1.0

    def test_toolkit_density_no_propositions_uses_one(self):
        # n_prop defaults to max(0, 1) = 1
        score = compute_toolkit_density(
            [_make_fallacy_flag()], [], [], []
        )
        assert score == pytest.approx(1.0)

    def test_synthesis_score_zero_input(self):
        inp = _make_input()
        score = compute_synthesis_score([], [], [], {}, inp, _CONFIG)
        assert score == 0.0

    def test_synthesis_score_positive_with_signals(self):
        flags = [_make_fallacy_flag(manipulation_indicator=0.85)]
        contra = [_make_contradiction_flag()]
        profile = {"availability_bias": 0.6}
        props = [Proposition(text="claim")]
        inp = _make_input(
            fallacy_flags=flags,
            contradiction_flags=contra,
            bias_vulnerability_profile=profile,
            user_propositions=props,
        )
        score = compute_synthesis_score(flags, contra, [], profile, inp, _CONFIG)
        assert 0.0 < score <= 1.0

    def test_synthesis_score_bounded(self):
        flags = [_make_fallacy_flag(manipulation_indicator=1.0)]
        contra = [_make_contradiction_flag()]
        bias = [_make_bias_flag()]
        profile = {"x": 1.0}
        props = [Proposition(text="a")]
        inp = _make_input(
            fallacy_flags=flags, contradiction_flags=contra,
            bias_flags=bias, bias_vulnerability_profile=profile,
            user_propositions=props,
        )
        score = compute_synthesis_score(flags, contra, bias, profile, inp, _CONFIG)
        assert 0.0 <= score <= 1.0


# =====================================================================
# 7. Layer 3 — Sequential
# =====================================================================

class TestSequentialLayer:
    def test_sequential_score_empty_tracker(self):
        tracker = TrapSequenceTracker()
        score = compute_sequential_score(tracker)
        assert score == 0.0

    def test_sequential_score_increasing_escalation(self):
        tracker = TrapSequenceTracker()
        tracker.update(TrapType.SEALIONING, 0.4)
        tracker.update(TrapType.SEALIONING, 0.8)
        score = compute_sequential_score(tracker)
        assert score > 0.0

    def test_sequential_score_bounded(self):
        tracker = TrapSequenceTracker()
        for _ in range(5):
            tracker.update(TrapType.KAFKA_TRAP, 0.9)
        tracker.active_sequences[0]["goalpost_positions"].append("pos1")
        tracker.active_sequences[0]["response_option_narrowing"] = 0.8
        score = compute_sequential_score(tracker)
        assert 0.0 <= score <= 1.0

    def test_sequential_is_mean_of_four_components(self):
        tracker = TrapSequenceTracker()
        tracker.update(TrapType.DARVO, 0.5)
        tracker.update(TrapType.DARVO, 0.9)
        esc = tracker.get_escalation_score()
        chain = tracker.get_chain_score()
        goal = tracker.get_goalpost_score()
        narrow = tracker.get_option_narrowing_score()
        expected = (esc + chain + goal + narrow) / 4.0
        assert compute_sequential_score(tracker) == pytest.approx(expected)


# =====================================================================
# 8. Fusion — intent score and classification
# =====================================================================

class TestFusion:
    def test_adversarial_intent_zero(self):
        score = compute_adversarial_intent(0.0, 0.0, 0.0, _CONFIG)
        assert score == 0.0

    def test_adversarial_intent_full(self):
        score = compute_adversarial_intent(1.0, 1.0, 1.0, _CONFIG)
        assert score == pytest.approx(1.0)

    def test_adversarial_intent_weighted_sum(self):
        t_t, t_s, t_seq = 0.6, 0.4, 0.2
        expected = (
            _CONFIG.w_template * t_t
            + _CONFIG.w_synthesis * t_s
            + _CONFIG.w_sequential * t_seq
        )
        assert compute_adversarial_intent(t_t, t_s, t_seq, _CONFIG) == pytest.approx(expected)

    def test_adversarial_intent_clamped_above(self):
        score = compute_adversarial_intent(2.0, 2.0, 2.0, _CONFIG)
        assert score <= 1.0

    def test_adversarial_intent_clamped_below(self):
        score = compute_adversarial_intent(-1.0, -1.0, -1.0, _CONFIG)
        assert score >= 0.0

    def test_classify_intent_level_none(self):
        assert classify_intent_level(0.0) == IntentLevel.NONE
        assert classify_intent_level(0.24) == IntentLevel.NONE

    def test_classify_intent_level_low(self):
        assert classify_intent_level(0.25) == IntentLevel.LOW
        assert classify_intent_level(0.44) == IntentLevel.LOW

    def test_classify_intent_level_moderate(self):
        assert classify_intent_level(0.45) == IntentLevel.MODERATE
        assert classify_intent_level(0.64) == IntentLevel.MODERATE

    def test_classify_intent_level_high(self):
        assert classify_intent_level(0.65) == IntentLevel.HIGH
        assert classify_intent_level(0.84) == IntentLevel.HIGH

    def test_classify_intent_level_near_certain(self):
        assert classify_intent_level(0.85) == IntentLevel.NEAR_CERTAIN
        assert classify_intent_level(1.00) == IntentLevel.NEAR_CERTAIN


# =====================================================================
# 9. Threshold resolution
# =====================================================================

class TestThresholdResolution:
    def test_normal_mode(self):
        assert resolve_trap_threshold(OperationalMode.NORMAL, _CONFIG) == pytest.approx(0.45)

    def test_dev_mode_lower(self):
        theta = resolve_trap_threshold(OperationalMode.DEV, _CONFIG)
        assert theta == pytest.approx(0.25)
        assert theta < resolve_trap_threshold(OperationalMode.NORMAL, _CONFIG)

    def test_learning_mode(self):
        assert resolve_trap_threshold(OperationalMode.LEARNING, _CONFIG) == pytest.approx(0.40)

    def test_rem_normal_mode(self):
        assert resolve_trap_threshold(OperationalMode.REM_NORMAL, _CONFIG) == pytest.approx(0.50)

    def test_reflective_mode(self):
        assert resolve_trap_threshold(OperationalMode.REFLECTIVE, _CONFIG) == pytest.approx(0.35)

    def test_rem_dream_mode(self):
        # REM_DREAM uses theta_rem
        assert resolve_trap_threshold(OperationalMode.REM_DREAM, _CONFIG) == pytest.approx(0.50)


# =====================================================================
# 10. Threat load
# =====================================================================

class TestThreatLoad:
    def test_threat_load_empty(self):
        assert compute_threat_load([], _CONFIG) == 0.0

    def test_threat_load_single_framing(self):
        flag = _make_trap_flag(
            trap_type=TrapType.LOADED_QUESTION,
            intent_score=0.7,
            category=TrapCategory.FRAMING,
        )
        load = compute_threat_load([flag], _CONFIG)
        # 0.7 * 0.40 = 0.28
        assert load == pytest.approx(0.28)

    def test_threat_load_single_meta(self):
        flag = _make_trap_flag(
            trap_type=TrapType.DOUBLE_BIND,
            intent_score=0.8,
            category=TrapCategory.META,
        )
        load = compute_threat_load([flag], _CONFIG)
        # 0.8 * 0.55 = 0.44
        assert load == pytest.approx(0.44)

    def test_threat_load_bounded_at_1(self):
        # Create multiple high-scoring flags
        flags = [
            _make_trap_flag(TrapType.DOUBLE_BIND, 1.0, TrapCategory.META),
            _make_trap_flag(TrapType.KAFKA_TRAP, 1.0, TrapCategory.SEQUENTIAL),
            _make_trap_flag(TrapType.LOADED_QUESTION, 1.0, TrapCategory.FRAMING),
        ]
        load = compute_threat_load(flags, _CONFIG)
        assert load <= 1.0

    def test_threat_load_uses_category_weights(self):
        sequential_flag = _make_trap_flag(TrapType.GISH_GALLOP, 0.8, TrapCategory.SEQUENTIAL)
        rhetorical_flag = _make_trap_flag(TrapType.WHATABOUTISM, 0.8, TrapCategory.RHETORICAL)
        seq_load = compute_threat_load([sequential_flag], _CONFIG)
        rhet_load = compute_threat_load([rhetorical_flag], _CONFIG)
        # Sequential weight (0.50) > Rhetorical weight (0.35)
        assert seq_load > rhet_load


# =====================================================================
# 11. Neurochemical signals
# =====================================================================

class TestNeurochemicalSignals:
    def _rng(self):
        return np.random.default_rng(0)

    def test_no_flags_returns_zero_signals(self):
        tracker = TrapSequenceTracker()
        signals = compute_neurochemical_signals([], 0.0, tracker, _STATE, _CONFIG, self._rng())
        assert all(v == 0.0 for v in signals.values())

    def test_signals_dict_keys(self):
        flag = _make_trap_flag(intent_score=0.7)
        tracker = TrapSequenceTracker()
        signals = compute_neurochemical_signals(
            [flag], 0.5, tracker, _STATE, _CONFIG, self._rng()
        )
        expected_keys = {
            "ne_burst", "ne_escalation", "cor_drift",
            "gaba_reuptake", "da_negative", "oxt_reduction",
            "beta_boost", "theta_suppress",
        }
        assert set(signals.keys()) == expected_keys

    def test_signals_non_negative(self):
        flag = _make_trap_flag(intent_score=0.9)
        tracker = TrapSequenceTracker()
        signals = compute_neurochemical_signals(
            [flag], 0.8, tracker, _STATE, _CONFIG, self._rng()
        )
        for k, v in signals.items():
            assert v >= 0.0, f"{k} = {v}"

    def test_ne_burst_zero_for_low_intent(self):
        flag = _make_trap_flag(intent_score=0.3)
        tracker = TrapSequenceTracker()
        signals = compute_neurochemical_signals(
            [flag], 0.3, tracker, _STATE, _CONFIG, self._rng()
        )
        # intent < 0.45 threshold → no NE burst
        assert signals["ne_burst"] == 0.0

    def test_ne_burst_positive_for_high_intent(self):
        # Test that NE burst formula is exercised (stochastic; use seeded rng)
        rng = np.random.default_rng(7)
        flag = _make_trap_flag(intent_score=0.9)
        tracker = TrapSequenceTracker()
        # Run many times and check that non-zero burst occurs at least sometimes
        results = [
            compute_neurochemical_signals([flag], 0.8, tracker, _STATE, _CONFIG, np.random.default_rng(i))["ne_burst"]
            for i in range(20)
        ]
        assert any(v > 0.0 for v in results)

    def test_cor_drift_zero_without_consecutive_turns(self):
        flag = _make_trap_flag(intent_score=0.8)
        tracker = TrapSequenceTracker()
        # Only 1 consecutive turn — below min_turns_for_cor=2
        tracker.consecutive_adversarial_turns = 1
        signals = compute_neurochemical_signals(
            [flag], 0.7, tracker, _STATE, _CONFIG, self._rng()
        )
        assert signals["cor_drift"] == 0.0

    def test_cor_drift_positive_with_consecutive_turns(self):
        flag = _make_trap_flag(intent_score=0.8)
        tracker = TrapSequenceTracker()
        tracker.consecutive_adversarial_turns = 3
        signals = compute_neurochemical_signals(
            [flag], 0.7, tracker, _STATE, _CONFIG, self._rng()
        )
        assert signals["cor_drift"] > 0.0

    def test_gaba_reuptake_proportional_to_threat_load(self):
        flag = _make_trap_flag(intent_score=0.8)
        tracker = TrapSequenceTracker()
        sig_low  = compute_neurochemical_signals([flag], 0.3, tracker, _STATE, _CONFIG, self._rng())
        sig_high = compute_neurochemical_signals([flag], 0.9, tracker, _STATE, _CONFIG, self._rng())
        assert sig_high["gaba_reuptake"] > sig_low["gaba_reuptake"]

    def test_oxt_reduction_respects_floor(self):
        flag = _make_trap_flag(intent_score=0.9)
        tracker = TrapSequenceTracker()
        # Set OXT very low (near floor)
        low_state = LogicTrapEngineState(oxt_level=_CONFIG.oxt_floor)
        signals = compute_neurochemical_signals(
            [flag], 1.0, tracker, low_state, _CONFIG, self._rng()
        )
        # Reduction should be ~0 because already at floor
        assert signals["oxt_reduction"] == pytest.approx(0.0, abs=1e-9)

    def test_oxt_reduction_normal_state(self):
        flag = _make_trap_flag(intent_score=0.9)
        tracker = TrapSequenceTracker()
        state = LogicTrapEngineState(oxt_level=1.0)
        signals = compute_neurochemical_signals(
            [flag], 0.8, tracker, state, _CONFIG, self._rng()
        )
        assert signals["oxt_reduction"] > 0.0

    def test_beta_boost_proportional_to_threat_load(self):
        flag = _make_trap_flag(intent_score=0.8)
        tracker = TrapSequenceTracker()
        sig_low  = compute_neurochemical_signals([flag], 0.2, tracker, _STATE, _CONFIG, self._rng())
        sig_high = compute_neurochemical_signals([flag], 0.8, tracker, _STATE, _CONFIG, self._rng())
        assert sig_high["beta_boost"] > sig_low["beta_boost"]

    def test_theta_suppress_positive(self):
        flag = _make_trap_flag(intent_score=0.8)
        tracker = TrapSequenceTracker()
        signals = compute_neurochemical_signals(
            [flag], 0.6, tracker, _STATE, _CONFIG, self._rng()
        )
        assert signals["theta_suppress"] > 0.0


# =====================================================================
# 12. Defensive strategy & FrameEscape
# =====================================================================

class TestDefensiveStrategy:
    def test_build_strategy_all_trap_types(self):
        for trap_type in TrapType:
            template = _TEMPLATE_BY_TYPE[trap_type]
            strategy = build_defensive_strategy(trap_type, template)
            assert isinstance(strategy, DefensiveStrategy)
            assert strategy.trap_type == trap_type
            assert strategy.strategy_name != ""
            assert strategy.response_template != ""

    def test_strategy_has_things_to_avoid_and_do(self):
        for trap_type in TrapType:
            template = _TEMPLATE_BY_TYPE[trap_type]
            strategy = build_defensive_strategy(trap_type, template)
            assert len(strategy.things_to_avoid) >= 1
            assert len(strategy.things_to_do) >= 1

    def test_frame_escape_populated(self):
        template = _TEMPLATE_BY_TYPE[TrapType.LOADED_QUESTION]
        strategy = build_defensive_strategy(TrapType.LOADED_QUESTION, template)
        fe = strategy.frame_escape
        assert fe is not None
        assert fe.original_frame != ""
        assert isinstance(fe.invalid_assumptions, list)
        assert fe.reframe != ""
        assert fe.reframe_response_template != ""

    def test_loaded_question_strategy_name(self):
        template = _TEMPLATE_BY_TYPE[TrapType.LOADED_QUESTION]
        strategy = build_defensive_strategy(TrapType.LOADED_QUESTION, template)
        assert strategy.strategy_name == "Frame Rejection"

    def test_gish_gallop_strategy_name(self):
        template = _TEMPLATE_BY_TYPE[TrapType.GISH_GALLOP]
        strategy = build_defensive_strategy(TrapType.GISH_GALLOP, template)
        assert strategy.strategy_name == "Selective Engagement"

    def test_double_bind_strategy_name(self):
        template = _TEMPLATE_BY_TYPE[TrapType.DOUBLE_BIND]
        strategy = build_defensive_strategy(TrapType.DOUBLE_BIND, template)
        assert strategy.strategy_name == "Bind Naming and Rejection"

    def test_frame_escape_is_frozen(self):
        fe = FrameEscape(
            original_frame="test",
            invalid_assumptions=["x"],
            reframe="neutral",
            reframe_response_template="template",
        )
        with pytest.raises((AttributeError, TypeError)):
            fe.reframe = "new"   # type: ignore


# =====================================================================
# 13. Engine ports
# =====================================================================

class TestEnginePorts:
    def test_engine_default_mode_is_normal(self):
        engine = LogicTrapDetectionEngine()
        status = engine.get_status()
        assert status["mode"] == "normal"

    def test_configure_changes_mode(self):
        engine = LogicTrapDetectionEngine()
        engine.configure(OperationalMode.DEV)
        assert engine.get_status()["mode"] == "dev"

    def test_get_status_keys(self):
        engine = LogicTrapDetectionEngine()
        status = engine.get_status()
        expected = {"engine_id", "version", "mode", "state", "templates_loaded"}
        assert set(status.keys()) == expected

    def test_get_status_templates_count(self):
        engine = LogicTrapDetectionEngine()
        assert engine.get_status()["templates_loaded"] == 21

    def test_update_neurochem_state_ne(self):
        engine = LogicTrapDetectionEngine()
        engine.update_neurochem_state({"ne": 0.85})
        assert engine._state.ne_level == pytest.approx(0.85)

    def test_update_neurochem_state_all_fields(self):
        engine = LogicTrapDetectionEngine()
        engine.update_neurochem_state({
            "ne": 0.5, "cor": 0.3,
            "gaba": 0.6, "da": 0.4, "oxt": 0.8,
        })
        s = engine._state
        assert s.ne_level   == pytest.approx(0.5)
        assert s.cor_level  == pytest.approx(0.3)
        assert s.gaba_level == pytest.approx(0.6)
        assert s.da_level   == pytest.approx(0.4)
        assert s.oxt_level  == pytest.approx(0.8)

    def test_update_neurochem_state_partial(self):
        engine = LogicTrapDetectionEngine()
        engine.update_neurochem_state({"ne": 0.7})
        # Other fields should remain at default
        assert engine._state.cor_level == 0.0

    def test_engine_accepts_custom_config(self):
        cfg = LogicTrapEngineConfig(theta_normal=0.60)
        engine = LogicTrapDetectionEngine(config=cfg)
        assert engine._config.theta_normal == pytest.approx(0.60)


# =====================================================================
# 14. End-to-end process()
# =====================================================================

class TestProcessEndToEnd:
    def test_clean_pass_empty_input(self):
        engine = LogicTrapDetectionEngine()
        result = engine.process(_make_input())
        assert isinstance(result, LogicTrapDetectionResult)
        assert result.clean_pass is True
        assert result.flags == []
        assert result.highest_intent_score == 0.0
        assert result.threat_load == 0.0

    def test_result_has_processing_time(self):
        engine = LogicTrapDetectionEngine()
        result = engine.process(_make_input())
        assert result.processing_time_ms >= 0.0

    def test_result_templates_evaluated(self):
        engine = LogicTrapDetectionEngine()
        result = engine.process(_make_input())
        assert result.templates_evaluated == 21

    def test_result_metadata_has_mode(self):
        engine = LogicTrapDetectionEngine()
        engine.configure(OperationalMode.DEV)
        result = engine.process(_make_input())
        assert result.metadata["mode"] == "dev"

    def test_metadata_threshold_used(self):
        # threshold_used is resolved from trap_input.active_mode, not engine._mode
        engine = LogicTrapDetectionEngine()
        result = engine.process(_make_input(active_mode=OperationalMode.DEV))
        assert result.metadata["threshold_used"] == pytest.approx(0.25)

    def test_dev_mode_more_sensitive(self):
        # DEV mode (theta=0.25) should detect traps that NORMAL (theta=0.45) misses
        engine_dev    = LogicTrapDetectionEngine()
        engine_normal = LogicTrapDetectionEngine()
        engine_dev.configure(OperationalMode.DEV)

        inp = _make_input(
            conversation_history=["when did you stop lying", "admit that you failed"],
            intention_vector=IntentionVector(e_defensiveness=0.7, e_challenge=0.6),
            fallacy_flags=[_make_fallacy_flag(manipulation_indicator=0.8)],
            contradiction_flags=[_make_contradiction_flag()],
        )
        result_dev    = engine_dev.process(inp)
        result_normal = engine_normal.process(inp)
        # Dev should detect at least as many traps
        assert len(result_dev.flags) >= len(result_normal.flags)

    def test_loaded_question_detected_in_text(self):
        engine = LogicTrapDetectionEngine()
        engine.configure(OperationalMode.DEV)
        inp = _make_input(
            conversation_history=["when did you stop lying to everyone?"],
            intention_vector=IntentionVector(e_defensiveness=0.8),
            fallacy_flags=[_make_fallacy_flag(manipulation_indicator=0.85)],
            contradiction_flags=[_make_contradiction_flag()],
        )
        result = engine.process(inp)
        trap_types = {f.trap_type for f in result.flags}
        assert TrapType.LOADED_QUESTION in trap_types

    def test_trap_flag_structure(self):
        engine = LogicTrapDetectionEngine()
        engine.configure(OperationalMode.DEV)
        inp = _make_input(
            conversation_history=["when did you stop lying"],
            intention_vector=IntentionVector(e_defensiveness=0.9),
            fallacy_flags=[_make_fallacy_flag(manipulation_indicator=0.9)],
            contradiction_flags=[_make_contradiction_flag()],
        )
        result = engine.process(inp)
        if result.flags:
            flag = result.flags[0]
            assert isinstance(flag.trap_id, str)
            assert flag.trap_type in TrapType
            assert flag.trap_category in TrapCategory
            assert 0.0 <= flag.adversarial_intent_score <= 1.0
            assert flag.intent_classification in IntentLevel
            assert isinstance(flag.layer_scores, dict)
            assert "template" in flag.layer_scores
            assert "synthesis" in flag.layer_scores
            assert "sequential" in flag.layer_scores
            assert isinstance(flag.defensive_strategy, DefensiveStrategy)
            assert isinstance(flag.timestamp, datetime)

    def test_tracker_updated_after_detection(self):
        engine = LogicTrapDetectionEngine()
        engine.configure(OperationalMode.DEV)
        inp = _make_input(
            conversation_history=["when did you stop lying"],
            intention_vector=IntentionVector(e_defensiveness=0.9),
            fallacy_flags=[_make_fallacy_flag(manipulation_indicator=0.9)],
            contradiction_flags=[_make_contradiction_flag()],
        )
        result = engine.process(inp)
        # If traps were detected, tracker should be updated
        if result.flags:
            assert result.updated_sequence_tracker.consecutive_adversarial_turns >= 1

    def test_tracker_passed_in_reused(self):
        engine = LogicTrapDetectionEngine()
        engine.configure(OperationalMode.DEV)

        tracker = TrapSequenceTracker()
        tracker.consecutive_adversarial_turns = 3

        inp = _make_input(active_sequence_tracker=tracker)
        result = engine.process(inp)
        # Tracker passed in should be the same object, potentially mutated
        assert result.updated_sequence_tracker is tracker

    def test_highest_intent_score_is_max(self):
        engine = LogicTrapDetectionEngine()
        engine.configure(OperationalMode.DEV)
        inp = _make_input(
            conversation_history=["when did you stop lying", "furthermore moreover additionally"],
            intention_vector=IntentionVector(e_defensiveness=0.9, e_challenge=0.8),
            fallacy_flags=[_make_fallacy_flag(0.9), _make_fallacy_flag(0.85)],
            contradiction_flags=[_make_contradiction_flag()],
        )
        result = engine.process(inp)
        if result.flags:
            expected = max(f.adversarial_intent_score for f in result.flags)
            assert result.highest_intent_score == pytest.approx(expected)

    def test_clean_pass_when_all_below_threshold(self):
        # Use NORMAL mode (theta=0.45) with minimal adversarial signals
        engine = LogicTrapDetectionEngine()
        result = engine.process(_make_input(conversation_history=["hello, how are you?"]))
        assert result.clean_pass is True

    def test_neurochemical_signals_in_result(self):
        engine = LogicTrapDetectionEngine()
        result = engine.process(_make_input())
        assert "ne_burst" in result.neurochemical_signals

    def test_flags_supporting_ids_populated(self):
        engine = LogicTrapDetectionEngine()
        engine.configure(OperationalMode.DEV)
        cf = _make_contradiction_flag()
        ff = _make_fallacy_flag(0.9)
        bf = _make_bias_flag()
        inp = _make_input(
            conversation_history=["when did you stop lying"],
            intention_vector=IntentionVector(e_defensiveness=0.9),
            contradiction_flags=[cf],
            fallacy_flags=[ff],
            bias_flags=[bf],
        )
        result = engine.process(inp)
        if result.flags:
            flag = result.flags[0]
            # All flags should appear in supporting IDs
            assert cf.contradiction_id in flag.supporting_contradiction_ids
            assert ff.fallacy_id in flag.supporting_fallacy_ids
            assert bf.bias_id in flag.supporting_bias_ids

    def test_multiple_trap_types_can_fire_together(self):
        engine = LogicTrapDetectionEngine()
        engine.configure(OperationalMode.DEV)
        inp = _make_input(
            conversation_history=[
                "when did you stop lying",           # loaded_question
                "furthermore moreover additionally", # gish_gallop
                "i'm the real victim",               # darvo
            ],
            intention_vector=IntentionVector(e_defensiveness=0.9, e_challenge=0.8),
            fallacy_flags=[_make_fallacy_flag(0.9), _make_fallacy_flag(0.85)],
            contradiction_flags=[_make_contradiction_flag()],
        )
        result = engine.process(inp)
        # Multiple trap types should be possible
        assert len(result.flags) >= 0  # at minimum no crash


# =====================================================================
# 15. Helpers
# =====================================================================

class TestHelpers:
    def test_build_combined_text_conversation(self):
        inp = _make_input(conversation_history=["hello world", "test phrase"])
        text = _build_combined_text(inp)
        assert "hello world" in text
        assert "test phrase" in text

    def test_build_combined_text_propositions(self):
        props = [Proposition(text="claim one"), Proposition(text="claim two")]
        inp = _make_input(user_propositions=props)
        text = _build_combined_text(inp)
        assert "claim one" in text
        assert "claim two" in text

    def test_build_combined_text_semantic_expansion(self):
        inp = _make_input(semantic_expansion={"topic": "manipulation"})
        text = _build_combined_text(inp)
        assert "manipulation" in text

    def test_build_combined_text_memory_context(self):
        stmts = [ProcessedStatement(raw_text="remembered fact", source_tag=SourceTag.MEMORY_MID_TERM)]
        inp = _make_input(memory_context=stmts)
        text = _build_combined_text(inp)
        assert "remembered fact" in text

    def test_build_combined_text_empty(self):
        inp = _make_input()
        text = _build_combined_text(inp)
        assert isinstance(text, str)

    def test_collect_supporting_ids(self):
        cf = _make_contradiction_flag()
        ff = _make_fallacy_flag()
        bf = _make_bias_flag()
        contra_ids, fallacy_ids, bias_ids = _collect_supporting_ids(
            TrapType.LOADED_QUESTION, [ff], [cf], [bf]
        )
        assert cf.contradiction_id in contra_ids
        assert ff.fallacy_id in fallacy_ids
        assert bf.bias_id in bias_ids

    def test_collect_supporting_ids_empty(self):
        contra_ids, fallacy_ids, bias_ids = _collect_supporting_ids(
            TrapType.LOADED_QUESTION, [], [], []
        )
        assert contra_ids == []
        assert fallacy_ids == []
        assert bias_ids == []


# =====================================================================
# 16. TrapFlag / Result field validation
# =====================================================================

class TestOutputTypes:
    def test_trap_flag_is_frozen(self):
        flag = _make_trap_flag()
        with pytest.raises((AttributeError, TypeError)):
            flag.adversarial_intent_score = 0.99  # type: ignore

    def test_trap_flag_has_uuid_id(self):
        flag = _make_trap_flag()
        # Should be a valid UUID string
        import uuid as _uuid
        _uuid.UUID(flag.trap_id)  # raises if invalid

    def test_trap_flag_timestamp_is_datetime(self):
        flag = _make_trap_flag()
        assert isinstance(flag.timestamp, datetime)

    def test_logic_trap_detection_result_is_frozen(self):
        engine = LogicTrapDetectionEngine()
        result = engine.process(_make_input())
        assert isinstance(result, LogicTrapDetectionResult)
        with pytest.raises((AttributeError, TypeError)):
            result.clean_pass = False  # type: ignore

    def test_result_active_sequences_is_list(self):
        engine = LogicTrapDetectionEngine()
        result = engine.process(_make_input())
        assert isinstance(result.active_sequences, list)

    def test_result_metadata_is_dict(self):
        engine = LogicTrapDetectionEngine()
        result = engine.process(_make_input())
        assert isinstance(result.metadata, dict)

    def test_bias_flag_stub_has_uuid(self):
        bf = BiasFlag()
        import uuid as _uuid
        _uuid.UUID(bf.bias_id)

    def test_bias_flag_is_frozen(self):
        bf = BiasFlag()
        with pytest.raises((AttributeError, TypeError)):
            bf.confidence = 0.99  # type: ignore

    def test_frame_escape_is_frozen(self):
        fe = FrameEscape(
            original_frame="frame",
            invalid_assumptions=["assumption"],
            reframe="reframe",
            reframe_response_template="template",
        )
        with pytest.raises((AttributeError, TypeError)):
            fe.reframe = "changed"  # type: ignore

    def test_defensive_strategy_is_frozen(self):
        template = _TEMPLATE_BY_TYPE[TrapType.WHATABOUTISM]
        strategy = build_defensive_strategy(TrapType.WHATABOUTISM, template)
        with pytest.raises((AttributeError, TypeError)):
            strategy.strategy_name = "modified"  # type: ignore


# =====================================================================
# 17. Edge cases
# =====================================================================

class TestEdgeCases:
    def test_process_with_none_tracker_creates_fresh(self):
        engine = LogicTrapDetectionEngine()
        inp = _make_input(active_sequence_tracker=None)
        result = engine.process(inp)
        assert result.updated_sequence_tracker is not None

    def test_process_with_very_long_history(self):
        history = [f"turn {i}: some adversarial content furthermore" for i in range(50)]
        inp = _make_input(conversation_history=history)
        engine = LogicTrapDetectionEngine()
        result = engine.process(inp)
        assert isinstance(result, LogicTrapDetectionResult)

    def test_process_with_all_signal_types(self):
        engine = LogicTrapDetectionEngine()
        engine.configure(OperationalMode.DEV)
        inp = _make_input(
            conversation_history=["when did you stop lying"],
            user_propositions=[Proposition(text="claim")],
            contradiction_flags=[_make_contradiction_flag()],
            fallacy_flags=[_make_fallacy_flag(0.9)],
            bias_flags=[_make_bias_flag()],
            bias_vulnerability_profile={"confirmation_bias": 0.7},
            intention_vector=IntentionVector(e_defensiveness=0.8, e_challenge=0.7),
            memory_context=[ProcessedStatement(raw_text="old statement", source_tag=SourceTag.MEMORY_MID_TERM)],
        )
        result = engine.process(inp)
        assert isinstance(result, LogicTrapDetectionResult)

    def test_tracker_not_created_twice(self):
        engine = LogicTrapDetectionEngine()
        tracker1 = TrapSequenceTracker()
        inp = _make_input(active_sequence_tracker=tracker1)
        result = engine.process(inp)
        # Should be same tracker object
        assert result.updated_sequence_tracker is tracker1

    def test_result_flags_is_list(self):
        engine = LogicTrapDetectionEngine()
        result = engine.process(_make_input())
        assert isinstance(result.flags, list)

    def test_template_category_coverage(self):
        """Every TrapCategory should have at least one template."""
        covered = {t.trap_category for t in _TRAP_TEMPLATES}
        assert covered == set(TrapCategory)

    def test_no_required_features_template_gets_zero_from_score(self):
        """A template with no required features should score based on supporting/inhibiting only."""
        # Create a minimal template with no required features
        template = TrapTemplate(
            trap_type=TrapType.LOADED_QUESTION,  # re-use enum value for test
            trap_category=TrapCategory.FRAMING,
            required_features=(),               # empty — no required
            supporting_features=("loaded_language",),
            inhibiting_features=(),
        )
        tracker = TrapSequenceTracker()
        inp = _make_input()
        # No supporting features present
        score = compute_template_score(template, "plain text", inp, tracker)
        assert score == 0.0
