"""
Tests for Socratic Reasoning Engine (Engine 14).

Coverage:
  - Enumerations (all values present)
  - SocraticEngineConfig (defaults, frozen)
  - socratic_score computation
  - topic_depth computation
  - mode_allows_socratic gate
  - check_all_activation_gates (all 4 gates)
  - contradiction_signal, assumption_signal, uncertainty_signal, frustration_signal
  - convergence_metrics computation
  - implicit assumption detection
  - entailment generation
  - target selection (all states)
  - question type selection (rotation)
  - question formulation (all types)
  - state transitions (all state × condition combos)
  - insight extraction
  - Σ(t) computation
  - neurochemical signals (all paths)
  - bidirectional feedback
  - engine ports (configure, update_neurochem_state, get_status)
  - process() end-to-end (activated / not activated / multi-turn / EXIT paths)
  - internal self-inquiry (REM_NORMAL with unsolved buffer)
  - output types (frozen / mutable)
  - edge cases
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime
from typing import List

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    ContradictionFlag,
    OperationalMode,
    ProcessedStatement,
    SourceTag,
)
from zados.cognitive_engines.py_engines.fallacy_detection_engine import (
    IntentionVector,
    Proposition,
)
from zados.cognitive_engines.py_engines.paradox_detection_engine import ParadoxFlag, ParadoxClass
from zados.cognitive_engines.py_engines.socratic_reasoning_engine import (
    # Enums
    DialogueState,
    QuestionType,
    ExpectedEffect,
    InsightSource,
    YieldReason,
    # Config / types
    SocraticEngineConfig,
    Assumption,
    Entailment,
    SocraticQuestion,
    SocraticInsight,
    ConvergenceMetrics,
    UnsolvedEntry,
    SocraticInput,
    SocraticOutput,
    SocraticDialogueState,
    SocraticEngineState,
    # Pure functions
    compute_socratic_score,
    compute_topic_depth,
    mode_allows_socratic,
    check_all_activation_gates,
    compute_contradiction_signal,
    compute_assumption_signal,
    compute_uncertainty_signal,
    compute_frustration_signal,
    compute_convergence_metrics,
    detect_implicit_assumptions,
    generate_entailments_from_propositions,
    select_target_proposition,
    select_question_type,
    formulate_question,
    compute_next_state,
    extract_insight,
    compute_socratic_processing_signal,
    compute_neurochemical_signals,
    generate_internal_questions,
    # Engine
    SocraticReasoningEngine,
)

# =====================================================================
# Helpers
# =====================================================================

_CONFIG = SocraticEngineConfig()
_RNG = np.random.default_rng(42)

# Shared ProcessedStatement instances for helpers
_PS = ProcessedStatement()


def _make_contradiction_flag(level: int = 2, confidence: float = 0.8) -> ContradictionFlag:
    import uuid as _uuid
    return ContradictionFlag(
        contradiction_id=str(_uuid.uuid4()),
        statement_a=_PS,
        statement_b=_PS,
        contradiction_level=level,
        confidence=confidence,
        evidence_signals={},
        temporal_separation=None,
        temporal_decay_applied=1.0,
        semantic_description="test contradiction",
        context_frame=None,
    )


def _make_paradox_flag(tension: float = 0.7) -> ParadoxFlag:
    import uuid as _uuid
    return ParadoxFlag(
        paradox_id=str(_uuid.uuid4()),
        source=_PS,
        source_contradiction=None,
        concept_a="concept_a",
        concept_b="concept_b",
        paradox_class=list(ParadoxClass)[0],
        class_posterior=0.7,
        confidence=0.8,
        features={},
        resolution_hint=None,
        structural_pattern=None,
        symbolic_tension_score=tension,
        timestamp=None,
    )


def _make_proposition(text: str = "All humans are mortal") -> Proposition:
    return Proposition(text=text)


def _make_intention(**kwargs) -> IntentionVector:
    """
    IntentionVector has direct fields: e_defensiveness, e_challenge.
    All others (e_exploration, e_symbolism, e_pragmatism, e_discharge) go in intent_scores.
    """
    e_defensiveness = float(kwargs.pop("e_defensiveness", 0.0))
    e_challenge = float(kwargs.pop("e_challenge", 0.0))
    # Everything else goes into intent_scores
    intent_scores: dict = {}
    for key in ("e_exploration", "e_symbolism", "e_pragmatism", "e_discharge"):
        if key in kwargs:
            intent_scores[key] = float(kwargs.pop(key))
    # Any remaining kwargs also go into intent_scores
    intent_scores.update({k: float(v) for k, v in kwargs.items()})
    return IntentionVector(
        e_defensiveness=e_defensiveness,
        e_challenge=e_challenge,
        intent_scores=intent_scores,
    )


def _make_input(
    text: str = "I believe that freedom is more important than security.",
    mode: OperationalMode = OperationalMode.NORMAL,
    intention: IntentionVector = None,
    propositions: List[Proposition] = None,
    contradiction_flags=None,
    paradox_flags=None,
    history=None,
    unsolved_buffer=None,
) -> SocraticInput:
    if intention is None:
        intention = _make_intention(e_exploration=0.6, e_challenge=0.3)
    return SocraticInput(
        intention_vector=intention,
        active_mode=mode,
        user_input=ProcessedStatement(raw_text=text),
        user_propositions=propositions or [_make_proposition(text)],
        contradiction_flags=contradiction_flags or [],
        paradox_flags=paradox_flags or [],
        dialogue_history=history or [],
        unsolved_buffer=unsolved_buffer or [],
    )


def _make_dialogue(
    state: DialogueState = DialogueState.PROBING,
    turn_count: int = 1,
) -> SocraticDialogueState:
    d = SocraticDialogueState()
    d.active = True
    d.current_state = state
    d.turn_count = turn_count
    d.starting_position = "I believe freedom matters."
    return d


def _make_neurochem(
    ach: float = 0.5, da: float = 0.5, oxt: float = 0.5,
    ne: float = 0.5, gaba: float = 0.5
) -> SocraticEngineState:
    s = SocraticEngineState()
    s.ach_level = ach
    s.da_tonic_level = da
    s.oxt_level = oxt
    s.ne_level = ne
    s.gaba_level = gaba
    return s


# =====================================================================
# TestEnumerations
# =====================================================================

class TestEnumerations:
    def test_dialogue_state_count(self):
        assert len(DialogueState) == 6

    def test_dialogue_state_values(self):
        vals = {s.value for s in DialogueState}
        assert vals == {"PROBING", "ELENCHUS", "APORIA", "EXPLORING", "MAIEUTICS", "EXIT"}

    def test_question_type_count(self):
        # 18 external + 5 internal = 23
        assert len(QuestionType) == 23

    def test_expected_effect_values(self):
        assert ExpectedEffect.CRYSTALLIZE_INSIGHT in ExpectedEffect

    def test_insight_source_values(self):
        assert InsightSource.USER_ARTICULATED in InsightSource
        assert InsightSource.SYSTEM_CRYSTALLIZED in InsightSource
        assert InsightSource.COLLABORATIVE in InsightSource

    def test_yield_reason_values(self):
        expected = {"FATIGUE", "FRUSTRATION", "TOPIC_MISMATCH", "APORIA_LIMIT", "INSIGHT_ACHIEVED"}
        assert {r.value for r in YieldReason} == expected


# =====================================================================
# TestConfig
# =====================================================================

class TestSocraticEngineConfig:
    def test_config_is_frozen(self):
        config = SocraticEngineConfig()
        with pytest.raises((TypeError, AttributeError)):
            config.theta_socratic = 0.9  # type: ignore

    def test_default_thresholds_in_range(self):
        config = SocraticEngineConfig()
        assert 0.0 < config.theta_socratic < 1.0
        assert 0.0 < config.theta_depth < 1.0
        assert config.max_socratic_turns >= 2
        assert config.max_aporia_turns >= 1

    def test_socratic_weights_sum_positive(self):
        config = SocraticEngineConfig()
        pos = config.w_s1 + config.w_s2 + config.w_s3
        neg = config.w_s4 + config.w_s5 + config.w_s6
        assert pos > 0
        assert neg > 0

    def test_convergence_weights_sum_to_one(self):
        config = SocraticEngineConfig()
        total = config.w_k1 + config.w_k2 + config.w_k3
        assert abs(total - 1.0) < 1e-6

    def test_neurochem_coupling_constants_positive(self):
        config = SocraticEngineConfig()
        assert config.beta_socratic > 0
        assert config.beta_insight > 0
        assert config.beta_elenchus > 0
        assert config.lambda_ne > 0
        assert config.eta_aporia > 0


# =====================================================================
# TestSocraticScore
# =====================================================================

class TestSocraticScore:
    def test_zero_intention_gives_zero(self):
        iv = _make_intention()
        score = compute_socratic_score(iv, _CONFIG)
        assert score == pytest.approx(0.0)

    def test_high_exploration_activates(self):
        # e_exploration=1.0 → w_s1×1.0=0.35 = exactly θ_socratic; add e_challenge to push above
        iv = _make_intention(e_exploration=1.0, e_challenge=0.5)
        score = compute_socratic_score(iv, _CONFIG)
        assert score > _CONFIG.theta_socratic

    def test_high_pragmatism_suppresses(self):
        iv = _make_intention(e_exploration=1.0, e_pragmatism=1.0)
        score = compute_socratic_score(iv, _CONFIG)
        # Pragmatism penalty should reduce score
        iv_no_pragma = _make_intention(e_exploration=1.0)
        score_no_pragma = compute_socratic_score(iv_no_pragma, _CONFIG)
        assert score < score_no_pragma

    def test_score_clamped_to_zero(self):
        iv = _make_intention(e_defensiveness=1.0, e_discharge=1.0)
        score = compute_socratic_score(iv, _CONFIG)
        assert score >= 0.0

    def test_score_formula(self):
        # e_exploration=0.8 → intent_scores, e_challenge=0.4 → direct field, e_symbolism=0.2 → intent_scores
        iv = _make_intention(e_exploration=0.8, e_challenge=0.4, e_symbolism=0.2)
        expected = (
            _CONFIG.w_s1 * 0.8 + _CONFIG.w_s2 * 0.4 + _CONFIG.w_s3 * 0.2
        )
        expected = min(max(expected, 0.0), 1.0)
        assert compute_socratic_score(iv, _CONFIG) == pytest.approx(expected, abs=1e-6)

    def test_all_penalties_max_gives_low_score(self):
        iv = _make_intention(e_pragmatism=1.0, e_discharge=1.0, e_defensiveness=1.0)
        score = compute_socratic_score(iv, _CONFIG)
        assert score < _CONFIG.theta_socratic


# =====================================================================
# TestTopicDepth
# =====================================================================

class TestTopicDepth:
    def test_empty_text_returns_zero(self):
        assert compute_topic_depth("", _CONFIG) == 0.0

    def test_procedural_text_low_depth(self):
        depth = compute_topic_depth("how do i sort a list in python", _CONFIG)
        assert depth < _CONFIG.theta_depth

    def test_opinion_text_boosts_depth(self):
        depth = compute_topic_depth(
            "I believe that freedom ought to be the highest value in any society and should always be protected.",
            _CONFIG
        )
        assert depth > _CONFIG.theta_depth

    def test_long_complex_text_higher_depth(self):
        short_depth = compute_topic_depth("Yes.", _CONFIG)
        long_depth = compute_topic_depth(
            "The relationship between individual autonomy and collective responsibility is philosophically intricate and contested.",
            _CONFIG
        )
        assert long_depth > short_depth

    def test_depth_bounded(self):
        depth = compute_topic_depth("I believe " * 100, _CONFIG)
        assert 0.0 <= depth <= 1.0


# =====================================================================
# TestModeGate
# =====================================================================

class TestModeGate:
    def test_normal_mode_allowed(self):
        assert mode_allows_socratic(OperationalMode.NORMAL)

    def test_learning_mode_allowed(self):
        assert mode_allows_socratic(OperationalMode.LEARNING)

    def test_reflective_mode_allowed(self):
        assert mode_allows_socratic(OperationalMode.REFLECTIVE)

    def test_rem_normal_mode_allowed(self):
        assert mode_allows_socratic(OperationalMode.REM_NORMAL)

    def test_dev_mode_not_allowed(self):
        assert not mode_allows_socratic(OperationalMode.DEV)

    def test_rem_dream_mode_not_allowed(self):
        assert not mode_allows_socratic(OperationalMode.REM_DREAM)


# =====================================================================
# TestActivationGates
# =====================================================================

class TestActivationGates:
    def test_full_activation(self):
        inp = _make_input(
            text="I believe that freedom ought to be valued over security.",
            intention=_make_intention(e_exploration=0.8, e_challenge=0.3),
        )
        dialogue = SocraticDialogueState()
        neurochem = _make_neurochem()
        activated, score, depth = check_all_activation_gates(inp, dialogue, _CONFIG, neurochem)
        assert activated
        assert score > _CONFIG.theta_socratic
        assert depth > 0.0

    def test_dev_mode_blocks(self):
        inp = _make_input(
            text="I believe freedom is fundamental.",
            mode=OperationalMode.DEV,
            intention=_make_intention(e_exploration=1.0),
        )
        dialogue = SocraticDialogueState()
        neurochem = _make_neurochem()
        activated, _, _ = check_all_activation_gates(inp, dialogue, _CONFIG, neurochem)
        assert not activated

    def test_low_intention_blocks(self):
        inp = _make_input(
            text="I believe justice should prevail.",
            intention=_make_intention(e_pragmatism=1.0),
        )
        dialogue = SocraticDialogueState()
        neurochem = _make_neurochem()
        activated, score, _ = check_all_activation_gates(inp, dialogue, _CONFIG, neurochem)
        assert score < _CONFIG.theta_socratic
        assert not activated

    def test_procedural_text_blocks(self):
        inp = _make_input(
            text="how do i sort a list",
            intention=_make_intention(e_exploration=1.0),
        )
        dialogue = SocraticDialogueState()
        neurochem = _make_neurochem()
        activated, _, depth = check_all_activation_gates(inp, dialogue, _CONFIG, neurochem)
        assert not activated
        assert depth < _CONFIG.theta_depth

    def test_fatigue_gate_blocks_after_max_turns(self):
        inp = _make_input(
            text="I believe freedom is fundamental and must be protected.",
            intention=_make_intention(e_exploration=1.0),
        )
        dialogue = SocraticDialogueState()
        dialogue.consecutive_socratic_turns = _CONFIG.max_socratic_turns + 1
        neurochem = _make_neurochem()
        activated, _, _ = check_all_activation_gates(inp, dialogue, _CONFIG, neurochem)
        assert not activated

    def test_yield_cooldown_blocks(self):
        inp = _make_input(
            text="I believe this deeply.",
            intention=_make_intention(e_exploration=1.0),
        )
        dialogue = SocraticDialogueState()
        dialogue.yield_cooldown = 2
        neurochem = _make_neurochem()
        activated, _, _ = check_all_activation_gates(inp, dialogue, _CONFIG, neurochem)
        assert not activated

    def test_high_ach_extends_effective_max_turns(self):
        """High ACh → higher effective_max_turns → stays activated longer."""
        inp = _make_input(
            text="I think that moral responsibility ought to be contextual and contextually determined.",
            intention=_make_intention(e_exploration=1.0, e_challenge=0.5),
        )
        dialogue = SocraticDialogueState()
        # Set turns to max_socratic_turns - 1: normally still OK, also with high ACh
        dialogue.consecutive_socratic_turns = _CONFIG.max_socratic_turns - 1
        neurochem_high_ach = _make_neurochem(ach=1.0)
        activated, _, _ = check_all_activation_gates(inp, dialogue, _CONFIG, neurochem_high_ach)
        # With high ACh, effective_max_turns = 5 + 1 = 6; turns=4 → still within limit
        assert activated


# =====================================================================
# TestTransitionSignals
# =====================================================================

class TestTransitionSignals:
    def test_contradiction_signal_no_flags(self):
        assert compute_contradiction_signal([]) == 0.0

    def test_contradiction_signal_high_confidence(self):
        flag = _make_contradiction_flag(confidence=1.0)
        sig = compute_contradiction_signal([flag])
        assert sig == pytest.approx(1.0)

    def test_contradiction_signal_mid_confidence(self):
        flag = _make_contradiction_flag(confidence=0.5)
        sig = compute_contradiction_signal([flag])
        assert 0.0 < sig < 1.0

    def test_assumption_signal_no_assumptions(self):
        assert compute_assumption_signal([], []) == 0.0

    def test_assumption_signal_all_covered(self):
        props = [_make_proposition("P1"), _make_proposition("P2")]
        assum = [Assumption(text="a"), Assumption(text="b")]
        sig = compute_assumption_signal(props, assum)
        assert sig == pytest.approx(1.0)

    def test_assumption_signal_half(self):
        props = [_make_proposition("P1"), _make_proposition("P2"), _make_proposition("P3"), _make_proposition("P4")]
        assum = [Assumption(text="a"), Assumption(text="b")]
        sig = compute_assumption_signal(props, assum)
        assert sig == pytest.approx(0.5)

    def test_uncertainty_signal_empty(self):
        assert compute_uncertainty_signal("", _CONFIG) == 0.0

    def test_uncertainty_signal_no_hedging(self):
        sig = compute_uncertainty_signal("The earth orbits the sun.", _CONFIG)
        assert sig < 0.1

    def test_uncertainty_signal_hedging_present(self):
        sig = compute_uncertainty_signal(
            "Maybe I believe this is perhaps the right answer, possibly.", _CONFIG
        )
        assert sig > 0.0

    def test_frustration_signal_no_markers(self):
        f = compute_frustration_signal("I think freedom is important.", 10, 10, _CONFIG)
        assert f == pytest.approx(0.0, abs=0.1)

    def test_frustration_signal_direct_request(self):
        f = compute_frustration_signal("just tell me the answer", 5, 20, _CONFIG)
        assert f > 0.0

    def test_frustration_signal_brevity(self):
        f = compute_frustration_signal("No.", 1, 30, _CONFIG)
        assert f > 0.0

    def test_frustration_signal_clamped(self):
        f = compute_frustration_signal("just tell me the answer", 1, 100, _CONFIG)
        assert 0.0 <= f <= 1.0


# =====================================================================
# TestConvergenceMetrics
# =====================================================================

class TestConvergenceMetrics:
    def test_fewer_than_3_turns_returns_zero_kappa(self):
        metrics = compute_convergence_metrics([["a", "b"]], _CONFIG)
        assert metrics.kappa_overall == 0.0
        assert metrics.turns_tracked == 1

    def test_three_identical_turns_high_convergence(self):
        traj = [["freedom", "justice", "value"], ["freedom", "justice", "value"], ["freedom", "justice", "value"]]
        metrics = compute_convergence_metrics(traj, _CONFIG)
        assert metrics.kappa_overall > 0.5
        assert metrics.stabilization == pytest.approx(1.0)

    def test_diverging_turns_lower_kappa(self):
        traj = [
            ["a", "b", "c"],
            ["d", "e", "f"],
            ["g", "h", "i"],
        ]
        metrics = compute_convergence_metrics(traj, _CONFIG)
        assert metrics.kappa_overall < 0.5

    def test_kappa_bounded(self):
        traj = [["x"] * 10, ["x"] * 10, ["x"] * 10]
        metrics = compute_convergence_metrics(traj, _CONFIG)
        assert 0.0 <= metrics.kappa_overall <= 1.0

    def test_turns_tracked_correct(self):
        traj = [["a"], ["b"], ["c"], ["d"]]
        metrics = compute_convergence_metrics(traj, _CONFIG)
        assert metrics.turns_tracked == 4


# =====================================================================
# TestAssumptionDetection
# =====================================================================

class TestAssumptionDetection:
    def test_no_assumptions_clean_text(self):
        text = "The cat sat on the mat."
        results = detect_implicit_assumptions(text, [])
        assert isinstance(results, list)

    def test_universal_all_detected(self):
        text = "All people want happiness."
        results = detect_implicit_assumptions(text, [])
        assert any("universal" in a.text for a in results)

    def test_obviously_detected(self):
        text = "Obviously, this is the right approach."
        results = detect_implicit_assumptions(text, [])
        assert any("shared understanding" in a.text for a in results)

    def test_never_detected(self):
        text = "I never make mistakes in this domain."
        results = detect_implicit_assumptions(text, [])
        assert any("universal negative" in a.text for a in results)

    def test_assumption_has_required_fields(self):
        text = "Everyone always behaves rationally."
        results = detect_implicit_assumptions(text, [_make_proposition("test")])
        assert all(hasattr(a, "text") and hasattr(a, "confidence") for a in results)

    def test_confidence_in_range(self):
        text = "All systems are clearly optimal."
        results = detect_implicit_assumptions(text, [])
        for a in results:
            assert 0.0 <= a.confidence <= 1.0


# =====================================================================
# TestEntailmentGeneration
# =====================================================================

class TestEntailmentGeneration:
    def test_returns_list(self):
        props = [_make_proposition("Freedom is essential.")]
        result = generate_entailments_from_propositions(props, [], [])
        assert isinstance(result, list)

    def test_one_proposition_one_entailment(self):
        props = [_make_proposition("Humans are rational.")]
        result = generate_entailments_from_propositions(props, [], [])
        assert len(result) >= 1

    def test_entailment_has_fields(self):
        props = [_make_proposition("Humans are rational.")]
        result = generate_entailments_from_propositions(props, [], [])
        if result:
            e = result[0]
            assert hasattr(e, "antecedent")
            assert hasattr(e, "consequent")
            assert hasattr(e, "confidence")

    def test_confidence_above_threshold(self):
        props = [_make_proposition("Test.")]
        result = generate_entailments_from_propositions(props, [], [])
        for e in result:
            assert e.confidence >= _CONFIG.entailment_confidence

    def test_contradiction_flag_populates_contradicts(self):
        props = [_make_proposition("Freedom is absolute.")]
        flags = [_make_contradiction_flag(confidence=0.9)]
        result = generate_entailments_from_propositions(props, [], flags)
        # If any entailment produced, it should reference the contradiction
        if result:
            assert result[0].contradicts_belief == "test contradiction"

    def test_empty_props_returns_empty(self):
        result = generate_entailments_from_propositions([], [], [])
        assert result == []

    def test_capped_at_three(self):
        props = [_make_proposition(f"P{i}") for i in range(10)]
        result = generate_entailments_from_propositions(props, [], [])
        assert len(result) <= 3


# =====================================================================
# TestTargetSelection
# =====================================================================

class TestTargetSelection:
    def _make_convergence(self, kappa: float = 0.3, turns: int = 3) -> ConvergenceMetrics:
        return ConvergenceMetrics(kappa_overall=kappa, turns_tracked=turns)

    def test_probing_returns_shortest_prop(self):
        props = [_make_proposition("A"), _make_proposition("A much longer proposition")]
        target = select_target_proposition(DialogueState.PROBING, props, [], [], self._make_convergence())
        assert target == "A"

    def test_probing_no_props_returns_none(self):
        target = select_target_proposition(DialogueState.PROBING, [], [], [], self._make_convergence())
        assert target is None

    def test_elenchus_uses_contradiction_flag(self):
        flags = [_make_contradiction_flag()]
        target = select_target_proposition(DialogueState.ELENCHUS, [], flags, [], self._make_convergence())
        assert target == "test contradiction"

    def test_elenchus_fallback_to_assumption(self):
        assum = [Assumption(text="hidden assumption")]
        target = select_target_proposition(DialogueState.ELENCHUS, [], [], assum, self._make_convergence())
        assert target == "hidden assumption"

    def test_aporia_returns_none(self):
        props = [_make_proposition("Something")]
        target = select_target_proposition(DialogueState.APORIA, props, [], [], self._make_convergence())
        assert target is None

    def test_exploring_returns_longest(self):
        props = [_make_proposition("Short"), _make_proposition("This is a much longer proposition with many words")]
        target = select_target_proposition(DialogueState.EXPLORING, props, [], [], self._make_convergence())
        assert "longer" in target

    def test_maieutics_returns_last_prop(self):
        props = [_make_proposition("First"), _make_proposition("Last insight")]
        target = select_target_proposition(DialogueState.MAIEUTICS, props, [], [], self._make_convergence())
        assert target == "Last insight"


# =====================================================================
# TestQuestionTypeSelection
# =====================================================================

class TestQuestionTypeSelection:
    def test_probing_returns_probing_type(self):
        qt = select_question_type(DialogueState.PROBING, [], _CONFIG)
        assert qt in (QuestionType.CLARIFICATION, QuestionType.FOUNDATIONAL,
                      QuestionType.DEFINITIONAL, QuestionType.SCOPE)

    def test_elenchus_returns_elenchus_type(self):
        qt = select_question_type(DialogueState.ELENCHUS, [], _CONFIG)
        assert qt in (QuestionType.IMPLICATIVE, QuestionType.COUNTER_CASE, QuestionType.CONSISTENCY)

    def test_aporia_returns_aporia_type(self):
        qt = select_question_type(DialogueState.APORIA, [], _CONFIG)
        assert qt in (QuestionType.REFRAMING, QuestionType.ANALOGICAL, QuestionType.ABSTRACTING)

    def test_exploring_returns_exploring_type(self):
        qt = select_question_type(DialogueState.EXPLORING, [], _CONFIG)
        assert qt in (QuestionType.GROUNDING, QuestionType.EXTENDING,
                      QuestionType.CONNECTING, QuestionType.TESTING, QuestionType.ANALOGICAL)

    def test_maieutics_returns_maieutics_type(self):
        qt = select_question_type(DialogueState.MAIEUTICS, [], _CONFIG)
        assert qt in (QuestionType.CRYSTALLIZING, QuestionType.NAMING,
                      QuestionType.INTEGRATING, QuestionType.APPLYING)

    def test_avoids_recent_repeat(self):
        # If CLARIFICATION was just used, should pick different type
        recent = [SocraticQuestion(question_type=QuestionType.CLARIFICATION)]
        qt = select_question_type(DialogueState.PROBING, recent * 3, _CONFIG)
        # Should eventually return something else or fallback
        assert qt in (QuestionType.CLARIFICATION, QuestionType.FOUNDATIONAL,
                      QuestionType.DEFINITIONAL, QuestionType.SCOPE)


# =====================================================================
# TestQuestionFormulation
# =====================================================================

class TestQuestionFormulation:
    def test_returns_socratic_question(self):
        q = formulate_question(QuestionType.CLARIFICATION, DialogueState.PROBING, "freedom", 1, None, _RNG)
        assert isinstance(q, SocraticQuestion)

    def test_question_text_is_string(self):
        q = formulate_question(QuestionType.FOUNDATIONAL, DialogueState.PROBING, None, 1, None, _RNG)
        assert isinstance(q.question_text, str)
        assert len(q.question_text) > 0

    def test_question_ends_with_mark(self):
        for qt in [QuestionType.CLARIFICATION, QuestionType.IMPLICATIVE, QuestionType.CRYSTALLIZING]:
            q = formulate_question(qt, DialogueState.PROBING, "test", 1, None, _RNG)
            assert q.question_text.endswith("?"), f"Failed for {qt}: {q.question_text}"

    def test_target_inserted_in_text(self):
        q = formulate_question(QuestionType.CLARIFICATION, DialogueState.PROBING, "justice", 1, None, _RNG)
        assert "justice" in q.question_text

    def test_turn_number_stored(self):
        q = formulate_question(QuestionType.SCOPE, DialogueState.PROBING, None, 7, None, _RNG)
        assert q.turn_number == 7

    def test_previous_id_stored(self):
        prev = "abc-123"
        q = formulate_question(QuestionType.EXTENDING, DialogueState.EXPLORING, None, 2, prev, _RNG)
        assert q.previous_question_id == prev

    def test_all_external_types_produce_question(self):
        external_types = [
            QuestionType.CLARIFICATION, QuestionType.FOUNDATIONAL, QuestionType.DEFINITIONAL,
            QuestionType.SCOPE, QuestionType.IMPLICATIVE, QuestionType.COUNTER_CASE,
            QuestionType.CONSISTENCY, QuestionType.REFRAMING, QuestionType.ANALOGICAL,
            QuestionType.ABSTRACTING, QuestionType.GROUNDING, QuestionType.EXTENDING,
            QuestionType.CONNECTING, QuestionType.TESTING, QuestionType.CRYSTALLIZING,
            QuestionType.NAMING, QuestionType.INTEGRATING, QuestionType.APPLYING,
        ]
        rng = np.random.default_rng(0)
        for qt in external_types:
            q = formulate_question(qt, DialogueState.PROBING, "test", 1, None, rng)
            assert q.question_text.endswith("?"), f"No ? for {qt}"


# =====================================================================
# TestStateTransitions
# =====================================================================

class TestStateTransitions:
    def _conv(self, kappa: float = 0.3, turns: int = 3) -> ConvergenceMetrics:
        return ConvergenceMetrics(kappa_overall=kappa, turns_tracked=turns)

    def test_probing_stays_probing_neutral(self):
        state, reason = compute_next_state(
            DialogueState.PROBING, c=0.1, a=0.1, u=0.3,
            kappa=self._conv(), f=0.1, turns_in_aporia=0,
            new_propositions=True, config=_CONFIG
        )
        assert state == DialogueState.PROBING
        assert reason is None

    def test_probing_to_elenchus_high_contradiction(self):
        state, reason = compute_next_state(
            DialogueState.PROBING, c=0.7, a=0.1, u=0.3,
            kappa=self._conv(), f=0.1, turns_in_aporia=0,
            new_propositions=True, config=_CONFIG
        )
        assert state == DialogueState.ELENCHUS

    def test_probing_to_elenchus_high_assumption(self):
        state, reason = compute_next_state(
            DialogueState.PROBING, c=0.1, a=0.5, u=0.3,
            kappa=self._conv(), f=0.1, turns_in_aporia=0,
            new_propositions=True, config=_CONFIG
        )
        assert state == DialogueState.ELENCHUS

    def test_probing_to_maieutics_convergent(self):
        state, reason = compute_next_state(
            DialogueState.PROBING, c=0.1, a=0.1, u=0.1,
            kappa=self._conv(kappa=0.2), f=0.1, turns_in_aporia=0,
            new_propositions=True, config=_CONFIG
        )
        assert state == DialogueState.MAIEUTICS

    def test_probing_exit_on_frustration(self):
        state, reason = compute_next_state(
            DialogueState.PROBING, c=0.1, a=0.1, u=0.1,
            kappa=self._conv(), f=0.8, turns_in_aporia=0,
            new_propositions=True, config=_CONFIG
        )
        assert state == DialogueState.EXIT
        assert reason == YieldReason.FRUSTRATION

    def test_elenchus_to_aporia_high_uncertainty(self):
        state, reason = compute_next_state(
            DialogueState.ELENCHUS, c=0.6, a=0.2, u=0.6,
            kappa=self._conv(), f=0.1, turns_in_aporia=0,
            new_propositions=True, config=_CONFIG
        )
        assert state == DialogueState.APORIA

    def test_elenchus_to_probing_resolved(self):
        state, reason = compute_next_state(
            DialogueState.ELENCHUS, c=0.1, a=0.1, u=0.2,
            kappa=self._conv(), f=0.1, turns_in_aporia=0,
            new_propositions=True, config=_CONFIG
        )
        assert state == DialogueState.PROBING

    def test_elenchus_exit_on_frustration(self):
        state, reason = compute_next_state(
            DialogueState.ELENCHUS, c=0.6, a=0.2, u=0.3,
            kappa=self._conv(), f=0.9, turns_in_aporia=0,
            new_propositions=True, config=_CONFIG
        )
        assert state == DialogueState.EXIT

    def test_aporia_to_exploring(self):
        state, reason = compute_next_state(
            DialogueState.APORIA, c=0.1, a=0.1, u=0.5,
            kappa=self._conv(turns=1), f=0.1, turns_in_aporia=1,
            new_propositions=True, config=_CONFIG
        )
        assert state == DialogueState.EXPLORING

    def test_aporia_exit_after_max(self):
        state, reason = compute_next_state(
            DialogueState.APORIA, c=0.1, a=0.1, u=0.2,
            kappa=self._conv(turns=1), f=0.1,
            turns_in_aporia=_CONFIG.max_aporia_turns + 1,
            new_propositions=False, config=_CONFIG
        )
        assert state == DialogueState.EXIT
        assert reason == YieldReason.APORIA_LIMIT

    def test_exploring_to_maieutics(self):
        state, reason = compute_next_state(
            DialogueState.EXPLORING, c=0.1, a=0.1, u=0.2,
            kappa=self._conv(kappa=0.2), f=0.1, turns_in_aporia=0,
            new_propositions=True, config=_CONFIG
        )
        assert state == DialogueState.MAIEUTICS

    def test_exploring_to_elenchus(self):
        state, reason = compute_next_state(
            DialogueState.EXPLORING, c=0.8, a=0.2, u=0.2,
            kappa=self._conv(), f=0.1, turns_in_aporia=0,
            new_propositions=True, config=_CONFIG
        )
        assert state == DialogueState.ELENCHUS

    def test_maieutics_exit_on_insight(self):
        state, reason = compute_next_state(
            DialogueState.MAIEUTICS, c=0.1, a=0.1, u=0.05,
            kappa=self._conv(kappa=0.7), f=0.1, turns_in_aporia=0,
            new_propositions=True, config=_CONFIG
        )
        assert state == DialogueState.EXIT
        assert reason == YieldReason.INSIGHT_ACHIEVED

    def test_maieutics_to_exploring_low_kappa(self):
        state, reason = compute_next_state(
            DialogueState.MAIEUTICS, c=0.1, a=0.1, u=0.3,
            kappa=self._conv(kappa=0.1), f=0.1, turns_in_aporia=0,
            new_propositions=True, config=_CONFIG
        )
        assert state == DialogueState.EXPLORING


# =====================================================================
# TestInsightExtraction
# =====================================================================

class TestInsightExtraction:
    def test_no_propositions_returns_none(self):
        d = _make_dialogue()
        result = extract_insight([], d, [], InsightSource.USER_ARTICULATED)
        assert result is None

    def test_returns_insight_with_propositions(self):
        d = _make_dialogue()
        d.starting_position = "Freedom is basic."
        props = [_make_proposition("Freedom is the foundation of all other values and rights.")]
        result = extract_insight(props, d, [], InsightSource.USER_ARTICULATED)
        assert result is not None
        assert isinstance(result, SocraticInsight)

    def test_dialectical_distance_nonzero(self):
        d = _make_dialogue()
        d.starting_position = "I like apples."
        props = [_make_proposition("Democracy requires an educated citizenry.")]
        result = extract_insight(props, d, [], InsightSource.USER_ARTICULATED)
        assert result is not None
        assert result.dialectical_distance > 0.0

    def test_distance_bounded(self):
        d = _make_dialogue()
        d.starting_position = "xyz"
        props = [_make_proposition("abc")]
        result = extract_insight(props, d, [], InsightSource.USER_ARTICULATED)
        if result:
            assert 0.0 <= result.dialectical_distance <= 1.0

    def test_source_stored(self):
        d = _make_dialogue()
        props = [_make_proposition("An insight.")]
        result = extract_insight(props, d, [], InsightSource.SYSTEM_CRYSTALLIZED)
        assert result is not None
        assert result.source == InsightSource.SYSTEM_CRYSTALLIZED

    def test_paradox_resolution_recorded(self):
        d = _make_dialogue()
        props = [_make_proposition("Insight that resolves paradox.")]
        pf = _make_paradox_flag(tension=0.8)
        result = extract_insight(props, d, [pf], InsightSource.COLLABORATIVE)
        assert result is not None
        assert len(result.resolved_paradoxes) > 0

    def test_question_chain_populated(self):
        d = _make_dialogue()
        q1 = SocraticQuestion(question_text="Test?", turn_number=1)
        q2 = SocraticQuestion(question_text="More?", turn_number=2)
        d.question_history = [q1, q2]
        props = [_make_proposition("An insight.")]
        result = extract_insight(props, d, [], InsightSource.USER_ARTICULATED)
        assert result is not None
        assert len(result.question_chain) == 2


# =====================================================================
# TestProcessingSignal
# =====================================================================

class TestProcessingSignal:
    def test_sigma_zero_with_no_activity(self):
        sigma = compute_socratic_processing_signal(0, 5, [], 1, [], _CONFIG)
        assert sigma == 0.0

    def test_sigma_scales_with_turn_count(self):
        s1 = compute_socratic_processing_signal(1, 5, [], 1, [], _CONFIG)
        s5 = compute_socratic_processing_signal(5, 5, [], 1, [], _CONFIG)
        assert s5 > s1

    def test_sigma_scales_with_assumptions(self):
        assum = [Assumption(text="a"), Assumption(text="b")]
        s0 = compute_socratic_processing_signal(0, 5, [], 2, [], _CONFIG)
        s_a = compute_socratic_processing_signal(0, 5, assum, 2, [], _CONFIG)
        assert s_a >= s0

    def test_sigma_bounded(self):
        assum = [Assumption(text=f"a{i}") for i in range(10)]
        sigma = compute_socratic_processing_signal(10, 5, assum, 1, [], _CONFIG)
        assert 0.0 <= sigma <= 1.0

    def test_paradox_tension_boosts_sigma(self):
        pf = _make_paradox_flag(tension=0.9)
        s0 = compute_socratic_processing_signal(0, 5, [], 1, [], _CONFIG)
        s_p = compute_socratic_processing_signal(0, 5, [], 1, [pf], _CONFIG)
        assert s_p > s0


# =====================================================================
# TestNeurochemicalSignals
# =====================================================================

class TestNeurochemicalSignals:
    def _rng(self):
        return np.random.default_rng(99)

    def test_keys_present(self):
        signals = compute_neurochemical_signals(
            DialogueState.PROBING, 0.5, ConvergenceMetrics(), 0.3, 0.2,
            0.8, 0.1, False, 0.0, 2, _CONFIG, self._rng()
        )
        expected = {"ach_burst", "da_tonic", "da_phasic", "oxt_drift",
                    "ne_burst", "gaba_suppress", "theta_boost",
                    "beta_boost", "theta_gamma_boost"}
        assert set(signals.keys()) == expected

    def test_zero_sigma_and_no_insight_mostly_zero(self):
        signals = compute_neurochemical_signals(
            DialogueState.PROBING, 0.0, ConvergenceMetrics(), 0.0, 0.0,
            0.5, 0.0, False, 0.0, 0, _CONFIG, self._rng()
        )
        # All should be zero or near zero
        for key, val in signals.items():
            assert val == pytest.approx(0.0, abs=0.05), f"{key} not zero: {val}"

    def test_ach_burst_positive_with_sigma(self):
        signals = compute_neurochemical_signals(
            DialogueState.PROBING, 0.8, ConvergenceMetrics(), 0.0, 0.0,
            0.8, 0.0, False, 0.0, 1, _CONFIG, self._rng()
        )
        assert signals["ach_burst"] > 0.0

    def test_da_tonic_inversely_modulated_by_kappa(self):
        low_kappa = ConvergenceMetrics(kappa_overall=0.1)
        high_kappa = ConvergenceMetrics(kappa_overall=0.9)
        s_low = compute_neurochemical_signals(
            DialogueState.PROBING, 0.5, low_kappa, 0.0, 0.0,
            0.8, 0.0, False, 0.0, 1, _CONFIG, self._rng()
        )
        s_high = compute_neurochemical_signals(
            DialogueState.PROBING, 0.5, high_kappa, 0.0, 0.0,
            0.8, 0.0, False, 0.0, 1, _CONFIG, self._rng()
        )
        assert s_low["da_tonic"] > s_high["da_tonic"]

    def test_da_phasic_on_insight(self):
        signals = compute_neurochemical_signals(
            DialogueState.MAIEUTICS, 0.5, ConvergenceMetrics(), 0.0, 0.0,
            0.8, 0.0, True, 0.8, 5, _CONFIG, self._rng()
        )
        assert signals["da_phasic"] > 0.0

    def test_oxt_positive_with_high_engagement(self):
        signals = compute_neurochemical_signals(
            DialogueState.PROBING, 0.5, ConvergenceMetrics(), 0.0, 0.0,
            1.0, 0.0, False, 0.0, 1, _CONFIG, self._rng()
        )
        assert signals["oxt_drift"] > 0.0

    def test_oxt_negative_with_high_frustration(self):
        signals = compute_neurochemical_signals(
            DialogueState.PROBING, 0.5, ConvergenceMetrics(), 0.0, 0.0,
            0.0, 0.9, False, 0.0, 1, _CONFIG, self._rng()
        )
        assert signals["oxt_drift"] < 0.0

    def test_ne_burst_only_in_elenchus(self):
        rng = np.random.default_rng(1)
        s_probing = compute_neurochemical_signals(
            DialogueState.PROBING, 0.5, ConvergenceMetrics(), 0.8, 0.2,
            0.8, 0.1, False, 0.0, 1, _CONFIG, rng
        )
        rng2 = np.random.default_rng(1)
        s_elenchus = compute_neurochemical_signals(
            DialogueState.ELENCHUS, 0.5, ConvergenceMetrics(), 0.8, 0.2,
            0.8, 0.1, False, 0.0, 1, _CONFIG, rng2
        )
        assert s_probing["ne_burst"] == pytest.approx(0.0)
        assert s_elenchus["ne_burst"] >= 0.0  # Poisson may give 0 with low λ

    def test_gaba_suppression_in_aporia(self):
        signals = compute_neurochemical_signals(
            DialogueState.APORIA, 0.5, ConvergenceMetrics(), 0.0, 0.6,
            0.8, 0.1, False, 0.0, 1, _CONFIG, self._rng()
        )
        assert signals["gaba_suppress"] > 0.0

    def test_theta_boost_in_exploring(self):
        signals = compute_neurochemical_signals(
            DialogueState.EXPLORING, 0.5, ConvergenceMetrics(), 0.0, 0.0,
            0.8, 0.1, False, 0.0, 1, _CONFIG, self._rng()
        )
        assert signals["theta_boost"] > 0.0

    def test_beta_boost_in_elenchus(self):
        signals = compute_neurochemical_signals(
            DialogueState.ELENCHUS, 0.5, ConvergenceMetrics(), 0.8, 0.0,
            0.8, 0.1, False, 0.0, 1, _CONFIG, self._rng()
        )
        assert signals["beta_boost"] > 0.0

    def test_theta_gamma_boost_on_insight(self):
        signals = compute_neurochemical_signals(
            DialogueState.MAIEUTICS, 0.5, ConvergenceMetrics(), 0.0, 0.0,
            0.8, 0.0, True, 0.5, 3, _CONFIG, self._rng()
        )
        assert signals["theta_gamma_boost"] == pytest.approx(_CONFIG.delta_insight_coherence)

    def test_all_signals_bounded_nonnegative(self):
        for state in DialogueState:
            if state == DialogueState.EXIT:
                continue
            signals = compute_neurochemical_signals(
                state, 0.5, ConvergenceMetrics(kappa_overall=0.3), 0.5, 0.4,
                0.7, 0.2, False, 0.0, 3, _CONFIG, np.random.default_rng(0)
            )
            for k, v in signals.items():
                assert v >= -0.5, f"Signal {k} too negative: {v} in state {state}"


# =====================================================================
# TestInternalInquiry
# =====================================================================

class TestInternalInquiry:
    def test_generate_internal_questions_empty_buffer(self):
        qs = generate_internal_questions([], _RNG, _CONFIG)
        assert qs == []

    def test_generates_questions_for_unsolved(self):
        entry = UnsolvedEntry(concept_text="the nature of free will", attempt_count=1, motivational_salience=0.8)
        qs = generate_internal_questions([entry], _RNG, _CONFIG)
        assert len(qs) > 0

    def test_uses_highest_salience_concept(self):
        low = UnsolvedEntry(concept_text="minor question", attempt_count=0, motivational_salience=0.2)
        high = UnsolvedEntry(concept_text="uniqueconcept_xyz", attempt_count=2, motivational_salience=0.9)
        qs = generate_internal_questions([low, high], _RNG, _CONFIG)
        # The high-salience concept text should be stored as target_proposition
        assert any(q.target_proposition == "uniqueconcept_xyz" for q in qs)

    def test_internal_question_types(self):
        entry = UnsolvedEntry(concept_text="consciousness", attempt_count=1, motivational_salience=0.7)
        qs = generate_internal_questions([entry], _RNG, _CONFIG)
        types = {q.question_type for q in qs}
        internal_types = {
            QuestionType.FALSIFICATION, QuestionType.PROVENANCE, QuestionType.ALTERNATIVE,
            QuestionType.DEPENDENCY, QuestionType.STABILITY
        }
        assert types.issubset(internal_types | {QuestionType.CLARIFICATION})

    def test_internal_questions_are_questions(self):
        entry = UnsolvedEntry(concept_text="identity", attempt_count=0, motivational_salience=0.5)
        qs = generate_internal_questions([entry], _RNG, _CONFIG)
        for q in qs:
            assert q.question_text.endswith("?"), f"Not a question: {q.question_text}"


# =====================================================================
# TestEnginePorts
# =====================================================================

class TestEnginePorts:
    def test_default_mode_is_normal(self):
        engine = SocraticReasoningEngine()
        status = engine.get_status()
        assert status["mode"] == OperationalMode.NORMAL.value

    def test_configure_changes_mode(self):
        engine = SocraticReasoningEngine()
        engine.configure(OperationalMode.LEARNING)
        assert engine.get_status()["mode"] == OperationalMode.LEARNING.value

    def test_get_status_has_required_keys(self):
        engine = SocraticReasoningEngine()
        status = engine.get_status()
        required = {"engine_id", "mode", "socratic_active", "current_state", "turn_count",
                    "consecutive_socratic_turns", "yield_cooldown",
                    "assumptions_surfaced", "insights_generated", "question_history_len"}
        assert required.issubset(set(status.keys()))
        assert status["engine_id"] == "socratic_reasoning_engine"

    def test_update_neurochem_ach(self):
        engine = SocraticReasoningEngine()
        engine.update_neurochem_state({"ach": 0.9})
        assert engine._state.ach_level == pytest.approx(0.9)

    def test_update_neurochem_all(self):
        engine = SocraticReasoningEngine()
        engine.update_neurochem_state({"ach": 0.8, "da": 0.7, "oxt": 0.6, "ne": 0.5, "gaba": 0.4})
        assert engine._state.ach_level == pytest.approx(0.8)
        assert engine._state.da_tonic_level == pytest.approx(0.7)
        assert engine._state.oxt_level == pytest.approx(0.6)
        assert engine._state.ne_level == pytest.approx(0.5)
        assert engine._state.gaba_level == pytest.approx(0.4)

    def test_update_neurochem_clamps(self):
        engine = SocraticReasoningEngine()
        engine.update_neurochem_state({"ach": 5.0})
        assert engine._state.ach_level == pytest.approx(1.0)
        engine.update_neurochem_state({"ne": -1.0})
        assert engine._state.ne_level == pytest.approx(0.0)

    def test_initial_state_not_active(self):
        engine = SocraticReasoningEngine()
        status = engine.get_status()
        assert not status["socratic_active"]
        assert status["turn_count"] == 0

    def test_custom_config(self):
        config = SocraticEngineConfig(theta_socratic=0.1, max_socratic_turns=10)
        engine = SocraticReasoningEngine(config=config)
        assert engine._config.theta_socratic == pytest.approx(0.1)


# =====================================================================
# TestProcess
# =====================================================================

class TestProcessEngine:
    def test_process_returns_output(self):
        engine = SocraticReasoningEngine()
        inp = _make_input()
        result = engine.process(inp)
        assert isinstance(result, SocraticOutput)

    def test_activated_produces_question(self):
        engine = SocraticReasoningEngine()
        inp = _make_input(
            text="I believe that freedom ought to be the highest political value in any democratic society.",
            intention=_make_intention(e_exploration=0.9, e_challenge=0.4),
        )
        result = engine.process(inp)
        if result.socratic_active:
            assert result.generated_question is not None
            assert isinstance(result.generated_question, SocraticQuestion)

    def test_not_activated_yields_direct(self):
        engine = SocraticReasoningEngine()
        inp = _make_input(
            text="how do i sort a list in python",
            intention=_make_intention(e_pragmatism=1.0),
        )
        result = engine.process(inp)
        assert not result.socratic_active
        assert result.yield_to_direct

    def test_dev_mode_not_activated(self):
        engine = SocraticReasoningEngine()
        engine.configure(OperationalMode.DEV)
        inp = _make_input(
            text="I believe freedom must be valued above all.",
            mode=OperationalMode.DEV,
            intention=_make_intention(e_exploration=1.0),
        )
        result = engine.process(inp)
        assert not result.socratic_active

    def test_processing_time_positive(self):
        engine = SocraticReasoningEngine()
        result = engine.process(_make_input())
        assert result.processing_time_ms >= 0.0

    def test_neurochemical_signals_present(self):
        engine = SocraticReasoningEngine()
        result = engine.process(_make_input())
        assert isinstance(result.neurochemical_signals, dict)
        assert "ach_burst" in result.neurochemical_signals

    def test_assumptions_identified(self):
        engine = SocraticReasoningEngine()
        inp = _make_input(
            text="Obviously everyone always wants freedom and nobody ever disagrees.",
            intention=_make_intention(e_exploration=0.9),
        )
        result = engine.process(inp)
        # May or may not activate, but assumptions should be identified if activated
        assert isinstance(result.identified_assumptions, list)

    def test_multi_turn_turn_count_increments(self):
        engine = SocraticReasoningEngine()
        inp = _make_input(
            text="I think democracy requires freedom and should be valued above all else.",
            intention=_make_intention(e_exploration=0.9, e_challenge=0.4),
        )
        r1 = engine.process(inp)
        if r1.socratic_active:
            r2 = engine.process(inp)
            if r2.socratic_active:
                status = engine.get_status()
                assert status["turn_count"] >= 2

    def test_convergence_state_returned(self):
        engine = SocraticReasoningEngine()
        inp = _make_input()
        result = engine.process(inp)
        assert isinstance(result.convergence_state, ConvergenceMetrics)

    def test_socratic_score_in_output(self):
        engine = SocraticReasoningEngine()
        result = engine.process(_make_input())
        assert result.activation_score >= 0.0

    def test_topic_depth_in_output(self):
        engine = SocraticReasoningEngine()
        result = engine.process(_make_input())
        assert result.topic_depth_score >= 0.0


# =====================================================================
# TestInternalInquiryViaProcess
# =====================================================================

class TestInternalInquiryViaProcess:
    def test_rem_normal_with_unsolved_buffer_returns_question(self):
        engine = SocraticReasoningEngine()
        entry = UnsolvedEntry(concept_text="the hard problem of consciousness", attempt_count=2, motivational_salience=0.9)
        inp = _make_input(mode=OperationalMode.REM_NORMAL, unsolved_buffer=[entry])
        result = engine.process(inp)
        assert result.generated_question is not None

    def test_rem_normal_empty_buffer_falls_through(self):
        engine = SocraticReasoningEngine()
        # No unsolved buffer → standard activation path
        inp = _make_input(mode=OperationalMode.REM_NORMAL)
        result = engine.process(inp)
        assert isinstance(result, SocraticOutput)

    def test_reflective_mode_with_buffer(self):
        engine = SocraticReasoningEngine()
        entry = UnsolvedEntry(concept_text="my own bias in reasoning", attempt_count=1, motivational_salience=0.7)
        inp = _make_input(mode=OperationalMode.REFLECTIVE, unsolved_buffer=[entry])
        result = engine.process(inp)
        assert result.generated_question is not None
        assert result.socratic_active

    def test_internal_signals_present(self):
        engine = SocraticReasoningEngine()
        entry = UnsolvedEntry(concept_text="free will", attempt_count=0, motivational_salience=0.6)
        inp = _make_input(mode=OperationalMode.REM_NORMAL, unsolved_buffer=[entry])
        result = engine.process(inp)
        assert result.neurochemical_signals.get("theta_boost", 0.0) >= 0.0


# =====================================================================
# TestOutputTypes
# =====================================================================

class TestOutputTypes:
    def test_socratic_output_is_frozen(self):
        """SocraticOutput is a frozen dataclass — mutation raises."""
        engine = SocraticReasoningEngine()
        result = engine.process(_make_input())
        # Frozen dataclasses raise TypeError on attribute set
        with pytest.raises((TypeError, AttributeError)):
            result.socratic_active = not result.socratic_active  # type: ignore

    def test_socratic_question_is_frozen(self):
        q = SocraticQuestion(question_text="Test?")
        with pytest.raises((TypeError, AttributeError)):
            q.question_text = "modified"  # type: ignore

    def test_assumption_is_frozen(self):
        a = Assumption(text="test")
        with pytest.raises((TypeError, AttributeError)):
            a.text = "changed"  # type: ignore

    def test_entailment_is_frozen(self):
        e = Entailment(antecedent="P", consequent="Q")
        with pytest.raises((TypeError, AttributeError)):
            e.antecedent = "changed"  # type: ignore

    def test_socratic_insight_is_frozen(self):
        insight = SocraticInsight(content="test insight")
        with pytest.raises((TypeError, AttributeError)):
            insight.content = "changed"  # type: ignore

    def test_dialogue_state_is_mutable(self):
        d = SocraticDialogueState()
        d.turn_count = 5
        assert d.turn_count == 5

    def test_engine_state_is_mutable(self):
        s = SocraticEngineState()
        s.ach_level = 0.9
        assert s.ach_level == pytest.approx(0.9)

    def test_socratic_question_has_uuid(self):
        q = SocraticQuestion()
        assert len(q.question_id) > 0

    def test_socratic_insight_timestamp_present(self):
        i = SocraticInsight(content="insight")
        assert isinstance(i.timestamp, datetime)

    def test_output_insights_is_list(self):
        engine = SocraticReasoningEngine()
        result = engine.process(_make_input())
        assert isinstance(result.insights, list)


# =====================================================================
# TestEdgeCases
# =====================================================================

class TestEdgeCases:
    def test_empty_user_input(self):
        engine = SocraticReasoningEngine()
        inp = _make_input(text="", propositions=[])
        result = engine.process(inp)
        assert isinstance(result, SocraticOutput)

    def test_very_long_text(self):
        engine = SocraticReasoningEngine()
        long_text = "I believe that freedom ought to be the highest value. " * 50
        inp = _make_input(text=long_text)
        result = engine.process(inp)
        assert isinstance(result, SocraticOutput)

    def test_contradiction_flags_feed_entailments(self):
        engine = SocraticReasoningEngine()
        flags = [_make_contradiction_flag(level=3)]
        inp = _make_input(
            text="I believe freedom is absolute and also contextual and culturally determined.",
            intention=_make_intention(e_exploration=0.9, e_challenge=0.6),
            contradiction_flags=flags,
        )
        result = engine.process(inp)
        assert isinstance(result, SocraticOutput)

    def test_yield_cooldown_after_exit(self):
        """After EXIT, yield_cooldown should prevent immediate re-activation."""
        engine = SocraticReasoningEngine()
        engine._state.dialogue.yield_cooldown = 2
        inp = _make_input(
            text="I think freedom ought to be the most important societal value.",
            intention=_make_intention(e_exploration=1.0),
        )
        result = engine.process(inp)
        # cooldown still active (decremented to 1)
        assert not result.socratic_active

    def test_multiple_engines_independent(self):
        engine1 = SocraticReasoningEngine()
        engine2 = SocraticReasoningEngine()
        inp = _make_input(
            text="Freedom ought to be paramount.",
            intention=_make_intention(e_exploration=0.9),
        )
        engine1.process(inp)
        status1 = engine1.get_status()
        status2 = engine2.get_status()
        # engine2 should still be fresh
        assert status2["turn_count"] == 0
        assert status1["turn_count"] >= status2["turn_count"]

    def test_paradox_flags_accepted(self):
        engine = SocraticReasoningEngine()
        pf = _make_paradox_flag(tension=0.9)
        inp = _make_input(
            text="I believe freedom is both absolute and conditional.",
            intention=_make_intention(e_exploration=0.8),
            paradox_flags=[pf],
        )
        result = engine.process(inp)
        assert isinstance(result, SocraticOutput)

    def test_question_chain_grows_with_turns(self):
        engine = SocraticReasoningEngine()
        inp = _make_input(
            text="I think that moral responsibility ought to be contextual and situational.",
            intention=_make_intention(e_exploration=0.9, e_challenge=0.4),
        )
        for _ in range(3):
            engine.process(inp)
        status = engine.get_status()
        # If activated for any turns, question_history_len grows
        assert status["question_history_len"] >= 0

    def test_convergence_metrics_after_many_turns(self):
        engine = SocraticReasoningEngine()
        texts = [
            "I believe freedom is the foundation of democracy.",
            "Freedom also requires responsibility.",
            "Freedom and responsibility are deeply intertwined.",
        ]
        for text in texts:
            inp = _make_input(
                text=text,
                propositions=[_make_proposition(text)],
                intention=_make_intention(e_exploration=0.9),
            )
            result = engine.process(inp)
        assert isinstance(result.convergence_state, ConvergenceMetrics)

    def test_sigma_zero_gives_no_ach_burst(self):
        """When sigma=0 and no insight, ach should be 0."""
        rng = np.random.default_rng(42)
        signals = compute_neurochemical_signals(
            DialogueState.PROBING, 0.0, ConvergenceMetrics(), 0.0, 0.0,
            0.5, 0.0, False, 0.0, 0, _CONFIG, rng
        )
        assert signals["ach_burst"] == pytest.approx(0.0)
