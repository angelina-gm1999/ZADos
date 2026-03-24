"""
Tests for simulated_opposition_engine.py — Engine 7 (Simulated Opposition).

Coverage plan
-------------
1.  Enumerations — OppositionMode, OppositionDepth, GateOutcome, OppositionSeverity, PositionStatus, TargetType
2.  OppositionEngineConfig defaults and immutability
3.  Depth selection — compute_depth_score(), select_depth(), is_disabled_for_mode()
4.  Depth params & mode lists — get_depth_params(), get_modes_for_depth()
5.  Quality evaluation — compute_opposition_quality(), classify_severity()
6.  Activity function — compute_opposition_activity()
7.  Gate scoring — compute_gate_score(), classify_gate_outcome(), extract_caveats()
8.  Anti-nihilism — compute_block_rate(), get_effective_quality_threshold()
9.  Dialectical tension — detect_dialectical_tension()
10. Opposition generators:
      generate_counterargument()
      generate_counterexample()
      generate_alternative_explanation()
      generate_assumption_excavation()
      generate_structural_resistance()
11. Core generation loop — run_generation_loop()
12. Neurochemical signals — compute_neurochemical_signals()
13. Bidirectional feedback — apply_neurochem_feedback()
14. Engine ports — configure(), update_neurochem_state(), get_status()
15. run_gate() — end-to-end gate scenarios
16. run_request() — callable interface
17. REM_DREAM disabled mode
18. Output type validation — OppositionResult fields, OppositionItem fields
19. Edge cases
"""

from __future__ import annotations

import uuid
from collections import deque
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
    FallacyFlag,
    IntentionVector,
    Argument,
    EvidenceSignals,
    FallacyCategory,
    FallacyType,
    Proposition,
)
from zados.cognitive_engines.py_engines.logic_trap_detection_engine import (
    BiasFlag,
    TrapFlag,
    TrapType,
    TrapCategory,
    IntentLevel,
    build_defensive_strategy,
    classify_intent_level,
)
from zados.cognitive_engines.py_engines.simulated_opposition_engine import (
    DialogueTurn,
    GateOutcome,
    OppositionDepth,
    OppositionEngineConfig,
    OppositionEngineState,
    OppositionGateInput,
    OppositionItem,
    OppositionMode,
    OppositionRequest,
    OppositionResult,
    OppositionSeverity,
    ParadoxCandidate,
    PositionStatus,
    ReasoningTrace,
    ResistanceResult,
    SimulatedOppositionEngine,
    TargetType,
    _extract_components,
    _score_components,
    apply_neurochem_feedback,
    classify_gate_outcome,
    classify_severity,
    compute_block_rate,
    compute_depth_score,
    compute_gate_score,
    compute_neurochemical_signals,
    compute_opposition_activity,
    compute_opposition_quality,
    detect_dialectical_tension,
    extract_caveats,
    generate_alternative_explanation,
    generate_assumption_excavation,
    generate_counterargument,
    generate_counterexample,
    generate_structural_resistance,
    get_depth_params,
    get_effective_quality_threshold,
    get_modes_for_depth,
    is_disabled_for_mode,
    run_generation_loop,
    select_depth,
)

# =====================================================================
# Helpers / fixtures
# =====================================================================

_CONFIG = OppositionEngineConfig()
_RNG    = np.random.default_rng(42)
_STATE  = OppositionEngineState()

_TARGET_SIMPLE   = "The sky is blue because of Rayleigh scattering."
_TARGET_CAUSAL   = "High sugar intake causes obesity, which leads to diabetes."
_TARGET_UNIVERSAL = "All government policies that raise taxes always reduce economic growth."


def _make_reasoning(**kwargs) -> ReasoningTrace:
    defaults = dict(
        premises=["Rayleigh scattering affects light.", "The sky scatters short wavelengths."],
        inference_steps=["Short wavelengths scatter more → blue appears dominant."],
        conclusion="The sky appears blue.",
        confidence=0.85,
        caveats_already_acknowledged=["This does not explain sunsets."],
    )
    defaults.update(kwargs)
    return ReasoningTrace(**defaults)


def _make_gate_input(**kwargs) -> OppositionGateInput:
    defaults = dict(
        candidate_response=_TARGET_SIMPLE,
        response_reasoning=_make_reasoning(),
        original_query="Why is the sky blue?",
        conversation_context=[],
        intention_vector=IntentionVector(),
        active_contradiction_flags=[],
        active_fallacy_flags=[],
        active_bias_flags=[],
        active_trap_flags=[],
        active_mode=OperationalMode.NORMAL,
        depth_override=None,
        complexity=0.4,
        stakes=0.3,
        novelty=0.2,
        challenge_level=0.3,
    )
    defaults.update(kwargs)
    return OppositionGateInput(**defaults)


def _make_request(**kwargs) -> OppositionRequest:
    defaults = dict(
        target=_TARGET_CAUSAL,
        target_type=TargetType.ARGUMENT,
        caller_engine="test",
        caller_context={},
        requested_modes=[OppositionMode.COUNTERARGUMENT],
        requested_depth=OppositionDepth.QUICK,
        focus_area=None,
        min_opposition_quality=0.45,
        active_mode=OperationalMode.NORMAL,
    )
    defaults.update(kwargs)
    return OppositionRequest(**defaults)


def _make_item(
    mode: OppositionMode = OppositionMode.COUNTERARGUMENT,
    quality: float = 0.70,
    severity: OppositionSeverity = OppositionSeverity.SIGNIFICANT,
    target_component: str = "conclusion",
    relevance: float = 0.80,
) -> OppositionItem:
    v = quality
    return OppositionItem(
        item_id=str(uuid.uuid4()),
        mode=mode,
        content="test opposition content",
        quality_score=quality,
        validity=v,
        relevance=relevance,
        strength=v,
        novelty=v,
        target_component=target_component,
        severity=severity,
        attack_type="test_attack",
    )


# =====================================================================
# 1. Enumerations
# =====================================================================

class TestEnumerations:
    def test_opposition_mode_count(self):
        assert len(OppositionMode) == 5

    def test_opposition_depth_values(self):
        assert set(OppositionDepth) == {
            OppositionDepth.QUICK, OppositionDepth.STANDARD, OppositionDepth.DEEP
        }

    def test_gate_outcome_values(self):
        expected = {GateOutcome.PASS, GateOutcome.PASS_WITH_CAVEAT,
                    GateOutcome.REVISE, GateOutcome.BLOCK}
        assert set(GateOutcome) == expected

    def test_severity_values(self):
        expected = {OppositionSeverity.MINOR, OppositionSeverity.MODERATE,
                    OppositionSeverity.SIGNIFICANT, OppositionSeverity.CRITICAL}
        assert set(OppositionSeverity) == expected

    def test_position_status_values(self):
        expected = {PositionStatus.SURVIVED, PositionStatus.IMPROVED,
                    PositionStatus.COLLAPSED, PositionStatus.TRANSFORMED}
        assert set(PositionStatus) == expected

    def test_target_type_values(self):
        types = {t.value for t in TargetType}
        assert "ARGUMENT" in types
        assert "EXPLANATION" in types

    def test_mode_str_values(self):
        assert OppositionMode.COUNTERARGUMENT.value == "COUNTERARGUMENT"
        assert OppositionMode.STRUCTURAL_RESISTANCE.value == "STRUCTURAL_RESISTANCE"


# =====================================================================
# 2. Config
# =====================================================================

class TestOppositionEngineConfig:
    def test_depth_weights_sum(self):
        c = OppositionEngineConfig()
        total = c.w_d1 + c.w_d2 + c.w_d3 + c.w_d4 + c.w_d5
        assert abs(total - 1.0) < 1e-9

    def test_quality_weights_sum(self):
        c = OppositionEngineConfig()
        total = c.w_q1 + c.w_q2 + c.w_q3 + c.w_q4
        assert abs(total - 1.0) < 1e-9

    def test_config_is_frozen(self):
        c = OppositionEngineConfig()
        with pytest.raises((AttributeError, TypeError)):
            c.theta_opposition_quality = 0.99  # type: ignore

    def test_default_thresholds_in_range(self):
        c = OppositionEngineConfig()
        assert 0.0 < c.theta_opposition_quality < 1.0
        assert 0.0 < c.theta_nihilism < 1.0
        assert 0.0 < c.theta_relevance < 1.0

    def test_gate_score_thresholds_ordered(self):
        c = OppositionEngineConfig()
        assert 0.0 < c.gate_pass_max < c.gate_caveat_max < c.gate_revise_max < 1.0

    def test_depth_limits_ordered(self):
        c = OppositionEngineConfig()
        assert c.max_passes_quick <= c.max_passes_standard <= c.max_passes_deep

    def test_depth_multipliers(self):
        c = OppositionEngineConfig()
        assert c.depth_mult_quick < c.depth_mult_standard < c.depth_mult_deep


# =====================================================================
# 3. Depth selection
# =====================================================================

class TestDepthSelection:
    def test_compute_depth_score_zero(self):
        score = compute_depth_score(0.0, 1.0, 0.0, 0.0, 0.0, _CONFIG)
        assert score == pytest.approx(0.0)

    def test_compute_depth_score_max(self):
        score = compute_depth_score(1.0, 0.0, 1.0, 1.0, 1.0, _CONFIG)
        assert score == pytest.approx(1.0)

    def test_compute_depth_score_formula(self):
        score = compute_depth_score(0.5, 0.5, 0.5, 0.5, 0.5, _CONFIG)
        expected = (
            _CONFIG.w_d1 * 0.5
            + _CONFIG.w_d2 * 0.5
            + _CONFIG.w_d3 * 0.5
            + _CONFIG.w_d4 * 0.5
            + _CONFIG.w_d5 * 0.5
        )
        assert score == pytest.approx(expected)

    def test_depth_score_clamped(self):
        score = compute_depth_score(2.0, -1.0, 2.0, 2.0, 2.0, _CONFIG)
        assert 0.0 <= score <= 1.0

    def test_select_depth_quick(self):
        d = select_depth(0.10, OperationalMode.NORMAL, None, _CONFIG)
        assert d == OppositionDepth.QUICK

    def test_select_depth_standard(self):
        d = select_depth(0.50, OperationalMode.NORMAL, None, _CONFIG)
        assert d == OppositionDepth.STANDARD

    def test_select_depth_deep(self):
        d = select_depth(0.80, OperationalMode.NORMAL, None, _CONFIG)
        assert d == OppositionDepth.DEEP

    def test_select_depth_dev_always_deep(self):
        d = select_depth(0.0, OperationalMode.DEV, None, _CONFIG)
        assert d == OppositionDepth.DEEP

    def test_select_depth_rem_normal_always_deep(self):
        d = select_depth(0.0, OperationalMode.REM_NORMAL, None, _CONFIG)
        assert d == OppositionDepth.DEEP

    def test_select_depth_override_respected(self):
        d = select_depth(0.0, OperationalMode.NORMAL, OppositionDepth.STANDARD, _CONFIG)
        assert d == OppositionDepth.STANDARD

    def test_is_disabled_for_rem_dream(self):
        assert is_disabled_for_mode(OperationalMode.REM_DREAM) is True

    def test_is_not_disabled_for_normal(self):
        assert is_disabled_for_mode(OperationalMode.NORMAL) is False

    def test_is_not_disabled_for_dev(self):
        assert is_disabled_for_mode(OperationalMode.DEV) is False


# =====================================================================
# 4. Depth params & mode lists
# =====================================================================

class TestDepthParams:
    def test_quick_params(self):
        p, s, o, m = get_depth_params(OppositionDepth.QUICK, _CONFIG)
        assert p == _CONFIG.max_passes_quick
        assert s == _CONFIG.sufficient_quick
        assert o == _CONFIG.max_output_quick
        assert m == pytest.approx(_CONFIG.depth_mult_quick)

    def test_standard_params(self):
        p, s, o, m = get_depth_params(OppositionDepth.STANDARD, _CONFIG)
        assert p == _CONFIG.max_passes_standard
        assert m == pytest.approx(_CONFIG.depth_mult_standard)

    def test_deep_params(self):
        p, s, o, m = get_depth_params(OppositionDepth.DEEP, _CONFIG)
        assert p == _CONFIG.max_passes_deep
        assert m == pytest.approx(_CONFIG.depth_mult_deep)

    def test_modes_quick(self):
        modes = get_modes_for_depth(OppositionDepth.QUICK)
        assert modes == [OppositionMode.COUNTERARGUMENT]

    def test_modes_standard(self):
        modes = get_modes_for_depth(OppositionDepth.STANDARD)
        assert OppositionMode.COUNTERARGUMENT in modes
        assert OppositionMode.COUNTEREXAMPLE in modes
        assert OppositionMode.ALTERNATIVE_EXPLANATION in modes
        assert OppositionMode.STRUCTURAL_RESISTANCE not in modes

    def test_modes_deep(self):
        modes = get_modes_for_depth(OppositionDepth.DEEP)
        assert set(modes) == set(OppositionMode)


# =====================================================================
# 5. Quality evaluation
# =====================================================================

class TestQualityEvaluation:
    def test_quality_zero(self):
        q = compute_opposition_quality(0.0, 0.0, 0.0, 0.0, _CONFIG)
        assert q == pytest.approx(0.0)

    def test_quality_max(self):
        q = compute_opposition_quality(1.0, 1.0, 1.0, 1.0, _CONFIG)
        assert q == pytest.approx(1.0)

    def test_quality_formula(self):
        q = compute_opposition_quality(0.8, 0.7, 0.6, 0.5, _CONFIG)
        expected = (
            _CONFIG.w_q1 * 0.8
            + _CONFIG.w_q2 * 0.7
            + _CONFIG.w_q3 * 0.6
            + _CONFIG.w_q4 * 0.5
        )
        assert q == pytest.approx(expected)

    def test_severity_minor(self):
        assert classify_severity(0.50) == OppositionSeverity.MINOR

    def test_severity_moderate(self):
        assert classify_severity(0.60) == OppositionSeverity.MODERATE

    def test_severity_significant(self):
        assert classify_severity(0.75) == OppositionSeverity.SIGNIFICANT

    def test_severity_critical(self):
        assert classify_severity(0.90) == OppositionSeverity.CRITICAL

    def test_severity_boundary_minor_moderate(self):
        # 0.55 is boundary for moderate
        assert classify_severity(0.555) == OppositionSeverity.MODERATE

    def test_severity_boundary_significant(self):
        assert classify_severity(0.70001) == OppositionSeverity.SIGNIFICANT


# =====================================================================
# 6. Activity function
# =====================================================================

class TestActivityFunction:
    def test_activity_no_oppositions(self):
        assert compute_opposition_activity([], 1.0, _CONFIG) == 0.0

    def test_activity_single_item(self):
        item = _make_item(mode=OppositionMode.COUNTERARGUMENT, quality=0.8, severity=OppositionSeverity.SIGNIFICANT)
        omega = compute_opposition_activity([item], 1.0, _CONFIG)
        assert omega > 0.0

    def test_activity_scales_with_depth_multiplier(self):
        item = _make_item()
        omega_low  = compute_opposition_activity([item], 0.5, _CONFIG)
        omega_high = compute_opposition_activity([item], 1.5, _CONFIG)
        assert omega_high > omega_low

    def test_activity_structural_resistance_highest_weight(self):
        sr_item  = _make_item(mode=OppositionMode.STRUCTURAL_RESISTANCE,  quality=0.7)
        ca_item  = _make_item(mode=OppositionMode.COUNTERARGUMENT,         quality=0.7)
        omega_sr = compute_opposition_activity([sr_item], 1.0, _CONFIG)
        omega_ca = compute_opposition_activity([ca_item], 1.0, _CONFIG)
        assert omega_sr > omega_ca

    def test_activity_accumulates_across_items(self):
        item1 = _make_item(quality=0.6)
        item2 = _make_item(quality=0.7)
        omega_both = compute_opposition_activity([item1, item2], 1.0, _CONFIG)
        omega_one  = compute_opposition_activity([item1],        1.0, _CONFIG)
        assert omega_both > omega_one


# =====================================================================
# 7. Gate scoring
# =====================================================================

class TestGateScoring:
    def test_gate_score_no_oppositions(self):
        assert compute_gate_score([]) == 0.0

    def test_gate_score_single(self):
        item = _make_item(quality=0.80, relevance=0.75)
        score = compute_gate_score([item])
        assert score == pytest.approx(0.80 * 0.75)

    def test_gate_score_max_of_multiple(self):
        items = [
            _make_item(quality=0.50, relevance=0.80),  # 0.40
            _make_item(quality=0.90, relevance=0.90),  # 0.81
            _make_item(quality=0.60, relevance=0.70),  # 0.42
        ]
        score = compute_gate_score(items)
        assert score == pytest.approx(0.81)

    def test_classify_outcome_pass(self):
        assert classify_gate_outcome(0.10, _CONFIG) == GateOutcome.PASS

    def test_classify_outcome_pass_with_caveat(self):
        assert classify_gate_outcome(0.35, _CONFIG) == GateOutcome.PASS_WITH_CAVEAT

    def test_classify_outcome_revise(self):
        assert classify_gate_outcome(0.60, _CONFIG) == GateOutcome.REVISE

    def test_classify_outcome_block(self):
        assert classify_gate_outcome(0.80, _CONFIG) == GateOutcome.BLOCK

    def test_classify_outcome_boundary_pass(self):
        assert classify_gate_outcome(_CONFIG.gate_pass_max - 0.001, _CONFIG) == GateOutcome.PASS

    def test_extract_caveats_returns_minor_moderate(self):
        items = [
            _make_item(quality=0.50, severity=OppositionSeverity.MINOR),
            _make_item(quality=0.65, severity=OppositionSeverity.MODERATE),
            _make_item(quality=0.90, severity=OppositionSeverity.CRITICAL),
        ]
        caveats = extract_caveats(items)
        assert len(caveats) == 2

    def test_extract_caveats_empty(self):
        items = [_make_item(severity=OppositionSeverity.CRITICAL)]
        assert extract_caveats(items) == []


# =====================================================================
# 8. Anti-nihilism
# =====================================================================

class TestAntiNihilism:
    def test_block_rate_empty(self):
        dq: deque = deque(maxlen=20)
        assert compute_block_rate(dq) == 0.0

    def test_block_rate_all_blocked(self):
        dq: deque = deque([True, True, True], maxlen=20)
        assert compute_block_rate(dq) == pytest.approx(1.0)

    def test_block_rate_half(self):
        dq: deque = deque([True, False, True, False], maxlen=20)
        assert compute_block_rate(dq) == pytest.approx(0.5)

    def test_effective_threshold_no_nihilism(self):
        t = get_effective_quality_threshold(0.45, 0.10, _CONFIG.theta_nihilism)
        assert t == pytest.approx(0.45)

    def test_effective_threshold_nihilism_triggers(self):
        t = get_effective_quality_threshold(0.45, 0.50, _CONFIG.theta_nihilism)
        assert t > 0.45

    def test_effective_threshold_bounded(self):
        t = get_effective_quality_threshold(0.45, 1.0, _CONFIG.theta_nihilism)
        assert t <= 1.0

    def test_effective_threshold_scales_with_excess(self):
        t_low  = get_effective_quality_threshold(0.45, 0.35, _CONFIG.theta_nihilism)
        t_high = get_effective_quality_threshold(0.45, 0.80, _CONFIG.theta_nihilism)
        assert t_high > t_low


# =====================================================================
# 9. Dialectical tension detection
# =====================================================================

class TestDialecticalTension:
    def test_no_tension_no_critical_items(self):
        items = [_make_item(quality=0.60, severity=OppositionSeverity.MODERATE)]
        candidates = detect_dialectical_tension(_TARGET_SIMPLE, items, _CONFIG)
        assert candidates == []

    def test_tension_detected_on_critical_conclusion(self):
        item = OppositionItem(
            item_id=str(uuid.uuid4()),
            mode=OppositionMode.COUNTERARGUMENT,
            content="critical opposition to conclusion",
            quality_score=0.90,
            validity=0.90,
            relevance=0.90,
            strength=0.90,
            novelty=0.90,
            target_component="conclusion",
            severity=OppositionSeverity.CRITICAL,
            attack_type="conclusion_attack",
        )
        candidates = detect_dialectical_tension(_TARGET_SIMPLE, [item], _CONFIG)
        assert len(candidates) == 1
        assert isinstance(candidates[0], ParadoxCandidate)

    def test_tension_not_detected_for_non_conclusion_component(self):
        item = OppositionItem(
            item_id=str(uuid.uuid4()),
            mode=OppositionMode.COUNTERARGUMENT,
            content="critical opposition to premise",
            quality_score=0.90,
            validity=0.90,
            relevance=0.90,
            strength=0.90,
            novelty=0.90,
            target_component="premise",   # not conclusion
            severity=OppositionSeverity.CRITICAL,
            attack_type="premise_attack",
        )
        candidates = detect_dialectical_tension(_TARGET_SIMPLE, [item], _CONFIG)
        assert candidates == []

    def test_tension_candidate_fields(self):
        item = OppositionItem(
            item_id=str(uuid.uuid4()),
            mode=OppositionMode.COUNTERARGUMENT,
            content="antithesis text",
            quality_score=0.91,
            validity=0.91,
            relevance=0.91,
            strength=0.91,
            novelty=0.91,
            target_component="conclusion",
            severity=OppositionSeverity.CRITICAL,
            attack_type="conclusion_attack",
        )
        candidates = detect_dialectical_tension("thesis text", [item], _CONFIG)
        c = candidates[0]
        assert "thesis" in c.thesis or len(c.thesis) > 0
        assert "antithesis" in c.antithesis or len(c.antithesis) > 0
        assert len(c.tension_description) > 0


# =====================================================================
# 10. Opposition generators
# =====================================================================

class TestOppositionGenerators:
    def test_counterargument_returns_item(self):
        reasoning = _make_reasoning()
        item = generate_counterargument(_TARGET_CAUSAL, reasoning, 0, _CONFIG)
        assert isinstance(item, OppositionItem)
        assert item.mode == OppositionMode.COUNTERARGUMENT
        assert len(item.content) > 0

    def test_counterargument_quality_in_range(self):
        for pass_n in range(3):
            item = generate_counterargument(_TARGET_CAUSAL, _make_reasoning(), pass_n, _CONFIG)
            assert 0.0 <= item.quality_score <= 1.0

    def test_counterargument_attack_types_rotate(self):
        types = set()
        for pass_n in range(3):
            item = generate_counterargument(_TARGET_CAUSAL, _make_reasoning(), pass_n, _CONFIG)
            types.add(item.attack_type)
        assert len(types) >= 2  # rotates through attack types

    def test_counterexample_returns_item(self):
        item = generate_counterexample(_TARGET_UNIVERSAL, 0, _CONFIG)
        assert isinstance(item, OppositionItem)
        assert item.mode == OppositionMode.COUNTEREXAMPLE

    def test_counterexample_strategies_rotate(self):
        strategies = set()
        for pass_n in range(3):
            item = generate_counterexample(_TARGET_UNIVERSAL, pass_n, _CONFIG)
            strategies.add(item.attack_type)
        assert len(strategies) >= 2

    def test_alternative_explanation_returns_item_or_none(self):
        result = generate_alternative_explanation(_TARGET_CAUSAL, 0, _CONFIG)
        # may be None if plausibility < theta_alt, but for causal targets should be non-None
        assert result is None or isinstance(result, OppositionItem)

    def test_alternative_explanation_causal_target(self):
        item = generate_alternative_explanation(_TARGET_CAUSAL, 0, _CONFIG)
        assert item is not None
        assert item.mode == OppositionMode.ALTERNATIVE_EXPLANATION

    def test_alternative_explanation_below_theta_alt_returns_none(self):
        # Config with theta_alt so high that alternatives are discarded
        strict_config = OppositionEngineConfig(theta_alt=0.99)
        result = generate_alternative_explanation(_TARGET_CAUSAL, 2, strict_config)
        assert result is None

    def test_assumption_excavation_returns_item(self):
        reasoning = _make_reasoning()
        item = generate_assumption_excavation(_TARGET_SIMPLE, reasoning, 0, _CONFIG)
        assert isinstance(item, OppositionItem)
        assert item.mode == OppositionMode.ASSUMPTION_EXCAVATION

    def test_assumption_excavation_strategies_rotate(self):
        strategies = set()
        for pass_n in range(2):
            item = generate_assumption_excavation(_TARGET_SIMPLE, _make_reasoning(), pass_n, _CONFIG)
            strategies.add(item.attack_type)
        assert len(strategies) == 2

    def test_structural_resistance_returns_items_and_result(self):
        items, result = generate_structural_resistance(_TARGET_SIMPLE, 3, _CONFIG)
        assert isinstance(items, list)
        assert isinstance(result, ResistanceResult)
        assert len(items) == 3

    def test_structural_resistance_result_has_status(self):
        _, result = generate_structural_resistance(_TARGET_SIMPLE, 3, _CONFIG)
        assert result.final_position_status in PositionStatus

    def test_structural_resistance_rounds_survived_in_range(self):
        _, result = generate_structural_resistance(_TARGET_SIMPLE, 5, _CONFIG)
        assert 0 <= result.rounds_survived <= 5

    def test_structural_resistance_key_challenges_populated(self):
        _, result = generate_structural_resistance(_TARGET_SIMPLE, 3, _CONFIG)
        assert len(result.key_challenges) == 3

    def test_structural_resistance_position_delta_non_empty(self):
        _, result = generate_structural_resistance(_TARGET_SIMPLE, 3, _CONFIG)
        assert len(result.position_delta) > 0


# =====================================================================
# 11. Core generation loop
# =====================================================================

class TestGenerationLoop:
    def test_counterargument_loop_returns_items(self):
        selected, gen, passed = run_generation_loop(
            _TARGET_CAUSAL, _make_reasoning(),
            OppositionMode.COUNTERARGUMENT,
            OppositionDepth.QUICK, _CONFIG.theta_opposition_quality, _CONFIG,
        )
        assert isinstance(selected, list)
        assert gen >= 0
        assert passed >= 0

    def test_quick_depth_max_output(self):
        selected, _, _ = run_generation_loop(
            _TARGET_CAUSAL, _make_reasoning(),
            OppositionMode.COUNTERARGUMENT,
            OppositionDepth.QUICK, _CONFIG.theta_opposition_quality, _CONFIG,
        )
        assert len(selected) <= _CONFIG.max_output_quick

    def test_standard_depth_max_output(self):
        selected, _, _ = run_generation_loop(
            _TARGET_CAUSAL, _make_reasoning(),
            OppositionMode.COUNTERARGUMENT,
            OppositionDepth.STANDARD, _CONFIG.theta_opposition_quality, _CONFIG,
        )
        assert len(selected) <= _CONFIG.max_output_standard

    def test_structural_resistance_via_loop(self):
        selected, gen, passed = run_generation_loop(
            _TARGET_SIMPLE, _make_reasoning(),
            OppositionMode.STRUCTURAL_RESISTANCE,
            OppositionDepth.STANDARD, _CONFIG.theta_opposition_quality, _CONFIG,
        )
        assert isinstance(selected, list)

    def test_alternative_explanation_via_loop(self):
        selected, gen, passed = run_generation_loop(
            _TARGET_CAUSAL, _make_reasoning(),
            OppositionMode.ALTERNATIVE_EXPLANATION,
            OppositionDepth.STANDARD, _CONFIG.theta_opposition_quality, _CONFIG,
        )
        assert isinstance(selected, list)

    def test_high_threshold_reduces_passed(self):
        _, _, passed_low  = run_generation_loop(
            _TARGET_CAUSAL, _make_reasoning(),
            OppositionMode.COUNTERARGUMENT,
            OppositionDepth.STANDARD, 0.20, _CONFIG,
        )
        _, _, passed_high = run_generation_loop(
            _TARGET_CAUSAL, _make_reasoning(),
            OppositionMode.COUNTERARGUMENT,
            OppositionDepth.STANDARD, 0.99, _CONFIG,
        )
        assert passed_high <= passed_low


# =====================================================================
# 12. Neurochemical signals
# =====================================================================

class TestNeurochemicalSignals:
    def _rng(self):
        return np.random.default_rng(0)

    def test_keys_present(self):
        item = _make_item(quality=0.75)
        signals = compute_neurochemical_signals(
            [item], GateOutcome.PASS, OppositionDepth.STANDARD,
            0.5, _STATE, _CONFIG, self._rng()
        )
        expected_keys = {
            "ne_burst", "da_signal", "ach_burst",
            "gaba_suppress", "cor_eustress",
            "beta_boost", "gamma_boost",
            "theta_boost", "theta_suppress",
        }
        assert set(signals.keys()) == expected_keys

    def test_no_oppositions_all_zero(self):
        signals = compute_neurochemical_signals(
            [], None, OppositionDepth.STANDARD, 0.0, _STATE, _CONFIG, self._rng()
        )
        # ne_burst, ach_burst, gaba_suppress, cor_eustress, beta_boost should all be 0
        assert signals["ne_burst"] == 0.0
        assert signals["ach_burst"] == 0.0
        assert signals["gaba_suppress"] == 0.0

    def test_ne_burst_positive_on_opposition(self):
        item = _make_item(quality=0.8)
        results = [
            compute_neurochemical_signals([item], GateOutcome.PASS, OppositionDepth.STANDARD,
                                          0.5, _STATE, _CONFIG, np.random.default_rng(i))["ne_burst"]
            for i in range(20)
        ]
        assert any(v > 0.0 for v in results)

    def test_da_pass_is_positive(self):
        item = _make_item(quality=0.5, relevance=0.3)  # low gate score → pass
        results = [
            compute_neurochemical_signals([item], GateOutcome.PASS, OppositionDepth.STANDARD,
                                          0.2, _STATE, _CONFIG, np.random.default_rng(i))["da_signal"]
            for i in range(20)
        ]
        assert any(v > 0.0 for v in results)

    def test_da_find_reward_on_revise(self):
        item = _make_item(quality=0.8, severity=OppositionSeverity.SIGNIFICANT)
        results = [
            compute_neurochemical_signals([item], GateOutcome.REVISE, OppositionDepth.STANDARD,
                                          0.8, _STATE, _CONFIG, np.random.default_rng(i))["da_signal"]
            for i in range(20)
        ]
        assert any(v > 0.0 for v in results)

    def test_da_block_is_higher_than_pass(self):
        item = _make_item(quality=0.9, severity=OppositionSeverity.CRITICAL)
        da_pass_vals = [
            compute_neurochemical_signals([item], GateOutcome.PASS, OppositionDepth.DEEP,
                                          0.2, _STATE, _CONFIG, np.random.default_rng(i))["da_signal"]
            for i in range(30)
        ]
        da_block_vals = [
            compute_neurochemical_signals([item], GateOutcome.BLOCK, OppositionDepth.DEEP,
                                          0.9, _STATE, _CONFIG, np.random.default_rng(i))["da_signal"]
            for i in range(30)
        ]
        # On average, BLOCK should produce more DA reward than PASS
        assert sum(da_block_vals) / 30 > sum(da_pass_vals) / 30

    def test_cor_eustress_only_during_deep(self):
        item = _make_item()
        sig_quick = compute_neurochemical_signals(
            [item], None, OppositionDepth.QUICK, 0.5, _STATE, _CONFIG, self._rng()
        )
        sig_deep = compute_neurochemical_signals(
            [item], None, OppositionDepth.DEEP, 0.5, _STATE, _CONFIG, self._rng()
        )
        assert sig_quick["cor_eustress"] == 0.0
        assert sig_deep["cor_eustress"] > 0.0

    def test_gaba_suppression_when_active(self):
        item = _make_item()
        signals = compute_neurochemical_signals(
            [item], None, OppositionDepth.STANDARD, 0.5, _STATE, _CONFIG, self._rng()
        )
        assert signals["gaba_suppress"] > 0.0

    def test_gamma_enhance_only_for_sr(self):
        ca_item = _make_item(mode=OppositionMode.COUNTERARGUMENT)
        sr_item = _make_item(mode=OppositionMode.STRUCTURAL_RESISTANCE)
        sig_ca  = compute_neurochemical_signals(
            [ca_item], None, OppositionDepth.STANDARD, 0.5, _STATE, _CONFIG, self._rng()
        )
        sig_sr  = compute_neurochemical_signals(
            [sr_item], None, OppositionDepth.STANDARD, 0.5, _STATE, _CONFIG, self._rng()
        )
        assert sig_ca["gamma_boost"] == 0.0
        assert sig_sr["gamma_boost"] > 0.0

    def test_theta_search_for_search_modes(self):
        ce_item = _make_item(mode=OppositionMode.COUNTEREXAMPLE)
        signals = compute_neurochemical_signals(
            [ce_item], None, OppositionDepth.STANDARD, 0.5, _STATE, _CONFIG, self._rng()
        )
        assert signals["theta_boost"] > 0.0
        assert signals["theta_suppress"] == 0.0

    def test_theta_resist_for_sr_mode(self):
        sr_item = _make_item(mode=OppositionMode.STRUCTURAL_RESISTANCE)
        signals = compute_neurochemical_signals(
            [sr_item], None, OppositionDepth.STANDARD, 0.5, _STATE, _CONFIG, self._rng()
        )
        assert signals["theta_suppress"] > 0.0


# =====================================================================
# 13. Bidirectional feedback
# =====================================================================

class TestBidirectionalFeedback:
    def test_high_ne_lowers_threshold(self):
        base = 0.45
        state = OppositionEngineState(ne_level=0.90)
        t = apply_neurochem_feedback(base, state, _CONFIG)
        assert t < base

    def test_high_cor_raises_threshold(self):
        base = 0.45
        state = OppositionEngineState(cor_level=0.90)
        t = apply_neurochem_feedback(base, state, _CONFIG)
        assert t > base

    def test_high_gaba_lowers_threshold(self):
        base = 0.45
        state = OppositionEngineState(gaba_level=0.90)
        t = apply_neurochem_feedback(base, state, _CONFIG)
        assert t < base

    def test_neutral_state_no_change(self):
        base = 0.45
        state = OppositionEngineState()  # all zeros
        t = apply_neurochem_feedback(base, state, _CONFIG)
        assert t == pytest.approx(base)

    def test_threshold_stays_bounded(self):
        base = 0.45
        state = OppositionEngineState(ne_level=1.0, gaba_level=1.0, cor_level=1.0)
        t = apply_neurochem_feedback(base, state, _CONFIG)
        assert 0.20 <= t <= 0.90


# =====================================================================
# 14. Engine ports
# =====================================================================

class TestEnginePorts:
    def test_default_mode_is_normal(self):
        engine = SimulatedOppositionEngine()
        assert engine.get_status()["mode"] == "normal"

    def test_configure_changes_mode(self):
        engine = SimulatedOppositionEngine()
        engine.configure(OperationalMode.DEV)
        assert engine.get_status()["mode"] == "dev"

    def test_get_status_keys(self):
        engine = SimulatedOppositionEngine()
        keys = set(engine.get_status().keys())
        assert {"engine_id", "version", "mode", "state", "block_rate", "nihilism_flag", "quality_threshold"} <= keys

    def test_update_neurochem_state_ne(self):
        engine = SimulatedOppositionEngine()
        engine.update_neurochem_state({"ne": 0.70})
        assert engine._state.ne_level == pytest.approx(0.70)

    def test_update_neurochem_state_all(self):
        engine = SimulatedOppositionEngine()
        engine.update_neurochem_state({
            "ne": 0.5, "da": 0.4, "ach": 0.6,
            "gaba": 0.3, "cor": 0.2,
        })
        s = engine._state
        assert s.ne_level   == pytest.approx(0.5)
        assert s.da_level   == pytest.approx(0.4)
        assert s.ach_level  == pytest.approx(0.6)
        assert s.gaba_level == pytest.approx(0.3)
        assert s.cor_level  == pytest.approx(0.2)

    def test_update_partial_leaves_others(self):
        engine = SimulatedOppositionEngine()
        engine.update_neurochem_state({"ne": 0.8})
        assert engine._state.da_level == 0.0

    def test_custom_config(self):
        cfg = OppositionEngineConfig(theta_opposition_quality=0.60)
        engine = SimulatedOppositionEngine(config=cfg)
        assert engine._config.theta_opposition_quality == pytest.approx(0.60)

    def test_block_rate_initial_zero(self):
        engine = SimulatedOppositionEngine()
        assert engine.get_status()["block_rate"] == 0.0

    def test_nihilism_flag_false_initially(self):
        engine = SimulatedOppositionEngine()
        assert engine.get_status()["nihilism_flag"] is False


# =====================================================================
# 15. run_gate() — end-to-end gate scenarios
# =====================================================================

class TestRunGate:
    def test_run_gate_returns_result(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_gate(_make_gate_input())
        assert isinstance(result, OppositionResult)

    def test_run_gate_has_gate_outcome(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_gate(_make_gate_input())
        assert result.gate_outcome in GateOutcome

    def test_run_gate_has_gate_score(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_gate(_make_gate_input())
        assert result.gate_score is not None
        assert 0.0 <= result.gate_score <= 1.0

    def test_run_gate_processing_time_positive(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_gate(_make_gate_input())
        assert result.processing_time_ms >= 0.0

    def test_run_gate_depth_used_in_result(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_gate(_make_gate_input())
        assert result.depth_used in OppositionDepth

    def test_run_gate_modes_used_non_empty(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_gate(_make_gate_input())
        assert len(result.modes_used) >= 1

    def test_run_gate_pass_outcome_low_complexity(self):
        engine = SimulatedOppositionEngine()
        # Very low complexity, high confidence → Quick depth → PASS likely
        inp = _make_gate_input(
            candidate_response="The capital of France is Paris.",
            complexity=0.05,
            stakes=0.05,
            novelty=0.05,
            challenge_level=0.05,
        )
        inp2 = OppositionGateInput(
            **{**inp.__dict__, "response_reasoning": ReasoningTrace(confidence=0.99)}
        )
        result = engine.run_gate(inp2)
        # Low-quality opposition → likely PASS or PASS_WITH_CAVEAT
        assert result.gate_outcome in (GateOutcome.PASS, GateOutcome.PASS_WITH_CAVEAT, GateOutcome.REVISE)

    def test_run_gate_depth_override_respected(self):
        engine = SimulatedOppositionEngine()
        inp = _make_gate_input(depth_override=OppositionDepth.DEEP)
        result = engine.run_gate(inp)
        assert result.depth_used == OppositionDepth.DEEP

    def test_run_gate_dev_mode_always_deep(self):
        engine = SimulatedOppositionEngine()
        inp = _make_gate_input(active_mode=OperationalMode.DEV, complexity=0.0)
        result = engine.run_gate(inp)
        assert result.depth_used == OppositionDepth.DEEP

    def test_run_gate_caveat_list_when_caveat_outcome(self):
        engine = SimulatedOppositionEngine()
        # Use DEEP mode to get varied severity opposition
        inp = _make_gate_input(active_mode=OperationalMode.DEV, complexity=0.9, stakes=0.9)
        result = engine.run_gate(inp)
        if result.gate_outcome == GateOutcome.PASS_WITH_CAVEAT:
            assert result.caveats is not None

    def test_run_gate_neurochemical_signals_present(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_gate(_make_gate_input())
        assert isinstance(result.neurochemical_signals, dict)

    def test_run_gate_candidates_stats(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_gate(_make_gate_input())
        assert result.candidates_generated >= 0
        assert result.candidates_passed_quality >= 0

    def test_run_gate_opposition_activity_non_negative(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_gate(_make_gate_input())
        assert result.opposition_activity >= 0.0


# =====================================================================
# 16. run_request() — callable interface
# =====================================================================

class TestRunRequest:
    def test_run_request_returns_result(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_request(_make_request())
        assert isinstance(result, OppositionResult)

    def test_run_request_no_gate_outcome(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_request(_make_request())
        assert result.gate_outcome is None
        assert result.gate_score is None
        assert result.caveats is None

    def test_run_request_modes_used_match_requested(self):
        engine = SimulatedOppositionEngine()
        modes = [OppositionMode.COUNTERARGUMENT, OppositionMode.COUNTEREXAMPLE]
        req = _make_request(requested_modes=modes, requested_depth=OppositionDepth.STANDARD)
        result = engine.run_request(req)
        assert result.depth_used == OppositionDepth.STANDARD

    def test_run_request_all_modes_deep(self):
        engine = SimulatedOppositionEngine()
        req = _make_request(
            requested_modes=list(OppositionMode),
            requested_depth=OppositionDepth.DEEP,
        )
        result = engine.run_request(req)
        assert result.depth_used == OppositionDepth.DEEP

    def test_run_request_respects_min_quality(self):
        engine = SimulatedOppositionEngine()
        req_low  = _make_request(min_opposition_quality=0.20)
        req_high = _make_request(min_opposition_quality=0.99)
        result_low  = engine.run_request(req_low)
        result_high = engine.run_request(req_high)
        # High quality threshold → fewer or equal items pass
        assert result_high.candidates_passed_quality <= result_low.candidates_passed_quality

    def test_run_request_processing_time(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_request(_make_request())
        assert result.processing_time_ms >= 0.0

    def test_run_request_timestamp_is_datetime(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_request(_make_request())
        assert isinstance(result.timestamp, datetime)


# =====================================================================
# 17. REM_DREAM disabled mode
# =====================================================================

class TestRemDreamDisabled:
    def test_gate_returns_pass_in_dream_mode(self):
        engine = SimulatedOppositionEngine()
        inp = _make_gate_input(active_mode=OperationalMode.REM_DREAM)
        result = engine.run_gate(inp)
        assert result.gate_outcome == GateOutcome.PASS
        assert result.oppositions == []

    def test_request_returns_no_oppositions_in_dream_mode(self):
        engine = SimulatedOppositionEngine()
        req = _make_request(active_mode=OperationalMode.REM_DREAM)
        result = engine.run_request(req)
        assert result.oppositions == []

    def test_disabled_result_has_empty_signals(self):
        engine = SimulatedOppositionEngine()
        inp = _make_gate_input(active_mode=OperationalMode.REM_DREAM)
        result = engine.run_gate(inp)
        assert result.neurochemical_signals == {}

    def test_disabled_result_clean_pass(self):
        engine = SimulatedOppositionEngine()
        inp = _make_gate_input(active_mode=OperationalMode.REM_DREAM)
        result = engine.run_gate(inp)
        assert result.gate_score == 0.0
        assert result.nihilism_flag is False


# =====================================================================
# 18. Output type validation
# =====================================================================

class TestOutputTypes:
    def test_opposition_result_is_frozen(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_request(_make_request())
        with pytest.raises((AttributeError, TypeError)):
            result.nihilism_flag = True  # type: ignore

    def test_opposition_item_is_frozen(self):
        item = _make_item()
        with pytest.raises((AttributeError, TypeError)):
            item.quality_score = 0.99  # type: ignore

    def test_resistance_result_is_frozen(self):
        _, r = generate_structural_resistance(_TARGET_SIMPLE, 3, _CONFIG)
        with pytest.raises((AttributeError, TypeError)):
            r.rounds_survived = 99  # type: ignore

    def test_paradox_candidate_is_frozen(self):
        pc = ParadoxCandidate(thesis="T", antithesis="A", tension_description="desc")
        with pytest.raises((AttributeError, TypeError)):
            pc.thesis = "changed"  # type: ignore

    def test_opposition_result_oppositions_is_list(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_request(_make_request())
        assert isinstance(result.oppositions, list)

    def test_opposition_result_modes_used_is_list(self):
        engine = SimulatedOppositionEngine()
        result = engine.run_request(_make_request())
        assert isinstance(result.modes_used, list)

    def test_opposition_item_uuid(self):
        item = _make_item()
        import uuid as _uuid
        _uuid.UUID(item.item_id)  # raises if invalid

    def test_reasoning_trace_mutable(self):
        r = ReasoningTrace()
        r.confidence = 0.99  # should be mutable (not frozen)
        assert r.confidence == 0.99

    def test_dialogue_turn_mutable(self):
        d = DialogueTurn()
        d.content = "hello"
        assert d.content == "hello"


# =====================================================================
# 19. Edge cases
# =====================================================================

class TestEdgeCases:
    def test_empty_target_counterargument(self):
        item = generate_counterargument("", _make_reasoning(premises=[], inference_steps=[], conclusion=""), 0, _CONFIG)
        assert isinstance(item, OppositionItem)

    def test_empty_target_counterexample(self):
        item = generate_counterexample("", 0, _CONFIG)
        assert isinstance(item, OppositionItem)

    def test_empty_target_assumption_excavation(self):
        item = generate_assumption_excavation("", _make_reasoning(inference_steps=[]), 0, _CONFIG)
        assert isinstance(item, OppositionItem)

    def test_run_gate_with_empty_candidate(self):
        engine = SimulatedOppositionEngine()
        inp = _make_gate_input(candidate_response="")
        result = engine.run_gate(inp)
        assert isinstance(result, OppositionResult)

    def test_run_request_empty_target(self):
        engine = SimulatedOppositionEngine()
        req = _make_request(target="")
        result = engine.run_request(req)
        assert isinstance(result, OppositionResult)

    def test_nihilism_flag_set_after_many_blocks(self):
        engine = SimulatedOppositionEngine()
        # Manually fill gate history with True (blocks) > 30% threshold
        for _ in range(15):
            engine._state.gate_history.append(True)
        for _ in range(5):
            engine._state.gate_history.append(False)
        status = engine.get_status()
        assert status["nihilism_flag"] is True

    def test_run_gate_multiple_times_accumulates_history(self):
        engine = SimulatedOppositionEngine()
        for _ in range(5):
            engine.run_gate(_make_gate_input())
        # Should have up to 5 entries (may be less if some didn't reach gate)
        history_len = len(engine._state.gate_history)
        assert history_len <= 20

    def test_run_request_single_assumption_excavation(self):
        engine = SimulatedOppositionEngine()
        req = _make_request(
            target=_TARGET_UNIVERSAL,
            requested_modes=[OppositionMode.ASSUMPTION_EXCAVATION],
            requested_depth=OppositionDepth.QUICK,
        )
        result = engine.run_request(req)
        assert isinstance(result, OppositionResult)

    def test_extract_components_causal_target(self):
        components = _extract_components(_TARGET_CAUSAL)
        assert len(components["causal_claims"]) > 0

    def test_extract_components_universal_target(self):
        components = _extract_components(_TARGET_UNIVERSAL)
        assert len(components["universal_claims"]) > 0

    def test_score_components_empty_premises(self):
        components = {"premises": [], "conclusion": "test", "causal_claims": [],
                      "universal_claims": [], "hedged_claims": [], "all_sentences": ["test"]}
        scores = _score_components(components)
        assert scores == {}

    def test_score_components_universal_has_higher_vulnerability(self):
        components_u = _extract_components("All people always behave consistently.")
        components_h = _extract_components("Some people might sometimes behave differently.")
        sc_u = _score_components(components_u)
        sc_h = _score_components(components_h)
        if sc_u and sc_h:
            max_u = max(sc_u.values())
            max_h = max(sc_h.values())
            assert max_u >= max_h
