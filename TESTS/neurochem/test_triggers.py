"""
Tests for conditional trigger evaluation (Appendix K.6).

Phase 33: Condition evaluator, namespace builder, trigger evaluation.
"""

import pytest

from zados.neurochem.neurosymbolic.triggers import (
    TriggerDefinition,
    TriggerResult,
    evaluate_condition,
    build_variable_namespace,
    evaluate_trigger,
    evaluate_all_triggers,
)


# ---------------------------------------------------------------------------
# Condition evaluator
# ---------------------------------------------------------------------------

class TestEvaluateCondition:
    """Tests for the recursive descent condition evaluator."""

    def test_simple_gt(self):
        assert evaluate_condition("x>0.5", {"x": 0.7}) is True

    def test_simple_gt_false(self):
        assert evaluate_condition("x>0.5", {"x": 0.3}) is False

    def test_simple_lt(self):
        assert evaluate_condition("x<0.5", {"x": 0.3}) is True

    def test_gte(self):
        assert evaluate_condition("x>=0.5", {"x": 0.5}) is True

    def test_lte(self):
        assert evaluate_condition("x<=0.5", {"x": 0.5}) is True

    def test_eq(self):
        assert evaluate_condition("x==0.5", {"x": 0.5}) is True

    def test_neq(self):
        assert evaluate_condition("x!=0.5", {"x": 0.6}) is True

    def test_and_both_true(self):
        assert evaluate_condition("x>0.5 AND y>0.5", {"x": 0.7, "y": 0.8}) is True

    def test_and_one_false(self):
        assert evaluate_condition("x>0.5 AND y>0.5", {"x": 0.7, "y": 0.3}) is False

    def test_or_one_true(self):
        assert evaluate_condition("x>0.5 OR y>0.5", {"x": 0.7, "y": 0.3}) is True

    def test_or_both_false(self):
        assert evaluate_condition("x>0.5 OR y>0.5", {"x": 0.3, "y": 0.2}) is False

    def test_not(self):
        assert evaluate_condition("NOT x>0.5", {"x": 0.3}) is True

    def test_not_true(self):
        assert evaluate_condition("NOT x>0.5", {"x": 0.7}) is False

    def test_nested_parens(self):
        assert evaluate_condition(
            "(x>0.5 AND y>0.5) OR z>0.9",
            {"x": 0.3, "y": 0.3, "z": 0.95},
        ) is True

    def test_complex_condition(self):
        assert evaluate_condition(
            "beta>0.6 AND S_DA-D2>0.7",
            {"beta": 0.8, "S_DA-D2": 0.9},
        ) is True

    def test_missing_variable_zero(self):
        """Missing variables default to 0.0."""
        assert evaluate_condition("x>0.5", {}) is False  # 0.0 > 0.5 → False

    def test_missing_variable_lt(self):
        assert evaluate_condition("x<0.5", {}) is True  # 0.0 < 0.5 → True

    def test_hyphenated_variable(self):
        """Variables with hyphens work (e.g., S_DA-D2)."""
        assert evaluate_condition("S_DA-D2>0.5", {"S_DA-D2": 0.8}) is True


# ---------------------------------------------------------------------------
# Variable namespace builder
# ---------------------------------------------------------------------------

class TestBuildNamespace:
    """Tests for build_variable_namespace."""

    def test_concentrations(self):
        ns = build_variable_namespace(concentrations={"DA": 0.5, "5HT": 0.3})
        assert ns["C_DA"] == 0.5
        assert ns["DA"] == 0.5
        assert ns["C_5HT"] == 0.3

    def test_saturations(self):
        ns = build_variable_namespace(saturations={"DA_D1": 0.7})
        assert ns["S_DA_D1"] == 0.7
        assert ns["S_DA-D1"] == 0.7  # hyphen variant

    def test_oscillations(self):
        ns = build_variable_namespace(oscillations={"theta": 0.4, "theta_gamma": 0.2})
        assert ns["phi_theta"] == 0.4
        assert ns["theta"] == 0.4
        assert ns["phi_theta_gamma"] == 0.2
        assert ns["theta_gamma"] == 0.2

    def test_metrics(self):
        ns = build_variable_namespace(metrics={"Fatigue": 0.6, "LogicMode": 0.8})
        assert ns["Fatigue"] == 0.6
        assert ns["LogicMode"] == 0.8

    def test_empty(self):
        ns = build_variable_namespace()
        assert ns == {}

    def test_combined(self):
        ns = build_variable_namespace(
            concentrations={"DA": 0.5},
            saturations={"DA_D1": 0.7},
            oscillations={"beta": 0.4},
            metrics={"Fatigue": 0.6},
        )
        assert "C_DA" in ns
        assert "S_DA_D1" in ns
        assert "phi_beta" in ns
        assert "Fatigue" in ns


# ---------------------------------------------------------------------------
# Trigger evaluation
# ---------------------------------------------------------------------------

class TestEvaluateTrigger:
    """Tests for evaluate_trigger."""

    def test_trigger_fires(self):
        trig = TriggerDefinition(
            condition_str="beta>0.6",
            actions=("INT(D2)",),
            activate_mode="LogicMode",
        )
        result = evaluate_trigger(trig, {"beta": 0.8})
        assert result.fired is True
        assert result.mode == "LogicMode"
        assert result.actions == ("INT(D2)",)

    def test_trigger_no_fire(self):
        trig = TriggerDefinition(
            condition_str="beta>0.6",
            actions=("INT(D2)",),
            activate_mode="LogicMode",
        )
        result = evaluate_trigger(trig, {"beta": 0.3})
        assert result.fired is False
        assert result.mode is None
        assert result.actions == ()

    def test_trigger_with_multiple_actions(self):
        trig = TriggerDefinition(
            condition_str="Fatigue>0.7",
            actions=("INT(D2)", "DSN(D1)"),
            activate_mode="Containment",
        )
        result = evaluate_trigger(trig, {"Fatigue": 0.8})
        assert result.fired is True
        assert len(result.actions) == 2

    def test_trigger_no_activate(self):
        trig = TriggerDefinition(
            condition_str="x>0.5",
            actions=("DA->D2:UP_ACT",),
        )
        result = evaluate_trigger(trig, {"x": 0.7})
        assert result.fired is True
        assert result.mode is None

    def test_trigger_result_frozen(self):
        r = TriggerResult(fired=True)
        with pytest.raises(AttributeError):
            r.fired = False

    def test_trigger_definition_frozen(self):
        t = TriggerDefinition(condition_str="x>0", actions=())
        with pytest.raises(AttributeError):
            t.condition_str = "y>0"


# ---------------------------------------------------------------------------
# Evaluate all triggers
# ---------------------------------------------------------------------------

class TestEvaluateAllTriggers:
    """Tests for evaluate_all_triggers."""

    def test_evaluate_all(self):
        triggers = [
            TriggerDefinition(condition_str="x>0.5", actions=("A",), activate_mode="M1"),
            TriggerDefinition(condition_str="y>0.5", actions=("B",), activate_mode="M2"),
        ]
        results = evaluate_all_triggers(triggers, {"x": 0.7, "y": 0.3})
        assert results[0].fired is True
        assert results[1].fired is False

    def test_evaluate_all_empty(self):
        results = evaluate_all_triggers([], {})
        assert results == []


# ---------------------------------------------------------------------------
# K.6.7 integration examples from spec
# ---------------------------------------------------------------------------

class TestSpecExamples:
    """Integration tests matching K.6.7 specification examples."""

    def test_contradiction_logic_mode(self):
        """IF(beta>0.6 AND S_DA-D2>0.7) => ACTIVATE(LogicMode)."""
        ns = build_variable_namespace(
            oscillations={"beta": 0.8},
            saturations={"DA_D2": 0.9},
        )
        trig = TriggerDefinition(
            condition_str="beta>0.6 AND S_DA-D2>0.7",
            actions=(),
            activate_mode="LogicMode",
        )
        result = evaluate_trigger(trig, ns)
        assert result.fired is True
        assert result.mode == "LogicMode"

    def test_theta_gamma_recursive_synthesis(self):
        """IF(theta_gamma>0.55 AND S_NMDA>0.5) => ACTIVATE(RecursiveSynthesis)."""
        ns = build_variable_namespace(
            oscillations={"theta_gamma": 0.6},
            saturations={"NMDA": 0.7},
        )
        trig = TriggerDefinition(
            condition_str="theta_gamma>0.55 AND S_NMDA>0.5",
            actions=(),
            activate_mode="RecursiveSynthesis",
        )
        result = evaluate_trigger(trig, ns)
        assert result.fired is True
        assert result.mode == "RecursiveSynthesis"

    def test_fatigue_containment(self):
        """IF(Fatigue>0.7) => INT(D2); ACTIVATE(Containment)."""
        ns = build_variable_namespace(
            metrics={"Fatigue": 0.8},
        )
        trig = TriggerDefinition(
            condition_str="Fatigue>0.7",
            actions=("INT(D2)",),
            activate_mode="Containment",
        )
        result = evaluate_trigger(trig, ns)
        assert result.fired is True
        assert result.mode == "Containment"
        assert "INT(D2)" in result.actions

    def test_fatigue_containment_not_firing(self):
        """Fatigue below threshold should not fire."""
        ns = build_variable_namespace(
            metrics={"Fatigue": 0.5},
        )
        trig = TriggerDefinition(
            condition_str="Fatigue>0.7",
            actions=("INT(D2)",),
            activate_mode="Containment",
        )
        result = evaluate_trigger(trig, ns)
        assert result.fired is False
