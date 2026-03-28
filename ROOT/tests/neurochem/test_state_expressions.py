"""
Tests for composite state expressions (Appendix K.5).

Phase 32: StateTerm, StateDefinition, variable resolution, evaluation.
"""

import pytest
import math

from zados.neurochem.neurosymbolic.state_expressions import (
    StateTerm,
    StateDefinition,
    resolve_term,
    evaluate_state,
    evaluate_all_states,
)


# ---------------------------------------------------------------------------
# Frozen dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    """Tests for frozen dataclasses."""

    def test_term_frozen(self):
        t = StateTerm(weight=0.5, variable="S_DA_D1")
        with pytest.raises(AttributeError):
            t.weight = 1.0

    def test_definition_frozen(self):
        d = StateDefinition(name="X", terms=())
        with pytest.raises(AttributeError):
            d.name = "Y"

    def test_definition_defaults(self):
        d = StateDefinition(name="X", terms=())
        assert d.bias == 0.0
        assert d.bounding == "clip"


# ---------------------------------------------------------------------------
# Variable resolution
# ---------------------------------------------------------------------------

class TestResolveTerm:
    """Tests for resolve_term."""

    def test_resolve_saturation(self):
        term = StateTerm(weight=0.5, variable="S_DA_D1")
        val = resolve_term(
            term,
            saturations={"DA_D1": 0.8},
            concentrations={},
            oscillations={},
        )
        assert val == pytest.approx(0.4)  # 0.5 * 0.8

    def test_resolve_concentration(self):
        term = StateTerm(weight=0.4, variable="C_DA")
        val = resolve_term(
            term,
            saturations={},
            concentrations={"DA": 0.6},
            oscillations={},
        )
        assert val == pytest.approx(0.24)  # 0.4 * 0.6

    def test_resolve_oscillation(self):
        term = StateTerm(weight=0.3, variable="phi_theta")
        val = resolve_term(
            term,
            saturations={},
            concentrations={},
            oscillations={"theta": 0.5},
        )
        assert val == pytest.approx(0.15)  # 0.3 * 0.5

    def test_resolve_gated_product(self):
        """phi_theta * S_NMDA → oscillation * saturation."""
        term = StateTerm(weight=1.0, variable="phi_theta*S_NMDA")
        val = resolve_term(
            term,
            saturations={"NMDA": 0.6},
            concentrations={},
            oscillations={"theta": 0.8},
        )
        assert val == pytest.approx(0.48)  # 1.0 * 0.8 * 0.6

    def test_resolve_cfc_gated(self):
        """phi_theta_gamma * S_NMDA → CFC oscillation * saturation."""
        term = StateTerm(weight=0.5, variable="phi_theta_gamma*S_NMDA")
        val = resolve_term(
            term,
            saturations={"NMDA": 0.4},
            concentrations={},
            oscillations={"theta_gamma": 0.6},
        )
        assert val == pytest.approx(0.12)  # 0.5 * 0.6 * 0.4

    def test_resolve_missing_variable_zero(self):
        term = StateTerm(weight=1.0, variable="S_UNKNOWN")
        val = resolve_term(
            term,
            saturations={},
            concentrations={},
            oscillations={},
        )
        assert val == pytest.approx(0.0)

    def test_resolve_negative_weight(self):
        term = StateTerm(weight=-0.3, variable="S_CB1")
        val = resolve_term(
            term,
            saturations={"CB1": 1.0},
            concentrations={},
            oscillations={},
        )
        assert val == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluateState:
    """Tests for evaluate_state."""

    def test_evaluate_simple(self):
        defn = StateDefinition(
            name="TestState",
            terms=(StateTerm(weight=1.0, variable="S_DA_D1"),),
        )
        val = evaluate_state(
            defn,
            saturations={"DA_D1": 0.7},
            concentrations={},
            oscillations={},
        )
        assert val == pytest.approx(0.7)

    def test_evaluate_with_bias(self):
        defn = StateDefinition(
            name="TestState",
            terms=(StateTerm(weight=0.5, variable="S_DA_D1"),),
            bias=0.1,
        )
        val = evaluate_state(
            defn,
            saturations={"DA_D1": 0.6},
            concentrations={},
            oscillations={},
        )
        # 0.1 + 0.5 * 0.6 = 0.4
        assert val == pytest.approx(0.4)

    def test_evaluate_clip_upper(self):
        defn = StateDefinition(
            name="TestState",
            terms=(
                StateTerm(weight=1.0, variable="S_A"),
                StateTerm(weight=1.0, variable="S_B"),
            ),
        )
        val = evaluate_state(
            defn,
            saturations={"A": 0.8, "B": 0.8},
            concentrations={},
            oscillations={},
        )
        assert val == 1.0  # clipped from 1.6

    def test_evaluate_clip_lower(self):
        defn = StateDefinition(
            name="TestState",
            terms=(StateTerm(weight=-2.0, variable="S_A"),),
        )
        val = evaluate_state(
            defn,
            saturations={"A": 0.8},
            concentrations={},
            oscillations={},
        )
        assert val == 0.0  # clipped from -1.6

    def test_evaluate_logistic(self):
        defn = StateDefinition(
            name="TestState",
            terms=(StateTerm(weight=1.0, variable="S_A"),),
            bounding="logistic",
        )
        val = evaluate_state(
            defn,
            saturations={"A": 0.5},
            concentrations={},
            oscillations={},
        )
        # logistic(0.5) with k=10, x0=0.5 → 1/(1+exp(0)) = 0.5
        assert val == pytest.approx(0.5)

    def test_evaluate_affine_clip(self):
        defn = StateDefinition(
            name="TestState",
            terms=(StateTerm(weight=-0.5, variable="S_A"),),
            bounding="affine_clip",
        )
        val = evaluate_state(
            defn,
            saturations={"A": 1.0},
            concentrations={},
            oscillations={},
        )
        assert val == pytest.approx(-0.5)  # in [-1, 1]

    def test_evaluate_negative_weights(self):
        defn = StateDefinition(
            name="TestState",
            terms=(
                StateTerm(weight=0.5, variable="S_A"),
                StateTerm(weight=-0.3, variable="S_B"),
            ),
        )
        val = evaluate_state(
            defn,
            saturations={"A": 0.8, "B": 0.5},
            concentrations={},
            oscillations={},
        )
        # 0.5 * 0.8 + (-0.3) * 0.5 = 0.4 - 0.15 = 0.25
        assert val == pytest.approx(0.25)

    def test_evaluate_empty_terms(self):
        defn = StateDefinition(name="Empty", terms=(), bias=0.3)
        val = evaluate_state(defn, {}, {}, {})
        assert val == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Evaluate all states
# ---------------------------------------------------------------------------

class TestEvaluateAllStates:
    """Tests for evaluate_all_states."""

    def test_evaluate_all(self):
        defs = [
            StateDefinition(
                name="A", terms=(StateTerm(weight=1.0, variable="S_X"),),
            ),
            StateDefinition(
                name="B", terms=(StateTerm(weight=0.5, variable="C_DA"),),
            ),
        ]
        result = evaluate_all_states(
            defs,
            saturations={"X": 0.6},
            concentrations={"DA": 0.8},
            oscillations={},
        )
        assert result["A"] == pytest.approx(0.6)
        assert result["B"] == pytest.approx(0.4)

    def test_evaluate_all_empty(self):
        result = evaluate_all_states([], {}, {}, {})
        assert result == {}


# ---------------------------------------------------------------------------
# K.5 specification examples
# ---------------------------------------------------------------------------

class TestSpecExamples:
    """Tests based on K.5 specification examples."""

    def test_creative_drive(self):
        """STATE(CreativeDrive) = 0.40*S_DA_D1 + 0.25*phi_theta_gamma*S_NMDA - 0.15*S_DA_D2."""
        defn = StateDefinition(
            name="CreativeDrive",
            terms=(
                StateTerm(weight=0.40, variable="S_DA_D1"),
                StateTerm(weight=0.25, variable="phi_theta_gamma*S_NMDA"),
                StateTerm(weight=-0.15, variable="S_DA_D2"),
            ),
        )
        val = evaluate_state(
            defn,
            saturations={"DA_D1": 0.8, "DA_D2": 0.3, "NMDA": 0.6},
            concentrations={},
            oscillations={"theta_gamma": 0.5},
        )
        # 0.40*0.8 + 0.25*(0.5*0.6) + (-0.15)*0.3
        # = 0.32 + 0.075 - 0.045 = 0.35
        assert val == pytest.approx(0.35)

    def test_logic_rigidity(self):
        """STATE(Rigidity) = 0.50*S_NE_B1 + 0.40*S_DA_D2 - 0.30*S_CB1."""
        defn = StateDefinition(
            name="Rigidity",
            terms=(
                StateTerm(weight=0.50, variable="S_NE_B1"),
                StateTerm(weight=0.40, variable="S_DA_D2"),
                StateTerm(weight=-0.30, variable="S_CB1"),
            ),
        )
        val = evaluate_state(
            defn,
            saturations={"NE_B1": 0.6, "DA_D2": 0.5, "CB1": 0.4},
            concentrations={},
            oscillations={},
        )
        # 0.50*0.6 + 0.40*0.5 - 0.30*0.4 = 0.30 + 0.20 - 0.12 = 0.38
        assert val == pytest.approx(0.38)

    def test_empathy_resonance(self):
        """STATE(EmpathyResonance) = 0.45*S_OXTR + 0.30*phi_theta - 0.20*C_cortisol."""
        defn = StateDefinition(
            name="EmpathyResonance",
            terms=(
                StateTerm(weight=0.45, variable="S_OXTR"),
                StateTerm(weight=0.30, variable="phi_theta"),
                StateTerm(weight=-0.20, variable="C_cortisol"),
            ),
        )
        val = evaluate_state(
            defn,
            saturations={"OXTR": 0.7},
            concentrations={"cortisol": 0.3},
            oscillations={"theta": 0.5},
        )
        # 0.45*0.7 + 0.30*0.5 - 0.20*0.3 = 0.315 + 0.15 - 0.06 = 0.405
        assert val == pytest.approx(0.405)
