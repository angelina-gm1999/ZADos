"""
Tests for mode selection hooks (Appendix M.5 + M.6).

Phase 35-37: ModeHookDefinition, mode namespace, composite gates, selector, spec examples.
"""

import pytest

from zados.neurochem.neurosymbolic.metrics import NeurochemicalMetrics
from zados.neurochem.neurosymbolic.mode_hooks import (
    ModeHookDefinition,
    ModeSelectionResult,
    DEFAULT_MODE_HOOKS,
    DEFAULT_THRESHOLDS,
    build_mode_namespace,
    evaluate_composite_gate,
    select_mode,
)
from zados.neurochem.neurosymbolic.triggers import evaluate_condition


# ---------------------------------------------------------------------------
# Phase 35: Mode definitions
# ---------------------------------------------------------------------------

class TestModeHookDefinition:
    """Tests for ModeHookDefinition frozen dataclass."""

    def test_frozen(self):
        hook = ModeHookDefinition(name="X", condition_str="a>0", priority_tier=0)
        with pytest.raises(AttributeError):
            hook.name = "Y"

    def test_defaults(self):
        hook = ModeHookDefinition(name="X", condition_str="a>0", priority_tier=0)
        assert hook.actions == ()
        assert hook.required_inputs == ()
        assert hook.composite_gate is None
        assert hook.composite_threshold is None


class TestDefaultModeHooks:
    """Tests for the default M.6 mode hook library."""

    def test_count(self):
        assert len(DEFAULT_MODE_HOOKS) == 14

    def test_names_unique(self):
        names = [h.name for h in DEFAULT_MODE_HOOKS]
        assert len(names) == len(set(names))

    def test_priority_tiers_valid(self):
        for hook in DEFAULT_MODE_HOOKS:
            assert hook.priority_tier in {0, 1, 2, 3}

    def test_tier_0_containment_modes(self):
        tier0 = [h for h in DEFAULT_MODE_HOOKS if h.priority_tier == 0]
        assert len(tier0) == 2
        names = {h.name for h in tier0}
        assert "Containment" in names
        assert "RecoveryReset" in names

    def test_tier_counts(self):
        tier_counts = {}
        for h in DEFAULT_MODE_HOOKS:
            tier_counts[h.priority_tier] = tier_counts.get(h.priority_tier, 0) + 1
        assert tier_counts[0] == 2   # safety
        assert tier_counts[1] == 3   # empathy
        assert tier_counts[2] == 6   # rigidity
        assert tier_counts[3] == 3   # drive

    def test_all_conditions_parseable(self):
        """Every condition_str should tokenize without error."""
        dummy_vars = {
            "M_hat": 0.5, "E_hat": 0.5, "R_hat": 0.5, "F_hat": 0.5,
            "phi_delta": 0.5, "phi_theta": 0.5, "phi_alpha": 0.5,
            "phi_beta": 0.5, "phi_gamma": 0.5,
            "phi_theta_gamma": 0.25, "phi_alpha_beta": 0.25,
            "S_5HT-1A": 0.5,
        }
        for hook in DEFAULT_MODE_HOOKS:
            # Should not raise
            evaluate_condition(hook.condition_str, dummy_vars)

    def test_all_have_composite_gates(self):
        """All default hooks have composite gate expressions."""
        for hook in DEFAULT_MODE_HOOKS:
            assert hook.composite_gate is not None, f"{hook.name} missing composite_gate"

    def test_thresholds_non_empty(self):
        assert len(DEFAULT_THRESHOLDS) > 0


class TestModeSelectionResult:
    """Tests for ModeSelectionResult frozen dataclass."""

    def test_frozen(self):
        r = ModeSelectionResult(active_mode="X")
        with pytest.raises(AttributeError):
            r.active_mode = "Y"

    def test_defaults(self):
        r = ModeSelectionResult()
        assert r.active_mode is None
        assert r.fired_modes == ()
        assert r.composite_scores is None


# ---------------------------------------------------------------------------
# Phase 36: Namespace builder
# ---------------------------------------------------------------------------

class TestBuildModeNamespace:
    """Tests for build_mode_namespace."""

    def test_metric_aliases(self):
        m = NeurochemicalMetrics(motivation=0.7, empathy=0.3, cognitive_rigidity=0.5, fatigue=0.2)
        ns = build_mode_namespace(m, {})
        assert ns["M_hat"] == pytest.approx(0.7)
        assert ns["E_hat"] == pytest.approx(0.3)
        assert ns["R_hat"] == pytest.approx(0.5)
        assert ns["F_hat"] == pytest.approx(0.2)

    def test_all_8_metrics(self):
        m = NeurochemicalMetrics(precision=0.6, openness=0.4)
        ns = build_mode_namespace(m, {})
        assert ns["precision"] == pytest.approx(0.6)
        assert ns["openness"] == pytest.approx(0.4)

    def test_oscillations(self):
        ns = build_mode_namespace(
            NeurochemicalMetrics(), {"theta": 0.5, "gamma": 0.8, "alpha": 0.3, "beta": 0.4},
        )
        assert ns["phi_theta"] == pytest.approx(0.5)
        assert ns["phi_gamma"] == pytest.approx(0.8)
        assert ns["theta"] == pytest.approx(0.5)  # short alias

    def test_cfc_products(self):
        ns = build_mode_namespace(
            NeurochemicalMetrics(), {"theta": 0.6, "gamma": 0.8, "alpha": 0.4, "beta": 0.5},
        )
        assert ns["phi_theta_gamma"] == pytest.approx(0.48)  # 0.6 * 0.8
        assert ns["phi_alpha_beta"] == pytest.approx(0.2)    # 0.4 * 0.5

    def test_saturations(self):
        ns = build_mode_namespace(
            NeurochemicalMetrics(), {},
            saturations={"DA_D2": 0.7, "5HT_1A": 0.3},
        )
        assert ns["S_DA_D2"] == pytest.approx(0.7)
        assert ns["S_DA-D2"] == pytest.approx(0.7)   # hyphen variant
        assert ns["S_5HT_1A"] == pytest.approx(0.3)
        assert ns["S_5HT-1A"] == pytest.approx(0.3)  # hyphen variant

    def test_concentrations(self):
        ns = build_mode_namespace(
            NeurochemicalMetrics(), {},
            concentrations={"DA": 0.6},
        )
        assert ns["C_DA"] == pytest.approx(0.6)

    def test_empty_optional_dicts(self):
        ns = build_mode_namespace(NeurochemicalMetrics(), {})
        assert "M_hat" in ns
        # Should not crash with no saturations/concentrations


# ---------------------------------------------------------------------------
# Phase 36: Composite gate evaluator
# ---------------------------------------------------------------------------

class TestEvaluateCompositeGate:
    """Tests for evaluate_composite_gate."""

    def test_simple_weighted_sum(self):
        val = evaluate_composite_gate(
            "0.5*M_hat + 0.5*phi_gamma",
            {"M_hat": 0.8, "phi_gamma": 0.6},
        )
        assert val == pytest.approx(0.7)  # 0.5*0.8 + 0.5*0.6

    def test_inverted_term(self):
        val = evaluate_composite_gate(
            "0.4*R_hat + 0.3*(1-phi_alpha) + 0.3*(1-S_5HT1A)",
            {"R_hat": 0.8, "phi_alpha": 0.2, "S_5HT1A": 0.3},
        )
        # 0.4*0.8 + 0.3*(1-0.2) + 0.3*(1-0.3) = 0.32 + 0.24 + 0.21 = 0.77
        assert val == pytest.approx(0.77)

    def test_missing_vars_zero(self):
        val = evaluate_composite_gate("0.5*M_hat + 0.5*UNKNOWN", {})
        assert val == pytest.approx(0.0)  # both vars missing → 0

    def test_three_term(self):
        val = evaluate_composite_gate(
            "0.4*M_hat + 0.3*phi_gamma + 0.3*(1-R_hat)",
            {"M_hat": 0.7, "phi_gamma": 0.6, "R_hat": 0.2},
        )
        # 0.4*0.7 + 0.3*0.6 + 0.3*(1-0.2) = 0.28 + 0.18 + 0.24 = 0.70
        assert val == pytest.approx(0.70)


# ---------------------------------------------------------------------------
# Phase 36: Mode selector
# ---------------------------------------------------------------------------

class TestSelectMode:
    """Tests for select_mode."""

    def test_no_hooks_fire(self):
        hooks = [
            ModeHookDefinition(name="X", condition_str="M_hat>0.9", priority_tier=3),
        ]
        result = select_mode(hooks, {"M_hat": 0.3})
        assert result.active_mode is None
        assert result.fired_modes == ()

    def test_single_mode_fires(self):
        hooks = [
            ModeHookDefinition(name="TestMode", condition_str="M_hat>0.5", priority_tier=3),
        ]
        result = select_mode(hooks, {"M_hat": 0.7})
        assert result.active_mode == "TestMode"
        assert "TestMode" in result.fired_modes

    def test_tier_priority(self):
        """Tier 0 beats tier 3."""
        hooks = [
            ModeHookDefinition(name="Creative", condition_str="M_hat>0.5", priority_tier=3),
            ModeHookDefinition(name="Safety", condition_str="F_hat>0.5", priority_tier=0),
        ]
        result = select_mode(hooks, {"M_hat": 0.7, "F_hat": 0.8})
        assert result.active_mode == "Safety"
        assert len(result.fired_modes) == 2

    def test_within_tier_composite_wins(self):
        """Higher composite score wins within same tier."""
        hooks = [
            ModeHookDefinition(
                name="ModeA", condition_str="x>0.5", priority_tier=2,
                composite_gate="0.5*x",
            ),
            ModeHookDefinition(
                name="ModeB", condition_str="x>0.5", priority_tier=2,
                composite_gate="1.0*x",
            ),
        ]
        result = select_mode(hooks, {"x": 0.8})
        assert result.active_mode == "ModeB"  # 1.0*0.8 > 0.5*0.8

    def test_within_tier_first_match_no_composite(self):
        """No composites → definition order wins."""
        hooks = [
            ModeHookDefinition(name="First", condition_str="x>0.5", priority_tier=2),
            ModeHookDefinition(name="Second", condition_str="x>0.5", priority_tier=2),
        ]
        result = select_mode(hooks, {"x": 0.8})
        assert result.active_mode == "First"

    def test_containment_overrides_creative(self):
        """Safety (tier 0) overrides drive (tier 3)."""
        hooks = [
            ModeHookDefinition(
                name="CreativeDivergence",
                condition_str="M_hat>0.6 AND phi_gamma>0.5",
                priority_tier=3,
            ),
            ModeHookDefinition(
                name="Containment",
                condition_str="F_hat>0.6 AND phi_delta>0.5",
                priority_tier=0,
            ),
        ]
        ns = {
            "M_hat": 0.8, "phi_gamma": 0.7,
            "F_hat": 0.7, "phi_delta": 0.6,
        }
        result = select_mode(hooks, ns)
        assert result.active_mode == "Containment"

    def test_empathy_overrides_logic(self):
        """Tier 1 (empathy) beats tier 2 (rigidity)."""
        hooks = [
            ModeHookDefinition(
                name="LogicMode",
                condition_str="R_hat>0.5 AND phi_beta>0.5",
                priority_tier=2,
            ),
            ModeHookDefinition(
                name="EmpathicAttunement",
                condition_str="E_hat>0.6 AND phi_theta>0.5",
                priority_tier=1,
            ),
        ]
        ns = {
            "R_hat": 0.7, "phi_beta": 0.6,
            "E_hat": 0.8, "phi_theta": 0.7,
        }
        result = select_mode(hooks, ns)
        assert result.active_mode == "EmpathicAttunement"

    def test_fired_modes_captures_all(self):
        hooks = [
            ModeHookDefinition(name="A", condition_str="x>0.5", priority_tier=3),
            ModeHookDefinition(name="B", condition_str="x>0.5", priority_tier=2),
            ModeHookDefinition(name="C", condition_str="x>0.9", priority_tier=1),  # won't fire
        ]
        result = select_mode(hooks, {"x": 0.7})
        assert set(result.fired_modes) == {"A", "B"}
        assert result.active_mode == "B"  # tier 2 < tier 3


# ---------------------------------------------------------------------------
# M.5.3 / M.6 spec integration examples
# ---------------------------------------------------------------------------

class TestSpecExamples:
    """Integration tests matching M.5.3 and M.6 table examples."""

    def test_creative_divergence_scenario(self):
        """High motivation, high gamma, low rigidity → CreativeDivergence."""
        metrics = NeurochemicalMetrics(
            motivation=0.75, cognitive_rigidity=0.25, fatigue=0.2,
        )
        osc = {"theta": 0.4, "gamma": 0.7, "alpha": 0.3, "beta": 0.3, "delta": 0.1}
        ns = build_mode_namespace(metrics, osc)
        result = select_mode(DEFAULT_MODE_HOOKS, ns)
        assert result.active_mode == "CreativeDivergence"

    def test_logic_scan_scenario(self):
        """High rigidity, low alpha, low 5HT1A → HypercriticalLogicScan."""
        metrics = NeurochemicalMetrics(
            cognitive_rigidity=0.75, fatigue=0.2,
        )
        osc = {"theta": 0.3, "gamma": 0.3, "alpha": 0.2, "beta": 0.6, "delta": 0.1}
        ns = build_mode_namespace(metrics, osc, saturations={"5HT_1A": 0.2})
        result = select_mode(DEFAULT_MODE_HOOKS, ns)
        assert result.active_mode == "HypercriticalLogicScan"

    def test_empathic_attunement_scenario(self):
        """High empathy, high theta, low rigidity → EmpathicAttunement."""
        metrics = NeurochemicalMetrics(
            empathy=0.8, cognitive_rigidity=0.2, fatigue=0.2,
        )
        osc = {"theta": 0.7, "gamma": 0.3, "alpha": 0.5, "beta": 0.3, "delta": 0.1}
        ns = build_mode_namespace(metrics, osc)
        result = select_mode(DEFAULT_MODE_HOOKS, ns)
        assert result.active_mode == "EmpathicAttunement"

    def test_containment_overrides_everything(self):
        """High fatigue + high delta → Containment beats any other mode."""
        metrics = NeurochemicalMetrics(
            motivation=0.8, empathy=0.7, cognitive_rigidity=0.6, fatigue=0.85,
        )
        osc = {"theta": 0.6, "gamma": 0.6, "alpha": 0.3, "beta": 0.6, "delta": 0.7}
        ns = build_mode_namespace(metrics, osc)
        result = select_mode(DEFAULT_MODE_HOOKS, ns)
        assert result.active_mode in {"Containment", "RecoveryReset"}
        # Must be tier 0
        tier0_names = {h.name for h in DEFAULT_MODE_HOOKS if h.priority_tier == 0}
        assert result.active_mode in tier0_names

    def test_recovery_reset_scenario(self):
        """Very high fatigue, high delta, low beta → RecoveryReset."""
        metrics = NeurochemicalMetrics(fatigue=0.9)
        osc = {"theta": 0.3, "gamma": 0.2, "alpha": 0.3, "beta": 0.1, "delta": 0.8}
        ns = build_mode_namespace(metrics, osc)
        result = select_mode(DEFAULT_MODE_HOOKS, ns)
        assert result.active_mode == "RecoveryReset"

    def test_conceptual_synthesis_scenario(self):
        """High theta-gamma coupling + high gamma + moderate motivation → ConceptualSynthesis."""
        metrics = NeurochemicalMetrics(
            motivation=0.55, cognitive_rigidity=0.5, fatigue=0.2,
        )
        osc = {"theta": 0.8, "gamma": 0.75, "alpha": 0.3, "beta": 0.3, "delta": 0.1}
        ns = build_mode_namespace(metrics, osc)
        result = select_mode(DEFAULT_MODE_HOOKS, ns)
        assert result.active_mode == "ConceptualSynthesis"
