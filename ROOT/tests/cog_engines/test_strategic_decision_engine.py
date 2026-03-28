"""Tests for Engine 21 -- Strategic Decision Engine."""
import math

import pytest

from zados.cognitive_engines.py_engines.strategic_decision_engine import (
    # Enums
    GoalStatus,
    RevisionReason,
    # Config
    SDConfig,
    # Internal records (used by export helpers)
    _GoalRecord,
    _StrategyRecord,
    _CommitmentRecord,
    _SDState,
    # Frozen output dataclasses
    Goal,
    StrategyScore,
    Commitment,
    RevisionEvent,
    StrategicDecisionNeurochem,
    StrategicDecisionInput,
    StrategicDecisionResult,
    # Pure functions
    compute_strategy_composite,
    compute_goal_selection_score,
    check_stagnation,
    check_feasibility_revision,
    check_risk_revision,
    compute_neurochem_signals,
    export_goal,
    export_commitment,
    export_strategy,
    # Engine
    StrategicDecisionEngine,
)
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# 1. Config Defaults
# =====================================================================


class TestSDConfig:
    """Verify SDConfig frozen defaults."""

    def test_frozen(self):
        cfg = SDConfig()
        with pytest.raises(Exception):
            cfg.max_goals = 999

    def test_max_goals_default(self):
        cfg = SDConfig()
        assert cfg.max_goals == 64

    def test_max_children_per_goal(self):
        cfg = SDConfig()
        assert cfg.max_children_per_goal == 8

    def test_max_depth(self):
        cfg = SDConfig()
        assert cfg.max_depth == 6

    def test_max_strategies_per_goal(self):
        cfg = SDConfig()
        assert cfg.max_strategies_per_goal == 12

    def test_max_active_strategies(self):
        cfg = SDConfig()
        assert cfg.max_active_strategies == 5

    def test_stagnation_threshold_modes(self):
        cfg = SDConfig()
        assert cfg.stagnation_threshold["normal"] == 5
        assert cfg.stagnation_threshold["rem_dream"] == 3

    def test_w_utility_modes(self):
        cfg = SDConfig()
        assert cfg.w_utility["normal"] == 0.40
        assert cfg.w_utility["dev"] == 0.45

    def test_exploration_bonus_modes(self):
        cfg = SDConfig()
        assert cfg.exploration_bonus["rem_dream"] == 0.30

    def test_selection_breadth_modes(self):
        cfg = SDConfig()
        assert cfg.selection_breadth["normal"] == 5
        assert cfg.selection_breadth["rem_dream"] == 8

    def test_nt_coupling_weights(self):
        cfg = SDConfig()
        assert cfg.mu_da == 0.25
        assert cfg.mu_5ht == 0.20
        assert cfg.mu_ne == 0.30
        assert cfg.mu_ach == 0.20
        assert cfg.mu_cor == 0.25
        assert cfg.mu_gaba == 0.15
        assert cfg.mu_cb1 == 0.20

    def test_write_port_coefficients(self):
        cfg = SDConfig()
        assert cfg.beta_da_novel == 0.10
        assert cfg.beta_5ht_progress == 0.08
        assert cfg.beta_ne_revision == 0.12

    def test_utility_decay_rate(self):
        cfg = SDConfig()
        assert cfg.utility_decay_rate == 0.02

    def test_progress_ema_alpha(self):
        cfg = SDConfig()
        assert cfg.progress_ema_alpha == 0.3


# =====================================================================
# 2. Pure Functions -- compute_strategy_composite
# =====================================================================


class TestComputeStrategyComposite:
    """Test the composite scoring formula."""

    def test_zero_inputs(self):
        score = compute_strategy_composite(0, 0, 0, 0, 0.4, 0.2, 0.25, 0.15)
        assert score == 0.0

    def test_perfect_utility_no_risk(self):
        score = compute_strategy_composite(
            expected_utility=1.0, risk=0.0, resource_cost=0.0,
            feasibility=1.0, w_utility=0.4, w_risk=0.2,
            w_feasibility=0.25, w_cost=0.15,
        )
        # 0.4*1.0*1.0 + 0.25*1.0 = 0.65
        assert abs(score - 0.65) < 1e-9

    def test_high_risk_lowers_score(self):
        base = compute_strategy_composite(1, 0, 0, 1, 0.4, 0.2, 0.25, 0.15)
        risky = compute_strategy_composite(1, 1, 0, 1, 0.4, 0.2, 0.25, 0.15)
        assert risky < base

    def test_high_cost_lowers_score(self):
        base = compute_strategy_composite(1, 0, 0, 1, 0.4, 0.2, 0.25, 0.15)
        costly = compute_strategy_composite(1, 0, 1, 1, 0.4, 0.2, 0.25, 0.15)
        assert costly < base

    def test_exploration_bonus_raises_score(self):
        base = compute_strategy_composite(0.5, 0.3, 0.2, 0.8, 0.4, 0.2, 0.25, 0.15,
                                          exploration_bonus=0.0)
        boosted = compute_strategy_composite(0.5, 0.3, 0.2, 0.8, 0.4, 0.2, 0.25, 0.15,
                                             exploration_bonus=0.1)
        assert boosted == pytest.approx(base + 0.1)

    def test_low_feasibility_reduces_utility_contribution(self):
        high_f = compute_strategy_composite(1, 0, 0, 1.0, 0.4, 0.2, 0.25, 0.15)
        low_f = compute_strategy_composite(1, 0, 0, 0.2, 0.4, 0.2, 0.25, 0.15)
        assert low_f < high_f


# =====================================================================
# 3. Pure Functions -- compute_goal_selection_score
# =====================================================================


class TestComputeGoalSelectionScore:
    """Test goal selection scoring."""

    def test_base_components(self):
        # priority=1, utility=1, feasibility=1, no stagnation/NT
        score = compute_goal_selection_score(1, 1, 1, 0, 0, 0, 0, 0)
        # base = 0.3 + 0.4 + 0.3 = 1.0, continuation=0
        assert abs(score - 1.0) < 1e-9

    def test_progress_continuation_bonus(self):
        no_prog = compute_goal_selection_score(0.5, 0.5, 0.5, 0.0, 0, 0, 0, 0)
        has_prog = compute_goal_selection_score(0.5, 0.5, 0.5, 0.8, 0, 0, 0, 0)
        assert has_prog > no_prog
        assert abs(has_prog - no_prog - 0.1 * 0.8) < 1e-9

    def test_stagnation_penalty(self):
        fresh = compute_goal_selection_score(0.5, 0.5, 0.5, 0, 0, 0, 0, 0)
        stagnant = compute_goal_selection_score(0.5, 0.5, 0.5, 0, 5, 0, 0, 0)
        assert stagnant < fresh
        assert abs(fresh - stagnant - 0.05 * 5) < 1e-9

    def test_cor_penalty_low_feasibility(self):
        no_cor = compute_goal_selection_score(0.5, 0.5, 0.2, 0, 0, 0, 0.0, 0)
        high_cor = compute_goal_selection_score(0.5, 0.5, 0.2, 0, 0, 0, 1.0, 0)
        # cor_penalty = 1.0 * 0.15 * max(0, 0.5 - 0.2) = 0.045
        assert high_cor < no_cor
        assert abs(no_cor - high_cor - 0.045) < 1e-9

    def test_cor_no_penalty_high_feasibility(self):
        no_cor = compute_goal_selection_score(0.5, 0.5, 0.8, 0, 0, 0, 0.0, 0)
        high_cor = compute_goal_selection_score(0.5, 0.5, 0.8, 0, 0, 0, 1.0, 0)
        # feasibility 0.8 > 0.5 so max(0, 0.5-0.8) = 0, no penalty
        assert abs(no_cor - high_cor) < 1e-9

    def test_gaba_suppression_low_utility(self):
        no_gaba = compute_goal_selection_score(0.5, 0.1, 0.5, 0, 0, 0, 0, 0.0)
        high_gaba = compute_goal_selection_score(0.5, 0.1, 0.5, 0, 0, 0, 0, 1.0)
        # gaba_suppress = 1.0 * 0.10 * max(0, 0.3 - 0.1) = 0.02
        assert high_gaba < no_gaba
        assert abs(no_gaba - high_gaba - 0.02) < 1e-9

    def test_gaba_no_suppression_high_utility(self):
        no_gaba = compute_goal_selection_score(0.5, 0.8, 0.5, 0, 0, 0, 0, 0.0)
        high_gaba = compute_goal_selection_score(0.5, 0.8, 0.5, 0, 0, 0, 0, 1.0)
        assert abs(no_gaba - high_gaba) < 1e-9


# =====================================================================
# 4. Pure Functions -- check_stagnation
# =====================================================================


class TestCheckStagnation:
    """Test the stagnation detection formula."""

    def test_not_stagnant_below_threshold(self):
        assert check_stagnation(3, 5, 0.0, 0.0, 0.3, 0.2) is False

    def test_stagnant_at_threshold(self):
        assert check_stagnation(5, 5, 0.0, 0.0, 0.3, 0.2) is True

    def test_ne_lowers_threshold(self):
        # base=5, ne=1.0, mu_ne=0.3 => eff = 5*(1-0.3)*(1+0) = 3.5
        assert check_stagnation(4, 5, 1.0, 0.0, 0.3, 0.2) is True

    def test_5ht_raises_threshold(self):
        # base=5, 5ht=1.0, mu_5ht=0.2 => eff = 5*(1-0)*(1+0.2) = 6.0
        assert check_stagnation(5, 5, 0.0, 1.0, 0.3, 0.2) is False
        assert check_stagnation(6, 5, 0.0, 1.0, 0.3, 0.2) is True

    def test_ne_and_5ht_combined(self):
        # base=5, ne=0.5, 5ht=0.5 => eff = 5*(1-0.15)*(1+0.1) = 5*0.85*1.1 = 4.675
        assert check_stagnation(4, 5, 0.5, 0.5, 0.3, 0.2) is False
        assert check_stagnation(5, 5, 0.5, 0.5, 0.3, 0.2) is True

    def test_effective_floor_at_one(self):
        # Even with extreme NE the threshold floors at 1
        assert check_stagnation(1, 5, 1.0, 0.0, 1.0, 0.0) is True
        assert check_stagnation(0, 5, 1.0, 0.0, 1.0, 0.0) is False


# =====================================================================
# 5. Pure Functions -- check_feasibility_revision / check_risk_revision
# =====================================================================


class TestCheckFeasibilityRevision:
    """Test feasibility floor revision check."""

    def test_above_floor_no_revision(self):
        assert check_feasibility_revision(0.5, 0.2, 0.0, 0.25) is False

    def test_below_floor_triggers_revision(self):
        assert check_feasibility_revision(0.1, 0.2, 0.0, 0.25) is True

    def test_cor_raises_floor(self):
        # floor=0.2, cor=1.0, mu_cor=0.25 => eff_floor = 0.2*(1+0.25) = 0.25
        assert check_feasibility_revision(0.22, 0.2, 1.0, 0.25) is True

    def test_floor_capped_at_095(self):
        # Very high cor and floor should cap at 0.95
        assert check_feasibility_revision(0.96, 0.9, 1.0, 1.0) is False


class TestCheckRiskRevision:
    """Test risk ceiling revision check."""

    def test_below_ceiling_no_revision(self):
        assert check_risk_revision(0.5, 0.8, 0.0, 0.25) is False

    def test_above_ceiling_triggers_revision(self):
        assert check_risk_revision(0.9, 0.8, 0.0, 0.25) is True

    def test_cor_lowers_ceiling(self):
        # ceiling=0.8, cor=1.0, mu_cor=0.25 => eff_ceil = 0.8*(1-0.125) = 0.7
        assert check_risk_revision(0.75, 0.8, 1.0, 0.25) is True

    def test_ceiling_floor_at_010(self):
        # Extreme COR should not push ceiling below 0.10
        assert check_risk_revision(0.05, 0.8, 1.0, 10.0) is False


# =====================================================================
# 6. Pure Functions -- compute_neurochem_signals
# =====================================================================


class TestComputeNeurochemSignals:
    """Test neurochem signal computation."""

    def test_no_events_baseline(self):
        cfg = SDConfig()
        nc = compute_neurochem_signals(
            novel_strategy_adopted=False, progress_made=0.0,
            revisions_triggered=0, max_risk_exposure=0.0,
            evaluation_depth=0, goals_pruned=0, stagnant_count=0,
            cfg=cfg,
        )
        assert nc.da_delta == 0.0
        assert nc._5ht_delta == 0.0
        assert nc.ne_delta == 0.0
        assert nc.cor_delta == 0.0
        assert nc.ach_delta == 0.0
        assert nc.gaba_delta == 0.0
        # beta_boost is always emitted
        assert nc.beta_boost == cfg.psi_beta

    def test_novel_strategy_da(self):
        cfg = SDConfig()
        nc = compute_neurochem_signals(
            novel_strategy_adopted=True, progress_made=0.0,
            revisions_triggered=0, max_risk_exposure=0.0,
            evaluation_depth=0, goals_pruned=0, stagnant_count=0,
            cfg=cfg,
        )
        assert nc.da_delta == pytest.approx(cfg.beta_da_novel)

    def test_progress_5ht(self):
        cfg = SDConfig()
        nc = compute_neurochem_signals(
            novel_strategy_adopted=False, progress_made=0.5,
            revisions_triggered=0, max_risk_exposure=0.0,
            evaluation_depth=0, goals_pruned=0, stagnant_count=0,
            cfg=cfg,
        )
        assert nc._5ht_delta == pytest.approx(cfg.beta_5ht_progress * 0.5)

    def test_revisions_ne(self):
        cfg = SDConfig()
        nc = compute_neurochem_signals(
            novel_strategy_adopted=False, progress_made=0.0,
            revisions_triggered=2, max_risk_exposure=0.0,
            evaluation_depth=0, goals_pruned=0, stagnant_count=0,
            cfg=cfg,
        )
        assert nc.ne_delta == pytest.approx(cfg.beta_ne_revision * 2)

    def test_stagnation_ne(self):
        cfg = SDConfig()
        nc = compute_neurochem_signals(
            novel_strategy_adopted=False, progress_made=0.0,
            revisions_triggered=0, max_risk_exposure=0.0,
            evaluation_depth=0, goals_pruned=0, stagnant_count=2,
            cfg=cfg,
        )
        assert nc.ne_delta == pytest.approx(cfg.beta_ne_revision * 0.5 * 2)

    def test_high_risk_cor(self):
        cfg = SDConfig()
        nc = compute_neurochem_signals(
            novel_strategy_adopted=False, progress_made=0.0,
            revisions_triggered=0, max_risk_exposure=0.8,
            evaluation_depth=0, goals_pruned=0, stagnant_count=0,
            cfg=cfg,
        )
        # cor_delta = 0.10 * (0.8-0.5) * 2.0 = 0.06
        assert nc.cor_delta == pytest.approx(cfg.beta_cor_risk * 0.3 * 2.0)

    def test_risk_below_050_no_cor(self):
        cfg = SDConfig()
        nc = compute_neurochem_signals(
            novel_strategy_adopted=False, progress_made=0.0,
            revisions_triggered=0, max_risk_exposure=0.4,
            evaluation_depth=0, goals_pruned=0, stagnant_count=0,
            cfg=cfg,
        )
        assert nc.cor_delta == 0.0

    def test_deep_evaluation_ach(self):
        cfg = SDConfig()
        nc = compute_neurochem_signals(
            novel_strategy_adopted=False, progress_made=0.0,
            revisions_triggered=0, max_risk_exposure=0.0,
            evaluation_depth=6, goals_pruned=0, stagnant_count=0,
            cfg=cfg,
        )
        assert nc.ach_delta > 0.0

    def test_deep_evaluation_theta_boost(self):
        cfg = SDConfig()
        nc = compute_neurochem_signals(
            novel_strategy_adopted=False, progress_made=0.0,
            revisions_triggered=0, max_risk_exposure=0.0,
            evaluation_depth=8, goals_pruned=0, stagnant_count=0,
            cfg=cfg,
        )
        assert nc.theta_boost > 0.0

    def test_pruned_goals_gaba(self):
        cfg = SDConfig()
        nc = compute_neurochem_signals(
            novel_strategy_adopted=False, progress_made=0.0,
            revisions_triggered=0, max_risk_exposure=0.0,
            evaluation_depth=0, goals_pruned=3, stagnant_count=0,
            cfg=cfg,
        )
        assert nc.gaba_delta == pytest.approx(cfg.beta_gaba_prune * 3)


# =====================================================================
# 7. Export Helpers
# =====================================================================


class TestExportHelpers:
    """Test export_goal, export_commitment, export_strategy."""

    def test_export_goal(self):
        rec = _GoalRecord(
            goal_id="g1", description="test goal", parent_id=None,
            priority=0.7, feasibility=0.8, utility=0.6,
            status=GoalStatus.ACTIVE, created_tick=5,
            last_progress_tick=8, progress=0.3, progress_velocity=0.05,
            stagnation_count=2, children_ids=["c1", "c2"],
            metadata={"k": "v"},
        )
        g = export_goal(rec)
        assert isinstance(g, Goal)
        assert g.goal_id == "g1"
        assert g.priority == 0.7
        assert g.children_ids == ("c1", "c2")
        assert g.metadata == {"k": "v"}

    def test_export_commitment(self):
        rec = _CommitmentRecord(
            goal_id="g1", strategy_id="s1", start_tick=3,
            progress=0.4, stagnation_count=1,
            last_progress_delta=0.1, progress_velocity=0.05,
        )
        c = export_commitment(rec)
        assert isinstance(c, Commitment)
        assert c.goal_id == "g1"
        assert c.progress == 0.4

    def test_export_strategy(self):
        rec = _StrategyRecord(
            strategy_id="s1", goal_id="g1", expected_utility=0.7,
            risk=0.3, resource_cost=0.2, composite=0.45,
            adopted=True, created_tick=2, metadata={},
        )
        s = export_strategy(rec)
        assert isinstance(s, StrategyScore)
        assert s.composite == 0.45


# =====================================================================
# 8. Enums
# =====================================================================


class TestEnums:
    def test_goal_status_values(self):
        assert GoalStatus.PENDING.value == "pending"
        assert GoalStatus.ACTIVE.value == "active"
        assert GoalStatus.COMPLETED.value == "completed"
        assert GoalStatus.ABANDONED.value == "abandoned"
        assert GoalStatus.STAGNANT.value == "stagnant"

    def test_revision_reason_values(self):
        assert RevisionReason.STAGNATION.value == "stagnation"
        assert RevisionReason.FEASIBILITY_DECAY.value == "feasibility_decay"
        assert RevisionReason.RISK_SPIKE.value == "risk_spike"
        assert RevisionReason.EXTERNAL_OVERRIDE.value == "external_override"
        assert RevisionReason.GOAL_COMPLETED.value == "goal_completed"
        assert RevisionReason.URGENCY_OVERRIDE.value == "urgency_override"


# =====================================================================
# 9. Engine Init and Identity
# =====================================================================


class TestEngineInit:
    """Test engine construction, identity, and repr."""

    def test_default_construction(self):
        e = StrategicDecisionEngine()
        assert e.engine_id == "strategic_decision_engine"
        assert e.cluster == "reasoning"

    def test_custom_config(self):
        cfg = SDConfig(max_goals=10)
        e = StrategicDecisionEngine(config=cfg)
        assert e._cfg.max_goals == 10

    def test_initial_state(self):
        e = StrategicDecisionEngine()
        assert e._cycle_count == 0
        assert len(e._goals) == 0
        assert len(e._strategies) == 0
        assert len(e._commitments) == 0

    def test_get_status_keys(self):
        e = StrategicDecisionEngine()
        status = e.get_status()
        assert status["engine_id"] == "strategic_decision_engine"
        assert status["cluster"] == "reasoning"
        assert status["mode"] == "normal"
        assert status["cycle_count"] == 0
        assert status["total_goals"] == 0
        assert "nt_levels" in status

    def test_get_status_nt_levels(self):
        e = StrategicDecisionEngine()
        nt = e.get_status()["nt_levels"]
        for key in ("da", "5ht", "ne", "ach", "cor", "gaba", "cb1"):
            assert key in nt
            assert nt[key] == 0.0


# =====================================================================
# 10. Mode Switching
# =====================================================================


class TestModeSwitching:
    def test_configure_mode(self):
        e = StrategicDecisionEngine()
        e.configure(OperationalMode.DEV)
        assert e._mode == OperationalMode.DEV
        assert e.get_status()["mode"] == "dev"

    def test_mode_affects_stagnation_threshold(self):
        e = StrategicDecisionEngine()
        e.configure(OperationalMode.REFLECTIVE)
        # reflective stagnation_threshold = 8
        assert e._get_mode_param(e._cfg.stagnation_threshold, 5) == 8

    def test_rem_dream_mode(self):
        e = StrategicDecisionEngine()
        e.configure(OperationalMode.REM_DREAM)
        assert e._mode_key() == "rem_dream"
        assert e._get_mode_param(e._cfg.stagnation_threshold, 5) == 3


# =====================================================================
# 11. NT Modulation (update_neurochem_state)
# =====================================================================


class TestNTModulation:
    def test_update_da(self):
        e = StrategicDecisionEngine()
        e.update_neurochem_state({"da": 0.7})
        assert e._state.da_level == pytest.approx(0.7)

    def test_update_5ht(self):
        e = StrategicDecisionEngine()
        e.update_neurochem_state({"5ht": 0.6})
        assert e._state._5ht_level == pytest.approx(0.6)

    def test_update_ne(self):
        e = StrategicDecisionEngine()
        e.update_neurochem_state({"ne": 0.8})
        assert e._state.ne_level == pytest.approx(0.8)

    def test_update_all_nts(self):
        e = StrategicDecisionEngine()
        e.update_neurochem_state({
            "da": 0.5, "5ht": 0.4, "ne": 0.3,
            "ach": 0.6, "cor": 0.2, "gaba": 0.1, "cb1": 0.9,
        })
        assert e._state.da_level == pytest.approx(0.5)
        assert e._state._5ht_level == pytest.approx(0.4)
        assert e._state.ne_level == pytest.approx(0.3)
        assert e._state.ach_level == pytest.approx(0.6)
        assert e._state.cor_level == pytest.approx(0.2)
        assert e._state.gaba_level == pytest.approx(0.1)
        assert e._state.cb1_level == pytest.approx(0.9)

    def test_clamp_above_one(self):
        e = StrategicDecisionEngine()
        e.update_neurochem_state({"da": 1.5})
        assert e._state.da_level == 1.0

    def test_clamp_below_zero(self):
        e = StrategicDecisionEngine()
        e.update_neurochem_state({"ne": -0.5})
        assert e._state.ne_level == 0.0

    def test_ignore_unknown_keys(self):
        e = StrategicDecisionEngine()
        e.update_neurochem_state({"unknown_nt": 0.5})
        # Should not raise and state stays default


# =====================================================================
# 12. Goal Management
# =====================================================================


class TestGoalManagement:
    """Test add_goal, remove_goal, complete_goal, abandon_goal."""

    def test_add_goal_returns_frozen_goal(self):
        e = StrategicDecisionEngine()
        g = e.add_goal("g1", "first goal", priority=0.8, utility=0.7)
        assert isinstance(g, Goal)
        assert g.goal_id == "g1"
        assert g.status == GoalStatus.PENDING

    def test_add_multiple_goals(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        e.add_goal("g2", "goal2")
        assert len(e._goals) == 2
        assert e._state.total_goals_created == 2

    def test_add_child_goal(self):
        e = StrategicDecisionEngine()
        e.add_goal("parent", "parent goal")
        g = e.add_goal("child", "child goal", parent_id="parent")
        assert g.parent_id == "parent"
        assert "child" in e._goals["parent"].children_ids

    def test_add_child_invalid_parent_falls_to_root(self):
        e = StrategicDecisionEngine()
        g = e.add_goal("g1", "orphan", parent_id="nonexistent")
        assert g.parent_id is None
        assert "g1" in e._root_ids

    def test_remove_goal(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        result = e.remove_goal("g1")
        assert result is True
        assert "g1" not in e._goals

    def test_remove_goal_with_children(self):
        e = StrategicDecisionEngine()
        e.add_goal("parent", "parent")
        e.add_goal("child1", "child1", parent_id="parent")
        e.add_goal("child2", "child2", parent_id="parent")
        e.remove_goal("parent")
        assert "parent" not in e._goals
        assert "child1" not in e._goals
        assert "child2" not in e._goals

    def test_remove_nonexistent_goal(self):
        e = StrategicDecisionEngine()
        assert e.remove_goal("nonexistent") is False

    def test_complete_goal(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        assert e.complete_goal("g1") is True
        assert e._goals["g1"].status == GoalStatus.COMPLETED
        assert e._goals["g1"].progress == 1.0
        assert e._state.total_goals_completed == 1

    def test_complete_nonexistent_goal(self):
        e = StrategicDecisionEngine()
        assert e.complete_goal("fake") is False

    def test_abandon_goal(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        assert e.abandon_goal("g1") is True
        assert e._goals["g1"].status == GoalStatus.ABANDONED
        assert e._state.total_goals_abandoned == 1

    def test_abandon_nonexistent_goal(self):
        e = StrategicDecisionEngine()
        assert e.abandon_goal("fake") is False

    def test_max_goals_eviction(self):
        cfg = SDConfig(max_goals=3)
        e = StrategicDecisionEngine(config=cfg)
        e.add_goal("g1", "low utility", utility=0.1)
        e.add_goal("g2", "mid utility", utility=0.5)
        e.add_goal("g3", "high utility", utility=0.9)
        # Adding a 4th should evict the lowest-utility pending goal (g1)
        e.add_goal("g4", "new goal", utility=0.6)
        assert "g1" not in e._goals
        assert len(e._goals) <= 3

    def test_max_children_eviction(self):
        cfg = SDConfig(max_children_per_goal=2)
        e = StrategicDecisionEngine(config=cfg)
        e.add_goal("parent", "parent")
        e.add_goal("c1", "child1", parent_id="parent", utility=0.1)
        e.add_goal("c2", "child2", parent_id="parent", utility=0.5)
        # Adding 3rd child should evict lowest-utility child (c1)
        e.add_goal("c3", "child3", parent_id="parent", utility=0.8)
        assert "c1" not in e._goals


# =====================================================================
# 13. Strategy Management
# =====================================================================


class TestStrategyManagement:
    def test_add_strategy(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        s = e.add_strategy("s1", "g1", expected_utility=0.8, risk=0.2)
        assert isinstance(s, StrategyScore)
        assert s.strategy_id == "s1"
        assert s.goal_id == "g1"
        assert e._state.total_strategies_scored == 1

    def test_add_strategy_invalid_goal(self):
        e = StrategicDecisionEngine()
        with pytest.raises(ValueError, match="not found"):
            e.add_strategy("s1", "nonexistent")

    def test_strategy_composite_computed(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", feasibility=0.9)
        s = e.add_strategy("s1", "g1", expected_utility=0.8, risk=0.1, resource_cost=0.1)
        assert s.composite != 0.0

    def test_max_strategies_per_goal_eviction(self):
        cfg = SDConfig(max_strategies_per_goal=2)
        e = StrategicDecisionEngine(config=cfg)
        e.add_goal("g1", "goal1")
        e.add_strategy("s1", "g1", expected_utility=0.1)
        e.add_strategy("s2", "g1", expected_utility=0.5)
        # Adding 3rd should evict lowest-composite
        e.add_strategy("s3", "g1", expected_utility=0.9)
        goal_strats = [s for s in e._strategies.values() if s.goal_id == "g1"]
        assert len(goal_strats) <= 2


# =====================================================================
# 14. Progress Tracking
# =====================================================================


class TestProgressTracking:
    def test_update_progress(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        p = e.update_progress("g1", 0.3)
        assert p == pytest.approx(0.3)
        assert e._goals["g1"].progress == pytest.approx(0.3)

    def test_progress_clamps_at_one(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        e.update_progress("g1", 0.7)
        p = e.update_progress("g1", 0.5)
        assert p == 1.0

    def test_progress_activates_pending(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        assert e._goals["g1"].status == GoalStatus.PENDING
        e.update_progress("g1", 0.1)
        assert e._goals["g1"].status == GoalStatus.ACTIVE

    def test_progress_resets_stagnation(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        e._goals["g1"].stagnation_count = 5
        e.update_progress("g1", 0.1)
        assert e._goals["g1"].stagnation_count == 0

    def test_progress_ema_velocity(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        e.update_progress("g1", 0.2)
        alpha = e._cfg.progress_ema_alpha
        expected_vel = alpha * 0.2 + (1 - alpha) * 0.0
        assert e._goals["g1"].progress_velocity == pytest.approx(expected_vel)

    def test_progress_nonexistent_goal(self):
        e = StrategicDecisionEngine()
        assert e.update_progress("fake", 0.5) == 0.0

    def test_negative_progress_ignored(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        e.update_progress("g1", 0.3)
        p = e.update_progress("g1", -0.2)
        # Negative delta clamped to 0, so progress stays at 0.3
        assert p == pytest.approx(0.3)

    def test_progress_updates_commitment(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        e.add_strategy("s1", "g1")
        e._create_or_update_commitment("g1", "s1")
        e.update_progress("g1", 0.2)
        assert e._commitments["g1"].progress == pytest.approx(0.2)


# =====================================================================
# 15. Sub-goal Selection
# =====================================================================


class TestSubgoalSelection:
    def test_no_goals_returns_none(self):
        e = StrategicDecisionEngine()
        goal, strat = e.select_next_action()
        assert goal is None
        assert strat is None

    def test_single_goal_selected(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "only goal", priority=0.8, utility=0.9, feasibility=0.7)
        goal, strat = e.select_next_action()
        assert goal is not None
        assert goal.goal_id == "g1"
        assert strat is None  # No strategies registered

    def test_highest_scored_goal_selected(self):
        e = StrategicDecisionEngine()
        e.add_goal("low", "low", priority=0.1, utility=0.1, feasibility=0.1)
        e.add_goal("high", "high", priority=0.9, utility=0.9, feasibility=0.9)
        goal, _ = e.select_next_action()
        assert goal.goal_id == "high"

    def test_strategy_returned_with_goal(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", priority=0.8, utility=0.8, feasibility=0.8)
        e.add_strategy("s1", "g1", expected_utility=0.7)
        goal, strat = e.select_next_action()
        assert goal is not None
        assert strat is not None
        assert strat.strategy_id == "s1"

    def test_completed_goals_excluded(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "completed", priority=0.9, utility=0.9)
        e.complete_goal("g1")
        e.add_goal("g2", "active", priority=0.5, utility=0.5)
        goal, _ = e.select_next_action()
        assert goal.goal_id == "g2"

    def test_abandoned_goals_excluded(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "abandoned", priority=0.9, utility=0.9)
        e.abandon_goal("g1")
        goal, _ = e.select_next_action()
        assert goal is None

    def test_stagnant_goals_still_selectable(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "stagnant goal")
        e._goals["g1"].status = GoalStatus.STAGNANT
        goal, _ = e.select_next_action()
        assert goal is not None


# =====================================================================
# 16. Commitment Tracking
# =====================================================================


class TestCommitmentTracking:
    def test_create_commitment(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        com = e._create_or_update_commitment("g1", "s1")
        assert com.goal_id == "g1"
        assert com.strategy_id == "s1"
        assert "g1" in e._commitments

    def test_update_existing_commitment(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        e._create_or_update_commitment("g1", "s1")
        com = e._create_or_update_commitment("g1", "s2")
        assert com.strategy_id == "s2"
        # Only one commitment per goal
        assert len(e._commitments) == 1

    def test_commitment_removed_on_complete(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        e._create_or_update_commitment("g1", "s1")
        e.complete_goal("g1")
        assert "g1" not in e._commitments

    def test_commitment_removed_on_abandon(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        e._create_or_update_commitment("g1", "s1")
        e.abandon_goal("g1")
        assert "g1" not in e._commitments


# =====================================================================
# 17. Stagnation Detection & Plan Revision
# =====================================================================


class TestStagnationAndRevision:
    def test_stagnation_detected_after_threshold(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        # Only one strategy so revision cannot succeed (no alternative)
        # This keeps the commitment alive and stagnation count accumulates
        e.add_strategy("s1", "g1", expected_utility=0.5)
        e._create_or_update_commitment("g1", "s1")
        # Run enough cycles to trigger stagnation (threshold=5 in normal mode)
        for _ in range(6):
            e._commitments["g1"].last_progress_delta = 0.0
            stagnant, revisions = e._evaluate_commitments()
        # After 5+ ticks with no progress, should detect stagnation
        assert len(stagnant) > 0 or e._goals["g1"].stagnation_count >= 5

    def test_revision_selects_better_alternative(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", feasibility=0.8)
        e.add_strategy("s_bad", "g1", expected_utility=0.2, risk=0.1)
        e.add_strategy("s_good", "g1", expected_utility=0.9, risk=0.1)
        rev = e._try_revision("g1", "s_bad", RevisionReason.STAGNATION)
        assert rev is not None
        assert rev.new_strategy_id == "s_good"
        assert rev.reason == RevisionReason.STAGNATION

    def test_revision_no_alternative(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        e.add_strategy("s1", "g1")
        rev = e._try_revision("g1", "s1", RevisionReason.STAGNATION)
        # Only one strategy -- no alternative
        assert rev is None

    def test_revision_only_if_better(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", feasibility=0.8)
        e.add_strategy("s_good", "g1", expected_utility=0.9, risk=0.1)
        e.add_strategy("s_bad", "g1", expected_utility=0.1, risk=0.5)
        # Current is s_good, alternative is s_bad -- should NOT revise
        rev = e._try_revision("g1", "s_good", RevisionReason.STAGNATION)
        assert rev is None

    def test_revision_reactivates_stagnant_goal(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", feasibility=0.8)
        e._goals["g1"].status = GoalStatus.STAGNANT
        e.add_strategy("s_old", "g1", expected_utility=0.2, risk=0.1)
        e.add_strategy("s_new", "g1", expected_utility=0.9, risk=0.1)
        e._create_or_update_commitment("g1", "s_old")
        rev = e._try_revision("g1", "s_old", RevisionReason.STAGNATION)
        assert rev is not None
        assert e._goals["g1"].status == GoalStatus.ACTIVE

    def test_revision_increments_total(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", feasibility=0.8)
        e.add_strategy("s1", "g1", expected_utility=0.2)
        e.add_strategy("s2", "g1", expected_utility=0.9)
        before = e._state.total_revisions
        e._try_revision("g1", "s1", RevisionReason.STAGNATION)
        assert e._state.total_revisions == before + 1


# =====================================================================
# 18. GABA-Driven Pruning
# =====================================================================


class TestGABAPruning:
    def test_no_pruning_low_gaba(self):
        e = StrategicDecisionEngine()
        e.update_neurochem_state({"gaba": 0.1})
        e.add_goal("g1", "low utility", utility=0.05)
        pruned = e._prune_low_utility_goals()
        assert len(pruned) == 0

    def test_prune_low_utility_high_gaba(self):
        e = StrategicDecisionEngine()
        e.update_neurochem_state({"gaba": 0.8})
        e.add_goal("g1", "very low utility", utility=0.05, feasibility=0.5)
        pruned = e._prune_low_utility_goals()
        assert "g1" in pruned
        assert e._goals["g1"].status == GoalStatus.ABANDONED

    def test_no_prune_if_progress(self):
        e = StrategicDecisionEngine()
        e.update_neurochem_state({"gaba": 0.8})
        e.add_goal("g1", "low utility in-progress", utility=0.05)
        e._goals["g1"].progress = 0.5  # significant progress
        pruned = e._prune_low_utility_goals()
        assert "g1" not in pruned

    def test_prune_skips_completed(self):
        e = StrategicDecisionEngine()
        e.update_neurochem_state({"gaba": 0.8})
        e.add_goal("g1", "done", utility=0.05)
        e.complete_goal("g1")
        pruned = e._prune_low_utility_goals()
        assert "g1" not in pruned


# =====================================================================
# 19. Utility Decay
# =====================================================================


class TestUtilityDecay:
    def test_decay_non_progressing(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", utility=0.5)
        e._goals["g1"].last_progress_tick = 0
        e._cycle_count = 5
        e._apply_utility_decay()
        assert e._goals["g1"].utility < 0.5

    def test_no_decay_with_progress(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", utility=0.5)
        e._goals["g1"].progress_velocity = 0.1  # active progress
        e._cycle_count = 5
        e._apply_utility_decay()
        assert e._goals["g1"].utility == pytest.approx(0.5)

    def test_no_decay_completed(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", utility=0.5)
        e.complete_goal("g1")
        e._cycle_count = 10
        e._apply_utility_decay()
        assert e._goals["g1"].utility == pytest.approx(0.5)

    def test_feasibility_also_decays(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", utility=0.5, feasibility=0.8)
        e._goals["g1"].last_progress_tick = 0
        e._cycle_count = 5
        e._apply_utility_decay()
        assert e._goals["g1"].feasibility < 0.8


# =====================================================================
# 20. process() Pipeline
# =====================================================================


class TestProcessPipeline:
    def test_empty_input(self):
        e = StrategicDecisionEngine()
        inp = StrategicDecisionInput()
        result = e.process(inp)
        assert isinstance(result, StrategicDecisionResult)
        assert result.engine_id == "strategic_decision_engine"
        assert result.cycle_count == 1
        assert result.selected_goal is None

    def test_process_registers_goals(self):
        e = StrategicDecisionEngine()
        inp = StrategicDecisionInput(
            new_goals=(
                {"goal_id": "g1", "description": "test", "priority": 0.8, "utility": 0.7},
            ),
        )
        result = e.process(inp)
        assert len(result.active_goals) == 1
        assert result.active_goals[0].goal_id == "g1"

    def test_process_registers_strategies(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        inp = StrategicDecisionInput(
            new_strategies=(
                {"strategy_id": "s1", "goal_id": "g1", "expected_utility": 0.8},
            ),
        )
        result = e.process(inp)
        assert len(result.scored_strategies) > 0

    def test_process_applies_progress(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        inp = StrategicDecisionInput(progress_updates={"g1": 0.3})
        result = e.process(inp)
        assert e._goals["g1"].progress == pytest.approx(0.3)

    def test_process_completes_goals(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        inp = StrategicDecisionInput(completed_goals=("g1",))
        result = e.process(inp)
        assert e._goals["g1"].status == GoalStatus.COMPLETED

    def test_process_abandons_goals(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        inp = StrategicDecisionInput(abandoned_goals=("g1",))
        result = e.process(inp)
        assert e._goals["g1"].status == GoalStatus.ABANDONED

    def test_process_selects_goal_and_strategy(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", priority=0.9, utility=0.9)
        e.add_strategy("s1", "g1", expected_utility=0.8)
        inp = StrategicDecisionInput()
        result = e.process(inp)
        assert result.selected_goal is not None
        assert result.selected_strategy is not None

    def test_process_creates_commitment(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", priority=0.9, utility=0.9)
        e.add_strategy("s1", "g1", expected_utility=0.8)
        inp = StrategicDecisionInput()
        result = e.process(inp)
        assert len(result.commitments) > 0

    def test_process_neurochem_signals(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", priority=0.9, utility=0.9)
        e.add_strategy("s1", "g1", expected_utility=0.8)
        inp = StrategicDecisionInput()
        result = e.process(inp)
        nc = result.neurochem_signals
        assert isinstance(nc, StrategicDecisionNeurochem)
        # beta_boost always present
        assert nc.beta_boost > 0.0

    def test_process_novel_strategy_da(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", priority=0.9, utility=0.9, feasibility=0.9)
        e.add_strategy("s1", "g1", expected_utility=0.8, risk=0.1)
        inp = StrategicDecisionInput()
        result = e.process(inp)
        # First adoption should be novel
        assert result.neurochem_signals.da_delta > 0.0

    def test_process_progress_5ht(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        inp = StrategicDecisionInput(progress_updates={"g1": 0.5})
        result = e.process(inp)
        assert result.neurochem_signals._5ht_delta > 0.0

    def test_process_metadata(self):
        e = StrategicDecisionEngine()
        inp = StrategicDecisionInput()
        result = e.process(inp)
        assert "total_goals" in result.metadata
        assert "nt_levels" in result.metadata

    def test_process_increments_cycle(self):
        e = StrategicDecisionEngine()
        e.process(StrategicDecisionInput())
        e.process(StrategicDecisionInput())
        assert e._cycle_count == 2

    def test_process_mode_in_result(self):
        e = StrategicDecisionEngine()
        e.configure(OperationalMode.DEV)
        result = e.process(StrategicDecisionInput())
        assert result.mode == "dev"

    def test_process_processing_time(self):
        e = StrategicDecisionEngine()
        result = e.process(StrategicDecisionInput())
        assert result.processing_time_ms >= 0.0

    def test_process_force_revision(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", feasibility=0.8)
        e.add_strategy("s1", "g1", expected_utility=0.2)
        e.add_strategy("s2", "g1", expected_utility=0.9)
        e._create_or_update_commitment("g1", "s1")
        inp = StrategicDecisionInput(force_revision_goals=("g1",))
        result = e.process(inp)
        # Should have triggered a revision
        assert len(result.revisions) > 0
        assert result.revisions[0].reason == RevisionReason.EXTERNAL_OVERRIDE


# =====================================================================
# 21. Neurochem Output (write-port signals)
# =====================================================================


class TestNeurochemOutput:
    """Test that the engine emits proper neurochem signals via process()."""

    def test_baseline_beta_boost(self):
        e = StrategicDecisionEngine()
        result = e.process(StrategicDecisionInput())
        assert result.neurochem_signals.beta_boost == pytest.approx(e._cfg.psi_beta)

    def test_theta_boost_deep_eval(self):
        e = StrategicDecisionEngine()
        # Add many goals so evaluation_depth > 5
        for i in range(8):
            e.add_goal(f"g{i}", f"goal {i}", priority=0.5, utility=0.5)
        result = e.process(StrategicDecisionInput())
        assert result.neurochem_signals.theta_boost > 0.0


# =====================================================================
# 22. Edge Cases
# =====================================================================


class TestEdgeCases:
    def test_all_goals_completed(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        e.add_goal("g2", "goal2")
        e.complete_goal("g1")
        e.complete_goal("g2")
        goal, strat = e.select_next_action()
        assert goal is None
        assert strat is None

    def test_process_with_all_completed(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        e.complete_goal("g1")
        result = e.process(StrategicDecisionInput())
        assert result.selected_goal is None

    def test_remove_goal_cleans_strategies(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        e.add_strategy("s1", "g1")
        e.add_strategy("s2", "g1")
        e.remove_goal("g1")
        assert "s1" not in e._strategies
        assert "s2" not in e._strategies

    def test_remove_goal_cleans_commitments(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        e._create_or_update_commitment("g1", "s1")
        e.remove_goal("g1")
        assert "g1" not in e._commitments

    def test_strategy_for_goal_with_zero_feasibility(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", feasibility=0.0)
        s = e.add_strategy("s1", "g1", expected_utility=1.0)
        # With 0 feasibility, utility*feasibility = 0
        assert s.composite <= 0.0 + e._cfg.w_feasibility["normal"] * 0.0 + 1e-9

    def test_deeply_nested_goal_removal(self):
        e = StrategicDecisionEngine()
        e.add_goal("root", "root")
        e.add_goal("mid", "mid", parent_id="root")
        e.add_goal("leaf", "leaf", parent_id="mid")
        e.remove_goal("root")
        assert len(e._goals) == 0

    def test_da_exploration_rem_dream(self):
        e = StrategicDecisionEngine()
        e.configure(OperationalMode.REM_DREAM)
        e.update_neurochem_state({"da": 0.8, "cb1": 0.9})
        e.add_goal("g1", "dream goal", priority=0.5, utility=0.5)
        e.add_goal("g2", "other", priority=0.5, utility=0.5)
        # Should not crash; exploration is amplified
        goal, _ = e.select_next_action()
        assert goal is not None

    def test_rescore_all_strategies(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1", feasibility=0.8)
        e.add_strategy("s1", "g1", expected_utility=0.5)
        e.add_strategy("s2", "g1", expected_utility=0.9)
        scored = e._rescore_all_strategies()
        assert len(scored) == 2
        # Higher utility strategy should score higher
        assert scored[0].expected_utility >= scored[1].expected_utility

    def test_evaluate_commitments_orphan_cleanup(self):
        e = StrategicDecisionEngine()
        # Create a commitment for a non-existent goal
        e._commitments["phantom"] = _CommitmentRecord(
            goal_id="phantom", strategy_id="s1", start_tick=0,
            progress=0.0, stagnation_count=0,
            last_progress_delta=0.0, progress_velocity=0.0,
        )
        e._evaluate_commitments()
        assert "phantom" not in e._commitments

    def test_get_status_after_operations(self):
        e = StrategicDecisionEngine()
        e.add_goal("g1", "goal1")
        e.add_goal("g2", "goal2")
        e.add_strategy("s1", "g1")
        e.update_progress("g1", 0.3)
        e.complete_goal("g2")
        status = e.get_status()
        assert status["total_goals"] == 2
        assert status["total_goals_completed"] == 1
        assert status["total_strategies_scored"] == 1
        assert "g1" in status["active_goal_ids"]
