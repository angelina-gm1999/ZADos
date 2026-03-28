"""Tests for Engine 13 — Simulation Brain Engine."""
import math

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.simulation_brain_engine import (
    SimulationBrainEngine, SBConfig, SBState,
    ScenarioSeed, ScenarioNode, ScenarioOutcome, RiskProfile,
    SimulationNeurochem, SimulationBrainInput, SimulationBrainResult,
    generate_seeds, rank_seeds,
    compute_uncertainty_drive, integrate_uncertainty,
    compute_forecast_temperature, compute_recursion_depth,
    compute_reward_volatility, compute_branch_count,
    evaluate_leaf, compute_risk_profile, compute_outcome_entropy,
    compute_recommendation, export_uncertainty,
    compute_simulation_neurochem,
    _softmax,
)
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# 1. Softmax
# =====================================================================

class TestSoftmax:
    def test_uniform(self):
        probs = _softmax([1.0, 1.0, 1.0], 1.0)
        assert len(probs) == 3
        assert abs(sum(probs) - 1.0) < 1e-9
        assert all(abs(p - 1/3) < 0.01 for p in probs)

    def test_temperature_sharpens(self):
        probs_hot = _softmax([1.0, 2.0, 3.0], 2.0)
        probs_cold = _softmax([1.0, 2.0, 3.0], 0.1)
        # Cold temperature → sharper distribution (max prob higher)
        assert max(probs_cold) > max(probs_hot)

    def test_empty(self):
        assert _softmax([], 1.0) == []

    def test_sums_to_one(self):
        probs = _softmax([0.5, 1.0, 0.3, 2.0], 0.8)
        assert abs(sum(probs) - 1.0) < 1e-9


# =====================================================================
# 2. Uncertainty Drive
# =====================================================================

class TestUncertaintyDrive:
    def test_basic(self):
        k = compute_uncertainty_drive(0.5, 0.3, 0.2)
        expected = 0.35 * 0.5 + 0.35 * 0.3 + 0.30 * 0.2
        assert abs(k - expected) < 1e-9

    def test_zero(self):
        assert compute_uncertainty_drive(0, 0, 0) == 0.0


# =====================================================================
# 3. Uncertainty Integration
# =====================================================================

class TestIntegration:
    def test_accumulates(self):
        d = integrate_uncertainty(0.5, 0.0, 5.0)
        assert d > 0

    def test_decays(self):
        d = integrate_uncertainty(0.0, 1.0, 5.0)
        assert d < 1.0

    def test_tiny_tau(self):
        d = integrate_uncertainty(0.5, 0.0, 0.0001)
        assert abs(d - 0.5) < 0.1  # Nearly immediate response


# =====================================================================
# 4. Forecast Temperature
# =====================================================================

class TestTemperature:
    def test_base(self):
        t = compute_forecast_temperature(0.8, 1.5, 0.0)
        assert abs(t - 0.8) < 1e-9

    def test_increases_with_uncertainty(self):
        t = compute_forecast_temperature(0.8, 1.5, 0.5)
        assert t > 0.8

    def test_minimum(self):
        t = compute_forecast_temperature(0.01, 1.5, -100)
        assert t >= 0.01


# =====================================================================
# 5. Recursion Depth
# =====================================================================

class TestRecursionDepth:
    def test_basic(self):
        d = compute_recursion_depth(3, 4, 3, 0.5, 0.3, 2, 10)
        # 3 + 4*0.5 - 3*0.3 = 3 + 2 - 0.9 = 4.1 → 4
        assert d == 4

    def test_high_theta_gamma(self):
        d = compute_recursion_depth(3, 4, 3, 1.0, 0.0, 2, 10)
        assert d == 7  # 3 + 4 - 0

    def test_clamped_min(self):
        d = compute_recursion_depth(0, 0, 10, 0.0, 1.0, 2, 10)
        assert d >= 2

    def test_clamped_max(self):
        d = compute_recursion_depth(3, 10, 0, 1.0, 0.0, 2, 10)
        assert d <= 10


# =====================================================================
# 6. Reward Volatility
# =====================================================================

class TestVolatility:
    def test_no_coupling(self):
        v = compute_reward_volatility(0.0)
        assert abs(v - 1.0) < 1e-9

    def test_full_coupling(self):
        v = compute_reward_volatility(1.0)
        assert abs(v - 0.5) < 1e-9

    def test_inverse_relationship(self):
        assert compute_reward_volatility(0.8) < compute_reward_volatility(0.2)


# =====================================================================
# 7. Branch Count
# =====================================================================

class TestBranchCount:
    def test_basic(self):
        n = compute_branch_count(0.8, 0, 5, 3, 3.0, 2.0, 2, 8)
        assert 2 <= n <= 8

    def test_high_temp_more_branches(self):
        n_hot = compute_branch_count(2.0, 0, 5, 3, 3.0, 2.0, 2, 8)
        n_cold = compute_branch_count(0.3, 0, 5, 3, 3.0, 2.0, 2, 8)
        assert n_hot >= n_cold

    def test_deeper_fewer_branches(self):
        n_shallow = compute_branch_count(0.8, 0, 5, 3, 3.0, 2.0, 2, 8)
        n_deep = compute_branch_count(0.8, 4, 5, 3, 3.0, 2.0, 2, 8)
        assert n_deep <= n_shallow


# =====================================================================
# 8. Seed Generation
# =====================================================================

class TestSeeds:
    def test_intent_seeds(self):
        inp = SimulationBrainInput(
            intent_descriptions=("goal_a", "goal_b"),
            intent_confidences=(0.8, 0.5),
        )
        seeds = generate_seeds(inp)
        assert len(seeds) == 2
        assert seeds[0].source == "intention"

    def test_mixed_sources(self):
        inp = SimulationBrainInput(
            intent_descriptions=("goal",),
            intent_confidences=(0.8,),
            alternative_interpretations=("alt",),
            alternative_plausibilities=(0.5,),
            memory_scenarios=("past",),
            memory_relevance_scores=(0.6,),
            contradiction_statements=(("a", "b"),),
        )
        seeds = generate_seeds(inp)
        sources = {s.source for s in seeds}
        assert "intention" in sources
        assert "interpretation" in sources
        assert "memory" in sources
        assert "contradiction" in sources

    def test_ranking(self):
        seeds = [
            ScenarioSeed(seed_id="s1", probability=0.9, novelty=0.1, risk_relevance=0.1),
            ScenarioSeed(seed_id="s2", probability=0.1, novelty=0.9, risk_relevance=0.9),
        ]
        ranked = rank_seeds(seeds)
        # s1 has higher prob component, s2 has higher novelty+risk
        assert len(ranked) == 2

    def test_empty(self):
        seeds = generate_seeds(SimulationBrainInput())
        assert len(seeds) == 0


# =====================================================================
# 9. Leaf Evaluation
# =====================================================================

class TestLeafEval:
    def test_high_consistency(self):
        node = ScenarioNode(
            consistency_score=0.9,
            cumulative_probability=0.5,
            reward_alignment={"logic": 0.8, "ethics": 0.7},
        )
        score = evaluate_leaf(node, 0, 0, None)
        assert 0 < score <= 1.0

    def test_low_consistency(self):
        node_high = ScenarioNode(consistency_score=0.9, cumulative_probability=0.5)
        node_low = ScenarioNode(consistency_score=0.1, cumulative_probability=0.5)
        s_high = evaluate_leaf(node_high, 0, 0, None)
        s_low = evaluate_leaf(node_low, 0, 0, None)
        assert s_high > s_low


# =====================================================================
# 10. Risk Profile
# =====================================================================

class TestRiskProfile:
    def test_empty(self):
        rp = compute_risk_profile([])
        assert rp.expected_utility == 0.0

    def test_single_outcome(self):
        outcomes = [ScenarioOutcome(probability=1.0, utility=0.7)]
        rp = compute_risk_profile(outcomes)
        assert abs(rp.expected_utility - 0.7) < 0.01
        assert abs(rp.outcome_variance) < 0.01

    def test_mixed_outcomes(self):
        outcomes = [
            ScenarioOutcome(outcome_id="a", probability=0.6, utility=0.8),
            ScenarioOutcome(outcome_id="b", probability=0.4, utility=0.2),
        ]
        rp = compute_risk_profile(outcomes)
        assert abs(rp.expected_utility - (0.6 * 0.8 + 0.4 * 0.2)) < 0.01
        assert rp.best_case_utility == 0.8
        assert rp.worst_case_utility == 0.2

    def test_tail_risk(self):
        outcomes = [
            ScenarioOutcome(probability=0.3, utility=0.1),  # Below disaster
            ScenarioOutcome(probability=0.7, utility=0.8),
        ]
        rp = compute_risk_profile(outcomes, theta_disaster=0.15)
        assert abs(rp.tail_risk - 0.3) < 0.01


# =====================================================================
# 11. Outcome Entropy
# =====================================================================

class TestEntropy:
    def test_single(self):
        outcomes = [ScenarioOutcome(probability=1.0)]
        assert compute_outcome_entropy(outcomes) < 0.01

    def test_uniform(self):
        outcomes = [ScenarioOutcome(probability=0.5), ScenarioOutcome(probability=0.5)]
        h = compute_outcome_entropy(outcomes)
        assert abs(h - math.log(2)) < 0.01

    def test_empty(self):
        assert compute_outcome_entropy([]) == 0.0


# =====================================================================
# 12. Recommendation
# =====================================================================

class TestRecommendation:
    def test_empty(self):
        action, conf, ci = compute_recommendation([])
        assert action == "defer"
        assert conf == 0.0

    def test_clear_winner(self):
        outcomes = [
            ScenarioOutcome(outcome_id="a", description="do_x", probability=0.9, utility=0.8),
            ScenarioOutcome(outcome_id="b", description="do_y", probability=0.1, utility=0.3),
        ]
        action, conf, ci = compute_recommendation(outcomes)
        assert action == "do_x"
        assert conf > 0


# =====================================================================
# 13. Uncertainty Export
# =====================================================================

class TestExport:
    def test_empty(self):
        u = export_uncertainty([], 5, 10)
        assert u["branch_uncertainty"] == 1.0

    def test_dominant_branch(self):
        outcomes = [
            ScenarioOutcome(probability=0.9, utility=0.7),
            ScenarioOutcome(probability=0.1, utility=0.3),
        ]
        u = export_uncertainty(outcomes, 5, 10)
        assert u["branch_uncertainty"] < 0.2  # One dominant


# =====================================================================
# 14. Neurochemical Signals
# =====================================================================

class TestNeurochem:
    def test_positive_forecast(self):
        nc = compute_simulation_neurochem(
            expected_utility=0.8, tail_risk=0.0, worst_severity=0.2,
            D_rec=5, D_base=3, D_max=10,
            temperature=0.8, T_0=0.8, outcome_convergence=0.8,
            cfg=SBConfig(),
        )
        assert nc.delta_da > 0  # Positive → DA boost
        assert nc.delta_ach > 0  # Deep → ACh

    def test_negative_forecast(self):
        nc = compute_simulation_neurochem(
            expected_utility=0.2, tail_risk=0.5, worst_severity=0.8,
            D_rec=3, D_base=3, D_max=10,
            temperature=0.8, T_0=0.8, outcome_convergence=0.3,
            cfg=SBConfig(),
        )
        assert nc.delta_da < 0   # Negative → DA drop
        assert nc.delta_ne > 0   # Threat → NE
        assert nc.delta_cor > 0  # Stress → cortisol

    def test_creative(self):
        nc = compute_simulation_neurochem(
            expected_utility=0.5, tail_risk=0.1, worst_severity=0.3,
            D_rec=3, D_base=3, D_max=10,
            temperature=1.5, T_0=0.8, outcome_convergence=0.5,
            cfg=SBConfig(),
        )
        assert nc.delta_cb1 > 0  # High T → CB1
        assert nc.gamma_burst > 0


# =====================================================================
# 15. Full Pipeline
# =====================================================================

class TestFullPipeline:
    def _engine(self, seed=42):
        return SimulationBrainEngine(rng=np.random.default_rng(seed))

    def test_basic_run(self):
        engine = self._engine()
        inp = SimulationBrainInput(
            intent_descriptions=("User wants to learn Python",),
            intent_confidences=(0.8,),
            theta_gamma_coupling=0.5,
        )
        result = engine.process(inp)
        assert result.engine_id == "simulation_brain_engine"
        assert result.total_nodes > 0
        assert len(result.outcomes) > 0
        assert result.processing_time_ms > 0
        assert result.forecast_temperature > 0

    def test_multiple_seeds(self):
        engine = self._engine()
        inp = SimulationBrainInput(
            intent_descriptions=("goal_a", "goal_b"),
            intent_confidences=(0.8, 0.6),
            alternative_interpretations=("alt_a",),
            alternative_plausibilities=(0.5,),
        )
        result = engine.process(inp)
        assert result.total_nodes > 1

    def test_contradiction_driven(self):
        engine = self._engine()
        inp = SimulationBrainInput(
            contradiction_statements=(("The sky is blue", "The sky is not blue"),),
            theta_gamma_coupling=0.5,
        )
        result = engine.process(inp)
        assert result.total_nodes > 0

    def test_empty_input(self):
        engine = self._engine()
        result = engine.process(SimulationBrainInput())
        assert result.total_nodes == 0
        assert len(result.outcomes) == 0

    def test_risk_profile_populated(self):
        engine = self._engine()
        inp = SimulationBrainInput(
            intent_descriptions=("test",),
            intent_confidences=(0.7,),
        )
        result = engine.process(inp)
        rp = result.risk_profile
        assert isinstance(rp.expected_utility, float)

    def test_uncertainty_export(self):
        engine = self._engine()
        inp = SimulationBrainInput(
            intent_descriptions=("test",),
            intent_confidences=(0.7,),
        )
        result = engine.process(inp)
        assert "branch_uncertainty" in result.simulation_uncertainty
        assert "outcome_entropy" in result.simulation_uncertainty

    def test_with_reward_context(self):
        engine = self._engine()
        inp = SimulationBrainInput(
            intent_descriptions=("test",),
            intent_confidences=(0.7,),
            reward_scores={"logic": 0.8, "attunement": 0.6, "ethics": 0.9, "innovation": 0.5},
        )
        result = engine.process(inp)
        assert result.total_nodes > 0


# =====================================================================
# 16. Mode Configuration
# =====================================================================

class TestModes:
    def test_configure(self):
        engine = SimulationBrainEngine()
        engine.configure(OperationalMode.DEV)
        assert engine.get_status()["mode"] == "dev"

    def test_dream_mode_params(self):
        cfg = SBConfig()
        assert cfg.T_0["rem_dream"] > cfg.T_0["normal"]
        assert cfg.D_max["rem_dream"] > cfg.D_max["normal"]
        assert cfg.theta_prune["rem_dream"] < cfg.theta_prune["normal"]
        assert cfg.max_nodes["rem_dream"] > cfg.max_nodes["normal"]

    def test_dream_mode_high_temperature(self):
        engine = SimulationBrainEngine(rng=np.random.default_rng(42))
        engine.configure(OperationalMode.REM_DREAM)
        inp = SimulationBrainInput(
            intent_descriptions=("dream scenario",),
            intent_confidences=(0.5,),
        )
        result = engine.process(inp)
        assert result.forecast_temperature > 1.0


# =====================================================================
# 17. NT Feedback
# =====================================================================

class TestNTFeedback:
    def test_update(self):
        engine = SimulationBrainEngine()
        engine.update_neurochem_state({"da": 0.5, "cb1": 0.7, "ne": 0.3})
        s = engine.get_status()
        assert s["nt_levels"]["da"] == 0.5
        assert s["nt_levels"]["cb1"] == 0.7

    def test_clamps(self):
        engine = SimulationBrainEngine()
        engine.update_neurochem_state({"da": 2.0, "ne": -1.0})
        s = engine.get_status()
        assert s["nt_levels"]["da"] == 1.0
        assert s["nt_levels"]["ne"] == 0.0


# =====================================================================
# 18. Introspection
# =====================================================================

class TestIntrospection:
    def test_status_keys(self):
        engine = SimulationBrainEngine()
        s = engine.get_status()
        assert "engine_id" in s
        assert "cluster" in s
        assert "total_simulations" in s
        assert "delta_uncertainty" in s
        assert "nt_levels" in s

    def test_cycle_increments(self):
        engine = SimulationBrainEngine(rng=np.random.default_rng(42))
        engine.process(SimulationBrainInput(
            intent_descriptions=("test",), intent_confidences=(0.7,),
        ))
        engine.process(SimulationBrainInput(
            intent_descriptions=("test",), intent_confidences=(0.7,),
        ))
        assert engine.get_status()["cycle_count"] == 2

    def test_nodes_accumulated(self):
        engine = SimulationBrainEngine(rng=np.random.default_rng(42))
        engine.process(SimulationBrainInput(
            intent_descriptions=("test",), intent_confidences=(0.7,),
        ))
        assert engine.get_status()["total_nodes_expanded"] > 0


# =====================================================================
# 19. Emotion Modulation
# =====================================================================

class TestEmotionMod:
    def test_curious_increases_temp(self):
        engine = SimulationBrainEngine(rng=np.random.default_rng(42))
        result_neutral = engine.process(SimulationBrainInput(
            intent_descriptions=("test",), intent_confidences=(0.7,),
        ))
        engine2 = SimulationBrainEngine(rng=np.random.default_rng(42))
        result_curious = engine2.process(SimulationBrainInput(
            intent_descriptions=("test",), intent_confidences=(0.7,),
            emotion_intensities={"curious": 1.0},
        ))
        assert result_curious.forecast_temperature >= result_neutral.forecast_temperature


# =====================================================================
# 20. Edge Cases
# =====================================================================

class TestEdgeCases:
    def test_max_nodes_limit(self):
        engine = SimulationBrainEngine(
            config=SBConfig(max_nodes={"normal": 10, "dev": 10, "learning": 10,
                                       "reflective": 10, "rem_normal": 10, "rem_dream": 10}),
            rng=np.random.default_rng(42),
        )
        inp = SimulationBrainInput(
            intent_descriptions=("a", "b", "c", "d", "e"),
            intent_confidences=(0.8, 0.7, 0.6, 0.5, 0.4),
        )
        result = engine.process(inp)
        assert result.total_nodes <= 10

    def test_high_uncertainty_drive(self):
        """With max-uncertainty inputs, δ_uncertainty ramps via leaky
        integrator across cycles until T(t) = T₀·(1+α·δ) exceeds 1.0."""
        engine = SimulationBrainEngine(rng=np.random.default_rng(42))
        inp = SimulationBrainInput(
            intent_descriptions=("test",),
            intent_confidences=(0.5,),
            e_disintegration=0.9,
            e_ambiguity=0.9,
            nt_variance=0.9,
        )
        # Run several cycles so delta_uncertainty builds up
        for _ in range(5):
            result = engine.process(inp)
        assert result.forecast_temperature > 1.0

    def test_deterministic_with_seed(self):
        """Same seed → same result."""
        inp = SimulationBrainInput(
            intent_descriptions=("test",), intent_confidences=(0.7,),
        )
        e1 = SimulationBrainEngine(rng=np.random.default_rng(99))
        r1 = e1.process(inp)
        e2 = SimulationBrainEngine(rng=np.random.default_rng(99))
        r2 = e2.process(inp)
        assert r1.total_nodes == r2.total_nodes
        assert abs(r1.expected_utility - r2.expected_utility) < 1e-9
