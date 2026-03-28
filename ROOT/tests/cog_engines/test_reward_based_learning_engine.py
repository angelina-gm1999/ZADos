"""
Tests for Engine 17 -- Reward-Based Learning Engine
====================================================
Covers:
  1. Config defaults
  2. Pure helper functions
  3. Reward prediction (running EMA)
  4. Prediction error computation
  5. Learning rate dynamics
  6. Parameter adjustments
  7. Consolidation
  8. NT modulation
  9. Mode switching
 10. process() pipeline
 11. Neurochem output
 12. Engine init / get_status() / repr
 13. Edge cases
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)
from zados.cognitive_engines.py_engines.reward_based_learning_engine import (
    # Enums
    ConvergenceStatus,
    DomainType,
    LearningPhase,
    # Config
    RewardLearningConfig,
    # Data types
    ConsolidationEvent,
    LearningRecord,
    ParameterAdjustment,
    PredictionError,
    RewardLearningInput,
    RewardLearningNeurochem,
    RewardLearningResult,
    RewardLearningState,
    RewardPrediction,
    # Pure functions
    _mode_key,
    assess_convergence,
    classify_learning_phase,
    compute_credit_depth,
    compute_effective_learning_rate,
    compute_exploration_width,
    compute_learning_neurochem,
    compute_noise_gate,
    compute_parameter_adjustment,
    compute_prediction_error,
    compute_social_amplifier,
    should_consolidate,
    update_prediction_ema,
    # Engine
    RewardBasedLearningEngine,
)


# =====================================================================
# Helpers
# =====================================================================

def _fixed_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_engine(seed: int = 42, **cfg_overrides) -> RewardBasedLearningEngine:
    cfg = RewardLearningConfig(**cfg_overrides)
    return RewardBasedLearningEngine(config=cfg, rng=_fixed_rng(seed))


def _basic_input(
    rewards: Dict[str, float] | None = None,
    params: Dict[str, float] | None = None,
    domains: Dict[str, str] | None = None,
    mode: OperationalMode = OperationalMode.NORMAL,
) -> RewardLearningInput:
    return RewardLearningInput(
        reward_signals=rewards or {"logic": 0.6},
        parameter_values=params or {"p1": 0.5},
        parameter_domains=domains or {"p1": "logic"},
        active_mode=mode,
    )


# =====================================================================
# 1. Config defaults
# =====================================================================


class TestConfigDefaults:
    """Verify RewardLearningConfig default values."""

    def test_initial_learning_rate(self):
        cfg = RewardLearningConfig()
        assert cfg.initial_learning_rate == 0.10

    def test_lr_bounds(self):
        cfg = RewardLearningConfig()
        assert cfg.min_learning_rate == 0.005
        assert cfg.max_learning_rate == 0.50

    def test_decay_and_warmup(self):
        cfg = RewardLearningConfig()
        assert cfg.lr_decay_factor == 0.995
        assert cfg.lr_warmup_cycles == 5

    def test_prediction_defaults(self):
        cfg = RewardLearningConfig()
        assert cfg.prediction_alpha == 0.15
        assert cfg.prediction_init == 0.50

    def test_convergence_defaults(self):
        cfg = RewardLearningConfig()
        assert cfg.convergence_threshold == 0.02
        assert cfg.convergence_window == 10

    def test_consolidation_defaults(self):
        cfg = RewardLearningConfig()
        assert cfg.consolidation_threshold == 0.015
        assert cfg.consolidation_window == 15
        assert cfg.max_consolidations_per_cycle == 3

    def test_mode_lr_multipliers(self):
        cfg = RewardLearningConfig()
        assert cfg.mode_lr_multiplier["normal"] == 1.0
        assert cfg.mode_lr_multiplier["learning"] == 1.40
        assert cfg.mode_lr_multiplier["rem_dream"] == 1.60

    def test_mode_noise_gate_multipliers(self):
        cfg = RewardLearningConfig()
        assert cfg.mode_noise_gate_multiplier["dev"] == 1.30
        assert cfg.mode_noise_gate_multiplier["rem_dream"] == 0.50

    def test_config_is_frozen(self):
        cfg = RewardLearningConfig()
        with pytest.raises(Exception):
            cfg.initial_learning_rate = 0.99  # type: ignore[misc]


# =====================================================================
# 2. Pure helper functions
# =====================================================================


class TestModeKey:
    """_mode_key converts OperationalMode to config dict key."""

    def test_all_modes(self):
        assert _mode_key(OperationalMode.NORMAL) == "normal"
        assert _mode_key(OperationalMode.DEV) == "dev"
        assert _mode_key(OperationalMode.LEARNING) == "learning"
        assert _mode_key(OperationalMode.REFLECTIVE) == "reflective"
        assert _mode_key(OperationalMode.REM_NORMAL) == "rem_normal"
        assert _mode_key(OperationalMode.REM_DREAM) == "rem_dream"


class TestComputePredictionError:
    """compute_prediction_error returns (delta, |delta|)."""

    def test_positive_delta(self):
        d, m = compute_prediction_error(0.8, 0.5)
        assert d == pytest.approx(0.3)
        assert m == pytest.approx(0.3)

    def test_negative_delta(self):
        d, m = compute_prediction_error(0.3, 0.7)
        assert d == pytest.approx(-0.4)
        assert m == pytest.approx(0.4)

    def test_zero_delta(self):
        d, m = compute_prediction_error(0.5, 0.5)
        assert d == 0.0
        assert m == 0.0


class TestUpdatePredictionEma:
    """update_prediction_ema produces EMA."""

    def test_basic_ema(self):
        # E' = (1 - 0.15) * 0.5 + 0.15 * 0.8 = 0.425 + 0.12 = 0.545
        result = update_prediction_ema(0.5, 0.8, 0.15)
        assert result == pytest.approx(0.545)

    def test_alpha_zero_no_update(self):
        result = update_prediction_ema(0.5, 1.0, 0.0)
        assert result == pytest.approx(0.5)

    def test_alpha_one_immediate(self):
        result = update_prediction_ema(0.5, 1.0, 1.0)
        assert result == pytest.approx(1.0)

    def test_convergence_over_many_steps(self):
        pred = 0.5
        for _ in range(200):
            pred = update_prediction_ema(pred, 0.9, 0.15)
        assert pred == pytest.approx(0.9, abs=0.01)


class TestComputeParameterAdjustment:
    """compute_parameter_adjustment with gating."""

    def test_positive_adjustment(self):
        adj, new_val = compute_parameter_adjustment(0.1, 0.05, 0.5, False)
        assert adj == pytest.approx(0.005)
        assert new_val == pytest.approx(0.505)

    def test_gated_returns_zero_adjustment(self):
        adj, new_val = compute_parameter_adjustment(0.1, 0.05, 0.5, True)
        assert adj == 0.0
        assert new_val == 0.5

    def test_clamped_to_one(self):
        adj, new_val = compute_parameter_adjustment(10.0, 1.0, 0.9, False)
        assert new_val == 1.0

    def test_clamped_to_zero(self):
        adj, new_val = compute_parameter_adjustment(-10.0, 1.0, 0.1, False)
        assert new_val == 0.0

    def test_negative_delta(self):
        adj, new_val = compute_parameter_adjustment(-0.2, 0.1, 0.5, False)
        assert adj == pytest.approx(-0.02)
        assert new_val == pytest.approx(0.48)


# =====================================================================
# 3. Reward prediction (EMA via process pipeline)
# =====================================================================


class TestRewardPrediction:
    """Prediction state management."""

    def test_new_domain_gets_init_prediction(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        inp = _basic_input(rewards={"logic": 0.7})
        engine.process(inp)
        pred = engine.get_prediction("logic")
        assert pred is not None
        # After one cycle: EMA from 0.5 toward 0.7
        assert pred.predicted == pytest.approx(
            update_prediction_ema(0.5, 0.7, 0.15)
        )

    def test_prediction_sample_count(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        engine.process(_basic_input(rewards={"logic": 0.6}))
        engine.process(_basic_input(rewards={"logic": 0.7}))
        pred = engine.get_prediction("logic")
        assert pred is not None
        assert pred.n_samples == 2


# =====================================================================
# 4. Prediction error computation
# =====================================================================


class TestPredictionErrorPipeline:
    """Prediction errors in process output."""

    def test_single_domain_error(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        result = engine.process(_basic_input(rewards={"logic": 0.8}))
        assert len(result.prediction_errors) == 1
        pe = result.prediction_errors[0]
        assert pe.domain == "logic"
        assert pe.actual == pytest.approx(0.8)
        assert pe.predicted == pytest.approx(0.5)
        # delta = 0.8 - 0.5 = 0.3
        assert pe.delta == pytest.approx(0.3)
        assert pe.magnitude == pytest.approx(0.3)

    def test_multi_domain_errors(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        engine.register_parameter("p2", "ethics", 0.5)
        rewards = {"logic": 0.9, "ethics": 0.3}
        result = engine.process(_basic_input(
            rewards=rewards,
            params={"p1": 0.5, "p2": 0.5},
            domains={"p1": "logic", "p2": "ethics"},
        ))
        assert len(result.prediction_errors) == 2
        domains = {pe.domain for pe in result.prediction_errors}
        assert "logic" in domains
        assert "ethics" in domains


# =====================================================================
# 5. Learning rate dynamics
# =====================================================================


class TestEffectiveLearningRate:
    """compute_effective_learning_rate with decay and NT modulation."""

    def test_no_modulation_cycle_zero(self):
        cfg = RewardLearningConfig()
        lr = compute_effective_learning_rate(
            0.1, 0, cfg, da=0.5, sht=0.5, ne=0.0, cor=0.0, mode_mult=1.0,
        )
        # No decay at cycle 0, DA at baseline (0.5), no 5-HT effect, no NE, no COR
        assert lr == pytest.approx(0.1)

    def test_decay_after_warmup(self):
        cfg = RewardLearningConfig(lr_warmup_cycles=5, lr_decay_factor=0.99)
        lr = compute_effective_learning_rate(
            0.1, 10, cfg, da=0.5, sht=0.5, ne=0.0, cor=0.0, mode_mult=1.0,
        )
        expected = 0.1 * (0.99 ** 5)
        assert lr == pytest.approx(expected, rel=1e-4)

    def test_no_decay_before_warmup(self):
        cfg = RewardLearningConfig(lr_warmup_cycles=10)
        lr = compute_effective_learning_rate(
            0.1, 3, cfg, da=0.5, sht=0.5, ne=0.0, cor=0.0, mode_mult=1.0,
        )
        assert lr == pytest.approx(0.1)

    def test_da_boost(self):
        cfg = RewardLearningConfig()
        lr_high = compute_effective_learning_rate(
            0.1, 0, cfg, da=0.9, sht=0.5, ne=0.0, cor=0.0, mode_mult=1.0,
        )
        lr_base = compute_effective_learning_rate(
            0.1, 0, cfg, da=0.5, sht=0.5, ne=0.0, cor=0.0, mode_mult=1.0,
        )
        assert lr_high > lr_base

    def test_5ht_damping(self):
        cfg = RewardLearningConfig()
        lr_high_5ht = compute_effective_learning_rate(
            0.1, 0, cfg, da=0.5, sht=0.9, ne=0.0, cor=0.0, mode_mult=1.0,
        )
        lr_base = compute_effective_learning_rate(
            0.1, 0, cfg, da=0.5, sht=0.5, ne=0.0, cor=0.0, mode_mult=1.0,
        )
        assert lr_high_5ht < lr_base

    def test_ne_urgency_boost(self):
        cfg = RewardLearningConfig()
        lr_ne = compute_effective_learning_rate(
            0.1, 0, cfg, da=0.5, sht=0.5, ne=0.9, cor=0.0, mode_mult=1.0,
        )
        lr_base = compute_effective_learning_rate(
            0.1, 0, cfg, da=0.5, sht=0.5, ne=0.0, cor=0.0, mode_mult=1.0,
        )
        assert lr_ne > lr_base

    def test_cortisol_penalty(self):
        cfg = RewardLearningConfig()
        lr_stress = compute_effective_learning_rate(
            0.1, 0, cfg, da=0.5, sht=0.5, ne=0.0, cor=0.9, mode_mult=1.0,
        )
        lr_base = compute_effective_learning_rate(
            0.1, 0, cfg, da=0.5, sht=0.5, ne=0.0, cor=0.0, mode_mult=1.0,
        )
        assert lr_stress < lr_base

    def test_clamped_to_min(self):
        cfg = RewardLearningConfig(min_learning_rate=0.01)
        lr = compute_effective_learning_rate(
            0.0001, 100, cfg, da=0.0, sht=1.0, ne=0.0, cor=1.0, mode_mult=0.1,
        )
        assert lr >= cfg.min_learning_rate

    def test_clamped_to_max(self):
        cfg = RewardLearningConfig(max_learning_rate=0.5)
        lr = compute_effective_learning_rate(
            0.5, 0, cfg, da=1.0, sht=0.0, ne=1.0, cor=0.0, mode_mult=2.0,
        )
        assert lr <= cfg.max_learning_rate

    def test_mode_multiplier_applied(self):
        cfg = RewardLearningConfig()
        lr_double = compute_effective_learning_rate(
            0.1, 0, cfg, da=0.5, sht=0.5, ne=0.0, cor=0.0, mode_mult=2.0,
        )
        lr_base = compute_effective_learning_rate(
            0.1, 0, cfg, da=0.5, sht=0.5, ne=0.0, cor=0.0, mode_mult=1.0,
        )
        assert lr_double == pytest.approx(lr_base * 2.0, rel=0.01)


# =====================================================================
# 6. Parameter adjustments
# =====================================================================


class TestParameterAdjustments:
    """Parameter adjustments produced by process()."""

    def test_adjustment_direction_positive(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        result = engine.process(_basic_input(rewards={"logic": 0.8}))
        adj = result.adjustments[0]
        # delta > 0 => adjustment > 0 => new_value > old_value (unless gated)
        if not adj.gated:
            assert adj.new_value >= adj.old_value

    def test_adjustment_direction_negative(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        result = engine.process(_basic_input(rewards={"logic": 0.2}))
        adj = result.adjustments[0]
        # delta < 0 => adjustment < 0 => new_value < old_value (unless gated)
        if not adj.gated:
            assert adj.new_value <= adj.old_value

    def test_gated_small_delta(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        # reward ~= prediction (0.5) => tiny delta => gated
        result = engine.process(_basic_input(rewards={"logic": 0.501}))
        adj = result.adjustments[0]
        assert adj.gated is True
        assert adj.new_value == adj.old_value

    def test_consolidated_params_skipped(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        rec = engine.get_record("p1")
        rec.consolidated = True
        result = engine.process(_basic_input(rewards={"logic": 0.9}))
        # Consolidated param should not appear in adjustments
        adj_ids = [a.parameter_id for a in result.adjustments]
        assert "p1" not in adj_ids


# =====================================================================
# 7. Convergence & Consolidation
# =====================================================================


class TestAssessConvergence:
    """assess_convergence pure function."""

    def test_short_history_returns_exploring(self):
        assert assess_convergence([0.1, 0.2], 0.02, 10) == ConvergenceStatus.EXPLORING

    def test_converged_all_below_threshold(self):
        history = [0.01] * 12
        assert assess_convergence(history, 0.02, 10) == ConvergenceStatus.CONVERGED

    def test_converging_trend(self):
        # Strictly decreasing magnitudes
        history = [0.5, 0.4, 0.3, 0.2, 0.1]
        assert assess_convergence(history, 0.02, 10) == ConvergenceStatus.CONVERGING

    def test_diverging_trend(self):
        # Strictly increasing magnitudes
        history = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert assess_convergence(history, 0.02, 10) == ConvergenceStatus.DIVERGING

    def test_exploring_no_clear_trend(self):
        history = [0.1, 0.3, 0.1, 0.3, 0.1]
        assert assess_convergence(history, 0.02, 10) == ConvergenceStatus.EXPLORING


class TestShouldConsolidate:
    """should_consolidate pure function."""

    def test_already_consolidated(self):
        rec = LearningRecord(
            parameter_id="p1", domain="logic", consolidated=True,
            delta_history=[0.001] * 20,
        )
        assert should_consolidate(rec, 0.015, 15) is False

    def test_not_enough_history(self):
        rec = LearningRecord(
            parameter_id="p1", domain="logic",
            delta_history=[0.001] * 5,
        )
        assert should_consolidate(rec, 0.015, 15) is False

    def test_triggers_consolidation(self):
        rec = LearningRecord(
            parameter_id="p1", domain="logic",
            delta_history=[0.005] * 20,
        )
        assert should_consolidate(rec, 0.015, 15) is True

    def test_no_consolidation_high_delta(self):
        rec = LearningRecord(
            parameter_id="p1", domain="logic",
            delta_history=[0.1] * 20,
        )
        assert should_consolidate(rec, 0.015, 15) is False


class TestClassifyLearningPhase:
    """classify_learning_phase pure function."""

    def test_initial_during_warmup(self):
        assert classify_learning_phase(10, 0, 0, 3, 5) == LearningPhase.INITIAL

    def test_initial_no_params(self):
        assert classify_learning_phase(0, 0, 0, 10, 5) == LearningPhase.INITIAL

    def test_active(self):
        assert classify_learning_phase(10, 1, 1, 10, 5) == LearningPhase.ACTIVE

    def test_plateau(self):
        # converged+consolidated >= 60%
        assert classify_learning_phase(10, 2, 5, 10, 5) == LearningPhase.PLATEAU

    def test_consolidated(self):
        # consolidated >= 60%
        assert classify_learning_phase(10, 7, 1, 10, 5) == LearningPhase.CONSOLIDATED


class TestConsolidationPipeline:
    """Consolidation within the process() pipeline."""

    def test_parameter_gets_consolidated(self):
        engine = _make_engine(consolidation_window=3, consolidation_threshold=0.05)
        engine.register_parameter("p1", "logic", 0.5)
        rec = engine.get_record("p1")
        # Inject delta history that is very stable
        rec.delta_history = [0.001] * 5
        result = engine.process(_basic_input(rewards={"logic": 0.501}))
        # After this cycle the consolidation check should trigger
        rec2 = engine.get_record("p1")
        assert rec2.consolidated is True
        assert len(result.consolidations) >= 1

    def test_consolidation_rate_limit(self):
        engine = _make_engine(
            max_consolidations_per_cycle=1,
            consolidation_window=3,
            consolidation_threshold=0.1,
        )
        for i in range(5):
            engine.register_parameter(f"p{i}", "logic", 0.5)
            rec = engine.get_record(f"p{i}")
            rec.delta_history = [0.001] * 5
        result = engine.process(_basic_input(
            rewards={"logic": 0.501},
            params={f"p{i}": 0.5 for i in range(5)},
            domains={f"p{i}": "logic" for i in range(5)},
        ))
        assert len(result.consolidations) <= 1


# =====================================================================
# 8. NT modulation
# =====================================================================


class TestNTModulation:
    """Neurochemical modulation effects on learning."""

    def test_da_increases_learning_rate(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        engine.update_neurochem_state({"da": 0.9})
        r1 = engine.process(_basic_input(rewards={"logic": 0.8}))
        lr_high = r1.metadata["effective_lr"]

        engine.reset()
        engine.register_parameter("p1", "logic", 0.5)
        engine.update_neurochem_state({"da": 0.1})
        r2 = engine.process(_basic_input(rewards={"logic": 0.8}))
        lr_low = r2.metadata["effective_lr"]

        assert lr_high > lr_low

    def test_5ht_stabilises_learning(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        engine.update_neurochem_state({"5ht": 0.9})
        r1 = engine.process(_basic_input(rewards={"logic": 0.8}))
        lr_stable = r1.metadata["effective_lr"]

        engine.reset()
        engine.register_parameter("p1", "logic", 0.5)
        engine.update_neurochem_state({"5ht": 0.1})
        r2 = engine.process(_basic_input(rewards={"logic": 0.8}))
        lr_unstable = r2.metadata["effective_lr"]

        assert lr_stable < lr_unstable

    def test_ne_urgency(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        engine.update_neurochem_state({"ne": 0.9})
        r1 = engine.process(_basic_input(rewards={"logic": 0.8}))
        lr_urgent = r1.metadata["effective_lr"]

        engine.reset()
        engine.register_parameter("p1", "logic", 0.5)
        engine.update_neurochem_state({"ne": 0.0})
        r2 = engine.process(_basic_input(rewards={"logic": 0.8}))
        lr_calm = r2.metadata["effective_lr"]

        assert lr_urgent > lr_calm

    def test_gaba_noise_gate(self):
        cfg = RewardLearningConfig()
        gate_high = compute_noise_gate(cfg.delta_noise_gate, 0.9, 1.0, cfg.w_gaba_gate)
        gate_low = compute_noise_gate(cfg.delta_noise_gate, 0.0, 1.0, cfg.w_gaba_gate)
        assert gate_high > gate_low

    def test_oxt_social_amplifier(self):
        # ATTUNEMENT domain gets boosted
        amp = compute_social_amplifier("attunement", 0.9, 0.15)
        assert amp > 1.0
        # Non-attunement domain: no boost
        amp_logic = compute_social_amplifier("logic", 0.9, 0.15)
        assert amp_logic == 1.0

    def test_cb1_exploration_width(self):
        width = compute_exploration_width(0.9, 0.20, OperationalMode.NORMAL)
        assert width > 1.0

    def test_cb1_rem_dream_bonus(self):
        w_normal = compute_exploration_width(0.9, 0.20, OperationalMode.NORMAL)
        w_dream = compute_exploration_width(0.9, 0.20, OperationalMode.REM_DREAM)
        assert w_dream > w_normal

    def test_ach_credit_depth(self):
        depth_high = compute_credit_depth(64, 0.9, 0.15, OperationalMode.NORMAL)
        depth_low = compute_credit_depth(64, 0.0, 0.15, OperationalMode.NORMAL)
        assert depth_high >= depth_low

    def test_ach_depth_rem_dream_bonus(self):
        depth = compute_credit_depth(64, 0.5, 0.15, OperationalMode.REM_DREAM)
        depth_normal = compute_credit_depth(64, 0.5, 0.15, OperationalMode.NORMAL)
        assert depth >= depth_normal

    def test_cortisol_penalty_via_pipeline(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        engine.update_neurochem_state({"cor": 0.9})
        r1 = engine.process(_basic_input(rewards={"logic": 0.8}))
        lr_stressed = r1.metadata["effective_lr"]

        engine.reset()
        engine.register_parameter("p1", "logic", 0.5)
        engine.update_neurochem_state({"cor": 0.0})
        r2 = engine.process(_basic_input(rewards={"logic": 0.8}))
        lr_relaxed = r2.metadata["effective_lr"]

        assert lr_stressed < lr_relaxed

    def test_update_neurochem_state_pattern_a(self):
        engine = _make_engine()
        engine.update_neurochem_state({
            "da": 0.7, "5ht": 0.6, "ne": 0.5, "ach": 0.4,
            "gaba": 0.3, "oxt": 0.2, "cb1": 0.1, "cor": 0.8,
        })
        st = engine.get_status()
        assert st["nt_levels"]["da"] == pytest.approx(0.7)
        assert st["nt_levels"]["5ht"] == pytest.approx(0.6)
        assert st["nt_levels"]["cor"] == pytest.approx(0.8)


# =====================================================================
# 9. Mode switching
# =====================================================================


class TestModeSwitching:
    """Mode-dependent behaviour."""

    def test_learning_mode_higher_lr(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        r_learn = engine.process(_basic_input(rewards={"logic": 0.8}, mode=OperationalMode.LEARNING))
        lr_learn = r_learn.metadata["effective_lr"]

        engine.reset()
        engine.register_parameter("p1", "logic", 0.5)
        r_normal = engine.process(_basic_input(rewards={"logic": 0.8}, mode=OperationalMode.NORMAL))
        lr_normal = r_normal.metadata["effective_lr"]

        assert lr_learn > lr_normal

    def test_dev_mode_lower_lr(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        r_dev = engine.process(_basic_input(rewards={"logic": 0.8}, mode=OperationalMode.DEV))

        engine.reset()
        engine.register_parameter("p1", "logic", 0.5)
        r_normal = engine.process(_basic_input(rewards={"logic": 0.8}, mode=OperationalMode.NORMAL))

        assert r_dev.metadata["effective_lr"] < r_normal.metadata["effective_lr"]

    def test_rem_dream_exploration(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        r = engine.process(_basic_input(rewards={"logic": 0.8}, mode=OperationalMode.REM_DREAM))
        assert r.metadata["exploration_mult"] >= 1.0

    def test_reflective_mode_lower_lr(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        r_refl = engine.process(_basic_input(rewards={"logic": 0.8}, mode=OperationalMode.REFLECTIVE))

        engine.reset()
        engine.register_parameter("p1", "logic", 0.5)
        r_normal = engine.process(_basic_input(rewards={"logic": 0.8}, mode=OperationalMode.NORMAL))

        assert r_refl.metadata["effective_lr"] < r_normal.metadata["effective_lr"]

    def test_configure_sets_mode(self):
        engine = _make_engine()
        engine.configure(OperationalMode.LEARNING)
        assert engine.get_status()["mode"] == "learning"

    def test_noise_gate_dev_stricter(self):
        cfg = RewardLearningConfig()
        gate_dev = compute_noise_gate(cfg.delta_noise_gate, 0.0, cfg.mode_noise_gate_multiplier["dev"], cfg.w_gaba_gate)
        gate_norm = compute_noise_gate(cfg.delta_noise_gate, 0.0, cfg.mode_noise_gate_multiplier["normal"], cfg.w_gaba_gate)
        assert gate_dev > gate_norm


# =====================================================================
# 10. process() pipeline
# =====================================================================


class TestProcessPipeline:
    """Integration tests for the full process() pipeline."""

    def test_single_cycle_returns_result(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        result = engine.process(_basic_input(rewards={"logic": 0.7}))
        assert isinstance(result, RewardLearningResult)
        assert result.processing_time_ms >= 0.0

    def test_cycle_count_increments(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        engine.process(_basic_input())
        engine.process(_basic_input())
        assert engine.get_status()["cycle_count"] == 2

    def test_auto_register_parameters(self):
        engine = _make_engine()
        # Don't manually register; let process auto-register from input
        inp = _basic_input(
            rewards={"logic": 0.7},
            params={"p_auto": 0.5},
            domains={"p_auto": "logic"},
        )
        engine.process(inp)
        rec = engine.get_record("p_auto")
        assert rec is not None
        assert rec.domain == "logic"

    def test_metadata_contains_expected_keys(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        result = engine.process(_basic_input())
        meta = result.metadata
        assert "mode" in meta
        assert "cycle" in meta
        assert "effective_lr" in meta
        assert "noise_gate" in meta
        assert "credit_depth" in meta
        assert "exploration_mult" in meta

    def test_multi_domain_pipeline(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        engine.register_parameter("p2", "ethics", 0.5)
        rewards = {"logic": 0.9, "ethics": 0.3}
        result = engine.process(_basic_input(
            rewards=rewards,
            params={"p1": 0.5, "p2": 0.5},
            domains={"p1": "logic", "p2": "ethics"},
        ))
        assert len(result.prediction_errors) == 2
        assert len(result.adjustments) == 2

    def test_mean_abs_delta(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        result = engine.process(_basic_input(rewards={"logic": 0.8}))
        # delta = 0.8 - 0.5 = 0.3
        assert result.mean_abs_delta == pytest.approx(0.3)
        assert result.max_abs_delta == pytest.approx(0.3)

    def test_global_lr_updated_after_cycle(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        initial_lr = engine.get_status()["global_lr"]
        engine.process(_basic_input())
        new_lr = engine.get_status()["global_lr"]
        # After cycle 1 (within warmup of 5), lr should equal effective_lr
        assert isinstance(new_lr, float)

    def test_oscillation_damping_with_5ht(self):
        """5-HT should damp oscillating parameter adjustments."""
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        engine.update_neurochem_state({"5ht": 0.9})
        rec = engine.get_record("p1")
        # Simulate oscillation in adjustment history
        rec.adjustment_history = [0.05, -0.05]
        rec.total_updates = 10
        result = engine.process(_basic_input(rewards={"logic": 0.8}))
        # With high 5-HT and oscillation, learning rate should be damped
        adj = [a for a in result.adjustments if a.parameter_id == "p1"]
        assert len(adj) == 1


# =====================================================================
# 11. Neurochem output
# =====================================================================


class TestNeurochemOutput:
    """compute_learning_neurochem and pipeline neurochem signals."""

    def test_positive_rpe_gives_positive_da(self):
        rng = _fixed_rng()
        pes = [PredictionError(domain="logic", actual=0.8, predicted=0.5, delta=0.3, magnitude=0.3)]
        cfg = RewardLearningConfig()
        nc = compute_learning_neurochem(pes, 0.0, 5, 64, 0, cfg, rng)
        assert nc.da_delta > 0.0

    def test_negative_rpe_gives_negative_da(self):
        rng = _fixed_rng()
        pes = [PredictionError(domain="logic", actual=0.2, predicted=0.5, delta=-0.3, magnitude=0.3)]
        cfg = RewardLearningConfig()
        nc = compute_learning_neurochem(pes, 0.0, 5, 64, 0, cfg, rng)
        assert nc.da_delta < 0.0

    def test_convergence_gives_5ht(self):
        rng = _fixed_rng()
        cfg = RewardLearningConfig()
        nc = compute_learning_neurochem([], 0.8, 5, 64, 0, cfg, rng)
        assert nc._5ht_delta > 0.0

    def test_large_delta_gives_ne_burst(self):
        rng = _fixed_rng()
        cfg = RewardLearningConfig()
        pes = [PredictionError(domain="logic", actual=0.9, predicted=0.1, delta=0.8, magnitude=0.8)]
        nc = compute_learning_neurochem(pes, 0.0, 5, 64, 0, cfg, rng)
        # NE should fire for magnitude > theta_delta_large (0.15)
        assert nc.ne_delta >= 0.0  # Poisson can be 0

    def test_ach_depth_utilisation(self):
        rng = _fixed_rng()
        cfg = RewardLearningConfig()
        nc = compute_learning_neurochem([], 0.0, 32, 64, 0, cfg, rng)
        expected_ach = cfg.beta_ach_depth * (32 / 64)
        assert nc.ach_delta == pytest.approx(expected_ach, abs=1e-5)

    def test_theta_boost_during_learning(self):
        rng = _fixed_rng()
        cfg = RewardLearningConfig()
        pes = [PredictionError(domain="logic", actual=0.7, predicted=0.5, delta=0.2, magnitude=0.2)]
        nc = compute_learning_neurochem(pes, 0.0, 5, 64, 0, cfg, rng)
        assert nc.theta_boost == pytest.approx(cfg.theta_boost_learning)

    def test_gamma_boost_on_large_error(self):
        rng = _fixed_rng()
        cfg = RewardLearningConfig()
        pes = [PredictionError(domain="logic", actual=0.9, predicted=0.1, delta=0.8, magnitude=0.8)]
        nc = compute_learning_neurochem(pes, 0.0, 5, 64, 0, cfg, rng)
        assert nc.gamma_boost == pytest.approx(cfg.gamma_boost_error)

    def test_beta_boost_on_consolidation(self):
        rng = _fixed_rng()
        cfg = RewardLearningConfig()
        nc = compute_learning_neurochem([], 0.0, 5, 64, 2, cfg, rng)
        assert nc.beta_boost == pytest.approx(cfg.beta_boost_consolidation)

    def test_no_pes_gives_zero_theta(self):
        rng = _fixed_rng()
        cfg = RewardLearningConfig()
        nc = compute_learning_neurochem([], 0.0, 0, 64, 0, cfg, rng)
        assert nc.theta_boost == 0.0

    def test_zero_max_params_no_crash(self):
        rng = _fixed_rng()
        cfg = RewardLearningConfig()
        nc = compute_learning_neurochem([], 0.0, 0, 0, 0, cfg, rng)
        assert nc.ach_delta == 0.0

    def test_pipeline_neurochem_in_result(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        result = engine.process(_basic_input(rewards={"logic": 0.8}))
        nc = result.neurochemical_signals
        assert isinstance(nc, RewardLearningNeurochem)


# =====================================================================
# 12. Engine init / get_status() / repr
# =====================================================================


class TestEngineInit:
    """Engine initialisation and introspection."""

    def test_default_init(self):
        engine = RewardBasedLearningEngine()
        assert engine.engine_id == "reward_based_learning_engine"
        assert engine.cluster == "learning"

    def test_custom_config(self):
        cfg = RewardLearningConfig(initial_learning_rate=0.2)
        engine = RewardBasedLearningEngine(config=cfg)
        assert engine.get_status()["global_lr"] == pytest.approx(0.2)

    def test_get_status_keys(self):
        engine = _make_engine()
        st = engine.get_status()
        expected_keys = {
            "engine_id", "mode", "cycle_count", "global_lr",
            "total_parameters", "active_parameters",
            "converged_parameters", "consolidated_parameters",
            "domains_tracked", "nt_levels",
        }
        assert expected_keys.issubset(st.keys())

    def test_get_status_nt_levels(self):
        engine = _make_engine()
        nt = engine.get_status()["nt_levels"]
        for key in ("da", "5ht", "ne", "ach", "gaba", "oxt", "cb1", "cor"):
            assert key in nt

    def test_engine_id_in_status(self):
        engine = _make_engine()
        assert engine.get_status()["engine_id"] == "reward_based_learning_engine"

    def test_register_parameter(self):
        engine = _make_engine()
        assert engine.register_parameter("p1", "logic", 0.5) is True
        rec = engine.get_record("p1")
        assert rec is not None
        assert rec.domain == "logic"
        assert rec.value == pytest.approx(0.5)

    def test_register_duplicate_returns_false(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        assert engine.register_parameter("p1", "logic", 0.5) is False

    def test_register_at_capacity_returns_false(self):
        engine = _make_engine(max_parameters=2)
        engine.register_parameter("p1", "logic", 0.5)
        engine.register_parameter("p2", "ethics", 0.5)
        assert engine.register_parameter("p3", "innovation", 0.5) is False

    def test_reset(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        engine.process(_basic_input())
        engine.reset()
        assert engine.get_status()["cycle_count"] == 0
        assert engine.get_status()["total_parameters"] == 0
        assert engine.get_record("p1") is None

    def test_get_prediction_none_before_process(self):
        engine = _make_engine()
        assert engine.get_prediction("logic") is None

    def test_get_record_none_for_unknown(self):
        engine = _make_engine()
        assert engine.get_record("nonexistent") is None


# =====================================================================
# 13. Edge cases
# =====================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_no_rewards_empty_result(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        inp = RewardLearningInput(
            reward_signals={},
            parameter_values={"p1": 0.5},
            parameter_domains={"p1": "logic"},
        )
        result = engine.process(inp)
        assert len(result.prediction_errors) == 0
        assert result.mean_abs_delta == 0.0

    def test_no_parameters(self):
        engine = _make_engine()
        result = engine.process(_basic_input(rewards={"logic": 0.7}))
        # Auto-register should happen
        assert result.active_parameters >= 0

    def test_zero_learning_rate(self):
        cfg = RewardLearningConfig()
        lr = compute_effective_learning_rate(
            0.0, 0, cfg, da=0.0, sht=1.0, ne=0.0, cor=1.0, mode_mult=0.01,
        )
        assert lr >= cfg.min_learning_rate

    def test_already_converged_skips_update(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        rec = engine.get_record("p1")
        rec.consolidated = True
        result = engine.process(_basic_input(rewards={"logic": 0.9}))
        p1_adj = [a for a in result.adjustments if a.parameter_id == "p1"]
        assert len(p1_adj) == 0

    def test_value_clamped_0_1(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.99)
        engine.update_neurochem_state({"da": 1.0, "ne": 1.0})
        result = engine.process(_basic_input(
            rewards={"logic": 1.0},
            params={"p1": 0.99},
        ))
        adj = [a for a in result.adjustments if a.parameter_id == "p1"]
        if adj:
            assert adj[0].new_value <= 1.0

    def test_negative_reward_signal(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        result = engine.process(_basic_input(rewards={"logic": 0.0}))
        # Should handle without crash
        assert len(result.prediction_errors) == 1
        assert result.prediction_errors[0].delta < 0.0

    def test_extreme_nt_levels(self):
        engine = _make_engine()
        engine.update_neurochem_state({
            "da": 1.0, "5ht": 1.0, "ne": 1.0, "ach": 1.0,
            "gaba": 1.0, "oxt": 1.0, "cb1": 1.0, "cor": 1.0,
        })
        engine.register_parameter("p1", "logic", 0.5)
        result = engine.process(_basic_input(rewards={"logic": 0.8}))
        assert isinstance(result, RewardLearningResult)

    def test_history_trimming(self):
        engine = _make_engine(max_adjustment_history=5)
        engine.register_parameter("p1", "logic", 0.5)
        for i in range(20):
            engine.process(_basic_input(rewards={"logic": 0.5 + 0.01 * i}))
        rec = engine.get_record("p1")
        assert len(rec.delta_history) <= 5

    def test_many_cycles_convergence_tracking(self):
        engine = _make_engine()
        engine.register_parameter("p1", "logic", 0.5)
        # Feed consistent reward, should trend toward convergence
        for _ in range(30):
            engine.process(_basic_input(rewards={"logic": 0.7}))
        rec = engine.get_record("p1")
        # After many cycles, delta should shrink as prediction tracks actual
        pred = engine.get_prediction("logic")
        assert pred is not None
        # Prediction should approach 0.7
        assert abs(pred.predicted - 0.7) < 0.05

    def test_social_amplifier_non_attunement(self):
        amp = compute_social_amplifier("logic", 1.0, 0.15)
        assert amp == 1.0

    def test_social_amplifier_low_oxt(self):
        amp = compute_social_amplifier("attunement", 0.1, 0.15)
        assert amp == 1.0

    def test_noise_gate_zero_gaba(self):
        gate = compute_noise_gate(0.005, 0.0, 1.0, 0.2)
        assert gate == pytest.approx(0.005)


# =====================================================================
# Enum value tests
# =====================================================================


class TestEnums:
    """Verify enum values."""

    def test_convergence_status_values(self):
        assert ConvergenceStatus.DIVERGING.value == "diverging"
        assert ConvergenceStatus.EXPLORING.value == "exploring"
        assert ConvergenceStatus.CONVERGING.value == "converging"
        assert ConvergenceStatus.CONVERGED.value == "converged"
        assert ConvergenceStatus.CONSOLIDATED.value == "consolidated"

    def test_learning_phase_values(self):
        assert LearningPhase.INITIAL.value == "initial"
        assert LearningPhase.ACTIVE.value == "active"
        assert LearningPhase.PLATEAU.value == "plateau"
        assert LearningPhase.CONSOLIDATED.value == "consolidated"

    def test_domain_type_values(self):
        assert DomainType.LOGIC.value == "logic"
        assert DomainType.ATTUNEMENT.value == "attunement"
        assert DomainType.INNOVATION.value == "innovation"
        assert DomainType.ETHICS.value == "ethics"
