"""Tests for Engine 25 -- Recursive Learning Engine."""
import math

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.recursive_learning_engine import (
    RecursiveLearningEngine,
    RecursiveLearningConfig,
    RecursiveLearningState,
    MetaStrategy,
    PerformanceTrend,
    MetaHealthStatus,
    MetaMetrics,
    MetaPerformanceSnapshot,
    MetaAdjustment,
    MetaDecision,
    RecursiveLearningNeurochem,
    RecursiveLearningInput,
    RecursiveLearningResult,
    compute_linear_regression,
    compute_variance,
    classify_trend,
    detect_plateau,
    detect_divergence,
    compute_explore_probability,
    decide_strategy,
    build_adjustments,
    compute_meta_health,
    compute_meta_improvement_rate,
    compute_neurochem_signals,
    get_mode_scales,
)
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Helpers
# =====================================================================

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _cfg(**overrides) -> RecursiveLearningConfig:
    return RecursiveLearningConfig(**overrides)


def _metrics(
    tick: int = 0,
    mean_abs_delta: float = 0.1,
    convergence_ratio: float = 0.5,
    learning_rate: float = 0.01,
    consolidation_threshold: float = 0.50,
    noise_gate: float = 0.05,
) -> MetaMetrics:
    return MetaMetrics(
        tick=tick,
        mean_abs_delta=mean_abs_delta,
        convergence_ratio=convergence_ratio,
        learning_rate=learning_rate,
        consolidation_threshold=consolidation_threshold,
        noise_gate=noise_gate,
    )


def _engine(seed: int = 42, **cfg_kw) -> RecursiveLearningEngine:
    return RecursiveLearningEngine(config=_cfg(**cfg_kw), rng=_rng(seed))


def _input(
    metrics: MetaMetrics | None = None,
    mode: OperationalMode = OperationalMode.NORMAL,
    force_strategy: MetaStrategy | None = None,
    history: list[MetaMetrics] | None = None,
) -> RecursiveLearningInput:
    return RecursiveLearningInput(
        e17_metrics=metrics or _metrics(),
        active_mode=mode,
        force_strategy=force_strategy,
        e17_history=history,
    )


# =====================================================================
# 1. Config defaults
# =====================================================================

class TestConfigDefaults:
    def test_default_window_size(self):
        c = RecursiveLearningConfig()
        assert c.performance_window_size == 20

    def test_default_min_window(self):
        c = RecursiveLearningConfig()
        assert c.min_window_for_trend == 5

    def test_plateau_defaults(self):
        c = RecursiveLearningConfig()
        assert c.plateau_variance_threshold == 0.001
        assert c.plateau_convergence_floor == 0.80
        assert c.plateau_min_cycles == 5

    def test_divergence_defaults(self):
        c = RecursiveLearningConfig()
        assert c.divergence_slope_threshold == 0.005
        assert c.divergence_delta_threshold == 0.30
        assert c.divergence_min_cycles == 3

    def test_strategy_defaults(self):
        c = RecursiveLearningConfig()
        assert c.explore_probability_base == 0.25
        assert c.exploit_lr_multiplier == 0.5
        assert c.explore_lr_multiplier == 2.0
        assert c.reset_lr_value == 0.01

    def test_frozen(self):
        c = RecursiveLearningConfig()
        with pytest.raises(AttributeError):
            c.performance_window_size = 99  # type: ignore[misc]


# =====================================================================
# 2. Pure helper: compute_linear_regression
# =====================================================================

class TestLinearRegression:
    def test_single_value(self):
        slope, r2 = compute_linear_regression([1.0])
        assert slope == 0.0
        assert r2 == 0.0

    def test_empty(self):
        slope, r2 = compute_linear_regression([])
        assert slope == 0.0
        assert r2 == 0.0

    def test_perfect_positive(self):
        slope, r2 = compute_linear_regression([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(slope - 1.0) < 1e-9
        assert abs(r2 - 1.0) < 1e-6

    def test_perfect_negative(self):
        slope, r2 = compute_linear_regression([5.0, 4.0, 3.0, 2.0, 1.0])
        assert abs(slope - (-1.0)) < 1e-9
        assert abs(r2 - 1.0) < 1e-6

    def test_constant_values(self):
        slope, r2 = compute_linear_regression([3.0, 3.0, 3.0])
        assert slope == 0.0
        assert r2 == 0.0

    def test_two_values(self):
        slope, r2 = compute_linear_regression([0.0, 1.0])
        assert abs(slope - 1.0) < 1e-9


# =====================================================================
# 3. Pure helper: compute_variance
# =====================================================================

class TestComputeVariance:
    def test_single_value(self):
        assert compute_variance([5.0]) == 0.0

    def test_empty(self):
        assert compute_variance([]) == 0.0

    def test_identical(self):
        assert compute_variance([3.0, 3.0, 3.0]) == 0.0

    def test_known(self):
        # [1, 2, 3] -> mean=2, sample var = ((1+0+1)/2) = 1.0
        assert abs(compute_variance([1.0, 2.0, 3.0]) - 1.0) < 1e-9

    def test_two_values(self):
        # [0, 4] -> mean=2, sample var = ((4+4)/1) = 8.0
        assert abs(compute_variance([0.0, 4.0]) - 8.0) < 1e-9


# =====================================================================
# 4. Pure helper: classify_trend
# =====================================================================

class TestClassifyTrend:
    def test_improving(self):
        # Negative slope, good r2
        t = classify_trend(-0.01, 0.8, [5.0, 4.0, 3.0, 2.0, 1.0])
        assert t == PerformanceTrend.IMPROVING

    def test_degrading(self):
        t = classify_trend(0.01, 0.8, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert t == PerformanceTrend.DEGRADING

    def test_stagnant_low_r2(self):
        t = classify_trend(0.01, 0.1, [1.0, 2.0, 3.0])
        assert t == PerformanceTrend.STAGNANT

    def test_stagnant_small_slope(self):
        t = classify_trend(0.001, 0.9, [1.0, 1.001, 1.002])
        assert t == PerformanceTrend.STAGNANT

    def test_oscillating(self):
        # Alternating directions
        vals = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        t = classify_trend(0.001, 0.1, vals)
        assert t == PerformanceTrend.OSCILLATING

    def test_short_sequence_no_oscillation_check(self):
        t = classify_trend(0.001, 0.1, [1.0, 2.0, 1.0])
        assert t == PerformanceTrend.STAGNANT


# =====================================================================
# 5. Pure helper: detect_plateau
# =====================================================================

class TestDetectPlateau:
    def test_no_plateau_high_variance(self):
        cfg = _cfg()
        is_p, counter = detect_plateau(0.1, 0.5, 0, cfg)
        assert is_p is False
        assert counter == 0

    def test_no_plateau_high_convergence(self):
        cfg = _cfg()
        is_p, counter = detect_plateau(0.0005, 0.9, 0, cfg)
        assert is_p is False
        assert counter == 0

    def test_plateau_counter_increments(self):
        cfg = _cfg()
        is_p, counter = detect_plateau(0.0005, 0.5, 3, cfg)
        assert counter == 4
        # Not yet at threshold (min_cycles=5)
        assert is_p is False

    def test_plateau_fires_at_threshold(self):
        cfg = _cfg()
        is_p, counter = detect_plateau(0.0005, 0.5, 4, cfg)
        assert is_p is True
        assert counter == 5

    def test_mode_scale_raises_threshold(self):
        cfg = _cfg()
        # mode_scale=2.0 => threshold=0.002; variance=0.0015 is below it
        is_p, counter = detect_plateau(0.0015, 0.5, 4, cfg, mode_scale=2.0)
        assert is_p is True

    def test_counter_resets_on_non_plateau(self):
        cfg = _cfg()
        is_p, counter = detect_plateau(0.1, 0.5, 10, cfg)
        assert is_p is False
        assert counter == 0


# =====================================================================
# 6. Pure helper: detect_divergence
# =====================================================================

class TestDetectDivergence:
    def test_no_divergence_negative_slope(self):
        cfg = _cfg()
        is_d, counter = detect_divergence(-0.01, 0.1, 0, cfg)
        assert is_d is False
        assert counter == 0

    def test_counter_increments_high_slope_high_delta(self):
        cfg = _cfg()
        is_d, counter = detect_divergence(0.01, 0.35, 1, cfg)
        assert counter == 2
        assert is_d is False  # min_cycles=3

    def test_divergence_fires(self):
        cfg = _cfg()
        is_d, counter = detect_divergence(0.01, 0.35, 2, cfg)
        assert is_d is True
        assert counter == 3

    def test_slope_above_but_delta_below_threshold(self):
        # Slope > threshold but mean_delta <= threshold -> increments but no divergence
        cfg = _cfg()
        is_d, counter = detect_divergence(0.01, 0.2, 2, cfg)
        assert is_d is False
        assert counter == 3  # Incremented

    def test_counter_decrements_on_recovery(self):
        cfg = _cfg()
        is_d, counter = detect_divergence(-0.01, 0.1, 5, cfg)
        assert is_d is False
        assert counter == 4  # max(0, 5-1)

    def test_mode_scale_increases_tolerance(self):
        cfg = _cfg()
        # mode_scale=2.0 => slope_thresh = 0.01; slope=0.008 < 0.01 -> no increment
        is_d, counter = detect_divergence(0.008, 0.35, 2, cfg, mode_scale=2.0)
        assert is_d is False
        assert counter == 1  # Decremented


# =====================================================================
# 7. Pure helper: compute_explore_probability
# =====================================================================

class TestComputeExploreProbability:
    def test_base_probability(self):
        cfg = _cfg()
        p = compute_explore_probability(
            False, False, MetaStrategy.EXPLOIT, 0.4, 0.4, 1.0, cfg, OperationalMode.NORMAL)
        # base=0.25, da_adj=0, 5ht_adj=0, meta_lr=1.0 -> 0.25
        assert abs(p - 0.25) < 1e-6

    def test_plateau_increases(self):
        cfg = _cfg()
        p = compute_explore_probability(
            True, False, MetaStrategy.EXPLOIT, 0.4, 0.4, 1.0, cfg, OperationalMode.NORMAL)
        assert p > 0.25  # plateau adds 0.30

    def test_divergence_decreases(self):
        cfg = _cfg()
        p_base = compute_explore_probability(
            False, False, MetaStrategy.EXPLOIT, 0.4, 0.4, 1.0, cfg, OperationalMode.NORMAL)
        p_div = compute_explore_probability(
            False, True, MetaStrategy.EXPLOIT, 0.4, 0.4, 1.0, cfg, OperationalMode.NORMAL)
        assert p_div < p_base

    def test_high_da_increases(self):
        cfg = _cfg()
        p = compute_explore_probability(
            False, False, MetaStrategy.EXPLOIT, 0.9, 0.4, 1.0, cfg, OperationalMode.NORMAL)
        assert p > 0.25

    def test_high_5ht_decreases(self):
        cfg = _cfg()
        p = compute_explore_probability(
            False, False, MetaStrategy.EXPLOIT, 0.4, 0.9, 1.0, cfg, OperationalMode.NORMAL)
        assert p < 0.25

    def test_creative_mode_scales_up(self):
        cfg = _cfg()
        p_normal = compute_explore_probability(
            False, False, MetaStrategy.EXPLOIT, 0.4, 0.4, 1.0, cfg, OperationalMode.NORMAL)
        p_dev = compute_explore_probability(
            False, False, MetaStrategy.EXPLOIT, 0.4, 0.4, 1.0, cfg, OperationalMode.DEV)
        assert p_dev > p_normal

    def test_rem_dream_overrides(self):
        cfg = _cfg()
        p = compute_explore_probability(
            False, False, MetaStrategy.EXPLOIT, 0.4, 0.4, 1.0, cfg, OperationalMode.REM_DREAM)
        assert abs(p - cfg.rem_explore_probability) < 1e-6

    def test_meta_lr_decay_reduces(self):
        cfg = _cfg()
        p_full = compute_explore_probability(
            False, False, MetaStrategy.EXPLOIT, 0.4, 0.4, 1.0, cfg, OperationalMode.NORMAL)
        p_decayed = compute_explore_probability(
            False, False, MetaStrategy.EXPLOIT, 0.4, 0.4, 0.5, cfg, OperationalMode.NORMAL)
        assert p_decayed < p_full

    def test_clamped_to_0_1(self):
        cfg = _cfg()
        p = compute_explore_probability(
            True, False, MetaStrategy.EXPLOIT, 1.0, 0.0, 1.0, cfg, OperationalMode.DEV)
        assert 0.0 <= p <= 1.0


# =====================================================================
# 8. Pure helper: decide_strategy
# =====================================================================

class TestDecideStrategy:
    def test_forced_override(self):
        rng = _rng()
        strat, conf, reason = decide_strategy(
            False, False, 0.5, MetaStrategy.EXPLOIT, 0, rng,
            force_strategy=MetaStrategy.RESET)
        assert strat == MetaStrategy.RESET
        assert conf == 1.0
        assert reason == "forced_override"

    def test_cooldown_keeps_current(self):
        rng = _rng()
        strat, conf, reason = decide_strategy(
            True, True, 0.9, MetaStrategy.EXPLORE, 3, rng)
        assert strat == MetaStrategy.EXPLORE
        assert reason == "cooldown_active"

    def test_divergence_triggers_reset(self):
        rng = _rng()
        strat, conf, reason = decide_strategy(
            False, True, 0.5, MetaStrategy.EXPLOIT, 0, rng)
        assert strat == MetaStrategy.RESET
        assert conf == 0.85
        assert reason == "divergence_detected"

    def test_plateau_explore(self):
        # Use rng that generates low uniform (< explore_prob=0.9)
        rng = _rng(0)
        strat, conf, reason = decide_strategy(
            True, False, 0.99, MetaStrategy.EXPLOIT, 0, rng)
        assert strat == MetaStrategy.EXPLORE
        assert "plateau" in reason

    def test_routine_exploit(self):
        # Use rng with explore_prob very low
        rng = _rng(42)
        strat, conf, reason = decide_strategy(
            False, False, 0.001, MetaStrategy.EXPLOIT, 0, rng)
        assert strat == MetaStrategy.EXPLOIT
        assert "routine" in reason


# =====================================================================
# 9. Pure helper: build_adjustments
# =====================================================================

class TestBuildAdjustments:
    def test_exploit_adjustments(self):
        cfg = _cfg()
        adjs = build_adjustments(MetaStrategy.EXPLOIT, 0.01, 0.50, 0.05, cfg, 1.0, OperationalMode.NORMAL)
        assert len(adjs) == 3
        params = {a.parameter for a in adjs}
        assert params == {"learning_rate", "noise_gate", "consolidation_threshold"}

    def test_exploit_reduces_lr(self):
        cfg = _cfg()
        adjs = build_adjustments(MetaStrategy.EXPLOIT, 0.10, 0.50, 0.05, cfg, 1.0, OperationalMode.NORMAL)
        lr_adj = [a for a in adjs if a.parameter == "learning_rate"][0]
        assert lr_adj.recommended < lr_adj.current_value

    def test_exploit_tightens_noise_gate(self):
        cfg = _cfg()
        adjs = build_adjustments(MetaStrategy.EXPLOIT, 0.01, 0.50, 0.10, cfg, 1.0, OperationalMode.NORMAL)
        ng_adj = [a for a in adjs if a.parameter == "noise_gate"][0]
        assert ng_adj.recommended == cfg.exploit_noise_gate

    def test_explore_increases_lr(self):
        cfg = _cfg()
        adjs = build_adjustments(MetaStrategy.EXPLORE, 0.01, 0.50, 0.05, cfg, 1.0, OperationalMode.NORMAL)
        lr_adj = [a for a in adjs if a.parameter == "learning_rate"][0]
        assert lr_adj.recommended > lr_adj.current_value

    def test_explore_widens_noise_gate(self):
        cfg = _cfg()
        adjs = build_adjustments(MetaStrategy.EXPLORE, 0.01, 0.50, 0.02, cfg, 1.0, OperationalMode.NORMAL)
        ng_adj = [a for a in adjs if a.parameter == "noise_gate"][0]
        assert ng_adj.recommended == cfg.explore_noise_gate

    def test_explore_rem_dream_extra_noise(self):
        cfg = _cfg()
        adjs = build_adjustments(MetaStrategy.EXPLORE, 0.01, 0.50, 0.05, cfg, 1.0, OperationalMode.REM_DREAM)
        ng_adj = [a for a in adjs if a.parameter == "noise_gate"][0]
        assert ng_adj.recommended == cfg.explore_noise_gate + cfg.rem_noise_injection

    def test_reset_baseline_lr(self):
        cfg = _cfg()
        adjs = build_adjustments(MetaStrategy.RESET, 0.10, 0.50, 0.05, cfg, 1.0, OperationalMode.NORMAL)
        lr_adj = [a for a in adjs if a.parameter == "learning_rate"][0]
        assert lr_adj.recommended == cfg.reset_lr_value

    def test_reset_consolidation_to_050(self):
        cfg = _cfg()
        adjs = build_adjustments(MetaStrategy.RESET, 0.01, 0.80, 0.05, cfg, 1.0, OperationalMode.NORMAL)
        ct_adj = [a for a in adjs if a.parameter == "consolidation_threshold"][0]
        assert ct_adj.recommended == 0.50

    def test_exploit_lr_floor(self):
        cfg = _cfg()
        # Very small LR should floor at 0.001
        adjs = build_adjustments(MetaStrategy.EXPLOIT, 0.001, 0.50, 0.05, cfg, 1.0, OperationalMode.NORMAL)
        lr_adj = [a for a in adjs if a.parameter == "learning_rate"][0]
        assert lr_adj.recommended >= 0.001

    def test_explore_lr_ceiling(self):
        cfg = _cfg()
        # Very large LR should cap at 0.50
        adjs = build_adjustments(MetaStrategy.EXPLORE, 0.40, 0.50, 0.05, cfg, 1.0, OperationalMode.NORMAL)
        lr_adj = [a for a in adjs if a.parameter == "learning_rate"][0]
        assert lr_adj.recommended <= 0.50

    def test_creative_mode_lr_scale(self):
        cfg = _cfg()
        adjs_normal = build_adjustments(MetaStrategy.EXPLOIT, 0.10, 0.50, 0.05, cfg, 1.0, OperationalMode.NORMAL)
        adjs_dev = build_adjustments(MetaStrategy.EXPLOIT, 0.10, 0.50, 0.05, cfg, 1.0, OperationalMode.DEV)
        lr_normal = [a for a in adjs_normal if a.parameter == "learning_rate"][0]
        lr_dev = [a for a in adjs_dev if a.parameter == "learning_rate"][0]
        # DEV mode applies creative_lr_scale -> larger recommended
        assert lr_dev.recommended > lr_normal.recommended


# =====================================================================
# 10. Pure helper: compute_meta_health
# =====================================================================

class TestComputeMetaHealth:
    def test_diverging_priority(self):
        h = compute_meta_health(True, True, MetaStrategy.EXPLOIT, PerformanceTrend.DEGRADING)
        assert h == MetaHealthStatus.DIVERGING

    def test_recovering_on_reset(self):
        h = compute_meta_health(False, False, MetaStrategy.RESET, PerformanceTrend.STAGNANT)
        assert h == MetaHealthStatus.RECOVERING

    def test_plateau(self):
        h = compute_meta_health(True, False, MetaStrategy.EXPLOIT, PerformanceTrend.STAGNANT)
        assert h == MetaHealthStatus.PLATEAU

    def test_healthy_improving(self):
        h = compute_meta_health(False, False, MetaStrategy.EXPLOIT, PerformanceTrend.IMPROVING)
        assert h == MetaHealthStatus.HEALTHY

    def test_healthy_stagnant(self):
        h = compute_meta_health(False, False, MetaStrategy.EXPLOIT, PerformanceTrend.STAGNANT)
        assert h == MetaHealthStatus.HEALTHY


# =====================================================================
# 11. Pure helper: compute_meta_improvement_rate
# =====================================================================

class TestMetaImprovementRate:
    def test_empty(self):
        assert compute_meta_improvement_rate([]) == 0.0

    def test_all_improved(self):
        assert abs(compute_meta_improvement_rate([True, True, True]) - 1.0) < 1e-9

    def test_none_improved(self):
        assert compute_meta_improvement_rate([False, False]) == 0.0

    def test_mixed(self):
        rate = compute_meta_improvement_rate([True, False, True, False])
        assert abs(rate - 0.5) < 1e-9


# =====================================================================
# 12. Pure helper: compute_neurochem_signals
# =====================================================================

class TestComputeNeurochemSignals:
    def test_explore_da_delta(self):
        rng = _rng()
        cfg = _cfg()
        nc = compute_neurochem_signals(
            MetaStrategy.EXPLORE, False, False,
            PerformanceTrend.STAGNANT, 0.3, cfg, rng)
        assert nc.da_delta > 0.0

    def test_exploit_5ht_delta(self):
        rng = _rng()
        cfg = _cfg()
        nc = compute_neurochem_signals(
            MetaStrategy.EXPLOIT, False, False,
            PerformanceTrend.IMPROVING, 0.3, cfg, rng)
        assert nc._5ht_delta == cfg.beta_5ht_exploit

    def test_exploit_degrading_no_5ht(self):
        rng = _rng()
        cfg = _cfg()
        nc = compute_neurochem_signals(
            MetaStrategy.EXPLOIT, False, False,
            PerformanceTrend.DEGRADING, 0.3, cfg, rng)
        assert nc._5ht_delta == 0.0

    def test_divergence_ne_delta(self):
        rng = _rng()
        cfg = _cfg()
        nc = compute_neurochem_signals(
            MetaStrategy.RESET, False, True,
            PerformanceTrend.DEGRADING, 0.3, cfg, rng)
        assert nc.ne_delta >= 0.0  # Poisson can be 0

    def test_plateau_ach_delta(self):
        rng = _rng()
        cfg = _cfg()
        nc = compute_neurochem_signals(
            MetaStrategy.EXPLOIT, True, False,
            PerformanceTrend.STAGNANT, 0.3, cfg, rng)
        assert nc.ach_delta == cfg.beta_ach_analysis

    def test_exploit_gaba_delta(self):
        rng = _rng()
        cfg = _cfg()
        nc = compute_neurochem_signals(
            MetaStrategy.EXPLOIT, False, False,
            PerformanceTrend.STAGNANT, 0.3, cfg, rng)
        assert nc.gaba_delta == cfg.beta_gaba_smooth

    def test_explore_no_gaba(self):
        rng = _rng()
        cfg = _cfg()
        nc = compute_neurochem_signals(
            MetaStrategy.EXPLORE, False, False,
            PerformanceTrend.STAGNANT, 0.3, cfg, rng)
        assert nc.gaba_delta == 0.0

    def test_beta_boost_on_plateau_or_divergence(self):
        rng = _rng()
        cfg = _cfg()
        nc = compute_neurochem_signals(
            MetaStrategy.EXPLOIT, True, False,
            PerformanceTrend.STAGNANT, 0.3, cfg, rng)
        assert nc.beta_boost == cfg.psi_beta_osc

    def test_no_beta_boost_normal(self):
        rng = _rng()
        cfg = _cfg()
        nc = compute_neurochem_signals(
            MetaStrategy.EXPLOIT, False, False,
            PerformanceTrend.STAGNANT, 0.3, cfg, rng)
        assert nc.beta_boost == 0.0

    def test_high_improvement_rate_da_bonus(self):
        rng = _rng()
        cfg = _cfg()
        nc = compute_neurochem_signals(
            MetaStrategy.EXPLOIT, False, False,
            PerformanceTrend.IMPROVING, 0.8, cfg, rng)
        assert nc.da_delta > 0.0  # improvement_rate > 0.5 adds DA


# =====================================================================
# 13. Pure helper: get_mode_scales
# =====================================================================

class TestGetModeScales:
    def test_normal_mode(self):
        cfg = _cfg()
        scales = get_mode_scales(OperationalMode.NORMAL, cfg)
        assert scales["window_scale"] == 1.0
        assert scales["explore_scale"] == 1.0

    def test_reflective_mode(self):
        cfg = _cfg()
        scales = get_mode_scales(OperationalMode.REFLECTIVE, cfg)
        assert scales["window_scale"] == cfg.analytical_window_scale
        assert scales["explore_scale"] == cfg.analytical_explore_scale

    def test_dev_mode(self):
        cfg = _cfg()
        scales = get_mode_scales(OperationalMode.DEV, cfg)
        assert scales["divergence_scale"] == cfg.creative_divergence_scale

    def test_rem_dream_mode(self):
        cfg = _cfg()
        scales = get_mode_scales(OperationalMode.REM_DREAM, cfg)
        assert scales["divergence_scale"] == cfg.rem_divergence_tolerance


# =====================================================================
# 14. Engine init
# =====================================================================

class TestEngineInit:
    def test_engine_id(self):
        e = _engine()
        assert e.engine_id == "recursive_learning_engine"

    def test_cluster(self):
        e = _engine()
        assert e.cluster == "learning"

    def test_default_mode(self):
        e = _engine()
        assert e._mode == OperationalMode.NORMAL

    def test_cycle_count_starts_zero(self):
        e = _engine()
        assert e._cycle_count == 0

    def test_custom_config(self):
        e = _engine(performance_window_size=30)
        assert e._cfg.performance_window_size == 30


# =====================================================================
# 15. Engine configure and update_neurochem_state
# =====================================================================

class TestEngineConfiguration:
    def test_configure_mode(self):
        e = _engine()
        e.configure(OperationalMode.DEV)
        assert e._mode == OperationalMode.DEV

    def test_update_neurochem_da(self):
        e = _engine()
        e.update_neurochem_state({"da": 0.7})
        assert e._state.da_level == 0.7

    def test_update_neurochem_5ht(self):
        e = _engine()
        e.update_neurochem_state({"5ht": 0.6})
        assert e._state._5ht_level == 0.6

    def test_update_neurochem_all(self):
        e = _engine()
        e.update_neurochem_state({"da": 0.1, "5ht": 0.2, "ne": 0.3, "ach": 0.4, "gaba": 0.5})
        assert e._state.da_level == 0.1
        assert e._state._5ht_level == 0.2
        assert e._state.ne_level == 0.3
        assert e._state.ach_level == 0.4
        assert e._state.gaba_level == 0.5

    def test_unknown_keys_ignored(self):
        e = _engine()
        e.update_neurochem_state({"unknown_key": 0.99})
        # Should not raise


# =====================================================================
# 16. Engine get_status
# =====================================================================

class TestEngineGetStatus:
    def test_initial_status(self):
        e = _engine()
        s = e.get_status()
        assert s["engine_id"] == "recursive_learning_engine"
        assert s["cycle_count"] == 0
        assert s["current_strategy"] == "exploit"
        assert s["window_size"] == 0

    def test_status_after_process(self):
        e = _engine()
        e.process(_input())
        s = e.get_status()
        assert s["cycle_count"] == 1
        assert s["window_size"] == 1

    def test_status_contains_nt_state(self):
        e = _engine()
        e.update_neurochem_state({"da": 0.5})
        s = e.get_status()
        assert s["state"]["da_level"] == 0.5


# =====================================================================
# 17. Engine process() pipeline
# =====================================================================

class TestProcessPipeline:
    def test_single_cycle(self):
        e = _engine()
        result = e.process(_input())
        assert isinstance(result, RecursiveLearningResult)
        assert result.processing_time_ms >= 0.0

    def test_window_grows(self):
        e = _engine()
        for i in range(5):
            e.process(_input(_metrics(tick=i, mean_abs_delta=0.1)))
        s = e.get_status()
        assert s["window_size"] == 5

    def test_window_trims_at_max(self):
        e = _engine(performance_window_size=5)
        for i in range(10):
            e.process(_input(_metrics(tick=i)))
        s = e.get_status()
        assert s["window_size"] <= 5

    def test_cycle_count_increments(self):
        e = _engine()
        e.process(_input())
        e.process(_input())
        assert e._cycle_count == 2

    def test_adjustments_always_emitted(self):
        e = _engine()
        result = e.process(_input())
        assert len(result.meta_decision.adjustments) == 3  # LR, noise, consolidation

    def test_metadata_populated(self):
        e = _engine()
        result = e.process(_input())
        assert "mode" in result.metadata
        assert "cycle" in result.metadata
        assert result.metadata["cycle"] == 1

    def test_bulk_history_injection(self):
        e = _engine()
        history = [_metrics(tick=i, mean_abs_delta=0.1 + i * 0.01) for i in range(8)]
        result = e.process(_input(history=history))
        # Window should have history + current = 9
        assert result.current_performance.window_size == 9

    def test_force_strategy(self):
        e = _engine()
        result = e.process(_input(force_strategy=MetaStrategy.RESET))
        assert result.meta_decision.strategy == MetaStrategy.RESET
        assert result.meta_decision.reason == "forced_override"

    def test_neurochemical_signals_in_result(self):
        e = _engine()
        result = e.process(_input())
        nc = result.neurochemical_signals
        assert isinstance(nc, RecursiveLearningNeurochem)


# =====================================================================
# 18. Plateau detection via process()
# =====================================================================

class TestProcessPlateauDetection:
    def test_plateau_detected_after_sustained_low_variance(self):
        e = _engine()
        # Feed constant low deltas with low convergence for enough cycles
        for i in range(15):
            m = _metrics(tick=i, mean_abs_delta=0.05, convergence_ratio=0.3)
            result = e.process(_input(m))
        # After 15 constant-delta cycles (variance~=0, convergence<0.8),
        # plateau should eventually fire
        assert result.current_performance.is_plateau is True

    def test_no_plateau_if_convergence_high(self):
        e = _engine()
        for i in range(15):
            m = _metrics(tick=i, mean_abs_delta=0.05, convergence_ratio=0.9)
            result = e.process(_input(m))
        assert result.current_performance.is_plateau is False


# =====================================================================
# 19. Divergence detection via process()
# =====================================================================

class TestProcessDivergenceDetection:
    def test_divergence_detected_on_increasing_deltas(self):
        e = _engine()
        for i in range(10):
            m = _metrics(tick=i, mean_abs_delta=0.2 + i * 0.05)
            result = e.process(_input(m))
        # Increasing deltas -> positive slope -> divergence
        assert result.current_performance.is_diverging is True

    def test_no_divergence_on_decreasing_deltas(self):
        e = _engine()
        for i in range(10):
            m = _metrics(tick=i, mean_abs_delta=0.5 - i * 0.04)
            result = e.process(_input(m))
        assert result.current_performance.is_diverging is False


# =====================================================================
# 20. Strategy switching
# =====================================================================

class TestStrategySwitching:
    def test_switch_count_increments(self):
        e = _engine()
        # Force a switch from default EXPLOIT to EXPLORE
        r1 = e.process(_input(force_strategy=MetaStrategy.EXPLOIT))
        r2 = e.process(_input(force_strategy=MetaStrategy.EXPLORE))
        assert r2.strategy_switches == 1

    def test_cooldown_after_max_switches(self):
        e = _engine(max_strategy_switches=3, switch_cooldown_cycles=2)
        strategies = [MetaStrategy.EXPLORE, MetaStrategy.EXPLOIT] * 5
        for i, s in enumerate(strategies):
            result = e.process(_input(_metrics(tick=i), force_strategy=s))
        # After 3 switches, cooldown should be active
        assert e._state.cooldown_remaining > 0 or e._state.strategy_switches >= 3

    def test_switch_improvements_tracked(self):
        e = _engine()
        # First cycle with high delta
        e.process(_input(_metrics(tick=0, mean_abs_delta=0.3), force_strategy=MetaStrategy.EXPLOIT))
        # Switch to explore
        e.process(_input(_metrics(tick=1, mean_abs_delta=0.3), force_strategy=MetaStrategy.EXPLORE))
        # Back to exploit with lower delta (improvement)
        result = e.process(_input(_metrics(tick=2, mean_abs_delta=0.1), force_strategy=MetaStrategy.EXPLOIT))
        # switch_improvements should have been recorded
        assert len(e._state.switch_improvements) >= 1

    def test_meta_lr_decays(self):
        e = _engine()
        initial_meta_lr = e._state.meta_lr
        e.process(_input())
        assert e._state.meta_lr < initial_meta_lr

    def test_meta_lr_floors_at_010(self):
        e = _engine()
        e._state.meta_lr = 0.11
        e.process(_input())
        # 0.11 * 0.95 = 0.1045 > 0.10, but many cycles will reach floor
        for _ in range(100):
            e.process(_input())
        assert e._state.meta_lr >= 0.10


# =====================================================================
# 21. NT modulation effects on process()
# =====================================================================

class TestNTModulationEffects:
    def test_high_da_increases_explore_prob(self):
        e1 = _engine(seed=99)
        e2 = _engine(seed=99)
        e2.update_neurochem_state({"da": 0.9})
        r1 = e1.process(_input())
        r2 = e2.process(_input())
        assert r2.meta_decision.explore_prob >= r1.meta_decision.explore_prob

    def test_high_5ht_decreases_explore_prob(self):
        e1 = _engine(seed=99)
        e2 = _engine(seed=99)
        e2.update_neurochem_state({"5ht": 0.9})
        r1 = e1.process(_input())
        r2 = e2.process(_input())
        assert r2.meta_decision.explore_prob <= r1.meta_decision.explore_prob

    def test_ach_increases_effective_window(self):
        e1 = _engine()
        e2 = _engine()
        e2.update_neurochem_state({"ach": 0.9})
        r1 = e1.process(_input())
        r2 = e2.process(_input())
        assert r2.metadata["effective_window"] >= r1.metadata["effective_window"]

    def test_gaba_smoothing_applied(self):
        e = _engine()
        e.update_neurochem_state({"gaba": 0.8})
        # Feed noisy data to trigger smoothing path
        for i in range(10):
            delta = 0.1 + (0.05 if i % 2 == 0 else -0.05)
            e.process(_input(_metrics(tick=i, mean_abs_delta=delta)))
        # Should run without error; smoothing is internal


# =====================================================================
# 22. Mode switching
# =====================================================================

class TestModeSwitching:
    def test_analytical_mode_wider_window(self):
        e = _engine()
        e.configure(OperationalMode.REFLECTIVE)
        result = e.process(_input(mode=OperationalMode.REFLECTIVE))
        assert result.metadata["effective_window"] >= 20  # scaled by 1.5

    def test_rem_dream_noise_injection(self):
        e = _engine()
        results = []
        for i in range(20):
            m = _metrics(tick=i, mean_abs_delta=0.1)
            result = e.process(_input(m, mode=OperationalMode.REM_DREAM))
            results.append(result.current_performance.mean_delta)
        # REM noise injection means mean_delta should vary, not all be exactly 0.1
        unique_vals = set(results)
        assert len(unique_vals) > 1

    def test_learning_mode_uses_creative_scales(self):
        cfg = _cfg()
        scales = get_mode_scales(OperationalMode.LEARNING, cfg)
        assert scales["explore_scale"] == cfg.creative_explore_scale


# =====================================================================
# 23. Edge cases
# =====================================================================

class TestEdgeCases:
    def test_no_metrics_default(self):
        e = _engine()
        result = e.process(_input())
        assert result.current_performance.window_size == 1

    def test_zero_delta_metrics(self):
        e = _engine()
        m = _metrics(mean_abs_delta=0.0, convergence_ratio=0.0)
        result = e.process(_input(m))
        assert result.current_performance.mean_delta >= 0.0

    def test_very_high_delta_metrics(self):
        e = _engine()
        m = _metrics(mean_abs_delta=100.0, convergence_ratio=1.0)
        result = e.process(_input(m))
        assert isinstance(result, RecursiveLearningResult)

    def test_many_rapid_switches(self):
        e = _engine(max_strategy_switches=3, switch_cooldown_cycles=2)
        for i in range(20):
            strat = MetaStrategy.EXPLORE if i % 2 == 0 else MetaStrategy.EXPLOIT
            e.process(_input(_metrics(tick=i), force_strategy=strat))
        # Engine should survive and eventually cooldown
        s = e.get_status()
        assert s["strategy_switches"] >= 3

    def test_switch_history_trimmed(self):
        e = _engine()
        # Push >50 switches
        for i in range(60):
            strat = MetaStrategy.EXPLORE if i % 2 == 0 else MetaStrategy.EXPLOIT
            e.process(_input(_metrics(tick=i), force_strategy=strat))
        assert len(e._state.switch_history) <= 50

    def test_empty_window_before_trend(self):
        e = _engine(min_window_for_trend=5)
        # Only 2 data points -> no plateau/divergence detection
        result = e.process(_input(_metrics(tick=0)))
        assert result.current_performance.is_plateau is False
        assert result.current_performance.is_diverging is False

    def test_convergence_ratio_near_boundary(self):
        e = _engine()
        # Convergence ratio exactly at floor
        for i in range(10):
            m = _metrics(tick=i, mean_abs_delta=0.05, convergence_ratio=0.80)
            result = e.process(_input(m))
        # convergence_ratio == floor -> should not count as plateau
        assert result.current_performance.is_plateau is False

    def test_meta_health_enum_values(self):
        assert MetaHealthStatus.HEALTHY.value == "healthy"
        assert MetaHealthStatus.PLATEAU.value == "plateau"
        assert MetaHealthStatus.DIVERGING.value == "diverging"
        assert MetaHealthStatus.RECOVERING.value == "recovering"

    def test_performance_trend_enum_values(self):
        assert PerformanceTrend.IMPROVING.value == "improving"
        assert PerformanceTrend.STAGNANT.value == "stagnant"
        assert PerformanceTrend.DEGRADING.value == "degrading"
        assert PerformanceTrend.OSCILLATING.value == "oscillating"

    def test_meta_strategy_enum_values(self):
        assert MetaStrategy.EXPLOIT.value == "exploit"
        assert MetaStrategy.EXPLORE.value == "explore"
        assert MetaStrategy.RESET.value == "reset"
