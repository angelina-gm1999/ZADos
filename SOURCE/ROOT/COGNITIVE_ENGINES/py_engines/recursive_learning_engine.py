"""
Engine 25 -- Recursive Learning Engine  (``recursive_learning_engine``)
=======================================================================
Meta-learning engine that monitors Engine 17 (Reward-Based Learning)
effectiveness and adjusts meta-parameters.  This is the "learning about
learning" engine -- it closes the outer loop of ZADOS's self-improvement
architecture.

Key mechanics:
  * **Performance Monitoring**: Tracks E17 metrics (mean_abs_delta,
    convergence_ratio) over a sliding window, computing trend slopes
    via simple linear regression.
  * **Plateau Detection**: Detects when learning has stalled --
    mean_abs_delta variance drops below threshold while convergence
    ratio remains below target.
  * **Divergence Detection**: Detects when learning is unstable --
    mean_abs_delta is increasing (positive trend slope) for a sustained
    period.
  * **Strategy Switching**: Three strategies: EXPLOIT (fine-tune current
    parameters), EXPLORE (broaden search with higher LR + noise),
    RESET (return to baseline when divergence is critical).
  * **Meta-Parameter Adjustment**: Emits recommended changes to E17's
    learning rate, consolidation threshold, and noise gate.  These are
    recommendations -- E17 decides whether to apply them.
  * **Meta-Convergence Tracking**: Tracks how often strategy switches
    occur and whether they improve performance, enabling second-order
    learning about which strategies work.

Lifecycle per cycle:
  Receive E17 metrics -> compute meta-performance indicators ->
  detect plateau/divergence -> decide exploit vs explore vs reset ->
  emit meta-parameter recommendations -> track meta-convergence

Neurochemical coupling:
  DA   -- drives strategy exploration (explore mode more likely at high DA)
  5-HT -- preserves working strategies (exploit mode at high 5-HT)
  NE   -- triggers urgent strategy switches (amplifies divergence signal)
  ACh  -- deepens performance analysis (increases window effective size)
  GABA -- suppresses noise in meta-metrics (smooths trend estimation)

Modes:
  DEFAULT    -- standard meta-learning with balanced explore/exploit
  ANALYTICAL -- deeper analysis, more conservative strategy switching
  CREATIVE   -- more aggressive exploration, lower exploit threshold
  REM_DREAM  -- relaxed constraints, high explore probability, noise injection
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from zados.cognitive_engines.constants import _clamp
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Enums
# =====================================================================


class MetaStrategy(str, Enum):
    """Meta-learning strategy for adjusting E17 parameters."""
    EXPLOIT = "exploit"   # Fine-tune: small LR, tight noise gate
    EXPLORE = "explore"   # Broaden: larger LR, wider noise gate
    RESET   = "reset"     # Return to baseline: divergence recovery


class PerformanceTrend(str, Enum):
    """Trend classification for meta-performance."""
    IMPROVING   = "improving"    # Negative slope (deltas shrinking)
    STAGNANT    = "stagnant"     # Near-zero slope
    DEGRADING   = "degrading"    # Positive slope (deltas growing)
    OSCILLATING = "oscillating"  # Alternating direction


class MetaHealthStatus(str, Enum):
    """Overall meta-learning health."""
    HEALTHY    = "healthy"      # Learning is converging
    PLATEAU    = "plateau"      # Learning stalled
    DIVERGING  = "diverging"    # Learning unstable
    RECOVERING = "recovering"   # Post-reset, stabilizing


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class RecursiveLearningConfig:
    """Immutable tuning knobs for the Recursive Learning Engine."""

    # --- Performance window ---
    performance_window_size: int    = 20     # Sliding window for trend detection
    min_window_for_trend:    int    = 5      # Minimum samples before trend analysis

    # --- Plateau detection ---
    plateau_variance_threshold: float = 0.001  # Variance below this = plateau
    plateau_convergence_floor:  float = 0.80   # Convergence ratio below this + low variance = plateau
    plateau_min_cycles:         int   = 5      # Sustained low-variance cycles before plateau flag

    # --- Divergence detection ---
    divergence_slope_threshold: float = 0.005  # Positive slope above this = diverging
    divergence_delta_threshold: float = 0.30   # mean_abs_delta above this = concerning
    divergence_min_cycles:      int   = 3      # Sustained positive slope cycles

    # --- Strategy parameters ---
    explore_probability_base: float = 0.25   # Base P(explore) when no strong signal
    exploit_lr_multiplier:    float = 0.5    # LR *= this during exploit
    explore_lr_multiplier:    float = 2.0    # LR *= this during explore
    reset_lr_value:           float = 0.01   # Absolute LR on reset
    exploit_noise_gate:       float = 0.02   # Tight noise gate for exploit
    explore_noise_gate:       float = 0.10   # Wide noise gate for explore
    reset_noise_gate:         float = 0.05   # Moderate noise gate on reset
    exploit_consolidation_adj: float = -0.05 # Lower consolidation threshold (tighter)
    explore_consolidation_adj: float = 0.10  # Raise consolidation threshold (looser)

    # --- Meta-convergence ---
    max_strategy_switches:     int   = 10    # Max switches before forced cooldown
    switch_cooldown_cycles:    int   = 5     # Cycles to wait after max switches
    meta_lr_decay:             float = 0.95  # Decay factor per meta-cycle
    meta_improvement_threshold: float = 0.02 # Min improvement to count as success

    # --- Neurochemical coupling weights ---
    w_da_explore:        float = 0.15   # DA influence on explore probability
    w_5ht_exploit:       float = 0.12   # 5-HT influence on exploit probability
    w_ne_urgency:        float = 0.10   # NE influence on divergence sensitivity
    w_ach_depth:         float = 0.08   # ACh influence on window effective size
    w_gaba_smooth:       float = 0.10   # GABA influence on trend smoothing

    # --- Neurochemical output ---
    beta_da_explore:     float = 0.10   # DA delta on explore decision
    beta_da_improvement: float = 0.12   # DA delta on meta-improvement
    beta_5ht_exploit:    float = 0.08   # 5-HT delta on exploit decision
    beta_ne_divergence:  float = 0.10   # NE delta on divergence detection
    beta_ach_analysis:   float = 0.06   # ACh delta on deep analysis
    beta_gaba_smooth:    float = 0.05   # GABA delta on smoothing
    psi_beta_osc:        float = 0.06   # Beta oscillation boost during analysis

    # --- Mode-specific adjustments ---
    # ANALYTICAL mode
    analytical_window_scale:   float = 1.5   # Wider effective window
    analytical_explore_scale:  float = 0.6   # Less exploration
    analytical_plateau_scale:  float = 0.7   # More sensitive plateau detection
    # CREATIVE mode
    creative_explore_scale:    float = 1.8   # More exploration
    creative_divergence_scale: float = 1.5   # Higher divergence tolerance
    creative_lr_scale:         float = 1.3   # Larger learning rate adjustments
    # REM_DREAM mode
    rem_explore_probability:   float = 0.70  # Very high explore probability
    rem_noise_injection:       float = 0.15  # Additional noise in meta-metrics
    rem_divergence_tolerance:  float = 2.0   # Much higher divergence tolerance

    # --- Stochastic distribution params ---
    gamma_alpha:  float = 2.0    # Gamma shape for DA/ACh
    gamma_theta:  float = 0.30   # Gamma scale
    poisson_lam:  float = 1.5    # Poisson lambda for NE


# =====================================================================
# Data types -- frozen outputs
# =====================================================================


@dataclass(frozen=True)
class MetaMetrics:
    """Snapshot of E17 performance at a given tick."""
    tick:               int   = 0
    mean_abs_delta:     float = 0.0   # Mean absolute parameter change
    convergence_ratio:  float = 0.0   # Fraction of params converged [0, 1]
    learning_rate:      float = 0.01  # Current E17 learning rate
    consolidation_threshold: float = 0.50  # Current E17 consolidation threshold
    noise_gate:         float = 0.05  # Current E17 noise gate


@dataclass(frozen=True)
class MetaPerformanceSnapshot:
    """Computed meta-performance indicators for the current window."""
    mean_delta:         float = 0.0   # Mean of mean_abs_delta over window
    delta_variance:     float = 0.0   # Variance of mean_abs_delta over window
    convergence_ratio:  float = 0.0   # Latest convergence ratio
    trend_slope:        float = 0.0   # Linear regression slope of deltas
    trend_r_squared:    float = 0.0   # R-squared of linear fit
    is_plateau:         bool  = False # Plateau detected
    is_diverging:       bool  = False # Divergence detected
    window_size:        int   = 0     # Actual window size used
    trend:              PerformanceTrend = PerformanceTrend.STAGNANT


@dataclass(frozen=True)
class MetaAdjustment:
    """Recommended change to a single E17 parameter."""
    parameter:     str   = ""     # E17 parameter name
    current_value: float = 0.0
    recommended:   float = 0.0
    delta:         float = 0.0    # recommended - current
    reason:        str   = ""


@dataclass(frozen=True)
class MetaDecision:
    """Strategy decision from one meta-learning cycle."""
    strategy:      MetaStrategy          = MetaStrategy.EXPLOIT
    confidence:    float                 = 0.0   # [0, 1] confidence in strategy choice
    reason:        str                   = ""
    adjustments:   List[MetaAdjustment]  = field(default_factory=list)
    explore_prob:  float                 = 0.0   # Computed explore probability


@dataclass(frozen=True)
class RecursiveLearningNeurochem:
    """
    Neurochemical coupling signals from one Recursive Learning cycle.

    Notation (Appendix S2-S3, S7-S9):
        da_delta     -> Delta C_DA(t)    : exploration reward / improvement reward
        _5ht_delta   -> Delta C_5HT(t)   : stability signal during exploit
        ne_delta     -> Delta C_NE(t)    : urgency on divergence detection
        ach_delta    -> Delta C_ACh(t)   : attentional deepening for analysis
        gaba_delta   -> Delta C_GABA(t)  : noise suppression during smoothing
        beta_boost   -> Delta phi_beta(t): analytical band enhancement (S7)
    """
    da_delta:    float = 0.0
    _5ht_delta:  float = 0.0
    ne_delta:    float = 0.0
    ach_delta:   float = 0.0
    gaba_delta:  float = 0.0
    beta_boost:  float = 0.0


@dataclass(frozen=True)
class RecursiveLearningInput:
    """Input bundle for one Recursive Learning Engine cycle."""
    e17_metrics:       MetaMetrics               = field(default_factory=MetaMetrics)
    e17_history:       Optional[List[MetaMetrics]] = None  # Optional bulk history injection
    active_mode:       OperationalMode            = OperationalMode.NORMAL
    force_strategy:    Optional[MetaStrategy]     = None   # Override for testing


@dataclass(frozen=True)
class RecursiveLearningResult:
    """Full output of one Recursive Learning Engine cycle."""
    current_performance:    MetaPerformanceSnapshot    = field(default_factory=MetaPerformanceSnapshot)
    meta_decision:          MetaDecision               = field(default_factory=MetaDecision)
    strategy_switches:      int                        = 0     # Total switches this session
    total_adjustments:      int                        = 0     # Total adjustments emitted
    meta_health:            MetaHealthStatus            = MetaHealthStatus.HEALTHY
    meta_improvement_rate:  float                      = 0.0   # Fraction of switches that improved
    current_strategy:       MetaStrategy               = MetaStrategy.EXPLOIT
    neurochemical_signals:  RecursiveLearningNeurochem  = field(default_factory=RecursiveLearningNeurochem)
    processing_time_ms:     float                      = 0.0
    metadata:               Dict[str, Any]             = field(default_factory=dict)


# =====================================================================
# Mutable engine state
# =====================================================================


@dataclass
class RecursiveLearningState:
    """Running state for the Recursive Learning Engine."""
    # Performance tracking
    performance_window:  List[float] = field(default_factory=list)  # mean_abs_delta history
    convergence_window:  List[float] = field(default_factory=list)  # convergence_ratio history
    lr_window:           List[float] = field(default_factory=list)  # learning_rate history

    # Strategy tracking
    current_strategy:    MetaStrategy = MetaStrategy.EXPLOIT
    strategy_switches:   int          = 0
    total_adjustments:   int          = 0
    switch_history:      List[str]    = field(default_factory=list)  # strategy names
    switch_improvements: List[bool]   = field(default_factory=list)  # did each switch help?
    cooldown_remaining:  int          = 0  # Cycles remaining in cooldown

    # Plateau / divergence counters
    plateau_counter:     int   = 0
    divergence_counter:  int   = 0
    pre_switch_delta:    float = 0.0  # mean_abs_delta before last switch

    # Meta-convergence
    meta_lr:             float = 1.0   # Meta learning rate (decays over time)

    # Neurochemical bidirectional
    da_level:    float = 0.0
    _5ht_level:  float = 0.0
    ne_level:    float = 0.0
    ach_level:   float = 0.0
    gaba_level:  float = 0.0


# =====================================================================
# Pure helper functions
# =====================================================================


def compute_linear_regression(values: List[float]) -> Tuple[float, float]:
    """
    Simple linear regression: y = slope * x + intercept.
    Returns (slope, r_squared).

    x-values are 0, 1, 2, ..., n-1.
    """
    n = len(values)
    if n < 2:
        return (0.0, 0.0)

    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    ss_xy = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    ss_xx = sum((i - x_mean) ** 2 for i in range(n))
    ss_yy = sum((v - y_mean) ** 2 for v in values)

    if ss_xx < 1e-12:
        return (0.0, 0.0)

    slope = ss_xy / ss_xx
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy) if ss_yy > 1e-12 else 0.0
    r_squared = _clamp(r_squared)

    return (slope, r_squared)


def compute_variance(values: List[float]) -> float:
    """Compute sample variance of a list of floats."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def classify_trend(slope: float, r_squared: float, values: List[float]) -> PerformanceTrend:
    """
    Classify performance trend based on regression slope and fit quality.

    slope < -threshold & good fit -> IMPROVING
    slope > +threshold & good fit -> DEGRADING
    alternating signs in recent diffs -> OSCILLATING
    else -> STAGNANT
    """
    slope_threshold = 0.002
    r_threshold = 0.25

    if r_squared >= r_threshold:
        if slope < -slope_threshold:
            return PerformanceTrend.IMPROVING
        if slope > slope_threshold:
            return PerformanceTrend.DEGRADING

    # Check for oscillation
    if len(values) >= 4:
        diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
        sign_changes = sum(
            1 for i in range(1, len(diffs))
            if (diffs[i] > 0) != (diffs[i - 1] > 0) and abs(diffs[i]) > 1e-6
        )
        if sign_changes >= len(diffs) * 0.5:
            return PerformanceTrend.OSCILLATING

    return PerformanceTrend.STAGNANT


def detect_plateau(
    variance: float,
    convergence_ratio: float,
    plateau_counter: int,
    cfg: RecursiveLearningConfig,
    mode_scale: float = 1.0,
) -> Tuple[bool, int]:
    """
    Detect learning plateau.

    Plateau = low variance in deltas (not improving or degrading)
    AND convergence ratio has not reached target.

    Returns (is_plateau, updated_counter).
    """
    threshold = cfg.plateau_variance_threshold * mode_scale
    if variance < threshold and convergence_ratio < cfg.plateau_convergence_floor:
        new_counter = plateau_counter + 1
        return (new_counter >= cfg.plateau_min_cycles, new_counter)
    return (False, 0)


def detect_divergence(
    slope: float,
    mean_delta: float,
    divergence_counter: int,
    cfg: RecursiveLearningConfig,
    mode_scale: float = 1.0,
) -> Tuple[bool, int]:
    """
    Detect learning divergence.

    Divergence = positive slope (deltas increasing) sustained over
    divergence_min_cycles AND mean_delta above threshold.

    Returns (is_diverging, updated_counter).
    """
    slope_thresh = cfg.divergence_slope_threshold * mode_scale
    if slope > slope_thresh and mean_delta > cfg.divergence_delta_threshold:
        new_counter = divergence_counter + 1
        return (new_counter >= cfg.divergence_min_cycles, new_counter)
    # Positive slope but delta not yet concerning
    if slope > slope_thresh:
        new_counter = divergence_counter + 1
        return (False, new_counter)
    return (False, max(0, divergence_counter - 1))


def compute_explore_probability(
    is_plateau: bool,
    is_diverging: bool,
    current_strategy: MetaStrategy,
    da_level: float,
    _5ht_level: float,
    meta_lr: float,
    cfg: RecursiveLearningConfig,
    mode: OperationalMode,
) -> float:
    """
    Compute probability of choosing EXPLORE over EXPLOIT.

    Base probability modulated by:
      - Plateau detection (increases explore)
      - DA level (increases explore)
      - 5-HT level (decreases explore / increases exploit)
      - Mode (CREATIVE increases, ANALYTICAL decreases)
      - Meta-LR decay (gradual convergence to exploit)
    """
    p = cfg.explore_probability_base

    # Plateau increases exploration
    if is_plateau:
        p += 0.30

    # Divergence favors reset, not explore
    if is_diverging:
        p -= 0.15

    # DA modulation: high DA -> more exploration
    p += cfg.w_da_explore * (da_level - 0.4)

    # 5-HT modulation: high 5-HT -> more exploitation (less explore)
    p -= cfg.w_5ht_exploit * (_5ht_level - 0.4)

    # Mode adjustments
    if mode in (OperationalMode.DEV, OperationalMode.LEARNING):
        p *= cfg.creative_explore_scale
    elif mode == OperationalMode.REM_DREAM:
        p = cfg.rem_explore_probability
    elif mode == OperationalMode.REFLECTIVE:
        p *= cfg.analytical_explore_scale

    # Meta-LR decay: over time, converge toward exploit
    p *= meta_lr

    return _clamp(p)


def decide_strategy(
    is_plateau: bool,
    is_diverging: bool,
    explore_prob: float,
    current_strategy: MetaStrategy,
    cooldown_remaining: int,
    rng: np.random.Generator,
    force_strategy: Optional[MetaStrategy] = None,
) -> Tuple[MetaStrategy, float, str]:
    """
    Decide meta-strategy.  Returns (strategy, confidence, reason).

    Priority:
      1. Forced strategy override (testing)
      2. Divergence -> RESET (if critical)
      3. Plateau -> sample EXPLORE with explore_prob
      4. Default -> EXPLOIT
    """
    if force_strategy is not None:
        return (force_strategy, 1.0, "forced_override")

    if cooldown_remaining > 0:
        return (current_strategy, 0.5, "cooldown_active")

    # Critical divergence -> RESET
    if is_diverging:
        return (MetaStrategy.RESET, 0.85, "divergence_detected")

    # Plateau -> probabilistic explore
    if is_plateau:
        if float(rng.uniform()) < explore_prob:
            return (MetaStrategy.EXPLORE, round(explore_prob, 4), "plateau_explore")
        return (MetaStrategy.EXPLOIT, round(1.0 - explore_prob, 4), "plateau_exploit")

    # No issues -> probabilistic but biased toward exploit
    if float(rng.uniform()) < explore_prob:
        return (MetaStrategy.EXPLORE, round(explore_prob, 4), "routine_explore")
    return (MetaStrategy.EXPLOIT, round(1.0 - explore_prob, 4), "routine_exploit")


def build_adjustments(
    strategy: MetaStrategy,
    current_lr: float,
    current_consolidation: float,
    current_noise_gate: float,
    cfg: RecursiveLearningConfig,
    meta_lr: float,
    mode: OperationalMode,
) -> List[MetaAdjustment]:
    """
    Build meta-parameter adjustments based on chosen strategy.

    Returns list of MetaAdjustment recommendations for E17.
    """
    adjustments: List[MetaAdjustment] = []
    lr_scale = cfg.creative_lr_scale if mode in (OperationalMode.DEV, OperationalMode.LEARNING) else 1.0

    if strategy == MetaStrategy.EXPLOIT:
        # Fine-tune: reduce LR, tighten noise gate
        new_lr = current_lr * cfg.exploit_lr_multiplier * lr_scale
        new_lr = max(0.001, new_lr)  # Floor
        new_noise = cfg.exploit_noise_gate
        new_consol = _clamp(current_consolidation + cfg.exploit_consolidation_adj)

        adjustments.append(MetaAdjustment(
            parameter="learning_rate",
            current_value=round(current_lr, 6),
            recommended=round(new_lr, 6),
            delta=round(new_lr - current_lr, 6),
            reason="exploit: reduce LR for fine-tuning",
        ))
        adjustments.append(MetaAdjustment(
            parameter="noise_gate",
            current_value=round(current_noise_gate, 6),
            recommended=round(new_noise, 6),
            delta=round(new_noise - current_noise_gate, 6),
            reason="exploit: tighten noise gate",
        ))
        adjustments.append(MetaAdjustment(
            parameter="consolidation_threshold",
            current_value=round(current_consolidation, 6),
            recommended=round(new_consol, 6),
            delta=round(new_consol - current_consolidation, 6),
            reason="exploit: lower consolidation threshold",
        ))

    elif strategy == MetaStrategy.EXPLORE:
        # Broaden: increase LR, widen noise gate
        new_lr = current_lr * cfg.explore_lr_multiplier * lr_scale
        new_lr = min(0.50, new_lr)  # Ceiling
        new_noise = cfg.explore_noise_gate
        new_consol = _clamp(current_consolidation + cfg.explore_consolidation_adj)

        if mode == OperationalMode.REM_DREAM:
            new_noise += cfg.rem_noise_injection

        adjustments.append(MetaAdjustment(
            parameter="learning_rate",
            current_value=round(current_lr, 6),
            recommended=round(new_lr, 6),
            delta=round(new_lr - current_lr, 6),
            reason="explore: increase LR for broader search",
        ))
        adjustments.append(MetaAdjustment(
            parameter="noise_gate",
            current_value=round(current_noise_gate, 6),
            recommended=round(new_noise, 6),
            delta=round(new_noise - current_noise_gate, 6),
            reason="explore: widen noise gate for diversity",
        ))
        adjustments.append(MetaAdjustment(
            parameter="consolidation_threshold",
            current_value=round(current_consolidation, 6),
            recommended=round(new_consol, 6),
            delta=round(new_consol - current_consolidation, 6),
            reason="explore: raise consolidation threshold",
        ))

    elif strategy == MetaStrategy.RESET:
        # Return to baseline
        new_lr = cfg.reset_lr_value
        new_noise = cfg.reset_noise_gate
        new_consol = 0.50  # Baseline consolidation

        adjustments.append(MetaAdjustment(
            parameter="learning_rate",
            current_value=round(current_lr, 6),
            recommended=round(new_lr, 6),
            delta=round(new_lr - current_lr, 6),
            reason="reset: return LR to baseline (divergence recovery)",
        ))
        adjustments.append(MetaAdjustment(
            parameter="noise_gate",
            current_value=round(current_noise_gate, 6),
            recommended=round(new_noise, 6),
            delta=round(new_noise - current_noise_gate, 6),
            reason="reset: moderate noise gate for stability",
        ))
        adjustments.append(MetaAdjustment(
            parameter="consolidation_threshold",
            current_value=round(current_consolidation, 6),
            recommended=round(new_consol, 6),
            delta=round(new_consol - current_consolidation, 6),
            reason="reset: baseline consolidation threshold",
        ))

    return adjustments


def compute_meta_health(
    is_plateau: bool,
    is_diverging: bool,
    current_strategy: MetaStrategy,
    trend: PerformanceTrend,
) -> MetaHealthStatus:
    """Classify overall meta-learning health."""
    if is_diverging:
        return MetaHealthStatus.DIVERGING
    if current_strategy == MetaStrategy.RESET:
        return MetaHealthStatus.RECOVERING
    if is_plateau:
        return MetaHealthStatus.PLATEAU
    if trend in (PerformanceTrend.IMPROVING, PerformanceTrend.STAGNANT):
        return MetaHealthStatus.HEALTHY
    return MetaHealthStatus.HEALTHY


def compute_meta_improvement_rate(switch_improvements: List[bool]) -> float:
    """Fraction of strategy switches that resulted in improvement."""
    if not switch_improvements:
        return 0.0
    return sum(1 for s in switch_improvements if s) / len(switch_improvements)


def compute_neurochem_signals(
    strategy: MetaStrategy,
    is_plateau: bool,
    is_diverging: bool,
    trend: PerformanceTrend,
    meta_improvement_rate: float,
    cfg: RecursiveLearningConfig,
    rng: np.random.Generator,
) -> RecursiveLearningNeurochem:
    """
    Neurochemical coupling from recursive learning output.

    DA   -- exploration reward or meta-improvement reward
    5-HT -- stability reinforcement during exploit
    NE   -- urgency burst on divergence (Poisson)
    ACh  -- attentional load during analysis
    GABA -- noise suppression during smoothing
    Beta -- oscillatory boost during analysis
    """
    # DA: exploration reward + meta-improvement
    da_delta = 0.0
    if strategy == MetaStrategy.EXPLORE:
        da_noise = float(rng.gamma(cfg.gamma_alpha, cfg.gamma_theta))
        da_delta = cfg.beta_da_explore * da_noise
    if meta_improvement_rate > 0.5:
        da_delta += cfg.beta_da_improvement * meta_improvement_rate

    # 5-HT: stability during exploit
    _5ht_delta = 0.0
    if strategy == MetaStrategy.EXPLOIT and trend != PerformanceTrend.DEGRADING:
        _5ht_delta = cfg.beta_5ht_exploit

    # NE: urgency on divergence
    ne_delta = 0.0
    if is_diverging:
        ne_impulse = float(rng.poisson(cfg.poisson_lam))
        ne_delta = cfg.beta_ne_divergence * ne_impulse

    # ACh: analysis depth
    ach_delta = cfg.beta_ach_analysis if (is_plateau or is_diverging) else 0.0

    # GABA: smoothing noise
    gaba_delta = cfg.beta_gaba_smooth if strategy == MetaStrategy.EXPLOIT else 0.0

    # Beta oscillation boost during active analysis
    beta_boost = cfg.psi_beta_osc if (is_plateau or is_diverging) else 0.0

    return RecursiveLearningNeurochem(
        da_delta=da_delta,
        _5ht_delta=_5ht_delta,
        ne_delta=ne_delta,
        ach_delta=ach_delta,
        gaba_delta=gaba_delta,
        beta_boost=beta_boost,
    )


def get_mode_scales(mode: OperationalMode, cfg: RecursiveLearningConfig) -> Dict[str, float]:
    """
    Return mode-dependent scale factors.

    Keys: window_scale, explore_scale, plateau_scale, divergence_scale
    """
    if mode == OperationalMode.REFLECTIVE:
        return {
            "window_scale": cfg.analytical_window_scale,
            "explore_scale": cfg.analytical_explore_scale,
            "plateau_scale": cfg.analytical_plateau_scale,
            "divergence_scale": 1.0,
        }
    if mode in (OperationalMode.DEV, OperationalMode.LEARNING):
        return {
            "window_scale": 1.0,
            "explore_scale": cfg.creative_explore_scale,
            "plateau_scale": 1.0,
            "divergence_scale": cfg.creative_divergence_scale,
        }
    if mode == OperationalMode.REM_DREAM:
        return {
            "window_scale": 1.0,
            "explore_scale": 1.0,  # Handled separately in explore_prob
            "plateau_scale": 1.5,
            "divergence_scale": cfg.rem_divergence_tolerance,
        }
    # NORMAL / REM_NORMAL
    return {
        "window_scale": 1.0,
        "explore_scale": 1.0,
        "plateau_scale": 1.0,
        "divergence_scale": 1.0,
    }


# =====================================================================
# Engine class
# =====================================================================


class RecursiveLearningEngine:
    """
    Engine 25 -- Recursive Learning Engine.

    Meta-learning engine that monitors E17 (Reward-Based Learning)
    effectiveness and adjusts meta-parameters.  Closes the outer loop
    of ZADOS's self-improvement architecture by learning about learning.

    API
    ---
    configure(mode)            -- set operational mode
    update_neurochem_state(d)  -- inject external NT levels (Pattern A)
    process(rl_input)          -- run meta-learning cycle
    get_status()               -- introspection
    """

    engine_id = "recursive_learning_engine"
    cluster   = "learning"

    def __init__(
        self,
        config: Optional[RecursiveLearningConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._cfg = config or RecursiveLearningConfig()
        self._rng = rng or np.random.default_rng()
        self._mode = OperationalMode.NORMAL
        self._state = RecursiveLearningState()
        self._cycle_count = 0

    # ----- Configuration --------------------------------------------------

    def configure(self, mode: OperationalMode) -> None:
        """Set operational mode."""
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """Inject current neurochemical levels for bidirectional feedback."""
        if "da" in state_dict:
            self._state.da_level = state_dict["da"]
        if "5ht" in state_dict:
            self._state._5ht_level = state_dict["5ht"]
        if "ne" in state_dict:
            self._state.ne_level = state_dict["ne"]
        if "ach" in state_dict:
            self._state.ach_level = state_dict["ach"]
        if "gaba" in state_dict:
            self._state.gaba_level = state_dict["gaba"]

    # ----- Main pipeline --------------------------------------------------

    def process(self, rl_input: RecursiveLearningInput) -> RecursiveLearningResult:
        """
        Run one meta-learning cycle.

        Pipeline stages:
          1. Ingest E17 metrics into sliding window
          2. Compute meta-performance indicators (trend, variance)
          3. Detect plateau / divergence
          4. Decide strategy (exploit / explore / reset)
          5. Build meta-parameter adjustments
          6. Track meta-convergence
          7. Compute neurochemical coupling
        """
        t0 = time.perf_counter()
        self._cycle_count += 1

        mode = rl_input.active_mode
        mode_scales = get_mode_scales(mode, self._cfg)

        # --- Stage 1: Ingest metrics into window ---

        # If bulk history provided, seed the window
        if rl_input.e17_history:
            for m in rl_input.e17_history:
                self._state.performance_window.append(m.mean_abs_delta)
                self._state.convergence_window.append(m.convergence_ratio)
                self._state.lr_window.append(m.learning_rate)

        # Append current metrics
        metrics = rl_input.e17_metrics
        self._state.performance_window.append(metrics.mean_abs_delta)
        self._state.convergence_window.append(metrics.convergence_ratio)
        self._state.lr_window.append(metrics.learning_rate)

        # Trim to window size (adjusted by mode + ACh)
        effective_window = int(
            self._cfg.performance_window_size
            * mode_scales["window_scale"]
            * (1.0 + self._cfg.w_ach_depth * self._state.ach_level)
        )
        effective_window = max(self._cfg.min_window_for_trend, effective_window)

        if len(self._state.performance_window) > effective_window:
            self._state.performance_window = self._state.performance_window[-effective_window:]
        if len(self._state.convergence_window) > effective_window:
            self._state.convergence_window = self._state.convergence_window[-effective_window:]
        if len(self._state.lr_window) > effective_window:
            self._state.lr_window = self._state.lr_window[-effective_window:]

        # --- Stage 2: Compute meta-performance indicators ---

        window = self._state.performance_window
        mean_delta = sum(window) / len(window) if window else 0.0
        delta_variance = compute_variance(window)

        # GABA smoothing: high GABA -> smooth the window before regression
        smoothed_window = list(window)
        if self._state.gaba_level > 0.3 and len(smoothed_window) >= 3:
            alpha = self._cfg.w_gaba_smooth * self._state.gaba_level
            for i in range(1, len(smoothed_window)):
                smoothed_window[i] = (1.0 - alpha) * smoothed_window[i] + alpha * smoothed_window[i - 1]

        slope, r_squared = compute_linear_regression(smoothed_window)
        trend = classify_trend(slope, r_squared, smoothed_window)

        latest_convergence = self._state.convergence_window[-1] if self._state.convergence_window else 0.0

        # --- Stage 3: Detect plateau / divergence ---

        # NE modulation: high NE -> more sensitive divergence detection
        ne_adj = 1.0 - self._cfg.w_ne_urgency * self._state.ne_level
        divergence_scale = mode_scales["divergence_scale"] * max(0.5, ne_adj)

        is_plateau = False
        is_diverging = False

        if len(window) >= self._cfg.min_window_for_trend:
            is_plateau, self._state.plateau_counter = detect_plateau(
                delta_variance,
                latest_convergence,
                self._state.plateau_counter,
                self._cfg,
                mode_scales["plateau_scale"],
            )

            is_diverging, self._state.divergence_counter = detect_divergence(
                slope,
                mean_delta,
                self._state.divergence_counter,
                self._cfg,
                divergence_scale,
            )

        # --- REM_DREAM noise injection ---
        if mode == OperationalMode.REM_DREAM:
            noise = float(self._rng.normal(0, self._cfg.rem_noise_injection))
            mean_delta = max(0.0, mean_delta + noise)

        # Build performance snapshot
        performance = MetaPerformanceSnapshot(
            mean_delta=round(mean_delta, 6),
            delta_variance=round(delta_variance, 6),
            convergence_ratio=round(latest_convergence, 4),
            trend_slope=round(slope, 6),
            trend_r_squared=round(r_squared, 4),
            is_plateau=is_plateau,
            is_diverging=is_diverging,
            window_size=len(window),
            trend=trend,
        )

        # --- Stage 4: Decide strategy ---

        # Cooldown management
        if self._state.cooldown_remaining > 0:
            self._state.cooldown_remaining -= 1

        explore_prob = compute_explore_probability(
            is_plateau,
            is_diverging,
            self._state.current_strategy,
            self._state.da_level,
            self._state._5ht_level,
            self._state.meta_lr,
            self._cfg,
            mode,
        )

        new_strategy, confidence, reason = decide_strategy(
            is_plateau,
            is_diverging,
            explore_prob,
            self._state.current_strategy,
            self._state.cooldown_remaining,
            self._rng,
            rl_input.force_strategy,
        )

        # Track strategy switches
        switched = new_strategy != self._state.current_strategy
        if switched:
            # Check if previous switch improved things
            if self._state.pre_switch_delta > 0:
                improved = mean_delta < self._state.pre_switch_delta - self._cfg.meta_improvement_threshold
                self._state.switch_improvements.append(improved)

            self._state.pre_switch_delta = mean_delta
            self._state.strategy_switches += 1
            self._state.switch_history.append(new_strategy.value)

            # Max switch check
            if self._state.strategy_switches >= self._cfg.max_strategy_switches:
                self._state.cooldown_remaining = self._cfg.switch_cooldown_cycles

        self._state.current_strategy = new_strategy

        # Meta-LR decay
        self._state.meta_lr *= self._cfg.meta_lr_decay
        self._state.meta_lr = max(0.10, self._state.meta_lr)

        # --- Stage 5: Build meta-parameter adjustments ---

        adjustments = build_adjustments(
            new_strategy,
            metrics.learning_rate,
            metrics.consolidation_threshold,
            metrics.noise_gate,
            self._cfg,
            self._state.meta_lr,
            mode,
        )
        self._state.total_adjustments += len(adjustments)

        meta_decision = MetaDecision(
            strategy=new_strategy,
            confidence=confidence,
            reason=reason,
            adjustments=adjustments,
            explore_prob=round(explore_prob, 4),
        )

        # --- Stage 6: Meta-convergence tracking ---

        meta_health = compute_meta_health(
            is_plateau, is_diverging, new_strategy, trend,
        )
        meta_improvement_rate = compute_meta_improvement_rate(
            self._state.switch_improvements,
        )

        # Trim switch history to prevent unbounded growth
        if len(self._state.switch_history) > 50:
            self._state.switch_history = self._state.switch_history[-50:]
        if len(self._state.switch_improvements) > 50:
            self._state.switch_improvements = self._state.switch_improvements[-50:]

        # --- Stage 7: Neurochemical coupling ---

        neurochem = compute_neurochem_signals(
            new_strategy,
            is_plateau,
            is_diverging,
            trend,
            meta_improvement_rate,
            self._cfg,
            self._rng,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        return RecursiveLearningResult(
            current_performance=performance,
            meta_decision=meta_decision,
            strategy_switches=self._state.strategy_switches,
            total_adjustments=self._state.total_adjustments,
            meta_health=meta_health,
            meta_improvement_rate=round(meta_improvement_rate, 4),
            current_strategy=new_strategy,
            neurochemical_signals=neurochem,
            processing_time_ms=round(elapsed, 3),
            metadata={
                "mode": mode.value,
                "cycle": self._cycle_count,
                "effective_window": effective_window,
                "meta_lr": round(self._state.meta_lr, 4),
                "cooldown_remaining": self._state.cooldown_remaining,
                "plateau_counter": self._state.plateau_counter,
                "divergence_counter": self._state.divergence_counter,
                "switched": switched,
            },
        )

    # ----- Introspection --------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return engine introspection data."""
        return {
            "engine_id": self.engine_id,
            "mode": self._mode.value,
            "cycle_count": self._cycle_count,
            "current_strategy": self._state.current_strategy.value,
            "strategy_switches": self._state.strategy_switches,
            "total_adjustments": self._state.total_adjustments,
            "meta_lr": round(self._state.meta_lr, 4),
            "cooldown_remaining": self._state.cooldown_remaining,
            "plateau_counter": self._state.plateau_counter,
            "divergence_counter": self._state.divergence_counter,
            "window_size": len(self._state.performance_window),
            "state": {
                "da_level": self._state.da_level,
                "_5ht_level": self._state._5ht_level,
                "ne_level": self._state.ne_level,
                "ach_level": self._state.ach_level,
                "gaba_level": self._state.gaba_level,
            },
        }
