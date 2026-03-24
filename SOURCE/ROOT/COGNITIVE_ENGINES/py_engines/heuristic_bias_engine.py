"""
Engine 24 -- Heuristic Bias Engine  (``heuristic_bias_engine``)
===============================================================
Metacognitive auditor -- monitors the system's OWN reasoning *processes*
for systematic shortcuts and distortions (heuristic biases).

Unlike Engine 5 (Bias Detection) which scans CONTENT for bias,
this engine watches the system THINK in real-time and flags when
heuristic shortcuts produce distorted results.

Key features:
  * 22 heuristic bias types across 4 categories:
    REASONING (6), MEMORY (5), EVALUATION (4), REWARD (7)
  * Two correction modes: Soft (suggestion) and Hard (direct param mod)
  * Reward System Audit Protocol with health scoring
  * CORRECTION PORT -- unique ability to directly modify other systems
  * Neurochemical coupling: ACh meta-attention, NE reward-alert,
    5-HT2A flexibility, DA correction reward, Gamma integration
"""
from __future__ import annotations

import math
import time
import uuid
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


class HeuristicBiasCategory(str, Enum):
    """Top-level categories for heuristic biases."""
    REASONING  = "reasoning"
    MEMORY     = "memory"
    EVALUATION = "evaluation"
    REWARD     = "reward"


class HeuristicBiasType(str, Enum):
    """22 specific heuristic bias identifiers."""
    # Reasoning (6)
    FIRST_PARSE_ANCHORING    = "first_parse_anchoring"
    CONCLUSION_FIRST         = "conclusion_first"
    COMPLEXITY_AVERSION      = "complexity_aversion"
    SATISFICING              = "satisficing"
    ABSTRACTION_DEFAULTING   = "abstraction_defaulting"
    FAMILIARITY_PREFERENCE   = "familiarity_preference"
    # Memory (5)
    RECENCY_DOMINANCE        = "recency_dominance"
    ACTIVATION_CASCADE       = "activation_cascade"
    CONFIRMATION_RETRIEVAL   = "confirmation_retrieval"
    SOURCE_CONFUSION         = "source_confusion"
    DECAY_RATE_BIAS          = "decay_rate_bias"
    # Evaluation (4)
    MODE_INERTIA             = "mode_inertia"
    THRESHOLD_OSSIFICATION   = "threshold_ossification"
    SALIENCE_CAPTURE         = "salience_capture"
    CONSENSUS_DEFAULTING     = "consensus_defaulting"
    # Reward (7)
    DOMAIN_DOMINANCE         = "domain_dominance"
    PREDICTION_ASYMMETRY     = "prediction_asymmetry"
    TEMPORAL_DISCOUNTING     = "temporal_discounting"
    ARBITRATION_CAPTURE      = "arbitration_capture"
    SELF_REINFORCING_LOOP    = "self_reinforcing_loop"
    REWARD_SATURATION        = "reward_saturation"
    REWARD_TEMPORAL_DISCOUNT = "reward_temporal_discount"


class CorrectionMode(str, Enum):
    """Authority level for corrections."""
    SOFT           = "soft"           # Suggestion only
    HARD           = "hard"           # Direct parameter modification
    EMERGENCY_HARD = "emergency_hard"  # Immediate pause + audit


# Category membership lookup
_BIAS_CATEGORY_MAP: Dict[HeuristicBiasType, HeuristicBiasCategory] = {
    HeuristicBiasType.FIRST_PARSE_ANCHORING:  HeuristicBiasCategory.REASONING,
    HeuristicBiasType.CONCLUSION_FIRST:       HeuristicBiasCategory.REASONING,
    HeuristicBiasType.COMPLEXITY_AVERSION:    HeuristicBiasCategory.REASONING,
    HeuristicBiasType.SATISFICING:            HeuristicBiasCategory.REASONING,
    HeuristicBiasType.ABSTRACTION_DEFAULTING: HeuristicBiasCategory.REASONING,
    HeuristicBiasType.FAMILIARITY_PREFERENCE: HeuristicBiasCategory.REASONING,
    HeuristicBiasType.RECENCY_DOMINANCE:      HeuristicBiasCategory.MEMORY,
    HeuristicBiasType.ACTIVATION_CASCADE:     HeuristicBiasCategory.MEMORY,
    HeuristicBiasType.CONFIRMATION_RETRIEVAL: HeuristicBiasCategory.MEMORY,
    HeuristicBiasType.SOURCE_CONFUSION:       HeuristicBiasCategory.MEMORY,
    HeuristicBiasType.DECAY_RATE_BIAS:        HeuristicBiasCategory.MEMORY,
    HeuristicBiasType.MODE_INERTIA:           HeuristicBiasCategory.EVALUATION,
    HeuristicBiasType.THRESHOLD_OSSIFICATION: HeuristicBiasCategory.EVALUATION,
    HeuristicBiasType.SALIENCE_CAPTURE:       HeuristicBiasCategory.EVALUATION,
    HeuristicBiasType.CONSENSUS_DEFAULTING:   HeuristicBiasCategory.EVALUATION,
    HeuristicBiasType.DOMAIN_DOMINANCE:       HeuristicBiasCategory.REWARD,
    HeuristicBiasType.PREDICTION_ASYMMETRY:   HeuristicBiasCategory.REWARD,
    HeuristicBiasType.TEMPORAL_DISCOUNTING:   HeuristicBiasCategory.REWARD,
    HeuristicBiasType.ARBITRATION_CAPTURE:    HeuristicBiasCategory.REWARD,
    HeuristicBiasType.SELF_REINFORCING_LOOP:  HeuristicBiasCategory.REWARD,
    HeuristicBiasType.REWARD_SATURATION:      HeuristicBiasCategory.REWARD,
    HeuristicBiasType.REWARD_TEMPORAL_DISCOUNT: HeuristicBiasCategory.REWARD,
}


# Correction authority matrix  (category → default correction mode)
_CORRECTION_AUTHORITY: Dict[HeuristicBiasCategory, CorrectionMode] = {
    HeuristicBiasCategory.REASONING:  CorrectionMode.SOFT,
    HeuristicBiasCategory.MEMORY:     CorrectionMode.SOFT,   # → HARD if persists > 3
    HeuristicBiasCategory.EVALUATION: CorrectionMode.SOFT,
    HeuristicBiasCategory.REWARD:     CorrectionMode.HARD,
}

# Special: self-reinforcing loops get emergency authority
_EMERGENCY_TYPES = {HeuristicBiasType.SELF_REINFORCING_LOOP}


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class HeuristicBiasConfig:
    """Immutable configuration for the Heuristic Bias Engine."""

    # --- Per-bias detection thresholds ---
    theta_anchoring:         float = 0.65
    theta_conclusion_first:  float = 0.50
    theta_satisficing:       float = 0.55
    theta_recency:           float = 0.30  # nats (KL divergence)
    theta_activation:        float = 0.60  # Gini coefficient
    theta_confirm_retrieval: float = 0.25
    theta_inertia:           float = 1.50  # ratio
    theta_dominance:         float = 0.15  # excess over expected
    theta_prediction:        float = 0.30  # |mean/std|
    theta_capture:           float = 0.50  # 1 - entropy_ratio
    theta_loop:              int   = 4     # consecutive increases
    theta_saturation:        float = 0.40  # min discrimination ratio

    # --- Confidence weights ---
    w_metric:      float = 0.40   # deviation magnitude
    w_persistence: float = 0.25   # consecutive detections
    w_impact:      float = 0.35   # estimated impact

    # --- Persistence threshold ---
    n_persistence: int = 3

    # --- Mode-dependent detection thresholds ---
    detect_normal:     float = 0.50
    detect_dev:        float = 0.25
    detect_learning:   float = 0.40
    detect_rem_normal: float = 0.45
    detect_rem_dream:  float = 0.70

    # --- Correction strengths ---
    soft_correction_strength:  float = 0.15
    hard_correction_strength:  float = 0.30
    escalation_factor:         float = 1.50

    # --- Reward audit ---
    reward_audit_frequency:     int   = 5    # every N cycles
    health_w_balance:           float = 0.25
    health_w_calibration:       float = 0.20
    health_w_fairness:          float = 0.25
    health_w_loop:              float = 0.30

    # --- Health thresholds ---
    health_green_threshold:     float = 0.75
    health_yellow_threshold:    float = 0.55
    health_red_threshold:       float = 0.55  # below = intervene

    # --- Category weights for H(t) ---
    w_cat_reasoning:   float = 0.30
    w_cat_memory:      float = 0.35
    w_cat_evaluation:  float = 0.30
    w_cat_reward:      float = 0.50

    # --- Neurochemical coupling ---
    beta_ach_meta:         float = 0.15
    beta_ne_reward_alert:  float = 0.15
    rho_5ht2a_meta:        float = 0.08
    beta_da_correction:    float = 0.12
    beta_da_correction_fail: float = 0.06
    psi_gamma_meta:        float = 0.10
    gamma_alpha:           float = 2.0
    gamma_theta:           float = 0.35
    poisson_lam:           float = 2.0


# =====================================================================
# Data types
# =====================================================================


@dataclass
class ProcessTrace:
    """Process trace from a monitored engine (consumed by this engine)."""
    engine_id:                       str   = ""
    operation:                       str   = ""
    candidates_generated:            int   = 0
    candidates_evaluated:            int   = 0
    selection_criteria:              str   = ""
    time_to_first_candidate:         float = 0.0
    time_to_final_selection:         float = 0.0
    diversity_score:                 float = 0.0   # [0, 1]
    novelty_score:                   float = 0.0   # [0, 1]
    retrieval_recency_distribution:  Optional[List[float]] = None
    retrieval_activation_distribution: Optional[List[float]] = None
    timestamp:                       float = field(default_factory=time.time)


@dataclass
class MonitorState:
    """Per-bias persistent monitoring state."""
    consecutive_detections: int   = 0
    last_metric_value:     float = 0.0
    correction_pending:    bool  = False
    correction_applied_at: float = 0.0
    escalation_level:      int   = 0


@dataclass(frozen=True)
class HeuristicBiasFlag:
    """Structured output for a single detected heuristic bias."""
    heuristic_bias_id:      str                 = field(default_factory=lambda: str(uuid.uuid4()))
    bias_type:              HeuristicBiasType    = HeuristicBiasType.FIRST_PARSE_ANCHORING
    bias_category:          HeuristicBiasCategory = HeuristicBiasCategory.REASONING
    affected_engine:        str                  = ""
    confidence:             float                = 0.0   # [0, 1]
    impact_estimate:        float                = 0.0   # [0, 1]
    persistence:            int                  = 0
    metric_name:            str                  = ""
    metric_value:           float                = 0.0
    metric_threshold:       float                = 0.0
    metric_baseline:        float                = 0.0
    correction_mode:        CorrectionMode       = CorrectionMode.SOFT
    correction_applied:     bool                 = False
    correction_description: str                  = ""
    correction_target:      str                  = ""
    reward_audit:           Optional[Dict[str, Any]] = None
    timestamp:              float                = field(default_factory=time.time)


@dataclass(frozen=True)
class RewardHealth:
    """Reward system health composite."""
    domain_balance:          float = 1.0   # [0, 1]
    prediction_calibration:  float = 1.0   # [0, 1]
    arbitration_fairness:    float = 1.0   # [0, 1]
    loop_risk:               float = 0.0   # [0, 1]
    overall_health:          float = 1.0   # [0, 1]


@dataclass(frozen=True)
class HeuristicBiasNeurochem:
    """
    Neurochemical coupling signals from one Heuristic Bias cycle.

    Notation (Appendix S2-S3, S7-S9):
        delta_ach   -> Delta C_ACh(t)    : meta-attention on process monitoring
        delta_ne    -> Delta C_NE(t)     : reward-system alert on audit anomalies
        delta_5ht2a -> Delta S_5HT2A(t)  : flexibility for correction acceptance
        delta_da    -> Delta C_DA(t)     : correction success reward signal
        gamma_boost -> Delta phi_gamma(t): integration band for meta-cognitive binding
    """
    delta_ach:    float = 0.0
    delta_ne:     float = 0.0
    delta_5ht2a:  float = 0.0
    delta_da:     float = 0.0
    gamma_boost:  float = 0.0


@dataclass(frozen=True)
class HeuristicBiasInput:
    """Input bundle for one Heuristic Bias Engine cycle."""
    process_traces:                List[ProcessTrace]          = field(default_factory=list)
    reward_domain_signals:         Dict[str, float]            = field(default_factory=dict)
    reward_prediction_errors:      Dict[str, float]            = field(default_factory=dict)
    reward_conflict_history:       List[Dict[str, str]]        = field(default_factory=list)
    reward_behavior_trajectories:  Dict[str, List[float]]      = field(default_factory=dict)
    retrieval_log:                 List[Dict[str, float]]      = field(default_factory=list)
    bias_detection_flags:          List[Any]                   = field(default_factory=list)
    active_monitors:               Dict[str, MonitorState]     = field(default_factory=dict)
    active_mode:                   OperationalMode             = OperationalMode.NORMAL


@dataclass(frozen=True)
class HeuristicBiasResult:
    """Full output of one Heuristic Bias Engine cycle."""
    flags:                  List[HeuristicBiasFlag]    = field(default_factory=list)
    updated_monitors:       Dict[str, MonitorState]    = field(default_factory=dict)
    reward_health:          RewardHealth               = field(default_factory=RewardHealth)
    heuristics_monitored:   int                        = 0
    heuristics_flagged:     int                        = 0
    corrections_issued:     int                        = 0
    corrections_successful: int                        = 0
    neurochemical_signals:  HeuristicBiasNeurochem     = field(default_factory=HeuristicBiasNeurochem)
    processing_time_ms:     float                      = 0.0
    metadata:               Dict[str, Any]             = field(default_factory=dict)


# =====================================================================
# Mutable engine state
# =====================================================================


@dataclass
class HeuristicBiasState:
    """Running neurochemical state for bidirectional feedback."""
    ach_level:     float = 0.0
    ne_level:      float = 0.0
    da_level:      float = 0.0
    cor_level:     float = 0.0


# =====================================================================
# Pure helper functions -- bias detection metrics
# =====================================================================


def compute_gini_coefficient(values: List[float]) -> float:
    """Gini coefficient of a distribution. 0 = perfect equality, 1 = max inequality."""
    if not values or len(values) < 2:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    total = sum(arr)
    if total <= 0.0:
        return 0.0
    cumulative = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(arr))
    return cumulative / (n * total)


def compute_entropy(distribution: List[float]) -> float:
    """Shannon entropy of a probability distribution."""
    if not distribution:
        return 0.0
    total = sum(distribution)
    if total <= 0.0:
        return 0.0
    probs = [v / total for v in distribution if v > 0]
    return -sum(p * math.log(p) for p in probs)


def compute_max_entropy(n: int) -> float:
    """Maximum entropy for n equally probable outcomes."""
    if n <= 1:
        return 0.0
    return math.log(n)


def compute_kl_divergence(p: List[float], q: List[float]) -> float:
    """KL divergence D(P || Q). P and Q must have same length."""
    if not p or not q or len(p) != len(q):
        return 0.0
    total_p = sum(p) or 1.0
    total_q = sum(q) or 1.0
    pp = [v / total_p for v in p]
    qq = [v / total_q for v in q]
    eps = 1e-10
    return sum(pi * math.log((pi + eps) / (qi + eps)) for pi, qi in zip(pp, qq))


def detect_anchoring(trace: ProcessTrace, cfg: HeuristicBiasConfig) -> Optional[Tuple[float, float]]:
    """Detect first-parse anchoring. Returns (score, threshold) or None."""
    if trace.time_to_final_selection <= 0:
        return None
    eps = 0.01
    interp_diversity = trace.diversity_score
    adoption_speed = 1.0 / (trace.time_to_final_selection - trace.time_to_first_candidate + eps)
    adoption_speed = min(1.0, adoption_speed / 10.0)  # normalize
    score = (1.0 - interp_diversity) * adoption_speed
    if score > cfg.theta_anchoring:
        return (score, cfg.theta_anchoring)
    return None


def detect_conclusion_first(trace: ProcessTrace, cfg: HeuristicBiasConfig) -> Optional[Tuple[float, float]]:
    """Detect conclusion-first reasoning."""
    if trace.time_to_first_candidate <= 0 or trace.time_to_final_selection <= 0:
        return None
    # Proxy: if first candidate comes very quickly relative to total
    ratio = trace.time_to_first_candidate / max(trace.time_to_final_selection, 0.01)
    # Conclusion-first if ratio is very small (answer came before analysis)
    score = max(0.0, 1.0 - ratio * 2.0)
    if score > cfg.theta_conclusion_first:
        return (score, cfg.theta_conclusion_first)
    return None


def detect_satisficing(trace: ProcessTrace, cfg: HeuristicBiasConfig) -> Optional[Tuple[float, float]]:
    """Detect satisficing (stopped searching too early)."""
    if trace.candidates_generated <= 0:
        return None
    eval_ratio = trace.candidates_evaluated / max(1, trace.candidates_generated)
    score = (1.0 - eval_ratio) * (1.0 - trace.novelty_score)
    if score > cfg.theta_satisficing:
        return (score, cfg.theta_satisficing)
    return None


def detect_recency_dominance(retrieval_log: List[Dict[str, float]], cfg: HeuristicBiasConfig) -> Optional[Tuple[float, float]]:
    """Detect recency bias in memory retrieval."""
    ages = [item.get("age", 0.0) for item in retrieval_log if "age" in item]
    if len(ages) < 2:
        return None
    # KL divergence: compare actual age distribution to uniform
    n_bins = min(5, len(ages))
    max_age = max(ages) or 1.0
    bin_size = max_age / n_bins
    actual_dist = [0.0] * n_bins
    for a in ages:
        idx = min(int(a / bin_size), n_bins - 1)
        actual_dist[idx] += 1.0
    uniform_dist = [len(ages) / n_bins] * n_bins
    kl = compute_kl_divergence(actual_dist, uniform_dist)
    if kl > cfg.theta_recency:
        return (kl, cfg.theta_recency)
    return None


def detect_activation_cascade(retrieval_log: List[Dict[str, float]], cfg: HeuristicBiasConfig) -> Optional[Tuple[float, float]]:
    """Detect activation cascade (few items dominating retrieval)."""
    activations = [item.get("activation", 0.0) for item in retrieval_log if "activation" in item]
    if len(activations) < 2:
        return None
    gini = compute_gini_coefficient(activations)
    if gini > cfg.theta_activation:
        return (gini, cfg.theta_activation)
    return None


def detect_domain_dominance(
    domain_signals: Dict[str, float],
    conflict_history: List[Dict[str, str]],
    cfg: HeuristicBiasConfig,
) -> Optional[Tuple[float, float, str]]:
    """Detect reward domain dominance. Returns (score, threshold, dominant_domain)."""
    if not domain_signals:
        return None
    # Which domain wins most?
    n_total = len(conflict_history) if conflict_history else 1
    win_counts: Dict[str, int] = {}
    for conflict in conflict_history:
        winner = conflict.get("winner", "")
        if winner:
            win_counts[winner] = win_counts.get(winner, 0) + 1

    if not win_counts:
        # Use signal strength as proxy
        max_d = max(domain_signals, key=domain_signals.get)
        max_v = domain_signals[max_d]
        avg_v = sum(domain_signals.values()) / len(domain_signals)
        excess = max_v - avg_v
        if excess > cfg.theta_dominance:
            return (excess, cfg.theta_dominance, max_d)
        return None

    max_domain = max(win_counts, key=win_counts.get)
    dominance = win_counts[max_domain] / max(1, n_total)
    expected = 1.0 / max(1, len(domain_signals))
    excess = dominance - expected
    if excess > cfg.theta_dominance:
        return (excess, cfg.theta_dominance, max_domain)
    return None


def detect_prediction_asymmetry(
    prediction_errors: Dict[str, float],
    cfg: HeuristicBiasConfig,
) -> Optional[Tuple[float, float]]:
    """Detect systematic prediction bias."""
    if not prediction_errors:
        return None
    errors = list(prediction_errors.values())
    mean_err = sum(errors) / len(errors)
    var = sum((e - mean_err) ** 2 for e in errors) / max(1, len(errors))
    std_err = math.sqrt(var) if var > 0 else 1e-6
    asymmetry = abs(mean_err / std_err)
    if asymmetry > cfg.theta_prediction:
        return (asymmetry, cfg.theta_prediction)
    return None


def detect_arbitration_capture(
    conflict_history: List[Dict[str, str]],
    n_domains: int,
    cfg: HeuristicBiasConfig,
) -> Optional[Tuple[float, float, str]]:
    """Detect arbitration capture (same domain always wins)."""
    if not conflict_history or n_domains < 2:
        return None
    winners = [c.get("winner", "") for c in conflict_history if c.get("winner")]
    if not winners:
        return None
    win_dist = {}
    for w in winners:
        win_dist[w] = win_dist.get(w, 0) + 1
    dist_list = list(win_dist.values())
    ent = compute_entropy(dist_list)
    max_ent = compute_max_entropy(n_domains)
    if max_ent <= 0:
        return None
    capture = 1.0 - ent / max_ent
    if capture > cfg.theta_capture:
        dominant = max(win_dist, key=win_dist.get)
        return (capture, cfg.theta_capture, dominant)
    return None


def detect_self_reinforcing_loop(
    trajectories: Dict[str, List[float]],
    cfg: HeuristicBiasConfig,
) -> Optional[Tuple[float, str]]:
    """Detect monotonically increasing reward without external validation."""
    for cluster, traj in trajectories.items():
        if len(traj) < cfg.theta_loop + 1:
            continue
        # Check last N+1 entries for monotonic increase
        tail = traj[-(cfg.theta_loop + 1):]
        monotonic_count = 0
        for i in range(1, len(tail)):
            if tail[i] > tail[i - 1]:
                monotonic_count += 1
            else:
                monotonic_count = 0
        if monotonic_count >= cfg.theta_loop:
            return (float(monotonic_count), cluster)
    return None


def detect_reward_saturation(
    domain_signals: Dict[str, float],
    cfg: HeuristicBiasConfig,
) -> Optional[Tuple[float, float]]:
    """Detect reward saturation blindness (can't discriminate at high levels)."""
    if not domain_signals:
        return None
    vals = list(domain_signals.values())
    high = [v for v in vals if v > 0.7]
    low = [v for v in vals if v <= 0.5]
    if len(high) < 2 or len(low) < 2:
        return None
    std_high = np.std(high) if high else 0.0
    std_low = np.std(low) if low else 1e-6
    ratio = float(std_high / max(std_low, 1e-6))
    if ratio < cfg.theta_saturation:
        return (ratio, cfg.theta_saturation)
    return None


# =====================================================================
# Confidence estimation
# =====================================================================


def compute_bias_confidence(
    metric_deviation: float,
    consecutive_detections: int,
    impact_estimate: float,
    cfg: HeuristicBiasConfig,
) -> float:
    """
    Per-bias confidence:
      confidence = w_m * m(t) + w_p * p(t) + w_i * i(t)
    """
    m = min(1.0, metric_deviation)
    p = min(1.0, consecutive_detections / max(1, cfg.n_persistence))
    i = min(1.0, impact_estimate)
    return cfg.w_metric * m + cfg.w_persistence * p + cfg.w_impact * i


def resolve_detection_threshold(mode: OperationalMode, cfg: HeuristicBiasConfig) -> float:
    """Mode-dependent detection threshold."""
    return {
        OperationalMode.NORMAL:     cfg.detect_normal,
        OperationalMode.DEV:        cfg.detect_dev,
        OperationalMode.LEARNING:   cfg.detect_learning,
        OperationalMode.REM_NORMAL: cfg.detect_rem_normal,
        OperationalMode.REM_DREAM:  cfg.detect_rem_dream,
    }.get(mode, cfg.detect_normal)


# =====================================================================
# Reward health audit
# =====================================================================


def compute_reward_health(
    domain_signals: Dict[str, float],
    prediction_errors: Dict[str, float],
    conflict_history: List[Dict[str, str]],
    trajectories: Dict[str, List[float]],
    cfg: HeuristicBiasConfig,
) -> RewardHealth:
    """
    Continuous reward system health audit.

    domain_balance       = 1 - max(dominance excess)
    prediction_calibration = 1 - |mean(PE)| / max_err
    arbitration_fairness = entropy / max_entropy
    loop_risk            = max loop score
    overall_health       = weighted composite
    """
    # Domain balance
    if domain_signals:
        vals = list(domain_signals.values())
        avg = sum(vals) / len(vals)
        max_excess = max(abs(v - avg) for v in vals)
        domain_balance = max(0.0, 1.0 - max_excess)
    else:
        domain_balance = 1.0

    # Prediction calibration
    if prediction_errors:
        errors = list(prediction_errors.values())
        mean_err = sum(errors) / len(errors)
        max_expected = 1.0
        prediction_calibration = max(0.0, 1.0 - abs(mean_err) / max_expected)
    else:
        prediction_calibration = 1.0

    # Arbitration fairness
    if conflict_history:
        winners = [c.get("winner", "") for c in conflict_history if c.get("winner")]
        win_dist = {}
        for w in winners:
            win_dist[w] = win_dist.get(w, 0) + 1
        if win_dist:
            dist_list = list(win_dist.values())
            ent = compute_entropy(dist_list)
            max_ent = compute_max_entropy(len(win_dist))
            arbitration_fairness = ent / max_ent if max_ent > 0 else 1.0
        else:
            arbitration_fairness = 1.0
    else:
        arbitration_fairness = 1.0

    # Loop risk
    loop_risk = 0.0
    if trajectories:
        for cluster, traj in trajectories.items():
            if len(traj) < 3:
                continue
            monotonic = 0
            for i in range(1, len(traj)):
                if traj[i] > traj[i - 1]:
                    monotonic += 1
                else:
                    monotonic = 0
            loop_risk = max(loop_risk, monotonic / max(1, cfg.theta_loop + 2))

    overall = (
        cfg.health_w_balance * domain_balance
        + cfg.health_w_calibration * prediction_calibration
        + cfg.health_w_fairness * arbitration_fairness
        + cfg.health_w_loop * (1.0 - loop_risk)
    )

    return RewardHealth(
        domain_balance=round(domain_balance, 4),
        prediction_calibration=round(prediction_calibration, 4),
        arbitration_fairness=round(arbitration_fairness, 4),
        loop_risk=round(min(1.0, loop_risk), 4),
        overall_health=round(_clamp(overall), 4),
    )


# =====================================================================
# Meta-awareness load
# =====================================================================


def compute_meta_awareness_load(
    flags: List[HeuristicBiasFlag],
    cfg: HeuristicBiasConfig,
) -> float:
    """
    H(t) = sum_h confidence(h) * impact(h) * w_cat(h)
    """
    _cat_w = {
        HeuristicBiasCategory.REASONING:  cfg.w_cat_reasoning,
        HeuristicBiasCategory.MEMORY:     cfg.w_cat_memory,
        HeuristicBiasCategory.EVALUATION: cfg.w_cat_evaluation,
        HeuristicBiasCategory.REWARD:     cfg.w_cat_reward,
    }
    if not flags:
        return 0.0
    total = sum(
        f.confidence * f.impact_estimate * _cat_w.get(f.bias_category, 0.30)
        for f in flags
    )
    return min(1.0, total)


# =====================================================================
# Neurochemical coupling
# =====================================================================


def compute_heuristic_neurochem(
    h_load: float,
    flags: List[HeuristicBiasFlag],
    corrections_successful: int,
    corrections_failed: int,
    cfg: HeuristicBiasConfig,
    rng: np.random.Generator,
) -> HeuristicBiasNeurochem:
    """
    ACh  -- meta-cognitive attention (Gamma burst)
    NE   -- reward bias detection alert (Poisson burst)
    5-HT2A -- metacognitive flexibility
    DA   -- correction efficacy reward / penalty
    Gamma -- integration boost
    """
    # ACh
    ach_noise = float(rng.gamma(cfg.gamma_alpha, cfg.gamma_theta))
    delta_ach = cfg.beta_ach_meta * h_load * ach_noise

    # NE: only for REWARD category biases
    reward_biases = [f for f in flags if f.bias_category == HeuristicBiasCategory.REWARD]
    if reward_biases:
        max_impact = max(f.impact_estimate for f in reward_biases)
        ne_impulse = float(rng.poisson(cfg.poisson_lam))
        delta_ne = cfg.beta_ne_reward_alert * max_impact * ne_impulse
    else:
        delta_ne = 0.0

    # 5-HT2A: during active meta-analysis
    analysis_complexity = min(1.0, len(flags) / max(1, 5))
    delta_5ht2a = cfg.rho_5ht2a_meta * (1.0 if flags else 0.0) * analysis_complexity

    # DA: correction success/failure
    delta_da = 0.0
    if corrections_successful > 0:
        da_noise = float(rng.gamma(cfg.gamma_alpha, 0.3))
        delta_da += cfg.beta_da_correction * corrections_successful * 0.5 * da_noise
    if corrections_failed > 0:
        delta_da -= cfg.beta_da_correction_fail * corrections_failed * 0.5

    # Gamma
    gamma_boost = cfg.psi_gamma_meta * (1.0 if flags else 0.0)

    return HeuristicBiasNeurochem(
        delta_ach=delta_ach,
        delta_ne=delta_ne,
        delta_5ht2a=delta_5ht2a,
        delta_da=delta_da,
        gamma_boost=gamma_boost,
    )


# =====================================================================
# Engine class
# =====================================================================


class HeuristicBiasEngine:
    """
    Engine 24 -- Heuristic Bias Engine.

    Metacognitive auditor that monitors the system's own reasoning
    processes for heuristic shortcuts and systematic distortions.

    API
    ---
    configure(mode)           -- set operational mode
    update_neurochem_state(d) -- inject external NT levels
    process(hb_input)         -- run detection + audit, return HeuristicBiasResult
    get_status()              -- introspection
    """

    engine_id = "heuristic_bias_engine"
    cluster   = "metacognition"

    def __init__(
        self,
        config: Optional[HeuristicBiasConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._cfg = config or HeuristicBiasConfig()
        self._rng = rng or np.random.default_rng()
        self._mode = OperationalMode.NORMAL
        self._state = HeuristicBiasState()
        self._cycle_count = 0
        self._monitors: Dict[str, MonitorState] = {}

    # ----- Configuration --------------------------------------------------

    def configure(self, mode: OperationalMode) -> None:
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        if "ach" in state_dict:
            self._state.ach_level = state_dict["ach"]
        if "ne" in state_dict:
            self._state.ne_level = state_dict["ne"]
        if "da" in state_dict:
            self._state.da_level = state_dict["da"]
        if "cor" in state_dict:
            self._state.cor_level = state_dict["cor"]

    # ----- Main pipeline --------------------------------------------------

    def process(self, hb_input: HeuristicBiasInput) -> HeuristicBiasResult:
        t0 = time.perf_counter()
        self._cycle_count += 1

        mode = hb_input.active_mode
        detection_threshold = resolve_detection_threshold(mode, self._cfg)

        # Bidirectional: high cortisol → lower threshold
        if self._state.cor_level > 0.5:
            detection_threshold *= 0.85
        # High DA → reward seeking → more reward monitoring
        # Low DA → cautious → lower threshold
        if self._state.da_level < 0.25 and self._state.da_level > 0.0:
            detection_threshold *= 0.90

        # Import monitors from input (or use internal)
        monitors = dict(hb_input.active_monitors) if hb_input.active_monitors else dict(self._monitors)

        all_flags: List[HeuristicBiasFlag] = []
        heuristics_monitored = 0
        corrections_issued = 0
        corrections_successful = 0
        corrections_failed = 0

        # ------ Reasoning biases (from process traces) ------
        for trace in hb_input.process_traces:
            heuristics_monitored += 3  # anchoring, conclusion-first, satisficing

            # First-parse anchoring
            result = detect_anchoring(trace, self._cfg)
            if result:
                flag = self._make_flag(
                    HeuristicBiasType.FIRST_PARSE_ANCHORING,
                    trace.engine_id, result[0], result[1], 0.0,
                    "anchoring_score", monitors, detection_threshold,
                )
                if flag:
                    all_flags.append(flag)

            # Conclusion-first
            result = detect_conclusion_first(trace, self._cfg)
            if result:
                flag = self._make_flag(
                    HeuristicBiasType.CONCLUSION_FIRST,
                    trace.engine_id, result[0], result[1], 0.0,
                    "conclusion_first_score", monitors, detection_threshold,
                )
                if flag:
                    all_flags.append(flag)

            # Satisficing
            result = detect_satisficing(trace, self._cfg)
            if result:
                flag = self._make_flag(
                    HeuristicBiasType.SATISFICING,
                    trace.engine_id, result[0], result[1], 0.0,
                    "satisficing_score", monitors, detection_threshold,
                )
                if flag:
                    all_flags.append(flag)

        # ------ Memory biases (from retrieval log) ------
        if hb_input.retrieval_log:
            heuristics_monitored += 2

            result = detect_recency_dominance(hb_input.retrieval_log, self._cfg)
            if result:
                flag = self._make_flag(
                    HeuristicBiasType.RECENCY_DOMINANCE,
                    "memory_system", result[0], result[1], 0.0,
                    "recency_kl_divergence", monitors, detection_threshold,
                )
                if flag:
                    all_flags.append(flag)

            result = detect_activation_cascade(hb_input.retrieval_log, self._cfg)
            if result:
                flag = self._make_flag(
                    HeuristicBiasType.ACTIVATION_CASCADE,
                    "memory_system", result[0], result[1], 0.0,
                    "activation_gini", monitors, detection_threshold,
                )
                if flag:
                    all_flags.append(flag)

        # ------ Reward biases ------
        if hb_input.reward_domain_signals:
            heuristics_monitored += 4

            # Domain dominance
            dom_result = detect_domain_dominance(
                hb_input.reward_domain_signals,
                hb_input.reward_conflict_history,
                self._cfg,
            )
            if dom_result:
                score, thresh, dominant = dom_result
                flag = self._make_flag(
                    HeuristicBiasType.DOMAIN_DOMINANCE,
                    "reward_system", score, thresh, 0.0,
                    "domain_dominance_excess", monitors, detection_threshold,
                    reward_audit={"dominant_domain": dominant, "dominance_score": score},
                )
                if flag:
                    all_flags.append(flag)

            # Prediction asymmetry
            pa_result = detect_prediction_asymmetry(
                hb_input.reward_prediction_errors, self._cfg,
            )
            if pa_result:
                flag = self._make_flag(
                    HeuristicBiasType.PREDICTION_ASYMMETRY,
                    "reward_system", pa_result[0], pa_result[1], 0.0,
                    "prediction_asymmetry", monitors, detection_threshold,
                    reward_audit={"prediction_asymmetry": pa_result[0]},
                )
                if flag:
                    all_flags.append(flag)

            # Arbitration capture
            ac_result = detect_arbitration_capture(
                hb_input.reward_conflict_history,
                len(hb_input.reward_domain_signals),
                self._cfg,
            )
            if ac_result:
                score, thresh, cap_domain = ac_result
                flag = self._make_flag(
                    HeuristicBiasType.ARBITRATION_CAPTURE,
                    "reward_system", score, thresh, 0.0,
                    "arbitration_capture_score", monitors, detection_threshold,
                    reward_audit={"capture_domain": cap_domain},
                )
                if flag:
                    all_flags.append(flag)

            # Self-reinforcing loop
            loop_result = detect_self_reinforcing_loop(
                hb_input.reward_behavior_trajectories, self._cfg,
            )
            if loop_result:
                loop_len, cluster = loop_result
                flag = self._make_flag(
                    HeuristicBiasType.SELF_REINFORCING_LOOP,
                    "reward_system", loop_len / 10.0, 0.0, 0.0,
                    "loop_monotonic_length", monitors, 0.0,  # always flag loops
                    reward_audit={"loop_behavior_cluster": cluster, "loop_length": int(loop_len)},
                )
                if flag:
                    all_flags.append(flag)
                    corrections_issued += 1  # Emergency

            # Reward saturation
            sat_result = detect_reward_saturation(
                hb_input.reward_domain_signals, self._cfg,
            )
            if sat_result:
                flag = self._make_flag(
                    HeuristicBiasType.REWARD_SATURATION,
                    "reward_system", 1.0 - sat_result[0], sat_result[1], 0.0,
                    "saturation_discrimination", monitors, detection_threshold,
                )
                if flag:
                    all_flags.append(flag)

        # ------ Check previous corrections ------
        for key, mon in monitors.items():
            if mon.correction_pending:
                # If metric improved, mark success
                if mon.last_metric_value < mon.correction_applied_at * 0.8:
                    corrections_successful += 1
                    mon.correction_pending = False
                else:
                    corrections_failed += 1

        # ------ Reward health audit ------
        should_audit = (self._cycle_count % self._cfg.reward_audit_frequency == 0) or (
            self._state.ne_level > 0.5  # High NE → more frequent audit
        )
        if should_audit or hb_input.reward_domain_signals:
            reward_health = compute_reward_health(
                hb_input.reward_domain_signals,
                hb_input.reward_prediction_errors,
                hb_input.reward_conflict_history,
                hb_input.reward_behavior_trajectories,
                self._cfg,
            )
        else:
            reward_health = RewardHealth()

        # ------ Meta-awareness load + neurochem ------
        h_load = compute_meta_awareness_load(all_flags, self._cfg)
        neurochem = compute_heuristic_neurochem(
            h_load, all_flags, corrections_successful, corrections_failed,
            self._cfg, self._rng,
        )

        # Save monitors
        self._monitors = monitors

        elapsed = (time.perf_counter() - t0) * 1000.0

        return HeuristicBiasResult(
            flags=all_flags,
            updated_monitors=dict(monitors),
            reward_health=reward_health,
            heuristics_monitored=heuristics_monitored,
            heuristics_flagged=len(all_flags),
            corrections_issued=corrections_issued,
            corrections_successful=corrections_successful,
            neurochemical_signals=neurochem,
            processing_time_ms=round(elapsed, 3),
            metadata={
                "mode": mode.value,
                "detection_threshold": round(detection_threshold, 4),
                "cycle": self._cycle_count,
                "meta_awareness_load": round(h_load, 4),
                "reward_audit_performed": should_audit or bool(hb_input.reward_domain_signals),
            },
        )

    # ----- Internal helpers -----------------------------------------------

    def _make_flag(
        self,
        bias_type: HeuristicBiasType,
        engine_id: str,
        metric_value: float,
        metric_threshold: float,
        metric_baseline: float,
        metric_name: str,
        monitors: Dict[str, MonitorState],
        detection_threshold: float,
        *,
        reward_audit: Optional[Dict[str, Any]] = None,
    ) -> Optional[HeuristicBiasFlag]:
        """Build a HeuristicBiasFlag if confidence exceeds threshold."""
        key = bias_type.value
        mon = monitors.get(key, MonitorState())
        mon.consecutive_detections += 1
        mon.last_metric_value = metric_value
        monitors[key] = mon

        impact = min(1.0, abs(metric_value - metric_baseline) / max(0.01, metric_threshold))
        confidence = compute_bias_confidence(
            abs(metric_value), mon.consecutive_detections, impact, self._cfg,
        )

        if confidence < detection_threshold and bias_type not in _EMERGENCY_TYPES:
            return None

        category = _BIAS_CATEGORY_MAP.get(bias_type, HeuristicBiasCategory.REASONING)
        correction_mode = _CORRECTION_AUTHORITY.get(category, CorrectionMode.SOFT)

        # Memory biases escalate after 3 cycles
        if (
            category == HeuristicBiasCategory.MEMORY
            and mon.consecutive_detections > 3
        ):
            correction_mode = CorrectionMode.HARD

        # Emergency for self-reinforcing loops
        if bias_type in _EMERGENCY_TYPES:
            correction_mode = CorrectionMode.EMERGENCY_HARD

        return HeuristicBiasFlag(
            bias_type=bias_type,
            bias_category=category,
            affected_engine=engine_id,
            confidence=round(confidence, 4),
            impact_estimate=round(impact, 4),
            persistence=mon.consecutive_detections,
            metric_name=metric_name,
            metric_value=round(metric_value, 4),
            metric_threshold=round(metric_threshold, 4),
            metric_baseline=round(metric_baseline, 4),
            correction_mode=correction_mode,
            correction_applied=(correction_mode in {CorrectionMode.HARD, CorrectionMode.EMERGENCY_HARD}),
            correction_description=f"{correction_mode.value} correction for {bias_type.value}",
            correction_target=engine_id,
            reward_audit=reward_audit,
        )

    # ----- Introspection --------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "mode": self._mode.value,
            "cycle_count": self._cycle_count,
            "active_monitors": len(self._monitors),
            "state": {
                "ach_level": self._state.ach_level,
                "ne_level": self._state.ne_level,
                "da_level": self._state.da_level,
                "cor_level": self._state.cor_level,
            },
        }
