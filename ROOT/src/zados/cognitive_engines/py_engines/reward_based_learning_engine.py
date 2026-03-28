"""
Engine 17 -- Reward-Based Learning Engine  (``reward_based_learning_engine``)
=============================================================================
Converts reward signals into parameter adjustments via temporal-difference
prediction error:

    delta = r_actual - r_predicted

The engine maintains per-domain reward predictions (running exponential
averages) and per-parameter learning records.  Each cycle it:

  1. **Receives** reward signals from the reward layer (one float per domain).
  2. **Computes** prediction errors  delta_d = r_d - E[r_d]  for each domain.
  3. **Updates** learning rates (adaptive, NT-modulated, with decay schedule).
  4. **Produces** parameter adjustments scaled by delta and learning rate.
  5. **Tracks** convergence (shrinking |delta| over a sliding window).
  6. **Consolidates** parameters whose adjustments have been stable for a
     configurable window, exporting them as "learned" (frozen) values.

Lifecycle per cycle:
    reward_signals --> prediction_errors --> update_learning_rates -->
    parameter_adjustments --> convergence_tracking --> consolidation

NT Coupling (Pattern A: ``update_neurochem_state(Dict[str, float])``):
    DA   -- modulates learning rate upward (high DA = fast learning, explore)
    5-HT -- stabilises learning (damps oscillation, prevents overshoot)
    NE   -- increases learning rate under urgency
    ACh  -- deepens credit assignment (more parameters tracked per domain)
    GABA -- gates small deltas as noise (noise suppression threshold)
    OXT  -- social reward amplifier (boosts social domain deltas)
    CB1  -- creative exploration (widens parameter search in REM_DREAM)
    COR  -- stress penalty (shrinks learning rate under high cortisol)

Mode behaviour:
    NORMAL      -- standard learning rates, standard noise gate
    DEV         -- conservative learning, tighter convergence
    LEARNING    -- boosted learning rates, wider exploration
    REFLECTIVE  -- reduced learning rate, consolidation-focused
    REM_NORMAL  -- standard with mild replay
    REM_DREAM   -- creative parameter exploration, wide search, relaxed gate

Output types (frozen dataclasses):
    PredictionError       -- per-domain actual vs predicted
    ParameterAdjustment   -- per-parameter delta and new value
    RewardLearningNeurochem -- NT + oscillatory output signals
    RewardLearningResult  -- full cycle output bundle
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


class ConvergenceStatus(str, Enum):
    """How well a parameter's learning has converged."""
    DIVERGING    = "diverging"      # |delta| growing
    EXPLORING    = "exploring"      # |delta| fluctuating, no trend
    CONVERGING   = "converging"     # |delta| shrinking
    CONVERGED    = "converged"      # |delta| below threshold for window
    CONSOLIDATED = "consolidated"   # Frozen -- no more updates


class LearningPhase(str, Enum):
    """Global learning phase classification."""
    INITIAL      = "initial"        # First N cycles, high variance
    ACTIVE       = "active"         # Learning ongoing
    PLATEAU      = "plateau"        # Minimal change, near convergence
    CONSOLIDATED = "consolidated"   # Majority consolidated


class DomainType(str, Enum):
    """Reward domain identifiers (from reward layer)."""
    LOGIC       = "logic"
    ATTUNEMENT  = "attunement"
    INNOVATION  = "innovation"
    ETHICS      = "ethics"


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class RewardLearningConfig:
    """Immutable configuration for the Reward-Based Learning Engine."""

    # --- Learning rate ---
    initial_learning_rate: float = 0.10
    min_learning_rate:     float = 0.005
    max_learning_rate:     float = 0.50
    lr_decay_factor:       float = 0.995    # Per-cycle multiplicative decay
    lr_warmup_cycles:      int   = 5        # Cycles before decay starts

    # --- Prediction ---
    prediction_alpha:      float = 0.15     # EMA smoothing for reward prediction
    prediction_init:       float = 0.50     # Initial predicted reward

    # --- Convergence ---
    convergence_threshold: float = 0.02     # |delta| below this = converged
    convergence_window:    int   = 10       # Must stay below for N cycles

    # --- Consolidation ---
    consolidation_threshold: float = 0.015  # Mean |delta| below this triggers consolidation
    consolidation_window:    int   = 15     # Must be stable for N cycles
    max_consolidations_per_cycle: int = 3   # Rate-limit consolidation

    # --- Parameter tracking ---
    max_parameters:          int   = 64     # Max concurrently tracked parameters
    max_adjustment_history:  int   = 50     # History length per parameter
    delta_noise_gate:        float = 0.005  # |delta| below this is suppressed as noise

    # --- NT modulation weights ---
    w_da_lr:          float = 0.30    # DA learning rate boost weight
    w_5ht_stability:  float = 0.25    # 5-HT damping weight
    w_ne_urgency:     float = 0.20    # NE urgency boost weight
    w_ach_depth:      float = 0.15    # ACh credit-assignment depth weight
    w_gaba_gate:      float = 0.20    # GABA noise gate weight
    w_oxt_social:     float = 0.15    # OXT social domain amplifier
    w_cb1_explore:    float = 0.20    # CB1 exploration width
    w_cor_penalty:    float = 0.15    # Cortisol learning rate penalty

    # --- NT coupling output ---
    beta_da_rpe:        float = 0.12    # DA output from prediction error
    beta_5ht_stable:    float = 0.08    # 5-HT output from convergence
    beta_ne_surprise:   float = 0.10    # NE output from large |delta|
    beta_ach_depth:     float = 0.06    # ACh output from credit assignment depth
    theta_delta_large:  float = 0.15    # |delta| threshold for "large"
    poisson_lam:        float = 1.5     # Poisson lambda for NE impulse
    gamma_da_alpha:     float = 2.0     # Gamma distribution alpha for DA
    gamma_da_theta:     float = 0.30    # Gamma distribution theta for DA

    # --- Oscillatory output ---
    theta_boost_learning:  float = 0.10   # Theta during active learning
    gamma_boost_error:     float = 0.08   # Gamma on large prediction errors
    beta_boost_consolidation: float = 0.06  # Beta on consolidation events

    # --- Mode configs (multipliers on base learning rate) ---
    mode_lr_multiplier: Dict[str, float] = field(default_factory=lambda: {
        "normal":      1.00,
        "dev":         0.70,
        "learning":    1.40,
        "reflective":  0.60,
        "rem_normal":  0.90,
        "rem_dream":   1.60,
    })
    mode_noise_gate_multiplier: Dict[str, float] = field(default_factory=lambda: {
        "normal":      1.00,
        "dev":         1.30,    # Stricter noise gate
        "learning":    0.80,    # Looser gate
        "reflective":  1.20,
        "rem_normal":  1.00,
        "rem_dream":   0.50,    # Very loose gate
    })
    mode_consolidation_multiplier: Dict[str, float] = field(default_factory=lambda: {
        "normal":      1.00,
        "dev":         0.90,
        "learning":    1.20,    # Faster consolidation when learning
        "reflective":  0.70,    # Slower -- want to verify
        "rem_normal":  1.00,
        "rem_dream":   0.60,    # Slow -- exploring
    })


# =====================================================================
# Data types -- frozen output
# =====================================================================


@dataclass(frozen=True)
class PredictionError:
    """Per-domain prediction error."""
    domain:    str   = ""
    actual:    float = 0.0
    predicted: float = 0.0
    delta:     float = 0.0    # actual - predicted
    magnitude: float = 0.0    # |delta|


@dataclass(frozen=True)
class ParameterAdjustment:
    """Per-parameter learning adjustment."""
    parameter_id:   str   = ""
    domain:         str   = ""
    delta:          float = 0.0
    learning_rate:  float = 0.0
    old_value:      float = 0.0
    new_value:      float = 0.0
    gated:          bool  = False   # True if suppressed by noise gate


@dataclass(frozen=True)
class ConsolidationEvent:
    """Record of a parameter being consolidated (frozen)."""
    parameter_id:  str   = ""
    domain:        str   = ""
    final_value:   float = 0.0
    mean_delta:    float = 0.0
    total_updates: int   = 0
    cycle:         int   = 0


@dataclass(frozen=True)
class RewardLearningNeurochem:
    """
    Neurochemical coupling signals from the Reward-Based Learning Engine.

    Notation (Appendix S2-S3, S9):
        da_delta      -> Delta C_DA(t)      : RPE-driven (positive for +delta, negative for -delta)
        _5ht_delta    -> Delta C_5HT(t)     : convergence / stability signal
        ne_delta      -> Delta C_NE(t)      : surprise / large prediction error (Poisson)
        ach_delta     -> Delta C_ACh(t)     : credit assignment depth signal
        theta_boost   -> Delta phi_theta(t) : active learning oscillatory band
        gamma_boost   -> Delta phi_gamma(t) : error integration oscillatory band
        beta_boost    -> Delta phi_beta(t)  : consolidation focus oscillatory band
    """
    da_delta:      float = 0.0
    _5ht_delta:    float = 0.0
    ne_delta:      float = 0.0
    ach_delta:     float = 0.0
    theta_boost:   float = 0.0
    gamma_boost:   float = 0.0
    beta_boost:    float = 0.0


@dataclass(frozen=True)
class RewardLearningInput:
    """Input bundle for one Reward-Based Learning cycle."""
    reward_signals:   Dict[str, float] = field(default_factory=dict)   # domain -> [0, 1]
    parameter_values: Dict[str, float] = field(default_factory=dict)   # param_id -> current value
    parameter_domains: Dict[str, str]  = field(default_factory=dict)   # param_id -> domain
    active_mode:      OperationalMode  = OperationalMode.NORMAL


@dataclass(frozen=True)
class RewardLearningResult:
    """Full output of one Reward-Based Learning cycle."""
    prediction_errors:      List[PredictionError]      = field(default_factory=list)
    adjustments:            List[ParameterAdjustment]   = field(default_factory=list)
    consolidations:         List[ConsolidationEvent]    = field(default_factory=list)
    mean_abs_delta:         float                       = 0.0
    max_abs_delta:          float                       = 0.0
    convergence_ratio:      float                       = 0.0    # Fraction of params converged/consolidated
    active_parameters:      int                         = 0
    consolidated_parameters: int                        = 0
    learning_phase:         LearningPhase               = LearningPhase.INITIAL
    neurochemical_signals:  RewardLearningNeurochem     = field(default_factory=RewardLearningNeurochem)
    processing_time_ms:     float                       = 0.0
    metadata:               Dict[str, Any]              = field(default_factory=dict)


# =====================================================================
# Mutable engine state
# =====================================================================


@dataclass
class LearningRecord:
    """Mutable per-parameter learning tracker."""
    parameter_id:        str            = ""
    domain:              str            = ""
    value:               float          = 0.0
    learning_rate:       float          = 0.10
    adjustment_history:  List[float]    = field(default_factory=list)
    delta_history:       List[float]    = field(default_factory=list)
    convergence_status:  ConvergenceStatus = ConvergenceStatus.EXPLORING
    convergence_counter: int            = 0    # Consecutive cycles below threshold
    consolidated:        bool           = False
    total_updates:       int            = 0
    created_cycle:       int            = 0


@dataclass
class RewardPrediction:
    """Mutable per-domain reward prediction (running EMA)."""
    domain:     str   = ""
    predicted:  float = 0.50    # E[r_d]
    n_samples:  int   = 0


@dataclass
class RewardLearningState:
    """Top-level mutable state for the engine."""
    predictions:     Dict[str, RewardPrediction] = field(default_factory=dict)
    records:         Dict[str, LearningRecord]   = field(default_factory=dict)
    global_lr:       float  = 0.10   # Global learning rate (before NT modulation)
    # Neurochemical bidirectional
    da_level:        float  = 0.0
    _5ht_level:      float  = 0.0
    ne_level:        float  = 0.0
    ach_level:       float  = 0.0
    gaba_level:      float  = 0.0
    oxt_level:       float  = 0.0
    cb1_level:       float  = 0.0
    cor_level:       float  = 0.0


# =====================================================================
# Pure helper functions
# =====================================================================


def _mode_key(mode: OperationalMode) -> str:
    """Convert OperationalMode to config dict key."""
    return {
        OperationalMode.NORMAL:     "normal",
        OperationalMode.DEV:        "dev",
        OperationalMode.LEARNING:   "learning",
        OperationalMode.REFLECTIVE: "reflective",
        OperationalMode.REM_NORMAL: "rem_normal",
        OperationalMode.REM_DREAM:  "rem_dream",
    }.get(mode, "normal")


def compute_prediction_error(
    actual: float,
    predicted: float,
) -> Tuple[float, float]:
    """
    Compute TD-style prediction error.

    delta = r_actual - r_predicted
    Returns (delta, |delta|).
    """
    delta = actual - predicted
    return (delta, abs(delta))


def update_prediction_ema(
    current_predicted: float,
    actual: float,
    alpha: float,
) -> float:
    """
    Update reward prediction via exponential moving average.

    E[r]_{t+1} = (1 - alpha) * E[r]_t + alpha * r_actual
    """
    return (1.0 - alpha) * current_predicted + alpha * actual


def compute_effective_learning_rate(
    base_lr: float,
    cycle: int,
    cfg: RewardLearningConfig,
    da: float,
    sht: float,
    ne: float,
    cor: float,
    mode_mult: float,
) -> float:
    """
    Compute effective learning rate with NT modulation and decay.

    lr_eff = base_lr * mode_mult * decay^max(0, cycle - warmup)
             * (1 + w_da * (DA - 0.5))        # DA boosts above baseline
             * (1 - w_5ht * max(0, 5HT - 0.5)) # 5-HT damps above baseline
             * (1 + w_ne * max(0, NE - 0.4))   # NE urgency boost
             * (1 - w_cor * max(0, COR - 0.3))  # Cortisol penalty

    Clamped to [min_lr, max_lr].
    """
    # Decay schedule
    decay_cycles = max(0, cycle - cfg.lr_warmup_cycles)
    decay = cfg.lr_decay_factor ** decay_cycles

    lr = base_lr * mode_mult * decay

    # DA modulation: above 0.5 baseline boosts, below dampens
    lr *= (1.0 + cfg.w_da_lr * (da - 0.5))

    # 5-HT stabilisation: high 5-HT reduces oscillation
    lr *= (1.0 - cfg.w_5ht_stability * max(0.0, sht - 0.5))

    # NE urgency: above threshold speeds learning
    lr *= (1.0 + cfg.w_ne_urgency * max(0.0, ne - 0.4))

    # Cortisol penalty: stress reduces learning rate
    lr *= (1.0 - cfg.w_cor_penalty * max(0.0, cor - 0.3))

    return _clamp(lr, cfg.min_learning_rate, cfg.max_learning_rate)


def compute_noise_gate(
    base_gate: float,
    gaba: float,
    mode_mult: float,
    w_gaba: float,
) -> float:
    """
    Effective noise gate threshold.

    gate = base_gate * mode_mult * (1 + w_gaba * max(0, GABA - 0.4))

    High GABA raises the gate -> more noise suppression.
    """
    gate = base_gate * mode_mult * (1.0 + w_gaba * max(0.0, gaba - 0.4))
    return max(0.0, gate)


def compute_credit_depth(
    base_max: int,
    ach: float,
    w_ach: float,
    mode: OperationalMode,
) -> int:
    """
    How many parameters to track per domain (credit assignment depth).

    depth = base_max * (1 + w_ach * (ACh - 0.3))
    REM_DREAM gets bonus depth for exploration.
    """
    multiplier = 1.0 + w_ach * max(0.0, ach - 0.3)
    if mode == OperationalMode.REM_DREAM:
        multiplier *= 1.3
    return max(1, int(base_max * multiplier))


def compute_parameter_adjustment(
    delta: float,
    learning_rate: float,
    current_value: float,
    gated: bool,
) -> Tuple[float, float]:
    """
    Compute parameter adjustment.

    adjustment = lr * delta  (gradient ascent on reward)
    new_value = clamp(current + adjustment, 0, 1)

    If gated (noise suppression), adjustment is 0.
    Returns (adjustment, new_value).
    """
    if gated:
        return (0.0, current_value)
    adjustment = learning_rate * delta
    new_value = _clamp(current_value + adjustment)
    return (adjustment, new_value)


def assess_convergence(
    delta_history: List[float],
    threshold: float,
    window: int,
) -> ConvergenceStatus:
    """
    Determine convergence status from recent |delta| history.

    CONVERGED  if last ``window`` entries all below ``threshold``.
    CONVERGING if trend is downward.
    DIVERGING  if trend is upward.
    EXPLORING  otherwise.
    """
    if len(delta_history) < 3:
        return ConvergenceStatus.EXPLORING

    recent = delta_history[-window:] if len(delta_history) >= window else delta_history

    # Check converged: all recent below threshold
    if len(recent) >= window and all(abs(d) < threshold for d in recent):
        return ConvergenceStatus.CONVERGED

    # Trend analysis on last 5 entries
    tail = delta_history[-5:]
    if len(tail) >= 3:
        increasing = sum(
            1 for i in range(1, len(tail)) if abs(tail[i]) > abs(tail[i - 1])
        )
        decreasing = sum(
            1 for i in range(1, len(tail)) if abs(tail[i]) < abs(tail[i - 1])
        )
        n = len(tail) - 1
        if decreasing >= n * 0.7:
            return ConvergenceStatus.CONVERGING
        if increasing >= n * 0.7:
            return ConvergenceStatus.DIVERGING

    return ConvergenceStatus.EXPLORING


def should_consolidate(
    record: LearningRecord,
    threshold: float,
    window: int,
) -> bool:
    """
    Determine if a parameter should be consolidated.

    Consolidates when: not already consolidated, has enough history,
    and mean |delta| over the window is below threshold.
    """
    if record.consolidated:
        return False
    if len(record.delta_history) < window:
        return False
    recent = record.delta_history[-window:]
    mean_abs = sum(abs(d) for d in recent) / len(recent)
    return mean_abs < threshold


def classify_learning_phase(
    total_params: int,
    consolidated_count: int,
    converged_count: int,
    cycle: int,
    warmup: int,
) -> LearningPhase:
    """
    Classify the global learning phase.

    INITIAL       if cycle <= warmup.
    CONSOLIDATED  if majority consolidated.
    PLATEAU       if majority converged (but not consolidated).
    ACTIVE        otherwise.
    """
    if cycle <= warmup:
        return LearningPhase.INITIAL

    if total_params == 0:
        return LearningPhase.INITIAL

    consolidated_ratio = consolidated_count / total_params
    converged_ratio = (converged_count + consolidated_count) / total_params

    if consolidated_ratio >= 0.6:
        return LearningPhase.CONSOLIDATED
    if converged_ratio >= 0.6:
        return LearningPhase.PLATEAU
    return LearningPhase.ACTIVE


def compute_social_amplifier(
    domain: str,
    oxt: float,
    w_oxt: float,
) -> float:
    """
    Social domain reward amplifier.

    If domain is ATTUNEMENT and OXT is high, amplify prediction error
    to speed social learning.  Returns a multiplier >= 1.0.
    """
    if domain != DomainType.ATTUNEMENT.value:
        return 1.0
    return 1.0 + w_oxt * max(0.0, oxt - 0.4)


def compute_exploration_width(
    cb1: float,
    w_cb1: float,
    mode: OperationalMode,
) -> float:
    """
    Exploration width multiplier.  High CB1 (especially in REM_DREAM)
    widens the effective parameter search space by boosting learning rate
    for novel (high-delta) parameters.

    Returns a multiplier >= 1.0.
    """
    base = 1.0 + w_cb1 * max(0.0, cb1 - 0.3)
    if mode == OperationalMode.REM_DREAM:
        base *= 1.4
    return base


def compute_learning_neurochem(
    prediction_errors: List[PredictionError],
    convergence_ratio: float,
    active_params: int,
    max_params: int,
    consolidated_this_cycle: int,
    cfg: RewardLearningConfig,
    rng: np.random.Generator,
) -> RewardLearningNeurochem:
    """
    Compute neurochemical output signals from learning activity.

    DA    -- RPE-driven: positive delta -> DA+, negative -> DA-
    5-HT  -- convergence: high convergence ratio -> 5-HT+
    NE    -- surprise: large |delta| -> Poisson NE burst
    ACh   -- depth: high credit assignment utilisation -> ACh+
    Theta -- active learning
    Gamma -- error integration (large deltas)
    Beta  -- consolidation events
    """
    # DA: mean signed prediction error drives dopamine
    if prediction_errors:
        mean_delta = sum(pe.delta for pe in prediction_errors) / len(prediction_errors)
        max_mag = max(pe.magnitude for pe in prediction_errors)
    else:
        mean_delta = 0.0
        max_mag = 0.0

    # DA: Gamma-distributed noise on positive RPE, direct on negative
    da_delta = 0.0
    if mean_delta > 0.0:
        da_noise = float(rng.gamma(cfg.gamma_da_alpha, cfg.gamma_da_theta))
        da_delta = cfg.beta_da_rpe * mean_delta * da_noise
    elif mean_delta < 0.0:
        da_delta = cfg.beta_da_rpe * mean_delta  # Negative, no noise

    # 5-HT: convergence signal
    _5ht_delta = cfg.beta_5ht_stable * convergence_ratio

    # NE: surprise on large prediction errors
    ne_delta = 0.0
    if max_mag > cfg.theta_delta_large:
        ne_impulse = float(rng.poisson(cfg.poisson_lam))
        ne_delta = cfg.beta_ne_surprise * max_mag * ne_impulse

    # ACh: credit assignment depth utilisation
    ach_delta = 0.0
    if max_params > 0:
        utilisation = active_params / max_params
        ach_delta = cfg.beta_ach_depth * utilisation

    # Oscillatory
    theta_boost = cfg.theta_boost_learning if prediction_errors else 0.0
    gamma_boost = cfg.gamma_boost_error if max_mag > cfg.theta_delta_large else 0.0
    beta_boost = cfg.beta_boost_consolidation if consolidated_this_cycle > 0 else 0.0

    return RewardLearningNeurochem(
        da_delta=round(da_delta, 6),
        _5ht_delta=round(_5ht_delta, 6),
        ne_delta=round(ne_delta, 6),
        ach_delta=round(ach_delta, 6),
        theta_boost=round(theta_boost, 6),
        gamma_boost=round(gamma_boost, 6),
        beta_boost=round(beta_boost, 6),
    )


# =====================================================================
# Engine class
# =====================================================================


class RewardBasedLearningEngine:
    """
    Engine 17 -- Reward-Based Learning Engine.

    Converts reward signals into parameter adjustments via prediction
    error.  Maintains per-domain reward predictions and per-parameter
    learning records with adaptive, NT-modulated learning rates.

    API
    ---
    configure(mode)            -- set operational mode
    update_neurochem_state(d)  -- inject external NT levels (Pattern A)
    process(rl_input)          -- run one learning cycle
    get_status()               -- introspection
    register_parameter(pid, domain, initial) -- add a tracked parameter
    get_record(pid)            -- inspect a learning record
    get_prediction(domain)     -- inspect a reward prediction
    reset()                    -- clear all state
    """

    engine_id = "reward_based_learning_engine"
    cluster   = "learning"

    def __init__(
        self,
        config: Optional[RewardLearningConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._cfg = config or RewardLearningConfig()
        self._rng = rng or np.random.default_rng()
        self._mode = OperationalMode.NORMAL
        self._state = RewardLearningState(
            global_lr=self._cfg.initial_learning_rate,
        )
        self._cycle_count = 0

    # ----- Configuration --------------------------------------------------

    def configure(self, mode: OperationalMode) -> None:
        """Set operational mode."""
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """Pattern A: inject external NT levels via canonical keys."""
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
        if "oxt" in state_dict:
            self._state.oxt_level = state_dict["oxt"]
        if "cb1" in state_dict:
            self._state.cb1_level = state_dict["cb1"]
        if "cor" in state_dict:
            self._state.cor_level = state_dict["cor"]

    # ----- Parameter registration -----------------------------------------

    def register_parameter(
        self,
        parameter_id: str,
        domain: str,
        initial_value: float = 0.5,
    ) -> bool:
        """
        Register a parameter for reward-based learning.

        Returns True if registered, False if at capacity or already exists.
        """
        if parameter_id in self._state.records:
            return False
        if len(self._state.records) >= self._cfg.max_parameters:
            return False
        self._state.records[parameter_id] = LearningRecord(
            parameter_id=parameter_id,
            domain=domain,
            value=_clamp(initial_value),
            learning_rate=self._state.global_lr,
            created_cycle=self._cycle_count,
        )
        return True

    def get_record(self, parameter_id: str) -> Optional[LearningRecord]:
        """Get a parameter's learning record."""
        return self._state.records.get(parameter_id)

    def get_prediction(self, domain: str) -> Optional[RewardPrediction]:
        """Get a domain's reward prediction state."""
        return self._state.predictions.get(domain)

    # ----- Reset ----------------------------------------------------------

    def reset(self) -> None:
        """Clear all state and start fresh."""
        self._state = RewardLearningState(
            global_lr=self._cfg.initial_learning_rate,
        )
        self._cycle_count = 0

    # ----- Main pipeline --------------------------------------------------

    def process(self, rl_input: RewardLearningInput) -> RewardLearningResult:
        """
        Run one reward-based learning cycle.

        Pipeline:
          1. Update / create reward predictions per domain
          2. Compute prediction errors
          3. Compute effective learning rates (NT-modulated)
          4. Produce parameter adjustments
          5. Track convergence
          6. Consolidate stable parameters
          7. Compute neurochemical output
        """
        t0 = time.perf_counter()
        self._cycle_count += 1

        mode = rl_input.active_mode
        mk = _mode_key(mode)

        # Mode multipliers
        lr_mult = self._cfg.mode_lr_multiplier.get(mk, 1.0)
        gate_mult = self._cfg.mode_noise_gate_multiplier.get(mk, 1.0)
        consol_mult = self._cfg.mode_consolidation_multiplier.get(mk, 1.0)

        # Exploration width from CB1
        explore_mult = compute_exploration_width(
            self._state.cb1_level, self._cfg.w_cb1_explore, mode,
        )

        # ---- Step 1: Update reward predictions ----------------------------
        for domain, actual in rl_input.reward_signals.items():
            if domain not in self._state.predictions:
                self._state.predictions[domain] = RewardPrediction(
                    domain=domain,
                    predicted=self._cfg.prediction_init,
                    n_samples=0,
                )
            pred = self._state.predictions[domain]
            pred.n_samples += 1

        # ---- Step 2: Compute prediction errors ----------------------------
        prediction_errors: List[PredictionError] = []
        domain_deltas: Dict[str, float] = {}

        for domain, actual in rl_input.reward_signals.items():
            pred = self._state.predictions[domain]
            delta, magnitude = compute_prediction_error(actual, pred.predicted)

            # Social amplifier
            social_amp = compute_social_amplifier(
                domain, self._state.oxt_level, self._cfg.w_oxt_social,
            )
            delta *= social_amp
            magnitude = abs(delta)

            prediction_errors.append(PredictionError(
                domain=domain,
                actual=round(actual, 6),
                predicted=round(pred.predicted, 6),
                delta=round(delta, 6),
                magnitude=round(magnitude, 6),
            ))
            domain_deltas[domain] = delta

            # Update prediction EMA
            pred.predicted = update_prediction_ema(
                pred.predicted, actual, self._cfg.prediction_alpha,
            )

        # ---- Step 3: Compute effective learning rates --------------------
        effective_lr = compute_effective_learning_rate(
            self._state.global_lr,
            self._cycle_count,
            self._cfg,
            self._state.da_level,
            self._state._5ht_level,
            self._state.ne_level,
            self._state.cor_level,
            lr_mult,
        )

        # Apply exploration width to learning rate for novel parameters
        effective_lr_explore = effective_lr * explore_mult

        # Noise gate
        noise_gate = compute_noise_gate(
            self._cfg.delta_noise_gate,
            self._state.gaba_level,
            gate_mult,
            self._cfg.w_gaba_gate,
        )

        # Credit assignment depth
        credit_depth = compute_credit_depth(
            self._cfg.max_parameters,
            self._state.ach_level,
            self._cfg.w_ach_depth,
            mode,
        )

        # ---- Step 4: Compute parameter adjustments -----------------------
        adjustments: List[ParameterAdjustment] = []

        # Auto-register parameters from input if not already registered
        for pid, pval in rl_input.parameter_values.items():
            if pid not in self._state.records:
                domain = rl_input.parameter_domains.get(pid, "")
                if domain and len(self._state.records) < credit_depth:
                    self.register_parameter(pid, domain, pval)

        for pid, record in list(self._state.records.items()):
            if record.consolidated:
                continue

            domain = record.domain
            delta = domain_deltas.get(domain, 0.0)

            # Use current value from input if provided, else internal
            current_val = rl_input.parameter_values.get(pid, record.value)

            # Noise gate
            gated = abs(delta) < noise_gate

            # Choose learning rate: novel params get exploration bonus
            param_lr = effective_lr_explore if record.total_updates < 5 else effective_lr

            # Per-parameter learning rate with 5-HT damping for oscillation
            if len(record.adjustment_history) >= 2:
                # Detect oscillation: last two adjustments have opposite signs
                last_two = record.adjustment_history[-2:]
                if last_two[0] * last_two[1] < 0:
                    # Oscillating -- 5-HT dampens further
                    damping = 1.0 - self._cfg.w_5ht_stability * max(
                        0.0, self._state._5ht_level - 0.3,
                    )
                    param_lr *= _clamp(damping, 0.3, 1.0)

            record.learning_rate = param_lr

            adjustment, new_value = compute_parameter_adjustment(
                delta, param_lr, current_val, gated,
            )

            record.value = new_value
            record.total_updates += 1
            record.delta_history.append(delta)
            if not gated:
                record.adjustment_history.append(adjustment)

            # Trim histories
            if len(record.delta_history) > self._cfg.max_adjustment_history:
                record.delta_history = record.delta_history[-self._cfg.max_adjustment_history:]
            if len(record.adjustment_history) > self._cfg.max_adjustment_history:
                record.adjustment_history = record.adjustment_history[-self._cfg.max_adjustment_history:]

            adjustments.append(ParameterAdjustment(
                parameter_id=pid,
                domain=domain,
                delta=round(delta, 6),
                learning_rate=round(param_lr, 6),
                old_value=round(current_val, 6),
                new_value=round(new_value, 6),
                gated=gated,
            ))

        # ---- Step 5: Track convergence -----------------------------------
        converged_count = 0
        consolidated_count = 0

        for record in self._state.records.values():
            if record.consolidated:
                consolidated_count += 1
                continue

            status = assess_convergence(
                record.delta_history,
                self._cfg.convergence_threshold,
                self._cfg.convergence_window,
            )
            record.convergence_status = status
            if status in (ConvergenceStatus.CONVERGED, ConvergenceStatus.CONSOLIDATED):
                converged_count += 1
                record.convergence_counter += 1
            else:
                record.convergence_counter = 0

        # ---- Step 6: Consolidate stable parameters -----------------------
        consolidations: List[ConsolidationEvent] = []
        consolidated_this_cycle = 0

        # Adjusted consolidation threshold by mode
        consol_threshold = self._cfg.consolidation_threshold * consol_mult
        consol_window = self._cfg.consolidation_window

        for pid, record in self._state.records.items():
            if consolidated_this_cycle >= self._cfg.max_consolidations_per_cycle:
                break
            if should_consolidate(record, consol_threshold, consol_window):
                record.consolidated = True
                record.convergence_status = ConvergenceStatus.CONSOLIDATED
                consolidated_count += 1
                consolidated_this_cycle += 1
                recent = record.delta_history[-consol_window:]
                mean_d = sum(abs(d) for d in recent) / len(recent) if recent else 0.0
                consolidations.append(ConsolidationEvent(
                    parameter_id=pid,
                    domain=record.domain,
                    final_value=round(record.value, 6),
                    mean_delta=round(mean_d, 6),
                    total_updates=record.total_updates,
                    cycle=self._cycle_count,
                ))

        # ---- Aggregate metrics -------------------------------------------
        total_params = len(self._state.records)
        active_params = total_params - consolidated_count

        all_deltas = [abs(pe.delta) for pe in prediction_errors]
        mean_abs_delta = (sum(all_deltas) / len(all_deltas)) if all_deltas else 0.0
        max_abs_delta = max(all_deltas) if all_deltas else 0.0

        convergence_ratio = 0.0
        if total_params > 0:
            convergence_ratio = (converged_count + consolidated_count) / total_params

        learning_phase = classify_learning_phase(
            total_params, consolidated_count, converged_count,
            self._cycle_count, self._cfg.lr_warmup_cycles,
        )

        # ---- Step 7: Neurochemical output --------------------------------
        neurochem = compute_learning_neurochem(
            prediction_errors,
            convergence_ratio,
            active_params,
            self._cfg.max_parameters,
            consolidated_this_cycle,
            self._cfg,
            self._rng,
        )

        # Update global learning rate for next cycle
        self._state.global_lr = effective_lr

        elapsed = (time.perf_counter() - t0) * 1000.0

        return RewardLearningResult(
            prediction_errors=prediction_errors,
            adjustments=adjustments,
            consolidations=consolidations,
            mean_abs_delta=round(mean_abs_delta, 6),
            max_abs_delta=round(max_abs_delta, 6),
            convergence_ratio=round(convergence_ratio, 4),
            active_parameters=active_params,
            consolidated_parameters=consolidated_count,
            learning_phase=learning_phase,
            neurochemical_signals=neurochem,
            processing_time_ms=round(elapsed, 3),
            metadata={
                "mode": mode.value,
                "cycle": self._cycle_count,
                "effective_lr": round(effective_lr, 6),
                "noise_gate": round(noise_gate, 6),
                "credit_depth": credit_depth,
                "exploration_mult": round(explore_mult, 4),
                "domains_active": list(rl_input.reward_signals.keys()),
                "consolidated_this_cycle": consolidated_this_cycle,
            },
        )

    # ----- Introspection --------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return engine introspection data."""
        total = len(self._state.records)
        consolidated = sum(1 for r in self._state.records.values() if r.consolidated)
        converged = sum(
            1 for r in self._state.records.values()
            if r.convergence_status == ConvergenceStatus.CONVERGED and not r.consolidated
        )
        return {
            "engine_id": self.engine_id,
            "mode": self._mode.value,
            "cycle_count": self._cycle_count,
            "global_lr": round(self._state.global_lr, 6),
            "total_parameters": total,
            "active_parameters": total - consolidated,
            "converged_parameters": converged,
            "consolidated_parameters": consolidated,
            "domains_tracked": list(self._state.predictions.keys()),
            "nt_levels": {
                "da": round(self._state.da_level, 4),
                "5ht": round(self._state._5ht_level, 4),
                "ne": round(self._state.ne_level, 4),
                "ach": round(self._state.ach_level, 4),
                "gaba": round(self._state.gaba_level, 4),
                "oxt": round(self._state.oxt_level, 4),
                "cb1": round(self._state.cb1_level, 4),
                "cor": round(self._state.cor_level, 4),
            },
        }
