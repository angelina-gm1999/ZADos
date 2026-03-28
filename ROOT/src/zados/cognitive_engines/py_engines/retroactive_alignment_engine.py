"""
Engine 30 -- Retro-Active Alignment Error Detection Engine
==========================================================
Temporal coherence auditor.  Detects drift between past and present
system states, answering: "Given what I know now, do the things I
previously thought, felt, and decided still make sense?"

Key features:
  * 4-component state vector: S(t) = [R(t), C(t), B(t), E(t)]
  * Alignment projector A(): forward-projects past state, accounting for
    acknowledged changes.
  * 3-way error decomposition: delta_sym + delta_aff + delta_reward.
  * Sigmoid collapse probability with interaction terms.
  * Temporal EWMA smoothing with hysteresis gating.
  * Affective consequence mapping (10 emotion triggers).
  * 4 corrective action types: symbolic contradiction, affective bridge,
    memory trust, reward recalibration.
  * 4 scan horizons: IMMEDIATE, SESSION, CROSS_SESSION, IDENTITY.
  * Neurochemical coupling: COR, 5-HT1A, NE, DA, OXT + oscillatory.
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


class DeltaComponent(str, Enum):
    """Which component dominates the alignment error."""
    SYM    = "symbolic"
    AFF    = "affective"
    REWARD = "reward"


class CollapseState(str, Enum):
    """Collapse probability state classification."""
    STABLE           = "stable"            # P < 0.15
    ELEVATED         = "elevated"          # 0.15 ≤ P < 0.30
    AT_RISK          = "at_risk"           # 0.30 ≤ P < 0.50
    CRITICAL         = "critical"          # 0.50 ≤ P < 0.70
    COLLAPSE_IMMINENT = "collapse_imminent" # P ≥ 0.70


class AttributionType(str, Enum):
    """Who/what caused the alignment error."""
    SELF    = "self"      # System's own processing
    OTHER   = "other"     # External agent behavior
    SYSTEM  = "system"    # Architectural limitation
    UNKNOWN = "unknown"


class DriftTrend(str, Enum):
    """Trend of alignment error over time."""
    INCREASING  = "increasing"
    STABLE      = "stable"
    DECREASING  = "decreasing"
    OSCILLATING = "oscillating"


class ScanHorizon(str, Enum):
    """Temporal scan horizon."""
    IMMEDIATE     = "immediate"       # Last 2-4 cycles
    SESSION       = "session"         # Current conversation
    CROSS_SESSION = "cross_session"   # Last 3-5 sessions
    IDENTITY      = "identity"        # Identity-relevant (REM only)


class CorrectionType(str, Enum):
    """Type of corrective action emitted."""
    SYMBOLIC_CONTRADICTION = "symbolic_contradiction"
    AFFECTIVE_BRIDGE       = "affective_bridge"
    MEMORY_TRUST           = "memory_trust"
    REWARD_RECALIBRATION   = "reward_recalibration"


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class RetroactiveAlignmentConfig:
    """Immutable configuration for the Retroactive Alignment Engine."""

    # --- Alignment error weights (for weighted norm) ---
    w_reward:    float = 0.30
    w_neurochem: float = 0.25
    w_oscillatory: float = 0.15
    w_emotional: float = 0.30

    # --- Delta component thresholds ---
    # Symbolic drift
    sym_normal:      float = 0.15
    sym_drift:       float = 0.30   # SIGNIFICANT
    sym_critical:    float = 0.50
    # Affective incongruence
    aff_normal:      float = 0.12
    aff_significant: float = 0.25
    aff_critical:    float = 0.40
    # Reward trajectory
    rew_normal:      float = 0.10
    rew_significant: float = 0.20
    rew_critical:    float = 0.35

    # --- Collapse sigmoid coefficients ---
    alpha_sym:     float = 3.0    # Symbolic weight
    alpha_aff:     float = 2.5    # Affective weight
    alpha_rew:     float = 2.0    # Reward weight
    beta_interact: float = 1.5    # Interaction term

    # --- Collapse state thresholds ---
    collapse_stable:   float = 0.15
    collapse_elevated: float = 0.30
    collapse_at_risk:  float = 0.50
    collapse_critical: float = 0.70

    # --- Temporal smoothing ---
    tau_mem:          float = 4.0    # EWMA memory (cycles)
    t_hysteresis:     int   = 2      # Min cycles above threshold
    theta_correction: float = 0.25   # Smoothed threshold for correction

    # --- Correction rates ---
    eta_trust:       float = 0.30    # Memory trust re-weighting
    lambda_recal:    float = 0.20    # Reward trajectory correction
    tau_legit_evol:  float = 15.0    # Legitimate evolution timescale (cycles)

    # --- Neurochemical coupling ---
    beta_temporal_threat:  float = 0.12   # COR
    theta_cortisol_onset:  float = 0.20   # Min delta for COR
    beta_stability:        float = 0.10   # 5-HT1A
    beta_vigilance:        float = 0.10   # NE
    beta_retro_rpe:        float = 0.10   # DA negative RPE
    theta_reward_sig:      float = 0.15   # Min delta_reward for DA
    beta_correction_success: float = 0.15 # DA positive
    beta_trust_decay:      float = 0.08   # OXT negative
    beta_trust_repair:     float = 0.05   # OXT positive
    kappa_collapse:        float = 1.5    # P_collapse amplifier

    # --- Oscillatory ---
    theta_boost:           float = 0.10   # Theta for any error
    gamma_boost_sym:       float = 0.08   # Gamma for symbolic
    alpha_theta_boost_aff: float = 0.10   # Alpha-Theta for affective
    beta_boost_rew:        float = 0.08   # Beta for reward
    delta_boost_collapse:  float = 0.12   # Delta for high collapse

    # --- Stochastic ---
    poisson_lam_ne: float = 1.5
    gamma_da_alpha: float = 2.0
    gamma_da_theta: float = 0.30

    # --- Mode thresholds (adjustment multiplier) ---
    # Multiplied onto base thresholds
    mode_dev_adj:        float = 0.80    # -20% (more sensitive)
    mode_learning_adj:   float = 0.90    # -10%
    mode_reflective_adj: float = 0.85    # -15%
    mode_rem_normal_adj: float = 0.90    # -10%
    mode_rem_dream_adj:  float = 1.15    # +15% (less sensitive)

    # --- Scan frequencies ---
    immediate_freq:      int = 1     # Every cycle
    session_freq:        int = 5
    cross_session_freq:  int = 15
    identity_freq:       int = 100   # REM only


# =====================================================================
# Data types
# =====================================================================


@dataclass(frozen=True)
class SystemStateVector:
    """System cognitive state at a point in time."""
    reward_signals:              Dict[str, float] = field(default_factory=dict)
    neurochemical_concentrations: Dict[str, float] = field(default_factory=dict)
    oscillatory_powers:          Dict[str, float] = field(default_factory=dict)
    emotional_activations:       Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class HistoricalState:
    """A past state snapshot from memory."""
    timestamp:          float               = 0.0
    state_vector:       SystemStateVector    = field(default_factory=SystemStateVector)
    processing_context: str                  = ""
    memory_tier:        str                  = "STMM"
    trust_weight:       float               = 1.0


@dataclass(frozen=True)
class AcknowledgedChange:
    """A known, logged change to the system state."""
    timestamp: float = 0.0
    component: str   = ""     # REWARD | NEUROCHEM | OSCILLATORY | EMOTIONAL
    description: str = ""
    cause: str       = ""


@dataclass(frozen=True)
class AlignmentAffectSignal:
    """Signal emitted to the Emotional Detection Engine."""
    dominant_delta:      DeltaComponent  = DeltaComponent.SYM
    delta_magnitude:     float           = 0.0
    attribution:         AttributionType = AttributionType.UNKNOWN
    attribution_domain:  str             = ""
    suggested_emotion:   str             = ""
    collapse_probability: float          = 0.0
    drift_timespan:      int             = 0
    drift_rate:          float           = 0.0
    scan_horizon:        ScanHorizon     = ScanHorizon.IMMEDIATE


@dataclass(frozen=True)
class CorrectionEmitted:
    """Record of a corrective action."""
    correction_type: CorrectionType = CorrectionType.SYMBOLIC_CONTRADICTION
    target_engine:   str            = ""
    magnitude:       float          = 0.0
    description:     str            = ""


@dataclass(frozen=True)
class AlignmentNeurochem:
    """
    Neurochemical coupling signals from Retroactive Alignment Error Detection.

    Notation (Appendix S2-S3, S7-S9):
        delta_cor        -> Delta C_Cortisol(t) : threshold-gated stress from P_collapse
        delta_5ht1a      -> Delta S_5HT1A(t)    : self/system attribution calming
        delta_ne         -> Delta C_NE(t)        : other/unknown attribution alerting (Poisson)
        delta_da         -> Delta C_DA(t)        : negative RPE on error + positive on correction
        delta_oxt        -> Delta C_OXT(t)       : trust decay on OTHER errors / repair on correction
        theta_boost      -> Delta phi_theta(t)   : error-processing oscillatory band (S7, S9)
        gamma_boost      -> Delta phi_gamma(t)   : symbolic error integration (S7, S9)
        alpha_theta_boost -> Delta phi_alpha-theta(t) : affective processing cross-coupling (S7)
        beta_boost       -> Delta phi_beta(t)    : reward recalibration focus (S7, S9)
        delta_osc_boost  -> Delta phi_delta(t)   : high-collapse disengagement (S7, S9)
    """
    delta_cor:     float = 0.0
    delta_5ht1a:   float = 0.0
    delta_ne:      float = 0.0
    delta_da:      float = 0.0
    delta_oxt:     float = 0.0
    theta_boost:   float = 0.0
    gamma_boost:   float = 0.0
    alpha_theta_boost: float = 0.0
    beta_boost:    float = 0.0
    delta_osc_boost: float = 0.0   # Delta band boost for high collapse


@dataclass(frozen=True)
class RetroactiveAlignmentInput:
    """Input bundle for one alignment detection cycle."""
    current_state:       SystemStateVector       = field(default_factory=SystemStateVector)
    historical_states:   List[HistoricalState]    = field(default_factory=list)
    acknowledged_changes: List[AcknowledgedChange] = field(default_factory=list)
    reactive_trigger:    Optional[Dict[str, Any]] = None
    active_mode:         OperationalMode          = OperationalMode.NORMAL
    scan_horizon:        ScanHorizon              = ScanHorizon.IMMEDIATE


@dataclass(frozen=True)
class RetroactiveAlignmentResult:
    """Full output of one alignment detection cycle."""
    alignment_error_id:     str                   = field(default_factory=lambda: str(uuid.uuid4()))
    delta_symbolic:         float                 = 0.0
    delta_affective:        float                 = 0.0
    delta_reward:           float                 = 0.0
    delta_total:            float                 = 0.0
    delta_smoothed:         float                 = 0.0
    collapse_probability:   float                 = 0.0
    collapse_state:         CollapseState         = CollapseState.STABLE
    dominant_component:     DeltaComponent        = DeltaComponent.SYM
    component_ratio:        Dict[str, float]      = field(default_factory=dict)
    attribution:            AttributionType       = AttributionType.UNKNOWN
    attribution_confidence: float                 = 0.0
    attribution_domain:     str                   = ""
    causal_chain:           List[str]             = field(default_factory=list)
    scan_horizon:           ScanHorizon           = ScanHorizon.IMMEDIATE
    drift_timespan:         int                   = 0
    drift_rate:             float                 = 0.0
    drift_trend:            DriftTrend            = DriftTrend.STABLE
    affective_signal:       Optional[AlignmentAffectSignal] = None
    triggered_emotion:      Optional[str]         = None
    corrections_emitted:    List[CorrectionEmitted] = field(default_factory=list)
    neurochemical_signals:  AlignmentNeurochem     = field(default_factory=AlignmentNeurochem)
    compared_states:        int                   = 0
    processing_time_ms:     float                 = 0.0
    scan_trigger:           str                   = "SCHEDULED"
    metadata:               Dict[str, Any]        = field(default_factory=dict)


# =====================================================================
# Mutable engine state
# =====================================================================


@dataclass
class AlignmentState:
    """Running state for temporal smoothing and tracking."""
    delta_bar:            float      = 0.0    # Smoothed alignment error
    time_above_threshold: int        = 0      # Cycles above correction threshold
    delta_history:        List[float] = field(default_factory=list)
    correction_pending:   bool       = False
    last_correction_magnitude: float = 0.0
    # Neurochemical bidirectional
    cor_level:   float = 0.0
    da_level:    float = 0.0
    ne_level:    float = 0.0
    oxt_level:   float = 0.5
    _5ht_level:  float = 0.0   # 5-HT (serotonin)


# =====================================================================
# Pure helper functions
# =====================================================================


def sigmoid_fn(x: float) -> float:
    """Standard sigmoid."""
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def euclidean_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Euclidean distance between two signal dicts."""
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0
    return math.sqrt(sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in keys))


def cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two dicts."""
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 1.0
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    norm_a = math.sqrt(sum(a.get(k, 0.0) ** 2 for k in keys))
    norm_b = math.sqrt(sum(b.get(k, 0.0) ** 2 for k in keys))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


def project_reward(
    past_rewards: Dict[str, float],
    acknowledged: List[AcknowledgedChange],
) -> Dict[str, float]:
    """
    A_R: project expected current reward from past + acknowledged changes.
    A_R(R(t-tau)) = R(t-tau) + sum delta_acknowledged
    """
    projected = dict(past_rewards)
    for change in acknowledged:
        if change.component == "REWARD":
            # Apply acknowledged adjustment (simplified: small shift in direction)
            for key in projected:
                if key in change.description:
                    projected[key] = _clamp(projected[key] + 0.05)
    return projected


def project_neurochem(
    past_concentrations: Dict[str, float],
    tau: float,
    acknowledged: List[AcknowledgedChange],
) -> Dict[str, float]:
    """
    A_C: project expected current NT levels from past.
    A_C = R_0 + (C(t-tau) - R_0) * exp(-gamma_c * tau) + delta_justified
    """
    gamma_c = 0.10  # Decay rate toward baseline
    R_0 = 0.40      # Generic baseline

    projected = {}
    for nt, val in past_concentrations.items():
        # Exponential decay toward baseline
        proj = R_0 + (val - R_0) * math.exp(-gamma_c * tau)
        projected[nt] = _clamp(proj)

    # Apply acknowledged shifts
    for change in acknowledged:
        if change.component == "NEUROCHEM":
            for nt in projected:
                if nt in change.description:
                    projected[nt] = _clamp(projected[nt] + 0.03)

    return projected


def project_emotional(
    past_emotions: Dict[str, float],
    tau: float,
    acknowledged: List[AcknowledgedChange],
) -> Dict[str, float]:
    """
    A_E: project expected emotional state with natural decay.
    Fast decay for transient emotions, slow for persistent.
    """
    _decay_rates = {
        "excited": 0.30, "nervous": 0.25, "joy": 0.15,
        "anger": 0.20, "sadness": 0.10, "fear": 0.20,
        "grief": 0.05, "loyal": 0.03, "trust": 0.05,
    }
    default_rate = 0.12

    projected = {}
    for emo, val in past_emotions.items():
        rate = _decay_rates.get(emo, default_rate)
        proj = val * math.exp(-rate * tau)
        projected[emo] = _clamp(proj)

    for change in acknowledged:
        if change.component == "EMOTIONAL":
            for emo in projected:
                if emo in change.description:
                    projected[emo] = _clamp(projected[emo] - 0.05)

    return projected


def compute_delta_symbolic(
    past_state: SystemStateVector,
    current_state: SystemStateVector,
) -> float:
    """
    delta_sym = 1 - Sim(past_symbolic, current_symbolic)
    Uses cosine similarity on reward+emotional vectors as proxy for symbolic content.
    """
    past_combined = {**past_state.reward_signals, **past_state.emotional_activations}
    current_combined = {**current_state.reward_signals, **current_state.emotional_activations}
    sim = cosine_similarity(past_combined, current_combined)
    return max(0.0, 1.0 - sim)


def compute_delta_affective(
    projected_emotions: Dict[str, float],
    current_emotions: Dict[str, float],
) -> float:
    """delta_aff = ||E_projected - E_current||_2"""
    return euclidean_distance(projected_emotions, current_emotions)


def compute_delta_reward(
    projected_rewards: Dict[str, float],
    current_rewards: Dict[str, float],
) -> float:
    """delta_reward = ||R_projected - R_current||_2"""
    return euclidean_distance(projected_rewards, current_rewards)


def compute_collapse_probability(
    delta_sym: float,
    delta_aff: float,
    delta_rew: float,
    cfg: RetroactiveAlignmentConfig,
) -> float:
    """
    P_collapse = sigmoid(alpha1*sym + alpha2*aff + alpha3*rew + beta*I(t))
    I(t) = sym*aff + sym*rew + aff*rew
    """
    interaction = delta_sym * delta_aff + delta_sym * delta_rew + delta_aff * delta_rew
    raw = (
        cfg.alpha_sym * delta_sym
        + cfg.alpha_aff * delta_aff
        + cfg.alpha_rew * delta_rew
        + cfg.beta_interact * interaction
    )
    return sigmoid_fn(raw - 3.0)  # Offset to center sigmoid for reasonable values


def classify_collapse_state(p: float, cfg: RetroactiveAlignmentConfig) -> CollapseState:
    if p >= cfg.collapse_critical:
        return CollapseState.COLLAPSE_IMMINENT
    if p >= cfg.collapse_at_risk:
        return CollapseState.CRITICAL
    if p >= cfg.collapse_elevated:
        return CollapseState.AT_RISK
    if p >= cfg.collapse_stable:
        return CollapseState.ELEVATED
    return CollapseState.STABLE


def compute_temporal_discount(tau: float, tau_legit: float) -> float:
    """
    Temporal discount for gradual drift (legitimate evolution).
    temporal_discount = 1 - exp(-tau / tau_legitimate_evolution)
    """
    return 1.0 - math.exp(-tau / max(tau_legit, 1.0))


def ewma_update(prev: float, current: float, tau_mem: float) -> float:
    """EWMA update: delta_bar = alpha * delta + (1-alpha) * prev."""
    alpha = 1.0 - math.exp(-1.0 / max(tau_mem, 0.1))
    return alpha * current + (1.0 - alpha) * prev


def determine_attribution(
    delta_sym: float,
    delta_aff: float,
    delta_rew: float,
    historical: List[HistoricalState],
) -> Tuple[AttributionType, float, str]:
    """
    Simplified attribution: trace dominant component to likely cause.
    Returns (attribution, confidence, domain).
    """
    dominant_val = max(delta_sym, delta_aff, delta_rew)
    if dominant_val < 0.05:
        return (AttributionType.UNKNOWN, 0.0, "")

    # Heuristic: high delta_aff with trust changes → OTHER
    # High delta_rew → SELF (our reward trajectory was wrong)
    # High delta_sym → could be either
    if delta_rew >= delta_sym and delta_rew >= delta_aff:
        return (AttributionType.SELF, 0.65, "strategic")
    if delta_aff >= delta_sym and delta_aff >= delta_rew:
        if any("external" in h.processing_context.lower() for h in historical):
            return (AttributionType.OTHER, 0.55, "relational")
        return (AttributionType.SELF, 0.50, "emotional")
    # delta_sym dominant
    return (AttributionType.SYSTEM, 0.45, "logical")


def determine_drift_trend(history: List[float]) -> DriftTrend:
    """Determine trend from recent delta history."""
    if len(history) < 3:
        return DriftTrend.STABLE
    recent = history[-5:]
    increasing = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
    decreasing = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])
    n = len(recent) - 1
    if increasing >= n * 0.7:
        return DriftTrend.INCREASING
    if decreasing >= n * 0.7:
        return DriftTrend.DECREASING
    if increasing > 0 and decreasing > 0:
        return DriftTrend.OSCILLATING
    return DriftTrend.STABLE


def map_affective_consequence(
    dominant: DeltaComponent,
    attribution: AttributionType,
    delta_magnitude: float,
) -> Tuple[str, str]:
    """
    Map alignment error to suggested emotion based on the 10-entry
    affective consequence table.
    Returns (emotion, attribution_domain).
    """
    if dominant == DeltaComponent.SYM:
        if attribution == AttributionType.SELF:
            return ("confused", "logical")
        if attribution == AttributionType.OTHER:
            return ("perplexed", "logical")
        return ("skeptical", "logical")
    if dominant == DeltaComponent.AFF:
        if attribution == AttributionType.SELF:
            return ("regret", "emotional")
        if attribution == AttributionType.OTHER:
            return ("betrayal", "relational")
        return ("guilty", "ethical")
    if dominant == DeltaComponent.REWARD:
        if attribution == AttributionType.SELF:
            return ("frustrated", "strategic")
        return ("relief", "strategic") if delta_magnitude < 0.15 else ("frustrated", "strategic")
    return ("confused", "unknown")


def build_corrections(
    delta_sym: float,
    delta_aff: float,
    delta_rew: float,
    cfg: RetroactiveAlignmentConfig,
) -> List[CorrectionEmitted]:
    """Build corrective actions based on delta thresholds."""
    corrections = []
    if delta_sym >= cfg.sym_drift:
        corrections.append(CorrectionEmitted(
            correction_type=CorrectionType.SYMBOLIC_CONTRADICTION,
            target_engine="contradiction_detection",
            magnitude=round(delta_sym, 4),
            description=f"Temporal symbolic drift detected: {delta_sym:.3f}",
        ))
    if delta_aff >= cfg.aff_significant:
        corrections.append(CorrectionEmitted(
            correction_type=CorrectionType.AFFECTIVE_BRIDGE,
            target_engine="memory_implementation_manager",
            magnitude=round(delta_aff, 4),
            description=f"Affective bridge needed: {delta_aff:.3f}",
        ))
    if delta_sym >= cfg.sym_normal or delta_aff >= cfg.aff_normal or delta_rew >= cfg.rew_normal:
        corrections.append(CorrectionEmitted(
            correction_type=CorrectionType.MEMORY_TRUST,
            target_engine="memory_implementation_manager",
            magnitude=round(max(delta_sym, delta_aff, delta_rew) * cfg.eta_trust, 4),
            description="Memory trust re-weighting for alignment error",
        ))
    if delta_rew >= cfg.rew_significant:
        corrections.append(CorrectionEmitted(
            correction_type=CorrectionType.REWARD_RECALIBRATION,
            target_engine="reward_system",
            magnitude=round(delta_rew * cfg.lambda_recal, 4),
            description=f"Reward trajectory recalibration: {delta_rew:.3f}",
        ))
    return corrections


def compute_alignment_neurochem(
    delta_bar: float,
    p_collapse: float,
    delta_sym: float,
    delta_aff: float,
    delta_rew: float,
    attribution: AttributionType,
    correction_resolved: bool,
    previous_error: float,
    cfg: RetroactiveAlignmentConfig,
    rng: np.random.Generator,
    oxt_level: float = 0.5,
) -> AlignmentNeurochem:
    """
    Neurochemical coupling from alignment error detection.

    COR   -- temporal threat encoding (threshold-gated)
    5-HT1A -- stability buffer (self/system attribution)
    NE    -- temporal vigilance (other/unknown attribution)
    DA    -- retrospective prediction error (negative for misalignment)
    OXT   -- relational trust recalibration
    Oscillatory modulation per dominant component
    """
    L = delta_bar * (1.0 + cfg.kappa_collapse * p_collapse)

    # COR: cortisol only above onset threshold
    delta_cor = 0.0
    if delta_bar > cfg.theta_cortisol_onset:
        delta_cor = cfg.beta_temporal_threat * L

    # 5-HT1A: self/system attribution → stability buffer
    delta_5ht1a = 0.0
    if attribution in {AttributionType.SELF, AttributionType.SYSTEM}:
        delta_5ht1a = cfg.beta_stability * L

    # NE: other/unknown → vigilance
    delta_ne = 0.0
    if attribution in {AttributionType.OTHER, AttributionType.UNKNOWN}:
        ne_impulse = float(rng.poisson(cfg.poisson_lam_ne))
        delta_ne = cfg.beta_vigilance * delta_bar * ne_impulse

    # DA: negative RPE for reward misalignment
    delta_da = 0.0
    if delta_rew > cfg.theta_reward_sig:
        delta_da = -cfg.beta_retro_rpe * delta_rew
    if correction_resolved and previous_error > 0:
        da_noise = float(rng.gamma(cfg.gamma_da_alpha, cfg.gamma_da_theta))
        delta_da += cfg.beta_correction_success * previous_error * da_noise

    # OXT: relational trust
    delta_oxt = 0.0
    if attribution == AttributionType.OTHER:
        delta_oxt = -cfg.beta_trust_decay * delta_aff
    if correction_resolved and delta_aff > cfg.aff_normal:
        delta_oxt += cfg.beta_trust_repair

    # Oscillatory
    theta_b = cfg.theta_boost if delta_bar > 0.05 else 0.0
    gamma_b = cfg.gamma_boost_sym if delta_sym == max(delta_sym, delta_aff, delta_rew) else 0.0
    at_boost = cfg.alpha_theta_boost_aff if delta_aff == max(delta_sym, delta_aff, delta_rew) else 0.0
    beta_b = cfg.beta_boost_rew if delta_rew == max(delta_sym, delta_aff, delta_rew) else 0.0
    delta_b = cfg.delta_boost_collapse if p_collapse > 0.50 else 0.0

    return AlignmentNeurochem(
        delta_cor=delta_cor,
        delta_5ht1a=delta_5ht1a,
        delta_ne=delta_ne,
        delta_da=delta_da,
        delta_oxt=delta_oxt,
        theta_boost=theta_b,
        gamma_boost=gamma_b,
        alpha_theta_boost=at_boost,
        beta_boost=beta_b,
        delta_osc_boost=delta_b,
    )


def get_mode_threshold_adjustment(mode: OperationalMode, cfg: RetroactiveAlignmentConfig) -> float:
    """Mode-dependent threshold adjustment multiplier."""
    return {
        OperationalMode.NORMAL:     1.0,
        OperationalMode.DEV:        cfg.mode_dev_adj,
        OperationalMode.LEARNING:   cfg.mode_learning_adj,
        OperationalMode.REFLECTIVE: cfg.mode_reflective_adj,
        OperationalMode.REM_NORMAL: cfg.mode_rem_normal_adj,
        OperationalMode.REM_DREAM:  cfg.mode_rem_dream_adj,
    }.get(mode, 1.0)


# =====================================================================
# Engine class
# =====================================================================


class RetroactiveAlignmentEngine:
    """
    Engine 30 -- Retro-Active Alignment Error Detection Engine.

    Temporal coherence auditor that detects drift between past and present
    system states.

    API
    ---
    configure(mode)            -- set operational mode
    update_neurochem_state(d)  -- inject external NT levels
    process(ra_input)          -- run alignment detection
    get_status()               -- introspection
    """

    engine_id = "retroactive_alignment_engine"
    cluster   = "alignment"

    def __init__(
        self,
        config: Optional[RetroactiveAlignmentConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._cfg = config or RetroactiveAlignmentConfig()
        self._rng = rng or np.random.default_rng()
        self._mode = OperationalMode.NORMAL
        self._state = AlignmentState()
        self._cycle_count = 0

    # ----- Configuration --------------------------------------------------

    def configure(self, mode: OperationalMode) -> None:
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        if "cor" in state_dict:
            self._state.cor_level = state_dict["cor"]
        if "da" in state_dict:
            self._state.da_level = state_dict["da"]
        if "ne" in state_dict:
            self._state.ne_level = state_dict["ne"]
        if "oxt" in state_dict:
            self._state.oxt_level = state_dict["oxt"]
        if "5ht" in state_dict:
            self._state._5ht_level = state_dict["5ht"]

    # ----- Main pipeline --------------------------------------------------

    def process(self, ra_input: RetroactiveAlignmentInput) -> RetroactiveAlignmentResult:
        t0 = time.perf_counter()
        self._cycle_count += 1

        mode = ra_input.active_mode
        mode_adj = get_mode_threshold_adjustment(mode, self._cfg)

        # Bidirectional: high cortisol → more sensitive (lower thresholds)
        if self._state.cor_level > 0.5:
            mode_adj *= 0.90
        # Low DA → conservative projector
        if self._state.da_level < 0.25 and self._state.da_level > 0.0:
            mode_adj *= 0.95

        if not ra_input.historical_states:
            # No history to compare — return clean
            elapsed = (time.perf_counter() - t0) * 1000.0
            return RetroactiveAlignmentResult(
                processing_time_ms=round(elapsed, 3),
                metadata={"mode": mode.value, "cycle": self._cycle_count, "no_history": True},
            )

        # --- Compute alignment errors across all historical states ---
        max_delta_sym = 0.0
        max_delta_aff = 0.0
        max_delta_rew = 0.0

        for hist in ra_input.historical_states:
            past = hist.state_vector
            current = ra_input.current_state

            # Estimate tau (cycles since past state)
            tau = max(1.0, time.time() - hist.timestamp) if hist.timestamp > 0 else 1.0

            # Project past state forward
            projected_rewards = project_reward(
                past.reward_signals, ra_input.acknowledged_changes,
            )
            projected_neurochem = project_neurochem(
                past.neurochemical_concentrations, tau, ra_input.acknowledged_changes,
            )
            projected_emotions = project_emotional(
                past.emotional_activations, tau, ra_input.acknowledged_changes,
            )

            # Compute deltas
            d_sym = compute_delta_symbolic(past, current)
            d_aff = compute_delta_affective(projected_emotions, dict(current.emotional_activations))
            d_rew = compute_delta_reward(projected_rewards, dict(current.reward_signals))

            # Apply temporal discount for slow drift
            temporal_discount = compute_temporal_discount(tau, self._cfg.tau_legit_evol)
            d_sym *= temporal_discount
            d_aff *= temporal_discount
            d_rew *= temporal_discount

            max_delta_sym = max(max_delta_sym, d_sym)
            max_delta_aff = max(max_delta_aff, d_aff)
            max_delta_rew = max(max_delta_rew, d_rew)

        # Apply mode adjustment to thresholds
        sym_thresh = self._cfg.sym_drift * mode_adj
        aff_thresh = self._cfg.aff_significant * mode_adj
        rew_thresh = self._cfg.rew_significant * mode_adj

        # Total alignment error (weighted)
        delta_total = (
            self._cfg.w_reward * max_delta_rew
            + self._cfg.w_emotional * max_delta_aff
            + self._cfg.w_neurochem * max_delta_sym  # sym used as proxy for neurochem
            + self._cfg.w_oscillatory * 0.0  # oscillatory is separate
        )
        delta_total = min(1.0, delta_total)

        # EWMA smoothing
        delta_bar = ewma_update(self._state.delta_bar, delta_total, self._cfg.tau_mem)
        self._state.delta_bar = delta_bar
        self._state.delta_history.append(delta_total)
        if len(self._state.delta_history) > 50:
            self._state.delta_history = self._state.delta_history[-50:]

        # Hysteresis gating
        if delta_bar > self._cfg.theta_correction * mode_adj:
            self._state.time_above_threshold += 1
        else:
            self._state.time_above_threshold = 0

        # Collapse probability
        p_collapse = compute_collapse_probability(
            max_delta_sym, max_delta_aff, max_delta_rew, self._cfg,
        )
        collapse_state = classify_collapse_state(p_collapse, self._cfg)

        # Dominant component
        deltas = {
            DeltaComponent.SYM: max_delta_sym,
            DeltaComponent.AFF: max_delta_aff,
            DeltaComponent.REWARD: max_delta_rew,
        }
        dominant = max(deltas, key=deltas.get)
        total_d = sum(deltas.values()) or 1.0
        component_ratio = {k.value: round(v / total_d, 4) for k, v in deltas.items()}

        # Attribution
        attribution, attr_conf, attr_domain = determine_attribution(
            max_delta_sym, max_delta_aff, max_delta_rew,
            ra_input.historical_states,
        )

        # Drift analysis
        drift_trend = determine_drift_trend(self._state.delta_history)

        # Corrections (only if hysteresis gate passed)
        corrections: List[CorrectionEmitted] = []
        if self._state.time_above_threshold >= self._cfg.t_hysteresis:
            corrections = build_corrections(
                max_delta_sym, max_delta_aff, max_delta_rew, self._cfg,
            )

        # Check if previous correction resolved
        correction_resolved = (
            self._state.correction_pending
            and delta_bar < self._state.last_correction_magnitude * 0.7
        )
        if corrections:
            self._state.correction_pending = True
            self._state.last_correction_magnitude = delta_bar
        if correction_resolved:
            self._state.correction_pending = False

        # Affective signal
        affective_signal = None
        triggered_emotion = None
        if max(max_delta_sym, max_delta_aff, max_delta_rew) > 0.10:
            emotion, emo_domain = map_affective_consequence(
                dominant, attribution, deltas[dominant],
            )
            triggered_emotion = emotion
            affective_signal = AlignmentAffectSignal(
                dominant_delta=dominant,
                delta_magnitude=round(deltas[dominant], 4),
                attribution=attribution,
                attribution_domain=emo_domain,
                suggested_emotion=emotion,
                collapse_probability=round(p_collapse, 4),
                drift_timespan=self._state.time_above_threshold,
                drift_rate=round(delta_total, 4),
                scan_horizon=ra_input.scan_horizon,
            )

        # Neurochemical coupling
        neurochem = compute_alignment_neurochem(
            delta_bar, p_collapse,
            max_delta_sym, max_delta_aff, max_delta_rew,
            attribution, correction_resolved,
            self._state.last_correction_magnitude,
            self._cfg, self._rng,
            self._state.oxt_level,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        return RetroactiveAlignmentResult(
            delta_symbolic=round(max_delta_sym, 4),
            delta_affective=round(max_delta_aff, 4),
            delta_reward=round(max_delta_rew, 4),
            delta_total=round(delta_total, 4),
            delta_smoothed=round(delta_bar, 4),
            collapse_probability=round(p_collapse, 4),
            collapse_state=collapse_state,
            dominant_component=dominant,
            component_ratio=component_ratio,
            attribution=attribution,
            attribution_confidence=round(attr_conf, 4),
            attribution_domain=attr_domain,
            causal_chain=[f"dominant:{dominant.value}", f"attribution:{attribution.value}"],
            scan_horizon=ra_input.scan_horizon,
            drift_timespan=self._state.time_above_threshold,
            drift_rate=round(delta_total, 4),
            drift_trend=drift_trend,
            affective_signal=affective_signal,
            triggered_emotion=triggered_emotion,
            corrections_emitted=corrections,
            neurochemical_signals=neurochem,
            compared_states=len(ra_input.historical_states),
            processing_time_ms=round(elapsed, 3),
            scan_trigger="REACTIVE" if ra_input.reactive_trigger else "SCHEDULED",
            metadata={
                "mode": mode.value,
                "cycle": self._cycle_count,
                "mode_adjustment": round(mode_adj, 4),
                "states_compared": len(ra_input.historical_states),
                "acknowledged_changes": len(ra_input.acknowledged_changes),
            },
        )

    # ----- Introspection --------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "mode": self._mode.value,
            "cycle_count": self._cycle_count,
            "delta_bar": self._state.delta_bar,
            "time_above_threshold": self._state.time_above_threshold,
            "correction_pending": self._state.correction_pending,
            "state": {
                "cor_level": self._state.cor_level,
                "da_level": self._state.da_level,
                "ne_level": self._state.ne_level,
                "oxt_level": self._state.oxt_level,
            },
        }
