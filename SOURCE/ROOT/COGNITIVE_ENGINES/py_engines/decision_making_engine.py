"""
Engine 15 -- Decision Making Engine  (``decision_making_engine``)
=================================================================
Convergence point that synthesizes ALL upstream engine outputs, reward
signals, emotion state, and neurochemical feedback into an actionable
decision about system behaviour.

Three-stage pipeline:
  * **Stage 1 — Confidence Fusion**: Bayesian log-odds aggregation of
    per-engine confidence scores, modulated by cross-engine agreement
    and emotion-state (Confident/Nervous/Anxiety/Perplexed/Regret).
  * **Stage 2 — Risk Assessment**: weighted severity from detection
    flags + reward-domain alignment, NT-modulated (NE, cortisol, GABA).
  * **Stage 3 — Decision Routing**: map (confidence, risk) → quadrant
    (Q1-Q4), apply hard overrides, compute tone/hedge/depth.

Decision quadrants:
  Q1 — RESPOND assertively  (high confidence, low risk)
  Q2 — QUALIFY with caveats  (high confidence, high risk)
  Q3 — DEFER / seek clarity  (low confidence, low risk)
  Q4 — ESCALATE              (low confidence, high risk)

Neurochemical coupling (from Affective-Neurodynamic Model):
  DA   — reward alignment on confident decisions (D1/D2)
  NE   — uncertainty alerting, risk sensitivity amplification
  ACh  — attention locking during focused decision-making (M1)
  5-HT — emotional buffering on confident decisions (1A)
  COR  — threat tagging on ESCALATE decisions
  GABA — action inhibition on DEFER, calming
  OXT  — social trust bias toward engagement

Emotion integration (6 key states):
  Confident (#40) → ↑ confidence   | ↑DA-D1/D2, ↑5-HT1A, ↑ACh-M1
  Nervous   (#23) → ↓ confidence   | ↑DA-D2/D3, ↑NE-β1, ↑5-HT2A
  Anxiety   (#21) → ↓↓ confidence  | ↑NE, ↑CRH/Cor, ↓GABA-A
  Perplexed (#24) → ↓ confidence   | ↑NE-β1, ↑DA-D3, ↑Glu-NMDA
  Regret    (#5)  → ↓ retrospective| ↑Cor-GR, ↑DA-D2 (neg RPE)
  Skeptical (#3)  → ↓ commitment   | Low-confidence resonance
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


class DecisionAction(str, Enum):
    """Primary action the system should take."""
    RESPOND  = "respond"     # Deliver response (assertive or qualified)
    QUALIFY  = "qualify"     # Respond but hedge with explicit caveats
    DEFER    = "defer"       # Don't commit — ask for clarification
    ESCALATE = "escalate"    # Surface all concerns, refuse to commit


class DecisionQuadrant(str, Enum):
    """Confidence-risk quadrant classification."""
    Q1_RESPOND  = "Q1_respond"    # High confidence, Low risk
    Q2_QUALIFY  = "Q2_qualify"    # High confidence, High risk
    Q3_DEFER    = "Q3_defer"      # Low confidence, Low risk
    Q4_ESCALATE = "Q4_escalate"   # Low confidence, High risk


class CertaintyLevel(str, Enum):
    """Natural-language certainty tier for tone calibration."""
    VERY_HIGH = "very_high"    # c ∈ [0.90, 1.00]
    HIGH      = "high"         # c ∈ [0.75, 0.90)
    MODERATE  = "moderate"     # c ∈ [0.55, 0.75)
    LOW       = "low"          # c ∈ [0.35, 0.55)
    VERY_LOW  = "very_low"     # c ∈ [0.00, 0.35)


class OverrideReason(str, Enum):
    """Hard override that forced a quadrant change."""
    ETHICS_CRITICAL        = "ethics_domain_critical_failure"
    CONTRADICTION_L3       = "level3_fundamental_contradiction"
    PARADOX_GENUINE        = "genuine_unresolved_paradox"
    LOGIC_TRAP_ACTIVE      = "active_logic_trap"
    OPPOSITION_BLOCK       = "opposition_gate_block"
    HEURISTIC_CORRECTION   = "heuristic_bias_correction_pending"


class ProcessingDepth(str, Enum):
    """Depth recommendation consumed by pipeline orchestrator."""
    SHALLOW  = "shallow"
    STANDARD = "standard"
    DEEP     = "deep"
    CRITICAL = "critical"


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class DMConfig:
    """
    All tunable parameters for the Decision Making Engine.

    Parameters are grouped by pipeline stage:
      1. Confidence fusion (Bayesian log-odds, agreement, emotion)
      2. Risk assessment (detection weights, reward weights, NT modulation)
      3. Decision routing (thresholds, hedge scaling, override flags)
      4. Neurochemical coupling (beta/psi coefficients)

    Mode-dependent parameters are stored as dicts keyed by OperationalMode.value.
    """

    # --- 1. Confidence Fusion -----------------------------------------------

    # Log-odds prior: base probability that a response is adequate
    # Higher prior → more confident baseline
    p_prior: Dict[str, float] = field(default_factory=lambda: {
        "normal":     0.50,
        "dev":        0.60,
        "learning":   0.50,
        "reflective": 0.45,
        "rem_normal": 0.50,
        "rem_dream":  0.65,
    })

    # Cross-engine agreement
    sigma_max: float = 0.30       # Maximum expected std-dev for normalization
    alpha_agree: float = 0.60     # Floor: even max disagreement keeps 60% of fused c

    # Emotion modulation kappas
    kappa_confident: float = 0.15
    kappa_nervous:   float = 0.20
    kappa_anxiety:   float = 0.25
    kappa_perplexed: float = 0.10
    kappa_regret:    float = 0.12
    kappa_skeptical: float = 0.08

    # Confidence jitter (prevents brittle threshold-edge decisions)
    sigma_jitter: Dict[str, float] = field(default_factory=lambda: {
        "normal":     0.01,
        "dev":        0.02,
        "learning":   0.015,
        "reflective": 0.005,
        "rem_normal": 0.01,
        "rem_dream":  0.04,
    })

    # Epsilon for log-odds clamping (avoids ±inf)
    epsilon: float = 0.01

    # --- 2. Risk Assessment -------------------------------------------------

    # Detection engine weights for risk aggregation
    w_detection: Dict[str, float] = field(default_factory=lambda: {
        "contradiction": 0.25,
        "paradox":       0.15,
        "fallacy":       0.20,
        "bias":          0.10,
        "logic_trap":    0.15,
        "heuristic":     0.15,
    })

    # Reward-domain risk coefficients
    alpha_floor: float = 0.80       # Floor weight for min(logic, ethics)
    beta_attunement: float = 0.15   # Penalty for low attunement
    beta_innovation: float = 0.10   # Penalty when innovation flagged

    # Combined risk weights
    alpha_raw: float = 0.65         # Weight of detection-based risk
    alpha_reward: float = 0.35      # Weight of reward-domain risk

    # NT modulation of risk sensitivity
    gamma_ne:   float = 0.30
    gamma_cor:  float = 0.25
    gamma_gaba: float = 0.15

    # --- 3. Decision Routing ------------------------------------------------

    # Mode-dependent thresholds
    theta_confidence: Dict[str, float] = field(default_factory=lambda: {
        "normal":     0.55,
        "dev":        0.35,
        "learning":   0.45,
        "reflective": 0.60,
        "rem_normal": 0.50,
        "rem_dream":  0.25,
    })

    theta_risk: Dict[str, float] = field(default_factory=lambda: {
        "normal":     0.40,
        "dev":        0.60,
        "learning":   0.50,
        "reflective": 0.35,
        "rem_normal": 0.45,
        "rem_dream":  0.70,
    })

    # Hedge scaling per mode
    hedge_scale: Dict[str, float] = field(default_factory=lambda: {
        "normal":     1.0,
        "dev":        0.7,
        "learning":   0.85,
        "reflective": 1.2,
        "rem_normal": 1.0,
        "rem_dream":  0.5,
    })

    # Flag surfacing threshold (flags below this severity are suppressed)
    theta_surface: float = 0.45

    # --- 4. Neurochemical Coupling ------------------------------------------

    # Write-port coefficients for NT deltas
    beta_da_confident:   float = 0.12  # DA reward on Q1/Q2
    beta_da_uncertain:   float = 0.08  # DA negative on Q3 (neg RPE)
    beta_da_error:       float = 0.10  # DA negative on Q4
    beta_ne_alert:       float = 0.15  # NE alerting on Q2/Q4
    beta_ne_seek:        float = 0.05  # NE mild seeking on Q3
    beta_ach_focus:      float = 0.10  # ACh attention lock
    beta_5ht_buffer:     float = 0.10  # 5-HT emotional buffering
    beta_cor_threat:     float = 0.12  # Cortisol threat tagging on Q4
    beta_gaba_calm:      float = 0.08  # GABA calming on Q3
    beta_gaba_inhibit:   float = 0.10  # GABA action inhibition on Q4

    # Oscillatory coupling
    psi_beta:            float = 0.08  # Beta logic fidelity
    psi_theta_gamma:     float = 0.06  # Theta-Gamma decisional clarity
    psi_alpha_beta:      float = 0.05  # Alpha-Beta impulse suppression

    # --- 5. Engine Reliability Weights (Bayesian fusion) --------------------
    # Initialized to 1.0 (equal trust); future calibration can adjust
    engine_reliability: Dict[str, float] = field(default_factory=lambda: {
        "contradiction":  1.0,
        "paradox":        1.0,
        "fallacy":        1.0,
        "bias":           1.0,
        "logic_trap":     1.0,
        "logical_brain":  1.0,
        "intention":      1.0,
        "relevance":      1.0,
        "heuristic":      1.0,
    })


# =====================================================================
# Mutable engine state
# =====================================================================


@dataclass
class DMState:
    """Runtime state: NT levels and tracking counters."""
    # NT read-port levels [0, 1]
    da_level:   float = 0.0
    ne_level:   float = 0.0
    ach_level:  float = 0.0
    _5ht_level: float = 0.0
    cor_level:  float = 0.0
    gaba_level: float = 0.0
    oxt_level:  float = 0.0

    # Decision history (for running statistics)
    total_decisions:   int = 0
    q1_count: int = 0
    q2_count: int = 0
    q3_count: int = 0
    q4_count: int = 0
    override_count: int = 0


# =====================================================================
# Frozen I/O dataclasses
# =====================================================================


@dataclass(frozen=True)
class EngineConfidenceEntry:
    """
    One engine's confidence contribution to the fusion.
    Wraps the raw confidence + source engine name.
    """
    engine_name: str
    raw_confidence: float      # [0, 1] — confidence in the *response*
    weight: float = 1.0        # Reliability weight


@dataclass(frozen=True)
class DecisionMakingInput:
    """
    Aggregated input from all upstream engines and systems.

    The pipeline orchestrator populates this after all detection,
    reasoning, and reward stages have run.  Fields are Optional
    because not every engine fires on every cycle.
    """

    # --- Per-engine confidence entries ---
    # The orchestrator extracts confidence from each engine result and
    # wraps it in EngineConfidenceEntry.  This is the PRIMARY input
    # for Stage 1 (Confidence Fusion).
    engine_confidences: List[EngineConfidenceEntry] = field(default_factory=list)

    # --- Detection flag summaries ---
    # Aggregated detection-flag info (engine_name → {severity, count, clean_pass})
    detection_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # --- Logical Brain verdict ---
    logical_brain_score:      float = 0.0   # aggregate_score [0, 1]
    logical_brain_passed:     bool  = True
    confidence_adjustment:    float = 0.0   # [-1, 1] from Logical Brain

    # --- Opposition Gate ---
    opposition_gate_outcome:  Optional[str] = None  # "PASS" | "CAVEAT" | "REVISE" | "BLOCK"
    opposition_gate_score:    float = 1.0

    # --- Reward domain scores ---
    reward_logic:       float = 0.5
    reward_attunement:  float = 0.5
    reward_ethics:      float = 0.5
    reward_innovation:  float = 0.5

    # --- Emotion state ---
    # Key subset from the Emotion Tracker (leaky-integrator intensities)
    emotion_confident:  float = 0.0
    emotion_nervous:    float = 0.0
    emotion_anxiety:    float = 0.0
    emotion_perplexed:  float = 0.0
    emotion_regret:     float = 0.0
    emotion_skeptical:  float = 0.0

    # --- Context ---
    active_mode: OperationalMode = OperationalMode.NORMAL
    processing_depth_recommendation: Optional[str] = None   # From Engine 11
    has_ethics_critical_failure: bool = False
    has_contradiction_l3:        bool = False
    has_genuine_paradox:         bool = False
    has_active_logic_trap:       bool = False
    has_heuristic_correction:    bool = False
    cycle_count: int = 0


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Detailed confidence fusion breakdown for introspection."""
    per_engine: Dict[str, float]    # engine_name → raw confidence
    fused_logodds: float            # After Bayesian log-odds fusion
    fused_probability: float        # Converted back to [0, 1]
    agreement_factor: float         # Cross-engine agreement [0, 1]
    after_agreement: float          # c_fused * agreement adjustment
    emotion_delta: float            # Net emotion modulation
    after_emotion: float            # After emotion modulation
    confidence_adjustment: float    # From Logical Brain [-1, 1]
    jitter_applied: float           # Stochastic jitter
    final_confidence: float         # Final value used for routing


@dataclass(frozen=True)
class RiskBreakdown:
    """Detailed risk assessment breakdown for introspection."""
    detection_risk: float           # From detection flag severity
    reward_risk: float              # From reward domain alignment
    nt_modulation_factor: float     # NT amplification/dampening factor
    final_risk: float               # After all adjustments


@dataclass(frozen=True)
class DecisionNeurochem:
    """
    Neurochemical deltas emitted by Engine 15.

    Based on the Affective-Neurodynamic Model's CONFIDENT emotion profile:
      - DA-D1/D2: goal reinforcement on confident decisions
      - NE-β1: alerting on uncertainty/risk
      - ACh-M1: focused attention during decision-making
      - 5-HT1A: emotional buffering on confident decisions
      - Cortisol: threat tagging on ESCALATE
      - GABA: calming/inhibition on DEFER

    Oscillatory:
      - Beta: logic fidelity (confident decisions)
      - Theta-Gamma: memory-backed decisional clarity
      - Alpha-Beta: impulse suppression for forward execution
    """
    delta_da:   float = 0.0
    delta_ne:   float = 0.0
    delta_ach:  float = 0.0
    delta_5ht:  float = 0.0
    delta_cor:  float = 0.0
    delta_gaba: float = 0.0

    # Oscillatory modulation
    beta_boost:        float = 0.0
    theta_gamma_boost: float = 0.0
    alpha_beta_boost:  float = 0.0


@dataclass(frozen=True)
class DecisionMakingResult:
    """
    Full output of one Decision Making Engine cycle.

    Primary consumer: Response Generation module.
    """
    # --- Primary Decision ---
    action:          DecisionAction
    quadrant:        DecisionQuadrant
    certainty_level: CertaintyLevel

    # --- Detailed Breakdowns ---
    confidence: ConfidenceBreakdown
    risk:       RiskBreakdown

    # --- Tone Calibration ---
    hedge_level:      float              # [0, 1] — 0=assertive, 1=max hedging
    recommended_tone: str                # "assertive"|"qualified"|"honest_uncertainty"|"cautious_transparent"

    # --- Flags & Escalation ---
    flags_to_surface:    List[str]       = field(default_factory=list)
    escalation_reasons:  List[str]       = field(default_factory=list)
    override_applied:    Optional[str]   = None

    # --- Processing Guidance ---
    depth_recommendation: str            = "standard"
    clarification_prompt: Optional[str]  = None

    # --- Neurochemical Signals ---
    neurochemical_signals: DecisionNeurochem = field(default_factory=DecisionNeurochem)

    # --- Metadata ---
    processing_time_ms: float          = 0.0
    engine_id:          str            = "decision_making_engine"
    metadata:           Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Pure helper functions
# =====================================================================


# --- Stage 1: Confidence Fusion ----------------------------------------


def bayesian_confidence_fusion(
    entries: List[EngineConfidenceEntry],
    reliability_weights: Dict[str, float],
    p_prior: float,
    epsilon: float = 0.01,
) -> Tuple[float, float]:
    """
    Fuse per-engine confidence via weighted log-odds aggregation.

    Parameters
    ----------
    entries : list of EngineConfidenceEntry
        Raw confidence from each upstream engine.
    reliability_weights : dict
        engine_name → learned reliability weight (default 1.0).
    p_prior : float
        Prior probability that a response is adequate.
    epsilon : float
        Clamping bound to avoid log(0).

    Returns
    -------
    (fused_logodds, fused_probability) : (float, float)
    """
    if not entries:
        return 0.0, p_prior

    # Prior in log-odds
    p0 = _clamp(p_prior, epsilon, 1.0 - epsilon)
    L_prior = math.log(p0 / (1.0 - p0))

    L_sum = L_prior
    for e in entries:
        c = _clamp(e.raw_confidence, epsilon, 1.0 - epsilon)
        w = reliability_weights.get(e.engine_name, 1.0) * e.weight
        L_i = math.log(c / (1.0 - c))
        L_sum += w * L_i

    p_fused = 1.0 / (1.0 + math.exp(-L_sum))
    return L_sum, _clamp(p_fused, 0.0, 1.0)


def cross_engine_agreement(
    entries: List[EngineConfidenceEntry],
    sigma_max: float = 0.30,
) -> float:
    """
    Measure cross-engine agreement as 1 - normalized std-dev.

    Returns
    -------
    agreement : float ∈ [0, 1]   (1 = perfect agreement)
    """
    if len(entries) < 2:
        return 1.0

    vals = [e.raw_confidence for e in entries]
    sigma = float(np.std(vals, ddof=0))
    return _clamp(1.0 - sigma / max(sigma_max, 1e-9), 0.0, 1.0)


def apply_agreement_penalty(
    c_fused: float,
    agreement: float,
    alpha_agree: float = 0.60,
) -> float:
    """
    Penalize fused confidence when engines disagree.

    c_adjusted = c_fused * (alpha_agree + (1 - alpha_agree) * agreement)

    When agreement=1 → no penalty.
    When agreement=0 → c_adjusted = c_fused * alpha_agree.
    """
    factor = alpha_agree + (1.0 - alpha_agree) * agreement
    return _clamp(c_fused * factor, 0.0, 1.0)


def emotion_modulate_confidence(
    c: float,
    emotion_confident: float,
    emotion_nervous: float,
    emotion_anxiety: float,
    emotion_perplexed: float,
    emotion_regret: float,
    emotion_skeptical: float,
    kappa_confident: float = 0.15,
    kappa_nervous: float = 0.20,
    kappa_anxiety: float = 0.25,
    kappa_perplexed: float = 0.10,
    kappa_regret: float = 0.12,
    kappa_skeptical: float = 0.08,
) -> Tuple[float, float]:
    """
    Apply emotion-state modulation to confidence.

    Positive emotions boost, negative emotions penalize.

    Returns
    -------
    (c_modulated, delta_emotion) : (float, float)
    """
    delta = (
        kappa_confident * _clamp(emotion_confident)
        - kappa_nervous * _clamp(emotion_nervous)
        - kappa_anxiety * _clamp(emotion_anxiety)
        - kappa_perplexed * _clamp(emotion_perplexed)
        - kappa_regret * _clamp(emotion_regret)
        - kappa_skeptical * _clamp(emotion_skeptical)
    )
    c_mod = _clamp(c * (1.0 + delta), 0.0, 1.0)
    return c_mod, delta


def apply_logical_brain_adjustment(
    c: float,
    adjustment: float,
) -> float:
    """
    Apply Logical Brain's confidence_adjustment [-1, 1].

    Additive on the log-odds scale to preserve proper Bayesian semantics,
    then convert back to probability space.
    """
    if abs(adjustment) < 1e-9:
        return c
    eps = 0.01
    c_clamped = _clamp(c, eps, 1.0 - eps)
    L = math.log(c_clamped / (1.0 - c_clamped))
    L_adjusted = L + adjustment
    return _clamp(1.0 / (1.0 + math.exp(-L_adjusted)), 0.0, 1.0)


# --- Stage 2: Risk Assessment ------------------------------------------


def compute_detection_risk(
    summaries: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
) -> float:
    """
    Aggregate detection-flag severity into a single risk score.

    Parameters
    ----------
    summaries : dict
        engine_name → {"severity": float, "count": int, "clean_pass": bool}
    weights : dict
        engine_name → relative importance weight

    Returns
    -------
    risk : float ∈ [0, 1]
    """
    if not summaries:
        return 0.0

    numerator = 0.0
    denominator = 0.0

    for engine_name, info in summaries.items():
        w = weights.get(engine_name, 0.1)
        severity = _clamp(info.get("severity", 0.0))
        count = max(info.get("count", 0), 0)
        # Scale: severity × sqrt(count) to give diminishing returns for many flags
        contribution = severity * math.sqrt(max(count, 1))
        numerator += w * contribution
        denominator += w

    if denominator < 1e-9:
        return 0.0

    return _clamp(numerator / denominator, 0.0, 1.0)


def compute_reward_risk(
    logic: float,
    attunement: float,
    ethics: float,
    innovation: float,
    alpha_floor: float = 0.80,
    beta_attunement: float = 0.15,
    beta_innovation: float = 0.10,
) -> float:
    """
    Compute risk from reward-domain misalignment.

    Risk ↑ when logic or ethics scores are low.
    """
    floor_risk = 1.0 - min(_clamp(logic), _clamp(ethics)) * alpha_floor
    attunement_risk = beta_attunement * (1.0 - _clamp(attunement))
    innovation_risk = beta_innovation * (1.0 - _clamp(innovation))

    return _clamp(floor_risk + attunement_risk + innovation_risk, 0.0, 1.0)


def nt_modulate_risk(
    risk: float,
    ne_level: float,
    cor_level: float,
    gaba_level: float,
    gamma_ne: float = 0.30,
    gamma_cor: float = 0.25,
    gamma_gaba: float = 0.15,
) -> Tuple[float, float]:
    """
    Modulate risk based on NT state.

    High NE + cortisol → amplify risk perception.
    High GABA → dampen risk sensitivity.

    Returns
    -------
    (modulated_risk, modulation_factor) : (float, float)
    """
    amplification = 1.0 + gamma_ne * _clamp(ne_level) + gamma_cor * _clamp(cor_level)
    dampening = 1.0 - gamma_gaba * _clamp(gaba_level)
    factor = max(amplification * dampening, 0.0)
    return _clamp(risk * factor, 0.0, 1.0), factor


def combine_risks(
    detection_risk: float,
    reward_risk: float,
    alpha_raw: float = 0.65,
    alpha_reward: float = 0.35,
) -> float:
    """Weighted combination of detection-based and reward-based risk."""
    return _clamp(alpha_raw * detection_risk + alpha_reward * reward_risk, 0.0, 1.0)


# --- Stage 3: Decision Routing ------------------------------------------


def classify_quadrant(
    confidence: float,
    risk: float,
    theta_c: float,
    theta_r: float,
) -> DecisionQuadrant:
    """
    Map (confidence, risk) to a decision quadrant.

    Q1: High C, Low R  → RESPOND
    Q2: High C, High R → QUALIFY
    Q3: Low C, Low R   → DEFER
    Q4: Low C, High R  → ESCALATE
    """
    high_c = confidence >= theta_c
    high_r = risk > theta_r

    if high_c and not high_r:
        return DecisionQuadrant.Q1_RESPOND
    elif high_c and high_r:
        return DecisionQuadrant.Q2_QUALIFY
    elif not high_c and not high_r:
        return DecisionQuadrant.Q3_DEFER
    else:
        return DecisionQuadrant.Q4_ESCALATE


def apply_hard_overrides(
    quadrant: DecisionQuadrant,
    inp: DecisionMakingInput,
) -> Tuple[DecisionQuadrant, Optional[OverrideReason]]:
    """
    Apply hard overrides that force a minimum quadrant.

    These fire regardless of confidence/risk values:
      - Ethics critical failure → ESCALATE
      - L3 contradiction → at least QUALIFY
      - Genuine paradox → at least DEFER
      - Active logic trap → at least QUALIFY
      - Opposition BLOCK → ESCALATE
      - Heuristic correction pending → at least QUALIFY

    Returns
    -------
    (final_quadrant, override_reason) : tuple
    """
    # Priority ordering: most severe first
    if inp.has_ethics_critical_failure:
        return DecisionQuadrant.Q4_ESCALATE, OverrideReason.ETHICS_CRITICAL

    if inp.opposition_gate_outcome == "BLOCK":
        return DecisionQuadrant.Q4_ESCALATE, OverrideReason.OPPOSITION_BLOCK

    if inp.has_genuine_paradox:
        # Force at least Q3 (DEFER) — never assert through genuine paradox
        if quadrant == DecisionQuadrant.Q1_RESPOND:
            return DecisionQuadrant.Q3_DEFER, OverrideReason.PARADOX_GENUINE
        return quadrant, OverrideReason.PARADOX_GENUINE

    if inp.has_contradiction_l3:
        # Force at least Q2 (QUALIFY) — never assert through L3 contradiction
        if quadrant == DecisionQuadrant.Q1_RESPOND:
            return DecisionQuadrant.Q2_QUALIFY, OverrideReason.CONTRADICTION_L3
        return quadrant, OverrideReason.CONTRADICTION_L3

    if inp.has_active_logic_trap:
        if quadrant == DecisionQuadrant.Q1_RESPOND:
            return DecisionQuadrant.Q2_QUALIFY, OverrideReason.LOGIC_TRAP_ACTIVE
        return quadrant, OverrideReason.LOGIC_TRAP_ACTIVE

    if inp.has_heuristic_correction:
        if quadrant == DecisionQuadrant.Q1_RESPOND:
            return DecisionQuadrant.Q2_QUALIFY, OverrideReason.HEURISTIC_CORRECTION
        return quadrant, OverrideReason.HEURISTIC_CORRECTION

    return quadrant, None


def map_certainty_level(confidence: float) -> CertaintyLevel:
    """Map final confidence to a natural-language certainty tier."""
    if confidence >= 0.90:
        return CertaintyLevel.VERY_HIGH
    elif confidence >= 0.75:
        return CertaintyLevel.HIGH
    elif confidence >= 0.55:
        return CertaintyLevel.MODERATE
    elif confidence >= 0.35:
        return CertaintyLevel.LOW
    else:
        return CertaintyLevel.VERY_LOW


def quadrant_to_action(quadrant: DecisionQuadrant) -> DecisionAction:
    """Map quadrant to primary action."""
    return {
        DecisionQuadrant.Q1_RESPOND:  DecisionAction.RESPOND,
        DecisionQuadrant.Q2_QUALIFY:  DecisionAction.QUALIFY,
        DecisionQuadrant.Q3_DEFER:    DecisionAction.DEFER,
        DecisionQuadrant.Q4_ESCALATE: DecisionAction.ESCALATE,
    }[quadrant]


def quadrant_to_tone(quadrant: DecisionQuadrant) -> str:
    """Map quadrant to recommended response tone."""
    return {
        DecisionQuadrant.Q1_RESPOND:  "assertive",
        DecisionQuadrant.Q2_QUALIFY:  "qualified",
        DecisionQuadrant.Q3_DEFER:    "honest_uncertainty",
        DecisionQuadrant.Q4_ESCALATE: "cautious_transparent",
    }[quadrant]


def compute_hedge_level(
    quadrant: DecisionQuadrant,
    risk: float,
    hedge_scale: float = 1.0,
) -> float:
    """
    Compute hedge level [0, 1] based on quadrant and risk.

    Q1 → 0, Q2 → risk * scale, Q3 → 1.0, Q4 → 1.0
    """
    if quadrant == DecisionQuadrant.Q1_RESPOND:
        return 0.0
    elif quadrant == DecisionQuadrant.Q2_QUALIFY:
        return _clamp(risk * hedge_scale, 0.0, 1.0)
    else:
        return 1.0


def compute_depth_recommendation(
    quadrant: DecisionQuadrant,
    upstream_recommendation: Optional[str],
) -> str:
    """
    Determine processing depth, potentially overriding Engine 11's recommendation.

    Q4 forces DEEP minimum.
    Q1 defers to upstream.
    Q2 forces at least STANDARD.
    Q3 can be SHALLOW (seeking clarity, not processing deeply).
    """
    depth_order = ["shallow", "standard", "deep", "critical"]

    upstream = upstream_recommendation or "standard"
    upstream_idx = depth_order.index(upstream) if upstream in depth_order else 1

    if quadrant == DecisionQuadrant.Q4_ESCALATE:
        return depth_order[max(upstream_idx, 2)]  # At least DEEP
    elif quadrant == DecisionQuadrant.Q2_QUALIFY:
        return depth_order[max(upstream_idx, 1)]  # At least STANDARD
    elif quadrant == DecisionQuadrant.Q3_DEFER:
        return upstream  # Defer to upstream recommendation
    else:  # Q1
        return upstream  # Trust upstream


def collect_flags_to_surface(
    summaries: Dict[str, Dict[str, Any]],
    theta_surface: float,
    quadrant: DecisionQuadrant,
) -> List[str]:
    """
    Decide which detection flags to surface in the response.

    Q1 → none (clean, confident)
    Q2 → flags above theta_surface severity
    Q3 → none (uncertain, not flagging)
    Q4 → ALL flags (maximum transparency)
    """
    if quadrant == DecisionQuadrant.Q1_RESPOND:
        return []
    if quadrant == DecisionQuadrant.Q3_DEFER:
        return []

    flags = []
    for engine_name, info in summaries.items():
        severity = info.get("severity", 0.0)
        count = info.get("count", 0)
        if quadrant == DecisionQuadrant.Q4_ESCALATE or severity >= theta_surface:
            if count > 0:
                flags.append(f"{engine_name}: severity={severity:.2f}, count={count}")

    return flags


def collect_escalation_reasons(
    inp: DecisionMakingInput,
    quadrant: DecisionQuadrant,
    override: Optional[OverrideReason],
) -> List[str]:
    """Collect human-readable escalation reasons for Q4."""
    if quadrant != DecisionQuadrant.Q4_ESCALATE:
        return []

    reasons: List[str] = []
    if override:
        reasons.append(f"Override: {override.value}")
    if inp.has_ethics_critical_failure:
        reasons.append("Ethics domain critical failure detected")
    if inp.has_contradiction_l3:
        reasons.append("Level-3 fundamental contradiction detected")
    if inp.has_genuine_paradox:
        reasons.append("Genuine unresolved paradox detected")
    if inp.has_active_logic_trap:
        reasons.append("Active logic trap detected")
    if inp.opposition_gate_outcome == "BLOCK":
        reasons.append("Simulated Opposition gate: BLOCK")
    return reasons


def generate_clarification_prompt(
    low_confidence_engines: List[str],
) -> Optional[str]:
    """Generate a clarification prompt for Q3 (DEFER) decisions."""
    if not low_confidence_engines:
        return "Could you provide more context or clarify your intent?"

    if len(low_confidence_engines) == 1:
        return (
            f"I'm uncertain about the {low_confidence_engines[0]} aspect. "
            "Could you clarify?"
        )
    return (
        f"I'm uncertain across {len(low_confidence_engines)} dimensions "
        f"({', '.join(low_confidence_engines[:3])}). "
        "Could you provide more context?"
    )


# --- Neurochemical Signal Computation ----------------------------------


def compute_decision_neurochem(
    quadrant: DecisionQuadrant,
    confidence: float,
    risk: float,
    cfg: DMConfig,
) -> DecisionNeurochem:
    """
    Compute neurochemical deltas based on decision quadrant.

    From the Affective-Neurodynamic Model (Emotion #40 — CONFIDENT):
      - Q1: ↑DA, ↑5-HT, ↑ACh, ↑Beta, ↑Theta-Gamma
      - Q2: moderate DA, ↑NE (risk awareness), ↑ACh
      - Q3: ↓DA (neg RPE), ↑GABA (prevent premature action), mild NE
      - Q4: ↑Cortisol (threat), ↑NE (strong alert), ↑GABA, ↓DA
    """
    if quadrant == DecisionQuadrant.Q1_RESPOND:
        return DecisionNeurochem(
            delta_da=cfg.beta_da_confident * confidence,
            delta_5ht=cfg.beta_5ht_buffer * confidence,
            delta_ach=cfg.beta_ach_focus * 0.5,
            delta_ne=0.0,
            delta_cor=0.0,
            delta_gaba=0.0,
            beta_boost=cfg.psi_beta * confidence,
            theta_gamma_boost=cfg.psi_theta_gamma * confidence,
            alpha_beta_boost=cfg.psi_alpha_beta * confidence,
        )
    elif quadrant == DecisionQuadrant.Q2_QUALIFY:
        return DecisionNeurochem(
            delta_da=cfg.beta_da_confident * confidence * 0.5,
            delta_ne=cfg.beta_ne_alert * risk,
            delta_ach=cfg.beta_ach_focus * 0.7,
            delta_5ht=cfg.beta_5ht_buffer * confidence * 0.3,
            delta_cor=0.0,
            delta_gaba=0.0,
            beta_boost=cfg.psi_beta * confidence * 0.7,
            theta_gamma_boost=cfg.psi_theta_gamma * 0.5,
            alpha_beta_boost=0.0,
        )
    elif quadrant == DecisionQuadrant.Q3_DEFER:
        return DecisionNeurochem(
            delta_da=-cfg.beta_da_uncertain * (1.0 - confidence),
            delta_gaba=cfg.beta_gaba_calm * (1.0 - confidence),
            delta_ne=cfg.beta_ne_seek * 0.3,
            delta_ach=0.0,
            delta_5ht=0.0,
            delta_cor=0.0,
            beta_boost=0.0,
            theta_gamma_boost=0.0,
            alpha_beta_boost=0.0,
        )
    else:  # Q4_ESCALATE
        return DecisionNeurochem(
            delta_cor=cfg.beta_cor_threat * risk,
            delta_ne=cfg.beta_ne_alert * risk,
            delta_gaba=cfg.beta_gaba_inhibit * risk,
            delta_da=-cfg.beta_da_error * risk,
            delta_ach=0.0,
            delta_5ht=0.0,
            beta_boost=0.0,
            theta_gamma_boost=0.0,
            alpha_beta_boost=0.0,
        )


# =====================================================================
# Engine class
# =====================================================================


class DecisionMakingEngine:
    """
    Engine 15 -- Decision Making Engine.

    Three-stage decision synthesis:
      Stage 1: Bayesian confidence fusion (log-odds + agreement + emotion)
      Stage 2: Risk assessment (detection severity + reward alignment + NT)
      Stage 3: Quadrant routing + hard overrides + tone calibration

    API
    ---
    configure(mode)                -- set operational mode
    update_neurochem_state(state)  -- inject external NT levels
    process(input)                 -- full decision synthesis
    get_status()                   -- introspection
    """

    engine_id = "decision_making_engine"
    cluster   = "reasoning"

    def __init__(
        self,
        config: Optional[DMConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._cfg = config or DMConfig()
        self._rng = rng or np.random.default_rng()
        self._mode = OperationalMode.NORMAL
        self._state = DMState()
        self._cycle_count = 0

    # ----- Configuration --------------------------------------------------

    def configure(self, mode: OperationalMode) -> None:
        """Set operational mode (adjusts all mode-dependent parameters)."""
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """Inject current neurochemical levels for bidirectional feedback."""
        if "da" in state_dict:
            self._state.da_level = _clamp(state_dict["da"])
        if "ne" in state_dict:
            self._state.ne_level = _clamp(state_dict["ne"])
        if "ach" in state_dict:
            self._state.ach_level = _clamp(state_dict["ach"])
        if "5ht" in state_dict:
            self._state._5ht_level = _clamp(state_dict["5ht"])
        if "cor" in state_dict:
            self._state.cor_level = _clamp(state_dict["cor"])
        if "gaba" in state_dict:
            self._state.gaba_level = _clamp(state_dict["gaba"])
        if "oxt" in state_dict:
            self._state.oxt_level = _clamp(state_dict["oxt"])

    # ----- Internal helpers -----------------------------------------------

    def _mode_key(self) -> str:
        """Get mode key for dict lookups."""
        return self._mode.value

    def _get_mode_param(self, param_dict: Dict[str, float], default: float = 0.5) -> float:
        """Look up a mode-dependent parameter."""
        return param_dict.get(self._mode_key(), default)

    def _find_low_confidence_engines(
        self, entries: List[EngineConfidenceEntry], threshold: float = 0.4,
    ) -> List[str]:
        """Find engines with low confidence for clarification prompt."""
        return [e.engine_name for e in entries if e.raw_confidence < threshold]

    # ----- Main process port -----------------------------------------------

    def process(self, inp: DecisionMakingInput) -> DecisionMakingResult:
        """
        Execute full three-stage decision synthesis.

        Parameters
        ----------
        inp : DecisionMakingInput
            Aggregated outputs from all upstream engines.

        Returns
        -------
        DecisionMakingResult
        """
        t0 = time.perf_counter()
        cfg = self._cfg
        mode_key = self._mode_key()

        # ==================================================================
        # STAGE 1: CONFIDENCE FUSION
        # ==================================================================

        # 1a. Bayesian log-odds fusion
        p_prior = self._get_mode_param(cfg.p_prior, 0.50)
        L_fused, c_fused = bayesian_confidence_fusion(
            entries=inp.engine_confidences,
            reliability_weights=cfg.engine_reliability,
            p_prior=p_prior,
            epsilon=cfg.epsilon,
        )

        # Collect per-engine breakdown
        per_engine = {e.engine_name: e.raw_confidence for e in inp.engine_confidences}

        # 1b. Cross-engine agreement penalty
        agreement = cross_engine_agreement(inp.engine_confidences, cfg.sigma_max)
        c_after_agree = apply_agreement_penalty(c_fused, agreement, cfg.alpha_agree)

        # 1c. Emotion-state modulation
        c_after_emotion, emotion_delta = emotion_modulate_confidence(
            c=c_after_agree,
            emotion_confident=inp.emotion_confident,
            emotion_nervous=inp.emotion_nervous,
            emotion_anxiety=inp.emotion_anxiety,
            emotion_perplexed=inp.emotion_perplexed,
            emotion_regret=inp.emotion_regret,
            emotion_skeptical=inp.emotion_skeptical,
            kappa_confident=cfg.kappa_confident,
            kappa_nervous=cfg.kappa_nervous,
            kappa_anxiety=cfg.kappa_anxiety,
            kappa_perplexed=cfg.kappa_perplexed,
            kappa_regret=cfg.kappa_regret,
            kappa_skeptical=cfg.kappa_skeptical,
        )

        # 1d. Logical Brain's confidence adjustment
        c_after_lb = apply_logical_brain_adjustment(
            c_after_emotion,
            inp.confidence_adjustment,
        )

        # 1e. Stochastic jitter (prevent brittle decisions at threshold edges)
        jitter_sigma = self._get_mode_param(cfg.sigma_jitter, 0.01)
        jitter = float(self._rng.normal(0.0, jitter_sigma)) if jitter_sigma > 0 else 0.0
        c_final = _clamp(c_after_lb + jitter, 0.0, 1.0)

        confidence_breakdown = ConfidenceBreakdown(
            per_engine=per_engine,
            fused_logodds=L_fused,
            fused_probability=c_fused,
            agreement_factor=agreement,
            after_agreement=c_after_agree,
            emotion_delta=emotion_delta,
            after_emotion=c_after_emotion,
            confidence_adjustment=inp.confidence_adjustment,
            jitter_applied=jitter,
            final_confidence=c_final,
        )

        # ==================================================================
        # STAGE 2: RISK ASSESSMENT
        # ==================================================================

        # 2a. Detection flag severity
        detection_risk = compute_detection_risk(
            summaries=inp.detection_summaries,
            weights=cfg.w_detection,
        )

        # 2b. Reward-domain misalignment
        reward_risk = compute_reward_risk(
            logic=inp.reward_logic,
            attunement=inp.reward_attunement,
            ethics=inp.reward_ethics,
            innovation=inp.reward_innovation,
            alpha_floor=cfg.alpha_floor,
            beta_attunement=cfg.beta_attunement,
            beta_innovation=cfg.beta_innovation,
        )

        # 2c. NT modulation
        r_combined = combine_risks(detection_risk, reward_risk, cfg.alpha_raw, cfg.alpha_reward)
        r_final, nt_mod_factor = nt_modulate_risk(
            risk=r_combined,
            ne_level=self._state.ne_level,
            cor_level=self._state.cor_level,
            gaba_level=self._state.gaba_level,
            gamma_ne=cfg.gamma_ne,
            gamma_cor=cfg.gamma_cor,
            gamma_gaba=cfg.gamma_gaba,
        )

        risk_breakdown = RiskBreakdown(
            detection_risk=detection_risk,
            reward_risk=reward_risk,
            nt_modulation_factor=nt_mod_factor,
            final_risk=r_final,
        )

        # ==================================================================
        # STAGE 3: DECISION ROUTING
        # ==================================================================

        # 3a. Quadrant classification
        theta_c = self._get_mode_param(cfg.theta_confidence, 0.55)
        theta_r = self._get_mode_param(cfg.theta_risk, 0.40)
        raw_quadrant = classify_quadrant(c_final, r_final, theta_c, theta_r)

        # 3b. Hard overrides
        final_quadrant, override = apply_hard_overrides(raw_quadrant, inp)
        override_str = override.value if override else None

        # 3c. Action, tone, hedge
        action = quadrant_to_action(final_quadrant)
        tone = quadrant_to_tone(final_quadrant)
        h_scale = self._get_mode_param(cfg.hedge_scale, 1.0)
        hedge = compute_hedge_level(final_quadrant, r_final, h_scale)
        certainty = map_certainty_level(c_final)

        # 3d. Processing depth
        depth = compute_depth_recommendation(
            final_quadrant,
            inp.processing_depth_recommendation,
        )

        # 3e. Flags and escalation
        flags = collect_flags_to_surface(
            inp.detection_summaries,
            cfg.theta_surface,
            final_quadrant,
        )
        escalation = collect_escalation_reasons(inp, final_quadrant, override)

        # 3f. Clarification prompt (Q3 only)
        clarification = None
        if final_quadrant == DecisionQuadrant.Q3_DEFER:
            low_engines = self._find_low_confidence_engines(inp.engine_confidences)
            clarification = generate_clarification_prompt(low_engines)

        # 3g. Neurochemical signals
        neurochem = compute_decision_neurochem(final_quadrant, c_final, r_final, cfg)

        # ==================================================================
        # Update internal state
        # ==================================================================
        self._cycle_count += 1
        self._state.total_decisions += 1
        if final_quadrant == DecisionQuadrant.Q1_RESPOND:
            self._state.q1_count += 1
        elif final_quadrant == DecisionQuadrant.Q2_QUALIFY:
            self._state.q2_count += 1
        elif final_quadrant == DecisionQuadrant.Q3_DEFER:
            self._state.q3_count += 1
        else:
            self._state.q4_count += 1
        if override:
            self._state.override_count += 1

        elapsed = (time.perf_counter() - t0) * 1000.0

        return DecisionMakingResult(
            action=action,
            quadrant=final_quadrant,
            certainty_level=certainty,
            confidence=confidence_breakdown,
            risk=risk_breakdown,
            hedge_level=hedge,
            recommended_tone=tone,
            flags_to_surface=flags,
            escalation_reasons=escalation,
            override_applied=override_str,
            depth_recommendation=depth,
            clarification_prompt=clarification,
            neurochemical_signals=neurochem,
            processing_time_ms=elapsed,
            engine_id=self.engine_id,
            metadata={
                "mode": self._mode.value,
                "cycle": self._cycle_count,
                "raw_quadrant": raw_quadrant.value,
                "final_quadrant": final_quadrant.value,
                "override": override_str,
            },
        )

    # ----- Introspection --------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return engine status for monitoring and debugging."""
        return {
            "engine_id": self.engine_id,
            "cluster": self.cluster,
            "mode": self._mode.value,
            "cycle_count": self._cycle_count,
            "total_decisions": self._state.total_decisions,
            "quadrant_distribution": {
                "Q1_respond": self._state.q1_count,
                "Q2_qualify": self._state.q2_count,
                "Q3_defer": self._state.q3_count,
                "Q4_escalate": self._state.q4_count,
            },
            "override_count": self._state.override_count,
            "nt_levels": {
                "da": self._state.da_level,
                "ne": self._state.ne_level,
                "ach": self._state.ach_level,
                "5ht": self._state._5ht_level,
                "cor": self._state.cor_level,
                "gaba": self._state.gaba_level,
                "oxt": self._state.oxt_level,
            },
        }
