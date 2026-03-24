"""
Engine 26 -- Uncertainty Pattern Engine  (``uncertainty_pattern_engine``)
========================================================================
Epistemic self-awareness module that tracks, quantifies, and propagates
uncertainty through the system's reasoning chains.

Three-phase pipeline:
  * **Phase 1 — Uncertainty Extraction**: Collect per-claim confidence
    from upstream engines, classify uncertainty type, build uncertainty map.
  * **Phase 2 — Propagation Analysis**: Trace inference chains, propagate
    uncertainty forward via Bayesian chain rule, detect amplification
    points and bottleneck premises.
  * **Phase 3 — Pattern Detection**: Detect overconfidence, topic-specific
    miscalibration, structural patterns (cascade, island, divergence,
    stagnation).

Uncertainty types:
  EPISTEMIC  — knowledge gap (reducible)
  ALEATORIC  — inherent randomness (irreducible)
  MODEL      — reasoning method uncertain (partially reducible)
  LINGUISTIC — natural language ambiguity (reducible via clarification)

Neurochemical coupling:
  NE   — increased uncertainty sensitivity
  DA   — confidence bias (mild uncertainty underestimation)
  ACh  — sharper uncertainty discrimination
  5-HT — uncertainty dampening (emotional buffering)
  COR  — uncertainty amplification (stress)
  GABA — propagation depth reduction (calming)

Emotion integration:
  Curious  (#36) → tolerance for epistemic uncertainty ↑
  Anxious  (#21) → aleatoric perception ↑, depth ↓
  Confident(#40) → mild uncertainty suppression (overconfidence risk)
  Perplexed(#24) → deep propagation analysis ↑
  Courageous(#39)→ tolerance for high-uncertainty action ↑
  Skeptical(#3)  → all uncertainty boosted, bottleneck detection ↑
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


class UncertaintyType(str, Enum):
    """Classification of uncertainty source."""
    EPISTEMIC  = "epistemic"    # Knowledge gap — reducible
    ALEATORIC  = "aleatoric"    # Inherent randomness — irreducible
    MODEL      = "model"        # Reasoning method uncertain
    LINGUISTIC = "linguistic"   # NL ambiguity — reducible via clarification


class PatternType(str, Enum):
    """Structural uncertainty pattern categories."""
    CASCADE    = "cascade"      # 3+ claims in chain all above theta
    ISLAND     = "island"       # High-confidence claim among low-confidence neighbors
    DIVERGENCE = "divergence"   # Two engines disagree on same claim
    STAGNATION = "stagnation"   # Same uncertainty level across N cycles


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class UPConfig:
    """
    All tunable parameters for the Uncertainty Pattern Engine.

    Mode-dependent parameters stored as dicts keyed by OperationalMode.value.
    """

    # --- Phase 1: Uncertainty Extraction ---

    # Evidence decay: each additional piece of evidence reduces uncertainty
    lambda_evidence: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.15, "dev": 0.10, "learning": 0.12,
        "reflective": 0.18, "rem_normal": 0.15, "rem_dream": 0.05,
    })

    # Classification thresholds
    theta_aleatoric: int = 3       # Prediction steps ahead → aleatoric
    theta_model: int = 5           # Reasoning chain length → model uncertainty
    theta_linguistic: float = 0.40 # Ambiguity score → linguistic

    # --- Phase 2: Propagation Analysis ---

    # Amplification detection
    theta_amplification: Dict[str, float] = field(default_factory=lambda: {
        "normal": 2.0, "dev": 3.0, "learning": 2.5,
        "reflective": 1.5, "rem_normal": 2.0, "rem_dream": 4.0,
    })

    # Bottleneck detection
    theta_bottleneck: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.70, "dev": 0.60, "learning": 0.65,
        "reflective": 0.75, "rem_normal": 0.70, "rem_dream": 0.50,
    })

    # Propagation depth gating
    D_base: Dict[str, int] = field(default_factory=lambda: {
        "normal": 5, "dev": 7, "learning": 6,
        "reflective": 4, "rem_normal": 5, "rem_dream": 3,
    })
    D_boost: Dict[str, int] = field(default_factory=lambda: {
        "normal": 3, "dev": 4, "learning": 3,
        "reflective": 2, "rem_normal": 3, "rem_dream": 2,
    })
    D_penalty: Dict[str, int] = field(default_factory=lambda: {
        "normal": 4, "dev": 2, "learning": 3,
        "reflective": 5, "rem_normal": 4, "rem_dream": 1,
    })

    # Default inference uncertainty (how uncertain each inference step is)
    u_inference_default: float = 0.05

    # --- Phase 3: Pattern Detection ---

    theta_calibration_alarm: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.15, "dev": 0.25, "learning": 0.20,
        "reflective": 0.10, "rem_normal": 0.15, "rem_dream": 0.40,
    })

    # System entropy alert threshold
    theta_alert: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.60, "dev": 0.75, "learning": 0.65,
        "reflective": 0.50, "rem_normal": 0.60, "rem_dream": 0.85,
    })

    # Pattern thresholds
    cascade_min_length: int = 3
    cascade_uncertainty_threshold: float = 0.50
    island_delta_threshold: float = 0.40
    divergence_delta_threshold: float = 0.30
    stagnation_cycles: int = 5
    stagnation_delta: float = 0.05

    # --- Neurochemical coupling ---

    gamma_ne: float = 0.20
    gamma_da: float = 0.15
    gamma_ach: float = 0.15
    gamma_cor: float = 0.25
    gamma_gaba: float = 0.10

    # Write-port NT coefficients
    beta_ne_uncertain: float = 0.12
    beta_da_dampen: float = 0.08
    beta_cor_cascade: float = 0.10
    beta_ach_focus: float = 0.10
    beta_gaba_stabilize: float = 0.06

    # Oscillatory
    psi_theta_gamma: float = 0.06
    psi_beta_suppress: float = 0.05

    # --- Emotion modulation ---

    kappa_anxious: float = 0.20
    kappa_skeptical: float = 0.15
    kappa_confident: float = 0.10
    kappa_courageous: float = 0.08
    zeta_curious: float = 0.25
    zeta_anxious: float = 0.20
    zeta_perplexed: float = 0.30

    # --- Stochastic ---
    sigma_noise: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.005, "dev": 0.01, "learning": 0.007,
        "reflective": 0.003, "rem_normal": 0.005, "rem_dream": 0.02,
    })


# =====================================================================
# Mutable engine state
# =====================================================================


@dataclass
class UPState:
    """Runtime state: NT levels, calibration history, counters."""
    # NT read-port levels [0, 1]
    ne_level:   float = 0.0
    da_level:   float = 0.0
    ach_level:  float = 0.0
    _5ht_level: float = 0.0
    cor_level:  float = 0.0
    gaba_level: float = 0.0

    # Historical calibration bins: bin_index -> (correct_count, total_count)
    calibration_bins: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    # Historical per-claim uncertainty (claim_id -> list of past uncertainties)
    claim_history: Dict[str, List[float]] = field(default_factory=dict)

    # Tracking
    total_analyses: int = 0
    total_amplifiers_found: int = 0
    total_bottlenecks_found: int = 0
    total_patterns_found: int = 0
    overconfidence_alerts: int = 0


# =====================================================================
# Frozen I/O dataclasses
# =====================================================================


@dataclass(frozen=True)
class ClaimWithConfidence:
    """A claim produced by an upstream engine with confidence metadata."""
    claim_id: str
    text: str = ""
    confidence: float = 0.5
    evidence_count: int = 0
    reasoning_chain_length: int = 0
    prediction_horizon: int = 0     # Steps ahead (0 for non-predictions)
    ambiguity_score: float = 0.0    # From tokenizer/semantic expander
    source_engine: str = ""


@dataclass(frozen=True)
class InferenceStep:
    """One step in an inference chain."""
    premise_id: str
    conclusion_id: str
    inference_type: str = "deductive"   # deductive | inductive | abductive
    inference_confidence: float = 0.95  # How reliable this inference method is


@dataclass(frozen=True)
class InferenceChain:
    """A sequence of inference steps forming a reasoning chain."""
    chain_id: str
    steps: Tuple[InferenceStep, ...] = ()


@dataclass(frozen=True)
class CalibrationData:
    """Historical accuracy data for calibration analysis."""
    # bin_index → (correct_count, total_count)
    bins: Dict[int, Tuple[int, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class UncertaintyEstimate:
    """Per-claim uncertainty estimate."""
    claim_id: str
    raw_uncertainty: float          # [0, 1]
    refined_uncertainty: float      # After evidence decay
    uncertainty_type: str           # UncertaintyType value
    source_engine: str
    evidence_count: int
    emotion_adjusted: float         # After emotion modulation


@dataclass(frozen=True)
class PropagationResult:
    """Result of propagating uncertainty through one inference chain."""
    chain_id: str
    chain_uncertainty: float        # Propagated uncertainty of final conclusion
    amplification_ratio: float      # How much uncertainty grew
    is_amplifier: bool
    bottleneck_premise: Optional[str] = None
    bottleneck_contribution: float = 0.0
    depth_reached: int = 0


@dataclass(frozen=True)
class UncertaintyPattern:
    """A detected structural uncertainty pattern."""
    pattern_type: str               # PatternType value
    affected_claims: Tuple[str, ...] = ()
    severity: float = 0.0           # [0, 1]
    description: str = ""


@dataclass(frozen=True)
class UncertaintyNeurochem:
    """Neurochemical deltas emitted by the Uncertainty Pattern Engine."""
    delta_ne:  float = 0.0
    delta_da:  float = 0.0
    delta_ach: float = 0.0
    delta_cor: float = 0.0
    delta_gaba: float = 0.0
    theta_gamma_boost: float = 0.0
    beta_suppress: float = 0.0


@dataclass(frozen=True)
class UncertaintyPatternInput:
    """Input to the Uncertainty Pattern Engine."""
    engine_claims: Dict[str, Tuple[ClaimWithConfidence, ...]] = field(default_factory=dict)
    inference_chains: Tuple[InferenceChain, ...] = ()
    historical_calibration: Optional[CalibrationData] = None
    ambiguity_scores: Optional[Dict[str, float]] = None
    theta_gamma_coupling: float = 0.5
    emotion_intensities: Optional[Dict[str, float]] = None
    active_mode: str = "normal"
    cycle_count: int = 0


@dataclass(frozen=True)
class UncertaintyPatternResult:
    """Output from the Uncertainty Pattern Engine."""
    # Uncertainty map
    uncertainty_map: Dict[str, UncertaintyEstimate] = field(default_factory=dict)
    system_entropy: float = 0.0

    # Propagation
    propagation_results: Tuple[PropagationResult, ...] = ()
    max_chain_uncertainty: float = 0.0
    total_amplifiers: int = 0
    total_bottlenecks: int = 0

    # Patterns
    patterns_detected: Tuple[UncertaintyPattern, ...] = ()
    calibration_error: float = 0.0
    overconfidence_alert: bool = False
    topic_alerts: Tuple[str, ...] = ()

    # Epistemic/Aleatoric split
    epistemic_fraction: float = 0.0
    aleatoric_fraction: float = 0.0
    reducible_uncertainty_claims: Tuple[str, ...] = ()

    # Neurochemical signals
    neurochemical_signals: UncertaintyNeurochem = field(default_factory=UncertaintyNeurochem)

    # Metadata
    propagation_depth_used: int = 0
    processing_time_ms: float = 0.0
    engine_id: str = "uncertainty_pattern_engine"
    metadata: Dict[str, Any] = field(default_factory=dict)


# =====================================================================
# Utility
# =====================================================================


# =====================================================================
# Phase 1: Uncertainty Extraction  (pure functions)
# =====================================================================


def beta_decay(evidence_count: int, lambda_ev: float = 0.15) -> float:
    """Evidence-based uncertainty decay: more evidence → lower uncertainty."""
    if evidence_count <= 0:
        return 1.0
    return math.exp(-lambda_ev * evidence_count)


def classify_uncertainty_type(
    claim: ClaimWithConfidence,
    theta_aleatoric: int = 3,
    theta_model: int = 5,
    theta_linguistic: float = 0.40,
) -> UncertaintyType:
    """Classify what kind of uncertainty a claim carries."""
    if claim.ambiguity_score > theta_linguistic:
        return UncertaintyType.LINGUISTIC
    if claim.prediction_horizon > theta_aleatoric:
        return UncertaintyType.ALEATORIC
    if claim.reasoning_chain_length > theta_model:
        return UncertaintyType.MODEL
    return UncertaintyType.EPISTEMIC


def refine_uncertainty(
    raw: float,
    evidence_count: int,
    lambda_ev: float = 0.15,
) -> float:
    """Refine raw uncertainty using evidence decay."""
    return _clamp(raw * beta_decay(evidence_count, lambda_ev))


def emotion_modulate_uncertainty(
    u: float,
    emotion_intensities: Optional[Dict[str, float]],
    kappa_anxious: float = 0.20,
    kappa_skeptical: float = 0.15,
    kappa_confident: float = 0.10,
    kappa_courageous: float = 0.08,
) -> float:
    """Modulate uncertainty based on current emotion state."""
    if not emotion_intensities:
        return u
    delta = (
        kappa_anxious * emotion_intensities.get("anxiety", 0.0)
        + kappa_skeptical * emotion_intensities.get("skeptical", 0.0)
        - kappa_confident * emotion_intensities.get("confident", 0.0)
        - kappa_courageous * emotion_intensities.get("courageous", 0.0)
    )
    return _clamp(u * (1.0 + delta))


def extract_uncertainty_map(
    engine_claims: Dict[str, Tuple[ClaimWithConfidence, ...]],
    lambda_ev: float,
    theta_aleatoric: int,
    theta_model: int,
    theta_linguistic: float,
    emotion_intensities: Optional[Dict[str, float]],
    kappas: Dict[str, float],
) -> Dict[str, UncertaintyEstimate]:
    """Build the complete uncertainty map from all engine claims."""
    umap: Dict[str, UncertaintyEstimate] = {}
    for engine_name, claims in engine_claims.items():
        for claim in claims:
            raw_u = 1.0 - _clamp(claim.confidence)
            utype = classify_uncertainty_type(
                claim, theta_aleatoric, theta_model, theta_linguistic,
            )
            refined = refine_uncertainty(raw_u, claim.evidence_count, lambda_ev)
            adjusted = emotion_modulate_uncertainty(
                refined, emotion_intensities,
                kappas.get("anxious", 0.20),
                kappas.get("skeptical", 0.15),
                kappas.get("confident", 0.10),
                kappas.get("courageous", 0.08),
            )
            umap[claim.claim_id] = UncertaintyEstimate(
                claim_id=claim.claim_id,
                raw_uncertainty=raw_u,
                refined_uncertainty=refined,
                uncertainty_type=utype.value,
                source_engine=claim.source_engine or engine_name,
                evidence_count=claim.evidence_count,
                emotion_adjusted=adjusted,
            )
    return umap


# =====================================================================
# System entropy
# =====================================================================


def compute_system_entropy(umap: Dict[str, UncertaintyEstimate]) -> float:
    """
    Shannon entropy over uncertainty map, normalized to [0, 1].

    H = -Σ [p·log(p) + (1-p)·log(1-p)]  /  (N·log(2))
    where p_c = 1 - u_c (probability of claim being correct).
    """
    if not umap:
        return 0.0
    eps = 1e-12
    total = 0.0
    for est in umap.values():
        p = _clamp(1.0 - est.emotion_adjusted, eps, 1.0 - eps)
        total += -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))
    n = len(umap)
    # Normalize by max entropy (N * log(2))
    max_entropy = n * math.log(2.0)
    if max_entropy < eps:
        return 0.0
    return _clamp(total / max_entropy)


# =====================================================================
# Phase 2: Propagation Analysis  (pure functions)
# =====================================================================


def propagate_chain(
    chain: InferenceChain,
    umap: Dict[str, UncertaintyEstimate],
    u_inference_default: float = 0.05,
) -> PropagationResult:
    """
    Propagate uncertainty through an inference chain.

    u_C = 1 - Π_i (1 - u_P_i) * (1 - u_inference_i)
    """
    if not chain.steps:
        return PropagationResult(
            chain_id=chain.chain_id,
            chain_uncertainty=0.0,
            amplification_ratio=0.0,
            is_amplifier=False,
            depth_reached=0,
        )

    # Track premise uncertainties
    premise_uncertainties: Dict[str, float] = {}
    product = 1.0

    for step in chain.steps:
        # Get premise uncertainty
        u_premise = 0.5  # Default if not in map
        if step.premise_id in umap:
            u_premise = umap[step.premise_id].emotion_adjusted
        premise_uncertainties[step.premise_id] = u_premise

        u_inf = 1.0 - step.inference_confidence
        product *= (1.0 - u_premise) * (1.0 - u_inf)

    chain_uncertainty = _clamp(1.0 - product)

    # Amplification ratio
    max_premise_u = max(premise_uncertainties.values()) if premise_uncertainties else 0.0
    if max_premise_u > 1e-9:
        amp_ratio = chain_uncertainty / max_premise_u
    else:
        amp_ratio = 0.0 if chain_uncertainty < 1e-9 else float("inf")

    # Bottleneck detection
    total_u = sum(premise_uncertainties.values())
    bottleneck_id = None
    bottleneck_contrib = 0.0
    if total_u > 1e-9:
        for pid, pu in premise_uncertainties.items():
            contrib = pu / total_u
            if contrib > bottleneck_contrib:
                bottleneck_contrib = contrib
                bottleneck_id = pid

    return PropagationResult(
        chain_id=chain.chain_id,
        chain_uncertainty=chain_uncertainty,
        amplification_ratio=amp_ratio,
        is_amplifier=False,  # Caller sets this based on threshold
        bottleneck_premise=bottleneck_id,
        bottleneck_contribution=bottleneck_contrib,
        depth_reached=len(chain.steps),
    )


def compute_propagation_depth(
    D_base: int,
    D_boost: int,
    D_penalty: int,
    theta_gamma: float,
    system_entropy: float,
) -> int:
    """Compute maximum propagation depth, mirroring E13's recursion depth gating."""
    raw = D_base + D_boost * theta_gamma - D_penalty * system_entropy
    return max(1, int(raw))


# =====================================================================
# Phase 3: Pattern Detection  (pure functions)
# =====================================================================


def compute_calibration_error(calibration: Optional[CalibrationData]) -> float:
    """
    Expected Calibration Error (ECE).

    ECE = Σ_k (|bin_k| / N) * |mean_confidence(bin_k) - empirical_accuracy(bin_k)|
    """
    if calibration is None or not calibration.bins:
        return 0.0

    total_samples = 0
    weighted_error = 0.0

    for _bin_idx, (correct, total) in calibration.bins.items():
        if total == 0:
            continue
        total_samples += total
        # Bin midpoint confidence (bin index 0-4 → 0.1, 0.3, 0.5, 0.7, 0.9)
        bin_confidence = (_bin_idx + 0.5) / 5.0
        empirical_accuracy = correct / total
        weighted_error += total * abs(bin_confidence - empirical_accuracy)

    if total_samples == 0:
        return 0.0
    return weighted_error / total_samples


def detect_cascade_pattern(
    chain: InferenceChain,
    umap: Dict[str, UncertaintyEstimate],
    min_length: int = 3,
    threshold: float = 0.50,
) -> Optional[UncertaintyPattern]:
    """Detect cascade: 3+ consecutive claims above uncertainty threshold."""
    if len(chain.steps) < min_length:
        return None

    high_u_claims: List[str] = []
    for step in chain.steps:
        pid = step.premise_id
        if pid in umap and umap[pid].emotion_adjusted > threshold:
            high_u_claims.append(pid)
        else:
            high_u_claims = []  # Reset streak
        if len(high_u_claims) >= min_length:
            severity = sum(umap[c].emotion_adjusted for c in high_u_claims) / len(high_u_claims)
            return UncertaintyPattern(
                pattern_type=PatternType.CASCADE.value,
                affected_claims=tuple(high_u_claims),
                severity=_clamp(severity),
                description=f"Cascading uncertainty: {len(high_u_claims)} claims in chain "
                            f"'{chain.chain_id}' all above {threshold:.2f}",
            )
    return None


def detect_island_pattern(
    umap: Dict[str, UncertaintyEstimate],
    chain: InferenceChain,
    delta_threshold: float = 0.40,
) -> List[UncertaintyPattern]:
    """Detect islands: high-confidence claim surrounded by low-confidence neighbors."""
    patterns: List[UncertaintyPattern] = []
    if len(chain.steps) < 2:
        return patterns

    claim_ids = [step.premise_id for step in chain.steps]
    # Add last conclusion
    if chain.steps:
        claim_ids.append(chain.steps[-1].conclusion_id)

    for i, cid in enumerate(claim_ids):
        if cid not in umap:
            continue
        u_current = umap[cid].emotion_adjusted
        # Low uncertainty = high confidence → potential island
        if u_current > 0.3:
            continue

        neighbors = []
        if i > 0 and claim_ids[i - 1] in umap:
            neighbors.append(umap[claim_ids[i - 1]].emotion_adjusted)
        if i < len(claim_ids) - 1 and claim_ids[i + 1] in umap:
            neighbors.append(umap[claim_ids[i + 1]].emotion_adjusted)

        if not neighbors:
            continue
        avg_neighbor_u = sum(neighbors) / len(neighbors)

        if avg_neighbor_u - u_current > delta_threshold:
            patterns.append(UncertaintyPattern(
                pattern_type=PatternType.ISLAND.value,
                affected_claims=(cid,),
                severity=_clamp(avg_neighbor_u - u_current),
                description=f"Confidence island: claim '{cid}' (u={u_current:.2f}) "
                            f"surrounded by uncertain neighbors (avg u={avg_neighbor_u:.2f})",
            ))
    return patterns


def detect_divergence_pattern(
    umap: Dict[str, UncertaintyEstimate],
    delta_threshold: float = 0.30,
) -> List[UncertaintyPattern]:
    """Detect divergence: same claim assessed differently by different engines."""
    # Group by claim text (since different engines may produce overlapping claims)
    # We check for claims with same claim_id from different sources
    by_text: Dict[str, List[UncertaintyEstimate]] = {}
    for est in umap.values():
        key = est.claim_id
        if key not in by_text:
            by_text[key] = []
        by_text[key].append(est)

    patterns: List[UncertaintyPattern] = []
    for _claim_key, estimates in by_text.items():
        if len(estimates) < 2:
            continue
        uncertainties = [e.emotion_adjusted for e in estimates]
        delta = max(uncertainties) - min(uncertainties)
        if delta > delta_threshold:
            patterns.append(UncertaintyPattern(
                pattern_type=PatternType.DIVERGENCE.value,
                affected_claims=tuple(e.claim_id for e in estimates),
                severity=_clamp(delta),
                description=f"Divergence: engines disagree on uncertainty for "
                            f"'{estimates[0].claim_id}' (range={delta:.2f})",
            ))
    return patterns


def detect_stagnation_pattern(
    umap: Dict[str, UncertaintyEstimate],
    history: Dict[str, List[float]],
    min_cycles: int = 5,
    delta: float = 0.05,
) -> List[UncertaintyPattern]:
    """Detect stagnation: uncertainty unchanged across N cycles."""
    patterns: List[UncertaintyPattern] = []
    for cid, est in umap.items():
        if cid not in history:
            continue
        hist = history[cid]
        if len(hist) < min_cycles:
            continue
        recent = hist[-min_cycles:]
        spread = max(recent) - min(recent)
        if spread < delta:
            patterns.append(UncertaintyPattern(
                pattern_type=PatternType.STAGNATION.value,
                affected_claims=(cid,),
                severity=_clamp(est.emotion_adjusted),
                description=f"Stagnation: claim '{cid}' uncertainty unchanged "
                            f"for {min_cycles} cycles (spread={spread:.4f})",
            ))
    return patterns


def compute_epistemic_fraction(umap: Dict[str, UncertaintyEstimate]) -> float:
    """Fraction of total uncertainty that is epistemic (reducible)."""
    if not umap:
        return 0.0
    total_u = sum(e.emotion_adjusted for e in umap.values())
    if total_u < 1e-12:
        return 0.0
    epistemic_u = sum(
        e.emotion_adjusted for e in umap.values()
        if e.uncertainty_type in (UncertaintyType.EPISTEMIC.value, UncertaintyType.LINGUISTIC.value)
    )
    return _clamp(epistemic_u / total_u)


def compute_aleatoric_fraction(umap: Dict[str, UncertaintyEstimate]) -> float:
    """Fraction of total uncertainty that is aleatoric (irreducible)."""
    if not umap:
        return 0.0
    total_u = sum(e.emotion_adjusted for e in umap.values())
    if total_u < 1e-12:
        return 0.0
    aleatoric_u = sum(
        e.emotion_adjusted for e in umap.values()
        if e.uncertainty_type == UncertaintyType.ALEATORIC.value
    )
    return _clamp(aleatoric_u / total_u)


def find_reducible_claims(umap: Dict[str, UncertaintyEstimate], top_n: int = 5) -> Tuple[str, ...]:
    """Find claims where gathering data would help most (high epistemic uncertainty)."""
    reducible = [
        (est.claim_id, est.emotion_adjusted)
        for est in umap.values()
        if est.uncertainty_type in (UncertaintyType.EPISTEMIC.value, UncertaintyType.LINGUISTIC.value)
    ]
    reducible.sort(key=lambda x: x[1], reverse=True)
    return tuple(cid for cid, _ in reducible[:top_n])


# =====================================================================
# Neurochemical signal computation (pure)
# =====================================================================


def compute_uncertainty_neurochem(
    system_entropy: float,
    theta_alert: float,
    cascade_count: int,
    max_cascades: int,
    has_bottleneck: bool,
    bottleneck_contribution: float,
    delta_h: float,       # Change in entropy (negative = improving)
    ece: float,
    cfg: UPConfig,
) -> UncertaintyNeurochem:
    """Compute neurochemical deltas based on uncertainty analysis results."""
    delta_ne = 0.0
    delta_da = 0.0
    delta_ach = 0.0
    delta_cor = 0.0
    delta_gaba = 0.0
    tg_boost = 0.0
    beta_sup = 0.0

    # High uncertainty state
    if system_entropy > theta_alert:
        delta_ne = cfg.beta_ne_uncertain * system_entropy
        delta_da = -cfg.beta_da_dampen * (system_entropy - theta_alert)
        if max_cascades > 0:
            delta_cor = cfg.beta_cor_cascade * min(cascade_count / max(max_cascades, 1), 1.0)

    # Bottleneck detected → focus attention
    if has_bottleneck:
        delta_ach = cfg.beta_ach_focus * bottleneck_contribution

    # Uncertainty improving → stabilize
    if delta_h < 0:
        delta_gaba = cfg.beta_gaba_stabilize * abs(delta_h)

    # Overconfidence → suppress beta
    if ece > 0:
        beta_sup = cfg.psi_beta_suppress * ece

    return UncertaintyNeurochem(
        delta_ne=delta_ne,
        delta_da=delta_da,
        delta_ach=delta_ach,
        delta_cor=delta_cor,
        delta_gaba=delta_gaba,
        theta_gamma_boost=tg_boost,
        beta_suppress=beta_sup,
    )


# =====================================================================
# Engine class
# =====================================================================


class UncertaintyPatternEngine:
    """
    Engine 26 -- Uncertainty Pattern Engine.

    Three-phase epistemic self-awareness:
      Phase 1: Uncertainty Extraction (per-claim, type classification, evidence decay)
      Phase 2: Propagation Analysis (chain rule, amplification, bottlenecks)
      Phase 3: Pattern Detection (overconfidence, cascade, island, divergence, stagnation)

    API
    ---
    configure(mode)                -- set operational mode
    update_neurochem_state(state)  -- inject external NT levels
    process(input)                 -- full uncertainty analysis
    get_status()                   -- introspection
    """

    engine_id = "uncertainty_pattern_engine"
    cluster   = "meta_self_awareness"

    def __init__(
        self,
        config: Optional[UPConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._cfg = config or UPConfig()
        self._rng = rng or np.random.default_rng()
        self._mode = OperationalMode.NORMAL
        self._state = UPState()
        self._cycle_count = 0
        self._prev_entropy = 0.0  # For delta tracking

    # ----- Configuration --------------------------------------------------

    def configure(self, mode: OperationalMode) -> None:
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        if "ne" in state_dict:
            self._state.ne_level = _clamp(state_dict["ne"])
        if "da" in state_dict:
            self._state.da_level = _clamp(state_dict["da"])
        if "ach" in state_dict:
            self._state.ach_level = _clamp(state_dict["ach"])
        if "5ht" in state_dict:
            self._state._5ht_level = _clamp(state_dict["5ht"])
        if "cor" in state_dict:
            self._state.cor_level = _clamp(state_dict["cor"])
        if "gaba" in state_dict:
            self._state.gaba_level = _clamp(state_dict["gaba"])

    # ----- Helpers --------------------------------------------------------

    def _mode_key(self) -> str:
        return self._mode.value

    def _get_mode_param(self, param_dict: Dict, default=0.5):
        return param_dict.get(self._mode_key(), default)

    # ----- Main process ---------------------------------------------------

    def process(self, inp: UncertaintyPatternInput) -> UncertaintyPatternResult:
        t0 = time.perf_counter()
        cfg = self._cfg
        mk = self._mode_key()

        # ============================================================
        # PHASE 1: UNCERTAINTY EXTRACTION
        # ============================================================

        lambda_ev = self._get_mode_param(cfg.lambda_evidence, 0.15)
        # NT modulation of evidence decay
        lambda_ev_eff = lambda_ev * (1.0 + cfg.gamma_da * self._state.da_level
                                     - cfg.gamma_cor * self._state.cor_level)

        kappas = {
            "anxious": cfg.kappa_anxious,
            "skeptical": cfg.kappa_skeptical,
            "confident": cfg.kappa_confident,
            "courageous": cfg.kappa_courageous,
        }
        umap = extract_uncertainty_map(
            engine_claims=inp.engine_claims,
            lambda_ev=lambda_ev_eff,
            theta_aleatoric=cfg.theta_aleatoric,
            theta_model=cfg.theta_model,
            theta_linguistic=cfg.theta_linguistic,
            emotion_intensities=inp.emotion_intensities,
            kappas=kappas,
        )

        # Apply stochastic noise
        sigma = self._get_mode_param(cfg.sigma_noise, 0.005)
        if sigma > 0:
            noisy_map: Dict[str, UncertaintyEstimate] = {}
            for cid, est in umap.items():
                noise = float(self._rng.normal(0.0, sigma))
                adj = _clamp(est.emotion_adjusted + noise)
                noisy_map[cid] = UncertaintyEstimate(
                    claim_id=est.claim_id,
                    raw_uncertainty=est.raw_uncertainty,
                    refined_uncertainty=est.refined_uncertainty,
                    uncertainty_type=est.uncertainty_type,
                    source_engine=est.source_engine,
                    evidence_count=est.evidence_count,
                    emotion_adjusted=adj,
                )
            umap = noisy_map

        system_entropy = compute_system_entropy(umap)

        # ============================================================
        # PHASE 2: PROPAGATION ANALYSIS
        # ============================================================

        d_base = self._get_mode_param(cfg.D_base, 5)
        d_boost = self._get_mode_param(cfg.D_boost, 3)
        d_penalty = self._get_mode_param(cfg.D_penalty, 4)
        max_depth = compute_propagation_depth(
            d_base, d_boost, d_penalty,
            inp.theta_gamma_coupling, system_entropy,
        )

        theta_amp = self._get_mode_param(cfg.theta_amplification, 2.0)
        # NT modulation of amplification threshold
        theta_amp_eff = theta_amp * (1.0 - cfg.gamma_ne * self._state.ne_level
                                     + cfg.gamma_gaba * self._state.gaba_level)

        theta_bn = self._get_mode_param(cfg.theta_bottleneck, 0.70)
        theta_bn_eff = theta_bn * (1.0 + cfg.gamma_ach * self._state.ach_level)

        prop_results: List[PropagationResult] = []
        total_amplifiers = 0
        total_bottlenecks = 0
        max_chain_u = 0.0

        for chain in inp.inference_chains:
            # Limit chain propagation by depth
            limited_steps = chain.steps[:max_depth] if len(chain.steps) > max_depth else chain.steps
            limited_chain = InferenceChain(chain_id=chain.chain_id, steps=limited_steps)

            pr = propagate_chain(limited_chain, umap, cfg.u_inference_default)

            is_amp = pr.amplification_ratio > theta_amp_eff if theta_amp_eff > 0 else False
            has_bn = pr.bottleneck_contribution > theta_bn_eff

            # Rebuild with flags
            pr = PropagationResult(
                chain_id=pr.chain_id,
                chain_uncertainty=pr.chain_uncertainty,
                amplification_ratio=pr.amplification_ratio,
                is_amplifier=is_amp,
                bottleneck_premise=pr.bottleneck_premise if has_bn else None,
                bottleneck_contribution=pr.bottleneck_contribution if has_bn else 0.0,
                depth_reached=pr.depth_reached,
            )
            prop_results.append(pr)

            if is_amp:
                total_amplifiers += 1
            if has_bn:
                total_bottlenecks += 1
            if pr.chain_uncertainty > max_chain_u:
                max_chain_u = pr.chain_uncertainty

        # ============================================================
        # PHASE 3: PATTERN DETECTION
        # ============================================================

        patterns: List[UncertaintyPattern] = []

        # Cascade patterns
        for chain in inp.inference_chains:
            cp = detect_cascade_pattern(
                chain, umap, cfg.cascade_min_length, cfg.cascade_uncertainty_threshold,
            )
            if cp:
                patterns.append(cp)

        # Island patterns
        for chain in inp.inference_chains:
            patterns.extend(detect_island_pattern(umap, chain, cfg.island_delta_threshold))

        # Divergence patterns
        patterns.extend(detect_divergence_pattern(umap, cfg.divergence_delta_threshold))

        # Stagnation patterns
        patterns.extend(detect_stagnation_pattern(
            umap, self._state.claim_history,
            cfg.stagnation_cycles, cfg.stagnation_delta,
        ))

        # Calibration error
        ece = compute_calibration_error(inp.historical_calibration)
        theta_cal = self._get_mode_param(cfg.theta_calibration_alarm, 0.15)
        overconfidence = ece > theta_cal

        # Epistemic/aleatoric split
        ep_frac = compute_epistemic_fraction(umap)
        al_frac = compute_aleatoric_fraction(umap)
        reducible = find_reducible_claims(umap)

        # ============================================================
        # NEUROCHEMICAL SIGNALS
        # ============================================================

        theta_al = self._get_mode_param(cfg.theta_alert, 0.60)
        delta_h = system_entropy - self._prev_entropy
        cascade_count = sum(1 for p in patterns if p.pattern_type == PatternType.CASCADE.value)

        bn_contrib = 0.0
        has_any_bn = total_bottlenecks > 0
        if has_any_bn:
            bn_contrib = max(
                (pr.bottleneck_contribution for pr in prop_results if pr.bottleneck_premise),
                default=0.0,
            )

        neurochem = compute_uncertainty_neurochem(
            system_entropy=system_entropy,
            theta_alert=theta_al,
            cascade_count=cascade_count,
            max_cascades=max(len(inp.inference_chains), 1),
            has_bottleneck=has_any_bn,
            bottleneck_contribution=bn_contrib,
            delta_h=delta_h,
            ece=ece,
            cfg=cfg,
        )

        # ============================================================
        # UPDATE STATE
        # ============================================================

        self._cycle_count += 1
        self._prev_entropy = system_entropy
        self._state.total_analyses += 1
        self._state.total_amplifiers_found += total_amplifiers
        self._state.total_bottlenecks_found += total_bottlenecks
        self._state.total_patterns_found += len(patterns)
        if overconfidence:
            self._state.overconfidence_alerts += 1

        # Update claim history
        for cid, est in umap.items():
            if cid not in self._state.claim_history:
                self._state.claim_history[cid] = []
            self._state.claim_history[cid].append(est.emotion_adjusted)
            # Keep history bounded
            if len(self._state.claim_history[cid]) > 20:
                self._state.claim_history[cid] = self._state.claim_history[cid][-20:]

        elapsed = (time.perf_counter() - t0) * 1000.0

        return UncertaintyPatternResult(
            uncertainty_map=umap,
            system_entropy=system_entropy,
            propagation_results=tuple(prop_results),
            max_chain_uncertainty=max_chain_u,
            total_amplifiers=total_amplifiers,
            total_bottlenecks=total_bottlenecks,
            patterns_detected=tuple(patterns),
            calibration_error=ece,
            overconfidence_alert=overconfidence,
            topic_alerts=(),
            epistemic_fraction=ep_frac,
            aleatoric_fraction=al_frac,
            reducible_uncertainty_claims=reducible,
            neurochemical_signals=neurochem,
            propagation_depth_used=max_depth,
            processing_time_ms=elapsed,
            engine_id=self.engine_id,
            metadata={
                "mode": self._mode.value,
                "cycle": self._cycle_count,
                "claims_analyzed": len(umap),
                "chains_analyzed": len(prop_results),
                "patterns_found": len(patterns),
            },
        )

    # ----- Introspection --------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "cluster": self.cluster,
            "mode": self._mode.value,
            "cycle_count": self._cycle_count,
            "total_analyses": self._state.total_analyses,
            "total_amplifiers_found": self._state.total_amplifiers_found,
            "total_bottlenecks_found": self._state.total_bottlenecks_found,
            "total_patterns_found": self._state.total_patterns_found,
            "overconfidence_alerts": self._state.overconfidence_alerts,
            "prev_entropy": self._prev_entropy,
            "nt_levels": {
                "ne": self._state.ne_level,
                "da": self._state.da_level,
                "ach": self._state.ach_level,
                "5ht": self._state._5ht_level,
                "cor": self._state.cor_level,
                "gaba": self._state.gaba_level,
            },
        }
