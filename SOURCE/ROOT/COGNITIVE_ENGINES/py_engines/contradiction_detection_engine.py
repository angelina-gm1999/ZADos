"""
Contradiction Detection Engine (Detection Cluster — Engine 1).

Single-responsibility detection module: identifies mutually incompatible
statements across user input, system output, and retrieved memory content.
Detects, flags, and logs — does NOT resolve or act.

Detection Levels
----------------
Level 1 — Direct Negation:
    Explicit negation of the same proposition.
Level 2 — Semantic Contradiction:
    Semantic role frames are incompatible without explicit negation.
Level 3 — Implicit Contextual Contradiction:
    Statements become incompatible only within a shared context frame,
    including temporal separation.

Stochastic Confidence Layer
----------------------------
Models contradiction detection as Bayesian inference:

    P(C | n, s, f, Δt) ∝ P(C) × P(n|C) × P(s|C) × P(f|C) × τ(Δt)

Missing evidence terms are simply omitted from the product (partial evidence).

Neurochemical Coupling
----------------------
- Emits contradiction load C(t) (leaky integral of κ_contradiction).
- C(t) → R_Logic penalty, DA D2 burst, 5-HT2C k_on, GABA reuptake, oscillatory modulation.
- Reads NE, GABA, DA D2 to adjust detection threshold θ_det bidirectionally.

Usage
-----
>>> from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
...     ContradictionDetectionEngine, ContradictionEngineConfig,
...     ProcessedStatement, SourceTag, ComparisonSet, OperationalMode,
... )
>>> config = ContradictionEngineConfig()
>>> engine = ContradictionDetectionEngine(config)
>>> stmt_a = ProcessedStatement(raw_text="The server is online.", source_tag=SourceTag.USER_INPUT)
>>> stmt_b = ProcessedStatement(raw_text="The server is not online.", source_tag=SourceTag.AI_OUTPUT)
>>> cset = ComparisonSet(primary_statements=[stmt_a], reference_set=[stmt_b])
>>> result = engine.process(cset)
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from zados.cognitive_engines.constants import _clamp


# =====================================================================
# Enumerations
# =====================================================================

class SourceTag(str, Enum):
    USER_INPUT          = "USER_INPUT"
    USER_INPUT_HISTORICAL = "USER_INPUT_HISTORICAL"
    AI_OUTPUT           = "AI_OUTPUT"
    AI_OUTPUT_HISTORICAL = "AI_OUTPUT_HISTORICAL"
    MEMORY_MID_TERM     = "MEMORY_MID_TERM"
    MEMORY_LONG_TERM    = "MEMORY_LONG_TERM"


class OperationalMode(str, Enum):
    NORMAL    = "normal"
    DEV       = "dev"
    LEARNING  = "learning"
    REFLECTIVE = "reflective"
    REM_NORMAL = "rem_normal"
    REM_DREAM  = "rem_dream"


# =====================================================================
# Configuration  (frozen dataclass — immutable)
# =====================================================================

@dataclass(frozen=True)
class ContradictionEngineConfig:
    """
    All tunable parameters for the Contradiction Detection Engine.

    Attributes
    ----------
    p_base : float
        Default base rate for contradiction prior P_base.
    alpha_mc : float
        Memory contrast activity scaling coefficient for prior.
    eps_fn : float
        False-negative rate for negation detection (Level 1).
    eps_fp : float
        False-positive rate for negation detection (Level 1).
    sigma_s : float
        NLP measurement noise std dev for semantic opposition (Level 2).
    sigma_f : float
        NLP measurement noise std dev for contextual evaluation (Level 3).
    mu_C : float
        Expected semantic opposition given true contradiction.
    mu_not_C : float
        Expected semantic opposition given no contradiction.
    mu_fC : float
        Expected contextual frame score given true contradiction.
    mu_f_not_C : float
        Expected contextual frame score given no contradiction.
    lambda_t : float
        Temporal decay rate (per day) for Δt modifier.
    theta_normal : float
        Detection threshold in Normal mode.
    theta_dev : float
        Detection threshold in Dev mode.
    theta_learning : float
        Detection threshold in Learning mode.
    theta_rem_normal : float
        Detection threshold in REM Normal mode.
    theta_rem_dream : float
        Detection threshold in REM Dream mode (raised — tolerates contradictions).
    lambda_1 : float
        κ_contradiction weight for semantic dissonance.
    lambda_2 : float
        κ_contradiction weight for logical contradiction.
    lambda_3 : float
        κ_contradiction weight for affective misalignment.
    tau_c : float
        Contradiction load decay constant (seconds).
    lambda_penalty : float
        C(t) → R_Logic penalty scaling.
    beta_contradict : float
        DA D2 burst coupling strength.
    alpha_DA : float
        Gamma distribution shape for DA burst.
    theta_DA : float
        Gamma distribution scale for DA burst.
    lambda_discord : float
        Affective misalignment → 5-HT2C k_on coupling.
    tau_aff : float
        Affective temporal convolution window (seconds).
    eta_contradict : float
        Contradiction load → GABA reuptake suppression.
    psi_contradict : float
        Contradiction load → beta band suppression.
    psi_theta : float
        Contradiction load → theta band enhancement.
    """
    # Prior
    p_base:       float = 0.10
    alpha_mc:     float = 0.50

    # Level 1 noise
    eps_fn:       float = 0.05
    eps_fp:       float = 0.02

    # Level 2 noise / means
    sigma_s:      float = 0.10
    mu_C:         float = 0.80
    mu_not_C:     float = 0.30

    # Level 3 noise / means
    sigma_f:      float = 0.15
    mu_fC:        float = 0.75
    mu_f_not_C:   float = 0.25

    # Temporal decay
    lambda_t:     float = 0.01   # per day

    # Detection thresholds per mode
    theta_normal:     float = 0.50
    theta_dev:        float = 0.30
    theta_learning:   float = 0.40
    theta_rem_normal: float = 0.50
    theta_rem_dream:  float = 0.70

    # κ_contradiction composition weights
    lambda_1:     float = 0.40   # semantic dissonance
    lambda_2:     float = 0.35   # logical contradiction
    lambda_3:     float = 0.25   # affective misalignment

    # Contradiction load
    tau_c:        float = 10.0   # seconds

    # Neurochemical coupling
    lambda_penalty:   float = 0.30
    beta_contradict:  float = 0.15
    alpha_DA:         float = 2.0
    theta_DA:         float = 0.5
    lambda_discord:   float = 0.20
    tau_aff:          float = 8.0
    eta_contradict:   float = 0.25
    psi_contradict:   float = 0.15
    psi_theta:        float = 0.10


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class ProcessedStatement:
    """
    A statement as it arrives after semantic expansion (step a).

    Attributes
    ----------
    raw_text : str
        Original text content.
    tokens : list of str
        Simple token list (space-split or NLP-produced).
    semantic_expansion : dict
        Expanded semantic associations from step (a).
    propositions : list of str
        Extracted logical propositions (predicate-argument strings).
    source_tag : SourceTag
        Origin of the statement.
    timestamp : datetime
        When the statement was produced.
    negation_markers : list of str
        Negation tokens found during NLP parse.
    semantic_roles : list of dict
        Semantic role frames (agent, action, patient, temporal, spatial).
    factuality_score : float
        Factual verifiability [0, 1]; 1.0 = clearly factual claim.
    affective_score : float
        Affective misalignment signal [0, 1]; 1.0 = high emotional charge.
    """
    raw_text:           str              = ""
    tokens:             List[str]        = field(default_factory=list)
    semantic_expansion: Dict             = field(default_factory=dict)
    propositions:       List[str]        = field(default_factory=list)
    source_tag:         SourceTag        = SourceTag.USER_INPUT
    timestamp:          datetime         = field(default_factory=datetime.utcnow)
    negation_markers:   List[str]        = field(default_factory=list)
    semantic_roles:     List[Dict]       = field(default_factory=list)
    factuality_score:   float            = 0.5
    affective_score:    float            = 0.0


@dataclass(frozen=True)
class ComparisonSet:
    """
    Input bundle for one engine invocation.

    Attributes
    ----------
    primary_statements : list of ProcessedStatement
        Current user input (post semantic expansion).
    reference_set : list of ProcessedStatement
        Memory contrast results + recent STM contents.
    memory_contrast_hit_ratio : float
        Ratio of memory contrast hits to total reference set size ∈ [0, 1].
        Used to adjust the contradiction prior.
    context_frame : dict
        Assembled context frame from short-term memory (for Level 3).
    """
    primary_statements:         List[ProcessedStatement] = field(default_factory=list)
    reference_set:              List[ProcessedStatement] = field(default_factory=list)
    memory_contrast_hit_ratio:  float                    = 0.0
    context_frame:              Dict                     = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceSignals:
    """
    Raw evidence signals from all three detection levels.

    Attributes
    ----------
    negation_signal : int or None
        Binary {0, 1} from Level 1; None if not evaluated.
    semantic_opposition : float or None
        Continuous [0, 1] from Level 2; None if not evaluated.
    contextual_incompatibility : float or None
        Continuous [0, 1] from Level 3; None if not evaluated.
    """
    negation_signal:             Optional[int]   = None
    semantic_opposition:         Optional[float] = None
    contextual_incompatibility:  Optional[float] = None


@dataclass(frozen=True)
class ContradictionFlag:
    """
    Structured output for a single detected contradiction.

    Attributes
    ----------
    contradiction_id : str
        UUID for this flag.
    statement_a : ProcessedStatement
        First conflicting statement.
    statement_b : ProcessedStatement
        Second conflicting statement.
    contradiction_level : int
        1 (direct negation), 2 (semantic), or 3 (contextual).
    confidence : float
        Posterior probability P(C | evidence, Δt) ∈ [0, 1].
    evidence_signals : EvidenceSignals
        Raw signals used in Bayesian fusion.
    temporal_separation : float or None
        Δt in days between statement timestamps.
    temporal_decay_applied : float
        τ(Δt) value applied; 1.0 if no temporal separation.
    semantic_description : str
        Brief human-readable description of the conflict.
    context_frame : dict or None
        Contextual information used for Level 3 (if applicable).
    timestamp : datetime
        When this flag was produced.
    """
    contradiction_id:    str
    statement_a:         ProcessedStatement
    statement_b:         ProcessedStatement
    contradiction_level: int
    confidence:          float
    evidence_signals:    EvidenceSignals
    temporal_separation: Optional[float]
    temporal_decay_applied: float
    semantic_description: str
    context_frame:        Optional[Dict]
    timestamp:            datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class ContradictionDetectionResult:
    """
    Full engine output for one process() call.

    Attributes
    ----------
    flags : list of ContradictionFlag
        All detected contradictions that met the threshold.
    comparisons_performed : int
        Total pairwise comparisons executed.
    clean_pass : bool
        True if no flags were emitted.
    detection_threshold_used : float
        θ_det active during this run.
    processing_time_ms : float
        Wall-clock time for the process() call.
    neurochemical_signals : dict
        Computed coupling signals ready for the neurochem write port.
    metadata : dict
        Counts and mode for downstream consumers.
    """
    flags:                    List[ContradictionFlag]
    comparisons_performed:    int
    clean_pass:               bool
    detection_threshold_used: float
    processing_time_ms:       float
    neurochemical_signals:    Dict
    metadata:                 Dict


# =====================================================================
# Mutable engine state  (bidirectional neurochem feedback)
# =====================================================================

@dataclass
class ContradictionEngineState:
    """
    Runtime state updated by neurochemical feedback.

    Attributes
    ----------
    contradiction_load : float
        C(t) — current leaky-integral contradiction load [0, ∞).
    affective_integral : float
        Running integral of AffectiveMisalignment × exp decay for 5-HT2C.
    last_tick_time : datetime or None
        Timestamp of the last process() call, used for dt computation.
    ne_level : float
        Current norepinephrine level read from neurochem port [0, 1].
    gaba_level : float
        Current GABA level [0, 1].
    da_d2_level : float
        Current DA D2 activation level [0, 1].
    """
    contradiction_load: float           = 0.0
    affective_integral: float           = 0.0
    last_tick_time:     Optional[datetime] = None
    ne_level:           float           = 0.0
    gaba_level:         float           = 0.0
    da_d2_level:        float           = 0.0

    def as_dict(self) -> Dict:
        return {
            "contradiction_load": self.contradiction_load,
            "affective_integral": self.affective_integral,
            "last_tick_time": self.last_tick_time.isoformat() if self.last_tick_time else None,
            "ne_level":        self.ne_level,
            "gaba_level":      self.gaba_level,
            "da_d2_level":     self.da_d2_level,
        }


# =====================================================================
# Pure functions — Bayesian confidence
# =====================================================================

def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
    """Unnormalized Gaussian PDF φ(x; μ, σ). σ must be > 0."""
    if sigma <= 0.0:
        return 1.0 if x == mu else 0.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def compute_prior(
    p_base: float,
    alpha_mc: float,
    memory_contrast_hit_ratio: float,
    da_d2_level: float = 0.0,
) -> float:
    """
    Compute contradiction prior P(C).

    P(C) = P_base × (1 + α_mc × r_mc)

    Bidirectional feedback: high DA D2 raises the prior (expects more
    contradictions when dopaminergic salience is elevated).

    Parameters
    ----------
    p_base : float
        Base rate (default 0.10).
    alpha_mc : float
        Memory contrast scaling coefficient.
    memory_contrast_hit_ratio : float
        r_mc ∈ [0, 1].
    da_d2_level : float
        Current DA D2 activation [0, 1]; increases prior linearly.

    Returns
    -------
    float
        Prior probability, clamped to [0, 1].
    """
    prior = p_base * (1.0 + alpha_mc * memory_contrast_hit_ratio)
    # DA D2 bidirectional feedback: scale up to 50 % extra at full activation
    prior = prior * (1.0 + 0.5 * da_d2_level)
    return _clamp(prior)


def compute_level1_likelihoods(
    n_obs: int,
    eps_fn: float,
    eps_fp: float,
) -> Tuple[float, float]:
    """
    P(n_obs | C) and P(n_obs | ¬C) for Level 1 negation signal.

    P(n_obs=1 | C)  = 1 - ε_fn   (true positive rate)
    P(n_obs=1 | ¬C) = ε_fp        (false positive rate)

    Parameters
    ----------
    n_obs : int
        Observed negation signal {0, 1}.
    eps_fn : float
        False-negative rate.
    eps_fp : float
        False-positive rate.

    Returns
    -------
    (P(n|C), P(n|¬C))
    """
    if n_obs == 1:
        return (1.0 - eps_fn, eps_fp)
    else:
        return (eps_fn, 1.0 - eps_fp)


def compute_level2_likelihoods(
    s_obs: float,
    mu_C: float,
    mu_not_C: float,
    sigma_s: float,
) -> Tuple[float, float]:
    """
    P(s_obs | C) and P(s_obs | ¬C) for Level 2 semantic opposition score.

    Uses Gaussian likelihood with NLP measurement noise σ_s.

    Parameters
    ----------
    s_obs : float
        Observed semantic opposition score [0, 1].
    mu_C, mu_not_C : float
        Expected opposition scores under contradiction / no-contradiction.
    sigma_s : float
        Measurement noise std dev.

    Returns
    -------
    (P(s|C), P(s|¬C))
    """
    p_s_given_C     = _gaussian_pdf(s_obs, mu_C,     sigma_s)
    p_s_given_not_C = _gaussian_pdf(s_obs, mu_not_C, sigma_s)
    return (p_s_given_C, p_s_given_not_C)


def compute_level3_likelihoods(
    f_obs: float,
    mu_fC: float,
    mu_f_not_C: float,
    sigma_f: float,
) -> Tuple[float, float]:
    """
    P(f_obs | C) and P(f_obs | ¬C) for Level 3 contextual incompatibility score.

    Parameters
    ----------
    f_obs : float
        Observed contextual incompatibility score [0, 1].
    mu_fC, mu_f_not_C : float
        Expected frame scores under contradiction / no-contradiction.
    sigma_f : float
        Measurement noise std dev (higher than σ_s).

    Returns
    -------
    (P(f|C), P(f|¬C))
    """
    p_f_given_C     = _gaussian_pdf(f_obs, mu_fC,     sigma_f)
    p_f_given_not_C = _gaussian_pdf(f_obs, mu_f_not_C, sigma_f)
    return (p_f_given_C, p_f_given_not_C)


def compute_temporal_decay(
    temporal_separation_days: Optional[float],
    lambda_t: float,
) -> float:
    """
    Temporal decay modifier τ(Δt) = exp(-λ_t × Δt).

    Parameters
    ----------
    temporal_separation_days : float or None
        Δt between statements in days; None → no decay (τ = 1.0).
    lambda_t : float
        Decay rate constant per day.

    Returns
    -------
    float
        τ(Δt) ∈ (0, 1].
    """
    if temporal_separation_days is None or temporal_separation_days <= 0.0:
        return 1.0
    return math.exp(-lambda_t * temporal_separation_days)


def fuse_bayesian_confidence(
    prior: float,
    evidence: EvidenceSignals,
    config: ContradictionEngineConfig,
    temporal_decay: float,
) -> float:
    """
    Fuse all evidence signals into a posterior contradiction probability.

    P(C | n, s, f, Δt) ∝ P(C) × P(n|C) × P(s|C) × P(f|C) × τ(Δt)

    Missing evidence terms (None) are omitted from the product.

    Parameters
    ----------
    prior : float
        P(C) — prior contradiction probability.
    evidence : EvidenceSignals
        Observed signals from all active detection levels.
    config : ContradictionEngineConfig
        Engine parameters.
    temporal_decay : float
        τ(Δt) modifier.

    Returns
    -------
    float
        Posterior probability ∈ [0, 1].
    """
    p_C    = prior
    p_notC = 1.0 - prior

    # Level 1
    if evidence.negation_signal is not None:
        l1_C, l1_notC = compute_level1_likelihoods(
            evidence.negation_signal, config.eps_fn, config.eps_fp
        )
        p_C    *= l1_C
        p_notC *= l1_notC

    # Level 2
    if evidence.semantic_opposition is not None:
        l2_C, l2_notC = compute_level2_likelihoods(
            evidence.semantic_opposition,
            config.mu_C, config.mu_not_C, config.sigma_s,
        )
        p_C    *= l2_C
        p_notC *= l2_notC

    # Level 3
    if evidence.contextual_incompatibility is not None:
        l3_C, l3_notC = compute_level3_likelihoods(
            evidence.contextual_incompatibility,
            config.mu_fC, config.mu_f_not_C, config.sigma_f,
        )
        p_C    *= l3_C
        p_notC *= l3_notC

    # Temporal decay applied only to the contradiction hypothesis
    p_C *= temporal_decay

    normalizer = p_C + p_notC
    if normalizer <= 0.0:
        return 0.0
    return p_C / normalizer


# =====================================================================
# Pure functions — NLP signal extraction (rule-based stubs)
# =====================================================================
# These functions implement deterministic rule-based heuristics that
# produce signals compatible with the Bayesian layer.  They operate
# on the structured fields of ProcessedStatement and do NOT require
# any external ML library — the full NLP stack (spaCy, sentence-
# transformers, NLI model) plugs in by populating those fields before
# the engine is called.
# =====================================================================

_NEGATION_TOKENS = frozenset({
    "not", "n't", "no", "never", "neither", "nor", "nobody",
    "nothing", "nowhere", "cannot", "can't", "won't", "don't",
    "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "shouldn't", "wouldn't", "couldn't",
})


def detect_negation_signal(
    stmt_a: ProcessedStatement,
    stmt_b: ProcessedStatement,
) -> int:
    """
    Level 1 — Direct Negation detection.

    Returns 1 if one statement has a negation marker that the other lacks
    AND both share at least one non-negation content token (same predicate).

    Parameters
    ----------
    stmt_a, stmt_b : ProcessedStatement
        Statements to compare.

    Returns
    -------
    int
        1 if direct negation structure detected, else 0.
    """
    tokens_a = {t.lower() for t in (stmt_a.tokens or stmt_a.raw_text.split())}
    tokens_b = {t.lower() for t in (stmt_b.tokens or stmt_b.raw_text.split())}

    neg_a = bool(tokens_a & _NEGATION_TOKENS)
    neg_b = bool(tokens_b & _NEGATION_TOKENS)

    # One negated, the other not → possible direct negation
    if neg_a == neg_b:
        return 0

    # Check shared content tokens (predicate overlap)
    content_a = tokens_a - _NEGATION_TOKENS
    content_b = tokens_b - _NEGATION_TOKENS
    shared = content_a & content_b
    return 1 if len(shared) >= 2 else 0


def compute_semantic_opposition(
    stmt_a: ProcessedStatement,
    stmt_b: ProcessedStatement,
) -> float:
    """
    Level 2 — Semantic Opposition scoring.

    Uses proposition overlap and semantic role comparison.
    Returns a score in [0, 1] where 1 = full semantic opposition.

    When a full NLP pipeline populates stmt.semantic_roles, this function
    compares agent/action/patient fields for shared referents and opposing
    values.  Without rich NLP fields, falls back to token-overlap inversion.

    Parameters
    ----------
    stmt_a, stmt_b : ProcessedStatement
        Statements with populated tokens, propositions, semantic_roles.

    Returns
    -------
    float
        Semantic opposition score s(A, B) ∈ [0, 1].
    """
    # If semantic_roles populated: compute role-level opposition
    if stmt_a.semantic_roles and stmt_b.semantic_roles:
        oppositions = []
        for role_a in stmt_a.semantic_roles:
            for role_b in stmt_b.semantic_roles:
                # Shared referent: same agent or same subject
                if role_a.get("agent") and role_a.get("agent") == role_b.get("agent"):
                    # Compare actions/predicates
                    act_a = str(role_a.get("action", "")).lower()
                    act_b = str(role_b.get("action", "")).lower()
                    if act_a and act_b:
                        # Exact same action = no opposition; different = some
                        opp = 0.0 if act_a == act_b else 0.6
                        oppositions.append(opp)
        if oppositions:
            return float(min(1.0, sum(oppositions) / len(oppositions)))

    # Fallback: Jaccard-inverted token overlap as proxy
    tokens_a = set(t.lower() for t in (stmt_a.tokens or stmt_a.raw_text.split()))
    tokens_b = set(t.lower() for t in (stmt_b.tokens or stmt_b.raw_text.split()))
    # Remove stop words and negation markers for content comparison
    stop = _NEGATION_TOKENS | {"the", "a", "an", "is", "are", "was", "were",
                                "be", "been", "being", "it", "its", "this",
                                "that", "and", "or", "but", "of", "in", "on"}
    content_a = tokens_a - stop
    content_b = tokens_b - stop
    if not content_a and not content_b:
        return 0.0
    union = content_a | content_b
    intersection = content_a & content_b
    jaccard = len(intersection) / len(union) if union else 0.0
    # High overlap with negation → high opposition; low overlap → low opposition
    # Account for negation: if one has neg markers and other does not
    neg_a = bool(tokens_a & _NEGATION_TOKENS)
    neg_b = bool(tokens_b & _NEGATION_TOKENS)
    if neg_a != neg_b and jaccard > 0.3:
        return min(1.0, jaccard * 1.5)
    return max(0.0, 1.0 - jaccard) * 0.5   # conservative without role data


def compute_contextual_incompatibility(
    stmt_a: ProcessedStatement,
    stmt_b: ProcessedStatement,
    context_frame: Dict,
) -> Optional[float]:
    """
    Level 3 — Implicit Contextual Contradiction scoring.

    Evaluates whether the context_frame makes co-truth of A and B impossible.
    Returns None if the context frame is empty (Level 3 unavailable).

    Parameters
    ----------
    stmt_a, stmt_b : ProcessedStatement
        Statements to compare.
    context_frame : dict
        Context assembled from short-term memory.

    Returns
    -------
    float or None
        Contextual incompatibility score f(A, B) ∈ [0, 1], or None.
    """
    if not context_frame:
        return None

    # Speaker consistency check: same speaker asserting opposite claims
    speaker_a = stmt_a.source_tag.value
    speaker_b = stmt_b.source_tag.value
    same_speaker = (speaker_a == speaker_b) or (
        "USER" in speaker_a and "USER" in speaker_b
    )

    # Temporal context: if both from same speaker with different timestamps
    # and content is contradictory → elevated contextual score
    base_score = 0.0
    if same_speaker:
        # Start with semantic opposition as base
        sem_opp = compute_semantic_opposition(stmt_a, stmt_b)
        base_score = sem_opp * 0.8   # contextual amplification

    # Add context-frame signals if present
    ctx_contradiction_signals = float(context_frame.get("contradiction_signal", 0.0))
    base_score = max(base_score, ctx_contradiction_signals)

    return _clamp(base_score)


def compute_temporal_separation_days(
    stmt_a: ProcessedStatement,
    stmt_b: ProcessedStatement,
) -> Optional[float]:
    """
    Compute Δt in days between two statement timestamps.

    Returns None if either timestamp is missing or difference is zero.
    """
    if stmt_a.timestamp is None or stmt_b.timestamp is None:
        return None
    delta = abs((stmt_a.timestamp - stmt_b.timestamp).total_seconds())
    days = delta / 86400.0
    return days if days > 0.0 else None


# =====================================================================
# Pure functions — Neurochemical coupling
# =====================================================================

def compute_kappa(
    semantic_dissonance: float,
    logical_contradiction_detected: float,
    affective_misalignment: float,
    lambda_1: float,
    lambda_2: float,
    lambda_3: float,
) -> float:
    """
    Compute instantaneous contradiction intensity κ_contradiction(t).

    κ(t) = λ₁ × SemanticDissonance(t)
           + λ₂ × LogicalContradictionDetected(t)
           + λ₃ × AffectiveMisalignment(t)

    Parameters
    ----------
    semantic_dissonance : float
        Mean semantic opposition score across all flagged pairs [0, 1].
    logical_contradiction_detected : float
        1.0 if Level 1 negation detected, else 0.0 (or weighted count).
    affective_misalignment : float
        Mean affective score differential across flagged pairs [0, 1].
    lambda_1, lambda_2, lambda_3 : float
        Composition weights (should sum to ≈ 1.0).

    Returns
    -------
    float
        κ_contradiction(t) ≥ 0.
    """
    return (
        lambda_1 * semantic_dissonance
        + lambda_2 * logical_contradiction_detected
        + lambda_3 * affective_misalignment
    )


def update_contradiction_load(
    current_load: float,
    kappa: float,
    dt: float,
    tau_c: float,
) -> float:
    """
    Update leaky-integral contradiction load C(t).

    Discretised: C(t+dt) = C(t) × exp(-dt/τ_c) + κ(t) × dt

    Parameters
    ----------
    current_load : float
        C(t) before this tick.
    kappa : float
        κ_contradiction(t).
    dt : float
        Timestep in seconds.
    tau_c : float
        Load decay time constant (seconds).

    Returns
    -------
    float
        Updated C(t+dt), clamped to [0, 1].
    """
    decay = math.exp(-dt / tau_c) if tau_c > 0.0 else 0.0
    new_load = current_load * decay + kappa * dt
    return _clamp(new_load)


def compute_neurochemical_signals(
    kappa: float,
    contradiction_load: float,
    affective_integral: float,
    config: ContradictionEngineConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    Compute all neurochemical coupling signals for one engine tick.

    Returns
    -------
    dict with keys:
        reward_logic_penalty : float    — R_Logic(t) subtraction
        da_d2_burst : float             — ΔC_DA(t) from Gamma-distributed burst
        k_on_5ht2c_multiplier : float   — k_on_5HT2C relative to k₀
        k_u_gaba_multiplier : float     — k_u_GABA relative to k_u₀
        beta_suppress_val : float       — multiplicative β-band factor
        theta_boost_val : float         — multiplicative θ-band factor
    """
    if rng is None:
        rng = np.random.default_rng()

    # R_Logic penalty
    reward_logic_penalty = config.lambda_penalty * contradiction_load

    # DA D2 burst: ΔC_DA = β × κ × ξ_DA,  ξ_DA ~ Gamma(α_DA, θ_DA)
    xi_da = rng.gamma(shape=config.alpha_DA, scale=config.theta_DA)
    da_d2_burst = config.beta_contradict * kappa * xi_da

    # 5-HT2C k_on modulation: k_on = k₀ × (1 + λ_discord × Aff_integral)
    k_on_5ht2c_multiplier = 1.0 + config.lambda_discord * affective_integral

    # GABA reuptake suppression: clamped to [k_u₀ × 0.3, k_u₀]
    k_u_gaba_multiplier = max(
        0.3,
        1.0 - config.eta_contradict * contradiction_load,
    )

    # Oscillatory modulation
    beta_suppress_val = 1.0 - config.psi_contradict * contradiction_load
    theta_boost_val = 1.0 + config.psi_theta * contradiction_load

    return {
        "reward_logic_penalty":    reward_logic_penalty,
        "da_d2_burst":             da_d2_burst,
        "k_on_5ht2c_multiplier":   k_on_5ht2c_multiplier,
        "k_u_gaba_multiplier":     k_u_gaba_multiplier,
        "beta_suppress":        beta_suppress_val,
        "theta_boost":       theta_boost_val,
    }


# =====================================================================
# Threshold — bidirectional neurochem feedback
# =====================================================================

def resolve_detection_threshold(
    mode: OperationalMode,
    config: ContradictionEngineConfig,
    ne_level: float = 0.0,
    gaba_level: float = 0.0,
) -> float:
    """
    Determine active detection threshold θ_det, adjusted by neurochem state.

    Bidirectional feedback rules:
    - High NE → lower θ_det (more vigilant under stress)
    - High GABA → raise θ_det (more tolerant under inhibitory dampening)

    Parameters
    ----------
    mode : OperationalMode
        Current operational mode.
    config : ContradictionEngineConfig
        Engine parameters.
    ne_level : float
        Current NE level [0, 1].
    gaba_level : float
        Current GABA level [0, 1].

    Returns
    -------
    float
        Effective θ_det, clamped to [0.05, 0.95].
    """
    base = {
        OperationalMode.NORMAL:     config.theta_normal,
        OperationalMode.DEV:        config.theta_dev,
        OperationalMode.LEARNING:   config.theta_learning,
        OperationalMode.REFLECTIVE: config.theta_normal,   # same as normal
        OperationalMode.REM_NORMAL: config.theta_rem_normal,
        OperationalMode.REM_DREAM:  config.theta_rem_dream,
    }.get(mode, config.theta_normal)

    # NE: reduces threshold by up to 0.10 at full activation
    ne_adjustment   = -0.10 * ne_level
    # GABA: raises threshold by up to 0.10 at full activation
    gaba_adjustment = +0.10 * gaba_level

    adjusted = base + ne_adjustment + gaba_adjustment
    return max(0.05, min(0.95, adjusted))


# =====================================================================
# Engine class
# =====================================================================

class ContradictionDetectionEngine:
    """
    Contradiction Detection Engine — Detection Cluster, Engine 1.

    Stateful wrapper around pure detection functions.  Maintains
    contradiction load C(t) and affective integral across calls.

    Parameters
    ----------
    config : ContradictionEngineConfig
        Immutable parameter set.
    rng : np.random.Generator, optional
        Random number generator (for stochastic DA burst).
    """

    engine_id = "contradiction_detection_engine"
    cluster   = "detection"

    def __init__(
        self,
        config: Optional[ContradictionEngineConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._config = config or ContradictionEngineConfig()
        self._rng    = rng or np.random.default_rng()
        self._state  = ContradictionEngineState()
        self._mode   = OperationalMode.NORMAL

    # ------------------------------------------------------------------
    # Configuration port
    # ------------------------------------------------------------------

    def configure(self, mode: OperationalMode) -> None:
        """Set the active operational mode."""
        self._mode = mode

    # ------------------------------------------------------------------
    # Neurochem read port
    # ------------------------------------------------------------------

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """
        Receive neurochem readout for bidirectional threshold adjustment.

        Canonical keys: ``"ne"``, ``"gaba"``, ``"da"`` (maps to DA-D2).
        """
        if "ne" in state_dict:
            self._state.ne_level = _clamp(state_dict["ne"])
        if "gaba" in state_dict:
            self._state.gaba_level = _clamp(state_dict["gaba"])
        if "da" in state_dict:
            self._state.da_d2_level = _clamp(state_dict["da"])

    # ------------------------------------------------------------------
    # Main process port
    # ------------------------------------------------------------------

    def process(
        self,
        comparison_set: ComparisonSet,
        dt: float = 1.0,
    ) -> ContradictionDetectionResult:
        """
        Run one full detection pass on the provided comparison set.

        Parameters
        ----------
        comparison_set : ComparisonSet
            Assembled input from upstream pipeline steps a–d.
        dt : float
            Timestep in seconds for contradiction load update.

        Returns
        -------
        ContradictionDetectionResult
            All flags, counts, and neurochemical coupling signals.
        """
        import time
        t_start = time.perf_counter()

        theta = resolve_detection_threshold(
            self._mode, self._config,
            ne_level=self._state.ne_level,
            gaba_level=self._state.gaba_level,
        )

        prior = compute_prior(
            self._config.p_base,
            self._config.alpha_mc,
            comparison_set.memory_contrast_hit_ratio,
            da_d2_level=self._state.da_d2_level,
        )

        primary    = comparison_set.primary_statements
        reference  = comparison_set.reference_set
        ctx        = comparison_set.context_frame

        flags: List[ContradictionFlag] = []
        comparisons = 0

        # Build list of all pairs to evaluate:
        # (a) cross-pairs: each primary vs each reference
        # (b) intra-primary: pairwise within primary set
        pairs: List[Tuple[ProcessedStatement, ProcessedStatement]] = []
        for p in primary:
            for r in reference:
                pairs.append((p, r))
        for i in range(len(primary)):
            for j in range(i + 1, len(primary)):
                pairs.append((primary[i], primary[j]))

        # Accumulators for κ_contradiction
        sem_scores: List[float] = []
        neg_scores: List[float] = []
        aff_scores: List[float] = []

        for stmt_a, stmt_b in pairs:
            comparisons += 1

            # Level 1 — Direct Negation
            n_obs = detect_negation_signal(stmt_a, stmt_b)

            # Level 2 — Semantic Opposition
            s_obs = compute_semantic_opposition(stmt_a, stmt_b)
            sem_scores.append(s_obs)

            # Level 3 — Contextual (if context frame available)
            f_obs = compute_contextual_incompatibility(stmt_a, stmt_b, ctx)

            # Temporal
            delta_t_days = compute_temporal_separation_days(stmt_a, stmt_b)
            tau_decay    = compute_temporal_decay(delta_t_days, self._config.lambda_t)

            # Evidence bundle
            evidence = EvidenceSignals(
                negation_signal=n_obs,
                semantic_opposition=s_obs,
                contextual_incompatibility=f_obs,
            )

            confidence = fuse_bayesian_confidence(
                prior, evidence, self._config, tau_decay
            )

            if confidence >= theta:
                # Determine highest active level
                if f_obs is not None and f_obs > 0.0:
                    level = 3
                elif s_obs > self._config.mu_not_C:
                    level = 2
                else:
                    level = 1

                desc = _build_semantic_description(stmt_a, stmt_b, level, confidence)

                flag = ContradictionFlag(
                    contradiction_id=str(uuid.uuid4()),
                    statement_a=stmt_a,
                    statement_b=stmt_b,
                    contradiction_level=level,
                    confidence=confidence,
                    evidence_signals=evidence,
                    temporal_separation=delta_t_days,
                    temporal_decay_applied=tau_decay,
                    semantic_description=desc,
                    context_frame=ctx if f_obs is not None else None,
                    timestamp=datetime.utcnow(),
                )
                flags.append(flag)

                neg_scores.append(float(n_obs))
                aff_scores.append(
                    abs(stmt_a.affective_score - stmt_b.affective_score)
                )

        # Compute κ and update contradiction load
        mean_sem = float(np.mean(sem_scores)) if sem_scores else 0.0
        mean_neg = float(np.mean(neg_scores)) if neg_scores else 0.0
        mean_aff = float(np.mean(aff_scores)) if aff_scores else 0.0

        kappa = compute_kappa(
            mean_sem, mean_neg, mean_aff,
            self._config.lambda_1,
            self._config.lambda_2,
            self._config.lambda_3,
        )

        self._state.contradiction_load = update_contradiction_load(
            self._state.contradiction_load, kappa, dt, self._config.tau_c
        )

        # Update affective integral (discrete leaky sum for 5-HT2C)
        decay_aff = math.exp(-dt / self._config.tau_aff) if self._config.tau_aff > 0 else 0.0
        self._state.affective_integral = (
            self._state.affective_integral * decay_aff + mean_aff * dt
        )

        neurochem = compute_neurochemical_signals(
            kappa,
            self._state.contradiction_load,
            self._state.affective_integral,
            self._config,
            rng=self._rng,
        )

        t_end = time.perf_counter()

        return ContradictionDetectionResult(
            flags=flags,
            comparisons_performed=comparisons,
            clean_pass=len(flags) == 0,
            detection_threshold_used=theta,
            processing_time_ms=(t_end - t_start) * 1000.0,
            neurochemical_signals=neurochem,
            metadata={
                "primary_statement_count": len(primary),
                "reference_set_size":      len(reference),
                "prior_used":              prior,
                "mode":                    self._mode.value,
            },
        )

    # ------------------------------------------------------------------
    # Status port
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        """Return current engine state for monitoring."""
        return {
            "engine_id":          self.engine_id,
            "mode":               self._mode.value,
            "contradiction_load": self._state.contradiction_load,
            "affective_integral": self._state.affective_integral,
            "ne_level":           self._state.ne_level,
            "gaba_level":         self._state.gaba_level,
            "da_d2_level":        self._state.da_d2_level,
        }


# =====================================================================
# Internal helpers
# =====================================================================

def _build_semantic_description(
    stmt_a: ProcessedStatement,
    stmt_b: ProcessedStatement,
    level: int,
    confidence: float,
) -> str:
    """Build a brief human-readable description of the detected conflict."""
    level_labels = {
        1: "Direct negation",
        2: "Semantic contradiction",
        3: "Contextual contradiction",
    }
    label = level_labels.get(level, "Contradiction")
    src_a = stmt_a.source_tag.value
    src_b = stmt_b.source_tag.value
    # Truncate raw text for readability
    text_a = stmt_a.raw_text[:60] + "..." if len(stmt_a.raw_text) > 60 else stmt_a.raw_text
    text_b = stmt_b.raw_text[:60] + "..." if len(stmt_b.raw_text) > 60 else stmt_b.raw_text
    return (
        f"{label} (conf={confidence:.2f}) between "
        f"[{src_a}] '{text_a}' and [{src_b}] '{text_b}'"
    )
