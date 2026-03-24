"""
Paradox Detection Engine (Detection Cluster — Engine 2).

Evaluates flagged contradictions (and independently detected semantic
tensions) to classify them as one of four paradox classes:

    R — Resolvable contradiction (not a paradox)
    A — Apparent paradox  (falsidical / veridical)
    G — Genuine paradox   (dialectical, both poles true)
    S — Structural paradox (antinomy / self-reference)

Primary input: ContradictionDetectionResult from Engine 1.
Secondary path: independent scan of semantic expansion for paradoxes
                not captured as explicit contradictions.

Philosophical Basis
-------------------
- Class R: Standard contradiction — law of non-contradiction applies.
- Class A: Quine's veridical/falsidical paradoxes — reframing dissolves.
- Class G: Hegelian dialectic — thesis/antithesis coexist productively.
- Class S: Russell/Gödel antinomies — structural self-reference.

Classification
--------------
For each candidate a feature vector (r, σ_ref, δ_abs, π) is extracted
and fed into a Bayesian classifier with Beta-distributed class-conditional
likelihoods.

    P(c | r, σ, δ, π) ∝ P(c) × Beta_r(r) × Beta_σ(σ) × Beta_δ(δ) × Beta_π(π)

Prior-encounter matching via cosine similarity of concept embeddings
shifts the prior toward the stored classification (ρ > θ_match).

Neurochemical Coupling
----------------------
Paradox load Π(t) = Σ w_class(f) × confidence(f) × tension_score(f)
- Genuine paradoxes → 5-HT2A temporal convolution (symbolic recursion)
- Structural paradoxes → CB1 baseline drift (stability + flexibility)
- Unsolved buffer → tonic DA drive (curiosity)
- Resolution events → phasic DA burst (reward prediction error)
- Genuine → θ enhancement; Structural → γ suppression

Usage
-----
>>> from zados.cognitive_engines.py_engines.paradox_detection_engine import (
...     ParadoxDetectionEngine, ParadoxEngineConfig,
...     ParadoxInput, OperationalMode,
... )
>>> from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
...     ContradictionDetectionResult, ContradictionDetectionEngine,
...     ComparisonSet, ProcessedStatement,
... )
>>> cfg = ParadoxEngineConfig()
>>> engine = ParadoxDetectionEngine(cfg)
>>> result = engine.process(paradox_input)
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
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    ContradictionDetectionResult,
    ContradictionFlag,
    OperationalMode,
    ProcessedStatement,
)


# =====================================================================
# Enumerations
# =====================================================================

class ParadoxClass(str, Enum):
    R = "R"   # Resolvable contradiction
    A = "A"   # Apparent paradox
    G = "G"   # Genuine dialectical paradox
    S = "S"   # Structural paradox (antinomy)


class ParadoxSource(str, Enum):
    CONTRADICTION_DERIVED  = "CONTRADICTION_DERIVED"
    INDEPENDENT_SINGLE     = "INDEPENDENT_SINGLE"
    INDEPENDENT_CONCEPTUAL = "INDEPENDENT_CONCEPTUAL"


# =====================================================================
# Configuration  (frozen — immutable)
# =====================================================================

@dataclass(frozen=True)
class BetaParams:
    """Alpha and beta parameters for a Beta distribution."""
    alpha: float
    beta: float


@dataclass(frozen=True)
class ClassLikelihoods:
    """
    Beta distribution parameters for each feature × class combination.

    Each attribute is a dict mapping ParadoxClass → BetaParams.
    """
    # Resolvability r(A,B)
    r_params: Dict[str, BetaParams] = field(default_factory=dict)
    # Self-reference σ_ref
    sigma_params: Dict[str, BetaParams] = field(default_factory=dict)
    # Abstraction divergence δ_abs
    delta_params: Dict[str, BetaParams] = field(default_factory=dict)
    # Dialectical productivity π
    pi_params: Dict[str, BetaParams] = field(default_factory=dict)


def _default_likelihoods() -> ClassLikelihoods:
    """Construct the spec-defined Beta parameters for all features."""
    return ClassLikelihoods(
        r_params={
            ParadoxClass.R: BetaParams(8, 2),
            ParadoxClass.A: BetaParams(5, 4),
            ParadoxClass.G: BetaParams(2, 7),
            ParadoxClass.S: BetaParams(2, 8),
        },
        sigma_params={
            ParadoxClass.R: BetaParams(1,   10),
            ParadoxClass.A: BetaParams(1,   8),
            ParadoxClass.G: BetaParams(1.5, 6),
            ParadoxClass.S: BetaParams(9,   2),
        },
        delta_params={
            ParadoxClass.R: BetaParams(3, 6),
            ParadoxClass.A: BetaParams(4, 4),
            ParadoxClass.G: BetaParams(7, 3),
            ParadoxClass.S: BetaParams(4, 5),
        },
        pi_params={
            ParadoxClass.R: BetaParams(2, 8),
            ParadoxClass.A: BetaParams(3, 5),
            ParadoxClass.G: BetaParams(8, 2),
            ParadoxClass.S: BetaParams(2, 6),
        },
    )


@dataclass(frozen=True)
class ParadoxEngineConfig:
    """
    All tunable parameters for the Paradox Detection Engine.

    Attributes
    ----------
    prior_R, prior_A, prior_G, prior_S : float
        Default class priors (must sum to 1.0).
    theta_normal, theta_dev, theta_learning, theta_rem_normal,
    theta_rem_dream : float
        Mode-dependent minimum confidence for emitting a flag.
    theta_oxy : float
        Minimum semantic opposition for oxymoron detection in Path 2.
    theta_conceptual : float
        Minimum π for concept-level paradox detection.
    theta_match : float
        Minimum cosine similarity for prior-encounter buffer match.
    theta_evidence : float
        Minimum similarity for evidence accumulation.
    theta_resolve : float
        Resolution proximity threshold → priority REM scheduling.
    w_r1..w_r4 : float
        Resolvability sub-feature weights.
    w_pi1..w_pi3 : float
        Dialectical productivity sub-feature weights.
    w_G, w_A, w_S, w_R : float
        Paradox load Π(t) class weights.
    lambda_paradox : float
        Π_G → 5-HT2A temporal convolution coupling.
    tau_sym : float
        5-HT2A convolution time window (seconds).
    rho_structural : float
        Π_S → CB1 baseline drift coupling.
    rho_identity : float
        High σ_ref → CB1 identity-protection coupling.
    gamma_CB1 : float
        CB1 relaxation rate to baseline.
    beta_curiosity : float
        Buffer salience → tonic DA coupling.
    beta_resolution : float
        Resolution event → phasic DA coupling.
    alpha_DA_res : float
        Gamma shape for phasic DA resolution burst.
    theta_DA_res : float
        Gamma scale for phasic DA resolution burst.
    psi_theta_paradox : float
        Π_G → θ-band multiplicative enhancement.
    psi_gamma_paradox : float
        Π_S → γ-band multiplicative suppression.
    delta_coherence : float
        Apparent resolution → Θγ coherence boost.
    m0_G, m0_A, m0_S : float
        Base motivational salience for buffer entries.
    alpha_attempt : float
        Salience growth per failed resolution attempt.
    lambda_age : float
        Age-dependent salience growth rate (per day).
    """
    # Priors
    prior_R: float = 0.60
    prior_A: float = 0.20
    prior_G: float = 0.12
    prior_S: float = 0.08

    # Detection thresholds per mode
    theta_normal:     float = 0.50
    theta_dev:        float = 0.30
    theta_learning:   float = 0.40
    theta_rem_normal: float = 0.45
    theta_rem_dream:  float = 0.25

    # Path 2 thresholds
    theta_oxy:        float = 0.70
    theta_conceptual: float = 0.60
    theta_match:      float = 0.80
    theta_evidence:   float = 0.60
    theta_resolve:    float = 0.80

    # Resolvability sub-weights
    w_r1: float = 0.30   # temporal
    w_r2: float = 0.20   # specificity
    w_r3: float = 0.30   # factual
    w_r4: float = 0.20   # contextual

    # Dialectical productivity sub-weights
    w_pi1: float = 0.30   # synthesis potential
    w_pi2: float = 0.40   # conceptual entailment
    w_pi3: float = 0.30   # persistence

    # Paradox load weights
    w_G: float = 0.60
    w_A: float = 0.20
    w_S: float = 0.40
    w_R: float = 0.05

    # 5-HT2A coupling
    lambda_paradox: float = 0.25
    tau_sym:        float = 12.0

    # CB1 coupling
    rho_structural: float = 0.18
    rho_identity:   float = 0.12
    gamma_CB1:      float = 0.10

    # DA coupling
    beta_curiosity:  float = 0.08
    beta_resolution: float = 0.30
    alpha_DA_res:    float = 3.0
    theta_DA_res:    float = 0.6

    # Oscillatory coupling
    psi_theta_paradox: float = 0.15
    psi_gamma_paradox: float = 0.10
    delta_coherence:   float = 0.08

    # Buffer salience model
    m0_G:          float = 0.30
    m0_A:          float = 0.15
    m0_S:          float = 0.10
    alpha_attempt: float = 0.15
    lambda_age:    float = 0.05   # per day


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class UnsolvedParadoxEntry:
    """
    An entry in the Unsolved Concepts Buffer for unresolved paradoxes.

    Attributes
    ----------
    paradox_flag : ParadoxFlag
        The originating flag.
    attempt_count : int
        Number of failed REM resolution attempts.
    first_encountered : datetime
        When this entry was first created.
    last_attempt : datetime or None
        Timestamp of the most recent resolution attempt.
    associated_evidence : list of dict
        Accumulated evidence segments with relevance scores.
    motivational_salience : float
        M(t) ∈ [0, 1] — drives DA tonic coupling.
    resolution_proximity : float
        Estimated closeness to resolution [0, 1].
    quarantined : bool
        True for Class S structural paradoxes.
    quarantine_reason : str or None
        Description of the quarantine reason.
    """
    paradox_flag:         "ParadoxFlag"
    attempt_count:        int                  = 0
    first_encountered:    datetime             = field(default_factory=datetime.utcnow)
    last_attempt:         Optional[datetime]   = None
    associated_evidence:  List[Dict]           = field(default_factory=list)
    motivational_salience: float               = 0.0
    resolution_proximity: float               = 0.0
    quarantined:          bool                 = False
    quarantine_reason:    Optional[str]        = None

    def tick_attempt(self) -> None:
        """Increment attempt counter and update last attempt timestamp."""
        self.attempt_count += 1
        self.last_attempt = datetime.utcnow()

    def add_evidence(
        self,
        content: str,
        source: str,
        relevance_score: float,
    ) -> None:
        """Append a new evidence segment."""
        self.associated_evidence.append({
            "content":        content,
            "source":         source,
            "timestamp":      datetime.utcnow().isoformat(),
            "relevance_score": relevance_score,
        })


@dataclass(frozen=True)
class ParadoxFeatures:
    """
    Feature vector for Bayesian classification of one candidate.

    Attributes
    ----------
    resolvability : float
        r(A, B) ∈ [0, 1] — higher = more resolvable.
    self_reference : float
        σ_ref(X) ∈ [0, 1] — higher = more self-referential.
    abstraction_divergence : float
        δ_abs(A, B) ∈ [0, 1] — higher = greater abstraction gap.
    dialectical_productivity : float
        π(A, B) ∈ [0, 1] — higher = more dialectically productive.
    prior_encounter_match : float
        ρ(A, B) ∈ [0, 1] — cosine similarity to buffer entries.
    """
    resolvability:           float = 0.5
    self_reference:          float = 0.0
    abstraction_divergence:  float = 0.5
    dialectical_productivity: float = 0.5
    prior_encounter_match:   float = 0.0


@dataclass(frozen=True)
class ParadoxFlag:
    """
    Structured output for one detected paradox.

    Attributes
    ----------
    paradox_id : str
        UUID for this flag.
    source : ParadoxSource
        Whether derived from a contradiction flag or independent detection.
    source_contradiction : ContradictionFlag or None
        The contradiction this was derived from (if any).
    concept_a, concept_b : str
        The two poles of the paradox.
    paradox_class : ParadoxClass
        Winning classification: R, A, G, or S.
    class_posterior : dict mapping str → float
        Full posterior distribution across all four classes.
    confidence : float
        P(class*) — probability of the winning class.
    features : ParadoxFeatures
        Feature vector used in classification.
    resolution_hint : str or None
        If Class A: suggested reframing direction.
    structural_pattern : str or None
        If Class S: identified structural pattern label.
    symbolic_tension_score : float
        π for Class G; 0.0 for Class R.
    timestamp : datetime
        When the flag was produced.
    """
    paradox_id:            str
    source:                ParadoxSource
    source_contradiction:  Optional[ContradictionFlag]
    concept_a:             str
    concept_b:             str
    paradox_class:         ParadoxClass
    class_posterior:       Dict[str, float]
    confidence:            float
    features:              ParadoxFeatures
    resolution_hint:       Optional[str]
    structural_pattern:    Optional[str]
    symbolic_tension_score: float
    timestamp:             datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class ParadoxDetectionResult:
    """
    Full engine output for one process() call.

    Attributes
    ----------
    flags : list of ParadoxFlag
        All flags that met the confidence threshold.
    contradictions_evaluated : int
        Number of ContradictionFlags evaluated (Path 1).
    paradoxes_from_contradictions : int
        Path 1 flags emitted.
    independent_paradoxes : int
        Path 2 flags emitted.
    genuine_count : int
        Class G flags.
    structural_count : int
        Class S flags.
    apparent_count : int
        Class A flags.
    reclassified_as_resolvable : int
        Inputs classified as Class R (not emitted as paradox flags).
    clean_pass : bool
        True if no G or S flags were emitted.
    detection_threshold_used : float
        Active θ during this run.
    processing_time_ms : float
        Wall-clock time.
    neurochemical_signals : dict
        Paradox load decomposition and all coupling outputs.
    metadata : dict
        Mode and scan details.
    """
    flags:                       List[ParadoxFlag]
    contradictions_evaluated:    int
    paradoxes_from_contradictions: int
    independent_paradoxes:       int
    genuine_count:               int
    structural_count:            int
    apparent_count:              int
    reclassified_as_resolvable:  int
    clean_pass:                  bool
    detection_threshold_used:    float
    processing_time_ms:          float
    neurochemical_signals:       Dict
    metadata:                    Dict


@dataclass(frozen=True)
class IntentionVector:
    """Minimal representation of the upstream intention map output."""
    primary_intent: str = ""
    intent_scores:  Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ParadoxInput:
    """
    Input bundle for one engine invocation.

    Attributes
    ----------
    contradiction_result : ContradictionDetectionResult
        Primary upstream input.
    semantic_expansion : dict
        From pipeline step (a).
    intention_vector : IntentionVector
        From pipeline step (c).
    memory_context : list of ProcessedStatement
        From pipeline step (d).
    unsolved_buffer : list of UnsolvedParadoxEntry
        Previously logged unresolved paradoxes.
    active_mode : OperationalMode
        Current system mode.
    """
    contradiction_result: ContradictionDetectionResult
    semantic_expansion:   Dict                         = field(default_factory=dict)
    intention_vector:     IntentionVector              = field(default_factory=IntentionVector)
    memory_context:       List[ProcessedStatement]     = field(default_factory=list)
    unsolved_buffer:      List[UnsolvedParadoxEntry]   = field(default_factory=list)
    active_mode:          OperationalMode              = OperationalMode.NORMAL


# =====================================================================
# Mutable engine state
# =====================================================================

@dataclass
class ParadoxEngineState:
    """
    Runtime state for neurochemical feedback.

    Attributes
    ----------
    ht2a_integral : float
        Running integral of Π_G × exp-decay for 5-HT2A coupling.
    cb1_level : float
        Current CB1 activation baseline.
    last_tick_time : datetime or None
        Timestamp of the last process() call.
    ht2a_activation : float
        Read from neurochem port [0, 1].
    cb1_read : float
        Read from neurochem port [0, 1].
    da_tonic : float
        Tonic DA level read from neurochem port [0, 1].
    ne_level : float
        NE level read from neurochem port [0, 1].
    """
    ht2a_integral:   float              = 0.0
    cb1_level:       float              = 0.0
    last_tick_time:  Optional[datetime] = None
    ht2a_activation: float              = 0.0
    cb1_read:        float              = 0.0
    da_tonic:        float              = 0.0
    ne_level:        float              = 0.0

    def as_dict(self) -> Dict:
        return {
            "ht2a_integral":   self.ht2a_integral,
            "cb1_level":       self.cb1_level,
            "last_tick_time":  self.last_tick_time.isoformat() if self.last_tick_time else None,
            "ht2a_activation": self.ht2a_activation,
            "cb1_read":        self.cb1_read,
            "da_tonic":        self.da_tonic,
            "ne_level":        self.ne_level,
        }


# =====================================================================
# Pure functions — Feature extraction
# =====================================================================

def compute_resolvability(
    stmt_a: ProcessedStatement,
    stmt_b: ProcessedStatement,
    config: ParadoxEngineConfig,
    temporal_separation_days: Optional[float] = None,
) -> float:
    """
    Compute resolvability score r(A, B) ∈ [0, 1].

    r = w_r1 × r_temporal + w_r2 × r_specificity
      + w_r3 × r_factual  + w_r4 × r_contextual

    Higher score = more resolvable (Class R territory).

    Parameters
    ----------
    stmt_a, stmt_b : ProcessedStatement
    config : ParadoxEngineConfig
    temporal_separation_days : float or None
        Δt between statements in days.

    Returns
    -------
    float
        r(A, B) ∈ [0, 1].
    """
    # r_temporal: sigmoid of Δt/τ_temporal (τ_temporal = 30 days default)
    tau_temporal = 30.0
    if temporal_separation_days is not None and temporal_separation_days > 0:
        r_temporal = 1.0 / (1.0 + math.exp(-temporal_separation_days / tau_temporal))
    else:
        r_temporal = 0.5   # unknown temporal gap → neutral

    # r_specificity: |spec(A) - spec(B)| / max(spec(A), spec(B))
    # Proxy: longer statements tend to be more specific
    len_a = max(1, len(stmt_a.tokens or stmt_a.raw_text.split()))
    len_b = max(1, len(stmt_b.tokens or stmt_b.raw_text.split()))
    spec_a = min(1.0, len_a / 20.0)
    spec_b = min(1.0, len_b / 20.0)
    denom = max(spec_a, spec_b)
    r_specificity = abs(spec_a - spec_b) / denom if denom > 0 else 0.0

    # r_factual: average factuality score
    r_factual = (stmt_a.factuality_score + stmt_b.factuality_score) / 2.0

    # r_contextual: average ambiguity proxy (1 - factuality as stand-in)
    r_contextual = 1.0 - r_factual

    r = (
        config.w_r1 * r_temporal
        + config.w_r2 * r_specificity
        + config.w_r3 * r_factual
        + config.w_r4 * r_contextual
    )
    return _clamp(r)


def compute_self_reference_index(text: str, tokens: Optional[List[str]] = None) -> float:
    """
    Compute self-reference index σ_ref(X) ∈ [0, 1].

    σ_ref = max(σ_explicit, σ_type, σ_regress)

    Detection is heuristic (rule-based) — the full NLP pipeline
    (coreference resolution) would populate richer signals.

    Parameters
    ----------
    text : str
        Raw statement text.
    tokens : list of str, optional
        Pre-tokenized form.

    Returns
    -------
    float
        σ_ref ∈ [0, 1].
    """
    text_lower = text.lower()
    toks = {t.lower() for t in (tokens or text.split())}

    # σ_explicit: explicit self-reference markers
    self_ref_phrases = [
        "this statement", "this sentence", "this claim",
        "this assertion", "itself", "self-referential",
        "i am lying", "this is false", "this is true",
    ]
    sigma_explicit = 0.9 if any(p in text_lower for p in self_ref_phrases) else 0.0

    # σ_type: type-level confusion — statement quantifies over all statements
    type_markers = ["all statements", "every statement", "no statement",
                    "all claims", "all rules", "all sets"]
    sigma_type = 0.8 if any(m in text_lower for m in type_markers) else 0.0

    # σ_regress: infinite regress — "the truth of X depends on X"
    regress_markers = ["depends on itself", "refers to itself",
                       "defines itself", "proves itself", "requires itself"]
    sigma_regress = 0.85 if any(m in text_lower for m in regress_markers) else 0.0

    # Additional: "liar paradox" linguistic pattern
    liar_signals = ["is false", "is a lie", "is untrue"]
    self_refs = {"this", "statement", "sentence", "claim", "i", "what i said"}
    has_liar = (
        any(s in text_lower for s in liar_signals)
        and bool(toks & self_refs)
    )
    if has_liar:
        sigma_explicit = max(sigma_explicit, 0.9)

    return _clamp(max(sigma_explicit, sigma_type, sigma_regress))


def compute_abstraction_divergence(
    stmt_a: ProcessedStatement,
    stmt_b: ProcessedStatement,
) -> float:
    """
    Compute abstraction level divergence δ_abs(A, B) ∈ [0, 1].

    level(X) = w_a1 × (1 - concreteness) + w_a2 × hypernym_depth
                + w_a3 × quantifier_scope

    Concreteness proxy: factuality_score (factual = concrete).
    Hypernym depth proxy: average token length (longer = more abstract).
    Quantifier scope proxy: presence of universal quantifiers.

    Parameters
    ----------
    stmt_a, stmt_b : ProcessedStatement

    Returns
    -------
    float
        δ_abs ∈ [0, 1].
    """
    w_a1, w_a2, w_a3 = 0.40, 0.30, 0.30
    max_level = 1.0

    def _level(stmt: ProcessedStatement) -> float:
        # (1 - concreteness): factuality ≈ concreteness
        concreteness = stmt.factuality_score
        l_concrete = 1.0 - concreteness

        # hypernym depth proxy: avg token length normalised
        toks = stmt.tokens or stmt.raw_text.split()
        avg_len = sum(len(t) for t in toks) / max(1, len(toks))
        l_hyp = min(1.0, avg_len / 10.0)

        # quantifier scope: universal quantifiers present
        text_lower = stmt.raw_text.lower()
        universals = {"all", "every", "always", "never", "none", "no one",
                      "everyone", "everything", "nothing", "anywhere"}
        toks_lower = {t.lower() for t in toks}
        l_quant = 1.0 if bool(toks_lower & universals) else 0.0

        return w_a1 * l_concrete + w_a2 * l_hyp + w_a3 * l_quant

    level_a = _level(stmt_a)
    level_b = _level(stmt_b)
    return min(1.0, abs(level_a - level_b) / max_level)


def compute_dialectical_productivity(
    stmt_a: ProcessedStatement,
    stmt_b: ProcessedStatement,
    config: ParadoxEngineConfig,
) -> float:
    """
    Compute dialectical productivity score π(A, B) ∈ [0, 1].

    π = w_π1 × synthesis_potential + w_π2 × conceptual_entailment
      + w_π3 × persistence

    Synthesis potential: 1 - |sim(M, A) - sim(M, B)| where M is the
    semantic midpoint (approximated by token union / 2).
    Conceptual entailment: bidirectional token inclusion proxy.
    Persistence: ratio of opposition across token permutations.

    Parameters
    ----------
    stmt_a, stmt_b : ProcessedStatement
    config : ParadoxEngineConfig

    Returns
    -------
    float
        π(A, B) ∈ [0, 1].
    """
    toks_a = set(t.lower() for t in (stmt_a.tokens or stmt_a.raw_text.split()))
    toks_b = set(t.lower() for t in (stmt_b.tokens or stmt_b.raw_text.split()))

    if not toks_a or not toks_b:
        return 0.5

    # Synthesis potential: how symmetric both poles are relative to midpoint
    union = toks_a | toks_b
    inter = toks_a & toks_b
    sim_a_mid = len(toks_a & union) / len(union)
    sim_b_mid = len(toks_b & union) / len(union)
    synthesis_potential = 1.0 - abs(sim_a_mid - sim_b_mid)

    # Conceptual entailment: bidirectional subset signal
    entail_a_to_b = len(inter) / len(toks_a) if toks_a else 0.0
    entail_b_to_a = len(inter) / len(toks_b) if toks_b else 0.0
    conceptual_entailment = min(entail_a_to_b, entail_b_to_a) * 2.0   # spec: × 2

    # Persistence: opposition consistency proxy — content token non-overlap
    non_shared_ratio = 1.0 - (len(inter) / len(union)) if union else 0.0
    persistence = non_shared_ratio   # high uniqueness = persistent opposition

    pi = (
        config.w_pi1 * synthesis_potential
        + config.w_pi2 * conceptual_entailment
        + config.w_pi3 * persistence
    )
    return _clamp(pi)


def compute_prior_encounter_match(
    concept_a: str,
    concept_b: str,
    unsolved_buffer: List[UnsolvedParadoxEntry],
    theta_match: float,
) -> Tuple[float, Optional[ParadoxClass]]:
    """
    Check the Unsolved Concepts Buffer for a prior encounter match.

    Uses token-overlap cosine similarity (no external embedding needed).
    Returns (ρ, stored_class) where ρ is the match score.

    Parameters
    ----------
    concept_a, concept_b : str
        Poles of the current candidate.
    unsolved_buffer : list of UnsolvedParadoxEntry
        Existing buffer entries.
    theta_match : float
        Minimum similarity for a match.

    Returns
    -------
    (float, ParadoxClass or None)
        Match score ρ and stored class if ρ > theta_match.
    """
    if not unsolved_buffer:
        return 0.0, None

    query_toks = set((concept_a + " " + concept_b).lower().split())
    best_sim = 0.0
    best_class: Optional[ParadoxClass] = None

    for entry in unsolved_buffer:
        stored_a = entry.paradox_flag.concept_a
        stored_b = entry.paradox_flag.concept_b
        stored_toks = set((stored_a + " " + stored_b).lower().split())
        union = query_toks | stored_toks
        inter = query_toks & stored_toks
        sim = len(inter) / len(union) if union else 0.0
        if sim > best_sim:
            best_sim = sim
            best_class = entry.paradox_flag.paradox_class

    if best_sim >= theta_match:
        return best_sim, best_class
    return best_sim, None


# =====================================================================
# Pure functions — Bayesian classifier
# =====================================================================

def _log_beta(a: float, b: float) -> float:
    """log B(a, b) = log Γ(a) + log Γ(b) − log Γ(a+b) via math.lgamma."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _beta_pdf_safe(x: float, a: float, b: float) -> float:
    """
    Beta(a, b) PDF evaluated at x, clamped to (ε, 1-ε) to avoid boundary issues.

    PDF(x; a, b) = x^(a-1) * (1-x)^(b-1) / B(a, b)
    Computed in log-space for numerical stability.
    """
    x_clamped = max(1e-9, min(1.0 - 1e-9, x))
    log_pdf = (
        (a - 1.0) * math.log(x_clamped)
        + (b - 1.0) * math.log(1.0 - x_clamped)
        - _log_beta(a, b)
    )
    return math.exp(log_pdf)


def compute_class_posteriors(
    features: ParadoxFeatures,
    config: ParadoxEngineConfig,
    likelihoods: ClassLikelihoods,
    unsolved_buffer: Optional[List[UnsolvedParadoxEntry]] = None,
    concept_a: str = "",
    concept_b: str = "",
) -> Dict[str, float]:
    """
    Compute normalised posterior probabilities for all four classes.

    P(c | r, σ, δ, π) ∝ P(c) × Beta_r(r) × Beta_σ(σ) × Beta_δ(δ) × Beta_π(π)

    Prior-encounter matching shifts P(c) toward stored classification
    when ρ > θ_match.

    Parameters
    ----------
    features : ParadoxFeatures
        Feature vector (r, σ, δ, π, ρ).
    config : ParadoxEngineConfig
    likelihoods : ClassLikelihoods
    unsolved_buffer : list, optional
    concept_a, concept_b : str
        Used for prior-encounter lookup.

    Returns
    -------
    dict mapping ParadoxClass.value → float
        Normalised posterior for each class.
    """
    base_priors = {
        ParadoxClass.R: config.prior_R,
        ParadoxClass.A: config.prior_A,
        ParadoxClass.G: config.prior_G,
        ParadoxClass.S: config.prior_S,
    }

    # Prior-encounter adjustment
    rho = features.prior_encounter_match
    stored_class: Optional[ParadoxClass] = None
    if unsolved_buffer and concept_a:
        rho, stored_class = compute_prior_encounter_match(
            concept_a, concept_b,
            unsolved_buffer,
            config.theta_match,
        )

    priors = {}
    for cls in ParadoxClass:
        if stored_class is not None and rho >= config.theta_match:
            indicator = 1.0 if cls == stored_class else 0.0
            priors[cls] = (1.0 - rho) * base_priors[cls] + rho * indicator
        else:
            priors[cls] = base_priors[cls]

    # Compute unnormalised posteriors
    unnorm: Dict[ParadoxClass, float] = {}
    for cls in ParadoxClass:
        r_p  = _beta_pdf_safe(features.resolvability,
                               likelihoods.r_params[cls].alpha,
                               likelihoods.r_params[cls].beta)
        s_p  = _beta_pdf_safe(features.self_reference,
                               likelihoods.sigma_params[cls].alpha,
                               likelihoods.sigma_params[cls].beta)
        d_p  = _beta_pdf_safe(features.abstraction_divergence,
                               likelihoods.delta_params[cls].alpha,
                               likelihoods.delta_params[cls].beta)
        pi_p = _beta_pdf_safe(features.dialectical_productivity,
                               likelihoods.pi_params[cls].alpha,
                               likelihoods.pi_params[cls].beta)
        unnorm[cls] = priors[cls] * r_p * s_p * d_p * pi_p

    total = sum(unnorm.values())
    if total <= 0.0:
        # Uniform fallback
        return {cls.value: 0.25 for cls in ParadoxClass}

    return {cls.value: unnorm[cls] / total for cls in ParadoxClass}


# =====================================================================
# Pure functions — Path 2 (independent detection)
# =====================================================================

def detect_oxymoron_candidates(
    stmt: ProcessedStatement,
    theta_oxy: float,
) -> List[Tuple[str, str, float]]:
    """
    Path 2 — Single-Statement: Oxymoron detection.

    Scans modifier-head pairs for semantic opposition using a heuristic
    antonym vocabulary.  Returns candidates as (modifier, head, score).

    Parameters
    ----------
    stmt : ProcessedStatement
    theta_oxy : float
        Minimum opposition score to flag.

    Returns
    -------
    list of (modifier, head, opposition_score)
    """
    # Heuristic antonym pairs (representative — full implementation would
    # use WordNet or a dedicated antonym database)
    antonym_pairs = {
        frozenset({"silent", "sound"}),
        frozenset({"living", "dead"}),
        frozenset({"dark", "light"}),
        frozenset({"cruel", "kind"}),
        frozenset({"bitter", "sweet"}),
        frozenset({"hot", "cold"}),
        frozenset({"open", "secret"}),
        frozenset({"alone", "together"}),
        frozenset({"beginning", "end"}),
        frozenset({"beautiful", "ugly"}),
        frozenset({"loving", "hate"}),
        frozenset({"freedom", "constraint"}),
        frozenset({"controlled", "chaos"}),
        frozenset({"honest", "lie"}),
        frozenset({"order", "anarchy"}),
    }

    toks = [t.lower() for t in (stmt.tokens or stmt.raw_text.split())]
    candidates = []
    for i, tok_a in enumerate(toks):
        for j, tok_b in enumerate(toks):
            if i >= j:
                continue
            pair = frozenset({tok_a, tok_b})
            if pair in antonym_pairs:
                candidates.append((tok_a, tok_b, 0.85))   # fixed score for known pairs

    return [(m, h, s) for m, h, s in candidates if s >= theta_oxy]


def detect_single_statement_self_reference(
    stmt: ProcessedStatement,
) -> float:
    """
    Path 2 — Single-Statement: Self-reference scan.

    Re-uses the σ_ref computation on an individual statement.

    Returns
    -------
    float
        σ_ref ∈ [0, 1].
    """
    return compute_self_reference_index(stmt.raw_text, stmt.tokens)


def detect_universal_particular_tension(stmt: ProcessedStatement) -> float:
    """
    Path 2 — Single-Statement: Universal-particular tension.

    Detects statements making universal claims while embedding particular
    exceptions ("All rules have exceptions").

    Returns
    -------
    float
        Tension score ∈ [0, 1].
    """
    text_lower = stmt.raw_text.lower()
    universals = {"all", "every", "always", "never", "none", "no one",
                  "everyone", "everything", "nothing"}
    exceptions = {"except", "exception", "but", "however", "unless",
                  "sometimes", "occasionally", "in some cases"}
    toks = {t.lower() for t in (stmt.tokens or stmt.raw_text.split())}
    has_universal  = bool(toks & universals)
    has_exception  = bool(toks & exceptions)
    if has_universal and has_exception:
        return 0.75
    return 0.0


# =====================================================================
# Pure functions — Neurochemical coupling
# =====================================================================

def compute_paradox_load(
    flags: List[ParadoxFlag],
    config: ParadoxEngineConfig,
) -> Dict[str, float]:
    """
    Compute paradox load Π(t) decomposed by class.

    Π(t) = Σ_f w_class(f) × confidence(f) × tension_score(f)

    Returns dict with keys: Π_G, Π_A, Π_S, Π_R, Π_total.
    """
    weights = {
        ParadoxClass.G: config.w_G,
        ParadoxClass.A: config.w_A,
        ParadoxClass.S: config.w_S,
        ParadoxClass.R: config.w_R,
    }
    loads: Dict[ParadoxClass, float] = {c: 0.0 for c in ParadoxClass}

    for flag in flags:
        tension = flag.symbolic_tension_score if flag.symbolic_tension_score > 0 else flag.confidence
        contribution = weights[flag.paradox_class] * flag.confidence * tension
        loads[flag.paradox_class] += contribution

    return {
        "Π_G":     loads[ParadoxClass.G],
        "Π_A":     loads[ParadoxClass.A],
        "Π_S":     loads[ParadoxClass.S],
        "Π_R":     loads[ParadoxClass.R],
        "Π_total": sum(loads.values()),
    }


def compute_motivational_salience(
    entry: UnsolvedParadoxEntry,
    config: ParadoxEngineConfig,
) -> float:
    """
    M(t) = M_0 × (1 + α_attempt × n_attempts) × (1 - exp(-λ_age × age_days))

    Clamped to [0, 1].

    Parameters
    ----------
    entry : UnsolvedParadoxEntry
    config : ParadoxEngineConfig

    Returns
    -------
    float
        Motivational salience M(t) ∈ [0, 1].
    """
    m0_map = {
        ParadoxClass.G: config.m0_G,
        ParadoxClass.A: config.m0_A,
        ParadoxClass.S: config.m0_S,
        ParadoxClass.R: 0.0,
    }
    m0 = m0_map.get(entry.paradox_flag.paradox_class, 0.0)
    n  = entry.attempt_count
    age_days = (datetime.utcnow() - entry.first_encountered).total_seconds() / 86400.0

    age_factor = 1.0 - math.exp(-config.lambda_age * age_days)
    salience = m0 * (1.0 + config.alpha_attempt * n) * age_factor
    return _clamp(salience)


def update_ht2a_integral(
    current_integral: float,
    pi_G: float,
    dt: float,
    tau_sym: float,
) -> float:
    """
    Update the leaky integral for 5-HT2A coupling.

    d/dt integral ≈ integral × exp(-dt/τ_sym) + Π_G × dt
    """
    decay = math.exp(-dt / tau_sym) if tau_sym > 0 else 0.0
    return current_integral * decay + pi_G * dt


def update_cb1_level(
    current_cb1: float,
    pi_S: float,
    sigma_ref: float,
    config: ParadoxEngineConfig,
    dt: float,
) -> float:
    """
    Update CB1 baseline level.

    dR_CB1/dt = ρ_structural × Π_S + ρ_identity × I(σ_ref > 0.7)
                - γ_CB1 × (R_CB1 - R₀)

    Discretised: CB1(t+dt) = CB1(t) + dt × dCB1/dt, clamped to [0, 1].
    """
    identity_signal = 1.0 if sigma_ref > 0.7 else 0.0
    r0 = 0.0   # baseline CB1 (neutral)
    d_cb1 = (
        config.rho_structural * pi_S
        + config.rho_identity * identity_signal
        - config.gamma_CB1 * (current_cb1 - r0)
    )
    new_cb1 = current_cb1 + dt * d_cb1
    return _clamp(new_cb1)


def compute_da_tonic_drive(
    unsolved_buffer: List[UnsolvedParadoxEntry],
    config: ParadoxEngineConfig,
) -> float:
    """
    ΔC_DA_tonic = β_curiosity × Σ_p M_p(t) × (1 - quarantined_p)

    Excludes quarantined (Class S) entries.
    """
    total = 0.0
    for entry in unsolved_buffer:
        if not entry.quarantined:
            salience = compute_motivational_salience(entry, config)
            total += salience
    return config.beta_curiosity * total


def compute_da_phasic_burst(
    resolved_entry: UnsolvedParadoxEntry,
    config: ParadoxEngineConfig,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    ΔC_DA_phasic = β_resolution × M_p(t_resolve) × n_attempts × ξ_DA

    ξ_DA ~ Gamma(α_DA_res, θ_DA_res).
    Reward prediction error: longer, harder problems → bigger burst.
    """
    if rng is None:
        rng = np.random.default_rng()
    salience = compute_motivational_salience(resolved_entry, config)
    xi_da = rng.gamma(shape=config.alpha_DA_res, scale=config.theta_DA_res)
    return config.beta_resolution * salience * resolved_entry.attempt_count * xi_da


def compute_paradox_neurochem_signals(
    pi_decomp: Dict[str, float],
    ht2a_integral: float,
    cb1_level: float,
    unsolved_buffer: List[UnsolvedParadoxEntry],
    config: ParadoxEngineConfig,
    flags: Optional[List[ParadoxFlag]] = None,
) -> Dict:
    """
    Assemble all neurochemical coupling signals for the write port.

    Returns
    -------
    dict with keys:
        pi_decomposition     — {Π_G, Π_A, Π_S, Π_R, Π_total}
        k_on_5ht2a_mult      — k_on_5HT2A relative multiplier
        cb1_baseline         — updated CB1 level
        da_tonic_drive       — tonic DA delta
        theta_boost_val      — multiplicative θ-band factor
        gamma_suppress_val   — multiplicative γ-band factor
    """
    pi_G = pi_decomp.get("Π_G", 0.0)
    pi_S = pi_decomp.get("Π_S", 0.0)

    # 5-HT2A: k_on = k₀ × (1 + λ × integral)
    k_on_5ht2a_mult = 1.0 + config.lambda_paradox * ht2a_integral

    # CB1 (latest tick already applied externally; expose current level)
    cb1_baseline = cb1_level

    # DA tonic
    da_tonic_drive = compute_da_tonic_drive(unsolved_buffer, config)

    # Theta enhancement: genuine paradox
    theta_boost_val = 1.0 + config.psi_theta_paradox * pi_G

    # Gamma suppression: structural paradox
    gamma_suppress_val = 1.0 - config.psi_gamma_paradox * pi_S

    return {
        "pi_decomposition":  pi_decomp,
        "k_on_5ht2a_mult":   k_on_5ht2a_mult,
        "cb1_baseline":      cb1_baseline,
        "da_tonic_drive":    da_tonic_drive,
        "theta_boost": theta_boost_val,
        "gamma_suppress": gamma_suppress_val,
    }


# =====================================================================
# Threshold — bidirectional neurochem feedback
# =====================================================================

def resolve_paradox_threshold(
    mode: OperationalMode,
    config: ParadoxEngineConfig,
    ht2a_activation: float = 0.0,
    cb1_read: float = 0.0,
    da_tonic: float = 0.0,
    ne_level: float = 0.0,
) -> float:
    """
    Determine active detection threshold, adjusted by neurochem state.

    Bidirectional feedback:
    - High 5-HT2A → lower G threshold (more sensitive to dialectical tensions)
    - High CB1    → lower S threshold (more aware of structural limits)
    - High DA tonic → raise threshold (prevent buffer overflow)
    - Low NE        → raise threshold slightly (relaxed state)

    Returns
    -------
    float
        Effective θ, clamped to [0.05, 0.95].
    """
    base = {
        OperationalMode.NORMAL:     config.theta_normal,
        OperationalMode.DEV:        config.theta_dev,
        OperationalMode.LEARNING:   config.theta_learning,
        OperationalMode.REFLECTIVE: config.theta_normal,
        OperationalMode.REM_NORMAL: config.theta_rem_normal,
        OperationalMode.REM_DREAM:  config.theta_rem_dream,
    }.get(mode, config.theta_normal)

    adjustment = (
        -0.05 * ht2a_activation   # 5-HT2A: more sensitive
        - 0.05 * cb1_read         # CB1: more structurally aware
        + 0.08 * da_tonic         # high tonic DA: buffer protection
        + 0.04 * (1.0 - ne_level) # low NE: relaxed tolerance
    )
    return max(0.05, min(0.95, base + adjustment))


# =====================================================================
# Internal helpers
# =====================================================================

def _extract_concept_pair(flag: ContradictionFlag) -> Tuple[str, str]:
    """Extract short concept labels from a contradiction flag."""
    # Use first proposition if available, else first N tokens
    def _concept(stmt: ProcessedStatement) -> str:
        if stmt.propositions:
            return stmt.propositions[0][:80]
        toks = (stmt.tokens or stmt.raw_text.split())[:6]
        return " ".join(toks)
    return _concept(flag.statement_a), _concept(flag.statement_b)


def _make_resolution_hint(features: ParadoxFeatures) -> Optional[str]:
    """Generate a resolution hint for Class A apparent paradoxes."""
    if features.abstraction_divergence > 0.6:
        return "Apparent conflict may dissolve by reconciling abstraction levels."
    if features.resolvability > 0.7:
        return "Additional context or temporal clarification likely resolves this."
    if features.dialectical_productivity < 0.3:
        return "One or both statements may contain a hidden false premise."
    return "Reframe at a higher abstraction level to surface the compatible interpretation."


def _identify_structural_pattern(features: ParadoxFeatures, concept_a: str) -> Optional[str]:
    """Identify the structural pattern label for Class S paradoxes."""
    if features.self_reference > 0.85:
        text_lower = concept_a.lower()
        if "false" in text_lower or "lie" in text_lower:
            return "liar_paradox"
        if "set" in text_lower or "class" in text_lower:
            return "russells_paradox"
        return "self_referential_loop"
    if features.self_reference > 0.5:
        return "infinite_regress"
    return "type_level_confusion"


# =====================================================================
# Engine class
# =====================================================================

class ParadoxDetectionEngine:
    """
    Paradox Detection Engine — Detection Cluster, Engine 2.

    Stateful wrapper around pure classification functions. Maintains
    5-HT2A integral and CB1 level across calls, and routes classified
    flags to buffer routing logic.

    Parameters
    ----------
    config : ParadoxEngineConfig
        Immutable parameter set.
    rng : np.random.Generator, optional
        Random number generator (for stochastic DA burst).
    """

    engine_id = "paradox_detection_engine"
    cluster   = "detection"

    def __init__(
        self,
        config: Optional[ParadoxEngineConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._config      = config or ParadoxEngineConfig()
        self._likelihoods = _default_likelihoods()
        self._rng         = rng or np.random.default_rng()
        self._state       = ParadoxEngineState()
        self._mode        = OperationalMode.NORMAL

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

        Canonical keys: ``"5ht"`` (→ 5-HT2A), ``"cb1"``, ``"da"``, ``"ne"``.
        """
        if "5ht" in state_dict:
            self._state.ht2a_activation = _clamp(state_dict["5ht"])
        if "cb1" in state_dict:
            self._state.cb1_read = _clamp(state_dict["cb1"])
        if "da" in state_dict:
            self._state.da_tonic = _clamp(state_dict["da"])
        if "ne" in state_dict:
            self._state.ne_level = _clamp(state_dict["ne"])

    # ------------------------------------------------------------------
    # Main process port
    # ------------------------------------------------------------------

    def process(
        self,
        paradox_input: ParadoxInput,
        dt: float = 1.0,
    ) -> ParadoxDetectionResult:
        """
        Run one full paradox detection pass.

        Parameters
        ----------
        paradox_input : ParadoxInput
            Assembled input from upstream (Engine 1 output + pipeline state).
        dt : float
            Timestep in seconds for state integration.

        Returns
        -------
        ParadoxDetectionResult
        """
        import time
        t_start = time.perf_counter()

        theta = resolve_paradox_threshold(
            self._mode, self._config,
            ht2a_activation=self._state.ht2a_activation,
            cb1_read=self._state.cb1_read,
            da_tonic=self._state.da_tonic,
            ne_level=self._state.ne_level,
        )

        flags: List[ParadoxFlag] = []
        contradictions_evaluated = 0
        path1_count = 0
        path2_count = 0
        reclassified_R = 0

        # ----------------------------------------------------------
        # Path 1: Contradiction-derived paradox detection
        # ----------------------------------------------------------
        for contra_flag in paradox_input.contradiction_result.flags:
            contradictions_evaluated += 1
            stmt_a = contra_flag.statement_a
            stmt_b = contra_flag.statement_b
            concept_a, concept_b = _extract_concept_pair(contra_flag)

            delta_t = contra_flag.temporal_separation

            features = self._extract_features(
                stmt_a, stmt_b, concept_a, concept_b,
                paradox_input.unsolved_buffer, delta_t,
            )
            posteriors = compute_class_posteriors(
                features, self._config, self._likelihoods,
                paradox_input.unsolved_buffer, concept_a, concept_b,
            )
            best_class_str = max(posteriors, key=posteriors.__getitem__)
            best_class     = ParadoxClass(best_class_str)
            confidence     = posteriors[best_class_str]

            if best_class == ParadoxClass.R:
                reclassified_R += 1
                continue   # Resolvable — not emitted as paradox

            if confidence < theta:
                continue

            flag = self._make_flag(
                source=ParadoxSource.CONTRADICTION_DERIVED,
                source_contradiction=contra_flag,
                concept_a=concept_a,
                concept_b=concept_b,
                paradox_class=best_class,
                class_posterior=posteriors,
                confidence=confidence,
                features=features,
            )
            flags.append(flag)
            path1_count += 1

        # ----------------------------------------------------------
        # Path 2: Independent detection
        # ----------------------------------------------------------
        path2_candidates = self._run_independent_detection(
            paradox_input, theta
        )
        for flag in path2_candidates:
            flags.append(flag)
            path2_count += 1

        # ----------------------------------------------------------
        # Neurochemical state updates
        # ----------------------------------------------------------
        pi_decomp = compute_paradox_load(flags, self._config)
        pi_G = pi_decomp["Π_G"]
        pi_S = pi_decomp["Π_S"]

        # Update 5-HT2A integral
        self._state.ht2a_integral = update_ht2a_integral(
            self._state.ht2a_integral, pi_G, dt, self._config.tau_sym
        )

        # Update CB1 level — use max σ_ref across flagged G/S
        max_sigma = max(
            (f.features.self_reference for f in flags
             if f.paradox_class in (ParadoxClass.G, ParadoxClass.S)),
            default=0.0,
        )
        self._state.cb1_level = update_cb1_level(
            self._state.cb1_level, pi_S, max_sigma, self._config, dt
        )

        neurochem = compute_paradox_neurochem_signals(
            pi_decomp,
            self._state.ht2a_integral,
            self._state.cb1_level,
            paradox_input.unsolved_buffer,
            self._config,
            flags=flags,
        )

        t_end = time.perf_counter()

        # Counts
        genuine_count    = sum(1 for f in flags if f.paradox_class == ParadoxClass.G)
        structural_count = sum(1 for f in flags if f.paradox_class == ParadoxClass.S)
        apparent_count   = sum(1 for f in flags if f.paradox_class == ParadoxClass.A)

        return ParadoxDetectionResult(
            flags=flags,
            contradictions_evaluated=contradictions_evaluated,
            paradoxes_from_contradictions=path1_count,
            independent_paradoxes=path2_count,
            genuine_count=genuine_count,
            structural_count=structural_count,
            apparent_count=apparent_count,
            reclassified_as_resolvable=reclassified_R,
            clean_pass=(genuine_count == 0 and structural_count == 0),
            detection_threshold_used=theta,
            processing_time_ms=(t_end - t_start) * 1000.0,
            neurochemical_signals=neurochem,
            metadata={
                "mode":                    self._mode.value,
                "prior_encounters_matched": sum(
                    1 for f in flags
                    if f.features.prior_encounter_match >= self._config.theta_match
                ),
                "independent_scan_performed": True,
            },
        )

    # ------------------------------------------------------------------
    # Status port
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        """Return current engine state for monitoring."""
        return {
            "engine_id":      self.engine_id,
            "mode":           self._mode.value,
            "ht2a_integral":  self._state.ht2a_integral,
            "cb1_level":      self._state.cb1_level,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        stmt_a: ProcessedStatement,
        stmt_b: ProcessedStatement,
        concept_a: str,
        concept_b: str,
        unsolved_buffer: List[UnsolvedParadoxEntry],
        delta_t_days: Optional[float],
    ) -> ParadoxFeatures:
        """Extract the full feature vector for a statement pair."""
        r      = compute_resolvability(stmt_a, stmt_b, self._config, delta_t_days)
        sigma  = max(
            compute_self_reference_index(stmt_a.raw_text, stmt_a.tokens),
            compute_self_reference_index(stmt_b.raw_text, stmt_b.tokens),
        )
        delta  = compute_abstraction_divergence(stmt_a, stmt_b)
        pi     = compute_dialectical_productivity(stmt_a, stmt_b, self._config)
        rho, _ = compute_prior_encounter_match(
            concept_a, concept_b, unsolved_buffer, self._config.theta_match
        )
        return ParadoxFeatures(
            resolvability=r,
            self_reference=sigma,
            abstraction_divergence=delta,
            dialectical_productivity=pi,
            prior_encounter_match=rho,
        )

    def _make_flag(
        self,
        source: ParadoxSource,
        source_contradiction: Optional[ContradictionFlag],
        concept_a: str,
        concept_b: str,
        paradox_class: ParadoxClass,
        class_posterior: Dict[str, float],
        confidence: float,
        features: ParadoxFeatures,
    ) -> ParadoxFlag:
        """Construct a ParadoxFlag with appropriate hints and patterns."""
        resolution_hint = (
            _make_resolution_hint(features)
            if paradox_class == ParadoxClass.A else None
        )
        structural_pattern = (
            _identify_structural_pattern(features, concept_a)
            if paradox_class == ParadoxClass.S else None
        )
        tension_score = features.dialectical_productivity if paradox_class == ParadoxClass.G else 0.0

        return ParadoxFlag(
            paradox_id=str(uuid.uuid4()),
            source=source,
            source_contradiction=source_contradiction,
            concept_a=concept_a,
            concept_b=concept_b,
            paradox_class=paradox_class,
            class_posterior=class_posterior,
            confidence=confidence,
            features=features,
            resolution_hint=resolution_hint,
            structural_pattern=structural_pattern,
            symbolic_tension_score=tension_score,
            timestamp=datetime.utcnow(),
        )

    def _run_independent_detection(
        self,
        paradox_input: ParadoxInput,
        theta: float,
    ) -> List[ParadoxFlag]:
        """
        Path 2: Scan semantic expansion for paradoxes independent of
        contradiction flags.

        5.4.1 Single-statement paradoxes: oxymoron + self-reference + universal-particular
        5.4.2 Conceptual paradox: concept-pair π evaluation
        """
        flags: List[ParadoxFlag] = []

        # Gather all statements (primary from semantic expansion + memory context)
        all_stmts: List[ProcessedStatement] = list(paradox_input.memory_context)
        # Also pull from contradiction result's primary set (re-examined independently)
        for contra_flag in paradox_input.contradiction_result.flags:
            all_stmts.append(contra_flag.statement_a)

        for stmt in all_stmts:
            # --- 5.4.1a: Oxymoron detection ---
            oxymorons = detect_oxymoron_candidates(stmt, self._config.theta_oxy)
            for mod, head, score in oxymorons:
                concept_a = mod
                concept_b = head
                # Build synthetic pair for feature extraction
                stub_b = ProcessedStatement(
                    raw_text=head, tokens=[head], source_tag=stmt.source_tag
                )
                features = self._extract_features(
                    stmt, stub_b, concept_a, concept_b,
                    paradox_input.unsolved_buffer, None,
                )
                # Override σ_ref = 0 for single-statement (per spec)
                features = ParadoxFeatures(
                    resolvability=features.resolvability,
                    self_reference=0.0,
                    abstraction_divergence=features.abstraction_divergence,
                    dialectical_productivity=features.dialectical_productivity,
                    prior_encounter_match=features.prior_encounter_match,
                )
                posteriors = compute_class_posteriors(
                    features, self._config, self._likelihoods,
                    paradox_input.unsolved_buffer, concept_a, concept_b,
                )
                best_cls_str = max(posteriors, key=posteriors.__getitem__)
                best_cls     = ParadoxClass(best_cls_str)
                conf         = posteriors[best_cls_str]
                if best_cls != ParadoxClass.R and conf >= theta:
                    flags.append(self._make_flag(
                        source=ParadoxSource.INDEPENDENT_SINGLE,
                        source_contradiction=None,
                        concept_a=concept_a,
                        concept_b=concept_b,
                        paradox_class=best_cls,
                        class_posterior=posteriors,
                        confidence=conf,
                        features=features,
                    ))

            # --- 5.4.1b: Self-reference scan ---
            sigma = detect_single_statement_self_reference(stmt)
            if sigma > 0.5:
                concept_a = stmt.raw_text[:60]
                concept_b = "self"
                features = ParadoxFeatures(
                    resolvability=0.15,           # self-referential → hard to resolve
                    self_reference=sigma,
                    abstraction_divergence=0.5,
                    dialectical_productivity=0.2,
                    prior_encounter_match=0.0,
                )
                posteriors = compute_class_posteriors(
                    features, self._config, self._likelihoods,
                    paradox_input.unsolved_buffer, concept_a, concept_b,
                )
                best_cls_str = max(posteriors, key=posteriors.__getitem__)
                best_cls     = ParadoxClass(best_cls_str)
                conf         = posteriors[best_cls_str]
                if best_cls != ParadoxClass.R and conf >= theta:
                    flags.append(self._make_flag(
                        source=ParadoxSource.INDEPENDENT_SINGLE,
                        source_contradiction=None,
                        concept_a=concept_a,
                        concept_b=concept_b,
                        paradox_class=best_cls,
                        class_posterior=posteriors,
                        confidence=conf,
                        features=features,
                    ))

            # --- 5.4.1c: Universal-particular tension ---
            tension = detect_universal_particular_tension(stmt)
            if tension > 0.0:
                concept_a = "universal claim"
                concept_b = "embedded exception"
                features = ParadoxFeatures(
                    resolvability=0.4,
                    self_reference=0.0,
                    abstraction_divergence=0.6,
                    dialectical_productivity=tension,
                    prior_encounter_match=0.0,
                )
                posteriors = compute_class_posteriors(
                    features, self._config, self._likelihoods,
                    paradox_input.unsolved_buffer, concept_a, concept_b,
                )
                best_cls_str = max(posteriors, key=posteriors.__getitem__)
                best_cls     = ParadoxClass(best_cls_str)
                conf         = posteriors[best_cls_str]
                if best_cls != ParadoxClass.R and conf >= theta:
                    flags.append(self._make_flag(
                        source=ParadoxSource.INDEPENDENT_SINGLE,
                        source_contradiction=None,
                        concept_a=concept_a,
                        concept_b=concept_b,
                        paradox_class=best_cls,
                        class_posterior=posteriors,
                        confidence=conf,
                        features=features,
                    ))

        # --- 5.4.2: Conceptual paradox detection (concept pairs) ---
        # Evaluate all statement pairs for high-π concept-level paradoxes
        for i in range(len(all_stmts)):
            for j in range(i + 1, len(all_stmts)):
                sa = all_stmts[i]
                sb = all_stmts[j]
                pi = compute_dialectical_productivity(sa, sb, self._config)
                if pi < self._config.theta_conceptual:
                    continue
                concept_a = (sa.tokens or sa.raw_text.split())[:3]
                concept_b = (sb.tokens or sb.raw_text.split())[:3]
                ca = " ".join(concept_a) if isinstance(concept_a, list) else str(concept_a)
                cb = " ".join(concept_b) if isinstance(concept_b, list) else str(concept_b)
                features = ParadoxFeatures(
                    resolvability=compute_resolvability(sa, sb, self._config),
                    self_reference=0.0,   # per spec: σ_ref = 0 for conceptual path
                    abstraction_divergence=compute_abstraction_divergence(sa, sb),
                    dialectical_productivity=pi,
                    prior_encounter_match=0.0,
                )
                posteriors = compute_class_posteriors(
                    features, self._config, self._likelihoods,
                    paradox_input.unsolved_buffer, ca, cb,
                )
                best_cls_str = max(posteriors, key=posteriors.__getitem__)
                best_cls     = ParadoxClass(best_cls_str)
                conf         = posteriors[best_cls_str]
                if best_cls != ParadoxClass.R and conf >= theta:
                    flags.append(self._make_flag(
                        source=ParadoxSource.INDEPENDENT_CONCEPTUAL,
                        source_contradiction=None,
                        concept_a=ca,
                        concept_b=cb,
                        paradox_class=best_cls,
                        class_posterior=posteriors,
                        confidence=conf,
                        features=features,
                    ))

        return flags
