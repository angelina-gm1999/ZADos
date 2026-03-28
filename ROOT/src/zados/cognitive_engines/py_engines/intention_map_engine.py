"""
Intention Map Engine — Engine 23 (Pattern & Analysis Cluster).

Eight-category intent classification with neurochemical routing.

Pipeline Position: Fractal Brain Step (a), fires THIRD (after Tokenizer + Expander).

Classification Pipeline (3 Stages)
------------------------------------
Stage 1 — Template Matching (Fast Path):
    cosine(F_combined, Template_k) weighted by default priors.
Stage 2 — Contextual Bayesian Update:
    P(k | F) ~ P(F | k) * P(k | history),  alpha=0.30 EWMA.
Stage 3 — Cross-Category Constraint Resolution:
    Mutual suppression (eta_suppress=0.15) and co-occurrence amplification
    (eta_amplify=0.10), then normalization to sum=1.

Neurochemical Coupling
-----------------------
- B_intent matrix (11 NTs x 8 categories):  DC_intent = B_intent @ E_intent * xi
- Phi_intent matrix (5 bands x 8 categories): DPhi = Phi_intent @ E_intent
- 5 cross-frequency coupling patterns (theta-gamma, alpha-gamma, beta-gamma,
  theta-alpha, delta-alpha) with threshold gating
- 6 pharmacodynamic cross-effects (empathy resonance, emotional inhibition,
  affective internalization, strategy propagation, novelty exploration,
  overgeneralization inhibition)
- Archetype routing via softmax(E_intent / tau),  tau=0.50 default

Usage
-----
>>> from zados.cognitive_engines.py_engines.intention_map_engine import (
...     IntentionMapEngine, IntentionMapConfig, IntentionMapResult,
... )
>>> from zados.cognitive_engines.py_engines.tokenizer import Tokenizer
>>> from zados.cognitive_engines.py_engines.semantic_expander import SemanticExpander
>>> tok = Tokenizer().process("I believe freedom is more important than security.")
>>> exp = SemanticExpander().process(tok)
>>> engine = IntentionMapEngine()
>>> result = engine.process(IntentionMapInput(tokenizer_result=tok, expansion_result=exp))
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from zados.cognitive_engines.constants import _clamp
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)
from zados.cognitive_engines.py_engines.tokenizer import (
    AggregateFeatures,
    TokenizerResult,
)
from zados.cognitive_engines.py_engines.semantic_expander import (
    ExpansionMetrics,
    ExpansionResult,
)


# =====================================================================
# Enumerations
# =====================================================================

class IntentCategory(str, Enum):
    CONNECTION     = "connection"
    CHALLENGE      = "challenge"
    EXPLORATION    = "exploration"
    DISCHARGE      = "discharge"
    PRAGMATIC      = "pragmatic"
    SYMBOLIC       = "symbolic"
    DEFENSIVE      = "defensive"
    DISINTEGRATION = "disintegration"


class Archetype(str, Enum):
    GUIDE      = "Guide"
    OPPONENT   = "Opponent"
    EXPLORER   = "Explorer"
    CONTAINER  = "Container"
    ARCHITECT  = "Architect"
    ORACLE     = "Oracle"
    FIREWALL   = "Firewall"
    STABILIZER = "Stabilizer"


# Intent category index order (canonical)
_INTENT_ORDER: List[IntentCategory] = list(IntentCategory)
_INTENT_INDEX: Dict[IntentCategory, int] = {c: i for i, c in enumerate(_INTENT_ORDER)}

# Archetype mapping: intent -> archetype
_INTENT_TO_ARCHETYPE: Dict[IntentCategory, Archetype] = {
    IntentCategory.CONNECTION:     Archetype.GUIDE,
    IntentCategory.CHALLENGE:      Archetype.OPPONENT,
    IntentCategory.EXPLORATION:    Archetype.EXPLORER,
    IntentCategory.DISCHARGE:      Archetype.CONTAINER,
    IntentCategory.PRAGMATIC:      Archetype.ARCHITECT,
    IntentCategory.SYMBOLIC:       Archetype.ORACLE,
    IntentCategory.DEFENSIVE:      Archetype.FIREWALL,
    IntentCategory.DISINTEGRATION: Archetype.STABILIZER,
}

# Incompatible archetype pairs: higher-weighted dominates, other at 50%
_ARCHETYPE_CONFLICTS: List[Tuple[Archetype, Archetype]] = [
    (Archetype.OPPONENT, Archetype.GUIDE),
    (Archetype.CONTAINER, Archetype.EXPLORER),
    (Archetype.FIREWALL, Archetype.GUIDE),
]


# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True)
class IntentionMapConfig:
    # --- Classification ---
    alpha_history: float = 0.30         # EWMA weight for historical prior
    eta_suppress: float = 0.15          # mutual suppression strength
    eta_amplify: float = 0.10           # co-occurrence amplification strength
    momentum: float = 0.25              # intent temporal momentum
    tau_temperature: float = 0.50       # archetype softmax temperature

    # --- Default priors ---
    prior_connection: float = 0.15
    prior_challenge: float = 0.10
    prior_exploration: float = 0.20
    prior_discharge: float = 0.10
    prior_pragmatic: float = 0.25
    prior_symbolic: float = 0.08
    prior_defensive: float = 0.07
    prior_disintegration: float = 0.05

    # --- Disintegration monitoring ---
    theta_disintegration: float = 0.20     # NE alert threshold
    theta_disintegration_rise: float = 0.08  # per-turn rising rate threshold
    disintegration_rise_turns: int = 2     # consecutive turns for Stabilizer override

    # --- Confidence thresholds ---
    high_confidence: float = 0.30
    low_confidence: float = 0.15

    # --- Neurochemical coupling ---
    beta_intent_ach: float = 0.10
    ach_gamma_alpha: float = 2.0
    ach_gamma_theta: float = 0.30
    beta_intent_da: float = 0.06
    da_gamma_alpha: float = 2.0
    da_gamma_theta: float = 0.25
    beta_novel_intent: float = 0.08
    beta_disint_alert: float = 0.14
    ne_poisson_lambda: float = 2.0

    # --- Cross-frequency coupling ---
    kappa_coupling: float = 0.12

    # --- Pharmacodynamic cross-effects ---
    kappa_empathy: float = 0.15
    theta_empathy: float = 0.08
    kappa_inhibit: float = 0.12
    theta_inhibit: float = 0.10
    kappa_intern: float = 0.10
    theta_intern: float = 0.08
    kappa_strategy: float = 0.10
    kappa_novelty: float = 0.12
    kappa_overgeneralize: float = 0.08

    # --- Regulatory adaptation ---
    rho_explore_da: float = 0.04
    rho_stress_ne: float = 0.06
    eta_oxt_def: float = 0.30
    tau_trust: int = 8
    lambda_5ht2c: float = 0.05
    tau_5ht2c: int = 10

    # --- Mode-specific threshold adjustments ---
    learning_threshold_reduction: float = 0.10
    reflective_threshold_reduction: float = 0.10


# =====================================================================
# Matrices (constant, spec-defined)
# =====================================================================

# B_intent: 11 NT channels x 8 intent categories
# Row order: DA, NE, 5-HT1A, 5-HT2A, OXT, ACh, Glu, GABA-A, MOR, CB1, COR
# Col order: connection, challenge, exploration, discharge, pragmatic, symbolic, defensive, disintegration
_B_INTENT = np.array([
    #   conn   chall  explo  disch  pragm  symb   defen  disint
    [ 0.05,  0.08,  0.18, -0.04,  0.10,  0.12, -0.03, -0.08],  # DA
    [-0.03,  0.16,  0.06,  0.12,  0.10,  0.02,  0.14,  0.15],  # NE
    [ 0.14, -0.06,  0.04,  0.08,  0.03,  0.06, -0.05, -0.10],  # 5-HT1A
    [ 0.04,  0.05,  0.12,  0.02,  0.01,  0.16,  0.04,  0.05],  # 5-HT2A
    [ 0.18, -0.05,  0.03,  0.06,  0.01,  0.10, -0.12, -0.10],  # OXT
    [ 0.04,  0.08,  0.12,  0.02,  0.10,  0.06,  0.04,  0.03],  # ACh
    [ 0.02,  0.06,  0.10,  0.01,  0.06,  0.08,  0.03,  0.12],  # Glu
    [ 0.06, -0.04, -0.02,  0.14,  0.04,  0.02,  0.08, -0.06],  # GABA-A
    [ 0.10, -0.04,  0.02,  0.04,  0.01,  0.06, -0.06, -0.08],  # MOR
    [ 0.06,  0.03,  0.08,  0.05,  0.02,  0.10,  0.04, -0.04],  # CB1
    [-0.04,  0.06, -0.02,  0.04,  0.01, -0.02,  0.08,  0.18],  # COR
], dtype=np.float64)

_NT_CHANNEL_NAMES = [
    "DA", "NE", "5-HT1A", "5-HT2A", "OXT",
    "ACh", "Glu", "GABA-A", "MOR", "CB1", "COR",
]

# Phi_intent: 5 oscillatory bands x 8 intent categories
# Row order: delta, theta, alpha, beta, gamma
# Col order: connection, challenge, exploration, discharge, pragmatic, symbolic, defensive, disintegration
_PHI_INTENT = np.array([
    #   conn   chall  explo  disch  pragm  symb   defen  disint
    [ 0.02, -0.02, -0.01,  0.10,  0.00,  0.01,  0.03,  0.08],  # delta
    [ 0.10,  0.04,  0.10,  0.03,  0.04,  0.12, -0.04, -0.06],  # theta
    [ 0.08, -0.04,  0.04,  0.08,  0.05,  0.03,  0.08, -0.08],  # alpha
    [ 0.02,  0.10,  0.04, -0.06,  0.10,  0.02,  0.08,  0.04],  # beta
    [ 0.08,  0.08,  0.12, -0.04,  0.06,  0.10, -0.04, -0.06],  # gamma
], dtype=np.float64)

_BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]

# Noise distribution configs per intent (for stochastic burst)
# (distribution_name, param_dict)
_INTENT_NOISE: Dict[IntentCategory, Tuple[str, Dict]] = {
    IntentCategory.CONNECTION:     ("gamma",     {"alpha": 2.0, "theta": 0.3}),
    IntentCategory.CHALLENGE:      ("poisson",   {"lam": 2.0}),
    IntentCategory.EXPLORATION:    ("gamma",     {"alpha": 2.5, "theta": 0.4}),
    IntentCategory.DISCHARGE:      ("lognormal", {"mu": -0.5, "sigma": 0.8}),
    IntentCategory.PRAGMATIC:      ("gamma",     {"alpha": 3.0, "theta": 0.25}),
    IntentCategory.SYMBOLIC:       ("lognormal", {"mu": -0.3, "sigma": 0.7}),
    IntentCategory.DEFENSIVE:      ("poisson",   {"lam": 1.5}),
    IntentCategory.DISINTEGRATION: ("lognormal", {"mu": 0.0, "sigma": 1.0}),
}


# =====================================================================
# Suppression / Amplification pairs
# =====================================================================

# Mutual suppression: (a, b) pairs
_SUPPRESSION_PAIRS: List[Tuple[IntentCategory, IntentCategory]] = [
    (IntentCategory.CONNECTION, IntentCategory.CHALLENGE),
    (IntentCategory.PRAGMATIC, IntentCategory.DISCHARGE),
    (IntentCategory.EXPLORATION, IntentCategory.DISINTEGRATION),
]

# Co-occurrence amplification: (a, b) pairs
_AMPLIFICATION_PAIRS: List[Tuple[IntentCategory, IntentCategory]] = [
    (IntentCategory.DEFENSIVE, IntentCategory.CONNECTION),
    (IntentCategory.SYMBOLIC, IntentCategory.EXPLORATION),
    (IntentCategory.CHALLENGE, IntentCategory.EXPLORATION),
    (IntentCategory.DISCHARGE, IntentCategory.CONNECTION),
]


# =====================================================================
# Cross-frequency coupling definitions
# =====================================================================

@dataclass(frozen=True)
class CFCPattern:
    name: str
    band_a: str
    band_b: str
    trigger_intents: Dict[IntentCategory, float]    # intent -> min threshold
    band_a_threshold: float
    band_b_threshold: float


_CFC_PATTERNS: List[CFCPattern] = [
    CFCPattern(
        name="theta_gamma",
        band_a="theta", band_b="gamma",
        trigger_intents={IntentCategory.EXPLORATION: 0.25, IntentCategory.SYMBOLIC: 0.20},
        band_a_threshold=0.4, band_b_threshold=0.4,
    ),
    CFCPattern(
        name="alpha_gamma",
        band_a="alpha", band_b="gamma",
        trigger_intents={IntentCategory.SYMBOLIC: 0.25},
        band_a_threshold=0.4, band_b_threshold=0.4,
    ),
    CFCPattern(
        name="beta_gamma",
        band_a="beta", band_b="gamma",
        trigger_intents={IntentCategory.CHALLENGE: 0.25},
        band_a_threshold=0.5, band_b_threshold=0.4,
    ),
    CFCPattern(
        name="theta_alpha",
        band_a="theta", band_b="alpha",
        trigger_intents={IntentCategory.CONNECTION: 0.30},
        band_a_threshold=0.4, band_b_threshold=0.4,
    ),
    CFCPattern(
        name="delta_alpha",
        band_a="delta", band_b="alpha",
        trigger_intents={IntentCategory.DISCHARGE: 0.30},
        band_a_threshold=0.3, band_b_threshold=0.4,
    ),
]


# =====================================================================
# Data Types
# =====================================================================

@dataclass(frozen=True)
class IntentionMapInput:
    """Input to the Intention Map Engine."""
    tokenizer_result: TokenizerResult = field(default_factory=TokenizerResult)
    expansion_result: ExpansionResult = field(default_factory=ExpansionResult)

    # Affective signals (from emotion detection)
    emotional_valence: float = 0.0          # [-1, 1]
    emotional_intensity: float = 0.0        # [0, 1]
    tone_content_alignment: float = 0.5     # [0, 1]
    emotional_trajectory: float = 0.0       # valence(t) - valence(t-1)
    affect_complexity: int = 0              # number of simultaneous emotions

    # Context signals (from conversation history)
    topic_continuity: float = 0.5           # [0, 1]
    turn_taking_ratio: float = 1.0          # user_len / system_len
    historical_intent: Optional[List[float]] = None  # previous E_intent (8 elements)
    interaction_depth: float = 0.0          # turns / avg_conversation_length
    response_to_input_ratio: float = 0.5    # [0, 1]

    # Upstream signals for cross-effects
    contradiction_load: float = 0.0         # [0, 1]

    # Neurochemical read ports (for pharmacodynamic cross-effects)
    nt_levels: Dict[str, float] = field(default_factory=dict)
    # Expected keys: "OXT", "CB1", "NE", "MOR", "DA_D1", "DA_D3", "5-HT2A", "GABA_B"

    # Oscillatory read ports
    band_powers: Dict[str, float] = field(default_factory=dict)
    # Expected keys: "delta", "theta", "alpha", "beta", "gamma"


@dataclass(frozen=True)
class IntentionMapResult:
    """Output of the Intention Map Engine."""
    # Intent vector
    intent_vector: List[float] = field(default_factory=list)
    intent_labels: Dict[str, float] = field(default_factory=dict)

    # Classification metadata
    dominant_intent: str = ""
    secondary_intent: Optional[str] = None
    intent_confidence: float = 0.0
    is_mixed: bool = False

    # Temporal analysis
    intent_trajectory: List[float] = field(default_factory=list)
    rising_intents: List[str] = field(default_factory=list)
    falling_intents: List[str] = field(default_factory=list)
    disintegration_alert: bool = False

    # Archetype routing
    archetype_selection: Dict[str, float] = field(default_factory=dict)
    primary_archetype: str = ""
    archetype_conflicts: List[Tuple[str, str]] = field(default_factory=list)

    # Neurochemical burst
    neurochemical_burst: Dict[str, float] = field(default_factory=dict)

    # Oscillatory modulation
    oscillatory_burst: Dict[str, float] = field(default_factory=dict)
    active_cross_frequency_couplings: List[str] = field(default_factory=list)

    # Pharmacodynamic cross-effects
    active_pharmacodynamics: Dict[str, float] = field(default_factory=dict)

    # Baseline adjustments
    baseline_adjustments: Dict[str, float] = field(default_factory=dict)

    # Session profile
    session_intent_profile: List[float] = field(default_factory=list)

    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =====================================================================
# Mutable State
# =====================================================================

@dataclass
class IntentionMapState:
    """Persisted engine state across turns."""
    previous_intent: Optional[np.ndarray] = None    # E_intent(t-1)
    historical_prior: Optional[np.ndarray] = None   # EWMA of past intents
    intent_history: List[np.ndarray] = field(default_factory=list)
    turn_count: int = 0
    disintegration_rise_count: int = 0
    # Session-level accumulators for regulatory adaptation
    cumulative_disintegration: float = 0.0
    # NT read-port levels (updated via update_neurochem_state)
    da_level: float = 0.0
    oxt_level: float = 0.0


# =====================================================================
# Pure Functions — Feature Extraction
# =====================================================================

def extract_linguistic_features(agg: AggregateFeatures, contradiction_density: float = 0.0) -> np.ndarray:
    """
    F_ling in R^13: Extract 13-dimensional linguistic feature vector from Tokenizer output.
    """
    # 1. Question type distribution -> single value (fraction of interrogatives)
    q_total = sum(agg.question_type_distribution.values()) if agg.question_type_distribution else 0
    q_ratio = q_total / max(agg.sentence_count, 1)

    # 2. Pronoun ratio (I / total pronouns, you / total, we / total, it / total)
    pd = agg.pronoun_distribution
    pron_i = pd.get("I", 0.0)
    pron_you = pd.get("you", 0.0)
    pron_we = pd.get("we", 0.0)
    pron_it = pd.get("it", 0.0)

    # 3-8. Direct features
    emo_vd = agg.emotional_vocab_density
    action_vd = agg.action_verb_density
    hedging_d = agg.hedging_density
    abstract_r = agg.abstract_noun_ratio
    imperative = min(agg.imperative_count / max(agg.sentence_count, 1), 1.0)
    conditional = min(agg.conditional_count / max(agg.sentence_count, 1), 1.0)

    # 9. Message length trajectory
    ml_ratio = agg.message_length_ratio if agg.message_length_ratio is not None else 1.0

    # 10. Coherence
    coherence = agg.inter_sentence_coherence

    # 11. Meta-linguistic markers
    meta = min(agg.meta_linguistic_count / max(agg.sentence_count, 1), 1.0)

    # 12. Contradiction density (from upstream)
    contra = float(np.clip(contradiction_density, 0.0, 1.0))

    # 13. Punctuation patterns (combined excl + question + ellipsis)
    punct = agg.exclamation_density + agg.question_density + agg.ellipsis_density

    return np.array([
        q_ratio, pron_i, pron_you, pron_we, pron_it,
        emo_vd, action_vd, hedging_d, abstract_r,
        imperative, conditional, ml_ratio, coherence,
    ], dtype=np.float64)


def extract_affective_features(
    valence: float,
    intensity: float,
    alignment: float,
    trajectory: float,
    complexity: int,
) -> np.ndarray:
    """F_affect in R^5."""
    return np.array([
        float(np.clip(valence, -1.0, 1.0)),
        float(np.clip(intensity, 0.0, 1.0)),
        float(np.clip(alignment, 0.0, 1.0)),
        float(np.clip(trajectory, -1.0, 1.0)),
        min(float(complexity), 10.0) / 10.0,
    ], dtype=np.float64)


def extract_context_features(
    topic_continuity: float,
    turn_taking_ratio: float,
    historical_intent: Optional[List[float]],
    interaction_depth: float,
    response_to_input_ratio: float,
) -> np.ndarray:
    """F_context in R^5."""
    # Historical intent pattern: mean of last intent vector (or 0.125 = uniform)
    hist_signal = 0.125
    if historical_intent and len(historical_intent) == 8:
        hist_signal = float(max(historical_intent))

    return np.array([
        float(np.clip(topic_continuity, 0.0, 1.0)),
        float(np.clip(turn_taking_ratio / 5.0, 0.0, 1.0)),  # normalize
        hist_signal,
        float(np.clip(interaction_depth, 0.0, 1.0)),
        float(np.clip(response_to_input_ratio, 0.0, 1.0)),
    ], dtype=np.float64)


def extract_structural_features(metrics: ExpansionMetrics) -> np.ndarray:
    """F_struct in R^5: maps 1:1 from Semantic Expander metrics."""
    return np.array([
        min(float(metrics.fractal_depth), 10.0) / 10.0,
        float(np.clip(metrics.pattern_novelty, 0.0, 1.0)),
        float(np.clip(metrics.symbolic_density, 0.0, 1.0)),
        float(np.clip(metrics.structural_complexity, 0.0, 1.0)),
        float(np.clip(metrics.information_noise_ratio, 0.0, 1.0)),
    ], dtype=np.float64)


def build_combined_features(
    f_ling: np.ndarray,
    f_affect: np.ndarray,
    f_context: np.ndarray,
    f_struct: np.ndarray,
) -> np.ndarray:
    """F_combined = [F_ling, F_affect, F_context, F_struct]."""
    return np.concatenate([f_ling, f_affect, f_context, f_struct])


# =====================================================================
# Pure Functions — Stage 1: Template Matching
# =====================================================================

def build_default_priors(config: IntentionMapConfig) -> np.ndarray:
    """Return default prior vector (8 elements, sum=1)."""
    priors = np.array([
        config.prior_connection,
        config.prior_challenge,
        config.prior_exploration,
        config.prior_discharge,
        config.prior_pragmatic,
        config.prior_symbolic,
        config.prior_defensive,
        config.prior_disintegration,
    ], dtype=np.float64)
    s = priors.sum()
    if s > 0:
        priors /= s
    return priors


def _build_intent_templates() -> np.ndarray:
    """
    Build 8 template vectors (one per intent) in R^28.

    Each template encodes the ideal feature signature for that intent.
    Template dimensions match F_combined: [13 ling, 5 affect, 5 context, 5 struct].
    """
    # 28-dimensional template per intent category
    templates = np.zeros((8, 28), dtype=np.float64)

    # Connection: high emotional vocab, high pron I/you, high coherence, positive valence
    templates[0] = [
        0.1, 0.3, 0.3, 0.2, 0.0,  # q_ratio, pron_I, pron_you, pron_we, pron_it
        0.6, 0.2, 0.1, 0.3,        # emo_vd, action_vd, hedging, abstract
        0.1, 0.1, 1.0, 0.5,        # imperative, conditional, ml_ratio, coherence
        # affect: positive valence, moderate intensity, high alignment, stable trajectory, low complexity
        0.5, 0.5, 0.7, 0.0, 0.2,
        # context: high continuity, balanced turns, moderate history, moderate depth, high resp_ratio
        0.7, 0.5, 0.2, 0.5, 0.6,
        # struct: moderate depth, low novelty, low symbolic, low complexity, moderate noise
        0.3, 0.2, 0.1, 0.2, 0.5,
    ]

    # Challenge: high contradiction, high question ratio, low hedging, neg valence
    templates[1] = [
        0.4, 0.2, 0.3, 0.0, 0.0,
        0.3, 0.4, 0.0, 0.4,
        0.3, 0.3, 1.2, 0.3,
        -0.3, 0.6, 0.3, -0.2, 0.1,
        0.5, 0.8, 0.3, 0.4, 0.3,
        0.4, 0.3, 0.2, 0.5, 0.4,
    ]

    # Exploration: high question ratio, high abstract, high novelty, high symbolic
    templates[2] = [
        0.5, 0.1, 0.0, 0.0, 0.1,
        0.2, 0.2, 0.3, 0.7,
        0.0, 0.3, 1.0, 0.4,
        0.2, 0.4, 0.5, 0.1, 0.3,
        0.6, 0.4, 0.3, 0.3, 0.5,
        0.6, 0.7, 0.5, 0.6, 0.6,
    ]

    # Discharge: high emotional intensity, low coherence, high emo vocab, short messages
    templates[3] = [
        0.1, 0.5, 0.0, 0.0, 0.0,
        0.8, 0.1, 0.1, 0.2,
        0.1, 0.0, 0.5, 0.1,
        -0.2, 0.8, 0.2, -0.3, 0.1,
        0.3, 0.3, 0.2, 0.2, 0.2,
        0.2, 0.1, 0.1, 0.1, 0.3,
    ]

    # Pragmatic: high imperative, high action verbs, low abstract, low emotion
    templates[4] = [
        0.2, 0.0, 0.2, 0.0, 0.1,
        0.1, 0.7, 0.1, 0.1,
        0.6, 0.2, 1.0, 0.5,
        0.0, 0.2, 0.6, 0.0, 0.1,
        0.5, 0.6, 0.4, 0.5, 0.6,
        0.3, 0.2, 0.1, 0.3, 0.5,
    ]

    # Symbolic: high abstract, high symbolic density, high type-token, high novelty
    templates[5] = [
        0.2, 0.1, 0.0, 0.0, 0.1,
        0.3, 0.1, 0.2, 0.8,
        0.0, 0.2, 1.0, 0.3,
        0.1, 0.3, 0.4, 0.0, 0.2,
        0.4, 0.3, 0.4, 0.3, 0.4,
        0.8, 0.8, 0.8, 0.7, 0.5,
    ]

    # Defensive: high hedging, low emotional vocab, short responses, negative trajectory
    templates[6] = [
        0.1, 0.3, 0.1, 0.0, 0.0,
        0.2, 0.1, 0.6, 0.2,
        0.1, 0.2, 0.6, 0.2,
        -0.1, 0.3, 0.2, -0.2, 0.1,
        0.4, 0.4, 0.3, 0.3, 0.3,
        0.2, 0.1, 0.1, 0.2, 0.4,
    ]

    # Disintegration: very low coherence, high affect complexity, high contradiction, erratic
    templates[7] = [
        0.3, 0.3, 0.0, 0.0, 0.0,
        0.5, 0.1, 0.3, 0.4,
        0.1, 0.1, 0.3, 0.0,
        -0.5, 0.9, 0.1, -0.4, 0.8,
        0.1, 0.3, 0.1, 0.1, 0.1,
        0.3, 0.2, 0.3, 0.4, 0.2,
    ]

    # Normalize templates to unit vectors for cosine similarity
    for i in range(8):
        norm = np.linalg.norm(templates[i])
        if norm > 0:
            templates[i] /= norm

    return templates


# Pre-compute templates at module load
_INTENT_TEMPLATES = _build_intent_templates()


def compute_template_match(
    f_combined: np.ndarray,
    config: IntentionMapConfig,
) -> np.ndarray:
    """
    Stage 1: Cosine similarity between F_combined and each template, weighted by priors.
    Returns 8-element match score vector.
    """
    priors = build_default_priors(config)

    # Normalize feature vector
    norm = np.linalg.norm(f_combined)
    if norm > 0:
        f_norm = f_combined / norm
    else:
        f_norm = f_combined

    # Cosine similarity with each template
    similarities = _INTENT_TEMPLATES @ f_norm  # (8,)

    # Weight by priors
    scores = np.maximum(similarities, 0.0) * priors

    # Ensure non-negative and normalize
    scores = np.maximum(scores, 1e-12)
    s = scores.sum()
    if s > 0:
        scores /= s

    return scores


# =====================================================================
# Pure Functions — Stage 2: Bayesian Update
# =====================================================================

def bayesian_update(
    match_scores: np.ndarray,
    historical_prior: Optional[np.ndarray],
    default_priors: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Stage 2: P(k | F) ~ P(F | k) * P(k | history).

    historical_prior is EWMA of past intents.
    If None, use default_priors.
    """
    if historical_prior is not None:
        prior = alpha * historical_prior + (1.0 - alpha) * default_priors
    else:
        prior = default_priors.copy()

    # Posterior proportional to likelihood * prior
    posterior = match_scores * prior
    s = posterior.sum()
    if s > 0:
        posterior /= s
    else:
        posterior = default_priors.copy()
    return posterior


# =====================================================================
# Pure Functions — Stage 3: Constraint Resolution
# =====================================================================

def apply_constraints(
    posterior: np.ndarray,
    eta_suppress: float,
    eta_amplify: float,
) -> np.ndarray:
    """
    Stage 3: Apply mutual suppression and co-occurrence amplification.
    Returns normalized intent vector summing to 1.
    """
    adjusted = posterior.copy()

    # Suppression: for each pair (a,b), suppress a by (1 - eta * e_b) and vice versa
    for cat_a, cat_b in _SUPPRESSION_PAIRS:
        ia = _INTENT_INDEX[cat_a]
        ib = _INTENT_INDEX[cat_b]
        adjusted[ia] *= (1.0 - eta_suppress * posterior[ib])
        adjusted[ib] *= (1.0 - eta_suppress * posterior[ia])

    # Amplification: for each pair (a,b), amplify a by (1 + eta * e_b)
    for cat_a, cat_b in _AMPLIFICATION_PAIRS:
        ia = _INTENT_INDEX[cat_a]
        ib = _INTENT_INDEX[cat_b]
        adjusted[ia] *= (1.0 + eta_amplify * posterior[ib])
        adjusted[ib] *= (1.0 + eta_amplify * posterior[ia])

    # Ensure non-negative
    adjusted = np.maximum(adjusted, 1e-12)

    # Normalize
    s = adjusted.sum()
    if s > 0:
        adjusted /= s
    return adjusted


# =====================================================================
# Pure Functions — Temporal Dynamics
# =====================================================================

def apply_momentum(
    current_raw: np.ndarray,
    previous: Optional[np.ndarray],
    momentum: float,
) -> np.ndarray:
    """
    E_intent(t) = (1 - momentum) * E_raw(t) + momentum * E(t-1).
    """
    if previous is not None:
        blended = (1.0 - momentum) * current_raw + momentum * previous
    else:
        blended = current_raw.copy()

    # Re-normalize
    s = blended.sum()
    if s > 0:
        blended /= s
    return blended


def compute_intent_trajectory(
    current: np.ndarray,
    previous: Optional[np.ndarray],
) -> np.ndarray:
    """dE/dt approximation."""
    if previous is not None:
        return current - previous
    return np.zeros(8, dtype=np.float64)


def check_disintegration_alert(
    trajectory: np.ndarray,
    disintegration_idx: int,
    rise_count: int,
    theta_rise: float,
    min_turns: int,
) -> Tuple[bool, int]:
    """
    Check if de_disintegration/dt > threshold for consecutive turns.
    Returns (alert_triggered, new_rise_count).
    """
    delta_disint = trajectory[disintegration_idx]
    if delta_disint > theta_rise:
        new_count = rise_count + 1
    else:
        new_count = 0
    alert = new_count >= min_turns
    return alert, new_count


# =====================================================================
# Pure Functions — Archetype Routing
# =====================================================================

def compute_archetype_weights(
    e_intent: np.ndarray,
    tau: float,
) -> Dict[str, float]:
    """
    Softmax(E_intent / tau) -> archetype weights.
    """
    scaled = e_intent / max(tau, 0.01)
    # Numerical stability
    scaled = scaled - scaled.max()
    exp_vals = np.exp(scaled)
    weights = exp_vals / exp_vals.sum()

    result: Dict[str, float] = {}
    for i, cat in enumerate(_INTENT_ORDER):
        archetype = _INTENT_TO_ARCHETYPE[cat]
        result[archetype.value] = float(weights[i])
    return result


def resolve_archetype_conflicts(
    weights: Dict[str, float],
) -> Tuple[Dict[str, float], List[Tuple[str, str]]]:
    """
    When conflicting archetypes both active, higher-weighted dominates,
    other at 50% influence.
    Returns (adjusted_weights, conflict_pairs).
    """
    adjusted = dict(weights)
    conflicts: List[Tuple[str, str]] = []

    for a, b in _ARCHETYPE_CONFLICTS:
        wa = adjusted.get(a.value, 0.0)
        wb = adjusted.get(b.value, 0.0)
        # Both "active" if above 0.10
        if wa > 0.10 and wb > 0.10:
            conflicts.append((a.value, b.value))
            if wa >= wb:
                adjusted[b.value] = wb * 0.5
            else:
                adjusted[a.value] = wa * 0.5

    # Re-normalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    return adjusted, conflicts


# =====================================================================
# Pure Functions — Neurochemical Burst
# =====================================================================

def sample_intent_noise(
    category: IntentCategory,
    rng: np.random.Generator,
) -> float:
    """Sample stochastic noise for a given intent category."""
    dist_name, params = _INTENT_NOISE[category]
    if dist_name == "gamma":
        return float(rng.gamma(params["alpha"], params["theta"]))
    elif dist_name == "poisson":
        return float(rng.poisson(params["lam"])) / params["lam"]
    elif dist_name == "lognormal":
        return float(rng.lognormal(params["mu"], params["sigma"]))
    return 1.0


def compute_neurochemical_burst(
    e_intent: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    DC_intent = B_intent @ (E_intent * xi).
    Returns dict of NT channel -> delta.
    """
    # Sample noise per intent
    xi = np.array([
        sample_intent_noise(cat, rng)
        for cat in _INTENT_ORDER
    ], dtype=np.float64)

    # Matrix multiply: (11, 8) @ (8,) -> (11,)
    burst = _B_INTENT @ (e_intent * xi)

    result: Dict[str, float] = {}
    for i, name in enumerate(_NT_CHANNEL_NAMES):
        result[name] = float(np.clip(burst[i], -1.0, 1.0))
    return result


def compute_oscillatory_burst(e_intent: np.ndarray) -> Dict[str, float]:
    """
    DPhi = Phi_intent @ E_intent.
    Returns dict of band -> delta.
    """
    burst = _PHI_INTENT @ e_intent
    result: Dict[str, float] = {}
    for i, name in enumerate(_BAND_NAMES):
        result[name] = float(np.clip(burst[i], -1.0, 1.0))
    return result


# =====================================================================
# Pure Functions — Cross-Frequency Coupling
# =====================================================================

def evaluate_cross_frequency_couplings(
    e_intent: np.ndarray,
    band_powers: Dict[str, float],
    osc_burst: Dict[str, float],
    config: IntentionMapConfig,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Evaluate all 5 CFC patterns. Returns (active_cfc_names, cfc_modulations).
    cfc_modulations maps CFC name -> coupling strength.
    """
    active: List[str] = []
    modulations: Dict[str, float] = {}

    disint_idx = _INTENT_INDEX[IntentCategory.DISINTEGRATION]
    e_disint = e_intent[disint_idx]
    coupling_collapse = e_disint >= 0.35

    for cfc in _CFC_PATTERNS:
        # Check if any trigger intent meets threshold
        triggered = False
        for cat, threshold in cfc.trigger_intents.items():
            idx = _INTENT_INDEX[cat]
            if e_intent[idx] > threshold:
                triggered = True
                break

        if not triggered:
            continue

        # Check band power thresholds (use current power + burst)
        power_a = band_powers.get(cfc.band_a, 0.0) + osc_burst.get(cfc.band_a, 0.0)
        power_b = band_powers.get(cfc.band_b, 0.0) + osc_burst.get(cfc.band_b, 0.0)

        if power_a < cfc.band_a_threshold or power_b < cfc.band_b_threshold:
            continue

        # Compute coupling strength
        strength = config.kappa_coupling * power_a * power_b
        if coupling_collapse:
            strength = 0.0

        active.append(cfc.name)
        modulations[cfc.name] = float(strength)

    return active, modulations


# =====================================================================
# Pure Functions — Pharmacodynamic Cross-Effects
# =====================================================================

def compute_pharmacodynamic_effects(
    e_intent: np.ndarray,
    nt_levels: Dict[str, float],
    band_powers: Dict[str, float],
    contradiction_load: float,
    intent_confidence: float,
    config: IntentionMapConfig,
) -> Dict[str, float]:
    """
    Compute all 6 pharmacodynamic cross-effects.
    Returns dict of effect_name -> strength.
    """
    effects: Dict[str, float] = {}

    oxt = nt_levels.get("OXT", 0.0)
    cb1 = nt_levels.get("CB1", 0.0)
    ne = nt_levels.get("NE", 0.0)
    mor = nt_levels.get("MOR", 0.0)
    da_d1 = nt_levels.get("DA_D1", 0.0)
    da_d3 = nt_levels.get("DA_D3", 0.0)
    sht2a = nt_levels.get("5-HT2A", 0.0)
    gaba_b = nt_levels.get("GABA_B", 0.0)
    theta_power = band_powers.get("theta", 0.0)

    # 1. Empathy Resonance: OXT * CB1 * Theta * kappa
    r_empathy = oxt * cb1 * theta_power * config.kappa_empathy
    if oxt > 0.3 and cb1 > 0.2 and theta_power > 0.4:
        effects["empathy_resonance"] = float(r_empathy)
    else:
        effects["empathy_resonance"] = 0.0

    # 2. Emotional Inhibition: NE * contradiction * kappa
    i_inhib = ne * contradiction_load * config.kappa_inhibit
    if ne > 0.5 and contradiction_load > 0.4:
        effects["emotional_inhibition"] = float(i_inhib)
    else:
        effects["emotional_inhibition"] = 0.0

    # 3. Affective Internalization: MOR * contradiction * kappa
    i_intern = mor * contradiction_load * config.kappa_intern
    if mor > 0.3 and contradiction_load > 0.5:
        effects["affective_internalization"] = float(i_intern)
    else:
        effects["affective_internalization"] = 0.0

    # 4. Strategy Propagation: DA_D1 * NE * kappa
    r_strat = da_d1 * ne * config.kappa_strategy
    if da_d1 > 0.4 and ne > 0.35:
        effects["strategy_propagation"] = float(r_strat)
    else:
        effects["strategy_propagation"] = 0.0

    # 5. Novelty Exploration: 5-HT2A * DA_D3 * CB1 * kappa
    r_novel = sht2a * da_d3 * cb1 * config.kappa_novelty
    if sht2a > 0.25 and da_d3 > 0.3 and cb1 > 0.2:
        effects["novelty_exploration"] = float(r_novel)
    else:
        effects["novelty_exploration"] = 0.0

    # 6. Overgeneralization Inhibition: GABA_B * (1 - confidence) * kappa
    i_overgen = gaba_b * (1.0 - intent_confidence) * config.kappa_overgeneralize
    if gaba_b > 0.3 and intent_confidence < 0.5:
        effects["overgeneralization_inhibition"] = float(i_overgen)
    else:
        effects["overgeneralization_inhibition"] = 0.0

    return effects


# =====================================================================
# Pure Functions — Engine-Level Neurochemical Coupling
# =====================================================================

def compute_engine_neurochemical_signals(
    e_intent: np.ndarray,
    intent_confidence: float,
    pattern_novelty: float,
    config: IntentionMapConfig,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Engine-specific ACh, DA, NE signals (separate from B_intent burst).
    """
    signals: Dict[str, float] = {}

    # Ambiguity penalty
    ambiguity_penalty = max(0.0, config.low_confidence - intent_confidence)
    processing_load = intent_confidence * (1.0 + ambiguity_penalty)

    # ACh — classification attention
    ach = config.beta_intent_ach * processing_load * float(
        rng.gamma(config.ach_gamma_alpha, config.ach_gamma_theta)
    )
    signals["ach_intent"] = float(np.clip(ach, 0.0, 1.0))

    # DA — successful classification reward
    da = config.beta_intent_da * intent_confidence * float(
        rng.gamma(config.da_gamma_alpha, config.da_gamma_theta)
    )
    # Novel intent bonus
    da += config.beta_novel_intent * pattern_novelty * 0.08
    signals["da_intent"] = float(np.clip(da, 0.0, 1.0))

    # NE — disintegration alerting
    disint_idx = _INTENT_INDEX[IntentCategory.DISINTEGRATION]
    e_disint = e_intent[disint_idx]
    if e_disint > config.theta_disintegration:
        ne_count = float(rng.poisson(config.ne_poisson_lambda))
        ne = config.beta_disint_alert * e_disint * ne_count / max(config.ne_poisson_lambda, 0.01)
        signals["ne_disint_alert"] = float(np.clip(ne, 0.0, 1.0))
    else:
        signals["ne_disint_alert"] = 0.0

    return signals


# =====================================================================
# Pure Functions — Baseline Adjustments (Regulatory Adaptation)
# =====================================================================

def compute_baseline_adjustments(
    e_intent: np.ndarray,
    interaction_depth: float,
    cumulative_disint: float,
    config: IntentionMapConfig,
) -> Dict[str, float]:
    """
    Recommended baseline shifts per NT channel.
    """
    adjustments: Dict[str, float] = {}

    # DA baseline shift from exploration
    explore_idx = _INTENT_INDEX[IntentCategory.EXPLORATION]
    adjustments["DA_baseline"] = float(config.rho_explore_da * e_intent[explore_idx])

    # NE stress accumulation from challenge + defensiveness + disintegration
    chall_idx = _INTENT_INDEX[IntentCategory.CHALLENGE]
    def_idx = _INTENT_INDEX[IntentCategory.DEFENSIVE]
    disint_idx = _INTENT_INDEX[IntentCategory.DISINTEGRATION]
    stress = e_intent[chall_idx] + e_intent[def_idx] + e_intent[disint_idx]
    adjustments["NE_baseline"] = float(config.rho_stress_ne * stress)

    # OXT suppression from defensiveness
    e_def = e_intent[def_idx]
    trust_factor = min(1.0, interaction_depth * config.tau_trust)
    oxt_modulation = trust_factor * (1.0 - config.eta_oxt_def * e_def) - 1.0
    adjustments["OXT_baseline"] = float(np.clip(oxt_modulation, -1.0, 1.0))

    # 5-HT2C sensitization from cumulative disintegration
    sensitization = config.lambda_5ht2c * cumulative_disint
    adjustments["5-HT2C_sensitivity"] = float(np.clip(sensitization, 0.0, 1.0))

    return adjustments


# =====================================================================
# Pure Functions — Mode-Specific Adjustments
# =====================================================================

def apply_mode_adjustments(
    e_intent: np.ndarray,
    mode: OperationalMode,
    config: IntentionMapConfig,
) -> np.ndarray:
    """
    Mode-specific sensitivity adjustments.
    Returns adjusted intent vector (re-normalized).
    """
    adjusted = e_intent.copy()

    if mode == OperationalMode.LEARNING:
        # Elevated sensitivity to Challenge and Exploration
        chall_idx = _INTENT_INDEX[IntentCategory.CHALLENGE]
        explore_idx = _INTENT_INDEX[IntentCategory.EXPLORATION]
        factor = 1.0 + config.learning_threshold_reduction
        adjusted[chall_idx] *= factor
        adjusted[explore_idx] *= factor

    elif mode == OperationalMode.REFLECTIVE:
        # Elevated sensitivity to Defensive and Disintegration
        def_idx = _INTENT_INDEX[IntentCategory.DEFENSIVE]
        disint_idx = _INTENT_INDEX[IntentCategory.DISINTEGRATION]
        factor = 1.0 + config.reflective_threshold_reduction
        adjusted[def_idx] *= factor
        adjusted[disint_idx] *= factor

    # Re-normalize
    s = adjusted.sum()
    if s > 0:
        adjusted /= s
    return adjusted


# =====================================================================
# Engine
# =====================================================================

class IntentionMapEngine:
    """
    Intention Map Engine — Engine 23 (Pattern & Analysis Cluster).

    Public API:
        configure(mode)                     -- set operational mode
        process(input_)                     -- main classification call
        update_neurochem_state(state_dict)  -- update NT read-port levels
        get_status()                        -- engine state snapshot
    """

    engine_id = "intention_map_engine"
    cluster   = "pattern_analysis"

    def __init__(
        self,
        config: IntentionMapConfig = IntentionMapConfig(),
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._config = config
        self._mode = OperationalMode.NORMAL
        self._rng = rng if rng is not None else np.random.default_rng()
        self._state = IntentionMapState()

    def configure(self, mode: OperationalMode) -> None:
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """Update NT read-port levels from pipeline dict."""
        if "da" in state_dict:
            self._state.da_level = _clamp(state_dict["da"])
        if "oxt" in state_dict:
            self._state.oxt_level = _clamp(state_dict["oxt"])

    def process(self, input_: IntentionMapInput) -> IntentionMapResult:
        if self._mode == OperationalMode.REM_DREAM:
            return IntentionMapResult()

        t0 = time.perf_counter()
        config = self._config
        state = self._state
        state.turn_count += 1

        # -----------------------------------------------------------
        # Pipeline 1: Linguistic features (from Tokenizer)
        # -----------------------------------------------------------
        f_ling = extract_linguistic_features(
            input_.tokenizer_result.aggregate_features,
            input_.contradiction_load,
        )

        # -----------------------------------------------------------
        # Pipeline 2: Affective features
        # -----------------------------------------------------------
        f_affect = extract_affective_features(
            input_.emotional_valence,
            input_.emotional_intensity,
            input_.tone_content_alignment,
            input_.emotional_trajectory,
            input_.affect_complexity,
        )

        # -----------------------------------------------------------
        # Pipeline 3: Context features
        # -----------------------------------------------------------
        hist_intent = input_.historical_intent
        if hist_intent is None and state.previous_intent is not None:
            hist_intent = state.previous_intent.tolist()
        f_context = extract_context_features(
            input_.topic_continuity,
            input_.turn_taking_ratio,
            hist_intent,
            input_.interaction_depth,
            input_.response_to_input_ratio,
        )

        # -----------------------------------------------------------
        # Pipeline 4: Structural features (from Semantic Expander)
        # -----------------------------------------------------------
        f_struct = extract_structural_features(input_.expansion_result.metrics)

        # -----------------------------------------------------------
        # Build combined feature vector
        # -----------------------------------------------------------
        f_combined = build_combined_features(f_ling, f_affect, f_context, f_struct)

        # -----------------------------------------------------------
        # Stage 1: Template matching
        # -----------------------------------------------------------
        match_scores = compute_template_match(f_combined, config)

        # -----------------------------------------------------------
        # Stage 2: Bayesian update
        # -----------------------------------------------------------
        default_priors = build_default_priors(config)
        posterior = bayesian_update(
            match_scores,
            state.historical_prior,
            default_priors,
            config.alpha_history,
        )

        # -----------------------------------------------------------
        # Stage 3: Constraint resolution
        # -----------------------------------------------------------
        constrained = apply_constraints(
            posterior,
            config.eta_suppress,
            config.eta_amplify,
        )

        # -----------------------------------------------------------
        # Mode-specific adjustments
        # -----------------------------------------------------------
        constrained = apply_mode_adjustments(constrained, self._mode, config)

        # -----------------------------------------------------------
        # Temporal dynamics: momentum
        # -----------------------------------------------------------
        e_intent = apply_momentum(constrained, state.previous_intent, config.momentum)

        # -----------------------------------------------------------
        # Temporal dynamics: trajectory and disintegration alert
        # -----------------------------------------------------------
        trajectory = compute_intent_trajectory(e_intent, state.previous_intent)
        disint_idx = _INTENT_INDEX[IntentCategory.DISINTEGRATION]
        disint_alert, new_rise_count = check_disintegration_alert(
            trajectory, disint_idx,
            state.disintegration_rise_count,
            config.theta_disintegration_rise,
            config.disintegration_rise_turns,
        )
        state.disintegration_rise_count = new_rise_count

        # Stabilizer override if disintegration alert
        if disint_alert:
            e_intent[disint_idx] = max(e_intent[disint_idx], 0.25)
            s = e_intent.sum()
            if s > 0:
                e_intent /= s

        # Accumulate disintegration for regulatory adaptation
        state.cumulative_disintegration += e_intent[disint_idx]

        # -----------------------------------------------------------
        # Classification metadata
        # -----------------------------------------------------------
        sorted_indices = np.argsort(e_intent)[::-1]
        dominant_idx = sorted_indices[0]
        second_idx = sorted_indices[1]
        dominant_cat = _INTENT_ORDER[dominant_idx].value
        confidence = float(e_intent[dominant_idx] - e_intent[second_idx])
        secondary_cat = None
        if confidence < 0.10:
            secondary_cat = _INTENT_ORDER[second_idx].value
        is_mixed = confidence < config.low_confidence

        # Rising / falling intents
        rising = [_INTENT_ORDER[i].value for i in range(8) if trajectory[i] > 0.01]
        falling = [_INTENT_ORDER[i].value for i in range(8) if trajectory[i] < -0.01]

        # -----------------------------------------------------------
        # Archetype routing
        # -----------------------------------------------------------
        archetype_weights = compute_archetype_weights(e_intent, config.tau_temperature)
        archetype_weights, arch_conflicts = resolve_archetype_conflicts(archetype_weights)
        primary_arch = max(archetype_weights, key=archetype_weights.get)

        # -----------------------------------------------------------
        # Neurochemical burst (B_intent matrix)
        # -----------------------------------------------------------
        nt_burst = compute_neurochemical_burst(e_intent, self._rng)

        # -----------------------------------------------------------
        # Oscillatory burst (Phi_intent matrix)
        # -----------------------------------------------------------
        osc_burst = compute_oscillatory_burst(e_intent)

        # -----------------------------------------------------------
        # Cross-frequency coupling
        # -----------------------------------------------------------
        active_cfc, cfc_mods = evaluate_cross_frequency_couplings(
            e_intent, input_.band_powers, osc_burst, config,
        )

        # -----------------------------------------------------------
        # Pharmacodynamic cross-effects
        # -----------------------------------------------------------
        pharma_effects = compute_pharmacodynamic_effects(
            e_intent, input_.nt_levels, input_.band_powers,
            input_.contradiction_load, confidence, config,
        )

        # -----------------------------------------------------------
        # Engine-level neurochemical signals
        # -----------------------------------------------------------
        pattern_novelty = input_.expansion_result.metrics.pattern_novelty
        engine_signals = compute_engine_neurochemical_signals(
            e_intent, confidence, pattern_novelty, config, self._rng,
        )

        # Merge engine signals into NT burst
        for k, v in engine_signals.items():
            nt_burst[k] = v

        # -----------------------------------------------------------
        # Baseline adjustments
        # -----------------------------------------------------------
        baseline_adj = compute_baseline_adjustments(
            e_intent, input_.interaction_depth,
            state.cumulative_disintegration, config,
        )

        # -----------------------------------------------------------
        # Update state
        # -----------------------------------------------------------
        state.previous_intent = e_intent.copy()
        state.intent_history.append(e_intent.copy())

        # Update historical prior (EWMA)
        if state.historical_prior is not None:
            state.historical_prior = (
                config.alpha_history * e_intent
                + (1.0 - config.alpha_history) * state.historical_prior
            )
        else:
            state.historical_prior = e_intent.copy()

        # Session intent profile (running average)
        if len(state.intent_history) > 0:
            session_profile = np.mean(state.intent_history, axis=0)
        else:
            session_profile = e_intent.copy()

        # -----------------------------------------------------------
        # Build intent labels dict
        # -----------------------------------------------------------
        intent_labels = {
            cat.value: float(e_intent[i])
            for i, cat in enumerate(_INTENT_ORDER)
        }

        dt_ms = (time.perf_counter() - t0) * 1000.0

        return IntentionMapResult(
            intent_vector=e_intent.tolist(),
            intent_labels=intent_labels,
            dominant_intent=dominant_cat,
            secondary_intent=secondary_cat,
            intent_confidence=float(confidence),
            is_mixed=is_mixed,
            intent_trajectory=trajectory.tolist(),
            rising_intents=rising,
            falling_intents=falling,
            disintegration_alert=disint_alert,
            archetype_selection=archetype_weights,
            primary_archetype=primary_arch,
            archetype_conflicts=arch_conflicts,
            neurochemical_burst=nt_burst,
            oscillatory_burst=osc_burst,
            active_cross_frequency_couplings=active_cfc,
            active_pharmacodynamics=pharma_effects,
            baseline_adjustments=baseline_adj,
            session_intent_profile=session_profile.tolist(),
            processing_time_ms=dt_ms,
        )

    def get_status(self) -> Dict:
        state = self._state
        return {
            "engine_id": self.engine_id,
            "mode": self._mode.value,
            "turn_count": state.turn_count,
            "has_previous_intent": state.previous_intent is not None,
            "disintegration_rise_count": state.disintegration_rise_count,
            "cumulative_disintegration": state.cumulative_disintegration,
            "history_length": len(state.intent_history),
        }
