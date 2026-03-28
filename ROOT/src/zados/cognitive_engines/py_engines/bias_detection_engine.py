"""
Engine 5 -- Bias Detection Engine  (``bias_detection_engine``)
=============================================================
Detects cognitive biases in **any** input flowing through the pipeline --
user reasoning, system reasoning, or memory content.

This engine is *external-facing*: it flags bias in CONTENT (what was said).
Compare with Engine 24 (Heuristic Bias Engine) which is *introspective*:
it monitors the system's OWN reasoning *processes* for shortcuts and
distortions.

Key design decisions (from user spec):
  * Hybrid Kahneman taxonomy -- not pure dual-process, but practical groupings.
  * Flag-only output -- no debiasing suggestions (that is downstream work).
  * Works on any ProcessedStatement from any pipeline source.
  * Neurochemical coupling: ACh attention, NE salience, 5-HT2A flexibility,
    DA novelty reward, cortisol threat.
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
    ProcessedStatement,
    SourceTag,
)


# =====================================================================
# Enums
# =====================================================================


class BiasCategory(str, Enum):
    """Top-level bias families (hybrid Kahneman)."""
    ANCHORING       = "anchoring"
    AVAILABILITY    = "availability"
    REPRESENTATIVENESS = "representativeness"
    FRAMING         = "framing"
    CONFIRMATION    = "confirmation"
    SOCIAL          = "social"
    TEMPORAL        = "temporal"
    SELF_SERVING    = "self_serving"


class BiasType(str, Enum):
    """Specific bias identifiers -- 24 types across 8 categories."""
    # Anchoring
    ANCHOR_NUMERIC          = "anchor_numeric"
    ANCHOR_NARRATIVE        = "anchor_narrative"
    ANCHOR_PRIMACY          = "anchor_primacy"
    # Availability
    AVAILABILITY_RECENCY    = "availability_recency"
    AVAILABILITY_VIVIDNESS  = "availability_vividness"
    AVAILABILITY_SALIENCE   = "availability_salience"
    # Representativeness
    CONJUNCTION_FALLACY     = "conjunction_fallacy"
    BASE_RATE_NEGLECT       = "base_rate_neglect"
    GAMBLER_FALLACY         = "gambler_fallacy"
    # Framing
    LOSS_AVERSION_FRAME     = "loss_aversion_frame"
    DEFAULT_EFFECT_FRAME    = "default_effect_frame"
    CONTRAST_EFFECT_FRAME   = "contrast_effect_frame"
    # Confirmation
    CONFIRMATION_SEARCH     = "confirmation_search"
    CONFIRMATION_INTERPRET  = "confirmation_interpret"
    DISCONFIRMATION_NEGLECT = "disconfirmation_neglect"
    # Social
    AUTHORITY_BIAS          = "authority_bias"
    BANDWAGON_EFFECT        = "bandwagon_effect"
    IN_GROUP_BIAS           = "in_group_bias"
    # Temporal
    HINDSIGHT_BIAS          = "hindsight_bias"
    PLANNING_FALLACY        = "planning_fallacy"
    SUNK_COST_BIAS          = "sunk_cost_bias"
    # Self-Serving
    SELF_SERVING_ATTRIB     = "self_serving_attribution"
    DUNNING_KRUGER          = "dunning_kruger"
    OPTIMISM_BIAS           = "optimism_bias"


class SeverityLevel(str, Enum):
    """How impactful the detected bias is."""
    LOW      = "low"       # Present but minor
    MODERATE = "moderate"  # Notable, may affect reasoning
    HIGH     = "high"      # Strong bias, likely distorting reasoning
    CRITICAL = "critical"  # Dominating the reasoning chain


# =====================================================================
# Configuration
# =====================================================================


# Template feature sets per bias type (keyword triggers + structural markers)
_BIAS_TEMPLATES: Dict[BiasType, Dict[str, Any]] = {
    BiasType.ANCHOR_NUMERIC: {
        "keywords": ["number", "figure", "initial", "first", "starting", "baseline"],
        "structural": "numeric_anchor_present",
        "weight": 1.0,
    },
    BiasType.ANCHOR_NARRATIVE: {
        "keywords": ["story", "narrative", "example", "case", "anecdote", "once"],
        "structural": "narrative_anchor",
        "weight": 0.9,
    },
    BiasType.ANCHOR_PRIMACY: {
        "keywords": ["first", "initial", "originally", "began", "started"],
        "structural": "primacy_ordering",
        "weight": 0.85,
    },
    BiasType.AVAILABILITY_RECENCY: {
        "keywords": ["recently", "just", "latest", "new", "current", "nowadays"],
        "structural": "recency_emphasis",
        "weight": 1.0,
    },
    BiasType.AVAILABILITY_VIVIDNESS: {
        "keywords": ["terrible", "amazing", "shocking", "horrifying", "incredible"],
        "structural": "vivid_language",
        "weight": 0.95,
    },
    BiasType.AVAILABILITY_SALIENCE: {
        "keywords": ["everyone", "always", "never", "obvious", "clearly"],
        "structural": "salience_overgeneralization",
        "weight": 0.90,
    },
    BiasType.CONJUNCTION_FALLACY: {
        "keywords": ["and", "also", "both", "specific", "particular", "detailed"],
        "structural": "conjunction_specificity",
        "weight": 0.85,
    },
    BiasType.BASE_RATE_NEGLECT: {
        "keywords": ["typical", "example", "specific", "instance", "case"],
        "structural": "missing_base_rate",
        "weight": 0.90,
    },
    BiasType.GAMBLER_FALLACY: {
        "keywords": ["due", "overdue", "streak", "pattern", "bound to", "must"],
        "structural": "sequence_expectation",
        "weight": 0.85,
    },
    BiasType.LOSS_AVERSION_FRAME: {
        "keywords": ["lose", "risk", "danger", "threat", "cost", "miss out"],
        "structural": "loss_frame",
        "weight": 1.0,
    },
    BiasType.DEFAULT_EFFECT_FRAME: {
        "keywords": ["default", "standard", "usual", "normal", "traditional"],
        "structural": "status_quo_frame",
        "weight": 0.80,
    },
    BiasType.CONTRAST_EFFECT_FRAME: {
        "keywords": ["compared", "versus", "relative", "better than", "worse than"],
        "structural": "contrast_juxtaposition",
        "weight": 0.85,
    },
    BiasType.CONFIRMATION_SEARCH: {
        "keywords": ["proves", "confirms", "supports", "evidence for", "shows that"],
        "structural": "one_sided_evidence",
        "weight": 1.0,
    },
    BiasType.CONFIRMATION_INTERPRET: {
        "keywords": ["obviously", "clearly", "of course", "naturally", "as expected"],
        "structural": "interpretive_certainty",
        "weight": 0.95,
    },
    BiasType.DISCONFIRMATION_NEGLECT: {
        "keywords": ["but", "however", "despite", "although", "yet", "ignore"],
        "structural": "disconfirm_dismiss",
        "weight": 0.90,
    },
    BiasType.AUTHORITY_BIAS: {
        "keywords": ["expert", "authority", "professor", "doctor", "study says"],
        "structural": "authority_appeal",
        "weight": 0.85,
    },
    BiasType.BANDWAGON_EFFECT: {
        "keywords": ["everyone", "majority", "most people", "popular", "trending"],
        "structural": "consensus_appeal",
        "weight": 0.80,
    },
    BiasType.IN_GROUP_BIAS: {
        "keywords": ["we", "our", "us", "them", "they", "those people"],
        "structural": "group_distinction",
        "weight": 0.85,
    },
    BiasType.HINDSIGHT_BIAS: {
        "keywords": ["knew it", "obvious", "predictable", "should have", "inevitable"],
        "structural": "retrospective_certainty",
        "weight": 0.90,
    },
    BiasType.PLANNING_FALLACY: {
        "keywords": ["easy", "quick", "simple", "just", "only takes", "straightforward"],
        "structural": "effort_underestimate",
        "weight": 0.85,
    },
    BiasType.SUNK_COST_BIAS: {
        "keywords": ["already", "invested", "spent", "committed", "too far", "wasted"],
        "structural": "past_investment_anchor",
        "weight": 0.90,
    },
    BiasType.SELF_SERVING_ATTRIB: {
        "keywords": ["i did", "my effort", "because i", "thanks to me", "their fault"],
        "structural": "asymmetric_attribution",
        "weight": 0.90,
    },
    BiasType.DUNNING_KRUGER: {
        "keywords": ["i know", "simple", "easy", "obvious", "anyone can"],
        "structural": "overconfidence_on_complexity",
        "weight": 0.85,
    },
    BiasType.OPTIMISM_BIAS: {
        "keywords": ["definitely", "certainly", "will work", "guaranteed", "no doubt"],
        "structural": "unrealistic_positive",
        "weight": 0.85,
    },
}


# Category → member bias types mapping
_CATEGORY_MEMBERS: Dict[BiasCategory, List[BiasType]] = {
    BiasCategory.ANCHORING: [
        BiasType.ANCHOR_NUMERIC, BiasType.ANCHOR_NARRATIVE, BiasType.ANCHOR_PRIMACY,
    ],
    BiasCategory.AVAILABILITY: [
        BiasType.AVAILABILITY_RECENCY, BiasType.AVAILABILITY_VIVIDNESS,
        BiasType.AVAILABILITY_SALIENCE,
    ],
    BiasCategory.REPRESENTATIVENESS: [
        BiasType.CONJUNCTION_FALLACY, BiasType.BASE_RATE_NEGLECT,
        BiasType.GAMBLER_FALLACY,
    ],
    BiasCategory.FRAMING: [
        BiasType.LOSS_AVERSION_FRAME, BiasType.DEFAULT_EFFECT_FRAME,
        BiasType.CONTRAST_EFFECT_FRAME,
    ],
    BiasCategory.CONFIRMATION: [
        BiasType.CONFIRMATION_SEARCH, BiasType.CONFIRMATION_INTERPRET,
        BiasType.DISCONFIRMATION_NEGLECT,
    ],
    BiasCategory.SOCIAL: [
        BiasType.AUTHORITY_BIAS, BiasType.BANDWAGON_EFFECT,
        BiasType.IN_GROUP_BIAS,
    ],
    BiasCategory.TEMPORAL: [
        BiasType.HINDSIGHT_BIAS, BiasType.PLANNING_FALLACY,
        BiasType.SUNK_COST_BIAS,
    ],
    BiasCategory.SELF_SERVING: [
        BiasType.SELF_SERVING_ATTRIB, BiasType.DUNNING_KRUGER,
        BiasType.OPTIMISM_BIAS,
    ],
}


def _bias_type_to_category(bt: BiasType) -> BiasCategory:
    for cat, members in _CATEGORY_MEMBERS.items():
        if bt in members:
            return cat
    return BiasCategory.ANCHORING  # fallback


@dataclass(frozen=True)
class BiasDetectionConfig:
    """Immutable tuning knobs for the Bias Detection Engine."""

    # --- Mode thresholds (posterior P ≥ theta → flag) ---
    theta_normal:     float = 0.45
    theta_dev:        float = 0.20
    theta_learning:   float = 0.35
    theta_reflective: float = 0.30
    theta_rem_normal: float = 0.40
    theta_rem_dream:  float = 0.60

    # --- Template matching weights ---
    w_keyword:     float = 0.35  # keyword hit density
    w_structural:  float = 0.30  # structural pattern match
    w_contextual:  float = 0.35  # contextual reinforcement (multi-bias co-occurrence)

    # --- Bayesian update ---
    prior_base:    float = 0.10  # P(bias) base rate per type
    alpha_update:  float = 0.30  # Bayesian evidence weight

    # --- Severity mapping ---
    severity_low:      float = 0.30
    severity_moderate:  float = 0.50
    severity_high:      float = 0.70
    severity_critical:  float = 0.85

    # --- Neurochemical coupling ---
    beta_ach_attention:  float = 0.12   # ACh attention load
    beta_ne_salience:    float = 0.10   # NE salience alert
    rho_5ht2a_flex:      float = 0.08   # 5-HT2A flexibility
    beta_da_novelty:     float = 0.10   # DA novel-bias detection reward
    beta_cor_threat:     float = 0.08   # Cortisol when threat-frame bias found
    psi_beta_osc:        float = 0.06   # Beta oscillation boost

    # --- Stochastic distribution params ---
    gamma_alpha:  float = 2.0   # Gamma shape for ACh/DA
    gamma_theta:  float = 0.30  # Gamma scale
    poisson_lam:  float = 1.5   # Poisson lambda for NE


# =====================================================================
# Data types -- frozen outputs
# =====================================================================


@dataclass(frozen=True)
class BiasFlag:
    """
    Structured output for a single detected cognitive bias.

    This replaces the stub BiasFlag in logic_trap_detection_engine.
    Downstream engines (Logic Trap, Heuristic Bias) consume these.
    """
    bias_id:          str            = field(default_factory=lambda: str(uuid.uuid4()))
    bias_type:        BiasType       = BiasType.ANCHOR_NUMERIC
    bias_category:    BiasCategory   = BiasCategory.ANCHORING
    source_tag:       SourceTag      = SourceTag.USER_INPUT
    confidence:       float          = 0.0     # posterior P(bias | evidence) [0, 1]
    severity:         SeverityLevel  = SeverityLevel.LOW
    keyword_score:    float          = 0.0     # keyword density [0, 1]
    structural_score: float          = 0.0     # structural pattern [0, 1]
    contextual_score: float          = 0.0     # co-occurrence reinforcement [0, 1]
    evidence_text:    str            = ""      # supporting snippet
    description:      str            = ""
    timestamp:        float          = field(default_factory=time.time)


@dataclass(frozen=True)
class BiasDetectionInput:
    """Input bundle for one engine invocation."""
    statements:         List[ProcessedStatement] = field(default_factory=list)
    memory_context:     Optional[Dict[str, Any]] = None  # retrieved memory items
    conversation_history: List[str]              = field(default_factory=list)
    active_mode:        OperationalMode          = OperationalMode.NORMAL


@dataclass(frozen=True)
class BiasDetectionNeurochem:
    """
    Neurochemical coupling signals from one Bias Detection cycle.

    Notation (Appendix S2-S3, S11):
        delta_ach   -> Delta C_ACh(t)      : attentional gating on bias scan
        delta_ne    -> Delta C_NE(t)       : salience amplification for flagged biases
        delta_5ht2a -> Delta S_5HT2A(t)    : cognitive flexibility / set-breaking
        delta_da    -> Delta C_DA(t)        : novelty reward for new bias detection
        delta_cor   -> Delta C_Cortisol(t)  : threat stress from high bias load B(t)
        beta_boost  -> Delta phi_beta(t)    : focused analytical band enhancement (S7)
    """
    delta_ach:     float = 0.0
    delta_ne:      float = 0.0
    delta_5ht2a:   float = 0.0
    delta_da:      float = 0.0
    delta_cor:     float = 0.0
    beta_boost:    float = 0.0


@dataclass(frozen=True)
class BiasDetectionResult:
    """Full output of one Bias Detection Engine cycle."""
    flags:                List[BiasFlag]         = field(default_factory=list)
    category_counts:      Dict[str, int]         = field(default_factory=dict)
    total_statements:     int                    = 0
    total_flagged:        int                    = 0
    bias_load:            float                  = 0.0   # B(t) composite [0, 1]
    clean_pass:           bool                   = True
    neurochemical_signals: BiasDetectionNeurochem       = field(default_factory=BiasDetectionNeurochem)
    processing_time_ms:   float                  = 0.0
    metadata:             Dict[str, Any]         = field(default_factory=dict)


# =====================================================================
# Mutable engine state
# =====================================================================


@dataclass
class BiasDetectionState:
    """Running state for neurochemical modulation."""
    ach_level:  float = 0.0
    ne_level:   float = 0.0
    da_level:   float = 0.0
    cor_level:  float = 0.0


# =====================================================================
# Pure helper functions
# =====================================================================


def compute_keyword_score(text: str, keywords: List[str]) -> float:
    """Fraction of template keywords found in *text* (case-insensitive)."""
    if not keywords:
        return 0.0
    lower = text.lower()
    hits = sum(1 for kw in keywords if kw in lower)
    return min(1.0, hits / max(1, len(keywords)))


def compute_structural_score(text: str, structural_marker: str) -> float:
    """
    Heuristic structural pattern detector.  Returns [0, 1].

    Each structural marker has a lightweight regex-free heuristic.
    """
    lower = text.lower()
    # Map structural markers to simple heuristics
    _heuristics = {
        "numeric_anchor_present": lambda t: any(c.isdigit() for c in t),
        "narrative_anchor": lambda t: any(w in t for w in ["once", "story", "imagine"]),
        "primacy_ordering": lambda t: any(w in t for w in ["first", "initial", "originally"]),
        "recency_emphasis": lambda t: any(w in t for w in ["recently", "just now", "latest"]),
        "vivid_language": lambda t: any(w in t for w in ["!", "terrible", "amazing", "shocking"]),
        "salience_overgeneralization": lambda t: any(w in t for w in ["everyone", "always", "never"]),
        "conjunction_specificity": lambda t: t.count(" and ") >= 2,
        "missing_base_rate": lambda t: not any(c.isdigit() for c in t) and "percent" not in t and "%" not in t,
        "sequence_expectation": lambda t: any(w in t for w in ["streak", "due", "bound to"]),
        "loss_frame": lambda t: any(w in t for w in ["lose", "risk", "danger", "threat"]),
        "status_quo_frame": lambda t: any(w in t for w in ["default", "standard", "usual"]),
        "contrast_juxtaposition": lambda t: any(w in t for w in ["compared", "versus", "better than"]),
        "one_sided_evidence": lambda t: any(w in t for w in ["proves", "confirms"]) and not any(w in t for w in ["however", "but", "although"]),
        "interpretive_certainty": lambda t: any(w in t for w in ["obviously", "clearly", "of course"]),
        "disconfirm_dismiss": lambda t: any(w in t for w in ["irrelevant", "doesn't matter", "ignore"]),
        "authority_appeal": lambda t: any(w in t for w in ["expert", "professor", "study"]),
        "consensus_appeal": lambda t: any(w in t for w in ["everyone", "most people", "popular"]),
        "group_distinction": lambda t: ("them" in t or "they" in t) and ("we" in t or "us" in t),
        "retrospective_certainty": lambda t: any(w in t for w in ["knew it", "predictable", "inevitable"]),
        "effort_underestimate": lambda t: any(w in t for w in ["easy", "quick", "simple"]) and any(w in t for w in ["just", "only"]),
        "past_investment_anchor": lambda t: any(w in t for w in ["already", "invested", "committed"]),
        "asymmetric_attribution": lambda t: (any(w in t for w in ["i did", "my effort"]) or any(w in t for w in ["their fault", "they failed"])),
        "overconfidence_on_complexity": lambda t: any(w in t for w in ["easy", "simple", "anyone can"]),
        "unrealistic_positive": lambda t: any(w in t for w in ["definitely", "guaranteed", "no doubt"]),
    }
    fn = _heuristics.get(structural_marker)
    if fn is None:
        return 0.0
    return 1.0 if fn(lower) else 0.0


def compute_contextual_score(
    bias_type: BiasType,
    other_scores: Dict[BiasType, float],
) -> float:
    """
    Co-occurrence reinforcement: if related biases in the same category are
    also elevated, this bias is more likely genuine.
    """
    cat = _bias_type_to_category(bias_type)
    members = _CATEGORY_MEMBERS.get(cat, [])
    sibling_scores = [other_scores[bt] for bt in members if bt != bias_type and bt in other_scores]
    if not sibling_scores:
        return 0.0
    return min(1.0, sum(sibling_scores) / len(sibling_scores))


def fuse_scores(
    keyword: float,
    structural: float,
    contextual: float,
    cfg: BiasDetectionConfig,
) -> float:
    """Weighted fusion → raw confidence [0, 1]."""
    return (
        cfg.w_keyword * keyword
        + cfg.w_structural * structural
        + cfg.w_contextual * contextual
    )


def bayesian_update(prior: float, evidence: float, alpha: float) -> float:
    """
    Simple Bayesian posterior:
        P(bias|E) = prior + alpha * evidence * (1 - prior)
    Clamped to [0, 1].
    """
    posterior = prior + alpha * evidence * (1.0 - prior)
    return _clamp(posterior)


def classify_severity(confidence: float, cfg: BiasDetectionConfig) -> SeverityLevel:
    """Map posterior confidence to severity tier."""
    if confidence >= cfg.severity_critical:
        return SeverityLevel.CRITICAL
    if confidence >= cfg.severity_high:
        return SeverityLevel.HIGH
    if confidence >= cfg.severity_moderate:
        return SeverityLevel.MODERATE
    return SeverityLevel.LOW


def resolve_threshold(mode: OperationalMode, cfg: BiasDetectionConfig) -> float:
    """Mode-dependent detection threshold."""
    return {
        OperationalMode.NORMAL:     cfg.theta_normal,
        OperationalMode.DEV:        cfg.theta_dev,
        OperationalMode.LEARNING:   cfg.theta_learning,
        OperationalMode.REFLECTIVE: cfg.theta_reflective,
        OperationalMode.REM_NORMAL: cfg.theta_rem_normal,
        OperationalMode.REM_DREAM:  cfg.theta_rem_dream,
    }.get(mode, cfg.theta_normal)


def compute_bias_load(flags: List[BiasFlag]) -> float:
    """
    Composite bias load B(t):
        B(t) = sum_f confidence(f) * severity_weight(f)
    Clamped to [0, 1].
    """
    if not flags:
        return 0.0
    _sev_w = {
        SeverityLevel.LOW: 0.25,
        SeverityLevel.MODERATE: 0.50,
        SeverityLevel.HIGH: 0.75,
        SeverityLevel.CRITICAL: 1.0,
    }
    total = sum(f.confidence * _sev_w.get(f.severity, 0.25) for f in flags)
    return min(1.0, total)


def compute_neurochem_signals(
    bias_load: float,
    flags: List[BiasFlag],
    cfg: BiasDetectionConfig,
    rng: np.random.Generator,
) -> BiasDetectionNeurochem:
    """
    Neurochemical coupling from bias detection output.

    ACh -- sustained attention for bias analysis
    NE  -- salience alert for high-confidence bias
    5-HT2A -- metacognitive flexibility during analysis
    DA  -- novelty reward when a bias is newly detected
    COR -- threat signal when framing / social biases are found
    Beta -- oscillatory boost during active analysis
    """
    if bias_load <= 0.0 and not flags:
        return BiasDetectionNeurochem()

    # ACh: meta-attentive load
    ach_noise = float(rng.gamma(cfg.gamma_alpha, cfg.gamma_theta))
    delta_ach = cfg.beta_ach_attention * bias_load * ach_noise

    # NE: fires for high-confidence biases
    high_conf = [f for f in flags if f.confidence >= 0.7]
    ne_impulse = float(rng.poisson(cfg.poisson_lam)) if high_conf else 0.0
    delta_ne = cfg.beta_ne_salience * (len(high_conf) / max(1, len(flags))) * ne_impulse

    # 5-HT2A: flexibility during analysis
    delta_5ht2a = cfg.rho_5ht2a_flex * (1.0 if flags else 0.0) * bias_load

    # DA: novelty reward for detecting biases
    da_noise = float(rng.gamma(cfg.gamma_alpha, cfg.gamma_theta))
    delta_da = cfg.beta_da_novelty * len(flags) / max(1, 10) * da_noise  # normalized by ~10

    # COR: threat signal for framing and social biases
    threat_cats = {BiasCategory.FRAMING, BiasCategory.SOCIAL}
    threat_flags = [f for f in flags if f.bias_category in threat_cats]
    delta_cor = cfg.beta_cor_threat * len(threat_flags) / max(1, len(flags)) * bias_load if threat_flags else 0.0

    # Beta oscillation boost
    beta_boost = cfg.psi_beta_osc * (1.0 if flags else 0.0)

    return BiasDetectionNeurochem(
        delta_ach=delta_ach,
        delta_ne=delta_ne,
        delta_5ht2a=delta_5ht2a,
        delta_da=delta_da,
        delta_cor=delta_cor,
        beta_boost=beta_boost,
    )


# =====================================================================
# Engine class
# =====================================================================


class BiasDetectionEngine:
    """
    Engine 5 -- Bias Detection Engine.

    Detects cognitive biases in processed statements from any pipeline source.
    Three-stage detection: template matching -> Bayesian update -> contextual
    reinforcement.

    API
    ---
    configure(mode)     -- set operational mode
    update_neurochem_state(state_dict) -- inject external NT levels
    process(bias_input) -- run detection pipeline, return BiasDetectionResult
    get_status()        -- introspection
    """

    engine_id = "bias_detection_engine"
    cluster   = "detection"

    def __init__(
        self,
        config: Optional[BiasDetectionConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._cfg = config or BiasDetectionConfig()
        self._rng = rng or np.random.default_rng()
        self._mode = OperationalMode.NORMAL
        self._state = BiasDetectionState()
        self._cycle_count = 0

    # ----- Configuration --------------------------------------------------

    def configure(self, mode: OperationalMode) -> None:
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """Inject current neurochemical levels for bidirectional feedback."""
        if "ach" in state_dict:
            self._state.ach_level = state_dict["ach"]
        if "ne" in state_dict:
            self._state.ne_level = state_dict["ne"]
        if "da" in state_dict:
            self._state.da_level = state_dict["da"]
        if "cor" in state_dict:
            self._state.cor_level = state_dict["cor"]

    # ----- Main pipeline --------------------------------------------------

    def process(self, bias_input: BiasDetectionInput) -> BiasDetectionResult:
        """
        Run the full bias detection pipeline on *bias_input*.

        Pipeline stages:
          1. Per-statement keyword + structural scoring for each bias type
          2. Bayesian posterior update using co-occurrence evidence
          3. Threshold gating -> emit BiasFlag per surviving detection
          4. Neurochemical coupling
        """
        t0 = time.perf_counter()
        self._cycle_count += 1

        mode = bias_input.active_mode
        threshold = resolve_threshold(mode, self._cfg)

        # Bidirectional feedback: high cortisol → lower thresholds
        if self._state.cor_level > 0.5:
            threshold *= 0.85
        # Low DA → more cautious → lower threshold (catch more)
        if self._state.da_level < 0.25 and self._state.da_level > 0.0:
            threshold *= 0.90

        all_flags: List[BiasFlag] = []

        for stmt in bias_input.statements:
            text = stmt.raw_text
            if not text.strip():
                continue

            # Stage 1: per-bias keyword + structural scoring
            raw_scores: Dict[BiasType, float] = {}
            keyword_scores: Dict[BiasType, float] = {}
            structural_scores: Dict[BiasType, float] = {}

            for bt, tmpl in _BIAS_TEMPLATES.items():
                kw_score = compute_keyword_score(text, tmpl["keywords"])
                st_score = compute_structural_score(text, tmpl["structural"])
                keyword_scores[bt] = kw_score
                structural_scores[bt] = st_score
                raw_scores[bt] = (
                    self._cfg.w_keyword * kw_score
                    + self._cfg.w_structural * st_score
                ) * tmpl.get("weight", 1.0)

            # Stage 2: contextual reinforcement + Bayesian update
            for bt in BiasType:
                ctx_score = compute_contextual_score(bt, raw_scores)
                fused = fuse_scores(
                    keyword_scores.get(bt, 0.0),
                    structural_scores.get(bt, 0.0),
                    ctx_score,
                    self._cfg,
                )
                posterior = bayesian_update(
                    self._cfg.prior_base,
                    fused,
                    self._cfg.alpha_update,
                )

                # Stage 3: threshold gating
                if posterior >= threshold:
                    severity = classify_severity(posterior, self._cfg)
                    flag = BiasFlag(
                        bias_type=bt,
                        bias_category=_bias_type_to_category(bt),
                        source_tag=stmt.source_tag,
                        confidence=round(posterior, 4),
                        severity=severity,
                        keyword_score=round(keyword_scores.get(bt, 0.0), 4),
                        structural_score=round(structural_scores.get(bt, 0.0), 4),
                        contextual_score=round(ctx_score, 4),
                        evidence_text=text[:200],
                        description=f"Detected {bt.value} bias ({severity.value})",
                    )
                    all_flags.append(flag)

        # Aggregate
        category_counts: Dict[str, int] = {}
        for f in all_flags:
            key = f.bias_category.value
            category_counts[key] = category_counts.get(key, 0) + 1

        bias_load = compute_bias_load(all_flags)

        neurochem = compute_neurochem_signals(
            bias_load, all_flags, self._cfg, self._rng,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        return BiasDetectionResult(
            flags=all_flags,
            category_counts=category_counts,
            total_statements=len(bias_input.statements),
            total_flagged=len(all_flags),
            bias_load=round(bias_load, 4),
            clean_pass=(len(all_flags) == 0),
            neurochemical_signals=neurochem,
            processing_time_ms=round(elapsed, 3),
            metadata={
                "mode": mode.value,
                "threshold": round(threshold, 4),
                "cycle": self._cycle_count,
            },
        )

    # ----- Introspection --------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "mode": self._mode.value,
            "cycle_count": self._cycle_count,
            "state": {
                "ach_level": self._state.ach_level,
                "ne_level": self._state.ne_level,
                "da_level": self._state.da_level,
                "cor_level": self._state.cor_level,
            },
        }
