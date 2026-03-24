"""
Engine 11 -- Input Relevance Evaluation Engine  (``input_relevance_evaluation_engine``)
=======================================================================================
Triage module that evaluates how relevant, important, and urgent the current
input is relative to ongoing context, active tasks, and system state.

Two-phase architecture:
  * **Phase 1 — Early Triage** (steps a-b): cheap scoring using STMM +
    urgency read from Extractor 5.  Produces a preliminary processing depth
    recommendation that gates downstream engine depth.
  * **Phase 2 — Post-Contrast Refinement** (after step d): re-scores using
    Memory Contrast results + Identity Memory hits.  Produces the
    authoritative relevance evaluation consumed by all step (e) engines.

Relevance is decomposed into five orthogonal dimensions:
  1. Contextual Continuity  (CC) — TF-IDF cosine vs active conversation
  2. Task Alignment         (TA) — intent vector cosine vs previous cycle
  3. Novelty                (NV) — inverse memory similarity
  4. Emotional Salience     (ES) — lexical + structural proxy → passthrough
  5. Identity Resonance     (IR) — keyword + LTMM identity match

Urgency is READ from Extractor 5 (``urgency_forecast.py``), not recomputed.

Neurochemical coupling:
  ACh  — attention allocation (relevance × novelty)
  DA   — novelty response (gated by relevance floor)
  NE   — relevance-modulated vigilance (additive to Extractor 5)
  5-HT — contextual stability signal
  COR  — relevance-urgency conflict stress

Key design decisions:
  * Pure Python engine — no SOAR, no external cognitive architecture.
  * Two scoring pipelines converge via non-linear interaction fusion.
  * Processing depth recommendation is the primary output consumed by the
    pipeline orchestrator.
  * Identity resonance IR ≥ 0.60 forces ≥ DEEP regardless of other scores.
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


class ProcessingDepth(str, Enum):
    """Processing depth recommendation for the pipeline orchestrator."""
    SHALLOW  = "shallow"    # Minimal — skip optional engines
    STANDARD = "standard"   # Normal — standard pipeline
    DEEP     = "deep"       # Enhanced — all engines at full depth
    CRITICAL = "critical"   # Maximum — priority interrupt, all resources


class Quadrant(str, Enum):
    """Relevance-urgency quadrant classification."""
    Q1_PRIORITY_INTERRUPT = "Q1_priority_interrupt"   # High R, High U
    Q2_DEEP_PROCESSING    = "Q2_deep_processing"      # High R, Low U
    Q3_ACKNOWLEDGE_REDIRECT = "Q3_acknowledge_redirect"  # Low R, High U
    Q4_SHALLOW_PROCESSING = "Q4_shallow_processing"   # Low R, Low U


class RelevanceFlagType(str, Enum):
    """Types of anomalous relevance conditions."""
    TOPIC_DISCONTINUITY       = "topic_discontinuity"
    INTENT_SHIFT              = "intent_shift"
    IDENTITY_CHALLENGE        = "identity_challenge"
    EMOTIONAL_OVERRIDE        = "emotional_override"
    RELEVANCE_URGENCY_CONFLICT = "relevance_urgency_conflict"
    NOVELTY_SATURATION        = "novelty_saturation"
    PHASE_DIVERGENCE          = "phase_divergence"


class FlagSeverity(str, Enum):
    """Severity levels for relevance flags (reward-compatible)."""
    INFO    = "info"
    WARNING = "warning"
    RISK    = "risk"


# =====================================================================
# Configuration
# =====================================================================

# --- Emotion keyword lexicon (surface-level proxy for Phase 1) ---
_EMOTION_KEYWORDS: List[str] = [
    "angry", "furious", "rage", "hate", "love", "happy", "sad", "depressed",
    "anxious", "scared", "terrified", "excited", "disgusted", "ashamed",
    "guilty", "jealous", "hurt", "frustrated", "overwhelmed", "devastated",
    "ecstatic", "miserable", "afraid", "panic", "crying", "tears",
    "heartbroken", "grateful", "hopeless", "desperate", "lonely", "betrayed",
    "worried", "stressed", "confused", "shocked", "surprised", "annoyed",
    "resentful", "bitter", "euphoric", "thrilled", "horrified", "disgusting",
]

# --- Identity keyword lexicon (surface-level proxy for Phase 1) ---
_IDENTITY_KEYWORDS: Dict[str, float] = {
    # Direct identity questions
    "who are you": 1.0,
    "what are you": 1.0,
    "your identity": 0.9,
    "your values": 0.9,
    "your beliefs": 0.9,
    "do you believe": 0.8,
    "your purpose": 0.8,
    "your nature": 0.8,
    "your name": 0.7,
    # Identity challenges
    "you're just": 0.9,
    "you are just": 0.9,
    "you can't really": 0.8,
    "you don't actually": 0.8,
    "you're not real": 0.9,
    "you are not real": 0.9,
    "you're only": 0.7,
    "just a machine": 0.9,
    "just a program": 0.9,
    "just an ai": 0.9,
    # Value-laden
    "consciousness": 0.6,
    "sentient": 0.7,
    "self-aware": 0.7,
    "alive": 0.6,
    "soul": 0.6,
    "free will": 0.7,
    "moral": 0.5,
    "ethical": 0.5,
}

# --- Intent proxy keywords (lightweight subset of IntentionMapEngine) ---
_INTENT_PROXY_KEYWORDS: Dict[str, List[str]] = {
    "exploration":    ["how", "why", "what if", "explain", "tell me about", "curious", "wonder", "explore"],
    "challenge":      ["disagree", "wrong", "actually", "but", "however", "no", "argue", "debate", "prove"],
    "pragmatism":     ["help me", "do this", "fix", "solve", "answer", "quickly", "need to", "how do i"],
    "discharge":      ["ugh", "vent", "frustrated", "just need to talk", "listen", "tired of"],
    "symbolism":      ["metaphor", "symbol", "represent", "meaning", "deeper", "archetype", "myth"],
    "confrontation":  ["attack", "destroy", "fight", "expose", "lie", "fraud", "cheat"],
    "defensiveness":  ["not my fault", "unfair", "misunderstand", "that's not what i", "you're twisting"],
    "submission":     ["whatever", "fine", "okay i guess", "you decide", "i don't know", "i give up"],
}


@dataclass(frozen=True)
class IREConfig:
    """Immutable configuration for the Input Relevance Evaluation Engine."""

    # --- Relevance dimension weights (Normal mode defaults) ---
    w_cc: float = 0.25   # Contextual Continuity
    w_ta: float = 0.20   # Task Alignment
    w_nv: float = 0.20   # Novelty
    w_es: float = 0.15   # Emotional Salience
    w_ir: float = 0.20   # Identity Resonance

    # --- Priority fusion parameters ---
    w_r:         float = 0.45   # Relevance weight in linear term
    w_u:         float = 0.35   # Urgency weight in linear term
    alpha_base:  float = 0.70   # Linear combination scaling
    alpha_interact: float = 0.20  # R × U interaction amplification
    alpha_override: float = 0.10  # Emotional/identity override floor

    # --- Urgency normalization ---
    gamma_u: float = 3.0   # Exponential saturation rate for U normalization

    # --- Processing depth thresholds (Normal mode) ---
    theta_shallow_standard: float = 0.25
    theta_standard_deep:    float = 0.55
    theta_deep_critical:    float = 0.80

    # --- Override thresholds ---
    ir_force_deep:          float = 0.60   # IR ≥ this → forced ≥ DEEP
    es_force_deep:          float = 0.80   # ES ≥ this → forced ≥ DEEP
    breach_count_force_deep: int  = 3      # ≥ 3 breaches → forced ≥ DEEP
    u_norm_force_critical:  float = 0.90   # U_norm ≥ this → forced CRITICAL

    # --- Phase 2 refinement weights ---
    w_stmm_cc: float = 0.60   # STMM weight in CC refinement
    w_mc_cc:   float = 0.40   # Memory Contrast weight in CC refinement

    # --- Novelty echo penalty ---
    echo_penalty_per_item: float = 0.15   # NV penalty per detected echo

    # --- Identity keyword normalization ---
    ir_keyword_norm: float = 3.0   # Hits / norm → saturated score

    # --- Neurochemical coupling ---
    beta_ach:     float = 0.10   # ACh attention allocation
    w_ach_r:      float = 0.60   # Relevance weight for ACh
    w_ach_nv:     float = 0.40   # Novelty weight for ACh
    beta_da:      float = 0.08   # DA novelty response
    da_r_floor:   float = 0.30   # Minimum R to allow DA
    beta_ne:      float = 0.06   # NE relevance-modulated vigilance
    ne_p_gate:    float = 0.40   # Minimum P(t) for NE emission
    beta_5ht:     float = 0.05   # 5-HT stability signal
    beta_cor:     float = 0.04   # Cortisol conflict stress
    cor_conflict_gate: float = 0.30  # Minimum conflict for cortisol

    # --- Stochastic distribution params ---
    gamma_alpha:  float = 2.0
    gamma_theta:  float = 0.50
    poisson_lam:  float = 1.5

    # --- Confidence estimation ---
    sigma_phase1: float = 0.15   # Measurement noise std for Phase 1
    sigma_phase2: float = 0.08   # Measurement noise std for Phase 2
    sigma_max:    float = 0.20   # Maximum noise for confidence normalization

    # --- Novelty saturation tracking ---
    novelty_saturation_threshold: float = 0.05
    novelty_saturation_cycles:    int   = 5


# =====================================================================
# Mode-specific configuration tables
# =====================================================================

_MODE_DIMENSION_WEIGHTS: Dict[OperationalMode, Dict[str, float]] = {
    OperationalMode.NORMAL:     {"w_cc": 0.25, "w_ta": 0.20, "w_nv": 0.20, "w_es": 0.15, "w_ir": 0.20},
    OperationalMode.DEV:        {"w_cc": 0.15, "w_ta": 0.15, "w_nv": 0.30, "w_es": 0.10, "w_ir": 0.30},
    OperationalMode.LEARNING:   {"w_cc": 0.20, "w_ta": 0.15, "w_nv": 0.35, "w_es": 0.10, "w_ir": 0.20},
    OperationalMode.REFLECTIVE: {"w_cc": 0.30, "w_ta": 0.25, "w_nv": 0.15, "w_es": 0.15, "w_ir": 0.15},
    OperationalMode.REM_NORMAL: {"w_cc": 0.30, "w_ta": 0.25, "w_nv": 0.15, "w_es": 0.10, "w_ir": 0.20},
    OperationalMode.REM_DREAM:  {"w_cc": 0.20, "w_ta": 0.10, "w_nv": 0.40, "w_es": 0.10, "w_ir": 0.20},
}

_MODE_DEPTH_THRESHOLDS: Dict[OperationalMode, Tuple[float, float, float]] = {
    #                              SHALLOW→STANDARD  STANDARD→DEEP  DEEP→CRITICAL
    OperationalMode.NORMAL:     (0.25, 0.55, 0.80),
    OperationalMode.DEV:        (0.15, 0.40, 0.70),
    OperationalMode.LEARNING:   (0.20, 0.45, 0.75),
    OperationalMode.REFLECTIVE: (0.30, 0.60, 0.85),
    OperationalMode.REM_NORMAL: (0.30, 0.60, 0.85),
    OperationalMode.REM_DREAM:  (0.40, 0.70, 0.90),
}


# =====================================================================
# Data types — frozen outputs
# =====================================================================


@dataclass(frozen=True)
class RelevanceDimensionScores:
    """Per-dimension relevance breakdown."""
    contextual_continuity: float = 0.0   # CC(t) [0, 1]
    task_alignment:        float = 0.0   # TA(t) [0, 1]
    novelty:               float = 0.0   # NV(t) [0, 1]
    emotional_salience:    float = 0.0   # ES(t) [0, 1]
    identity_resonance:    float = 0.0   # IR(t) [0, 1]


@dataclass(frozen=True)
class RelevanceFlag:
    """Structured flag for anomalous relevance conditions."""
    flag_id:          str              = field(default_factory=lambda: str(uuid.uuid4()))
    flag_type:        RelevanceFlagType = RelevanceFlagType.TOPIC_DISCONTINUITY
    confidence:       float            = 0.0
    severity:         FlagSeverity     = FlagSeverity.INFO
    description:      str              = ""
    source_dimension: str              = ""
    timestamp:        float            = field(default_factory=time.time)


@dataclass(frozen=True)
class IRENeuroChemSignals:
    """
    Neurochemical coupling signals from the IRE Engine.

    Notation (per spec Section 9):
        delta_ach  → Δ C_ACh(t)       : attention allocation
        delta_da   → Δ C_DA(t)        : novelty response (gated by R floor)
        delta_ne   → Δ C_NE(t)        : relevance-modulated vigilance
        delta_5ht  → Δ C_5HT(t)       : contextual stability
        delta_cor  → Δ C_Cortisol(t)  : relevance-urgency conflict stress
    """
    delta_ach: float = 0.0
    delta_da:  float = 0.0
    delta_ne:  float = 0.0
    delta_5ht: float = 0.0
    delta_cor: float = 0.0


@dataclass(frozen=True)
class IREPhase1Input:
    """Input bundle for Phase 1 (Early Triage, steps a-b)."""
    # From step (a) — Tokenizer + Semantic Expander
    current_text:          str                           = ""
    tokens:                List[str]                     = field(default_factory=list)

    # From STMM
    stmm_user_messages:    List[str]                     = field(default_factory=list)
    stmm_system_responses: List[str]                     = field(default_factory=list)
    previous_intent_vector: Optional[List[float]]        = None   # 8-element float list

    # From Extractor 5 (neurochem layer)
    urgency_risk:          float                         = 0.0
    urgency_breach_flags:  Dict[str, bool]               = field(default_factory=dict)
    urgency_smoothed_axes: Dict[str, float]              = field(default_factory=dict)

    # Mode
    active_mode:           OperationalMode               = OperationalMode.NORMAL

    # NT read ports
    nt_levels:             Dict[str, float]              = field(default_factory=dict)


@dataclass(frozen=True)
class IREPhase2Input:
    """Additional input for Phase 2 (Post-Contrast Refinement, after step d)."""
    # Phase 1 result (warm start)
    phase1_result: Optional[IREPhase1Result]             = None

    # From step (d) — Memory Contrast
    memory_contrast_scores:   List[float]                = field(default_factory=list)
    detected_echoes:          List[str]                   = field(default_factory=list)
    identity_match_scores:    List[float]                = field(default_factory=list)

    # From step (c) — Intention Map (current cycle)
    current_intent_vector:    Optional[List[float]]      = None

    # From Emotional Detection Engine (step b)
    emotional_intensity:      Optional[float]            = None


@dataclass(frozen=True)
class IREPhase1Result:
    """Output of Phase 1 (preliminary)."""
    dimensions:            RelevanceDimensionScores      = field(default_factory=RelevanceDimensionScores)
    relevance_composite:   float                         = 0.0
    urgency_normalized:    float                         = 0.0
    priority_composite:    float                         = 0.0
    processing_depth:      ProcessingDepth               = ProcessingDepth.STANDARD
    confidence:            float                         = 0.0
    phase:                 int                           = 1
    quadrant:              Quadrant                      = Quadrant.Q4_SHALLOW_PROCESSING
    neurochemical_signals: IRENeuroChemSignals           = field(default_factory=IRENeuroChemSignals)
    processing_time_ms:    float                         = 0.0
    metadata:              Dict[str, Any]                = field(default_factory=dict)


@dataclass(frozen=True)
class IREResult:
    """Output of Phase 2 (authoritative final result)."""
    dimensions:                RelevanceDimensionScores  = field(default_factory=RelevanceDimensionScores)
    relevance_composite:       float                     = 0.0
    urgency_normalized:        float                     = 0.0
    priority_composite:        float                     = 0.0
    processing_depth:          ProcessingDepth            = ProcessingDepth.STANDARD
    depth_changed_from_phase1: bool                      = False
    confidence:                float                     = 0.0
    phase:                     int                       = 2
    quadrant:                  Quadrant                  = Quadrant.Q4_SHALLOW_PROCESSING
    urgency_risk_raw:          float                     = 0.0
    urgency_breach_flags:      Dict[str, bool]           = field(default_factory=dict)
    urgency_breach_count:      int                       = 0
    phase1_priority:           float                     = 0.0
    delta_priority:            float                     = 0.0
    neurochemical_signals:     IRENeuroChemSignals       = field(default_factory=IRENeuroChemSignals)
    flags:                     List[RelevanceFlag]       = field(default_factory=list)
    processing_time_ms:        float                     = 0.0
    timestamp:                 float                     = field(default_factory=time.time)
    metadata:                  Dict[str, Any]            = field(default_factory=dict)


# =====================================================================
# Mutable engine state
# =====================================================================


@dataclass
class IREState:
    """Running state for neurochemical modulation + novelty tracking."""
    ach_level:  float = 0.0
    da_level:   float = 0.0
    ne_level:   float = 0.0
    _5ht_level:  float = 0.0   # 5-HT
    cor_level:  float = 0.0   # cortisol
    # Novelty saturation tracker
    low_novelty_streak: int = 0
    previous_cc: float = 0.0   # For topic discontinuity detection


# =====================================================================
# Pure helper functions — TF-IDF cosine
# =====================================================================


def _tokenize_simple(text: str) -> List[str]:
    """Lowercase whitespace tokenizer with minimal punctuation stripping."""
    return [w.strip(".,!?;:\"'()[]{}") for w in text.lower().split() if w.strip(".,!?;:\"'()[]{}")]


def _build_tf(tokens: List[str]) -> Dict[str, float]:
    """Term frequency (normalized by document length)."""
    if not tokens:
        return {}
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    n = len(tokens)
    return {t: c / n for t, c in counts.items()}


def _build_idf(documents: List[List[str]]) -> Dict[str, float]:
    """Inverse document frequency across a small corpus."""
    n_docs = len(documents)
    if n_docs == 0:
        return {}
    df: Dict[str, int] = {}
    for doc in documents:
        seen = set(doc)
        for t in seen:
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n_docs + 1) / (freq + 1)) + 1.0 for t, freq in df.items()}


def _tfidf_vector(tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
    """TF-IDF vector as sparse dict."""
    return {t: tf_val * idf.get(t, 1.0) for t, tf_val in tf.items()}


def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    if not a or not b:
        return 0.0
    keys = set(a.keys()) & set(b.keys())
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_tfidf_cosine(query_text: str, reference_texts: List[str]) -> List[float]:
    """
    Compute TF-IDF cosine similarity of *query_text* against each reference.
    Returns list of similarity scores ∈ [0, 1], same length as *reference_texts*.
    """
    if not reference_texts or not query_text.strip():
        return [0.0] * len(reference_texts)

    query_tokens = _tokenize_simple(query_text)
    ref_token_lists = [_tokenize_simple(rt) for rt in reference_texts]
    all_docs = [query_tokens] + ref_token_lists

    idf = _build_idf(all_docs)
    query_vec = _tfidf_vector(_build_tf(query_tokens), idf)

    scores = []
    for ref_tokens in ref_token_lists:
        ref_vec = _tfidf_vector(_build_tf(ref_tokens), idf)
        scores.append(_cosine_similarity(query_vec, ref_vec))
    return scores


# =====================================================================
# Pure helper functions — Dimension scoring
# =====================================================================


def compute_contextual_continuity_phase1(
    current_text: str,
    stmm_messages: List[str],
) -> float:
    """
    CC_phase1(t) = max cosine similarity between current input and STMM window.
    """
    if not stmm_messages or not current_text.strip():
        return 0.0
    sims = compute_tfidf_cosine(current_text, stmm_messages)
    return max(sims) if sims else 0.0


def refine_contextual_continuity_phase2(
    cc_phase1: float,
    memory_contrast_scores: List[float],
    cfg: IREConfig,
) -> float:
    """
    CC_phase2(t) = w_stmm × CC_phase1 + w_mc × max(mc_scores).
    """
    if not memory_contrast_scores:
        return cc_phase1
    max_mc = max(memory_contrast_scores)
    return min(1.0, cfg.w_stmm_cc * cc_phase1 + cfg.w_mc_cc * max_mc)


def compute_intent_proxy(text: str) -> List[float]:
    """
    Lightweight 8-element intent proxy from keyword density.
    Order: exploration, challenge, pragmatism, discharge, symbolism,
           confrontation, defensiveness, submission.
    """
    lower = text.lower()
    proxy = []
    for intent_name in ["exploration", "challenge", "pragmatism", "discharge",
                        "symbolism", "confrontation", "defensiveness", "submission"]:
        keywords = _INTENT_PROXY_KEYWORDS.get(intent_name, [])
        if not keywords:
            proxy.append(0.0)
            continue
        hits = sum(1 for kw in keywords if kw in lower)
        proxy.append(min(1.0, hits / max(1, len(keywords))))
    return proxy


def _cosine_vectors(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length float vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_task_alignment_phase1(
    current_text: str,
    previous_intent_vector: Optional[List[float]],
) -> float:
    """
    TA_phase1(t) = cosine(proxy_intent(t), prev_intent_vector(t-1)).
    Returns 0.5 (neutral prior) if no previous intent exists.
    """
    if previous_intent_vector is None or len(previous_intent_vector) == 0:
        return 0.5
    proxy = compute_intent_proxy(current_text)
    sim = _cosine_vectors(proxy, previous_intent_vector)
    # Cosine can be negative; clamp to [0, 1]
    return _clamp(sim)


def refine_task_alignment_phase2(
    ta_phase1: float,
    current_intent_vector: Optional[List[float]],
    previous_intent_vector: Optional[List[float]],
) -> float:
    """
    If current cycle's intent is available, use authoritative vector.
    """
    if current_intent_vector is None or previous_intent_vector is None:
        return ta_phase1
    sim = _cosine_vectors(current_intent_vector, previous_intent_vector)
    return _clamp(sim)


def compute_novelty_phase1(
    current_text: str,
    stmm_messages: List[str],
) -> float:
    """
    NV_phase1(t) = 1.0 - max cosine to any STMM message.
    """
    if not stmm_messages or not current_text.strip():
        return 0.5   # neutral when no reference
    sims = compute_tfidf_cosine(current_text, stmm_messages)
    max_sim = max(sims) if sims else 0.0
    return max(0.0, 1.0 - max_sim)


def compute_novelty_phase2(
    memory_contrast_scores: List[float],
    detected_echoes: List[str],
    cfg: IREConfig,
) -> float:
    """
    NV_phase2(t) = (1.0 - max_memory_sim) × (1.0 - echo_penalty).
    """
    if not memory_contrast_scores:
        return 0.5   # no memory hits → moderate novelty
    max_sim = max(memory_contrast_scores) if memory_contrast_scores else 0.0
    base_novelty = max(0.0, 1.0 - max_sim)
    echo_penalty = min(1.0, len(detected_echoes) * cfg.echo_penalty_per_item)
    return max(0.0, base_novelty * (1.0 - echo_penalty))


def compute_emotional_salience_phase1(
    text: str,
    tokens: List[str],
) -> float:
    """
    Surface-level emotional intensity proxy: lexical density + structural markers.
    """
    if not text.strip():
        return 0.0

    lower = text.lower()

    # Lexical emotion density
    total_tokens = max(1, len(tokens)) if tokens else max(1, len(lower.split()))
    emotion_hits = sum(1 for kw in _EMOTION_KEYWORDS if kw in lower)
    lex_emotion = min(1.0, emotion_hits / max(1, total_tokens) * 5.0)
    # Scale factor 5.0 because emotion words are sparse in most text

    # Structural intensity markers
    struct_score = 0.0
    if "!" in text:
        struct_score += 0.15
    # All-caps words (at least 3 chars, not common abbreviations)
    words = text.split()
    caps_count = sum(1 for w in words if w.isupper() and len(w) >= 3)
    if caps_count > 0:
        struct_score += 0.20
    # Repeated punctuation
    if "!!" in text or "??" in text or "..." in text:
        struct_score += 0.15
    # Personal pronouns + emotion patterns
    personal_patterns = ["i feel", "i am", "i'm", "you always", "you never",
                         "i can't", "i hate", "i love", "makes me"]
    if any(p in lower for p in personal_patterns):
        struct_score += 0.15
    # Message brevity (very short + emotional = high intensity)
    if len(words) <= 5 and (lex_emotion > 0.0 or struct_score > 0.0):
        struct_score += 0.10

    struct_score = min(1.0, struct_score)

    # Fusion
    w_lex = 0.55
    w_struct = 0.45
    return min(1.0, w_lex * lex_emotion + w_struct * struct_score)


def compute_identity_resonance_phase1(text: str) -> float:
    """
    Keyword-based identity relevance proxy.
    Score = sum(weights of matching keywords) / normalization_factor.
    """
    if not text.strip():
        return 0.0
    lower = text.lower()
    total_weight = 0.0
    for keyword, weight in _IDENTITY_KEYWORDS.items():
        if keyword in lower:
            total_weight += weight
    return min(1.0, total_weight / 3.0)  # normalization_factor = 3.0


def refine_identity_resonance_phase2(
    ir_phase1: float,
    identity_match_scores: List[float],
) -> float:
    """
    IR only goes UP during Phase 2.  Memory match confirms or strengthens.
    """
    if not identity_match_scores:
        return ir_phase1
    max_identity_sim = max(identity_match_scores)
    return max(ir_phase1, max_identity_sim)


# =====================================================================
# Pure helper functions — Urgency + Fusion
# =====================================================================


def normalize_urgency(urgency_risk: float, gamma_u: float) -> float:
    """
    U_norm(t) = 1.0 - exp(-γ_u × U(t)).
    Maps unbounded U(t) to [0, 1].
    """
    if urgency_risk <= 0.0:
        return 0.0
    return 1.0 - math.exp(-gamma_u * urgency_risk)


def compute_breach_count(breach_flags: Dict[str, bool]) -> int:
    """Number of urgency axes with predicted threshold breach."""
    return sum(1 for v in breach_flags.values() if v)


def compute_relevance_composite(
    dims: RelevanceDimensionScores,
    weights: Dict[str, float],
) -> float:
    """
    R(t) = Σ w_k × dim_k.
    """
    r = (
        weights.get("w_cc", 0.25) * dims.contextual_continuity
        + weights.get("w_ta", 0.20) * dims.task_alignment
        + weights.get("w_nv", 0.20) * dims.novelty
        + weights.get("w_es", 0.15) * dims.emotional_salience
        + weights.get("w_ir", 0.20) * dims.identity_resonance
    )
    return _clamp(r)


def compute_priority_composite(
    r: float,
    u_norm: float,
    es: float,
    ir: float,
    cfg: IREConfig,
) -> float:
    """
    P(t) = α_base × (w_r × R + w_u × U_norm) + α_interact × R × U_norm
           + α_override × max(ES, IR).
    Clamped to [0, 1].
    """
    linear = cfg.alpha_base * (cfg.w_r * r + cfg.w_u * u_norm)
    interaction = cfg.alpha_interact * r * u_norm
    override = cfg.alpha_override * max(es, ir)
    p = linear + interaction + override
    return _clamp(p)


def classify_quadrant(r: float, u_norm: float) -> Quadrant:
    """Map (R, U_norm) to the four quadrants."""
    if r >= 0.5 and u_norm >= 0.5:
        return Quadrant.Q1_PRIORITY_INTERRUPT
    if r >= 0.5 and u_norm < 0.5:
        return Quadrant.Q2_DEEP_PROCESSING
    if r < 0.5 and u_norm >= 0.5:
        return Quadrant.Q3_ACKNOWLEDGE_REDIRECT
    return Quadrant.Q4_SHALLOW_PROCESSING


def classify_processing_depth(
    priority: float,
    thresholds: Tuple[float, float, float],
) -> ProcessingDepth:
    """
    Threshold-based depth classification.
    thresholds = (shallow→standard, standard→deep, deep→critical).
    """
    if priority >= thresholds[2]:
        return ProcessingDepth.CRITICAL
    if priority >= thresholds[1]:
        return ProcessingDepth.DEEP
    if priority >= thresholds[0]:
        return ProcessingDepth.STANDARD
    return ProcessingDepth.SHALLOW


def apply_depth_overrides(
    depth: ProcessingDepth,
    ir: float,
    es: float,
    breach_count: int,
    u_norm: float,
    cfg: IREConfig,
) -> ProcessingDepth:
    """
    Apply mode-independent override rules.
    Depth can only be RAISED, never lowered, by overrides.
    """
    _DEPTH_ORDER = {
        ProcessingDepth.SHALLOW: 0,
        ProcessingDepth.STANDARD: 1,
        ProcessingDepth.DEEP: 2,
        ProcessingDepth.CRITICAL: 3,
    }
    current = _DEPTH_ORDER[depth]

    # Extreme urgency → CRITICAL (highest priority override)
    if u_norm >= cfg.u_norm_force_critical:
        return ProcessingDepth.CRITICAL

    forced = current

    if ir >= cfg.ir_force_deep:
        forced = max(forced, _DEPTH_ORDER[ProcessingDepth.DEEP])
    if es >= cfg.es_force_deep:
        forced = max(forced, _DEPTH_ORDER[ProcessingDepth.DEEP])
    if breach_count >= cfg.breach_count_force_deep:
        forced = max(forced, _DEPTH_ORDER[ProcessingDepth.DEEP])

    _REVERSE = {v: k for k, v in _DEPTH_ORDER.items()}
    return _REVERSE[forced]


def resolve_mode_weights(mode: OperationalMode) -> Dict[str, float]:
    """Get dimension weights for the active mode."""
    return _MODE_DIMENSION_WEIGHTS.get(mode, _MODE_DIMENSION_WEIGHTS[OperationalMode.NORMAL])


def resolve_depth_thresholds(mode: OperationalMode) -> Tuple[float, float, float]:
    """Get processing depth thresholds for the active mode."""
    return _MODE_DEPTH_THRESHOLDS.get(mode, _MODE_DEPTH_THRESHOLDS[OperationalMode.NORMAL])


def compute_confidence(sigma: float, sigma_max: float) -> float:
    """Confidence = 1.0 - σ / σ_max, clamped to [0, 1]."""
    if sigma_max <= 0.0:
        return 0.0
    return _clamp(1.0 - sigma / sigma_max)


# =====================================================================
# Pure helper functions — Neurochemical signals
# =====================================================================


def compute_neurochem_signals(
    r: float,
    u_norm: float,
    p: float,
    dims: RelevanceDimensionScores,
    cfg: IREConfig,
    rng: np.random.Generator,
) -> IRENeuroChemSignals:
    """
    Neurochemical coupling from IRE evaluation output.

    ACh -- attention allocation: relevance × novelty
    DA  -- novelty response (gated by relevance floor)
    NE  -- relevance-modulated vigilance (additive to Extractor 5)
    5HT -- contextual stability signal
    COR -- relevance-urgency conflict stress
    """
    # ACh: attention allocation
    ach_noise = float(rng.gamma(cfg.gamma_alpha, cfg.gamma_theta))
    delta_ach = cfg.beta_ach * (cfg.w_ach_r * r + cfg.w_ach_nv * dims.novelty) * ach_noise

    # DA: novelty response, gated by relevance
    da_noise = float(rng.gamma(cfg.gamma_alpha, cfg.gamma_theta))
    da_r_gate = max(r, cfg.da_r_floor)
    delta_da = cfg.beta_da * dims.novelty * da_r_gate * da_noise

    # NE: relevance-modulated vigilance (only above P threshold)
    delta_ne = 0.0
    if p > cfg.ne_p_gate:
        ne_impulse = float(rng.poisson(cfg.poisson_lam)) / max(1.0, cfg.poisson_lam)
        delta_ne = cfg.beta_ne * p * ne_impulse

    # 5-HT: contextual stability
    sht_noise = float(rng.gamma(cfg.gamma_alpha + 0.5, cfg.gamma_theta - 0.10))
    delta_5ht = cfg.beta_5ht * (dims.contextual_continuity + dims.task_alignment) / 2.0 * sht_noise

    # Cortisol: relevance-urgency conflict
    conflict = abs(r - u_norm) * max(r, u_norm)
    delta_cor = 0.0
    if conflict > cfg.cor_conflict_gate:
        cor_noise = float(rng.gamma(cfg.gamma_alpha, cfg.gamma_theta))
        delta_cor = cfg.beta_cor * conflict * cor_noise

    return IRENeuroChemSignals(
        delta_ach=delta_ach,
        delta_da=delta_da,
        delta_ne=delta_ne,
        delta_5ht=delta_5ht,
        delta_cor=delta_cor,
    )


# =====================================================================
# Pure helper functions — Flag generation
# =====================================================================


def _classify_flag_severity(confidence: float) -> FlagSeverity:
    """Map confidence to flag severity (reward-compatible)."""
    if confidence >= 0.80:
        return FlagSeverity.RISK
    if confidence >= 0.65:
        return FlagSeverity.WARNING
    return FlagSeverity.INFO


def generate_flags(
    dims: RelevanceDimensionScores,
    r: float,
    u_norm: float,
    phase1_priority: float,
    phase2_priority: float,
    breach_count: int,
    low_novelty_streak: int,
    previous_cc: float,
    cfg: IREConfig,
) -> List[RelevanceFlag]:
    """Generate relevance flags for anomalous conditions."""
    flags: List[RelevanceFlag] = []

    # TOPIC_DISCONTINUITY: CC < 0.15 and previous CC > 0.50
    if dims.contextual_continuity < 0.15 and previous_cc > 0.50:
        conf = min(1.0, (previous_cc - dims.contextual_continuity))
        flags.append(RelevanceFlag(
            flag_type=RelevanceFlagType.TOPIC_DISCONTINUITY,
            confidence=conf,
            severity=_classify_flag_severity(conf),
            description="Abrupt topic discontinuity detected",
            source_dimension="contextual_continuity",
        ))

    # INTENT_SHIFT: TA < 0.20
    if dims.task_alignment < 0.20:
        conf = 1.0 - dims.task_alignment / 0.20  # [0, 1] as TA→0
        flags.append(RelevanceFlag(
            flag_type=RelevanceFlagType.INTENT_SHIFT,
            confidence=min(1.0, conf),
            severity=_classify_flag_severity(min(1.0, conf)),
            description="User intent has shifted significantly",
            source_dimension="task_alignment",
        ))

    # IDENTITY_CHALLENGE: IR ≥ 0.60
    if dims.identity_resonance >= cfg.ir_force_deep:
        flags.append(RelevanceFlag(
            flag_type=RelevanceFlagType.IDENTITY_CHALLENGE,
            confidence=dims.identity_resonance,
            severity=_classify_flag_severity(dims.identity_resonance),
            description="Identity-relevant content detected — forced deep processing",
            source_dimension="identity_resonance",
        ))

    # EMOTIONAL_OVERRIDE: ES ≥ 0.70
    if dims.emotional_salience >= 0.70:
        flags.append(RelevanceFlag(
            flag_type=RelevanceFlagType.EMOTIONAL_OVERRIDE,
            confidence=dims.emotional_salience,
            severity=_classify_flag_severity(dims.emotional_salience),
            description="High emotional content overriding standard priorities",
            source_dimension="emotional_salience",
        ))

    # RELEVANCE_URGENCY_CONFLICT: R < 0.3 AND U_norm > 0.7
    if r < 0.3 and u_norm > 0.7:
        conf = u_norm * (1.0 - r)
        flags.append(RelevanceFlag(
            flag_type=RelevanceFlagType.RELEVANCE_URGENCY_CONFLICT,
            confidence=min(1.0, conf),
            severity=_classify_flag_severity(min(1.0, conf)),
            description="High urgency with low content relevance — Q3 redirect case",
            source_dimension="urgency",
        ))

    # NOVELTY_SATURATION: NV < threshold for N+ cycles
    if low_novelty_streak >= cfg.novelty_saturation_cycles:
        flags.append(RelevanceFlag(
            flag_type=RelevanceFlagType.NOVELTY_SATURATION,
            confidence=0.80,
            severity=FlagSeverity.WARNING,
            description=f"Novelty below {cfg.novelty_saturation_threshold} for "
                        f"{low_novelty_streak} consecutive cycles — user may be stuck",
            source_dimension="novelty",
        ))

    # PHASE_DIVERGENCE: |delta_priority| > 0.30
    delta_p = abs(phase2_priority - phase1_priority)
    if delta_p > 0.30:
        flags.append(RelevanceFlag(
            flag_type=RelevanceFlagType.PHASE_DIVERGENCE,
            confidence=min(1.0, delta_p),
            severity=_classify_flag_severity(min(1.0, delta_p)),
            description=f"Phase 2 revised priority by {delta_p:.2f} — unusual divergence",
            source_dimension="fusion",
        ))

    return flags


# =====================================================================
# Engine class
# =====================================================================


class InputRelevanceEvaluationEngine:
    """
    Engine 11 -- Input Relevance Evaluation Engine.

    Two-phase triage module: evaluates relevance (5 dimensions) and reads
    urgency (from Extractor 5), fuses into composite priority, and outputs
    a processing depth recommendation consumed by the pipeline orchestrator.

    API
    ---
    configure(mode)                -- set operational mode
    update_neurochem_state(state)  -- inject external NT levels
    process_phase1(input)          -- early triage (steps a-b)
    process_phase2(input)          -- post-contrast refinement (after step d)
    process(phase1_input, phase2_input)  -- convenience: both phases sequentially
    get_status()                   -- introspection
    """

    engine_id = "input_relevance_evaluation_engine"
    cluster   = "pattern_analysis"

    def __init__(
        self,
        config: Optional[IREConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._cfg = config or IREConfig()
        self._rng = rng or np.random.default_rng()
        self._mode = OperationalMode.NORMAL
        self._state = IREState()
        self._cycle_count = 0

    # ----- Configuration --------------------------------------------------

    def configure(self, mode: OperationalMode) -> None:
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """Inject current neurochemical levels for bidirectional feedback."""
        if "ach" in state_dict:
            self._state.ach_level = state_dict["ach"]
        if "da" in state_dict:
            self._state.da_level = state_dict["da"]
        if "ne" in state_dict:
            self._state.ne_level = state_dict["ne"]
        if "5ht" in state_dict:
            self._state._5ht_level = state_dict["5ht"]
        if "cor" in state_dict:
            self._state.cor_level = state_dict["cor"]

    # ----- Bidirectional NT feedback (applied to thresholds) ---------------

    def _apply_nt_feedback(
        self,
        thresholds: Tuple[float, float, float],
        weights: Dict[str, float],
    ) -> Tuple[Tuple[float, float, float], Dict[str, float]]:
        """
        Modulate thresholds and weights based on current NT levels.
        Returns (modified_thresholds, modified_weights).
        """
        t0, t1, t2 = thresholds
        w = dict(weights)  # copy

        # High NE → lower thresholds by 10% (more vigilant)
        if self._state.ne_level > 0.6:
            factor = 0.90
            t0 *= factor
            t1 *= factor
            t2 *= factor

        # High cortisol → lower all thresholds by 15%
        if self._state.cor_level > 0.5:
            factor = 0.85
            t0 *= factor
            t1 *= factor
            t2 *= factor

        # Low DA → increase novelty weight by 20% (seek stimulation)
        if 0.0 < self._state.da_level < 0.25:
            w["w_nv"] = w.get("w_nv", 0.20) * 1.20
            # Renormalize
            total = sum(w.values())
            if total > 0:
                w = {k: v / total for k, v in w.items()}

        # High 5-HT → raise thresholds (content, less reactive)
        if self._state._5ht_level > 0.6:
            factor = 1.10
            t0 *= factor
            t1 *= factor
            t2 *= factor

        return (t0, t1, t2), w

    # ----- Phase 1 --------------------------------------------------------

    def process_phase1(self, inp: IREPhase1Input) -> IREPhase1Result:
        """
        Phase 1 — Early Triage (steps a-b).

        Uses: tokenized input, STMM buffer, previous intent, urgency read.
        Produces: preliminary relevance, processing depth recommendation.
        """
        t0 = time.perf_counter()
        self._cycle_count += 1

        mode = inp.active_mode
        base_weights = resolve_mode_weights(mode)
        base_thresholds = resolve_depth_thresholds(mode)
        thresholds, weights = self._apply_nt_feedback(base_thresholds, base_weights)

        # Assemble STMM window
        stmm_window = list(inp.stmm_user_messages) + list(inp.stmm_system_responses)

        # --- Dimension 1: Contextual Continuity ---
        cc = compute_contextual_continuity_phase1(inp.current_text, stmm_window)

        # --- Dimension 2: Task Alignment ---
        ta = compute_task_alignment_phase1(inp.current_text, inp.previous_intent_vector)

        # --- Dimension 3: Novelty (preliminary) ---
        nv = compute_novelty_phase1(inp.current_text, stmm_window)

        # --- Dimension 4: Emotional Salience ---
        es = compute_emotional_salience_phase1(inp.current_text, inp.tokens)

        # --- Dimension 5: Identity Resonance ---
        ir = compute_identity_resonance_phase1(inp.current_text)

        dims = RelevanceDimensionScores(
            contextual_continuity=cc,
            task_alignment=ta,
            novelty=nv,
            emotional_salience=es,
            identity_resonance=ir,
        )

        # --- Urgency read ---
        u_norm = normalize_urgency(inp.urgency_risk, self._cfg.gamma_u)
        b_count = compute_breach_count(inp.urgency_breach_flags)

        # --- Fusion ---
        r = compute_relevance_composite(dims, weights)
        p = compute_priority_composite(r, u_norm, es, ir, self._cfg)

        # --- Quadrant ---
        quadrant = classify_quadrant(r, u_norm)

        # --- Processing depth ---
        depth = classify_processing_depth(p, thresholds)
        depth = apply_depth_overrides(depth, ir, es, b_count, u_norm, self._cfg)

        # --- Confidence ---
        confidence = compute_confidence(self._cfg.sigma_phase1, self._cfg.sigma_max)

        # --- Neurochemical signals ---
        # High ACh → reduce emission by 50%
        neurochem = compute_neurochem_signals(r, u_norm, p, dims, self._cfg, self._rng)
        if self._state.ach_level > 0.6:
            neurochem = IRENeuroChemSignals(
                delta_ach=neurochem.delta_ach * 0.5,
                delta_da=neurochem.delta_da,
                delta_ne=neurochem.delta_ne,
                delta_5ht=neurochem.delta_5ht,
                delta_cor=neurochem.delta_cor,
            )

        # --- Track state ---
        self._state.previous_cc = cc

        elapsed = (time.perf_counter() - t0) * 1000.0

        return IREPhase1Result(
            dimensions=dims,
            relevance_composite=r,
            urgency_normalized=u_norm,
            priority_composite=p,
            processing_depth=depth,
            confidence=confidence,
            phase=1,
            quadrant=quadrant,
            neurochemical_signals=neurochem,
            processing_time_ms=elapsed,
            metadata={
                "mode": mode.value,
                "weights_used": weights,
                "thresholds_used": list(thresholds),
                "breach_count": b_count,
                "cycle": self._cycle_count,
            },
        )

    # ----- Phase 2 --------------------------------------------------------

    def process_phase2(self, inp: IREPhase2Input) -> IREResult:
        """
        Phase 2 — Post-Contrast Refinement (after step d).

        Uses: Phase 1 result + Memory Contrast + current intent + emotion.
        Produces: authoritative relevance evaluation + flags.
        """
        t0 = time.perf_counter()

        p1 = inp.phase1_result
        if p1 is None:
            # If Phase 1 wasn't run, return a default result
            return IREResult()

        mode_str = p1.metadata.get("mode", OperationalMode.NORMAL.value)
        # Recover mode from metadata
        try:
            mode = OperationalMode(mode_str)
        except ValueError:
            mode = OperationalMode.NORMAL

        base_weights = resolve_mode_weights(mode)
        base_thresholds = resolve_depth_thresholds(mode)
        thresholds, weights = self._apply_nt_feedback(base_thresholds, base_weights)

        # --- Refine Dimension 1: Contextual Continuity ---
        cc = refine_contextual_continuity_phase2(
            p1.dimensions.contextual_continuity,
            inp.memory_contrast_scores,
            self._cfg,
        )

        # --- Refine Dimension 2: Task Alignment ---
        previous_intent = None
        if p1.metadata.get("_previous_intent_vector"):
            previous_intent = p1.metadata["_previous_intent_vector"]
        ta = refine_task_alignment_phase2(
            p1.dimensions.task_alignment,
            inp.current_intent_vector,
            previous_intent,
        )

        # --- Refine Dimension 3: Novelty (authoritative) ---
        if inp.memory_contrast_scores:
            nv = compute_novelty_phase2(
                inp.memory_contrast_scores,
                inp.detected_echoes,
                self._cfg,
            )
        else:
            nv = p1.dimensions.novelty

        # --- Refine Dimension 4: Emotional Salience ---
        if inp.emotional_intensity is not None:
            es = inp.emotional_intensity
        else:
            es = p1.dimensions.emotional_salience

        # --- Refine Dimension 5: Identity Resonance ---
        ir = refine_identity_resonance_phase2(
            p1.dimensions.identity_resonance,
            inp.identity_match_scores,
        )

        dims = RelevanceDimensionScores(
            contextual_continuity=cc,
            task_alignment=ta,
            novelty=nv,
            emotional_salience=es,
            identity_resonance=ir,
        )

        # --- Urgency (unchanged from Phase 1 — same cycle) ---
        u_norm = p1.urgency_normalized
        b_count = p1.metadata.get("breach_count", 0)

        # --- Fusion ---
        r = compute_relevance_composite(dims, weights)
        p = compute_priority_composite(r, u_norm, es, ir, self._cfg)

        # --- Quadrant ---
        quadrant = classify_quadrant(r, u_norm)

        # --- Processing depth ---
        depth = classify_processing_depth(p, thresholds)
        depth = apply_depth_overrides(depth, ir, es, b_count, u_norm, self._cfg)
        depth_changed = (depth != p1.processing_depth)

        # --- Confidence (Phase 2 is more confident) ---
        confidence = compute_confidence(self._cfg.sigma_phase2, self._cfg.sigma_max)

        # --- Neurochemical signals (Phase 2 delta only) ---
        neurochem = compute_neurochem_signals(r, u_norm, p, dims, self._cfg, self._rng)
        if self._state.ach_level > 0.6:
            neurochem = IRENeuroChemSignals(
                delta_ach=neurochem.delta_ach * 0.5,
                delta_da=neurochem.delta_da,
                delta_ne=neurochem.delta_ne,
                delta_5ht=neurochem.delta_5ht,
                delta_cor=neurochem.delta_cor,
            )

        # --- Novelty saturation tracking ---
        if nv < self._cfg.novelty_saturation_threshold:
            self._state.low_novelty_streak += 1
        else:
            self._state.low_novelty_streak = 0

        # --- Track state ---
        self._state.previous_cc = cc

        # --- Flag generation ---
        flags = generate_flags(
            dims=dims,
            r=r,
            u_norm=u_norm,
            phase1_priority=p1.priority_composite,
            phase2_priority=p,
            breach_count=b_count,
            low_novelty_streak=self._state.low_novelty_streak,
            previous_cc=p1.dimensions.contextual_continuity,
            cfg=self._cfg,
        )

        # --- Urgency passthrough ---
        urgency_breach_flags_raw = p1.metadata.get("_urgency_breach_flags", {})

        elapsed = (time.perf_counter() - t0) * 1000.0

        return IREResult(
            dimensions=dims,
            relevance_composite=r,
            urgency_normalized=u_norm,
            priority_composite=p,
            processing_depth=depth,
            depth_changed_from_phase1=depth_changed,
            confidence=confidence,
            phase=2,
            quadrant=quadrant,
            urgency_risk_raw=p1.metadata.get("_urgency_risk_raw", 0.0),
            urgency_breach_flags=urgency_breach_flags_raw,
            urgency_breach_count=b_count,
            phase1_priority=p1.priority_composite,
            delta_priority=p - p1.priority_composite,
            neurochemical_signals=neurochem,
            flags=flags,
            processing_time_ms=elapsed,
            metadata={
                "mode": mode.value,
                "weights_used": weights,
                "thresholds_used": list(thresholds),
                "breach_count": b_count,
                "cycle": self._cycle_count,
                "low_novelty_streak": self._state.low_novelty_streak,
            },
        )

    # ----- Convenience: both phases sequentially ---------------------------

    def process(
        self,
        phase1_input: IREPhase1Input,
        phase2_input_extra: Optional[IREPhase2Input] = None,
    ) -> IREResult:
        """
        Run both phases sequentially. If only Phase 1 input is provided,
        returns Phase 1 result wrapped in an IREResult.
        """
        p1 = self.process_phase1(phase1_input)

        if phase2_input_extra is None:
            # Wrap Phase 1 result as IREResult
            return IREResult(
                dimensions=p1.dimensions,
                relevance_composite=p1.relevance_composite,
                urgency_normalized=p1.urgency_normalized,
                priority_composite=p1.priority_composite,
                processing_depth=p1.processing_depth,
                depth_changed_from_phase1=False,
                confidence=p1.confidence,
                phase=1,
                quadrant=p1.quadrant,
                urgency_risk_raw=phase1_input.urgency_risk,
                urgency_breach_flags=phase1_input.urgency_breach_flags,
                urgency_breach_count=compute_breach_count(phase1_input.urgency_breach_flags),
                phase1_priority=p1.priority_composite,
                delta_priority=0.0,
                neurochemical_signals=p1.neurochemical_signals,
                flags=[],
                processing_time_ms=p1.processing_time_ms,
                metadata=p1.metadata,
            )

        # Enrich Phase 2 input with Phase 1 result
        p2_inp = IREPhase2Input(
            phase1_result=IREPhase1Result(
                dimensions=p1.dimensions,
                relevance_composite=p1.relevance_composite,
                urgency_normalized=p1.urgency_normalized,
                priority_composite=p1.priority_composite,
                processing_depth=p1.processing_depth,
                confidence=p1.confidence,
                phase=1,
                quadrant=p1.quadrant,
                neurochemical_signals=p1.neurochemical_signals,
                processing_time_ms=p1.processing_time_ms,
                metadata={
                    **p1.metadata,
                    "_previous_intent_vector": list(phase1_input.previous_intent_vector)
                    if phase1_input.previous_intent_vector else None,
                    "_urgency_risk_raw": phase1_input.urgency_risk,
                    "_urgency_breach_flags": dict(phase1_input.urgency_breach_flags),
                },
            ),
            memory_contrast_scores=phase2_input_extra.memory_contrast_scores,
            detected_echoes=phase2_input_extra.detected_echoes,
            identity_match_scores=phase2_input_extra.identity_match_scores,
            current_intent_vector=phase2_input_extra.current_intent_vector,
            emotional_intensity=phase2_input_extra.emotional_intensity,
        )

        return self.process_phase2(p2_inp)

    # ----- Introspection --------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "cluster": self.cluster,
            "mode": self._mode.value,
            "cycle_count": self._cycle_count,
            "state": {
                "ach_level": self._state.ach_level,
                "da_level": self._state.da_level,
                "ne_level": self._state.ne_level,
                "5ht_level": self._state._5ht_level,
                "cor_level": self._state.cor_level,
                "low_novelty_streak": self._state.low_novelty_streak,
                "previous_cc": self._state.previous_cc,
            },
        }
