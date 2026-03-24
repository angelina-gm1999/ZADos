"""
Engine 22 -- Contextual Learning Engine  (``contextual_learning_engine``)
=========================================================================
Learns and recognises conversational contexts, then applies
context-specific parameter adjustments to downstream processing.

The engine maintains an internal library of *context records* -- each a
fingerprint of (topic, emotion_state, intent) that uniquely identifies a
conversational context.  When a new input arrives the engine:

  1. **Fingerprints** the input into a composite hash of its constituent
     signals (topic, emotion, intent).
  2. **Matches** the fingerprint against all stored records using weighted
     cosine similarity.
  3. **Looks up** parameter adjustments associated with the best match.
  4. **Encodes** a new record when no sufficiently similar context exists.
  5. **Strengthens** existing records on re-encounter, increasing their
     confidence and refining their stored parameter adjustments.

NT Coupling
-----------
- ACh strengthens encoding (faster context learning).
- OXT increases social-context sensitivity.
- CB1 lowers recognition threshold (more flexible matching).
- DA rewards novel context discovery.
- 5-HT stabilises existing context records.
- NE broadens the set of features considered during fingerprinting.

Oscillatory
-----------
- theta_boost emitted during encoding (episodic binding).
- gamma_boost emitted during recognition (associative recall).

Mode Support
------------
DEFAULT      — balanced recognition / encoding
ANALYTICAL   — higher recognition threshold, slower encoding
CREATIVE     — lower threshold, faster encoding, broader matching
REM_DREAM    — very loose matching, aggressive encoding, dream consolidation

Usage
-----
>>> from zados.cognitive_engines.py_engines.contextual_learning_engine import (
...     ContextualLearningEngine, ContextLearningConfig,
...     ContextInput,
... )
>>> engine = ContextualLearningEngine()
>>> inp = ContextInput(
...     topic="quantum computing",
...     emotion_state={"curiosity": 0.8, "excitement": 0.5},
...     intent="learn",
...     raw_text="Tell me about quantum entanglement.",
... )
>>> result = engine.process(inp)
>>> result.novel_context
True
"""
from __future__ import annotations

import hashlib
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


class ContextStatus(str, Enum):
    """Lifecycle status of a stored context record."""
    ACTIVE    = "active"       # Regularly encountered
    DORMANT   = "dormant"      # Not seen for a while, still retained
    DECAYED   = "decayed"      # Below retention threshold, candidate for pruning
    ARCHIVED  = "archived"     # Explicitly preserved (identity-relevant)


class MatchQuality(str, Enum):
    """Quality tier of a context match."""
    EXACT       = "exact"        # similarity >= 0.95
    STRONG      = "strong"       # similarity >= 0.80
    MODERATE    = "moderate"     # similarity >= recognition_threshold
    WEAK        = "weak"         # below threshold but above floor
    NONE        = "none"         # no meaningful match


class EncodingStrength(str, Enum):
    """How strongly a new context is encoded."""
    STRONG   = "strong"     # High ACh + novel + emotional
    MODERATE = "moderate"   # Default encoding
    WEAK     = "weak"       # Low attention / familiar-ish


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class SimilarityWeights:
    """Weights for the three fingerprint components in similarity calculation."""
    topic_weight:   float = 0.45
    emotion_weight: float = 0.30
    intent_weight:  float = 0.25


@dataclass(frozen=True)
class ModeConfig:
    """Per-mode parameter overrides."""
    recognition_threshold: float = 0.60
    encoding_strength:     float = 0.50
    decay_rate:            float = 0.01
    broadening_factor:     float = 1.0   # multiplier on feature set breadth


@dataclass(frozen=True)
class ContextLearningConfig:
    """Immutable configuration for the Contextual Learning Engine."""

    # --- Core parameters ---
    recognition_threshold:  float = 0.60   # similarity >= this → recognised
    max_contexts:           int   = 512    # maximum stored context records
    encoding_strength:      float = 0.50   # base encoding confidence
    decay_rate:             float = 0.01   # per-tick confidence decay
    decay_half_life_ticks:  int   = 500    # ticks until confidence halves
    dormancy_threshold:     float = 0.15   # below this → DORMANT
    prune_threshold:        float = 0.05   # below this → DECAYED (prune candidate)

    # --- Similarity weights ---
    similarity_weights: SimilarityWeights = field(default_factory=SimilarityWeights)

    # --- Strengthening ---
    strengthening_rate:     float = 0.10   # confidence boost per re-encounter
    max_confidence:         float = 1.0
    adjustment_blend_rate:  float = 0.20   # EMA rate for blending new adjustments

    # --- Mode configs ---
    mode_default: ModeConfig = field(default_factory=lambda: ModeConfig(
        recognition_threshold=0.60, encoding_strength=0.50,
        decay_rate=0.01, broadening_factor=1.0,
    ))
    mode_analytical: ModeConfig = field(default_factory=lambda: ModeConfig(
        recognition_threshold=0.75, encoding_strength=0.35,
        decay_rate=0.005, broadening_factor=0.8,
    ))
    mode_creative: ModeConfig = field(default_factory=lambda: ModeConfig(
        recognition_threshold=0.45, encoding_strength=0.65,
        decay_rate=0.015, broadening_factor=1.4,
    ))
    mode_rem_dream: ModeConfig = field(default_factory=lambda: ModeConfig(
        recognition_threshold=0.35, encoding_strength=0.80,
        decay_rate=0.005, broadening_factor=2.0,
    ))

    # --- Neurochemical coupling ---
    beta_ach_encoding:      float = 0.15   # ACh → encoding strength boost
    beta_oxt_social:        float = 0.12   # OXT → social feature weight boost
    beta_cb1_flexibility:   float = 0.10   # CB1 → threshold lowering
    beta_da_novelty:        float = 0.12   # DA → novelty reward
    beta_5ht_stability:     float = 0.08   # 5-HT → record stability
    beta_ne_broadening:     float = 0.10   # NE → feature broadening

    # --- Oscillatory coupling ---
    psi_theta_encoding:     float = 0.08   # theta boost during encoding
    psi_gamma_recognition:  float = 0.06   # gamma boost during recognition

    # --- Stochastic distribution params ---
    gamma_alpha:  float = 2.0    # Gamma shape for DA/ACh noise
    gamma_theta:  float = 0.30   # Gamma scale
    poisson_lam:  float = 1.5    # Poisson lambda for NE


# =====================================================================
# Data types -- frozen outputs
# =====================================================================


@dataclass(frozen=True)
class ContextFingerprint:
    """
    Hash-based fingerprint of a conversational context.

    Each component is hashed independently, then combined into a composite.
    The raw feature vectors are retained for similarity computation.
    """
    context_id:      str              = ""
    topic_hash:      str              = ""
    emotion_hash:    str              = ""
    intent_hash:     str              = ""
    composite_hash:  str              = ""
    topic_vector:    Tuple[float, ...] = ()
    emotion_vector:  Tuple[float, ...] = ()
    intent_vector:   Tuple[float, ...] = ()


@dataclass(frozen=True)
class ContextMatchResult:
    """Result of matching current input against a stored context record."""
    context_id:            str                = ""
    similarity:            float              = 0.0
    match_quality:         MatchQuality       = MatchQuality.NONE
    is_novel:              bool               = True
    parameter_adjustments: Dict[str, float]   = field(default_factory=dict)
    confidence:            float              = 0.0
    encounter_count:       int                = 0
    topic_similarity:      float              = 0.0
    emotion_similarity:    float              = 0.0
    intent_similarity:     float              = 0.0


@dataclass(frozen=True)
class ContextLearningNeurochem:
    """
    Neurochemical coupling signals from one Contextual Learning cycle.

    Notation (Appendix S2-S3, S7):
        da_delta      -> Delta C_DA(t)     : novelty reward for new context
        ach_delta     -> Delta C_ACh(t)    : attentional encoding gate
        oxt_delta     -> Delta C_OXT(t)    : social context sensitivity
        _5ht_delta    -> Delta C_5HT(t)    : record stabilisation signal
        theta_boost   -> Delta phi_theta(t): episodic encoding oscillatory
        gamma_boost   -> Delta phi_gamma(t): recognition binding oscillatory
    """
    da_delta:     float = 0.0
    ach_delta:    float = 0.0
    oxt_delta:    float = 0.0
    _5ht_delta:   float = 0.0
    theta_boost:  float = 0.0
    gamma_boost:  float = 0.0


@dataclass(frozen=True)
class ContextLearningResult:
    """Full output of one Contextual Learning Engine cycle."""
    current_fingerprint:   ContextFingerprint       = field(default_factory=ContextFingerprint)
    matches:               List[ContextMatchResult]  = field(default_factory=list)
    best_match:            Optional[ContextMatchResult] = None
    novel_context:         bool                      = True
    active_adjustments:    Dict[str, float]          = field(default_factory=dict)
    total_contexts:        int                       = 0
    contexts_active:       int                       = 0
    contexts_dormant:      int                       = 0
    contexts_pruned:       int                       = 0
    encoding_performed:    bool                      = False
    strengthening_applied: bool                      = False
    neurochemical_signals: ContextLearningNeurochem  = field(default_factory=ContextLearningNeurochem)
    processing_time_ms:    float                     = 0.0
    metadata:              Dict[str, Any]            = field(default_factory=dict)


# =====================================================================
# Context input
# =====================================================================


@dataclass
class ContextInput:
    """Input bundle for one Contextual Learning Engine cycle."""
    topic:                str                  = ""
    emotion_state:        Dict[str, float]     = field(default_factory=dict)
    intent:               str                  = ""
    raw_text:             str                  = ""
    parameter_adjustments: Dict[str, float]    = field(default_factory=dict)
    social_markers:       List[str]            = field(default_factory=list)
    active_mode:          OperationalMode      = OperationalMode.NORMAL


# =====================================================================
# Mutable context record (stored in engine library)
# =====================================================================


@dataclass
class ContextRecord:
    """Mutable context record in the engine's internal library."""
    context_id:            str              = field(default_factory=lambda: str(uuid.uuid4()))
    fingerprint:           Optional[ContextFingerprint] = None
    topic:                 str              = ""
    emotion_state:         Dict[str, float] = field(default_factory=dict)
    intent:                str              = ""
    confidence:            float            = 0.5
    encounter_count:       int              = 1
    parameter_adjustments: Dict[str, float] = field(default_factory=dict)
    last_seen_tick:        int              = 0
    created_tick:          int              = 0
    status:                ContextStatus    = ContextStatus.ACTIVE
    social_context:        bool             = False


# =====================================================================
# Mutable engine state
# =====================================================================


@dataclass
class ContextLearningState:
    """Running neurochemical state for bidirectional feedback."""
    ach_level:  float = 0.0
    ne_level:   float = 0.0
    da_level:   float = 0.0
    _5ht_level: float = 0.0
    oxt_level:  float = 0.0
    cb1_level:  float = 0.0


# =====================================================================
# Pure helper functions
# =====================================================================


def _hash_string(s: str) -> str:
    """Deterministic SHA-256 hex digest of a string (first 16 chars)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _hash_dict(d: Dict[str, float]) -> str:
    """Deterministic hash of a float dict (sorted keys, rounded values)."""
    canonical = "|".join(
        f"{k}={round(v, 4)}" for k, v in sorted(d.items())
    )
    return _hash_string(canonical)


def _text_to_vector(text: str, dim: int = 32) -> Tuple[float, ...]:
    """
    Lightweight text → dense vector via character n-gram hashing.

    Not a real embedding -- just a deterministic pseudo-vector sufficient
    for cosine similarity between text snippets.  No external ML deps.
    """
    vec = [0.0] * dim
    if not text:
        return tuple(vec)
    lower = text.lower().strip()
    # Character trigrams
    for i in range(max(1, len(lower) - 2)):
        trigram = lower[i:i + 3]
        h = int(hashlib.md5(trigram.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 128) % 2 == 0 else -1.0
        vec[idx] += sign
    # Normalise
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return tuple(v / norm for v in vec)


def _emotion_to_vector(emotion: Dict[str, float], dim: int = 16) -> Tuple[float, ...]:
    """
    Emotion state dict → dense vector via hashed emotion names.

    Each emotion name hashes to a slot; its value is added there.
    """
    vec = [0.0] * dim
    if not emotion:
        return tuple(vec)
    for name, value in emotion.items():
        h = int(hashlib.md5(name.lower().encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += value
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return tuple(v / norm for v in vec)


def build_fingerprint(
    topic: str,
    emotion_state: Dict[str, float],
    intent: str,
) -> ContextFingerprint:
    """Build a ContextFingerprint from raw input components."""
    topic_hash = _hash_string(topic.lower().strip()) if topic else ""
    emotion_hash = _hash_dict(emotion_state) if emotion_state else ""
    intent_hash = _hash_string(intent.lower().strip()) if intent else ""
    composite = _hash_string(f"{topic_hash}|{emotion_hash}|{intent_hash}")

    topic_vec = _text_to_vector(topic)
    emotion_vec = _emotion_to_vector(emotion_state)
    intent_vec = _text_to_vector(intent)

    return ContextFingerprint(
        context_id=str(uuid.uuid4()),
        topic_hash=topic_hash,
        emotion_hash=emotion_hash,
        intent_hash=intent_hash,
        composite_hash=composite,
        topic_vector=topic_vec,
        emotion_vector=emotion_vec,
        intent_vector=intent_vec,
    )


def cosine_similarity(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    """
    Cosine similarity between two tuples.  Returns [0, 1].

    Negative cosine values are clamped to 0 (anti-correlated vectors
    are treated as maximally dissimilar in this context).
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    sim = dot / (norm_a * norm_b)
    return _clamp(sim)


def weighted_similarity(
    fp_a: ContextFingerprint,
    fp_b: ContextFingerprint,
    weights: SimilarityWeights,
) -> Tuple[float, float, float, float]:
    """
    Weighted similarity between two fingerprints.

    Returns (composite, topic_sim, emotion_sim, intent_sim).
    """
    topic_sim = cosine_similarity(fp_a.topic_vector, fp_b.topic_vector)
    emotion_sim = cosine_similarity(fp_a.emotion_vector, fp_b.emotion_vector)
    intent_sim = cosine_similarity(fp_a.intent_vector, fp_b.intent_vector)

    composite = (
        weights.topic_weight * topic_sim
        + weights.emotion_weight * emotion_sim
        + weights.intent_weight * intent_sim
    )
    return (composite, topic_sim, emotion_sim, intent_sim)


def classify_match_quality(similarity: float, threshold: float) -> MatchQuality:
    """Map similarity score to a quality tier."""
    if similarity >= 0.95:
        return MatchQuality.EXACT
    if similarity >= 0.80:
        return MatchQuality.STRONG
    if similarity >= threshold:
        return MatchQuality.MODERATE
    if similarity >= threshold * 0.6:
        return MatchQuality.WEAK
    return MatchQuality.NONE


def resolve_mode_config(
    mode: OperationalMode,
    cfg: ContextLearningConfig,
) -> ModeConfig:
    """Return the ModeConfig for the given operational mode."""
    return {
        OperationalMode.NORMAL:     cfg.mode_default,
        OperationalMode.DEV:        cfg.mode_default,
        OperationalMode.LEARNING:   cfg.mode_default,
        OperationalMode.REFLECTIVE: cfg.mode_analytical,
        OperationalMode.REM_NORMAL: cfg.mode_default,
        OperationalMode.REM_DREAM:  cfg.mode_rem_dream,
    }.get(mode, cfg.mode_default)


def compute_decay(
    confidence: float,
    ticks_since_seen: int,
    decay_rate: float,
    half_life: int,
) -> float:
    """
    Exponential decay of context record confidence.

    decay(c, dt) = c * exp(-lambda * dt)
    where lambda = ln(2) / half_life
    """
    if half_life <= 0 or ticks_since_seen <= 0:
        return confidence
    lam = math.log(2.0) / half_life
    return confidence * math.exp(-lam * ticks_since_seen)


def strengthen_record(
    record: ContextRecord,
    new_adjustments: Dict[str, float],
    rate: float,
    blend_rate: float,
    max_conf: float,
    current_tick: int,
) -> None:
    """
    Strengthen a context record on re-encounter.

    - Increment encounter count.
    - Boost confidence (bounded by max_conf).
    - Blend new parameter adjustments via EMA.
    - Update last_seen_tick.
    """
    record.encounter_count += 1
    record.confidence = min(max_conf, record.confidence + rate * (1.0 - record.confidence))
    record.last_seen_tick = current_tick
    record.status = ContextStatus.ACTIVE

    # EMA blend of parameter adjustments
    for key, val in new_adjustments.items():
        old = record.parameter_adjustments.get(key, val)
        record.parameter_adjustments[key] = old + blend_rate * (val - old)


def encode_new_record(
    fingerprint: ContextFingerprint,
    topic: str,
    emotion_state: Dict[str, float],
    intent: str,
    parameter_adjustments: Dict[str, float],
    encoding_strength: float,
    current_tick: int,
    social_context: bool = False,
) -> ContextRecord:
    """Create a new ContextRecord from the current input."""
    return ContextRecord(
        context_id=fingerprint.context_id,
        fingerprint=fingerprint,
        topic=topic,
        emotion_state=dict(emotion_state),
        intent=intent,
        confidence=_clamp(encoding_strength),
        encounter_count=1,
        parameter_adjustments=dict(parameter_adjustments),
        last_seen_tick=current_tick,
        created_tick=current_tick,
        status=ContextStatus.ACTIVE,
        social_context=social_context,
    )


def apply_decay_to_library(
    library: Dict[str, ContextRecord],
    current_tick: int,
    decay_rate: float,
    half_life: int,
    dormancy_threshold: float,
    prune_threshold: float,
) -> int:
    """
    Apply time-based decay to all context records.

    Returns the number of records pruned.
    """
    pruned = 0
    to_remove: List[str] = []

    for cid, record in library.items():
        if record.status == ContextStatus.ARCHIVED:
            continue  # never decay archived records

        ticks = current_tick - record.last_seen_tick
        if ticks > 0:
            record.confidence = compute_decay(
                record.confidence, ticks, decay_rate, half_life,
            )

        # Status transitions
        if record.confidence < prune_threshold:
            record.status = ContextStatus.DECAYED
            to_remove.append(cid)
            pruned += 1
        elif record.confidence < dormancy_threshold:
            record.status = ContextStatus.DORMANT

    for cid in to_remove:
        del library[cid]

    return pruned


def evict_if_needed(
    library: Dict[str, ContextRecord],
    max_contexts: int,
) -> int:
    """
    Evict lowest-confidence records when library exceeds capacity.

    Returns the number of records evicted.
    """
    if len(library) <= max_contexts:
        return 0

    n_evict = len(library) - max_contexts
    # Sort by (status != ARCHIVED, confidence ascending)
    candidates = sorted(
        library.items(),
        key=lambda kv: (
            0 if kv[1].status != ContextStatus.ARCHIVED else 1,
            kv[1].confidence,
        ),
    )
    evicted = 0
    for cid, record in candidates:
        if record.status == ContextStatus.ARCHIVED:
            continue
        del library[cid]
        evicted += 1
        if evicted >= n_evict:
            break
    return evicted


def compute_neurochem_signals(
    novel_context: bool,
    best_similarity: float,
    encoding_performed: bool,
    strengthening_applied: bool,
    social_context: bool,
    n_matches: int,
    cfg: ContextLearningConfig,
    rng: np.random.Generator,
) -> ContextLearningNeurochem:
    """
    Compute neurochemical coupling signals for one cycle.

    DA  -- novelty reward when a genuinely new context is encoded.
    ACh -- attentional gating during encoding.
    OXT -- social-context sensitivity modulation.
    5-HT -- stabilisation signal when an existing context is strengthened.
    theta_boost -- episodic binding during encoding.
    gamma_boost -- associative recall during recognition.
    """
    # DA: novelty reward
    da_delta = 0.0
    if novel_context and encoding_performed:
        da_noise = float(rng.gamma(cfg.gamma_alpha, cfg.gamma_theta))
        da_delta = cfg.beta_da_novelty * da_noise

    # ACh: attentional encoding gate
    ach_delta = 0.0
    if encoding_performed:
        ach_noise = float(rng.gamma(cfg.gamma_alpha, cfg.gamma_theta))
        ach_delta = cfg.beta_ach_encoding * ach_noise

    # OXT: social-context sensitivity
    oxt_delta = 0.0
    if social_context:
        oxt_delta = cfg.beta_oxt_social * (1.0 if encoding_performed else 0.5)

    # 5-HT: stabilisation on strengthening
    _5ht_delta = 0.0
    if strengthening_applied:
        _5ht_delta = cfg.beta_5ht_stability * best_similarity

    # theta: episodic binding during encoding
    theta_boost = 0.0
    if encoding_performed:
        theta_boost = cfg.psi_theta_encoding

    # gamma: associative recall during recognition
    gamma_boost = 0.0
    if not novel_context and n_matches > 0:
        gamma_boost = cfg.psi_gamma_recognition * min(1.0, n_matches / 3.0)

    return ContextLearningNeurochem(
        da_delta=da_delta,
        ach_delta=ach_delta,
        oxt_delta=oxt_delta,
        _5ht_delta=_5ht_delta,
        theta_boost=theta_boost,
        gamma_boost=gamma_boost,
    )


# =====================================================================
# Engine class
# =====================================================================


class ContextualLearningEngine:
    """
    Engine 22 -- Contextual Learning Engine.

    Learns and recognises conversational contexts, then applies
    context-specific parameter adjustments to downstream processing.

    The engine maintains an internal *context library* of fingerprinted
    context records.  On each cycle it fingerprints the input, searches
    the library for matches, and either strengthens an existing record
    or encodes a new one.

    API
    ---
    configure(mode)               -- set operational mode
    update_neurochem_state(d)     -- inject external NT levels (Pattern A)
    process(context_input)        -- run learn/recognise cycle
    get_status()                  -- introspection
    get_context_count()           -- number of stored contexts
    get_context_library_summary() -- summary of library contents
    archive_context(context_id)   -- protect a context from decay/eviction
    """

    engine_id = "contextual_learning_engine"
    cluster   = "learning"

    def __init__(
        self,
        config: Optional[ContextLearningConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._cfg = config or ContextLearningConfig()
        self._rng = rng or np.random.default_rng()
        self._mode = OperationalMode.NORMAL
        self._state = ContextLearningState()
        self._cycle_count = 0
        self._tick = 0
        self._library: Dict[str, ContextRecord] = {}

    # ----- Configuration --------------------------------------------------

    def configure(self, mode: OperationalMode) -> None:
        """Set the operational mode."""
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """
        Inject current neurochemical levels for bidirectional feedback.

        Accepts canonical NT keys: "ach", "ne", "da", "5ht", "oxt", "cb1".
        """
        if "ach" in state_dict:
            self._state.ach_level = state_dict["ach"]
        if "ne" in state_dict:
            self._state.ne_level = state_dict["ne"]
        if "da" in state_dict:
            self._state.da_level = state_dict["da"]
        if "5ht" in state_dict:
            self._state._5ht_level = state_dict["5ht"]
        if "oxt" in state_dict:
            self._state.oxt_level = state_dict["oxt"]
        if "cb1" in state_dict:
            self._state.cb1_level = state_dict["cb1"]

    # ----- Main pipeline --------------------------------------------------

    def process(self, context_input: ContextInput) -> ContextLearningResult:
        """
        Run the full contextual-learning cycle on *context_input*.

        Pipeline stages:
          1. Build fingerprint from (topic, emotion_state, intent).
          2. Apply time-based decay to library.
          3. Match fingerprint against all stored records.
          4. If recognised → strengthen existing record.
          5. If novel → encode new context record.
          6. Compute neurochemical coupling signals.
        """
        t0 = time.perf_counter()
        self._cycle_count += 1
        self._tick += 1

        mode = context_input.active_mode
        mode_cfg = resolve_mode_config(mode, self._cfg)

        # --- Bidirectional NT modulation ---
        # CB1 lowers recognition threshold (more flexible)
        recognition_threshold = mode_cfg.recognition_threshold
        if self._state.cb1_level > 0.0:
            recognition_threshold = max(
                0.20,
                recognition_threshold - self._cfg.beta_cb1_flexibility * self._state.cb1_level,
            )

        # ACh boosts encoding strength
        encoding_strength = mode_cfg.encoding_strength
        if self._state.ach_level > 0.0:
            encoding_strength = min(
                1.0,
                encoding_strength + self._cfg.beta_ach_encoding * self._state.ach_level,
            )

        # NE broadens feature consideration (adjusts similarity weights)
        sim_weights = self._cfg.similarity_weights
        ne_broadening = self._state.ne_level * self._cfg.beta_ne_broadening
        if ne_broadening > 0.0:
            # Flatten the weights toward uniform when NE is high
            avg_w = (sim_weights.topic_weight + sim_weights.emotion_weight + sim_weights.intent_weight) / 3.0
            blend = min(1.0, ne_broadening)
            sim_weights = SimilarityWeights(
                topic_weight=sim_weights.topic_weight + blend * (avg_w - sim_weights.topic_weight),
                emotion_weight=sim_weights.emotion_weight + blend * (avg_w - sim_weights.emotion_weight),
                intent_weight=sim_weights.intent_weight + blend * (avg_w - sim_weights.intent_weight),
            )

        # OXT boosts social feature sensitivity
        if self._state.oxt_level > 0.0:
            oxt_boost = self._cfg.beta_oxt_social * self._state.oxt_level
            sim_weights = SimilarityWeights(
                topic_weight=sim_weights.topic_weight * (1.0 - oxt_boost * 0.3),
                emotion_weight=sim_weights.emotion_weight + oxt_boost * 0.15,
                intent_weight=sim_weights.intent_weight + oxt_boost * 0.15,
            )

        # 5-HT stabilises existing records (reduces decay)
        effective_decay = mode_cfg.decay_rate
        if self._state._5ht_level > 0.0:
            effective_decay *= max(0.2, 1.0 - self._cfg.beta_5ht_stability * self._state._5ht_level)

        # --- Stage 1: Fingerprint ---
        fingerprint = build_fingerprint(
            context_input.topic,
            context_input.emotion_state,
            context_input.intent,
        )

        # --- Stage 2: Decay ---
        pruned = apply_decay_to_library(
            self._library,
            self._tick,
            effective_decay,
            self._cfg.decay_half_life_ticks,
            self._cfg.dormancy_threshold,
            self._cfg.prune_threshold,
        )

        # --- Stage 3: Match ---
        matches: List[ContextMatchResult] = []
        for cid, record in self._library.items():
            if record.fingerprint is None:
                continue
            composite, topic_sim, emotion_sim, intent_sim = weighted_similarity(
                fingerprint, record.fingerprint, sim_weights,
            )
            quality = classify_match_quality(composite, recognition_threshold)
            if quality != MatchQuality.NONE:
                matches.append(ContextMatchResult(
                    context_id=record.context_id,
                    similarity=round(composite, 4),
                    match_quality=quality,
                    is_novel=False,
                    parameter_adjustments=dict(record.parameter_adjustments),
                    confidence=round(record.confidence, 4),
                    encounter_count=record.encounter_count,
                    topic_similarity=round(topic_sim, 4),
                    emotion_similarity=round(emotion_sim, 4),
                    intent_similarity=round(intent_sim, 4),
                ))

        # Sort by similarity descending
        matches.sort(key=lambda m: m.similarity, reverse=True)

        # --- Stage 4/5: Recognise or Encode ---
        novel_context = True
        encoding_performed = False
        strengthening_applied = False
        best_match: Optional[ContextMatchResult] = None
        active_adjustments: Dict[str, float] = {}

        # Check for above-threshold match
        recognised_matches = [
            m for m in matches
            if m.similarity >= recognition_threshold
        ]

        if recognised_matches:
            # Recognised -- strengthen best match
            best_match = recognised_matches[0]
            novel_context = False

            record = self._library.get(best_match.context_id)
            if record is not None:
                strengthen_record(
                    record,
                    context_input.parameter_adjustments,
                    self._cfg.strengthening_rate,
                    self._cfg.adjustment_blend_rate,
                    self._cfg.max_confidence,
                    self._tick,
                )
                strengthening_applied = True
                active_adjustments = dict(record.parameter_adjustments)
        else:
            # Novel -- encode new record
            has_social = bool(context_input.social_markers)
            new_record = encode_new_record(
                fingerprint=fingerprint,
                topic=context_input.topic,
                emotion_state=context_input.emotion_state,
                intent=context_input.intent,
                parameter_adjustments=context_input.parameter_adjustments,
                encoding_strength=encoding_strength,
                current_tick=self._tick,
                social_context=has_social,
            )
            self._library[new_record.context_id] = new_record
            encoding_performed = True
            active_adjustments = dict(context_input.parameter_adjustments)

            # Build a novel match result for the output
            best_match = ContextMatchResult(
                context_id=new_record.context_id,
                similarity=0.0,
                match_quality=MatchQuality.NONE,
                is_novel=True,
                parameter_adjustments=dict(context_input.parameter_adjustments),
                confidence=round(new_record.confidence, 4),
                encounter_count=1,
            )

        # Evict if over capacity
        evicted = evict_if_needed(self._library, self._cfg.max_contexts)

        # --- Stage 6: Neurochem ---
        best_similarity = best_match.similarity if best_match and not novel_context else 0.0
        has_social = bool(context_input.social_markers)

        neurochem = compute_neurochem_signals(
            novel_context=novel_context,
            best_similarity=best_similarity,
            encoding_performed=encoding_performed,
            strengthening_applied=strengthening_applied,
            social_context=has_social,
            n_matches=len(recognised_matches) if not novel_context else 0,
            cfg=self._cfg,
            rng=self._rng,
        )

        # Library stats
        n_active = sum(1 for r in self._library.values() if r.status == ContextStatus.ACTIVE)
        n_dormant = sum(1 for r in self._library.values() if r.status == ContextStatus.DORMANT)

        elapsed = (time.perf_counter() - t0) * 1000.0

        return ContextLearningResult(
            current_fingerprint=fingerprint,
            matches=matches,
            best_match=best_match,
            novel_context=novel_context,
            active_adjustments=active_adjustments,
            total_contexts=len(self._library),
            contexts_active=n_active,
            contexts_dormant=n_dormant,
            contexts_pruned=pruned + evicted,
            encoding_performed=encoding_performed,
            strengthening_applied=strengthening_applied,
            neurochemical_signals=neurochem,
            processing_time_ms=round(elapsed, 3),
            metadata={
                "mode": mode.value,
                "recognition_threshold": round(recognition_threshold, 4),
                "encoding_strength": round(encoding_strength, 4),
                "effective_decay": round(effective_decay, 6),
                "cycle": self._cycle_count,
                "tick": self._tick,
            },
        )

    # ----- Library management ---------------------------------------------

    def get_context_count(self) -> int:
        """Return the number of stored context records."""
        return len(self._library)

    def get_context_library_summary(self) -> Dict[str, Any]:
        """Return a summary of the context library contents."""
        statuses: Dict[str, int] = {}
        total_encounters = 0
        total_confidence = 0.0
        social_count = 0

        for record in self._library.values():
            key = record.status.value
            statuses[key] = statuses.get(key, 0) + 1
            total_encounters += record.encounter_count
            total_confidence += record.confidence
            if record.social_context:
                social_count += 1

        n = len(self._library) or 1
        return {
            "total_contexts": len(self._library),
            "status_distribution": statuses,
            "avg_confidence": round(total_confidence / n, 4),
            "avg_encounters": round(total_encounters / n, 2),
            "social_contexts": social_count,
            "capacity_used": round(len(self._library) / max(1, self._cfg.max_contexts), 4),
        }

    def archive_context(self, context_id: str) -> bool:
        """
        Protect a context record from decay and eviction.

        Returns True if the context was found and archived.
        """
        record = self._library.get(context_id)
        if record is None:
            return False
        record.status = ContextStatus.ARCHIVED
        return True

    def get_record(self, context_id: str) -> Optional[ContextRecord]:
        """Retrieve a context record by ID.  Returns None if not found."""
        return self._library.get(context_id)

    def clear_library(self) -> int:
        """
        Remove all non-archived context records.

        Returns the number of records removed.
        """
        to_remove = [
            cid for cid, r in self._library.items()
            if r.status != ContextStatus.ARCHIVED
        ]
        for cid in to_remove:
            del self._library[cid]
        return len(to_remove)

    # ----- Introspection --------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return engine introspection dict."""
        return {
            "engine_id": self.engine_id,
            "cluster": self.cluster,
            "mode": self._mode.value,
            "cycle_count": self._cycle_count,
            "tick": self._tick,
            "total_contexts": len(self._library),
            "state": {
                "ach_level": self._state.ach_level,
                "ne_level": self._state.ne_level,
                "da_level": self._state.da_level,
                "_5ht_level": self._state._5ht_level,
                "oxt_level": self._state.oxt_level,
                "cb1_level": self._state.cb1_level,
            },
        }
