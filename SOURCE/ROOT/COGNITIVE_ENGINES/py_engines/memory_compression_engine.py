"""
Engine 29 -- Memory Compression Engine  (``memory_compression_engine``)
=======================================================================

Determines compression strategy for memory packets transitioning between
tiers (STMM → MTMM → LTMM).  For each packet, the engine computes an
information-theoretic profile and selects a compression policy:

  - **VERBATIM**   -- keep everything (high salience / identity / unresolved)
  - **SEMANTIC**   -- keep meaning, drop exact wording
  - **SYMBOLIC**   -- reduce to symbolic tags + key metrics
  - **PRUNE**      -- discard entirely (below retention threshold)

Scoring axes:
  1. **Entropy**            -- information density of the content
  2. **Redundancy**         -- overlap with existing memory
  3. **Salience**           -- emotional + reward significance
  4. **Recency**            -- time since creation
  5. **Access Frequency**   -- how often referenced

Neurochemical coupling:
  ACh  -- preserves high-attention items (raises salience weight)
  5-HT -- protects emotionally significant items
  GABA -- accelerates pruning (raises redundancy weight)
  DA   -- preserves novel items (raises entropy weight)
  COR  -- stress preserves unresolved items

Complements:
  ``memory/short_term/compressor.py``  -- MemoryExitCompressor (STMM→MTMM format)
  ``memory/long_term/consolidation.py`` -- MemoryConsolidationEngine (promotion criteria)
  This engine provides the *compression policy* consumed by both.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from zados.cognitive_engines.constants import _clamp


# =========================================================================
# 1.  Enums
# =========================================================================

class CompressionPolicy(str, Enum):
    VERBATIM = "verbatim"   # Full preservation
    SEMANTIC = "semantic"   # Meaning-preserving compression
    SYMBOLIC = "symbolic"   # Reduced to symbols + metrics
    PRUNE    = "prune"      # Discard entirely


class TransitionType(str, Enum):
    STMM_TO_MTMM = "stmm_to_mtmm"
    MTMM_TO_LTMM = "mtmm_to_ltmm"
    LTMM_COLD    = "ltmm_cold"       # Cold storage compression


# =========================================================================
# 2.  Configuration
# =========================================================================

@dataclass(frozen=True)
class MemoryCompressionConfig:
    """Immutable configuration for the Memory Compression Engine."""

    # --- Axis weights for policy scoring ---
    w_entropy:       float = 0.25   # Information density
    w_redundancy:    float = 0.20   # Overlap with existing memory
    w_salience:      float = 0.25   # Emotional + reward significance
    w_recency:       float = 0.15   # Time decay
    w_access_freq:   float = 0.15   # How often referenced

    # --- Policy thresholds (on composite retention score) ---
    threshold_verbatim: float = 0.75   # Above → VERBATIM
    threshold_semantic: float = 0.50   # Above → SEMANTIC
    threshold_symbolic: float = 0.25   # Above → SYMBOLIC
    # Below threshold_symbolic → PRUNE

    # --- Recency decay ---
    recency_half_life_ticks: float = 100.0

    # --- Entropy estimation ---
    entropy_normalization:   float = 4.0   # Max expected entropy (bits/token)

    # --- Override rules ---
    identity_force_verbatim:     bool  = True   # Identity-relevant → always VERBATIM
    unresolved_force_semantic:   bool  = True   # Unresolved items → at least SEMANTIC
    emotional_threshold:         float = 0.70   # Above → at least SEMANTIC
    critical_flag_force_verbatim: bool = True   # CRITICAL flag → VERBATIM

    # --- NT modulation weights ---
    w_ach_salience:     float = 0.30   # ACh raises salience weight
    w_5ht_emotional:    float = 0.35   # 5-HT protects emotional
    w_gaba_redundancy:  float = 0.30   # GABA raises redundancy weight
    w_da_entropy:       float = 0.25   # DA raises entropy weight
    w_cor_unresolved:   float = 0.20   # Cortisol protects unresolved

    # --- Transition-specific modifiers ---
    transition_modifiers: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "stmm_to_mtmm": {
            "threshold_semantic": 0.40,   # Lower threshold (keep more)
        },
        "mtmm_to_ltmm": {
            "threshold_verbatim": 0.80,   # Higher bar for VERBATIM
            "threshold_symbolic": 0.30,   # Higher bar for SYMBOLIC
        },
        "ltmm_cold": {
            "threshold_verbatim": 0.90,
            "threshold_semantic": 0.60,
            "threshold_symbolic": 0.35,
        },
    })


# =========================================================================
# 3.  Input / Output types
# =========================================================================

@dataclass(frozen=True)
class CompressionAxisScores:
    """Per-axis breakdown for a single packet."""
    entropy:       float = 0.0   # [0, 1] Information density
    redundancy:    float = 0.0   # [0, 1] Overlap with existing
    salience:      float = 0.0   # [0, 1] Emotional + reward
    recency:       float = 0.0   # [0, 1] Time-decay factor
    access_freq:   float = 0.0   # [0, 1] Normalised access count


@dataclass(frozen=True)
class CompressionDecision:
    """Compression policy decision for a single packet."""
    packet_id:           str                = ""
    policy:              CompressionPolicy  = CompressionPolicy.SEMANTIC
    retention_score:     float              = 0.0
    axes:                CompressionAxisScores = field(default_factory=CompressionAxisScores)
    override_reason:     str                = ""     # Non-empty if an override was applied
    transition_type:     TransitionType     = TransitionType.STMM_TO_MTMM
    metadata:            Dict[str, Any]     = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryCompressionNeurochem:
    """Neurochemical output from compression cycle."""
    _5ht_delta:     float = 0.0   # Emotional preservation activity
    gaba_delta:     float = 0.0   # Pruning activity
    ach_delta:      float = 0.0   # Focused retention
    da_delta:       float = 0.0   # Novel content preservation
    alpha_boost:    float = 0.0   # Compression → alpha rhythm

    def as_dict(self) -> Dict[str, float]:
        return {
            "_5ht_delta":  self._5ht_delta,
            "gaba_delta":  self.gaba_delta,
            "ach_delta":   self.ach_delta,
            "da_delta":    self.da_delta,
            "alpha_boost": self.alpha_boost,
        }


@dataclass(frozen=True)
class MemoryCompressionResult:
    """Full output of one compression cycle."""
    decisions:           List[CompressionDecision] = field(default_factory=list)
    policy_counts:       Dict[str, int]            = field(default_factory=dict)
    mean_retention:      float                     = 0.0
    pruned_count:        int                       = 0
    verbatim_count:      int                       = 0
    neurochem_signals:   MemoryCompressionNeurochem = field(
        default_factory=MemoryCompressionNeurochem)
    processing_time_ms:  float                     = 0.0
    tick:                int                       = 0


# =========================================================================
# 4.  Packet descriptor (input to engine)
# =========================================================================

@dataclass
class PacketDescriptor:
    """
    Describes a memory packet for compression evaluation.
    Can be constructed from a MemoryPacket or provided manually.
    """
    packet_id:             str   = ""
    text_length:           int   = 0
    unique_tokens:         int   = 0
    total_tokens:          int   = 0
    emotional_significance: float = 0.0
    reward_mean:           float = 0.0
    has_unresolved:        bool  = False
    has_contradictions:    bool  = False
    is_identity_relevant:  bool  = False
    flags:                 List[str] = field(default_factory=list)
    creation_tick:         int   = 0
    access_count:          int   = 0
    overlap_score:         float = 0.0   # Pre-computed redundancy [0, 1]
    trust_weight:          float = 1.0


# =========================================================================
# 5.  Pure scoring functions
# =========================================================================

def compute_entropy_score(
    unique_tokens: int,
    total_tokens: int,
    normalisation: float,
) -> float:
    """
    Estimate information entropy from token distribution.
    H ≈ log2(unique_tokens) normalised by max expected entropy.
    """
    if unique_tokens <= 0 or total_tokens <= 0:
        return 0.0
    # Approximate entropy: assuming uniform distribution over unique tokens
    h = math.log2(max(1, unique_tokens))
    return _clamp(h / max(0.1, normalisation))


def compute_redundancy_score(overlap: float) -> float:
    """Redundancy = pre-computed overlap with existing memory."""
    return _clamp(overlap)


def compute_salience_score(
    emotional_significance: float,
    reward_mean: float,
    trust_weight: float,
) -> float:
    """Salience = weighted blend of emotional significance and reward."""
    raw = 0.5 * emotional_significance + 0.3 * reward_mean + 0.2 * (1.0 - trust_weight)
    return _clamp(raw)


def compute_recency_score(
    ticks_since_creation: int,
    half_life: float,
) -> float:
    """Exponential decay from creation."""
    if ticks_since_creation <= 0:
        return 1.0
    lam = math.log(2.0) / max(1.0, half_life)
    return math.exp(-lam * ticks_since_creation)


def compute_access_frequency_score(
    access_count: int,
    max_expected: int = 20,
) -> float:
    """Normalised access frequency."""
    return _clamp(access_count / max(1, max_expected))


def compute_retention_score(
    axes: CompressionAxisScores,
    weights: Dict[str, float],
) -> float:
    """Weighted retention score. High = keep more, Low = compress more."""
    # Redundancy is inverted: high redundancy → low retention
    return _clamp(
        weights.get("w_entropy", 0.25) * axes.entropy
        + weights.get("w_redundancy", 0.20) * (1.0 - axes.redundancy)
        + weights.get("w_salience", 0.25) * axes.salience
        + weights.get("w_recency", 0.15) * axes.recency
        + weights.get("w_access_freq", 0.15) * axes.access_freq
    )


def classify_policy(
    retention_score: float,
    thresholds: Dict[str, float],
) -> CompressionPolicy:
    """Threshold-based policy classification."""
    if retention_score >= thresholds.get("threshold_verbatim", 0.75):
        return CompressionPolicy.VERBATIM
    if retention_score >= thresholds.get("threshold_semantic", 0.50):
        return CompressionPolicy.SEMANTIC
    if retention_score >= thresholds.get("threshold_symbolic", 0.25):
        return CompressionPolicy.SYMBOLIC
    return CompressionPolicy.PRUNE


def apply_policy_overrides(
    policy: CompressionPolicy,
    desc: PacketDescriptor,
    cfg: MemoryCompressionConfig,
    cor_level: float,
) -> Tuple[CompressionPolicy, str]:
    """
    Apply override rules. Policy can only be RAISED, never lowered.
    Returns (new_policy, override_reason).
    """
    _POLICY_ORDER = {
        CompressionPolicy.PRUNE: 0,
        CompressionPolicy.SYMBOLIC: 1,
        CompressionPolicy.SEMANTIC: 2,
        CompressionPolicy.VERBATIM: 3,
    }
    _REVERSE = {v: k for k, v in _POLICY_ORDER.items()}

    current = _POLICY_ORDER[policy]
    reason = ""

    # Identity → VERBATIM
    if cfg.identity_force_verbatim and desc.is_identity_relevant:
        if current < _POLICY_ORDER[CompressionPolicy.VERBATIM]:
            current = _POLICY_ORDER[CompressionPolicy.VERBATIM]
            reason = "identity_relevant"

    # Critical flag → VERBATIM
    if cfg.critical_flag_force_verbatim:
        flag_names = {f.split(":")[0].upper() for f in desc.flags}
        if "CRITICAL" in flag_names or "IDENTITY" in flag_names:
            if current < _POLICY_ORDER[CompressionPolicy.VERBATIM]:
                current = _POLICY_ORDER[CompressionPolicy.VERBATIM]
                reason = "critical_flag"

    # Unresolved → at least SEMANTIC
    if cfg.unresolved_force_semantic and (desc.has_unresolved or desc.has_contradictions):
        # Cortisol further protects unresolved items
        min_policy = CompressionPolicy.SEMANTIC
        if cor_level > 0.6:
            min_policy = CompressionPolicy.VERBATIM
        if current < _POLICY_ORDER[min_policy]:
            current = _POLICY_ORDER[min_policy]
            reason = "unresolved_items"

    # High emotional significance → at least SEMANTIC
    if desc.emotional_significance >= cfg.emotional_threshold:
        if current < _POLICY_ORDER[CompressionPolicy.SEMANTIC]:
            current = _POLICY_ORDER[CompressionPolicy.SEMANTIC]
            reason = "emotional_override"

    return _REVERSE[current], reason


def compute_effective_weights(
    cfg: MemoryCompressionConfig,
    ach: float,
    sht: float,
    gaba: float,
    da: float,
) -> Dict[str, float]:
    """NT-modulated axis weights."""
    w = {
        "w_entropy":     cfg.w_entropy + cfg.w_da_entropy * da,
        "w_redundancy":  cfg.w_redundancy + cfg.w_gaba_redundancy * gaba,
        "w_salience":    cfg.w_salience + cfg.w_ach_salience * ach + cfg.w_5ht_emotional * sht,
        "w_recency":     cfg.w_recency,
        "w_access_freq": cfg.w_access_freq,
    }
    # Renormalise
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}
    return w


def compute_compression_neurochem(
    verbatim_count: int,
    pruned_count: int,
    emotional_preserved: int,
    novel_preserved: int,
    total_processed: int,
) -> MemoryCompressionNeurochem:
    """Compute NT output from compression events."""
    return MemoryCompressionNeurochem(
        _5ht_delta=min(0.2, emotional_preserved * 0.04),
        gaba_delta=min(0.2, pruned_count * 0.03),
        ach_delta=0.05 if verbatim_count > 0 else 0.0,
        da_delta=min(0.15, novel_preserved * 0.03),
        alpha_boost=min(0.15, total_processed * 0.01),
    )


# =========================================================================
# 6.  Engine class
# =========================================================================

class MemoryCompressionEngine:
    """
    Engine 29 -- Memory Compression.

    Evaluates memory packets and assigns compression policies
    based on information-theoretic scoring and neurochemical modulation.

    API
    ---
    update_neurochem_state(state)  -- inject NT levels (Pattern A)
    evaluate(descriptors, transition_type)  -- evaluate packets
    process(input_data)            -- pipeline entry point
    get_status()                   -- introspection
    """

    engine_id = "memory_compression_engine"
    cluster   = "homeostasis"

    def __init__(
        self,
        config: Optional[MemoryCompressionConfig] = None,
    ) -> None:
        self._cfg = config or MemoryCompressionConfig()

        # NT levels (Pattern A)
        self.ach_level:  float = 0.5
        self._5ht_level: float = 0.5
        self.gaba_level: float = 0.5
        self.da_level:   float = 0.5
        self.cor_level:  float = 0.5

        # State
        self._tick:     int = 0
        self._total_evaluated: int = 0
        self._total_pruned:    int = 0
        self._total_verbatim:  int = 0

    # -----------------------------------------------------------------
    # Pattern A: NT State
    # -----------------------------------------------------------------

    def update_neurochem_state(self, nt_state: Dict[str, float]) -> None:
        self.ach_level  = _clamp(nt_state.get("ach",  self.ach_level))
        self._5ht_level = _clamp(nt_state.get("5ht",  self._5ht_level))
        self.gaba_level = _clamp(nt_state.get("gaba", self.gaba_level))
        self.da_level   = _clamp(nt_state.get("da",   self.da_level))
        self.cor_level  = _clamp(nt_state.get("cor",  self.cor_level))

    # -----------------------------------------------------------------
    # Core: Evaluate packets
    # -----------------------------------------------------------------

    def evaluate(
        self,
        descriptors: List[PacketDescriptor],
        transition_type: TransitionType = TransitionType.STMM_TO_MTMM,
    ) -> MemoryCompressionResult:
        """Evaluate a batch of packet descriptors and assign compression policies."""
        t0 = time.perf_counter()
        self._tick += 1

        # Get transition-specific threshold overrides
        transition_overrides = self._cfg.transition_modifiers.get(transition_type.value, {})
        thresholds = {
            "threshold_verbatim": transition_overrides.get(
                "threshold_verbatim", self._cfg.threshold_verbatim),
            "threshold_semantic": transition_overrides.get(
                "threshold_semantic", self._cfg.threshold_semantic),
            "threshold_symbolic": transition_overrides.get(
                "threshold_symbolic", self._cfg.threshold_symbolic),
        }

        # Compute effective weights
        eff_weights = compute_effective_weights(
            self._cfg, self.ach_level, self._5ht_level, self.gaba_level, self.da_level,
        )

        decisions: List[CompressionDecision] = []
        emotional_preserved = 0
        novel_preserved = 0

        for desc in descriptors:
            # Compute axes
            entropy = compute_entropy_score(
                desc.unique_tokens, desc.total_tokens, self._cfg.entropy_normalization,
            )
            redundancy = compute_redundancy_score(desc.overlap_score)
            salience = compute_salience_score(
                desc.emotional_significance, desc.reward_mean, desc.trust_weight,
            )
            recency = compute_recency_score(
                max(0, self._tick - desc.creation_tick),
                self._cfg.recency_half_life_ticks,
            )
            access_freq = compute_access_frequency_score(desc.access_count)

            axes = CompressionAxisScores(
                entropy=entropy,
                redundancy=redundancy,
                salience=salience,
                recency=recency,
                access_freq=access_freq,
            )

            # Retention score
            retention = compute_retention_score(axes, eff_weights)

            # Policy classification
            policy = classify_policy(retention, thresholds)

            # Apply overrides
            policy, override_reason = apply_policy_overrides(
                policy, desc, self._cfg, self.cor_level,
            )

            decisions.append(CompressionDecision(
                packet_id=desc.packet_id,
                policy=policy,
                retention_score=retention,
                axes=axes,
                override_reason=override_reason,
                transition_type=transition_type,
                metadata={"effective_weights": eff_weights},
            ))

            # Track neurochem-relevant counters
            if desc.emotional_significance >= self._cfg.emotional_threshold and policy != CompressionPolicy.PRUNE:
                emotional_preserved += 1
            if entropy > 0.6 and policy != CompressionPolicy.PRUNE:
                novel_preserved += 1

        # Policy counts
        policy_counts: Dict[str, int] = {}
        for d in decisions:
            policy_counts[d.policy.value] = policy_counts.get(d.policy.value, 0) + 1

        verbatim_count = policy_counts.get(CompressionPolicy.VERBATIM.value, 0)
        pruned_count = policy_counts.get(CompressionPolicy.PRUNE.value, 0)
        retention_scores = [d.retention_score for d in decisions]
        mean_ret = sum(retention_scores) / max(1, len(retention_scores))

        # Update lifetime counters
        self._total_evaluated += len(descriptors)
        self._total_pruned += pruned_count
        self._total_verbatim += verbatim_count

        # Neurochem output
        signals = compute_compression_neurochem(
            verbatim_count, pruned_count, emotional_preserved,
            novel_preserved, len(descriptors),
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        return MemoryCompressionResult(
            decisions=decisions,
            policy_counts=policy_counts,
            mean_retention=mean_ret,
            pruned_count=pruned_count,
            verbatim_count=verbatim_count,
            neurochem_signals=signals,
            processing_time_ms=elapsed,
            tick=self._tick,
        )

    # -----------------------------------------------------------------
    # process() -- Pipeline Entry Point
    # -----------------------------------------------------------------

    def process(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Pipeline entry point."""
        input_data = input_data or {}

        # 1. NT state
        if "nt_state" in input_data:
            self.update_neurochem_state(input_data["nt_state"])

        # 2. Build descriptors
        descriptors = []
        for pkt_data in input_data.get("packets", []):
            descriptors.append(PacketDescriptor(
                packet_id=pkt_data.get("packet_id", ""),
                text_length=pkt_data.get("text_length", 0),
                unique_tokens=pkt_data.get("unique_tokens", 0),
                total_tokens=pkt_data.get("total_tokens", 0),
                emotional_significance=pkt_data.get("emotional_significance", 0.0),
                reward_mean=pkt_data.get("reward_mean", 0.0),
                has_unresolved=pkt_data.get("has_unresolved", False),
                has_contradictions=pkt_data.get("has_contradictions", False),
                is_identity_relevant=pkt_data.get("is_identity_relevant", False),
                flags=pkt_data.get("flags", []),
                creation_tick=pkt_data.get("creation_tick", 0),
                access_count=pkt_data.get("access_count", 0),
                overlap_score=pkt_data.get("overlap_score", 0.0),
                trust_weight=pkt_data.get("trust_weight", 1.0),
            ))

        # 3. Transition type
        transition_str = input_data.get("transition_type", "stmm_to_mtmm")
        try:
            transition = TransitionType(transition_str)
        except ValueError:
            transition = TransitionType.STMM_TO_MTMM

        # 4. Evaluate
        result = self.evaluate(descriptors, transition)

        return {
            "decisions": [
                {
                    "packet_id": d.packet_id,
                    "policy": d.policy.value,
                    "retention_score": d.retention_score,
                    "override_reason": d.override_reason,
                    "axes": {
                        "entropy": d.axes.entropy,
                        "redundancy": d.axes.redundancy,
                        "salience": d.axes.salience,
                        "recency": d.axes.recency,
                        "access_freq": d.axes.access_freq,
                    },
                }
                for d in result.decisions
            ],
            "policy_counts": result.policy_counts,
            "mean_retention": result.mean_retention,
            "pruned_count": result.pruned_count,
            "verbatim_count": result.verbatim_count,
            "neurochem_signals": result.neurochem_signals.as_dict(),
            "tick": result.tick,
            "processing_time_ms": result.processing_time_ms,
        }

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine_id":        self.engine_id,
            "cluster":          self.cluster,
            "tick":             self._tick,
            "total_evaluated":  self._total_evaluated,
            "total_pruned":     self._total_pruned,
            "total_verbatim":   self._total_verbatim,
            "nt_levels": {
                "ach":  self.ach_level,
                "5ht":  self._5ht_level,
                "gaba": self.gaba_level,
                "da":   self.da_level,
                "cor":  self.cor_level,
            },
        }

    def __repr__(self) -> str:
        return (
            f"MemoryCompressionEngine(tick={self._tick}, "
            f"evaluated={self._total_evaluated}, pruned={self._total_pruned})"
        )
