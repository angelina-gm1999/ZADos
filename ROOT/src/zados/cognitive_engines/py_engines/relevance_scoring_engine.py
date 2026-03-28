"""
Engine 8 -- Relevance Scoring Engine  (``relevance_scoring_engine``)
=====================================================================

Assigns a composite relevance score to every concept, token, or atom
that flows through the processing pipeline.  The score is a weighted
blend of six orthogonal axes:

  1. **Recency**            -- exponential decay from last-access timestamp
  2. **Frequency**          -- normalised access count over a sliding window
  3. **Semantic Proximity** -- TruthValue.strength from AtomSpace relations
  4. **Attention Weight**   -- ECAN STI normalised to [0, 1]
  5. **Contextual Fit**     -- cosine similarity to the active context vector
  6. **Novelty Bonus**      -- inverse of frequency + recency (rewards new items)

Neurochemical coupling:
  ACh  -- tightens threshold (only high-scoring items pass)
  NE   -- broadens scope (lowers threshold, more items pass)
  DA   -- boosts novelty axis weight
  5-HT -- stabilises weights (reduces NT sensitivity)
  GABA -- raises threshold (stronger pruning)

Interacts with:
  E9  (AtomSpace)  -- reads TruthValues / atom metadata
  E16 (ECAN)       -- reads STI / attentional focus membership
  E11 (IRE)        -- downstream consumer of relevance scores
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from zados.cognitive_engines.constants import _clamp


# =========================================================================
# 1.  Configuration
# =========================================================================

@dataclass(frozen=True)
class RelevanceScoringConfig:
    """Immutable configuration for the Relevance Scoring Engine."""

    # --- Axis weights (sum to 1.0 in default mode) ---
    w_recency:           float = 0.20
    w_frequency:         float = 0.15
    w_semantic_proximity: float = 0.20
    w_attention_weight:  float = 0.20
    w_contextual_fit:    float = 0.15
    w_novelty_bonus:     float = 0.10

    # --- Recency decay ---
    recency_half_life:   float = 50.0   # ticks until 50% decay
    recency_lambda:      float = 0.0    # computed in __post_init__

    # --- Frequency window ---
    frequency_window:    int   = 100    # sliding window size (ticks)

    # --- Threshold ---
    relevance_threshold: float = 0.30   # items below this are pruned

    # --- Novelty ---
    novelty_freq_cap:    float = 10.0   # frequency count at which novelty = 0
    novelty_recency_cap: float = 5.0    # ticks-since-access at which novelty plateaus

    # --- NT modulation weights ---
    w_ach_threshold:     float = 0.25   # ACh tightens threshold
    w_ne_threshold:      float = 0.20   # NE lowers threshold
    w_da_novelty:        float = 0.30   # DA boosts novelty weight
    w_5ht_stability:     float = 0.20   # 5-HT dampens NT effects
    w_gaba_threshold:    float = 0.25   # GABA raises threshold

    # --- Mode overrides ---
    mode_configs: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "ANALYTICAL": {
            "w_semantic_proximity": 0.30,
            "w_attention_weight":  0.25,
            "w_novelty_bonus":     0.05,
            "relevance_threshold": 0.40,
        },
        "CREATIVE": {
            "w_novelty_bonus":     0.25,
            "w_contextual_fit":    0.10,
            "relevance_threshold": 0.20,
        },
        "REM_DREAM": {
            "w_novelty_bonus":     0.30,
            "w_recency":           0.10,
            "relevance_threshold": 0.15,
        },
        "DEFAULT": {},
    })

    def __post_init__(self) -> None:
        hl = self.recency_half_life if self.recency_half_life > 0 else 50.0
        lam = math.log(2.0) / hl
        object.__setattr__(self, "recency_lambda", lam)


# =========================================================================
# 2.  Frozen output types
# =========================================================================

@dataclass(frozen=True)
class RelevanceAxisScores:
    """Per-axis breakdown for a single item."""
    recency:            float = 0.0
    frequency:          float = 0.0
    semantic_proximity: float = 0.0
    attention_weight:   float = 0.0
    contextual_fit:     float = 0.0
    novelty_bonus:      float = 0.0


@dataclass(frozen=True)
class ScoredItem:
    """A single item with its composite relevance score."""
    item_id:       str                = ""
    composite:     float              = 0.0
    axes:          RelevanceAxisScores = field(default_factory=RelevanceAxisScores)
    above_threshold: bool             = True


@dataclass(frozen=True)
class RelevanceScoringNeurochem:
    """Neurochemical output from the scoring cycle."""
    da_delta:       float = 0.0   # Novel items found
    ach_delta:      float = 0.0   # High-focus items
    ne_delta:       float = 0.0   # Scope expansion signal
    _5ht_delta:     float = 0.0   # Stability from consistent patterns
    gamma_boost:    float = 0.0   # Active scoring → gamma
    beta_boost:     float = 0.0   # Focused threshold → beta

    def as_dict(self) -> Dict[str, float]:
        return {
            "da_delta":    self.da_delta,
            "ach_delta":   self.ach_delta,
            "ne_delta":    self.ne_delta,
            "_5ht_delta":  self._5ht_delta,
            "gamma_boost": self.gamma_boost,
            "beta_boost":  self.beta_boost,
        }


@dataclass(frozen=True)
class RelevanceScoringResult:
    """Full output of one scoring cycle."""
    scored_items:          List[ScoredItem]            = field(default_factory=list)
    above_threshold_count: int                         = 0
    below_threshold_count: int                         = 0
    mean_relevance:        float                       = 0.0
    max_relevance:         float                       = 0.0
    effective_threshold:   float                       = 0.0
    neurochem_signals:     RelevanceScoringNeurochem   = field(default_factory=RelevanceScoringNeurochem)
    processing_time_ms:    float                       = 0.0
    tick:                  int                         = 0


# =========================================================================
# 3.  Item tracker (mutable internal state)
# =========================================================================

@dataclass
class _ItemRecord:
    """Internal tracking record for a single scoreable item."""
    item_id:         str
    last_access_tick: int   = 0
    access_count:    int   = 0
    access_history:  List[int] = field(default_factory=list)
    semantic_score:  float = 0.0   # From AtomSpace TV strength
    sti_normalized:  float = 0.0   # From ECAN STI
    in_af:           bool  = False  # In attentional focus
    context_sim:     float = 0.0   # Contextual cosine
    metadata:        Dict[str, Any] = field(default_factory=dict)


# =========================================================================
# 4.  Pure scoring functions
# =========================================================================

def compute_recency_score(
    ticks_since_access: int,
    lam: float,
) -> float:
    """Exponential decay: exp(-lambda * delta_t)."""
    if ticks_since_access <= 0:
        return 1.0
    return math.exp(-lam * ticks_since_access)


def compute_frequency_score(
    access_history: List[int],
    current_tick: int,
    window: int,
) -> float:
    """Normalised frequency within sliding window."""
    if window <= 0:
        return 0.0
    count = sum(1 for t in access_history if current_tick - t <= window)
    # Normalise so that hitting every tick = 1.0
    return min(1.0, count / max(1, window) * 10.0)


def compute_novelty_bonus(
    frequency_score: float,
    recency_score: float,
    freq_cap: float,
    recency_cap: float,
) -> float:
    """Novelty = inverse of frequency and recency (new things score high)."""
    inv_freq = max(0.0, 1.0 - frequency_score)
    inv_recent = max(0.0, 1.0 - recency_score)
    # Novel items have LOW frequency and LOW recency (just appeared)
    # Actually novel = first appearance → high recency, low frequency
    # So novelty = high recency * low frequency
    return _clamp(recency_score * inv_freq)


def compute_composite_relevance(
    axes: RelevanceAxisScores,
    weights: Dict[str, float],
) -> float:
    """Weighted sum of axis scores."""
    return _clamp(
        weights.get("w_recency", 0.20) * axes.recency
        + weights.get("w_frequency", 0.15) * axes.frequency
        + weights.get("w_semantic_proximity", 0.20) * axes.semantic_proximity
        + weights.get("w_attention_weight", 0.20) * axes.attention_weight
        + weights.get("w_contextual_fit", 0.15) * axes.contextual_fit
        + weights.get("w_novelty_bonus", 0.10) * axes.novelty_bonus
    )


def compute_effective_threshold(
    base: float,
    ach: float,
    ne: float,
    gaba: float,
    w_ach: float,
    w_ne: float,
    w_gaba: float,
) -> float:
    """NT-modulated relevance threshold."""
    t = base * (1.0 + w_ach * ach) * (1.0 + w_gaba * gaba) * max(0.2, 1.0 - w_ne * ne)
    return _clamp(t, 0.05, 0.95)


def compute_effective_weights(
    base_weights: Dict[str, float],
    da: float,
    sht: float,
    w_da_novelty: float,
    w_5ht_stability: float,
) -> Dict[str, float]:
    """NT-modulated axis weights."""
    w = dict(base_weights)

    # DA boosts novelty weight, dampened by 5-HT
    stability_factor = max(0.2, 1.0 - w_5ht_stability * sht)
    da_boost = w_da_novelty * da * stability_factor
    w["w_novelty_bonus"] = w.get("w_novelty_bonus", 0.10) + da_boost

    # Renormalise
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}

    return w


def compute_scoring_neurochem(
    novel_count: int,
    focused_count: int,
    total_scored: int,
    mean_relevance: float,
) -> RelevanceScoringNeurochem:
    """Compute NT output from scoring cycle events."""
    return RelevanceScoringNeurochem(
        da_delta=min(0.3, novel_count * 0.03),
        ach_delta=0.05 if 0 < focused_count <= 10 else 0.0,
        ne_delta=min(0.2, total_scored * 0.005),
        _5ht_delta=0.04 if mean_relevance > 0.5 else 0.0,
        gamma_boost=min(0.2, total_scored * 0.01),
        beta_boost=0.05 if focused_count > 0 else 0.0,
    )


# =========================================================================
# 5.  Context vector helpers (sparse cosine)
# =========================================================================

def _sparse_cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
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
    return max(0.0, dot / (norm_a * norm_b))


# =========================================================================
# 6.  Engine class
# =========================================================================

class RelevanceScoringEngine:
    """
    Engine 8 -- Relevance Scoring.

    Maintains internal tracking of items (concepts, tokens, atoms) and
    produces composite relevance scores each cycle.

    API
    ---
    update_neurochem_state(state)  -- inject NT levels (Pattern A)
    register_item(item_id, ...)   -- register a new item for scoring
    mark_accessed(item_id)        -- record an access event
    update_item_signals(...)      -- update semantic/ECAN/context signals
    set_context_vector(vec)       -- set the active context for contextual_fit
    process(input_data)           -- run a scoring cycle
    get_status()                  -- introspection
    """

    engine_id = "relevance_scoring_engine"
    cluster   = "pattern_analysis"

    def __init__(
        self,
        config: Optional[RelevanceScoringConfig] = None,
    ) -> None:
        self._cfg = config or RelevanceScoringConfig()

        # NT levels (Pattern A)
        self.ach_level:  float = 0.5
        self.ne_level:   float = 0.5
        self.da_level:   float = 0.5
        self._5ht_level: float = 0.5
        self.gaba_level: float = 0.5

        # State
        self._mode:         str = "DEFAULT"
        self._tick:         int = 0
        self._items: Dict[str, _ItemRecord] = {}
        self._context_vector: Dict[str, float] = {}

    # -----------------------------------------------------------------
    # Pattern A: Neurochemical State Update
    # -----------------------------------------------------------------

    def update_neurochem_state(self, nt_state: Dict[str, float]) -> None:
        self.ach_level  = _clamp(nt_state.get("ach",  self.ach_level))
        self.ne_level   = _clamp(nt_state.get("ne",   self.ne_level))
        self.da_level   = _clamp(nt_state.get("da",   self.da_level))
        self._5ht_level = _clamp(nt_state.get("5ht",  self._5ht_level))
        self.gaba_level = _clamp(nt_state.get("gaba", self.gaba_level))

    # -----------------------------------------------------------------
    # Item management
    # -----------------------------------------------------------------

    def register_item(
        self,
        item_id: str,
        semantic_score: float = 0.0,
        sti_normalized: float = 0.0,
        in_af: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register or update a scoreable item."""
        if item_id in self._items:
            rec = self._items[item_id]
            rec.semantic_score = semantic_score
            rec.sti_normalized = sti_normalized
            rec.in_af = in_af
            if metadata:
                rec.metadata.update(metadata)
        else:
            self._items[item_id] = _ItemRecord(
                item_id=item_id,
                last_access_tick=self._tick,
                semantic_score=semantic_score,
                sti_normalized=sti_normalized,
                in_af=in_af,
                metadata=metadata or {},
            )

    def mark_accessed(self, item_id: str) -> None:
        """Record an access event for an item."""
        if item_id in self._items:
            rec = self._items[item_id]
            rec.last_access_tick = self._tick
            rec.access_count += 1
            rec.access_history.append(self._tick)
            # Trim history to window size
            if len(rec.access_history) > self._cfg.frequency_window:
                rec.access_history = rec.access_history[-self._cfg.frequency_window:]

    def update_item_signals(
        self,
        item_id: str,
        semantic_score: Optional[float] = None,
        sti_normalized: Optional[float] = None,
        in_af: Optional[bool] = None,
        context_sim: Optional[float] = None,
    ) -> None:
        """Update external signal values for an item."""
        if item_id not in self._items:
            return
        rec = self._items[item_id]
        if semantic_score is not None:
            rec.semantic_score = _clamp(semantic_score)
        if sti_normalized is not None:
            rec.sti_normalized = _clamp(sti_normalized)
        if in_af is not None:
            rec.in_af = in_af
        if context_sim is not None:
            rec.context_sim = _clamp(context_sim)

    def set_context_vector(self, context: Dict[str, float]) -> None:
        """Set the active context vector for contextual_fit scoring."""
        self._context_vector = dict(context)

    def remove_item(self, item_id: str) -> None:
        """Remove an item from tracking."""
        self._items.pop(item_id, None)

    # -----------------------------------------------------------------
    # Mode configuration
    # -----------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Set operational mode (affects weights and thresholds)."""
        self._mode = mode

    # -----------------------------------------------------------------
    # Core: Score all items
    # -----------------------------------------------------------------

    def _get_base_weights(self) -> Dict[str, float]:
        """Get base weights for the current mode."""
        mode_overrides = self._cfg.mode_configs.get(self._mode, {})
        return {
            "w_recency":            mode_overrides.get("w_recency", self._cfg.w_recency),
            "w_frequency":          mode_overrides.get("w_frequency", self._cfg.w_frequency),
            "w_semantic_proximity": mode_overrides.get("w_semantic_proximity", self._cfg.w_semantic_proximity),
            "w_attention_weight":   mode_overrides.get("w_attention_weight", self._cfg.w_attention_weight),
            "w_contextual_fit":     mode_overrides.get("w_contextual_fit", self._cfg.w_contextual_fit),
            "w_novelty_bonus":      mode_overrides.get("w_novelty_bonus", self._cfg.w_novelty_bonus),
        }

    def _get_base_threshold(self) -> float:
        """Get base threshold for the current mode."""
        mode_overrides = self._cfg.mode_configs.get(self._mode, {})
        return mode_overrides.get("relevance_threshold", self._cfg.relevance_threshold)

    def score_all(self) -> RelevanceScoringResult:
        """Score all registered items and return results."""
        t0 = time.perf_counter()
        self._tick += 1

        # Compute effective weights and threshold
        base_weights = self._get_base_weights()
        eff_weights = compute_effective_weights(
            base_weights, self.da_level, self._5ht_level,
            self._cfg.w_da_novelty, self._cfg.w_5ht_stability,
        )
        eff_threshold = compute_effective_threshold(
            self._get_base_threshold(),
            self.ach_level, self.ne_level, self.gaba_level,
            self._cfg.w_ach_threshold, self._cfg.w_ne_threshold,
            self._cfg.w_gaba_threshold,
        )

        scored: List[ScoredItem] = []
        novel_count = 0
        focused_count = 0

        for rec in self._items.values():
            # Axis 1: Recency
            ticks_since = max(0, self._tick - rec.last_access_tick)
            recency = compute_recency_score(ticks_since, self._cfg.recency_lambda)

            # Axis 2: Frequency
            frequency = compute_frequency_score(
                rec.access_history, self._tick, self._cfg.frequency_window,
            )

            # Axis 3: Semantic proximity (from AtomSpace TV)
            semantic = rec.semantic_score

            # Axis 4: Attention weight (from ECAN STI)
            attention = rec.sti_normalized

            # Axis 5: Contextual fit
            if self._context_vector and rec.metadata.get("features"):
                context_fit = _sparse_cosine(self._context_vector, rec.metadata["features"])
            else:
                context_fit = rec.context_sim

            # Axis 6: Novelty bonus
            novelty = compute_novelty_bonus(
                frequency, recency,
                self._cfg.novelty_freq_cap, self._cfg.novelty_recency_cap,
            )

            axes = RelevanceAxisScores(
                recency=recency,
                frequency=frequency,
                semantic_proximity=semantic,
                attention_weight=attention,
                contextual_fit=context_fit,
                novelty_bonus=novelty,
            )

            composite = compute_composite_relevance(axes, eff_weights)
            above = composite >= eff_threshold

            scored.append(ScoredItem(
                item_id=rec.item_id,
                composite=composite,
                axes=axes,
                above_threshold=above,
            ))

            if novelty > 0.5:
                novel_count += 1
            if rec.in_af:
                focused_count += 1

        # Statistics
        above_count = sum(1 for s in scored if s.above_threshold)
        below_count = len(scored) - above_count
        composites = [s.composite for s in scored]
        mean_rel = sum(composites) / max(1, len(composites))
        max_rel = max(composites, default=0.0)

        # Neurochem output
        signals = compute_scoring_neurochem(
            novel_count, focused_count, len(scored), mean_rel,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        return RelevanceScoringResult(
            scored_items=scored,
            above_threshold_count=above_count,
            below_threshold_count=below_count,
            mean_relevance=mean_rel,
            max_relevance=max_rel,
            effective_threshold=eff_threshold,
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

        # 2. Mode
        if "mode" in input_data:
            self.set_mode(input_data["mode"])

        # 3. Register new items
        for item in input_data.get("items", []):
            self.register_item(
                item_id=item["item_id"],
                semantic_score=item.get("semantic_score", 0.0),
                sti_normalized=item.get("sti_normalized", 0.0),
                in_af=item.get("in_af", False),
                metadata=item.get("metadata"),
            )

        # 4. Mark accesses
        for item_id in input_data.get("accessed", []):
            self.mark_accessed(item_id)

        # 5. Update signals
        for sig in input_data.get("signals", []):
            self.update_item_signals(
                item_id=sig["item_id"],
                semantic_score=sig.get("semantic_score"),
                sti_normalized=sig.get("sti_normalized"),
                in_af=sig.get("in_af"),
                context_sim=sig.get("context_sim"),
            )

        # 6. Context vector
        if "context_vector" in input_data:
            self.set_context_vector(input_data["context_vector"])

        # 7. Score
        result = self.score_all()

        return {
            "scored_items": [
                {
                    "item_id": s.item_id,
                    "composite": s.composite,
                    "above_threshold": s.above_threshold,
                    "axes": {
                        "recency": s.axes.recency,
                        "frequency": s.axes.frequency,
                        "semantic_proximity": s.axes.semantic_proximity,
                        "attention_weight": s.axes.attention_weight,
                        "contextual_fit": s.axes.contextual_fit,
                        "novelty_bonus": s.axes.novelty_bonus,
                    },
                }
                for s in result.scored_items
            ],
            "above_threshold_count": result.above_threshold_count,
            "below_threshold_count": result.below_threshold_count,
            "mean_relevance": result.mean_relevance,
            "max_relevance": result.max_relevance,
            "effective_threshold": result.effective_threshold,
            "neurochem_signals": result.neurochem_signals.as_dict(),
            "tick": result.tick,
            "processing_time_ms": result.processing_time_ms,
        }

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        return {
            "engine_id":    self.engine_id,
            "cluster":      self.cluster,
            "mode":         self._mode,
            "tick":         self._tick,
            "tracked_items": len(self._items),
            "nt_levels": {
                "ach":  self.ach_level,
                "ne":   self.ne_level,
                "da":   self.da_level,
                "5ht":  self._5ht_level,
                "gaba": self.gaba_level,
            },
        }

    def __repr__(self) -> str:
        return f"RelevanceScoringEngine(mode={self._mode}, items={len(self._items)})"
