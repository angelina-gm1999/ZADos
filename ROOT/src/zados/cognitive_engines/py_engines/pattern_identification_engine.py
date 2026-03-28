"""
Engine 19 -- Pattern Identification Engine  (``pattern_identification_engine``)
===============================================================================

Detects recurring patterns in the input stream across four dimensions:

  1. **Temporal**    -- recurring sequences with period estimation
  2. **Structural**  -- repeated syntactic / organisational structures
  3. **Semantic**    -- recurring meaning clusters / topic recurrence
  4. **Behavioral**  -- repeated user intent / interaction patterns

Detection pipeline (per cycle):
  Tokenize input → sliding window extraction → hash fingerprinting →
  fingerprint matching against stored patterns → score update →
  pattern promotion / decay

Neurochemical coupling:
  DA   -- rewards novel pattern discovery (phasic spike on first detection)
  5-HT -- stabilises established patterns (reduces decay)
  ACh  -- tightens pattern matching thresholds
  NE   -- broadens window (more elements per window)
  GABA -- accelerates decay of low-confidence patterns

Interacts with:
  E9  (AtomSpace) -- writes confirmed patterns as ConceptNodes with TV
  E20 (Pattern Comparison) -- downstream consumer
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from zados.cognitive_engines.constants import _clamp


# =========================================================================
# 1.  Enums
# =========================================================================

class PatternType(str, Enum):
    TEMPORAL   = "temporal"
    STRUCTURAL = "structural"
    SEMANTIC   = "semantic"
    BEHAVIORAL = "behavioral"


class PatternStatus(str, Enum):
    CANDIDATE  = "candidate"    # Detected but not yet confirmed
    CONFIRMED  = "confirmed"    # Occurred enough times with confidence
    DECAYING   = "decaying"     # No recent observations, losing confidence


# =========================================================================
# 2.  Configuration
# =========================================================================

@dataclass(frozen=True)
class PatternIdentificationConfig:
    """Immutable configuration for the Pattern Identification Engine."""

    # --- Sliding window ---
    window_size:        int   = 5      # Elements per window
    window_step:        int   = 1      # Step between windows
    max_windows:        int   = 50     # Max windows to keep per type

    # --- Fingerprinting ---
    min_ngram_size:     int   = 2      # Minimum n-gram for fingerprinting
    max_ngram_size:     int   = 4      # Maximum n-gram

    # --- Pattern lifecycle ---
    confirmation_threshold: int   = 3   # Occurrences to confirm
    decay_rate:             float = 0.05 # Confidence decay per tick without observation
    min_confidence:         float = 0.10 # Below this → remove pattern
    max_patterns_per_type:  int   = 200  # Capacity per PatternType
    initial_confidence:     float = 0.30 # Starting confidence for new candidates

    # --- Semantic similarity threshold ---
    semantic_sim_threshold: float = 0.60 # Min similarity to count as semantic match

    # --- NT modulation ---
    w_da_discovery:     float = 0.30  # DA on novel pattern
    w_5ht_decay:        float = 0.40  # 5-HT reduces decay
    w_ach_threshold:    float = 0.25  # ACh tightens confirmation threshold
    w_ne_window:        float = 0.20  # NE broadens window
    w_gaba_decay:       float = 0.30  # GABA accelerates decay

    # --- Mode overrides ---
    mode_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "ANALYTICAL": {
            "confirmation_threshold": 2,
            "window_size": 7,
        },
        "CREATIVE": {
            "confirmation_threshold": 4,
            "semantic_sim_threshold": 0.45,
        },
        "REM_DREAM": {
            "confirmation_threshold": 2,
            "decay_rate": 0.02,
            "semantic_sim_threshold": 0.40,
        },
        "DEFAULT": {},
    })


# =========================================================================
# 3.  Frozen output types
# =========================================================================

@dataclass(frozen=True)
class DetectedPattern:
    """A single identified pattern."""
    pattern_id:     str           = ""
    pattern_type:   PatternType   = PatternType.TEMPORAL
    status:         PatternStatus = PatternStatus.CANDIDATE
    fingerprint:    str           = ""
    confidence:     float         = 0.0
    occurrence_count: int         = 0
    first_seen_tick: int          = 0
    last_seen_tick:  int          = 0
    elements:       Tuple[str, ...] = ()
    period:         int           = 0     # For temporal patterns
    metadata:       Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatternIdentificationNeurochem:
    """Neurochemical output from pattern identification cycle."""
    da_delta:       float = 0.0   # Novel pattern discovery
    _5ht_delta:     float = 0.0   # Pattern consolidation
    ach_delta:      float = 0.0   # Focused matching
    gamma_boost:    float = 0.0   # Active pattern detection
    theta_boost:    float = 0.0   # Temporal pattern resonance

    def as_dict(self) -> Dict[str, float]:
        return {
            "da_delta":     self.da_delta,
            "_5ht_delta":   self._5ht_delta,
            "ach_delta":    self.ach_delta,
            "gamma_boost":  self.gamma_boost,
            "theta_boost":  self.theta_boost,
        }


@dataclass(frozen=True)
class PatternIdentificationResult:
    """Full output of one pattern identification cycle."""
    new_patterns:       List[DetectedPattern] = field(default_factory=list)
    confirmed_patterns: List[DetectedPattern] = field(default_factory=list)
    decayed_patterns:   List[str]             = field(default_factory=list)
    total_patterns:     int                   = 0
    by_type:            Dict[str, int]        = field(default_factory=dict)
    neurochem_signals:  PatternIdentificationNeurochem = field(
        default_factory=PatternIdentificationNeurochem)
    processing_time_ms: float                 = 0.0
    tick:               int                   = 0


# =========================================================================
# 4.  Internal mutable pattern record
# =========================================================================

@dataclass
class _PatternRecord:
    """Internal mutable pattern record."""
    pattern_id:      str
    pattern_type:    PatternType
    status:          PatternStatus
    fingerprint:     str
    confidence:      float
    occurrence_count: int
    first_seen_tick: int
    last_seen_tick:  int
    elements:        Tuple[str, ...]
    period:          int = 0
    metadata:        Dict[str, Any] = field(default_factory=dict)

    def to_detected(self) -> DetectedPattern:
        return DetectedPattern(
            pattern_id=self.pattern_id,
            pattern_type=self.pattern_type,
            status=self.status,
            fingerprint=self.fingerprint,
            confidence=self.confidence,
            occurrence_count=self.occurrence_count,
            first_seen_tick=self.first_seen_tick,
            last_seen_tick=self.last_seen_tick,
            elements=self.elements,
            period=self.period,
            metadata=dict(self.metadata),
        )


# =========================================================================
# 5.  Pure helper functions
# =========================================================================

def compute_fingerprint(elements: Tuple[str, ...]) -> str:
    """Hash fingerprint from an element tuple."""
    content = "|".join(elements)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def extract_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from a token list."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def extract_sliding_windows(
    tokens: List[str],
    window_size: int,
    step: int,
) -> List[Tuple[str, ...]]:
    """Extract sliding windows from token sequence."""
    if len(tokens) < window_size:
        return [tuple(tokens)] if tokens else []
    windows = []
    for i in range(0, len(tokens) - window_size + 1, step):
        windows.append(tuple(tokens[i:i + window_size]))
    return windows


def estimate_period(occurrence_ticks: List[int]) -> int:
    """Estimate temporal period from occurrence timestamps."""
    if len(occurrence_ticks) < 2:
        return 0
    sorted_ticks = sorted(occurrence_ticks)
    gaps = [sorted_ticks[i + 1] - sorted_ticks[i] for i in range(len(sorted_ticks) - 1)]
    if not gaps:
        return 0
    # Use median gap as period estimate
    gaps.sort()
    return gaps[len(gaps) // 2]


def compute_effective_decay(
    base_decay: float,
    sht: float,
    gaba: float,
    w_5ht: float,
    w_gaba: float,
) -> float:
    """NT-modulated decay rate. 5-HT stabilises, GABA accelerates."""
    return base_decay * max(0.1, 1.0 - w_5ht * sht) * (1.0 + w_gaba * gaba)


def compute_effective_confirmation(
    base_threshold: int,
    ach: float,
    w_ach: float,
) -> int:
    """NT-modulated confirmation threshold. ACh tightens."""
    return max(1, int(base_threshold * (1.0 + w_ach * ach)))


def compute_effective_window_size(
    base_size: int,
    ne: float,
    w_ne: float,
) -> int:
    """NT-modulated window size. NE broadens."""
    return max(2, int(base_size * (1.0 + w_ne * ne)))


def compute_pattern_neurochem(
    new_count: int,
    confirmed_count: int,
    temporal_count: int,
    total_active: int,
) -> PatternIdentificationNeurochem:
    """Compute NT output from pattern identification events."""
    return PatternIdentificationNeurochem(
        da_delta=min(0.3, new_count * 0.05),
        _5ht_delta=min(0.2, confirmed_count * 0.04),
        ach_delta=0.05 if total_active > 5 else 0.0,
        gamma_boost=min(0.2, total_active * 0.008),
        theta_boost=min(0.15, temporal_count * 0.03),
    )


def _simple_tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokenizer."""
    return [w.strip(".,!?;:\"'()[]{}") for w in text.lower().split()
            if w.strip(".,!?;:\"'()[]{}")]


def _token_cosine(a: List[str], b: List[str]) -> float:
    """Bag-of-words cosine similarity."""
    if not a or not b:
        return 0.0
    a_counts: Dict[str, int] = {}
    for t in a:
        a_counts[t] = a_counts.get(t, 0) + 1
    b_counts: Dict[str, int] = {}
    for t in b:
        b_counts[t] = b_counts.get(t, 0) + 1
    keys = set(a_counts) & set(b_counts)
    if not keys:
        return 0.0
    dot = sum(a_counts[k] * b_counts[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a_counts.values()))
    nb = math.sqrt(sum(v * v for v in b_counts.values()))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


# =========================================================================
# 6.  Engine class
# =========================================================================

class PatternIdentificationEngine:
    """
    Engine 19 -- Pattern Identification.

    Detects temporal, structural, semantic, and behavioral patterns
    in the input stream using sliding-window hash fingerprinting.

    API
    ---
    update_neurochem_state(state)  -- inject NT levels (Pattern A)
    process(input_data)            -- run detection cycle
    get_patterns(pattern_type)     -- query stored patterns
    get_status()                   -- introspection
    """

    engine_id = "pattern_identification_engine"
    cluster   = "pattern_analysis"

    def __init__(
        self,
        config: Optional[PatternIdentificationConfig] = None,
    ) -> None:
        self._cfg = config or PatternIdentificationConfig()

        # NT levels (Pattern A)
        self.da_level:   float = 0.5
        self._5ht_level: float = 0.5
        self.ach_level:  float = 0.5
        self.ne_level:   float = 0.5
        self.gaba_level: float = 0.5

        # State
        self._mode:    str = "DEFAULT"
        self._tick:    int = 0
        self._patterns: Dict[str, _PatternRecord] = {}  # fingerprint → record
        self._fingerprint_index: Dict[PatternType, Set[str]] = {
            pt: set() for pt in PatternType
        }
        # History for temporal detection
        self._temporal_history: List[Tuple[str, ...]] = []
        # Semantic history for semantic detection
        self._semantic_history: List[List[str]] = []
        # Behavioral history (intent labels)
        self._behavioral_history: List[str] = []

    # -----------------------------------------------------------------
    # Pattern A: NT State
    # -----------------------------------------------------------------

    def update_neurochem_state(self, nt_state: Dict[str, float]) -> None:
        self.da_level   = _clamp(nt_state.get("da",   self.da_level))
        self._5ht_level = _clamp(nt_state.get("5ht",  self._5ht_level))
        self.ach_level  = _clamp(nt_state.get("ach",  self.ach_level))
        self.ne_level   = _clamp(nt_state.get("ne",   self.ne_level))
        self.gaba_level = _clamp(nt_state.get("gaba", self.gaba_level))

    # -----------------------------------------------------------------
    # Mode
    # -----------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def _get_mode_override(self, key: str, default: Any) -> Any:
        overrides = self._cfg.mode_configs.get(self._mode, {})
        return overrides.get(key, default)

    # -----------------------------------------------------------------
    # Core: Pattern detection
    # -----------------------------------------------------------------

    def _detect_temporal(self, tokens: List[str]) -> List[_PatternRecord]:
        """Detect temporal patterns via sliding window fingerprinting."""
        eff_window = compute_effective_window_size(
            self._get_mode_override("window_size", self._cfg.window_size),
            self.ne_level, self._cfg.w_ne_window,
        )
        windows = extract_sliding_windows(tokens, eff_window, self._cfg.window_step)
        self._temporal_history.extend(windows)
        # Trim history
        max_hist = self._cfg.max_windows * 2
        if len(self._temporal_history) > max_hist:
            self._temporal_history = self._temporal_history[-max_hist:]

        new_patterns: List[_PatternRecord] = []

        for window in windows:
            fp = compute_fingerprint(window)
            if fp in self._patterns:
                rec = self._patterns[fp]
                rec.occurrence_count += 1
                rec.last_seen_tick = self._tick
                rec.confidence = min(1.0, rec.confidence + 0.1)
                # Estimate period
                if "occurrence_ticks" not in rec.metadata:
                    rec.metadata["occurrence_ticks"] = []
                rec.metadata["occurrence_ticks"].append(self._tick)
                rec.period = estimate_period(rec.metadata["occurrence_ticks"])
            else:
                rec = _PatternRecord(
                    pattern_id=fp,
                    pattern_type=PatternType.TEMPORAL,
                    status=PatternStatus.CANDIDATE,
                    fingerprint=fp,
                    confidence=self._cfg.initial_confidence,
                    occurrence_count=1,
                    first_seen_tick=self._tick,
                    last_seen_tick=self._tick,
                    elements=window,
                    metadata={"occurrence_ticks": [self._tick]},
                )
                self._patterns[fp] = rec
                self._fingerprint_index[PatternType.TEMPORAL].add(fp)
                new_patterns.append(rec)

        return new_patterns

    def _detect_structural(self, tokens: List[str]) -> List[_PatternRecord]:
        """Detect structural patterns via n-gram fingerprinting."""
        new_patterns: List[_PatternRecord] = []
        min_n = self._cfg.min_ngram_size
        max_n = min(self._cfg.max_ngram_size, len(tokens))

        for n in range(min_n, max_n + 1):
            ngrams = extract_ngrams(tokens, n)
            for ng in ngrams:
                fp = "s_" + compute_fingerprint(ng)
                if fp in self._patterns:
                    rec = self._patterns[fp]
                    rec.occurrence_count += 1
                    rec.last_seen_tick = self._tick
                    rec.confidence = min(1.0, rec.confidence + 0.08)
                else:
                    rec = _PatternRecord(
                        pattern_id=fp,
                        pattern_type=PatternType.STRUCTURAL,
                        status=PatternStatus.CANDIDATE,
                        fingerprint=fp,
                        confidence=self._cfg.initial_confidence,
                        occurrence_count=1,
                        first_seen_tick=self._tick,
                        last_seen_tick=self._tick,
                        elements=ng,
                    )
                    self._patterns[fp] = rec
                    self._fingerprint_index[PatternType.STRUCTURAL].add(fp)
                    new_patterns.append(rec)

        return new_patterns

    def _detect_semantic(self, tokens: List[str]) -> List[_PatternRecord]:
        """Detect semantic patterns via bag-of-words cosine to history."""
        new_patterns: List[_PatternRecord] = []
        sim_threshold = self._get_mode_override(
            "semantic_sim_threshold", self._cfg.semantic_sim_threshold,
        )

        for hist_tokens in self._semantic_history:
            sim = _token_cosine(tokens, hist_tokens)
            if sim >= sim_threshold:
                # Create a combined fingerprint
                combined = tuple(sorted(set(tokens) & set(hist_tokens)))
                if not combined:
                    continue
                fp = "sem_" + compute_fingerprint(combined)
                if fp in self._patterns:
                    rec = self._patterns[fp]
                    rec.occurrence_count += 1
                    rec.last_seen_tick = self._tick
                    rec.confidence = min(1.0, rec.confidence + 0.12)
                else:
                    rec = _PatternRecord(
                        pattern_id=fp,
                        pattern_type=PatternType.SEMANTIC,
                        status=PatternStatus.CANDIDATE,
                        fingerprint=fp,
                        confidence=self._cfg.initial_confidence + sim * 0.2,
                        occurrence_count=1,
                        first_seen_tick=self._tick,
                        last_seen_tick=self._tick,
                        elements=combined,
                        metadata={"similarity": sim},
                    )
                    self._patterns[fp] = rec
                    self._fingerprint_index[PatternType.SEMANTIC].add(fp)
                    new_patterns.append(rec)

        # Add current tokens to history
        self._semantic_history.append(tokens)
        max_hist = self._cfg.max_windows
        if len(self._semantic_history) > max_hist:
            self._semantic_history = self._semantic_history[-max_hist:]

        return new_patterns

    def _detect_behavioral(self, intent: str) -> List[_PatternRecord]:
        """Detect behavioral patterns from intent sequence."""
        new_patterns: List[_PatternRecord] = []
        if not intent:
            return new_patterns

        self._behavioral_history.append(intent)
        max_hist = self._cfg.max_windows
        if len(self._behavioral_history) > max_hist:
            self._behavioral_history = self._behavioral_history[-max_hist:]

        # Detect recurring intent sequences (bigrams and trigrams)
        for n in range(2, min(4, len(self._behavioral_history) + 1)):
            ngrams = extract_ngrams(self._behavioral_history, n)
            for ng in ngrams:
                fp = "beh_" + compute_fingerprint(ng)
                if fp in self._patterns:
                    rec = self._patterns[fp]
                    rec.occurrence_count += 1
                    rec.last_seen_tick = self._tick
                    rec.confidence = min(1.0, rec.confidence + 0.10)
                else:
                    rec = _PatternRecord(
                        pattern_id=fp,
                        pattern_type=PatternType.BEHAVIORAL,
                        status=PatternStatus.CANDIDATE,
                        fingerprint=fp,
                        confidence=self._cfg.initial_confidence,
                        occurrence_count=1,
                        first_seen_tick=self._tick,
                        last_seen_tick=self._tick,
                        elements=ng,
                    )
                    self._patterns[fp] = rec
                    self._fingerprint_index[PatternType.BEHAVIORAL].add(fp)
                    new_patterns.append(rec)

        return new_patterns

    # -----------------------------------------------------------------
    # Core: Pattern lifecycle (confirmation + decay)
    # -----------------------------------------------------------------

    def _promote_patterns(self) -> List[_PatternRecord]:
        """Promote candidates that meet confirmation threshold."""
        eff_threshold = compute_effective_confirmation(
            self._get_mode_override("confirmation_threshold", self._cfg.confirmation_threshold),
            self.ach_level, self._cfg.w_ach_threshold,
        )
        promoted = []
        for rec in self._patterns.values():
            if rec.status == PatternStatus.CANDIDATE and rec.occurrence_count >= eff_threshold:
                rec.status = PatternStatus.CONFIRMED
                promoted.append(rec)
        return promoted

    def _decay_patterns(self) -> List[str]:
        """Decay patterns not seen recently. Remove dead ones."""
        eff_decay = compute_effective_decay(
            self._get_mode_override("decay_rate", self._cfg.decay_rate),
            self._5ht_level, self.gaba_level,
            self._cfg.w_5ht_decay, self._cfg.w_gaba_decay,
        )
        removed: List[str] = []
        to_remove: List[str] = []

        for fp, rec in self._patterns.items():
            if rec.last_seen_tick < self._tick:
                ticks_absent = self._tick - rec.last_seen_tick
                rec.confidence -= eff_decay * ticks_absent
                if rec.confidence <= self._cfg.min_confidence:
                    to_remove.append(fp)
                elif rec.status == PatternStatus.CONFIRMED:
                    rec.status = PatternStatus.DECAYING

        for fp in to_remove:
            rec = self._patterns.pop(fp)
            self._fingerprint_index[rec.pattern_type].discard(fp)
            removed.append(fp)

        return removed

    def _enforce_capacity(self) -> None:
        """Remove lowest-confidence patterns if over capacity."""
        for pt in PatternType:
            fps = self._fingerprint_index[pt]
            if len(fps) <= self._cfg.max_patterns_per_type:
                continue
            # Sort by confidence, remove lowest
            records = [(fp, self._patterns[fp].confidence) for fp in fps if fp in self._patterns]
            records.sort(key=lambda x: x[1])
            excess = len(records) - self._cfg.max_patterns_per_type
            for fp, _ in records[:excess]:
                self._patterns.pop(fp, None)
                fps.discard(fp)

    # -----------------------------------------------------------------
    # Core: Full cycle
    # -----------------------------------------------------------------

    def detect(
        self,
        tokens: Optional[List[str]] = None,
        text: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> PatternIdentificationResult:
        """Run a full pattern detection cycle."""
        t0 = time.perf_counter()
        self._tick += 1

        # Tokenize if needed
        if tokens is None and text:
            tokens = _simple_tokenize(text)
        tokens = tokens or []

        # Detect across all dimensions
        new_temporal   = self._detect_temporal(tokens) if tokens else []
        new_structural = self._detect_structural(tokens) if tokens else []
        new_semantic   = self._detect_semantic(tokens) if tokens else []
        new_behavioral = self._detect_behavioral(intent or "") if intent else []

        all_new = new_temporal + new_structural + new_semantic + new_behavioral

        # Promote candidates
        promoted = self._promote_patterns()

        # Decay absent patterns
        removed = self._decay_patterns()

        # Enforce capacity
        self._enforce_capacity()

        # Count by type
        by_type: Dict[str, int] = {}
        for pt in PatternType:
            by_type[pt.value] = len(self._fingerprint_index[pt])

        total = sum(by_type.values())

        # Count temporal patterns for neurochem
        temporal_count = by_type.get(PatternType.TEMPORAL.value, 0)

        # Neurochem output
        signals = compute_pattern_neurochem(
            new_count=len(all_new),
            confirmed_count=len(promoted),
            temporal_count=temporal_count,
            total_active=total,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        return PatternIdentificationResult(
            new_patterns=[r.to_detected() for r in all_new],
            confirmed_patterns=[r.to_detected() for r in promoted],
            decayed_patterns=removed,
            total_patterns=total,
            by_type=by_type,
            neurochem_signals=signals,
            processing_time_ms=elapsed,
            tick=self._tick,
        )

    # -----------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------

    def get_patterns(
        self,
        pattern_type: Optional[PatternType] = None,
        status: Optional[PatternStatus] = None,
        min_confidence: float = 0.0,
    ) -> List[DetectedPattern]:
        """Query stored patterns with optional filters."""
        results = []
        for rec in self._patterns.values():
            if pattern_type and rec.pattern_type != pattern_type:
                continue
            if status and rec.status != status:
                continue
            if rec.confidence < min_confidence:
                continue
            results.append(rec.to_detected())
        return sorted(results, key=lambda p: p.confidence, reverse=True)

    def get_pattern_by_id(self, pattern_id: str) -> Optional[DetectedPattern]:
        """Look up a specific pattern by ID/fingerprint."""
        rec = self._patterns.get(pattern_id)
        return rec.to_detected() if rec else None

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

        # 3. Detect
        result = self.detect(
            tokens=input_data.get("tokens"),
            text=input_data.get("text"),
            intent=input_data.get("intent"),
        )

        return {
            "new_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "pattern_type": p.pattern_type.value,
                    "confidence": p.confidence,
                    "elements": list(p.elements),
                }
                for p in result.new_patterns
            ],
            "confirmed_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "pattern_type": p.pattern_type.value,
                    "confidence": p.confidence,
                    "occurrence_count": p.occurrence_count,
                }
                for p in result.confirmed_patterns
            ],
            "decayed_patterns": result.decayed_patterns,
            "total_patterns": result.total_patterns,
            "by_type": result.by_type,
            "neurochem_signals": result.neurochem_signals.as_dict(),
            "tick": result.tick,
            "processing_time_ms": result.processing_time_ms,
        }

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        by_type = {pt.value: len(fps) for pt, fps in self._fingerprint_index.items()}
        return {
            "engine_id":   self.engine_id,
            "cluster":     self.cluster,
            "mode":        self._mode,
            "tick":        self._tick,
            "total_patterns": sum(by_type.values()),
            "by_type":     by_type,
            "nt_levels": {
                "da":   self.da_level,
                "5ht":  self._5ht_level,
                "ach":  self.ach_level,
                "ne":   self.ne_level,
                "gaba": self.gaba_level,
            },
        }

    def __repr__(self) -> str:
        total = sum(len(fps) for fps in self._fingerprint_index.values())
        return f"PatternIdentificationEngine(mode={self._mode}, patterns={total})"
