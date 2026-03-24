"""
Engine 20 -- Pattern Comparison Engine  (``pattern_comparison_engine``)
======================================================================

Compares input patterns (typically from Engine 19 — Pattern Identification)
against a stored library of reference templates.  Three complementary
similarity algorithms produce a composite match score for each
input--template pair:

  1. **Jaccard similarity**  — element-set overlap (|A & B| / |A | B|)
  2. **Weighted cosine**     — confidence-weighted bag-of-elements inner product
  3. **Structural alignment** — positional order agreement (longest common
     subsequence length / max length)

The engine tracks a *template library* — a mutable collection of labelled
reference patterns with confidence scores and decay dynamics.  Templates
that are matched frequently grow in confidence; those that are not matched
decay toward a floor and are eventually evicted.

Novelty detection
-----------------
An input pattern whose best composite score against all templates falls
*below* the match threshold is classified as **novel**.  Novel patterns
are surfaced to downstream engines (e.g. E9 AtomSpace) and produce a
phasic DA spike via the neurochemical coupling.

Similarity ranking
------------------
For every input pattern, the engine returns a ranked list of the top-*k*
most similar templates, ordered by composite score.  The ranking respects
mode-dependent thresholds and NT modulation.

Neurochemical coupling (Appendix S2--S9)
-----------------------------------------
  ACh  — tightens match threshold (higher ACh requires closer matches)
  CB1  — relaxes match threshold (higher CB1 allows looser matches)
  DA   — boosts novelty reward for unmatched patterns
  5-HT — stabilises template library (reduces template decay rate)
  NE   — broadens search scope (compares against more templates per cycle)

Operational modes
-----------------
  DEFAULT    — balanced threshold and search
  ANALYTICAL — stricter threshold, larger k
  CREATIVE   — relaxed threshold, low-confidence templates still considered
  REM_DREAM  — very loose matching, slow decay, wide search

Interacts with
--------------
  E19 (Pattern Identification) — upstream producer of input patterns
  E9  (AtomSpace)              — confirmed templates written as ConceptNodes
  E10 (PLN)                    — inference over template relationships
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from zados.cognitive_engines.constants import _clamp


# =========================================================================
# 1.  Configuration
# =========================================================================

@dataclass(frozen=True)
class PatternComparisonConfig:
    """Immutable configuration for the Pattern Comparison Engine."""

    # --- Match scoring weights ---
    jaccard_weight:     float = 0.35
    cosine_weight:      float = 0.35
    alignment_weight:   float = 0.30

    # --- Match threshold (composite score must exceed to count) ---
    match_threshold:    float = 0.40

    # --- Template library ---
    max_templates:      int   = 500
    template_decay_rate: float = 0.03   # Confidence decay per tick without match
    template_min_confidence: float = 0.05  # Below this → evict
    template_initial_confidence: float = 0.50
    template_match_boost: float = 0.08  # Confidence boost when template is matched

    # --- Similarity ranking ---
    top_k:              int   = 5       # Top-k templates returned per input

    # --- Novelty detection ---
    novelty_da_spike:   float = 0.15    # DA delta for novel pattern

    # --- NT modulation weights ---
    w_ach_threshold:    float = 0.25    # ACh tightens match threshold
    w_cb1_threshold:    float = 0.20    # CB1 relaxes match threshold
    w_da_novelty:       float = 0.30    # DA boosts novelty score
    w_5ht_decay:        float = 0.40    # 5-HT reduces template decay
    w_ne_search:        float = 0.25    # NE broadens search (top-k)

    # --- Neurochem output scaling ---
    nc_da_scale:        float = 0.05    # DA per novel pattern
    nc_ach_scale:       float = 0.04    # ACh per focused match cycle
    nc_5ht_scale:       float = 0.03    # 5-HT per stable template interaction
    nc_gamma_scale:     float = 0.008   # Gamma boost per active comparison
    nc_theta_scale:     float = 0.006   # Theta boost for temporal template echo

    # --- Mode overrides ---
    mode_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "DEFAULT": {},
        "ANALYTICAL": {
            "match_threshold": 0.55,
            "top_k": 10,
            "template_decay_rate": 0.04,
        },
        "CREATIVE": {
            "match_threshold": 0.25,
            "top_k": 8,
            "template_min_confidence": 0.02,
        },
        "REM_DREAM": {
            "match_threshold": 0.20,
            "top_k": 15,
            "template_decay_rate": 0.01,
            "template_min_confidence": 0.02,
        },
    })


# =========================================================================
# 2.  Frozen output types
# =========================================================================

@dataclass(frozen=True)
class PatternMatch:
    """Result of comparing one input pattern against one template."""
    input_pattern_id:  str   = ""
    template_id:       str   = ""
    template_label:    str   = ""
    jaccard_score:     float = 0.0
    cosine_score:      float = 0.0
    alignment_score:   float = 0.0
    composite_score:   float = 0.0
    is_novel:          bool  = False


@dataclass(frozen=True)
class PatternComparisonNeurochem:
    """
    Neurochemical output from one comparison cycle.

    Notation (Appendix S2--S3, S7):
        da_delta    -> Delta C_DA(t)       : novelty reward for unmatched patterns
        ach_delta   -> Delta C_ACh(t)      : attentional gating during match scan
        _5ht_delta  -> Delta C_5HT(t)      : template stabilisation signal
        gamma_boost -> Delta phi_gamma(t)  : active comparison band enhancement
        theta_boost -> Delta phi_theta(t)  : temporal resonance echo
    """
    da_delta:     float = 0.0
    ach_delta:    float = 0.0
    _5ht_delta:   float = 0.0
    gamma_boost:  float = 0.0
    theta_boost:  float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "da_delta":     self.da_delta,
            "ach_delta":    self.ach_delta,
            "_5ht_delta":   self._5ht_delta,
            "gamma_boost":  self.gamma_boost,
            "theta_boost":  self.theta_boost,
        }


@dataclass(frozen=True)
class PatternComparisonResult:
    """Full output of one pattern comparison cycle."""
    matches:            List[PatternMatch]  = field(default_factory=list)
    novel_patterns:     List[str]           = field(default_factory=list)
    total_compared:     int                 = 0
    total_matched:      int                 = 0
    total_novel:        int                 = 0
    mean_similarity:    float               = 0.0
    top_matches:        List[PatternMatch]  = field(default_factory=list)
    neurochem_signals:  PatternComparisonNeurochem = field(
        default_factory=PatternComparisonNeurochem)
    templates_active:   int                 = 0
    templates_decayed:  int                 = 0
    tick:               int                 = 0
    processing_time_ms: float               = 0.0
    metadata:           Dict[str, Any]      = field(default_factory=dict)


# =========================================================================
# 3.  Internal mutable template record
# =========================================================================

@dataclass
class _TemplateRecord:
    """
    Internal mutable record for a reference template in the library.

    Stored in the engine's ``_templates`` dict keyed by ``template_id``.
    """
    template_id:       str
    label:             str
    elements:          Tuple[str, ...]
    confidence:        float
    last_matched_tick: int
    match_count:       int
    created_tick:      int
    metadata:          Dict[str, Any] = field(default_factory=dict)

    # Optional: per-element confidence scores for weighted cosine
    element_weights:   Optional[Tuple[float, ...]] = None


# =========================================================================
# 4.  Pure helper functions — similarity algorithms
# =========================================================================

def compute_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """
    Jaccard similarity coefficient:  |A & B| / |A | B|.

    Returns 0.0 when both sequences are empty (by convention).
    """
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def compute_weighted_cosine(
    a_elements: Sequence[str],
    a_weights: Optional[Sequence[float]],
    b_elements: Sequence[str],
    b_weights: Optional[Sequence[float]],
) -> float:
    """
    Weighted bag-of-elements cosine similarity.

    Each element receives a weight (default 1.0).  The bags are formed by
    summing weights for each unique element, then cosine similarity is
    computed over the resulting weight vectors.
    """
    if not a_elements or not b_elements:
        return 0.0

    # Build weighted bags
    def _build_bag(
        elems: Sequence[str],
        weights: Optional[Sequence[float]],
    ) -> Dict[str, float]:
        bag: Dict[str, float] = {}
        for i, elem in enumerate(elems):
            w = weights[i] if weights and i < len(weights) else 1.0
            bag[elem] = bag.get(elem, 0.0) + w
        return bag

    bag_a = _build_bag(a_elements, a_weights)
    bag_b = _build_bag(b_elements, b_weights)

    keys = set(bag_a) & set(bag_b)
    if not keys:
        return 0.0

    dot = sum(bag_a[k] * bag_b[k] for k in keys)
    norm_a = math.sqrt(sum(v * v for v in bag_a.values()))
    norm_b = math.sqrt(sum(v * v for v in bag_b.values()))

    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0

    return dot / (norm_a * norm_b)


def compute_structural_alignment(a: Sequence[str], b: Sequence[str]) -> float:
    """
    Structural alignment score based on Longest Common Subsequence (LCS).

    Score = LCS_length / max(len(a), len(b)).  Captures positional order
    agreement beyond simple set overlap.
    """
    if not a or not b:
        return 0.0

    n, m = len(a), len(b)
    # DP table for LCS length (space-optimised to two rows)
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (m + 1)

    lcs_len = prev[m]
    return lcs_len / max(n, m)


def compute_composite_score(
    jaccard: float,
    cosine: float,
    alignment: float,
    cfg: PatternComparisonConfig,
) -> float:
    """Weighted fusion of the three similarity scores → [0, 1]."""
    score = (
        cfg.jaccard_weight * jaccard
        + cfg.cosine_weight * cosine
        + cfg.alignment_weight * alignment
    )
    return _clamp(score)


def compute_effective_threshold(
    base_threshold: float,
    ach: float,
    cb1: float,
    w_ach: float,
    w_cb1: float,
) -> float:
    """
    NT-modulated match threshold.

    ACh tightens (increases threshold), CB1 relaxes (decreases threshold).
    Result is clamped to [0.05, 0.95] to avoid extreme saturation.
    """
    modulated = base_threshold * (1.0 + w_ach * ach) * (1.0 - w_cb1 * cb1)
    return _clamp(modulated, lo=0.05, hi=0.95)


def compute_effective_top_k(
    base_k: int,
    ne: float,
    w_ne: float,
) -> int:
    """NE-modulated top-k.  Higher NE → broader search."""
    return max(1, int(base_k * (1.0 + w_ne * ne)))


def compute_effective_decay(
    base_decay: float,
    sht: float,
    w_5ht: float,
) -> float:
    """5-HT-modulated decay rate.  Higher 5-HT → lower decay (stabilisation)."""
    return base_decay * max(0.05, 1.0 - w_5ht * sht)


def compute_comparison_neurochem(
    novel_count: int,
    matched_count: int,
    total_compared: int,
    temporal_template_hits: int,
    da_level: float,
    cfg: PatternComparisonConfig,
) -> PatternComparisonNeurochem:
    """
    Compute neurochemical output signals from one comparison cycle.

    DA  — phasic spike per novel pattern, boosted by current DA level
    ACh — sustained attention proportional to number of comparisons
    5-HT — stabilisation signal when templates are successfully matched
    Gamma — active comparison band boost
    Theta — temporal resonance when temporal templates are echoed
    """
    # DA: novelty reward, boosted by existing DA (positive feedback)
    da_base = novel_count * cfg.nc_da_scale
    da_boost = da_base * (1.0 + cfg.w_da_novelty * da_level)
    da_delta = _clamp(da_boost, lo=0.0, hi=0.50)

    # ACh: sustained attention during comparison scan
    ach_delta = _clamp(min(0.30, total_compared * cfg.nc_ach_scale), lo=0.0)

    # 5-HT: stabilisation from successful matches
    _5ht_delta = _clamp(min(0.20, matched_count * cfg.nc_5ht_scale), lo=0.0)

    # Gamma: active comparison integration
    gamma_boost = _clamp(min(0.20, total_compared * cfg.nc_gamma_scale), lo=0.0)

    # Theta: temporal template resonance
    theta_boost = _clamp(min(0.15, temporal_template_hits * cfg.nc_theta_scale), lo=0.0)

    return PatternComparisonNeurochem(
        da_delta=da_delta,
        ach_delta=ach_delta,
        _5ht_delta=_5ht_delta,
        gamma_boost=gamma_boost,
        theta_boost=theta_boost,
    )


# =========================================================================
# 5.  Engine class
# =========================================================================

class PatternComparisonEngine:
    """
    Engine 20 -- Pattern Comparison.

    Compares input patterns against a stored template library using three
    similarity algorithms (Jaccard, weighted cosine, structural alignment)
    and produces composite match scores, novelty flags, and neurochemical
    coupling signals.

    API
    ---
    update_neurochem_state(state)    -- inject NT levels (Pattern A)
    add_template(...)                -- register a reference template
    remove_template(template_id)     -- remove a template
    compare(patterns)                -- run comparison cycle
    process(input_data)              -- pipeline entry point
    get_status()                     -- introspection
    """

    engine_id = "pattern_comparison_engine"
    cluster   = "pattern_analysis"

    def __init__(
        self,
        config: Optional[PatternComparisonConfig] = None,
    ) -> None:
        self._cfg = config or PatternComparisonConfig()

        # NT levels (Pattern A)
        self.da_level:    float = 0.5
        self._5ht_level:  float = 0.5
        self.ach_level:   float = 0.5
        self.ne_level:    float = 0.5
        self.gaba_level:  float = 0.5
        self.cb1_level:   float = 0.5

        # State
        self._mode:      str = "DEFAULT"
        self._tick:      int = 0
        self._templates: Dict[str, _TemplateRecord] = {}

        # Cumulative statistics
        self._total_comparisons: int = 0
        self._total_novel:       int = 0
        self._total_matched:     int = 0

    # -----------------------------------------------------------------
    # Pattern A: NT State
    # -----------------------------------------------------------------

    def update_neurochem_state(self, nt_state: Dict[str, float]) -> None:
        """Inject current neurochemical levels for bidirectional feedback."""
        self.da_level    = _clamp(nt_state.get("da",    self.da_level))
        self._5ht_level  = _clamp(nt_state.get("5ht",   self._5ht_level))
        self.ach_level   = _clamp(nt_state.get("ach",   self.ach_level))
        self.ne_level    = _clamp(nt_state.get("ne",    self.ne_level))
        self.gaba_level  = _clamp(nt_state.get("gaba",  self.gaba_level))
        self.cb1_level   = _clamp(nt_state.get("cb1",   self.cb1_level))

    # -----------------------------------------------------------------
    # Mode
    # -----------------------------------------------------------------

    def set_mode(self, mode: str) -> None:
        """Set operational mode (DEFAULT, ANALYTICAL, CREATIVE, REM_DREAM)."""
        self._mode = mode

    def _get_mode_override(self, key: str, default: Any) -> Any:
        """Return mode-specific parameter override, falling back to *default*."""
        overrides = self._cfg.mode_configs.get(self._mode, {})
        return overrides.get(key, default)

    # -----------------------------------------------------------------
    # Template library management
    # -----------------------------------------------------------------

    def add_template(
        self,
        template_id: str,
        elements: Sequence[str],
        label: str = "",
        confidence: Optional[float] = None,
        element_weights: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> _TemplateRecord:
        """
        Register a reference template in the library.

        Parameters
        ----------
        template_id : str
            Unique identifier for this template (caller-assigned or UUID).
        elements : Sequence[str]
            Ordered element tuple defining the template pattern.
        label : str, optional
            Human-readable label for the template.
        confidence : float, optional
            Initial confidence.  Defaults to ``template_initial_confidence``.
        element_weights : Sequence[float], optional
            Per-element weights for the weighted cosine similarity.
        metadata : dict, optional
            Arbitrary metadata attached to the template.

        Returns
        -------
        _TemplateRecord
            The created (or updated) template record.
        """
        conf = confidence if confidence is not None else self._cfg.template_initial_confidence
        conf = _clamp(conf)

        elem_tuple = tuple(elements)
        w_tuple = tuple(element_weights) if element_weights else None

        if template_id in self._templates:
            # Update existing
            rec = self._templates[template_id]
            rec.elements = elem_tuple
            rec.label = label or rec.label
            rec.confidence = conf
            rec.element_weights = w_tuple
            if metadata:
                rec.metadata.update(metadata)
            return rec

        rec = _TemplateRecord(
            template_id=template_id,
            label=label,
            elements=elem_tuple,
            confidence=conf,
            last_matched_tick=self._tick,
            match_count=0,
            created_tick=self._tick,
            element_weights=w_tuple,
            metadata=metadata or {},
        )
        self._templates[template_id] = rec

        # Enforce capacity
        self._enforce_capacity()

        return rec

    def remove_template(self, template_id: str) -> bool:
        """Remove a template from the library.  Returns True if found."""
        return self._templates.pop(template_id, None) is not None

    def get_template(self, template_id: str) -> Optional[_TemplateRecord]:
        """Look up a template by ID.  Returns None if not found."""
        return self._templates.get(template_id)

    def get_templates(
        self,
        min_confidence: float = 0.0,
        label_prefix: Optional[str] = None,
    ) -> List[_TemplateRecord]:
        """Return templates matching optional filters, sorted by confidence."""
        results = []
        for rec in self._templates.values():
            if rec.confidence < min_confidence:
                continue
            if label_prefix and not rec.label.startswith(label_prefix):
                continue
            results.append(rec)
        return sorted(results, key=lambda r: r.confidence, reverse=True)

    def _enforce_capacity(self) -> None:
        """Evict lowest-confidence templates if library exceeds capacity."""
        if len(self._templates) <= self._cfg.max_templates:
            return
        records = sorted(self._templates.values(), key=lambda r: r.confidence)
        excess = len(self._templates) - self._cfg.max_templates
        for rec in records[:excess]:
            self._templates.pop(rec.template_id, None)

    # -----------------------------------------------------------------
    # Core: compare input patterns against templates
    # -----------------------------------------------------------------

    def compare(
        self,
        patterns: List[Dict[str, Any]],
    ) -> PatternComparisonResult:
        """
        Compare a list of input patterns against the template library.

        Each pattern dict must contain:
          - ``"pattern_id"`` : str
          - ``"elements"``   : List[str]
        Optional:
          - ``"confidence"`` : float  (used as element weight fallback)
          - ``"element_weights"`` : List[float]

        Returns a PatternComparisonResult with matches, novel patterns,
        similarity rankings, and neurochemical signals.
        """
        t0 = time.perf_counter()
        self._tick += 1

        # Resolve mode-dependent parameters
        eff_threshold = compute_effective_threshold(
            self._get_mode_override("match_threshold", self._cfg.match_threshold),
            self.ach_level,
            self.cb1_level,
            self._cfg.w_ach_threshold,
            self._cfg.w_cb1_threshold,
        )
        eff_k = compute_effective_top_k(
            self._get_mode_override("top_k", self._cfg.top_k),
            self.ne_level,
            self._cfg.w_ne_search,
        )
        eff_decay = compute_effective_decay(
            self._get_mode_override("template_decay_rate", self._cfg.template_decay_rate),
            self._5ht_level,
            self._cfg.w_5ht_decay,
        )
        min_conf = self._get_mode_override(
            "template_min_confidence", self._cfg.template_min_confidence,
        )

        all_matches: List[PatternMatch] = []
        novel_pattern_ids: List[str] = []
        composite_scores: List[float] = []
        temporal_template_hits = 0

        for pat in patterns:
            pat_id = pat.get("pattern_id", str(uuid.uuid4()))
            pat_elements = tuple(pat.get("elements", []))
            pat_weights = pat.get("element_weights")
            pat_conf = pat.get("confidence", 1.0)

            if not pat_elements:
                continue

            # If no explicit element weights, use uniform scaled by confidence
            if pat_weights is None:
                pat_weights_tuple: Optional[Tuple[float, ...]] = None
            else:
                pat_weights_tuple = tuple(pat_weights)

            best_score = 0.0
            per_template_matches: List[PatternMatch] = []

            for tmpl in self._templates.values():
                # Compute three similarity scores
                jaccard = compute_jaccard(pat_elements, tmpl.elements)
                cosine = compute_weighted_cosine(
                    pat_elements, pat_weights_tuple,
                    tmpl.elements, tmpl.element_weights,
                )
                alignment = compute_structural_alignment(pat_elements, tmpl.elements)
                composite = compute_composite_score(jaccard, cosine, alignment, self._cfg)

                match = PatternMatch(
                    input_pattern_id=pat_id,
                    template_id=tmpl.template_id,
                    template_label=tmpl.label,
                    jaccard_score=round(jaccard, 4),
                    cosine_score=round(cosine, 4),
                    alignment_score=round(alignment, 4),
                    composite_score=round(composite, 4),
                    is_novel=False,
                )
                per_template_matches.append(match)

                if composite > best_score:
                    best_score = composite

                # If match passes threshold, update template stats
                if composite >= eff_threshold:
                    tmpl.match_count += 1
                    tmpl.last_matched_tick = self._tick
                    tmpl.confidence = _clamp(
                        tmpl.confidence + self._cfg.template_match_boost
                    )
                    # Track temporal template hits
                    if tmpl.metadata.get("pattern_type") == "temporal":
                        temporal_template_hits += 1

            # Sort by composite score descending, take top-k
            per_template_matches.sort(
                key=lambda m: m.composite_score, reverse=True,
            )
            top_k_matches = per_template_matches[:eff_k]

            # Novelty detection
            is_novel = best_score < eff_threshold or len(self._templates) == 0
            if is_novel:
                novel_pattern_ids.append(pat_id)
                # Mark top match (if any) as novel context
                if top_k_matches:
                    # Replace with novel-flagged version
                    top_match = top_k_matches[0]
                    top_k_matches[0] = PatternMatch(
                        input_pattern_id=top_match.input_pattern_id,
                        template_id=top_match.template_id,
                        template_label=top_match.template_label,
                        jaccard_score=top_match.jaccard_score,
                        cosine_score=top_match.cosine_score,
                        alignment_score=top_match.alignment_score,
                        composite_score=top_match.composite_score,
                        is_novel=True,
                    )
                else:
                    # No templates at all — still create a novelty marker
                    top_k_matches = [PatternMatch(
                        input_pattern_id=pat_id,
                        template_id="",
                        template_label="",
                        composite_score=0.0,
                        is_novel=True,
                    )]

            all_matches.extend(top_k_matches)
            if per_template_matches:
                composite_scores.append(best_score)

        # Template decay
        decayed_count = self._decay_templates(eff_decay, min_conf)

        # Aggregate statistics
        total_compared = len([p for p in patterns if p.get("elements")])
        total_matched = total_compared - len(novel_pattern_ids)
        mean_sim = (
            sum(composite_scores) / len(composite_scores)
            if composite_scores else 0.0
        )

        # Update cumulative stats
        self._total_comparisons += total_compared
        self._total_novel += len(novel_pattern_ids)
        self._total_matched += total_matched

        # Select overall top matches across all input patterns
        top_overall = sorted(
            all_matches, key=lambda m: m.composite_score, reverse=True,
        )[:eff_k]

        # Neurochemical output
        neurochem = compute_comparison_neurochem(
            novel_count=len(novel_pattern_ids),
            matched_count=total_matched,
            total_compared=total_compared,
            temporal_template_hits=temporal_template_hits,
            da_level=self.da_level,
            cfg=self._cfg,
        )

        elapsed = (time.perf_counter() - t0) * 1000.0

        return PatternComparisonResult(
            matches=all_matches,
            novel_patterns=novel_pattern_ids,
            total_compared=total_compared,
            total_matched=total_matched,
            total_novel=len(novel_pattern_ids),
            mean_similarity=round(mean_sim, 4),
            top_matches=top_overall,
            neurochem_signals=neurochem,
            templates_active=len(self._templates),
            templates_decayed=decayed_count,
            tick=self._tick,
            processing_time_ms=round(elapsed, 3),
            metadata={
                "mode": self._mode,
                "effective_threshold": round(eff_threshold, 4),
                "effective_top_k": eff_k,
            },
        )

    # -----------------------------------------------------------------
    # Template decay
    # -----------------------------------------------------------------

    def _decay_templates(self, decay_rate: float, min_confidence: float) -> int:
        """
        Apply confidence decay to templates not matched on this tick.

        Templates whose confidence falls below *min_confidence* are evicted.
        Returns the number of evicted templates.
        """
        to_remove: List[str] = []

        for tmpl_id, rec in self._templates.items():
            if rec.last_matched_tick < self._tick:
                ticks_absent = self._tick - rec.last_matched_tick
                rec.confidence -= decay_rate * ticks_absent
                if rec.confidence < min_confidence:
                    to_remove.append(tmpl_id)

        for tmpl_id in to_remove:
            self._templates.pop(tmpl_id, None)

        return len(to_remove)

    # -----------------------------------------------------------------
    # process() — Pipeline Entry Point
    # -----------------------------------------------------------------

    def process(self, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Pipeline entry point.

        Accepts a dict with optional keys:

          ``"nt_state"``   : Dict[str, float]    — NT levels (Pattern A)
          ``"mode"``       : str                  — operational mode
          ``"patterns"``   : List[Dict]           — input patterns to compare
          ``"templates"``  : List[Dict]           — templates to add before comparing

        Each pattern dict should contain ``"pattern_id"`` and ``"elements"``.
        Each template dict should contain ``"template_id"`` and ``"elements"``,
        and optionally ``"label"``, ``"confidence"``, ``"element_weights"``.

        Returns a dict with match results, novelty flags, statistics,
        and neurochemical signals.
        """
        input_data = input_data or {}

        # 1. NT state
        if "nt_state" in input_data:
            self.update_neurochem_state(input_data["nt_state"])

        # 2. Mode
        if "mode" in input_data:
            self.set_mode(input_data["mode"])

        # 3. Add templates if provided
        for tmpl in input_data.get("templates", []):
            self.add_template(
                template_id=tmpl.get("template_id", str(uuid.uuid4())),
                elements=tmpl.get("elements", []),
                label=tmpl.get("label", ""),
                confidence=tmpl.get("confidence"),
                element_weights=tmpl.get("element_weights"),
                metadata=tmpl.get("metadata"),
            )

        # 4. Run comparison
        patterns = input_data.get("patterns", [])
        result = self.compare(patterns)

        return {
            "matches": [
                {
                    "input_pattern_id":  m.input_pattern_id,
                    "template_id":       m.template_id,
                    "template_label":    m.template_label,
                    "jaccard_score":     m.jaccard_score,
                    "cosine_score":      m.cosine_score,
                    "alignment_score":   m.alignment_score,
                    "composite_score":   m.composite_score,
                    "is_novel":          m.is_novel,
                }
                for m in result.matches
            ],
            "novel_patterns":   result.novel_patterns,
            "total_compared":   result.total_compared,
            "total_matched":    result.total_matched,
            "total_novel":      result.total_novel,
            "mean_similarity":  result.mean_similarity,
            "top_matches": [
                {
                    "input_pattern_id":  m.input_pattern_id,
                    "template_id":       m.template_id,
                    "composite_score":   m.composite_score,
                    "is_novel":          m.is_novel,
                }
                for m in result.top_matches
            ],
            "neurochem_signals": result.neurochem_signals.as_dict(),
            "templates_active":  result.templates_active,
            "templates_decayed": result.templates_decayed,
            "tick":              result.tick,
            "processing_time_ms": result.processing_time_ms,
        }

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return engine status for monitoring and diagnostics."""
        return {
            "engine_id":           self.engine_id,
            "cluster":             self.cluster,
            "mode":                self._mode,
            "tick":                self._tick,
            "templates_active":    len(self._templates),
            "total_comparisons":   self._total_comparisons,
            "total_novel":         self._total_novel,
            "total_matched":       self._total_matched,
            "nt_levels": {
                "da":   self.da_level,
                "5ht":  self._5ht_level,
                "ach":  self.ach_level,
                "ne":   self.ne_level,
                "gaba": self.gaba_level,
                "cb1":  self.cb1_level,
            },
        }

    def __repr__(self) -> str:
        return (
            f"PatternComparisonEngine("
            f"mode={self._mode}, "
            f"templates={len(self._templates)}, "
            f"tick={self._tick})"
        )
