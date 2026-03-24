"""
Engine 18 -- Data Analysis Engine  (``data_analysis_engine``)
==============================================================
Extracts structured entity-relation-entity triples, dependency structures,
and co-occurrence relationships from text input flowing through the pipeline.

This engine transforms raw or pre-processed text into a relational graph
representation suitable for downstream reasoning engines (PLN, AtomSpace,
Simulation Brain, etc.).

Extraction pipeline (4 stages):
  * **Stage 1 -- Entity Extraction**: rule-based NER proxy using capitalized
    words, quoted terms, tagged tokens, and pronoun anchoring.  Produces
    ``ExtractedEntity`` instances with confidence and type classification.
  * **Stage 2 -- Relation Extraction**: verb-based predicate detection,
    preposition links, copula patterns, and causal connectors.  Produces
    ``ExtractedRelation`` triples (subject, predicate, object).
  * **Stage 3 -- Co-occurrence Matrix**: sliding-window co-occurrence
    counting across entity pairs, producing ``CoOccurrence`` records.
  * **Stage 4 -- Dependency Depth Estimation**: clause-nesting depth via
    subordination marker counting and parenthetical tracking.

Neurochemical coupling (bidirectional):
  ACh  -- deepens analysis (more relation types extracted, tighter matching)
  NE   -- broadens entity scope (lower entity detection threshold)
  DA   -- rewards novel entity/relation discovery
  5-HT -- stabilises existing analysis (reduces noise, raises thresholds)
  GABA -- suppresses weak relations (raises relation confidence floor)

Output types (all frozen dataclasses):
  ExtractedEntity   -- entity_id, text, entity_type, confidence, metadata
  ExtractedRelation -- relation_id, subject_id, predicate, object_id, confidence
  CoOccurrence      -- entity_a, entity_b, count, window_size
  DataAnalysisNeurochem -- da_delta, ach_delta, ne_delta, _5ht_delta, gamma_boost
  DataAnalysisResult -- entities, relations, co_occurrences, dependency_depth, ...

Operational modes (mapped from OperationalMode):
  NORMAL     -- balanced thresholds
  DEV        -- lowered entity threshold, verbose metadata
  LEARNING   -- amplified novelty reward, moderate thresholds
  REFLECTIVE -- tighter matching, higher confidence floors
  REM_NORMAL -- standard with mildly relaxed entity scope
  REM_DREAM  -- creative: very low thresholds, maximal entity/relation yield

Usage
-----
>>> from zados.cognitive_engines.py_engines.data_analysis_engine import (
...     DataAnalysisEngine, DataAnalysisConfig, DataAnalysisInput,
...     DataAnalysisResult,
... )
>>> engine = DataAnalysisEngine()
>>> inp = DataAnalysisInput(raw_text="Alice told Bob that the Server crashed.")
>>> result = engine.process(inp)
>>> len(result.entities)
3
"""

from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from zados.cognitive_engines.constants import _clamp
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Enums
# =====================================================================


class EntityType(str, Enum):
    """Classification of extracted entities."""
    PERSON       = "person"
    ORGANIZATION = "organization"
    LOCATION     = "location"
    CONCEPT      = "concept"
    ARTIFACT     = "artifact"
    EVENT        = "event"
    QUANTITY     = "quantity"
    TEMPORAL     = "temporal"
    UNKNOWN      = "unknown"


class RelationType(str, Enum):
    """Classification of extracted relations."""
    ACTION        = "action"         # subject verb object
    ATTRIBUTE     = "attribute"      # subject is/has property
    CAUSAL        = "causal"         # because, therefore, causes
    SPATIAL       = "spatial"        # in, on, at, near, above
    TEMPORAL_REL  = "temporal_rel"   # before, after, during, when
    POSSESSION    = "possession"     # has, owns, belongs
    COMPARISON    = "comparison"     # like, unlike, greater, less
    PART_WHOLE    = "part_whole"     # part of, contains, includes
    COPULA        = "copula"         # is, are, was, were (identity)
    PREPOSITION   = "preposition"    # generic prepositional link


class ConfidenceTier(str, Enum):
    """Discrete confidence classification."""
    HIGH    = "high"       # >= 0.75
    MEDIUM  = "medium"     # >= 0.45
    LOW     = "low"        # >= 0.20
    TRACE   = "trace"      # < 0.20


# =====================================================================
# Constants -- Linguistic patterns
# =====================================================================

# Common verbs for relation extraction (predicate detection)
_ACTION_VERBS: List[str] = [
    "told", "said", "gave", "sent", "asked", "showed", "taught", "built",
    "created", "destroyed", "moved", "placed", "removed", "changed",
    "started", "stopped", "helped", "called", "wrote", "read", "found",
    "lost", "hit", "crashed", "killed", "saved", "broke", "fixed",
    "opened", "closed", "ran", "walked", "flew", "drove", "carried",
    "pulled", "pushed", "threw", "caught", "made", "used", "took",
    "brought", "bought", "sold", "paid", "earned", "saw", "heard",
    "felt", "thought", "knew", "believed", "wanted", "needed", "liked",
    "loved", "hated", "feared", "expected", "planned", "decided",
    "chose", "accepted", "rejected", "agreed", "disagreed", "argued",
    "discussed", "explained", "described", "mentioned", "suggested",
    "recommended", "proposed", "announced", "reported", "claimed",
    "denied", "confirmed", "revealed", "discovered", "invented",
    "designed", "implemented", "deployed", "tested", "analyzed",
    "evaluated", "compared", "measured", "observed", "detected",
    "processed", "computed", "generated", "produced", "consumed",
    "contains", "includes", "requires", "provides", "supports",
    "enables", "prevents", "causes", "affects", "influences",
]

_COPULA_VERBS: FrozenSet[str] = frozenset({
    "is", "are", "was", "were", "be", "been", "being",
    "becomes", "became", "seems", "appears", "remains",
})

_CAUSAL_CONNECTORS: FrozenSet[str] = frozenset({
    "because", "therefore", "consequently", "thus", "hence",
    "causes", "caused", "causing", "results", "resulted",
    "leads", "led", "leading", "due", "since", "so",
})

_TEMPORAL_MARKERS: FrozenSet[str] = frozenset({
    "before", "after", "during", "when", "while", "until",
    "since", "then", "previously", "subsequently", "meanwhile",
    "yesterday", "today", "tomorrow", "now", "later", "earlier",
})

_SPATIAL_PREPOSITIONS: FrozenSet[str] = frozenset({
    "in", "on", "at", "near", "above", "below", "under",
    "over", "between", "among", "beside", "behind", "inside",
    "outside", "across", "through", "around", "within",
})

_POSSESSION_VERBS: FrozenSet[str] = frozenset({
    "has", "have", "had", "having", "owns", "owned", "owning",
    "possesses", "possessed", "belongs", "belonged",
})

_COMPARISON_MARKERS: FrozenSet[str] = frozenset({
    "like", "unlike", "similar", "different", "greater", "less",
    "more", "fewer", "better", "worse", "equal", "same",
    "compared", "than", "as",
})

_PART_WHOLE_MARKERS: FrozenSet[str] = frozenset({
    "part", "component", "element", "member", "section",
    "contains", "includes", "comprises", "consists",
    "composed", "made",
})

_SUBORDINATION_MARKERS: FrozenSet[str] = frozenset({
    "that", "which", "who", "whom", "whose", "where", "when",
    "while", "although", "though", "because", "since", "if",
    "unless", "whether", "whereas", "whereby",
})

# Stopwords for entity filtering
_STOPWORDS: FrozenSet[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are",
    "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "this", "that", "these", "those",
    "it", "its", "my", "your", "his", "her", "our", "their",
    "not", "no", "nor", "so", "if", "then", "than", "too",
    "very", "just", "about", "up", "out", "into", "over",
    "after", "before", "between", "through", "during", "above",
    "below", "each", "every", "all", "both", "few", "more",
    "most", "other", "some", "such", "only", "own", "same",
    "also", "how", "what", "when", "where", "why", "who",
    "which", "here", "there", "i", "me", "we", "us", "you",
    "he", "she", "they", "them", "him", "her",
})

# Person-name heuristic: common titles
_PERSON_TITLES: FrozenSet[str] = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sir", "lord", "lady",
    "president", "senator", "governor", "captain", "general",
})

# Temporal quantity patterns
_TEMPORAL_PATTERN = re.compile(
    r"\b\d+\s*(?:second|minute|hour|day|week|month|year|decade|century)s?\b",
    re.IGNORECASE,
)

# Numeric quantity pattern
_QUANTITY_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*(?:%|percent|kg|lb|km|mi|m|cm|mm|gb|mb|tb|hz|mhz|ghz))?\b",
    re.IGNORECASE,
)

# Quoted string pattern
_QUOTED_PATTERN = re.compile(r'"([^"]+)"')

# Capitalized multi-word phrase (e.g., "New York", "United States")
_CAPITALIZED_PHRASE_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

# Single capitalized word (not at sentence start)
_CAPITALIZED_WORD_PATTERN = re.compile(r"(?<=[.!?]\s)[A-Z][a-z]+|\b[A-Z][a-z]{2,}\b")


# =====================================================================
# Configuration
# =====================================================================


@dataclass(frozen=True)
class DataAnalysisConfig:
    """Immutable tuning knobs for the Data Analysis Engine."""

    # --- Entity extraction ---
    entity_confidence_threshold: float = 0.30
    max_entities:                int   = 64
    min_entity_length:          int   = 2
    capitalize_bonus:           float = 0.25   # Confidence bonus for capitalized
    quoted_bonus:               float = 0.30   # Confidence bonus for quoted terms
    title_bonus:                float = 0.20   # Confidence bonus if preceded by title

    # --- Relation extraction ---
    relation_confidence_threshold: float = 0.25
    max_relations:                 int   = 128
    verb_base_confidence:          float = 0.55
    copula_base_confidence:        float = 0.50
    causal_base_confidence:        float = 0.60
    preposition_base_confidence:   float = 0.35

    # --- Co-occurrence ---
    co_occurrence_window: int   = 5    # tokens
    min_co_occurrence:    int   = 1    # minimum count to report

    # --- Dependency depth ---
    max_dependency_depth: int   = 10

    # --- Mode-specific entity thresholds ---
    entity_threshold_normal:     float = 0.30
    entity_threshold_dev:        float = 0.15
    entity_threshold_learning:   float = 0.25
    entity_threshold_reflective: float = 0.40
    entity_threshold_rem_normal: float = 0.25
    entity_threshold_rem_dream:  float = 0.10

    # --- Mode-specific relation thresholds ---
    relation_threshold_normal:     float = 0.25
    relation_threshold_dev:        float = 0.15
    relation_threshold_learning:   float = 0.20
    relation_threshold_reflective: float = 0.35
    relation_threshold_rem_normal: float = 0.25
    relation_threshold_rem_dream:  float = 0.10

    # --- Neurochemical coupling weights ---
    beta_ach_depth:       float = 0.12   # ACh → analysis depth
    beta_ne_scope:        float = 0.10   # NE → entity scope broadening
    beta_da_novelty:      float = 0.10   # DA → novel discovery reward
    beta_5ht_stability:   float = 0.08   # 5-HT → noise reduction
    beta_gaba_suppress:   float = 0.08   # GABA → weak relation suppression
    psi_gamma_osc:        float = 0.06   # Gamma oscillation boost

    # --- Stochastic distribution params ---
    gamma_alpha:  float = 2.0    # Gamma shape for DA/ACh
    gamma_theta:  float = 0.30   # Gamma scale
    poisson_lam:  float = 1.5    # Poisson lambda for NE

    # --- NT feedback modulation ---
    ach_threshold_tighten:  float = 0.10   # Entity threshold reduction per ACh unit above 0.5
    ne_threshold_loosen:    float = 0.12   # Entity threshold reduction per NE unit above 0.5
    sht_threshold_raise:    float = 0.08   # Threshold increase per 5-HT unit above 0.5
    gaba_relation_raise:    float = 0.10   # Relation threshold increase per GABA unit above 0.5
    da_novelty_bonus:       float = 0.15   # Extra confidence for novel entities when DA high

    # --- Novelty tracking ---
    novelty_history_size: int = 100   # Number of recent entities to track for novelty


# =====================================================================
# Mode-specific threshold tables
# =====================================================================

_MODE_ENTITY_THRESHOLDS: Dict[OperationalMode, str] = {
    OperationalMode.NORMAL:     "entity_threshold_normal",
    OperationalMode.DEV:        "entity_threshold_dev",
    OperationalMode.LEARNING:   "entity_threshold_learning",
    OperationalMode.REFLECTIVE: "entity_threshold_reflective",
    OperationalMode.REM_NORMAL: "entity_threshold_rem_normal",
    OperationalMode.REM_DREAM:  "entity_threshold_rem_dream",
}

_MODE_RELATION_THRESHOLDS: Dict[OperationalMode, str] = {
    OperationalMode.NORMAL:     "relation_threshold_normal",
    OperationalMode.DEV:        "relation_threshold_dev",
    OperationalMode.LEARNING:   "relation_threshold_learning",
    OperationalMode.REFLECTIVE: "relation_threshold_reflective",
    OperationalMode.REM_NORMAL: "relation_threshold_rem_normal",
    OperationalMode.REM_DREAM:  "relation_threshold_rem_dream",
}


# =====================================================================
# Data types -- frozen outputs
# =====================================================================


@dataclass(frozen=True)
class ExtractedEntity:
    """
    Single entity extracted from text.

    entity_type is inferred from heuristic rules (capitalization, context
    keywords, title proximity, numeric patterns).
    """
    entity_id:   str         = field(default_factory=lambda: str(uuid.uuid4()))
    text:        str         = ""
    entity_type: EntityType  = EntityType.UNKNOWN
    confidence:  float       = 0.0
    span_start:  int         = -1     # character offset in source
    span_end:    int         = -1
    is_novel:    bool        = False  # True if not seen in recent history
    metadata:    Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedRelation:
    """
    Entity-relation-entity triple extracted from text.

    subject_id and object_id reference ExtractedEntity.entity_id values.
    """
    relation_id:    str            = field(default_factory=lambda: str(uuid.uuid4()))
    subject_id:     str            = ""
    predicate:      str            = ""
    object_id:      str            = ""
    relation_type:  RelationType   = RelationType.ACTION
    confidence:     float          = 0.0
    evidence_text:  str            = ""     # supporting snippet
    metadata:       Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CoOccurrence:
    """
    Co-occurrence record between two entities within a sliding window.
    """
    entity_a:    str   = ""   # entity_id of first entity
    entity_b:    str   = ""   # entity_id of second entity
    count:       int   = 0    # number of co-occurrences
    window_size: int   = 5    # window size used


@dataclass(frozen=True)
class DataAnalysisNeurochem:
    """
    Neurochemical coupling signals from one Data Analysis cycle.

    Notation (Appendix S2-S3):
        da_delta    -> Delta C_DA(t)       : novelty reward for new entities/relations
        ach_delta   -> Delta C_ACh(t)      : attentional depth during analysis
        ne_delta    -> Delta C_NE(t)       : scope-broadening salience
        _5ht_delta  -> Delta C_5HT(t)      : stability signal
        gamma_boost -> Delta phi_gamma(t)  : oscillatory boost during extraction
    """
    da_delta:    float = 0.0
    ach_delta:   float = 0.0
    ne_delta:    float = 0.0
    _5ht_delta:  float = 0.0
    gamma_boost: float = 0.0


@dataclass(frozen=True)
class DataAnalysisResult:
    """Full output of one Data Analysis Engine cycle."""
    entities:              List[ExtractedEntity]   = field(default_factory=list)
    relations:             List[ExtractedRelation]  = field(default_factory=list)
    co_occurrences:        List[CoOccurrence]       = field(default_factory=list)
    dependency_depth:      int                      = 0
    entity_count:          int                      = 0
    relation_count:        int                      = 0
    novel_entity_count:    int                      = 0
    novel_relation_count:  int                      = 0
    entity_type_counts:    Dict[str, int]           = field(default_factory=dict)
    relation_type_counts:  Dict[str, int]           = field(default_factory=dict)
    confidence_mean:       float                    = 0.0
    analysis_depth_score:  float                    = 0.0
    neurochemical_signals: DataAnalysisNeurochem    = field(default_factory=DataAnalysisNeurochem)
    processing_time_ms:    float                    = 0.0
    metadata:              Dict[str, Any]           = field(default_factory=dict)


@dataclass(frozen=True)
class DataAnalysisInput:
    """Input bundle for one Data Analysis Engine invocation."""
    raw_text:      str                         = ""
    tokens:        Optional[List[str]]         = None   # pre-tokenized if available
    active_mode:   OperationalMode             = OperationalMode.NORMAL
    context_texts: List[str]                   = field(default_factory=list)


# =====================================================================
# Mutable engine state
# =====================================================================


@dataclass
class DataAnalysisState:
    """Running state for neurochemical modulation and novelty tracking."""
    ach_level:   float = 0.0
    ne_level:    float = 0.0
    da_level:    float = 0.0
    _5ht_level:  float = 0.0
    gaba_level:  float = 0.0
    # Novelty tracking: set of recently seen entity text (lowered)
    recent_entities: List[str] = field(default_factory=list)


# =====================================================================
# Pure helper functions -- Tokenization
# =====================================================================


def tokenize_simple(text: str) -> List[str]:
    """
    Lowercase whitespace tokenizer with minimal punctuation stripping.
    Returns list of tokens preserving order.
    """
    return [
        w.strip(".,!?;:\"'()[]{}") for w in text.split()
        if w.strip(".,!?;:\"'()[]{}")
    ]


def tokenize_preserving_case(text: str) -> List[str]:
    """
    Whitespace tokenizer preserving case (needed for entity extraction).
    Strips trailing punctuation but keeps case.
    """
    return [
        w.strip(".,!?;:\"'()[]{}") for w in text.split()
        if w.strip(".,!?;:\"'()[]{}")
    ]


# =====================================================================
# Pure helper functions -- Entity extraction
# =====================================================================


def _classify_entity_type(text: str, context_lower: str) -> EntityType:
    """
    Heuristic entity type classification.

    Rules (in priority order):
      1. Temporal quantity pattern -> TEMPORAL
      2. Numeric quantity pattern -> QUANTITY
      3. Preceded by person title -> PERSON
      4. Location keywords nearby -> LOCATION
      5. Organization suffixes -> ORGANIZATION
      6. Event keywords -> EVENT
      7. Single capitalized word with > 3 chars -> default CONCEPT
      8. Otherwise UNKNOWN
    """
    lower = text.lower()

    # 1. Temporal
    if _TEMPORAL_PATTERN.search(text):
        return EntityType.TEMPORAL

    # 2. Quantity
    if _QUANTITY_PATTERN.fullmatch(text.strip()):
        return EntityType.QUANTITY

    # 3. Person title check (look for title immediately before entity in context)
    for title in _PERSON_TITLES:
        pattern = title + " " + lower
        if pattern in context_lower:
            return EntityType.PERSON

    # 4. Location keywords
    location_markers = {"city", "country", "state", "province", "river",
                        "mountain", "ocean", "sea", "lake", "street",
                        "avenue", "boulevard", "park", "island"}
    for marker in location_markers:
        if marker in context_lower:
            idx = context_lower.find(lower)
            marker_idx = context_lower.find(marker)
            if idx >= 0 and marker_idx >= 0 and abs(idx - marker_idx) < 30:
                return EntityType.LOCATION

    # 5. Organization suffixes
    org_suffixes = ("inc", "corp", "ltd", "llc", "co", "company",
                    "foundation", "institute", "university", "organization",
                    "department", "agency", "committee", "council")
    if any(lower.endswith(s) or lower.startswith(s) for s in org_suffixes):
        return EntityType.ORGANIZATION
    for s in org_suffixes:
        if s in context_lower:
            idx = context_lower.find(lower)
            s_idx = context_lower.find(s)
            if idx >= 0 and s_idx >= 0 and abs(idx - s_idx) < 20:
                return EntityType.ORGANIZATION

    # 6. Event keywords
    event_markers = {"event", "conference", "meeting", "summit", "ceremony",
                     "election", "war", "battle", "incident", "crash",
                     "disaster", "outbreak", "revolution"}
    for marker in event_markers:
        if marker in context_lower:
            idx = context_lower.find(lower)
            m_idx = context_lower.find(marker)
            if idx >= 0 and m_idx >= 0 and abs(idx - m_idx) < 25:
                return EntityType.EVENT

    # 7. Capitalized word default
    if text[0].isupper() and len(text) > 2:
        return EntityType.CONCEPT

    return EntityType.UNKNOWN


def extract_entities(
    text: str,
    tokens_with_case: List[str],
    threshold: float,
    cfg: DataAnalysisConfig,
    recent_entities: Set[str],
    da_novelty_bonus: float,
) -> List[ExtractedEntity]:
    """
    Rule-based entity extraction.

    Sources of entity candidates:
      1. Capitalized multi-word phrases ("New York", "United States")
      2. Single capitalized words (not at sentence start for first word only)
      3. Quoted terms ("the server")
      4. Numeric quantities and temporal expressions
    """
    entities: List[ExtractedEntity] = []
    seen_spans: Set[Tuple[int, int]] = set()
    context_lower = text.lower()

    def _add_entity(
        ent_text: str,
        span_start: int,
        span_end: int,
        base_confidence: float,
        bonus: float = 0.0,
    ) -> None:
        if len(entities) >= cfg.max_entities:
            return
        if len(ent_text) < cfg.min_entity_length:
            return
        if ent_text.lower() in _STOPWORDS:
            return
        # Avoid overlapping spans
        for s, e in seen_spans:
            if not (span_end <= s or span_start >= e):
                return

        confidence = _clamp(base_confidence + bonus)

        # Novelty bonus
        is_novel = ent_text.lower() not in recent_entities
        if is_novel and da_novelty_bonus > 0.0:
            confidence = _clamp(confidence + da_novelty_bonus * 0.5)

        if confidence < threshold:
            return

        etype = _classify_entity_type(ent_text, context_lower)

        entities.append(ExtractedEntity(
            text=ent_text,
            entity_type=etype,
            confidence=round(confidence, 4),
            span_start=span_start,
            span_end=span_end,
            is_novel=is_novel,
            metadata={"source": "rule_based"},
        ))
        seen_spans.add((span_start, span_end))

    # 1. Capitalized multi-word phrases
    for m in _CAPITALIZED_PHRASE_PATTERN.finditer(text):
        phrase = m.group()
        _add_entity(phrase, m.start(), m.end(), 0.65, cfg.capitalize_bonus)

    # 2. Quoted terms
    for m in _QUOTED_PATTERN.finditer(text):
        quoted = m.group(1)
        if len(quoted) >= cfg.min_entity_length:
            _add_entity(quoted, m.start(1), m.end(1), 0.55, cfg.quoted_bonus)

    # 3. Temporal expressions
    for m in _TEMPORAL_PATTERN.finditer(text):
        _add_entity(m.group(), m.start(), m.end(), 0.60)

    # 4. Numeric quantities
    for m in _QUANTITY_PATTERN.finditer(text):
        _add_entity(m.group(), m.start(), m.end(), 0.50)

    # 5. Single capitalized words (skip first word of sentences)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            clean = word.strip(".,!?;:\"'()[]{}()")
            if not clean:
                continue
            if clean[0].isupper() and len(clean) >= 3 and clean.lower() not in _STOPWORDS:
                # Skip if it's the very first word of the sentence (likely
                # just sentence-initial capitalization)
                if i == 0 and len(words) > 1:
                    # Only skip if it's not also a proper noun pattern
                    # (detected by presence in another sentence mid-position)
                    other_occurrences = sum(
                        1 for s2 in sentences if s2 != sentence
                        for j, w2 in enumerate(s2.split())
                        if j > 0 and w2.strip(".,!?;:\"'()[]{}") == clean
                    )
                    if other_occurrences == 0:
                        continue

                # Find span in original text
                idx = text.find(clean)
                if idx >= 0:
                    _add_entity(
                        clean,
                        idx,
                        idx + len(clean),
                        0.45,
                        cfg.capitalize_bonus * 0.5,
                    )

    return entities


# =====================================================================
# Pure helper functions -- Relation extraction
# =====================================================================


def _find_nearest_entity(
    position: int,
    entities: List[ExtractedEntity],
    direction: str,
    max_distance: int = 100,
) -> Optional[ExtractedEntity]:
    """
    Find the nearest entity to *position* in the given direction
    ('left' or 'right').
    """
    best: Optional[ExtractedEntity] = None
    best_dist = max_distance + 1

    for ent in entities:
        if direction == "left":
            if ent.span_end <= position:
                dist = position - ent.span_end
                if dist < best_dist:
                    best_dist = dist
                    best = ent
        else:  # right
            if ent.span_start >= position:
                dist = ent.span_start - position
                if dist < best_dist:
                    best_dist = dist
                    best = ent

    return best


def _classify_relation_type(predicate: str) -> RelationType:
    """Classify a predicate word/phrase into a RelationType."""
    lower = predicate.lower()

    if lower in _COPULA_VERBS:
        return RelationType.COPULA
    if lower in _CAUSAL_CONNECTORS:
        return RelationType.CAUSAL
    if lower in _SPATIAL_PREPOSITIONS:
        return RelationType.SPATIAL
    if lower in _TEMPORAL_MARKERS:
        return RelationType.TEMPORAL_REL
    if lower in _POSSESSION_VERBS:
        return RelationType.POSSESSION
    if lower in _COMPARISON_MARKERS:
        return RelationType.COMPARISON
    if lower in _PART_WHOLE_MARKERS:
        return RelationType.PART_WHOLE

    # Check if it's in the action verbs list
    if lower in {v.lower() for v in _ACTION_VERBS}:
        return RelationType.ACTION

    # Generic preposition fallback
    if lower in _SPATIAL_PREPOSITIONS | _TEMPORAL_MARKERS:
        return RelationType.PREPOSITION

    return RelationType.ACTION  # default


def extract_relations(
    text: str,
    entities: List[ExtractedEntity],
    threshold: float,
    cfg: DataAnalysisConfig,
    ach_depth_bonus: float,
) -> List[ExtractedRelation]:
    """
    Extract entity-relation-entity triples from text.

    Strategies:
      1. Verb-based: for each action/copula verb found, attach nearest
         left entity (subject) and nearest right entity (object).
      2. Causal connectors: link entities across causal markers.
      3. Preposition links: entity PREP entity.
    """
    if not entities:
        return []

    relations: List[ExtractedRelation] = []
    words = text.split()
    seen_triples: Set[Tuple[str, str, str]] = set()

    def _add_relation(
        subject: ExtractedEntity,
        predicate: str,
        obj: ExtractedEntity,
        base_confidence: float,
        evidence: str = "",
    ) -> None:
        if len(relations) >= cfg.max_relations:
            return

        # Dedup
        triple_key = (subject.entity_id, predicate.lower(), obj.entity_id)
        if triple_key in seen_triples:
            return

        # Don't create self-referential relations
        if subject.entity_id == obj.entity_id:
            return

        rel_type = _classify_relation_type(predicate)
        confidence = _clamp(base_confidence + ach_depth_bonus * 0.3)

        if confidence < threshold:
            return

        relations.append(ExtractedRelation(
            subject_id=subject.entity_id,
            predicate=predicate,
            object_id=obj.entity_id,
            relation_type=rel_type,
            confidence=round(confidence, 4),
            evidence_text=evidence[:200] if evidence else "",
            metadata={"strategy": "verb_based"},
        ))
        seen_triples.add(triple_key)

    # Strategy 1: Verb-based relation extraction
    all_verbs = set(_ACTION_VERBS) | _COPULA_VERBS | _POSSESSION_VERBS
    for i, word in enumerate(words):
        clean = word.strip(".,!?;:\"'()[]{}").lower()
        if clean in all_verbs:
            # Find character position of this word in text
            pos = 0
            for j in range(i):
                pos = text.find(words[j], pos) + len(words[j])
            verb_pos = text.find(word, pos)
            if verb_pos < 0:
                continue

            subj = _find_nearest_entity(verb_pos, entities, "left")
            obj = _find_nearest_entity(verb_pos + len(word), entities, "right")

            if subj is not None and obj is not None:
                # Determine base confidence by verb type
                if clean in _COPULA_VERBS:
                    base = cfg.copula_base_confidence
                elif clean in _POSSESSION_VERBS:
                    base = cfg.verb_base_confidence
                else:
                    base = cfg.verb_base_confidence

                # Extract evidence window
                start = max(0, verb_pos - 30)
                end = min(len(text), verb_pos + len(word) + 30)
                evidence = text[start:end]

                _add_relation(subj, clean, obj, base, evidence)

    # Strategy 2: Causal connector relations
    for connector in _CAUSAL_CONNECTORS:
        idx = text.lower().find(connector)
        while idx >= 0:
            subj = _find_nearest_entity(idx, entities, "left")
            obj = _find_nearest_entity(idx + len(connector), entities, "right")
            if subj is not None and obj is not None:
                start = max(0, idx - 30)
                end = min(len(text), idx + len(connector) + 30)
                _add_relation(
                    subj, connector, obj,
                    cfg.causal_base_confidence,
                    text[start:end],
                )
            idx = text.lower().find(connector, idx + len(connector))

    # Strategy 3: Preposition links
    for prep in _SPATIAL_PREPOSITIONS:
        pattern = f" {prep} "
        idx = text.lower().find(pattern)
        while idx >= 0:
            subj = _find_nearest_entity(idx, entities, "left")
            obj = _find_nearest_entity(idx + len(pattern), entities, "right")
            if subj is not None and obj is not None:
                start = max(0, idx - 20)
                end = min(len(text), idx + len(pattern) + 20)
                _add_relation(
                    subj, prep, obj,
                    cfg.preposition_base_confidence,
                    text[start:end],
                )
            idx = text.lower().find(pattern, idx + len(pattern))

    return relations


# =====================================================================
# Pure helper functions -- Co-occurrence
# =====================================================================


def compute_co_occurrences(
    text: str,
    entities: List[ExtractedEntity],
    window_size: int,
    min_count: int,
) -> List[CoOccurrence]:
    """
    Sliding-window co-occurrence counting.

    Tokenizes text, maps tokens to entity IDs (by substring match),
    then counts pairs within windows of *window_size* tokens.
    """
    if len(entities) < 2:
        return []

    tokens = tokenize_simple(text)
    if len(tokens) < 2:
        return []

    # Map each token index to the set of entity IDs it belongs to
    token_entity_map: List[Set[str]] = [set() for _ in tokens]
    for ent in entities:
        ent_lower = ent.text.lower()
        ent_tokens = ent_lower.split()
        for i in range(len(tokens) - len(ent_tokens) + 1):
            if tokens[i:i + len(ent_tokens)] == ent_tokens:
                for j in range(i, i + len(ent_tokens)):
                    token_entity_map[j].add(ent.entity_id)

    # Also do single-token matching for entities that are single words
    for i, tok in enumerate(tokens):
        for ent in entities:
            if tok == ent.text.lower() and len(ent.text.split()) == 1:
                token_entity_map[i].add(ent.entity_id)

    # Count co-occurrences within sliding windows
    pair_counts: Dict[Tuple[str, str], int] = {}
    for i in range(len(tokens)):
        entities_at_i = token_entity_map[i]
        if not entities_at_i:
            continue
        for j in range(i + 1, min(i + window_size, len(tokens))):
            entities_at_j = token_entity_map[j]
            if not entities_at_j:
                continue
            for a in entities_at_i:
                for b in entities_at_j:
                    if a == b:
                        continue
                    key = (min(a, b), max(a, b))
                    pair_counts[key] = pair_counts.get(key, 0) + 1

    # Build output
    result: List[CoOccurrence] = []
    for (a, b), count in sorted(pair_counts.items(), key=lambda x: -x[1]):
        if count >= min_count:
            result.append(CoOccurrence(
                entity_a=a,
                entity_b=b,
                count=count,
                window_size=window_size,
            ))

    return result


# =====================================================================
# Pure helper functions -- Dependency depth
# =====================================================================


def estimate_dependency_depth(
    text: str,
    max_depth: int,
) -> int:
    """
    Estimate clause-nesting depth via subordination marker counting
    and parenthetical/bracket tracking.

    Heuristic: depth = base(1) + subordination_count + nesting_depth.
    """
    if not text.strip():
        return 0

    lower = text.lower()

    # Count subordination markers
    sub_count = 0
    for marker in _SUBORDINATION_MARKERS:
        pattern = f" {marker} "
        sub_count += lower.count(pattern)

    # Track parenthetical nesting
    max_paren = 0
    current_paren = 0
    for ch in text:
        if ch in ("(", "[", "{"):
            current_paren += 1
            max_paren = max(max_paren, current_paren)
        elif ch in (")", "]", "}"):
            current_paren = max(0, current_paren - 1)

    # Count comma-separated clauses as mild depth
    comma_clauses = max(0, text.count(",") - 1)  # -1 to avoid counting lists

    # Base depth is always 1 for non-empty text
    depth = 1 + min(sub_count, 5) + max_paren + min(comma_clauses // 3, 2)

    return min(depth, max_depth)


# =====================================================================
# Pure helper functions -- Confidence and scoring
# =====================================================================


def classify_confidence_tier(confidence: float) -> ConfidenceTier:
    """Map confidence to discrete tier."""
    if confidence >= 0.75:
        return ConfidenceTier.HIGH
    if confidence >= 0.45:
        return ConfidenceTier.MEDIUM
    if confidence >= 0.20:
        return ConfidenceTier.LOW
    return ConfidenceTier.TRACE


def compute_analysis_depth_score(
    entity_count: int,
    relation_count: int,
    dependency_depth: int,
    co_occurrence_count: int,
) -> float:
    """
    Composite score [0, 1] reflecting how deeply the text was analyzed.

    D(t) = w_e * norm(entities) + w_r * norm(relations) +
           w_d * norm(depth) + w_c * norm(co_occurrences)
    """
    w_e = 0.30
    w_r = 0.35
    w_d = 0.20
    w_c = 0.15

    # Saturating normalization
    norm_e = 1.0 - math.exp(-entity_count / 5.0)
    norm_r = 1.0 - math.exp(-relation_count / 8.0)
    norm_d = min(1.0, dependency_depth / 6.0)
    norm_c = 1.0 - math.exp(-co_occurrence_count / 5.0)

    return _clamp(w_e * norm_e + w_r * norm_r + w_d * norm_d + w_c * norm_c)


def resolve_entity_threshold(
    mode: OperationalMode,
    cfg: DataAnalysisConfig,
) -> float:
    """Get entity confidence threshold for the active mode."""
    attr_name = _MODE_ENTITY_THRESHOLDS.get(mode, "entity_threshold_normal")
    return getattr(cfg, attr_name, cfg.entity_confidence_threshold)


def resolve_relation_threshold(
    mode: OperationalMode,
    cfg: DataAnalysisConfig,
) -> float:
    """Get relation confidence threshold for the active mode."""
    attr_name = _MODE_RELATION_THRESHOLDS.get(mode, "relation_threshold_normal")
    return getattr(cfg, attr_name, cfg.relation_confidence_threshold)


# =====================================================================
# Pure helper functions -- Neurochemical coupling
# =====================================================================


def compute_neurochem_signals(
    entities: List[ExtractedEntity],
    relations: List[ExtractedRelation],
    novel_entity_count: int,
    novel_relation_count: int,
    analysis_depth: float,
    cfg: DataAnalysisConfig,
    rng: np.random.Generator,
) -> DataAnalysisNeurochem:
    """
    Neurochemical coupling from data analysis output.

    DA   -- novelty reward for discovering new entities/relations
    ACh  -- attention depth during analysis (proportional to richness)
    NE   -- scope signal (fires when many entities found)
    5-HT -- stability signal (inverse of noise: high confidence = stable)
    Gamma-- oscillatory boost during active extraction
    """
    if not entities and not relations:
        return DataAnalysisNeurochem()

    n_ent = len(entities)
    n_rel = len(relations)
    total_novel = novel_entity_count + novel_relation_count

    # DA: novelty reward (stochastic)
    da_noise = float(rng.gamma(cfg.gamma_alpha, cfg.gamma_theta))
    da_delta = cfg.beta_da_novelty * (total_novel / max(1, n_ent + n_rel)) * da_noise

    # ACh: attentional depth
    ach_noise = float(rng.gamma(cfg.gamma_alpha, cfg.gamma_theta))
    ach_delta = cfg.beta_ach_depth * analysis_depth * ach_noise

    # NE: entity scope signal (Poisson burst for many entities)
    ne_impulse = 0.0
    if n_ent >= 3:
        ne_impulse = float(rng.poisson(cfg.poisson_lam)) / max(1.0, cfg.poisson_lam)
    ne_delta = cfg.beta_ne_scope * (n_ent / max(1, cfg.max_entities)) * ne_impulse

    # 5-HT: stability (mean confidence)
    if entities:
        mean_conf = sum(e.confidence for e in entities) / n_ent
    else:
        mean_conf = 0.0
    _5ht_delta = cfg.beta_5ht_stability * mean_conf

    # Gamma oscillation boost
    gamma_boost = cfg.psi_gamma_osc * (1.0 if (entities or relations) else 0.0)

    return DataAnalysisNeurochem(
        da_delta=da_delta,
        ach_delta=ach_delta,
        ne_delta=ne_delta,
        _5ht_delta=_5ht_delta,
        gamma_boost=gamma_boost,
    )


# =====================================================================
# Engine class
# =====================================================================


class DataAnalysisEngine:
    """
    Engine 18 -- Data Analysis Engine.

    Extracts structured entity-relation-entity triples, dependency
    structures, and co-occurrence relationships from text.

    Four-stage pipeline:
      1. Entity Extraction  (rule-based NER proxy)
      2. Relation Extraction (verb/preposition patterns)
      3. Co-occurrence Matrix (sliding window)
      4. Dependency Depth Estimation (clause nesting)

    API
    ---
    configure(mode)                 -- set operational mode
    update_neurochem_state(state)   -- inject external NT levels
    process(input_data)             -- run extraction pipeline
    get_status()                    -- introspection
    """

    engine_id = "data_analysis_engine"
    cluster   = "pattern_analysis"

    def __init__(
        self,
        config: Optional[DataAnalysisConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._cfg = config or DataAnalysisConfig()
        self._rng = rng or np.random.default_rng()
        self._mode = OperationalMode.NORMAL
        self._state = DataAnalysisState()
        self._cycle_count = 0

    # ----- Configuration --------------------------------------------------

    def configure(self, mode: OperationalMode) -> None:
        """Set operational mode (affects thresholds and analysis depth)."""
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """
        Inject current neurochemical levels for bidirectional feedback.

        Canonical keys: "ach", "ne", "da", "5ht", "gaba"
        """
        if "ach" in state_dict:
            self._state.ach_level = state_dict["ach"]
        if "ne" in state_dict:
            self._state.ne_level = state_dict["ne"]
        if "da" in state_dict:
            self._state.da_level = state_dict["da"]
        if "5ht" in state_dict:
            self._state._5ht_level = state_dict["5ht"]
        if "gaba" in state_dict:
            self._state.gaba_level = state_dict["gaba"]

    # ----- Bidirectional NT feedback (threshold modulation) ----------------

    def _modulate_entity_threshold(self, base_threshold: float) -> float:
        """
        Modulate entity extraction threshold based on current NT levels.

        ACh > 0.5 -> tighten (more precise, more relation types)
        NE  > 0.5 -> loosen (broader scope)
        5-HT > 0.5 -> raise (more conservative, less noise)
        """
        t = base_threshold

        # ACh: deeper analysis -> slightly lower threshold (more entities for
        # richer relation extraction)
        if self._state.ach_level > 0.5:
            excess = self._state.ach_level - 0.5
            t -= self._cfg.ach_threshold_tighten * excess

        # NE: broadens scope -> lower threshold
        if self._state.ne_level > 0.5:
            excess = self._state.ne_level - 0.5
            t -= self._cfg.ne_threshold_loosen * excess

        # 5-HT: stabilises -> raise threshold (less noise)
        if self._state._5ht_level > 0.5:
            excess = self._state._5ht_level - 0.5
            t += self._cfg.sht_threshold_raise * excess

        return _clamp(t, 0.05, 0.90)

    def _modulate_relation_threshold(self, base_threshold: float) -> float:
        """
        Modulate relation extraction threshold.

        ACh > 0.5 -> lower (more relation types extracted)
        GABA > 0.5 -> raise (suppress weak relations)
        5-HT > 0.5 -> raise (reduce noise)
        """
        t = base_threshold

        # ACh: deeper -> lower threshold
        if self._state.ach_level > 0.5:
            excess = self._state.ach_level - 0.5
            t -= self._cfg.ach_threshold_tighten * excess * 0.8

        # GABA: suppress weak relations -> raise threshold
        if self._state.gaba_level > 0.5:
            excess = self._state.gaba_level - 0.5
            t += self._cfg.gaba_relation_raise * excess

        # 5-HT: stabilise -> raise threshold
        if self._state._5ht_level > 0.5:
            excess = self._state._5ht_level - 0.5
            t += self._cfg.sht_threshold_raise * excess * 0.6

        return _clamp(t, 0.05, 0.90)

    def _compute_da_novelty_bonus(self) -> float:
        """
        DA-driven novelty bonus for newly discovered entities.
        Active only when DA > 0.4.
        """
        if self._state.da_level > 0.4:
            return self._cfg.da_novelty_bonus * (self._state.da_level - 0.4) / 0.6
        return 0.0

    # ----- Novelty tracking -----------------------------------------------

    def _update_novelty_history(self, entities: List[ExtractedEntity]) -> None:
        """Add newly extracted entities to the novelty history."""
        for ent in entities:
            self._state.recent_entities.append(ent.text.lower())
        # Trim to max size
        max_size = self._cfg.novelty_history_size
        if len(self._state.recent_entities) > max_size:
            self._state.recent_entities = self._state.recent_entities[-max_size:]

    # ----- Main pipeline --------------------------------------------------

    def process(self, input_data: DataAnalysisInput) -> DataAnalysisResult:
        """
        Run the full data analysis pipeline on *input_data*.

        Pipeline stages:
          1. Entity extraction (rule-based NER proxy)
          2. Relation extraction (verb + preposition patterns)
          3. Co-occurrence matrix (sliding window)
          4. Dependency depth estimation (clause nesting)
          5. Neurochemical coupling
        """
        t0 = time.perf_counter()
        self._cycle_count += 1

        text = input_data.raw_text
        if not text.strip():
            elapsed = (time.perf_counter() - t0) * 1000.0
            return DataAnalysisResult(
                processing_time_ms=round(elapsed, 3),
                metadata={
                    "mode": input_data.active_mode.value,
                    "cycle": self._cycle_count,
                    "empty_input": True,
                },
            )

        mode = input_data.active_mode

        # Resolve base thresholds for current mode
        base_entity_thresh = resolve_entity_threshold(mode, self._cfg)
        base_relation_thresh = resolve_relation_threshold(mode, self._cfg)

        # Apply NT modulation to thresholds
        entity_thresh = self._modulate_entity_threshold(base_entity_thresh)
        relation_thresh = self._modulate_relation_threshold(base_relation_thresh)

        # DA-driven novelty bonus
        da_novelty = self._compute_da_novelty_bonus()

        # Tokenize preserving case (for entity extraction)
        tokens_with_case = (
            input_data.tokens if input_data.tokens
            else tokenize_preserving_case(text)
        )

        # Build recent entity set for novelty checking
        recent_set = set(self._state.recent_entities)

        # ---- Stage 1: Entity Extraction ----
        entities = extract_entities(
            text=text,
            tokens_with_case=tokens_with_case,
            threshold=entity_thresh,
            cfg=self._cfg,
            recent_entities=recent_set,
            da_novelty_bonus=da_novelty,
        )

        # ---- Stage 2: Relation Extraction ----
        ach_depth = (
            self._cfg.beta_ach_depth * max(0.0, self._state.ach_level - 0.3)
        )
        relations = extract_relations(
            text=text,
            entities=entities,
            threshold=relation_thresh,
            cfg=self._cfg,
            ach_depth_bonus=ach_depth,
        )

        # ---- Stage 3: Co-occurrence Matrix ----
        co_occurrences = compute_co_occurrences(
            text=text,
            entities=entities,
            window_size=self._cfg.co_occurrence_window,
            min_count=self._cfg.min_co_occurrence,
        )

        # ---- Stage 4: Dependency Depth ----
        dep_depth = estimate_dependency_depth(text, self._cfg.max_dependency_depth)

        # ---- Aggregation ----
        novel_ent_count = sum(1 for e in entities if e.is_novel)

        # Novel relation heuristic: relations connecting novel entities
        novel_entity_ids = {e.entity_id for e in entities if e.is_novel}
        novel_rel_count = sum(
            1 for r in relations
            if r.subject_id in novel_entity_ids or r.object_id in novel_entity_ids
        )

        entity_type_counts: Dict[str, int] = {}
        for e in entities:
            key = e.entity_type.value
            entity_type_counts[key] = entity_type_counts.get(key, 0) + 1

        relation_type_counts: Dict[str, int] = {}
        for r in relations:
            key = r.relation_type.value
            relation_type_counts[key] = relation_type_counts.get(key, 0) + 1

        # Mean confidence
        all_confs = [e.confidence for e in entities] + [r.confidence for r in relations]
        conf_mean = (sum(all_confs) / len(all_confs)) if all_confs else 0.0

        # Analysis depth score
        depth_score = compute_analysis_depth_score(
            len(entities), len(relations), dep_depth, len(co_occurrences),
        )

        # ---- Neurochemical coupling ----
        neurochem = compute_neurochem_signals(
            entities=entities,
            relations=relations,
            novel_entity_count=novel_ent_count,
            novel_relation_count=novel_rel_count,
            analysis_depth=depth_score,
            cfg=self._cfg,
            rng=self._rng,
        )

        # ---- Update novelty history ----
        self._update_novelty_history(entities)

        elapsed = (time.perf_counter() - t0) * 1000.0

        return DataAnalysisResult(
            entities=entities,
            relations=relations,
            co_occurrences=co_occurrences,
            dependency_depth=dep_depth,
            entity_count=len(entities),
            relation_count=len(relations),
            novel_entity_count=novel_ent_count,
            novel_relation_count=novel_rel_count,
            entity_type_counts=entity_type_counts,
            relation_type_counts=relation_type_counts,
            confidence_mean=round(conf_mean, 4),
            analysis_depth_score=round(depth_score, 4),
            neurochemical_signals=neurochem,
            processing_time_ms=round(elapsed, 3),
            metadata={
                "mode": mode.value,
                "entity_threshold": round(entity_thresh, 4),
                "relation_threshold": round(relation_thresh, 4),
                "da_novelty_bonus": round(da_novelty, 4),
                "cycle": self._cycle_count,
                "nt_state": {
                    "ach": self._state.ach_level,
                    "ne": self._state.ne_level,
                    "da": self._state.da_level,
                    "5ht": self._state._5ht_level,
                    "gaba": self._state.gaba_level,
                },
            },
        )

    # ----- Introspection --------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return engine introspection data."""
        return {
            "engine_id": self.engine_id,
            "cluster": self.cluster,
            "mode": self._mode.value,
            "cycle_count": self._cycle_count,
            "state": {
                "ach_level": self._state.ach_level,
                "ne_level": self._state.ne_level,
                "da_level": self._state.da_level,
                "5ht_level": self._state._5ht_level,
                "gaba_level": self._state.gaba_level,
                "recent_entities_count": len(self._state.recent_entities),
            },
        }
