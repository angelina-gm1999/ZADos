"""
Semantic Expander — Pipeline Infrastructure (Pattern & Analysis Cluster).

Outward expansion: takes Tokenizer's structural decomposition and builds
semantic neighborhoods around each concept.

Pipeline Position: Fractal Brain Step (a), fires SECOND (after Tokenizer).

Expansion Pipelines (5)
------------------------
Pipeline 1 — WordNet Relational:  synonyms, hypernyms, hyponyms, antonyms, etc.
Pipeline 2 — Embedding Neighborhood:  cosine-similarity top-K neighbors
Pipeline 3 — ConceptNet (OPTIONAL): relational knowledge graph assertions
Pipeline 4 — Morphological Family: derivational relatives via shared roots
Pipeline 5 — Collocation Extraction: noun chunks, phrasal verbs

Output: ExpansionGraph with per-token expansion sets, concept cloud, shared
concepts, and 9 expansion metrics (fractal_depth, pattern_novelty, etc.)

External NLP dependencies (spaCy vectors, NLTK WordNet, ConceptNet) are
accessed through injectable methods.  The default implementation uses
heuristic fallbacks so the engine is fully testable without multi-GB files.

Neurochemical Coupling
-----------------------
- DA Gamma burst ← pattern_novelty
- 5-HT2A Gamma burst ← symbolic_density
- ACh Gamma burst ← expansion load
- GLU burst ← shared_concept_ratio (cross-token integration demand)
- Beta + Alpha oscillatory boosts

Usage
-----
>>> from zados.cognitive_engines.py_engines.semantic_expander import (
...     SemanticExpander, SemanticExpanderConfig, ExpansionResult,
... )
>>> from zados.cognitive_engines.py_engines.tokenizer import Tokenizer
>>> tok_result = Tokenizer().process("Freedom requires responsibility.")
>>> result = SemanticExpander().process(tok_result)
"""

from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)
from zados.cognitive_engines.py_engines.tokenizer import (
    TokenData,
    TokenizerResult,
)


# =====================================================================
# Enumerations
# =====================================================================

class RelationType(str, Enum):
    SYNONYM       = "synonym"
    HYPERNYM      = "hypernym"
    HYPONYM       = "hyponym"
    MERONYM       = "meronym"
    HOLONYM       = "holonym"
    ANTONYM       = "antonym"
    DERIVATIONAL  = "derivationally_related"
    ALSO_SEE      = "also_see"
    EMBEDDING     = "embedding_neighbor"
    CONCEPTNET    = "conceptnet"
    MORPHOLOGICAL = "morphological"
    COLLOCATION   = "collocation"


# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True)
class SemanticExpanderConfig:
    # WordNet depth limits
    hypernym_depth: int = 3
    hyponym_depth: int = 2
    meronym_depth: int = 2
    holonym_depth: int = 2

    # Embedding neighborhood
    embedding_k: int = 15
    theta_similarity: float = 0.55

    # ConceptNet (optional)
    conceptnet_max_assertions: int = 20
    conceptnet_min_weight: float = 1.0

    # Distance → relevance
    synonym_distance: float = 0.0
    hop1_distance: float = 0.3
    hop2_distance: float = 0.5
    hop3_distance: float = 0.7
    antonym_distance: float = 0.4
    primary_synset_multiplier: float = 1.2

    # Neurochemical coupling
    n_expansion_baseline: int = 100
    lambda_depth: float = 0.10
    da_novelty_coupling: float = 0.05
    da_gamma_alpha: float = 2.0
    da_gamma_theta: float = 0.3
    sht2a_symbolic_coupling: float = 0.04
    sht2a_gamma_alpha: float = 2.0
    sht2a_gamma_theta: float = 0.25
    ach_coupling: float = 0.05
    ach_gamma_alpha: float = 2.5
    ach_gamma_theta: float = 0.25
    glu_coupling: float = 0.04
    glu_shared_threshold: float = 0.15
    beta_boost: float = 0.03
    alpha_symbolic_coupling: float = 0.02

    # Common / high-frequency words (for novelty computation)
    high_frequency_concepts: frozenset = frozenset({
        "thing", "person", "time", "way", "day", "man", "world",
        "life", "part", "place", "case", "group", "company", "system",
        "number", "point", "state", "home", "fact", "year", "work",
        "do", "make", "go", "get", "have", "be", "say", "know",
        "think", "take", "come", "see", "look", "want", "give", "use",
        "find", "tell", "ask", "try", "leave", "call", "keep",
        "good", "new", "first", "last", "long", "great", "little",
        "own", "other", "old", "right", "big", "high", "small",
    })


# =====================================================================
# Data Types
# =====================================================================

@dataclass(frozen=True)
class ExpansionNode:
    concept: str = ""
    relation_type: str = RelationType.SYNONYM.value
    relation_depth: int = 0
    relevance: float = 1.0
    definition: str = ""


@dataclass(frozen=True)
class EmbeddingNeighbor:
    concept: str = ""
    similarity: float = 0.0


@dataclass(frozen=True)
class MorphologicalRelative:
    word: str = ""
    pos: str = ""
    relation_to_source: str = "same_root"
    shared_root: str = ""


@dataclass(frozen=True)
class NounChunk:
    text: str = ""
    root_token: str = ""
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class Collocation:
    text: str = ""
    tokens: Tuple[str, ...] = ()
    coll_type: str = "noun_chunk"


@dataclass(frozen=True)
class TokenExpansionSet:
    source_token: str = ""
    source_index: int = 0
    wordnet_expansions: List[ExpansionNode] = field(default_factory=list)
    embedding_neighbors: List[EmbeddingNeighbor] = field(default_factory=list)
    morphological_relatives: List[MorphologicalRelative] = field(default_factory=list)
    total_expansion_count: int = 0
    mean_relevance: float = 0.0
    # True when token was expanded via heuristic fallbacks (no NLP backend).
    # Downstream consumers should treat results with reduced confidence.
    used_fallback: bool = True


@dataclass(frozen=True)
class ConceptNode:
    concept: str = ""
    sources: Tuple[str, ...] = ()
    relations: Tuple[str, ...] = ()
    max_relevance: float = 0.0
    source_count: int = 0


@dataclass(frozen=True)
class SharedConcept:
    concept: str = ""
    connecting_tokens: Tuple[int, ...] = ()
    relation_types: Tuple[str, ...] = ()
    bridging_strength: float = 0.0


@dataclass(frozen=True)
class ExpansionMetrics:
    fractal_depth: int = 0
    pattern_novelty: float = 0.0
    symbolic_density: float = 0.0
    structural_complexity: float = 0.0
    information_noise_ratio: float = 0.0
    expansion_breadth: float = 0.0
    expansion_coherence: float = 0.0
    polysemy_burden: float = 0.0
    shared_concept_ratio: float = 0.0


@dataclass(frozen=True)
class ExpansionResult:
    token_expansions: Dict[int, TokenExpansionSet] = field(default_factory=dict)
    concept_cloud: Dict[str, ConceptNode] = field(default_factory=dict)
    shared_concepts: List[SharedConcept] = field(default_factory=list)
    collocations: List[Collocation] = field(default_factory=list)
    metrics: ExpansionMetrics = field(default_factory=ExpansionMetrics)
    processing_time_ms: float = 0.0
    neurochemical_signals: Dict[str, float] = field(default_factory=dict)
    # Fraction of tokens that used heuristic fallbacks (0.0 = full NLP, 1.0 = all fallback).
    # When > 0.5, downstream engines should discount expansion-derived signals.
    fallback_ratio: float = 1.0


# =====================================================================
# Pure Functions — Pipeline 1: WordNet Expansion (heuristic)
# =====================================================================

# Built-in synonym / antonym / hypernym dictionaries (minimal for testing)
_SYNONYM_MAP: Dict[str, List[str]] = {
    "freedom": ["liberty", "independence", "autonomy"],
    "responsibility": ["duty", "obligation", "accountability"],
    "believe": ["think", "consider", "hold"],
    "important": ["significant", "crucial", "vital"],
    "security": ["safety", "protection", "defense"],
    "justice": ["fairness", "equity", "righteousness"],
    "truth": ["fact", "reality", "verity"],
    "love": ["affection", "devotion", "attachment"],
}

_ANTONYM_MAP: Dict[str, List[str]] = {
    "freedom": ["captivity", "slavery"],
    "important": ["trivial", "insignificant"],
    "security": ["danger", "risk"],
    "truth": ["falsehood", "lie"],
    "love": ["hate", "indifference"],
    "good": ["bad", "evil"],
}

_HYPERNYM_MAP: Dict[str, List[str]] = {
    "freedom": ["right", "condition", "state"],
    "responsibility": ["obligation", "duty", "concept"],
    "justice": ["principle", "virtue", "concept"],
    "security": ["condition", "state", "need"],
    "truth": ["knowledge", "reality", "concept"],
}


def expand_wordnet(
    lemma: str,
    config: SemanticExpanderConfig,
) -> List[ExpansionNode]:
    """Expand via built-in dictionaries (Phase 1). In Phase 2+, use NLTK WordNet."""
    nodes: List[ExpansionNode] = []
    lemma_lower = lemma.lower()

    # Synonyms (distance 0)
    for syn in _SYNONYM_MAP.get(lemma_lower, []):
        nodes.append(ExpansionNode(
            concept=syn,
            relation_type=RelationType.SYNONYM.value,
            relation_depth=1,
            relevance=1.0 - config.synonym_distance,
            definition=f"synonym of {lemma}",
        ))

    # Antonyms (distance 0.4)
    for ant in _ANTONYM_MAP.get(lemma_lower, []):
        nodes.append(ExpansionNode(
            concept=ant,
            relation_type=RelationType.ANTONYM.value,
            relation_depth=1,
            relevance=1.0 - config.antonym_distance,
            definition=f"antonym of {lemma}",
        ))

    # Hypernyms (up to 3 hops)
    hypernyms = _HYPERNYM_MAP.get(lemma_lower, [])
    for depth_idx, hyp in enumerate(hypernyms[:config.hypernym_depth]):
        hop = depth_idx + 1
        if hop == 1:
            dist = config.hop1_distance
        elif hop == 2:
            dist = config.hop2_distance
        else:
            dist = config.hop3_distance
        nodes.append(ExpansionNode(
            concept=hyp,
            relation_type=RelationType.HYPERNYM.value,
            relation_depth=hop,
            relevance=1.0 - dist,
            definition=f"hypernym of {lemma} (depth {hop})",
        ))

    return nodes


# =====================================================================
# Pure Functions — Pipeline 2: Embedding Neighborhood (stub)
# =====================================================================

def expand_embedding_neighborhood(
    lemma: str,
    config: SemanticExpanderConfig,
) -> List[EmbeddingNeighbor]:
    """
    Stub: generate synthetic neighbors based on character overlap.
    In Phase 2+, uses spaCy word vectors + cosine similarity matrix.
    """
    neighbors: List[EmbeddingNeighbor] = []
    # Heuristic: generate related-sounding words
    if len(lemma) >= 4:
        prefix = lemma[:3]
        # Just return the lemma with suffix variations as stubs
        for suffix, sim in [("ness", 0.75), ("ity", 0.70), ("ism", 0.65), ("ist", 0.60)]:
            candidate = prefix + suffix
            if candidate != lemma and sim >= config.theta_similarity:
                neighbors.append(EmbeddingNeighbor(concept=candidate, similarity=sim))
    return neighbors[:config.embedding_k]


# =====================================================================
# Pure Functions — Pipeline 4: Morphological Family
# =====================================================================

def expand_morphological_family(
    token_text: str,
    root: str,
    pos: str,
) -> List[MorphologicalRelative]:
    """Expand via derivational morphology."""
    relatives: List[MorphologicalRelative] = []
    if not root or len(root) < 3:
        return relatives

    # Generate common derivational forms
    derivations = [
        (root + "tion", "NOUN", "derivation"),
        (root + "ment", "NOUN", "derivation"),
        (root + "ness", "NOUN", "derivation"),
        (root + "able", "ADJ", "derivation"),
        (root + "ive", "ADJ", "derivation"),
        (root + "ly", "ADV", "derivation"),
        (root + "er", "NOUN", "derivation"),
        (root + "ing", "VERB", "derivation"),
    ]
    for word, d_pos, rel in derivations:
        if word != token_text.lower() and len(word) > 3:
            relatives.append(MorphologicalRelative(
                word=word, pos=d_pos, relation_to_source=rel, shared_root=root,
            ))

    return relatives[:6]  # cap


# =====================================================================
# Pure Functions — Pipeline 5: Collocation Extraction
# =====================================================================

def extract_collocations(tokens: List[TokenData]) -> List[Collocation]:
    """Extract noun chunks and simple collocations from token list."""
    collocations: List[Collocation] = []
    # Simple consecutive noun-adj-noun chunks
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.pos_coarse in ("NOUN", "PROPN"):
            chunk_tokens = [t.text]
            j = i + 1
            while j < len(tokens) and tokens[j].pos_coarse in ("NOUN", "ADJ", "PROPN"):
                chunk_tokens.append(tokens[j].text)
                j += 1
            if len(chunk_tokens) > 1:
                collocations.append(Collocation(
                    text=" ".join(chunk_tokens),
                    tokens=tuple(chunk_tokens),
                    coll_type="noun_chunk",
                ))
            i = j
        else:
            i += 1
    return collocations


# =====================================================================
# Pure Functions — Graph Assembly
# =====================================================================

def build_concept_cloud(
    token_expansions: Dict[int, TokenExpansionSet],
) -> Dict[str, ConceptNode]:
    """Build unified concept cloud from all token expansions."""
    cloud: Dict[str, Dict] = {}  # concept -> {sources, relations, max_rel, count}

    for idx, texp in token_expansions.items():
        source = texp.source_token
        for node in texp.wordnet_expansions:
            c = node.concept
            if c not in cloud:
                cloud[c] = {"sources": set(), "relations": set(), "max_rel": 0.0, "count": 0}
            cloud[c]["sources"].add(source)
            cloud[c]["relations"].add(node.relation_type)
            cloud[c]["max_rel"] = max(cloud[c]["max_rel"], node.relevance)
            cloud[c]["count"] += 1

        for nb in texp.embedding_neighbors:
            c = nb.concept
            if c not in cloud:
                cloud[c] = {"sources": set(), "relations": set(), "max_rel": 0.0, "count": 0}
            cloud[c]["sources"].add(source)
            cloud[c]["relations"].add(RelationType.EMBEDDING.value)
            cloud[c]["max_rel"] = max(cloud[c]["max_rel"], nb.similarity)
            cloud[c]["count"] += 1

        for mr in texp.morphological_relatives:
            c = mr.word
            if c not in cloud:
                cloud[c] = {"sources": set(), "relations": set(), "max_rel": 0.0, "count": 0}
            cloud[c]["sources"].add(source)
            cloud[c]["relations"].add(RelationType.MORPHOLOGICAL.value)
            cloud[c]["max_rel"] = max(cloud[c]["max_rel"], 0.7)
            cloud[c]["count"] += 1

    result: Dict[str, ConceptNode] = {}
    for concept, data in cloud.items():
        result[concept] = ConceptNode(
            concept=concept,
            sources=tuple(data["sources"]),
            relations=tuple(data["relations"]),
            max_relevance=data["max_rel"],
            source_count=len(data["sources"]),
        )
    return result


def find_shared_concepts(concept_cloud: Dict[str, ConceptNode]) -> List[SharedConcept]:
    """Find concepts that bridge 2+ source tokens."""
    shared: List[SharedConcept] = []
    for concept, node in concept_cloud.items():
        if node.source_count >= 2:
            shared.append(SharedConcept(
                concept=concept,
                connecting_tokens=(),  # would need token indices
                relation_types=node.relations,
                bridging_strength=node.max_relevance,
            ))
    return shared


# =====================================================================
# Pure Functions — Expansion Metrics
# =====================================================================

def compute_expansion_metrics(
    token_expansions: Dict[int, TokenExpansionSet],
    concept_cloud: Dict[str, ConceptNode],
    shared_concepts: List[SharedConcept],
    content_token_count: int,
    config: SemanticExpanderConfig,
) -> ExpansionMetrics:
    """Compute all 9 expansion metrics."""
    total_concepts = len(concept_cloud)
    total_concepts = max(total_concepts, 1)

    # Fractal depth: max hypernym depth with novel concepts
    max_depth = 0
    for texp in token_expansions.values():
        for node in texp.wordnet_expansions:
            if node.relation_type == RelationType.HYPERNYM.value:
                max_depth = max(max_depth, node.relation_depth)

    # Pattern novelty: proportion NOT in high-frequency set
    if total_concepts > 0:
        novel = sum(1 for c in concept_cloud if c.lower() not in config.high_frequency_concepts)
        pattern_novelty = novel / total_concepts
    else:
        pattern_novelty = 0.0

    # Symbolic density: proportion with abstract hypernyms (relevance < 0.5)
    abstract_count = sum(
        1 for texp in token_expansions.values()
        for node in texp.wordnet_expansions
        if node.relation_type == RelationType.HYPERNYM.value and node.relevance < 0.5
    )
    symbolic_density = min(abstract_count / max(total_concepts, 1), 1.0)

    # Structural complexity: graph density (edges / possible edges)
    total_edges = sum(texp.total_expansion_count for texp in token_expansions.values())
    possible_edges = len(token_expansions) * total_concepts
    structural_complexity = min(total_edges / max(possible_edges, 1), 1.0)

    # Information noise ratio: high-relevance (>0.6) / total
    high_rel = sum(
        1 for node in concept_cloud.values() if node.max_relevance > 0.6
    )
    information_noise_ratio = high_rel / total_concepts

    # Expansion breadth: mean expansion count per content token
    content_count = max(content_token_count, 1)
    expansion_breadth = sum(texp.total_expansion_count for texp in token_expansions.values()) / content_count

    # Polysemy burden
    polysemy_scores = [
        texp.wordnet_expansions[0].relation_depth if texp.wordnet_expansions else 0
        for texp in token_expansions.values()
    ]
    # stub: just use expansion count as proxy
    polysemy_burden = sum(len(texp.wordnet_expansions) for texp in token_expansions.values()) / max(len(token_expansions), 1)

    # Shared concept ratio
    shared_ratio = len(shared_concepts) / total_concepts

    return ExpansionMetrics(
        fractal_depth=max_depth,
        pattern_novelty=float(np.clip(pattern_novelty, 0.0, 1.0)),
        symbolic_density=float(np.clip(symbolic_density, 0.0, 1.0)),
        structural_complexity=float(np.clip(structural_complexity, 0.0, 1.0)),
        information_noise_ratio=float(np.clip(information_noise_ratio, 0.0, 1.0)),
        expansion_breadth=float(expansion_breadth),
        expansion_coherence=0.0,  # requires real embeddings
        polysemy_burden=float(polysemy_burden),
        shared_concept_ratio=float(np.clip(shared_ratio, 0.0, 1.0)),
    )


# =====================================================================
# Pure Functions — Neurochemical Coupling
# =====================================================================

def compute_expansion_load(
    total_expansion_count: int,
    fractal_depth: int,
    config: SemanticExpanderConfig,
) -> float:
    """L_exp(t) = total_expansion_count / N_baseline × (1 + λ_depth × fractal_depth)."""
    load = (total_expansion_count / max(config.n_expansion_baseline, 1)
            * (1 + config.lambda_depth * fractal_depth))
    return float(np.clip(load, 0.0, 10.0))


def compute_expander_neurochemical_signals(
    metrics: ExpansionMetrics,
    load: float,
    config: SemanticExpanderConfig,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Compute all expander neurochemical signals."""
    signals: Dict[str, float] = {
        "da_novelty_burst": 0.0,
        "sht2a_symbolic_burst": 0.0,
        "ach_burst": 0.0,
        "glu_burst": 0.0,
        "beta_boost": config.beta_boost,
        "alpha_boost": 0.0,
    }

    # DA — novelty
    if metrics.pattern_novelty > 0.0:
        da = config.da_novelty_coupling * metrics.pattern_novelty * float(rng.gamma(config.da_gamma_alpha, config.da_gamma_theta))
        signals["da_novelty_burst"] = float(np.clip(da, 0.0, 1.0))

    # 5-HT2A — symbolic depth
    if metrics.symbolic_density > 0.0:
        sht = config.sht2a_symbolic_coupling * metrics.symbolic_density * float(rng.gamma(config.sht2a_gamma_alpha, config.sht2a_gamma_theta))
        signals["sht2a_symbolic_burst"] = float(np.clip(sht, 0.0, 1.0))

    # ACh — expansion attention
    if load > 0.0:
        ach = config.ach_coupling * load * float(rng.gamma(config.ach_gamma_alpha, config.ach_gamma_theta))
        signals["ach_burst"] = float(np.clip(ach, 0.0, 1.0))

    # GLU — integration demand (shared concepts > threshold)
    if metrics.shared_concept_ratio > config.glu_shared_threshold:
        signals["glu_burst"] = float(np.clip(
            config.glu_coupling * metrics.shared_concept_ratio, 0.0, 1.0
        ))

    # Alpha boost from symbolic density
    signals["alpha_boost"] = float(np.clip(
        config.alpha_symbolic_coupling * metrics.symbolic_density, 0.0, 0.5
    ))

    return signals


# =====================================================================
# Engine
# =====================================================================

class SemanticExpander:
    """
    Semantic Expander — Pipeline Infrastructure.

    Public API:
        configure(mode)                    — set operational mode
        process(tokenizer_result)          — main expansion call
        get_status()                       — engine state snapshot
    """

    def __init__(
        self,
        config: SemanticExpanderConfig = SemanticExpanderConfig(),
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._config = config
        self._mode = OperationalMode.NORMAL
        self._rng = rng if rng is not None else np.random.default_rng()

    def configure(self, mode: OperationalMode) -> None:
        self._mode = mode

    def process(self, tokenizer_result: TokenizerResult) -> ExpansionResult:
        if self._mode == OperationalMode.REM_DREAM:
            # Expanded mode: deeper expansion
            effective_config = SemanticExpanderConfig(
                hypernym_depth=self._config.hypernym_depth + 2,
                embedding_k=self._config.embedding_k + 10,
                theta_similarity=0.45,
            )
        else:
            effective_config = self._config

        t0 = time.perf_counter()

        content_tokens = [t for t in tokenizer_result.tokens if t.is_content_word]
        token_expansions: Dict[int, TokenExpansionSet] = {}

        for t in content_tokens:
            lemma = t.lemma
            idx = t.token_index

            # Pipeline 1: WordNet
            wn_nodes = expand_wordnet(lemma, effective_config)

            # Pipeline 2: Embedding neighborhood
            emb_neighbors = expand_embedding_neighborhood(lemma, effective_config)

            # Pipeline 4: Morphological family
            root = t.morphology.root if t.morphology else lemma
            morph_relatives = expand_morphological_family(t.text, root, t.pos_coarse)

            total = len(wn_nodes) + len(emb_neighbors) + len(morph_relatives)
            relevances = (
                [n.relevance for n in wn_nodes]
                + [n.similarity for n in emb_neighbors]
                + [0.7] * len(morph_relatives)
            )
            mean_rel = sum(relevances) / max(len(relevances), 1)

            # Flag fallback: True when the WordNet pipeline returned nothing
            # (token not in the heuristic dictionaries → no real synonyms/hypernyms)
            tok_used_fallback = len(wn_nodes) == 0

            token_expansions[idx] = TokenExpansionSet(
                source_token=t.text,
                source_index=idx,
                wordnet_expansions=wn_nodes,
                embedding_neighbors=emb_neighbors,
                morphological_relatives=morph_relatives,
                total_expansion_count=total,
                mean_relevance=float(mean_rel),
                used_fallback=tok_used_fallback,
            )

        # Pipeline 5: Collocations
        collocations = extract_collocations(tokenizer_result.tokens)

        # Graph assembly
        concept_cloud = build_concept_cloud(token_expansions)
        shared = find_shared_concepts(concept_cloud)

        # Metrics
        metrics = compute_expansion_metrics(
            token_expansions, concept_cloud, shared,
            len(content_tokens), effective_config,
        )

        # Neurochemical
        total_exp = sum(texp.total_expansion_count for texp in token_expansions.values())
        load = compute_expansion_load(total_exp, metrics.fractal_depth, effective_config)
        signals = compute_expander_neurochemical_signals(metrics, load, effective_config, self._rng)

        dt_ms = (time.perf_counter() - t0) * 1000.0

        # Compute fallback ratio: what fraction of tokens used heuristic-only?
        n_tokens = len(token_expansions)
        n_fallback = sum(1 for t in token_expansions.values() if t.used_fallback)
        fb_ratio = n_fallback / max(n_tokens, 1)

        return ExpansionResult(
            token_expansions=token_expansions,
            concept_cloud=concept_cloud,
            shared_concepts=shared,
            collocations=collocations,
            metrics=metrics,
            processing_time_ms=dt_ms,
            neurochemical_signals=signals,
            fallback_ratio=fb_ratio,
        )

    def get_status(self) -> Dict:
        return {"mode": self._mode.value}
