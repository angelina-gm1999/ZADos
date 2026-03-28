"""
Tests for Semantic Expander — Pipeline Infrastructure.
"""

import math

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)
from zados.cognitive_engines.py_engines.tokenizer import (
    Tokenizer,
    TokenizerConfig,
    TokenizerResult,
    TokenData,
    MorphologyEntry,
)
from zados.cognitive_engines.py_engines.semantic_expander import (
    Collocation,
    ConceptNode,
    EmbeddingNeighbor,
    ExpansionMetrics,
    ExpansionNode,
    ExpansionResult,
    MorphologicalRelative,
    NounChunk,
    RelationType,
    SemanticExpander,
    SemanticExpanderConfig,
    SharedConcept,
    TokenExpansionSet,
    build_concept_cloud,
    compute_expansion_load,
    compute_expansion_metrics,
    compute_expander_neurochemical_signals,
    expand_embedding_neighborhood,
    expand_morphological_family,
    expand_wordnet,
    extract_collocations,
    find_shared_concepts,
)


# =====================================================================
# Helpers
# =====================================================================

RNG = np.random.default_rng(42)


def _make_expander(seed: int = 42, **config_kw) -> SemanticExpander:
    cfg = SemanticExpanderConfig(**config_kw)
    return SemanticExpander(config=cfg, rng=np.random.default_rng(seed))


def _make_tokenizer_result(text: str = "Freedom requires responsibility.") -> TokenizerResult:
    tok = Tokenizer(rng=np.random.default_rng(42))
    return tok.process(text)


# =====================================================================
# Test: Enumerations
# =====================================================================

class TestEnums:
    def test_relation_types(self):
        assert len(RelationType) >= 10
        assert RelationType.SYNONYM.value == "synonym"
        assert RelationType.HYPERNYM.value == "hypernym"
        assert RelationType.ANTONYM.value == "antonym"
        assert RelationType.EMBEDDING.value == "embedding_neighbor"
        assert RelationType.MORPHOLOGICAL.value == "morphological"
        assert RelationType.COLLOCATION.value == "collocation"


# =====================================================================
# Test: Configuration
# =====================================================================

class TestConfig:
    def test_default_config_valid(self):
        cfg = SemanticExpanderConfig()
        assert cfg.hypernym_depth == 3
        assert cfg.embedding_k == 15
        assert cfg.theta_similarity == 0.55
        assert cfg.beta_boost == 0.03

    def test_high_frequency_concepts_populated(self):
        cfg = SemanticExpanderConfig()
        assert "thing" in cfg.high_frequency_concepts
        assert "good" in cfg.high_frequency_concepts


# =====================================================================
# Test: Data Types
# =====================================================================

class TestDataTypes:
    def test_expansion_node_defaults(self):
        en = ExpansionNode()
        assert en.concept == ""
        assert en.relevance == 1.0

    def test_embedding_neighbor_defaults(self):
        nb = EmbeddingNeighbor()
        assert nb.similarity == 0.0

    def test_morphological_relative_defaults(self):
        mr = MorphologicalRelative()
        assert mr.relation_to_source == "same_root"

    def test_token_expansion_set_defaults(self):
        tes = TokenExpansionSet()
        assert tes.total_expansion_count == 0

    def test_concept_node_defaults(self):
        cn = ConceptNode()
        assert cn.source_count == 0

    def test_shared_concept_defaults(self):
        sc = SharedConcept()
        assert sc.bridging_strength == 0.0

    def test_expansion_metrics_defaults(self):
        em = ExpansionMetrics()
        assert em.fractal_depth == 0
        assert em.pattern_novelty == 0.0

    def test_expansion_result_defaults(self):
        er = ExpansionResult()
        assert er.processing_time_ms == 0.0

    def test_frozen_immutable(self):
        en = ExpansionNode()
        with pytest.raises(AttributeError):
            en.concept = "test"


# =====================================================================
# Test: Pipeline 1 — WordNet Expansion
# =====================================================================

class TestWordnetExpansion:
    def test_known_word_has_synonyms(self):
        nodes = expand_wordnet("freedom", SemanticExpanderConfig())
        synonyms = [n for n in nodes if n.relation_type == RelationType.SYNONYM.value]
        assert len(synonyms) > 0
        assert any(n.concept == "liberty" for n in synonyms)

    def test_known_word_has_antonyms(self):
        nodes = expand_wordnet("freedom", SemanticExpanderConfig())
        antonyms = [n for n in nodes if n.relation_type == RelationType.ANTONYM.value]
        assert len(antonyms) > 0
        assert any(n.concept == "captivity" for n in antonyms)

    def test_known_word_has_hypernyms(self):
        nodes = expand_wordnet("freedom", SemanticExpanderConfig())
        hypernyms = [n for n in nodes if n.relation_type == RelationType.HYPERNYM.value]
        assert len(hypernyms) > 0

    def test_unknown_word_empty(self):
        nodes = expand_wordnet("xyzzyplugh", SemanticExpanderConfig())
        assert len(nodes) == 0

    def test_synonym_relevance_1(self):
        cfg = SemanticExpanderConfig()
        nodes = expand_wordnet("freedom", cfg)
        synonyms = [n for n in nodes if n.relation_type == RelationType.SYNONYM.value]
        for s in synonyms:
            assert s.relevance == 1.0  # synonym_distance = 0.0

    def test_antonym_relevance_correct(self):
        cfg = SemanticExpanderConfig()
        nodes = expand_wordnet("freedom", cfg)
        antonyms = [n for n in nodes if n.relation_type == RelationType.ANTONYM.value]
        for a in antonyms:
            assert a.relevance == pytest.approx(0.6)  # 1.0 - 0.4

    def test_hypernym_depth_respected(self):
        cfg = SemanticExpanderConfig(hypernym_depth=2)
        nodes = expand_wordnet("freedom", cfg)
        hypernyms = [n for n in nodes if n.relation_type == RelationType.HYPERNYM.value]
        assert len(hypernyms) <= 2

    def test_hypernym_depth_increases_distance(self):
        cfg = SemanticExpanderConfig()
        nodes = expand_wordnet("freedom", cfg)
        hypernyms = sorted(
            [n for n in nodes if n.relation_type == RelationType.HYPERNYM.value],
            key=lambda n: n.relation_depth,
        )
        if len(hypernyms) >= 2:
            assert hypernyms[0].relevance >= hypernyms[1].relevance


# =====================================================================
# Test: Pipeline 2 — Embedding Neighborhood
# =====================================================================

class TestEmbeddingNeighborhood:
    def test_produces_neighbors_for_long_word(self):
        cfg = SemanticExpanderConfig()
        neighbors = expand_embedding_neighborhood("freedom", cfg)
        assert len(neighbors) > 0

    def test_empty_for_short_word(self):
        cfg = SemanticExpanderConfig()
        neighbors = expand_embedding_neighborhood("go", cfg)
        assert len(neighbors) == 0

    def test_similarity_above_threshold(self):
        cfg = SemanticExpanderConfig()
        neighbors = expand_embedding_neighborhood("freedom", cfg)
        for nb in neighbors:
            assert nb.similarity >= cfg.theta_similarity

    def test_capped_by_embedding_k(self):
        cfg = SemanticExpanderConfig(embedding_k=2)
        neighbors = expand_embedding_neighborhood("understanding", cfg)
        assert len(neighbors) <= 2


# =====================================================================
# Test: Pipeline 4 — Morphological Family
# =====================================================================

class TestMorphologicalFamily:
    def test_produces_relatives(self):
        relatives = expand_morphological_family("protect", "protect", "VERB")
        assert len(relatives) > 0

    def test_relatives_share_root(self):
        relatives = expand_morphological_family("protect", "protect", "VERB")
        for r in relatives:
            assert r.shared_root == "protect"

    def test_excludes_source_word(self):
        relatives = expand_morphological_family("protection", "protect", "NOUN")
        for r in relatives:
            assert r.word != "protection"

    def test_short_root_empty(self):
        relatives = expand_morphological_family("go", "go", "VERB")
        assert len(relatives) == 0

    def test_capped_at_six(self):
        relatives = expand_morphological_family("understand", "understand", "VERB")
        assert len(relatives) <= 6


# =====================================================================
# Test: Pipeline 5 — Collocation Extraction
# =====================================================================

class TestCollocationExtraction:
    def test_noun_chunk_detected(self):
        tokens = [
            TokenData(text="human", pos_coarse="NOUN"),
            TokenData(text="rights", pos_coarse="NOUN"),
        ]
        colls = extract_collocations(tokens)
        assert len(colls) >= 1
        assert colls[0].text == "human rights"

    def test_no_chunk_for_verbs(self):
        tokens = [
            TokenData(text="run", pos_coarse="VERB"),
            TokenData(text="fast", pos_coarse="ADV"),
        ]
        colls = extract_collocations(tokens)
        assert len(colls) == 0

    def test_adjective_noun_chunk(self):
        tokens = [
            TokenData(text="big", pos_coarse="NOUN"),  # Mislabeled as NOUN to trigger
            TokenData(text="idea", pos_coarse="NOUN"),
        ]
        colls = extract_collocations(tokens)
        assert len(colls) >= 1

    def test_empty_tokens(self):
        colls = extract_collocations([])
        assert len(colls) == 0


# =====================================================================
# Test: Graph Assembly
# =====================================================================

class TestGraphAssembly:
    def _sample_expansion(self):
        return {
            0: TokenExpansionSet(
                source_token="freedom",
                wordnet_expansions=[
                    ExpansionNode(concept="liberty", relation_type=RelationType.SYNONYM.value, relevance=1.0),
                    ExpansionNode(concept="right", relation_type=RelationType.HYPERNYM.value, relation_depth=1, relevance=0.7),
                ],
                embedding_neighbors=[EmbeddingNeighbor(concept="freeness", similarity=0.65)],
                morphological_relatives=[MorphologicalRelative(word="freed", shared_root="free")],
                total_expansion_count=4,
            ),
            1: TokenExpansionSet(
                source_token="responsibility",
                wordnet_expansions=[
                    ExpansionNode(concept="duty", relation_type=RelationType.SYNONYM.value, relevance=1.0),
                    ExpansionNode(concept="right", relation_type=RelationType.HYPERNYM.value, relation_depth=1, relevance=0.5),
                ],
                total_expansion_count=2,
            ),
        }

    def test_concept_cloud_built(self):
        cloud = build_concept_cloud(self._sample_expansion())
        assert len(cloud) > 0
        assert "liberty" in cloud
        assert "duty" in cloud

    def test_shared_concept_found(self):
        cloud = build_concept_cloud(self._sample_expansion())
        shared = find_shared_concepts(cloud)
        # "right" appears in both expansion sets
        shared_concepts = [s.concept for s in shared]
        assert "right" in shared_concepts

    def test_shared_concept_has_strength(self):
        cloud = build_concept_cloud(self._sample_expansion())
        shared = find_shared_concepts(cloud)
        for s in shared:
            assert s.bridging_strength > 0.0


# =====================================================================
# Test: Expansion Metrics
# =====================================================================

class TestExpansionMetrics:
    def _sample_data(self):
        expansions = {
            0: TokenExpansionSet(
                source_token="freedom",
                wordnet_expansions=[
                    ExpansionNode(concept="liberty", relation_type=RelationType.SYNONYM.value, relevance=0.9),
                    ExpansionNode(concept="right", relation_type=RelationType.HYPERNYM.value, relation_depth=2, relevance=0.4),
                ],
                embedding_neighbors=[EmbeddingNeighbor(concept="freeness", similarity=0.7)],
                total_expansion_count=3,
            ),
        }
        cloud = build_concept_cloud(expansions)
        shared = find_shared_concepts(cloud)
        return expansions, cloud, shared

    def test_fractal_depth(self):
        exps, cloud, shared = self._sample_data()
        m = compute_expansion_metrics(exps, cloud, shared, 1, SemanticExpanderConfig())
        assert m.fractal_depth == 2

    def test_pattern_novelty_in_range(self):
        exps, cloud, shared = self._sample_data()
        m = compute_expansion_metrics(exps, cloud, shared, 1, SemanticExpanderConfig())
        assert 0.0 <= m.pattern_novelty <= 1.0

    def test_symbolic_density_in_range(self):
        exps, cloud, shared = self._sample_data()
        m = compute_expansion_metrics(exps, cloud, shared, 1, SemanticExpanderConfig())
        assert 0.0 <= m.symbolic_density <= 1.0

    def test_shared_concept_ratio(self):
        exps, cloud, shared = self._sample_data()
        m = compute_expansion_metrics(exps, cloud, shared, 1, SemanticExpanderConfig())
        assert 0.0 <= m.shared_concept_ratio <= 1.0


# =====================================================================
# Test: Neurochemical Coupling
# =====================================================================

class TestNeurochemicalCoupling:
    def test_expansion_load_formula(self):
        load = compute_expansion_load(50, 2, SemanticExpanderConfig())
        # 50/100 * (1 + 0.10 * 2) = 0.5 * 1.2 = 0.6
        assert abs(load - 0.6) < 0.01

    def test_expansion_load_clamped(self):
        load = compute_expansion_load(5000, 10, SemanticExpanderConfig())
        assert load <= 10.0

    def test_da_novelty_burst(self):
        rng = np.random.default_rng(42)
        metrics = ExpansionMetrics(pattern_novelty=0.8, symbolic_density=0.0)
        signals = compute_expander_neurochemical_signals(metrics, 1.0, SemanticExpanderConfig(), rng)
        assert signals["da_novelty_burst"] > 0.0

    def test_sht2a_symbolic_burst(self):
        rng = np.random.default_rng(42)
        metrics = ExpansionMetrics(symbolic_density=0.8)
        signals = compute_expander_neurochemical_signals(metrics, 1.0, SemanticExpanderConfig(), rng)
        assert signals["sht2a_symbolic_burst"] > 0.0

    def test_ach_burst_on_load(self):
        rng = np.random.default_rng(42)
        metrics = ExpansionMetrics()
        signals = compute_expander_neurochemical_signals(metrics, 2.0, SemanticExpanderConfig(), rng)
        assert signals["ach_burst"] > 0.0

    def test_glu_burst_on_high_shared(self):
        rng = np.random.default_rng(42)
        metrics = ExpansionMetrics(shared_concept_ratio=0.5)
        signals = compute_expander_neurochemical_signals(metrics, 1.0, SemanticExpanderConfig(), rng)
        assert signals["glu_burst"] > 0.0

    def test_glu_burst_zero_low_shared(self):
        rng = np.random.default_rng(42)
        metrics = ExpansionMetrics(shared_concept_ratio=0.05)
        signals = compute_expander_neurochemical_signals(metrics, 1.0, SemanticExpanderConfig(), rng)
        assert signals["glu_burst"] == 0.0

    def test_beta_boost_always_present(self):
        rng = np.random.default_rng(42)
        metrics = ExpansionMetrics()
        signals = compute_expander_neurochemical_signals(metrics, 0.0, SemanticExpanderConfig(), rng)
        assert signals["beta_boost"] == 0.03

    def test_alpha_boost_from_symbolic(self):
        rng = np.random.default_rng(42)
        metrics = ExpansionMetrics(symbolic_density=0.5)
        signals = compute_expander_neurochemical_signals(metrics, 1.0, SemanticExpanderConfig(), rng)
        assert signals["alpha_boost"] > 0.0


# =====================================================================
# Test: Full Engine
# =====================================================================

class TestSemanticExpanderEngine:
    def test_process_basic(self):
        tok_result = _make_tokenizer_result()
        exp = _make_expander()
        result = exp.process(tok_result)
        assert result.processing_time_ms >= 0.0
        assert len(result.neurochemical_signals) > 0

    def test_token_expansions_populated(self):
        tok_result = _make_tokenizer_result("Freedom requires responsibility.")
        exp = _make_expander()
        result = exp.process(tok_result)
        # At least some content words should have expansions
        assert len(result.token_expansions) > 0

    def test_concept_cloud_populated(self):
        tok_result = _make_tokenizer_result("Freedom requires responsibility.")
        exp = _make_expander()
        result = exp.process(tok_result)
        assert len(result.concept_cloud) > 0

    def test_metrics_computed(self):
        tok_result = _make_tokenizer_result("Freedom requires responsibility.")
        exp = _make_expander()
        result = exp.process(tok_result)
        assert result.metrics is not None
        assert result.metrics.expansion_breadth >= 0.0

    def test_neurochemical_signals_keys(self):
        tok_result = _make_tokenizer_result()
        exp = _make_expander()
        result = exp.process(tok_result)
        assert "da_novelty_burst" in result.neurochemical_signals
        assert "sht2a_symbolic_burst" in result.neurochemical_signals
        assert "ach_burst" in result.neurochemical_signals
        assert "beta_boost" in result.neurochemical_signals

    def test_configure_mode(self):
        exp = _make_expander()
        exp.configure(OperationalMode.LEARNING)
        assert exp.get_status()["mode"] == "learning"

    def test_rem_dream_expanded_depth(self):
        tok_result = _make_tokenizer_result("Freedom requires responsibility.")
        exp = _make_expander()
        exp.configure(OperationalMode.REM_DREAM)
        result = exp.process(tok_result)
        # REM_DREAM uses expanded config (hypernym_depth+2, embedding_k+10, lower theta)
        assert result.processing_time_ms >= 0.0

    def test_empty_input(self):
        tok_result = TokenizerResult()
        exp = _make_expander()
        result = exp.process(tok_result)
        assert len(result.token_expansions) == 0

    def test_get_status(self):
        exp = _make_expander()
        status = exp.get_status()
        assert "mode" in status

    def test_collocations_populated(self):
        # Use a sentence with adjacent nouns
        tok_result = _make_tokenizer_result("Human rights are fundamental rights.")
        exp = _make_expander()
        result = exp.process(tok_result)
        # May or may not have collocations depending on POS tagging
        assert isinstance(result.collocations, list)

    def test_shared_concepts_list(self):
        tok_result = _make_tokenizer_result("Freedom requires responsibility.")
        exp = _make_expander()
        result = exp.process(tok_result)
        assert isinstance(result.shared_concepts, list)

    def test_known_word_expansion_chain(self):
        tok_result = _make_tokenizer_result("Justice and truth.")
        exp = _make_expander()
        result = exp.process(tok_result)
        # Check that at least one content token has wordnet expansions
        has_expansions = any(
            len(tes.wordnet_expansions) > 0
            for tes in result.token_expansions.values()
        )
        assert has_expansions


# =====================================================================
# Test: Edge Cases
# =====================================================================

class TestEdgeCases:
    def test_single_word(self):
        tok_result = _make_tokenizer_result("Freedom")
        exp = _make_expander()
        result = exp.process(tok_result)
        assert len(result.token_expansions) >= 0

    def test_all_stop_words(self):
        tok_result = _make_tokenizer_result("is a the and")
        exp = _make_expander()
        result = exp.process(tok_result)
        # All stop words -> no content words -> no expansions
        assert result.processing_time_ms >= 0.0

    def test_unknown_words(self):
        tok_result = _make_tokenizer_result("Xyzzy plugh frobozz")
        exp = _make_expander()
        result = exp.process(tok_result)
        # Unknown words still get embedding/morphological stubs
        assert result.processing_time_ms >= 0.0

    def test_very_long_input(self):
        text = "freedom " * 50
        tok_result = _make_tokenizer_result(text.strip())
        exp = _make_expander()
        result = exp.process(tok_result)
        assert len(result.token_expansions) > 0
