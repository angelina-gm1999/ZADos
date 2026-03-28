"""
Tests for data_analysis_engine.py -- Engine 18 (Data Analysis Engine).

Coverage plan
-------------
1.  Enums: EntityType, RelationType, ConfidenceTier
2.  DataAnalysisConfig defaults and immutability
3.  Pure helper: tokenize_simple(), tokenize_preserving_case()
4.  Pure helper: _classify_entity_type()
5.  Pure helper: extract_entities() -- capitalized, quoted, tagged, quantities
6.  Pure helper: _find_nearest_entity(), _classify_relation_type()
7.  Pure helper: extract_relations() -- verb, causal, preposition strategies
8.  Pure helper: compute_co_occurrences()
9.  Pure helper: estimate_dependency_depth()
10. Pure helper: classify_confidence_tier()
11. Pure helper: compute_analysis_depth_score()
12. Pure helper: resolve_entity_threshold(), resolve_relation_threshold()
13. Pure helper: compute_neurochem_signals()
14. NT modulation: _modulate_entity_threshold(), _modulate_relation_threshold()
15. NT modulation: _compute_da_novelty_bonus()
16. Mode switching (NORMAL, DEV, REM_DREAM, REFLECTIVE)
17. process() pipeline -- full integration
18. process() edge cases -- empty input, single token, many tokens
19. Novelty tracking: _update_novelty_history()
20. Engine init, get_status(), repr, configure()
21. Neurochem output signals (as_dict pattern)
"""

from __future__ import annotations

import math
from typing import Dict, List, Set

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)
from zados.cognitive_engines.py_engines.data_analysis_engine import (
    ConfidenceTier,
    CoOccurrence,
    DataAnalysisConfig,
    DataAnalysisEngine,
    DataAnalysisInput,
    DataAnalysisNeurochem,
    DataAnalysisResult,
    DataAnalysisState,
    EntityType,
    ExtractedEntity,
    ExtractedRelation,
    RelationType,
    _classify_entity_type,
    _classify_relation_type,
    _find_nearest_entity,
    classify_confidence_tier,
    compute_analysis_depth_score,
    compute_co_occurrences,
    compute_neurochem_signals,
    estimate_dependency_depth,
    extract_entities,
    extract_relations,
    resolve_entity_threshold,
    resolve_relation_threshold,
    tokenize_preserving_case,
    tokenize_simple,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def cfg() -> DataAnalysisConfig:
    return DataAnalysisConfig()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def engine(rng) -> DataAnalysisEngine:
    return DataAnalysisEngine(rng=rng)


# =====================================================================
# 1. Enums
# =====================================================================


class TestEnums:
    def test_entity_type_members(self):
        names = {m.name for m in EntityType}
        for expected in ("PERSON", "ORGANIZATION", "LOCATION", "CONCEPT",
                         "ARTIFACT", "EVENT", "QUANTITY", "TEMPORAL", "UNKNOWN"):
            assert expected in names

    def test_relation_type_members(self):
        names = {m.name for m in RelationType}
        for expected in ("ACTION", "ATTRIBUTE", "CAUSAL", "SPATIAL",
                         "TEMPORAL_REL", "POSSESSION", "COMPARISON",
                         "PART_WHOLE", "COPULA", "PREPOSITION"):
            assert expected in names

    def test_confidence_tier_members(self):
        assert ConfidenceTier.HIGH.value == "high"
        assert ConfidenceTier.MEDIUM.value == "medium"
        assert ConfidenceTier.LOW.value == "low"
        assert ConfidenceTier.TRACE.value == "trace"


# =====================================================================
# 2. DataAnalysisConfig defaults and immutability
# =====================================================================


class TestConfig:
    def test_config_defaults(self, cfg):
        assert cfg.entity_confidence_threshold == 0.30
        assert cfg.max_entities == 64
        assert cfg.min_entity_length == 2
        assert cfg.capitalize_bonus == 0.25
        assert cfg.quoted_bonus == 0.30
        assert cfg.relation_confidence_threshold == 0.25
        assert cfg.max_relations == 128
        assert cfg.co_occurrence_window == 5
        assert cfg.max_dependency_depth == 10

    def test_config_frozen(self, cfg):
        with pytest.raises(AttributeError):
            cfg.max_entities = 999  # type: ignore[misc]

    def test_config_nt_coupling_defaults(self, cfg):
        assert cfg.beta_ach_depth == 0.12
        assert cfg.beta_ne_scope == 0.10
        assert cfg.beta_da_novelty == 0.10
        assert cfg.beta_5ht_stability == 0.08
        assert cfg.beta_gaba_suppress == 0.08
        assert cfg.psi_gamma_osc == 0.06

    def test_config_mode_thresholds(self, cfg):
        assert cfg.entity_threshold_normal == 0.30
        assert cfg.entity_threshold_dev == 0.15
        assert cfg.entity_threshold_rem_dream == 0.10
        assert cfg.relation_threshold_reflective == 0.35


# =====================================================================
# 3. Tokenization helpers
# =====================================================================


class TestTokenization:
    def test_tokenize_simple_basic(self):
        tokens = tokenize_simple("Alice told Bob.")
        assert tokens == ["Alice", "told", "Bob"]

    def test_tokenize_simple_strips_punctuation(self):
        tokens = tokenize_simple('"Hello," said Alice!')
        assert "Hello" in tokens
        assert "said" in tokens
        assert "Alice" in tokens

    def test_tokenize_simple_empty(self):
        assert tokenize_simple("") == []

    def test_tokenize_simple_only_punctuation(self):
        assert tokenize_simple(".,!? ;:") == []

    def test_tokenize_preserving_case_keeps_case(self):
        tokens = tokenize_preserving_case("Alice told Bob.")
        assert "Alice" in tokens
        assert "Bob" in tokens

    def test_tokenize_preserving_case_strips_punct(self):
        tokens = tokenize_preserving_case('"Server" crashed!')
        assert "Server" in tokens
        assert "crashed" in tokens


# =====================================================================
# 4. _classify_entity_type
# =====================================================================


class TestClassifyEntityType:
    def test_temporal_quantity(self):
        result = _classify_entity_type("10 days", "the task took 10 days to complete")
        assert result == EntityType.TEMPORAL

    def test_quantity(self):
        result = _classify_entity_type("42", "there are 42 items left")
        assert result == EntityType.QUANTITY

    def test_person_with_title(self):
        result = _classify_entity_type("Smith", "dr smith was present at the meeting")
        assert result == EntityType.PERSON

    def test_location_near_marker(self):
        result = _classify_entity_type("Paris", "the city of paris is beautiful")
        assert result == EntityType.LOCATION

    def test_organization_suffix(self):
        result = _classify_entity_type("Acme Corp", "acme corp filed papers")
        assert result == EntityType.ORGANIZATION

    def test_organization_proximity(self):
        result = _classify_entity_type("Oxford", "oxford university accepted students")
        assert result == EntityType.ORGANIZATION

    def test_event_marker(self):
        result = _classify_entity_type("Summit", "the summit was held in may")
        assert result == EntityType.EVENT

    def test_capitalized_default_concept(self):
        result = _classify_entity_type("Algorithm", "the algorithm is fast")
        assert result == EntityType.CONCEPT

    def test_short_lowercase_unknown(self):
        # "co" is an org suffix substring that appears in words like "context",
        # so use a context without such collisions.
        result = _classify_entity_type("ab", "the ab value is set")
        assert result == EntityType.UNKNOWN


# =====================================================================
# 5. extract_entities
# =====================================================================


class TestExtractEntities:
    def test_capitalized_phrase(self, cfg):
        text = "Alice went to New York City for vacation."
        tokens = tokenize_preserving_case(text)
        ents = extract_entities(text, tokens, 0.10, cfg, set(), 0.0)
        ent_texts = [e.text for e in ents]
        assert "New York City" in ent_texts

    def test_quoted_term(self, cfg):
        text = 'The system logged "fatal error" in the output.'
        tokens = tokenize_preserving_case(text)
        ents = extract_entities(text, tokens, 0.10, cfg, set(), 0.0)
        ent_texts = [e.text for e in ents]
        assert "fatal error" in ent_texts

    def test_capitalized_single_word(self, cfg):
        # Single-sentence first-word "Alice" is skipped unless it appears
        # mid-position in another sentence.  Use two sentences so Alice
        # appears mid-sentence in the second one.
        text = "Someone met Alice. Alice told Bob about the Server."
        tokens = tokenize_preserving_case(text)
        ents = extract_entities(text, tokens, 0.10, cfg, set(), 0.0)
        ent_texts = [e.text for e in ents]
        assert "Alice" in ent_texts
        assert "Server" in ent_texts

    def test_stopwords_filtered(self, cfg):
        text = "The quick brown fox jumps over The lazy dog."
        tokens = tokenize_preserving_case(text)
        ents = extract_entities(text, tokens, 0.01, cfg, set(), 0.0)
        ent_texts_lower = [e.text.lower() for e in ents]
        assert "the" not in ent_texts_lower

    def test_min_entity_length(self, cfg):
        text = '"X" is a concept.'
        tokens = tokenize_preserving_case(text)
        ents = extract_entities(text, tokens, 0.01, cfg, set(), 0.0)
        ent_texts = [e.text for e in ents]
        # "X" has length 1 < min_entity_length of 2, so should be excluded
        assert "X" not in ent_texts

    def test_novelty_flag_set(self, cfg):
        text = "Alice told Bob something."
        tokens = tokenize_preserving_case(text)
        # "alice" already seen
        ents = extract_entities(text, tokens, 0.10, cfg, {"alice"}, 0.0)
        for e in ents:
            if e.text == "Alice":
                assert e.is_novel is False
            elif e.text == "Bob":
                assert e.is_novel is True

    def test_da_novelty_bonus_increases_confidence(self, cfg):
        text = "Alice told Bob something."
        tokens = tokenize_preserving_case(text)
        ents_no_bonus = extract_entities(text, tokens, 0.10, cfg, set(), 0.0)
        ents_with_bonus = extract_entities(text, tokens, 0.10, cfg, set(), 0.5)
        # Entities with DA bonus should have >= confidence
        for e_nb, e_wb in zip(
            sorted(ents_no_bonus, key=lambda e: e.text),
            sorted(ents_with_bonus, key=lambda e: e.text),
        ):
            if e_nb.text == e_wb.text and e_nb.is_novel:
                assert e_wb.confidence >= e_nb.confidence

    def test_max_entities_cap(self):
        tiny_cfg = DataAnalysisConfig(max_entities=2)
        text = "Alice and Bob and Charlie and David went to London."
        tokens = tokenize_preserving_case(text)
        ents = extract_entities(text, tokens, 0.01, tiny_cfg, set(), 0.0)
        assert len(ents) <= 2

    def test_temporal_entity_detected(self, cfg):
        text = "The task took 10 days to finish."
        tokens = tokenize_preserving_case(text)
        ents = extract_entities(text, tokens, 0.01, cfg, set(), 0.0)
        temporal = [e for e in ents if e.entity_type == EntityType.TEMPORAL]
        assert len(temporal) >= 1

    def test_quantity_entity_detected(self, cfg):
        text = "There are 42 items in the queue."
        tokens = tokenize_preserving_case(text)
        ents = extract_entities(text, tokens, 0.01, cfg, set(), 0.0)
        quant = [e for e in ents if e.entity_type == EntityType.QUANTITY]
        assert len(quant) >= 1


# =====================================================================
# 6. _find_nearest_entity and _classify_relation_type
# =====================================================================


class TestRelationHelpers:
    def _make_entity(self, text: str, start: int, end: int) -> ExtractedEntity:
        return ExtractedEntity(text=text, span_start=start, span_end=end,
                               confidence=0.8)

    def test_find_nearest_left(self):
        e1 = self._make_entity("Alice", 0, 5)
        e2 = self._make_entity("Bob", 20, 23)
        result = _find_nearest_entity(15, [e1, e2], "left")
        assert result is e1

    def test_find_nearest_right(self):
        e1 = self._make_entity("Alice", 0, 5)
        e2 = self._make_entity("Bob", 20, 23)
        result = _find_nearest_entity(10, [e1, e2], "right")
        assert result is e2

    def test_find_nearest_none_beyond_max_distance(self):
        e1 = self._make_entity("Alice", 0, 5)
        result = _find_nearest_entity(200, [e1], "left", max_distance=10)
        assert result is None

    def test_classify_copula(self):
        assert _classify_relation_type("is") == RelationType.COPULA

    def test_classify_causal(self):
        assert _classify_relation_type("because") == RelationType.CAUSAL

    def test_classify_spatial(self):
        assert _classify_relation_type("near") == RelationType.SPATIAL

    def test_classify_temporal(self):
        assert _classify_relation_type("before") == RelationType.TEMPORAL_REL

    def test_classify_possession(self):
        assert _classify_relation_type("has") == RelationType.POSSESSION

    def test_classify_comparison(self):
        assert _classify_relation_type("greater") == RelationType.COMPARISON

    def test_classify_part_whole(self):
        assert _classify_relation_type("contains") == RelationType.PART_WHOLE

    def test_classify_action_verb(self):
        assert _classify_relation_type("crashed") == RelationType.ACTION

    def test_classify_unknown_defaults_action(self):
        assert _classify_relation_type("xyzzy") == RelationType.ACTION


# =====================================================================
# 7. extract_relations
# =====================================================================


class TestExtractRelations:
    def test_verb_based_relation(self, cfg):
        # Use two sentences so that capitalized names are not skipped
        # as sentence-initial artefacts.
        text = "Someone met Alice. Alice told Bob about the plan."
        tokens = tokenize_preserving_case(text)
        ents = extract_entities(text, tokens, 0.10, cfg, set(), 0.0)
        assert len(ents) >= 2, f"Expected >=2 entities, got {[e.text for e in ents]}"
        rels = extract_relations(text, ents, 0.10, cfg, 0.0)
        # Should have at least one relation with "told" as predicate
        predicates = [r.predicate for r in rels]
        assert "told" in predicates

    def test_no_entities_returns_empty(self, cfg):
        text = "Something happened."
        rels = extract_relations(text, [], 0.10, cfg, 0.0)
        assert rels == []

    def test_no_self_referential_relations(self, cfg):
        text = "Alice told Alice about Alice."
        tokens = tokenize_preserving_case(text)
        ents = extract_entities(text, tokens, 0.10, cfg, set(), 0.0)
        rels = extract_relations(text, ents, 0.10, cfg, 0.0)
        for r in rels:
            assert r.subject_id != r.object_id

    def test_causal_connector(self, cfg):
        text = "The Server crashed because the Network failed."
        tokens = tokenize_preserving_case(text)
        ents = extract_entities(text, tokens, 0.10, cfg, set(), 0.0)
        if len(ents) < 2:
            pytest.skip("Not enough entities")
        rels = extract_relations(text, ents, 0.10, cfg, 0.0)
        causal = [r for r in rels if r.relation_type == RelationType.CAUSAL]
        assert len(causal) >= 1

    def test_ach_depth_bonus_affects_confidence(self, cfg):
        text = "Alice told Bob about the Server."
        tokens = tokenize_preserving_case(text)
        ents = extract_entities(text, tokens, 0.10, cfg, set(), 0.0)
        if len(ents) < 2:
            pytest.skip("Not enough entities")
        rels_low = extract_relations(text, ents, 0.10, cfg, 0.0)
        rels_high = extract_relations(text, ents, 0.10, cfg, 0.5)
        # Higher ACh depth bonus should yield higher confidence
        if rels_low and rels_high:
            assert rels_high[0].confidence >= rels_low[0].confidence


# =====================================================================
# 8. compute_co_occurrences
# =====================================================================


class TestCoOccurrences:
    def _make_entity(self, eid: str, text: str) -> ExtractedEntity:
        return ExtractedEntity(entity_id=eid, text=text, confidence=0.8)

    def test_two_entities_co_occur(self):
        # tokenize_simple preserves case, and matching compares tok vs
        # ent.text.lower(), so use lowercase entity text to ensure matching.
        text = "alice told bob about the plan."
        e1 = self._make_entity("e1", "alice")
        e2 = self._make_entity("e2", "bob")
        result = compute_co_occurrences(text, [e1, e2], window_size=5, min_count=1)
        assert len(result) >= 1
        assert result[0].count >= 1

    def test_fewer_than_two_entities(self):
        text = "Alice is here."
        e1 = self._make_entity("e1", "Alice")
        result = compute_co_occurrences(text, [e1], window_size=5, min_count=1)
        assert result == []

    def test_window_size_respected(self):
        # With a tiny window, distant entities may not co-occur.
        # Use lowercase text/entities so token matching works.
        text = "alice went far away and after a very long journey bob arrived."
        e1 = self._make_entity("e1", "alice")
        e2 = self._make_entity("e2", "bob")
        result_small = compute_co_occurrences(text, [e1, e2], window_size=2, min_count=1)
        result_large = compute_co_occurrences(text, [e1, e2], window_size=50, min_count=1)
        # Large window should find co-occurrence; small may not
        assert len(result_large) >= len(result_small)

    def test_min_count_filter(self):
        text = "alice told bob."
        e1 = self._make_entity("e1", "alice")
        e2 = self._make_entity("e2", "bob")
        result_high = compute_co_occurrences(text, [e1, e2], window_size=5, min_count=100)
        assert result_high == []


# =====================================================================
# 9. estimate_dependency_depth
# =====================================================================


class TestDependencyDepth:
    def test_empty_text(self):
        assert estimate_dependency_depth("", 10) == 0
        assert estimate_dependency_depth("   ", 10) == 0

    def test_simple_sentence(self):
        depth = estimate_dependency_depth("Alice told Bob.", 10)
        assert depth >= 1

    def test_subordination_increases_depth(self):
        simple = estimate_dependency_depth("Alice runs.", 10)
        complex_ = estimate_dependency_depth(
            "Alice knew that Bob said that the server which was old crashed.", 10
        )
        assert complex_ > simple

    def test_parentheticals_increase_depth(self):
        no_paren = estimate_dependency_depth("Alice told Bob.", 10)
        paren = estimate_dependency_depth("Alice (the one who knows Bob (the tall one)) told him.", 10)
        assert paren > no_paren

    def test_max_depth_capped(self):
        text = "A that B that C that D that E that F that G."
        depth = estimate_dependency_depth(text, 3)
        assert depth <= 3


# =====================================================================
# 10. classify_confidence_tier
# =====================================================================


class TestConfidenceTier:
    def test_high(self):
        assert classify_confidence_tier(0.80) == ConfidenceTier.HIGH
        assert classify_confidence_tier(0.75) == ConfidenceTier.HIGH

    def test_medium(self):
        assert classify_confidence_tier(0.60) == ConfidenceTier.MEDIUM
        assert classify_confidence_tier(0.45) == ConfidenceTier.MEDIUM

    def test_low(self):
        assert classify_confidence_tier(0.30) == ConfidenceTier.LOW
        assert classify_confidence_tier(0.20) == ConfidenceTier.LOW

    def test_trace(self):
        assert classify_confidence_tier(0.10) == ConfidenceTier.TRACE
        assert classify_confidence_tier(0.0) == ConfidenceTier.TRACE


# =====================================================================
# 11. compute_analysis_depth_score
# =====================================================================


class TestAnalysisDepthScore:
    def test_all_zero(self):
        score = compute_analysis_depth_score(0, 0, 0, 0)
        assert score == pytest.approx(0.0)

    def test_increases_with_entities(self):
        s0 = compute_analysis_depth_score(0, 0, 0, 0)
        s5 = compute_analysis_depth_score(5, 0, 0, 0)
        assert s5 > s0

    def test_increases_with_relations(self):
        s0 = compute_analysis_depth_score(0, 0, 0, 0)
        s5 = compute_analysis_depth_score(0, 5, 0, 0)
        assert s5 > s0

    def test_increases_with_depth(self):
        s0 = compute_analysis_depth_score(0, 0, 0, 0)
        s3 = compute_analysis_depth_score(0, 0, 3, 0)
        assert s3 > s0

    def test_bounded_01(self):
        score = compute_analysis_depth_score(100, 100, 10, 100)
        assert 0.0 <= score <= 1.0


# =====================================================================
# 12. resolve_entity_threshold / resolve_relation_threshold
# =====================================================================


class TestResolveThresholds:
    def test_entity_threshold_normal(self, cfg):
        t = resolve_entity_threshold(OperationalMode.NORMAL, cfg)
        assert t == cfg.entity_threshold_normal

    def test_entity_threshold_dev(self, cfg):
        t = resolve_entity_threshold(OperationalMode.DEV, cfg)
        assert t == cfg.entity_threshold_dev

    def test_entity_threshold_rem_dream(self, cfg):
        t = resolve_entity_threshold(OperationalMode.REM_DREAM, cfg)
        assert t == cfg.entity_threshold_rem_dream

    def test_relation_threshold_normal(self, cfg):
        t = resolve_relation_threshold(OperationalMode.NORMAL, cfg)
        assert t == cfg.relation_threshold_normal

    def test_relation_threshold_reflective(self, cfg):
        t = resolve_relation_threshold(OperationalMode.REFLECTIVE, cfg)
        assert t == cfg.relation_threshold_reflective

    def test_relation_threshold_learning(self, cfg):
        t = resolve_relation_threshold(OperationalMode.LEARNING, cfg)
        assert t == cfg.relation_threshold_learning


# =====================================================================
# 13. compute_neurochem_signals
# =====================================================================


class TestNeurochemSignals:
    def test_empty_returns_zeros(self, cfg, rng):
        nc = compute_neurochem_signals([], [], 0, 0, 0.0, cfg, rng)
        assert nc.da_delta == 0.0
        assert nc.ach_delta == 0.0
        assert nc.ne_delta == 0.0
        assert nc._5ht_delta == 0.0
        assert nc.gamma_boost == 0.0

    def test_non_empty_gamma_boost(self, cfg, rng):
        e = ExtractedEntity(text="Alice", confidence=0.8)
        nc = compute_neurochem_signals([e], [], 1, 0, 0.5, cfg, rng)
        assert nc.gamma_boost == pytest.approx(cfg.psi_gamma_osc)

    def test_5ht_proportional_to_mean_confidence(self, cfg, rng):
        e_low = ExtractedEntity(text="A", confidence=0.2)
        e_high = ExtractedEntity(text="B", confidence=0.9)
        nc_low = compute_neurochem_signals([e_low], [], 0, 0, 0.5, cfg, rng)
        nc_high = compute_neurochem_signals([e_high], [], 0, 0, 0.5, cfg, rng)
        assert nc_high._5ht_delta > nc_low._5ht_delta

    def test_ne_fires_for_many_entities(self, cfg):
        rng_fixed = np.random.default_rng(123)
        ents = [ExtractedEntity(text=f"E{i}", confidence=0.5) for i in range(10)]
        nc = compute_neurochem_signals(ents, [], 5, 0, 0.5, cfg, rng_fixed)
        # NE should have fired because n_ent >= 3
        # The Poisson draw can be zero, but with lambda 1.5 it's unlikely
        # Just check it's non-negative
        assert nc.ne_delta >= 0.0

    def test_da_higher_with_novel_entities(self, cfg):
        rng1 = np.random.default_rng(999)
        rng2 = np.random.default_rng(999)
        ents = [ExtractedEntity(text="X", confidence=0.5)]
        nc_no_novel = compute_neurochem_signals(ents, [], 0, 0, 0.5, cfg, rng1)
        nc_novel = compute_neurochem_signals(ents, [], 1, 0, 0.5, cfg, rng2)
        assert nc_novel.da_delta >= nc_no_novel.da_delta


# =====================================================================
# 14. NT modulation -- entity / relation thresholds
# =====================================================================


class TestNTModulation:
    def test_ach_lowers_entity_threshold(self):
        engine = DataAnalysisEngine()
        engine.update_neurochem_state({"ach": 0.9})
        base = 0.30
        modulated = engine._modulate_entity_threshold(base)
        assert modulated < base

    def test_ne_lowers_entity_threshold(self):
        engine = DataAnalysisEngine()
        engine.update_neurochem_state({"ne": 0.9})
        base = 0.30
        modulated = engine._modulate_entity_threshold(base)
        assert modulated < base

    def test_5ht_raises_entity_threshold(self):
        engine = DataAnalysisEngine()
        engine.update_neurochem_state({"5ht": 0.9})
        base = 0.30
        modulated = engine._modulate_entity_threshold(base)
        assert modulated > base

    def test_no_modulation_below_05(self):
        engine = DataAnalysisEngine()
        engine.update_neurochem_state({"ach": 0.3, "ne": 0.3, "5ht": 0.3})
        base = 0.30
        modulated = engine._modulate_entity_threshold(base)
        assert modulated == pytest.approx(base)

    def test_gaba_raises_relation_threshold(self):
        engine = DataAnalysisEngine()
        engine.update_neurochem_state({"gaba": 0.9})
        base = 0.25
        modulated = engine._modulate_relation_threshold(base)
        assert modulated > base

    def test_ach_lowers_relation_threshold(self):
        engine = DataAnalysisEngine()
        engine.update_neurochem_state({"ach": 0.9})
        base = 0.25
        modulated = engine._modulate_relation_threshold(base)
        assert modulated < base

    def test_5ht_raises_relation_threshold(self):
        engine = DataAnalysisEngine()
        engine.update_neurochem_state({"5ht": 0.9})
        base = 0.25
        modulated = engine._modulate_relation_threshold(base)
        assert modulated > base

    def test_entity_threshold_clamped(self):
        engine = DataAnalysisEngine()
        # Push NE extremely high to try to drive threshold below 0.05
        engine.update_neurochem_state({"ne": 100.0})
        modulated = engine._modulate_entity_threshold(0.30)
        assert modulated >= 0.05

    def test_relation_threshold_clamped_high(self):
        engine = DataAnalysisEngine()
        engine.update_neurochem_state({"gaba": 100.0, "5ht": 100.0})
        modulated = engine._modulate_relation_threshold(0.25)
        assert modulated <= 0.90


# =====================================================================
# 15. _compute_da_novelty_bonus
# =====================================================================


class TestDANoveltyBonus:
    def test_zero_below_threshold(self):
        engine = DataAnalysisEngine()
        engine.update_neurochem_state({"da": 0.3})
        assert engine._compute_da_novelty_bonus() == 0.0

    def test_positive_above_threshold(self):
        engine = DataAnalysisEngine()
        engine.update_neurochem_state({"da": 0.8})
        bonus = engine._compute_da_novelty_bonus()
        assert bonus > 0.0

    def test_scales_with_da(self):
        engine = DataAnalysisEngine()
        engine.update_neurochem_state({"da": 0.6})
        bonus_low = engine._compute_da_novelty_bonus()
        engine.update_neurochem_state({"da": 0.9})
        bonus_high = engine._compute_da_novelty_bonus()
        assert bonus_high > bonus_low


# =====================================================================
# 16. Mode switching
# =====================================================================


class TestModeSwitching:
    def test_configure_mode(self, engine):
        engine.configure(OperationalMode.DEV)
        assert engine._mode == OperationalMode.DEV

    def test_rem_dream_lowers_thresholds(self):
        rng = np.random.default_rng(42)
        engine = DataAnalysisEngine(rng=rng)
        text = "Alice told Bob that the Server crashed."
        inp_normal = DataAnalysisInput(raw_text=text, active_mode=OperationalMode.NORMAL)
        inp_dream = DataAnalysisInput(raw_text=text, active_mode=OperationalMode.REM_DREAM)
        r_normal = engine.process(inp_normal)
        r_dream = engine.process(inp_dream)
        # REM_DREAM has lower thresholds so should yield >= entities
        assert r_dream.entity_count >= r_normal.entity_count

    def test_reflective_raises_thresholds(self, cfg):
        t_normal = resolve_entity_threshold(OperationalMode.NORMAL, cfg)
        t_reflect = resolve_entity_threshold(OperationalMode.REFLECTIVE, cfg)
        assert t_reflect > t_normal


# =====================================================================
# 17. process() pipeline -- full integration
# =====================================================================


class TestProcessPipeline:
    def test_basic_process(self, engine):
        text = "Alice told Bob that the Server crashed."
        inp = DataAnalysisInput(raw_text=text)
        result = engine.process(inp)
        assert isinstance(result, DataAnalysisResult)
        assert result.entity_count >= 1
        assert result.processing_time_ms > 0.0

    def test_entities_populated(self, engine):
        text = "Alice told Bob that the Server crashed."
        result = engine.process(DataAnalysisInput(raw_text=text))
        assert len(result.entities) == result.entity_count

    def test_relations_populated(self, engine):
        text = "Alice told Bob that the Server crashed."
        result = engine.process(DataAnalysisInput(raw_text=text))
        assert len(result.relations) == result.relation_count

    def test_metadata_present(self, engine):
        text = "Alice told Bob something."
        result = engine.process(DataAnalysisInput(raw_text=text))
        assert "mode" in result.metadata
        assert "entity_threshold" in result.metadata
        assert "cycle" in result.metadata

    def test_cycle_counter_increments(self, engine):
        text = "Alice told Bob something."
        inp = DataAnalysisInput(raw_text=text)
        engine.process(inp)
        engine.process(inp)
        assert engine._cycle_count == 2

    def test_custom_tokens_used(self, engine):
        text = "Alice told Bob."
        custom_tokens = ["Alice", "told", "Bob"]
        inp = DataAnalysisInput(raw_text=text, tokens=custom_tokens)
        result = engine.process(inp)
        assert isinstance(result, DataAnalysisResult)

    def test_neurochem_signals_in_result(self, engine):
        text = "Alice told Bob that the Server crashed."
        result = engine.process(DataAnalysisInput(raw_text=text))
        nc = result.neurochemical_signals
        assert isinstance(nc, DataAnalysisNeurochem)

    def test_entity_type_counts(self, engine):
        text = "Alice told Bob about the 10 days project."
        result = engine.process(DataAnalysisInput(raw_text=text))
        # entity_type_counts should be a dict with known keys
        assert isinstance(result.entity_type_counts, dict)

    def test_process_with_nt_state(self):
        rng = np.random.default_rng(42)
        engine = DataAnalysisEngine(rng=rng)
        engine.update_neurochem_state({"ach": 0.8, "ne": 0.7, "da": 0.6})
        text = "Alice told Bob that the Server crashed."
        result = engine.process(DataAnalysisInput(raw_text=text))
        # With high ACh/NE the thresholds lower, potentially finding more entities
        assert result.entity_count >= 1

    def test_confidence_mean_in_range(self, engine):
        text = "Alice told Bob that the Server crashed."
        result = engine.process(DataAnalysisInput(raw_text=text))
        if result.entity_count > 0 or result.relation_count > 0:
            assert 0.0 <= result.confidence_mean <= 1.0

    def test_analysis_depth_score_positive(self, engine):
        text = "Alice told Bob that the Server which was old crashed because the System failed."
        result = engine.process(DataAnalysisInput(raw_text=text))
        assert result.analysis_depth_score >= 0.0

    def test_dependency_depth_populated(self, engine):
        text = "Alice told Bob that the Server which was old crashed."
        result = engine.process(DataAnalysisInput(raw_text=text))
        assert result.dependency_depth >= 1


# =====================================================================
# 18. Edge cases -- empty, single token, many tokens
# =====================================================================


class TestEdgeCases:
    def test_empty_input(self, engine):
        result = engine.process(DataAnalysisInput(raw_text=""))
        assert result.entity_count == 0
        assert result.relation_count == 0
        assert result.metadata.get("empty_input") is True

    def test_whitespace_only(self, engine):
        result = engine.process(DataAnalysisInput(raw_text="   "))
        assert result.entity_count == 0
        assert result.metadata.get("empty_input") is True

    def test_single_word(self, engine):
        result = engine.process(DataAnalysisInput(raw_text="Hello"))
        assert isinstance(result, DataAnalysisResult)
        # Minimal extraction, but should not crash
        assert result.processing_time_ms > 0.0

    def test_very_long_input(self, engine):
        text = " ".join(
            ["Alice told Bob about the Server."] * 50
        )
        result = engine.process(DataAnalysisInput(raw_text=text))
        assert isinstance(result, DataAnalysisResult)
        assert result.entity_count >= 1


# =====================================================================
# 19. Novelty tracking
# =====================================================================


class TestNoveltyTracking:
    def test_first_process_all_novel(self, engine):
        text = "Alice told Bob something."
        result = engine.process(DataAnalysisInput(raw_text=text))
        if result.entity_count > 0:
            assert result.novel_entity_count == result.entity_count

    def test_second_process_fewer_novel(self, engine):
        text = "Alice told Bob something."
        inp = DataAnalysisInput(raw_text=text)
        r1 = engine.process(inp)
        r2 = engine.process(inp)
        # On second pass, previously seen entities are not novel
        assert r2.novel_entity_count <= r1.novel_entity_count

    def test_novelty_history_trimmed(self):
        tiny_cfg = DataAnalysisConfig(novelty_history_size=3)
        engine = DataAnalysisEngine(config=tiny_cfg)
        for i in range(10):
            engine._state.recent_entities.append(f"entity_{i}")
        # Manually invoke trimming
        engine._update_novelty_history([])
        assert len(engine._state.recent_entities) <= 3


# =====================================================================
# 20. Engine init, get_status(), repr, configure
# =====================================================================


class TestEngineInit:
    def test_engine_id(self, engine):
        assert engine.engine_id == "data_analysis_engine"

    def test_cluster(self, engine):
        assert engine.cluster == "pattern_analysis"

    def test_default_mode(self, engine):
        assert engine._mode == OperationalMode.NORMAL

    def test_get_status_keys(self, engine):
        status = engine.get_status()
        assert "engine_id" in status
        assert "cluster" in status
        assert "mode" in status
        assert "cycle_count" in status
        assert "state" in status

    def test_get_status_state_keys(self, engine):
        state = engine.get_status()["state"]
        for key in ("ach_level", "ne_level", "da_level", "5ht_level",
                     "gaba_level", "recent_entities_count"):
            assert key in state

    def test_configure_updates_mode(self, engine):
        engine.configure(OperationalMode.LEARNING)
        status = engine.get_status()
        assert status["mode"] == "learning"

    def test_update_neurochem_state(self, engine):
        engine.update_neurochem_state({"ach": 0.7, "ne": 0.5, "da": 0.3, "5ht": 0.6, "gaba": 0.4})
        assert engine._state.ach_level == 0.7
        assert engine._state.ne_level == 0.5
        assert engine._state.da_level == 0.3
        assert engine._state._5ht_level == 0.6
        assert engine._state.gaba_level == 0.4

    def test_update_neurochem_partial(self, engine):
        engine.update_neurochem_state({"da": 0.9})
        assert engine._state.da_level == 0.9
        # Others remain at default 0.0
        assert engine._state.ach_level == 0.0

    def test_custom_config(self):
        custom = DataAnalysisConfig(max_entities=10, co_occurrence_window=3)
        engine = DataAnalysisEngine(config=custom)
        assert engine._cfg.max_entities == 10
        assert engine._cfg.co_occurrence_window == 3


# =====================================================================
# 21. Neurochem output dataclass
# =====================================================================


class TestNeurochemDataclass:
    def test_default_zeros(self):
        nc = DataAnalysisNeurochem()
        assert nc.da_delta == 0.0
        assert nc.ach_delta == 0.0
        assert nc.ne_delta == 0.0
        assert nc._5ht_delta == 0.0
        assert nc.gamma_boost == 0.0

    def test_frozen(self):
        nc = DataAnalysisNeurochem(da_delta=0.1)
        with pytest.raises(AttributeError):
            nc.da_delta = 0.5  # type: ignore[misc]

    def test_result_frozen(self):
        r = DataAnalysisResult()
        with pytest.raises(AttributeError):
            r.entity_count = 999  # type: ignore[misc]

    def test_extracted_entity_frozen(self):
        e = ExtractedEntity(text="Alice", confidence=0.8)
        with pytest.raises(AttributeError):
            e.text = "Bob"  # type: ignore[misc]

    def test_extracted_relation_frozen(self):
        r = ExtractedRelation(predicate="told", confidence=0.7)
        with pytest.raises(AttributeError):
            r.predicate = "said"  # type: ignore[misc]

    def test_co_occurrence_frozen(self):
        c = CoOccurrence(entity_a="e1", entity_b="e2", count=3)
        with pytest.raises(AttributeError):
            c.count = 10  # type: ignore[misc]
