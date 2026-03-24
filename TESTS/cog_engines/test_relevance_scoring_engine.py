"""
Tests for Engine 8 — Relevance Scoring Engine.

Coverage:
  - Config & defaults
  - Pure scoring functions
  - NT modulation (ACh threshold, NE threshold, DA novelty, 5-HT stability, GABA threshold)
  - Item registration & lifecycle
  - Context vector cosine scoring
  - Composite relevance computation
  - Threshold filtering (above/below)
  - Mode switching (ANALYTICAL, CREATIVE, REM_DREAM)
  - process() pipeline
  - Neurochem output
  - Edge cases
"""
import math
import pytest

from zados.cognitive_engines.py_engines.relevance_scoring_engine import (
    RelevanceAxisScores,
    RelevanceScoringConfig,
    RelevanceScoringEngine,
    RelevanceScoringNeurochem,
    RelevanceScoringResult,
    ScoredItem,
    _sparse_cosine,
    compute_composite_relevance,
    compute_effective_threshold,
    compute_effective_weights,
    compute_frequency_score,
    compute_novelty_bonus,
    compute_recency_score,
    compute_scoring_neurochem,
)


# =====================================================================
# Config
# =====================================================================

class TestConfig:
    def test_default_weights_sum_to_one(self):
        cfg = RelevanceScoringConfig()
        total = (cfg.w_recency + cfg.w_frequency + cfg.w_semantic_proximity
                 + cfg.w_attention_weight + cfg.w_contextual_fit + cfg.w_novelty_bonus)
        assert abs(total - 1.0) < 0.01

    def test_recency_lambda_computed(self):
        cfg = RelevanceScoringConfig(recency_half_life=50.0)
        expected = math.log(2.0) / 50.0
        assert abs(cfg.recency_lambda - expected) < 1e-9

    def test_custom_half_life(self):
        cfg = RelevanceScoringConfig(recency_half_life=100.0)
        expected = math.log(2.0) / 100.0
        assert abs(cfg.recency_lambda - expected) < 1e-9

    def test_mode_configs_present(self):
        cfg = RelevanceScoringConfig()
        assert "ANALYTICAL" in cfg.mode_configs
        assert "CREATIVE" in cfg.mode_configs
        assert "REM_DREAM" in cfg.mode_configs


# =====================================================================
# Pure scoring functions
# =====================================================================

class TestRecencyScore:
    def test_zero_ticks(self):
        assert compute_recency_score(0, 0.01) == 1.0

    def test_negative_ticks(self):
        assert compute_recency_score(-5, 0.01) == 1.0

    def test_positive_decay(self):
        score = compute_recency_score(50, math.log(2.0) / 50.0)
        assert abs(score - 0.5) < 0.01

    def test_large_ticks_near_zero(self):
        score = compute_recency_score(1000, 0.01)
        assert score < 0.01


class TestFrequencyScore:
    def test_no_history(self):
        assert compute_frequency_score([], 10, 100) == 0.0

    def test_zero_window(self):
        assert compute_frequency_score([1, 2, 3], 10, 0) == 0.0

    def test_all_in_window(self):
        history = list(range(10))
        score = compute_frequency_score(history, 10, 100)
        assert score > 0.0

    def test_capped_at_one(self):
        history = list(range(100))
        score = compute_frequency_score(history, 100, 100)
        assert score <= 1.0


class TestNoveltyBonus:
    def test_high_recency_low_frequency(self):
        # First appearance: high recency (1.0), low frequency (0.0)
        novelty = compute_novelty_bonus(0.0, 1.0, 10.0, 5.0)
        assert novelty == 1.0

    def test_low_recency_high_frequency(self):
        novelty = compute_novelty_bonus(1.0, 0.0, 10.0, 5.0)
        assert novelty == 0.0

    def test_both_moderate(self):
        novelty = compute_novelty_bonus(0.5, 0.5, 10.0, 5.0)
        assert 0.0 < novelty < 1.0


class TestCompositeRelevance:
    def test_all_zeros(self):
        axes = RelevanceAxisScores()
        weights = {"w_recency": 0.2, "w_frequency": 0.15, "w_semantic_proximity": 0.2,
                   "w_attention_weight": 0.2, "w_contextual_fit": 0.15, "w_novelty_bonus": 0.1}
        assert compute_composite_relevance(axes, weights) == 0.0

    def test_all_ones(self):
        axes = RelevanceAxisScores(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        weights = {"w_recency": 0.2, "w_frequency": 0.15, "w_semantic_proximity": 0.2,
                   "w_attention_weight": 0.2, "w_contextual_fit": 0.15, "w_novelty_bonus": 0.1}
        assert abs(compute_composite_relevance(axes, weights) - 1.0) < 0.01

    def test_weighted_correctly(self):
        axes = RelevanceAxisScores(recency=1.0)
        weights = {"w_recency": 0.5, "w_frequency": 0.0, "w_semantic_proximity": 0.0,
                   "w_attention_weight": 0.0, "w_contextual_fit": 0.0, "w_novelty_bonus": 0.0}
        assert abs(compute_composite_relevance(axes, weights) - 0.5) < 0.01


class TestEffectiveThreshold:
    def test_neutral_nt(self):
        # All at 0 → base threshold unchanged
        t = compute_effective_threshold(0.30, 0.0, 0.0, 0.0, 0.25, 0.20, 0.25)
        assert abs(t - 0.30) < 0.01

    def test_high_ach_raises(self):
        base = compute_effective_threshold(0.30, 0.0, 0.0, 0.0, 0.25, 0.20, 0.25)
        high_ach = compute_effective_threshold(0.30, 1.0, 0.0, 0.0, 0.25, 0.20, 0.25)
        assert high_ach > base

    def test_high_ne_lowers(self):
        base = compute_effective_threshold(0.30, 0.0, 0.0, 0.0, 0.25, 0.20, 0.25)
        high_ne = compute_effective_threshold(0.30, 0.0, 1.0, 0.0, 0.25, 0.20, 0.25)
        assert high_ne < base

    def test_high_gaba_raises(self):
        base = compute_effective_threshold(0.30, 0.0, 0.0, 0.0, 0.25, 0.20, 0.25)
        high_gaba = compute_effective_threshold(0.30, 0.0, 0.0, 1.0, 0.25, 0.20, 0.25)
        assert high_gaba > base

    def test_clamped_bounds(self):
        t = compute_effective_threshold(0.30, 1.0, 0.0, 1.0, 5.0, 0.0, 5.0)
        assert 0.05 <= t <= 0.95


class TestEffectiveWeights:
    def test_high_da_boosts_novelty(self):
        base = {"w_recency": 0.2, "w_frequency": 0.15, "w_semantic_proximity": 0.2,
                "w_attention_weight": 0.2, "w_contextual_fit": 0.15, "w_novelty_bonus": 0.1}
        w_low = compute_effective_weights(base, 0.0, 0.5, 0.30, 0.20)
        w_high = compute_effective_weights(base, 1.0, 0.5, 0.30, 0.20)
        # Novelty share should be higher with high DA
        assert w_high["w_novelty_bonus"] > w_low["w_novelty_bonus"]

    def test_5ht_dampens_da_effect(self):
        base = {"w_recency": 0.2, "w_frequency": 0.15, "w_semantic_proximity": 0.2,
                "w_attention_weight": 0.2, "w_contextual_fit": 0.15, "w_novelty_bonus": 0.1}
        w_no_5ht = compute_effective_weights(base, 1.0, 0.0, 0.30, 0.20)
        w_hi_5ht = compute_effective_weights(base, 1.0, 1.0, 0.30, 0.20)
        # With high 5-HT, DA effect on novelty should be dampened
        assert w_hi_5ht["w_novelty_bonus"] < w_no_5ht["w_novelty_bonus"]

    def test_normalised(self):
        base = {"w_recency": 0.2, "w_frequency": 0.15, "w_semantic_proximity": 0.2,
                "w_attention_weight": 0.2, "w_contextual_fit": 0.15, "w_novelty_bonus": 0.1}
        w = compute_effective_weights(base, 0.8, 0.3, 0.30, 0.20)
        assert abs(sum(w.values()) - 1.0) < 0.01


class TestSparseCosineSimilarity:
    def test_identical_vectors(self):
        v = {"a": 1.0, "b": 2.0}
        assert abs(_sparse_cosine(v, v) - 1.0) < 0.01

    def test_orthogonal_vectors(self):
        assert _sparse_cosine({"a": 1.0}, {"b": 1.0}) == 0.0

    def test_empty_vectors(self):
        assert _sparse_cosine({}, {"a": 1.0}) == 0.0
        assert _sparse_cosine({"a": 1.0}, {}) == 0.0


class TestScoringNeurochem:
    def test_novel_items_produce_da(self):
        nc = compute_scoring_neurochem(5, 0, 10, 0.3)
        assert nc.da_delta > 0

    def test_focused_items_produce_ach(self):
        nc = compute_scoring_neurochem(0, 5, 10, 0.3)
        assert nc.ach_delta > 0

    def test_high_mean_produces_5ht(self):
        nc = compute_scoring_neurochem(0, 0, 10, 0.6)
        assert nc._5ht_delta > 0


# =====================================================================
# Engine integration
# =====================================================================

class TestEngineInit:
    def test_default_construction(self):
        e = RelevanceScoringEngine()
        assert e.engine_id == "relevance_scoring_engine"
        assert e.cluster == "pattern_analysis"
        assert e._tick == 0

    def test_get_status(self):
        e = RelevanceScoringEngine()
        s = e.get_status()
        assert s["engine_id"] == "relevance_scoring_engine"
        assert s["tracked_items"] == 0
        assert "nt_levels" in s

    def test_repr(self):
        e = RelevanceScoringEngine()
        assert "RelevanceScoringEngine" in repr(e)


class TestItemManagement:
    def test_register_item(self):
        e = RelevanceScoringEngine()
        e.register_item("item_1", semantic_score=0.7)
        assert "item_1" in e._items
        assert e._items["item_1"].semantic_score == 0.7

    def test_register_duplicate_updates(self):
        e = RelevanceScoringEngine()
        e.register_item("item_1", semantic_score=0.3)
        e.register_item("item_1", semantic_score=0.8)
        assert e._items["item_1"].semantic_score == 0.8

    def test_mark_accessed(self):
        e = RelevanceScoringEngine()
        e.register_item("item_1")
        e.mark_accessed("item_1")
        assert e._items["item_1"].access_count == 1

    def test_mark_accessed_nonexistent(self):
        e = RelevanceScoringEngine()
        e.mark_accessed("ghost")  # Should not raise

    def test_update_signals(self):
        e = RelevanceScoringEngine()
        e.register_item("item_1")
        e.update_item_signals("item_1", semantic_score=0.9, in_af=True)
        assert e._items["item_1"].semantic_score == 0.9
        assert e._items["item_1"].in_af is True

    def test_remove_item(self):
        e = RelevanceScoringEngine()
        e.register_item("item_1")
        e.remove_item("item_1")
        assert "item_1" not in e._items


class TestScoring:
    def test_score_empty(self):
        e = RelevanceScoringEngine()
        result = e.score_all()
        assert len(result.scored_items) == 0
        assert result.mean_relevance == 0.0

    def test_score_single_item(self):
        e = RelevanceScoringEngine()
        e.register_item("item_1", semantic_score=0.8, sti_normalized=0.6)
        result = e.score_all()
        assert len(result.scored_items) == 1
        assert result.scored_items[0].item_id == "item_1"
        assert result.scored_items[0].composite > 0

    def test_recently_accessed_high_recency(self):
        e = RelevanceScoringEngine()
        e.register_item("item_1")
        e.mark_accessed("item_1")
        result = e.score_all()
        # Should have high recency since just accessed
        axes = result.scored_items[0].axes
        assert axes.recency > 0.5

    def test_threshold_filtering(self):
        cfg = RelevanceScoringConfig(relevance_threshold=0.50)
        e = RelevanceScoringEngine(config=cfg)
        e.register_item("high", semantic_score=0.9, sti_normalized=0.9)
        e.register_item("low", semantic_score=0.0, sti_normalized=0.0)
        result = e.score_all()
        above = [s for s in result.scored_items if s.above_threshold]
        below = [s for s in result.scored_items if not s.above_threshold]
        assert result.above_threshold_count == len(above)
        assert result.below_threshold_count == len(below)

    def test_tick_increments(self):
        e = RelevanceScoringEngine()
        e.register_item("a")
        r1 = e.score_all()
        r2 = e.score_all()
        assert r2.tick == r1.tick + 1


class TestContextVector:
    def test_context_vector_scoring(self):
        e = RelevanceScoringEngine()
        e.register_item("item_1", metadata={"features": {"ml": 1.0, "ai": 0.8}})
        e.set_context_vector({"ml": 1.0, "ai": 1.0, "python": 0.5})
        result = e.score_all()
        assert result.scored_items[0].axes.contextual_fit > 0.0

    def test_no_context_vector(self):
        e = RelevanceScoringEngine()
        e.register_item("item_1")
        result = e.score_all()
        assert result.scored_items[0].axes.contextual_fit == 0.0


class TestNTModulation:
    def test_nt_update(self):
        e = RelevanceScoringEngine()
        e.update_neurochem_state({"ach": 0.9, "ne": 0.2, "da": 0.8})
        assert abs(e.ach_level - 0.9) < 0.01
        assert abs(e.ne_level - 0.2) < 0.01
        assert abs(e.da_level - 0.8) < 0.01

    def test_nt_clamps(self):
        e = RelevanceScoringEngine()
        e.update_neurochem_state({"ach": 1.5, "ne": -0.5})
        assert e.ach_level == 1.0
        assert e.ne_level == 0.0

    def test_high_ach_raises_threshold(self):
        e = RelevanceScoringEngine()
        e.register_item("a")
        e.update_neurochem_state({"ach": 0.0, "ne": 0.0, "gaba": 0.0})
        r1 = e.score_all()
        e.update_neurochem_state({"ach": 1.0})
        r2 = e.score_all()
        assert r2.effective_threshold > r1.effective_threshold

    def test_high_ne_lowers_threshold(self):
        e = RelevanceScoringEngine()
        e.register_item("a")
        e.update_neurochem_state({"ach": 0.0, "ne": 0.0, "gaba": 0.0})
        r1 = e.score_all()
        e.update_neurochem_state({"ne": 1.0})
        r2 = e.score_all()
        assert r2.effective_threshold < r1.effective_threshold


class TestModes:
    def test_analytical_mode(self):
        e = RelevanceScoringEngine()
        e.set_mode("ANALYTICAL")
        e.register_item("a", semantic_score=0.5)
        result = e.score_all()
        assert result.effective_threshold > 0.30  # Higher threshold

    def test_creative_mode(self):
        e = RelevanceScoringEngine()
        e.set_mode("CREATIVE")
        e.register_item("a")
        result = e.score_all()
        assert result.effective_threshold < 0.30  # Lower threshold

    def test_unknown_mode_uses_defaults(self):
        e = RelevanceScoringEngine()
        e.set_mode("NONEXISTENT")
        e.register_item("a")
        result = e.score_all()
        # Should use default config
        assert result.tick == 1


class TestProcessPipeline:
    def test_process_with_items(self):
        e = RelevanceScoringEngine()
        result = e.process({
            "items": [
                {"item_id": "a", "semantic_score": 0.8},
                {"item_id": "b", "semantic_score": 0.3},
            ],
        })
        assert len(result["scored_items"]) == 2
        assert "neurochem_signals" in result

    def test_process_with_nt_state(self):
        e = RelevanceScoringEngine()
        e.process({
            "nt_state": {"ach": 0.9, "da": 0.1},
            "items": [{"item_id": "x"}],
        })
        assert abs(e.ach_level - 0.9) < 0.01

    def test_process_with_mode(self):
        e = RelevanceScoringEngine()
        e.process({
            "mode": "CREATIVE",
            "items": [{"item_id": "x"}],
        })
        assert e._mode == "CREATIVE"

    def test_process_with_access(self):
        e = RelevanceScoringEngine()
        e.register_item("a")
        result = e.process({"accessed": ["a"]})
        assert e._items["a"].access_count == 1

    def test_process_with_signals(self):
        e = RelevanceScoringEngine()
        e.register_item("a")
        e.process({
            "signals": [{"item_id": "a", "semantic_score": 0.95}],
        })
        assert abs(e._items["a"].semantic_score - 0.95) < 0.01

    def test_process_with_context_vector(self):
        e = RelevanceScoringEngine()
        e.process({
            "context_vector": {"topic": 1.0},
            "items": [{"item_id": "a"}],
        })
        assert e._context_vector == {"topic": 1.0}

    def test_process_empty(self):
        e = RelevanceScoringEngine()
        result = e.process({})
        assert result["tick"] == 1

    def test_process_none(self):
        e = RelevanceScoringEngine()
        result = e.process(None)
        assert result["tick"] == 1


class TestNeurochemOutput:
    def test_neurochem_as_dict(self):
        nc = RelevanceScoringNeurochem(da_delta=0.1, ach_delta=0.05)
        d = nc.as_dict()
        assert d["da_delta"] == 0.1
        assert d["ach_delta"] == 0.05

    def test_result_contains_neurochem(self):
        e = RelevanceScoringEngine()
        e.register_item("a", sti_normalized=0.8)
        e.update_item_signals("a", in_af=True)
        result = e.score_all()
        assert isinstance(result.neurochem_signals, RelevanceScoringNeurochem)


class TestEdgeCases:
    def test_many_items(self):
        e = RelevanceScoringEngine()
        for i in range(100):
            e.register_item(f"item_{i}", semantic_score=i / 100.0)
        result = e.score_all()
        assert len(result.scored_items) == 100

    def test_access_history_trimming(self):
        cfg = RelevanceScoringConfig(frequency_window=10)
        e = RelevanceScoringEngine(config=cfg)
        e.register_item("a")
        for _ in range(20):
            e.mark_accessed("a")
        assert len(e._items["a"].access_history) <= 10

    def test_processing_time_recorded(self):
        e = RelevanceScoringEngine()
        e.register_item("a")
        result = e.score_all()
        assert result.processing_time_ms >= 0.0

    def test_max_relevance(self):
        e = RelevanceScoringEngine()
        e.register_item("high", semantic_score=1.0, sti_normalized=1.0)
        e.register_item("low", semantic_score=0.0, sti_normalized=0.0)
        result = e.score_all()
        assert result.max_relevance >= result.mean_relevance
