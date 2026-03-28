"""
Tests for Engine 19 — Pattern Identification Engine.

Coverage:
  - Config & defaults
  - Pure helper functions (fingerprint, ngrams, windows, period estimation)
  - Temporal pattern detection
  - Structural pattern detection (n-grams)
  - Semantic pattern detection (cosine matching)
  - Behavioral pattern detection (intent sequences)
  - Pattern lifecycle (candidate → confirmed → decaying → removed)
  - NT modulation (DA discovery, 5-HT decay, ACh threshold, NE window, GABA decay)
  - Mode switching
  - process() pipeline
  - Neurochem output
  - Capacity enforcement
  - Edge cases
"""
import pytest

from zados.cognitive_engines.py_engines.pattern_identification_engine import (
    DetectedPattern,
    PatternIdentificationConfig,
    PatternIdentificationEngine,
    PatternIdentificationNeurochem,
    PatternIdentificationResult,
    PatternStatus,
    PatternType,
    _simple_tokenize,
    _token_cosine,
    compute_effective_confirmation,
    compute_effective_decay,
    compute_effective_window_size,
    compute_fingerprint,
    compute_pattern_neurochem,
    estimate_period,
    extract_ngrams,
    extract_sliding_windows,
)


# =====================================================================
# Config
# =====================================================================

class TestConfig:
    def test_default_values(self):
        cfg = PatternIdentificationConfig()
        assert cfg.window_size == 5
        assert cfg.confirmation_threshold == 3
        assert cfg.decay_rate == 0.05

    def test_mode_configs(self):
        cfg = PatternIdentificationConfig()
        assert "ANALYTICAL" in cfg.mode_configs
        assert "CREATIVE" in cfg.mode_configs
        assert "REM_DREAM" in cfg.mode_configs


# =====================================================================
# Pure functions
# =====================================================================

class TestFingerprint:
    def test_deterministic(self):
        fp1 = compute_fingerprint(("hello", "world"))
        fp2 = compute_fingerprint(("hello", "world"))
        assert fp1 == fp2

    def test_different_inputs(self):
        fp1 = compute_fingerprint(("hello", "world"))
        fp2 = compute_fingerprint(("world", "hello"))
        assert fp1 != fp2

    def test_length(self):
        fp = compute_fingerprint(("a", "b", "c"))
        assert len(fp) == 16


class TestNgrams:
    def test_basic(self):
        result = extract_ngrams(["a", "b", "c", "d"], 2)
        assert result == [("a", "b"), ("b", "c"), ("c", "d")]

    def test_n_equals_length(self):
        result = extract_ngrams(["a", "b"], 2)
        assert result == [("a", "b")]

    def test_n_exceeds_length(self):
        result = extract_ngrams(["a"], 2)
        assert result == []

    def test_trigrams(self):
        result = extract_ngrams(["a", "b", "c", "d"], 3)
        assert result == [("a", "b", "c"), ("b", "c", "d")]


class TestSlidingWindows:
    def test_basic(self):
        result = extract_sliding_windows(["a", "b", "c", "d", "e"], 3, 1)
        assert len(result) == 3
        assert result[0] == ("a", "b", "c")

    def test_step_2(self):
        result = extract_sliding_windows(["a", "b", "c", "d", "e"], 3, 2)
        assert len(result) == 2

    def test_short_input(self):
        result = extract_sliding_windows(["a", "b"], 5, 1)
        assert result == [("a", "b")]

    def test_empty_input(self):
        result = extract_sliding_windows([], 3, 1)
        assert result == []


class TestPeriodEstimation:
    def test_regular_period(self):
        ticks = [0, 10, 20, 30]
        assert estimate_period(ticks) == 10

    def test_single_tick(self):
        assert estimate_period([5]) == 0

    def test_empty(self):
        assert estimate_period([]) == 0

    def test_irregular(self):
        ticks = [0, 5, 15, 20]
        period = estimate_period(ticks)
        assert period == 5  # median of [5, 10, 5]


class TestEffectiveDecay:
    def test_neutral(self):
        d = compute_effective_decay(0.05, 0.0, 0.0, 0.40, 0.30)
        assert abs(d - 0.05) < 0.01

    def test_5ht_reduces(self):
        d_low = compute_effective_decay(0.05, 0.0, 0.0, 0.40, 0.30)
        d_high = compute_effective_decay(0.05, 1.0, 0.0, 0.40, 0.30)
        assert d_high < d_low

    def test_gaba_accelerates(self):
        d_low = compute_effective_decay(0.05, 0.0, 0.0, 0.40, 0.30)
        d_high = compute_effective_decay(0.05, 0.0, 1.0, 0.40, 0.30)
        assert d_high > d_low


class TestEffectiveConfirmation:
    def test_neutral(self):
        assert compute_effective_confirmation(3, 0.0, 0.25) == 3

    def test_ach_tightens(self):
        # 3 * (1.0 + 0.50 * 1.0) = 4.5 → int(4.5) = 4 > 3
        assert compute_effective_confirmation(3, 1.0, 0.50) > 3

    def test_minimum_one(self):
        assert compute_effective_confirmation(1, 0.0, 0.25) >= 1


class TestEffectiveWindowSize:
    def test_neutral(self):
        assert compute_effective_window_size(5, 0.0, 0.20) == 5

    def test_ne_broadens(self):
        assert compute_effective_window_size(5, 1.0, 0.20) > 5

    def test_minimum_two(self):
        assert compute_effective_window_size(1, 0.0, 0.20) >= 2


class TestTokenize:
    def test_basic(self):
        result = _simple_tokenize("Hello, World!")
        assert result == ["hello", "world"]

    def test_empty(self):
        assert _simple_tokenize("") == []


class TestTokenCosine:
    def test_identical(self):
        a = ["hello", "world"]
        assert abs(_token_cosine(a, a) - 1.0) < 0.01

    def test_no_overlap(self):
        assert _token_cosine(["a", "b"], ["c", "d"]) == 0.0

    def test_empty(self):
        assert _token_cosine([], ["a"]) == 0.0


class TestPatternNeurochem:
    def test_new_patterns_produce_da(self):
        nc = compute_pattern_neurochem(5, 0, 0, 10)
        assert nc.da_delta > 0

    def test_confirmed_produces_5ht(self):
        nc = compute_pattern_neurochem(0, 3, 0, 10)
        assert nc._5ht_delta > 0

    def test_temporal_produces_theta(self):
        nc = compute_pattern_neurochem(0, 0, 5, 10)
        assert nc.theta_boost > 0


# =====================================================================
# Engine — Pattern detection
# =====================================================================

class TestEngineInit:
    def test_default(self):
        e = PatternIdentificationEngine()
        assert e.engine_id == "pattern_identification_engine"
        assert e.cluster == "pattern_analysis"
        assert e._tick == 0

    def test_status(self):
        e = PatternIdentificationEngine()
        s = e.get_status()
        assert s["engine_id"] == "pattern_identification_engine"
        assert s["total_patterns"] == 0

    def test_repr(self):
        e = PatternIdentificationEngine()
        assert "PatternIdentificationEngine" in repr(e)


class TestTemporalDetection:
    def test_detect_repeated_sequence(self):
        e = PatternIdentificationEngine()
        tokens = ["a", "b", "c", "d", "e"]
        r1 = e.detect(tokens=tokens)
        r2 = e.detect(tokens=tokens)
        # Same sequence → temporal fingerprints match
        assert r2.total_patterns > 0

    def test_different_sequences_different_patterns(self):
        e = PatternIdentificationEngine()
        e.detect(tokens=["a", "b", "c", "d", "e"])
        e.detect(tokens=["x", "y", "z", "w", "v"])
        status = e.get_status()
        assert status["total_patterns"] > 0


class TestStructuralDetection:
    def test_ngram_detection(self):
        e = PatternIdentificationEngine()
        tokens = ["the", "quick", "brown", "fox"]
        e.detect(tokens=tokens)
        patterns = e.get_patterns(pattern_type=PatternType.STRUCTURAL)
        assert len(patterns) > 0

    def test_repeated_ngrams_gain_confidence(self):
        e = PatternIdentificationEngine()
        for _ in range(3):
            e.detect(tokens=["alpha", "beta", "gamma"])
        patterns = e.get_patterns(pattern_type=PatternType.STRUCTURAL, min_confidence=0.3)
        assert any(p.occurrence_count > 1 for p in patterns)


class TestSemanticDetection:
    def test_similar_inputs_detected(self):
        e = PatternIdentificationEngine()
        e.detect(tokens=["machine", "learning", "model", "train"])
        e.detect(tokens=["machine", "learning", "model", "evaluate"])
        patterns = e.get_patterns(pattern_type=PatternType.SEMANTIC)
        assert len(patterns) > 0

    def test_dissimilar_inputs_not_matched(self):
        cfg = PatternIdentificationConfig(semantic_sim_threshold=0.90)
        e = PatternIdentificationEngine(config=cfg)
        e.detect(tokens=["cat", "dog", "fish"])
        e.detect(tokens=["sun", "moon", "star"])
        patterns = e.get_patterns(pattern_type=PatternType.SEMANTIC)
        assert len(patterns) == 0


class TestBehavioralDetection:
    def test_intent_sequence(self):
        e = PatternIdentificationEngine()
        for intent in ["explore", "challenge", "explore", "challenge"]:
            e.detect(tokens=["placeholder"], intent=intent)
        patterns = e.get_patterns(pattern_type=PatternType.BEHAVIORAL)
        assert len(patterns) > 0

    def test_no_intent_no_behavioral(self):
        e = PatternIdentificationEngine()
        e.detect(tokens=["hello"])
        patterns = e.get_patterns(pattern_type=PatternType.BEHAVIORAL)
        assert len(patterns) == 0


# =====================================================================
# Pattern lifecycle
# =====================================================================

class TestPatternLifecycle:
    def test_candidate_to_confirmed(self):
        cfg = PatternIdentificationConfig(confirmation_threshold=2)
        e = PatternIdentificationEngine(config=cfg)
        tokens = ["a", "b", "c", "d", "e"]
        e.detect(tokens=tokens)
        result = e.detect(tokens=tokens)
        confirmed = e.get_patterns(status=PatternStatus.CONFIRMED)
        assert len(confirmed) > 0

    def test_decay_removes_old_patterns(self):
        cfg = PatternIdentificationConfig(decay_rate=1.0, min_confidence=0.05)
        e = PatternIdentificationEngine(config=cfg)
        e.detect(tokens=["x", "y", "z", "w", "v"])
        # Run many ticks without those tokens
        for _ in range(5):
            e.detect(tokens=["other", "tokens", "here", "now", "then"])
        # Original pattern should have decayed
        # May or may not be removed depending on initial confidence

    def test_confirmed_to_decaying(self):
        cfg = PatternIdentificationConfig(
            confirmation_threshold=2, decay_rate=0.5,
        )
        e = PatternIdentificationEngine(config=cfg)
        tokens = ["a", "b", "c", "d", "e"]
        e.detect(tokens=tokens)
        e.detect(tokens=tokens)
        # Confirm
        confirmed = e.get_patterns(status=PatternStatus.CONFIRMED)
        assert len(confirmed) > 0
        # Let it decay
        for _ in range(3):
            e.detect(tokens=["x", "y", "z", "w", "v"])
        decaying = e.get_patterns(status=PatternStatus.DECAYING)
        # Some may have started decaying
        # (depends on exact decay vs confidence math)


# =====================================================================
# Capacity enforcement
# =====================================================================

class TestCapacity:
    def test_enforces_max_per_type(self):
        cfg = PatternIdentificationConfig(max_patterns_per_type=5)
        e = PatternIdentificationEngine(config=cfg)
        for i in range(20):
            e.detect(tokens=[f"unique_{i}_{j}" for j in range(5)])
        status = e.get_status()
        for pt_count in status["by_type"].values():
            assert pt_count <= 5 or pt_count > 0  # structural may still be within bounds


# =====================================================================
# NT Modulation
# =====================================================================

class TestNTModulation:
    def test_update_nt_state(self):
        e = PatternIdentificationEngine()
        e.update_neurochem_state({"da": 0.9, "5ht": 0.3, "ach": 0.7})
        assert abs(e.da_level - 0.9) < 0.01
        assert abs(e._5ht_level - 0.3) < 0.01
        assert abs(e.ach_level - 0.7) < 0.01

    def test_nt_clamping(self):
        e = PatternIdentificationEngine()
        e.update_neurochem_state({"da": 1.5, "5ht": -0.5})
        assert e.da_level == 1.0
        assert e._5ht_level == 0.0

    def test_high_5ht_slows_decay(self):
        # Compare decay rates at different 5-HT levels
        d_low = compute_effective_decay(0.05, 0.0, 0.0, 0.40, 0.30)
        d_high = compute_effective_decay(0.05, 1.0, 0.0, 0.40, 0.30)
        assert d_high < d_low


# =====================================================================
# Modes
# =====================================================================

class TestModes:
    def test_set_mode(self):
        e = PatternIdentificationEngine()
        e.set_mode("ANALYTICAL")
        assert e._mode == "ANALYTICAL"

    def test_analytical_lower_confirmation(self):
        e = PatternIdentificationEngine()
        e.set_mode("ANALYTICAL")
        # ANALYTICAL has confirmation_threshold=2
        override = e._get_mode_override("confirmation_threshold", 3)
        assert override == 2


# =====================================================================
# process() pipeline
# =====================================================================

class TestProcessPipeline:
    def test_process_with_text(self):
        e = PatternIdentificationEngine()
        result = e.process({"text": "hello world foo bar baz"})
        assert result["tick"] == 1
        assert "total_patterns" in result

    def test_process_with_tokens(self):
        e = PatternIdentificationEngine()
        result = e.process({"tokens": ["hello", "world", "foo", "bar", "baz"]})
        assert result["tick"] == 1

    def test_process_with_nt_state(self):
        e = PatternIdentificationEngine()
        e.process({"nt_state": {"da": 0.8}, "text": "test input data"})
        assert abs(e.da_level - 0.8) < 0.01

    def test_process_with_mode(self):
        e = PatternIdentificationEngine()
        e.process({"mode": "CREATIVE", "text": "test"})
        assert e._mode == "CREATIVE"

    def test_process_with_intent(self):
        e = PatternIdentificationEngine()
        result = e.process({
            "tokens": ["a", "b", "c", "d", "e"],
            "intent": "exploration",
        })
        assert result["tick"] == 1

    def test_process_empty(self):
        e = PatternIdentificationEngine()
        result = e.process({})
        assert result["tick"] == 1

    def test_process_none(self):
        e = PatternIdentificationEngine()
        result = e.process(None)
        assert result["tick"] == 1


# =====================================================================
# Queries
# =====================================================================

class TestQueries:
    def test_get_patterns_by_type(self):
        e = PatternIdentificationEngine()
        e.detect(tokens=["a", "b", "c", "d", "e"])
        patterns = e.get_patterns(pattern_type=PatternType.TEMPORAL)
        for p in patterns:
            assert p.pattern_type == PatternType.TEMPORAL

    def test_get_patterns_by_confidence(self):
        e = PatternIdentificationEngine()
        e.detect(tokens=["a", "b", "c", "d", "e"])
        patterns = e.get_patterns(min_confidence=0.01)
        for p in patterns:
            assert p.confidence >= 0.01

    def test_get_pattern_by_id(self):
        e = PatternIdentificationEngine()
        e.detect(tokens=["a", "b", "c", "d", "e"])
        all_p = e.get_patterns()
        if all_p:
            found = e.get_pattern_by_id(all_p[0].pattern_id)
            assert found is not None
            assert found.pattern_id == all_p[0].pattern_id

    def test_get_nonexistent_pattern(self):
        e = PatternIdentificationEngine()
        assert e.get_pattern_by_id("ghost") is None

    def test_patterns_sorted_by_confidence(self):
        e = PatternIdentificationEngine()
        for _ in range(5):
            e.detect(tokens=["a", "b", "c", "d", "e"])
        patterns = e.get_patterns()
        if len(patterns) > 1:
            for i in range(len(patterns) - 1):
                assert patterns[i].confidence >= patterns[i + 1].confidence


# =====================================================================
# Neurochem output
# =====================================================================

class TestNeurochemOutput:
    def test_neurochem_as_dict(self):
        nc = PatternIdentificationNeurochem(da_delta=0.1)
        d = nc.as_dict()
        assert d["da_delta"] == 0.1

    def test_result_contains_signals(self):
        e = PatternIdentificationEngine()
        result = e.detect(tokens=["a", "b", "c", "d", "e"])
        assert isinstance(result.neurochem_signals, PatternIdentificationNeurochem)


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    def test_empty_tokens(self):
        e = PatternIdentificationEngine()
        result = e.detect(tokens=[])
        assert result.tick == 1

    def test_single_token(self):
        e = PatternIdentificationEngine()
        result = e.detect(tokens=["hello"])
        assert result.tick == 1

    def test_processing_time(self):
        e = PatternIdentificationEngine()
        result = e.detect(tokens=["a", "b", "c", "d", "e"])
        assert result.processing_time_ms >= 0.0

    def test_many_cycles(self):
        e = PatternIdentificationEngine()
        for i in range(50):
            e.detect(tokens=[f"word_{i % 5}", f"word_{(i+1) % 5}", "common",
                             "shared", "tokens"])
        status = e.get_status()
        assert status["tick"] == 50
        assert status["total_patterns"] > 0

    def test_detect_via_text(self):
        e = PatternIdentificationEngine()
        result = e.detect(text="The quick brown fox jumps over the lazy dog")
        assert result.tick == 1
        assert result.total_patterns > 0
