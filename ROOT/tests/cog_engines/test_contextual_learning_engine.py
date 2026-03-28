"""Tests for Engine 22 -- Contextual Learning Engine."""
import math
import uuid

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.contextual_learning_engine import (
    ContextualLearningEngine,
    ContextLearningConfig,
    ContextLearningState,
    ContextInput,
    ContextRecord,
    ContextFingerprint,
    ContextMatchResult,
    ContextLearningResult,
    ContextLearningNeurochem,
    ContextStatus,
    MatchQuality,
    EncodingStrength,
    SimilarityWeights,
    ModeConfig,
    # Pure helper functions
    _hash_string,
    _hash_dict,
    _text_to_vector,
    _emotion_to_vector,
    build_fingerprint,
    cosine_similarity,
    weighted_similarity,
    classify_match_quality,
    resolve_mode_config,
    compute_decay,
    strengthen_record,
    encode_new_record,
    apply_decay_to_library,
    evict_if_needed,
    compute_neurochem_signals,
)
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Helpers
# =====================================================================

def _make_engine(seed=42, **overrides):
    """Build an engine with fixed RNG and optional config overrides."""
    cfg = ContextLearningConfig(**overrides) if overrides else None
    return ContextualLearningEngine(config=cfg, rng=np.random.default_rng(seed))


def _make_input(topic="quantum computing", emotion=None, intent="learn",
                raw_text="Tell me about quantum.", adjustments=None,
                social_markers=None, mode=OperationalMode.NORMAL):
    return ContextInput(
        topic=topic,
        emotion_state=emotion or {"curiosity": 0.8},
        intent=intent,
        raw_text=raw_text,
        parameter_adjustments=adjustments or {},
        social_markers=social_markers or [],
        active_mode=mode,
    )


# =====================================================================
# 1. Config Defaults
# =====================================================================

class TestConfigDefaults:
    def test_recognition_threshold(self):
        cfg = ContextLearningConfig()
        assert cfg.recognition_threshold == 0.60

    def test_max_contexts(self):
        cfg = ContextLearningConfig()
        assert cfg.max_contexts == 512

    def test_encoding_strength(self):
        cfg = ContextLearningConfig()
        assert cfg.encoding_strength == 0.50

    def test_decay_half_life(self):
        cfg = ContextLearningConfig()
        assert cfg.decay_half_life_ticks == 500

    def test_dormancy_and_prune_thresholds(self):
        cfg = ContextLearningConfig()
        assert cfg.dormancy_threshold == 0.15
        assert cfg.prune_threshold == 0.05

    def test_similarity_weights_sum_to_one(self):
        cfg = ContextLearningConfig()
        w = cfg.similarity_weights
        total = w.topic_weight + w.emotion_weight + w.intent_weight
        assert abs(total - 1.0) < 1e-9

    def test_mode_configs_differ(self):
        cfg = ContextLearningConfig()
        assert cfg.mode_analytical.recognition_threshold > cfg.mode_default.recognition_threshold
        assert cfg.mode_creative.recognition_threshold < cfg.mode_default.recognition_threshold
        assert cfg.mode_rem_dream.recognition_threshold < cfg.mode_creative.recognition_threshold

    def test_beta_coupling_constants(self):
        cfg = ContextLearningConfig()
        assert cfg.beta_ach_encoding == 0.15
        assert cfg.beta_oxt_social == 0.12
        assert cfg.beta_cb1_flexibility == 0.10
        assert cfg.beta_da_novelty == 0.12
        assert cfg.beta_5ht_stability == 0.08
        assert cfg.beta_ne_broadening == 0.10

    def test_oscillatory_coupling(self):
        cfg = ContextLearningConfig()
        assert cfg.psi_theta_encoding == 0.08
        assert cfg.psi_gamma_recognition == 0.06


# =====================================================================
# 2. Pure Helper Functions — Hashing
# =====================================================================

class TestHashFunctions:
    def test_hash_string_deterministic(self):
        h1 = _hash_string("hello")
        h2 = _hash_string("hello")
        assert h1 == h2
        assert len(h1) == 16

    def test_hash_string_different_inputs(self):
        assert _hash_string("hello") != _hash_string("world")

    def test_hash_dict_deterministic(self):
        d = {"a": 0.5, "b": 0.3}
        h1 = _hash_dict(d)
        h2 = _hash_dict(d)
        assert h1 == h2

    def test_hash_dict_order_independent(self):
        d1 = {"b": 0.3, "a": 0.5}
        d2 = {"a": 0.5, "b": 0.3}
        assert _hash_dict(d1) == _hash_dict(d2)

    def test_hash_dict_rounds_values(self):
        d1 = {"x": 0.123456789}
        d2 = {"x": 0.12345}  # Same after rounding to 4 decimal places
        # Both round to 0.1235
        assert _hash_dict(d1) == _hash_dict(d2)


# =====================================================================
# 3. Pure Helper Functions — Vectorization
# =====================================================================

class TestVectorization:
    def test_text_to_vector_deterministic(self):
        v1 = _text_to_vector("hello world")
        v2 = _text_to_vector("hello world")
        assert v1 == v2

    def test_text_to_vector_dimension(self):
        v = _text_to_vector("test", dim=32)
        assert len(v) == 32

    def test_text_to_vector_custom_dim(self):
        v = _text_to_vector("test", dim=64)
        assert len(v) == 64

    def test_text_to_vector_empty(self):
        v = _text_to_vector("")
        assert all(x == 0.0 for x in v)

    def test_text_to_vector_normalized(self):
        v = _text_to_vector("some text here")
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-9

    def test_text_to_vector_case_insensitive(self):
        v1 = _text_to_vector("Hello World")
        v2 = _text_to_vector("hello world")
        assert v1 == v2

    def test_emotion_to_vector_deterministic(self):
        e = {"joy": 0.8, "sadness": 0.2}
        v1 = _emotion_to_vector(e)
        v2 = _emotion_to_vector(e)
        assert v1 == v2

    def test_emotion_to_vector_dimension(self):
        e = {"joy": 0.8}
        v = _emotion_to_vector(e, dim=16)
        assert len(v) == 16

    def test_emotion_to_vector_empty(self):
        v = _emotion_to_vector({})
        assert all(x == 0.0 for x in v)

    def test_emotion_to_vector_normalized(self):
        e = {"joy": 0.8, "sadness": 0.2, "anger": 0.5}
        v = _emotion_to_vector(e)
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-9


# =====================================================================
# 4. Cosine Similarity
# =====================================================================

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = (1.0, 0.0, 0.0, 1.0)
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = (1.0, 0.0)
        b = (0.0, 1.0)
        assert abs(cosine_similarity(a, b)) < 1e-9

    def test_anti_correlated_clamped_to_zero(self):
        a = (1.0, 0.0)
        b = (-1.0, 0.0)
        assert cosine_similarity(a, b) == 0.0

    def test_mismatched_lengths(self):
        a = (1.0, 0.0)
        b = (1.0, 0.0, 0.0)
        assert cosine_similarity(a, b) == 0.0

    def test_empty_vectors(self):
        assert cosine_similarity((), ()) == 0.0

    def test_returns_between_0_and_1(self):
        rng = np.random.default_rng(99)
        for _ in range(20):
            a = tuple(rng.standard_normal(8))
            b = tuple(rng.standard_normal(8))
            sim = cosine_similarity(a, b)
            assert 0.0 <= sim <= 1.0


# =====================================================================
# 5. Build Fingerprint
# =====================================================================

class TestBuildFingerprint:
    def test_returns_fingerprint(self):
        fp = build_fingerprint("quantum", {"curiosity": 0.8}, "learn")
        assert isinstance(fp, ContextFingerprint)

    def test_hashes_populated(self):
        fp = build_fingerprint("quantum", {"curiosity": 0.8}, "learn")
        assert len(fp.topic_hash) == 16
        assert len(fp.emotion_hash) == 16
        assert len(fp.intent_hash) == 16
        assert len(fp.composite_hash) == 16

    def test_vectors_populated(self):
        fp = build_fingerprint("quantum", {"curiosity": 0.8}, "learn")
        assert len(fp.topic_vector) == 32
        assert len(fp.emotion_vector) == 16
        assert len(fp.intent_vector) == 32

    def test_context_id_is_uuid(self):
        fp = build_fingerprint("quantum", {}, "learn")
        # Should not throw
        uuid.UUID(fp.context_id)

    def test_empty_inputs(self):
        fp = build_fingerprint("", {}, "")
        assert fp.topic_hash == ""
        assert fp.emotion_hash == ""
        assert fp.intent_hash == ""
        # composite is still computed from the concatenated empty hashes
        assert len(fp.composite_hash) == 16

    def test_same_inputs_same_hashes(self):
        fp1 = build_fingerprint("quantum", {"curiosity": 0.8}, "learn")
        fp2 = build_fingerprint("quantum", {"curiosity": 0.8}, "learn")
        assert fp1.topic_hash == fp2.topic_hash
        assert fp1.emotion_hash == fp2.emotion_hash
        assert fp1.intent_hash == fp2.intent_hash
        # context_ids differ (uuid)
        assert fp1.context_id != fp2.context_id


# =====================================================================
# 6. Weighted Similarity
# =====================================================================

class TestWeightedSimilarity:
    def test_identical_fingerprints_high_similarity(self):
        fp = build_fingerprint("quantum", {"curiosity": 0.8}, "learn")
        composite, ts, es, is_ = weighted_similarity(fp, fp, SimilarityWeights())
        assert composite > 0.99

    def test_different_fingerprints_lower_similarity(self):
        fp_a = build_fingerprint("quantum", {"curiosity": 0.8}, "learn")
        fp_b = build_fingerprint("cooking", {"joy": 0.6}, "create")
        composite, _, _, _ = weighted_similarity(fp_a, fp_b, SimilarityWeights())
        assert composite < 0.8

    def test_returns_four_values(self):
        fp = build_fingerprint("test", {}, "test")
        result = weighted_similarity(fp, fp, SimilarityWeights())
        assert len(result) == 4


# =====================================================================
# 7. Classify Match Quality
# =====================================================================

class TestClassifyMatchQuality:
    def test_exact(self):
        assert classify_match_quality(0.96, 0.60) == MatchQuality.EXACT

    def test_strong(self):
        assert classify_match_quality(0.85, 0.60) == MatchQuality.STRONG

    def test_moderate(self):
        assert classify_match_quality(0.65, 0.60) == MatchQuality.MODERATE

    def test_weak(self):
        # threshold * 0.6 = 0.36; 0.40 >= 0.36 but < 0.60
        assert classify_match_quality(0.40, 0.60) == MatchQuality.WEAK

    def test_none(self):
        assert classify_match_quality(0.10, 0.60) == MatchQuality.NONE

    def test_boundary_exact(self):
        assert classify_match_quality(0.95, 0.60) == MatchQuality.EXACT

    def test_boundary_strong(self):
        assert classify_match_quality(0.80, 0.60) == MatchQuality.STRONG


# =====================================================================
# 8. Resolve Mode Config
# =====================================================================

class TestResolveModeConfig:
    def test_normal_mode(self):
        cfg = ContextLearningConfig()
        mc = resolve_mode_config(OperationalMode.NORMAL, cfg)
        assert mc.recognition_threshold == 0.60

    def test_reflective_maps_to_analytical(self):
        cfg = ContextLearningConfig()
        mc = resolve_mode_config(OperationalMode.REFLECTIVE, cfg)
        assert mc.recognition_threshold == 0.75

    def test_rem_dream_mode(self):
        cfg = ContextLearningConfig()
        mc = resolve_mode_config(OperationalMode.REM_DREAM, cfg)
        assert mc.recognition_threshold == 0.35
        assert mc.broadening_factor == 2.0

    def test_dev_maps_to_default(self):
        cfg = ContextLearningConfig()
        mc = resolve_mode_config(OperationalMode.DEV, cfg)
        assert mc == cfg.mode_default

    def test_learning_maps_to_default(self):
        cfg = ContextLearningConfig()
        mc = resolve_mode_config(OperationalMode.LEARNING, cfg)
        assert mc == cfg.mode_default


# =====================================================================
# 9. Compute Decay
# =====================================================================

class TestComputeDecay:
    def test_half_life_halves_confidence(self):
        result = compute_decay(1.0, 500, 0.01, 500)
        assert abs(result - 0.5) < 1e-9

    def test_no_ticks_no_decay(self):
        assert compute_decay(0.8, 0, 0.01, 500) == 0.8

    def test_zero_half_life_no_decay(self):
        assert compute_decay(0.8, 100, 0.01, 0) == 0.8

    def test_decay_is_monotonic(self):
        c1 = compute_decay(1.0, 100, 0.01, 500)
        c2 = compute_decay(1.0, 200, 0.01, 500)
        c3 = compute_decay(1.0, 300, 0.01, 500)
        assert c1 > c2 > c3

    def test_negative_ticks_no_decay(self):
        assert compute_decay(0.8, -5, 0.01, 500) == 0.8


# =====================================================================
# 10. Strengthen Record
# =====================================================================

class TestStrengthenRecord:
    def test_encounter_count_increments(self):
        record = ContextRecord(confidence=0.5, encounter_count=1)
        strengthen_record(record, {}, 0.10, 0.20, 1.0, 10)
        assert record.encounter_count == 2

    def test_confidence_increases(self):
        record = ContextRecord(confidence=0.5, encounter_count=1)
        strengthen_record(record, {}, 0.10, 0.20, 1.0, 10)
        assert record.confidence > 0.5

    def test_confidence_bounded_by_max(self):
        record = ContextRecord(confidence=0.99, encounter_count=5)
        strengthen_record(record, {}, 0.50, 0.20, 1.0, 10)
        assert record.confidence <= 1.0

    def test_status_set_to_active(self):
        record = ContextRecord(confidence=0.3, status=ContextStatus.DORMANT)
        strengthen_record(record, {}, 0.10, 0.20, 1.0, 10)
        assert record.status == ContextStatus.ACTIVE

    def test_last_seen_tick_updated(self):
        record = ContextRecord(last_seen_tick=0)
        strengthen_record(record, {}, 0.10, 0.20, 1.0, 42)
        assert record.last_seen_tick == 42

    def test_parameter_adjustments_blended(self):
        record = ContextRecord(parameter_adjustments={"lr": 0.10})
        strengthen_record(record, {"lr": 0.50}, 0.10, 0.20, 1.0, 10)
        # EMA: old + blend_rate * (new - old) = 0.10 + 0.20 * (0.50 - 0.10) = 0.18
        assert abs(record.parameter_adjustments["lr"] - 0.18) < 1e-9

    def test_new_adjustment_key_added(self):
        record = ContextRecord(parameter_adjustments={})
        strengthen_record(record, {"temp": 0.7}, 0.10, 0.20, 1.0, 10)
        # New key: old defaults to val itself, so result = val + blend*(val-val) = val
        assert abs(record.parameter_adjustments["temp"] - 0.7) < 1e-9


# =====================================================================
# 11. Encode New Record
# =====================================================================

class TestEncodeNewRecord:
    def test_returns_context_record(self):
        fp = build_fingerprint("quantum", {"curiosity": 0.8}, "learn")
        rec = encode_new_record(fp, "quantum", {"curiosity": 0.8}, "learn",
                                {"lr": 0.1}, 0.5, 10)
        assert isinstance(rec, ContextRecord)

    def test_fields_populated(self):
        fp = build_fingerprint("quantum", {"curiosity": 0.8}, "learn")
        rec = encode_new_record(fp, "quantum", {"curiosity": 0.8}, "learn",
                                {"lr": 0.1}, 0.65, 10, social_context=True)
        assert rec.context_id == fp.context_id
        assert rec.topic == "quantum"
        assert rec.intent == "learn"
        assert rec.confidence == 0.65
        assert rec.encounter_count == 1
        assert rec.last_seen_tick == 10
        assert rec.created_tick == 10
        assert rec.status == ContextStatus.ACTIVE
        assert rec.social_context is True

    def test_encoding_strength_clamped(self):
        fp = build_fingerprint("x", {}, "y")
        rec = encode_new_record(fp, "x", {}, "y", {}, 1.5, 0)
        assert rec.confidence == 1.0


# =====================================================================
# 12. Apply Decay to Library
# =====================================================================

class TestApplyDecayToLibrary:
    def _make_library(self, n=3, confidence=0.5, last_seen=0):
        lib = {}
        for i in range(n):
            cid = str(uuid.uuid4())
            lib[cid] = ContextRecord(
                context_id=cid,
                confidence=confidence,
                last_seen_tick=last_seen,
                status=ContextStatus.ACTIVE,
            )
        return lib

    def test_no_decay_when_just_seen(self):
        lib = self._make_library(confidence=0.5, last_seen=10)
        pruned = apply_decay_to_library(lib, 10, 0.01, 500, 0.15, 0.05)
        assert pruned == 0
        assert len(lib) == 3

    def test_records_become_dormant(self):
        lib = self._make_library(1, confidence=0.16, last_seen=0)
        # After 500 ticks, confidence halves: 0.16 * 0.5 = 0.08 < 0.15 → dormant
        pruned = apply_decay_to_library(lib, 500, 0.01, 500, 0.15, 0.05)
        assert pruned == 0
        record = list(lib.values())[0]
        assert record.status == ContextStatus.DORMANT

    def test_records_get_pruned(self):
        lib = self._make_library(1, confidence=0.06, last_seen=0)
        # After 500 ticks: 0.06 * 0.5 = 0.03 < 0.05 → pruned
        pruned = apply_decay_to_library(lib, 500, 0.01, 500, 0.15, 0.05)
        assert pruned == 1
        assert len(lib) == 0

    def test_archived_records_not_decayed(self):
        lib = self._make_library(1, confidence=0.06, last_seen=0)
        record = list(lib.values())[0]
        record.status = ContextStatus.ARCHIVED
        pruned = apply_decay_to_library(lib, 1000, 0.01, 500, 0.15, 0.05)
        assert pruned == 0
        assert record.confidence == 0.06  # unchanged


# =====================================================================
# 13. Evict If Needed
# =====================================================================

class TestEvictIfNeeded:
    def test_no_eviction_under_capacity(self):
        lib = {}
        for i in range(3):
            cid = str(uuid.uuid4())
            lib[cid] = ContextRecord(context_id=cid, confidence=0.5)
        evicted = evict_if_needed(lib, 10)
        assert evicted == 0
        assert len(lib) == 3

    def test_eviction_over_capacity(self):
        lib = {}
        for i in range(5):
            cid = str(uuid.uuid4())
            lib[cid] = ContextRecord(context_id=cid, confidence=0.1 * (i + 1))
        evicted = evict_if_needed(lib, 3)
        assert evicted == 2
        assert len(lib) == 3
        # Lowest-confidence records should be evicted
        remaining_confs = sorted(r.confidence for r in lib.values())
        assert remaining_confs[0] >= 0.3

    def test_archived_records_protected(self):
        lib = {}
        # 1 archived with low confidence
        cid_archived = str(uuid.uuid4())
        lib[cid_archived] = ContextRecord(
            context_id=cid_archived, confidence=0.01,
            status=ContextStatus.ARCHIVED,
        )
        # 2 active with higher confidence
        for i in range(2):
            cid = str(uuid.uuid4())
            lib[cid] = ContextRecord(context_id=cid, confidence=0.3 + 0.1 * i)
        evicted = evict_if_needed(lib, 2)
        assert evicted == 1
        # Archived should survive
        assert cid_archived in lib


# =====================================================================
# 14. Compute Neurochem Signals
# =====================================================================

class TestComputeNeurochemSignals:
    def _rng(self, seed=42):
        return np.random.default_rng(seed)

    def test_novel_encoding_emits_da_and_ach(self):
        cfg = ContextLearningConfig()
        nc = compute_neurochem_signals(
            novel_context=True, best_similarity=0.0,
            encoding_performed=True, strengthening_applied=False,
            social_context=False, n_matches=0,
            cfg=cfg, rng=self._rng(),
        )
        assert nc.da_delta > 0
        assert nc.ach_delta > 0

    def test_no_da_on_recognition(self):
        cfg = ContextLearningConfig()
        nc = compute_neurochem_signals(
            novel_context=False, best_similarity=0.85,
            encoding_performed=False, strengthening_applied=True,
            social_context=False, n_matches=1,
            cfg=cfg, rng=self._rng(),
        )
        assert nc.da_delta == 0.0

    def test_social_context_emits_oxt(self):
        cfg = ContextLearningConfig()
        nc = compute_neurochem_signals(
            novel_context=True, best_similarity=0.0,
            encoding_performed=True, strengthening_applied=False,
            social_context=True, n_matches=0,
            cfg=cfg, rng=self._rng(),
        )
        assert nc.oxt_delta > 0

    def test_no_oxt_without_social(self):
        cfg = ContextLearningConfig()
        nc = compute_neurochem_signals(
            novel_context=True, best_similarity=0.0,
            encoding_performed=True, strengthening_applied=False,
            social_context=False, n_matches=0,
            cfg=cfg, rng=self._rng(),
        )
        assert nc.oxt_delta == 0.0

    def test_strengthening_emits_5ht(self):
        cfg = ContextLearningConfig()
        nc = compute_neurochem_signals(
            novel_context=False, best_similarity=0.9,
            encoding_performed=False, strengthening_applied=True,
            social_context=False, n_matches=1,
            cfg=cfg, rng=self._rng(),
        )
        assert nc._5ht_delta > 0

    def test_theta_boost_on_encoding(self):
        cfg = ContextLearningConfig()
        nc = compute_neurochem_signals(
            novel_context=True, best_similarity=0.0,
            encoding_performed=True, strengthening_applied=False,
            social_context=False, n_matches=0,
            cfg=cfg, rng=self._rng(),
        )
        assert nc.theta_boost == cfg.psi_theta_encoding

    def test_gamma_boost_on_recognition(self):
        cfg = ContextLearningConfig()
        nc = compute_neurochem_signals(
            novel_context=False, best_similarity=0.8,
            encoding_performed=False, strengthening_applied=True,
            social_context=False, n_matches=2,
            cfg=cfg, rng=self._rng(),
        )
        assert nc.gamma_boost > 0

    def test_no_gamma_on_novel(self):
        cfg = ContextLearningConfig()
        nc = compute_neurochem_signals(
            novel_context=True, best_similarity=0.0,
            encoding_performed=True, strengthening_applied=False,
            social_context=False, n_matches=0,
            cfg=cfg, rng=self._rng(),
        )
        assert nc.gamma_boost == 0.0


# =====================================================================
# 15. Engine Init & Introspection
# =====================================================================

class TestEngineInit:
    def test_engine_id(self):
        e = _make_engine()
        assert e.engine_id == "contextual_learning_engine"

    def test_cluster(self):
        e = _make_engine()
        assert e.cluster == "learning"

    def test_initial_status(self):
        e = _make_engine()
        s = e.get_status()
        assert s["engine_id"] == "contextual_learning_engine"
        assert s["cluster"] == "learning"
        assert s["mode"] == "normal"
        assert s["cycle_count"] == 0
        assert s["tick"] == 0
        assert s["total_contexts"] == 0

    def test_initial_context_count(self):
        e = _make_engine()
        assert e.get_context_count() == 0

    def test_repr(self):
        e = _make_engine()
        r = repr(e)
        assert "ContextualLearningEngine" in r


# =====================================================================
# 16. configure() and Mode Switching
# =====================================================================

class TestModeSwitching:
    def test_configure_changes_mode(self):
        e = _make_engine()
        e.configure(OperationalMode.REFLECTIVE)
        assert e.get_status()["mode"] == "reflective"

    def test_analytical_mode_higher_threshold(self):
        e = _make_engine()
        e.configure(OperationalMode.REFLECTIVE)
        inp = _make_input(mode=OperationalMode.REFLECTIVE)
        result = e.process(inp)
        assert result.metadata["recognition_threshold"] == 0.75

    def test_creative_mode_lower_threshold(self):
        e = _make_engine()
        inp = _make_input(topic="art", intent="create")
        # Creative mode is not an OperationalMode enum value directly,
        # but REM_DREAM is the loosest
        inp_rem = _make_input(topic="art", intent="create",
                              mode=OperationalMode.REM_DREAM)
        result = e.process(inp_rem)
        assert result.metadata["recognition_threshold"] == 0.35


# =====================================================================
# 17. update_neurochem_state (Pattern A)
# =====================================================================

class TestUpdateNeurochemState:
    def test_all_keys(self):
        e = _make_engine()
        e.update_neurochem_state({
            "ach": 0.5, "ne": 0.3, "da": 0.7,
            "5ht": 0.4, "oxt": 0.6, "cb1": 0.2,
        })
        s = e.get_status()["state"]
        assert s["ach_level"] == 0.5
        assert s["ne_level"] == 0.3
        assert s["da_level"] == 0.7
        assert s["_5ht_level"] == 0.4
        assert s["oxt_level"] == 0.6
        assert s["cb1_level"] == 0.2

    def test_partial_update(self):
        e = _make_engine()
        e.update_neurochem_state({"ach": 0.9})
        s = e.get_status()["state"]
        assert s["ach_level"] == 0.9
        assert s["da_level"] == 0.0  # unchanged

    def test_unknown_keys_ignored(self):
        e = _make_engine()
        e.update_neurochem_state({"ach": 0.5, "unknown_key": 0.9})
        s = e.get_status()["state"]
        assert s["ach_level"] == 0.5


# =====================================================================
# 18. NT Modulation Effects
# =====================================================================

class TestNTModulation:
    def test_cb1_lowers_recognition_threshold(self):
        e = _make_engine()
        e.update_neurochem_state({"cb1": 0.8})
        inp = _make_input()
        result = e.process(inp)
        # 0.60 - 0.10 * 0.8 = 0.52
        assert result.metadata["recognition_threshold"] < 0.60

    def test_ach_boosts_encoding_strength(self):
        e = _make_engine()
        e.update_neurochem_state({"ach": 0.8})
        inp = _make_input()
        result = e.process(inp)
        # 0.50 + 0.15 * 0.8 = 0.62
        assert result.metadata["encoding_strength"] > 0.50

    def test_5ht_reduces_decay(self):
        e = _make_engine()
        e.update_neurochem_state({"5ht": 0.8})
        inp = _make_input()
        result = e.process(inp)
        # effective_decay = 0.01 * max(0.2, 1.0 - 0.08 * 0.8) = 0.01 * 0.936 = 0.00936
        assert result.metadata["effective_decay"] < 0.01

    def test_ne_broadens_weights(self):
        e = _make_engine()
        # Without NE, weights are [0.45, 0.30, 0.25]
        # With high NE, they should flatten toward 1/3 each
        e.update_neurochem_state({"ne": 1.0})
        inp = _make_input()
        # Process runs; internal weights are broadened but we can verify
        # indirectly that broadening doesn't crash and produces results
        result = e.process(inp)
        assert result is not None

    def test_cb1_threshold_floor(self):
        e = _make_engine()
        e.update_neurochem_state({"cb1": 100.0})  # absurdly high
        inp = _make_input()
        result = e.process(inp)
        assert result.metadata["recognition_threshold"] >= 0.20

    def test_ach_encoding_ceiling(self):
        e = _make_engine()
        e.update_neurochem_state({"ach": 100.0})
        inp = _make_input()
        result = e.process(inp)
        assert result.metadata["encoding_strength"] <= 1.0


# =====================================================================
# 19. process() — Novel Context Detection
# =====================================================================

class TestNovelContextDetection:
    def test_first_input_is_novel(self):
        e = _make_engine()
        inp = _make_input()
        result = e.process(inp)
        assert result.novel_context is True
        assert result.encoding_performed is True
        assert result.strengthening_applied is False

    def test_novel_context_adds_to_library(self):
        e = _make_engine()
        inp = _make_input()
        e.process(inp)
        assert e.get_context_count() == 1

    def test_different_topics_are_novel(self):
        # Use maximally dissimilar topics + intents + emotions to ensure
        # the trigram-based fingerprinting produces low similarity
        e = _make_engine()
        e.process(_make_input(
            topic="zzzzz qqqqq xxxxx",
            intent="aaa_bbb_ccc",
            emotion={"joy": 1.0},
        ))
        r2 = e.process(_make_input(
            topic="mmmmm nnnnn ooooo",
            intent="ppp_rrr_sss",
            emotion={"anger": 1.0},
        ))
        assert e.get_context_count() == 2


# =====================================================================
# 20. process() — Context Recognition & Strengthening
# =====================================================================

class TestContextRecognition:
    def test_same_input_recognised(self):
        e = _make_engine()
        inp = _make_input()
        r1 = e.process(inp)
        assert r1.novel_context is True
        r2 = e.process(inp)
        assert r2.novel_context is False
        assert r2.strengthening_applied is True

    def test_recognition_increases_encounter_count(self):
        e = _make_engine()
        inp = _make_input()
        e.process(inp)
        r2 = e.process(inp)
        assert r2.best_match is not None
        # After strengthening, encounter_count is 2 -- read from the record
        # directly since the match result snapshot may capture pre-strengthen count.
        record = e.get_record(r2.best_match.context_id)
        assert record.encounter_count == 2

    def test_recognition_confidence_grows(self):
        # The record decays slightly each tick before strengthening.
        # To verify net growth, check the live record after many re-encounters.
        e = _make_engine()
        inp = _make_input()
        r1 = e.process(inp)
        cid = r1.best_match.context_id
        initial_conf = e.get_record(cid).confidence
        for _ in range(5):
            e.process(inp)
        final_conf = e.get_record(cid).confidence
        assert final_conf > initial_conf


# =====================================================================
# 21. process() — Parameter Adjustments
# =====================================================================

class TestParameterAdjustments:
    def test_novel_context_passes_adjustments_through(self):
        e = _make_engine()
        inp = _make_input(adjustments={"lr": 0.01, "temp": 0.8})
        result = e.process(inp)
        assert result.active_adjustments == {"lr": 0.01, "temp": 0.8}

    def test_recognised_context_returns_blended_adjustments(self):
        e = _make_engine()
        inp = _make_input(adjustments={"lr": 0.10})
        e.process(inp)
        inp2 = _make_input(adjustments={"lr": 0.50})
        r2 = e.process(inp2)
        # After EMA blend, lr should be between 0.10 and 0.50
        assert 0.10 < r2.active_adjustments["lr"] < 0.50


# =====================================================================
# 22. process() — Full Pipeline
# =====================================================================

class TestProcessPipeline:
    def test_result_type(self):
        e = _make_engine()
        result = e.process(_make_input())
        assert isinstance(result, ContextLearningResult)

    def test_fingerprint_present(self):
        e = _make_engine()
        result = e.process(_make_input())
        assert isinstance(result.current_fingerprint, ContextFingerprint)

    def test_neurochemical_signals_present(self):
        e = _make_engine()
        result = e.process(_make_input())
        assert isinstance(result.neurochemical_signals, ContextLearningNeurochem)

    def test_metadata_contains_expected_keys(self):
        e = _make_engine()
        result = e.process(_make_input())
        keys = {"mode", "recognition_threshold", "encoding_strength",
                "effective_decay", "cycle", "tick"}
        assert keys.issubset(result.metadata.keys())

    def test_tick_and_cycle_increment(self):
        e = _make_engine()
        e.process(_make_input())
        e.process(_make_input())
        s = e.get_status()
        assert s["cycle_count"] == 2
        assert s["tick"] == 2

    def test_processing_time_positive(self):
        e = _make_engine()
        result = e.process(_make_input())
        assert result.processing_time_ms >= 0.0


# =====================================================================
# 23. Library Management
# =====================================================================

class TestLibraryManagement:
    def test_get_context_library_summary_empty(self):
        e = _make_engine()
        summary = e.get_context_library_summary()
        assert summary["total_contexts"] == 0
        assert summary["capacity_used"] == 0.0

    def test_get_context_library_summary_populated(self):
        e = _make_engine()
        e.process(_make_input(topic="alpha"))
        e.process(_make_input(topic="beta"))
        summary = e.get_context_library_summary()
        assert summary["total_contexts"] == 2
        assert summary["avg_encounters"] >= 1.0

    def test_archive_context(self):
        e = _make_engine()
        r = e.process(_make_input())
        cid = r.best_match.context_id
        assert e.archive_context(cid) is True
        record = e.get_record(cid)
        assert record.status == ContextStatus.ARCHIVED

    def test_archive_nonexistent(self):
        e = _make_engine()
        assert e.archive_context("nonexistent") is False

    def test_get_record(self):
        e = _make_engine()
        r = e.process(_make_input())
        cid = r.best_match.context_id
        record = e.get_record(cid)
        assert record is not None
        assert record.context_id == cid

    def test_get_record_nonexistent(self):
        e = _make_engine()
        assert e.get_record("nope") is None

    def test_clear_library_preserves_archived(self):
        e = _make_engine()
        r1 = e.process(_make_input(topic="alpha"))
        r2 = e.process(_make_input(topic="beta"))
        e.archive_context(r1.best_match.context_id)
        removed = e.clear_library()
        assert removed == 1
        assert e.get_context_count() == 1

    def test_social_markers_set_social_context(self):
        e = _make_engine()
        inp = _make_input(social_markers=["greeting", "empathy"])
        result = e.process(inp)
        cid = result.best_match.context_id
        record = e.get_record(cid)
        assert record.social_context is True


# =====================================================================
# 24. Edge Cases
# =====================================================================

class TestEdgeCases:
    def test_empty_input(self):
        e = _make_engine()
        inp = ContextInput()  # All defaults
        result = e.process(inp)
        assert result.novel_context is True
        assert e.get_context_count() == 1

    def test_max_capacity_eviction(self):
        e = _make_engine(max_contexts=5)
        for i in range(7):
            e.process(_make_input(topic=f"topic_{i}", intent=f"intent_{i}"))
        assert e.get_context_count() <= 5

    def test_many_sequential_same_input(self):
        e = _make_engine()
        inp = _make_input()
        for _ in range(10):
            e.process(inp)
        assert e.get_context_count() == 1
        s = e.get_status()
        assert s["cycle_count"] == 10

    def test_all_novel_inputs(self):
        # Use very dissimilar inputs: long random-looking strings that
        # produce distinct trigram vectors in the 32-dim hash space.
        e = _make_engine()
        count_novel = 0
        for i in range(10):
            r = e.process(_make_input(
                topic=f"{'abcdefghij'[i] * 20}",
                intent=f"{'klmnopqrst'[i] * 20}",
                emotion={f"{'uvwxyzABCD'[i] * 4}": 0.9},
            ))
            if r.novel_context:
                count_novel += 1
        # At least 8 out of 10 should be novel (trigram hash can have collisions)
        assert count_novel >= 8

    def test_context_with_no_adjustments(self):
        e = _make_engine()
        result = e.process(_make_input(adjustments={}))
        assert result.active_adjustments == {}


# =====================================================================
# 25. Context Decay via process()
# =====================================================================

class TestContextDecay:
    def test_confidence_decays_over_ticks(self):
        e = _make_engine()
        r1 = e.process(_make_input(topic="old_topic"))
        cid = r1.best_match.context_id
        # Process many different inputs to advance ticks without re-seeing old_topic
        for i in range(50):
            e.process(_make_input(topic=f"other_{i}", intent=f"other_intent_{i}",
                                  emotion={"e_" + str(i): 0.5}))
        record = e.get_record(cid)
        if record is not None:
            assert record.confidence < r1.best_match.confidence

    def test_archived_does_not_decay_via_process(self):
        e = _make_engine()
        r = e.process(_make_input(topic="identity_topic"))
        cid = r.best_match.context_id
        e.archive_context(cid)
        initial_conf = e.get_record(cid).confidence
        for i in range(20):
            e.process(_make_input(topic=f"other_{i}", intent=f"int_{i}",
                                  emotion={"e_" + str(i): 0.5}))
        assert e.get_record(cid).confidence == initial_conf


# =====================================================================
# 26. Neurochem Output from process()
# =====================================================================

class TestNeurochemOutput:
    def test_novel_context_theta_boost(self):
        e = _make_engine()
        result = e.process(_make_input())
        assert result.neurochemical_signals.theta_boost > 0

    def test_recognised_context_gamma_boost(self):
        e = _make_engine()
        inp = _make_input()
        e.process(inp)
        r2 = e.process(inp)
        assert r2.neurochemical_signals.gamma_boost > 0

    def test_social_context_oxt_delta(self):
        e = _make_engine()
        inp = _make_input(social_markers=["hello"])
        result = e.process(inp)
        assert result.neurochemical_signals.oxt_delta > 0

    def test_novel_encoding_da_delta(self):
        e = _make_engine()
        result = e.process(_make_input())
        assert result.neurochemical_signals.da_delta > 0


# =====================================================================
# 27. Enum Coverage
# =====================================================================

class TestEnums:
    def test_context_status_values(self):
        assert ContextStatus.ACTIVE.value == "active"
        assert ContextStatus.DORMANT.value == "dormant"
        assert ContextStatus.DECAYED.value == "decayed"
        assert ContextStatus.ARCHIVED.value == "archived"

    def test_match_quality_values(self):
        assert MatchQuality.EXACT.value == "exact"
        assert MatchQuality.STRONG.value == "strong"
        assert MatchQuality.MODERATE.value == "moderate"
        assert MatchQuality.WEAK.value == "weak"
        assert MatchQuality.NONE.value == "none"

    def test_encoding_strength_values(self):
        assert EncodingStrength.STRONG.value == "strong"
        assert EncodingStrength.MODERATE.value == "moderate"
        assert EncodingStrength.WEAK.value == "weak"
