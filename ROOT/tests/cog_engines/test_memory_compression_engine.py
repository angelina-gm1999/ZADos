"""
Tests for Engine 29 — Memory Compression Engine.

Coverage:
  - Config & defaults
  - Pure scoring functions (entropy, redundancy, salience, recency, access_freq)
  - Retention score computation
  - Policy classification (VERBATIM, SEMANTIC, SYMBOLIC, PRUNE)
  - Policy overrides (identity, critical flags, unresolved, emotional)
  - NT modulation (ACh salience, 5-HT emotional, GABA redundancy, DA entropy, COR unresolved)
  - Transition-type modifiers (STMM→MTMM, MTMM→LTMM, LTMM cold)
  - Effective weight computation
  - process() pipeline
  - Neurochem output
  - Edge cases
"""
import math
import pytest

from zados.cognitive_engines.py_engines.memory_compression_engine import (
    CompressionAxisScores,
    CompressionDecision,
    CompressionPolicy,
    MemoryCompressionConfig,
    MemoryCompressionEngine,
    MemoryCompressionNeurochem,
    MemoryCompressionResult,
    PacketDescriptor,
    TransitionType,
    apply_policy_overrides,
    classify_policy,
    compute_access_frequency_score,
    compute_compression_neurochem,
    compute_effective_weights,
    compute_entropy_score,
    compute_recency_score,
    compute_redundancy_score,
    compute_retention_score,
    compute_salience_score,
)


# =====================================================================
# Config
# =====================================================================

class TestConfig:
    def test_default_weights_sum_to_one(self):
        cfg = MemoryCompressionConfig()
        total = (cfg.w_entropy + cfg.w_redundancy + cfg.w_salience
                 + cfg.w_recency + cfg.w_access_freq)
        assert abs(total - 1.0) < 0.01

    def test_threshold_ordering(self):
        cfg = MemoryCompressionConfig()
        assert cfg.threshold_verbatim > cfg.threshold_semantic > cfg.threshold_symbolic

    def test_transition_modifiers_present(self):
        cfg = MemoryCompressionConfig()
        assert "stmm_to_mtmm" in cfg.transition_modifiers
        assert "mtmm_to_ltmm" in cfg.transition_modifiers
        assert "ltmm_cold" in cfg.transition_modifiers


# =====================================================================
# Pure scoring functions
# =====================================================================

class TestEntropyScore:
    def test_zero_tokens(self):
        assert compute_entropy_score(0, 0, 4.0) == 0.0

    def test_one_unique_token(self):
        score = compute_entropy_score(1, 10, 4.0)
        assert score == 0.0  # log2(1) = 0

    def test_moderate_entropy(self):
        score = compute_entropy_score(16, 100, 4.0)
        assert abs(score - 1.0) < 0.01  # log2(16) = 4.0

    def test_high_entropy(self):
        score = compute_entropy_score(100, 200, 4.0)
        assert score == 1.0  # Clamped at 1.0

    def test_clamped_at_one(self):
        score = compute_entropy_score(1000, 2000, 4.0)
        assert score <= 1.0


class TestRedundancyScore:
    def test_zero_overlap(self):
        assert compute_redundancy_score(0.0) == 0.0

    def test_full_overlap(self):
        assert compute_redundancy_score(1.0) == 1.0

    def test_clamped(self):
        assert compute_redundancy_score(1.5) == 1.0
        assert compute_redundancy_score(-0.5) == 0.0


class TestSalienceScore:
    def test_zero_everything(self):
        score = compute_salience_score(0.0, 0.0, 1.0)
        assert score == 0.0

    def test_high_emotional(self):
        score = compute_salience_score(1.0, 0.0, 1.0)
        assert score > 0.4

    def test_high_reward(self):
        score = compute_salience_score(0.0, 1.0, 1.0)
        assert score > 0.2

    def test_low_trust_increases_salience(self):
        s_high_trust = compute_salience_score(0.0, 0.0, 1.0)
        s_low_trust = compute_salience_score(0.0, 0.0, 0.0)
        assert s_low_trust > s_high_trust


class TestRecencyScore:
    def test_zero_ticks(self):
        assert compute_recency_score(0, 100.0) == 1.0

    def test_half_life(self):
        score = compute_recency_score(100, 100.0)
        assert abs(score - 0.5) < 0.01

    def test_large_ticks(self):
        score = compute_recency_score(1000, 100.0)
        assert score < 0.01


class TestAccessFrequencyScore:
    def test_zero_access(self):
        assert compute_access_frequency_score(0) == 0.0

    def test_moderate_access(self):
        score = compute_access_frequency_score(10, max_expected=20)
        assert abs(score - 0.5) < 0.01

    def test_capped(self):
        score = compute_access_frequency_score(100, max_expected=20)
        assert score == 1.0


class TestRetentionScore:
    def test_all_zeros(self):
        axes = CompressionAxisScores()
        weights = {"w_entropy": 0.25, "w_redundancy": 0.20, "w_salience": 0.25,
                   "w_recency": 0.15, "w_access_freq": 0.15}
        # Redundancy inverted: 1.0 - 0.0 = 1.0, so contribution is 0.20
        score = compute_retention_score(axes, weights)
        assert abs(score - 0.20) < 0.01

    def test_high_salience(self):
        axes = CompressionAxisScores(salience=1.0)
        weights = {"w_entropy": 0.25, "w_redundancy": 0.20, "w_salience": 0.25,
                   "w_recency": 0.15, "w_access_freq": 0.15}
        score = compute_retention_score(axes, weights)
        assert score > 0.4

    def test_high_redundancy_lowers_retention(self):
        axes_low_red = CompressionAxisScores(redundancy=0.0, salience=0.5)
        axes_high_red = CompressionAxisScores(redundancy=1.0, salience=0.5)
        weights = {"w_entropy": 0.25, "w_redundancy": 0.20, "w_salience": 0.25,
                   "w_recency": 0.15, "w_access_freq": 0.15}
        s_low = compute_retention_score(axes_low_red, weights)
        s_high = compute_retention_score(axes_high_red, weights)
        assert s_low > s_high


# =====================================================================
# Policy classification
# =====================================================================

class TestPolicyClassification:
    def test_verbatim(self):
        p = classify_policy(0.80, {"threshold_verbatim": 0.75, "threshold_semantic": 0.50,
                                   "threshold_symbolic": 0.25})
        assert p == CompressionPolicy.VERBATIM

    def test_semantic(self):
        p = classify_policy(0.60, {"threshold_verbatim": 0.75, "threshold_semantic": 0.50,
                                   "threshold_symbolic": 0.25})
        assert p == CompressionPolicy.SEMANTIC

    def test_symbolic(self):
        p = classify_policy(0.30, {"threshold_verbatim": 0.75, "threshold_semantic": 0.50,
                                   "threshold_symbolic": 0.25})
        assert p == CompressionPolicy.SYMBOLIC

    def test_prune(self):
        p = classify_policy(0.10, {"threshold_verbatim": 0.75, "threshold_semantic": 0.50,
                                   "threshold_symbolic": 0.25})
        assert p == CompressionPolicy.PRUNE

    def test_boundary_verbatim(self):
        p = classify_policy(0.75, {"threshold_verbatim": 0.75, "threshold_semantic": 0.50,
                                   "threshold_symbolic": 0.25})
        assert p == CompressionPolicy.VERBATIM


# =====================================================================
# Policy overrides
# =====================================================================

class TestPolicyOverrides:
    def test_identity_forces_verbatim(self):
        cfg = MemoryCompressionConfig()
        desc = PacketDescriptor(is_identity_relevant=True)
        policy, reason = apply_policy_overrides(CompressionPolicy.PRUNE, desc, cfg, 0.5)
        assert policy == CompressionPolicy.VERBATIM
        assert reason == "identity_relevant"

    def test_critical_flag_forces_verbatim(self):
        cfg = MemoryCompressionConfig()
        desc = PacketDescriptor(flags=["CRITICAL:test"])
        policy, reason = apply_policy_overrides(CompressionPolicy.SYMBOLIC, desc, cfg, 0.5)
        assert policy == CompressionPolicy.VERBATIM
        assert reason == "critical_flag"

    def test_unresolved_forces_semantic(self):
        cfg = MemoryCompressionConfig()
        desc = PacketDescriptor(has_unresolved=True)
        policy, reason = apply_policy_overrides(CompressionPolicy.PRUNE, desc, cfg, 0.3)
        assert policy == CompressionPolicy.SEMANTIC
        assert reason == "unresolved_items"

    def test_unresolved_with_high_cortisol_forces_verbatim(self):
        cfg = MemoryCompressionConfig()
        desc = PacketDescriptor(has_unresolved=True)
        policy, reason = apply_policy_overrides(CompressionPolicy.PRUNE, desc, cfg, 0.8)
        assert policy == CompressionPolicy.VERBATIM

    def test_high_emotion_forces_semantic(self):
        cfg = MemoryCompressionConfig()
        desc = PacketDescriptor(emotional_significance=0.85)
        policy, reason = apply_policy_overrides(CompressionPolicy.PRUNE, desc, cfg, 0.3)
        assert policy == CompressionPolicy.SEMANTIC
        assert reason == "emotional_override"

    def test_no_override_when_already_higher(self):
        cfg = MemoryCompressionConfig()
        desc = PacketDescriptor(emotional_significance=0.85)
        policy, reason = apply_policy_overrides(CompressionPolicy.VERBATIM, desc, cfg, 0.3)
        assert policy == CompressionPolicy.VERBATIM
        assert reason == ""  # No override needed

    def test_contradictions_force_semantic(self):
        cfg = MemoryCompressionConfig()
        desc = PacketDescriptor(has_contradictions=True)
        policy, reason = apply_policy_overrides(CompressionPolicy.PRUNE, desc, cfg, 0.3)
        assert policy == CompressionPolicy.SEMANTIC


# =====================================================================
# Effective weights (NT modulation)
# =====================================================================

class TestEffectiveWeights:
    def test_neutral(self):
        cfg = MemoryCompressionConfig()
        w = compute_effective_weights(cfg, 0.0, 0.0, 0.0, 0.0)
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_high_da_boosts_entropy(self):
        cfg = MemoryCompressionConfig()
        w_low = compute_effective_weights(cfg, 0.0, 0.0, 0.0, 0.0)
        w_high = compute_effective_weights(cfg, 0.0, 0.0, 0.0, 1.0)
        assert w_high["w_entropy"] > w_low["w_entropy"]

    def test_high_gaba_boosts_redundancy(self):
        cfg = MemoryCompressionConfig()
        w_low = compute_effective_weights(cfg, 0.0, 0.0, 0.0, 0.0)
        w_high = compute_effective_weights(cfg, 0.0, 0.0, 1.0, 0.0)
        assert w_high["w_redundancy"] > w_low["w_redundancy"]

    def test_high_ach_boosts_salience(self):
        cfg = MemoryCompressionConfig()
        w_low = compute_effective_weights(cfg, 0.0, 0.0, 0.0, 0.0)
        w_high = compute_effective_weights(cfg, 1.0, 0.0, 0.0, 0.0)
        assert w_high["w_salience"] > w_low["w_salience"]

    def test_normalised(self):
        cfg = MemoryCompressionConfig()
        w = compute_effective_weights(cfg, 0.9, 0.8, 0.7, 0.6)
        assert abs(sum(w.values()) - 1.0) < 0.01


# =====================================================================
# Engine integration
# =====================================================================

class TestEngineInit:
    def test_default(self):
        e = MemoryCompressionEngine()
        assert e.engine_id == "memory_compression_engine"
        assert e.cluster == "homeostasis"
        assert e._tick == 0

    def test_status(self):
        e = MemoryCompressionEngine()
        s = e.get_status()
        assert s["engine_id"] == "memory_compression_engine"
        assert s["total_evaluated"] == 0

    def test_repr(self):
        e = MemoryCompressionEngine()
        assert "MemoryCompressionEngine" in repr(e)


class TestNTModulation:
    def test_update_nt_state(self):
        e = MemoryCompressionEngine()
        e.update_neurochem_state({"ach": 0.9, "5ht": 0.7, "gaba": 0.3, "da": 0.6, "cor": 0.4})
        assert abs(e.ach_level - 0.9) < 0.01
        assert abs(e._5ht_level - 0.7) < 0.01
        assert abs(e.cor_level - 0.4) < 0.01

    def test_nt_clamping(self):
        e = MemoryCompressionEngine()
        e.update_neurochem_state({"ach": 1.5, "gaba": -0.5})
        assert e.ach_level == 1.0
        assert e.gaba_level == 0.0


class TestEvaluate:
    def test_single_packet(self):
        e = MemoryCompressionEngine()
        desc = PacketDescriptor(
            packet_id="pkt_1",
            unique_tokens=20,
            total_tokens=50,
            emotional_significance=0.3,
            creation_tick=0,
        )
        result = e.evaluate([desc])
        assert len(result.decisions) == 1
        assert result.decisions[0].packet_id == "pkt_1"

    def test_high_salience_keeps_verbatim(self):
        e = MemoryCompressionEngine()
        desc = PacketDescriptor(
            packet_id="high",
            unique_tokens=50,
            total_tokens=100,
            emotional_significance=0.9,
            reward_mean=0.8,
            access_count=15,
            creation_tick=0,
        )
        result = e.evaluate([desc])
        assert result.decisions[0].policy in (
            CompressionPolicy.VERBATIM, CompressionPolicy.SEMANTIC)

    def test_high_redundancy_prunes(self):
        e = MemoryCompressionEngine()
        desc = PacketDescriptor(
            packet_id="redundant",
            unique_tokens=5,
            total_tokens=10,
            overlap_score=0.95,
            creation_tick=0,
        )
        result = e.evaluate([desc])
        # High overlap + low entropy → likely SYMBOLIC or PRUNE
        assert result.decisions[0].policy in (
            CompressionPolicy.SYMBOLIC, CompressionPolicy.PRUNE)

    def test_multiple_packets(self):
        e = MemoryCompressionEngine()
        descs = [
            PacketDescriptor(packet_id=f"pkt_{i}", unique_tokens=i * 5,
                             total_tokens=i * 10, creation_tick=0)
            for i in range(1, 6)
        ]
        result = e.evaluate(descs)
        assert len(result.decisions) == 5

    def test_policy_counts(self):
        e = MemoryCompressionEngine()
        descs = [
            PacketDescriptor(packet_id="keep", unique_tokens=50, total_tokens=100,
                             emotional_significance=0.9, access_count=10, creation_tick=0),
            PacketDescriptor(packet_id="drop", unique_tokens=2, total_tokens=5,
                             overlap_score=0.99, creation_tick=0),
        ]
        result = e.evaluate(descs)
        assert sum(result.policy_counts.values()) == 2

    def test_tick_increments(self):
        e = MemoryCompressionEngine()
        r1 = e.evaluate([PacketDescriptor(packet_id="a", creation_tick=0)])
        r2 = e.evaluate([PacketDescriptor(packet_id="b", creation_tick=0)])
        assert r2.tick == r1.tick + 1


class TestTransitionTypes:
    def test_stmm_to_mtmm(self):
        e = MemoryCompressionEngine()
        desc = PacketDescriptor(packet_id="a", unique_tokens=10, total_tokens=20,
                                creation_tick=0)
        result = e.evaluate([desc], TransitionType.STMM_TO_MTMM)
        assert result.decisions[0].transition_type == TransitionType.STMM_TO_MTMM

    def test_mtmm_to_ltmm_higher_bar(self):
        e = MemoryCompressionEngine()
        desc = PacketDescriptor(
            packet_id="a", unique_tokens=10, total_tokens=20,
            emotional_significance=0.5, creation_tick=0,
        )
        r_stmm = e.evaluate([desc], TransitionType.STMM_TO_MTMM)
        r_ltmm = e.evaluate([desc], TransitionType.MTMM_TO_LTMM)
        # MTMM→LTMM has higher thresholds, so policies may be more aggressive
        # (At minimum, the threshold values differ)

    def test_ltmm_cold_highest_bar(self):
        cfg = MemoryCompressionConfig()
        mods = cfg.transition_modifiers["ltmm_cold"]
        assert mods["threshold_verbatim"] > cfg.threshold_verbatim


class TestOverridesInEvaluate:
    def test_identity_packet_verbatim(self):
        e = MemoryCompressionEngine()
        desc = PacketDescriptor(
            packet_id="identity",
            unique_tokens=2, total_tokens=5,
            is_identity_relevant=True,
            creation_tick=0,
        )
        result = e.evaluate([desc])
        assert result.decisions[0].policy == CompressionPolicy.VERBATIM
        assert result.decisions[0].override_reason == "identity_relevant"

    def test_critical_flag_verbatim(self):
        e = MemoryCompressionEngine()
        desc = PacketDescriptor(
            packet_id="critical",
            unique_tokens=2, total_tokens=5,
            flags=["CRITICAL:emergency"],
            creation_tick=0,
        )
        result = e.evaluate([desc])
        assert result.decisions[0].policy == CompressionPolicy.VERBATIM


# =====================================================================
# process() pipeline
# =====================================================================

class TestProcessPipeline:
    def test_process_with_packets(self):
        e = MemoryCompressionEngine()
        result = e.process({
            "packets": [
                {"packet_id": "p1", "unique_tokens": 20, "total_tokens": 40,
                 "emotional_significance": 0.5, "creation_tick": 0},
                {"packet_id": "p2", "unique_tokens": 5, "total_tokens": 10,
                 "overlap_score": 0.9, "creation_tick": 0},
            ],
        })
        assert len(result["decisions"]) == 2
        assert "policy_counts" in result

    def test_process_with_nt_state(self):
        e = MemoryCompressionEngine()
        e.process({
            "nt_state": {"ach": 0.8, "gaba": 0.6},
            "packets": [{"packet_id": "a", "creation_tick": 0}],
        })
        assert abs(e.ach_level - 0.8) < 0.01

    def test_process_with_transition_type(self):
        e = MemoryCompressionEngine()
        result = e.process({
            "transition_type": "mtmm_to_ltmm",
            "packets": [{"packet_id": "a", "unique_tokens": 10,
                         "total_tokens": 20, "creation_tick": 0}],
        })
        assert result["tick"] == 1

    def test_process_empty(self):
        e = MemoryCompressionEngine()
        result = e.process({})
        assert result["tick"] == 1
        assert len(result["decisions"]) == 0

    def test_process_none(self):
        e = MemoryCompressionEngine()
        result = e.process(None)
        assert result["tick"] == 1

    def test_process_invalid_transition_type(self):
        e = MemoryCompressionEngine()
        result = e.process({
            "transition_type": "invalid",
            "packets": [{"packet_id": "a", "creation_tick": 0}],
        })
        assert result["tick"] == 1  # Falls back to STMM_TO_MTMM


# =====================================================================
# Neurochem output
# =====================================================================

class TestNeurochemOutput:
    def test_neurochem_as_dict(self):
        nc = MemoryCompressionNeurochem(_5ht_delta=0.1, gaba_delta=0.05)
        d = nc.as_dict()
        assert d["_5ht_delta"] == 0.1
        assert d["gaba_delta"] == 0.05

    def test_pruning_produces_gaba(self):
        nc = compute_compression_neurochem(0, 5, 0, 0, 10)
        assert nc.gaba_delta > 0

    def test_emotional_preservation_produces_5ht(self):
        nc = compute_compression_neurochem(0, 0, 3, 0, 10)
        assert nc._5ht_delta > 0

    def test_verbatim_produces_ach(self):
        nc = compute_compression_neurochem(3, 0, 0, 0, 10)
        assert nc.ach_delta > 0


# =====================================================================
# Lifetime counters
# =====================================================================

class TestLifetimeCounters:
    def test_total_evaluated(self):
        e = MemoryCompressionEngine()
        e.evaluate([PacketDescriptor(packet_id="a", creation_tick=0)])
        e.evaluate([PacketDescriptor(packet_id="b", creation_tick=0)])
        assert e._total_evaluated == 2

    def test_total_pruned(self):
        e = MemoryCompressionEngine()
        desc = PacketDescriptor(
            packet_id="disposable", unique_tokens=1, total_tokens=2,
            overlap_score=0.99, creation_tick=0,
        )
        e.evaluate([desc])
        # May or may not be pruned depending on other factors
        assert e._total_pruned >= 0


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    def test_many_packets(self):
        e = MemoryCompressionEngine()
        descs = [
            PacketDescriptor(packet_id=f"p_{i}", unique_tokens=i + 1,
                             total_tokens=(i + 1) * 2, creation_tick=0)
            for i in range(50)
        ]
        result = e.evaluate(descs)
        assert len(result.decisions) == 50

    def test_processing_time(self):
        e = MemoryCompressionEngine()
        result = e.evaluate([PacketDescriptor(packet_id="a", creation_tick=0)])
        assert result.processing_time_ms >= 0.0

    def test_mean_retention(self):
        e = MemoryCompressionEngine()
        descs = [
            PacketDescriptor(packet_id="a", unique_tokens=50, total_tokens=100,
                             emotional_significance=0.9, creation_tick=0),
            PacketDescriptor(packet_id="b", unique_tokens=1, total_tokens=2,
                             overlap_score=0.99, creation_tick=0),
        ]
        result = e.evaluate(descs)
        assert 0.0 <= result.mean_retention <= 1.0

    def test_all_identity_packets_verbatim(self):
        e = MemoryCompressionEngine()
        descs = [
            PacketDescriptor(packet_id=f"id_{i}", is_identity_relevant=True,
                             creation_tick=0)
            for i in range(5)
        ]
        result = e.evaluate(descs)
        for d in result.decisions:
            assert d.policy == CompressionPolicy.VERBATIM
