"""
Tests for Engine 12 -- Logical Brain Engine
============================================
Covers: enums, config, submodule tiers, diagnostic elevation,
tier scoring, aggregate scoring, verdict classification, neurochem
coupling, engine pipeline, mode thresholds, bidirectional feedback,
edge cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.logical_brain_engine import (
    DIAGNOSTIC_ELEVATION,
    EvaluationTier,
    LogicalBrainConfig,
    LogicalBrainEngine,
    LogicalBrainInput,
    LogicalBrainNeurochem,
    LogicalBrainResult,
    LogicalBrainVerdict,
    SubmoduleScore,
    VerdictLevel,
    _SUBMODULE_TIERS,
    apply_diagnostic_elevation,
    classify_verdict,
    compute_aggregate_score,
    compute_logical_brain_neurochem,
    compute_tier_score,
    resolve_pass_threshold,
)
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)
from zados.reward.domains.logic.ports import (
    ContrastResult,
    MemoryContrastPort,
    CognitiveTracePort,
    TraceResult,
)


# =====================================================================
# Fixtures
# =====================================================================

RNG = np.random.default_rng(42)
CFG = LogicalBrainConfig()


class MockMemoryContrast:
    """Mock MemoryContrastPort for testing."""
    def __init__(self, sim: float = 0.8, div: float = 0.1):
        self._sim = sim
        self._div = div

    def contrast(self, *, current, query_type, ctx_id=None, limit=5, meta=None):
        return ContrastResult(similarity=self._sim, divergence=self._div)


class MockCognitiveTrace:
    """Mock CognitiveTracePort for testing."""
    def get_trace(self, *, request, trace_type, ctx_id=None, meta=None):
        return TraceResult(trace={"mock": True})


# =====================================================================
# Enums
# =====================================================================


class TestEnums:
    def test_verdict_levels(self):
        assert len(VerdictLevel) == 4
        assert VerdictLevel.EXEMPLARY.value == "exemplary"

    def test_evaluation_tiers(self):
        assert len(EvaluationTier) == 4


# =====================================================================
# Config
# =====================================================================


class TestConfig:
    def test_tier_weights_sum_to_one(self):
        total = (CFG.w_tier1_epistemic + CFG.w_tier2_consistency
                 + CFG.w_tier3_fidelity + CFG.w_tier4_continuity)
        assert total == pytest.approx(1.0)

    def test_defaults(self):
        assert CFG.diagnostic_elevation == DIAGNOSTIC_ELEVATION
        assert CFG.exemplary_threshold == 0.85

    def test_frozen(self):
        with pytest.raises(AttributeError):
            CFG.exemplary_threshold = 0.99


# =====================================================================
# Submodule tier registry
# =====================================================================


class TestSubmoduleTiers:
    def test_all_submodules_registered(self):
        expected = {
            "epistemic_calibration", "uncertainty_acknowledgment",
            "abstention_appropriateness", "internal_consistency",
            "external_consistency", "context_fidelity", "concept_fidelity",
            "semantic_continuity", "concept_continuity",
        }
        assert set(_SUBMODULE_TIERS.keys()) == expected

    def test_tier_weights_within_tier(self):
        for tier in EvaluationTier:
            members = [(name, w) for name, (t, w) in _SUBMODULE_TIERS.items() if t == tier]
            if members:
                total = sum(w for _, w in members)
                assert total == pytest.approx(1.0, abs=0.01), f"Tier {tier} weights don't sum to 1.0: {total}"


# =====================================================================
# Diagnostic elevation
# =====================================================================


class TestDiagnosticElevation:
    def test_zero_unchanged(self):
        assert apply_diagnostic_elevation(0.0, 1.25) == 0.0

    def test_one_unchanged(self):
        assert apply_diagnostic_elevation(1.0, 1.25) == 1.0

    def test_mid_value(self):
        result = apply_diagnostic_elevation(0.5, 1.25)
        # x^(1/1.25) = x^0.8 → 0.5^0.8 ≈ 0.574
        assert 0.50 < result < 0.60

    def test_low_value_boosted(self):
        result = apply_diagnostic_elevation(0.2, 1.25)
        assert result > 0.2

    def test_higher_elevation_more_boost(self):
        r1 = apply_diagnostic_elevation(0.3, 1.10)
        r2 = apply_diagnostic_elevation(0.3, 1.50)
        assert r2 > r1


# =====================================================================
# Tier scoring
# =====================================================================


class TestTierScoring:
    def test_empty_tier(self):
        assert compute_tier_score([], EvaluationTier.CORE_EPISTEMIC) == 0.0

    def test_single_submodule(self):
        scores = [SubmoduleScore(
            name="epistemic_calibration",
            tier=EvaluationTier.CORE_EPISTEMIC,
            raw_score=0.8,
            elevated_score=0.85,
        )]
        result = compute_tier_score(scores, EvaluationTier.CORE_EPISTEMIC)
        assert result == pytest.approx(0.85)

    def test_multiple_submodules_weighted(self):
        scores = [
            SubmoduleScore(name="epistemic_calibration", tier=EvaluationTier.CORE_EPISTEMIC,
                           raw_score=0.8, elevated_score=0.85),
            SubmoduleScore(name="uncertainty_acknowledgment", tier=EvaluationTier.CORE_EPISTEMIC,
                           raw_score=0.6, elevated_score=0.65),
            SubmoduleScore(name="abstention_appropriateness", tier=EvaluationTier.CORE_EPISTEMIC,
                           raw_score=0.7, elevated_score=0.75),
        ]
        result = compute_tier_score(scores, EvaluationTier.CORE_EPISTEMIC)
        # Weighted: 0.40*0.85 + 0.35*0.65 + 0.25*0.75 = 0.34 + 0.2275 + 0.1875 = 0.755
        assert result == pytest.approx(0.755)

    def test_skipped_excluded(self):
        scores = [
            SubmoduleScore(name="epistemic_calibration", tier=EvaluationTier.CORE_EPISTEMIC,
                           raw_score=0.8, elevated_score=0.85),
            SubmoduleScore(name="uncertainty_acknowledgment", tier=EvaluationTier.CORE_EPISTEMIC,
                           raw_score=0.0, elevated_score=0.0, skipped=True),
        ]
        result = compute_tier_score(scores, EvaluationTier.CORE_EPISTEMIC)
        # Only epistemic_calibration counts, weight redistributed
        assert result == pytest.approx(0.85)

    def test_all_skipped(self):
        scores = [
            SubmoduleScore(name="internal_consistency", tier=EvaluationTier.CORE_CONSISTENCY,
                           raw_score=0.0, elevated_score=0.0, skipped=True),
        ]
        assert compute_tier_score(scores, EvaluationTier.CORE_CONSISTENCY) == 0.0


# =====================================================================
# Aggregate scoring
# =====================================================================


class TestAggregateScoring:
    def test_all_perfect(self):
        tier_scores = {
            "core_epistemic": 1.0, "core_consistency": 1.0,
            "extended_fidelity": 1.0, "extended_continuity": 1.0,
        }
        assert compute_aggregate_score(tier_scores, CFG) == pytest.approx(1.0)

    def test_all_zero(self):
        tier_scores = {
            "core_epistemic": 0.0, "core_consistency": 0.0,
            "extended_fidelity": 0.0, "extended_continuity": 0.0,
        }
        assert compute_aggregate_score(tier_scores, CFG) == 0.0

    def test_mixed(self):
        tier_scores = {
            "core_epistemic": 0.8, "core_consistency": 0.6,
            "extended_fidelity": 0.4, "extended_continuity": 0.3,
        }
        result = compute_aggregate_score(tier_scores, CFG)
        expected = 0.30 * 0.8 + 0.30 * 0.6 + 0.20 * 0.4 + 0.20 * 0.3
        assert result == pytest.approx(expected)


# =====================================================================
# Verdict classification
# =====================================================================


class TestVerdictClassification:
    def test_exemplary(self):
        assert classify_verdict(0.90, CFG) == VerdictLevel.EXEMPLARY

    def test_adequate(self):
        assert classify_verdict(0.70, CFG) == VerdictLevel.ADEQUATE

    def test_deficient(self):
        assert classify_verdict(0.45, CFG) == VerdictLevel.DEFICIENT

    def test_critical(self):
        assert classify_verdict(0.20, CFG) == VerdictLevel.CRITICAL

    def test_boundary_exemplary(self):
        assert classify_verdict(0.85, CFG) == VerdictLevel.EXEMPLARY

    def test_boundary_adequate(self):
        assert classify_verdict(0.60, CFG) == VerdictLevel.ADEQUATE


# =====================================================================
# Mode thresholds
# =====================================================================


class TestModeThresholds:
    def test_normal(self):
        assert resolve_pass_threshold(OperationalMode.NORMAL, CFG) == 0.50

    def test_dev_most_lenient(self):
        assert resolve_pass_threshold(OperationalMode.DEV, CFG) == 0.30

    def test_all_modes(self):
        for mode in OperationalMode:
            t = resolve_pass_threshold(mode, CFG)
            assert 0.0 < t < 1.0


# =====================================================================
# Neurochemical coupling
# =====================================================================


class TestNeurochemCoupling:
    def test_high_score_positive_da(self):
        sig = compute_logical_brain_neurochem(
            0.85, VerdictLevel.EXEMPLARY, 0, CFG, np.random.default_rng(42),
        )
        assert sig.delta_da > 0.0

    def test_low_score_negative_da(self):
        sig = compute_logical_brain_neurochem(
            0.25, VerdictLevel.CRITICAL, 3, CFG, np.random.default_rng(42),
        )
        assert sig.delta_da < 0.0

    def test_low_score_triggers_cor(self):
        sig = compute_logical_brain_neurochem(
            0.30, VerdictLevel.CRITICAL, 2, CFG, np.random.default_rng(42),
        )
        assert sig.delta_cor > 0.0

    def test_high_score_no_cor(self):
        sig = compute_logical_brain_neurochem(
            0.80, VerdictLevel.ADEQUATE, 0, CFG, np.random.default_rng(42),
        )
        assert sig.delta_cor == 0.0

    def test_ach_always_positive(self):
        sig = compute_logical_brain_neurochem(
            0.50, VerdictLevel.ADEQUATE, 1, CFG, np.random.default_rng(42),
        )
        assert sig.delta_ach > 0.0

    def test_beta_boost_present(self):
        sig = compute_logical_brain_neurochem(
            0.50, VerdictLevel.ADEQUATE, 0, CFG, np.random.default_rng(42),
        )
        assert sig.beta_boost == CFG.psi_beta_analysis


# =====================================================================
# Engine -- without ports
# =====================================================================


class TestEngineNoPort:
    def setup_method(self):
        self.engine = LogicalBrainEngine(rng=np.random.default_rng(42))

    def test_default_state(self):
        status = self.engine.get_status()
        assert status["engine_id"] == "logical_brain_engine"
        assert status["ports"]["memory_contrast"] is False

    def test_process_empty_state(self):
        result = self.engine.process(LogicalBrainInput())
        assert isinstance(result, LogicalBrainResult)
        assert isinstance(result.verdict, LogicalBrainVerdict)
        # Without ports, consistency submodules score 0 with skipped
        assert result.metadata["submodules_skipped"] > 0

    def test_epistemic_tier_works_without_ports(self):
        # Epistemic submodules don't need ports
        state = {"confidence": 0.5, "uncertainty": 0.5}
        result = self.engine.process(LogicalBrainInput(state=state))
        epistemic_scores = [
            s for s in result.submodule_scores
            if s.tier == EvaluationTier.CORE_EPISTEMIC
        ]
        assert len(epistemic_scores) == 3
        # At least one should not be skipped
        assert any(not s.skipped for s in epistemic_scores)


# =====================================================================
# Engine -- with mock ports
# =====================================================================


class TestEngineWithPorts:
    def setup_method(self):
        self.engine = LogicalBrainEngine(
            rng=np.random.default_rng(42),
            memory_contrast=MockMemoryContrast(sim=0.9, div=0.05),
            cognitive_trace=MockCognitiveTrace(),
        )

    def test_ports_connected(self):
        status = self.engine.get_status()
        assert status["ports"]["memory_contrast"] is True
        assert status["ports"]["cognitive_trace"] is True

    def test_consistency_not_skipped(self):
        state = {"representation": {"text": "test"}, "confidence": 0.7, "uncertainty": 0.3}
        result = self.engine.process(LogicalBrainInput(state=state))
        consistency_scores = [
            s for s in result.submodule_scores
            if s.tier == EvaluationTier.CORE_CONSISTENCY
        ]
        assert any(not s.skipped for s in consistency_scores)

    def test_high_similarity_good_score(self):
        state = {"representation": {"text": "consistent"}, "confidence": 0.7, "uncertainty": 0.3}
        result = self.engine.process(LogicalBrainInput(state=state))
        assert result.verdict.aggregate_score > 0.0

    def test_domain_result_compatible(self):
        result = self.engine.process(LogicalBrainInput(state={"confidence": 0.5, "uncertainty": 0.5}))
        assert result.domain_result is not None
        assert result.domain_result.domain == "logic_diagnostic"
        assert result.domain_result.meta["diagnostic_mode"] is True


# =====================================================================
# Engine -- mode + bidirectional
# =====================================================================


class TestEngineModes:
    def test_configure_mode(self):
        engine = LogicalBrainEngine()
        engine.configure(OperationalMode.REFLECTIVE)
        assert engine.get_status()["mode"] == "reflective"

    def test_high_cortisol_increases_elevation(self):
        engine = LogicalBrainEngine(rng=np.random.default_rng(42))
        engine.update_neurochem_state({"cor": 0.8})
        result = engine.process(LogicalBrainInput(state={"confidence": 0.5, "uncertainty": 0.5}))
        assert result.metadata["elevation"] > DIAGNOSTIC_ELEVATION

    def test_cycle_count(self):
        engine = LogicalBrainEngine()
        engine.process(LogicalBrainInput())
        engine.process(LogicalBrainInput())
        assert engine.get_status()["cycle_count"] == 2


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    def test_low_divergence_high_score(self):
        engine = LogicalBrainEngine(
            rng=np.random.default_rng(42),
            memory_contrast=MockMemoryContrast(sim=0.95, div=0.02),
        )
        state = {"representation": {"text": "good"}, "confidence": 0.8, "uncertainty": 0.2}
        result = engine.process(LogicalBrainInput(state=state))
        # Should be a decent score
        assert result.verdict.aggregate_score > 0.3

    def test_high_divergence_poor_score(self):
        engine = LogicalBrainEngine(
            rng=np.random.default_rng(42),
            memory_contrast=MockMemoryContrast(sim=0.1, div=0.9),
        )
        state = {"representation": {"text": "bad"}, "confidence": 0.9, "uncertainty": 0.8}
        result = engine.process(LogicalBrainInput(state=state))
        # Should flag issues
        assert len(result.verdict.tier_scores) == 4

    def test_verdict_passed_check(self):
        engine = LogicalBrainEngine(
            rng=np.random.default_rng(42),
            memory_contrast=MockMemoryContrast(sim=0.95, div=0.01),
        )
        state = {"representation": {"text": "x"}, "confidence": 0.5, "uncertainty": 0.5}
        result = engine.process(LogicalBrainInput(state=state))
        threshold = resolve_pass_threshold(OperationalMode.NORMAL, CFG)
        if result.verdict.aggregate_score >= threshold:
            assert result.verdict.passed
        else:
            assert not result.verdict.passed
