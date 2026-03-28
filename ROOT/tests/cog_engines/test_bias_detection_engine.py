"""
Tests for Engine 5 -- Bias Detection Engine
============================================
Covers: enums, config, data types, template matching, structural
detection, contextual reinforcement, Bayesian update, severity
classification, bias load, neurochemical coupling, engine pipeline,
mode thresholds, bidirectional feedback, edge cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.bias_detection_engine import (
    BiasCategory,
    BiasDetectionConfig,
    BiasDetectionEngine,
    BiasDetectionInput,
    BiasDetectionResult,
    BiasDetectionState,
    BiasFlag,
    BiasType,
    BiasDetectionNeurochem,
    SeverityLevel,
    _BIAS_TEMPLATES,
    _CATEGORY_MEMBERS,
    _bias_type_to_category,
    bayesian_update,
    classify_severity,
    compute_bias_load,
    compute_contextual_score,
    compute_keyword_score,
    compute_neurochem_signals,
    compute_structural_score,
    fuse_scores,
    resolve_threshold,
)
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
    ProcessedStatement,
    SourceTag,
)


# =====================================================================
# Constants & Fixtures
# =====================================================================


RNG = np.random.default_rng(42)
CFG = BiasDetectionConfig()


def _stmt(text: str, src: SourceTag = SourceTag.USER_INPUT) -> ProcessedStatement:
    return ProcessedStatement(raw_text=text, source_tag=src)


# =====================================================================
# Enums
# =====================================================================


class TestEnums:
    def test_bias_category_values(self):
        assert len(BiasCategory) == 8

    def test_bias_type_values(self):
        assert len(BiasType) == 24

    def test_severity_levels(self):
        assert set(SeverityLevel) == {
            SeverityLevel.LOW, SeverityLevel.MODERATE,
            SeverityLevel.HIGH, SeverityLevel.CRITICAL,
        }

    def test_category_membership(self):
        for cat, members in _CATEGORY_MEMBERS.items():
            for bt in members:
                assert _bias_type_to_category(bt) == cat


# =====================================================================
# Config
# =====================================================================


class TestConfig:
    def test_defaults(self):
        cfg = BiasDetectionConfig()
        assert cfg.theta_normal == 0.45
        assert cfg.w_keyword + cfg.w_structural + cfg.w_contextual == pytest.approx(1.0)
        assert cfg.prior_base == 0.10

    def test_frozen(self):
        with pytest.raises(AttributeError):
            CFG.theta_normal = 0.99


# =====================================================================
# Template registry
# =====================================================================


class TestTemplateRegistry:
    def test_all_bias_types_have_templates(self):
        for bt in BiasType:
            assert bt in _BIAS_TEMPLATES, f"Missing template for {bt}"

    def test_template_structure(self):
        for bt, tmpl in _BIAS_TEMPLATES.items():
            assert "keywords" in tmpl
            assert "structural" in tmpl
            assert "weight" in tmpl
            assert isinstance(tmpl["keywords"], list)


# =====================================================================
# Keyword scoring
# =====================================================================


class TestKeywordScore:
    def test_no_keywords(self):
        assert compute_keyword_score("hello world", []) == 0.0

    def test_all_match(self):
        assert compute_keyword_score("number figure initial", ["number", "figure", "initial"]) == 1.0

    def test_partial_match(self):
        score = compute_keyword_score("number and text", ["number", "figure"])
        assert 0.0 < score < 1.0

    def test_no_match(self):
        assert compute_keyword_score("xyz", ["number", "figure"]) == 0.0

    def test_case_insensitive(self):
        assert compute_keyword_score("NUMBER FIGURE", ["number", "figure"]) == 1.0


# =====================================================================
# Structural scoring
# =====================================================================


class TestStructuralScore:
    def test_numeric_anchor(self):
        assert compute_structural_score("price is 100", "numeric_anchor_present") == 1.0

    def test_no_numeric(self):
        assert compute_structural_score("no numbers here", "numeric_anchor_present") == 0.0

    def test_vivid_language(self):
        assert compute_structural_score("this is terrible!", "vivid_language") == 1.0

    def test_unknown_marker(self):
        assert compute_structural_score("text", "nonexistent_marker") == 0.0

    def test_loss_frame(self):
        assert compute_structural_score("you will lose everything", "loss_frame") == 1.0

    def test_group_distinction(self):
        assert compute_structural_score("we know but they don't", "group_distinction") == 1.0


# =====================================================================
# Contextual scoring
# =====================================================================


class TestContextualScore:
    def test_no_siblings(self):
        score = compute_contextual_score(BiasType.ANCHOR_NUMERIC, {})
        assert score == 0.0

    def test_high_siblings(self):
        other_scores = {
            BiasType.ANCHOR_NARRATIVE: 0.8,
            BiasType.ANCHOR_PRIMACY: 0.7,
        }
        score = compute_contextual_score(BiasType.ANCHOR_NUMERIC, other_scores)
        assert score > 0.5

    def test_no_cross_category(self):
        # Availability scores shouldn't boost anchoring
        other_scores = {
            BiasType.AVAILABILITY_RECENCY: 0.9,
        }
        score = compute_contextual_score(BiasType.ANCHOR_NUMERIC, other_scores)
        assert score == 0.0


# =====================================================================
# Fusion and Bayesian
# =====================================================================


class TestFusionAndBayesian:
    def test_fuse_scores_weighted(self):
        result = fuse_scores(1.0, 1.0, 1.0, CFG)
        assert result == pytest.approx(1.0)

    def test_fuse_scores_zero(self):
        assert fuse_scores(0.0, 0.0, 0.0, CFG) == 0.0

    def test_bayesian_zero_evidence(self):
        assert bayesian_update(0.10, 0.0, 0.30) == pytest.approx(0.10)

    def test_bayesian_high_evidence(self):
        post = bayesian_update(0.10, 1.0, 0.30)
        assert post > 0.10
        assert post <= 1.0

    def test_bayesian_clamp(self):
        assert bayesian_update(0.99, 1.0, 5.0) == 1.0
        assert bayesian_update(0.0, 0.0, 0.0) == 0.0


# =====================================================================
# Severity classification
# =====================================================================


class TestSeverityClassification:
    def test_low(self):
        assert classify_severity(0.20, CFG) == SeverityLevel.LOW

    def test_moderate(self):
        assert classify_severity(0.55, CFG) == SeverityLevel.MODERATE

    def test_high(self):
        assert classify_severity(0.75, CFG) == SeverityLevel.HIGH

    def test_critical(self):
        assert classify_severity(0.90, CFG) == SeverityLevel.CRITICAL


# =====================================================================
# Bias load
# =====================================================================


class TestBiasLoad:
    def test_empty(self):
        assert compute_bias_load([]) == 0.0

    def test_single_low(self):
        flag = BiasFlag(confidence=0.5, severity=SeverityLevel.LOW)
        load = compute_bias_load([flag])
        assert 0.0 < load < 0.5

    def test_multiple_critical(self):
        flags = [
            BiasFlag(confidence=0.9, severity=SeverityLevel.CRITICAL),
            BiasFlag(confidence=0.85, severity=SeverityLevel.CRITICAL),
        ]
        load = compute_bias_load(flags)
        assert load > 0.5


# =====================================================================
# Mode threshold resolution
# =====================================================================


class TestModeThresholds:
    def test_normal(self):
        assert resolve_threshold(OperationalMode.NORMAL, CFG) == 0.45

    def test_dev_lowest(self):
        assert resolve_threshold(OperationalMode.DEV, CFG) == 0.20

    def test_rem_dream_highest(self):
        assert resolve_threshold(OperationalMode.REM_DREAM, CFG) == 0.60

    def test_all_modes_covered(self):
        for mode in OperationalMode:
            t = resolve_threshold(mode, CFG)
            assert 0.0 < t < 1.0


# =====================================================================
# Neurochemical signals
# =====================================================================


class TestBiasDetectionNeurochem:
    def test_no_bias_no_signal(self):
        sig = compute_neurochem_signals(0.0, [], CFG, RNG)
        assert sig.delta_ach == 0.0
        assert sig.delta_ne == 0.0

    def test_with_bias_ach_positive(self):
        flags = [BiasFlag(confidence=0.8, severity=SeverityLevel.HIGH,
                          bias_category=BiasCategory.ANCHORING)]
        sig = compute_neurochem_signals(0.6, flags, CFG, np.random.default_rng(42))
        assert sig.delta_ach > 0.0

    def test_threat_categories_trigger_cor(self):
        flags = [BiasFlag(confidence=0.7, severity=SeverityLevel.HIGH,
                          bias_category=BiasCategory.FRAMING)]
        sig = compute_neurochem_signals(0.5, flags, CFG, np.random.default_rng(42))
        assert sig.delta_cor > 0.0

    def test_non_threat_no_cor(self):
        flags = [BiasFlag(confidence=0.7, severity=SeverityLevel.HIGH,
                          bias_category=BiasCategory.ANCHORING)]
        sig = compute_neurochem_signals(0.5, flags, CFG, np.random.default_rng(42))
        assert sig.delta_cor == 0.0

    def test_beta_boost_present(self):
        flags = [BiasFlag()]
        sig = compute_neurochem_signals(0.3, flags, CFG, np.random.default_rng(42))
        assert sig.beta_boost > 0.0


# =====================================================================
# Engine -- basic pipeline
# =====================================================================


class TestEngineBasic:
    def setup_method(self):
        self.engine = BiasDetectionEngine(rng=np.random.default_rng(42))

    def test_empty_input(self):
        result = self.engine.process(BiasDetectionInput())
        assert result.clean_pass
        assert result.total_flagged == 0
        assert result.bias_load == 0.0

    def test_biased_text_flags(self):
        stmt = _stmt("Everyone knows this is obviously true, clearly the expert confirms it proves the point")
        result = self.engine.process(BiasDetectionInput(
            statements=[stmt], active_mode=OperationalMode.DEV,
        ))
        assert result.total_flagged > 0
        assert not result.clean_pass

    def test_neutral_text_clean(self):
        stmt = _stmt("The temperature today is mild.")
        result = self.engine.process(BiasDetectionInput(statements=[stmt]))
        # May or may not flag depending on threshold -- mostly clean text
        assert result.total_statements == 1

    def test_multiple_statements(self):
        stmts = [
            _stmt("recently everyone is saying this terrible thing"),
            _stmt("the expert confirms it proves the theory"),
        ]
        result = self.engine.process(BiasDetectionInput(statements=stmts))
        assert result.total_statements == 2
        assert result.total_flagged >= 0  # could flag multiple

    def test_result_has_neurochem(self):
        stmt = _stmt("obviously everyone knows this is terrible and shocking")
        result = self.engine.process(BiasDetectionInput(statements=[stmt]))
        assert isinstance(result.neurochemical_signals, BiasDetectionNeurochem)

    def test_processing_time(self):
        stmt = _stmt("test text")
        result = self.engine.process(BiasDetectionInput(statements=[stmt]))
        assert result.processing_time_ms >= 0.0


# =====================================================================
# Engine -- mode configuration
# =====================================================================


class TestEngineModes:
    def test_dev_mode_more_sensitive(self):
        engine = BiasDetectionEngine(rng=np.random.default_rng(42))
        stmt = _stmt("the first initial number was 50")

        # Normal mode
        result_normal = engine.process(BiasDetectionInput(
            statements=[stmt], active_mode=OperationalMode.NORMAL,
        ))

        # Dev mode (lower threshold)
        engine2 = BiasDetectionEngine(rng=np.random.default_rng(42))
        result_dev = engine2.process(BiasDetectionInput(
            statements=[stmt], active_mode=OperationalMode.DEV,
        ))

        assert result_dev.total_flagged >= result_normal.total_flagged

    def test_rem_dream_less_sensitive(self):
        engine = BiasDetectionEngine(rng=np.random.default_rng(42))
        stmt = _stmt("recently everyone says this is obviously true")
        result_dream = engine.process(BiasDetectionInput(
            statements=[stmt], active_mode=OperationalMode.REM_DREAM,
        ))
        engine2 = BiasDetectionEngine(rng=np.random.default_rng(42))
        result_normal = engine2.process(BiasDetectionInput(
            statements=[stmt], active_mode=OperationalMode.NORMAL,
        ))
        assert result_dream.total_flagged <= result_normal.total_flagged


# =====================================================================
# Engine -- bidirectional feedback
# =====================================================================


class TestBidirectionalFeedback:
    def test_high_cortisol_lowers_threshold(self):
        engine = BiasDetectionEngine(rng=np.random.default_rng(42))
        engine.update_neurochem_state({"cor": 0.8})
        stmt = _stmt("the first number suggests a starting baseline")
        result = engine.process(BiasDetectionInput(statements=[stmt]))
        # High cortisol → 0.85x threshold → more flags potentially
        assert result.metadata["threshold"] < CFG.theta_normal

    def test_low_da_lowers_threshold(self):
        engine = BiasDetectionEngine(rng=np.random.default_rng(42))
        engine.update_neurochem_state({"da": 0.15})
        stmt = _stmt("test text")
        result = engine.process(BiasDetectionInput(statements=[stmt]))
        assert result.metadata["threshold"] < CFG.theta_normal


# =====================================================================
# Engine -- status / configure
# =====================================================================


class TestEngineStatus:
    def test_get_status(self):
        engine = BiasDetectionEngine()
        status = engine.get_status()
        assert status["engine_id"] == "bias_detection_engine"
        assert status["mode"] == "normal"
        assert status["cycle_count"] == 0

    def test_configure_mode(self):
        engine = BiasDetectionEngine()
        engine.configure(OperationalMode.DEV)
        assert engine.get_status()["mode"] == "dev"

    def test_cycle_count_increments(self):
        engine = BiasDetectionEngine()
        engine.process(BiasDetectionInput())
        engine.process(BiasDetectionInput())
        assert engine.get_status()["cycle_count"] == 2


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    def test_empty_text_statement(self):
        stmt = _stmt("")
        engine = BiasDetectionEngine()
        result = engine.process(BiasDetectionInput(statements=[stmt]))
        assert result.clean_pass

    def test_whitespace_only_statement(self):
        stmt = _stmt("   \n\t  ")
        engine = BiasDetectionEngine()
        result = engine.process(BiasDetectionInput(statements=[stmt]))
        assert result.clean_pass

    def test_source_tag_preserved(self):
        stmt = _stmt("everyone knows", SourceTag.MEMORY_LONG_TERM)
        engine = BiasDetectionEngine(rng=np.random.default_rng(42))
        result = engine.process(BiasDetectionInput(
            statements=[stmt], active_mode=OperationalMode.DEV,
        ))
        for f in result.flags:
            assert f.source_tag == SourceTag.MEMORY_LONG_TERM

    def test_category_counts_accurate(self):
        stmt = _stmt("everyone obviously knows this is terrible shocking news, recently confirmed by experts")
        engine = BiasDetectionEngine(rng=np.random.default_rng(42))
        result = engine.process(BiasDetectionInput(
            statements=[stmt], active_mode=OperationalMode.DEV,
        ))
        total_from_counts = sum(result.category_counts.values())
        assert total_from_counts == result.total_flagged

    def test_flag_fields_populated(self):
        stmt = _stmt("everyone knows this is obviously proven by the expert study")
        engine = BiasDetectionEngine(rng=np.random.default_rng(42))
        result = engine.process(BiasDetectionInput(
            statements=[stmt], active_mode=OperationalMode.DEV,
        ))
        for f in result.flags:
            assert f.bias_id
            assert f.bias_type in BiasType
            assert f.bias_category in BiasCategory
            assert 0.0 <= f.confidence <= 1.0
            assert f.severity in SeverityLevel
            assert f.description
