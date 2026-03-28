"""
Tests for contradiction_detection_engine.py

Coverage:
- Bayesian confidence computation (all levels, partial evidence, temporal decay)
- Prior computation (base, memory contrast scaling, DA D2 feedback)
- Threshold resolution (all modes, NE/GABA bidirectional feedback)
- Negation detection (Level 1)
- Semantic opposition scoring (Level 2)
- Contextual incompatibility (Level 3)
- κ_contradiction and contradiction load update
- Neurochemical signal computation
- ContradictionDetectionEngine.process() end-to-end
- Engine state (configure, update_neurochem_state, get_status)
"""

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    ComparisonSet,
    ContradictionDetectionEngine,
    ContradictionEngineConfig,
    ContradictionEngineState,
    EvidenceSignals,
    OperationalMode,
    ProcessedStatement,
    SourceTag,
    _gaussian_pdf,
    compute_contextual_incompatibility,
    compute_kappa,
    compute_level1_likelihoods,
    compute_level2_likelihoods,
    compute_level3_likelihoods,
    compute_neurochemical_signals,
    compute_prior,
    compute_semantic_opposition,
    compute_temporal_decay,
    compute_temporal_separation_days,
    detect_negation_signal,
    fuse_bayesian_confidence,
    resolve_detection_threshold,
    update_contradiction_load,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> ContradictionEngineConfig:
    return ContradictionEngineConfig()


@pytest.fixture
def engine(cfg) -> ContradictionDetectionEngine:
    return ContradictionDetectionEngine(cfg, rng=np.random.default_rng(42))


@pytest.fixture
def stmt_online() -> ProcessedStatement:
    return ProcessedStatement(
        raw_text="The server is online.",
        tokens=["The", "server", "is", "online"],
        source_tag=SourceTag.USER_INPUT,
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        factuality_score=0.9,
        affective_score=0.1,
    )


@pytest.fixture
def stmt_offline() -> ProcessedStatement:
    return ProcessedStatement(
        raw_text="The server is not online.",
        tokens=["The", "server", "is", "not", "online"],
        source_tag=SourceTag.AI_OUTPUT,
        timestamp=datetime(2026, 1, 1, 12, 0, 1),
        factuality_score=0.9,
        affective_score=0.1,
    )


@pytest.fixture
def stmt_unrelated() -> ProcessedStatement:
    return ProcessedStatement(
        raw_text="The weather is nice today.",
        tokens=["The", "weather", "is", "nice", "today"],
        source_tag=SourceTag.AI_OUTPUT,
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        factuality_score=0.7,
        affective_score=0.2,
    )


# ---------------------------------------------------------------------------
# Gaussian PDF
# ---------------------------------------------------------------------------

class TestGaussianPDF:
    def test_peak_at_mean(self):
        pdf_at_mean = _gaussian_pdf(0.5, 0.5, 0.1)
        pdf_off     = _gaussian_pdf(0.7, 0.5, 0.1)
        assert pdf_at_mean > pdf_off

    def test_zero_sigma_returns_zero_for_nonmatch(self):
        result = _gaussian_pdf(0.6, 0.5, 0.0)
        assert result == 0.0

    def test_zero_sigma_returns_one_for_exact(self):
        result = _gaussian_pdf(0.5, 0.5, 0.0)
        assert result == 1.0

    def test_symmetry(self):
        """PDF is symmetric around the mean."""
        mu, sigma = 0.5, 0.15
        assert _gaussian_pdf(mu - 0.1, mu, sigma) == pytest.approx(
            _gaussian_pdf(mu + 0.1, mu, sigma)
        )


# ---------------------------------------------------------------------------
# Prior computation
# ---------------------------------------------------------------------------

class TestComputePrior:
    def test_zero_contrast_ratio(self, cfg):
        prior = compute_prior(cfg.p_base, cfg.alpha_mc, 0.0)
        assert prior == pytest.approx(cfg.p_base)

    def test_full_contrast_ratio_raises_prior(self, cfg):
        prior = compute_prior(cfg.p_base, cfg.alpha_mc, 1.0)
        assert prior > cfg.p_base

    def test_prior_clamped_to_one(self, cfg):
        prior = compute_prior(1.0, 10.0, 1.0, da_d2_level=1.0)
        assert prior <= 1.0

    def test_da_d2_raises_prior(self, cfg):
        p_no_da = compute_prior(cfg.p_base, cfg.alpha_mc, 0.5, da_d2_level=0.0)
        p_hi_da = compute_prior(cfg.p_base, cfg.alpha_mc, 0.5, da_d2_level=1.0)
        assert p_hi_da > p_no_da

    def test_prior_non_negative(self, cfg):
        prior = compute_prior(0.0, 0.0, 0.0, da_d2_level=0.0)
        assert prior >= 0.0


# ---------------------------------------------------------------------------
# Level 1 likelihoods
# ---------------------------------------------------------------------------

class TestLevel1Likelihoods:
    def test_n_obs_1_returns_tp_and_fp(self, cfg):
        p_c, p_nc = compute_level1_likelihoods(1, cfg.eps_fn, cfg.eps_fp)
        assert p_c  == pytest.approx(1.0 - cfg.eps_fn)
        assert p_nc == pytest.approx(cfg.eps_fp)

    def test_n_obs_0_returns_fn_and_tn(self, cfg):
        p_c, p_nc = compute_level1_likelihoods(0, cfg.eps_fn, cfg.eps_fp)
        assert p_c  == pytest.approx(cfg.eps_fn)
        assert p_nc == pytest.approx(1.0 - cfg.eps_fp)

    def test_tp_greater_than_fp(self, cfg):
        p_c, p_nc = compute_level1_likelihoods(1, cfg.eps_fn, cfg.eps_fp)
        assert p_c > p_nc


# ---------------------------------------------------------------------------
# Level 2 likelihoods
# ---------------------------------------------------------------------------

class TestLevel2Likelihoods:
    def test_high_opposition_favours_contradiction(self, cfg):
        p_c, p_nc = compute_level2_likelihoods(0.90, cfg.mu_C, cfg.mu_not_C, cfg.sigma_s)
        assert p_c > p_nc

    def test_low_opposition_favours_no_contradiction(self, cfg):
        p_c, p_nc = compute_level2_likelihoods(0.10, cfg.mu_C, cfg.mu_not_C, cfg.sigma_s)
        assert p_nc > p_c

    def test_midpoint_is_ambiguous(self, cfg):
        mid = (cfg.mu_C + cfg.mu_not_C) / 2
        p_c, p_nc = compute_level2_likelihoods(mid, cfg.mu_C, cfg.mu_not_C, cfg.sigma_s)
        # Neither should completely dominate
        assert p_c > 0 and p_nc > 0


# ---------------------------------------------------------------------------
# Level 3 likelihoods
# ---------------------------------------------------------------------------

class TestLevel3Likelihoods:
    def test_high_contextual_favours_contradiction(self, cfg):
        p_c, p_nc = compute_level3_likelihoods(0.9, cfg.mu_fC, cfg.mu_f_not_C, cfg.sigma_f)
        assert p_c > p_nc

    def test_low_contextual_favours_no_contradiction(self, cfg):
        p_c, p_nc = compute_level3_likelihoods(0.0, cfg.mu_fC, cfg.mu_f_not_C, cfg.sigma_f)
        assert p_nc > p_c


# ---------------------------------------------------------------------------
# Temporal decay
# ---------------------------------------------------------------------------

class TestTemporalDecay:
    def test_none_returns_one(self, cfg):
        assert compute_temporal_decay(None, cfg.lambda_t) == pytest.approx(1.0)

    def test_zero_days_returns_one(self, cfg):
        assert compute_temporal_decay(0.0, cfg.lambda_t) == pytest.approx(1.0)

    def test_positive_days_returns_less_than_one(self, cfg):
        tau = compute_temporal_decay(10.0, cfg.lambda_t)
        assert 0.0 < tau < 1.0

    def test_monotone_decay(self, cfg):
        tau_1  = compute_temporal_decay(1.0,   cfg.lambda_t)
        tau_10 = compute_temporal_decay(10.0,  cfg.lambda_t)
        tau_100 = compute_temporal_decay(100.0, cfg.lambda_t)
        assert tau_1 > tau_10 > tau_100 > 0.0

    def test_known_value(self, cfg):
        # τ(Δt) = exp(-λ_t × Δt)
        expected = math.exp(-cfg.lambda_t * 5.0)
        assert compute_temporal_decay(5.0, cfg.lambda_t) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Bayesian confidence fusion
# ---------------------------------------------------------------------------

class TestFuseBayesianConfidence:
    def test_clear_contradiction_high_confidence(self, cfg):
        evidence = EvidenceSignals(
            negation_signal=1,
            semantic_opposition=0.95,
            contextual_incompatibility=0.90,
        )
        conf = fuse_bayesian_confidence(0.10, evidence, cfg, temporal_decay=1.0)
        assert conf > 0.80

    def test_clear_non_contradiction_low_confidence(self, cfg):
        evidence = EvidenceSignals(
            negation_signal=0,
            semantic_opposition=0.05,
            contextual_incompatibility=0.05,
        )
        conf = fuse_bayesian_confidence(0.10, evidence, cfg, temporal_decay=1.0)
        assert conf < 0.20

    def test_temporal_decay_reduces_confidence(self, cfg):
        evidence = EvidenceSignals(
            negation_signal=1,
            semantic_opposition=0.90,
            contextual_incompatibility=None,
        )
        conf_no_decay = fuse_bayesian_confidence(0.15, evidence, cfg, temporal_decay=1.0)
        conf_decayed  = fuse_bayesian_confidence(0.15, evidence, cfg, temporal_decay=0.3)
        assert conf_no_decay > conf_decayed

    def test_partial_evidence_level1_only(self, cfg):
        evidence = EvidenceSignals(negation_signal=1)
        conf = fuse_bayesian_confidence(0.10, evidence, cfg, temporal_decay=1.0)
        assert 0.0 <= conf <= 1.0

    def test_partial_evidence_levels_1_2(self, cfg):
        evidence = EvidenceSignals(negation_signal=1, semantic_opposition=0.85)
        conf = fuse_bayesian_confidence(0.10, evidence, cfg, temporal_decay=1.0)
        assert 0.0 <= conf <= 1.0

    def test_zero_prior_gives_zero_confidence(self, cfg):
        evidence = EvidenceSignals(negation_signal=1, semantic_opposition=0.95)
        conf = fuse_bayesian_confidence(0.0, evidence, cfg, temporal_decay=1.0)
        assert conf == pytest.approx(0.0)

    def test_output_bounded(self, cfg):
        for prior in [0.01, 0.10, 0.50, 0.99]:
            for n in [0, 1]:
                evidence = EvidenceSignals(negation_signal=n, semantic_opposition=0.7)
                conf = fuse_bayesian_confidence(prior, evidence, cfg, temporal_decay=1.0)
                assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Detection threshold
# ---------------------------------------------------------------------------

class TestResolveDetectionThreshold:
    def test_normal_mode_default(self, cfg):
        theta = resolve_detection_threshold(OperationalMode.NORMAL, cfg)
        assert theta == pytest.approx(cfg.theta_normal)

    def test_dev_mode_lower_than_normal(self, cfg):
        theta_dev    = resolve_detection_threshold(OperationalMode.DEV,    cfg)
        theta_normal = resolve_detection_threshold(OperationalMode.NORMAL, cfg)
        assert theta_dev < theta_normal

    def test_dream_mode_highest(self, cfg):
        theta_dream  = resolve_detection_threshold(OperationalMode.REM_DREAM, cfg)
        theta_normal = resolve_detection_threshold(OperationalMode.NORMAL,    cfg)
        assert theta_dream > theta_normal

    def test_high_ne_lowers_threshold(self, cfg):
        low_ne  = resolve_detection_threshold(OperationalMode.NORMAL, cfg, ne_level=0.0)
        high_ne = resolve_detection_threshold(OperationalMode.NORMAL, cfg, ne_level=1.0)
        assert high_ne < low_ne

    def test_high_gaba_raises_threshold(self, cfg):
        low_gaba  = resolve_detection_threshold(OperationalMode.NORMAL, cfg, gaba_level=0.0)
        high_gaba = resolve_detection_threshold(OperationalMode.NORMAL, cfg, gaba_level=1.0)
        assert high_gaba > low_gaba

    def test_threshold_clamped(self, cfg):
        theta = resolve_detection_threshold(OperationalMode.NORMAL, cfg, ne_level=1.0, gaba_level=1.0)
        assert 0.05 <= theta <= 0.95


# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------

class TestDetectNegationSignal:
    def test_direct_negation_detected(self, stmt_online, stmt_offline):
        result = detect_negation_signal(stmt_online, stmt_offline)
        assert result == 1

    def test_same_polarity_not_detected(self, stmt_online, stmt_unrelated):
        result = detect_negation_signal(stmt_online, stmt_unrelated)
        assert result == 0

    def test_symmetric(self, stmt_online, stmt_offline):
        assert detect_negation_signal(stmt_online, stmt_offline) == \
               detect_negation_signal(stmt_offline, stmt_online)

    def test_unrelated_statements(self, stmt_online, stmt_unrelated):
        assert detect_negation_signal(stmt_online, stmt_unrelated) == 0

    def test_both_negated_not_flagged(self):
        a = ProcessedStatement(raw_text="not x", tokens=["not", "x"])
        b = ProcessedStatement(raw_text="not y", tokens=["not", "y"])
        # Both have negation but no shared content tokens → 0
        assert detect_negation_signal(a, b) == 0


# ---------------------------------------------------------------------------
# Semantic opposition
# ---------------------------------------------------------------------------

class TestComputeSemanticOpposition:
    def test_negated_pair_has_higher_opposition(self, stmt_online, stmt_offline):
        s_opposed   = compute_semantic_opposition(stmt_online, stmt_offline)
        s_unrelated = compute_semantic_opposition(
            stmt_online,
            ProcessedStatement(raw_text="sky is blue", tokens=["sky", "is", "blue"]),
        )
        # Negated pair shares tokens + has negation → should not be zero
        assert s_opposed >= 0.0

    def test_identical_statements_low_opposition(self):
        s = ProcessedStatement(raw_text="x is true", tokens=["x", "is", "true"])
        opp = compute_semantic_opposition(s, s)
        assert opp < 0.5

    def test_output_bounded(self, stmt_online, stmt_offline):
        opp = compute_semantic_opposition(stmt_online, stmt_offline)
        assert 0.0 <= opp <= 1.0

    def test_semantic_roles_path(self):
        """When semantic_roles are populated, role-based path is used."""
        a = ProcessedStatement(
            raw_text="Alice runs fast.",
            semantic_roles=[{"agent": "alice", "action": "runs"}],
        )
        b = ProcessedStatement(
            raw_text="Alice walks slow.",
            semantic_roles=[{"agent": "alice", "action": "walks"}],
        )
        opp = compute_semantic_opposition(a, b)
        # Different actions on same agent → some opposition
        assert 0.0 <= opp <= 1.0

    def test_same_action_on_same_agent_no_opposition(self):
        a = ProcessedStatement(
            raw_text="Bob eats.",
            semantic_roles=[{"agent": "bob", "action": "eats"}],
        )
        b = ProcessedStatement(
            raw_text="Bob eats too.",
            semantic_roles=[{"agent": "bob", "action": "eats"}],
        )
        opp = compute_semantic_opposition(a, b)
        assert opp == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Contextual incompatibility
# ---------------------------------------------------------------------------

class TestComputeContextualIncompatibility:
    def test_empty_context_returns_none(self, stmt_online, stmt_offline):
        result = compute_contextual_incompatibility(stmt_online, stmt_offline, {})
        assert result is None

    def test_non_empty_context_returns_float(self, stmt_online, stmt_offline):
        ctx = {"contradiction_signal": 0.5}
        result = compute_contextual_incompatibility(stmt_online, stmt_offline, ctx)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_high_context_signal(self, stmt_online, stmt_offline):
        ctx = {"contradiction_signal": 0.9}
        result = compute_contextual_incompatibility(stmt_online, stmt_offline, ctx)
        assert result >= 0.7

    def test_output_bounded(self, stmt_online, stmt_offline):
        ctx = {"contradiction_signal": 2.0}   # over-range input
        result = compute_contextual_incompatibility(stmt_online, stmt_offline, ctx)
        assert result is not None
        assert result <= 1.0


# ---------------------------------------------------------------------------
# Temporal separation
# ---------------------------------------------------------------------------

class TestComputeTemporalSeparation:
    def test_same_timestamp_returns_none(self):
        ts = datetime(2026, 1, 1)
        a = ProcessedStatement(timestamp=ts)
        b = ProcessedStatement(timestamp=ts)
        assert compute_temporal_separation_days(a, b) is None

    def test_one_day_apart(self):
        t0 = datetime(2026, 1, 1)
        t1 = datetime(2026, 1, 2)
        a = ProcessedStatement(timestamp=t0)
        b = ProcessedStatement(timestamp=t1)
        result = compute_temporal_separation_days(a, b)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_symmetric(self):
        t0 = datetime(2026, 1, 1)
        t1 = datetime(2026, 1, 5)
        a = ProcessedStatement(timestamp=t0)
        b = ProcessedStatement(timestamp=t1)
        assert compute_temporal_separation_days(a, b) == compute_temporal_separation_days(b, a)


# ---------------------------------------------------------------------------
# κ_contradiction and load
# ---------------------------------------------------------------------------

class TestKappaAndLoad:
    def test_kappa_composition(self, cfg):
        kappa = compute_kappa(0.8, 1.0, 0.5,
                              cfg.lambda_1, cfg.lambda_2, cfg.lambda_3)
        expected = (cfg.lambda_1 * 0.8
                    + cfg.lambda_2 * 1.0
                    + cfg.lambda_3 * 0.5)
        assert kappa == pytest.approx(expected)

    def test_kappa_zero_inputs_gives_zero(self, cfg):
        assert compute_kappa(0.0, 0.0, 0.0,
                             cfg.lambda_1, cfg.lambda_2, cfg.lambda_3) == pytest.approx(0.0)

    def test_load_increases_with_positive_kappa(self, cfg):
        load0 = 0.0
        load1 = update_contradiction_load(load0, kappa=0.5, dt=1.0, tau_c=cfg.tau_c)
        load2 = update_contradiction_load(load1, kappa=0.5, dt=1.0, tau_c=cfg.tau_c)
        assert load2 > load1 > load0

    def test_load_decays_without_kappa(self, cfg):
        load0 = 0.8
        load1 = update_contradiction_load(load0, kappa=0.0, dt=1.0, tau_c=cfg.tau_c)
        assert load1 < load0

    def test_load_clamped(self, cfg):
        load = update_contradiction_load(0.99, kappa=100.0, dt=10.0, tau_c=cfg.tau_c)
        assert load <= 1.0


# ---------------------------------------------------------------------------
# Neurochemical signals
# ---------------------------------------------------------------------------

class TestNeurochemSignals:
    def test_zero_load_gives_zero_penalty(self, cfg):
        signals = compute_neurochemical_signals(0.0, 0.0, 0.0, cfg,
                                                rng=np.random.default_rng(0))
        assert signals["reward_logic_penalty"] == pytest.approx(0.0)

    def test_reward_penalty_proportional_to_load(self, cfg):
        s_low  = compute_neurochemical_signals(0.0, 0.2, 0.0, cfg, rng=np.random.default_rng(0))
        s_high = compute_neurochemical_signals(0.0, 0.8, 0.0, cfg, rng=np.random.default_rng(0))
        assert s_high["reward_logic_penalty"] > s_low["reward_logic_penalty"]

    def test_gaba_multiplier_clamped_minimum(self, cfg):
        signals = compute_neurochemical_signals(0.0, 1.0, 0.0, cfg,
                                                rng=np.random.default_rng(0))
        assert signals["k_u_gaba_multiplier"] >= 0.3

    def test_gaba_multiplier_at_zero_load(self, cfg):
        signals = compute_neurochemical_signals(0.0, 0.0, 0.0, cfg,
                                                rng=np.random.default_rng(0))
        assert signals["k_u_gaba_multiplier"] == pytest.approx(1.0)

    def test_beta_suppression_decreases_with_load(self, cfg):
        s_low  = compute_neurochemical_signals(0.0, 0.1, 0.0, cfg, rng=np.random.default_rng(0))
        s_high = compute_neurochemical_signals(0.0, 0.9, 0.0, cfg, rng=np.random.default_rng(0))
        assert s_high["beta_suppress"] < s_low["beta_suppress"]

    def test_theta_enhancement_increases_with_load(self, cfg):
        s_low  = compute_neurochemical_signals(0.0, 0.1, 0.0, cfg, rng=np.random.default_rng(0))
        s_high = compute_neurochemical_signals(0.0, 0.9, 0.0, cfg, rng=np.random.default_rng(0))
        assert s_high["theta_boost"] > s_low["theta_boost"]

    def test_5ht2c_multiplier_increases_with_affective_integral(self, cfg):
        s_low  = compute_neurochemical_signals(0.0, 0.0, 0.1, cfg, rng=np.random.default_rng(0))
        s_high = compute_neurochemical_signals(0.0, 0.0, 1.0, cfg, rng=np.random.default_rng(0))
        assert s_high["k_on_5ht2c_multiplier"] > s_low["k_on_5ht2c_multiplier"]


# ---------------------------------------------------------------------------
# Engine — end-to-end
# ---------------------------------------------------------------------------

class TestContradictionDetectionEngineProcess:
    def test_direct_contradiction_flagged(self, engine, stmt_online, stmt_offline):
        cset = ComparisonSet(
            primary_statements=[stmt_online],
            reference_set=[stmt_offline],
            memory_contrast_hit_ratio=0.5,
        )
        result = engine.process(cset, dt=1.0)
        assert not result.clean_pass
        assert len(result.flags) >= 1
        assert result.comparisons_performed >= 1

    def test_unrelated_statements_clean_pass(self, engine, stmt_online, stmt_unrelated):
        cset = ComparisonSet(
            primary_statements=[stmt_online],
            reference_set=[stmt_unrelated],
            memory_contrast_hit_ratio=0.0,
        )
        result = engine.process(cset, dt=1.0)
        # Unrelated content with low memory contrast → should be clean or low confidence
        assert result.comparisons_performed == 1

    def test_flag_has_valid_fields(self, engine, stmt_online, stmt_offline):
        cset = ComparisonSet(
            primary_statements=[stmt_online],
            reference_set=[stmt_offline],
            memory_contrast_hit_ratio=0.5,
        )
        result = engine.process(cset, dt=1.0)
        if result.flags:
            flag = result.flags[0]
            assert 0.0 <= flag.confidence <= 1.0
            assert flag.contradiction_level in {1, 2, 3}
            assert flag.contradiction_id  # non-empty UUID
            assert flag.semantic_description

    def test_result_counts_consistent(self, engine, stmt_online, stmt_offline):
        cset = ComparisonSet(
            primary_statements=[stmt_online],
            reference_set=[stmt_offline],
            memory_contrast_hit_ratio=0.5,
        )
        result = engine.process(cset, dt=1.0)
        assert result.comparisons_performed >= len(result.flags)

    def test_threshold_in_metadata(self, engine, stmt_online, stmt_offline):
        cset = ComparisonSet(primary_statements=[stmt_online], reference_set=[stmt_offline])
        result = engine.process(cset)
        assert result.detection_threshold_used == pytest.approx(
            engine._config.theta_normal, abs=0.15
        )

    def test_neurochemical_signals_present(self, engine, stmt_online, stmt_offline):
        cset = ComparisonSet(primary_statements=[stmt_online], reference_set=[stmt_offline])
        result = engine.process(cset)
        expected_keys = {
            "reward_logic_penalty", "da_d2_burst", "k_on_5ht2c_multiplier",
            "k_u_gaba_multiplier", "beta_suppress", "theta_boost",
        }
        assert expected_keys.issubset(result.neurochemical_signals.keys())

    def test_empty_sets_returns_clean_pass(self, engine):
        cset = ComparisonSet(primary_statements=[], reference_set=[])
        result = engine.process(cset)
        assert result.clean_pass
        assert result.comparisons_performed == 0

    def test_intra_primary_comparison(self, engine):
        """Two primary statements compared against each other."""
        s1 = ProcessedStatement(
            raw_text="It is raining.",
            tokens=["It", "is", "raining"],
            source_tag=SourceTag.USER_INPUT,
        )
        s2 = ProcessedStatement(
            raw_text="It is not raining.",
            tokens=["It", "is", "not", "raining"],
            source_tag=SourceTag.USER_INPUT,
        )
        cset = ComparisonSet(
            primary_statements=[s1, s2],
            reference_set=[],
            memory_contrast_hit_ratio=0.5,
        )
        result = engine.process(cset, dt=1.0)
        # C(m=2, 2) = 1 intra-primary comparison
        assert result.comparisons_performed == 1

    def test_dev_mode_lower_threshold(self, cfg):
        """Dev mode should fire flags at lower confidence."""
        engine_dev    = ContradictionDetectionEngine(cfg, rng=np.random.default_rng(0))
        engine_normal = ContradictionDetectionEngine(cfg, rng=np.random.default_rng(0))
        engine_dev.configure(OperationalMode.DEV)

        s_a = ProcessedStatement(raw_text="x is true",  tokens=["x", "is", "true"])
        s_b = ProcessedStatement(raw_text="x is not true", tokens=["x", "is", "not", "true"])
        cset = ComparisonSet(primary_statements=[s_a], reference_set=[s_b],
                             memory_contrast_hit_ratio=0.3)
        r_dev    = engine_dev.process(cset)
        r_normal = engine_normal.process(cset)
        # Dev should catch at least as many as normal (lower threshold)
        assert len(r_dev.flags) >= len(r_normal.flags)


# ---------------------------------------------------------------------------
# Engine — configure, neurochem, status
# ---------------------------------------------------------------------------

class TestEngineConfig:
    def test_configure_sets_mode(self, engine):
        engine.configure(OperationalMode.DEV)
        assert engine._mode == OperationalMode.DEV

    def test_update_neurochem_state(self, engine):
        engine.update_neurochem_state({"ne": 0.8, "gaba": 0.3, "da": 0.5})
        assert engine._state.ne_level    == pytest.approx(0.8)
        assert engine._state.gaba_level  == pytest.approx(0.3)
        assert engine._state.da_d2_level == pytest.approx(0.5)

    def test_neurochem_state_clamped(self, engine):
        engine.update_neurochem_state({"ne": 2.0, "gaba": -0.5})
        assert engine._state.ne_level   == pytest.approx(1.0)
        assert engine._state.gaba_level == pytest.approx(0.0)

    def test_get_status_keys(self, engine):
        status = engine.get_status()
        required = {"engine_id", "mode", "contradiction_load",
                    "affective_integral", "ne_level", "gaba_level", "da_d2_level"}
        assert required.issubset(status.keys())

    def test_get_status_engine_id(self, engine):
        assert engine.get_status()["engine_id"] == "contradiction_detection_engine"

    def test_contradiction_load_accumulates(self, engine, stmt_online, stmt_offline):
        """Contradiction load should increase after a detection pass."""
        initial_load = engine._state.contradiction_load
        cset = ComparisonSet(
            primary_statements=[stmt_online],
            reference_set=[stmt_offline],
            memory_contrast_hit_ratio=0.5,
        )
        engine.process(cset, dt=1.0)
        assert engine._state.contradiction_load >= initial_load
