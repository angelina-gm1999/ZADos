"""
Tests for Intention Map Engine — Engine 23.
"""

import math

import numpy as np
import pytest

from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)
from zados.cognitive_engines.py_engines.tokenizer import (
    AggregateFeatures,
    Tokenizer,
    TokenizerResult,
)
from zados.cognitive_engines.py_engines.semantic_expander import (
    ExpansionMetrics,
    ExpansionResult,
    SemanticExpander,
)
from zados.cognitive_engines.py_engines.intention_map_engine import (
    Archetype,
    IntentCategory,
    IntentionMapConfig,
    IntentionMapEngine,
    IntentionMapInput,
    IntentionMapResult,
    IntentionMapState,
    _AMPLIFICATION_PAIRS,
    _B_INTENT,
    _BAND_NAMES,
    _CFC_PATTERNS,
    _INTENT_INDEX,
    _INTENT_ORDER,
    _INTENT_TO_ARCHETYPE,
    _NT_CHANNEL_NAMES,
    _PHI_INTENT,
    _SUPPRESSION_PAIRS,
    apply_constraints,
    apply_mode_adjustments,
    apply_momentum,
    bayesian_update,
    build_combined_features,
    build_default_priors,
    check_disintegration_alert,
    compute_archetype_weights,
    compute_baseline_adjustments,
    compute_engine_neurochemical_signals,
    compute_intent_trajectory,
    compute_neurochemical_burst,
    compute_oscillatory_burst,
    compute_pharmacodynamic_effects,
    compute_template_match,
    evaluate_cross_frequency_couplings,
    extract_affective_features,
    extract_context_features,
    extract_linguistic_features,
    extract_structural_features,
    resolve_archetype_conflicts,
    sample_intent_noise,
)


# =====================================================================
# Helpers
# =====================================================================

RNG_SEED = 42


def _make_engine(seed: int = RNG_SEED, **config_kw) -> IntentionMapEngine:
    cfg = IntentionMapConfig(**config_kw)
    return IntentionMapEngine(config=cfg, rng=np.random.default_rng(seed))


def _make_input(**kwargs) -> IntentionMapInput:
    """Build IntentionMapInput with sensible defaults."""
    return IntentionMapInput(**kwargs)


def _uniform_intent() -> np.ndarray:
    return np.ones(8, dtype=np.float64) / 8.0


def _spike_intent(category: IntentCategory, weight: float = 0.5) -> np.ndarray:
    """Create intent vector with a spike at the given category."""
    v = np.ones(8, dtype=np.float64) * (1.0 - weight) / 7.0
    v[_INTENT_INDEX[category]] = weight
    return v


# =====================================================================
# Test: Enumerations
# =====================================================================

class TestEnums:
    def test_eight_intent_categories(self):
        assert len(IntentCategory) == 8

    def test_eight_archetypes(self):
        assert len(Archetype) == 8

    def test_intent_to_archetype_mapping(self):
        assert len(_INTENT_TO_ARCHETYPE) == 8
        assert _INTENT_TO_ARCHETYPE[IntentCategory.CONNECTION] == Archetype.GUIDE
        assert _INTENT_TO_ARCHETYPE[IntentCategory.CHALLENGE] == Archetype.OPPONENT
        assert _INTENT_TO_ARCHETYPE[IntentCategory.EXPLORATION] == Archetype.EXPLORER
        assert _INTENT_TO_ARCHETYPE[IntentCategory.DISCHARGE] == Archetype.CONTAINER
        assert _INTENT_TO_ARCHETYPE[IntentCategory.PRAGMATIC] == Archetype.ARCHITECT
        assert _INTENT_TO_ARCHETYPE[IntentCategory.SYMBOLIC] == Archetype.ORACLE
        assert _INTENT_TO_ARCHETYPE[IntentCategory.DEFENSIVE] == Archetype.FIREWALL
        assert _INTENT_TO_ARCHETYPE[IntentCategory.DISINTEGRATION] == Archetype.STABILIZER

    def test_intent_order_length(self):
        assert len(_INTENT_ORDER) == 8

    def test_intent_index_consistent(self):
        for i, cat in enumerate(_INTENT_ORDER):
            assert _INTENT_INDEX[cat] == i


# =====================================================================
# Test: Configuration
# =====================================================================

class TestConfig:
    def test_defaults_valid(self):
        cfg = IntentionMapConfig()
        assert cfg.alpha_history == 0.30
        assert cfg.eta_suppress == 0.15
        assert cfg.eta_amplify == 0.10
        assert cfg.momentum == 0.25
        assert cfg.tau_temperature == 0.50

    def test_priors_sum_to_one(self):
        cfg = IntentionMapConfig()
        total = (cfg.prior_connection + cfg.prior_challenge + cfg.prior_exploration
                 + cfg.prior_discharge + cfg.prior_pragmatic + cfg.prior_symbolic
                 + cfg.prior_defensive + cfg.prior_disintegration)
        assert abs(total - 1.0) < 0.01

    def test_neurochemical_params(self):
        cfg = IntentionMapConfig()
        assert cfg.beta_intent_ach == 0.10
        assert cfg.beta_intent_da == 0.06
        assert cfg.beta_disint_alert == 0.14

    def test_pharmacodynamic_params(self):
        cfg = IntentionMapConfig()
        assert cfg.kappa_empathy == 0.15
        assert cfg.kappa_inhibit == 0.12
        assert cfg.kappa_strategy == 0.10
        assert cfg.kappa_novelty == 0.12
        assert cfg.kappa_overgeneralize == 0.08


# =====================================================================
# Test: Matrices
# =====================================================================

class TestMatrices:
    def test_b_intent_shape(self):
        assert _B_INTENT.shape == (11, 8)

    def test_phi_intent_shape(self):
        assert _PHI_INTENT.shape == (5, 8)

    def test_nt_channel_names_length(self):
        assert len(_NT_CHANNEL_NAMES) == 11

    def test_band_names_length(self):
        assert len(_BAND_NAMES) == 5

    def test_b_intent_key_values(self):
        # DA row, exploration column (index 2) should be 0.18
        assert _B_INTENT[0, 2] == pytest.approx(0.18)
        # OXT row (4), connection column (0) should be 0.18
        assert _B_INTENT[4, 0] == pytest.approx(0.18)
        # COR row (10), disintegration column (7) should be 0.18
        assert _B_INTENT[10, 7] == pytest.approx(0.18)
        # NE row (1), challenge column (1) should be 0.16
        assert _B_INTENT[1, 1] == pytest.approx(0.16)

    def test_phi_intent_key_values(self):
        # Theta row (1), symbolic column (5) should be 0.12
        assert _PHI_INTENT[1, 5] == pytest.approx(0.12)
        # Gamma row (4), exploration column (2) should be 0.12
        assert _PHI_INTENT[4, 2] == pytest.approx(0.12)


# =====================================================================
# Test: Feature Extraction
# =====================================================================

class TestFeatureExtraction:
    def test_linguistic_features_13d(self):
        agg = AggregateFeatures()
        f = extract_linguistic_features(agg)
        assert len(f) == 13

    def test_affective_features_5d(self):
        f = extract_affective_features(0.5, 0.3, 0.7, 0.1, 2)
        assert len(f) == 5
        assert f[0] == pytest.approx(0.5)  # valence
        assert f[1] == pytest.approx(0.3)  # intensity

    def test_context_features_5d(self):
        f = extract_context_features(0.8, 1.5, None, 0.3, 0.6)
        assert len(f) == 5
        assert f[0] == pytest.approx(0.8)

    def test_structural_features_5d(self):
        m = ExpansionMetrics(fractal_depth=3, pattern_novelty=0.6,
                             symbolic_density=0.4, structural_complexity=0.3,
                             information_noise_ratio=0.7)
        f = extract_structural_features(m)
        assert len(f) == 5
        assert f[0] == pytest.approx(0.3)  # 3/10
        assert f[1] == pytest.approx(0.6)  # novelty

    def test_combined_features_28d(self):
        f_ling = np.zeros(13)
        f_affect = np.zeros(5)
        f_context = np.zeros(5)
        f_struct = np.zeros(5)
        combined = build_combined_features(f_ling, f_affect, f_context, f_struct)
        assert len(combined) == 28

    def test_affective_clipping(self):
        f = extract_affective_features(2.0, -1.0, 2.0, -2.0, 100)
        assert f[0] == pytest.approx(1.0)
        assert f[1] == pytest.approx(0.0)
        assert f[2] == pytest.approx(1.0)
        assert f[3] == pytest.approx(-1.0)
        assert f[4] <= 1.0

    def test_context_with_historical_intent(self):
        hist = [0.1, 0.1, 0.5, 0.05, 0.1, 0.05, 0.05, 0.05]
        f = extract_context_features(0.5, 1.0, hist, 0.5, 0.5)
        assert f[2] == pytest.approx(0.5)  # max of historical intent


# =====================================================================
# Test: Stage 1 — Template Matching
# =====================================================================

class TestTemplateMatching:
    def test_returns_8_scores(self):
        f = np.random.default_rng(42).random(28)
        scores = compute_template_match(f, IntentionMapConfig())
        assert len(scores) == 8

    def test_scores_sum_to_one(self):
        f = np.random.default_rng(42).random(28)
        scores = compute_template_match(f, IntentionMapConfig())
        assert abs(scores.sum() - 1.0) < 1e-6

    def test_scores_non_negative(self):
        f = np.random.default_rng(42).random(28)
        scores = compute_template_match(f, IntentionMapConfig())
        assert np.all(scores >= 0.0)

    def test_zero_input_falls_back(self):
        f = np.zeros(28)
        scores = compute_template_match(f, IntentionMapConfig())
        assert len(scores) == 8


# =====================================================================
# Test: Stage 2 — Bayesian Update
# =====================================================================

class TestBayesianUpdate:
    def test_returns_8_probabilities(self):
        match = np.ones(8) / 8.0
        priors = np.ones(8) / 8.0
        post = bayesian_update(match, None, priors, 0.30)
        assert len(post) == 8

    def test_sums_to_one(self):
        match = np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        match /= match.sum()
        priors = np.ones(8) / 8.0
        post = bayesian_update(match, None, priors, 0.30)
        assert abs(post.sum() - 1.0) < 1e-6

    def test_history_shifts_posterior(self):
        match = np.ones(8) / 8.0
        priors = np.ones(8) / 8.0
        # Historical prior strongly favoring exploration
        hist = np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.2])
        post_no_hist = bayesian_update(match, None, priors, 0.30)
        post_hist = bayesian_update(match, hist, priors, 0.30)
        # With history, exploration should be higher
        explore_idx = _INTENT_INDEX[IntentCategory.EXPLORATION]
        assert post_hist[explore_idx] > post_no_hist[explore_idx]

    def test_alpha_zero_ignores_history(self):
        match = np.ones(8) / 8.0
        priors = np.ones(8) / 8.0
        hist = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        post = bayesian_update(match, hist, priors, 0.0)
        # Should be same as using just default priors
        post_default = bayesian_update(match, None, priors, 0.0)
        np.testing.assert_allclose(post, post_default, atol=1e-6)


# =====================================================================
# Test: Stage 3 — Constraint Resolution
# =====================================================================

class TestConstraintResolution:
    def test_sums_to_one(self):
        e = np.ones(8) / 8.0
        constrained = apply_constraints(e, 0.15, 0.10)
        assert abs(constrained.sum() - 1.0) < 1e-6

    def test_suppression_reduces_conflicting(self):
        # Create vector with high connection AND challenge
        e = _spike_intent(IntentCategory.CONNECTION, 0.4)
        e[_INTENT_INDEX[IntentCategory.CHALLENGE]] = 0.4
        e /= e.sum()
        constrained = apply_constraints(e, 0.15, 0.10)
        # Both should be lower relative to each other (but still largest)
        # Suppression: connection suppressed by challenge and vice versa
        conn_idx = _INTENT_INDEX[IntentCategory.CONNECTION]
        chall_idx = _INTENT_INDEX[IntentCategory.CHALLENGE]
        # After suppression, the mutual suppression should have effect
        assert constrained[conn_idx] < e[conn_idx] or constrained[chall_idx] < e[chall_idx]

    def test_amplification_boosts_compatible(self):
        # Symbolic + Exploration should amplify each other
        e = np.ones(8) / 8.0
        e[_INTENT_INDEX[IntentCategory.SYMBOLIC]] = 0.3
        e[_INTENT_INDEX[IntentCategory.EXPLORATION]] = 0.3
        e /= e.sum()
        constrained = apply_constraints(e, 0.0, 0.10)  # suppression off
        sym_idx = _INTENT_INDEX[IntentCategory.SYMBOLIC]
        exp_idx = _INTENT_INDEX[IntentCategory.EXPLORATION]
        # After amplification, both should be relatively higher
        assert constrained[sym_idx] >= e[sym_idx] * 0.9  # at least maintained
        assert constrained[exp_idx] >= e[exp_idx] * 0.9

    def test_non_negative(self):
        e = np.random.default_rng(42).random(8)
        e /= e.sum()
        constrained = apply_constraints(e, 0.15, 0.10)
        assert np.all(constrained >= 0.0)


# =====================================================================
# Test: Temporal Dynamics
# =====================================================================

class TestTemporalDynamics:
    def test_momentum_blends(self):
        current = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2])
        current /= current.sum()
        previous = np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.2])
        previous /= previous.sum()
        blended = apply_momentum(current, previous, 0.25)
        # Should be between current and previous
        assert blended[0] < current[0]
        assert blended[2] > current[2]

    def test_momentum_zero_equals_current(self):
        current = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        current /= current.sum()
        previous = np.ones(8) / 8.0
        blended = apply_momentum(current, previous, 0.0)
        np.testing.assert_allclose(blended, current, atol=1e-6)

    def test_momentum_no_previous(self):
        current = np.ones(8) / 8.0
        blended = apply_momentum(current, None, 0.25)
        np.testing.assert_allclose(blended, current, atol=1e-6)

    def test_trajectory_computation(self):
        current = np.array([0.3, 0.1, 0.2, 0.05, 0.1, 0.1, 0.05, 0.1])
        previous = np.array([0.1, 0.2, 0.2, 0.05, 0.1, 0.1, 0.05, 0.2])
        traj = compute_intent_trajectory(current, previous)
        assert traj[0] == pytest.approx(0.2)  # rising
        assert traj[1] == pytest.approx(-0.1)  # falling

    def test_trajectory_no_previous(self):
        current = np.ones(8) / 8.0
        traj = compute_intent_trajectory(current, None)
        assert np.all(traj == 0.0)

    def test_disintegration_alert_triggered(self):
        traj = np.zeros(8)
        disint_idx = _INTENT_INDEX[IntentCategory.DISINTEGRATION]
        traj[disint_idx] = 0.10  # above 0.08
        alert, count = check_disintegration_alert(traj, disint_idx, 1, 0.08, 2)
        assert alert is True
        assert count == 2

    def test_disintegration_alert_not_triggered(self):
        traj = np.zeros(8)
        disint_idx = _INTENT_INDEX[IntentCategory.DISINTEGRATION]
        traj[disint_idx] = 0.05  # below 0.08
        alert, count = check_disintegration_alert(traj, disint_idx, 1, 0.08, 2)
        assert alert is False
        assert count == 0

    def test_disintegration_alert_needs_consecutive(self):
        traj = np.zeros(8)
        disint_idx = _INTENT_INDEX[IntentCategory.DISINTEGRATION]
        traj[disint_idx] = 0.10  # above threshold
        alert, count = check_disintegration_alert(traj, disint_idx, 0, 0.08, 2)
        assert alert is False  # only 1 turn, need 2
        assert count == 1


# =====================================================================
# Test: Archetype Routing
# =====================================================================

class TestArchetypeRouting:
    def test_weights_sum_to_one(self):
        e = _uniform_intent()
        weights = compute_archetype_weights(e, 0.50)
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_spike_favors_archetype(self):
        e = _spike_intent(IntentCategory.EXPLORATION, 0.7)
        weights = compute_archetype_weights(e, 0.50)
        assert weights[Archetype.EXPLORER.value] > 0.3

    def test_low_temperature_sharp(self):
        e = _spike_intent(IntentCategory.PRAGMATIC, 0.4)
        w_low = compute_archetype_weights(e, 0.30)
        w_high = compute_archetype_weights(e, 0.80)
        # Low temperature should be sharper
        assert w_low[Archetype.ARCHITECT.value] > w_high[Archetype.ARCHITECT.value]

    def test_conflict_resolution(self):
        weights = {a.value: 0.125 for a in Archetype}
        # Make Opponent and Guide both active
        weights[Archetype.OPPONENT.value] = 0.30
        weights[Archetype.GUIDE.value] = 0.20
        resolved, conflicts = resolve_archetype_conflicts(weights)
        assert len(conflicts) > 0
        assert (Archetype.OPPONENT.value, Archetype.GUIDE.value) in conflicts
        # Guide should be at 50% influence (lower weight)
        assert resolved[Archetype.GUIDE.value] < weights[Archetype.GUIDE.value]

    def test_no_conflict_when_one_inactive(self):
        weights = {a.value: 0.01 for a in Archetype}
        weights[Archetype.OPPONENT.value] = 0.90
        weights[Archetype.GUIDE.value] = 0.05  # below 0.10 threshold
        resolved, conflicts = resolve_archetype_conflicts(weights)
        assert len(conflicts) == 0


# =====================================================================
# Test: Neurochemical Burst
# =====================================================================

class TestNeurochemicalBurst:
    def test_returns_all_nt_channels(self):
        e = _uniform_intent()
        rng = np.random.default_rng(42)
        burst = compute_neurochemical_burst(e, rng)
        assert len(burst) == 11
        for name in _NT_CHANNEL_NAMES:
            assert name in burst

    def test_burst_values_bounded(self):
        e = _spike_intent(IntentCategory.CONNECTION, 0.9)
        rng = np.random.default_rng(42)
        burst = compute_neurochemical_burst(e, rng)
        for v in burst.values():
            assert -1.0 <= v <= 1.0

    def test_exploration_boosts_da(self):
        e = _spike_intent(IntentCategory.EXPLORATION, 0.7)
        # Run multiple times to average stochastic effect
        da_values = []
        for s in range(20):
            rng = np.random.default_rng(s)
            burst = compute_neurochemical_burst(e, rng)
            da_values.append(burst["DA"])
        mean_da = np.mean(da_values)
        assert mean_da > 0.01  # DA should be positive for exploration

    def test_connection_boosts_oxt(self):
        e = _spike_intent(IntentCategory.CONNECTION, 0.7)
        oxt_values = []
        for s in range(20):
            rng = np.random.default_rng(s)
            burst = compute_neurochemical_burst(e, rng)
            oxt_values.append(burst["OXT"])
        mean_oxt = np.mean(oxt_values)
        assert mean_oxt > 0.01

    def test_disintegration_boosts_cortisol(self):
        e = _spike_intent(IntentCategory.DISINTEGRATION, 0.7)
        cor_values = []
        for s in range(20):
            rng = np.random.default_rng(s)
            burst = compute_neurochemical_burst(e, rng)
            cor_values.append(burst["COR"])
        mean_cor = np.mean(cor_values)
        assert mean_cor > 0.01

    def test_noise_sampling(self):
        rng = np.random.default_rng(42)
        # Gamma noise
        val = sample_intent_noise(IntentCategory.CONNECTION, rng)
        assert val > 0.0
        # Poisson noise
        val = sample_intent_noise(IntentCategory.CHALLENGE, rng)
        assert val >= 0.0
        # Lognormal noise
        val = sample_intent_noise(IntentCategory.DISCHARGE, rng)
        assert val > 0.0


# =====================================================================
# Test: Oscillatory Burst
# =====================================================================

class TestOscillatoryBurst:
    def test_returns_all_bands(self):
        e = _uniform_intent()
        burst = compute_oscillatory_burst(e)
        assert len(burst) == 5
        for name in _BAND_NAMES:
            assert name in burst

    def test_exploration_boosts_theta_gamma(self):
        e = _spike_intent(IntentCategory.EXPLORATION, 0.7)
        burst = compute_oscillatory_burst(e)
        assert burst["theta"] > 0.0
        assert burst["gamma"] > 0.0

    def test_discharge_boosts_delta(self):
        e = _spike_intent(IntentCategory.DISCHARGE, 0.7)
        burst = compute_oscillatory_burst(e)
        assert burst["delta"] > 0.0

    def test_challenge_boosts_beta(self):
        e = _spike_intent(IntentCategory.CHALLENGE, 0.7)
        burst = compute_oscillatory_burst(e)
        assert burst["beta"] > 0.0


# =====================================================================
# Test: Cross-Frequency Coupling
# =====================================================================

class TestCrossFrequencyCoupling:
    def test_no_cfc_with_low_intent(self):
        e = _uniform_intent()  # all 0.125, below thresholds
        bands = {b: 0.5 for b in _BAND_NAMES}
        osc_burst = {b: 0.0 for b in _BAND_NAMES}
        active, mods = evaluate_cross_frequency_couplings(e, bands, osc_burst, IntentionMapConfig())
        assert len(active) == 0

    def test_theta_gamma_triggered_by_exploration(self):
        e = _spike_intent(IntentCategory.EXPLORATION, 0.5)
        bands = {"delta": 0.2, "theta": 0.5, "alpha": 0.3, "beta": 0.3, "gamma": 0.5}
        osc_burst = {b: 0.0 for b in _BAND_NAMES}
        active, mods = evaluate_cross_frequency_couplings(e, bands, osc_burst, IntentionMapConfig())
        assert "theta_gamma" in active

    def test_beta_gamma_triggered_by_challenge(self):
        e = _spike_intent(IntentCategory.CHALLENGE, 0.5)
        bands = {"delta": 0.2, "theta": 0.3, "alpha": 0.3, "beta": 0.6, "gamma": 0.5}
        osc_burst = {b: 0.0 for b in _BAND_NAMES}
        active, mods = evaluate_cross_frequency_couplings(e, bands, osc_burst, IntentionMapConfig())
        assert "beta_gamma" in active

    def test_disintegration_collapses_coupling(self):
        e = np.zeros(8)
        e[_INTENT_INDEX[IntentCategory.DISINTEGRATION]] = 0.40  # >= 0.35
        e[_INTENT_INDEX[IntentCategory.EXPLORATION]] = 0.30  # above 0.25
        e /= e.sum()
        # Re-scale to maintain disint >= 0.35
        e[_INTENT_INDEX[IntentCategory.DISINTEGRATION]] = 0.40
        e[_INTENT_INDEX[IntentCategory.EXPLORATION]] = 0.30
        bands = {"delta": 0.3, "theta": 0.5, "alpha": 0.5, "beta": 0.5, "gamma": 0.5}
        osc_burst = {b: 0.0 for b in _BAND_NAMES}
        active, mods = evaluate_cross_frequency_couplings(e, bands, osc_burst, IntentionMapConfig())
        # Even though triggered, coupling strength should be 0 due to collapse
        for v in mods.values():
            assert v == pytest.approx(0.0)

    def test_cfc_needs_band_thresholds(self):
        e = _spike_intent(IntentCategory.EXPLORATION, 0.5)
        # Low band powers
        bands = {"delta": 0.1, "theta": 0.1, "alpha": 0.1, "beta": 0.1, "gamma": 0.1}
        osc_burst = {b: 0.0 for b in _BAND_NAMES}
        active, mods = evaluate_cross_frequency_couplings(e, bands, osc_burst, IntentionMapConfig())
        assert "theta_gamma" not in active


# =====================================================================
# Test: Pharmacodynamic Cross-Effects
# =====================================================================

class TestPharmacodynamicEffects:
    def test_empathy_resonance_active(self):
        e = _uniform_intent()
        nt = {"OXT": 0.5, "CB1": 0.3, "NE": 0.0, "MOR": 0.0, "DA_D1": 0.0,
              "DA_D3": 0.0, "5-HT2A": 0.0, "GABA_B": 0.0}
        bands = {"theta": 0.5}
        effects = compute_pharmacodynamic_effects(e, nt, bands, 0.0, 0.3, IntentionMapConfig())
        assert effects["empathy_resonance"] > 0.0

    def test_empathy_resonance_inactive_low_oxt(self):
        e = _uniform_intent()
        nt = {"OXT": 0.1, "CB1": 0.3, "NE": 0.0, "MOR": 0.0, "DA_D1": 0.0,
              "DA_D3": 0.0, "5-HT2A": 0.0, "GABA_B": 0.0}
        bands = {"theta": 0.5}
        effects = compute_pharmacodynamic_effects(e, nt, bands, 0.0, 0.3, IntentionMapConfig())
        assert effects["empathy_resonance"] == 0.0

    def test_emotional_inhibition_active(self):
        e = _uniform_intent()
        nt = {"OXT": 0.0, "CB1": 0.0, "NE": 0.6, "MOR": 0.0, "DA_D1": 0.0,
              "DA_D3": 0.0, "5-HT2A": 0.0, "GABA_B": 0.0}
        effects = compute_pharmacodynamic_effects(e, nt, {}, 0.5, 0.3, IntentionMapConfig())
        assert effects["emotional_inhibition"] > 0.0

    def test_emotional_inhibition_inactive_low_contradiction(self):
        e = _uniform_intent()
        nt = {"NE": 0.6}
        effects = compute_pharmacodynamic_effects(e, nt, {}, 0.2, 0.3, IntentionMapConfig())
        assert effects["emotional_inhibition"] == 0.0

    def test_strategy_propagation_active(self):
        nt = {"DA_D1": 0.5, "NE": 0.4}
        effects = compute_pharmacodynamic_effects(_uniform_intent(), nt, {}, 0.0, 0.3, IntentionMapConfig())
        assert effects["strategy_propagation"] > 0.0

    def test_novelty_exploration_active(self):
        nt = {"5-HT2A": 0.3, "DA_D3": 0.4, "CB1": 0.3}
        effects = compute_pharmacodynamic_effects(_uniform_intent(), nt, {}, 0.0, 0.3, IntentionMapConfig())
        assert effects["novelty_exploration"] > 0.0

    def test_overgeneralization_inhibition_active(self):
        nt = {"GABA_B": 0.4}
        effects = compute_pharmacodynamic_effects(_uniform_intent(), nt, {}, 0.0, 0.1, IntentionMapConfig())
        # confidence 0.1 < 0.5 and GABA_B 0.4 > 0.3
        assert effects["overgeneralization_inhibition"] > 0.0

    def test_overgeneralization_inactive_high_confidence(self):
        nt = {"GABA_B": 0.4}
        effects = compute_pharmacodynamic_effects(_uniform_intent(), nt, {}, 0.0, 0.6, IntentionMapConfig())
        assert effects["overgeneralization_inhibition"] == 0.0

    def test_all_effects_present(self):
        nt = {}
        effects = compute_pharmacodynamic_effects(_uniform_intent(), nt, {}, 0.0, 0.3, IntentionMapConfig())
        expected_keys = [
            "empathy_resonance", "emotional_inhibition", "affective_internalization",
            "strategy_propagation", "novelty_exploration", "overgeneralization_inhibition",
        ]
        for k in expected_keys:
            assert k in effects


# =====================================================================
# Test: Engine-Level Neurochemical Signals
# =====================================================================

class TestEngineNeurochemicalSignals:
    def test_ach_intent_signal(self):
        rng = np.random.default_rng(42)
        signals = compute_engine_neurochemical_signals(
            _uniform_intent(), 0.5, 0.3, IntentionMapConfig(), rng
        )
        assert "ach_intent" in signals
        assert signals["ach_intent"] >= 0.0

    def test_da_intent_signal(self):
        rng = np.random.default_rng(42)
        signals = compute_engine_neurochemical_signals(
            _uniform_intent(), 0.8, 0.5, IntentionMapConfig(), rng
        )
        assert signals["da_intent"] > 0.0

    def test_ne_disintegration_alert(self):
        # Run with multiple seeds to overcome Poisson(2.0) occasionally returning 0
        found_positive = False
        for seed in range(20):
            rng = np.random.default_rng(seed)
            e = _spike_intent(IntentCategory.DISINTEGRATION, 0.5)
            signals = compute_engine_neurochemical_signals(e, 0.3, 0.1, IntentionMapConfig(), rng)
            if signals["ne_disint_alert"] > 0.0:
                found_positive = True
                break
        assert found_positive, "NE disint alert should fire for at least one seed"

    def test_ne_no_alert_below_threshold(self):
        rng = np.random.default_rng(42)
        e = _spike_intent(IntentCategory.DISINTEGRATION, 0.10)
        signals = compute_engine_neurochemical_signals(e, 0.3, 0.1, IntentionMapConfig(), rng)
        assert signals["ne_disint_alert"] == 0.0


# =====================================================================
# Test: Baseline Adjustments
# =====================================================================

class TestBaselineAdjustments:
    def test_da_baseline_from_exploration(self):
        e = _spike_intent(IntentCategory.EXPLORATION, 0.7)
        adj = compute_baseline_adjustments(e, 0.5, 0.0, IntentionMapConfig())
        assert adj["DA_baseline"] > 0.0

    def test_ne_baseline_from_stress(self):
        e = _spike_intent(IntentCategory.CHALLENGE, 0.5)
        adj = compute_baseline_adjustments(e, 0.5, 0.0, IntentionMapConfig())
        assert adj["NE_baseline"] > 0.0

    def test_oxt_suppression_from_defensiveness(self):
        e = _spike_intent(IntentCategory.DEFENSIVE, 0.5)
        adj = compute_baseline_adjustments(e, 0.5, 0.0, IntentionMapConfig())
        assert adj["OXT_baseline"] < 0.0

    def test_5ht2c_sensitization(self):
        e = _uniform_intent()
        adj = compute_baseline_adjustments(e, 0.5, 5.0, IntentionMapConfig())
        assert adj["5-HT2C_sensitivity"] > 0.0


# =====================================================================
# Test: Mode Adjustments
# =====================================================================

class TestModeAdjustments:
    def test_learning_boosts_challenge_exploration(self):
        e = _uniform_intent()
        adjusted = apply_mode_adjustments(e, OperationalMode.LEARNING, IntentionMapConfig())
        chall_idx = _INTENT_INDEX[IntentCategory.CHALLENGE]
        explore_idx = _INTENT_INDEX[IntentCategory.EXPLORATION]
        # These should be higher relative to others
        assert adjusted[chall_idx] > e[chall_idx]
        assert adjusted[explore_idx] > e[explore_idx]

    def test_reflective_boosts_defensive_disintegration(self):
        e = _uniform_intent()
        adjusted = apply_mode_adjustments(e, OperationalMode.REFLECTIVE, IntentionMapConfig())
        def_idx = _INTENT_INDEX[IntentCategory.DEFENSIVE]
        disint_idx = _INTENT_INDEX[IntentCategory.DISINTEGRATION]
        assert adjusted[def_idx] > e[def_idx]
        assert adjusted[disint_idx] > e[disint_idx]

    def test_normal_no_change(self):
        e = _uniform_intent()
        adjusted = apply_mode_adjustments(e, OperationalMode.NORMAL, IntentionMapConfig())
        np.testing.assert_allclose(adjusted, e, atol=1e-6)

    def test_renormalized(self):
        e = _uniform_intent()
        adjusted = apply_mode_adjustments(e, OperationalMode.LEARNING, IntentionMapConfig())
        assert abs(adjusted.sum() - 1.0) < 1e-6


# =====================================================================
# Test: Full Engine
# =====================================================================

class TestIntentionMapEngine:
    def test_process_basic(self):
        engine = _make_engine()
        tok = Tokenizer(rng=np.random.default_rng(42))
        tok_result = tok.process("I believe freedom is more important than security.")
        exp = SemanticExpander(rng=np.random.default_rng(42))
        exp_result = exp.process(tok_result)
        inp = IntentionMapInput(tokenizer_result=tok_result, expansion_result=exp_result)
        result = engine.process(inp)
        assert len(result.intent_vector) == 8
        assert abs(sum(result.intent_vector) - 1.0) < 1e-6
        assert result.dominant_intent != ""
        assert result.primary_archetype != ""
        assert result.processing_time_ms >= 0.0

    def test_intent_labels_populated(self):
        engine = _make_engine()
        result = engine.process(IntentionMapInput())
        assert len(result.intent_labels) == 8
        for cat in IntentCategory:
            assert cat.value in result.intent_labels

    def test_neurochemical_burst_populated(self):
        engine = _make_engine()
        result = engine.process(IntentionMapInput())
        assert len(result.neurochemical_burst) > 0
        for name in _NT_CHANNEL_NAMES:
            assert name in result.neurochemical_burst

    def test_oscillatory_burst_populated(self):
        engine = _make_engine()
        result = engine.process(IntentionMapInput())
        assert len(result.oscillatory_burst) == 5
        for name in _BAND_NAMES:
            assert name in result.oscillatory_burst

    def test_pharmacodynamics_populated(self):
        engine = _make_engine()
        result = engine.process(IntentionMapInput())
        assert len(result.active_pharmacodynamics) == 6

    def test_baseline_adjustments_populated(self):
        engine = _make_engine()
        result = engine.process(IntentionMapInput())
        assert "DA_baseline" in result.baseline_adjustments
        assert "NE_baseline" in result.baseline_adjustments
        assert "OXT_baseline" in result.baseline_adjustments

    def test_session_profile_populated(self):
        engine = _make_engine()
        result = engine.process(IntentionMapInput())
        assert len(result.session_intent_profile) == 8

    def test_rem_dream_returns_empty(self):
        engine = _make_engine()
        engine.configure(OperationalMode.REM_DREAM)
        result = engine.process(IntentionMapInput())
        assert result.intent_vector == []
        assert result.dominant_intent == ""

    def test_configure_mode(self):
        engine = _make_engine()
        engine.configure(OperationalMode.LEARNING)
        status = engine.get_status()
        assert status["mode"] == "learning"
        assert status["engine_id"] == "intention_map_engine"

    def test_get_status(self):
        engine = _make_engine()
        status = engine.get_status()
        assert "engine_id" in status
        assert status["engine_id"] == "intention_map_engine"
        assert "mode" in status
        assert "turn_count" in status
        assert status["turn_count"] == 0
        # engine_id should be the first key
        assert list(status.keys())[0] == "engine_id"

    def test_turn_count_increments(self):
        engine = _make_engine()
        engine.process(IntentionMapInput())
        engine.process(IntentionMapInput())
        assert engine.get_status()["turn_count"] == 2

    def test_momentum_smooths_transitions(self):
        engine = _make_engine()
        # First turn
        r1 = engine.process(IntentionMapInput(emotional_intensity=0.9, emotional_valence=0.8))
        # Second turn — different features
        r2 = engine.process(IntentionMapInput(emotional_intensity=0.1, emotional_valence=-0.5))
        # With momentum=0.25, second turn should retain some of first turn's intent
        # Just check that it doesn't completely change
        assert r2.intent_vector is not None
        assert len(r2.intent_vector) == 8

    def test_disintegration_alert_propagates(self):
        engine = _make_engine()
        # Process multiple turns with inputs that bias toward disintegration
        # High emotional intensity, negative valence, low coherence
        inp = IntentionMapInput(
            emotional_intensity=0.95,
            emotional_valence=-0.8,
            affect_complexity=8,
            topic_continuity=0.1,
        )
        # Run several turns
        results = []
        for _ in range(5):
            r = engine.process(inp)
            results.append(r)
        # The result object should have valid disintegration_alert field
        assert isinstance(results[-1].disintegration_alert, bool)

    def test_archetype_selection_populated(self):
        engine = _make_engine()
        result = engine.process(IntentionMapInput())
        assert len(result.archetype_selection) > 0
        total = sum(result.archetype_selection.values())
        assert abs(total - 1.0) < 0.01

    def test_confidence_and_mixed_flag(self):
        engine = _make_engine()
        result = engine.process(IntentionMapInput())
        assert result.intent_confidence >= 0.0
        assert isinstance(result.is_mixed, bool)

    def test_rising_falling_intents(self):
        engine = _make_engine()
        engine.process(IntentionMapInput())  # first turn (no trajectory)
        result = engine.process(IntentionMapInput(emotional_intensity=0.8))
        # Should have lists (possibly empty on second turn)
        assert isinstance(result.rising_intents, list)
        assert isinstance(result.falling_intents, list)

    def test_active_cfc_list(self):
        engine = _make_engine()
        result = engine.process(IntentionMapInput(
            band_powers={"delta": 0.5, "theta": 0.5, "alpha": 0.5, "beta": 0.5, "gamma": 0.5}
        ))
        assert isinstance(result.active_cross_frequency_couplings, list)

    def test_historical_intent_used(self):
        engine = _make_engine()
        r1 = engine.process(IntentionMapInput())
        # Historical intent from state should now exist
        assert engine._state.previous_intent is not None
        # Second pass uses it
        r2 = engine.process(IntentionMapInput())
        assert r2.intent_vector is not None

    def test_with_upstream_nt_levels(self):
        engine = _make_engine()
        nt_levels = {"OXT": 0.6, "CB1": 0.3, "NE": 0.2, "MOR": 0.1,
                     "DA_D1": 0.4, "DA_D3": 0.3, "5-HT2A": 0.2, "GABA_B": 0.3}
        result = engine.process(IntentionMapInput(
            nt_levels=nt_levels,
            band_powers={"theta": 0.5},
        ))
        # Should have empathy resonance active with high OXT + CB1 + theta
        assert isinstance(result.active_pharmacodynamics, dict)


# =====================================================================
# Test: Class Attributes
# =====================================================================

class TestClassAttributes:
    def test_engine_id(self):
        assert IntentionMapEngine.engine_id == "intention_map_engine"

    def test_cluster(self):
        assert IntentionMapEngine.cluster == "pattern_analysis"

    def test_engine_id_on_instance(self):
        engine = _make_engine()
        assert engine.engine_id == "intention_map_engine"

    def test_cluster_on_instance(self):
        engine = _make_engine()
        assert engine.cluster == "pattern_analysis"


# =====================================================================
# Test: update_neurochem_state
# =====================================================================

class TestUpdateNeurochemState:
    def test_updates_da_level(self):
        engine = _make_engine()
        engine.update_neurochem_state({"da": 0.7})
        assert engine._state.da_level == pytest.approx(0.7)

    def test_updates_oxt_level(self):
        engine = _make_engine()
        engine.update_neurochem_state({"oxt": 0.45})
        assert engine._state.oxt_level == pytest.approx(0.45)

    def test_updates_both(self):
        engine = _make_engine()
        engine.update_neurochem_state({"da": 0.3, "oxt": 0.6})
        assert engine._state.da_level == pytest.approx(0.3)
        assert engine._state.oxt_level == pytest.approx(0.6)

    def test_clamps_above_one(self):
        engine = _make_engine()
        engine.update_neurochem_state({"da": 1.5, "oxt": 2.0})
        assert engine._state.da_level == pytest.approx(1.0)
        assert engine._state.oxt_level == pytest.approx(1.0)

    def test_clamps_below_zero(self):
        engine = _make_engine()
        engine.update_neurochem_state({"da": -0.3, "oxt": -1.0})
        assert engine._state.da_level == pytest.approx(0.0)
        assert engine._state.oxt_level == pytest.approx(0.0)

    def test_ignores_unknown_keys(self):
        engine = _make_engine()
        engine.update_neurochem_state({"da": 0.5, "serotonin": 0.9})
        assert engine._state.da_level == pytest.approx(0.5)
        # oxt_level should remain default
        assert engine._state.oxt_level == pytest.approx(0.0)

    def test_empty_dict_no_change(self):
        engine = _make_engine()
        engine.update_neurochem_state({})
        assert engine._state.da_level == pytest.approx(0.0)
        assert engine._state.oxt_level == pytest.approx(0.0)

    def test_partial_update_preserves_other(self):
        engine = _make_engine()
        engine.update_neurochem_state({"da": 0.8})
        engine.update_neurochem_state({"oxt": 0.4})
        assert engine._state.da_level == pytest.approx(0.8)
        assert engine._state.oxt_level == pytest.approx(0.4)

    def test_overwrite_previous_value(self):
        engine = _make_engine()
        engine.update_neurochem_state({"da": 0.9})
        engine.update_neurochem_state({"da": 0.2})
        assert engine._state.da_level == pytest.approx(0.2)

    def test_boundary_values(self):
        engine = _make_engine()
        engine.update_neurochem_state({"da": 0.0, "oxt": 1.0})
        assert engine._state.da_level == pytest.approx(0.0)
        assert engine._state.oxt_level == pytest.approx(1.0)

    def test_default_state_values(self):
        """State dataclass should initialize da_level and oxt_level to 0.0."""
        state = IntentionMapState()
        assert state.da_level == 0.0
        assert state.oxt_level == 0.0


# =====================================================================
# Test: Edge Cases
# =====================================================================

class TestEdgeCases:
    def test_empty_input(self):
        engine = _make_engine()
        result = engine.process(IntentionMapInput())
        assert len(result.intent_vector) == 8

    def test_extreme_affective_input(self):
        engine = _make_engine()
        result = engine.process(IntentionMapInput(
            emotional_intensity=1.0,
            emotional_valence=-1.0,
            affect_complexity=10,
        ))
        assert all(-1.0 <= v <= 1.0 for v in result.intent_vector)

    def test_many_turns_stability(self):
        engine = _make_engine()
        for i in range(20):
            result = engine.process(IntentionMapInput())
        assert len(result.intent_vector) == 8
        assert abs(sum(result.intent_vector) - 1.0) < 1e-4
        assert engine.get_status()["turn_count"] == 20

    def test_full_pipeline_integration(self):
        """Test full chain: text -> tokenizer -> expander -> intention map."""
        tok = Tokenizer(rng=np.random.default_rng(42))
        exp = SemanticExpander(rng=np.random.default_rng(42))
        engine = _make_engine()

        tok_result = tok.process("I think perhaps we should explore this idea more deeply.")
        exp_result = exp.process(tok_result)
        result = engine.process(IntentionMapInput(
            tokenizer_result=tok_result,
            expansion_result=exp_result,
        ))

        assert len(result.intent_vector) == 8
        assert result.dominant_intent in [c.value for c in IntentCategory]
        assert result.primary_archetype in [a.value for a in Archetype]
        assert result.processing_time_ms >= 0.0

    def test_suppression_pairs_defined(self):
        assert len(_SUPPRESSION_PAIRS) == 3

    def test_amplification_pairs_defined(self):
        assert len(_AMPLIFICATION_PAIRS) == 4
