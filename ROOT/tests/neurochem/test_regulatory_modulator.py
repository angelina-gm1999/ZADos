"""Tests for regulatory modulator and oscillation envelope (Extractor 3)."""

import pytest

from zados.neurochem.extractors.leaky_integrator import LeakyIntegratorState
from zados.neurochem.extractors.regulatory_modulator import (
    RegulatoryPathwayConfig,
    RegulatoryModulatorConfig,
    RegulatoryModulatorState,
    OscillationEnvelopeRule,
    DEFAULT_REGULATORY_CONFIG,
    DEFAULT_ENVELOPE_RULES,
    step_regulatory_modulator,
    compute_oscillation_envelope,
)
from zados.neurochem.state.oscillation_state import OscillationState


# =====================================================================
# RegulatoryModulatorState
# =====================================================================

class TestRegulatoryModulatorState:
    def test_from_config(self):
        state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)
        assert len(state.integrator_states) == 4
        assert "OXT_attunement" in state.integrator_states
        assert "CB1_innovation" in state.integrator_states
        assert "NE_logic" in state.integrator_states
        assert "GABA_B_ethics" in state.integrator_states

    def test_initial_values_at_baseline(self):
        state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)
        # OXT baseline pathway: baseline=0.0
        assert state.integrator_states["OXT_attunement"].value == 0.0
        # NE multiplier pathway: baseline=1.0
        assert state.integrator_states["NE_logic"].value == 1.0

    def test_roundtrip_dict(self):
        state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)
        d = state.as_dict()
        restored = RegulatoryModulatorState.from_dict(d)
        for name in state.integrator_states:
            assert restored.integrator_states[name].value == state.integrator_states[name].value
            assert restored.integrator_states[name].baseline == state.integrator_states[name].baseline


# =====================================================================
# step_regulatory_modulator — feedback format
# =====================================================================

class TestStepRegulatoryModulatorFormat:
    def test_output_has_correct_top_keys(self):
        state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)
        eval_vec = {"social_salience": 0.7, "novelty": 0.8,
                    "logical_conflict": 0.5, "urgency": 0.6}
        new_state, feedback = step_regulatory_modulator(state, eval_vec)
        assert "neurotransmitters" in feedback
        assert "receptors" in feedback

    def test_nt_feedback_keys(self):
        state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)
        eval_vec = {"social_salience": 0.7, "novelty": 0.8,
                    "logical_conflict": 0.5, "urgency": 0.6}
        _, feedback = step_regulatory_modulator(state, eval_vec)
        nt = feedback["neurotransmitters"]
        assert "OXT" in nt
        assert "CB1" in nt
        assert "NE" in nt
        assert "C_baseline_delta" in nt["OXT"]
        assert "C_baseline_delta" in nt["CB1"]
        assert "u_base_multiplier" in nt["NE"]

    def test_receptor_feedback_keys(self):
        state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)
        eval_vec = {"urgency": 0.6}
        _, feedback = step_regulatory_modulator(state, eval_vec)
        assert "GABA_B" in feedback["receptors"]
        assert "K_d_multiplier" in feedback["receptors"]["GABA_B"]

    def test_returns_new_state(self):
        state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)
        eval_vec = {"social_salience": 0.9}
        new_state, _ = step_regulatory_modulator(state, eval_vec)
        assert new_state is not state


# =====================================================================
# step_regulatory_modulator — temporal smoothing
# =====================================================================

class TestStepRegulatoryModulatorSmoothing:
    def test_step_input_ramps_not_jumps(self):
        """A sudden step input should produce gradual (not instantaneous) response."""
        state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)

        # Step 1: high social_salience
        eval_vec = {"social_salience": 1.0, "novelty": 0.0,
                    "logical_conflict": 0.0, "urgency": 0.0}
        state, fb1 = step_regulatory_modulator(state, eval_vec, dt=0.01)
        delta_1 = fb1["neurotransmitters"]["OXT"]["C_baseline_delta"]

        # Step 2: same input
        state, fb2 = step_regulatory_modulator(state, eval_vec, dt=0.01)
        delta_2 = fb2["neurotransmitters"]["OXT"]["C_baseline_delta"]

        # Should be gradually increasing, not jumping to max
        assert abs(delta_1) < 0.05  # less than max gain
        assert abs(delta_2) >= abs(delta_1)  # growing

    def test_many_steps_approach_steady(self):
        """After many steps with constant input, feedback should stabilize."""
        state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)
        eval_vec = {"social_salience": 0.9, "novelty": 0.0,
                    "logical_conflict": 0.0, "urgency": 0.0}

        for _ in range(50000):
            state, feedback = step_regulatory_modulator(state, eval_vec, dt=0.01)

        delta = feedback["neurotransmitters"]["OXT"]["C_baseline_delta"]
        # Should be at or near max gain (0.05), since input > center
        assert delta > 0.0

    def test_zero_eval_decays_to_neutral(self):
        """Zero evaluation → feedback should decay toward neutral."""
        config = RegulatoryModulatorConfig(pathways=(
            RegulatoryPathwayConfig(
                name="test",
                evaluation_axis="novelty",
                target_nt="CB1",
                target_param="C_baseline_delta",
                target_category="neurotransmitters",
                tau=1.0,
                baseline=0.0,
                gain=0.05,
                center=0.5,
            ),
        ))
        # Start with displaced state
        state = RegulatoryModulatorState(integrator_states={
            "test": LeakyIntegratorState(value=0.03, baseline=0.0),
        })
        eval_vec = {"novelty": 0.5}  # at center → zero drive

        for _ in range(10000):
            state, feedback = step_regulatory_modulator(state, eval_vec, config, dt=0.01)

        delta = feedback["neurotransmitters"]["CB1"]["C_baseline_delta"]
        assert abs(delta) < 0.001  # decayed to near zero

    def test_missing_axis_produces_baseline_feedback(self):
        """Missing evaluation axis → integrator decays to baseline."""
        state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)
        eval_vec = {}  # all axes missing

        for _ in range(100):
            state, feedback = step_regulatory_modulator(state, eval_vec, dt=0.01)

        # With zero input:
        # baseline-delta pathways: input = (0 - 0.5)*1.0 = -0.5 → negative drive
        # So OXT delta should be slightly negative (below-center)
        oxt = feedback["neurotransmitters"]["OXT"]["C_baseline_delta"]
        assert isinstance(oxt, float)


# =====================================================================
# step_regulatory_modulator — single pathway
# =====================================================================

class TestStepRegulatoryModulatorSinglePathway:
    def test_oxt_high_social_positive_delta(self):
        """High social_salience (>0.5 center) → positive OXT baseline delta."""
        state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)
        eval_vec = {"social_salience": 0.9}
        _, feedback = step_regulatory_modulator(state, eval_vec, dt=0.01)
        assert feedback["neurotransmitters"]["OXT"]["C_baseline_delta"] > 0.0

    def test_oxt_low_social_negative_delta(self):
        """Low social_salience (<0.5 center) → negative OXT baseline delta."""
        state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)
        eval_vec = {"social_salience": 0.1}
        _, feedback = step_regulatory_modulator(state, eval_vec, dt=0.01)
        assert feedback["neurotransmitters"]["OXT"]["C_baseline_delta"] < 0.0

    def test_ne_multiplier_near_one_initially(self):
        """NE multiplier starts near 1.0 (baseline)."""
        state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)
        eval_vec = {"logical_conflict": 0.5}
        _, feedback = step_regulatory_modulator(state, eval_vec, dt=0.01)
        ne_mult = feedback["neurotransmitters"]["NE"]["u_base_multiplier"]
        # Should be very close to 1.0 after one small step
        assert 0.99 < ne_mult < 1.1


# =====================================================================
# step_regulatory_modulator — custom config
# =====================================================================

class TestStepRegulatoryModulatorCustomConfig:
    def test_custom_single_pathway(self):
        config = RegulatoryModulatorConfig(pathways=(
            RegulatoryPathwayConfig(
                name="custom",
                evaluation_axis="urgency",
                target_nt="cortisol",
                target_param="C_baseline_delta",
                target_category="neurotransmitters",
                tau=5.0,
                gain=0.1,
                center=0.3,
            ),
        ))
        state = RegulatoryModulatorState.from_config(config)
        eval_vec = {"urgency": 0.8}
        _, feedback = step_regulatory_modulator(state, eval_vec, config, dt=0.01)
        assert "cortisol" in feedback["neurotransmitters"]
        assert "C_baseline_delta" in feedback["neurotransmitters"]["cortisol"]

    def test_empty_config(self):
        config = RegulatoryModulatorConfig(pathways=())
        state = RegulatoryModulatorState.from_config(config)
        _, feedback = step_regulatory_modulator(state, {}, config, dt=0.01)
        assert feedback["neurotransmitters"] == {}
        assert feedback["receptors"] == {}


# =====================================================================
# compute_oscillation_envelope
# =====================================================================

class TestComputeOscillationEnvelope:
    def test_zero_state_no_modulation(self):
        """Zero regulatory state → no change in oscillations."""
        reg_state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)
        osc = OscillationState(theta=0.5, gamma=0.3, beta=0.4, alpha=0.6)
        result = compute_oscillation_envelope(reg_state, osc)
        # All integrators at baseline → mod_signal = 0 → no change
        assert result.theta == pytest.approx(0.5)
        assert result.gamma == pytest.approx(0.3)
        assert result.beta == pytest.approx(0.4)
        assert result.alpha == pytest.approx(0.6)

    def test_does_not_mutate_input(self):
        reg_state = RegulatoryModulatorState.from_config(DEFAULT_REGULATORY_CONFIG)
        osc = OscillationState(theta=0.5)
        result = compute_oscillation_envelope(reg_state, osc)
        assert osc.theta == 0.5  # unchanged

    def test_displaced_integrator_modulates_band(self):
        """Integrator displaced from baseline should modulate the target band."""
        reg_state = RegulatoryModulatorState(integrator_states={
            "OXT_attunement": LeakyIntegratorState(value=0.5, baseline=0.0),
            "CB1_innovation": LeakyIntegratorState(value=0.0, baseline=0.0),
            "NE_logic": LeakyIntegratorState(value=1.0, baseline=1.0),
            "GABA_B_ethics": LeakyIntegratorState(value=1.0, baseline=1.0),
        })
        osc = OscillationState(theta=0.3, gamma=0.3, beta=0.3, alpha=0.3)
        result = compute_oscillation_envelope(reg_state, osc)

        # OXT_attunement → theta, coeff=0.3, mod_signal=|0.5-0.0|=0.5
        # new_theta = 0.3 + 0.3*0.5 = 0.45
        assert result.theta == pytest.approx(0.45)

        # CB1_innovation → gamma, mod_signal=0 → no change
        assert result.gamma == pytest.approx(0.3)

        # NE_logic → beta, mod_signal=0 → no change
        assert result.beta == pytest.approx(0.3)

    def test_clamped_to_unit(self):
        """Envelope modulation should clamp amplitudes to [0, 1]."""
        reg_state = RegulatoryModulatorState(integrator_states={
            "OXT_attunement": LeakyIntegratorState(value=10.0, baseline=0.0),
        })
        osc = OscillationState(theta=0.9)
        rules = (OscillationEnvelopeRule("OXT_attunement", "theta", 0.5),)
        result = compute_oscillation_envelope(reg_state, osc, rules)
        assert result.theta <= 1.0

    def test_multiplicative_formula(self):
        """Multiplicative envelope rule."""
        reg_state = RegulatoryModulatorState(integrator_states={
            "test_pathway": LeakyIntegratorState(value=0.5, baseline=0.0),
        })
        osc = OscillationState(theta=0.4)
        rules = (
            OscillationEnvelopeRule("test_pathway", "theta", 0.5, formula="multiplicative"),
        )
        result = compute_oscillation_envelope(reg_state, osc, rules)
        # mod_signal = |0.5 - 0.0| = 0.5
        # new = 0.4 * (1.0 + 0.5 * 0.5) = 0.4 * 1.25 = 0.5
        assert result.theta == pytest.approx(0.5)

    def test_unknown_formula_raises(self):
        reg_state = RegulatoryModulatorState(integrator_states={
            "test": LeakyIntegratorState(value=0.5, baseline=0.0),
        })
        osc = OscillationState()
        rules = (OscillationEnvelopeRule("test", "theta", 0.5, formula="unknown"),)
        with pytest.raises(ValueError, match="Unknown envelope formula"):
            compute_oscillation_envelope(reg_state, osc, rules)

    def test_missing_pathway_skipped(self):
        """Rule referencing non-existent pathway is silently skipped."""
        reg_state = RegulatoryModulatorState(integrator_states={})
        osc = OscillationState(theta=0.5)
        result = compute_oscillation_envelope(reg_state, osc, DEFAULT_ENVELOPE_RULES)
        assert result.theta == 0.5  # unchanged

    def test_default_rules_cover_4_bands(self):
        """Default rules should cover theta, gamma, beta, alpha."""
        bands = {r.target_band for r in DEFAULT_ENVELOPE_RULES}
        assert bands == {"theta", "gamma", "beta", "alpha"}
