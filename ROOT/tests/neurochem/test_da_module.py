"""
Tests for DAModule and engine dispatch.

Phase 11: DA-specific module behavior, engine dispatch to module vs generic,
backward compatibility verification.
"""

import pytest

from zados.neurochem.neurotransmitters.dopamine import DAModule
from zados.neurochem.neurotransmitters.base import (
    OscillationCouplingRule,
    ReleaseDriveSpec,
)
from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.neurotransmitters.configs import register_neurotransmitter
from zados.neurochem.state import OscillationState


# =====================================================================
# DAModule Unit Tests
# =====================================================================

class TestDAModuleProperties:
    """Test DAModule property specifications."""

    def test_name(self):
        module = DAModule()
        assert module.name == "DA"

    def test_release_spec_keys(self):
        module = DAModule()
        spec = module.release_spec
        assert "novelty" in spec.signal_keys
        assert "rpe" in spec.signal_keys
        assert "effort" in spec.signal_keys
        assert "emotion_drive" in spec.signal_keys

    def test_release_spec_weights_match_keys(self):
        module = DAModule()
        spec = module.release_spec
        assert len(spec.signal_keys) == len(spec.weights)

    def test_release_spec_weights_sum_reasonable(self):
        module = DAModule()
        spec = module.release_spec
        assert 0.9 <= sum(spec.weights) <= 1.1

    def test_oscillation_rules_exist(self):
        module = DAModule()
        rules = module.oscillation_rules
        assert len(rules) >= 3

    def test_gamma_release_rule(self):
        module = DAModule()
        release_rules = [r for r in module.oscillation_rules if r.target == "release"]
        assert len(release_rules) >= 1
        gamma_rules = [r for r in release_rules if r.band == "gamma"]
        assert len(gamma_rules) == 1

    def test_alpha_noise_suppression_rules(self):
        module = DAModule()
        noise_rules = [r for r in module.oscillation_rules
                       if r.target in ("sigma_tonic", "sigma_phasic")]
        assert len(noise_rules) >= 2
        for rule in noise_rules:
            assert rule.band == "alpha"
            assert rule.coefficient < 0  # suppression

    def test_theta_kd_rule(self):
        module = DAModule()
        kd_rules = [r for r in module.oscillation_rules if r.target == "K_d"]
        assert len(kd_rules) == 1
        assert kd_rules[0].band == "theta"


class TestDAModuleReleaseDrive:
    """Test DAModule.compute_release_drive behavior."""

    def test_novelty_only(self):
        module = DAModule()
        drive = module.compute_release_drive({"novelty": 1.0})
        assert drive > 0.0

    def test_rpe_only(self):
        module = DAModule()
        drive = module.compute_release_drive({"rpe": 1.0})
        assert drive > 0.0

    def test_combined_signals(self):
        module = DAModule()
        drive = module.compute_release_drive({
            "novelty": 0.8, "rpe": 0.5, "effort": 0.3,
        })
        assert drive > 0.0

    def test_emotion_drive_contributes(self):
        module = DAModule()
        without = module.compute_release_drive({"novelty": 0.5})
        with_emotion = module.compute_release_drive({
            "novelty": 0.5, "emotion_drive": 0.5,
        })
        assert with_emotion > without

    def test_negative_rpe_reduces_drive(self):
        module = DAModule()
        positive = module.compute_release_drive({"rpe": 0.5})
        negative = module.compute_release_drive({"rpe": -0.5})
        assert positive > negative

    def test_all_negative_clamps_to_zero(self):
        module = DAModule()
        drive = module.compute_release_drive({
            "novelty": -1.0, "rpe": -1.0, "effort": -1.0,
            "emotion_drive": -1.0,
        })
        assert drive == 0.0

    def test_empty_signals_zero(self):
        module = DAModule()
        assert module.compute_release_drive({}) == 0.0


class TestDAModuleOscillationCoupling:
    """Test DAModule.apply_oscillation_coupling behavior."""

    def test_gamma_boosts_release(self):
        module = DAModule()
        params = {"release": 1.0, "sigma_tonic": 0.05, "sigma_phasic": 0.1, "K_d": 0.3}
        result = module.apply_oscillation_coupling(params, {"gamma": 0.8})
        assert result["release"] > 1.0

    def test_alpha_suppresses_noise(self):
        module = DAModule()
        params = {"release": 1.0, "sigma_tonic": 0.05, "sigma_phasic": 0.1, "K_d": 0.3}
        result = module.apply_oscillation_coupling(params, {"alpha": 0.7})
        assert result["sigma_tonic"] < 0.05
        assert result["sigma_phasic"] < 0.1

    def test_theta_modulates_kd(self):
        module = DAModule()
        params = {"release": 1.0, "sigma_tonic": 0.05, "sigma_phasic": 0.1, "K_d": 0.3}
        result = module.apply_oscillation_coupling(params, {"theta": 0.6})
        # K_d: 0.3 * (1 + (-0.3) * 0.6) = 0.3 * 0.82 = 0.246
        assert result["K_d"] < 0.3

    def test_zero_oscillations_no_change(self):
        module = DAModule()
        params = {"release": 1.0, "sigma_tonic": 0.05, "sigma_phasic": 0.1, "K_d": 0.3}
        result = module.apply_oscillation_coupling(params, {
            "gamma": 0.0, "alpha": 0.0, "theta": 0.0,
        })
        assert abs(result["release"] - 1.0) < 1e-9
        assert abs(result["sigma_tonic"] - 0.05) < 1e-9
        assert abs(result["K_d"] - 0.3) < 1e-9


class TestDAModulePrimaryBand:
    """Test DAModule primary release band accessors."""

    def test_primary_release_band_is_gamma(self):
        module = DAModule()
        assert module.get_primary_release_band() == "gamma"

    def test_primary_release_coefficient_positive(self):
        module = DAModule()
        assert module.get_primary_release_coefficient() > 0.0


# =====================================================================
# Engine Dispatch Tests
# =====================================================================

class TestEngineDispatch:
    """Test that engine dispatches to module when registered."""

    def test_engine_has_nt_modules_dict(self):
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        assert hasattr(engine, "_nt_modules")
        assert isinstance(engine._nt_modules, dict)

    def test_register_nt_module(self):
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        module = DAModule()
        engine.register_nt_module(module)
        assert "DA" in engine._nt_modules
        assert engine._nt_modules["DA"] is module

    def test_dispatch_to_module(self):
        """Engine uses module path when module is registered."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(engine, "DA")
        engine.register_nt_module(DAModule())

        osc = OscillationState(gamma=0.5, theta=0.3, alpha=0.2)
        engine.set_oscillation_state(osc)

        # Step with DA signals
        engine.step({"DA": {"novelty": 0.8, "rpe": 0.3}})

        state = engine.registry.get_neurotransmitter("DA")
        # Should have updated (concentration > 0)
        assert state.C > 0.0

    def test_dispatch_generic_fallback(self):
        """Engine uses generic path when no module registered."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(engine, "DA")
        # NOT registering a module

        engine.step({"DA": {"novelty": 0.8, "rpe": 0.3}})

        state = engine.registry.get_neurotransmitter("DA")
        assert state.C > 0.0

    def test_module_responds_to_emotion_drive(self):
        """DA module processes emotion_drive signal."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(engine, "DA")
        engine.register_nt_module(DAModule())

        # Step with emotion_drive only
        engine.step({"DA": {"emotion_drive": 0.8}})

        state = engine.registry.get_neurotransmitter("DA")
        assert state.C > 0.0

    def test_mixed_module_and_generic(self):
        """Only DA has a module; NE uses generic path."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(engine, "DA")
        register_neurotransmitter(engine, "NE")
        engine.register_nt_module(DAModule())

        # Both get stepped
        engine.step({
            "DA": {"novelty": 0.5},
            "NE": {"novelty": 0.3},
        })

        da_state = engine.registry.get_neurotransmitter("DA")
        ne_state = engine.registry.get_neurotransmitter("NE")
        assert da_state.C > 0.0
        assert ne_state.C > 0.0


class TestEngineDispatchBoundedness:
    """Ensure module path produces bounded concentrations."""

    def test_concentration_bounded_01(self):
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(engine, "DA")
        engine.register_nt_module(DAModule())

        osc = OscillationState(gamma=0.8, theta=0.5, alpha=0.3)
        engine.set_oscillation_state(osc)

        for _ in range(100):
            engine.step({"DA": {"novelty": 0.9, "rpe": 0.8, "effort": 0.5}})

        state = engine.registry.get_neurotransmitter("DA")
        # Individual components bounded [0, 1]
        assert 0.0 <= state.C_tonic <= 1.0
        assert 0.0 <= state.C_phasic <= 1.0
        assert 0.0 <= state.F <= 1.0

    def test_no_signals_stable(self):
        """With no input, DA should drift toward baseline."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(engine, "DA")
        engine.register_nt_module(DAModule())

        for _ in range(200):
            engine.step()

        state = engine.registry.get_neurotransmitter("DA")
        assert 0.0 <= state.C_tonic <= 1.0
        assert state.C_phasic >= 0.0  # Should decay toward 0

    def test_module_with_all_oscillations(self):
        """Full oscillation state doesn't cause errors."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(engine, "DA")
        engine.register_nt_module(DAModule())

        osc = OscillationState(
            delta=0.3, theta=0.5, alpha=0.4, beta=0.6, gamma=0.7,
        )
        engine.set_oscillation_state(osc)

        for _ in range(50):
            engine.step({"DA": {
                "novelty": 0.5, "rpe": 0.2, "effort": 0.3,
                "emotion_drive": 0.4,
            }})

        state = engine.registry.get_neurotransmitter("DA")
        assert 0.0 <= state.C_tonic <= 1.0
        assert 0.0 <= state.C_phasic <= 1.0


class TestBackwardCompatibility:
    """Ensure existing usage patterns still work after refactor."""

    def test_engine_without_modules_works(self):
        """Engine with no modules registered behaves as before."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(engine, "DA")

        engine.step({"DA": {"novelty": 0.5, "rpe": 0.3, "effort": 0.1}})

        state = engine.registry.get_neurotransmitter("DA")
        assert state.C > 0.0

    def test_readout_works_with_module(self):
        """get_neurosymbolic_readout works when module is registered."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(engine, "DA")
        engine.register_nt_module(DAModule())

        osc = OscillationState(theta=0.3, gamma=0.5)
        engine.set_oscillation_state(osc)

        engine.step({"DA": {"novelty": 0.5}})
        readout = engine.get_neurosymbolic_readout()
        assert "motivation" in readout

    def test_feedback_works_with_module(self):
        """apply_feedback works when module is registered."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_neurotransmitter(engine, "DA")
        engine.register_nt_module(DAModule())

        # Feedback doesn't target DA, but should not crash
        engine.apply_feedback({
            "neurotransmitters": {
                "DA": {"C_baseline_delta": 0.02},
            },
            "receptors": {},
        })

        config = engine.registry.get_config("DA")
        assert abs(config["C_baseline"] - 0.52) < 1e-9
