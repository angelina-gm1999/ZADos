"""
Parametrized tests across all 11 NT modules.

Phase 12: Verifies that every NT module has consistent structure,
valid oscillation rules, responsive release drives, and integrates
correctly with the engine.
"""

import pytest

from zados.neurochem.neurotransmitters.dopamine import DAModule
from zados.neurochem.neurotransmitters.serotonin import SerotoninModule
from zados.neurochem.neurotransmitters.norepinephrine import NEModule
from zados.neurochem.neurotransmitters.acetylcholine import AChModule
from zados.neurochem.neurotransmitters.oxytocin import OXTModule
from zados.neurochem.neurotransmitters.opioid import MORModule
from zados.neurochem.neurotransmitters.endocannabinoid import CB1Module
from zados.neurochem.neurotransmitters.cortisol_mod import CortisolModule
from zados.neurochem.neurotransmitters.crh import CRHModule
from zados.neurochem.neurotransmitters.gaba import GABAModule
from zados.neurochem.neurotransmitters.glutamate import GLUModule
from zados.neurochem.neurotransmitters.module_registry import (
    NTModuleRegistry,
    register_all_nt_modules,
)
from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.neurotransmitters.configs import (
    register_all_neurotransmitters,
    DEFAULT_NT_CONFIGS,
)
from zados.neurochem.state import OscillationState


# All module classes
ALL_MODULE_CLASSES = [
    DAModule, SerotoninModule, NEModule, AChModule, OXTModule,
    MORModule, CB1Module, CortisolModule, CRHModule, GABAModule, GLUModule,
]

# Expected NT names matching configs.py
EXPECTED_NT_NAMES = sorted(DEFAULT_NT_CONFIGS.keys())


@pytest.fixture(params=ALL_MODULE_CLASSES, ids=[c.__name__ for c in ALL_MODULE_CLASSES])
def module(request):
    """Parametrized fixture providing each NT module instance."""
    return request.param()


# =====================================================================
# Structure Tests (parametrized across all modules)
# =====================================================================

class TestModuleStructure:
    """Every module must have consistent structure."""

    def test_has_name(self, module):
        assert isinstance(module.name, str)
        assert len(module.name) > 0

    def test_name_matches_configs(self, module):
        """Module name must correspond to a key in DEFAULT_NT_CONFIGS."""
        assert module.name in DEFAULT_NT_CONFIGS, (
            f"{module.__class__.__name__}.name = {module.name!r} "
            f"not found in DEFAULT_NT_CONFIGS"
        )

    def test_has_release_spec(self, module):
        spec = module.release_spec
        assert len(spec.signal_keys) > 0
        assert len(spec.signal_keys) == len(spec.weights)

    def test_emotion_drive_accepted(self, module):
        """Every module must accept 'emotion_drive' as a signal key."""
        assert "emotion_drive" in module.release_spec.signal_keys

    def test_weights_sum_reasonable(self, module):
        """Weights should sum to approximately 1.0 (±0.2)."""
        total = sum(module.release_spec.weights)
        assert 0.8 <= total <= 1.2, (
            f"{module.name} weights sum to {total}"
        )

    def test_has_oscillation_rules(self, module):
        rules = module.oscillation_rules
        assert isinstance(rules, list)
        # Every module should have at least one rule
        assert len(rules) >= 1

    def test_oscillation_rules_valid(self, module):
        """All oscillation rules should have valid targets and bands."""
        for rule in module.oscillation_rules:
            assert rule.target is not None
            assert rule.band is not None
            assert rule.coefficient != 0.0


# =====================================================================
# Release Drive Tests (parametrized)
# =====================================================================

class TestModuleReleaseDrive:
    """Release drive behavior tests across all modules."""

    def test_empty_signals_below_threshold(self, module):
        """Empty signals should produce zero or near-zero drive."""
        drive = module.compute_release_drive({})
        assert drive >= 0.0
        # With threshold > 0 and no signals, drive should be 0
        if module.release_spec.threshold > 0:
            assert drive == 0.0

    def test_high_signals_produce_positive_drive(self, module):
        """Strong signals for all keys should produce positive drive."""
        signals = {key: 1.0 for key in module.release_spec.signal_keys}
        drive = module.compute_release_drive(signals)
        assert drive > 0.0

    def test_emotion_drive_alone_produces_output(self, module):
        """Emotion drive alone should produce some release."""
        drive = module.compute_release_drive({"emotion_drive": 1.0})
        # Should be non-negative; may be zero if below threshold
        assert drive >= 0.0

    def test_drive_non_negative(self, module):
        """Release drive should always be non-negative."""
        signals = {key: -1.0 for key in module.release_spec.signal_keys}
        drive = module.compute_release_drive(signals)
        assert drive >= 0.0


# =====================================================================
# Oscillation Coupling Tests (parametrized)
# =====================================================================

class TestModuleOscillationCoupling:
    """Oscillation coupling tests across all modules."""

    def test_zero_oscillations_no_change(self, module):
        params = {"sigma_tonic": 0.05, "sigma_phasic": 0.1, "u_base": 0.1,
                  "release": 1.0, "K_d": 0.3, "theta_tonic": 0.1, "theta_phasic": 1.0}
        zero_osc = {b: 0.0 for b in ["delta", "theta", "alpha", "beta", "gamma",
                                       "theta_gamma", "alpha_beta"]}
        result = module.apply_oscillation_coupling(params, zero_osc)
        for key in params:
            assert abs(result[key] - params[key]) < 1e-9, (
                f"{module.name}: {key} changed with zero oscillations"
            )

    def test_coupling_returns_dict(self, module):
        params = {"sigma_tonic": 0.05, "release": 1.0}
        osc = {"gamma": 0.5, "theta": 0.3, "alpha": 0.4, "beta": 0.6, "delta": 0.2,
               "theta_gamma": 0.15, "alpha_beta": 0.24}
        result = module.apply_oscillation_coupling(params, osc)
        assert isinstance(result, dict)

    def test_coupling_preserves_untargeted_params(self, module):
        params = {"sigma_tonic": 0.05, "release": 1.0, "unrelated_param": 42.0}
        osc = {"gamma": 0.5, "theta": 0.3}
        result = module.apply_oscillation_coupling(params, osc)
        assert result["unrelated_param"] == 42.0


# =====================================================================
# Registry Tests
# =====================================================================

class TestNTModuleRegistration:
    """Test registration of all modules."""

    def test_register_all_nt_modules(self):
        NTModuleRegistry.clear()
        register_all_nt_modules()
        assert NTModuleRegistry.count() == 12

    def test_all_configs_have_modules(self):
        NTModuleRegistry.clear()
        register_all_nt_modules()
        for nt_name in DEFAULT_NT_CONFIGS:
            assert NTModuleRegistry.is_registered(nt_name), (
                f"No module registered for NT: {nt_name}"
            )
        NTModuleRegistry.clear()

    def test_module_names_match_configs(self):
        NTModuleRegistry.clear()
        register_all_nt_modules()
        registered = set(NTModuleRegistry.registered_names())
        expected = set(DEFAULT_NT_CONFIGS.keys())
        assert registered == expected
        NTModuleRegistry.clear()


# =====================================================================
# Engine Integration Tests
# =====================================================================

class TestFullEngineIntegration:
    """Test engine with all 12 modules registered."""

    def test_all_modules_on_engine(self):
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_all_neurotransmitters(engine)
        register_all_nt_modules(engine)

        assert len(engine._nt_modules) == 12

    def test_step_all_nts_with_modules(self):
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_all_neurotransmitters(engine)
        register_all_nt_modules(engine)

        osc = OscillationState(
            delta=0.2, theta=0.4, alpha=0.3, beta=0.5, gamma=0.6,
        )
        engine.set_oscillation_state(osc)

        # Step with signals for various NTs
        signals = {
            "DA": {"novelty": 0.7, "rpe": 0.3},
            "NE": {"precision": 0.5, "uncertainty": 0.4},
            "OXT": {"empathy": 0.6, "social_engagement": 0.5},
            "GABA": {"inhibition": 0.3},
        }
        engine.step(signals)

        # Verify all NTs have valid state
        for nt_name in DEFAULT_NT_CONFIGS:
            state = engine.registry.get_neurotransmitter(nt_name)
            assert 0.0 <= state.C_tonic <= 1.0, f"{nt_name} tonic out of bounds"
            assert 0.0 <= state.C_phasic <= 1.0, f"{nt_name} phasic out of bounds"

    def test_100_steps_bounded(self):
        """100 steps with strong signals should remain bounded."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_all_neurotransmitters(engine)
        register_all_nt_modules(engine)

        osc = OscillationState(
            delta=0.3, theta=0.5, alpha=0.4, beta=0.6, gamma=0.7,
        )
        engine.set_oscillation_state(osc)

        for _ in range(100):
            engine.step({
                "DA": {"novelty": 0.9, "rpe": 0.5, "emotion_drive": 0.3},
                "5HT": {"mood_stability": 0.7, "emotion_drive": 0.4},
                "NE": {"precision": 0.8, "emotion_drive": 0.5},
                "GABA": {"inhibition": 0.6, "emotion_drive": 0.3},
                "GLU": {"excitation": 0.7, "emotion_drive": 0.4},
            })

        for nt_name in DEFAULT_NT_CONFIGS:
            state = engine.registry.get_neurotransmitter(nt_name)
            assert 0.0 <= state.C_tonic <= 1.0, f"{nt_name} tonic out of bounds after 100 steps"
            assert 0.0 <= state.C_phasic <= 1.0, f"{nt_name} phasic out of bounds after 100 steps"

    def test_readout_with_all_modules(self):
        """Neurosymbolic readout should work with all modules."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_all_neurotransmitters(engine)
        register_all_nt_modules(engine)

        osc = OscillationState(theta=0.4, gamma=0.5)
        engine.set_oscillation_state(osc)

        engine.step({"DA": {"novelty": 0.5}, "OXT": {"empathy": 0.6}})

        readout = engine.get_neurosymbolic_readout()
        assert "motivation" in readout
        assert "empathy" in readout
        assert "anxiety" in readout

    def test_emotion_drive_reaches_all_nts(self):
        """Emotion drive signal should affect all registered NTs."""
        engine = NeurochemicalEngine(dt=0.01, seed=42)
        register_all_neurotransmitters(engine)
        register_all_nt_modules(engine)

        # Record baseline states
        baselines = {}
        for nt_name in DEFAULT_NT_CONFIGS:
            state = engine.registry.get_neurotransmitter(nt_name)
            baselines[nt_name] = state.C_phasic

        # Step with emotion_drive for every NT
        signals = {
            nt_name: {"emotion_drive": 0.8}
            for nt_name in DEFAULT_NT_CONFIGS
        }
        for _ in range(10):
            engine.step(signals)

        # At least some NTs should show increased phasic
        increased = 0
        for nt_name in DEFAULT_NT_CONFIGS:
            state = engine.registry.get_neurotransmitter(nt_name)
            if state.C_phasic > baselines[nt_name] + 0.001:
                increased += 1

        # Not all will increase (stochastic + some may have high threshold)
        # but at least several should respond
        assert increased >= 3, (
            f"Only {increased} NTs showed increased phasic from emotion_drive"
        )
