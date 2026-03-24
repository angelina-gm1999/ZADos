"""
Tests for emotion-driven receptor plasticity.

Phase 21: emotion_plasticity_rules defined in ReceptorSpec are now applied
via compute_plasticity_deltas + apply_plasticity_delta pure functions,
orchestrated by engine.apply_emotion_event().
"""

import pytest

from zados.neurochem.receptors.plasticity import (
    compute_plasticity_deltas,
    apply_plasticity_delta,
)
from zados.neurochem.receptors.dopamine_receptors import DopamineReceptors
from zados.neurochem.receptors.serotonin_receptors import SerotoninReceptors
from zados.neurochem.state.receptor_state import ReceptorState
from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.state import NeurotransmitterState


# ---------------------------------------------------------------------------
# compute_plasticity_deltas
# ---------------------------------------------------------------------------

class TestComputePlasticityDeltas:
    """Tests for the pure compute_plasticity_deltas function."""

    def test_joy_returns_da_deltas(self):
        """'joy' should produce deltas for DA_D1 and DA_D2."""
        modules = {"DA": DopamineReceptors()}
        deltas = compute_plasticity_deltas("joy", modules)
        assert "DA_D1" in deltas
        assert deltas["DA_D1"]["sigma_delta"] == pytest.approx(0.1)
        assert deltas["DA_D1"]["rho_delta"] == pytest.approx(0.05)
        assert "DA_D2" in deltas
        assert deltas["DA_D2"]["sigma_delta"] == pytest.approx(0.05)

    def test_unknown_emotion_returns_empty(self):
        """Unknown emotion should return empty dict."""
        modules = {"DA": DopamineReceptors()}
        deltas = compute_plasticity_deltas("nonexistent_emotion", modules)
        assert deltas == {}

    def test_multiple_modules(self):
        """Deltas from multiple modules are collected."""
        modules = {
            "DA": DopamineReceptors(),
            "5HT": SerotoninReceptors(),
        }
        # "fear" affects DA_D1 (sigma_delta=-0.05), DA_D2 (sigma_delta=-0.1)
        # and potentially 5HT receptors
        deltas = compute_plasticity_deltas("fear", modules)
        assert "DA_D1" in deltas
        assert deltas["DA_D1"]["sigma_delta"] == pytest.approx(-0.05)

    def test_emotion_with_no_matching_receptors(self):
        """Emotion that exists in spec but no modules registered for it."""
        modules = {}
        deltas = compute_plasticity_deltas("joy", modules)
        assert deltas == {}


# ---------------------------------------------------------------------------
# apply_plasticity_delta
# ---------------------------------------------------------------------------

class TestApplyPlasticityDelta:
    """Tests for the pure apply_plasticity_delta function."""

    def test_sigma_increases(self):
        """sigma_delta > 0 should increase sigma."""
        state = ReceptorState(receptor_id="DA_D1", sigma=0.5)
        new_state = apply_plasticity_delta(state, {"sigma_delta": 0.1})
        assert new_state.sigma == pytest.approx(0.6)
        # Original not mutated
        assert state.sigma == pytest.approx(0.5)

    def test_rho_increases(self):
        """rho_delta > 0 should increase rho."""
        state = ReceptorState(receptor_id="DA_D1", rho=0.7)
        new_state = apply_plasticity_delta(state, {"rho_delta": 0.05})
        assert new_state.rho == pytest.approx(0.75)

    def test_sigma_and_rho_combined(self):
        """Both sigma and rho deltas applied together."""
        state = ReceptorState(receptor_id="DA_D1", sigma=0.5, rho=0.5)
        new_state = apply_plasticity_delta(
            state, {"sigma_delta": 0.1, "rho_delta": 0.2}
        )
        assert new_state.sigma == pytest.approx(0.6)
        assert new_state.rho == pytest.approx(0.7)

    def test_clamped_upper(self):
        """Deltas should be clamped to [0, 1]."""
        state = ReceptorState(receptor_id="DA_D1", sigma=0.95)
        new_state = apply_plasticity_delta(state, {"sigma_delta": 0.2})
        assert new_state.sigma == pytest.approx(1.0)

    def test_clamped_lower(self):
        """Negative deltas clamped at 0."""
        state = ReceptorState(receptor_id="DA_D1", sigma=0.05)
        new_state = apply_plasticity_delta(state, {"sigma_delta": -0.2})
        assert new_state.sigma == pytest.approx(0.0)

    def test_intensity_scaling(self):
        """intensity=0.5 should halve the deltas."""
        state = ReceptorState(receptor_id="DA_D1", sigma=0.5)
        new_state = apply_plasticity_delta(
            state, {"sigma_delta": 0.2}, intensity=0.5
        )
        assert new_state.sigma == pytest.approx(0.6)  # 0.5 + 0.2*0.5

    def test_empty_deltas(self):
        """Empty deltas dict should not change state."""
        state = ReceptorState(receptor_id="DA_D1", sigma=0.5, rho=0.7)
        new_state = apply_plasticity_delta(state, {})
        assert new_state.sigma == pytest.approx(0.5)
        assert new_state.rho == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Engine integration: apply_emotion_event
# ---------------------------------------------------------------------------

class TestEngineApplyEmotionEvent:
    """Tests for engine.apply_emotion_event()."""

    @pytest.fixture
    def engine_with_da_receptors(self):
        """Engine with DA NT + DA_D1 receptor + DopamineReceptors module."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        da_state = NeurotransmitterState(C_tonic=0.5, C_phasic=0.0, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da_state)
        receptor = ReceptorState(receptor_id="DA_D1", sigma=0.5, rho=0.5)
        engine.add_receptor("DA_D1", initial_state=receptor, config={"K_d": 0.5})
        engine.register_receptor_module(DopamineReceptors())
        return engine

    def test_joy_changes_da_d1_sigma(self, engine_with_da_receptors):
        """'joy' should increase DA_D1 sigma."""
        engine = engine_with_da_receptors
        engine.apply_emotion_event("joy")
        state = engine.registry.get_receptor("DA_D1")
        # DA_D1 joy: sigma_delta=0.1, rho_delta=0.05
        assert state.sigma == pytest.approx(0.6)
        assert state.rho == pytest.approx(0.55)

    def test_joy_with_intensity(self, engine_with_da_receptors):
        """intensity=0.5 should halve the plasticity effect."""
        engine = engine_with_da_receptors
        engine.apply_emotion_event("joy", intensity=0.5)
        state = engine.registry.get_receptor("DA_D1")
        assert state.sigma == pytest.approx(0.55)  # 0.5 + 0.1*0.5
        assert state.rho == pytest.approx(0.525)    # 0.5 + 0.05*0.5

    def test_no_modules_no_crash(self):
        """No receptor modules registered should not crash."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        engine.apply_emotion_event("joy")  # Should not raise

    def test_unknown_emotion_no_change(self, engine_with_da_receptors):
        """Unknown emotion should leave state unchanged."""
        engine = engine_with_da_receptors
        sigma_before = engine.registry.get_receptor("DA_D1").sigma
        engine.apply_emotion_event("nonexistent_emotion")
        sigma_after = engine.registry.get_receptor("DA_D1").sigma
        assert sigma_after == pytest.approx(sigma_before)

    def test_unregistered_receptor_skipped(self):
        """Receptor in module spec but not in registry should be skipped."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        engine.register_receptor_module(DopamineReceptors())
        # No DA_D1 in registry, but module defines it
        engine.apply_emotion_event("joy")  # Should not raise
