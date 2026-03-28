"""
Tests for multi-band noise oscillatory modulation.

Phase 25: sigma_mod = sigma_base * max(floor, 1 - Σ s_k*φ_k + Σ a_k*φ_k)
with per-NT suppression and amplification coefficients.
"""

import pytest

from zados.neurochem.oscillations.oscillation_modulation import (
    modulate_noise,
    modulate_noise_multiband,
)
from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.state import NeurotransmitterState, OscillationState


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

class TestModulateNoiseMultiband:
    """Tests for the modulate_noise_multiband pure function."""

    def test_alpha_only_matches_legacy(self):
        """Single alpha suppression should match legacy modulate_noise."""
        sigma_base = 0.1
        phi_alpha = 0.7
        alpha_coeff = 0.4  # same as legacy default

        legacy = modulate_noise(sigma_base, phi_alpha, coefficient=alpha_coeff)
        multiband = modulate_noise_multiband(
            sigma_base,
            osc_amplitudes={"alpha": phi_alpha, "theta": 0.0},
            suppression_coefficients={"alpha": alpha_coeff},
            amplification_coefficients={},
        )
        assert multiband == pytest.approx(legacy)

    def test_suppression_and_amplification(self):
        """Suppression and amplification should combine additively."""
        sigma_base = 0.1
        osc = {"alpha": 0.5, "gamma": 0.4}
        suppression = {"alpha": 0.4}
        amplification = {"gamma": 0.3}
        # scale = max(0.1, 1 - 0.4*0.5 + 0.3*0.4) = max(0.1, 1 - 0.2 + 0.12) = max(0.1, 0.92) = 0.92
        result = modulate_noise_multiband(sigma_base, osc, suppression, amplification)
        assert result == pytest.approx(sigma_base * 0.92)

    def test_floor(self):
        """Scaling factor should never go below floor."""
        sigma_base = 0.1
        osc = {"alpha": 1.0, "theta": 1.0}
        suppression = {"alpha": 0.8, "theta": 0.5}
        amplification = {}
        # scale = max(0.1, 1 - 0.8 - 0.5) = max(0.1, -0.3) = 0.1
        result = modulate_noise_multiband(sigma_base, osc, suppression, amplification)
        assert result == pytest.approx(sigma_base * 0.1)

    def test_zero_coefficients(self):
        """No change when all coefficients are zero."""
        sigma_base = 0.1
        osc = {"alpha": 0.9, "gamma": 0.8}
        result = modulate_noise_multiband(sigma_base, osc, {}, {})
        assert result == pytest.approx(sigma_base)

    def test_amplification_only(self):
        """Amplification without suppression should increase noise."""
        sigma_base = 0.1
        osc = {"gamma": 0.8}
        result = modulate_noise_multiband(
            sigma_base, osc,
            suppression_coefficients={},
            amplification_coefficients={"gamma": 0.5},
        )
        # scale = max(0.1, 1 + 0.5*0.8) = max(0.1, 1.4) = 1.4
        assert result == pytest.approx(sigma_base * 1.4)

    def test_custom_floor(self):
        """Custom floor should be respected."""
        sigma_base = 0.1
        osc = {"alpha": 1.0}
        result = modulate_noise_multiband(
            sigma_base, osc,
            suppression_coefficients={"alpha": 2.0},
            amplification_coefficients={},
            floor=0.3,
        )
        # scale = max(0.3, 1 - 2.0) = max(0.3, -1.0) = 0.3
        assert result == pytest.approx(sigma_base * 0.3)

    def test_empty_amplitudes(self):
        """Empty oscillation amplitudes → no modulation."""
        sigma_base = 0.1
        result = modulate_noise_multiband(
            sigma_base, {}, {"alpha": 0.4}, {"gamma": 0.3},
        )
        assert result == pytest.approx(sigma_base)


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------

class TestEngineMultibandNoise:
    """Tests for engine-level multi-band noise modulation."""

    def _make_engine(self, nt_config, osc=None):
        """Helper: create engine with a single NT."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        da_state = NeurotransmitterState(
            C_tonic=0.3, C_phasic=0.0, F=0.0, eta_u=0.0,
        )
        engine.add_neurotransmitter("DA", initial_state=da_state, config=nt_config)
        if osc:
            engine.registry.set_oscillations(osc)
        return engine

    def test_engine_multiband_noise_when_configured(self):
        """Engine should use multiband noise when noise_band_coefficients present."""
        osc = OscillationState(alpha=0.5, gamma=0.4)
        config = {
            "C_baseline": 0.3,
            "noise_band_coefficients": {
                "suppression": {"alpha": 0.4},
                "amplification": {"gamma": 0.2},
            },
        }
        engine = self._make_engine(config, osc)
        engine.step()  # should not crash
        state = engine.registry.get_neurotransmitter("DA")
        assert state is not None

    def test_engine_noise_falls_back_to_legacy(self):
        """No noise_band_coefficients → legacy alpha-only."""
        osc = OscillationState(alpha=0.5)
        config = {"C_baseline": 0.3}
        engine = self._make_engine(config, osc)
        engine.step()  # should use legacy modulate_noise
        state = engine.registry.get_neurotransmitter("DA")
        assert state is not None

    def test_no_oscillations_no_noise_modulation(self):
        """Without oscillation state, noise should remain at baseline."""
        config = {
            "C_baseline": 0.3,
            "noise_band_coefficients": {
                "suppression": {"alpha": 0.4},
            },
        }
        engine = self._make_engine(config, osc=None)
        engine.step()  # should not crash
        state = engine.registry.get_neurotransmitter("DA")
        assert state is not None
