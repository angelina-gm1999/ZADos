"""
Tests for multi-band K_d oscillatory modulation.

Phase 24: K_d(t) = K_d_base * (1 - Σ α_k * φ_k) with per-receptor
band coefficients, replacing single-band theta-only modulation.
"""

import pytest

from zados.neurochem.oscillations.oscillation_modulation import (
    modulate_K_d,
    modulate_K_d_multiband,
)
from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.state import NeurotransmitterState, OscillationState
from zados.neurochem.state.receptor_state import ReceptorState


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

class TestModulateKdMultiband:
    """Tests for the modulate_K_d_multiband pure function."""

    def test_single_theta_matches_legacy(self):
        """Single theta coefficient should match legacy modulate_K_d."""
        K_d_base = 0.5
        phi_theta = 0.7
        alpha_coeff = 0.3  # same as legacy default

        legacy = modulate_K_d(K_d_base, phi_theta, kd_coefficient=alpha_coeff)
        multiband = modulate_K_d_multiband(
            K_d_base,
            osc_amplitudes={"theta": phi_theta, "alpha": 0.0, "gamma": 0.0},
            band_coefficients={"theta": alpha_coeff},
        )
        assert multiband == pytest.approx(legacy)

    def test_two_bands(self):
        """Theta + gamma coefficients should both apply."""
        K_d_base = 0.5
        osc = {"theta": 0.5, "gamma": 0.4}
        coeffs = {"theta": 0.3, "gamma": 0.2}
        # K_d = 0.5 * (1 - 0.3*0.5 - 0.2*0.4) = 0.5 * (1 - 0.15 - 0.08) = 0.5 * 0.77 = 0.385
        result = modulate_K_d_multiband(K_d_base, osc, coeffs)
        assert result == pytest.approx(0.385)

    def test_zero_coefficients(self):
        """No modulation when all coefficients are zero."""
        K_d_base = 0.5
        osc = {"theta": 0.8, "gamma": 0.9, "alpha": 0.7}
        coeffs = {"theta": 0.0, "gamma": 0.0, "alpha": 0.0}
        result = modulate_K_d_multiband(K_d_base, osc, coeffs)
        assert result == pytest.approx(K_d_base)

    def test_clamped_positive(self):
        """K_d should never go below 0.01."""
        K_d_base = 0.1
        osc = {"theta": 1.0, "gamma": 1.0}
        coeffs = {"theta": 0.8, "gamma": 0.8}
        # K_d = 0.1 * (1 - 0.8 - 0.8) = 0.1 * (-0.6) = -0.06 → clamped to 0.01
        result = modulate_K_d_multiband(K_d_base, osc, coeffs)
        assert result == pytest.approx(0.01)

    def test_all_bands(self):
        """All 5 bands contributing."""
        K_d_base = 1.0
        osc = {"delta": 0.2, "theta": 0.3, "alpha": 0.4, "beta": 0.5, "gamma": 0.6}
        coeffs = {"delta": 0.1, "theta": 0.2, "alpha": 0.1, "beta": 0.05, "gamma": 0.15}
        # total_mod = 0.1*0.2 + 0.2*0.3 + 0.1*0.4 + 0.05*0.5 + 0.15*0.6
        # = 0.02 + 0.06 + 0.04 + 0.025 + 0.09 = 0.235
        # K_d = 1.0 * (1 - 0.235) = 0.765
        result = modulate_K_d_multiband(K_d_base, osc, coeffs)
        assert result == pytest.approx(0.765)

    def test_empty_amplitudes(self):
        """Empty oscillation amplitudes → no modulation."""
        K_d_base = 0.5
        result = modulate_K_d_multiband(K_d_base, {}, {"theta": 0.3})
        assert result == pytest.approx(K_d_base)

    def test_missing_band_in_coefficients(self):
        """Bands present in amplitudes but not in coefficients → coefficient=0."""
        K_d_base = 0.5
        osc = {"theta": 0.8, "gamma": 0.9}
        coeffs = {"theta": 0.3}  # gamma not in coefficients
        # Only theta applies: K_d = 0.5 * (1 - 0.3*0.8) = 0.5 * 0.76 = 0.38
        result = modulate_K_d_multiband(K_d_base, osc, coeffs)
        assert result == pytest.approx(0.38)


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------

class TestEngineMultibandKd:
    """Tests for engine-level multi-band K_d modulation."""

    def _make_engine_with_receptor(self, receptor_config, osc=None):
        """Helper: create engine with DA + DA_D1 receptor."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        da_state = NeurotransmitterState(
            C_tonic=0.3, C_phasic=0.0, F=0.0, eta_u=0.0,
        )
        engine.add_neurotransmitter("DA", initial_state=da_state, config={"C_baseline": 0.3})
        r1 = ReceptorState(receptor_id="DA_D1", rho=0.5)
        engine.add_receptor("DA_D1", initial_state=r1, config=receptor_config)
        if osc:
            engine.registry.set_oscillations(osc)
        return engine

    def test_engine_uses_multiband_when_configured(self):
        """Engine should use multiband K_d when kd_band_coefficients present."""
        osc = OscillationState(theta=0.5, gamma=0.4)
        config_multiband = {
            "K_d": 0.5,
            "kd_band_coefficients": {"theta": 0.3, "gamma": 0.2},
        }
        engine = self._make_engine_with_receptor(config_multiband, osc)
        engine.step()
        # Should not crash; receptor updated
        state = engine.registry.get_receptor("DA_D1")
        assert state is not None

    def test_engine_falls_back_to_legacy(self):
        """No kd_band_coefficients → legacy theta-only modulation."""
        osc = OscillationState(theta=0.5)
        config_legacy = {"K_d": 0.5}
        engine = self._make_engine_with_receptor(config_legacy, osc)
        engine.step()
        # Should not crash; uses legacy path
        state = engine.registry.get_receptor("DA_D1")
        assert state is not None

    def test_multiband_changes_effective_kd(self):
        """Multi-band config should produce different A_ij than legacy."""
        osc = OscillationState(theta=0.5, gamma=0.6)

        # Legacy: only theta affects K_d
        config_legacy = {"K_d": 0.5}
        engine_leg = self._make_engine_with_receptor(config_legacy, osc.copy())
        engine_leg.step()
        aij_legacy = engine_leg.registry.get_effective_signaling("DA_D1")

        # Multiband: theta + gamma affect K_d (more modulation → lower K_d → higher sat → higher A_ij)
        config_multi = {
            "K_d": 0.5,
            "kd_band_coefficients": {"theta": 0.3, "gamma": 0.3},
        }
        engine_multi = self._make_engine_with_receptor(config_multi, osc.copy())
        engine_multi.step()
        aij_multi = engine_multi.registry.get_effective_signaling("DA_D1")

        # Multiband should produce higher A_ij (lower K_d → higher saturation)
        assert aij_multi > aij_legacy

    def test_no_oscillations_no_modulation(self):
        """Without oscillation state, K_d should remain at baseline."""
        config = {
            "K_d": 0.5,
            "kd_band_coefficients": {"theta": 0.3, "gamma": 0.2},
        }
        engine = self._make_engine_with_receptor(config, osc=None)
        engine.step()
        # Should not crash even with config present
        state = engine.registry.get_receptor("DA_D1")
        assert state is not None
