"""
Tests for oscillation modulation pure functions.

Phase 10: Foundation tests — K_d modulation, release modulation,
noise modulation, effective signaling proxy, tonic baseline modulation.
"""

import pytest

from zados.neurochem.oscillations.oscillation_modulation import (
    modulate_K_d,
    modulate_release,
    modulate_noise,
    modulate_reuptake,
    modulate_tonic_baseline,
    compute_g_chi,
    compute_effective_signaling_proxy,
)


# =====================================================================
# modulate_K_d Tests
# =====================================================================

class TestModulateKd:
    """Tests for theta -> K_d modulation."""

    def test_zero_theta_no_change(self):
        assert modulate_K_d(0.3, phi_theta=0.0) == 0.3

    def test_high_theta_reduces_K_d(self):
        # K_d(t) = 0.3 * (1 - 0.3 * 1.0) = 0.3 * 0.7 = 0.21
        result = modulate_K_d(0.3, phi_theta=1.0)
        assert abs(result - 0.21) < 1e-9

    def test_moderate_theta(self):
        # K_d(t) = 0.5 * (1 - 0.3 * 0.5) = 0.5 * 0.85 = 0.425
        result = modulate_K_d(0.5, phi_theta=0.5)
        assert abs(result - 0.425) < 1e-9

    def test_custom_kd_coefficient(self):
        # K_d(t) = 0.4 * (1 - 0.5 * 0.8) = 0.4 * 0.6 = 0.24
        result = modulate_K_d(0.4, phi_theta=0.8, kd_coefficient=0.5)
        assert abs(result - 0.24) < 1e-9

    def test_result_always_positive(self):
        # Even with max modulation, K_d should be >= 0.01
        result = modulate_K_d(0.01, phi_theta=1.0, kd_coefficient=0.99)
        assert result >= 0.01

    def test_clamp_prevents_negative(self):
        # Force extreme parameters
        result = modulate_K_d(0.01, phi_theta=1.0, kd_coefficient=2.0)
        assert result == 0.01

    def test_preserves_monotonicity(self):
        # Higher theta should give lower K_d
        K_d_low_theta = modulate_K_d(0.5, phi_theta=0.2)
        K_d_high_theta = modulate_K_d(0.5, phi_theta=0.8)
        assert K_d_high_theta < K_d_low_theta


# =====================================================================
# modulate_release Tests
# =====================================================================

class TestModulateRelease:
    """Tests for gamma -> release modulation."""

    def test_zero_gamma_no_change(self):
        assert modulate_release(0.5, phi_gamma=0.0) == 0.5

    def test_high_gamma_boosts_release(self):
        # R_mod = 0.5 * (1 + 0.5 * 1.0) = 0.5 * 1.5 = 0.75
        result = modulate_release(0.5, phi_gamma=1.0)
        assert abs(result - 0.75) < 1e-9

    def test_custom_coefficient(self):
        # R_mod = 1.0 * (1 + 0.8 * 0.5) = 1.0 * 1.4 = 1.4
        result = modulate_release(1.0, phi_gamma=0.5, coefficient=0.8)
        assert abs(result - 1.4) < 1e-9

    def test_zero_base_release(self):
        # 0 * anything = 0
        result = modulate_release(0.0, phi_gamma=1.0)
        assert result == 0.0

    def test_preserves_monotonicity(self):
        low = modulate_release(0.5, phi_gamma=0.2)
        high = modulate_release(0.5, phi_gamma=0.8)
        assert high > low


# =====================================================================
# modulate_noise Tests
# =====================================================================

class TestModulateNoise:
    """Tests for alpha -> noise suppression."""

    def test_zero_alpha_no_change(self):
        assert modulate_noise(0.05, phi_alpha=0.0) == 0.05

    def test_high_alpha_suppresses_noise(self):
        # sigma_mod = 0.05 * max(0.1, 1 - 0.4*1.0) = 0.05 * 0.6 = 0.03
        result = modulate_noise(0.05, phi_alpha=1.0)
        assert abs(result - 0.03) < 1e-9

    def test_moderate_alpha(self):
        # sigma_mod = 0.1 * max(0.1, 1 - 0.4*0.5) = 0.1 * 0.8 = 0.08
        result = modulate_noise(0.1, phi_alpha=0.5)
        assert abs(result - 0.08) < 1e-9

    def test_minimum_noise_floor(self):
        # With very high coefficient, noise should not go below 10% of base
        result = modulate_noise(0.1, phi_alpha=1.0, coefficient=2.0)
        # max(0.1, 1 - 2.0*1.0) = max(0.1, -1.0) = 0.1
        # 0.1 * 0.1 = 0.01
        assert abs(result - 0.01) < 1e-9

    def test_preserves_monotonicity(self):
        # Higher alpha -> lower noise
        low_alpha = modulate_noise(0.1, phi_alpha=0.2)
        high_alpha = modulate_noise(0.1, phi_alpha=0.8)
        assert high_alpha < low_alpha

    def test_always_non_negative(self):
        result = modulate_noise(0.05, phi_alpha=1.0, coefficient=5.0)
        assert result >= 0.0


# =====================================================================
# modulate_reuptake Tests
# =====================================================================

class TestModulateReuptake:
    """Tests for beta -> reuptake modulation."""

    def test_zero_beta_no_change(self):
        assert modulate_reuptake(0.1, phi_beta=0.0) == 0.1

    def test_high_beta_increases_reuptake(self):
        # u_mod = 0.1 * (1 + 0.3 * 1.0) = 0.1 * 1.3 = 0.13
        result = modulate_reuptake(0.1, phi_beta=1.0)
        assert abs(result - 0.13) < 1e-9


# =====================================================================
# modulate_tonic_baseline Tests
# =====================================================================

class TestModulateTonicBaseline:
    """Tests for delta -> tonic baseline modulation."""

    def test_zero_delta_no_change(self):
        assert modulate_tonic_baseline(0.5, phi_delta=0.0) == 0.5

    def test_high_delta_lowers_baseline(self):
        # C_mod = 0.5 * (1 - 0.2 * 1.0) = 0.5 * 0.8 = 0.4
        result = modulate_tonic_baseline(0.5, phi_delta=1.0)
        assert abs(result - 0.4) < 1e-9

    def test_clamp_to_positive(self):
        result = modulate_tonic_baseline(0.01, phi_delta=1.0, coefficient=2.0)
        assert result >= 0.01

    def test_clamp_upper_bound(self):
        result = modulate_tonic_baseline(1.5, phi_delta=0.0)
        assert result <= 1.0


# =====================================================================
# compute_g_chi Tests
# =====================================================================

class TestComputeGChi:
    """Tests for functional state gating factor."""

    def test_active(self):
        assert compute_g_chi("ACTIVE") == 1.0

    def test_desensitized(self):
        assert compute_g_chi("DESENSITIZED") == 0.5

    def test_internalized(self):
        assert compute_g_chi("INTERNALIZED") == 0.1

    def test_upregulated(self):
        assert compute_g_chi("UPREGULATED") == 1.2

    def test_unknown_defaults_to_active(self):
        assert compute_g_chi("UNKNOWN") == 1.0


# =====================================================================
# compute_effective_signaling_proxy Tests
# =====================================================================

class TestEffectiveSignalingProxy:
    """Tests for A_ij = rho * sigma * g(chi) * S computation."""

    def test_all_ones(self):
        result = compute_effective_signaling_proxy(1.0, 1.0, 1.0, 1.0)
        assert result == 1.0

    def test_all_zeros(self):
        result = compute_effective_signaling_proxy(0.0, 1.0, 1.0, 1.0)
        assert result == 0.0

    def test_typical_active_receptor(self):
        # rho=0.8, sigma=0.9, g_chi=1.0 (active), S=0.7
        result = compute_effective_signaling_proxy(0.8, 0.9, 1.0, 0.7)
        assert abs(result - 0.504) < 1e-9

    def test_desensitized_receptor(self):
        # Same but g_chi=0.5 (desensitized) -> halved signaling
        active = compute_effective_signaling_proxy(0.8, 0.9, 1.0, 0.7)
        desens = compute_effective_signaling_proxy(0.8, 0.9, 0.5, 0.7)
        assert abs(desens - active * 0.5) < 1e-9

    def test_upregulated_above_one(self):
        # g_chi=1.2 (upregulated) allows result > 1.0
        result = compute_effective_signaling_proxy(1.0, 1.0, 1.2, 1.0)
        assert result == 1.2

    def test_non_negative(self):
        result = compute_effective_signaling_proxy(0.0, 0.0, 0.0, 0.0)
        assert result >= 0.0

    def test_low_density_reduces_signaling(self):
        high_rho = compute_effective_signaling_proxy(0.9, 0.8, 1.0, 0.6)
        low_rho = compute_effective_signaling_proxy(0.3, 0.8, 1.0, 0.6)
        assert low_rho < high_rho
