"""
Tests for gamma_gprotein inclusion in the A_ij effective signaling formula.

Phase 18: Both compute_effective_signaling (receptors/base.py) and
compute_effective_signaling_proxy (oscillation_modulation.py) now include
gamma_gprotein as a multiplicative factor in A_ij.

Formula: A_ij = rho * sigma * gamma * g(chi) * S * w
"""

import pytest

from zados.neurochem.oscillations.oscillation_modulation import (
    compute_effective_signaling_proxy,
    compute_g_chi,
)
from zados.neurochem.receptors.dopamine_receptors import DopamineReceptors


# ---------------------------------------------------------------------------
# compute_effective_signaling_proxy (standalone pure function)
# ---------------------------------------------------------------------------

class TestProxyGamma:
    """Tests for gamma_gprotein in compute_effective_signaling_proxy."""

    def test_proxy_gamma_1_matches_old_behavior(self):
        """gamma=1.0 (default) gives the same result as the old formula."""
        result = compute_effective_signaling_proxy(
            rho=0.8, sigma=0.7, g_chi=1.0, saturation=0.6,
        )
        expected = 0.8 * 0.7 * 1.0 * 0.6
        assert result == pytest.approx(expected)

        # Explicit gamma=1.0 should give the same result
        result_explicit = compute_effective_signaling_proxy(
            rho=0.8, sigma=0.7, g_chi=1.0, saturation=0.6,
            gamma_gprotein=1.0,
        )
        assert result_explicit == pytest.approx(expected)

    def test_proxy_gamma_zero_zeroes_signaling(self):
        """gamma=0.0 should zero out the entire A_ij."""
        result = compute_effective_signaling_proxy(
            rho=1.0, sigma=1.0, g_chi=1.0, saturation=1.0,
            gamma_gprotein=0.0,
        )
        assert result == pytest.approx(0.0)

    def test_proxy_gamma_half_halves_signaling(self):
        """gamma=0.5 should halve A_ij relative to gamma=1.0."""
        full = compute_effective_signaling_proxy(
            rho=0.8, sigma=0.7, g_chi=1.0, saturation=0.6,
            gamma_gprotein=1.0,
        )
        half = compute_effective_signaling_proxy(
            rho=0.8, sigma=0.7, g_chi=1.0, saturation=0.6,
            gamma_gprotein=0.5,
        )
        assert half == pytest.approx(full * 0.5)

    def test_proxy_gamma_reduces_signaling(self):
        """Decreasing gamma monotonically decreases A_ij."""
        params = dict(rho=0.9, sigma=0.8, g_chi=1.0, saturation=0.7)
        a_full = compute_effective_signaling_proxy(**params, gamma_gprotein=1.0)
        a_mid = compute_effective_signaling_proxy(**params, gamma_gprotein=0.6)
        a_low = compute_effective_signaling_proxy(**params, gamma_gprotein=0.2)
        assert a_full > a_mid > a_low > 0.0

    def test_proxy_gamma_with_desensitized_state(self):
        """gamma combines with g(chi) for DESENSITIZED state."""
        g_chi = compute_g_chi("DESENSITIZED")  # 0.5
        result = compute_effective_signaling_proxy(
            rho=1.0, sigma=1.0, g_chi=g_chi, saturation=1.0,
            gamma_gprotein=0.5,
        )
        # 1.0 * 1.0 * 0.5 * 0.5 * 1.0 = 0.25
        assert result == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# ReceptorFamilyModule.compute_effective_signaling (via DopamineReceptors)
# ---------------------------------------------------------------------------

class TestFamilyModuleGamma:
    """Tests for gamma_gprotein in ReceptorFamilyModule.compute_effective_signaling."""

    @pytest.fixture
    def da_module(self):
        return DopamineReceptors()

    def test_gamma_1_matches_old_behavior(self, da_module):
        """gamma=1.0 (default) preserves old behavior."""
        result = da_module.compute_effective_signaling(
            receptor_id="DA_D1",
            rho=1.0,
            sigma=1.0,
            functional_state="ACTIVE",
            saturation=1.0,
        )
        # DA_D1 weight = 1.0, g(ACTIVE) = 1.0
        assert result == pytest.approx(1.0)

        result_explicit = da_module.compute_effective_signaling(
            receptor_id="DA_D1",
            rho=1.0,
            sigma=1.0,
            functional_state="ACTIVE",
            saturation=1.0,
            gamma_gprotein=1.0,
        )
        assert result_explicit == pytest.approx(1.0)

    def test_gamma_zero_zeroes_signaling(self, da_module):
        """gamma=0.0 zeroes out signaling even with perfect receptor conditions."""
        result = da_module.compute_effective_signaling(
            receptor_id="DA_D1",
            rho=1.0,
            sigma=1.0,
            functional_state="ACTIVE",
            saturation=1.0,
            gamma_gprotein=0.0,
        )
        assert result == pytest.approx(0.0)

    def test_gamma_half_with_weight(self, da_module):
        """gamma=0.5 interacts correctly with receptor weight."""
        # DA_D2 weight = 0.9
        result = da_module.compute_effective_signaling(
            receptor_id="DA_D2",
            rho=1.0,
            sigma=1.0,
            functional_state="ACTIVE",
            saturation=1.0,
            gamma_gprotein=0.5,
        )
        # 1.0 * 1.0 * 0.5 * 1.0 * 1.0 * 0.9 = 0.45
        assert result == pytest.approx(0.45)

    def test_gamma_in_full_formula(self, da_module):
        """All factors combine correctly: rho * sigma * gamma * g(chi) * S * w."""
        result = da_module.compute_effective_signaling(
            receptor_id="DA_D4",  # weight = 0.85
            rho=0.8,
            sigma=0.6,
            functional_state="UPREGULATED",  # g(chi) = 1.2
            saturation=0.5,
            gamma_gprotein=0.7,
        )
        expected = 0.8 * 0.6 * 0.7 * 1.2 * 0.5 * 0.85
        assert result == pytest.approx(expected)

    def test_gamma_unknown_receptor_uses_default_weight(self, da_module):
        """Unknown receptor_id uses weight=1.0."""
        result = da_module.compute_effective_signaling(
            receptor_id="DA_UNKNOWN",
            rho=1.0,
            sigma=1.0,
            functional_state="ACTIVE",
            saturation=1.0,
            gamma_gprotein=0.3,
        )
        # weight=1.0, g=1.0, so result = 0.3
        assert result == pytest.approx(0.3)
