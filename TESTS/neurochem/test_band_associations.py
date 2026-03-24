"""
Tests for oscillation band association data and accessors.

Phase 10: Foundation tests — data structure completeness, accessor functions.
"""

import pytest

from zados.neurochem.oscillations.band_associations import (
    NT_BAND_ASSOCIATIONS,
    BAND_MODULATION_DEFAULTS,
    get_primary_bands,
    get_secondary_bands,
    get_all_associated_bands,
    get_nts_for_band,
)


# =====================================================================
# Data Completeness Tests
# =====================================================================

class TestNTBandAssociationsCompleteness:
    """All 11 NTs must be present with valid band associations."""

    EXPECTED_NTS = [
        "DA", "5HT", "NE", "ACh", "OXT", "MOR", "CB1",
        "cortisol", "CRH", "GABA", "GLU", "histamine",
    ]

    VALID_BANDS = {"delta", "theta", "alpha", "beta", "gamma",
                   "theta_gamma", "alpha_beta"}

    def test_all_12_nts_present(self):
        for nt in self.EXPECTED_NTS:
            assert nt in NT_BAND_ASSOCIATIONS, f"Missing NT: {nt}"

    def test_no_extra_nts(self):
        for nt in NT_BAND_ASSOCIATIONS:
            assert nt in self.EXPECTED_NTS, f"Unexpected NT: {nt}"

    def test_all_have_primary_key(self):
        for nt, entry in NT_BAND_ASSOCIATIONS.items():
            assert "primary" in entry, f"{nt} missing 'primary' key"

    def test_all_have_secondary_key(self):
        for nt, entry in NT_BAND_ASSOCIATIONS.items():
            assert "secondary" in entry, f"{nt} missing 'secondary' key"

    def test_primary_bands_are_valid(self):
        for nt, entry in NT_BAND_ASSOCIATIONS.items():
            for band in entry["primary"]:
                assert band in self.VALID_BANDS, (
                    f"{nt} has invalid primary band: {band}"
                )

    def test_secondary_bands_are_valid(self):
        for nt, entry in NT_BAND_ASSOCIATIONS.items():
            for band in entry["secondary"]:
                assert band in self.VALID_BANDS, (
                    f"{nt} has invalid secondary band: {band}"
                )

    def test_every_nt_has_at_least_one_primary(self):
        for nt, entry in NT_BAND_ASSOCIATIONS.items():
            assert len(entry["primary"]) >= 1, (
                f"{nt} has no primary bands"
            )

    def test_no_overlap_primary_secondary(self):
        for nt, entry in NT_BAND_ASSOCIATIONS.items():
            overlap = set(entry["primary"]) & set(entry["secondary"])
            assert len(overlap) == 0, (
                f"{nt} has overlapping bands: {overlap}"
            )


# =====================================================================
# Specific Association Tests (per PDF Appendix I)
# =====================================================================

class TestSpecificAssociations:
    """Verify specific NT -> band mappings match PDF spec."""

    def test_da_primary(self):
        assert "gamma" in NT_BAND_ASSOCIATIONS["DA"]["primary"]
        assert "theta" in NT_BAND_ASSOCIATIONS["DA"]["primary"]

    def test_serotonin_primary(self):
        assert "theta" in NT_BAND_ASSOCIATIONS["5HT"]["primary"]
        assert "alpha" in NT_BAND_ASSOCIATIONS["5HT"]["primary"]

    def test_ne_primary(self):
        assert "beta" in NT_BAND_ASSOCIATIONS["NE"]["primary"]

    def test_ach_primary(self):
        assert "beta" in NT_BAND_ASSOCIATIONS["ACh"]["primary"]
        assert len(NT_BAND_ASSOCIATIONS["ACh"]["primary"]) == 1

    def test_oxt_primary(self):
        assert "theta" in NT_BAND_ASSOCIATIONS["OXT"]["primary"]

    def test_mor_primary(self):
        assert "delta" in NT_BAND_ASSOCIATIONS["MOR"]["primary"]

    def test_cb1_primary(self):
        assert "delta" in NT_BAND_ASSOCIATIONS["CB1"]["primary"]

    def test_cortisol_primary(self):
        assert "beta" in NT_BAND_ASSOCIATIONS["cortisol"]["primary"]

    def test_crh_primary(self):
        assert "beta" in NT_BAND_ASSOCIATIONS["CRH"]["primary"]

    def test_gaba_primary(self):
        assert "alpha" in NT_BAND_ASSOCIATIONS["GABA"]["primary"]
        assert "delta" in NT_BAND_ASSOCIATIONS["GABA"]["primary"]

    def test_glu_primary(self):
        assert "gamma" in NT_BAND_ASSOCIATIONS["GLU"]["primary"]
        assert "theta_gamma" in NT_BAND_ASSOCIATIONS["GLU"]["primary"]


# =====================================================================
# Band Modulation Defaults Tests
# =====================================================================

class TestBandModulationDefaults:
    """Tests for BAND_MODULATION_DEFAULTS data."""

    def test_all_five_bands_present(self):
        for band in ["gamma", "theta", "alpha", "beta", "delta"]:
            assert band in BAND_MODULATION_DEFAULTS

    def test_gamma_targets_release(self):
        assert BAND_MODULATION_DEFAULTS["gamma"] == "release"

    def test_theta_targets_K_d(self):
        assert BAND_MODULATION_DEFAULTS["theta"] == "K_d"

    def test_alpha_targets_noise(self):
        assert BAND_MODULATION_DEFAULTS["alpha"] == "noise"

    def test_beta_targets_desensitization(self):
        assert BAND_MODULATION_DEFAULTS["beta"] == "desensitization"

    def test_delta_targets_tonic(self):
        assert BAND_MODULATION_DEFAULTS["delta"] == "tonic"


# =====================================================================
# Accessor Function Tests
# =====================================================================

class TestAccessorFunctions:
    """Tests for get_primary_bands, get_secondary_bands, etc."""

    def test_get_primary_bands_da(self):
        result = get_primary_bands("DA")
        assert "gamma" in result
        assert "theta" in result

    def test_get_secondary_bands_da(self):
        result = get_secondary_bands("DA")
        assert "beta" in result

    def test_get_all_associated_bands(self):
        result = get_all_associated_bands("DA")
        assert "gamma" in result
        assert "theta" in result
        assert "beta" in result

    def test_get_primary_bands_unknown_nt(self):
        assert get_primary_bands("UNKNOWN") == []

    def test_get_secondary_bands_unknown_nt(self):
        assert get_secondary_bands("UNKNOWN") == []

    def test_get_all_associated_bands_unknown(self):
        assert get_all_associated_bands("UNKNOWN") == []

    def test_get_nts_for_gamma(self):
        result = get_nts_for_band("gamma")
        assert "DA" in result
        assert "GLU" in result

    def test_get_nts_for_beta(self):
        result = get_nts_for_band("beta")
        assert "NE" in result
        assert "ACh" in result
        assert "cortisol" in result
        assert "CRH" in result
        assert "histamine" in result

    def test_get_nts_for_theta(self):
        result = get_nts_for_band("theta")
        assert "DA" in result
        assert "5HT" in result
        assert "OXT" in result

    def test_get_nts_for_delta(self):
        result = get_nts_for_band("delta")
        assert "MOR" in result
        assert "CB1" in result
        assert "GABA" in result

    def test_get_nts_for_alpha(self):
        result = get_nts_for_band("alpha")
        assert "5HT" in result
        assert "GABA" in result

    def test_get_nts_for_unknown_band(self):
        assert get_nts_for_band("nonexistent") == []

    def test_primary_returns_new_list(self):
        """Ensure returned list is a copy, not a reference."""
        result1 = get_primary_bands("DA")
        result2 = get_primary_bands("DA")
        result1.append("fake")
        assert "fake" not in result2


class TestModuleRegistryImport:
    """Test that module_registry imports and works."""

    def test_import_registry(self):
        from zados.neurochem.neurotransmitters.module_registry import (
            NTModuleRegistry,
        )
        # Should start empty (or be clearable)
        NTModuleRegistry.clear()
        assert NTModuleRegistry.count() == 0

    def test_register_and_get(self):
        from zados.neurochem.neurotransmitters.module_registry import (
            NTModuleRegistry,
        )
        from zados.neurochem.neurotransmitters.base import (
            NeurotransmitterModule,
            ReleaseDriveSpec,
            OscillationCouplingRule,
        )

        class TestModule(NeurotransmitterModule):
            @property
            def name(self): return "TEST"
            @property
            def release_spec(self):
                return ReleaseDriveSpec(signal_keys=["a"], weights=[1.0])
            @property
            def oscillation_rules(self): return []

        NTModuleRegistry.clear()
        module = TestModule()
        NTModuleRegistry.register(module)

        assert NTModuleRegistry.is_registered("TEST")
        assert NTModuleRegistry.get("TEST") is module
        assert NTModuleRegistry.count() == 1
        assert "TEST" in NTModuleRegistry.registered_names()

        NTModuleRegistry.clear()

    def test_get_unregistered_returns_none(self):
        from zados.neurochem.neurotransmitters.module_registry import (
            NTModuleRegistry,
        )
        NTModuleRegistry.clear()
        assert NTModuleRegistry.get("NONEXISTENT") is None
