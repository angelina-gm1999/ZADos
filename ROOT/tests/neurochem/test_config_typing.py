"""
Tests for typed config wrappers and validation.

Phase 16: Verifies NTConfig, ReceptorConfig, and validation utilities.
"""

import pytest

from zados.neurochem.config.nt_config import NTConfig
from zados.neurochem.config.receptor_config import ReceptorConfig
from zados.neurochem.config.validation import (
    validate_nt_config,
    validate_receptor_config,
    validate_all_configs,
    NT_REQUIRED_KEYS,
    RECEPTOR_REQUIRED_KEYS,
)
from zados.neurochem.neurotransmitters.configs import (
    DEFAULT_NT_CONFIGS,
    DEFAULT_RECEPTOR_CONFIGS,
)


# =====================================================================
# NTConfig Tests
# =====================================================================

class TestNTConfig:
    """Test typed NT config wrapper."""

    def test_from_dict(self):
        config = NTConfig.from_dict(DEFAULT_NT_CONFIGS["DA"])
        assert config.C_baseline == 0.5
        assert config.u_base == 0.1

    def test_as_dict_roundtrip(self):
        original = DEFAULT_NT_CONFIGS["DA"]
        config = NTConfig.from_dict(original)
        result = config.as_dict()
        for key in NT_REQUIRED_KEYS:
            assert abs(result[key] - original[key]) < 1e-9

    def test_frozen(self):
        config = NTConfig.from_dict(DEFAULT_NT_CONFIGS["DA"])
        with pytest.raises(AttributeError):
            config.C_baseline = 0.9

    def test_total_clearance_rate(self):
        config = NTConfig.from_dict(DEFAULT_NT_CONFIGS["DA"])
        expected = 0.1 + 0.05 + 0.02  # u_base + d_base + c_base
        assert abs(config.total_clearance_rate - expected) < 1e-9

    def test_tonic_snr(self):
        config = NTConfig.from_dict(DEFAULT_NT_CONFIGS["DA"])
        expected = 0.5 / 0.05  # C_baseline / sigma_tonic = 10.0
        assert abs(config.tonic_snr - expected) < 1e-9

    def test_all_defaults_parseable(self):
        """Every default NT config should parse into NTConfig."""
        for nt_name, config_dict in DEFAULT_NT_CONFIGS.items():
            config = NTConfig.from_dict(config_dict)
            assert config.C_baseline > 0.0, f"{nt_name} has zero baseline"

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            NTConfig.from_dict({"C_baseline": 0.5})  # Missing other keys


# =====================================================================
# ReceptorConfig Tests
# =====================================================================

class TestReceptorConfig:
    """Test typed receptor config wrapper."""

    def test_from_dict(self):
        config = ReceptorConfig.from_dict(DEFAULT_RECEPTOR_CONFIGS["DA_D1"])
        assert config.K_d == 0.4
        assert config.parent_nt == "DA"
        assert config.exposure_tau == 10.0

    def test_as_dict_roundtrip(self):
        original = DEFAULT_RECEPTOR_CONFIGS["DA_D1"]
        config = ReceptorConfig.from_dict(original)
        result = config.as_dict()
        assert result["K_d"] == original["K_d"]
        assert result["parent_nt"] == original["parent_nt"]
        assert result["exposure_tau"] == original["exposure_tau"]

    def test_frozen(self):
        config = ReceptorConfig.from_dict(DEFAULT_RECEPTOR_CONFIGS["DA_D1"])
        with pytest.raises(AttributeError):
            config.K_d = 0.9

    def test_affinity(self):
        config = ReceptorConfig.from_dict(DEFAULT_RECEPTOR_CONFIGS["DA_D1"])
        assert abs(config.affinity - 2.5) < 1e-9  # 1/0.4

    def test_high_affinity_receptor(self):
        """DA_D3 has lowest K_d (0.2) = highest affinity."""
        d3 = ReceptorConfig.from_dict(DEFAULT_RECEPTOR_CONFIGS["DA_D3"])
        d1 = ReceptorConfig.from_dict(DEFAULT_RECEPTOR_CONFIGS["DA_D1"])
        assert d3.affinity > d1.affinity

    def test_all_defaults_parseable(self):
        """Every default receptor config should parse."""
        for rid, config_dict in DEFAULT_RECEPTOR_CONFIGS.items():
            config = ReceptorConfig.from_dict(config_dict)
            assert config.K_d > 0.0, f"{rid} has zero K_d"


# =====================================================================
# Validation Tests
# =====================================================================

class TestValidation:
    """Test config validation utilities."""

    def test_valid_nt_config(self):
        errors = validate_nt_config(DEFAULT_NT_CONFIGS["DA"], "DA")
        assert len(errors) == 0

    def test_all_default_nt_configs_valid(self):
        for nt_name, config in DEFAULT_NT_CONFIGS.items():
            errors = validate_nt_config(config, nt_name)
            assert len(errors) == 0, f"{nt_name}: {errors}"

    def test_missing_key(self):
        errors = validate_nt_config({"C_baseline": 0.5}, "TEST")
        assert len(errors) > 0
        assert any("Missing" in e for e in errors)

    def test_out_of_bounds(self):
        bad_config = dict(DEFAULT_NT_CONFIGS["DA"])
        bad_config["C_baseline"] = 5.0  # Out of [0, 1]
        errors = validate_nt_config(bad_config, "TEST")
        assert len(errors) > 0
        assert any("out of bounds" in e for e in errors)

    def test_invalid_type(self):
        bad_config = dict(DEFAULT_NT_CONFIGS["DA"])
        bad_config["C_baseline"] = "not_a_number"
        errors = validate_nt_config(bad_config, "TEST")
        assert len(errors) > 0
        assert any("expected numeric" in e for e in errors)

    def test_sigma_ordering_warning(self):
        bad_config = dict(DEFAULT_NT_CONFIGS["DA"])
        bad_config["sigma_phasic"] = 0.01  # Less than sigma_tonic (0.05)
        errors = validate_nt_config(bad_config, "TEST")
        assert len(errors) > 0
        assert any("sigma_phasic" in e for e in errors)

    def test_valid_receptor_config(self):
        errors = validate_receptor_config(
            DEFAULT_RECEPTOR_CONFIGS["DA_D1"], "DA_D1",
        )
        assert len(errors) == 0

    def test_all_default_receptor_configs_valid(self):
        for rid, config in DEFAULT_RECEPTOR_CONFIGS.items():
            errors = validate_receptor_config(config, rid)
            assert len(errors) == 0, f"{rid}: {errors}"

    def test_receptor_missing_key(self):
        errors = validate_receptor_config({"K_d": 0.4}, "TEST")
        assert len(errors) > 0

    def test_receptor_invalid_parent_nt(self):
        errors = validate_receptor_config({
            "K_d": 0.4, "parent_nt": "", "exposure_tau": 10.0,
        }, "TEST")
        assert len(errors) > 0

    def test_validate_all_configs(self):
        errors = validate_all_configs(
            DEFAULT_NT_CONFIGS, DEFAULT_RECEPTOR_CONFIGS,
        )
        assert len(errors) == 0
