"""
Tests for NeurotransmitterModule base classes and dataclasses.

Phase 10: Foundation tests — base class contracts, dataclass validation,
default implementations.
"""

import pytest
from typing import List, Dict

from zados.neurochem.neurotransmitters.base import (
    OscillationCouplingRule,
    ReleaseDriveSpec,
    NeurotransmitterModule,
)


# =====================================================================
# Concrete test module (minimal implementation for testing ABC)
# =====================================================================

class MockNTModule(NeurotransmitterModule):
    """Concrete implementation for testing."""

    @property
    def name(self) -> str:
        return "MOCK"

    @property
    def release_spec(self) -> ReleaseDriveSpec:
        return ReleaseDriveSpec(
            signal_keys=["signal_a", "signal_b", "emotion_drive"],
            weights=[0.5, 0.3, 0.2],
            threshold=0.1,
        )

    @property
    def oscillation_rules(self) -> List[OscillationCouplingRule]:
        return [
            OscillationCouplingRule(target="release", band="gamma", coefficient=0.5),
            OscillationCouplingRule(target="sigma_tonic", band="alpha", coefficient=-0.4),
        ]


# =====================================================================
# OscillationCouplingRule Tests
# =====================================================================

class TestOscillationCouplingRule:
    """Tests for OscillationCouplingRule dataclass."""

    def test_valid_construction(self):
        rule = OscillationCouplingRule(
            target="release", band="gamma", coefficient=0.5
        )
        assert rule.target == "release"
        assert rule.band == "gamma"
        assert rule.coefficient == 0.5
        assert rule.formula == "multiplicative"  # default

    def test_additive_formula(self):
        rule = OscillationCouplingRule(
            target="K_d", band="theta", coefficient=-0.3, formula="additive"
        )
        assert rule.formula == "additive"

    def test_negative_coefficient(self):
        rule = OscillationCouplingRule(
            target="sigma_tonic", band="alpha", coefficient=-0.4
        )
        assert rule.coefficient == -0.4

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError, match="Invalid target"):
            OscillationCouplingRule(
                target="invalid_target", band="gamma", coefficient=0.5
            )

    def test_invalid_band_raises(self):
        with pytest.raises(ValueError, match="Invalid band"):
            OscillationCouplingRule(
                target="release", band="invalid_band", coefficient=0.5
            )

    def test_invalid_formula_raises(self):
        with pytest.raises(ValueError, match="Invalid formula"):
            OscillationCouplingRule(
                target="release", band="gamma", coefficient=0.5,
                formula="invalid_formula"
            )

    def test_all_valid_targets(self):
        valid_targets = [
            "release", "reuptake", "sigma_tonic", "sigma_phasic",
            "K_d", "u_base", "d_base", "c_base",
            "theta_tonic", "theta_phasic",
        ]
        for target in valid_targets:
            rule = OscillationCouplingRule(
                target=target, band="gamma", coefficient=0.1
            )
            assert rule.target == target

    def test_all_valid_bands(self):
        valid_bands = [
            "delta", "theta", "alpha", "beta", "gamma",
            "theta_gamma", "alpha_beta",
        ]
        for band in valid_bands:
            rule = OscillationCouplingRule(
                target="release", band=band, coefficient=0.1
            )
            assert rule.band == band

    def test_frozen(self):
        rule = OscillationCouplingRule(
            target="release", band="gamma", coefficient=0.5
        )
        with pytest.raises(AttributeError):
            rule.coefficient = 0.9


# =====================================================================
# ReleaseDriveSpec Tests
# =====================================================================

class TestReleaseDriveSpec:
    """Tests for ReleaseDriveSpec dataclass."""

    def test_valid_construction(self):
        spec = ReleaseDriveSpec(
            signal_keys=["novelty", "rpe"],
            weights=[0.6, 0.4],
        )
        assert spec.signal_keys == ["novelty", "rpe"]
        assert spec.weights == [0.6, 0.4]
        assert spec.threshold == 0.0  # default

    def test_custom_threshold(self):
        spec = ReleaseDriveSpec(
            signal_keys=["a"], weights=[1.0], threshold=0.2
        )
        assert spec.threshold == 0.2

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="must match"):
            ReleaseDriveSpec(
                signal_keys=["a", "b"],
                weights=[1.0],
            )

    def test_empty_lists(self):
        spec = ReleaseDriveSpec(signal_keys=[], weights=[])
        assert len(spec.signal_keys) == 0

    def test_frozen(self):
        spec = ReleaseDriveSpec(
            signal_keys=["a"], weights=[1.0]
        )
        with pytest.raises(AttributeError):
            spec.threshold = 0.5


# =====================================================================
# NeurotransmitterModule Tests (via MockNTModule)
# =====================================================================

class TestNeurotransmitterModule:
    """Tests for NeurotransmitterModule abstract base class."""

    def test_name(self):
        module = MockNTModule()
        assert module.name == "MOCK"

    def test_release_spec(self):
        module = MockNTModule()
        spec = module.release_spec
        assert len(spec.signal_keys) == 3
        assert "emotion_drive" in spec.signal_keys

    def test_oscillation_rules(self):
        module = MockNTModule()
        rules = module.oscillation_rules
        assert len(rules) == 2
        assert rules[0].target == "release"
        assert rules[0].band == "gamma"
        assert rules[1].target == "sigma_tonic"

    # --- compute_release_drive ---

    def test_release_drive_weighted_sum(self):
        module = MockNTModule()
        drive = module.compute_release_drive({
            "signal_a": 1.0,
            "signal_b": 1.0,
            "emotion_drive": 1.0,
        })
        # 0.5*1.0 + 0.3*1.0 + 0.2*1.0 = 1.0, minus threshold 0.1 = 0.9
        assert abs(drive - 0.9) < 1e-9

    def test_release_drive_partial_signals(self):
        module = MockNTModule()
        drive = module.compute_release_drive({
            "signal_a": 0.5,
        })
        # 0.5*0.5 + 0.3*0.0 + 0.2*0.0 = 0.25, minus 0.1 = 0.15
        assert abs(drive - 0.15) < 1e-9

    def test_release_drive_below_threshold(self):
        module = MockNTModule()
        drive = module.compute_release_drive({
            "signal_a": 0.1,
        })
        # 0.5*0.1 = 0.05, minus 0.1 = -0.05 -> clamped to 0.0
        assert drive == 0.0

    def test_release_drive_empty_signals(self):
        module = MockNTModule()
        drive = module.compute_release_drive({})
        # All zeros, threshold 0.1 -> 0.0
        assert drive == 0.0

    def test_release_drive_non_negative(self):
        module = MockNTModule()
        drive = module.compute_release_drive({
            "signal_a": -1.0,
            "signal_b": -1.0,
            "emotion_drive": -1.0,
        })
        assert drive >= 0.0

    # --- apply_oscillation_coupling ---

    def test_oscillation_coupling_multiplicative(self):
        module = MockNTModule()
        params = {"release": 1.0, "sigma_tonic": 0.05}
        osc = {"gamma": 0.8, "alpha": 0.5}

        result = module.apply_oscillation_coupling(params, osc)

        # release: 1.0 * (1 + 0.5*0.8) = 1.4
        assert abs(result["release"] - 1.4) < 1e-9

        # sigma_tonic: 0.05 * (1 + (-0.4)*0.5) = 0.05 * 0.8 = 0.04
        assert abs(result["sigma_tonic"] - 0.04) < 1e-9

    def test_oscillation_coupling_zero_amplitudes(self):
        module = MockNTModule()
        params = {"release": 1.0, "sigma_tonic": 0.05}
        osc = {"gamma": 0.0, "alpha": 0.0}

        result = module.apply_oscillation_coupling(params, osc)

        assert result["release"] == 1.0
        assert result["sigma_tonic"] == 0.05

    def test_oscillation_coupling_missing_band(self):
        module = MockNTModule()
        params = {"release": 1.0, "sigma_tonic": 0.05}
        osc = {}  # no bands present

        result = module.apply_oscillation_coupling(params, osc)

        # Missing band defaults to 0.0, so no modulation
        assert result["release"] == 1.0
        assert result["sigma_tonic"] == 0.05

    def test_oscillation_coupling_preserves_other_params(self):
        module = MockNTModule()
        params = {"release": 1.0, "sigma_tonic": 0.05, "u_base": 0.1}
        osc = {"gamma": 0.5, "alpha": 0.3}

        result = module.apply_oscillation_coupling(params, osc)

        # u_base not targeted by any rule, should be preserved
        assert result["u_base"] == 0.1

    def test_oscillation_coupling_returns_copy(self):
        module = MockNTModule()
        params = {"release": 1.0, "sigma_tonic": 0.05}
        osc = {"gamma": 0.5, "alpha": 0.3}

        result = module.apply_oscillation_coupling(params, osc)

        # Original should not be modified
        assert params["release"] == 1.0

    # --- get_primary_release_band ---

    def test_get_primary_release_band(self):
        module = MockNTModule()
        assert module.get_primary_release_band() == "gamma"

    def test_get_primary_release_coefficient(self):
        module = MockNTModule()
        assert module.get_primary_release_coefficient() == 0.5


class TestNTModuleWithoutReleaseRule:
    """Test module with no release-targeting oscillation rules."""

    class NoReleaseModule(NeurotransmitterModule):
        @property
        def name(self): return "NR"
        @property
        def release_spec(self):
            return ReleaseDriveSpec(signal_keys=["x"], weights=[1.0])
        @property
        def oscillation_rules(self):
            return [
                OscillationCouplingRule(target="sigma_tonic", band="alpha", coefficient=-0.3),
            ]

    def test_no_primary_release_band(self):
        module = self.NoReleaseModule()
        assert module.get_primary_release_band() is None

    def test_no_primary_release_coefficient(self):
        module = self.NoReleaseModule()
        assert module.get_primary_release_coefficient() == 0.0


class TestAdditiveOscillationCoupling:
    """Test additive formula variant."""

    class AdditiveModule(NeurotransmitterModule):
        @property
        def name(self): return "ADD"
        @property
        def release_spec(self):
            return ReleaseDriveSpec(signal_keys=["x"], weights=[1.0])
        @property
        def oscillation_rules(self):
            return [
                OscillationCouplingRule(
                    target="K_d", band="theta", coefficient=-0.1,
                    formula="additive"
                ),
            ]

    def test_additive_coupling(self):
        module = self.AdditiveModule()
        params = {"K_d": 0.5}
        osc = {"theta": 0.7}

        result = module.apply_oscillation_coupling(params, osc)

        # K_d: 0.5 + (-0.1 * 0.7) = 0.5 - 0.07 = 0.43
        assert abs(result["K_d"] - 0.43) < 1e-9
