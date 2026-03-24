"""
Tests for oscillation-dependent threshold shifts.

Phase 27: ThresholdBandSpec + modulate_threshold.
Saturation thresholds for CTMC transitions are shifted by oscillation bands.
"""

import pytest

from zados.neurochem.oscillations.transition_modulation import (
    ThresholdBandSpec,
    modulate_threshold,
)
from zados.neurochem.kinetics.receptor_dynamics import (
    compute_transition_rates,
)
from zados.neurochem.state.receptor_state import ReceptorFunctionalState
from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.state import NeurotransmitterState, OscillationState
from zados.neurochem.state.receptor_state import ReceptorState


# ---------------------------------------------------------------------------
# ThresholdBandSpec dataclass
# ---------------------------------------------------------------------------

class TestThresholdBandSpec:
    """Tests for the ThresholdBandSpec frozen dataclass."""

    def test_frozen(self):
        spec = ThresholdBandSpec("theta", -0.1)
        with pytest.raises(AttributeError):
            spec.shift_coefficient = 0.0

    def test_fields(self):
        spec = ThresholdBandSpec("alpha", 0.15)
        assert spec.band == "alpha"
        assert spec.shift_coefficient == 0.15


# ---------------------------------------------------------------------------
# modulate_threshold
# ---------------------------------------------------------------------------

class TestModulateThreshold:
    """Tests for the pure modulate_threshold function."""

    def test_no_specs_returns_base(self):
        """Empty specs → base threshold unchanged."""
        result = modulate_threshold(0.7, {"theta": 0.5}, [])
        assert result == pytest.approx(0.7)

    def test_single_band_shifts(self):
        """Single band shifts threshold."""
        specs = [ThresholdBandSpec("theta", -0.1)]
        # base=0.7, shift = -0.1 * 0.5 = -0.05 → 0.65
        result = modulate_threshold(0.7, {"theta": 0.5}, specs)
        assert result == pytest.approx(0.65)

    def test_positive_shift_raises_threshold(self):
        """Positive coefficient raises threshold (harder to trigger)."""
        specs = [ThresholdBandSpec("alpha", 0.2)]
        # base=0.3, shift = 0.2 * 0.8 = 0.16 → 0.46
        result = modulate_threshold(0.3, {"alpha": 0.8}, specs)
        assert result == pytest.approx(0.46)

    def test_clamped_lower(self):
        """Threshold should not go below 0."""
        specs = [ThresholdBandSpec("gamma", -2.0)]
        result = modulate_threshold(0.3, {"gamma": 1.0}, specs)
        assert result == pytest.approx(0.0)

    def test_clamped_upper(self):
        """Threshold should not exceed 1."""
        specs = [ThresholdBandSpec("gamma", 2.0)]
        result = modulate_threshold(0.8, {"gamma": 1.0}, specs)
        assert result == pytest.approx(1.0)

    def test_multi_band_shifts(self):
        """Multiple bands shift threshold additively."""
        specs = [
            ThresholdBandSpec("theta", -0.1),
            ThresholdBandSpec("gamma", -0.05),
        ]
        osc = {"theta": 0.5, "gamma": 0.4}
        # base=0.7, shift = -0.1*0.5 + -0.05*0.4 = -0.05 + -0.02 = -0.07 → 0.63
        result = modulate_threshold(0.7, osc, specs)
        assert result == pytest.approx(0.63)

    def test_missing_band_no_effect(self):
        """Band in spec but not in amplitudes → no shift."""
        specs = [ThresholdBandSpec("delta", -0.2)]
        result = modulate_threshold(0.7, {"theta": 0.5}, specs)
        assert result == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Integration with compute_transition_rates
# ---------------------------------------------------------------------------

class TestTransitionRatesWithThresholdModulation:
    """Tests for threshold modulation within compute_transition_rates."""

    def test_theta_lowers_desens_threshold(self):
        """Theta oscillation lowers desensitization threshold → triggers earlier."""
        threshold_specs = {
            "theta_desens": [ThresholdBandSpec("theta", -0.15)],
        }
        osc = {"theta": 0.8}
        # base theta_desens = 0.7, shift = -0.15*0.8 = -0.12 → eff = 0.58
        # saturation=0.6 > 0.58 → should fire (would not fire at 0.7)
        rates = compute_transition_rates(
            current_state=ReceptorFunctionalState.ACTIVE,
            saturation=0.6,
            exposure_trace=0.0,
            time_in_state=10.0,
            thresholds={"theta_desens": 0.7, "t0_desens": 5.0, "beta_desens_scaling": 0.0},
            osc_amplitudes=osc,
            threshold_modulation_specs=threshold_specs,
        )
        assert ReceptorFunctionalState.DESENSITIZED in rates

    def test_without_threshold_mod_no_fire(self):
        """Same scenario without threshold modulation → should NOT fire."""
        rates = compute_transition_rates(
            current_state=ReceptorFunctionalState.ACTIVE,
            saturation=0.6,
            exposure_trace=0.0,
            time_in_state=10.0,
            thresholds={"theta_desens": 0.7, "t0_desens": 5.0, "beta_desens_scaling": 0.0},
        )
        assert ReceptorFunctionalState.DESENSITIZED not in rates

    def test_threshold_mod_upreg_exit(self):
        """Threshold modulation on upreg_exit lowers exit threshold."""
        threshold_specs = {
            "theta_upreg_exit": [ThresholdBandSpec("gamma", -0.2)],
        }
        osc = {"gamma": 0.5}
        # base theta_upreg_exit=0.4, shift = -0.2*0.5 = -0.1 → eff = 0.3
        # saturation=0.35 > 0.3 → should exit upregulated
        rates = compute_transition_rates(
            current_state=ReceptorFunctionalState.UPREGULATED,
            saturation=0.35,
            exposure_trace=0.0,
            time_in_state=10.0,
            thresholds={"theta_upreg_exit": 0.4, "t_upreg_exit": 5.0},
            osc_amplitudes=osc,
            threshold_modulation_specs=threshold_specs,
        )
        assert ReceptorFunctionalState.ACTIVE in rates


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineThresholdModulation:
    """Tests for engine-level threshold modulation."""

    def test_engine_with_threshold_specs(self):
        """Engine should pass threshold_modulation_specs to CTMC."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        da_state = NeurotransmitterState(C_tonic=0.3, C_phasic=0.0, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da_state, config={"C_baseline": 0.3})
        r1 = ReceptorState(receptor_id="DA_D1", rho=0.5)
        config = {
            "K_d": 0.5,
            "threshold_modulation_specs": {
                "theta_desens": [ThresholdBandSpec("theta", -0.1)],
            },
        }
        engine.add_receptor("DA_D1", initial_state=r1, config=config)
        engine.registry.set_oscillations(OscillationState(theta=0.6))
        engine.step()  # should not crash
        state = engine.registry.get_receptor("DA_D1")
        assert state is not None
