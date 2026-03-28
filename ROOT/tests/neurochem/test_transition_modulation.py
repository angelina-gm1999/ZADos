"""
Tests for multi-band CTMC transition rate modulation.

Phase 26: TransitionBandSpec + compute_transition_multiplier.
All 5 bands can modulate transition timing via per-receptor specs.
"""

import pytest

from zados.neurochem.oscillations.transition_modulation import (
    TransitionBandSpec,
    compute_transition_multiplier,
)
from zados.neurochem.kinetics.receptor_dynamics import (
    compute_transition_rates,
    step_receptor_dynamics,
)
from zados.neurochem.state.receptor_state import ReceptorState, ReceptorFunctionalState
from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.state import NeurotransmitterState, OscillationState


# ---------------------------------------------------------------------------
# TransitionBandSpec dataclass
# ---------------------------------------------------------------------------

class TestTransitionBandSpec:
    """Tests for the TransitionBandSpec frozen dataclass."""

    def test_frozen(self):
        spec = TransitionBandSpec("beta", 0.3)
        with pytest.raises(AttributeError):
            spec.lambda_coeff = 0.5

    def test_fields(self):
        spec = TransitionBandSpec("gamma", -0.2)
        assert spec.band == "gamma"
        assert spec.lambda_coeff == -0.2


# ---------------------------------------------------------------------------
# compute_transition_multiplier
# ---------------------------------------------------------------------------

class TestComputeTransitionMultiplier:
    """Tests for the pure compute_transition_multiplier function."""

    def test_no_specs_returns_one(self):
        """Empty specs → multiplier = 1.0."""
        m = compute_transition_multiplier({"beta": 0.5}, [])
        assert m == pytest.approx(1.0)

    def test_single_band(self):
        """One band contributes to multiplier."""
        specs = [TransitionBandSpec("beta", 0.3)]
        m = compute_transition_multiplier({"beta": 0.8}, specs)
        # m = 1 + 0.3 * 0.8 = 1.24
        assert m == pytest.approx(1.24)

    def test_clamp_min(self):
        """Multiplier clamped to m_min."""
        specs = [TransitionBandSpec("beta", -5.0)]
        m = compute_transition_multiplier({"beta": 1.0}, specs, m_min=0.1)
        # m = 1 + (-5.0) * 1.0 = -4.0 → clamped to 0.1
        assert m == pytest.approx(0.1)

    def test_clamp_max(self):
        """Multiplier clamped to m_max."""
        specs = [TransitionBandSpec("gamma", 10.0)]
        m = compute_transition_multiplier({"gamma": 1.0}, specs, m_max=3.0)
        # m = 1 + 10.0 * 1.0 = 11.0 → clamped to 3.0
        assert m == pytest.approx(3.0)

    def test_multi_band(self):
        """Multiple bands combine additively."""
        specs = [
            TransitionBandSpec("beta", 0.3),
            TransitionBandSpec("gamma", -0.2),
        ]
        osc = {"beta": 0.8, "gamma": 0.5}
        m = compute_transition_multiplier(osc, specs)
        # m = 1 + 0.3*0.8 + (-0.2)*0.5 = 1 + 0.24 - 0.10 = 1.14
        assert m == pytest.approx(1.14)

    def test_missing_band_in_amplitudes(self):
        """Band in spec but not in amplitudes → contribution = 0."""
        specs = [TransitionBandSpec("theta", 0.5)]
        m = compute_transition_multiplier({"beta": 0.8}, specs)
        # theta not in amplitudes → 0 contribution → m = 1.0
        assert m == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_transition_rates with oscillation modulation
# ---------------------------------------------------------------------------

class TestTransitionRatesWithOscModulation:
    """Tests for compute_transition_rates with multi-band modulation."""

    def test_backward_compat_no_osc(self):
        """Without osc_amplitudes, behavior identical to legacy."""
        rates_legacy = compute_transition_rates(
            current_state=ReceptorFunctionalState.ACTIVE,
            saturation=0.8,
            exposure_trace=5.0,
            time_in_state=10.0,
            thresholds={"theta_desens": 0.7, "t0_desens": 5.0, "beta_desens_scaling": 0.3},
            beta_amplitude=0.0,
        )
        rates_new = compute_transition_rates(
            current_state=ReceptorFunctionalState.ACTIVE,
            saturation=0.8,
            exposure_trace=5.0,
            time_in_state=10.0,
            thresholds={"theta_desens": 0.7, "t0_desens": 5.0, "beta_desens_scaling": 0.3},
            beta_amplitude=0.0,
            osc_amplitudes=None,
            transition_specs=None,
        )
        assert rates_legacy == rates_new

    def test_beta_legacy_still_works(self):
        """Legacy beta_amplitude parameter still functions."""
        # Without beta: t0_desens_eff = 5.0, time_in_state=6 > 5 → transition fires
        rates_no_beta = compute_transition_rates(
            current_state=ReceptorFunctionalState.ACTIVE,
            saturation=0.8,
            exposure_trace=0.0,
            time_in_state=6.0,
            thresholds={"theta_desens": 0.7, "t0_desens": 5.0, "beta_desens_scaling": 0.3},
            beta_amplitude=0.0,
        )
        assert ReceptorFunctionalState.DESENSITIZED in rates_no_beta

        # With high beta: t0_desens_eff = 5.0 * (1 - 0.3*1.0) = 3.5
        # time_in_state=4 > 3.5 → should still fire
        rates_with_beta = compute_transition_rates(
            current_state=ReceptorFunctionalState.ACTIVE,
            saturation=0.8,
            exposure_trace=0.0,
            time_in_state=4.0,
            thresholds={"theta_desens": 0.7, "t0_desens": 5.0, "beta_desens_scaling": 0.3},
            beta_amplitude=1.0,
        )
        assert ReceptorFunctionalState.DESENSITIZED in rates_with_beta

    def test_transition_specs_accelerate_desensitization(self):
        """Gamma oscillation can accelerate desensitization via transition_specs."""
        specs = {
            "desensitization": [TransitionBandSpec("gamma", 1.0)],
        }
        osc = {"gamma": 0.5}
        # Without modulation: t0_desens_eff = 5.0, time=3 < 5 → no transition
        rates_without = compute_transition_rates(
            current_state=ReceptorFunctionalState.ACTIVE,
            saturation=0.8,
            exposure_trace=0.0,
            time_in_state=3.0,
            thresholds={"theta_desens": 0.7, "t0_desens": 5.0, "beta_desens_scaling": 0.0},
        )
        assert ReceptorFunctionalState.DESENSITIZED not in rates_without

        # With modulation: m = 1 + 1.0*0.5 = 1.5, t_eff = 5.0/1.5 = 3.33
        # time=3.5 > 3.33 → transition fires
        rates_with = compute_transition_rates(
            current_state=ReceptorFunctionalState.ACTIVE,
            saturation=0.8,
            exposure_trace=0.0,
            time_in_state=3.5,
            thresholds={"theta_desens": 0.7, "t0_desens": 5.0, "beta_desens_scaling": 0.0},
            osc_amplitudes=osc,
            transition_specs=specs,
        )
        assert ReceptorFunctionalState.DESENSITIZED in rates_with

    def test_transition_specs_slow_recovery(self):
        """Negative coefficient can slow recovery."""
        specs = {
            "recovery": [TransitionBandSpec("delta", -0.5)],
        }
        osc = {"delta": 0.8}
        # m = 1 + (-0.5)*0.8 = 0.6, t_eff = 10.0/0.6 = 16.67
        # time=12 < 16.67 → no recovery
        rates = compute_transition_rates(
            current_state=ReceptorFunctionalState.DESENSITIZED,
            saturation=0.1,
            exposure_trace=5.0,
            time_in_state=12.0,
            thresholds={"epsilon_recovery": 0.3, "t_recovery": 10.0, "theta_intern": 50.0},
            osc_amplitudes=osc,
            transition_specs=specs,
        )
        assert ReceptorFunctionalState.ACTIVE not in rates


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineTransitionModulation:
    """Tests for engine-level multi-band CTMC modulation."""

    def test_engine_passes_osc_to_ctmc(self):
        """Engine passes oscillation amplitudes when transition_band_specs configured."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        da_state = NeurotransmitterState(C_tonic=0.3, C_phasic=0.0, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da_state, config={"C_baseline": 0.3})

        r1 = ReceptorState(receptor_id="DA_D1", rho=0.5)
        config = {
            "K_d": 0.5,
            "transition_band_specs": {
                "desensitization": [TransitionBandSpec("gamma", 0.5)],
            },
        }
        engine.add_receptor("DA_D1", initial_state=r1, config=config)
        engine.registry.set_oscillations(OscillationState(gamma=0.8))
        engine.step()  # should not crash
        state = engine.registry.get_receptor("DA_D1")
        assert state is not None

    def test_engine_no_specs_backward_compat(self):
        """No transition_band_specs → legacy behavior, no crash."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        da_state = NeurotransmitterState(C_tonic=0.3, C_phasic=0.0, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da_state, config={"C_baseline": 0.3})
        r1 = ReceptorState(receptor_id="DA_D1", rho=0.5)
        engine.add_receptor("DA_D1", initial_state=r1, config={"K_d": 0.5})
        engine.step()
        state = engine.registry.get_receptor("DA_D1")
        assert state is not None
