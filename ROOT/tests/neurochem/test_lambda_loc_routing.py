"""
Tests for lambda_loc concentration routing.

Phase 22: Receptors can see different concentration pools based on their
synaptic localization (lambda_loc). Presynaptic receptors see full
C_tonic + C_phasic; extrasynaptic see only C_tonic (volume transmission).

The routing is opt-in via engine's use_lambda_loc_routing flag.
"""

import pytest

from zados.neurochem.kinetics.receptor_dynamics import compute_effective_concentration
from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.state import NeurotransmitterState, ReceptorState


# ---------------------------------------------------------------------------
# Pure function: compute_effective_concentration
# ---------------------------------------------------------------------------

class TestComputeEffectiveConcentration:
    """Tests for the pure compute_effective_concentration function."""

    def test_presynaptic_sees_full_c(self):
        """lambda=0 → C_tonic + C_phasic."""
        result = compute_effective_concentration(
            C_tonic=0.3, C_phasic=0.5, lambda_loc=0.0,
        )
        assert result == pytest.approx(0.8)

    def test_synaptic_sees_half_phasic(self):
        """lambda=0.5 → C_tonic + 0.5 * C_phasic."""
        result = compute_effective_concentration(
            C_tonic=0.3, C_phasic=0.5, lambda_loc=0.5,
        )
        assert result == pytest.approx(0.55)

    def test_extrasynaptic_sees_only_tonic(self):
        """lambda=1.0 → C_tonic only."""
        result = compute_effective_concentration(
            C_tonic=0.3, C_phasic=0.5, lambda_loc=1.0,
        )
        assert result == pytest.approx(0.3)

    def test_no_phasic_always_tonic(self):
        """C_phasic=0 → always C_tonic regardless of lambda."""
        for lam in [0.0, 0.5, 1.0]:
            result = compute_effective_concentration(
                C_tonic=0.4, C_phasic=0.0, lambda_loc=lam,
            )
            assert result == pytest.approx(0.4)

    def test_no_tonic_presynaptic_sees_phasic(self):
        """C_tonic=0, lambda=0 → C_phasic only."""
        result = compute_effective_concentration(
            C_tonic=0.0, C_phasic=0.6, lambda_loc=0.0,
        )
        assert result == pytest.approx(0.6)

    def test_no_tonic_extrasynaptic_sees_nothing(self):
        """C_tonic=0, lambda=1.0 → 0."""
        result = compute_effective_concentration(
            C_tonic=0.0, C_phasic=0.6, lambda_loc=1.0,
        )
        assert result == pytest.approx(0.0)

    def test_result_non_negative(self):
        """Result should never be negative."""
        result = compute_effective_concentration(
            C_tonic=0.0, C_phasic=0.0, lambda_loc=0.5,
        )
        assert result >= 0.0


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineLambdaLocRouting:
    """Tests for engine-level lambda_loc routing."""

    def test_default_no_routing(self):
        """Default engine (flag=False) uses total C = C_tonic + C_phasic."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        da_state = NeurotransmitterState(C_tonic=0.3, C_phasic=0.5, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da_state, config={"C_baseline": 0.3})

        # Extrasynaptic receptor should still see full C with routing off
        receptor = ReceptorState(receptor_id="DA_D1", lambda_loc=1.0)
        engine.add_receptor("DA_D1", initial_state=receptor, config={"K_d": 0.5})

        # Also add a presynaptic receptor for comparison
        receptor2 = ReceptorState(receptor_id="DA_D2", lambda_loc=0.0)
        engine.add_receptor("DA_D2", initial_state=receptor2, config={"K_d": 0.5})

        engine.step()

        # Without routing, both should see the same concentration
        # (the A_ij difference is only from the proxy which uses same sat)
        state1 = engine.registry.get_receptor("DA_D1")
        state2 = engine.registry.get_receptor("DA_D2")
        # Both should have similar exposure traces since they see same concentration
        assert state1.exposure_trace == pytest.approx(state2.exposure_trace, rel=0.01)

    def test_with_routing_enabled(self):
        """With routing on, extrasynaptic receptor sees less concentration."""
        engine = NeurochemicalEngine(dt=0.1, seed=42, use_lambda_loc_routing=True)
        da_state = NeurotransmitterState(C_tonic=0.3, C_phasic=0.5, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da_state, config={"C_baseline": 0.3})

        # Presynaptic: sees C_tonic + C_phasic = 0.8
        receptor_pre = ReceptorState(receptor_id="DA_D1", lambda_loc=0.0)
        engine.add_receptor("DA_D1", initial_state=receptor_pre, config={"K_d": 0.5})

        # Extrasynaptic: sees only C_tonic = 0.3
        receptor_extra = ReceptorState(receptor_id="DA_D2", lambda_loc=1.0)
        engine.add_receptor("DA_D2", initial_state=receptor_extra, config={"K_d": 0.5})

        engine.step()

        state_pre = engine.registry.get_receptor("DA_D1")
        state_extra = engine.registry.get_receptor("DA_D2")

        # Presynaptic sees higher concentration → higher exposure trace
        assert state_pre.exposure_trace > state_extra.exposure_trace

    def test_extrasynaptic_ignores_phasic_bursts(self):
        """lambda=1.0 receptor should be unaffected by phasic burst changes."""
        engine = NeurochemicalEngine(dt=0.1, seed=42, use_lambda_loc_routing=True)

        # Start with no phasic
        da_state = NeurotransmitterState(C_tonic=0.4, C_phasic=0.0, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da_state, config={"C_baseline": 0.4})
        receptor = ReceptorState(receptor_id="DA_D1", lambda_loc=1.0)
        engine.add_receptor("DA_D1", initial_state=receptor, config={"K_d": 0.5})
        engine.step()
        trace_no_phasic = engine.registry.get_receptor("DA_D1").exposure_trace

        # Reset and try with phasic burst
        engine2 = NeurochemicalEngine(dt=0.1, seed=42, use_lambda_loc_routing=True)
        da_state2 = NeurotransmitterState(C_tonic=0.4, C_phasic=0.5, F=0.0, eta_u=0.0)
        engine2.add_neurotransmitter("DA", initial_state=da_state2, config={"C_baseline": 0.4})
        receptor2 = ReceptorState(receptor_id="DA_D1", lambda_loc=1.0)
        engine2.add_receptor("DA_D1", initial_state=receptor2, config={"K_d": 0.5})
        engine2.step()
        trace_with_phasic = engine2.registry.get_receptor("DA_D1").exposure_trace

        # Extrasynaptic receptor should see the same concentration in both cases
        # (lambda=1.0 → only C_tonic = 0.4), so exposure traces should match
        assert trace_no_phasic == pytest.approx(trace_with_phasic, rel=0.01)
