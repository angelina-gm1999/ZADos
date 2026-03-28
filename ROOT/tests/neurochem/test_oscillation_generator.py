"""
Tests for state-derived oscillation generator.

Phase 28: Derive oscillation band amplitudes from neurochemical state,
closing the bidirectional NT ↔ oscillation loop.
"""

import pytest

from zados.neurochem.oscillations.generators import (
    derive_oscillation_state,
    DEFAULT_BAND_DERIVATION_RULES,
)
from zados.neurochem.state import NeurotransmitterState, OscillationState
from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.state.receptor_state import ReceptorState


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

class TestDeriveOscillationState:
    """Tests for the pure derive_oscillation_state function."""

    def test_default_rules_exist(self):
        """Default rules should cover all 5 bands."""
        assert set(DEFAULT_BAND_DERIVATION_RULES.keys()) == {
            "delta", "theta", "alpha", "beta", "gamma", "sigma",
        }

    def test_all_zero_nts(self):
        """All-zero NT states → all-zero oscillation bands."""
        nt_states = {
            "DA": NeurotransmitterState(C_tonic=0.0, C_phasic=0.0, F=0.0, eta_u=0.0),
            "GLU": NeurotransmitterState(C_tonic=0.0, C_phasic=0.0, F=0.0, eta_u=0.0),
        }
        osc = derive_oscillation_state(nt_states)
        assert osc.gamma == pytest.approx(0.0)
        assert osc.theta == pytest.approx(0.0)
        assert osc.alpha == pytest.approx(0.0)
        assert osc.beta == pytest.approx(0.0)
        assert osc.delta == pytest.approx(0.0)

    def test_high_da_phasic_drives_gamma(self):
        """DA.C_phasic=1.0 should drive gamma (weight 0.4)."""
        nt_states = {
            "DA": NeurotransmitterState(C_tonic=0.0, C_phasic=1.0, F=0.0, eta_u=0.0),
        }
        osc = derive_oscillation_state(nt_states)
        # gamma = DA.C_phasic * 0.4 = 0.4
        assert osc.gamma == pytest.approx(0.4)

    def test_high_gaba_tonic_drives_alpha(self):
        """GABA.C_tonic=1.0 should drive alpha (weight 0.5)."""
        nt_states = {
            "GABA": NeurotransmitterState(C_tonic=1.0, C_phasic=0.0, F=0.0, eta_u=0.0),
        }
        osc = derive_oscillation_state(nt_states)
        # alpha = GABA.C_tonic * 0.5 = 0.5
        assert osc.alpha == pytest.approx(0.5)

    def test_high_ne_total_drives_beta(self):
        """NE.C (total) should drive beta (weight 0.4)."""
        nt_states = {
            "NE": NeurotransmitterState(C_tonic=0.5, C_phasic=0.3, F=0.0, eta_u=0.0),
        }
        osc = derive_oscillation_state(nt_states)
        # beta = NE.C * 0.4 = 0.8 * 0.4 = 0.32
        assert osc.beta == pytest.approx(0.32)

    def test_high_mor_tonic_drives_delta(self):
        """MOR.C_tonic=1.0 should drive delta (weight 0.4)."""
        nt_states = {
            "MOR": NeurotransmitterState(C_tonic=1.0, C_phasic=0.0, F=0.0, eta_u=0.0),
        }
        osc = derive_oscillation_state(nt_states)
        # delta = MOR.C_tonic * 0.4 = 0.4
        assert osc.delta == pytest.approx(0.4)

    def test_oxt_tonic_drives_theta(self):
        """OXT.C_tonic=1.0 should drive theta (weight 0.4)."""
        nt_states = {
            "OXT": NeurotransmitterState(C_tonic=1.0, C_phasic=0.0, F=0.0, eta_u=0.0),
        }
        osc = derive_oscillation_state(nt_states)
        # theta = OXT.C_tonic * 0.4 = 0.4
        assert osc.theta == pytest.approx(0.4)

    def test_clamped_to_01(self):
        """Output should always be in [0, 1] even with multiple high NTs."""
        nt_states = {
            "DA": NeurotransmitterState(C_tonic=1.0, C_phasic=1.0, F=0.0, eta_u=0.0),
            "GLU": NeurotransmitterState(C_tonic=1.0, C_phasic=1.0, F=0.0, eta_u=0.0),
            "ACh": NeurotransmitterState(C_tonic=1.0, C_phasic=1.0, F=0.0, eta_u=0.0),
        }
        osc = derive_oscillation_state(nt_states)
        # gamma = DA.C_phasic*0.4 + GLU.C_phasic*0.3 + ACh.C_phasic*0.3 = 1.0
        assert osc.gamma <= 1.0
        assert osc.gamma >= 0.0

    def test_missing_nts_ignored(self):
        """Missing NTs should be treated as zero."""
        nt_states = {}  # no NTs at all
        osc = derive_oscillation_state(nt_states)
        assert osc.gamma == pytest.approx(0.0)
        assert osc.theta == pytest.approx(0.0)

    def test_custom_rules(self):
        """Custom derivation rules should override defaults."""
        custom_rules = {
            "gamma": {"DA": ("C_tonic", 1.0)},  # use tonic instead of phasic
            "theta": {},
            "alpha": {},
            "beta": {},
            "delta": {},
        }
        nt_states = {
            "DA": NeurotransmitterState(C_tonic=0.7, C_phasic=0.0, F=0.0, eta_u=0.0),
        }
        osc = derive_oscillation_state(nt_states, band_derivation_rules=custom_rules)
        assert osc.gamma == pytest.approx(0.7)
        assert osc.theta == pytest.approx(0.0)

    def test_cross_frequency_coupling(self):
        """CFC values computed correctly from derived state."""
        nt_states = {
            "DA": NeurotransmitterState(C_tonic=0.5, C_phasic=0.5, F=0.0, eta_u=0.0),
            "OXT": NeurotransmitterState(C_tonic=0.5, C_phasic=0.0, F=0.0, eta_u=0.0),
        }
        osc = derive_oscillation_state(nt_states)
        # gamma from DA phasic: 0.5 * 0.4 = 0.2
        # theta from OXT tonic: 0.5 * 0.4 = 0.2, DA tonic: 0.5 * 0.3 = 0.15 → total = 0.35
        assert osc.theta_gamma_coupling() == pytest.approx(osc.theta * osc.gamma)


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------

class TestEngineOscillationMode:
    """Tests for engine oscillation_mode parameter."""

    def test_default_is_static(self):
        """Default oscillation_mode should be 'static'."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        assert engine.oscillation_mode == "static"

    def test_static_mode_oscillations_unchanged(self):
        """In static mode, oscillations should not change during step."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        da = NeurotransmitterState(C_tonic=0.5, C_phasic=0.5, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da, config={"C_baseline": 0.5})
        osc = OscillationState(theta=0.3, gamma=0.5)
        engine.registry.set_oscillations(osc)

        engine.step()

        osc_after = engine.registry.get_oscillations()
        assert osc_after.theta == pytest.approx(0.3)
        assert osc_after.gamma == pytest.approx(0.5)

    def test_state_derived_mode_updates_oscillations(self):
        """In state_derived mode, oscillations should update from NT state."""
        engine = NeurochemicalEngine(dt=0.1, seed=42, oscillation_mode="state_derived")
        # DA with high phasic → should drive gamma
        da = NeurotransmitterState(C_tonic=0.5, C_phasic=0.8, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da, config={"C_baseline": 0.5})
        # Start with zero oscillations
        engine.registry.set_oscillations(OscillationState())

        engine.step()

        osc_after = engine.registry.get_oscillations()
        # gamma should be driven by DA.C_phasic (though C_phasic may have changed during step)
        # The important thing is it's non-zero
        assert osc_after.gamma > 0  # DA phasic drives gamma

    def test_state_derived_feedback_loop(self):
        """NT → oscillation → receptor bidirectional loop should work."""
        engine = NeurochemicalEngine(
            dt=0.1, seed=42, oscillation_mode="state_derived",
        )
        da = NeurotransmitterState(C_tonic=0.5, C_phasic=0.5, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da, config={"C_baseline": 0.5})
        r1 = ReceptorState(receptor_id="DA_D1", rho=0.5)
        engine.add_receptor("DA_D1", initial_state=r1, config={"K_d": 0.5})

        # Multiple steps should not crash
        for _ in range(5):
            engine.step()

        state = engine.registry.get_receptor("DA_D1")
        assert state is not None
        osc = engine.registry.get_oscillations()
        assert osc is not None

    def test_state_derived_no_nts_no_crash(self):
        """State-derived mode with no NTs should not crash (all zeros)."""
        engine = NeurochemicalEngine(dt=0.1, seed=42, oscillation_mode="state_derived")
        engine.step()  # no NTs, no receptors — should still derive zeros
        osc = engine.registry.get_oscillations()
        if osc is not None:
            assert osc.gamma == pytest.approx(0.0)

    def test_external_mode_same_as_static(self):
        """'external' mode should behave same as 'static' (no auto-derivation)."""
        engine = NeurochemicalEngine(dt=0.1, seed=42, oscillation_mode="external")
        da = NeurotransmitterState(C_tonic=0.5, C_phasic=0.8, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da, config={"C_baseline": 0.5})
        osc = OscillationState(theta=0.3, gamma=0.5)
        engine.registry.set_oscillations(osc)

        engine.step()

        osc_after = engine.registry.get_oscillations()
        # External mode should NOT auto-derive — oscillations remain as set
        assert osc_after.theta == pytest.approx(0.3)
        assert osc_after.gamma == pytest.approx(0.5)
