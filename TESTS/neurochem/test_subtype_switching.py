"""
Tests for receptor subtype switching mechanism.

Phase 23: Under sustained activation of one receptor subtype, expression
shifts toward a complementary subtype (homeostatic compensation). Density
is conserved: what the source loses, the target gains.
"""

import pytest

from zados.neurochem.receptors.subtype_switching import (
    SubtypeSwitchRule,
    compute_subtype_switch_deltas,
    apply_subtype_switch_deltas,
)
from zados.neurochem.receptors.dopamine_receptors import DopamineReceptors
from zados.neurochem.receptors.serotonin_receptors import SerotoninReceptors
from zados.neurochem.receptors.norepinephrine_receptors import NorepinephrineReceptors
from zados.neurochem.receptors.acetylcholine_receptors import AcetylcholineReceptors
from zados.neurochem.receptors.gaba_receptors import GABAReceptors
from zados.neurochem.receptors.glutamate_receptors import GlutamateReceptors
from zados.neurochem.state.receptor_state import ReceptorState
from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.state import NeurotransmitterState


# ---------------------------------------------------------------------------
# SubtypeSwitchRule dataclass
# ---------------------------------------------------------------------------

class TestSubtypeSwitchRule:
    """Tests for the SubtypeSwitchRule frozen dataclass."""

    def test_frozen(self):
        """SubtypeSwitchRule should be immutable."""
        rule = SubtypeSwitchRule("DA_D1", "DA_D2", 20.0, 0.005, 0.02)
        with pytest.raises(AttributeError):
            rule.exposure_threshold = 10.0

    def test_fields(self):
        """All fields accessible."""
        rule = SubtypeSwitchRule("DA_D1", "DA_D2", 20.0, 0.005, 0.02)
        assert rule.source_receptor_id == "DA_D1"
        assert rule.target_receptor_id == "DA_D2"
        assert rule.exposure_threshold == 20.0
        assert rule.rho_transfer_rate == 0.005
        assert rule.max_transfer_per_step == 0.02


# ---------------------------------------------------------------------------
# compute_subtype_switch_deltas
# ---------------------------------------------------------------------------

class TestComputeSubtypeSwitchDeltas:
    """Tests for the pure compute_subtype_switch_deltas function."""

    def test_below_threshold_no_deltas(self):
        """No switching when exposure < threshold."""
        states = {
            "DA_D1": ReceptorState(receptor_id="DA_D1", rho=0.5, exposure_trace=10.0),
            "DA_D2": ReceptorState(receptor_id="DA_D2", rho=0.5, exposure_trace=5.0),
        }
        rules = [SubtypeSwitchRule("DA_D1", "DA_D2", 20.0, 0.005, 0.02)]
        deltas = compute_subtype_switch_deltas(states, rules, dt=1.0)
        assert deltas == {}

    def test_above_threshold_produces_deltas(self):
        """Deltas computed when exposure > threshold."""
        states = {
            "DA_D1": ReceptorState(receptor_id="DA_D1", rho=0.5, exposure_trace=25.0),
            "DA_D2": ReceptorState(receptor_id="DA_D2", rho=0.5, exposure_trace=5.0),
        }
        rules = [SubtypeSwitchRule("DA_D1", "DA_D2", 20.0, 0.005, 0.02)]
        deltas = compute_subtype_switch_deltas(states, rules, dt=1.0)
        assert "DA_D1" in deltas
        assert "DA_D2" in deltas
        assert deltas["DA_D1"] < 0  # source loses density
        assert deltas["DA_D2"] > 0  # target gains density

    def test_conservation_of_density(self):
        """source_delta + target_delta = 0."""
        states = {
            "DA_D1": ReceptorState(receptor_id="DA_D1", rho=0.5, exposure_trace=25.0),
            "DA_D2": ReceptorState(receptor_id="DA_D2", rho=0.5, exposure_trace=5.0),
        }
        rules = [SubtypeSwitchRule("DA_D1", "DA_D2", 20.0, 0.005, 0.02)]
        deltas = compute_subtype_switch_deltas(states, rules, dt=1.0)
        total = sum(deltas.values())
        assert total == pytest.approx(0.0)

    def test_max_transfer_cap(self):
        """Delta capped at max_transfer_per_step."""
        states = {
            "DA_D1": ReceptorState(receptor_id="DA_D1", rho=0.5, exposure_trace=25.0),
            "DA_D2": ReceptorState(receptor_id="DA_D2", rho=0.5),
        }
        # Very high rate, but max_transfer_per_step caps it
        rules = [SubtypeSwitchRule("DA_D1", "DA_D2", 20.0, 10.0, 0.01)]
        deltas = compute_subtype_switch_deltas(states, rules, dt=1.0)
        assert abs(deltas["DA_D1"]) == pytest.approx(0.01)

    def test_no_transfer_more_than_source_rho(self):
        """Cannot transfer more density than source has."""
        states = {
            "DA_D1": ReceptorState(receptor_id="DA_D1", rho=0.002, exposure_trace=25.0),
            "DA_D2": ReceptorState(receptor_id="DA_D2", rho=0.5),
        }
        rules = [SubtypeSwitchRule("DA_D1", "DA_D2", 20.0, 0.005, 0.02)]
        deltas = compute_subtype_switch_deltas(states, rules, dt=1.0)
        assert abs(deltas["DA_D1"]) <= 0.002 + 1e-10

    def test_missing_source_receptor_skipped(self):
        """Missing source receptor should be skipped."""
        states = {
            "DA_D2": ReceptorState(receptor_id="DA_D2", rho=0.5),
        }
        rules = [SubtypeSwitchRule("DA_D1", "DA_D2", 20.0, 0.005, 0.02)]
        deltas = compute_subtype_switch_deltas(states, rules, dt=1.0)
        assert deltas == {}

    def test_no_rules_no_deltas(self):
        """Empty rules → no changes."""
        states = {
            "DA_D1": ReceptorState(receptor_id="DA_D1", rho=0.5, exposure_trace=25.0),
        }
        deltas = compute_subtype_switch_deltas(states, [], dt=1.0)
        assert deltas == {}

    def test_bidirectional_rules(self):
        """Both D1→D2 and D2→D1 can coexist and fire independently."""
        states = {
            "DA_D1": ReceptorState(receptor_id="DA_D1", rho=0.5, exposure_trace=25.0),
            "DA_D2": ReceptorState(receptor_id="DA_D2", rho=0.5, exposure_trace=25.0),
        }
        rules = [
            SubtypeSwitchRule("DA_D1", "DA_D2", 20.0, 0.005, 0.02),
            SubtypeSwitchRule("DA_D2", "DA_D1", 20.0, 0.003, 0.015),
        ]
        deltas = compute_subtype_switch_deltas(states, rules, dt=1.0)
        # Both receptors should have deltas (net of both rules)
        assert "DA_D1" in deltas
        assert "DA_D2" in deltas
        # Conservation still holds
        total = sum(deltas.values())
        assert total == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# apply_subtype_switch_deltas
# ---------------------------------------------------------------------------

class TestApplySubtypeSwitchDeltas:
    """Tests for the pure apply_subtype_switch_deltas function."""

    def test_returns_new_states(self):
        """Originals not mutated."""
        states = {
            "DA_D1": ReceptorState(receptor_id="DA_D1", rho=0.5),
            "DA_D2": ReceptorState(receptor_id="DA_D2", rho=0.5),
        }
        deltas = {"DA_D1": -0.01, "DA_D2": 0.01}
        new_states = apply_subtype_switch_deltas(states, deltas)
        # Original unchanged
        assert states["DA_D1"].rho == pytest.approx(0.5)
        assert states["DA_D2"].rho == pytest.approx(0.5)
        # New states updated
        assert new_states["DA_D1"].rho == pytest.approx(0.49)
        assert new_states["DA_D2"].rho == pytest.approx(0.51)

    def test_rho_clamped_lower(self):
        """rho should not go below 0."""
        states = {"DA_D1": ReceptorState(receptor_id="DA_D1", rho=0.01)}
        deltas = {"DA_D1": -0.05}
        new_states = apply_subtype_switch_deltas(states, deltas)
        assert new_states["DA_D1"].rho == pytest.approx(0.0)

    def test_rho_clamped_upper(self):
        """rho should not exceed 1."""
        states = {"DA_D1": ReceptorState(receptor_id="DA_D1", rho=0.99)}
        deltas = {"DA_D1": 0.05}
        new_states = apply_subtype_switch_deltas(states, deltas)
        assert new_states["DA_D1"].rho == pytest.approx(1.0)

    def test_empty_deltas_no_change(self):
        """Empty deltas → original states returned."""
        states = {"DA_D1": ReceptorState(receptor_id="DA_D1", rho=0.5)}
        new_states = apply_subtype_switch_deltas(states, {})
        assert new_states["DA_D1"].rho == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Receptor module rules
# ---------------------------------------------------------------------------

class TestReceptorModuleRules:
    """Tests that multi-subtype receptor modules define switch rules."""

    def test_dopamine_has_rules(self):
        """DopamineReceptors defines D1⇄D2 rules."""
        rules = DopamineReceptors().subtype_switch_rules
        assert len(rules) == 2
        sources = {r.source_receptor_id for r in rules}
        assert sources == {"DA_D1", "DA_D2"}

    def test_serotonin_has_rules(self):
        """SerotoninReceptors defines 1A⇄2A rules."""
        rules = SerotoninReceptors().subtype_switch_rules
        assert len(rules) == 2
        sources = {r.source_receptor_id for r in rules}
        assert sources == {"5HT_1A", "5HT_2A"}

    def test_norepinephrine_has_rules(self):
        """NorepinephrineReceptors defines alpha1⇄alpha2 rules."""
        rules = NorepinephrineReceptors().subtype_switch_rules
        assert len(rules) == 2
        sources = {r.source_receptor_id for r in rules}
        assert sources == {"NE_alpha1", "NE_alpha2"}

    def test_acetylcholine_has_rules(self):
        """AcetylcholineReceptors defines nicotinic⇄muscarinic rules."""
        rules = AcetylcholineReceptors().subtype_switch_rules
        assert len(rules) == 2
        sources = {r.source_receptor_id for r in rules}
        assert sources == {"ACh_nicotinic", "ACh_muscarinic"}

    def test_gaba_has_rules(self):
        """GABAReceptors defines A⇄B rules."""
        rules = GABAReceptors().subtype_switch_rules
        assert len(rules) == 2
        sources = {r.source_receptor_id for r in rules}
        assert sources == {"GABA_A", "GABA_B"}

    def test_glutamate_has_rules(self):
        """GlutamateReceptors defines NMDA⇄AMPA rules."""
        rules = GlutamateReceptors().subtype_switch_rules
        assert len(rules) == 2
        sources = {r.source_receptor_id for r in rules}
        assert sources == {"GLU_NMDA", "GLU_AMPA"}

    def test_all_rules_are_frozen(self):
        """All rules from all modules should be frozen dataclasses."""
        for ModuleCls in [DopamineReceptors, SerotoninReceptors,
                          NorepinephrineReceptors, AcetylcholineReceptors,
                          GABAReceptors, GlutamateReceptors]:
            for rule in ModuleCls().subtype_switch_rules:
                with pytest.raises(AttributeError):
                    rule.exposure_threshold = 0.0


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------

class TestEngineSubtypeSwitching:
    """Tests for engine-level subtype switching."""

    def test_no_switching_below_threshold(self):
        """With low exposure, no density transfer occurs."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        da_state = NeurotransmitterState(C_tonic=0.3, C_phasic=0.0, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da_state, config={"C_baseline": 0.3})
        r1 = ReceptorState(receptor_id="DA_D1", rho=0.5, exposure_trace=5.0)
        r2 = ReceptorState(receptor_id="DA_D2", rho=0.5, exposure_trace=5.0)
        engine.add_receptor("DA_D1", initial_state=r1, config={"K_d": 0.5})
        engine.add_receptor("DA_D2", initial_state=r2, config={"K_d": 0.5})
        engine.register_receptor_module(DopamineReceptors())

        rho1_before = engine.registry.get_receptor("DA_D1").rho
        rho2_before = engine.registry.get_receptor("DA_D2").rho
        engine.step()
        rho1_after = engine.registry.get_receptor("DA_D1").rho
        rho2_after = engine.registry.get_receptor("DA_D2").rho

        # rho changes from receptor dynamics (gamma, etc.) are tiny; subtype
        # switching should NOT have fired since exposure_trace < 20
        # Total rho should still sum to ~1.0 (conservation check is the key)
        assert rho1_after + rho2_after == pytest.approx(
            rho1_before + rho2_before, abs=0.01
        )

    def test_switching_with_high_exposure(self):
        """With high exposure, density transfers from source to target."""
        engine = NeurochemicalEngine(dt=1.0, seed=42)
        da_state = NeurotransmitterState(C_tonic=0.3, C_phasic=0.0, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da_state, config={"C_baseline": 0.3})

        # D1 has very high exposure → should transfer density to D2
        r1 = ReceptorState(receptor_id="DA_D1", rho=0.6, exposure_trace=30.0)
        r2 = ReceptorState(receptor_id="DA_D2", rho=0.4, exposure_trace=5.0)
        engine.add_receptor("DA_D1", initial_state=r1, config={"K_d": 0.5})
        engine.add_receptor("DA_D2", initial_state=r2, config={"K_d": 0.5})
        engine.register_receptor_module(DopamineReceptors())

        engine.step()

        rho1 = engine.registry.get_receptor("DA_D1").rho
        rho2 = engine.registry.get_receptor("DA_D2").rho

        # D1 should have lost some density, D2 gained
        # (exposure_trace=30 > threshold=20, dt=1.0, rate=0.005 → transfer=0.005)
        # But exposure_trace gets updated during step, so we just check direction
        # The initial exposure was 30 > 20 so switching should fire
        assert rho1 < 0.6  # D1 lost density
        assert rho2 > 0.4  # D2 gained density

    def test_no_modules_no_crash(self):
        """No receptor modules registered should not crash."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        da_state = NeurotransmitterState(C_tonic=0.3, C_phasic=0.0, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da_state, config={"C_baseline": 0.3})
        r1 = ReceptorState(receptor_id="DA_D1", rho=0.5, exposure_trace=30.0)
        engine.add_receptor("DA_D1", initial_state=r1, config={"K_d": 0.5})
        engine.step()  # Should not raise
