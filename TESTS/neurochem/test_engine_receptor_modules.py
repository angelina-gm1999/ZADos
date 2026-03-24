"""
Tests for wiring receptor family modules into the engine.

Phase 19: The engine now computes and stores effective signaling (A_ij)
for each receptor after step_receptor_dynamics. When a family module is
registered, it uses the module's per-subtype weights; otherwise it falls
back to the generic proxy.
"""

import pytest

from zados.neurochem.core.engine import NeurochemicalEngine
from zados.neurochem.core.registry import NeurochemicalRegistry
from zados.neurochem.state import NeurotransmitterState, ReceptorState, OscillationState
from zados.neurochem.receptors.dopamine_receptors import DopamineReceptors
from zados.neurochem.receptors.receptor_registry import (
    ReceptorModuleRegistry,
    register_all_receptor_modules,
)
from zados.neurochem.neurotransmitters.configs import register_all_receptor_modules_on_engine


# ---------------------------------------------------------------------------
# Registry effective signaling storage
# ---------------------------------------------------------------------------

class TestRegistryEffectiveSignaling:
    """Tests for set/get/get_all on NeurochemicalRegistry."""

    def test_set_and_get(self):
        reg = NeurochemicalRegistry()
        reg.set_effective_signaling("DA_D1", 0.75)
        assert reg.get_effective_signaling("DA_D1") == pytest.approx(0.75)

    def test_get_default_zero(self):
        reg = NeurochemicalRegistry()
        assert reg.get_effective_signaling("DA_D1") == pytest.approx(0.0)

    def test_get_all(self):
        reg = NeurochemicalRegistry()
        reg.set_effective_signaling("DA_D1", 0.5)
        reg.set_effective_signaling("DA_D2", 0.3)
        result = reg.get_all_effective_signaling()
        assert result == {"DA_D1": pytest.approx(0.5), "DA_D2": pytest.approx(0.3)}

    def test_get_all_returns_copy(self):
        reg = NeurochemicalRegistry()
        reg.set_effective_signaling("DA_D1", 0.5)
        copy = reg.get_all_effective_signaling()
        copy["DA_D1"] = 999.0
        assert reg.get_effective_signaling("DA_D1") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Engine receptor module registration
# ---------------------------------------------------------------------------

class TestEngineReceptorModuleRegistration:
    """Tests for register_receptor_module on the engine."""

    def test_register_receptor_module(self):
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        da_module = DopamineReceptors()
        engine.register_receptor_module(da_module)
        assert "DA" in engine._receptor_modules
        assert engine._receptor_modules["DA"] is da_module

    def test_register_multiple_modules(self):
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        da_module = DopamineReceptors()
        engine.register_receptor_module(da_module)
        assert len(engine._receptor_modules) == 1


# ---------------------------------------------------------------------------
# A_ij computation in _update_receptor
# ---------------------------------------------------------------------------

class TestUpdateReceptorComputesAij:
    """Tests for A_ij computation during engine step."""

    @pytest.fixture
    def engine_with_da(self):
        """Engine with DA NT + DA_D1 receptor + DopamineReceptors module."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        # Register DA with non-zero concentration
        da_state = NeurotransmitterState(C_tonic=0.5, C_phasic=0.2, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da_state, config={"C_baseline": 0.5})
        # Register DA_D1 receptor
        receptor_state = ReceptorState(receptor_id="DA_D1")
        engine.add_receptor("DA_D1", initial_state=receptor_state, config={"K_d": 0.5})
        # Register family module
        engine.register_receptor_module(DopamineReceptors())
        return engine

    def test_aij_computed_after_step(self, engine_with_da):
        """After a step, A_ij should be stored in the registry."""
        engine_with_da.step()
        a_ij = engine_with_da.registry.get_effective_signaling("DA_D1")
        assert a_ij > 0.0

    def test_aij_zero_when_zero_concentration(self):
        """A_ij should be 0 when NT concentration is 0."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        da_state = NeurotransmitterState(C_tonic=0.0, C_phasic=0.0, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da_state, config={"C_baseline": 0.0})
        receptor_state = ReceptorState(receptor_id="DA_D1")
        engine.add_receptor("DA_D1", initial_state=receptor_state, config={"K_d": 0.5})
        engine.register_receptor_module(DopamineReceptors())
        engine.step()
        # Concentration is 0 → saturation = 0 → A_ij = 0
        # (drift may add tiny C_tonic, but phasic = 0 and baseline = 0)
        a_ij = engine.registry.get_effective_signaling("DA_D1")
        assert a_ij == pytest.approx(0.0, abs=0.01)

    def test_aij_reflects_desensitized_state(self):
        """DESENSITIZED state should reduce A_ij via g(chi)=0.5."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        da_state = NeurotransmitterState(C_tonic=0.8, C_phasic=0.0, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da_state, config={"C_baseline": 0.8})

        # Active receptor
        active_state = ReceptorState(receptor_id="DA_D1")
        engine.add_receptor("DA_D1", initial_state=active_state, config={"K_d": 0.5})

        # Desensitized receptor (same rho, sigma but different chi)
        from zados.neurochem.state.receptor_state import ReceptorFunctionalState
        desens_state = ReceptorState(
            receptor_id="DA_D2",
            chi=ReceptorFunctionalState.DESENSITIZED,
            sigma=0.5,  # desensitized sigma
        )
        engine.add_receptor("DA_D2", initial_state=desens_state, config={"K_d": 0.5})

        engine.register_receptor_module(DopamineReceptors())
        engine.step()

        a_d1 = engine.registry.get_effective_signaling("DA_D1")
        a_d2 = engine.registry.get_effective_signaling("DA_D2")
        # D1 (ACTIVE, g=1.0) should have higher A_ij than D2 (DESENSITIZED, g=0.5, sigma=0.5)
        assert a_d1 > a_d2

    def test_aij_uses_weight_from_spec(self, engine_with_da):
        """DA_D2 (weight=0.9) should have lower A_ij than DA_D1 (weight=1.0) all else equal."""
        engine = engine_with_da
        # Add DA_D2 with same state
        receptor_d2 = ReceptorState(receptor_id="DA_D2")
        engine.add_receptor("DA_D2", initial_state=receptor_d2, config={"K_d": 0.5})
        engine.step()

        a_d1 = engine.registry.get_effective_signaling("DA_D1")
        a_d2 = engine.registry.get_effective_signaling("DA_D2")
        # Same conditions except weight: D1=1.0, D2=0.9
        assert a_d1 > a_d2
        assert a_d2 == pytest.approx(a_d1 * 0.9, rel=0.01)

    def test_fallback_proxy_without_module(self):
        """Without a registered module, engine uses proxy (no weight)."""
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        da_state = NeurotransmitterState(C_tonic=0.6, C_phasic=0.0, F=0.0, eta_u=0.0)
        engine.add_neurotransmitter("DA", initial_state=da_state, config={"C_baseline": 0.6})
        receptor_state = ReceptorState(receptor_id="DA_D1")
        engine.add_receptor("DA_D1", initial_state=receptor_state, config={"K_d": 0.5})
        # No register_receptor_module — uses proxy
        engine.step()
        a_ij = engine.registry.get_effective_signaling("DA_D1")
        # Should be > 0 (proxy uses weight=1.0 implicitly via rho*sigma*g*sat)
        assert a_ij > 0.0


# ---------------------------------------------------------------------------
# register_all_receptor_modules_on_engine helper
# ---------------------------------------------------------------------------

class TestRegisterAllHelper:
    """Tests for the configs.py helper."""

    def setup_method(self):
        """Clear the global registry between tests."""
        ReceptorModuleRegistry.clear()

    def test_registers_all_11_modules(self):
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        register_all_receptor_modules_on_engine(engine)
        assert len(engine._receptor_modules) == 11
        # Verify all expected NTs are present
        expected_nts = {"DA", "5HT", "NE", "ACh", "OXT", "MOR", "CB1", "CRH", "GABA", "GLU", "histamine"}
        assert set(engine._receptor_modules.keys()) == expected_nts

    def test_modules_are_functional(self):
        engine = NeurochemicalEngine(dt=0.1, seed=42)
        register_all_receptor_modules_on_engine(engine)
        # Each module should have receptor_specs
        for parent_nt, module in engine._receptor_modules.items():
            assert len(module.receptor_specs) > 0, f"{parent_nt} has no specs"
