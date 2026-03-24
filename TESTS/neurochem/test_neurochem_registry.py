import pytest
from zados.neurochem.core.registry import NeurochemicalRegistry


def test_registry_init():
    """Registry initializes empty."""
    reg = NeurochemicalRegistry()
    assert reg.neurotransmitter_names() == []
    assert reg.receptor_ids() == []
    assert reg.get_oscillations() is None


def test_register_neurotransmitter():
    """Can register and retrieve neurotransmitters."""
    reg = NeurochemicalRegistry()
    
    state_mock = {"C": 0.5, "F": 0.0}
    config_mock = {"ku0": 0.1, "R0": 0.2}
    
    reg.register_neurotransmitter("dopamine", state_mock, config_mock)
    
    assert "dopamine" in reg.neurotransmitter_names()
    assert reg.get_neurotransmitter("dopamine") == state_mock
    assert reg.get_config("dopamine") == config_mock


def test_register_receptor():
    """Can register and retrieve receptors."""
    reg = NeurochemicalRegistry()
    
    receptor_mock = {"rho": 1.0, "state": "active"}
    reg.register_receptor("DA_D1", receptor_mock)
    
    assert "DA_D1" in reg.receptor_ids()
    assert reg.get_receptor("DA_D1") == receptor_mock


def test_set_oscillations():
    """Can set and retrieve oscillation state."""
    reg = NeurochemicalRegistry()
    
    osc_mock = {"gamma": 0.7, "theta": 0.3}
    reg.set_oscillations(osc_mock)
    
    assert reg.get_oscillations() == osc_mock


def test_iter_neurotransmitters():
    """Can iterate over all neurotransmitters."""
    reg = NeurochemicalRegistry()
    
    reg.register_neurotransmitter("dopamine", {"C": 0.5})
    reg.register_neurotransmitter("serotonin", {"C": 0.3})
    
    names = [name for name, _ in reg.iter_neurotransmitters()]
    assert set(names) == {"dopamine", "serotonin"}


def test_get_missing_neurotransmitter_raises():
    """Getting unregistered transmitter raises KeyError."""
    reg = NeurochemicalRegistry()
    
    with pytest.raises(KeyError, match="not registered"):
        reg.get_neurotransmitter("nonexistent")


def test_get_missing_config_raises():
    """Getting config for unregistered transmitter raises KeyError."""
    reg = NeurochemicalRegistry()
    
    with pytest.raises(KeyError, match="not found"):
        reg.get_config("nonexistent")