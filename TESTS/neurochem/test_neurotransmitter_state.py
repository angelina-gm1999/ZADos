import pytest
from zados.neurochem.state import NeurotransmitterState


def test_state_initialization_defaults():
    """State initializes with default values."""
    state = NeurotransmitterState()
    assert state.C_tonic == 0.5
    assert state.C_phasic == 0.0
    assert state.F == 0.0
    assert state.eta_u == 1.0


def test_state_initialization_custom():
    """State can be initialized with custom values."""
    state = NeurotransmitterState(C_tonic=0.8, C_phasic=0.2, F=0.3, eta_u=0.9)
    assert state.C_tonic == 0.8
    assert state.C_phasic == 0.2
    assert state.F == 0.3
    assert state.eta_u == 0.9


def test_total_concentration():
    """Total concentration is sum of tonic and phasic."""
    state = NeurotransmitterState(C_tonic=0.6, C_phasic=0.3)
    assert state.C == pytest.approx(0.9)


def test_update_concentration_tonic():
    """Can update tonic concentration."""
    state = NeurotransmitterState(C_tonic=0.5)
    state.update_concentration(0.2, is_phasic=False)
    assert state.C_tonic == pytest.approx(0.7)
    assert state.C_phasic == 0.0


def test_update_concentration_phasic():
    """Can update phasic concentration."""
    state = NeurotransmitterState(C_phasic=0.1)
    state.update_concentration(0.3, is_phasic=True)
    assert state.C_phasic == pytest.approx(0.4)


def test_update_concentration_enforces_nonnegativity():
    """Concentration cannot go negative."""
    state = NeurotransmitterState(C_tonic=0.3)
    state.update_concentration(-0.5, is_phasic=False)
    assert state.C_tonic == 0.0


def test_set_concentration():
    """Can set total concentration (goes to tonic, zeros phasic)."""
    state = NeurotransmitterState(C_tonic=0.3, C_phasic=0.2)
    state.set_concentration(0.8)
    assert state.C_tonic == 0.8
    assert state.C_phasic == 0.0


def test_update_fatigue():
    """Can update fatigue level."""
    state = NeurotransmitterState(F=0.2)
    state.update_fatigue(0.3)
    assert state.F == pytest.approx(0.5)


def test_fatigue_bounds():
    """Fatigue is clamped to [0, 1]."""
    state = NeurotransmitterState(F=0.8)
    state.update_fatigue(0.5)
    assert state.F == 1.0
    
    state.update_fatigue(-2.0)
    assert state.F == 0.0


def test_update_transporter_efficiency():
    """Can update transporter efficiency."""
    state = NeurotransmitterState(eta_u=0.9)
    state.update_transporter_efficiency(-0.2)
    assert state.eta_u == pytest.approx(0.7)


def test_transporter_efficiency_bounds():
    """Transporter efficiency is clamped to [0, 1]."""
    state = NeurotransmitterState(eta_u=0.3)
    state.update_transporter_efficiency(-0.5)
    assert state.eta_u == 0.0
    
    state.update_transporter_efficiency(1.5)
    assert state.eta_u == 1.0


def test_as_dict():
    """State can be exported to dictionary."""
    state = NeurotransmitterState(C_tonic=0.6, C_phasic=0.2, F=0.4, eta_u=0.8)
    d = state.as_dict()
    
    assert d["C_tonic"] == 0.6
    assert d["C_phasic"] == 0.2
    assert d["C"] == pytest.approx(0.8)
    assert d["F"] == 0.4
    assert d["eta_u"] == 0.8


def test_from_dict():
    """State can be created from dictionary."""
    data = {"C_tonic": 0.7, "C_phasic": 0.1, "F": 0.3, "eta_u": 0.9}
    state = NeurotransmitterState.from_dict(data)
    
    assert state.C_tonic == 0.7
    assert state.C_phasic == 0.1
    assert state.F == 0.3
    assert state.eta_u == 0.9


def test_from_dict_uses_defaults():
    """from_dict uses defaults for missing keys."""
    data = {"C_tonic": 0.9}
    state = NeurotransmitterState.from_dict(data)
    
    assert state.C_tonic == 0.9
    assert state.C_phasic == 0.0
    assert state.F == 0.0
    assert state.eta_u == 1.0


def test_copy():
    """State can be deep copied."""
    state1 = NeurotransmitterState(C_tonic=0.7, F=0.3)
    state2 = state1.copy()
    
    assert state2.C_tonic == 0.7
    assert state2.F == 0.3
    
    # Verify independence
    state2.update_concentration(0.1)
    assert state1.C_tonic == 0.7
    assert state2.C_tonic == pytest.approx(0.8)


def test_post_init_enforces_bounds():
    """__post_init__ enforces bounds on initialization."""
    state = NeurotransmitterState(C_tonic=-0.5, F=1.5, eta_u=-0.2)
    assert state.C_tonic == 0.0
    assert state.F == 1.0
    assert state.eta_u == 0.0