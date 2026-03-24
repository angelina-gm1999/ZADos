import pytest
from zados.neurochem.state import OscillationState


def test_oscillation_state_initialization_defaults():
    """Oscillation state initializes with default values."""
    state = OscillationState()
    assert state.delta == 0.0
    assert state.theta == 0.0
    assert state.alpha == 0.0
    assert state.beta == 0.0
    assert state.gamma == 0.0
    assert state.phase["delta"] == 0.0
    assert state.phase["theta"] == 0.0


def test_oscillation_state_initialization_custom():
    """Oscillation state can be initialized with custom values."""
    state = OscillationState(
        delta=0.3,
        theta=0.5,
        alpha=0.2,
        beta=0.4,
        gamma=0.7,
    )
    assert state.delta == 0.3
    assert state.theta == 0.5
    assert state.alpha == 0.2
    assert state.beta == 0.4
    assert state.gamma == 0.7


def test_set_band():
    """Can set band amplitude."""
    state = OscillationState()
    state.set_band("theta", 0.6)
    assert state.theta == 0.6


def test_set_band_clamps_to_bounds():
    """Setting band clamps to [0, 1]."""
    state = OscillationState()
    state.set_band("gamma", 1.5)
    assert state.gamma == 1.0
    
    state.set_band("gamma", -0.3)
    assert state.gamma == 0.0


def test_set_band_invalid_raises():
    """Setting invalid band raises error."""
    state = OscillationState()
    with pytest.raises(ValueError, match="Unknown band"):
        state.set_band("omega", 0.5)


def test_get_band():
    """Can get band amplitude."""
    state = OscillationState(alpha=0.7)
    assert state.get_band("alpha") == 0.7


def test_get_band_invalid_raises():
    """Getting invalid band raises error."""
    state = OscillationState()
    with pytest.raises(ValueError, match="Unknown band"):
        state.get_band("zeta")


def test_set_phase():
    """Can set band phase."""
    state = OscillationState()
    state.set_phase("beta", 3.14)
    assert state.phase["beta"] == pytest.approx(3.14)


def test_set_phase_invalid_raises():
    """Setting phase for invalid band raises error."""
    state = OscillationState()
    with pytest.raises(ValueError, match="Unknown band"):
        state.set_phase("invalid", 1.0)


def test_get_phase():
    """Can get band phase."""
    state = OscillationState()
    state.phase["delta"] = 2.5
    assert state.get_phase("delta") == pytest.approx(2.5)


def test_get_phase_invalid_raises():
    """Getting phase for invalid band raises error."""
    state = OscillationState()
    with pytest.raises(ValueError, match="Unknown band"):
        state.get_phase("invalid")


def test_theta_gamma_coupling():
    """Theta-gamma coupling is product of amplitudes."""
    state = OscillationState(theta=0.6, gamma=0.8)
    assert state.theta_gamma_coupling() == pytest.approx(0.48)


def test_alpha_beta_coupling():
    """Alpha-beta coupling is product of amplitudes."""
    state = OscillationState(alpha=0.5, beta=0.4)
    assert state.alpha_beta_coupling() == pytest.approx(0.2)


def test_normalize():
    """Normalize makes band amplitudes sum to 1."""
    state = OscillationState(delta=0.2, theta=0.3, alpha=0.1, beta=0.2, gamma=0.2)
    state.normalize()
    
    total = state.delta + state.theta + state.alpha + state.beta + state.gamma
    assert total == pytest.approx(1.0)
    
    # Check proportions preserved (theta=0.3 was largest)
    assert state.theta > state.delta
    assert state.theta > state.alpha
    assert state.theta > state.beta


def test_normalize_handles_zero_total():
    """Normalize handles all-zero state gracefully."""
    state = OscillationState()
    state.normalize()  # Should not crash
    assert state.delta == 0.0


def test_as_dict():
    """State can be exported to dictionary."""
    state = OscillationState(
        delta=0.2,
        theta=0.5,
        alpha=0.3,
        beta=0.4,
        gamma=0.6,
    )
    state.set_phase("theta", 1.57)
    
    d = state.as_dict()
    
    assert d["delta"] == 0.2
    assert d["theta"] == 0.5
    assert d["alpha"] == 0.3
    assert d["beta"] == 0.4
    assert d["gamma"] == 0.6
    assert d["phase"]["theta"] == pytest.approx(1.57)
    assert d["theta_gamma_coupling"] == pytest.approx(0.3)
    assert d["alpha_beta_coupling"] == pytest.approx(0.12)


def test_from_dict():
    """State can be created from dictionary."""
    data = {
        "delta": 0.3,
        "theta": 0.6,
        "alpha": 0.4,
        "beta": 0.5,
        "gamma": 0.8,
        "phase": {
            "delta": 0.5,
            "theta": 1.2,
            "alpha": 2.1,
            "beta": 0.8,
            "gamma": 3.0,
        },
    }
    state = OscillationState.from_dict(data)
    
    assert state.delta == 0.3
    assert state.theta == 0.6
    assert state.alpha == 0.4
    assert state.beta == 0.5
    assert state.gamma == 0.8
    assert state.phase["theta"] == pytest.approx(1.2)


def test_from_dict_uses_defaults():
    """from_dict uses defaults for missing keys."""
    data = {"theta": 0.7}
    state = OscillationState.from_dict(data)
    
    assert state.theta == 0.7
    assert state.delta == 0.0
    assert state.gamma == 0.0


def test_copy():
    """State can be deep copied."""
    state1 = OscillationState(theta=0.5, gamma=0.8)
    state1.set_phase("theta", 2.0)
    
    state2 = state1.copy()
    
    assert state2.theta == 0.5
    assert state2.gamma == 0.8
    assert state2.phase["theta"] == pytest.approx(2.0)
    
    # Verify independence
    state2.set_band("theta", 0.9)
    assert state1.theta == 0.5
    assert state2.theta == 0.9


def test_post_init_enforces_bounds():
    """__post_init__ enforces bounds on initialization."""
    state = OscillationState(delta=-0.5, theta=1.5, alpha=2.0, beta=-1.0, gamma=0.5)
    assert state.delta == 0.0
    assert state.theta == 1.0
    assert state.alpha == 1.0
    assert state.beta == 0.0
    assert state.gamma == 0.5


def test_bands_class_method():
    """bands() returns list of all band names."""
    bands = OscillationState.bands()
    assert bands == ["delta", "theta", "alpha", "beta", "gamma", "sigma"]