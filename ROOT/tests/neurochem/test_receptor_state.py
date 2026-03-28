import pytest
import math
from zados.neurochem.state import ReceptorState, ReceptorFunctionalState


def test_receptor_state_initialization_defaults():
    """Receptor state initializes with default values."""
    state = ReceptorState(receptor_id="DA_D1")
    assert state.receptor_id == "DA_D1"
    assert state.rho == 1.0
    assert state.sigma == 1.0
    assert state.lambda_loc == 0.5
    assert state.gamma_gprotein == 1.0
    assert state.chi == ReceptorFunctionalState.ACTIVE
    assert state.exposure_trace == 0.0
    assert state.time_in_state == 0.0


def test_receptor_state_initialization_custom():
    """Receptor state can be initialized with custom values."""
    state = ReceptorState(
        receptor_id="5HT_2A",
        rho=0.8,
        sigma=0.9,
        lambda_loc=0.7,
        gamma_gprotein=0.85,
        chi=ReceptorFunctionalState.DESENSITIZED,
        exposure_trace=5.0,
        time_in_state=10.0,
    )
    assert state.receptor_id == "5HT_2A"
    assert state.rho == 0.8
    assert state.sigma == 0.9
    assert state.lambda_loc == 0.7
    assert state.gamma_gprotein == 0.85
    assert state.chi == ReceptorFunctionalState.DESENSITIZED
    assert state.exposure_trace == 5.0
    assert state.time_in_state == 10.0


def test_update_density():
    """Can update receptor density."""
    state = ReceptorState(receptor_id="DA_D2", rho=0.6)
    state.update_density(0.2)
    assert state.rho == pytest.approx(0.8)


def test_density_bounds():
    """Density is clamped to [0, 1]."""
    state = ReceptorState(receptor_id="DA_D3", rho=0.9)
    state.update_density(0.3)
    assert state.rho == 1.0
    
    state.update_density(-1.5)
    assert state.rho == 0.0


def test_update_sensitivity():
    """Can update receptor sensitivity."""
    state = ReceptorState(receptor_id="DA_D1", sigma=0.7)
    state.update_sensitivity(0.1)
    assert state.sigma == pytest.approx(0.8)


def test_sensitivity_bounds():
    """Sensitivity is clamped to [0, 1]."""
    state = ReceptorState(receptor_id="DA_D1", sigma=0.3)
    state.update_sensitivity(-0.5)
    assert state.sigma == 0.0
    
    state.update_sensitivity(1.5)
    assert state.sigma == 1.0


def test_set_functional_state():
    """Can transition to a new functional state."""
    state = ReceptorState(receptor_id="DA_D1", chi=ReceptorFunctionalState.ACTIVE)
    state.set_functional_state(ReceptorFunctionalState.DESENSITIZED)
    
    assert state.chi == ReceptorFunctionalState.DESENSITIZED
    assert state.time_in_state == 0.0


def test_set_functional_state_resets_timer():
    """Transitioning to new state resets time_in_state."""
    state = ReceptorState(receptor_id="DA_D1", time_in_state=50.0)
    state.set_functional_state(ReceptorFunctionalState.INTERNALIZED)
    assert state.time_in_state == 0.0

def test_set_functional_state_same_state():
    """Setting same state does not reset timer."""
    state = ReceptorState(
        receptor_id="DA_D1",
        chi=ReceptorFunctionalState.ACTIVE,
        time_in_state=25.0
    )
    state.set_functional_state(ReceptorFunctionalState.ACTIVE)
    assert state.time_in_state == 25.0


def test_update_exposure_trace():
    """Exposure trace updates with decay and accumulation."""
    state = ReceptorState(receptor_id="DA_D1", exposure_trace=10.0)
    
    saturation = 0.8
    dt = 1.0
    tau = 10.0
    
    state.update_exposure_trace(saturation, dt, tau)
    
    expected = 10.0 * math.exp(-dt / tau) + saturation * dt
    assert state.exposure_trace == pytest.approx(expected)


def test_increment_time_in_state():
    """Can increment time spent in current state."""
    state = ReceptorState(receptor_id="DA_D1", time_in_state=5.0)
    state.increment_time_in_state(2.5)
    assert state.time_in_state == pytest.approx(7.5)


def test_saturation():
    """Saturation function computes correctly."""
    state = ReceptorState(receptor_id="DA_D1")
    
    # Half-saturation: C = K_d
    assert state.saturation(concentration=0.5, K_d=0.5) == pytest.approx(0.5)
    
    # Low concentration
    assert state.saturation(concentration=0.1, K_d=1.0) == pytest.approx(0.1 / 1.1)
    
    # High concentration (near saturation)
    assert state.saturation(concentration=10.0, K_d=1.0) == pytest.approx(10.0 / 11.0)


def test_as_dict():
    """State can be exported to dictionary."""
    state = ReceptorState(
        receptor_id="DA_D2",
        rho=0.7,
        sigma=0.8,
        lambda_loc=0.6,
        gamma_gprotein=0.9,
        chi=ReceptorFunctionalState.DESENSITIZED,
        exposure_trace=3.5,
        time_in_state=12.0,
    )
    d = state.as_dict()
    
    assert d["receptor_id"] == "DA_D2"
    assert d["rho"] == 0.7
    assert d["sigma"] == 0.8
    assert d["lambda_loc"] == 0.6
    assert d["gamma_gprotein"] == 0.9
    assert d["chi"] == "desensitized"
    assert d["exposure_trace"] == 3.5
    assert d["time_in_state"] == 12.0


def test_from_dict():
    """State can be created from dictionary."""
    data = {
        "receptor_id": "5HT_1A",
        "rho": 0.85,
        "sigma": 0.75,
        "lambda_loc": 0.4,
        "gamma_gprotein": 0.95,
        "chi": "internalized",
        "exposure_trace": 8.0,
        "time_in_state": 20.0,
    }
    state = ReceptorState.from_dict(data)
    
    assert state.receptor_id == "5HT_1A"
    assert state.rho == 0.85
    assert state.sigma == 0.75
    assert state.lambda_loc == 0.4
    assert state.gamma_gprotein == 0.95
    assert state.chi == ReceptorFunctionalState.INTERNALIZED
    assert state.exposure_trace == 8.0
    assert state.time_in_state == 20.0


def test_from_dict_uses_defaults():
    """from_dict uses defaults for missing keys."""
    data = {"receptor_id": "NE_alpha1"}
    state = ReceptorState.from_dict(data)
    
    assert state.receptor_id == "NE_alpha1"
    assert state.rho == 1.0
    assert state.sigma == 1.0
    assert state.chi == ReceptorFunctionalState.ACTIVE


def test_copy():
    """State can be deep copied."""
    state1 = ReceptorState(
        receptor_id="DA_D1",
        rho=0.6,
        chi=ReceptorFunctionalState.DESENSITIZED
    )
    state2 = state1.copy()
    
    assert state2.receptor_id == "DA_D1"
    assert state2.rho == 0.6
    assert state2.chi == ReceptorFunctionalState.DESENSITIZED
    
    # Verify independence
    state2.update_density(0.2)
    assert state1.rho == 0.6
    assert state2.rho == pytest.approx(0.8)


def test_post_init_enforces_bounds():
    """__post_init__ enforces bounds on initialization."""
    state = ReceptorState(
        receptor_id="test",
        rho=-0.5,
        sigma=1.5,
        lambda_loc=2.0,
        gamma_gprotein=-0.2,
        exposure_trace=-5.0,
        time_in_state=-1.0,
    )
    assert state.rho == 0.0
    assert state.sigma == 1.0
    assert state.lambda_loc == 1.0
    assert state.gamma_gprotein == 0.0
    assert state.exposure_trace == 0.0
    assert state.time_in_state == 0.0


def test_functional_state_enum():
    """ReceptorFunctionalState enum has correct values."""
    assert ReceptorFunctionalState.ACTIVE.value == "active"
    assert ReceptorFunctionalState.DESENSITIZED.value == "desensitized"
    assert ReceptorFunctionalState.INTERNALIZED.value == "internalized"
    assert ReceptorFunctionalState.UPREGULATED.value == "upregulated"