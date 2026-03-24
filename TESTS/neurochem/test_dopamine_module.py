import numpy as np
import pytest
from zados.neurochem.neurotransmitters.dopamine import Dopamine

# Mock modulate_parameters (or use the real one if implemented)
from zados.neurochem.oscillations import modulation_links

def dummy_modulate_parameters(params, oscillations):
    # Return parameters unchanged for testing (unless you want to mock actual modulation)
    return params

# Save the original function
_original_modulate_parameters = modulation_links.modulate_parameters


@pytest.fixture(autouse=True)
def mock_modulate_parameters():
    """Mock modulate_parameters for dopamine tests, restore after."""
    modulation_links.modulate_parameters = dummy_modulate_parameters
    yield
    modulation_links.modulate_parameters = _original_modulate_parameters


def test_dopamine_basic_step():
    params = {
        "R0": 0.1,
        "beta_nov": 0.5,
        "beta_rew": 0.8,
        "ku0": 0.2,
        "gamma": 0.1,
        "epsilon": 0.05,
        "kd": 0.01,
        "kl": 0.02,
        "alpha": 0.03,
        "C0": 0.4,
        "F0": 0.1,
    }

    da = Dopamine(params=params, rng=np.random.default_rng(seed=42))

    novelty = 0.7
    rpe = 0.5
    dt = 0.01
    oscillations = {"gamma": 0.6}

    prev_state = da.state().copy()
    updated_c = da.step(novelty, rpe, dt, oscillations)
    state = da.state()

    assert isinstance(updated_c, float)
    assert updated_c >= 0.0
    assert state["C"] != prev_state["C"] or state["F"] != prev_state["F"]


def test_dopamine_zero_inputs():
    params = {
        "R0": 0.0,
        "beta_nov": 0.0,
        "beta_rew": 0.0,
        "ku0": 0.2,
        "gamma": 0.1,
        "epsilon": 0.05,
        "kd": 0.01,
        "kl": 0.02,
        "alpha": 0.0,  # no noise
        "C0": 0.4,
        "F0": 0.0,
    }

    da = Dopamine(params=params)

    novelty = 0.0
    rpe = 0.0
    dt = 0.01
    oscillations = {}

    c_before = da.C
    da.step(novelty, rpe, dt, oscillations)
    c_after = da.C

    # With no input, C should decrease due to reuptake, diffusion, degradation
    assert c_after < c_before


def test_dopamine_noise_effect():
    params = {
        "R0": 0.1,
        "beta_nov": 0.0,
        "beta_rew": 0.0,
        "ku0": 0.2,
        "gamma": 0.1,
        "epsilon": 0.01,
        "kd": 0.0,
        "kl": 0.0,
        "alpha": 0.5,  # high noise
        "C0": 0.5,
        "F0": 0.0,
    }

    da = Dopamine(params=params, rng=np.random.default_rng(seed=123))
    novelty = 0.0
    rpe = 0.0
    dt = 0.01
    oscillations = {}

    values = []
    for _ in range(5):
        values.append(da.step(novelty, rpe, dt, oscillations))

    assert any(abs(values[i] - values[i - 1]) > 1e-4 for i in range(1, len(values)))
