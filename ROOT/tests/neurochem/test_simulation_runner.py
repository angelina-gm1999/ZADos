from zados.neurochem.core.simulation import SimulationRunner
import numpy as np


def test_simulation_runs_without_crashing():
    def novelty(t): return 0.2
    def rpe(t): return 0.1
    def osc(t): return {"gamma": 0.5}

    params = {
        "R0": 0.1,
        "beta_nov": 0.4,
        "beta_rew": 0.6,
        "ku0": 0.2,
        "gamma": 0.1,
        "epsilon": 0.05,
        "kd": 0.01,
        "kl": 0.02,
        "alpha": 0.03,
        "C0": 0.5,
        "F0": 0.0,
    }

    sim = SimulationRunner(
        dopamine_params=params,
        novelty_fn=novelty,
        rpe_fn=rpe,
        oscillation_fn=osc,
        dt=0.01,
        T=0.5,
        rng=np.random.default_rng(seed=42),
    )

    sim.run()
    history = sim.get_history()

    assert len(history) > 0
    assert all("C" in step for step in history)
    assert all("F" in step for step in history)
    assert all("t" in step for step in history)
    assert isinstance(history[-1]["C"], float)
    assert 0.0 <= history[-1]["C"] <= 5.0