from __future__ import annotations
from typing import Callable, Dict, List
import numpy as np

from zados.neurochem.neurotransmitters.dopamine import Dopamine


class SimulationRunner:
    """
    Runs the core simulation loop for neuromodulator dynamics.
    Currently supports only Dopamine.
    """

    def __init__(
        self,
        dopamine_params: dict,
        novelty_fn: Callable[[float], float],
        rpe_fn: Callable[[float], float],
        oscillation_fn: Callable[[float], dict],
        dt: float = 0.01,
        T: float = 10.0,
        rng: np.random.Generator = None,
    ):
        self.dt = dt
        self.T = T
        self.timesteps = int(T / dt)
        self.time = np.linspace(0, T, self.timesteps)

        self.dopamine = Dopamine(dopamine_params, rng=rng or np.random.default_rng())

        self.novelty_fn = novelty_fn
        self.rpe_fn = rpe_fn
        self.osc_fn = oscillation_fn

        self.history: List[Dict[str, float]] = []

    def run(self):
        for t in self.time:
            novelty = self.novelty_fn(t)
            rpe = self.rpe_fn(t)
            oscillations = self.osc_fn(t)

            self.dopamine.step(novelty, rpe, self.dt, oscillations)
            self.history.append({
                "t": t,
                **self.dopamine.state()
            })

    def get_history(self) -> List[Dict[str, float]]:
        return self.history
