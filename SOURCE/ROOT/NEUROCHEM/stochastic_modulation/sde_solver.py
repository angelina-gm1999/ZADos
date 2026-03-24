from __future__ import annotations
import numpy as np
from typing import Callable


def euler_maruyama_step(
    C: float,
    t: float,
    dt: float,
    deterministic: Callable[[float, float], float],
    stochastic: Callable[[float], float],
    rng: np.random.Generator | None = None
) -> float:
    """
    Perform one Euler–Maruyama update step.

    Args:
        C (float): Current state (e.g., neurotransmitter concentration).
        t (float): Current time.
        dt (float): Timestep size.
        deterministic (Callable): Drift term function f(C, t).
        stochastic (Callable): Diffusion term function g(C).
        rng (np.random.Generator | None): Optional RNG.

    Returns:
        float: Updated state.
    """
    if rng is None:
        rng = np.random.default_rng()

    dW = rng.normal(loc=0.0, scale=np.sqrt(dt))
    dC_det = deterministic(C, t) * dt
    dC_stoch = stochastic(C) * dW

    return C + dC_det + dC_stoch
