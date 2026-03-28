from __future__ import annotations
import numpy as np


def concentration_scaled_noise(
    C: float,
    alpha: float,
    rng: np.random.Generator | None = None
) -> float:
    """
    Returns multiplicative noise scaled by concentration.

    Args:
        C (float): Current concentration.
        alpha (float): Noise gain factor.
        rng (np.random.Generator | None): Optional RNG for reproducibility.

    Returns:
        float: Noise value for current timestep.
    """
    if rng is None:
        rng = np.random.default_rng()

    xi = rng.normal(loc=0.0, scale=1.0)
    return alpha * C * xi
