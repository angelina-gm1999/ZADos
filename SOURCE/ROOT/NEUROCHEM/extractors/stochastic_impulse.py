"""
Stochastic Impulse Generators (Extractor 2 — noise component).

Provides distribution-based impulse samplers for phasic burst generation.
Each sampler produces a non-negative impulse ξ(t) drawn from a
context-shaped distribution, with volatility-sensitive parameterisation.

Supported distributions:
- Gamma: graded bursts with volatility-adaptive shape
- Poisson: discrete event-like triggering
- Lognormal: heavy-tailed affective spillover

Usage
-----
>>> from zados.neurochem.extractors.stochastic_impulse import sample_impulse
>>> xi = sample_impulse(0.7, d_eval_dt=0.2, distribution="gamma", rng=rng)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def sample_gamma_impulse(
    eval_value: float,
    d_eval_dt: float = 0.0,
    base_shape: float = 2.0,
    base_scale: float = 0.5,
    volatility_sensitivity: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Gamma-distributed impulse with volatility-adaptive shape.

    Shape parameter k grows with |de/dt|, producing heavier tails
    during rapid evaluation changes.

    Parameters
    ----------
    eval_value : float
        Current evaluation axis value e_k(t). Scales output amplitude.
    d_eval_dt : float
        Rate of change of evaluation axis (|de/dt|). Controls volatility.
    base_shape : float
        Base Gamma shape parameter k_0.
    base_scale : float
        Gamma scale parameter θ.
    volatility_sensitivity : float
        How much |de/dt| shifts the shape: k = k_0 + v·|de/dt|.
    rng : np.random.Generator, optional
        Numpy RNG. If None, uses default unseeded generator.

    Returns
    -------
    float
        Non-negative impulse value.
        Normalized by E[X] = k * θ so raw sample is O(1) before
        scaling by eval_value.
    """
    if rng is None:
        rng = np.random.default_rng()

    if eval_value <= 0.0:
        return 0.0

    shape = base_shape + volatility_sensitivity * abs(d_eval_dt)
    raw = float(rng.gamma(shape, base_scale))
    # Normalise by E[X] = k * θ so output is roughly O(1)
    expected = shape * base_scale
    if expected > 0.0:
        raw /= expected
    return raw * eval_value


def sample_poisson_impulse(
    eval_value: float,
    rate_scale: float = 5.0,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Poisson-count impulse for discrete event-like bursts.

    count ~ Poisson(λ = rate_scale * eval_value)
    output = count / rate_scale

    Expected value ≈ eval_value.

    Parameters
    ----------
    eval_value : float
        Current evaluation axis value.
    rate_scale : float
        Scales the Poisson rate λ.
    rng : np.random.Generator, optional
        Numpy RNG.

    Returns
    -------
    float
        Non-negative impulse (normalised Poisson count).
    """
    if rng is None:
        rng = np.random.default_rng()

    if eval_value <= 0.0:
        return 0.0

    lam = rate_scale * eval_value
    count = int(rng.poisson(lam))
    return count / rate_scale if rate_scale > 0.0 else 0.0


def sample_lognormal_impulse(
    eval_value: float,
    d_eval_dt: float = 0.0,
    base_mu: float = -0.5,
    base_sigma: float = 0.5,
    volatility_sensitivity: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Lognormal impulse for heavy-tailed bursts.

    σ adapts to volatility: σ = base_sigma + v·|de/dt|, producing
    wider tails under rapid evaluation shifts.
    Normalized by median = exp(μ) so raw sample is O(1) before
    scaling by eval_value.

    Parameters
    ----------
    eval_value : float
        Current evaluation axis value.
    d_eval_dt : float
        Rate of change of evaluation axis.
    base_mu : float
        Log-mean parameter μ.
    base_sigma : float
        Base log-std parameter σ_0.
    volatility_sensitivity : float
        How much |de/dt| widens σ.
    rng : np.random.Generator, optional
        Numpy RNG.

    Returns
    -------
    float
        Non-negative impulse value.
    """
    if rng is None:
        rng = np.random.default_rng()

    if eval_value <= 0.0:
        return 0.0

    sigma = base_sigma + volatility_sensitivity * abs(d_eval_dt)
    raw = float(rng.lognormal(base_mu, max(sigma, 1e-6)))
    # Normalise by median (exp(mu)) so output is roughly O(1)
    import math
    median = math.exp(base_mu)
    if median > 0.0:
        raw /= median
    return raw * eval_value


def sample_impulse(
    eval_value: float,
    d_eval_dt: float = 0.0,
    distribution: str = "gamma",
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> float:
    """
    Dispatch to the appropriate impulse sampler by distribution name.

    Parameters
    ----------
    eval_value : float
        Current evaluation axis value e_k(t).
    d_eval_dt : float
        Rate of change |de/dt|.
    distribution : str
        One of "gamma", "poisson", "lognormal".
    rng : np.random.Generator, optional
        Numpy RNG.
    **kwargs
        Extra keyword arguments forwarded to the specific sampler.

    Returns
    -------
    float
        Non-negative impulse.

    Raises
    ------
    ValueError
        If distribution name is not recognised.
    """
    if distribution == "gamma":
        return sample_gamma_impulse(eval_value, d_eval_dt, rng=rng, **kwargs)
    elif distribution == "poisson":
        return sample_poisson_impulse(eval_value, rng=rng, **kwargs)
    elif distribution == "lognormal":
        return sample_lognormal_impulse(eval_value, d_eval_dt, rng=rng, **kwargs)
    else:
        raise ValueError(f"Unknown impulse distribution: {distribution!r}")
