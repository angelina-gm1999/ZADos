"""
Leaky Integrator Primitives (Extractor 3 — temporal smoothing).

Generic leaky integrator and exponential moving average functions used by
the regulatory modulator and emotion tracker for temporal smoothing.

The core equation is:

    dR/dt = gain · input − (R − R_0) / τ

Discretised via forward Euler:

    R(t+dt) = R(t) + dt · [gain · input − (R(t) − R_0) / τ]

Usage
-----
>>> from zados.neurochem.extractors.leaky_integrator import (
...     LeakyIntegratorState, leaky_integrator_step,
...     exponential_moving_average_step, batch_leaky_integrator_step,
... )
>>> state = LeakyIntegratorState(value=0.0, baseline=0.0)
>>> state = leaky_integrator_step(state, input_signal=1.0, dt=0.01, tau=10.0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional


# =====================================================================
# State
# =====================================================================

@dataclass
class LeakyIntegratorState:
    """
    State of a single leaky integrator.

    Attributes
    ----------
    value : float
        Current integrator output R(t).
    baseline : float
        Equilibrium value R_0 the integrator decays towards.
    """
    value: float = 0.0
    baseline: float = 0.0

    def as_dict(self) -> dict:
        """Export to dictionary."""
        return {"value": self.value, "baseline": self.baseline}

    @classmethod
    def from_dict(cls, data: dict) -> LeakyIntegratorState:
        """Restore from dictionary."""
        return cls(
            value=data.get("value", 0.0),
            baseline=data.get("baseline", 0.0),
        )


# =====================================================================
# Pure functions
# =====================================================================

def leaky_integrator_step(
    state: LeakyIntegratorState,
    input_signal: float,
    dt: float,
    tau: float = 10.0,
    gain: float = 1.0,
) -> LeakyIntegratorState:
    """
    Advance a leaky integrator by one timestep (forward Euler).

    dR/dt = gain · input − (R − R_0) / τ
    R(t+dt) = R(t) + dt · [gain · input − (R(t) − R_0) / τ]

    Parameters
    ----------
    state : LeakyIntegratorState
        Current integrator state.
    input_signal : float
        Driving input I(t).
    dt : float
        Timestep size (seconds).
    tau : float
        Time constant (seconds). Larger τ = slower response.
    gain : float
        Input coupling strength.

    Returns
    -------
    LeakyIntegratorState
        Updated state (new object, original not mutated).
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")

    decay = (state.value - state.baseline) / tau
    drive = gain * input_signal
    new_value = state.value + dt * (drive - decay)

    return LeakyIntegratorState(value=new_value, baseline=state.baseline)


def exponential_moving_average_step(
    current: float,
    new_sample: float,
    dt: float,
    tau: float,
) -> float:
    """
    Exponential moving average (EMA) update.

    X(t+dt) = X(t) · exp(−dt/τ) + sample · (1 − exp(−dt/τ))

    Parameters
    ----------
    current : float
        Current EMA value.
    new_sample : float
        New observation.
    dt : float
        Timestep.
    tau : float
        Time constant (seconds).

    Returns
    -------
    float
        Updated EMA value.
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be positive, got {tau}")

    alpha = 1.0 - math.exp(-dt / tau)
    return current * (1.0 - alpha) + new_sample * alpha


def batch_leaky_integrator_step(
    states: Dict[str, LeakyIntegratorState],
    inputs: Dict[str, float],
    dt: float,
    taus: Dict[str, float],
    gains: Optional[Dict[str, float]] = None,
) -> Dict[str, LeakyIntegratorState]:
    """
    Step multiple leaky integrators in parallel.

    Only steps integrators whose keys appear in both ``states`` and
    ``inputs``.  Missing inputs are treated as 0.0.

    Parameters
    ----------
    states : dict
        Maps name → LeakyIntegratorState.
    inputs : dict
        Maps name → input signal float.
    dt : float
        Timestep.
    taus : dict
        Maps name → time constant.
    gains : dict, optional
        Maps name → gain.  Missing entries default to 1.0.

    Returns
    -------
    dict
        Updated states dict (new state objects).
    """
    result: Dict[str, LeakyIntegratorState] = {}
    for name, state in states.items():
        inp = inputs.get(name, 0.0)
        tau = taus.get(name, 10.0)
        g = 1.0 if gains is None else gains.get(name, 1.0)
        result[name] = leaky_integrator_step(state, inp, dt, tau=tau, gain=g)
    return result
