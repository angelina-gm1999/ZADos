"""
Urgency Forecast Module (Extractor 5 — threshold-crossing prediction).

Performs temporal smoothing of evaluation axes, linear-exponential
forecasting, threshold-crossing detection, and generates both reactive
(Poisson NE/DA spikes) and modulatory (NE reuptake ↑, GABA_B tone ↓)
outputs.

Signal flow
-----------
1. Map 5 urgency axes onto the 8-axis evaluation vector via weighted
   sums with optional inversion.
2. Temporally smooth each axis through a leaky integrator (reuses
   ``leaky_integrator_step()``).
3. Compute linear-exponential forecast:
   ê_k(t+δ) = ẽ_k(t) + α_k · (dẽ_k/dt) · (1 − exp(−δ/τ_k))
4. Detect threshold breach: B_k(t) = 1 if ê_k(t+δ) > Θ_k.
5. Compute global urgency risk: U(t) = max_k (ê_k − Θ_k)₊.
6. Reactive output: ΔC_NE(t) = β_urg · U(t) · Poisson(λ_urg).
7. Modulatory output (persistent breach): NE u_base ↑, GABA_B K_d ↓.

Usage
-----
>>> from zados.neurochem.extractors.urgency_forecast import (
...     step_urgency_forecast, DEFAULT_URGENCY_FORECAST_CONFIG,
...     UrgencyForecastState,
... )
>>> state = UrgencyForecastState.from_config(DEFAULT_URGENCY_FORECAST_CONFIG)
>>> state, risk, bursts, feedback = step_urgency_forecast(
...     state, adjusted_eval, dt=0.01,
...     config=DEFAULT_URGENCY_FORECAST_CONFIG, rng=rng,
... )
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from zados.neurochem.extractors.leaky_integrator import (
    LeakyIntegratorState,
    leaky_integrator_step,
)
from zados.neurochem.extractors.stochastic_impulse import sample_poisson_impulse


# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True)
class UrgencyAxisSourceDef:
    """
    Definition of a single evaluation-axis source feeding an urgency axis.

    Attributes
    ----------
    eval_axis : str
        Name of the evaluation vector axis to read.
    weight : float
        Coupling weight (multiplied with the axis value).
    invert : bool
        If True, use ``(1 - eval_value)`` instead of ``eval_value``.
    """
    eval_axis: str
    weight: float = 1.0
    invert: bool = False


@dataclass(frozen=True)
class UrgencyAxisConfig:
    """
    Configuration for a single urgency forecast axis.

    Attributes
    ----------
    name : str
        Axis identifier (e.g. "logical_pressure").
    sources : tuple of UrgencyAxisSourceDef
        Evaluation-axis sources with weights and optional inversion.
    alpha : float
        Forecast gain — sensitivity to rate of change dẽ/dt.
    tau_smooth : float
        Time constant for the temporal-smoothing leaky integrator (seconds).
    tau_forecast : float
        Decay time constant for the linear-exponential forecast (seconds).
    threshold : float
        Breach threshold Θ_k in [0, 1].
    """
    name: str
    sources: Tuple[UrgencyAxisSourceDef, ...] = ()
    alpha: float = 1.0
    tau_smooth: float = 2.0
    tau_forecast: float = 3.0
    threshold: float = 0.7


@dataclass(frozen=True)
class UrgencyForecastConfig:
    """
    Global configuration for the urgency forecast module.

    Attributes
    ----------
    axes : tuple of UrgencyAxisConfig
        The urgency axes to monitor (default: 5).
    prediction_window : float
        Forecast horizon δ in seconds (T in the spec).
    urgency_epsilon : float
        Minimum U(t) to trigger reactive burst output.
    beta_urg : float
        Reactive NE burst scaling factor.
    lambda_urg : float
        Poisson rate parameter for NE spike generation.
    da_burst_fraction : float
        DA burst = this fraction × NE burst.
    ne_gain_boost : float
        Modulatory NE u_base multiplier increase during persistent breach.
    gaba_tone_reduction : float
        Modulatory GABA_B K_d multiplier (< 1.0 = more binding) during
        persistent breach.
    persistence_steps : int
        Number of consecutive breach ticks required before modulatory
        output activates.
    """
    axes: Tuple[UrgencyAxisConfig, ...] = ()
    prediction_window: float = 5.0
    urgency_epsilon: float = 0.1
    beta_urg: float = 0.3
    lambda_urg: float = 5.0
    da_burst_fraction: float = 0.15
    ne_gain_boost: float = 0.15
    gaba_tone_reduction: float = 0.85
    persistence_steps: int = 3


# --- Default urgency axes ---
# 5 axes mapped onto the 8-axis evaluation vector via weighted sums.
# "invert" axes use (1 - eval_value) to represent deficit / compression.

DEFAULT_URGENCY_AXES: Tuple[UrgencyAxisConfig, ...] = (
    UrgencyAxisConfig(
        name="logical_pressure",
        sources=(UrgencyAxisSourceDef("logical_conflict", weight=1.0),),
        alpha=1.2,
        tau_smooth=2.0,
        tau_forecast=3.0,
        threshold=0.7,
    ),
    UrgencyAxisConfig(
        name="emotional_compression",
        sources=(UrgencyAxisSourceDef("emotional_valence", weight=1.0, invert=True),),
        alpha=0.8,
        tau_smooth=3.0,
        tau_forecast=4.0,
        threshold=0.75,
    ),
    UrgencyAxisConfig(
        name="discord_build",
        sources=(
            UrgencyAxisSourceDef("urgency", weight=0.6),
            UrgencyAxisSourceDef("logical_conflict", weight=0.4),
        ),
        alpha=1.0,
        tau_smooth=2.5,
        tau_forecast=3.5,
        threshold=0.65,
    ),
    UrgencyAxisConfig(
        name="expectation_violation",
        sources=(
            UrgencyAxisSourceDef("novelty", weight=0.5),
            UrgencyAxisSourceDef("reward_alignment", weight=0.5, invert=True),
        ),
        alpha=1.5,
        tau_smooth=1.5,
        tau_forecast=2.0,
        threshold=0.7,
    ),
    UrgencyAxisConfig(
        name="narrative_entropy",
        sources=(UrgencyAxisSourceDef("coherence", weight=1.0, invert=True),),
        alpha=0.9,
        tau_smooth=3.0,
        tau_forecast=4.0,
        threshold=0.7,
    ),
)

DEFAULT_URGENCY_FORECAST_CONFIG = UrgencyForecastConfig(
    axes=DEFAULT_URGENCY_AXES,
)


# =====================================================================
# State
# =====================================================================

@dataclass
class UrgencyForecastState:
    """
    Mutable state for the urgency forecast module.

    Attributes
    ----------
    smoothed : dict
        Maps axis_name → LeakyIntegratorState (temporal smoother).
    prev_smoothed : dict
        Maps axis_name → ẽ_k(t−dt) for finite-difference derivative.
    breach_counters : dict
        Maps axis_name → int (consecutive breach tick count).
    """
    smoothed: Dict[str, LeakyIntegratorState] = field(default_factory=dict)
    prev_smoothed: Dict[str, float] = field(default_factory=dict)
    breach_counters: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: UrgencyForecastConfig) -> UrgencyForecastState:
        """Create initial state from config with all integrators at baseline."""
        smoothed = {}
        prev_smoothed = {}
        breach_counters = {}
        for axis_cfg in config.axes:
            smoothed[axis_cfg.name] = LeakyIntegratorState(value=0.0, baseline=0.0)
            prev_smoothed[axis_cfg.name] = 0.0
            breach_counters[axis_cfg.name] = 0
        return cls(
            smoothed=smoothed,
            prev_smoothed=prev_smoothed,
            breach_counters=breach_counters,
        )

    def as_dict(self) -> dict:
        """Export to dictionary for checkpointing."""
        return {
            "smoothed": {
                k: v.as_dict() for k, v in self.smoothed.items()
            },
            "prev_smoothed": dict(self.prev_smoothed),
            "breach_counters": dict(self.breach_counters),
        }

    @classmethod
    def from_dict(cls, data: dict) -> UrgencyForecastState:
        """Restore from dictionary."""
        smoothed = {
            k: LeakyIntegratorState.from_dict(v)
            for k, v in data.get("smoothed", {}).items()
        }
        return cls(
            smoothed=smoothed,
            prev_smoothed=dict(data.get("prev_smoothed", {})),
            breach_counters=dict(data.get("breach_counters", {})),
        )


# =====================================================================
# Pure functions
# =====================================================================

def compute_urgency_axis_value(
    adjusted_eval: Dict[str, float],
    axis_config: UrgencyAxisConfig,
) -> float:
    """
    Compute a single urgency axis value from the evaluation vector.

    e_k = Σ_j w_j · transform(eval[source_j])

    where transform = (1 − x) if invert else x.

    Parameters
    ----------
    adjusted_eval : dict
        Maps eval_axis_name → float [0, 1].
    axis_config : UrgencyAxisConfig
        Configuration defining sources, weights, and inversion.

    Returns
    -------
    float
        Urgency axis value, clamped to [0, 1].
    """
    value = 0.0
    for src in axis_config.sources:
        raw = adjusted_eval.get(src.eval_axis, 0.0)
        transformed = (1.0 - raw) if src.invert else raw
        value += src.weight * transformed
    return max(0.0, min(1.0, value))


def forecast_peak(
    smoothed_current: float,
    smoothed_prev: float,
    dt: float,
    alpha: float,
    tau_forecast: float,
    prediction_window: float,
) -> float:
    """
    Linear-exponential forecast of an urgency axis.

    ê_k(t+δ) = ẽ_k(t) + α · (dẽ_k/dt) · (1 − exp(−δ/τ_k))

    The forecast peak occurs at δ* = prediction_window since the factor
    (1 − exp(−δ/τ)) is monotonically increasing.

    Parameters
    ----------
    smoothed_current : float
        ẽ_k(t) — current smoothed value.
    smoothed_prev : float
        ẽ_k(t−dt) — previous smoothed value.
    dt : float
        Timestep (seconds). Must be > 0.
    alpha : float
        Forecast gain (sensitivity to rate of change).
    tau_forecast : float
        Forecast decay time constant (seconds).
    prediction_window : float
        Forecast horizon δ (seconds).

    Returns
    -------
    float
        Forecasted value, clamped to [0, 1].
    """
    if dt <= 0.0:
        return max(0.0, min(1.0, smoothed_current))

    d_smooth = (smoothed_current - smoothed_prev) / dt

    if tau_forecast <= 0.0:
        forecast_factor = 1.0
    else:
        forecast_factor = 1.0 - math.exp(-prediction_window / tau_forecast)

    predicted = smoothed_current + alpha * d_smooth * forecast_factor
    return max(0.0, min(1.0, predicted))


def detect_breach(forecast_value: float, threshold: float) -> bool:
    """
    Detect whether a forecast exceeds its threshold.

    B_k(t) = 1 if ê_k(t+δ) > Θ_k.

    Parameters
    ----------
    forecast_value : float
        Forecasted axis value ê_k.
    threshold : float
        Breach threshold Θ_k.

    Returns
    -------
    bool
        True if forecast strictly exceeds threshold.
    """
    return forecast_value > threshold


def compute_urgency_risk(
    forecasts: Dict[str, float],
    thresholds: Dict[str, float],
) -> float:
    """
    Compute global urgency risk as the maximum rectified breach.

    U(t) = max_k (ê_k − Θ_k)₊

    Parameters
    ----------
    forecasts : dict
        Maps axis_name → forecasted value ê_k.
    thresholds : dict
        Maps axis_name → threshold Θ_k.

    Returns
    -------
    float
        Non-negative urgency risk. 0.0 if no breaches.
    """
    max_risk = 0.0
    for name, forecast in forecasts.items():
        threshold = thresholds.get(name, 1.0)
        excess = forecast - threshold
        if excess > max_risk:
            max_risk = excess
    return max_risk


def compute_reactive_burst(
    urgency_risk: float,
    config: UrgencyForecastConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Compute reactive NE/DA burst deltas from urgency risk.

    If U(t) >= ε:
        ΔC_NE = β_urg · U(t) · Poisson(λ_urg · U(t)) / λ_urg
        ΔC_DA = da_burst_fraction · ΔC_NE

    Uses ``sample_poisson_impulse()`` for NE spike generation,
    consistent with the reactivity matrix pattern.

    Parameters
    ----------
    urgency_risk : float
        Global urgency risk U(t).
    config : UrgencyForecastConfig
        Module configuration.
    rng : np.random.Generator, optional
        Numpy RNG for stochastic sampling.

    Returns
    -------
    dict
        Maps NT name → burst delta. Empty if below epsilon.
    """
    if urgency_risk < config.urgency_epsilon:
        return {}

    poisson_sample = sample_poisson_impulse(
        urgency_risk,
        rate_scale=config.lambda_urg,
        rng=rng,
    )
    ne_delta = config.beta_urg * urgency_risk * poisson_sample
    da_delta = config.da_burst_fraction * ne_delta

    result: Dict[str, float] = {}
    if ne_delta > 0.0:
        result["NE"] = ne_delta
    if da_delta > 0.0:
        result["DA"] = da_delta
    return result


def compute_modulatory_feedback(
    breach_counters: Dict[str, int],
    config: UrgencyForecastConfig,
) -> Dict[str, Dict]:
    """
    Compute modulatory feedback from persistent threshold breaches.

    If any axis has breach_counter >= persistence_steps:
        - NE u_base_multiplier = 1.0 + ne_gain_boost  (increase reuptake)
        - GABA_B K_d_multiplier = gaba_tone_reduction  (lower K_d = more binding)

    Parameters
    ----------
    breach_counters : dict
        Maps axis_name → consecutive breach tick count.
    config : UrgencyForecastConfig
        Module configuration.

    Returns
    -------
    dict
        Feedback params in standard format:
        {"neurotransmitters": {...}, "receptors": {...}}.
        Empty sub-dicts if no persistent breach.
    """
    feedback: Dict[str, Dict] = {"neurotransmitters": {}, "receptors": {}}

    persistent = any(
        count >= config.persistence_steps
        for count in breach_counters.values()
    )
    if not persistent:
        return feedback

    feedback["neurotransmitters"]["NE"] = {
        "u_base_multiplier": 1.0 + config.ne_gain_boost,
    }
    feedback["receptors"]["GABA_B"] = {
        "K_d_multiplier": config.gaba_tone_reduction,
    }
    return feedback


def step_urgency_forecast(
    state: UrgencyForecastState,
    adjusted_eval: Dict[str, float],
    dt: float,
    config: UrgencyForecastConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[UrgencyForecastState, float, Dict[str, float], Dict[str, Dict]]:
    """
    Execute one full urgency forecast tick.

    Sequence:
    1. Compute raw urgency axis values from evaluation vector.
    2. Temporally smooth each axis via leaky integrator.
    3. Compute linear-exponential forecast for each axis.
    4. Detect threshold breaches and update counters.
    5. Compute global urgency risk U(t).
    6. Generate reactive NE/DA burst deltas.
    7. Generate modulatory feedback (persistent breach).

    Parameters
    ----------
    state : UrgencyForecastState
        Current module state.
    adjusted_eval : dict
        Maps eval_axis_name → float [0, 1] (post-4M-adjustment).
    dt : float
        Timestep (seconds).
    config : UrgencyForecastConfig
        Module configuration.
    rng : np.random.Generator, optional
        Numpy RNG for stochastic sampling.

    Returns
    -------
    tuple of (UrgencyForecastState, float, dict, dict)
        - new_state : Updated module state.
        - urgency_risk : Global risk U(t) ≥ 0.
        - burst_deltas : {nt_name: delta} for reactive bursts.
        - feedback_params : Standard feedback format for modulatory output.
    """
    new_smoothed = dict(state.smoothed)
    new_prev_smoothed = dict(state.prev_smoothed)
    new_breach_counters = dict(state.breach_counters)

    forecasts: Dict[str, float] = {}
    thresholds: Dict[str, float] = {}

    for axis_cfg in config.axes:
        name = axis_cfg.name

        # 1. Raw urgency axis value
        raw_value = compute_urgency_axis_value(adjusted_eval, axis_cfg)

        # 2. Temporal smoothing via leaky integrator
        integrator = state.smoothed.get(
            name, LeakyIntegratorState(value=0.0, baseline=0.0)
        )
        new_integrator = leaky_integrator_step(
            integrator,
            input_signal=raw_value,
            dt=dt,
            tau=axis_cfg.tau_smooth,
        )
        smoothed_current = new_integrator.value
        smoothed_prev = state.prev_smoothed.get(name, 0.0)

        # 3. Linear-exponential forecast
        forecast_val = forecast_peak(
            smoothed_current,
            smoothed_prev,
            dt,
            alpha=axis_cfg.alpha,
            tau_forecast=axis_cfg.tau_forecast,
            prediction_window=config.prediction_window,
        )

        # 4. Threshold breach detection
        breached = detect_breach(forecast_val, axis_cfg.threshold)
        if breached:
            new_breach_counters[name] = new_breach_counters.get(name, 0) + 1
        else:
            new_breach_counters[name] = 0

        # Store for risk computation
        forecasts[name] = forecast_val
        thresholds[name] = axis_cfg.threshold

        # Update state
        new_smoothed[name] = new_integrator
        new_prev_smoothed[name] = smoothed_current

    # 5. Global urgency risk
    urgency_risk = compute_urgency_risk(forecasts, thresholds)

    # 6. Reactive burst deltas
    burst_deltas = compute_reactive_burst(urgency_risk, config, rng=rng)

    # 7. Modulatory feedback
    feedback_params = compute_modulatory_feedback(new_breach_counters, config)

    new_state = UrgencyForecastState(
        smoothed=new_smoothed,
        prev_smoothed=new_prev_smoothed,
        breach_counters=new_breach_counters,
    )

    return new_state, urgency_risk, burst_deltas, feedback_params
