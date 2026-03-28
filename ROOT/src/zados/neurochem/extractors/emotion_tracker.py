"""
Emotion Saturation Tracker (Extractor 4 — state tracking).

Maintains per-emotion leaky integrators that track emotional intensity
over time. Provides dominant emotion detection and saturation metrics.

Each emotion integrator follows:

    dE_k/dt = λ_k · I_k − E_k / τ_k

where E_k is the saturation level, I_k is the emotion input strength,
λ_k is the gain, and τ_k is the decay time constant.

Usage
-----
>>> from zados.neurochem.extractors.emotion_tracker import (
...     EmotionTrackerState, step_emotion_tracker,
...     get_dominant_emotion, get_emotion_saturations,
... )
>>> state = EmotionTrackerState.from_emotion_ids(["joy", "curiosity", "anxiety"])
>>> state = step_emotion_tracker(state, {"joy": 0.8, "curiosity": 0.5}, dt=0.01)
>>> dominant = get_dominant_emotion(state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from zados.neurochem.extractors.leaky_integrator import (
    LeakyIntegratorState,
    leaky_integrator_step,
)


# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True)
class EmotionTrackerConfig:
    """
    Configuration for a single emotion's leaky integrator.

    Attributes
    ----------
    emotion_id : str
        Emotion identifier (e.g., "joy", "curiosity").
    tau : float
        Decay time constant (seconds). Larger → slower decay.
    gain : float
        Input coupling strength.
    saturation_cap : float
        Maximum saturation level (clamping).
    """
    emotion_id: str
    tau: float = 10.0
    gain: float = 1.0
    saturation_cap: float = 1.0


# Default configs for all 12 structural emotions
DEFAULT_EMOTION_TRACKER_CONFIGS: Dict[str, EmotionTrackerConfig] = {
    "joy":         EmotionTrackerConfig("joy",         tau=8.0,  gain=1.0),
    "curiosity":   EmotionTrackerConfig("curiosity",   tau=6.0,  gain=1.2),
    "anxiety":     EmotionTrackerConfig("anxiety",     tau=12.0, gain=1.0),
    "fear":        EmotionTrackerConfig("fear",        tau=15.0, gain=1.5),
    "anger":       EmotionTrackerConfig("anger",       tau=10.0, gain=1.3),
    "sadness":     EmotionTrackerConfig("sadness",     tau=20.0, gain=0.8),
    "calm":        EmotionTrackerConfig("calm",        tau=15.0, gain=0.6),
    "empathy":     EmotionTrackerConfig("empathy",     tau=12.0, gain=1.0),
    "trust":       EmotionTrackerConfig("trust",       tau=18.0, gain=0.7),
    "surprise":    EmotionTrackerConfig("surprise",    tau=4.0,  gain=1.5),
    "contentment": EmotionTrackerConfig("contentment", tau=15.0, gain=0.5),
    "focus":       EmotionTrackerConfig("focus",       tau=8.0,  gain=1.0),
}


# =====================================================================
# State
# =====================================================================

@dataclass
class EmotionTrackerState:
    """
    State container for the emotion saturation tracker.

    Holds one ``LeakyIntegratorState`` per tracked emotion.

    Attributes
    ----------
    integrators : dict
        Maps emotion_id → LeakyIntegratorState.
    """
    integrators: Dict[str, LeakyIntegratorState] = field(default_factory=dict)

    @classmethod
    def from_emotion_ids(
        cls,
        ids: Optional[List[str]] = None,
    ) -> EmotionTrackerState:
        """
        Initialize tracker state for the given emotion IDs.

        All integrators start at zero (no emotion active).

        Parameters
        ----------
        ids : list of str, optional
            Emotion IDs to track. If None, uses all 12 defaults.

        Returns
        -------
        EmotionTrackerState
            Initialized state.
        """
        if ids is None:
            ids = sorted(DEFAULT_EMOTION_TRACKER_CONFIGS.keys())
        integrators = {eid: LeakyIntegratorState(0.0, 0.0) for eid in ids}
        return cls(integrators=integrators)

    def as_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "integrators": {
                eid: state.as_dict()
                for eid, state in self.integrators.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> EmotionTrackerState:
        """Restore from dictionary."""
        integrators = {}
        for eid, state_dict in data.get("integrators", {}).items():
            integrators[eid] = LeakyIntegratorState.from_dict(state_dict)
        return cls(integrators=integrators)


# =====================================================================
# Pure functions
# =====================================================================

def step_emotion_tracker(
    state: EmotionTrackerState,
    emotion_inputs: Dict[str, float],
    dt: float,
    configs: Optional[Dict[str, EmotionTrackerConfig]] = None,
) -> EmotionTrackerState:
    """
    Step all emotion integrators by one timestep.

    For each tracked emotion:
        dE_k/dt = λ_k · I_k − E_k / τ_k

    Emotions present in ``state.integrators`` but not in ``emotion_inputs``
    decay naturally (input = 0).

    Parameters
    ----------
    state : EmotionTrackerState
        Current tracker state.
    emotion_inputs : dict
        Maps emotion_id → input strength. Typically [0, 1].
    dt : float
        Timestep.
    configs : dict, optional
        Maps emotion_id → EmotionTrackerConfig.
        Defaults to DEFAULT_EMOTION_TRACKER_CONFIGS.

    Returns
    -------
    EmotionTrackerState
        Updated state with clamped saturation values.
    """
    if configs is None:
        configs = DEFAULT_EMOTION_TRACKER_CONFIGS

    new_integrators = {}
    for eid, integrator in state.integrators.items():
        inp = emotion_inputs.get(eid, 0.0)
        cfg = configs.get(eid, EmotionTrackerConfig(eid))

        new_int = leaky_integrator_step(
            integrator,
            inp,
            dt,
            tau=cfg.tau,
            gain=cfg.gain,
        )

        # Clamp to [0, saturation_cap]
        new_int.value = max(0.0, min(cfg.saturation_cap, new_int.value))

        new_integrators[eid] = new_int

    return EmotionTrackerState(integrators=new_integrators)


def get_dominant_emotion(state: EmotionTrackerState) -> Tuple[str, float]:
    """
    Get the emotion with the highest saturation level.

    Parameters
    ----------
    state : EmotionTrackerState
        Current tracker state.

    Returns
    -------
    tuple of (str, float)
        (emotion_id, saturation_value).
        Returns ("none", 0.0) if no emotions are tracked.
    """
    if not state.integrators:
        return ("none", 0.0)

    best_eid = ""
    best_val = -1.0
    for eid, integrator in state.integrators.items():
        if integrator.value > best_val:
            best_val = integrator.value
            best_eid = eid

    return (best_eid, best_val)


def get_saturation(state: EmotionTrackerState) -> float:
    """
    Get overall emotional saturation (mean of all integrator values).

    Parameters
    ----------
    state : EmotionTrackerState
        Current state.

    Returns
    -------
    float
        Mean saturation across all tracked emotions.
    """
    if not state.integrators:
        return 0.0
    total = sum(i.value for i in state.integrators.values())
    return total / len(state.integrators)


def get_emotion_saturations(state: EmotionTrackerState) -> Dict[str, float]:
    """
    Get all per-emotion saturation values.

    Parameters
    ----------
    state : EmotionTrackerState
        Current state.

    Returns
    -------
    dict
        Maps emotion_id → saturation float.
    """
    return {eid: i.value for eid, i in state.integrators.items()}
