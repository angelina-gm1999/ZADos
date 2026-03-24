"""
Emotion 4M/4R Splitter (Extractor 4M — Modulatory / 4R — Reactive).

Splits per-emotion saturation levels into two pathways:

- **4M (Modulatory)**: Slow, tonic baseline adjustments. Each emotion's
  modulatory fraction contributes additive adjustments to evaluation
  vector axes (E(t)) before the regulatory modulator step.

- **4R (Reactive)**: Fast, phasic bursts. Each emotion's reactive fraction
  scales the existing EmotionNTRecipe signals for injection via
  ``emotion_profile_to_signals()``.

Usage
-----
>>> from zados.neurochem.extractors.emotion_splitter import (
...     split_emotion_effects, DEFAULT_EMOTION_SPLIT_CONFIGS,
... )
>>> modulatory, reactive = split_emotion_effects(emotion_tracker_state, configs)
>>> # modulatory: {eval_axis → adjustment}  (add to E(t))
>>> # reactive:   {emotion_id → scaled_strength}  (feed to emotion_profile_to_signals)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from zados.neurochem.extractors.emotion_tracker import (
    EmotionTrackerState,
    get_emotion_saturations,
)


# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True)
class EmotionSplitConfig:
    """
    Configuration for splitting a single emotion into 4M/4R pathways.

    Attributes
    ----------
    emotion_id : str
        Emotion identifier.
    modulatory_fraction : float
        Fraction routed to modulatory (tonic) pathway [0, 1].
    reactive_fraction : float
        Fraction routed to reactive (phasic) pathway [0, 1].
        Should satisfy: modulatory_fraction + reactive_fraction = 1.0
    modulatory_target_axes : dict
        Maps evaluation axis name → coupling weight.
        The emotion's modulatory contribution is:
            Σ (saturation * modulatory_fraction * coupling_weight)
        added to each target axis.
    reactive_boost_gain : float
        Gain applied to the reactive pathway signal.
    """
    emotion_id: str
    modulatory_fraction: float = 0.5
    reactive_fraction: float = 0.5
    modulatory_target_axes: Dict[str, float] = field(default_factory=dict)
    reactive_boost_gain: float = 1.0


# Default split configs for all 12 emotions.
# modulatory_target_axes defines which evaluation axes each emotion influences.
# Note: Negative coupling weights are intentional for certain emotions:
#   sadness:  emotional_valence=-0.3, reward_alignment=-0.2  (dampens positive evaluation)
#   calm:     urgency=-0.2  (reduces perceived urgency)
# These create inhibitory modulatory effects, shifting evaluation axes downward.
DEFAULT_EMOTION_SPLIT_CONFIGS: Dict[str, EmotionSplitConfig] = {
    "joy": EmotionSplitConfig(
        emotion_id="joy",
        modulatory_fraction=0.6,
        reactive_fraction=0.4,
        modulatory_target_axes={
            "reward_alignment": 0.4,
            "emotional_valence": 0.3,
        },
    ),
    "curiosity": EmotionSplitConfig(
        emotion_id="curiosity",
        modulatory_fraction=0.3,
        reactive_fraction=0.7,
        modulatory_target_axes={
            "novelty": 0.5,
            "coherence": 0.2,
        },
    ),
    "anxiety": EmotionSplitConfig(
        emotion_id="anxiety",
        modulatory_fraction=0.7,
        reactive_fraction=0.3,
        modulatory_target_axes={
            "urgency": 0.4,
            "logical_conflict": 0.2,
        },
    ),
    "fear": EmotionSplitConfig(
        emotion_id="fear",
        modulatory_fraction=0.4,
        reactive_fraction=0.6,
        modulatory_target_axes={
            "urgency": 0.5,
        },
        reactive_boost_gain=1.5,
    ),
    "anger": EmotionSplitConfig(
        emotion_id="anger",
        modulatory_fraction=0.4,
        reactive_fraction=0.6,
        modulatory_target_axes={
            "urgency": 0.3,
            "logical_conflict": 0.3,
        },
        reactive_boost_gain=1.3,
    ),
    "sadness": EmotionSplitConfig(
        emotion_id="sadness",
        modulatory_fraction=0.7,
        reactive_fraction=0.3,
        modulatory_target_axes={
            "emotional_valence": -0.3,
            "reward_alignment": -0.2,
        },
    ),
    "calm": EmotionSplitConfig(
        emotion_id="calm",
        modulatory_fraction=0.8,
        reactive_fraction=0.2,
        modulatory_target_axes={
            "coherence": 0.3,
            "urgency": -0.2,
        },
    ),
    "empathy": EmotionSplitConfig(
        emotion_id="empathy",
        modulatory_fraction=0.6,
        reactive_fraction=0.4,
        modulatory_target_axes={
            "social_salience": 0.4,
            "emotional_valence": 0.3,
        },
    ),
    "trust": EmotionSplitConfig(
        emotion_id="trust",
        modulatory_fraction=0.7,
        reactive_fraction=0.3,
        modulatory_target_axes={
            "social_salience": 0.4,
            "identity_resonance": 0.2,
        },
    ),
    "surprise": EmotionSplitConfig(
        emotion_id="surprise",
        modulatory_fraction=0.3,
        reactive_fraction=0.7,
        modulatory_target_axes={
            "novelty": 0.3,
            "urgency": 0.2,
        },
        reactive_boost_gain=1.2,
    ),
    "contentment": EmotionSplitConfig(
        emotion_id="contentment",
        modulatory_fraction=0.8,
        reactive_fraction=0.2,
        modulatory_target_axes={
            "coherence": 0.2,
            "reward_alignment": 0.3,
        },
    ),
    "focus": EmotionSplitConfig(
        emotion_id="focus",
        modulatory_fraction=0.5,
        reactive_fraction=0.5,
        modulatory_target_axes={
            "coherence": 0.4,
            "urgency": 0.1,
        },
    ),
}


# =====================================================================
# Pure functions
# =====================================================================

def compute_modulatory_adjustments(
    saturations: Dict[str, float],
    configs: Dict[str, EmotionSplitConfig],
) -> Dict[str, float]:
    """
    Aggregate modulatory effects on evaluation axes from all emotions.

    For each emotion with saturation S_k:
        For each target axis a with coupling weight w_a:
            adjustment[a] += S_k * modulatory_fraction * w_a

    Parameters
    ----------
    saturations : dict
        Maps emotion_id → saturation float.
    configs : dict
        Maps emotion_id → EmotionSplitConfig.

    Returns
    -------
    dict
        Maps eval_axis_name → additive adjustment float.
    """
    adjustments: Dict[str, float] = {}

    for eid, saturation in saturations.items():
        cfg = configs.get(eid)
        if cfg is None or saturation <= 0.0:
            continue

        mod_strength = saturation * cfg.modulatory_fraction

        for axis, weight in cfg.modulatory_target_axes.items():
            contribution = mod_strength * weight
            if axis in adjustments:
                adjustments[axis] += contribution
            else:
                adjustments[axis] = contribution

    return adjustments


def compute_reactive_signals(
    saturations: Dict[str, float],
    configs: Dict[str, EmotionSplitConfig],
) -> Dict[str, float]:
    """
    Compute reactive emotion profile for phasic NT bursts.

    For each emotion with saturation S_k:
        reactive_profile[emotion_id] = S_k * reactive_fraction * reactive_boost_gain

    The output is suitable as input to ``emotion_profile_to_signals()``.

    Parameters
    ----------
    saturations : dict
        Maps emotion_id → saturation float.
    configs : dict
        Maps emotion_id → EmotionSplitConfig.

    Returns
    -------
    dict
        Maps emotion_id → reactive strength.
    """
    profile: Dict[str, float] = {}

    for eid, saturation in saturations.items():
        cfg = configs.get(eid)
        if cfg is None or saturation <= 0.0:
            continue

        reactive_strength = saturation * cfg.reactive_fraction * cfg.reactive_boost_gain
        if reactive_strength > 0.0:
            profile[eid] = reactive_strength

    return profile


def split_emotion_effects(
    emotion_tracker_state: EmotionTrackerState,
    configs: Optional[Dict[str, EmotionSplitConfig]] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Split emotion tracker state into modulatory and reactive pathways.

    Parameters
    ----------
    emotion_tracker_state : EmotionTrackerState
        Current emotion tracker state.
    configs : dict, optional
        Maps emotion_id → EmotionSplitConfig.
        Defaults to DEFAULT_EMOTION_SPLIT_CONFIGS.

    Returns
    -------
    tuple of (dict, dict)
        (modulatory_adjustments, reactive_profile)
        - modulatory_adjustments: {eval_axis → additive float} — add to E(t)
        - reactive_profile: {emotion_id → strength} — feed to emotion_profile_to_signals()
    """
    if configs is None:
        configs = DEFAULT_EMOTION_SPLIT_CONFIGS

    saturations = get_emotion_saturations(emotion_tracker_state)

    modulatory = compute_modulatory_adjustments(saturations, configs)
    reactive = compute_reactive_signals(saturations, configs)

    return modulatory, reactive
