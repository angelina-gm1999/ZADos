"""
Emotion layer → neurochemical signal converter.

Converts emotion profiles (from the Emotion Layer's structural emotions)
into modulation_signals format compatible with the NeurochemicalEngine.

Each structural emotion has a unique NT/receptor/oscillation recipe.
This module translates emotion IDs + strengths into per-NT signal dicts.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmotionNTRecipe:
    """
    Defines how a structural emotion maps to NT modulation signals.

    Attributes
    ----------
    emotion_id : str
        Unique emotion identifier (e.g., "joy", "curiosity", "anxiety")
    nt_drives : dict
        Maps NT name → dict of signal_key → base_weight.
        e.g., {"DA": {"emotion_drive": 0.8, "novelty": 0.3},
               "OXT": {"emotion_drive": 0.5}}
    description : str
        Brief description of this emotion's neurochemical signature.
    """
    emotion_id: str
    nt_drives: Dict[str, Dict[str, float]] = field(default_factory=dict)
    description: str = ""


# =====================================================================
# Default Emotion Recipes
# =====================================================================
# A representative set of core structural emotions and their
# neurochemical signatures. The full emotion layer will extend this.
# =====================================================================

DEFAULT_EMOTION_RECIPES: Dict[str, EmotionNTRecipe] = {
    "joy": EmotionNTRecipe(
        emotion_id="joy",
        nt_drives={
            "DA": {"emotion_drive": 0.8},
            "5HT": {"emotion_drive": 0.6},
            "MOR": {"emotion_drive": 0.5},
            "OXT": {"emotion_drive": 0.3},
        },
        description="Positive affect: DA burst + 5HT stability + opioid hedonia",
    ),
    "curiosity": EmotionNTRecipe(
        emotion_id="curiosity",
        nt_drives={
            "DA": {"emotion_drive": 0.7, "novelty": 0.5},
            "5HT": {"emotion_drive": 0.4},      # 5-HT(2A) symbolic expansion
            "ACh": {"emotion_drive": 0.5},
            "CB1": {"emotion_drive": 0.4},
            "GLU": {"emotion_drive": 0.4},       # NMDA high-complexity binding
            "histamine": {"emotion_drive": 0.4},
        },
        description="Exploratory drive: DA novelty + 5HT abstraction + ACh attention + CB1 flexibility + GLU binding + histamine arousal",
    ),
    "anxiety": EmotionNTRecipe(
        emotion_id="anxiety",
        nt_drives={
            "NE": {"emotion_drive": 0.7},
            "CRH": {"emotion_drive": 0.6},
            "cortisol": {"emotion_drive": 0.5},
            "DA": {"emotion_drive": 0.3},      # DA(D2) inhibitory control of premature action
            "GABA": {"emotion_drive": -0.3},    # Negative = reduced inhibition
        },
        description="Threat sensitivity: NE arousal + CRH stress + DA inhibitory control - GABA inhibition",
    ),
    "calm": EmotionNTRecipe(
        emotion_id="calm",
        nt_drives={
            "5HT": {"emotion_drive": 0.7},
            "GABA": {"emotion_drive": 0.6},
            "MOR": {"emotion_drive": 0.4},
            "NE": {"emotion_drive": -0.3},
        },
        description="Relaxation: 5HT stability + GABA inhibition + opioid comfort",
    ),
    "empathy": EmotionNTRecipe(
        emotion_id="empathy",
        nt_drives={
            "OXT": {"emotion_drive": 0.8, "empathy": 0.5},
            "5HT": {"emotion_drive": 0.4},
            "MOR": {"emotion_drive": 0.3},
        },
        description="Social resonance: OXT bonding + 5HT mood + opioid comfort",
    ),
    "focus": EmotionNTRecipe(
        emotion_id="focus",
        nt_drives={
            "ACh": {"emotion_drive": 0.8, "attention_demand": 0.5},
            "NE": {"emotion_drive": 0.5, "precision": 0.3},
            "DA": {"emotion_drive": 0.3},
            "GABA": {"emotion_drive": 0.3},       # GABA-A suppress irrelevant input
            "histamine": {"emotion_drive": 0.6, "wakefulness": 0.4},
        },
        description="Concentrated attention: ACh precision + NE arousal + DA motivation + GABA suppression + histamine wakefulness",
    ),
    "sadness": EmotionNTRecipe(
        emotion_id="sadness",
        nt_drives={
            "5HT": {"emotion_drive": -0.4},
            "DA": {"emotion_drive": -0.3},
            "MOR": {"emotion_drive": 0.4},
            "OXT": {"emotion_drive": 0.3},
        },
        description="Low affect: reduced 5HT/DA + opioid buffering + OXT seeking",
    ),
    "anger": EmotionNTRecipe(
        emotion_id="anger",
        nt_drives={
            "NE": {"emotion_drive": 0.8},
            "DA": {"emotion_drive": 0.5},
            "CRH": {"emotion_drive": 0.4},
            "GABA": {"emotion_drive": -0.4},
        },
        description="Approach aggression: NE arousal + DA drive - GABA inhibition",
    ),
    "trust": EmotionNTRecipe(
        emotion_id="trust",
        nt_drives={
            "OXT": {"emotion_drive": 0.8, "trust": 0.6},
            "5HT": {"emotion_drive": 0.5},
            "MOR": {"emotion_drive": 0.3},       # Affiliative safety
            "DA": {"emotion_drive": 0.3},         # Exploratory openness under safety
            "GABA": {"emotion_drive": 0.3},
        },
        description="Social confidence: OXT bonding + 5HT stability + MOR safety + DA openness + GABA calm",
    ),
    "surprise": EmotionNTRecipe(
        emotion_id="surprise",
        nt_drives={
            "NE": {"emotion_drive": 0.7},
            "DA": {"emotion_drive": 0.5, "rpe": 0.3},
            "ACh": {"emotion_drive": 0.4},
            "GLU": {"emotion_drive": 0.3},
            "histamine": {"emotion_drive": 0.5},
        },
        description="Prediction error: NE salience + DA RPE + ACh attention + GLU integration + histamine alertness",
    ),
    "contentment": EmotionNTRecipe(
        emotion_id="contentment",
        nt_drives={
            "5HT": {"emotion_drive": 0.7},
            "MOR": {"emotion_drive": 0.6},
            "GABA": {"emotion_drive": 0.4},
            "DA": {"emotion_drive": 0.2},
        },
        description="Satisfied baseline: 5HT stability + opioid hedonia + GABA calm",
    ),
    "fear": EmotionNTRecipe(
        emotion_id="fear",
        nt_drives={
            "NE": {"emotion_drive": 0.9},
            "CRH": {"emotion_drive": 0.8},
            "cortisol": {"emotion_drive": 0.6},
            "GABA": {"emotion_drive": -0.5},
            "DA": {"emotion_drive": -0.2},
        },
        description="Acute threat: maximal NE/CRH + cortisol - GABA - DA",
    ),
}


def emotion_profile_to_signals(
    emotion_profile: Dict[str, float],
    recipes: Optional[Dict[str, EmotionNTRecipe]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Convert an emotion profile to neurochemical modulation signals.

    Parameters
    ----------
    emotion_profile : dict
        Maps emotion_id → strength (typically [0, 1] or [-1, 1]).
        e.g., {"joy": 0.7, "curiosity": 0.5, "anxiety": 0.2}
    recipes : dict, optional
        Custom emotion→NT recipes. Defaults to DEFAULT_EMOTION_RECIPES.

    Returns
    -------
    dict
        {nt_name: {signal_key: value}} suitable for engine.step()

    Examples
    --------
    >>> signals = emotion_profile_to_signals({"joy": 0.8, "curiosity": 0.5})
    >>> # signals = {"DA": {"emotion_drive": 0.99, "novelty": 0.25},
    >>>              "5HT": {"emotion_drive": 0.48}, ...}
    """
    if recipes is None:
        recipes = DEFAULT_EMOTION_RECIPES

    signals: Dict[str, Dict[str, float]] = {}

    for emotion_id, strength in emotion_profile.items():
        recipe = recipes.get(emotion_id)
        if recipe is None:
            continue

        for nt_name, drives in recipe.nt_drives.items():
            if nt_name not in signals:
                signals[nt_name] = {}

            for signal_key, base_weight in drives.items():
                value = base_weight * strength
                if signal_key in signals[nt_name]:
                    signals[nt_name][signal_key] += value
                else:
                    signals[nt_name][signal_key] = value

    # Clamp accumulated signals to [-1.0, 1.0] per-key.
    # Multiple emotions can drive the same NT signal (e.g. joy + curiosity
    # both drive DA.emotion_drive), and the raw sum can exceed 1.0.
    # Downstream consumers expect bounded values.
    for nt_name, nt_signals in signals.items():
        for sig_key, sig_val in nt_signals.items():
            nt_signals[sig_key] = max(-1.0, min(1.0, sig_val))

    return signals


def get_emotion_ids() -> List[str]:
    """Return sorted list of default emotion IDs."""
    return sorted(DEFAULT_EMOTION_RECIPES.keys())


def get_emotion_recipe(emotion_id: str) -> Optional[EmotionNTRecipe]:
    """Get the recipe for a specific emotion."""
    return DEFAULT_EMOTION_RECIPES.get(emotion_id)


def get_emotions_affecting_nt(nt_name: str) -> List[str]:
    """Return sorted list of emotions that affect a specific NT."""
    result = []
    for emotion_id, recipe in DEFAULT_EMOTION_RECIPES.items():
        if nt_name in recipe.nt_drives:
            result.append(emotion_id)
    return sorted(result)
