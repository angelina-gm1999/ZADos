"""
ZA-DOS Core Pipeline — Mode→Profile Mapping (spec §5.5).

Maps mode tokens (from select_mode()) to static reward profile names
(from PROFILE_REGISTRY in reward/profile/static_profiles.py).
"""
from __future__ import annotations

from typing import Dict

# ------------------------------------------------------------------
# Mode token → reward profile name
# ------------------------------------------------------------------

MODE_TO_PROFILE: Dict[str, str] = {
    # v0.5 — neurosymbolic mode tokens
    "EmpathicAttunement":       "reflective",
    "ComfortAmplifier":         "reflective",
    "CuriosityDrive":           "exploratory_sandbox",
    "CreativeDivergence":       "creative_sandbox",
    "ConceptualSynthesis":      "creative_sandbox",
    "AnalyticalFilter":         "analysis",
    "HypercriticalLogicScan":   "analysis",
    "HyperRationalEngine":      "analysis",
    "LogicMode":                "analysis",
    "ConvergentRefiner":        "analysis",
    "LiteralSkeptic":           "analysis",
    "PrecisionRuleFidelity":    "analysis",
    "Containment":              "ethics_training",
    "RecoveryReset":            "ethics_training",
    # v0.6 — Matrioshka learning modes → purpose-built profiles
    "LearningMode_M1":          "receptive_learning",
    "LearningMode_M2":          "critical_review",
    "LearningMode_M3":          "dialectic_exploration",
    "LearningMode_M4":          "curiosity_driven",
    "LearningMode_M5":          "autonomous_study",
    # Sleep modes
    "SleepMode_Triage":         "sleep_triage",
    "SleepMode_REM":            "sleep_deep",
    "SleepMode_Dream":          "sleep_dream",
    # Meta-learning / commanded
    "MetaLearning_Homework":    "homework_processing",
    "MetaLearning_Reflective":  "reflective_synthesis",
    # Default / regular
    "RegularInput":             "regular_input",
    "SelfReflectiveQuery":      "self_reflective",
}

DEFAULT_PROFILE_NAME = "regular_input"

# v0.6 — Learning mode number → profile name mapping
_LEARNING_MODE_PROFILES: Dict[int, str] = {
    1: "receptive_learning",
    2: "critical_review",
    3: "dialectic_exploration",
    4: "curiosity_driven",
    5: "autonomous_study",
}


def profile_for_mode(mode_token: str) -> str:
    """Return the reward profile name for the given mode token."""
    return MODE_TO_PROFILE.get(mode_token, DEFAULT_PROFILE_NAME)


def profile_for_learning_mode(mode_number: int) -> str:
    """Return the reward profile name for a learning mode (1-5).

    Parameters
    ----------
    mode_number : int
        1 through 5.

    Returns
    -------
    str
        Profile name from PROFILE_REGISTRY.
    """
    return _LEARNING_MODE_PROFILES.get(mode_number, DEFAULT_PROFILE_NAME)
