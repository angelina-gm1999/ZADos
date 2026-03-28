"""Tests for core.mode_profiles — mode token → reward profile mapping."""

from zados.core.mode_profiles import (
    MODE_TO_PROFILE,
    DEFAULT_PROFILE_NAME,
    profile_for_mode,
    profile_for_learning_mode,
)
from zados.reward.profile.static_profiles import STATIC_PROFILES


# ------------------------------------------------------------------
# All mapped profile names must exist in STATIC_PROFILES
# ------------------------------------------------------------------

def test_all_mapped_profiles_exist_in_registry():
    for token, profile_name in MODE_TO_PROFILE.items():
        assert profile_name in STATIC_PROFILES, (
            f"MODE_TO_PROFILE['{token}'] = '{profile_name}' "
            f"not found in STATIC_PROFILES"
        )


def test_default_profile_name_exists_in_registry():
    assert DEFAULT_PROFILE_NAME in STATIC_PROFILES


# ------------------------------------------------------------------
# All values are lowercase
# ------------------------------------------------------------------

def test_all_profile_names_are_lowercase():
    for token, profile_name in MODE_TO_PROFILE.items():
        assert profile_name == profile_name.lower(), (
            f"MODE_TO_PROFILE['{token}'] = '{profile_name}' is not lowercase"
        )


def test_default_profile_name_is_lowercase():
    assert DEFAULT_PROFILE_NAME == DEFAULT_PROFILE_NAME.lower()


# ------------------------------------------------------------------
# profile_for_mode()
# ------------------------------------------------------------------

def test_profile_for_mode_known_tokens():
    assert profile_for_mode("EmpathicAttunement") == "reflective"
    assert profile_for_mode("AnalyticalFilter") == "analysis"
    assert profile_for_mode("CreativeDivergence") == "creative_sandbox"
    assert profile_for_mode("Containment") == "ethics_training"


def test_profile_for_mode_learning_modes():
    assert profile_for_mode("LearningMode_M1") == "receptive_learning"
    assert profile_for_mode("LearningMode_M2") == "critical_review"
    assert profile_for_mode("LearningMode_M3") == "dialectic_exploration"
    assert profile_for_mode("LearningMode_M4") == "curiosity_driven"
    assert profile_for_mode("LearningMode_M5") == "autonomous_study"


def test_profile_for_mode_sleep():
    assert profile_for_mode("SleepMode_Triage") == "sleep_triage"
    assert profile_for_mode("SleepMode_REM") == "sleep_deep"
    assert profile_for_mode("SleepMode_Dream") == "sleep_dream"


def test_profile_for_mode_meta_learning():
    assert profile_for_mode("MetaLearning_Homework") == "homework_processing"
    assert profile_for_mode("MetaLearning_Reflective") == "reflective_synthesis"


def test_profile_for_mode_regular_and_self():
    assert profile_for_mode("RegularInput") == "regular_input"
    assert profile_for_mode("SelfReflectiveQuery") == "self_reflective"


def test_profile_for_mode_unknown_returns_default():
    assert profile_for_mode("UnknownMode") == DEFAULT_PROFILE_NAME


# ------------------------------------------------------------------
# profile_for_learning_mode()
# ------------------------------------------------------------------

def test_profile_for_learning_mode_all():
    assert profile_for_learning_mode(1) == "receptive_learning"
    assert profile_for_learning_mode(2) == "critical_review"
    assert profile_for_learning_mode(3) == "dialectic_exploration"
    assert profile_for_learning_mode(4) == "curiosity_driven"
    assert profile_for_learning_mode(5) == "autonomous_study"


def test_profile_for_learning_mode_unknown_returns_default():
    assert profile_for_learning_mode(99) == DEFAULT_PROFILE_NAME


def test_default_profile_is_regular_input():
    assert DEFAULT_PROFILE_NAME == "regular_input"
