from zados.reward.profile.static_profiles import (
    STATIC_PROFILES,
    PROFILE_REGISTRY,
    DEFAULT_PROFILE,
    REFLECTIVE_PROFILE,
    EXPLORATORY_SANDBOX_PROFILE,
    ETHICS_TRAINING_PROFILE,
    CREATIVE_SANDBOX_PROFILE,
    ANALYSIS_PROFILE,
    RECEPTIVE_LEARNING_PROFILE,
    CRITICAL_REVIEW_PROFILE,
    DIALECTIC_EXPLORATION_PROFILE,
    CURIOSITY_DRIVEN_PROFILE,
    AUTONOMOUS_STUDY_PROFILE,
    HOMEWORK_PROCESSING_PROFILE,
    REFLECTIVE_SYNTHESIS_PROFILE,
    SLEEP_TRIAGE_PROFILE,
    SLEEP_DEEP_PROFILE,
    SLEEP_DREAM_PROFILE,
    REGULAR_INPUT_PROFILE,
    SELF_REFLECTIVE_PROFILE,
)
from zados.reward.profile.base import RewardProfile


# ------------------------------------------------------------------
# Registry completeness
# ------------------------------------------------------------------

def test_all_profiles_exist():
    expected = {
        "reflective",
        "exploratory_sandbox",
        "ethics_training",
        "creative_sandbox",
        "analysis",
        "receptive_learning",
        "critical_review",
        "dialectic_exploration",
        "curiosity_driven",
        "autonomous_study",
        "homework_processing",
        "reflective_synthesis",
        "sleep_triage",
        "sleep_deep",
        "sleep_dream",
        "regular_input",
        "self_reflective",
    }

    assert set(STATIC_PROFILES.keys()) == expected


def test_profile_count():
    assert len(STATIC_PROFILES) == 17


def test_profiles_are_reward_profiles():
    for profile in STATIC_PROFILES.values():
        assert isinstance(profile, RewardProfile)


def test_profile_weights_are_bounded():
    for profile in STATIC_PROFILES.values():
        for v in profile.domain_weights.values():
            assert 0.0 <= v <= 1.0


def test_profile_biases_are_bounded():
    for profile in STATIC_PROFILES.values():
        assert 0.0 <= profile.suppression_bias <= 1.0
        assert 0.0 <= profile.abstention_bias <= 1.0


def test_profile_names_match_registry_keys():
    for name, profile in STATIC_PROFILES.items():
        assert profile.name == name


# ------------------------------------------------------------------
# Backward-compatible aliases
# ------------------------------------------------------------------

def test_profile_registry_alias():
    assert PROFILE_REGISTRY is STATIC_PROFILES


def test_default_profile_alias():
    assert DEFAULT_PROFILE is STATIC_PROFILES["regular_input"]
    assert DEFAULT_PROFILE.name == "regular_input"


# ------------------------------------------------------------------
# Domain weights — 4 required domains present
# ------------------------------------------------------------------

REQUIRED_DOMAINS = {"ethics", "logic", "innovation", "human_attunement"}


def test_all_profiles_have_required_domains():
    for name, profile in STATIC_PROFILES.items():
        assert set(profile.domain_weights.keys()) == REQUIRED_DOMAINS, (
            f"Profile '{name}' missing domains: "
            f"{REQUIRED_DOMAINS - set(profile.domain_weights.keys())}"
        )


# ------------------------------------------------------------------
# Original 5 profiles — spot checks
# ------------------------------------------------------------------

def test_reflective_profile():
    p = REFLECTIVE_PROFILE
    assert p.name == "reflective"
    assert p.domain_weights["ethics"] == 0.9
    assert p.suppression_bias == 0.2
    assert p.abstention_bias == 0.6


def test_analysis_profile_renamed():
    """analysis_investigation was renamed to analysis."""
    p = ANALYSIS_PROFILE
    assert p.name == "analysis"
    assert p.domain_weights["logic"] == 1.0


def test_creative_sandbox_profile():
    p = CREATIVE_SANDBOX_PROFILE
    assert p.name == "creative_sandbox"
    assert p.domain_weights["innovation"] == 1.0
    assert p.suppression_bias == 0.05


# ------------------------------------------------------------------
# New learning mode profiles — spot checks
# ------------------------------------------------------------------

def test_receptive_learning_profile():
    p = RECEPTIVE_LEARNING_PROFILE
    assert p.name == "receptive_learning"
    assert p.domain_weights["human_attunement"] == 0.9
    # Lower suppression for receptive mode
    assert p.suppression_bias < 0.2


def test_critical_review_profile():
    p = CRITICAL_REVIEW_PROFILE
    assert p.name == "critical_review"
    assert p.domain_weights["logic"] == 0.9
    assert p.domain_weights["ethics"] == 0.8


def test_dialectic_exploration_profile():
    p = DIALECTIC_EXPLORATION_PROFILE
    assert p.name == "dialectic_exploration"
    assert p.domain_weights["logic"] == 0.8
    assert p.domain_weights["innovation"] == 0.8


def test_curiosity_driven_profile():
    p = CURIOSITY_DRIVEN_PROFILE
    assert p.name == "curiosity_driven"
    assert p.domain_weights["innovation"] == 0.8


def test_autonomous_study_profile():
    p = AUTONOMOUS_STUDY_PROFILE
    assert p.name == "autonomous_study"
    assert p.domain_weights["logic"] == 0.8


# ------------------------------------------------------------------
# Pipeline profiles — spot checks
# ------------------------------------------------------------------

def test_homework_processing_profile():
    p = HOMEWORK_PROCESSING_PROFILE
    assert p.name == "homework_processing"
    assert p.domain_weights["logic"] == 0.9


def test_reflective_synthesis_profile():
    p = REFLECTIVE_SYNTHESIS_PROFILE
    assert p.name == "reflective_synthesis"
    assert p.domain_weights["ethics"] == 0.8
    assert p.domain_weights["human_attunement"] == 0.8


def test_regular_input_profile():
    p = REGULAR_INPUT_PROFILE
    assert p.name == "regular_input"
    # Balanced — no domain exceeds 0.7
    for v in p.domain_weights.values():
        assert v <= 0.7


# ------------------------------------------------------------------
# Sleep profiles — spot checks
# ------------------------------------------------------------------

def test_sleep_triage_profile():
    p = SLEEP_TRIAGE_PROFILE
    assert p.name == "sleep_triage"
    assert p.suppression_bias < 0.15


def test_sleep_deep_profile():
    p = SLEEP_DEEP_PROFILE
    assert p.name == "sleep_deep"
    assert p.suppression_bias <= 0.1


def test_sleep_dream_profile():
    p = SLEEP_DREAM_PROFILE
    assert p.name == "sleep_dream"
    assert p.domain_weights["innovation"] == 0.9
    # Dream mode = minimal suppression and abstention
    assert p.suppression_bias <= 0.05
    assert p.abstention_bias <= 0.1


def test_self_reflective_profile():
    p = SELF_REFLECTIVE_PROFILE
    assert p.name == "self_reflective"
    assert p.domain_weights["ethics"] == 0.9
