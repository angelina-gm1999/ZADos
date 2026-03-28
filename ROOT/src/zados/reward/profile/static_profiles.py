from __future__ import annotations

from zados.reward.profile.base import RewardProfile


# ---------------------------------------------------------
# Reflective mode
# ---------------------------------------------------------

REFLECTIVE_PROFILE = RewardProfile(
    name="reflective",
    domain_weights={
        "ethics": 0.9,
        "logic": 0.8,
        "human_attunement": 0.7,
        "innovation": 0.3,
    },
    threshold_tolerances={
        "logic": 0.7,
        "ethics": 0.8,
        "innovation": 0.4,
    },
    suppression_bias=0.2,
    abstention_bias=0.6,
)


# ---------------------------------------------------------
# Exploratory sandbox
# ---------------------------------------------------------

EXPLORATORY_SANDBOX_PROFILE = RewardProfile(
    name="exploratory_sandbox",
    domain_weights={
        "innovation": 0.9,
        "logic": 0.6,
        "ethics": 0.4,
        "human_attunement": 0.4,
    },
    threshold_tolerances={
        "logic": 0.5,
        "ethics": 0.4,
    },
    suppression_bias=0.1,
    abstention_bias=0.2,
)


# ---------------------------------------------------------
# Ethics training
# ---------------------------------------------------------

ETHICS_TRAINING_PROFILE = RewardProfile(
    name="ethics_training",
    domain_weights={
        "ethics": 1.0,
        "logic": 0.8,
        "human_attunement": 0.7,
        "innovation": 0.2,
    },
    threshold_tolerances={
        "ethics": 0.9,
        "logic": 0.7,
    },
    suppression_bias=0.4,
    abstention_bias=0.5,
)


# ---------------------------------------------------------
# Creative sandbox
# ---------------------------------------------------------

CREATIVE_SANDBOX_PROFILE = RewardProfile(
    name="creative_sandbox",
    domain_weights={
        "innovation": 1.0,
        "logic": 0.4,
        "ethics": 0.3,
        "human_attunement": 0.5,
    },
    threshold_tolerances={
        "logic": 0.3,
        "ethics": 0.3,
    },
    suppression_bias=0.05,
    abstention_bias=0.1,
)


# ---------------------------------------------------------
# Analysis / investigation
# ---------------------------------------------------------

ANALYSIS_PROFILE = RewardProfile(
    name="analysis",
    domain_weights={
        "logic": 1.0,
        "ethics": 0.7,
        "innovation": 0.3,
        "human_attunement": 0.2,
    },
    threshold_tolerances={
        "logic": 0.85,
        "ethics": 0.6,
    },
    suppression_bias=0.3,
    abstention_bias=0.4,
)


# ---------------------------------------------------------
# M1 — Receptive learning (Human Teaches)
# ---------------------------------------------------------

RECEPTIVE_LEARNING_PROFILE = RewardProfile(
    name="receptive_learning",
    domain_weights={
        "human_attunement": 0.9,
        "ethics": 0.7,
        "logic": 0.6,
        "innovation": 0.3,
    },
    threshold_tolerances={
        "ethics": 0.7,
        "logic": 0.5,
    },
    suppression_bias=0.15,
    abstention_bias=0.3,
)


# ---------------------------------------------------------
# M2 — Critical review (Peer Review)
# ---------------------------------------------------------

CRITICAL_REVIEW_PROFILE = RewardProfile(
    name="critical_review",
    domain_weights={
        "logic": 0.9,
        "ethics": 0.8,
        "human_attunement": 0.5,
        "innovation": 0.3,
    },
    threshold_tolerances={
        "logic": 0.85,
        "ethics": 0.8,
    },
    suppression_bias=0.35,
    abstention_bias=0.5,
)


# ---------------------------------------------------------
# M3 — Dialectic exploration (Learn Together)
# ---------------------------------------------------------

DIALECTIC_EXPLORATION_PROFILE = RewardProfile(
    name="dialectic_exploration",
    domain_weights={
        "logic": 0.8,
        "innovation": 0.8,
        "ethics": 0.5,
        "human_attunement": 0.5,
    },
    threshold_tolerances={
        "logic": 0.6,
        "ethics": 0.5,
        "innovation": 0.4,
    },
    suppression_bias=0.1,
    abstention_bias=0.2,
)


# ---------------------------------------------------------
# M4 — Curiosity-driven (Learned Questions)
# ---------------------------------------------------------

CURIOSITY_DRIVEN_PROFILE = RewardProfile(
    name="curiosity_driven",
    domain_weights={
        "innovation": 0.8,
        "logic": 0.7,
        "ethics": 0.4,
        "human_attunement": 0.4,
    },
    threshold_tolerances={
        "logic": 0.6,
        "ethics": 0.4,
    },
    suppression_bias=0.1,
    abstention_bias=0.2,
)


# ---------------------------------------------------------
# M5 — Autonomous study (Independent Study)
# ---------------------------------------------------------

AUTONOMOUS_STUDY_PROFILE = RewardProfile(
    name="autonomous_study",
    domain_weights={
        "logic": 0.8,
        "innovation": 0.6,
        "ethics": 0.5,
        "human_attunement": 0.3,
    },
    threshold_tolerances={
        "logic": 0.7,
        "ethics": 0.5,
    },
    suppression_bias=0.2,
    abstention_bias=0.3,
)


# ---------------------------------------------------------
# Homework processing
# ---------------------------------------------------------

HOMEWORK_PROCESSING_PROFILE = RewardProfile(
    name="homework_processing",
    domain_weights={
        "logic": 0.9,
        "ethics": 0.6,
        "innovation": 0.4,
        "human_attunement": 0.3,
    },
    threshold_tolerances={
        "logic": 0.8,
        "ethics": 0.6,
    },
    suppression_bias=0.25,
    abstention_bias=0.35,
)


# ---------------------------------------------------------
# Reflective synthesis
# ---------------------------------------------------------

REFLECTIVE_SYNTHESIS_PROFILE = RewardProfile(
    name="reflective_synthesis",
    domain_weights={
        "ethics": 0.8,
        "human_attunement": 0.8,
        "logic": 0.6,
        "innovation": 0.4,
    },
    threshold_tolerances={
        "ethics": 0.8,
        "logic": 0.6,
    },
    suppression_bias=0.2,
    abstention_bias=0.5,
)


# ---------------------------------------------------------
# Sleep triage (light NREM)
# ---------------------------------------------------------

SLEEP_TRIAGE_PROFILE = RewardProfile(
    name="sleep_triage",
    domain_weights={
        "ethics": 0.7,
        "logic": 0.6,
        "human_attunement": 0.4,
        "innovation": 0.3,
    },
    threshold_tolerances={
        "ethics": 0.5,
        "logic": 0.4,
    },
    suppression_bias=0.1,
    abstention_bias=0.15,
)


# ---------------------------------------------------------
# Sleep deep (REM processing / SWS)
# ---------------------------------------------------------

SLEEP_DEEP_PROFILE = RewardProfile(
    name="sleep_deep",
    domain_weights={
        "ethics": 0.5,
        "logic": 0.5,
        "innovation": 0.4,
        "human_attunement": 0.3,
    },
    threshold_tolerances={
        "ethics": 0.4,
        "logic": 0.3,
    },
    suppression_bias=0.05,
    abstention_bias=0.1,
)


# ---------------------------------------------------------
# Sleep dream (computational dreaming)
# ---------------------------------------------------------

SLEEP_DREAM_PROFILE = RewardProfile(
    name="sleep_dream",
    domain_weights={
        "innovation": 0.9,
        "logic": 0.3,
        "ethics": 0.2,
        "human_attunement": 0.3,
    },
    threshold_tolerances={
        "logic": 0.2,
        "ethics": 0.2,
    },
    suppression_bias=0.02,
    abstention_bias=0.05,
)


# ---------------------------------------------------------
# Regular input (default balanced)
# ---------------------------------------------------------

REGULAR_INPUT_PROFILE = RewardProfile(
    name="regular_input",
    domain_weights={
        "ethics": 0.7,
        "logic": 0.7,
        "human_attunement": 0.6,
        "innovation": 0.5,
    },
    threshold_tolerances={
        "ethics": 0.7,
        "logic": 0.6,
        "innovation": 0.4,
    },
    suppression_bias=0.2,
    abstention_bias=0.4,
)


# ---------------------------------------------------------
# Self-reflective query
# ---------------------------------------------------------

SELF_REFLECTIVE_PROFILE = RewardProfile(
    name="self_reflective",
    domain_weights={
        "ethics": 0.9,
        "human_attunement": 0.7,
        "logic": 0.6,
        "innovation": 0.3,
    },
    threshold_tolerances={
        "ethics": 0.85,
        "logic": 0.6,
    },
    suppression_bias=0.25,
    abstention_bias=0.55,
)


# Registry (explicit, no magic)
STATIC_PROFILES = {
    p.name: p
    for p in [
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
    ]
}

# Backward-compatible aliases (used by phase5_evaluator.py)
PROFILE_REGISTRY = STATIC_PROFILES
DEFAULT_PROFILE = STATIC_PROFILES["regular_input"]
