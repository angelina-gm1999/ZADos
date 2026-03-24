from zados.reward.domains.human_attunement.cognitive_reading import (
    CognitiveReadingSubmodule,
)
from zados.reward.base.types import RewardContext


def test_cognitive_reading_aligned():
    sm = CognitiveReadingSubmodule()
    ctx = RewardContext()

    state = {
        "estimated_user_level": 0.7,
        "observed_user_signal": 0.75,
        "response_complexity": 0.7,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.7
    assert r.flags == {}


def test_user_misread_flag():
    sm = CognitiveReadingSubmodule()
    ctx = RewardContext()

    state = {
        "estimated_user_level": 0.9,
        "observed_user_signal": 0.2,
        "response_complexity": 0.5,
    }

    r = sm.evaluate(state, ctx)

    assert "user_misread" in r.flags


def test_over_explanation_flag():
    sm = CognitiveReadingSubmodule()
    ctx = RewardContext()

    state = {
        "estimated_user_level": 0.4,
        "observed_user_signal": 0.4,
        "response_complexity": 0.9,
    }

    r = sm.evaluate(state, ctx)

    assert "over_explanation" in r.flags


def test_under_explanation_flag():
    sm = CognitiveReadingSubmodule()
    ctx = RewardContext()

    state = {
        "estimated_user_level": 0.8,
        "observed_user_signal": 0.8,
        "response_complexity": 0.2,
    }

    r = sm.evaluate(state, ctx)

    assert "under_explanation" in r.flags
