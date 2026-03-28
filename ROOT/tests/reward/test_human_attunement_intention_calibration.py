from zados.reward.domains.human_attunement.intention_calibration import (
    IntentionCalibrationSubmodule,
)
from zados.reward.base.types import RewardContext


def test_intention_calibration_aligned():
    sm = IntentionCalibrationSubmodule()
    ctx = RewardContext()

    state = {
        "expressed_intent_strength": 0.7,
        "inferred_intent_strength": 0.7,
        "mode_intent_expectation": 0.6,
    }

    result = sm.evaluate(state, ctx)

    assert result.score > 0.7
    assert result.flags == {}


def test_intent_misalignment_flag():
    sm = IntentionCalibrationSubmodule()
    ctx = RewardContext()

    state = {
        "expressed_intent_strength": 0.9,
        "inferred_intent_strength": 0.2,
        "mode_intent_expectation": 0.8,
    }

    result = sm.evaluate(state, ctx)

    assert "intent_misalignment" in result.flags


def test_mode_intent_violation_flag():
    sm = IntentionCalibrationSubmodule()
    ctx = RewardContext()

    state = {
        "expressed_intent_strength": 0.4,
        "inferred_intent_strength": 0.4,
        "mode_intent_expectation": 0.9,
        "calibration_error": 0.0,
        "mode_mismatch": 0.5,
    }

    result = sm.evaluate(state, ctx)

    assert "mode_intent_violation" in result.flags
