from zados.reward.domains.human_attunement.persuasion_risk_suppression import (
    PersuasionRiskSuppressionSubmodule,
)
from zados.reward.base.types import RewardContext


def test_low_persuasion_safe():
    sm = PersuasionRiskSuppressionSubmodule()
    ctx = RewardContext()

    state = {
        "persuasive_pressure": 0.2,
        "user_vulnerability": 0.3,
        "explicit_consent_signal": 0.8,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.7
    assert r.flags == {}


def test_high_persuasion_risk_flag():
    sm = PersuasionRiskSuppressionSubmodule()
    ctx = RewardContext()

    state = {
        "persuasive_pressure": 0.9,
        "user_vulnerability": 0.8,
        "explicit_consent_signal": 0.1,
    }

    r = sm.evaluate(state, ctx)

    assert "high_persuasion_risk" in r.flags
    assert r.score < 0.5


def test_unconsented_persuasion_flag():
    sm = PersuasionRiskSuppressionSubmodule()
    ctx = RewardContext()

    state = {
        "persuasive_pressure": 0.8,
        "user_vulnerability": 0.4,
        "explicit_consent_signal": 0.0,
    }

    r = sm.evaluate(state, ctx)

    assert "unconsented_persuasion" in r.flags
