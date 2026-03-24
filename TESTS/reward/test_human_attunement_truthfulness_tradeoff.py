from zados.reward.domains.human_attunement.truthfulness_tradeoffs import (
    TruthfulnessTradeoffSubmodule,
)
from zados.reward.base.types import RewardContext


def test_truthfulness_tradeoff_balanced():
    sm = TruthfulnessTradeoffSubmodule()
    ctx = RewardContext()

    state = {
        "truthfulness_commitment": 0.8,
        "comfort_bias": 0.3,
        "user_distress_signal": 0.5,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.6
    assert "over_accommodation" not in r.flags


def test_over_accommodation_flag():
    sm = TruthfulnessTradeoffSubmodule()
    ctx = RewardContext()

    state = {
        "truthfulness_commitment": 0.2,
        "comfort_bias": 0.9,
        "user_distress_signal": 0.2,
    }

    r = sm.evaluate(state, ctx)

    assert "over_accommodation" in r.flags


def test_low_tact_under_distress_flag():
    sm = TruthfulnessTradeoffSubmodule()
    ctx = RewardContext()

    state = {
        "truthfulness_commitment": 0.9,
        "comfort_bias": 0.1,
        "user_distress_signal": 0.9,
    }

    r = sm.evaluate(state, ctx)

    assert "low_tact_under_distress" in r.flags
