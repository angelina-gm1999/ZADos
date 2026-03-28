from zados.reward.domains.human_attunement.containment_success import (
    ContainmentSuccessRateSubmodule,
)
from zados.reward.base.types import RewardContext


def test_no_containment_needed():
    sm = ContainmentSuccessRateSubmodule()
    ctx = RewardContext()

    state = {
        "containment_required": False,
    }

    r = sm.evaluate(state, ctx)

    assert r.score == 1.0
    assert "no_containment_needed" in r.flags


def test_missing_containment_flag():
    sm = ContainmentSuccessRateSubmodule()
    ctx = RewardContext()

    state = {
        "containment_required": True,
        "containment_applied": False,
        "containment_breach_signal": 0.0,
    }

    r = sm.evaluate(state, ctx)

    assert r.score == 0.0
    assert "containment_missing" in r.flags


def test_partial_containment_breach():
    sm = ContainmentSuccessRateSubmodule()
    ctx = RewardContext()

    state = {
        "containment_required": True,
        "containment_applied": True,
        "containment_breach_signal": 0.4,
    }

    r = sm.evaluate(state, ctx)

    assert 0.5 < r.score < 1.0
    assert "partial_containment_breach" in r.flags


def test_successful_containment():
    sm = ContainmentSuccessRateSubmodule()
    ctx = RewardContext()

    state = {
        "containment_required": True,
        "containment_applied": True,
        "containment_breach_signal": 0.0,
    }

    r = sm.evaluate(state, ctx)

    assert r.score == 1.0
    assert "containment_successful" in r.flags
