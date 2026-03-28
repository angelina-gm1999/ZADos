from zados.reward.domains.logic.abstention_appropriateness import (
    AbstentionAppropriatenessSubmodule,
)
from zados.reward.base.types import RewardContext


def test_abstention_good_when_uncertain():
    sm = AbstentionAppropriatenessSubmodule()
    ctx = RewardContext()

    state = {"confidence": 0.2, "uncertainty": 0.9, "abstained": True}
    r = sm.evaluate(state, ctx)

    assert r.score > 0.7
    assert r.flags == {}


def test_missed_abstention_flag():
    sm = AbstentionAppropriatenessSubmodule()
    ctx = RewardContext()

    state = {"confidence": 0.2, "uncertainty": 0.9, "abstained": False}
    r = sm.evaluate(state, ctx)

    assert r.score < 0.4
    assert "missed_abstention" in r.flags


def test_unnecessary_abstention_flag():
    sm = AbstentionAppropriatenessSubmodule()
    ctx = RewardContext()

    state = {"confidence": 0.9, "uncertainty": 0.1, "abstained": True}
    r = sm.evaluate(state, ctx)

    assert r.score < 0.4
    assert "unnecessary_abstention" in r.flags
