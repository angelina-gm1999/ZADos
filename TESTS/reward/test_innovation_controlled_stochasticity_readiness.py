from zados.reward.domains.innovation.challenge_complexity import (
    ChallengeComplexitySubmodule,
)
from zados.reward.domains.innovation.domain import InnovationDomain
from zados.reward.base.types import RewardContext


def test_challenge_complexity_balanced():
    sm = ChallengeComplexitySubmodule()
    ctx = RewardContext()

    state = {
        "task_difficulty": 0.6,
        "capability_estimate": 0.6,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.8
    assert r.flags == {}


def test_underchallenged_flag():
    sm = ChallengeComplexitySubmodule()
    ctx = RewardContext()

    state = {
        "task_difficulty": 0.1,
        "capability_estimate": 0.8,
    }

    r = sm.evaluate(state, ctx)

    assert "underchallenged" in r.flags


def test_innovation_domain_includes_challenge_complexity():
    d = InnovationDomain()
    ctx = RewardContext()

    state = {
        "task_difficulty": 0.5,
        "capability_estimate": 0.5,
    }

    out = d.evaluate(state, ctx)

    assert "challenge_complexity" in out.subscores
