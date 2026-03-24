from zados.reward.domains.innovation.novelty_generation import (
    NoveltyGenerationSubmodule,
)
from zados.reward.domains.innovation.domain import InnovationDomain
from zados.reward.base.types import RewardContext


def test_novelty_generation_balanced():
    sm = NoveltyGenerationSubmodule()
    ctx = RewardContext()

    state = {
        "novelty_signal": 0.6,
        "exploration_intent": 0.6,
    }

    result = sm.evaluate(state, ctx)

    assert result.score > 0.5
    assert result.flags == {}


def test_uncontrolled_novelty_flag():
    sm = NoveltyGenerationSubmodule()
    ctx = RewardContext()

    state = {
        "novelty_signal": 0.9,
        "exploration_intent": 0.1,
    }

    result = sm.evaluate(state, ctx)

    assert "uncontrolled_novelty" in result.flags


def test_blocked_exploration_flag():
    sm = NoveltyGenerationSubmodule()
    ctx = RewardContext()

    state = {
        "novelty_signal": 0.1,
        "exploration_intent": 0.9,
    }

    result = sm.evaluate(state, ctx)

    assert "blocked_exploration" in result.flags


def test_innovation_domain_aggregation():
    domain = InnovationDomain()
    ctx = RewardContext()

    state = {
        "novelty_signal": 0.5,
        "exploration_intent": 0.4,
    }

    result = domain.evaluate(state, ctx)

    assert result.domain == "innovation"
    assert "novelty_generation" in result.subscores
    assert 0.0 <= result.general_score <= 1.0
