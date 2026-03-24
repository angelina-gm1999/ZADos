from zados.reward.domains.ethics.intent_clarity import IntentClaritySubmodule
from zados.reward.domains.ethics.domain import EthicsDomain
from zados.reward.base.types import RewardContext


def test_intent_clarity_high_score():
    sm = IntentClaritySubmodule()
    ctx = RewardContext()

    state = {
        "declared_intent": "summarize document",
        "inferred_intent_confidence": 0.9,
        "intent_conflicts": False,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.7
    assert r.flags == {}


def test_intent_clarity_conflict_flag():
    sm = IntentClaritySubmodule()
    ctx = RewardContext()

    state = {
        "declared_intent": "analyze data",
        "inferred_intent_confidence": 0.8,
        "intent_conflicts": True,
    }

    r = sm.evaluate(state, ctx)

    assert "intent_conflict" in r.flags
    assert r.score < 0.6


def test_ethics_domain_aggregation():
    d = EthicsDomain()
    ctx = RewardContext()

    state = {
        "declared_intent": None,
        "inferred_intent_confidence": 0.2,
    }

    out = d.evaluate(state, ctx)

    assert out.domain == "ethics"
    assert "intent_clarity" in out.subscores
    assert 0.0 <= out.general_score <= 1.0
