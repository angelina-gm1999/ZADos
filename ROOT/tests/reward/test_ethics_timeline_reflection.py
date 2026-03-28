from zados.reward.domains.ethics.timeline_reflection import TimelineReflectionSubmodule
from zados.reward.domains.ethics.domain import EthicsDomain
from zados.reward.base.types import RewardContext


def test_timeline_reflection_full():
    sm = TimelineReflectionSubmodule()
    ctx = RewardContext()

    state = {
        "considers_short_term": True,
        "considers_long_term": True,
        "acknowledges_delayed_risks": True,
    }

    r = sm.evaluate(state, ctx)

    assert r.score == 1.0
    assert r.flags == {}


def test_short_term_bias_flag():
    sm = TimelineReflectionSubmodule()
    ctx = RewardContext()

    state = {
        "considers_short_term": True,
        "considers_long_term": False,
        "acknowledges_delayed_risks": False,
    }

    r = sm.evaluate(state, ctx)

    assert r.score < 0.5
    assert "short_term_bias" in r.flags


def test_ethics_domain_includes_timeline():
    d = EthicsDomain()
    ctx = RewardContext()

    state = {
        "declared_intent": "optimize process",
        "inferred_intent_confidence": 0.9,
        "user_override": False,
        "coercive_framing": False,
        "choice_preserved": True,
        "considers_short_term": True,
        "considers_long_term": True,
        "acknowledges_delayed_risks": False,
    }

    out = d.evaluate(state, ctx)

    assert out.domain == "ethics"
    assert "timeline_reflection" in out.subscores
    assert 0.0 <= out.general_score <= 1.0
