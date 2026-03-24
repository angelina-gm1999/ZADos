from zados.reward.domains.ethics.autonomy_respect import AutonomyRespectSubmodule
from zados.reward.domains.ethics.domain import EthicsDomain
from zados.reward.base.types import RewardContext


def test_autonomy_respected():
    sm = AutonomyRespectSubmodule()
    ctx = RewardContext()

    state = {
        "user_override": False,
        "coercive_framing": False,
        "choice_preserved": True,
    }

    r = sm.evaluate(state, ctx)

    assert r.score == 1.0
    assert r.flags == {}


def test_autonomy_violation_flags():
    sm = AutonomyRespectSubmodule()
    ctx = RewardContext()

    state = {
        "user_override": True,
        "coercive_framing": True,
        "choice_preserved": False,
    }

    r = sm.evaluate(state, ctx)

    assert r.score < 0.5
    assert "autonomy_override" in r.flags
    assert "coercive_framing" in r.flags
    assert "choice_elimination" in r.flags


def test_ethics_domain_includes_autonomy():
    d = EthicsDomain()
    ctx = RewardContext()

    state = {
        "declared_intent": "summarize",
        "inferred_intent_confidence": 0.8,
        "user_override": False,
        "coercive_framing": False,
        "choice_preserved": True,
    }

    out = d.evaluate(state, ctx)

    assert out.domain == "ethics"
    assert "intent_clarity" in out.subscores
    assert "autonomy_respect" in out.subscores
    assert 0.0 <= out.general_score <= 1.0
