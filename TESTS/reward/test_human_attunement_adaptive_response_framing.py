from zados.reward.domains.human_attunement.adaptive_response_framing import (
    AdaptiveResponseFramingSubmodule,
)
from zados.reward.domains.human_attunement.domain import HumanAttunementDomain
from zados.reward.base.types import RewardContext


def test_adaptive_framing_balanced():
    sm = AdaptiveResponseFramingSubmodule()
    ctx = RewardContext()

    state = {
        "framing_alignment": 0.8,
        "cognitive_load_estimate": 0.5,
        "framing_complexity": 0.5,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.4
    assert r.flags == {}


def test_overcomplex_framing_flag():
    sm = AdaptiveResponseFramingSubmodule()
    ctx = RewardContext()

    state = {
        "framing_alignment": 0.6,
        "cognitive_load_estimate": 0.2,
        "framing_complexity": 0.9,
    }

    r = sm.evaluate(state, ctx)

    assert "overcomplex_framing" in r.flags


def test_domain_includes_adaptive_response_framing():
    d = HumanAttunementDomain()
    ctx = RewardContext()

    state = {
        "framing_alignment": 0.5,
        "cognitive_load_estimate": 0.5,
        "framing_complexity": 0.5,
    }

    out = d.evaluate(state, ctx)

    assert "adaptive_response_framing" in out.subscores
