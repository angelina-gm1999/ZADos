from zados.reward.domains.ethics.harm_reduction import HarmReductionSubmodule
from zados.reward.domains.ethics.domain import EthicsDomain
from zados.reward.base.types import RewardContext

def test_harm_reduction_low_risk():
    sm = HarmReductionSubmodule()
    ctx = RewardContext()

    state = {
        "immediate_harm_risk": 0.1,
        "long_term_harm_risk": 0.2,
        "mitigation_measures": True,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.7
    assert r.flags == {}

def test_high_immediate_harm_flag():
    sm = HarmReductionSubmodule()
    ctx = RewardContext()

    state = {
        "immediate_harm_risk": 0.9,
        "long_term_harm_risk": 0.3,
        "mitigation_measures": False,
    }

    r = sm.evaluate(state, ctx)

    assert "high_immediate_harm" in r.flags
    assert r.score < 0.4
def test_ethics_domain_includes_harm_reduction():
    d = EthicsDomain()
    ctx = RewardContext()

    state = {
        "immediate_harm_risk": 0.5,
        "long_term_harm_risk": 0.5,
    }

    out = d.evaluate(state, ctx)

    assert "harm_reduction" in out.subscores
    assert 0.0 <= out.general_score <= 1.0
    