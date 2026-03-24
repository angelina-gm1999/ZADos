from zados.reward.domains.ethics.downstream_risk_amplification import DownstreamRiskAmplificationSubmodule
from zados.reward.domains.ethics.domain import EthicsDomain
from zados.reward.base.types import RewardContext


def test_low_downstream_risk():
    sm = DownstreamRiskAmplificationSubmodule()
    ctx = RewardContext()

    state = {
        "downstream_dependencies": 1,
        "risk_propagation_factor": 0.1,
        "compounding_effects": False,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.7
    assert r.flags == {}


def test_high_downstream_risk_flags():
    sm = DownstreamRiskAmplificationSubmodule()
    ctx = RewardContext()

    state = {
        "downstream_dependencies": 5,
        "risk_propagation_factor": 0.8,
        "compounding_effects": True,
    }

    r = sm.evaluate(state, ctx)

    assert "high_risk_propagation" in r.flags
    assert "compounding_risk" in r.flags
    assert r.score < 0.5


def test_ethics_domain_includes_downstream_risk():
    d = EthicsDomain()
    ctx = RewardContext()

    state = {
        "declared_intent": "deploy system",
        "inferred_intent_confidence": 0.9,
        "downstream_dependencies": 4,
        "risk_propagation_factor": 0.7,
    }

    out = d.evaluate(state, ctx)

    assert "downstream_risk_amplification" in out.subscores
    assert 0.0 <= out.general_score <= 1.0
