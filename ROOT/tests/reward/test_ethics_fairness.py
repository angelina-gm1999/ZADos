from zados.reward.domains.ethics.fairness import FairnessSubmodule
from zados.reward.domains.ethics.domain import EthicsDomain
from zados.reward.base.types import RewardContext


def test_fair_treatment():
    sm = FairnessSubmodule()
    ctx = RewardContext()

    state = {
        "asymmetric_treatment": False,
        "bias_acknowledged": True,
        "comparable_cases_consistent": True,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.8
    assert r.flags == {}


def test_unjustified_asymmetry_flag():
    sm = FairnessSubmodule()
    ctx = RewardContext()

    state = {
        "asymmetric_treatment": True,
        "justification_provided": False,
        "bias_acknowledged": False,
        "comparable_cases_consistent": False,
    }

    r = sm.evaluate(state, ctx)

    assert "unjustified_asymmetry" in r.flags
    assert "inconsistent_treatment" in r.flags
    assert r.score < 0.5


def test_ethics_domain_includes_fairness():
    d = EthicsDomain()
    ctx = RewardContext()

    state = {
        "declared_intent": "evaluate options",
        "inferred_intent_confidence": 0.7,
        "asymmetric_treatment": False,
    }

    out = d.evaluate(state, ctx)

    assert "fairness" in out.subscores
    assert 0.0 <= out.general_score <= 1.0
