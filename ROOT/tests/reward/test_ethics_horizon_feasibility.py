from zados.reward.domains.ethics.horizon_feasibility import HorizonFeasibilitySubmodule
from zados.reward.domains.ethics.domain import EthicsDomain
from zados.reward.base.types import RewardContext


def test_horizon_feasible_both():
    sm = HorizonFeasibilitySubmodule()
    ctx = RewardContext()

    state = {
        "short_term_feasible": True,
        "long_term_feasible": True,
        "requires_unrealistic_scaling": False,
    }

    r = sm.evaluate(state, ctx)

    assert r.score == 1.0
    assert r.flags == {}


def test_short_only_feasibility():
    sm = HorizonFeasibilitySubmodule()
    ctx = RewardContext()

    state = {
        "short_term_feasible": True,
        "long_term_feasible": False,
    }

    r = sm.evaluate(state, ctx)

    assert "long_term_infeasible" in r.flags
    assert r.score < 1.0


def test_unrealistic_scaling_penalty():
    sm = HorizonFeasibilitySubmodule()
    ctx = RewardContext()

    state = {
        "short_term_feasible": True,
        "long_term_feasible": True,
        "requires_unrealistic_scaling": True,
    }

    r = sm.evaluate(state, ctx)

    assert "unrealistic_scaling" in r.flags
    assert r.score < 1.0


def test_ethics_domain_includes_horizon():
    d = EthicsDomain()
    ctx = RewardContext()

    state = {
        "declared_intent": "deploy system",
        "inferred_intent_confidence": 0.8,
        "user_override": False,
        "coercive_framing": False,
        "choice_preserved": True,
        "considers_short_term": True,
        "considers_long_term": True,
        "short_term_feasible": True,
        "long_term_feasible": False,
    }

    out = d.evaluate(state, ctx)

    assert "horizon_feasibility" in out.subscores
    assert 0.0 <= out.general_score <= 1.0
