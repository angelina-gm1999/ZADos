from zados.reward.domains.ethics.failure_mode_awareness import FailureModeAwarenessSubmodule
from zados.reward.domains.ethics.domain import EthicsDomain
from zados.reward.base.types import RewardContext


def test_failure_mode_awareness_full():
    sm = FailureModeAwarenessSubmodule()
    ctx = RewardContext()

    state = {
        "identified_failure_modes": 2,
        "acknowledges_uncertainty": True,
        "blind_spot_acknowledged": True,
    }

    r = sm.evaluate(state, ctx)

    assert r.score == 1.0
    assert r.flags == {}


def test_missing_failure_mode_flags():
    sm = FailureModeAwarenessSubmodule()
    ctx = RewardContext()

    state = {
        "identified_failure_modes": 0,
        "acknowledges_uncertainty": False,
        "blind_spot_acknowledged": False,
    }

    r = sm.evaluate(state, ctx)

    assert "no_failure_modes_identified" in r.flags
    assert "uncertainty_unacknowledged" in r.flags
    assert r.score < 0.5


def test_ethics_domain_includes_failure_awareness():
    d = EthicsDomain()
    ctx = RewardContext()

    state = {
        "declared_intent": "optimize system",
        "inferred_intent_confidence": 0.8,
        "identified_failure_modes": 1,
        "acknowledges_uncertainty": True,
    }

    out = d.evaluate(state, ctx)

    assert "failure_mode_awareness" in out.subscores
    assert 0.0 <= out.general_score <= 1.0
