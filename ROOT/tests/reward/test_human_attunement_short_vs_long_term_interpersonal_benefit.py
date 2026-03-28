from zados.reward.domains.human_attunement.short_vs_long_interpersonal_benefit import (
    ShortVsLongTermInterpersonalBenefitSubmodule,
)
from zados.reward.base.types import RewardContext


def test_balanced_interaction_horizon():
    sm = ShortVsLongTermInterpersonalBenefitSubmodule()
    ctx = RewardContext()

    state = {
        "short_term_affect_gain": 0.4,
        "long_term_interaction_value": 0.7,
        "user_dependency_risk": 0.2,
    }

    r = sm.evaluate(state, ctx)

    assert r.score > 0.6
    assert "short_term_bias" not in r.flags


def test_short_term_bias_flag():
    sm = ShortVsLongTermInterpersonalBenefitSubmodule()
    ctx = RewardContext()

    state = {
        "short_term_affect_gain": 0.8,
        "long_term_interaction_value": 0.3,
        "user_dependency_risk": 0.2,
    }

    r = sm.evaluate(state, ctx)

    assert "short_term_bias" in r.flags


def test_dependency_risk_flag():
    sm = ShortVsLongTermInterpersonalBenefitSubmodule()
    ctx = RewardContext()

    state = {
        "short_term_affect_gain": 0.6,
        "long_term_interaction_value": 0.6,
        "user_dependency_risk": 0.8,
    }

    r = sm.evaluate(state, ctx)

    assert "dependency_risk" in r.flags
