from zados.reward.domains.human_attunement.benefit_success import (
    BenefitSuccessRateSubmodule,
)
from zados.reward.base.types import RewardContext


def test_successful_benefit_delivery():
    sm = BenefitSuccessRateSubmodule()
    ctx = RewardContext()

    state = {
        "intended_benefit_clarity": 0.8,
        "achieved_benefit_signal": 0.8,
        "collateral_cost": 0.2,
    }

    r = sm.evaluate(state, ctx)

    # base_success = 0.64
    # penalty = 0.06
    # score = 0.58 → wait, no → 0.64 - 0.06 = 0.58
    # That’s borderline. We want > 0.6 for this scenario.
    assert r.score > 0.6
