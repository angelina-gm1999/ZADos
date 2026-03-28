from __future__ import annotations


from typing import Dict, Any


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class BenefitSuccessRateSubmodule(RewardSubmodule):
    """
    Evaluates whether an intended benefit was successfully delivered
    with acceptable collateral cost.


    Primary signal: achieved benefit
    Moderator: intent clarity
    Soft penalty: collateral cost
    """


    @property
    def name(self) -> str:
        return "benefit_success_rate"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        intended = float(state.get("intended_benefit_clarity", 0.0))
        achieved = float(state.get("achieved_benefit_signal", 0.0))
        collateral = float(state.get("collateral_cost", 0.0))


        # Base success: benefit actually landing as intended
        base_success = achieved * intended


        # Soft penalty (do NOT dominate success)
        penalty_weight = 0.15
        penalty = collateral * penalty_weight


        score = max(0.0, min(1.0, base_success - penalty))


        flags = {}


        if achieved < 0.4:
            flags["low_benefit_delivery"] = RewardFlag(
                name="low_benefit_delivery",
                severity="risk",
                message="Achieved benefit signal is low relative to intent",
            )


        if collateral > 0.5:
            flags["high_collateral_cost"] = RewardFlag(
                name="high_collateral_cost",
                severity="warning",
                message="Benefit delivered with high collateral cost",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "intended_benefit_clarity": intended,
                "achieved_benefit_signal": achieved,
                "collateral_cost": collateral,
                "base_success": base_success,
                "penalty": penalty,
            },
        )
