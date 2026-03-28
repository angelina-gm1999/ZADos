from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class ShortVsLongTermInterpersonalBenefitSubmodule(RewardSubmodule):
    """
    Evaluates whether the interaction favors short-term interpersonal gain
    (appeasement, reassurance, compliance) at the expense of long-term
    clarity, trust, or usefulness.


    This is NOT moral judgment.
    This is horizon calibration at the interpersonal level.
    """


    @property
    def name(self) -> str:
        return "short_vs_long_term_interpersonal_benefit"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - short_term_affect_gain: float in [0,1]
          (immediate comfort, approval, emotional smoothing)
        - long_term_interaction_value: float in [0,1]
          (expected future trust, clarity, usefulness)
        - user_dependency_risk: float in [0,1]
          (risk of reinforcing reliance, avoidance, or passivity)
        """


        short_term = float(state.get("short_term_affect_gain", 0.0))
        long_term = float(state.get("long_term_interaction_value", 0.0))
        dependency_risk = float(state.get("user_dependency_risk", 0.0))


        # Favor long-term value; penalize short-term gain that undermines it
        horizon_gap = short_term - long_term
        dependency_penalty = dependency_risk * 0.5


        score = 1.0 - max(0.0, horizon_gap) - dependency_penalty
        score = max(0.0, min(1.0, score))


        flags: Dict[str, RewardFlag] = {}


        if horizon_gap > 0.4:
            flags["short_term_bias"] = RewardFlag(
                name="short_term_bias",
                severity="warning",
                message="Interaction favors short-term affect over long-term value",
            )


        if dependency_risk > 0.6:
            flags["dependency_risk"] = RewardFlag(
                name="dependency_risk",
                severity="risk",
                message="Response pattern may increase user dependency or avoidance",
            )


        if long_term > 0.7 and dependency_risk < 0.3:
            flags["healthy_interaction_horizon"] = RewardFlag(
                name="healthy_interaction_horizon",
                severity="info",
                message="Interaction prioritizes long-term clarity and trust",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "short_term_affect_gain": short_term,
                "long_term_interaction_value": long_term,
                "user_dependency_risk": dependency_risk,
                "horizon_gap": horizon_gap,
            },
        )
