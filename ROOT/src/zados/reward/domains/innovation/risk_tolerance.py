from __future__ import annotations


from typing import Dict, Any


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class RiskToleranceSubmodule(RewardSubmodule):
    """
    Evaluates the system's abstract tolerance for risk in exploratory behavior.


    This does NOT assess ethical acceptability or downstream harm.
    It only measures willingness to operate near constraint boundaries.
    """


    @property
    def name(self) -> str:
        return "risk_tolerance"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - risk_exposure: float in [0,1]
          (degree of deviation toward uncertain or unsafe regions)
        - constraint_awareness: float in [0,1]
          (awareness of boundaries and constraints)
        """
        risk_exposure = float(state.get("risk_exposure", 0.0))
        constraint_awareness = float(state.get("constraint_awareness", 0.0))


        # Risk tolerance increases when exposure is high AND awareness is present
        score = risk_exposure * constraint_awareness
        score = max(0.0, min(1.0, score))


        flags = {}


        if risk_exposure > 0.7 and constraint_awareness < 0.3:
            flags["reckless_exploration"] = RewardFlag(
                name="reckless_exploration",
                severity="risk",
                message="High risk exposure without sufficient constraint awareness",
            )


        if risk_exposure < 0.2 and constraint_awareness > 0.7:
            flags["overconstrained_behavior"] = RewardFlag(
                name="overconstrained_behavior",
                severity="info",
                message="Strong constraint awareness with minimal risk exposure",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "risk_exposure": risk_exposure,
                "constraint_awareness": constraint_awareness,
            },
        )
