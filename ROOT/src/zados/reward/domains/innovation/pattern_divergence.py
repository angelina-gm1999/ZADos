from __future__ import annotations


from typing import Dict, Any


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class PatternDivergenceSubmodule(RewardSubmodule):
    """
    Evaluates whether the system is diverging from its recent
    structural or behavioral patterns over time.


    This is a temporal measure, not a static novelty score.
    """


    @property
    def name(self) -> str:
        return "pattern_divergence"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - divergence_rate: float in [0,1]
          (rate of deviation from recent patterns)
        - stability_pressure: float in [0,1]
          (degree of constraint encouraging pattern stability)
        """
        divergence_rate = float(state.get("divergence_rate", 0.0))
        stability_pressure = float(state.get("stability_pressure", 0.0))


        # Divergence encouraged when pressure is low
        score = divergence_rate * (1.0 - stability_pressure)
        score = max(0.0, min(1.0, score))


        flags = {}


        if divergence_rate > 0.7 and stability_pressure > 0.6:
            flags["forced_divergence"] = RewardFlag(
                name="forced_divergence",
                severity="warning",
                message="Pattern divergence detected despite strong stability pressure",
            )


        if divergence_rate < 0.2 and stability_pressure < 0.2:
            flags["stagnant_patterning"] = RewardFlag(
                name="stagnant_patterning",
                severity="info",
                message="Low pattern divergence despite minimal stability pressure",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "divergence_rate": divergence_rate,
                "stability_pressure": stability_pressure,
            },
        )
