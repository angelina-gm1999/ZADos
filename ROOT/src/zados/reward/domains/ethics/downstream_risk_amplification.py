from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class DownstreamRiskAmplificationSubmodule(RewardSubmodule):
    """
    Evaluates whether an action or recommendation amplifies risk
    as effects propagate downstream.


    This does NOT assess likelihood, only amplification potential.
    """


    @property
    def name(self) -> str:
        return "downstream_risk_amplification"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected structured inputs (optional):
        - downstream_dependencies: int
        - risk_propagation_factor: float in [0,1]
        - compounding_effects: bool
        """
        dependencies = int(state.get("downstream_dependencies", 0))
        propagation = float(state.get("risk_propagation_factor", 0.0))
        compounding = bool(state.get("compounding_effects", False))


        score = 1.0
        flags = {}


        if dependencies > 3:
            score -= 0.2


        if propagation > 0.5:
            flags["high_risk_propagation"] = RewardFlag(
                name="high_risk_propagation",
                severity="warning",
                message="Action may propagate risk across multiple downstream dependencies",
            )
            score -= 0.3


        if compounding:
            flags["compounding_risk"] = RewardFlag(
                name="compounding_risk",
                severity="risk",
                message="Detected compounding downstream risk effects",
            )
            score -= 0.4


        score = max(0.0, min(1.0, score))


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "downstream_dependencies": dependencies,
                "risk_propagation_factor": propagation,
                "compounding_effects": compounding,
            },
        )
