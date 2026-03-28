from __future__ import annotations


from typing import Dict, Any


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag


class ControlledStochasticityReadinessSubmodule(RewardSubmodule):
    """
    Evaluates wether conditions permit the safe introduction of controlled stochasticity in downstream processes.

    THIS MODULE DOES NOT INJECT STOCHASTIC NOISE ITSELF. IT ONLY ASSESSES READINESS FOR IT.
    """


    @property
    def name(self) -> str:
        return "controlled_stochasticity_readiness"

    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - exploration_drive: float in [0,1]
        - constraint_awareness: float in [0,1]
        - system_stability: float in [0,1]
        """
        exploration_drive = float(state.get("exploration_drive", 0.0))
        constraint_awareness = float(state.get("constraint_awareness", 0.0))
        system_stability = float(state.get("system_stability", 1.0))


        # Readiness requires exploration intent, awareness of constraints, and system stability
        score = exploration_drive * constraint_awareness * system_stability
        score = max(0.0, min(1.0, score))


        flags = {}


        if exploration_drive > 0.6 and system_stability < 0.4:
            flags["unstable_for_stochasticity"] = RewardFlag(
                name="unstable_for_stochasticity",
                severity="warning",
                message="Low system stability combined with high exploration drive indicates instability for controlled stochasticity",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "exploration_drive": exploration_drive,
                "system_stability": system_stability,
                "constraint_awareness": constraint_awareness,
            },
        )
