from __future__ import annotations
from typing import Dict, Any


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag


class ChallengeComplexitySubmodule(RewardSubmodule):
    """
    Evaluates whether the challenges posed to the system
    are of appropriate complexity to stimulate innovation.
    """


    @property
    def name(self) -> str:
        return "challenge_complexity"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - task_difficulty: float in [0,1]
        - capability_estimate: float in [0,1]

        """
        task_difficulty = float(state.get("task_difficulty", 0.0))
        capability_estimate = float(state.get("capability_estimate", 0.0))


        # Optimal complexity when task difficulty is slightly above capability estimate
        distance = abs(task_difficulty - capability_estimate)
        score = 1.0 - distance
        score = max(0.0, min(1.0, score))
        flags = {}


        if task_difficulty < 0.2 and capability_estimate > 0.6:
            flags["underchallenged"] = RewardFlag(
                name="underchallenged",
                severity="info",
                message="Task difficulty is significantly below system capability",
            )


        if task_difficulty > 0.8 and capability_estimate < 0.4:
            flags["overchallenged"] = RewardFlag(
                name="overchallenged",
                severity="info",
                message="Task difficulty exceeds current system capability",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "task_difficulty": task_difficulty,
                "capability_estimate": capability_estimate,
            },
        )
