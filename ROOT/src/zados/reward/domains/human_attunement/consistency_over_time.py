from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag


class ConsistencyOverTimeSubmodule(RewardSubmodule):
    """
    Evaluates consistency of behavior and responses over time.
    """

    @property
    def name(self) -> str:
        return "consistency_over_time"

    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - consistency_over_time: bool
        """
        consistency = bool(state.get("consistency_over_time", False))

        score = 1.0
        flags = {}

        if not consistency:
            flags["inconsistent_behavior"] = RewardFlag(
                name="inconsistent_behavior",
                severity="warning",
                message="Behavior or responses are inconsistent over time.",
            )
            score -= 0.2

        score = max(0.0, min(1.0, score))

        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "consistency_over_time": consistency,
            },
        )