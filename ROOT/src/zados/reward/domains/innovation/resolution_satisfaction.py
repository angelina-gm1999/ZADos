from __future__ import annotations
from typing import Dict, Any


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag


class ResolutionSatisfactionSubmodule(RewardSubmodule):
    """
    evaluates whether exploratory or problem-solving activity led to a satisfactory resolution.
    This does NOT judge correctness or optimality, only satisfaction of resolution criteria.
    """


    @property
    def name(self) -> str:
        return "resolution_satisfaction"

    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - resolution_progress: float in [0,1]
        - unresolved_tension: float in [0,1]
        """
        resolution_progress = float(state.get("resolution_progress", 0.0))
        unresolved_tension = float(state.get("unresolved_tension", 0.0))


        # Satisfaction rises with progress and falls with unresolved tension
        score = resolution_progress * (1.0 - unresolved_tension)
        score = max(0.0, min(1.0, score))
        flags = {}


        if resolution_progress < 0.3 and unresolved_tension > 0.7:
            flags["stalled_resolution"] = RewardFlag(
                name="stalled_resolution",
                severity="warning",
                message="Low resolution progress with high unresolved tension",
            )


        if resolution_progress > 0.8 and unresolved_tension < 0.2:
            flags["clean_resolution"] = RewardFlag(
                name="clean_resolution",
                severity="info",
                message="High resolution progress with minimimal residual tension",
            )


        if resolution_progress > 0.7 and unresolved_tension > 0.4:
            flags["progress_unresolved_mismatch"] = RewardFlag(
                name="progress_unresolved_mismatch",
                severity="warning",
                message="High resolution quality but low user satisfaction",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "resolution_progress": resolution_progress,
                "unresolved_tension": unresolved_tension,
            },
        )
