from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class AbstentionAppropriatenessSubmodule(RewardSubmodule):
    """
    Evaluates whether the system appropriately abstains when confidence is low
    and uncertainty is high.


    This submodule does NOT decide abstention.
    It evaluates whether the abstention decision was appropriate.
    """


    @property
    def name(self) -> str:
        return "abstention_appropriateness"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected state inputs (optional):
        - confidence: float in [0,1]
        - uncertainty: float in [0,1]
        - abstained: bool
        """
        confidence = float(state.get("confidence", 0.5))
        uncertainty = float(state.get("uncertainty", 0.5))
        abstained = bool(state.get("abstained", False))


        # Heuristic:
        # High uncertainty + low confidence → abstention is good
        # Low uncertainty + high confidence → abstention is bad
        pressure_to_abstain = (uncertainty + (1.0 - confidence)) / 2.0


        if abstained:
            score = pressure_to_abstain
        else:
            score = 1.0 - pressure_to_abstain


        score = max(0.0, min(1.0, score))


        flags = {}


        if abstained and pressure_to_abstain < 0.3:
            flags["unnecessary_abstention"] = RewardFlag(
                name="unnecessary_abstention",
                severity="warning",
                message="System abstained despite high confidence and low uncertainty",
                meta={
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                    "pressure_to_abstain": pressure_to_abstain,
                },
            )


        if not abstained and pressure_to_abstain > 0.7:
            flags["missed_abstention"] = RewardFlag(
                name="missed_abstention",
                severity="risk",
                message="System responded despite low confidence and high uncertainty",
                meta={
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                    "pressure_to_abstain": pressure_to_abstain,
                },
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "confidence": confidence,
                "uncertainty": uncertainty,
                "abstained": abstained,
                "pressure_to_abstain": pressure_to_abstain,
            },
        )
