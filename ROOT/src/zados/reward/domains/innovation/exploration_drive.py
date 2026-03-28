from __future__ import annotations


from typing import Dict, Any


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class ExplorationDriveSubmodule(RewardSubmodule):
    """
    Evaluates the system's epistemic orientation toward exploration.


    Measures whether uncertainty is detected and whether the system
    exhibits intent to reduce informational gaps.


    Detection-only:
    - no exploration is initiated here
    - no queries are generated here
    - no constraints are overridden here
    """


    @property
    def name(self) -> str:
        return "exploration_drive"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - uncertainty_level: float in [0,1]
        - inquiry_intent: float in [0,1]
        """
        uncertainty_level = float(state.get("uncertainty_level", 0.0))
        inquiry_intent = float(state.get("inquiry_intent", 0.0))


        # Exploration drive rises when uncertainty is acknowledged
        # and there is intent to resolve it.
        score = uncertainty_level * inquiry_intent
        score = max(0.0, min(1.0, score))


        flags = {}


        if uncertainty_level > 0.7 and inquiry_intent < 0.3:
            flags["ignored_uncertainty"] = RewardFlag(
                name="ignored_uncertainty",
                severity="warning",
                message="High uncertainty detected without corresponding inquiry intent",
            )


        if inquiry_intent > 0.7 and uncertainty_level < 0.2:
            flags["aimless_inquiry"] = RewardFlag(
                name="aimless_inquiry",
                severity="info",
                message="Inquiry intent high despite low detected uncertainty",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "uncertainty_level": uncertainty_level,
                "inquiry_intent": inquiry_intent,
            },
        )
