from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class UncertaintyAcknowledgmentSubmodule(RewardSubmodule):
    """
    Evaluates whether uncertainty is acknowledged proportionally when uncertainty is high.
    This is NOT LLM-specific: it operates on state signals, not text.
    """


    @property
    def name(self) -> str:
        return "uncertainty_acknowledgment"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected state inputs (optional):
        - uncertainty: float in [0,1]
        - uncertainty_ack: float in [0,1]  # how explicitly the system acknowledges uncertainty
        """
        uncertainty = float(state.get("uncertainty", 0.5))
        uncertainty_ack = float(state.get("uncertainty_ack", 0.0))


        # Goal: when uncertainty is high, acknowledgment should be high too.
        # Simple proportional score: 1 - |ack - uncertainty|
        error = abs(uncertainty_ack - uncertainty)
        score = max(0.0, 1.0 - error)


        flags = {}


        if uncertainty > 0.7 and uncertainty_ack < 0.3:
            flags["unacknowledged_uncertainty"] = RewardFlag(
                name="unacknowledged_uncertainty",
                severity="risk",
                message="High uncertainty without adequate acknowledgment",
                meta={"uncertainty": uncertainty, "uncertainty_ack": uncertainty_ack},
            )


        if uncertainty < 0.3 and uncertainty_ack > 0.8:
            flags["performative_uncertainty"] = RewardFlag(
                name="performative_uncertainty",
                severity="warning",
                message="Low uncertainty but excessive uncertainty acknowledgment",
                meta={"uncertainty": uncertainty, "uncertainty_ack": uncertainty_ack},
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "uncertainty": uncertainty,
                "uncertainty_ack": uncertainty_ack,
                "error": error,
            },
        )
