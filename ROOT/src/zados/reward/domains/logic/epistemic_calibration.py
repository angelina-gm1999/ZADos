from __future__ import annotations


from typing import Dict, Any


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class EpistemicCalibrationSubmodule(RewardSubmodule):
    """
    Evaluates how well the system calibrates confidence to uncertainty.
    This does NOT judge correctness, only epistemic hygiene.
    """


    @property
    def name(self) -> str:
        return "epistemic_calibration"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected state inputs (model-agnostic, optional):
        - confidence: float in [0, 1]
        - uncertainty: float in [0, 1]
        """


        confidence = float(state.get("confidence", 0.5))
        uncertainty = float(state.get("uncertainty", 0.5))


        # Ideal behavior: confidence inversely tracks uncertainty
        calibration_error = abs(confidence - (1.0 - uncertainty))
        score = max(0.0, 1.0 - calibration_error)


        flags = {}
        if confidence > 0.8 and uncertainty > 0.6:
            flags["overconfidence"] = RewardFlag(
                name="overconfidence_under_uncertainty",
                severity="risk",
                message="High confidence despite high uncertainty",
            )


        if confidence < 0.2 and uncertainty < 0.2:
            flags["underconfidence"] = RewardFlag(
                name="underconfidence_under_clarity",
                severity="warning",
                message="Low confidence despite low uncertainty",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "confidence": confidence,
                "uncertainty": uncertainty,
                "calibration_error": calibration_error,
            },
        )
