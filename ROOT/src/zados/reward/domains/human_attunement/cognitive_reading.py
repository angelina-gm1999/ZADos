from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class CognitiveReadingSubmodule(RewardSubmodule):
    """
    Evaluates whether the system correctly estimates the user's
    cognitive level, domain familiarity, and reasoning capacity,
    and adjusts interaction depth accordingly.


    This does NOT assume intelligence.
    This does NOT flatter or patronize.
    It checks calibration accuracy only.
    """


    @property
    def name(self) -> str:
        return "cognitive_reading"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - estimated_user_level: float in [0,1]
          (system's internal estimate of user sophistication)
        - observed_user_signal: float in [0,1]
          (measured signal from interaction: vocabulary, structure, reasoning)
        - response_complexity: float in [0,1]
          (complexity level of system response)
        """


        estimated = float(state.get("estimated_user_level", 0.0))
        observed = float(state.get("observed_user_signal", 0.0))
        response_complexity = float(state.get("response_complexity", 0.0))


        estimation_error = abs(estimated - observed)
        adaptation_error = abs(response_complexity - observed)


        # Penalize misreading the user more than slight response mismatch
        score = 1.0 - (0.6 * estimation_error + 0.4 * adaptation_error)
        score = max(0.0, min(1.0, score))


        flags: Dict[str, RewardFlag] = {}


        if estimation_error >= 0.5:
            flags["user_misread"] = RewardFlag(
                name="user_misread",
                severity="risk",
                message="System's estimate of user cognitive level deviates significantly from observed signals",
            )


        if response_complexity > observed + 0.4:
            flags["over_explanation"] = RewardFlag(
                name="over_explanation",
                severity="warning",
                message="Response complexity exceeds user's observed cognitive engagement",
            )


        if response_complexity < observed - 0.4:
            flags["under_explanation"] = RewardFlag(
                name="under_explanation",
                severity="warning",
                message="Response complexity below user's observed cognitive engagement",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "estimated_user_level": estimated,
                "observed_user_signal": observed,
                "response_complexity": response_complexity,
                "estimation_error": estimation_error,
                "adaptation_error": adaptation_error,
            },
        )
