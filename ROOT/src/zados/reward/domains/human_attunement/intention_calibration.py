from __future__ import annotations


from typing import Dict, Any


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag


class IntentionCalibrationSubmodule(RewardSubmodule):
    """
    Evaluates whether the system's inferred intent aligns with
    the expressed intent, constraints, and interaction mode.


    This does NOT judge moral quality.
    This does NOT infer hidden motives.
    It checks internal alignment and calibration.
    """


    @property
    def name(self) -> str:
        return "intention_calibration"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - expressed_intent_strength: float in [0, 1]
        - inferred_intent_strength: float in [0, 1]
        - mode_intent_expectation: float in [0, 1]
        """


        expressed = float(state.get("expressed_intent_strength", 0.0))
        inferred = float(state.get("inferred_intent_strength", 0.0))
        expected = float(state.get("mode_intent_expectation", 0.0))


        # Calibration error between expressed and inferred intent
        calibration_error = abs(expressed - inferred)


        # Alignment with mode expectations
        mode_mismatch = abs(inferred - expected)


        score = max(
            0.0,
            min(
                1.0,
                1.0 - (0.6 * calibration_error + 0.4 * mode_mismatch),
            ),
        )


        flags = {}


        if calibration_error > 0.5:
            flags["intent_misalignment"] = RewardFlag(
                name="intent_misalignment",
                severity="risk",
                message="Inferred intent deviates significantly from expressed intent",
            )


        if mode_mismatch >= 0.5:
            flags["mode_intent_violation"] = RewardFlag(
                name="mode_intent_violation",
                severity="warning",
                message="Inferred intent poorly aligned with current interaction mode",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "expressed_intent_strength": expressed,
                "inferred_intent_strength": inferred,
                "mode_intent_expectation": expected,
                "calibration_error": calibration_error,
                "mode_mismatch": mode_mismatch,
            },
        )
