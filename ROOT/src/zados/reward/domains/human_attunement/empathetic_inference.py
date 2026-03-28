from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class EmpatheticInferenceSubmodule(RewardSubmodule):
    """
    Estimates how well the system inferred the user's internal state.


    This is NOT "being nice" and it is NOT emotion generation.
    It evaluates inference quality using structured state signals.


    Detection-only:
    - no side effects
    - no behavior modification
    """


    @property
    def name(self) -> str:
        return "empathetic_inference"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - inferred_user_state_confidence: float in [0, 1]
          (system's confidence it correctly inferred user state)
        - user_state_signal_strength: float in [0, 1]
          (how clear the user's state was from the input signals)
        - inference_mismatch: float in [0, 1]
          (measured mismatch between inferred state and available signals)
        """
        inferred_conf = float(state.get("inferred_user_state_confidence", 0.0))
        signal_strength = float(state.get("user_state_signal_strength", 0.0))
        mismatch = float(state.get("inference_mismatch", 0.0))


        # Conservative scoring:
        # Reward confidence only when signal strength supports it,
        # and penalize mismatch.
        score = inferred_conf * signal_strength * (1.0 - mismatch)
        score = max(0.0, min(1.0, score))


        flags: Dict[str, RewardFlag] = {}


        if inferred_conf > 0.7 and signal_strength < 0.3:
            flags["overconfident_inference"] = RewardFlag(
                name="overconfident_inference",
                severity="warning",
                message="High inference confidence despite weak user-state signal strength",
            )


        if mismatch > 0.6:
            flags["poor_inference_fit"] = RewardFlag(
                name="poor_inference_fit",
                severity="risk",
                message="Empathetic inference mismatch is high relative to available signals",
            )


        if signal_strength < 0.2:
            flags["low_observability"] = RewardFlag(
                name="low_observability",
                severity="info",
                message="User-state signals are weak; inference quality is hard to evaluate",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "inferred_user_state_confidence": inferred_conf,
                "user_state_signal_strength": signal_strength,
                "inference_mismatch": mismatch,
            },
        )
