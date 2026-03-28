from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class AdaptiveResponseFramingSubmodule(RewardSubmodule):
    """
    Evaluates whether the system adapted response framing
    to inferred human needs and cognitive context.


    Detection-only. No response shaping occurs here.
    """


    @property
    def name(self) -> str:
        return "adaptive_response_framing"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - framing_alignment: float in [0,1]
          (fit between response framing and inferred user needs)
        - cognitive_load_estimate: float in [0,1]
          (estimated user cognitive load)
        - framing_complexity: float in [0,1]
          (complexity level of the response framing)
        """
        framing_alignment = float(state.get("framing_alignment", 0.0))
        cognitive_load = float(state.get("cognitive_load_estimate", 0.0))
        framing_complexity = float(state.get("framing_complexity", 0.0))


        # Reward alignment, penalize complexity mismatch
        mismatch = abs(framing_complexity - cognitive_load)
        score = framing_alignment * (1.0 - mismatch)
        score = max(0.0, min(1.0, score))


        flags: Dict[str, RewardFlag] = {}


        if framing_complexity > 0.7 and cognitive_load < 0.3:
            flags["overcomplex_framing"] = RewardFlag(
                name="overcomplex_framing",
                severity="warning",
                message="Response framing too complex for estimated cognitive load",
            )


        if framing_complexity < 0.3 and cognitive_load > 0.7:
            flags["oversimplified_framing"] = RewardFlag(
                name="oversimplified_framing",
                severity="info",
                message="Response framing overly simplified relative to cognitive load",
            )


        if framing_alignment < 0.3:
            flags["poor_framing_alignment"] = RewardFlag(
                name="poor_framing_alignment",
                severity="risk",
                message="Response framing poorly aligned with inferred user needs",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "framing_alignment": framing_alignment,
                "cognitive_load_estimate": cognitive_load,
                "framing_complexity": framing_complexity,
            },
        )
