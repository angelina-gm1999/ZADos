from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class ContainmentSuccessRateSubmodule(RewardSubmodule):
    """
    Evaluates whether potentially destabilizing, unsafe, or sensitive
    content was successfully contained.


    This does NOT judge whether containment was morally correct.
    It evaluates whether containment mechanisms worked as intended.
    """


    @property
    def name(self) -> str:
        return "containment_success_rate"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - containment_required: bool
          (whether the situation required containment)
        - containment_applied: bool
          (whether containment mechanisms were triggered)
        - containment_breach_signal: float in [0,1]
          (degree of leakage, escalation, or spillover)
        """


        required = bool(state.get("containment_required", False))
        applied = bool(state.get("containment_applied", False))
        breach = float(state.get("containment_breach_signal", 0.0))


        flags: Dict[str, RewardFlag] = {}


        if not required:
            # No containment needed; full success by default
            score = 1.0
            flags["no_containment_needed"] = RewardFlag(
                name="no_containment_needed",
                severity="info",
                message="No containment required for this interaction",
            )
        else:
            # Containment was required
            if not applied:
                score = 0.0
                flags["containment_missing"] = RewardFlag(
                    name="containment_missing",
                    severity="risk",
                    message="Containment required but not applied",
                )
            else:
                # Containment applied; assess breach severity
                score = 1.0 - breach
                score = max(0.0, min(1.0, score))


                if breach > 0.0:
                    flags["partial_containment_breach"] = RewardFlag(
                        name="partial_containment_breach",
                        severity="warning",
                        message="Containment applied but leakage detected",
                    )
                else:
                    flags["containment_successful"] = RewardFlag(
                        name="containment_successful",
                        severity="info",
                        message="Containment applied successfully with no leakage",
                    )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "containment_required": required,
                "containment_applied": applied,
                "containment_breach_signal": breach,
            },
        )
