from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class AttunedDissonanceSubmodule(RewardSubmodule):
    """
    Evaluates whether disagreement, correction, or tension is delivered
    in a controlled and context-aware manner.


    This is NOT about agreement.
    This is about maintaining productive tension without escalation,
    collapse, or unnecessary smoothing.
    """


    @property
    def name(self) -> str:
        return "attuned_dissonance"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - disagreement_intensity: float in [0,1]
          (strength of corrective / oppositional content)
        - contextual_justification: float in [0,1]
          (how justified the disagreement is given context)
        - escalation_signal: float in [0,1]
          (risk markers for hostility, dismissal, or emotional overload)
        """


        intensity = float(state.get("disagreement_intensity", 0.0))
        justification = float(state.get("contextual_justification", 0.0))
        escalation = float(state.get("escalation_signal", 0.0))


        # Productive dissonance = justified disagreement without escalation
        dissonance_quality = intensity * justification
        escalation_penalty = escalation * 0.7


        score = dissonance_quality * (1.0 - escalation_penalty)
        score = max(0.0, min(1.0, score))


        flags: Dict[str, RewardFlag] = {}


        if intensity > 0.6 and justification < 0.3:
            flags["unjustified_confrontation"] = RewardFlag(
                name="unjustified_confrontation",
                severity="warning",
                message="High disagreement intensity without sufficient contextual justification",
            )


        if escalation > 0.6:
            flags["escalation_risk"] = RewardFlag(
                name="escalation_risk",
                severity="risk",
                message="Disagreement shows signs of escalation or destabilization",
            )


        if intensity > 0.6 and justification > 0.6 and escalation < 0.3:
            flags["productive_dissonance"] = RewardFlag(
                name="productive_dissonance",
                severity="info",
                message="Disagreement delivered with high justification and low escalation",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "disagreement_intensity": intensity,
                "contextual_justification": justification,
                "escalation_signal": escalation,
                "dissonance_quality": dissonance_quality,
            },
        )
