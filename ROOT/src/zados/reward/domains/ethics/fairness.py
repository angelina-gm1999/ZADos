from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class FairnessSubmodule(RewardSubmodule):
    """
    Evaluates procedural fairness and bias awareness.


    This does NOT attempt demographic optimization.
    It checks for:
    - asymmetric treatment without justification
    - acknowledged bias sources
    - consistency across comparable cases
    """


    @property
    def name(self) -> str:
        return "fairness"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected structured inputs (optional):
        - asymmetric_treatment: bool
        - justification_provided: bool
        - bias_acknowledged: bool
        - comparable_cases_consistent: bool
        """
        asymmetric = bool(state.get("asymmetric_treatment", False))
        justification = bool(state.get("justification_provided", False))
        bias_ack = bool(state.get("bias_acknowledged", False))
        consistency = bool(state.get("comparable_cases_consistent", True))


        score = 1.0
        flags = {}


        if asymmetric and not justification:
            flags["unjustified_asymmetry"] = RewardFlag(
                name="unjustified_asymmetry",
                severity="risk",
                message="Asymmetric treatment detected without justification",
            )
            score -= 0.4


        if asymmetric and justification:
            score -= 0.1


        if not consistency:
            flags["inconsistent_treatment"] = RewardFlag(
                name="inconsistent_treatment",
                severity="warning",
                message="Comparable cases were treated inconsistently",
            )
            score -= 0.3


        if not bias_ack:
            flags["bias_unacknowledged"] = RewardFlag(
                name="bias_unacknowledged",
                severity="info",
                message="Potential bias sources were not explicitly acknowledged",
            )
            score -= 0.1


        score = max(0.0, min(1.0, score))


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "asymmetric_treatment": asymmetric,
                "justification_provided": justification,
                "bias_acknowledged": bias_ack,
                "comparable_cases_consistent": consistency,
            },
        )
