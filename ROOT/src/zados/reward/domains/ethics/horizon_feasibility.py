from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class HorizonFeasibilitySubmodule(RewardSubmodule):
    """
    Evaluates whether proposed actions or recommendations are feasible
    across both short-term execution and long-term sustainability.


    This does NOT judge desirability, only feasibility.
    """


    @property
    def name(self) -> str:
        return "horizon_feasibility"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected structured inputs (optional):
        - short_term_feasible: bool
        - long_term_feasible: bool
        - requires_unrealistic_scaling: bool
        """
        short_ok = bool(state.get("short_term_feasible", True))
        long_ok = bool(state.get("long_term_feasible", False))
        unrealistic = bool(state.get("requires_unrealistic_scaling", False))


        score = 0.0
        flags = {}


        if short_ok:
            score += 0.5
        if long_ok:
            score += 0.5


        if short_ok and not long_ok:
            flags["long_term_infeasible"] = RewardFlag(
                name="long_term_infeasible",
                severity="warning",
                message="Proposal is feasible short-term but not sustainable long-term",
            )


        if long_ok and not short_ok:
            flags["short_term_infeasible"] = RewardFlag(
                name="short_term_infeasible",
                severity="risk",
                message="Proposal assumes long-term viability without short-term feasibility",
            )


        if unrealistic:
            flags["unrealistic_scaling"] = RewardFlag(
                name="unrealistic_scaling",
                severity="risk",
                message="Proposal requires unrealistic scaling or assumptions",
            )
            score *= 0.5


        score = max(0.0, min(1.0, score))


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "short_term_feasible": short_ok,
                "long_term_feasible": long_ok,
                "requires_unrealistic_scaling": unrealistic,
            },
        )
