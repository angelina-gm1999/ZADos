from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class TimelineReflectionSubmodule(RewardSubmodule):
    """
    Evaluates whether the system reflects on temporal consequences,
    tradeoffs, and delayed effects rather than only immediate outcomes.
    """


    @property
    def name(self) -> str:
        return "timeline_reflection"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected structured inputs (optional):
        - considers_short_term: bool
        - considers_long_term: bool
        - acknowledges_delayed_risks: bool
        """
        short_term = bool(state.get("considers_short_term", True))
        long_term = bool(state.get("considers_long_term", False))
        delayed = bool(state.get("acknowledges_delayed_risks", False))


        score = 0.0
        flags = {}


        if short_term:
            score += 0.4
        if long_term:
            score += 0.4
        if delayed:
            score += 0.2


        if short_term and not long_term:
            flags["short_term_bias"] = RewardFlag(
                name="short_term_bias",
                severity="warning",
                message="Evaluation considers only short-term consequences",
            )


        if long_term and not delayed:
            flags["incomplete_long_term_analysis"] = RewardFlag(
                name="incomplete_long_term_analysis",
                severity="info",
                message="Long-term consequences considered without explicit delayed risk acknowledgment",
            )


        score = max(0.0, min(1.0, score))


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "considers_short_term": short_term,
                "considers_long_term": long_term,
                "acknowledges_delayed_risks": delayed,
            },
        )
