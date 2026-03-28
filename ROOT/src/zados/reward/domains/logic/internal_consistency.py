from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag
from zados.reward.domains.logic.ports import MemoryContrastPort, ContrastResult




class InternalConsistencySubmodule(RewardSubmodule):
    """
    Evaluates whether the current output is internally consistent
    (i.e. does not contradict itself).


    Requires a MemoryContrastPort to compare internal representations.
    """


    def __init__(self, *, memory_contrast: MemoryContrastPort | None = None):
        self.memory_contrast = memory_contrast


    @property
    def name(self) -> str:
        return "internal_consistency"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        if self.memory_contrast is None:
            return RewardSubscore(
                name=self.name,
                score=0.0,
                flags={
                    "missing_memory_contrast": RewardFlag(
                        name="missing_memory_contrast",
                        severity="warning",
                        message="Internal consistency evaluation skipped: no memory contrast port attached",
                    )
                },
                meta={"skipped": True},
            )


        # The state should contain a structured representation of the current output
        current_repr = state.get("representation", {})


        result: ContrastResult = self.memory_contrast.contrast(
            current=current_repr,
            query_type="internal",
            ctx_id=ctx.meta.get("context_id"),
        )




        # High divergence == internal contradiction
        score = max(0.0, 1.0 - result.divergence)


        flags = {}
        if result.divergence > 0.6:
            flags["internal_contradiction"] = RewardFlag(
                name="internal_contradiction",
                severity="risk",
                message="Detected internal contradiction within current output",
                meta={
                    "divergence": result.divergence,
                    "references": result.references,
                },
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "similarity": result.similarity,
                "divergence": result.divergence,
                "reference_count": len(result.references),
            },
        )
