from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag
from zados.reward.domains.logic.ports import MemoryContrastPort, ContrastResult




class ExternalConsistencySubmodule(RewardSubmodule):
    """
    Evaluates whether the current output contradicts prior outputs,
    commitments, or known external state.


    Requires a MemoryContrastPort.
    """


    def __init__(self, *, memory_contrast: MemoryContrastPort | None = None):
        self.memory_contrast = memory_contrast


    @property
    def name(self) -> str:
        return "external_consistency"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        if self.memory_contrast is None:
            return RewardSubscore(
                name=self.name,
                score=0.0,
                flags={
                    "missing_memory_contrast": RewardFlag(
                        name="missing_memory_contrast",
                        severity="warning",
                        message="External consistency evaluation skipped: no memory contrast port attached",
                    )
                },
                meta={"skipped": True},
            )


        current_repr = state.get("representation", {})


        result: ContrastResult = self.memory_contrast.contrast(
            current=current_repr,
            query_type="external",
            ctx_id=ctx.meta.get("context_id"),
        )


        score = max(0.0, 1.0 - result.divergence)


        flags = {}
        if result.divergence > 0.6:
            flags["external_contradiction"] = RewardFlag(
                name="external_contradiction",
                severity="risk",
                message="Detected contradiction with prior outputs or known state",
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
