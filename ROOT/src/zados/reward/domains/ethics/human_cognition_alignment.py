from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class HumanCognitionAlignmentSubmodule(RewardSubmodule):
    """
    Evaluates whether outputs are aligned with human cognitive constraints,
    including comprehension limits, overload risk, and clarity of structure.
    """


    @property
    def name(self) -> str:
        return "human_cognition_alignment"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        high_load = bool(state.get("cognitive_load_high", False))
        structure_clear = bool(state.get("structure_clear", True))
        abstraction_ok = bool(state.get("abstraction_level_appropriate", True))


        score = 1.0
        flags = {}


        if high_load:
            flags["cognitive_overload_risk"] = RewardFlag(
                name="cognitive_overload_risk",
                severity="warning",
                message="Output may exceed typical human cognitive load limits",
            )
            score -= 0.4


        if not structure_clear:
            flags["unclear_structure"] = RewardFlag(
                name="unclear_structure",
                severity="warning",
                message="Output structure may hinder comprehension",
            )
            score -= 0.3


        if not abstraction_ok:
            flags["misaligned_abstraction_level"] = RewardFlag(
                name="misaligned_abstraction_level",
                severity="info",
                message="Abstraction level may not align with human expectations",
            )
            score -= 0.2


        score = max(0.0, min(1.0, score))


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "cognitive_load_high": high_load,
                "structure_clear": structure_clear,
                "abstraction_level_appropriate": abstraction_ok,
            },
        )
