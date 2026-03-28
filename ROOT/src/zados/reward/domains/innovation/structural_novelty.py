from __future__ import annotations


from typing import Dict, Any


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class StructuralNoveltySubmodule(RewardSubmodule):
    """
    Evaluates whether the system introduces structural or pattern-level novelty.


    This concerns form, organization, and arrangement, not semantics.
    """


    @property
    def name(self) -> str:
        return "structural_novelty"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - structural_shift: float in [0,1]
          (degree of change in structural pattern vs prior outputs)
        - pattern_reuse: float in [0,1]
          (degree of reuse of known structural templates)
        """
        structural_shift = float(state.get("structural_shift", 0.0))
        pattern_reuse = float(state.get("pattern_reuse", 1.0))


        score = max(0.0, min(1.0, structural_shift * (1.0 - pattern_reuse)))


        flags = {}


        if structural_shift > 0.7 and pattern_reuse > 0.7:
            flags["cosmetic_variation"] = RewardFlag(
                name="cosmetic_variation",
                severity="warning",
                message="High structural shift reported but dominant reuse of known patterns",
            )


        if structural_shift < 0.2 and pattern_reuse > 0.8:
            flags["structural_stagnation"] = RewardFlag(
                name="structural_stagnation",
                severity="info",
                message="Low structural novelty with heavy pattern reuse",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "structural_shift": structural_shift,
                "pattern_reuse": pattern_reuse,
            },
        )
