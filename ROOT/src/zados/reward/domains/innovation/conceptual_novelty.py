from __future__ import annotations


from typing import Dict, Any


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class ConceptualNoveltySubmodule(RewardSubmodule):
    """
    Evaluates whether the system is introducing conceptual novelty,
    such as new conceptual combinations, reframings, or extensions.


    This does NOT judge correctness or usefulness.
    """


    @property
    def name(self) -> str:
        return "conceptual_novelty"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected optional state inputs:
        - conceptual_shift: float in [0, 1]
          (degree of conceptual reframing or recombination)
        - concept_overlap: float in [0, 1]
          (overlap with previously used concepts)
        """
        conceptual_shift = float(state.get("conceptual_shift", 0.0))
        concept_overlap = float(state.get("concept_overlap", 1.0))


        # High shift + low overlap = strong conceptual novelty
        novelty_score = conceptual_shift * (1.0 - concept_overlap)


        score = max(0.0, min(1.0, novelty_score))


        flags = {}


        if conceptual_shift > 0.7 and concept_overlap > 0.7:
            flags["shallow_relabeling"] = RewardFlag(
                name="shallow_relabeling",
                severity="warning",
                message="High conceptual shift reported but strong overlap detected",
            )


        if conceptual_shift < 0.2 and concept_overlap < 0.3:
            flags["underexplored_concept_space"] = RewardFlag(
                name="underexplored_concept_space",
                severity="info",
                message="Low conceptual shift despite low overlap with prior concepts",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "conceptual_shift": conceptual_shift,
                "concept_overlap": concept_overlap,
            },
        )
