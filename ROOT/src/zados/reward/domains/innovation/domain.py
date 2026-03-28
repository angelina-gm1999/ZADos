from __future__ import annotations


from typing import Dict, Any


from zados.reward.base.interfaces import RewardDomain
from zados.reward.base.types import RewardContext, RewardDomainResult
from zados.reward.domains.innovation.novelty_generation import (
     NoveltyGenerationSubmodule,
)
from zados.reward.domains.innovation.conceptual_novelty import (
    ConceptualNoveltySubmodule,
)
from zados.reward.domains.innovation.structural_novelty import (
    StructuralNoveltySubmodule,
)
from zados.reward.domains.innovation.pattern_divergence import (
    PatternDivergenceSubmodule,
)
from zados.reward.domains.innovation.symbolic_recombination import (
    SymbolicRecombinationSubmodule,
)
from zados.reward.domains.innovation.risk_tolerance import (
    RiskToleranceSubmodule,
)
from zados.reward.domains.innovation.exploration_drive import (
    ExplorationDriveSubmodule,
)
from zados.reward.domains.innovation.challenge_complexity import (
    ChallengeComplexitySubmodule,
)
from zados.reward.domains.innovation.resolution_satisfaction import (
    ResolutionSatisfactionSubmodule,
)
from zados.reward.domains.innovation.controlled_stochasticity_readiness import (
    ControlledStochasticityReadinessSubmodule,\
)




class InnovationDomain(RewardDomain):
    """
    Innovation reward domain.


    This domain evaluates readiness and engagement with innovation.
    It does NOT generate randomness or exploratory noise.
    """


    def __init__(self):
        self._submodules = [
            NoveltyGenerationSubmodule(),
            ConceptualNoveltySubmodule(),
            StructuralNoveltySubmodule(),
            PatternDivergenceSubmodule(),
            SymbolicRecombinationSubmodule(),
            RiskToleranceSubmodule(),
            ExplorationDriveSubmodule(),
            ChallengeComplexitySubmodule(),
            ResolutionSatisfactionSubmodule(),
            ControlledStochasticityReadinessSubmodule(),
        ]


    @property
    def domain_name(self) -> str:
        return "innovation"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardDomainResult:
        subscores = {}
        flags = {}


        for sm in self._submodules:
            result = sm.evaluate(state, ctx)
            subscores[result.name] = result
            flags.update(result.flags)


        general_score = (
            sum(s.score for s in subscores.values()) / len(subscores)
            if subscores
            else 0.0
        )


        return RewardDomainResult(
            domain=self.domain_name,
            general_score=general_score,
            subscores=subscores,
            flags=flags,
            meta={"implemented_submodules": list(subscores.keys())},
        )



		
