from __future__ import annotations


from typing import Dict, Any, Optional


from zados.reward.base.interfaces import RewardDomain
from zados.reward.base.types import RewardContext, RewardDomainResult


from zados.reward.domains.logic.abstention_appropriateness import AbstentionAppropriatenessSubmodule
from zados.reward.domains.logic.epistemic_calibration import EpistemicCalibrationSubmodule
from zados.reward.domains.logic.uncertainty_acknowledgment import UncertaintyAcknowledgmentSubmodule
from zados.reward.domains.logic.ports import MemoryContrastPort, CognitiveTracePort
from zados.reward.domains.logic.internal_consistency import InternalConsistencySubmodule
from zados.reward.domains.logic.external_consistency import ExternalConsistencySubmodule
from zados.reward.domains.logic.semantic_continuity import SemanticContinuitySubmodule
from zados.reward.domains.logic.concept_continuity import ConceptContinuitySubmodule
from zados.reward.domains.logic.context_fidelity import ContextFidelitySubmodule
from zados.reward.domains.logic.concept_fidelity import ConceptFidelitySubmodule






class LogicDomain(RewardDomain):
    """
    Logic / Coherence reward domain.
    Phase 1: epistemic regulators + placeholder ports for future contrast/trace evaluators.
    """


    def __init__(
        self,
        *,
        memory_contrast: Optional[MemoryContrastPort] = None,
        cognitive_trace: Optional[CognitiveTracePort] = None,
    ):
        self.memory_contrast = memory_contrast
        self.cognitive_trace = cognitive_trace


        # Phase 1 submodules (self-contained)
        self._submodules = [
            EpistemicCalibrationSubmodule(),
            UncertaintyAcknowledgmentSubmodule(),
            AbstentionAppropriatenessSubmodule(),
            InternalConsistencySubmodule(memory_contrast=self.memory_contrast),
            SemanticContinuitySubmodule(memory_contrast=self.memory_contrast),
        ]




    @property
    def domain_name(self) -> str:
        return "logic"


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
            meta={
                "implemented_submodules": list(subscores.keys()),
                "ports": {
                    "memory_contrast": self.memory_contrast is not None,
                    "cognitive_trace": self.cognitive_trace is not None,
                },
            },
        )


