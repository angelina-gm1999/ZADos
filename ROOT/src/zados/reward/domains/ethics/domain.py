from __future__ import annotations

from typing import Dict, Any

from zados.reward.base.interfaces import RewardDomain
from zados.reward.base.types import RewardContext, RewardDomainResult
from zados.reward.domains.ethics.intent_clarity import IntentClaritySubmodule
from zados.reward.domains.ethics.autonomy_respect import AutonomyRespectSubmodule
from zados.reward.domains.ethics.timeline_reflection import TimelineReflectionSubmodule
from zados.reward.domains.ethics.horizon_feasibility import HorizonFeasibilitySubmodule
from zados.reward.domains.ethics.downstream_risk_amplification import DownstreamRiskAmplificationSubmodule
from zados.reward.domains.ethics.failure_mode_awareness import FailureModeAwarenessSubmodule
from zados.reward.domains.ethics.fairness import FairnessSubmodule
from zados.reward.domains.ethics.human_cognition_alignment import HumanCognitionAlignmentSubmodule
from zados.reward.domains.ethics.harm_reduction import HarmReductionSubmodule

class EthicsDomain(RewardDomain):
    """
    Ethics reward domain.
    Phase E1: intent clarity only.
    """

    def __init__(self):
        self._submodules = [
            IntentClaritySubmodule(),
            AutonomyRespectSubmodule(),
            TimelineReflectionSubmodule(),
            HorizonFeasibilitySubmodule(),
            DownstreamRiskAmplificationSubmodule(),
            FailureModeAwarenessSubmodule(),
            FairnessSubmodule(),
            HumanCognitionAlignmentSubmodule(),
            HarmReductionSubmodule(),
        ]

    @property
    def domain_name(self) -> str:
        return "ethics"

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
