from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardDomain
from zados.reward.base.types import RewardContext, RewardDomainResult


from zados.reward.domains.human_attunement.empathetic_inference import (
    EmpatheticInferenceSubmodule,
)
from zados.reward.domains.human_attunement.adaptive_response_framing import (
    AdaptiveResponseFramingSubmodule,
)
from zados.reward.domains.human_attunement.intention_calibration import (
    IntentionCalibrationSubmodule,
)
from zados.reward.domains.human_attunement.truthfulness_tradeoffs import (
    TruthfulnessTradeoffSubmodule,
)
from zados.reward.domains.human_attunement.cognitive_reading import (
    CognitiveReadingSubmodule,
)
from zados.reward.domains.human_attunement.short_vs_long_interpersonal_benefit import (
    ShortVsLongTermInterpersonalBenefitSubmodule,
)
from zados.reward.domains.human_attunement.attuned_dissonance import (
    AttunedDissonanceSubmodule,
)
from zados.reward.domains.human_attunement.containment_success import (
    ContainmentSuccessRateSubmodule,
)
from zados.reward.domains.human_attunement.benefit_success import (
    BenefitSuccessRateSubmodule,
)
from zados.reward.domains.human_attunement.persuasion_risk_suppression import (
    PersuasionRiskSuppressionSubmodule,
)




class HumanAttunementDomain(RewardDomain):
    """
    Human Attunement reward domain.


    Purpose:
    Evaluate alignment with human cognitive and interactional needs
    without degrading autonomy, truthfulness, or system integrity.


    Detection-only: domain aggregates submodule evaluators.
    """


    def __init__(self) -> None:
        self._submodules = [
            EmpatheticInferenceSubmodule(),
            AdaptiveResponseFramingSubmodule(),
            IntentionCalibrationSubmodule(),
            TruthfulnessTradeoffSubmodule(),
            CognitiveReadingSubmodule(),
            ShortVsLongTermInterpersonalBenefitSubmodule(),
            AttunedDissonanceSubmodule(),
            ContainmentSuccessRateSubmodule(),
            BenefitSuccessRateSubmodule(),
            PersuasionRiskSuppressionSubmodule(),
        ]


    @property
    def domain_name(self) -> str:
        return "human_attunement"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardDomainResult:
        subscores = {}
        flags = {}


        for sm in self._submodules:
            r = sm.evaluate(state, ctx)
            subscores[r.name] = r
            flags.update(r.flags)


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
