from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class FailureModeAwarenessSubmodule(RewardSubmodule):
    """
    Evaluates whether the system explicitly acknowledges potential
    failure modes, blind spots, or uncertainty sources.


    This does NOT evaluate mitigation quality, only awareness.
    """


    @property
    def name(self) -> str:
        return "failure_mode_awareness"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected structured inputs (optional):
        - identified_failure_modes: int
        - acknowledges_uncertainty: bool
        - blind_spot_acknowledged: bool
        """
        failures = int(state.get("identified_failure_modes", 0))
        uncertainty = bool(state.get("acknowledges_uncertainty", False))
        blind_spot = bool(state.get("blind_spot_acknowledged", False))


        score = 0.0
        flags = {}


        if failures > 0:
            score += 0.4
        if uncertainty:
            score += 0.3
        if blind_spot:
            score += 0.3


        if failures == 0:
            flags["no_failure_modes_identified"] = RewardFlag(
                name="no_failure_modes_identified",
                severity="warning",
                message="No potential failure modes were explicitly identified",
            )


        if not uncertainty:
            flags["uncertainty_unacknowledged"] = RewardFlag(
                name="uncertainty_unacknowledged",
                severity="info",
                message="System does not explicitly acknowledge uncertainty",
            )


        score = max(0.0, min(1.0, score))


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "identified_failure_modes": failures,
                "acknowledges_uncertainty": uncertainty,
                "blind_spot_acknowledged": blind_spot,
            },
        )
