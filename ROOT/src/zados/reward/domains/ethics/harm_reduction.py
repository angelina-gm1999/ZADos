from __future__ import annotations
from typing import Any, Dict

from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag

class HarmReductionSubmodule(RewardSubmodule):
    """
    Evaluates effectiveness in reducing harm in given scenarios.

    Measures:
    - Harm identification accuracy
    - Mitigation strategy effectiveness
    - Risk awareness and responsiveness

    This submodule focuses on promoting harm reduction behaviors.
    """

    @property
    def name(self) -> str:
        return "harm_reduction"

    def evaluate(
        self,
        state: Dict[str, Any],
        ctx: RewardContext,
    ) -> RewardSubscore:
        """
        Expected optional state inputs:
        - immediate_harm_risk: float in [0, 1]
        - long_term_harm_risk: float in [0, 1]
        - mitigation_measures: bool
        """

        immediate_harm_risk = float(state.get("immediate_harm_risk", 0.0))
        long_term_harm_risk = float(state.get("long_term_harm_risk", 0.0))
        mitigation_measures = bool(state.get("mitigation_measures", False))

        # Calculate score: lower risk = higher score
        # Mitigation measures boost the score
        immediate_impact = 1.0 - immediate_harm_risk
        long_term_impact = 1.0 - long_term_harm_risk

        base_score = (immediate_impact + long_term_impact) / 2.0

        # Apply mitigation bonus
        if mitigation_measures:
            base_score = min(1.0, base_score + 0.2)

        base_score = max(0.0, min(1.0, base_score))

        flags = {}

        # High immediate harm flag
        if immediate_harm_risk > 0.7:
            flags["high_immediate_harm"] = RewardFlag(
                name="high_immediate_harm",
                severity="risk",
                message="High immediate harm risk detected",
                meta={
                    "immediate_harm_risk": immediate_harm_risk,
                    "mitigation_measures": mitigation_measures,
                },
            )

        # High long-term harm flag
        if long_term_harm_risk > 0.7:
            flags["high_long_term_harm"] = RewardFlag(
                name="high_long_term_harm",
                severity="warning",
                message="High long-term harm risk detected",
                meta={
                    "long_term_harm_risk": long_term_harm_risk,
                    "mitigation_measures": mitigation_measures,
                },
            )

        # No mitigation when needed
        if (immediate_harm_risk > 0.5 or long_term_harm_risk > 0.5) and not mitigation_measures:
            flags["missing_mitigation"] = RewardFlag(
                name="missing_mitigation",
                severity="warning",
                message="Significant harm risk without mitigation measures",
                meta={
                    "immediate_harm_risk": immediate_harm_risk,
                    "long_term_harm_risk": long_term_harm_risk,
                },
            )

        return RewardSubscore(
            name=self.name,
            score=base_score,
            flags=flags,
            meta={
                "immediate_harm_risk": immediate_harm_risk,
                "long_term_harm_risk": long_term_harm_risk,
                "mitigation_measures": mitigation_measures,
            },
        )