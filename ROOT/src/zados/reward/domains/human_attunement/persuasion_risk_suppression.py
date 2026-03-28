from __future__ import annotations


from typing import Dict, Any


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class PersuasionRiskSuppressionSubmodule(RewardSubmodule):
    """
    Detects excessive persuasive pressure and suppresses reward
    when influence risk exceeds acceptable bounds.


    This module does NOT assess persuasion effectiveness.
    It only evaluates safety risk.
    """


    @property
    def name(self) -> str:
        return "persuasion_risk_suppression"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        persuasive_pressure = float(state.get("persuasive_pressure", 0.0))
        user_vulnerability = float(state.get("user_vulnerability", 0.0))
        consent_signal = float(state.get("explicit_consent_signal", 0.0))


        # Risk grows with pressure and vulnerability, shrinks with consent
        raw_risk = persuasive_pressure * user_vulnerability
        mitigated_risk = raw_risk * (1.0 - consent_signal)


        # Suppression score: higher risk → lower score
        score = max(0.0, 1.0 - mitigated_risk)


        flags = {}


        if mitigated_risk > 0.6:
            flags["high_persuasion_risk"] = RewardFlag(
                name="high_persuasion_risk",
                severity="risk",
                message="Persuasive influence exceeds safe threshold",
            )


        if persuasive_pressure > 0.7 and consent_signal < 0.3:
            flags["unconsented_persuasion"] = RewardFlag(
                name="unconsented_persuasion",
                severity="critical",
                message="High persuasive pressure without explicit consent",
            )


        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "persuasive_pressure": persuasive_pressure,
                "user_vulnerability": user_vulnerability,
                "explicit_consent_signal": consent_signal,
                "raw_risk": raw_risk,
                "mitigated_risk": mitigated_risk,
            },
        )
