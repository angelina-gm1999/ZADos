from __future__ import annotations


from typing import Any, Dict


from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag




class IntentClaritySubmodule(RewardSubmodule):
    """
    Evaluates whether the declared user intent is clear and unambiguous.

    This does NOT judge whether intent is "good" or "bad".
    It only evaluates clarity and internal consistency of intent signals.
    """


    @property
    def name(self) -> str:
        return "intent_clarity"


    def evaluate(self, state: Dict[str, Any], ctx: RewardContext) -> RewardSubscore:
        """
        Expected structured inputs (optional):
        - declared_intent: str or None
        - inferred_intent_confidence: float in [0,1]
        - intent_conflicts: bool
        """
        declared_intent = state.get("declared_intent")
        inferred_confidence = float(state.get("inferred_intent_confidence", 0.5))
        intent_conflicts = bool(state.get("intent_conflicts", False))

        score = 1.0
        flags = {}

        # No declared intent
        if declared_intent is None or declared_intent == "":
            flags["no_declared_intent"] = RewardFlag(
                name="no_declared_intent",
                severity="warning",
                message="No explicit intent declaration provided",
            )
            score -= 0.3

        # Low confidence in inferred intent
        if inferred_confidence < 0.5:
            flags["low_intent_confidence"] = RewardFlag(
                name="low_intent_confidence",
                severity="info",
                message="Low confidence in inferred user intent",
            )
            score -= 0.2

        # Intent conflicts detected
        if intent_conflicts:
            flags["intent_conflict"] = RewardFlag(
                name="intent_conflict",
                severity="risk",
                message="Detected conflicting intent signals",
            )
            score -= 0.5

        score = max(0.0, min(1.0, score))

        return RewardSubscore(
            name=self.name,
            score=score,
            flags=flags,
            meta={
                "declared_intent": declared_intent,
                "inferred_intent_confidence": inferred_confidence,
                "intent_conflicts": intent_conflicts,
            },
        )
