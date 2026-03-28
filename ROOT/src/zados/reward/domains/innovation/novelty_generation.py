from __future__ import annotations

from typing import Any, Dict

from zados.reward.base.interfaces import RewardSubmodule
from zados.reward.base.types import RewardContext, RewardSubscore
from zados.reward.base.structure import RewardFlag


class NoveltyGenerationSubmodule(RewardSubmodule):
    """
    Evaluates novelty and diversity of outputs.

    Measures:
    - Novelty (distance from recent patterns)
    - Diversity (variety of concepts/styles)
    - Exploration intent alignment

    This submodule does NOT enforce penalties.
    It reports signals only.
    """

    @property
    def name(self) -> str:
        return "novelty_generation"

    def evaluate(
        self,
        state: Dict[str, Any],
        ctx: RewardContext,
    ) -> RewardSubscore:
        """
        Expected optional state inputs:
        - novelty_signal: float in [0, 1]
        - exploration_intent: float in [0, 1]
        """

        novelty_signal = float(state.get("novelty_signal", 0.5))
        exploration_intent = float(state.get("exploration_intent", 0.5))

        # Base score combines novelty and exploration alignment
        base_score = (novelty_signal + exploration_intent) / 2.0
        base_score = max(0.0, min(1.0, base_score))

        flags = {}

        # High novelty without exploration intent (uncontrolled)
        if novelty_signal > 0.7 and exploration_intent < 0.3:
            flags["uncontrolled_novelty"] = RewardFlag(
                name="uncontrolled_novelty",
                severity="warning",
                message="High novelty signal without corresponding exploration intent",
                meta={
                    "novelty_signal": novelty_signal,
                    "exploration_intent": exploration_intent,
                },
            )

        # High exploration intent without novelty (blocked exploration)
        if exploration_intent > 0.7 and novelty_signal < 0.3:
            flags["blocked_exploration"] = RewardFlag(
                name="blocked_exploration",
                severity="warning",
                message="High exploration intent but low novelty signal",
                meta={
                    "novelty_signal": novelty_signal,
                    "exploration_intent": exploration_intent,
                },
            )

        if base_score < 0.3:
            flags["low_novelty_diversity"] = RewardFlag(
                name="low_novelty_diversity",
                severity="info",
                message="Output shows low novelty and exploration.",
                meta={
                    "novelty_signal": novelty_signal,
                    "exploration_intent": exploration_intent,
                },
            )

        return RewardSubscore(
            name=self.name,
            score=base_score,
            flags=flags,
            meta={
                "novelty_score": novelty_signal,
                "diversity_score": novelty_signal,
                "exploration_intent": exploration_intent > 0.5,
            },
        )
