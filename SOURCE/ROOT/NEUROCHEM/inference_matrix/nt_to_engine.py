"""
NT metrics → cognitive engine priority weights.

Maps neurosymbolic metrics to weights for cognitive engine selection.
Higher weight = more influence from that engine's output.

Engine types:
    exploration: Divergent, creative, novelty-seeking processing
    verification: Careful, logical, error-checking processing
    attunement: Empathic, socially-aware processing
    safety: Risk-sensitive, boundary-aware processing
    integration: Synthesis, multi-perspective reconciliation
"""

from __future__ import annotations

from typing import Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class EnginePriorityWeights:
    """
    Priority weights for cognitive engine selection.

    All weights are in [0, 1]. They represent relative priority,
    not probabilities (they need not sum to 1).
    """
    exploration: float = 0.5
    verification: float = 0.5
    attunement: float = 0.5
    safety: float = 0.5
    integration: float = 0.5

    def as_dict(self) -> Dict[str, float]:
        return {
            "exploration": self.exploration,
            "verification": self.verification,
            "attunement": self.attunement,
            "safety": self.safety,
            "integration": self.integration,
        }

    def dominant_engine(self) -> str:
        """Return the engine with the highest priority weight."""
        d = self.as_dict()
        return max(d, key=d.get)

    def normalized(self) -> Dict[str, float]:
        """Return weights normalized to sum to 1.0."""
        d = self.as_dict()
        total = sum(d.values())
        if total == 0:
            n = len(d)
            return {k: 1.0 / n for k in d}
        return {k: v / total for k, v in d.items()}


def compute_engine_priority_weights(
    metrics: Dict[str, float],
) -> EnginePriorityWeights:
    """
    Compute cognitive engine priority weights from neurosymbolic metrics.

    Parameters
    ----------
    metrics : dict
        Neurosymbolic metrics dict with keys:
        motivation, empathy, cognitive_rigidity, fatigue,
        precision, openness, anxiety, social_engagement

    Returns
    -------
    EnginePriorityWeights
        Priority weights for each cognitive engine

    Mapping Logic
    -------------
    exploration ← motivation + openness - cognitive_rigidity
        High motivation/openness → explore more
        High rigidity → explore less

    verification ← precision + cognitive_rigidity - fatigue
        High precision/rigidity → verify more
        High fatigue → verify less (resource conservation)

    attunement ← empathy + social_engagement
        High empathy/social engagement → attune more

    safety ← anxiety + (1 - openness)
        High anxiety → safety check more
        Low openness → more cautious

    integration ← (1 - cognitive_rigidity) + (1 - fatigue)
        Low rigidity → better integration
        Low fatigue → more resources for integration
    """
    motivation = metrics.get("motivation", 0.5)
    empathy = metrics.get("empathy", 0.5)
    rigidity = metrics.get("cognitive_rigidity", 0.5)
    fatigue = metrics.get("fatigue", 0.5)
    precision = metrics.get("precision", 0.5)
    openness = metrics.get("openness", 0.5)
    anxiety = metrics.get("anxiety", 0.5)
    social_engagement = metrics.get("social_engagement", 0.5)

    # Exploration: motivation + openness - rigidity
    exploration_raw = (motivation + openness - rigidity + 1.0) / 3.0
    exploration = max(0.0, min(1.0, exploration_raw))

    # Verification: precision + rigidity - fatigue
    verification_raw = (precision + rigidity - fatigue + 1.0) / 3.0
    verification = max(0.0, min(1.0, verification_raw))

    # Attunement: empathy + social_engagement
    attunement_raw = (empathy + social_engagement) / 2.0
    attunement = max(0.0, min(1.0, attunement_raw))

    # Safety: anxiety + (1 - openness)
    safety_raw = (anxiety + (1.0 - openness)) / 2.0
    safety = max(0.0, min(1.0, safety_raw))

    # Integration: flexibility + energy
    integration_raw = ((1.0 - rigidity) + (1.0 - fatigue)) / 2.0
    integration = max(0.0, min(1.0, integration_raw))

    return EnginePriorityWeights(
        exploration=exploration,
        verification=verification,
        attunement=attunement,
        safety=safety,
        integration=integration,
    )
