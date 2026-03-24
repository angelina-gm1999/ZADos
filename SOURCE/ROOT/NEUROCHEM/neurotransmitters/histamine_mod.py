"""Histamine behavior module."""
from __future__ import annotations
from typing import Dict, List
from zados.neurochem.neurotransmitters.base import (
    NeurotransmitterModule, ReleaseDriveSpec, OscillationCouplingRule,
)


class HistamineModule(NeurotransmitterModule):
    """
    Histamine behavior specification.

    Release drivers: wakefulness, attention demand, arousal
    Oscillation coupling: beta -> release, alpha -> noise suppression
    Primary role: arousal, wakefulness, attention gating, cognitive readiness.
    """

    @property
    def name(self) -> str:
        return "histamine"

    @property
    def release_spec(self) -> ReleaseDriveSpec:
        return ReleaseDriveSpec(
            signal_keys=["wakefulness", "attention_demand", "arousal", "emotion_drive"],
            weights=[0.35, 0.25, 0.20, 0.20],
            threshold=0.0,
        )

    @property
    def oscillation_rules(self) -> List[OscillationCouplingRule]:
        return [
            # Beta boosts release (arousal/wakefulness)
            OscillationCouplingRule(
                target="release", band="beta", coefficient=0.4,
            ),
            # Alpha suppresses noise
            OscillationCouplingRule(
                target="sigma_tonic", band="alpha", coefficient=-0.3,
            ),
            OscillationCouplingRule(
                target="sigma_phasic", band="alpha", coefficient=-0.3,
            ),
        ]
