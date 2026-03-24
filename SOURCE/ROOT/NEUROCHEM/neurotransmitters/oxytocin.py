"""Oxytocin (OXT) behavior module."""
from __future__ import annotations
from typing import Dict, List
from zados.neurochem.neurotransmitters.base import (
    NeurotransmitterModule, ReleaseDriveSpec, OscillationCouplingRule,
)

class OXTModule(NeurotransmitterModule):
    """
    Oxytocin behavior specification.

    Release drivers: empathy, social engagement, trust
    Oscillation coupling: theta -> release
    Primary role: social bonding, trust resonance, attunement.
    """
    @property
    def name(self) -> str: return "OXT"

    @property
    def release_spec(self) -> ReleaseDriveSpec:
        return ReleaseDriveSpec(
            signal_keys=["empathy", "social_engagement", "trust", "emotion_drive"],
            weights=[0.30, 0.25, 0.20, 0.25],
            threshold=0.05,
        )

    @property
    def oscillation_rules(self) -> List[OscillationCouplingRule]:
        return [
            OscillationCouplingRule(target="release", band="theta", coefficient=0.4),
            OscillationCouplingRule(target="sigma_tonic", band="alpha", coefficient=-0.2),
        ]
