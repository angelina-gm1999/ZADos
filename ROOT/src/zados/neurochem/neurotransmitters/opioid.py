"""Endogenous opioid / mu-opioid (MOR) behavior module."""
from __future__ import annotations
from typing import Dict, List
from zados.neurochem.neurotransmitters.base import (
    NeurotransmitterModule, ReleaseDriveSpec, OscillationCouplingRule,
)

class MORModule(NeurotransmitterModule):
    """
    Mu-opioid receptor system behavior specification.

    Release drivers: hedonic tone, comfort, satisfaction
    Oscillation coupling: delta -> tonic modulation
    Primary role: hedonic tone, comfort, affective buffering.
    """
    @property
    def name(self) -> str: return "MOR"

    @property
    def release_spec(self) -> ReleaseDriveSpec:
        return ReleaseDriveSpec(
            signal_keys=["hedonic_tone", "comfort", "satisfaction", "emotion_drive"],
            weights=[0.30, 0.25, 0.20, 0.25],
            threshold=0.05,
        )

    @property
    def oscillation_rules(self) -> List[OscillationCouplingRule]:
        return [
            OscillationCouplingRule(target="theta_tonic", band="delta", coefficient=-0.2),
            OscillationCouplingRule(target="sigma_tonic", band="delta", coefficient=-0.15),
        ]
