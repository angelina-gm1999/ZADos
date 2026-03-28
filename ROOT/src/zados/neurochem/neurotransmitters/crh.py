"""CRH (Corticotropin-Releasing Hormone) behavior module."""
from __future__ import annotations
from typing import Dict, List
from zados.neurochem.neurotransmitters.base import (
    NeurotransmitterModule, ReleaseDriveSpec, OscillationCouplingRule,
)

class CRHModule(NeurotransmitterModule):
    """
    CRH behavior specification.

    Release drivers: acute stress, pressure scaling
    Oscillation coupling: beta -> release
    Primary role: acute stress drive, pressure scaling.
    """
    @property
    def name(self) -> str: return "CRH"

    @property
    def release_spec(self) -> ReleaseDriveSpec:
        return ReleaseDriveSpec(
            signal_keys=["acute_stress", "pressure_scaling", "emotion_drive"],
            weights=[0.40, 0.30, 0.30],
            threshold=0.05,
        )

    @property
    def oscillation_rules(self) -> List[OscillationCouplingRule]:
        return [
            OscillationCouplingRule(target="release", band="beta", coefficient=0.3),
        ]
