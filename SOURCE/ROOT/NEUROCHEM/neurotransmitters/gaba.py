"""GABA behavior module."""
from __future__ import annotations
from typing import Dict, List
from zados.neurochem.neurotransmitters.base import (
    NeurotransmitterModule, ReleaseDriveSpec, OscillationCouplingRule,
)

class GABAModule(NeurotransmitterModule):
    """
    GABA behavior specification.

    Release drivers: inhibition demand, boundary proximity, suppression
    Oscillation coupling: alpha -> release (inhibitory gating), delta -> tonic
    Primary role: suppression, gating, stabilization.
    """
    @property
    def name(self) -> str: return "GABA"

    @property
    def release_spec(self) -> ReleaseDriveSpec:
        return ReleaseDriveSpec(
            signal_keys=["inhibition", "boundary_proximity", "suppression", "emotion_drive"],
            weights=[0.30, 0.25, 0.20, 0.25],
            threshold=0.05,
        )

    @property
    def oscillation_rules(self) -> List[OscillationCouplingRule]:
        return [
            OscillationCouplingRule(target="release", band="alpha", coefficient=0.35),
            OscillationCouplingRule(target="theta_tonic", band="delta", coefficient=-0.2),
        ]
