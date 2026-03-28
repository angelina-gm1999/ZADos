"""Endocannabinoid (CB1) behavior module."""
from __future__ import annotations
from typing import Dict, List
from zados.neurochem.neurotransmitters.base import (
    NeurotransmitterModule, ReleaseDriveSpec, OscillationCouplingRule,
)

class CB1Module(NeurotransmitterModule):
    """
    Endocannabinoid (CB1) behavior specification.

    Release drivers: flexibility, filter suppression, affective continuity
    Oscillation coupling: delta -> tonic, alpha_beta -> cross-frequency coupling
    Primary role: flexibility, filter inhibition, affective continuity.
    """
    @property
    def name(self) -> str: return "CB1"

    @property
    def release_spec(self) -> ReleaseDriveSpec:
        return ReleaseDriveSpec(
            signal_keys=["flexibility", "filter_suppression", "continuity", "emotion_drive"],
            weights=[0.30, 0.25, 0.20, 0.25],
            threshold=0.05,
        )

    @property
    def oscillation_rules(self) -> List[OscillationCouplingRule]:
        return [
            OscillationCouplingRule(target="theta_tonic", band="delta", coefficient=-0.2),
            OscillationCouplingRule(target="sigma_tonic", band="alpha_beta", coefficient=-0.2),
        ]
