"""Glutamate (GLU) behavior module."""
from __future__ import annotations
from typing import Dict, List
from zados.neurochem.neurotransmitters.base import (
    NeurotransmitterModule, ReleaseDriveSpec, OscillationCouplingRule,
)

class GLUModule(NeurotransmitterModule):
    """
    Glutamate behavior specification.

    Release drivers: excitation, integration demand, signal propagation
    Oscillation coupling: gamma -> release, beta -> release
    Primary role: fast signal propagation, high-resolution integration.
    """
    @property
    def name(self) -> str: return "GLU"

    @property
    def release_spec(self) -> ReleaseDriveSpec:
        return ReleaseDriveSpec(
            signal_keys=["excitation", "integration_demand", "signal_propagation", "emotion_drive"],
            weights=[0.30, 0.25, 0.20, 0.25],
            threshold=0.05,
        )

    @property
    def oscillation_rules(self) -> List[OscillationCouplingRule]:
        return [
            OscillationCouplingRule(target="release", band="gamma", coefficient=0.35),
            OscillationCouplingRule(target="release", band="beta", coefficient=0.2),
            OscillationCouplingRule(target="sigma_tonic", band="alpha", coefficient=-0.3),
        ]
