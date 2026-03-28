"""Cortisol behavior module."""
from __future__ import annotations
from typing import Dict, List
from zados.neurochem.neurotransmitters.base import (
    NeurotransmitterModule, ReleaseDriveSpec, OscillationCouplingRule,
)

class CortisolModule(NeurotransmitterModule):
    """
    Cortisol behavior specification.

    Release drivers: stress level, time pressure, tradeoff load
    Oscillation coupling: beta -> release
    Primary role: time-horizon pressure, tradeoff enforcement, stress weighting.
    """
    @property
    def name(self) -> str: return "cortisol"

    @property
    def release_spec(self) -> ReleaseDriveSpec:
        return ReleaseDriveSpec(
            signal_keys=["stress_level", "time_pressure", "tradeoff_load", "emotion_drive"],
            weights=[0.30, 0.25, 0.20, 0.25],
            threshold=0.05,
        )

    @property
    def oscillation_rules(self) -> List[OscillationCouplingRule]:
        return [
            OscillationCouplingRule(target="release", band="beta", coefficient=0.25),
            OscillationCouplingRule(target="theta_tonic", band="delta", coefficient=-0.15),
        ]
