"""Serotonin (5-HT) behavior module."""
from __future__ import annotations
from typing import Dict, List
from zados.neurochem.neurotransmitters.base import (
    NeurotransmitterModule, ReleaseDriveSpec, OscillationCouplingRule,
)

class SerotoninModule(NeurotransmitterModule):
    """
    Serotonin (5-HT) behavior specification.

    Release drivers: mood stability, ambiguity tolerance, time-horizon weighting
    Oscillation coupling: theta -> release, alpha -> noise suppression
    Primary role: affect regulation, ambiguity buffering, long-horizon weighting.
    """
    @property
    def name(self) -> str: return "5HT"

    @property
    def release_spec(self) -> ReleaseDriveSpec:
        return ReleaseDriveSpec(
            signal_keys=["mood_stability", "ambiguity", "horizon_weight", "emotion_drive"],
            weights=[0.30, 0.25, 0.20, 0.25],
            threshold=0.05,
        )

    @property
    def oscillation_rules(self) -> List[OscillationCouplingRule]:
        return [
            OscillationCouplingRule(target="release", band="theta", coefficient=0.4),
            OscillationCouplingRule(target="sigma_tonic", band="alpha", coefficient=-0.4),
            OscillationCouplingRule(target="sigma_phasic", band="alpha", coefficient=-0.3),
        ]
