"""Acetylcholine (ACh) behavior module."""
from __future__ import annotations
from typing import Dict, List
from zados.neurochem.neurotransmitters.base import (
    NeurotransmitterModule, ReleaseDriveSpec, OscillationCouplingRule,
)

class AChModule(NeurotransmitterModule):
    """
    Acetylcholine behavior specification.

    Release drivers: attention demand, rule fidelity, precision weighting
    Oscillation coupling: beta -> release, gamma -> release
    Primary role: precision, attention, rule fidelity.
    """
    @property
    def name(self) -> str: return "ACh"

    @property
    def release_spec(self) -> ReleaseDriveSpec:
        return ReleaseDriveSpec(
            signal_keys=["attention_demand", "rule_fidelity", "precision_weight", "emotion_drive"],
            weights=[0.30, 0.25, 0.20, 0.25],
            threshold=0.05,
        )

    @property
    def oscillation_rules(self) -> List[OscillationCouplingRule]:
        return [
            OscillationCouplingRule(target="release", band="beta", coefficient=0.3),
            OscillationCouplingRule(target="release", band="gamma", coefficient=0.25),
            OscillationCouplingRule(target="sigma_tonic", band="alpha", coefficient=-0.3),
        ]
