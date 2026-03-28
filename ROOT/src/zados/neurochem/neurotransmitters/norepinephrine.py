"""Norepinephrine (NE) behavior module."""
from __future__ import annotations
from typing import Dict, List
from zados.neurochem.neurotransmitters.base import (
    NeurotransmitterModule, ReleaseDriveSpec, OscillationCouplingRule,
)

class NEModule(NeurotransmitterModule):
    """
    Norepinephrine behavior specification.

    Release drivers: precision demand, uncertainty, contradiction detection
    Oscillation coupling: beta -> release, alpha -> noise suppression
    Primary role: arousal, salience, contradiction detection, gain control.
    """
    @property
    def name(self) -> str: return "NE"

    @property
    def release_spec(self) -> ReleaseDriveSpec:
        return ReleaseDriveSpec(
            signal_keys=["precision", "uncertainty", "contradiction", "emotion_drive"],
            weights=[0.30, 0.25, 0.20, 0.25],
            threshold=0.05,
        )

    @property
    def oscillation_rules(self) -> List[OscillationCouplingRule]:
        return [
            OscillationCouplingRule(target="release", band="beta", coefficient=0.3),
            OscillationCouplingRule(target="sigma_tonic", band="alpha", coefficient=-0.4),
            OscillationCouplingRule(target="sigma_phasic", band="alpha", coefficient=-0.3),
        ]
