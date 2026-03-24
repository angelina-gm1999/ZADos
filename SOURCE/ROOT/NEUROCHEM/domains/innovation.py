"""
Innovation domain → NT signal mapping.

Innovation domain primarily drives:
    DA: novelty, rpe, effort (motivation, reward prediction)
    CB1: flexibility, filter_suppression (cognitive flexibility)

Subscore mapping:
    novelty_generation → DA novelty
    conceptual_novelty → DA novelty
    exploration_drive → DA rpe (positive = worth exploring)
    resolution_satisfaction → DA rpe (progress = positive RPE)
    pattern_divergence → CB1 flexibility
    structural_creativity → CB1 filter_suppression
"""

from __future__ import annotations

from typing import Dict, List

from zados.neurochem.domains.base import DomainNTMapping, NTSignalMapping


class InnovationMapping(DomainNTMapping):
    """
    Innovation domain maps to DA (motivation/novelty) and CB1 (flexibility).

    High innovation scores drive dopaminergic phasic bursts via novelty
    and RPE signals, and endocannabinoid flexibility via filter suppression.
    """

    @property
    def domain_name(self) -> str:
        return "innovation"

    @property
    def target_nts(self) -> List[str]:
        return ["DA", "CB1"]

    @property
    def signal_mappings(self) -> Dict[str, List[NTSignalMapping]]:
        return {
            "novelty_generation": [
                NTSignalMapping(nt_name="DA", signal_key="novelty", weight=0.5),
            ],
            "conceptual_novelty": [
                NTSignalMapping(nt_name="DA", signal_key="novelty", weight=0.5),
            ],
            "exploration_drive": [
                NTSignalMapping(
                    nt_name="DA", signal_key="rpe",
                    weight=0.5, offset=-0.25,  # Center around 0
                ),
            ],
            "resolution_satisfaction": [
                NTSignalMapping(
                    nt_name="DA", signal_key="rpe",
                    weight=0.5, offset=-0.25,  # Center around 0
                ),
            ],
            "pattern_divergence": [
                NTSignalMapping(nt_name="CB1", signal_key="flexibility", weight=0.8),
            ],
            "structural_creativity": [
                NTSignalMapping(
                    nt_name="CB1", signal_key="filter_suppression", weight=0.7,
                ),
            ],
        }
