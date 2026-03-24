"""
Human Attunement domain → NT signal mapping.

Human Attunement domain primarily drives:
    OXT: empathy, social_engagement, trust (social bonding)
    5HT: mood_stability, ambiguity (affect regulation)
    MOR: hedonic_tone, comfort (affective buffering)

Subscore mapping:
    empathetic_inference → OXT empathy
    cognitive_reading → OXT empathy
    intention_calibration → OXT social_engagement, OXT trust
    attuned_dissonance → OXT social_engagement
    emotional_resonance → 5HT mood_stability, MOR hedonic_tone
    rapport_calibration → 5HT mood_stability
    comfort_provision → MOR comfort
"""

from __future__ import annotations

from typing import Dict, List

from zados.neurochem.domains.base import DomainNTMapping, NTSignalMapping


class HumanAttunementMapping(DomainNTMapping):
    """
    Human Attunement domain maps to OXT (social bonding),
    5HT (mood stability), and MOR (hedonic comfort).

    Strong attunement scores drive oxytocin-mediated social engagement,
    serotonergic mood stabilization, and opioid comfort signaling.
    """

    @property
    def domain_name(self) -> str:
        return "human_attunement"

    @property
    def target_nts(self) -> List[str]:
        return ["OXT", "5HT", "MOR"]

    @property
    def signal_mappings(self) -> Dict[str, List[NTSignalMapping]]:
        return {
            "empathetic_inference": [
                NTSignalMapping(nt_name="OXT", signal_key="empathy", weight=0.5),
            ],
            "cognitive_reading": [
                NTSignalMapping(nt_name="OXT", signal_key="empathy", weight=0.5),
            ],
            "intention_calibration": [
                NTSignalMapping(
                    nt_name="OXT", signal_key="social_engagement", weight=0.5,
                ),
                NTSignalMapping(nt_name="OXT", signal_key="trust", weight=0.4),
            ],
            "attuned_dissonance": [
                NTSignalMapping(
                    nt_name="OXT", signal_key="social_engagement", weight=0.5,
                ),
            ],
            "emotional_resonance": [
                NTSignalMapping(
                    nt_name="5HT", signal_key="mood_stability", weight=0.6,
                ),
                NTSignalMapping(
                    nt_name="MOR", signal_key="hedonic_tone", weight=0.5,
                ),
            ],
            "rapport_calibration": [
                NTSignalMapping(
                    nt_name="5HT", signal_key="mood_stability", weight=0.4,
                ),
            ],
            "comfort_provision": [
                NTSignalMapping(nt_name="MOR", signal_key="comfort", weight=0.7),
            ],
        }
