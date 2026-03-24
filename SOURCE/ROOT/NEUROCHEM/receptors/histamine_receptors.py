"""
Histamine receptor family module.

Histamine receptors: HIST_H1, HIST_H2, HIST_H3, HIST_H4
- H1: Gq-coupled (excitatory), arousal, wakefulness
- H2: Gs-coupled (excitatory), gastric acid, cardiac, cognitive
- H3: Gi-coupled (inhibitory), autoreceptor, presynaptic feedback
- H4: Gi-coupled (modulatory), immune modulation, peripheral

Emotion plasticity:
- Focus/alertness: H1 sensitivity ↑, H3 sensitivity ↓ (disinhibition)
- Fatigue/sleep: H1 sensitivity ↓, H3 sensitivity ↑ (suppression)
- Curiosity: H1 density ↑ (arousal for exploration)
"""

from __future__ import annotations

from typing import Dict

from zados.neurochem.receptors.base import ReceptorFamilyModule, ReceptorSpec


class HistamineReceptors(ReceptorFamilyModule):
    """
    Histamine receptor family: H1, H2, H3, H4.

    Tuberomammillary nucleus projection system.
    H1/H2 are postsynaptic excitatory; H3 is a presynaptic
    autoreceptor providing negative feedback; H4 is primarily
    peripheral/immune.
    """

    @property
    def parent_nt(self) -> str:
        return "histamine"

    @property
    def receptor_specs(self) -> Dict[str, ReceptorSpec]:
        return {
            "HIST_H1": ReceptorSpec(
                receptor_id="HIST_H1",
                ionotropic=False,
                signaling_type="excitatory",
                effective_signaling_weight=1.0,
                emotion_plasticity_rules={
                    "focus": {"sigma_delta": 0.1, "rho_delta": 0.05},
                    "curiosity": {"rho_delta": 0.06},
                    "calm": {"sigma_delta": -0.08},
                    "sadness": {"sigma_delta": -0.06},
                },
            ),
            "HIST_H2": ReceptorSpec(
                receptor_id="HIST_H2",
                ionotropic=False,
                signaling_type="excitatory",
                effective_signaling_weight=0.8,
                emotion_plasticity_rules={
                    "focus": {"sigma_delta": 0.06},
                    "anxiety": {"sigma_delta": 0.04},
                    "calm": {"sigma_delta": -0.05},
                },
            ),
            "HIST_H3": ReceptorSpec(
                receptor_id="HIST_H3",
                ionotropic=False,
                signaling_type="inhibitory",
                effective_signaling_weight=0.6,
                emotion_plasticity_rules={
                    "focus": {"sigma_delta": -0.08},
                    "calm": {"sigma_delta": 0.06},
                    "sadness": {"sigma_delta": 0.05},
                },
            ),
            "HIST_H4": ReceptorSpec(
                receptor_id="HIST_H4",
                ionotropic=False,
                signaling_type="modulatory",
                effective_signaling_weight=0.4,
                emotion_plasticity_rules={
                    "anxiety": {"sigma_delta": 0.04},
                },
            ),
        }
