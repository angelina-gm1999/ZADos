"""
Norepinephrine receptor family module.

NE receptors: alpha1, alpha2, beta1, beta2
- alpha1: Gq-coupled (excitatory), arousal and attention
- alpha2: Gi-coupled (inhibitory), autoreceptor — gain control
- beta1: Gs-coupled (excitatory), cardiac/arousal amplification
- beta2: Gs-coupled (excitatory), peripheral/bronchial

Emotion plasticity:
- Alertness/vigilance: alpha1 sensitivity ↑
- Fear/threat: beta1 sensitivity ↑ (fight-or-flight)
- Calm: alpha2 sensitivity ↑ (inhibitory gain control)
"""

from __future__ import annotations

from typing import Dict

from zados.neurochem.receptors.base import ReceptorFamilyModule, ReceptorSpec
from zados.neurochem.receptors.subtype_switching import SubtypeSwitchRule


class NorepinephrineReceptors(ReceptorFamilyModule):
    """
    Norepinephrine receptor family: alpha1, alpha2, beta1, beta2.

    alpha1: Excitatory, arousal amplification, attention focusing.
    alpha2: Inhibitory autoreceptor, gain control, noise gating.
    beta1: Excitatory, stress-responsive arousal, fight-or-flight.
    beta2: Excitatory, peripheral/modulatory.
    """

    @property
    def parent_nt(self) -> str:
        return "NE"

    @property
    def receptor_specs(self) -> Dict[str, ReceptorSpec]:
        return {
            "NE_alpha1": ReceptorSpec(
                receptor_id="NE_alpha1",
                ionotropic=False,
                signaling_type="excitatory",
                effective_signaling_weight=1.0,
                emotion_plasticity_rules={
                    "alertness": {"sigma_delta": 0.1, "rho_delta": 0.05},
                    "vigilance": {"sigma_delta": 0.08},
                    "calm": {"sigma_delta": -0.05},
                },
            ),
            "NE_alpha2": ReceptorSpec(
                receptor_id="NE_alpha2",
                ionotropic=False,
                signaling_type="inhibitory",
                effective_signaling_weight=0.9,
                emotion_plasticity_rules={
                    "calm": {"sigma_delta": 0.1},
                    "focus": {"sigma_delta": 0.06},
                    "panic": {"sigma_delta": -0.1},
                },
            ),
            "NE_beta1": ReceptorSpec(
                receptor_id="NE_beta1",
                ionotropic=False,
                signaling_type="excitatory",
                effective_signaling_weight=0.95,
                emotion_plasticity_rules={
                    "fear": {"sigma_delta": 0.12},
                    "threat": {"sigma_delta": 0.1},
                    "safety": {"sigma_delta": -0.08},
                },
            ),
            "NE_beta2": ReceptorSpec(
                receptor_id="NE_beta2",
                ionotropic=False,
                signaling_type="excitatory",
                effective_signaling_weight=0.85,
                emotion_plasticity_rules={
                    "arousal": {"sigma_delta": 0.06},
                },
            ),
        }

    @property
    def subtype_switch_rules(self):
        """alpha1 ⇄ alpha2 homeostatic compensation (excitatory ⇄ inhibitory)."""
        return [
            SubtypeSwitchRule(
                source_receptor_id="NE_alpha1",
                target_receptor_id="NE_alpha2",
                exposure_threshold=20.0,
                rho_transfer_rate=0.004,
                max_transfer_per_step=0.018,
            ),
            SubtypeSwitchRule(
                source_receptor_id="NE_alpha2",
                target_receptor_id="NE_alpha1",
                exposure_threshold=20.0,
                rho_transfer_rate=0.003,
                max_transfer_per_step=0.015,
            ),
        ]
