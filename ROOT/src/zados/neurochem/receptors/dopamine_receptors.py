"""
Dopamine receptor family module.

DA receptors: D1, D2, D3, D4, D5
- D1, D5: Gs-coupled (excitatory), increase cAMP
- D2, D3, D4: Gi-coupled (inhibitory), decrease cAMP
- All metabotropic (GPCRs)

Emotion plasticity:
- Joy/excitement: D1 sensitivity ↑, D2 sensitivity ↑
- Fear/anxiety: D2 sensitivity ↓ (avoidance dampening)
- Curiosity: D4 sensitivity ↑ (novelty-seeking receptor)
"""

from __future__ import annotations

from typing import Dict

from zados.neurochem.receptors.base import ReceptorFamilyModule, ReceptorSpec
from zados.neurochem.receptors.subtype_switching import SubtypeSwitchRule


class DopamineReceptors(ReceptorFamilyModule):
    """
    Dopamine receptor family: D1-D5.

    D1-like (D1, D5): Excitatory, Gs-coupled, fast signaling
    D2-like (D2, D3, D4): Inhibitory, Gi-coupled, modulatory

    D1 drives motivated approach; D2 gates impulsive action;
    D3 provides high-affinity tonic sensing; D4 novelty-seeking;
    D5 cortical excitation.
    """

    @property
    def parent_nt(self) -> str:
        return "DA"

    @property
    def receptor_specs(self) -> Dict[str, ReceptorSpec]:
        return {
            "DA_D1": ReceptorSpec(
                receptor_id="DA_D1",
                ionotropic=False,
                signaling_type="excitatory",
                effective_signaling_weight=1.0,
                emotion_plasticity_rules={
                    "joy": {"sigma_delta": 0.1, "rho_delta": 0.05},
                    "excitement": {"sigma_delta": 0.08},
                    "fear": {"sigma_delta": -0.05},
                },
            ),
            "DA_D2": ReceptorSpec(
                receptor_id="DA_D2",
                ionotropic=False,
                signaling_type="inhibitory",
                effective_signaling_weight=0.9,
                emotion_plasticity_rules={
                    "joy": {"sigma_delta": 0.05},
                    "fear": {"sigma_delta": -0.1},
                    "caution": {"sigma_delta": 0.08},
                },
            ),
            "DA_D3": ReceptorSpec(
                receptor_id="DA_D3",
                ionotropic=False,
                signaling_type="inhibitory",
                effective_signaling_weight=0.8,
                emotion_plasticity_rules={
                    "contentment": {"sigma_delta": 0.05},
                },
            ),
            "DA_D4": ReceptorSpec(
                receptor_id="DA_D4",
                ionotropic=False,
                signaling_type="modulatory",
                effective_signaling_weight=0.85,
                emotion_plasticity_rules={
                    "curiosity": {"sigma_delta": 0.12, "rho_delta": 0.05},
                    "boredom": {"sigma_delta": -0.08},
                },
            ),
            "DA_D5": ReceptorSpec(
                receptor_id="DA_D5",
                ionotropic=False,
                signaling_type="excitatory",
                effective_signaling_weight=0.95,
                emotion_plasticity_rules={
                    "excitement": {"sigma_delta": 0.08},
                },
            ),
        }

    @property
    def subtype_switch_rules(self):
        """D1 ⇄ D2 homeostatic compensation under sustained activation."""
        return [
            SubtypeSwitchRule(
                source_receptor_id="DA_D1",
                target_receptor_id="DA_D2",
                exposure_threshold=20.0,
                rho_transfer_rate=0.005,
                max_transfer_per_step=0.02,
            ),
            SubtypeSwitchRule(
                source_receptor_id="DA_D2",
                target_receptor_id="DA_D1",
                exposure_threshold=20.0,
                rho_transfer_rate=0.003,
                max_transfer_per_step=0.015,
            ),
        ]
