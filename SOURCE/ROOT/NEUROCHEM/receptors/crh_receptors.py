"""
CRH (Corticotropin-Releasing Hormone) receptor family module.

CRH receptor: CRH_R1
- Gs-coupled (excitatory), GPCR
- HPA axis activation, acute stress signaling

Emotion plasticity:
- Stress/threat: CRH_R1 sensitivity ↑ (acute stress amplification)
- Safety/calm: CRH_R1 sensitivity ↓ (stress dampening)
- Anxiety: CRH_R1 density ↑ (chronic stress sensitization)
"""

from __future__ import annotations

from typing import Dict

from zados.neurochem.receptors.base import ReceptorFamilyModule, ReceptorSpec


class CRHReceptors(ReceptorFamilyModule):
    """
    CRH receptor family: CRH_R1.

    Corticotropin-releasing hormone receptor type 1.
    Gs-coupled GPCR mediating acute stress drive,
    HPA axis activation, and pressure scaling.

    Fast peptide dynamics relative to cortisol.
    """

    @property
    def parent_nt(self) -> str:
        return "CRH"

    @property
    def receptor_specs(self) -> Dict[str, ReceptorSpec]:
        return {
            "CRH_R1": ReceptorSpec(
                receptor_id="CRH_R1",
                ionotropic=False,
                signaling_type="excitatory",
                effective_signaling_weight=1.0,
                emotion_plasticity_rules={
                    "stress": {"sigma_delta": 0.12, "rho_delta": 0.06},
                    "threat": {"sigma_delta": 0.1},
                    "anxiety": {"sigma_delta": 0.08, "rho_delta": 0.05},
                    "safety": {"sigma_delta": -0.1},
                    "calm": {"sigma_delta": -0.08},
                },
            ),
        }
