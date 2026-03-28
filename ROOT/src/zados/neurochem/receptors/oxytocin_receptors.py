"""
Oxytocin receptor family module.

OXT receptor: OXTR (single receptor)
- Gq-coupled (excitatory), GPCR
- Slow peptide dynamics, sustained signaling

Emotion plasticity:
- Trust/bonding: OXTR sensitivity ↑
- Betrayal/isolation: OXTR sensitivity ↓
- Empathy/compassion: OXTR density ↑ (upregulation)
"""

from __future__ import annotations

from typing import Dict

from zados.neurochem.receptors.base import ReceptorFamilyModule, ReceptorSpec


class OxytocinReceptors(ReceptorFamilyModule):
    """
    Oxytocin receptor family: OXTR.

    Single receptor subtype. Gq-coupled GPCR mediating
    social bonding, trust resonance, empathic attunement,
    and prosocial behavior amplification.

    Slow dynamics (peptide signaling) with sustained effects.
    """

    @property
    def parent_nt(self) -> str:
        return "OXT"

    @property
    def receptor_specs(self) -> Dict[str, ReceptorSpec]:
        return {
            "OXTR": ReceptorSpec(
                receptor_id="OXTR",
                ionotropic=False,
                signaling_type="excitatory",
                effective_signaling_weight=1.0,
                emotion_plasticity_rules={
                    "trust": {"sigma_delta": 0.12, "rho_delta": 0.08},
                    "bonding": {"sigma_delta": 0.1},
                    "empathy": {"sigma_delta": 0.08, "rho_delta": 0.05},
                    "compassion": {"sigma_delta": 0.06},
                    "betrayal": {"sigma_delta": -0.12, "rho_delta": -0.05},
                    "isolation": {"sigma_delta": -0.08},
                },
            ),
        }
