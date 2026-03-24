"""
Endogenous opioid receptor family module.

MOR receptor: MOR_mu (μ-opioid receptor)
- Gi-coupled (inhibitory), GPCR
- Mediates analgesia, hedonic tone, reward consummation

Emotion plasticity:
- Contentment/comfort: MOR_mu sensitivity ↑
- Pain/distress: MOR_mu sensitivity ↓ (protective desensitization)
- Pleasure/hedonia: MOR_mu density ↑
"""

from __future__ import annotations

from typing import Dict

from zados.neurochem.receptors.base import ReceptorFamilyModule, ReceptorSpec


class OpioidReceptors(ReceptorFamilyModule):
    """
    Endogenous opioid receptor family: MOR_mu.

    μ-opioid receptor (MOR). Gi-coupled GPCR mediating
    hedonic tone, comfort signaling, affective buffering,
    and reward consummation.

    Slow peptide dynamics with tonic modulation emphasis.
    """

    @property
    def parent_nt(self) -> str:
        return "MOR"

    @property
    def receptor_specs(self) -> Dict[str, ReceptorSpec]:
        return {
            "MOR_mu": ReceptorSpec(
                receptor_id="MOR_mu",
                ionotropic=False,
                signaling_type="inhibitory",
                effective_signaling_weight=1.0,
                emotion_plasticity_rules={
                    "contentment": {"sigma_delta": 0.1, "rho_delta": 0.05},
                    "comfort": {"sigma_delta": 0.08},
                    "pleasure": {"sigma_delta": 0.06, "rho_delta": 0.04},
                    "pain": {"sigma_delta": -0.1},
                    "distress": {"sigma_delta": -0.08},
                },
            ),
        }
