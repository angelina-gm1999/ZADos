"""
Endocannabinoid receptor family module.

CB1 receptor: CB1 (cannabinoid type 1)
- Gi-coupled (inhibitory), GPCR
- Retrograde signaling: released postsynaptically, acts presynaptically
- Very slow lipid signaling dynamics

Note: CB1 receptor_id == NT name. Config merge handled in configs.py.

Emotion plasticity:
- Flexibility/openness: CB1 sensitivity ↑
- Rigidity/control: CB1 sensitivity ↓
- Relaxation: CB1 density ↑
"""

from __future__ import annotations

from typing import Dict

from zados.neurochem.receptors.base import ReceptorFamilyModule, ReceptorSpec


class CannabinoidReceptors(ReceptorFamilyModule):
    """
    Endocannabinoid receptor family: CB1.

    Cannabinoid type 1 receptor. Gi-coupled GPCR mediating
    retrograde inhibition, cognitive flexibility,
    filter suppression, and affective continuity.

    Very slow lipid signaling dynamics.
    Unique retrograde signaling: postsynaptic release → presynaptic action.
    """

    @property
    def parent_nt(self) -> str:
        return "CB1"

    @property
    def receptor_specs(self) -> Dict[str, ReceptorSpec]:
        return {
            "CB1": ReceptorSpec(
                receptor_id="CB1",
                ionotropic=False,
                signaling_type="inhibitory",
                effective_signaling_weight=1.0,
                emotion_plasticity_rules={
                    "flexibility": {"sigma_delta": 0.1, "rho_delta": 0.05},
                    "openness": {"sigma_delta": 0.08},
                    "relaxation": {"sigma_delta": 0.06, "rho_delta": 0.04},
                    "rigidity": {"sigma_delta": -0.1},
                    "control": {"sigma_delta": -0.06},
                },
            ),
        }
