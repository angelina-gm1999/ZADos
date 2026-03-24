"""
GABA receptor family module.

GABA receptors: GABA_A, GABA_B
- GABA_A: Ionotropic (inhibitory), fast ligand-gated Cl- channel
  Mediates fast phasic inhibition, shunting inhibition
- GABA_B: Metabotropic (inhibitory), Gi-coupled GPCR
  Mediates slow tonic inhibition, K+ conductance increase

Emotion plasticity:
- Calm/safety: GABA_A sensitivity ↑ (fast inhibitory gating)
- Anxiety: GABA_A sensitivity ↓ (disinhibition / anxiogenic)
- Restraint: GABA_B sensitivity ↑ (tonic suppression)
"""

from __future__ import annotations

from typing import Dict

from zados.neurochem.receptors.base import ReceptorFamilyModule, ReceptorSpec
from zados.neurochem.receptors.subtype_switching import SubtypeSwitchRule


class GABAReceptors(ReceptorFamilyModule):
    """
    GABA receptor family: GABA_A, GABA_B.

    GABA_A: Fast ionotropic inhibition. Cl- channel.
    Phasic inhibition, shunting, cortical gating, anxiolytic action.

    GABA_B: Slow metabotropic inhibition. Gi-coupled.
    Tonic inhibition, K+ conductance, presynaptic suppression.
    Ethics/boundary enforcement via K_d feedback pathway.
    """

    @property
    def parent_nt(self) -> str:
        return "GABA"

    @property
    def receptor_specs(self) -> Dict[str, ReceptorSpec]:
        return {
            "GABA_A": ReceptorSpec(
                receptor_id="GABA_A",
                ionotropic=True,  # Fast Cl- channel
                signaling_type="inhibitory",
                effective_signaling_weight=1.0,
                emotion_plasticity_rules={
                    "calm": {"sigma_delta": 0.1, "rho_delta": 0.05},
                    "safety": {"sigma_delta": 0.08},
                    "anxiety": {"sigma_delta": -0.12},
                    "panic": {"sigma_delta": -0.15},
                    "restraint": {"sigma_delta": 0.06},
                },
            ),
            "GABA_B": ReceptorSpec(
                receptor_id="GABA_B",
                ionotropic=False,
                signaling_type="inhibitory",
                effective_signaling_weight=0.9,
                emotion_plasticity_rules={
                    "restraint": {"sigma_delta": 0.1},
                    "caution": {"sigma_delta": 0.08},
                    "calm": {"sigma_delta": 0.05},
                    "impulsivity": {"sigma_delta": -0.1},
                },
            ),
        }

    @property
    def subtype_switch_rules(self):
        """GABA_A ⇄ GABA_B compensation (fast ionotropic ⇄ slow metabotropic)."""
        return [
            SubtypeSwitchRule(
                source_receptor_id="GABA_A",
                target_receptor_id="GABA_B",
                exposure_threshold=20.0,
                rho_transfer_rate=0.004,
                max_transfer_per_step=0.018,
            ),
            SubtypeSwitchRule(
                source_receptor_id="GABA_B",
                target_receptor_id="GABA_A",
                exposure_threshold=20.0,
                rho_transfer_rate=0.003,
                max_transfer_per_step=0.015,
            ),
        ]
