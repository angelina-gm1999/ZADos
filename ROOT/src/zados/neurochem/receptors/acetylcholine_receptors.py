"""
Acetylcholine receptor family module.

ACh receptors: nicotinic, muscarinic
- nicotinic: Ionotropic (excitatory), fast ligand-gated ion channel
  Na+/K+ permeable, rapid depolarization
- muscarinic: Metabotropic (modulatory), GPCR
  M1-M5 subtypes collapsed to one representative

Emotion plasticity:
- Focus/attention: nicotinic sensitivity ↑ (fast cholinergic bursts)
- Curiosity: muscarinic sensitivity ↑ (sustained attention)
- Fatigue: both sensitivity ↓
"""

from __future__ import annotations

from typing import Dict

from zados.neurochem.receptors.base import ReceptorFamilyModule, ReceptorSpec
from zados.neurochem.receptors.subtype_switching import SubtypeSwitchRule


class AcetylcholineReceptors(ReceptorFamilyModule):
    """
    Acetylcholine receptor family: nicotinic, muscarinic.

    Nicotinic: Fast ionotropic, excitatory. Rapid attention bursts,
    precision enhancement, rule fidelity signaling.

    Muscarinic: Slow metabotropic, modulatory. Sustained attention,
    cortical plasticity, memory encoding support.
    """

    @property
    def parent_nt(self) -> str:
        return "ACh"

    @property
    def receptor_specs(self) -> Dict[str, ReceptorSpec]:
        return {
            "ACh_nicotinic": ReceptorSpec(
                receptor_id="ACh_nicotinic",
                ionotropic=True,  # Fast ligand-gated ion channel
                signaling_type="excitatory",
                effective_signaling_weight=1.0,
                emotion_plasticity_rules={
                    "focus": {"sigma_delta": 0.1, "rho_delta": 0.05},
                    "attention": {"sigma_delta": 0.08},
                    "fatigue": {"sigma_delta": -0.1, "rho_delta": -0.05},
                },
            ),
            "ACh_muscarinic": ReceptorSpec(
                receptor_id="ACh_muscarinic",
                ionotropic=False,
                signaling_type="modulatory",
                effective_signaling_weight=0.9,
                emotion_plasticity_rules={
                    "curiosity": {"sigma_delta": 0.08},
                    "engagement": {"sigma_delta": 0.06},
                    "fatigue": {"sigma_delta": -0.08},
                },
            ),
        }

    @property
    def subtype_switch_rules(self):
        """nicotinic ⇄ muscarinic compensation (ionotropic ⇄ metabotropic)."""
        return [
            SubtypeSwitchRule(
                source_receptor_id="ACh_nicotinic",
                target_receptor_id="ACh_muscarinic",
                exposure_threshold=20.0,
                rho_transfer_rate=0.004,
                max_transfer_per_step=0.018,
            ),
            SubtypeSwitchRule(
                source_receptor_id="ACh_muscarinic",
                target_receptor_id="ACh_nicotinic",
                exposure_threshold=20.0,
                rho_transfer_rate=0.003,
                max_transfer_per_step=0.015,
            ),
        ]
