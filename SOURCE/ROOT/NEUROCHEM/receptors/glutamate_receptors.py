"""
Glutamate receptor family module.

GLU receptors: NMDA, AMPA, Kainate, mGluR
- AMPA: Ionotropic (excitatory), fast Na+/K+ channel
  Mediates fast excitatory postsynaptic potentials
- Kainate: Ionotropic (excitatory), Na+/K+ channel
  Modulatory role in synaptic transmission
- NMDA: Ionotropic (excitatory), Ca2+/Na+ channel
  Voltage-dependent Mg2+ block, coincidence detector
  Critical for plasticity (LTP/LTD)
- mGluR: Metabotropic (modulatory), GPCR
  Slow neuromodulation, synaptic plasticity

Emotion plasticity:
- Learning/engagement: NMDA sensitivity ↑ (plasticity gate)
- Overwhelm/excitotoxicity: AMPA sensitivity ↓ (protective)
- Focus: mGluR sensitivity ↑ (sustained modulation)
"""

from __future__ import annotations

from typing import Dict

from zados.neurochem.receptors.base import ReceptorFamilyModule, ReceptorSpec
from zados.neurochem.receptors.subtype_switching import SubtypeSwitchRule


class GlutamateReceptors(ReceptorFamilyModule):
    """
    Glutamate receptor family: NMDA, AMPA, Kainate, mGluR.

    AMPA: Fast ionotropic excitation. Primary mediator of
    fast excitatory transmission. Signal propagation.

    Kainate: Ionotropic excitation. Modulatory role in
    presynaptic release probability and network oscillations.

    NMDA: Ionotropic excitation with voltage-dependent Mg2+ block.
    Coincidence detector for Hebbian plasticity (LTP/LTD).
    Critical for learning and memory consolidation.

    mGluR: Slow metabotropic modulation. Long-term synaptic
    plasticity, homeostatic scaling, neuroprotection.
    """

    @property
    def parent_nt(self) -> str:
        return "GLU"

    @property
    def receptor_specs(self) -> Dict[str, ReceptorSpec]:
        return {
            "GLU_NMDA": ReceptorSpec(
                receptor_id="GLU_NMDA",
                ionotropic=True,  # Ca2+/Na+ channel with Mg2+ block
                signaling_type="excitatory",
                effective_signaling_weight=1.0,
                emotion_plasticity_rules={
                    "learning": {"sigma_delta": 0.12, "rho_delta": 0.05},
                    "engagement": {"sigma_delta": 0.08},
                    "overwhelm": {"sigma_delta": -0.1},
                    "fatigue": {"sigma_delta": -0.06},
                },
            ),
            "GLU_AMPA": ReceptorSpec(
                receptor_id="GLU_AMPA",
                ionotropic=True,  # Fast Na+/K+ channel
                signaling_type="excitatory",
                effective_signaling_weight=1.0,
                emotion_plasticity_rules={
                    "alertness": {"sigma_delta": 0.08},
                    "overwhelm": {"sigma_delta": -0.12},
                    "excitotoxic_stress": {"sigma_delta": -0.15},
                },
            ),
            "GLU_KAINATE": ReceptorSpec(
                receptor_id="GLU_KAINATE",
                ionotropic=True,  # Na+/K+ channel
                signaling_type="excitatory",
                effective_signaling_weight=0.85,
                emotion_plasticity_rules={
                    "engagement": {"sigma_delta": 0.06},
                },
            ),
            "GLU_mGluR": ReceptorSpec(
                receptor_id="GLU_mGluR",
                ionotropic=False,  # Metabotropic GPCR
                signaling_type="modulatory",
                effective_signaling_weight=0.9,
                emotion_plasticity_rules={
                    "focus": {"sigma_delta": 0.1},
                    "plasticity": {"sigma_delta": 0.08, "rho_delta": 0.04},
                    "rigidity": {"sigma_delta": -0.06},
                },
            ),
        }

    @property
    def subtype_switch_rules(self):
        """NMDA ⇄ AMPA compensation (plasticity ⇄ fast transmission)."""
        return [
            SubtypeSwitchRule(
                source_receptor_id="GLU_NMDA",
                target_receptor_id="GLU_AMPA",
                exposure_threshold=20.0,
                rho_transfer_rate=0.004,
                max_transfer_per_step=0.018,
            ),
            SubtypeSwitchRule(
                source_receptor_id="GLU_AMPA",
                target_receptor_id="GLU_NMDA",
                exposure_threshold=20.0,
                rho_transfer_rate=0.003,
                max_transfer_per_step=0.015,
            ),
        ]
