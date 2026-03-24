"""
Serotonin receptor family module.

5-HT receptors: 1A, 1B, 2A, 2C, 3
- 5HT_1A: Gi-coupled (inhibitory), autoreceptor — mood stability
- 5HT_1B: Gi-coupled (inhibitory), terminal autoreceptor
- 5HT_2A: Gq-coupled (excitatory), psychedelic/flexibility receptor
- 5HT_2C: Gq-coupled (modulatory), anxiety/appetite regulation
- 5HT_3: Ionotropic (excitatory), fast ligand-gated ion channel

Emotion plasticity:
- Calm/serenity: 1A sensitivity ↑ (stability)
- Anxiety: 2C sensitivity ↑, 1A sensitivity ↓
- Openness: 2A sensitivity ↑ (flexibility / perceptual broadening)
"""

from __future__ import annotations

from typing import Dict

from zados.neurochem.receptors.base import ReceptorFamilyModule, ReceptorSpec
from zados.neurochem.receptors.subtype_switching import SubtypeSwitchRule


class SerotoninReceptors(ReceptorFamilyModule):
    """
    Serotonin (5-HT) receptor family: 1A, 1B, 2A, 2C, 3.

    5HT_1A: Inhibitory autoreceptor, mood stabilization, affect regulation.
    5HT_1B: Terminal autoreceptor, release gating.
    5HT_2A: Excitatory, cognitive flexibility, perceptual broadening.
    5HT_2C: Modulatory, anxiety regulation, impulse control.
    5HT_3: Ionotropic (only non-GPCR serotonin receptor), fast excitatory.
    """

    @property
    def parent_nt(self) -> str:
        return "5HT"

    @property
    def receptor_specs(self) -> Dict[str, ReceptorSpec]:
        return {
            "5HT_1A": ReceptorSpec(
                receptor_id="5HT_1A",
                ionotropic=False,
                signaling_type="inhibitory",
                effective_signaling_weight=1.0,
                emotion_plasticity_rules={
                    "calm": {"sigma_delta": 0.1},
                    "serenity": {"sigma_delta": 0.08},
                    "anxiety": {"sigma_delta": -0.1},
                    "distress": {"sigma_delta": -0.08},
                },
            ),
            "5HT_1B": ReceptorSpec(
                receptor_id="5HT_1B",
                ionotropic=False,
                signaling_type="inhibitory",
                effective_signaling_weight=0.85,
                emotion_plasticity_rules={
                    "calm": {"sigma_delta": 0.05},
                },
            ),
            "5HT_2A": ReceptorSpec(
                receptor_id="5HT_2A",
                ionotropic=False,
                signaling_type="excitatory",
                effective_signaling_weight=1.0,
                emotion_plasticity_rules={
                    "openness": {"sigma_delta": 0.12, "rho_delta": 0.05},
                    "curiosity": {"sigma_delta": 0.08},
                    "rigidity": {"sigma_delta": -0.1},
                },
            ),
            "5HT_2C": ReceptorSpec(
                receptor_id="5HT_2C",
                ionotropic=False,
                signaling_type="modulatory",
                effective_signaling_weight=0.9,
                emotion_plasticity_rules={
                    "anxiety": {"sigma_delta": 0.1},
                    "fear": {"sigma_delta": 0.08},
                    "calm": {"sigma_delta": -0.05},
                },
            ),
            "5HT_3": ReceptorSpec(
                receptor_id="5HT_3",
                ionotropic=True,  # Only ionotropic 5-HT receptor
                signaling_type="excitatory",
                effective_signaling_weight=0.8,
                emotion_plasticity_rules={
                    "nausea_distress": {"sigma_delta": 0.1},
                },
            ),
        }

    @property
    def subtype_switch_rules(self):
        """1A ⇄ 2A homeostatic compensation (inhibitory ⇄ excitatory balance)."""
        return [
            SubtypeSwitchRule(
                source_receptor_id="5HT_1A",
                target_receptor_id="5HT_2A",
                exposure_threshold=20.0,
                rho_transfer_rate=0.004,
                max_transfer_per_step=0.018,
            ),
            SubtypeSwitchRule(
                source_receptor_id="5HT_2A",
                target_receptor_id="5HT_1A",
                exposure_threshold=20.0,
                rho_transfer_rate=0.004,
                max_transfer_per_step=0.018,
            ),
        ]
