"""
Logic domain → NT signal mapping.

Logic domain primarily drives:
    NE: precision, uncertainty, contradiction (arousal, salience)
    ACh: attention_demand, rule_fidelity (attention, precision)
    GLU: excitation, integration_demand (fast signal integration)

Subscore mapping:
    epistemic_calibration → NE precision (inverted: low calibration → high precision need)
    internal_consistency → NE precision (inverted), NE contradiction
    external_consistency → NE precision (inverted)
    uncertainty_acknowledgment → NE uncertainty (inverted: low ack → high uncertainty)
    formal_validity → ACh rule_fidelity
    inferential_rigor → ACh attention_demand, GLU integration_demand
"""

from __future__ import annotations

from typing import Dict, List

from zados.neurochem.domains.base import DomainNTMapping, NTSignalMapping


class LogicMapping(DomainNTMapping):
    """
    Logic domain maps to NE (precision/salience), ACh (attention),
    and GLU (integration).

    Low logic scores (inconsistency, poor calibration) drive NE
    error detection. High inferential rigor demands drive ACh attention
    and GLU fast integration.
    """

    @property
    def domain_name(self) -> str:
        return "logic"

    @property
    def target_nts(self) -> List[str]:
        return ["NE", "ACh", "GLU"]

    @property
    def signal_mappings(self) -> Dict[str, List[NTSignalMapping]]:
        return {
            "epistemic_calibration": [
                # Low calibration → high precision need
                NTSignalMapping(
                    nt_name="NE", signal_key="precision",
                    weight=0.5, invert=True,
                ),
            ],
            "internal_consistency": [
                # Low consistency → high precision need
                NTSignalMapping(
                    nt_name="NE", signal_key="precision",
                    weight=0.25, invert=True,
                ),
                # Low consistency → contradiction detection
                NTSignalMapping(
                    nt_name="NE", signal_key="contradiction",
                    weight=0.8, invert=True,
                ),
            ],
            "external_consistency": [
                NTSignalMapping(
                    nt_name="NE", signal_key="precision",
                    weight=0.25, invert=True,
                ),
            ],
            "uncertainty_acknowledgment": [
                # Low acknowledgment → high uncertainty signal
                NTSignalMapping(
                    nt_name="NE", signal_key="uncertainty",
                    weight=1.0, invert=True,
                ),
            ],
            "formal_validity": [
                NTSignalMapping(
                    nt_name="ACh", signal_key="rule_fidelity",
                    weight=0.8,
                ),
            ],
            "inferential_rigor": [
                NTSignalMapping(
                    nt_name="ACh", signal_key="attention_demand",
                    weight=0.7,
                ),
                NTSignalMapping(
                    nt_name="GLU", signal_key="integration_demand",
                    weight=0.6,
                ),
            ],
        }
