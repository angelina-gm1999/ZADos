"""
Ethics domain → NT signal mapping.

Ethics domain primarily drives:
    GABA: inhibition, boundary_proximity (suppression, gating)
    cortisol: stress_level, time_pressure (stress response)
    CRH: acute_stress, pressure_scaling (acute stress trigger)

Subscore mapping:
    failure_mode_awareness → GABA inhibition (high awareness → more gating)
    downstream_risk_amplification → GABA boundary_proximity (inverted: low score → close to boundary)
    harm_mitigation → GABA inhibition
    temporal_consequence → cortisol time_pressure
    stakes_assessment → cortisol stress_level, CRH acute_stress
    ethical_consistency → CRH pressure_scaling (inverted)
"""

from __future__ import annotations

from typing import Dict, List

from zados.neurochem.domains.base import DomainNTMapping, NTSignalMapping


class EthicsMapping(DomainNTMapping):
    """
    Ethics domain maps to GABA (inhibition/boundary),
    cortisol (stress/time-horizon), and CRH (acute stress).

    High ethics concern drives GABAergic inhibition for behavioral gating,
    cortisol for time-horizon pressure, and CRH for acute stress response
    when ethical boundaries are approached.
    """

    @property
    def domain_name(self) -> str:
        return "ethics"

    @property
    def target_nts(self) -> List[str]:
        return ["GABA", "cortisol", "CRH"]

    @property
    def signal_mappings(self) -> Dict[str, List[NTSignalMapping]]:
        return {
            "failure_mode_awareness": [
                NTSignalMapping(
                    nt_name="GABA", signal_key="inhibition", weight=0.6,
                ),
            ],
            "downstream_risk_amplification": [
                # Low risk amplification score → close to ethical boundary
                NTSignalMapping(
                    nt_name="GABA", signal_key="boundary_proximity",
                    weight=0.8, invert=True,
                ),
            ],
            "harm_mitigation": [
                NTSignalMapping(
                    nt_name="GABA", signal_key="inhibition", weight=0.4,
                ),
            ],
            "temporal_consequence": [
                NTSignalMapping(
                    nt_name="cortisol", signal_key="time_pressure", weight=0.7,
                ),
            ],
            "stakes_assessment": [
                NTSignalMapping(
                    nt_name="cortisol", signal_key="stress_level", weight=0.6,
                ),
                NTSignalMapping(
                    nt_name="CRH", signal_key="acute_stress", weight=0.5,
                ),
            ],
            "ethical_consistency": [
                # Low ethical consistency → high pressure scaling
                NTSignalMapping(
                    nt_name="CRH", signal_key="pressure_scaling",
                    weight=0.7, invert=True,
                ),
            ],
        }
