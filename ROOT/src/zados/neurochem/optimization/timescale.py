"""
Timescale separation and sparse update scheduling (Appendix N.2).

Exploits the fast/slow variable hierarchy to reduce redundant computation.
Fast variables (concentrations, saturations, oscillation amplitudes) update
every tick; slow variables (receptor plasticity, fatigue, tonic drift) update
every M ticks with scaled dt.

Usage
-----
>>> from zados.neurochem.optimization.timescale import (
...     TimescaleConfig, SparseUpdateScheduler,
... )
>>> scheduler = SparseUpdateScheduler(TimescaleConfig(M_receptor=100))
>>> if scheduler.should_update("receptor", step_number=200):
...     update_receptors(scaled_dt=scheduler.get_scaled_dt("receptor", dt))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TimescaleConfig:
    """
    Configuration for sparse update intervals (N.2.3).

    Parameters
    ----------
    M_receptor : int
        Receptor plasticity (rho, sigma, chi) update interval in ticks.
    M_fatigue : int
        Fatigue accumulation update interval in ticks.
    M_oscillation : int
        Oscillation derivation update interval (state_derived mode).
    """
    M_receptor: int = 100
    M_fatigue: int = 50
    M_oscillation: int = 10


DEFAULT_TIMESCALE_CONFIG = TimescaleConfig()


# Map variable group names to TimescaleConfig field names
_GROUP_TO_FIELD: Dict[str, str] = {
    "receptor": "M_receptor",
    "fatigue": "M_fatigue",
    "oscillation": "M_oscillation",
}


class SparseUpdateScheduler:
    """
    Manages sparse update schedules for slow variables (N.2.2).

    Fast variables (concentrations, phasic bursts) are not managed —
    they always update every tick. Only slow variables are gated.
    """

    def __init__(self, config: TimescaleConfig = DEFAULT_TIMESCALE_CONFIG):
        self.config = config

    def should_update(self, variable_group: str, step_number: int) -> bool:
        """
        Check if a variable group should update on this step.

        Parameters
        ----------
        variable_group : str
            One of: "receptor", "fatigue", "oscillation".
        step_number : int
            Current step number (0-indexed).

        Returns
        -------
        bool
            True if this step is an update tick for the group.
        """
        interval = self.get_interval(variable_group)
        if interval <= 1:
            return True
        return step_number % interval == 0

    def get_interval(self, variable_group: str) -> int:
        """
        Return the M value for a variable group.

        Parameters
        ----------
        variable_group : str
            One of: "receptor", "fatigue", "oscillation".

        Returns
        -------
        int
            Update interval in ticks.

        Raises
        ------
        KeyError
            If variable_group is not recognized.
        """
        field_name = _GROUP_TO_FIELD.get(variable_group)
        if field_name is None:
            raise KeyError(
                f"Unknown variable group '{variable_group}'. "
                f"Valid groups: {list(_GROUP_TO_FIELD.keys())}"
            )
        return getattr(self.config, field_name)

    def get_scaled_dt(self, variable_group: str, dt: float) -> float:
        """
        Return M * dt for slow-variable integration.

        When a slow variable updates every M ticks, it needs to integrate
        over M * dt worth of time to compensate for the sparse schedule.

        Parameters
        ----------
        variable_group : str
            One of: "receptor", "fatigue", "oscillation".
        dt : float
            Base time step.

        Returns
        -------
        float
            Scaled time step (M * dt).
        """
        return self.get_interval(variable_group) * dt
