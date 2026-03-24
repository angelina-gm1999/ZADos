"""
Typed neurotransmitter configuration.

Wraps the dict-based NT configs from neurotransmitters/configs.py
as frozen dataclasses for type safety and IDE support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class NTConfig:
    """
    Typed neurotransmitter kinetic configuration.

    Attributes
    ----------
    C_baseline : float
        Tonic baseline concentration [0, 1]
    theta_tonic : float
        Tonic mean-reversion rate (Ornstein-Uhlenbeck)
    theta_phasic : float
        Phasic decay rate
    sigma_tonic : float
        Tonic noise amplitude
    sigma_phasic : float
        Phasic noise amplitude
    u_base : float
        Reuptake rate coefficient
    d_base : float
        Degradation rate coefficient
    c_base : float
        Clearance/diffusion rate coefficient
    """
    C_baseline: float
    theta_tonic: float
    theta_phasic: float
    sigma_tonic: float
    sigma_phasic: float
    u_base: float
    d_base: float
    c_base: float

    def as_dict(self) -> Dict[str, float]:
        """Convert to dict format expected by engine."""
        return {
            "C_baseline": self.C_baseline,
            "theta_tonic": self.theta_tonic,
            "theta_phasic": self.theta_phasic,
            "sigma_tonic": self.sigma_tonic,
            "sigma_phasic": self.sigma_phasic,
            "u_base": self.u_base,
            "d_base": self.d_base,
            "c_base": self.c_base,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> NTConfig:
        """
        Create NTConfig from dict.

        Parameters
        ----------
        config : dict
            Dict with keys matching NTConfig fields.

        Returns
        -------
        NTConfig

        Raises
        ------
        KeyError
            If required keys are missing.
        """
        return cls(
            C_baseline=float(config["C_baseline"]),
            theta_tonic=float(config["theta_tonic"]),
            theta_phasic=float(config["theta_phasic"]),
            sigma_tonic=float(config["sigma_tonic"]),
            sigma_phasic=float(config["sigma_phasic"]),
            u_base=float(config["u_base"]),
            d_base=float(config["d_base"]),
            c_base=float(config["c_base"]),
        )

    @property
    def total_clearance_rate(self) -> float:
        """Total clearance rate: reuptake + degradation + diffusion."""
        return self.u_base + self.d_base + self.c_base

    @property
    def tonic_snr(self) -> float:
        """Tonic signal-to-noise ratio (baseline / noise)."""
        if self.sigma_tonic == 0:
            return float("inf")
        return self.C_baseline / self.sigma_tonic
