"""
Typed receptor configuration.

Wraps the dict-based receptor configs from neurotransmitters/configs.py
as frozen dataclasses for type safety and IDE support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class ReceptorConfig:
    """
    Typed receptor kinetic configuration.

    Attributes
    ----------
    K_d : float
        Dissociation constant (Michaelis-Menten half-saturation).
        Lower K_d = higher affinity.
    parent_nt : str
        Parent neurotransmitter name (e.g., "DA", "5HT")
    exposure_tau : float
        Exposure time constant for desensitization dynamics.
    """
    K_d: float
    parent_nt: str
    exposure_tau: float

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dict format expected by engine."""
        return {
            "K_d": self.K_d,
            "parent_nt": self.parent_nt,
            "exposure_tau": self.exposure_tau,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> ReceptorConfig:
        """
        Create ReceptorConfig from dict.

        Parameters
        ----------
        config : dict
            Dict with keys matching ReceptorConfig fields.

        Returns
        -------
        ReceptorConfig
        """
        return cls(
            K_d=float(config["K_d"]),
            parent_nt=str(config["parent_nt"]),
            exposure_tau=float(config["exposure_tau"]),
        )

    @property
    def affinity(self) -> float:
        """Receptor affinity (inverse of K_d). Higher = stronger binding."""
        if self.K_d == 0:
            return float("inf")
        return 1.0 / self.K_d
