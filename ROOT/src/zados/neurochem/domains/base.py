"""
Base classes for domain-to-NT mapping modules.

A DomainNTMapping defines how a cognitive evaluation domain's subscores
translate into neurotransmitter modulation signals. This bridges the
reward system's evaluations with the neurochemical engine's per-NT
module signal inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass(frozen=True)
class NTSignalMapping:
    """
    Specification for how a subscore maps to an NT signal.

    Attributes
    ----------
    nt_name : str
        Target neurotransmitter (e.g., "DA", "5HT")
    signal_key : str
        Signal key in the NT module's release spec (e.g., "novelty", "precision")
    weight : float
        Scaling weight applied to the subscore. Default 1.0.
    invert : bool
        If True, signal = weight * (1.0 - subscore). Useful for
        mapping "low consistency" → "high precision need".
    offset : float
        Constant offset added after scaling. Default 0.0.
        Useful for centering signals (e.g., RPE around 0).
    """
    nt_name: str
    signal_key: str
    weight: float = 1.0
    invert: bool = False
    offset: float = 0.0

    def compute(self, subscore: float) -> float:
        """
        Transform a subscore into an NT modulation signal value.

        Parameters
        ----------
        subscore : float
            Raw subscore value, typically in [0, 1]

        Returns
        -------
        float
            Transformed signal value
        """
        if self.invert:
            value = self.weight * (1.0 - subscore)
        else:
            value = self.weight * subscore
        return value + self.offset


class DomainNTMapping(ABC):
    """
    Abstract base for domain-to-NT signal mapping.

    Each domain (Innovation, Logic, Human Attunement, Ethics) gets
    a concrete subclass that defines:
    - Which NTs this domain primarily drives
    - How subscores map to NT signal keys
    - Optional domain-level modulation logic
    """

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Name of the cognitive evaluation domain."""
        ...

    @property
    @abstractmethod
    def target_nts(self) -> List[str]:
        """List of NT names this domain primarily affects."""
        ...

    @property
    @abstractmethod
    def signal_mappings(self) -> Dict[str, List[NTSignalMapping]]:
        """
        Map of subscore_name -> list of NTSignalMapping.

        Each subscore can map to multiple NT signals.
        """
        ...

    def map_subscores(
        self,
        subscores: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """
        Map domain subscores to NT modulation signals.

        Parameters
        ----------
        subscores : dict
            Domain subscores, where each value is either a float
            or a dict with a "score" key.

        Returns
        -------
        dict
            {nt_name: {signal_key: value, ...}, ...}
        """
        result: Dict[str, Dict[str, float]] = {}

        for subscore_name, mappings in self.signal_mappings.items():
            # Extract score from subscore (handle both float and dict)
            raw = subscores.get(subscore_name, None)
            if raw is None:
                continue
            if isinstance(raw, dict):
                score = raw.get("score", 0.0)
            else:
                score = float(raw)

            for mapping in mappings:
                value = mapping.compute(score)
                if mapping.nt_name not in result:
                    result[mapping.nt_name] = {}
                # If multiple subscores map to the same signal key,
                # accumulate (average later if needed)
                if mapping.signal_key in result[mapping.nt_name]:
                    result[mapping.nt_name][mapping.signal_key] += value
                else:
                    result[mapping.nt_name][mapping.signal_key] = value

        return result

    def get_mappings_for_nt(self, nt_name: str) -> List[NTSignalMapping]:
        """Get all signal mappings that target a specific NT."""
        result = []
        for mappings in self.signal_mappings.values():
            for m in mappings:
                if m.nt_name == nt_name:
                    result.append(m)
        return result
