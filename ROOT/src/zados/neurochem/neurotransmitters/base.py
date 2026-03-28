"""
Base classes for per-neurotransmitter behavior modules.

Each NT gets a concrete subclass that specifies:
- Release drive logic (what signals drive phasic bursts, with weights)
- Oscillation coupling rules (which bands modulate which kinetic parameters)
- Optional custom release computation for nonlinear drive logic

The engine calls these modules during its update loop; the modules
do NOT manage state themselves — the engine remains the orchestrator.

Usage
-----
>>> class DAModule(NeurotransmitterModule):
...     @property
...     def name(self) -> str: return "DA"
...     @property
...     def release_spec(self) -> ReleaseDriveSpec: ...
...     @property
...     def oscillation_rules(self) -> list[OscillationCouplingRule]: ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class OscillationCouplingRule:
    """
    A single oscillation band -> kinetic parameter coupling rule.

    Attributes
    ----------
    target : str
        Which kinetic parameter to modulate.
        Valid targets: "release", "reuptake", "sigma_tonic",
        "sigma_phasic", "K_d", "u_base", "d_base", "c_base",
        "theta_tonic", "theta_phasic"
    band : str
        Which oscillation band amplitude to read.
        Valid bands: "delta", "theta", "alpha", "beta", "gamma",
        "theta_gamma" (cross-frequency coupling),
        "alpha_beta" (cross-frequency coupling)
    coefficient : float
        Strength of modulation. Positive = enhance, negative = suppress.
    formula : str
        How to apply the modulation:
        "multiplicative": param *= (1.0 + coefficient * phi_band)
        "additive": param += coefficient * phi_band
    """
    target: str
    band: str
    coefficient: float
    formula: str = "multiplicative"

    def __post_init__(self):
        valid_targets = {
            "release", "reuptake", "sigma_tonic", "sigma_phasic",
            "K_d", "u_base", "d_base", "c_base",
            "theta_tonic", "theta_phasic",
        }
        valid_bands = {
            "delta", "theta", "alpha", "beta", "gamma",
            "theta_gamma", "alpha_beta",
        }
        valid_formulas = {"multiplicative", "additive"}

        if self.target not in valid_targets:
            raise ValueError(
                f"Invalid target {self.target!r}. Must be one of {sorted(valid_targets)}"
            )
        if self.band not in valid_bands:
            raise ValueError(
                f"Invalid band {self.band!r}. Must be one of {sorted(valid_bands)}"
            )
        if self.formula not in valid_formulas:
            raise ValueError(
                f"Invalid formula {self.formula!r}. Must be one of {sorted(valid_formulas)}"
            )


@dataclass(frozen=True)
class ReleaseDriveSpec:
    """
    Specification of what signals drive a neurotransmitter's phasic release.

    Attributes
    ----------
    signal_keys : list of str
        Ordered list of signal key names this NT reads from
        modulation_signals[nt_name].
    weights : list of float
        Corresponding weights for combining into total drive.
        Must have same length as signal_keys.
    threshold : float
        Minimum total drive required to trigger phasic burst.
        Values below threshold produce zero release drive.
    """
    signal_keys: List[str]
    weights: List[float]
    threshold: float = 0.0

    def __post_init__(self):
        if len(self.signal_keys) != len(self.weights):
            raise ValueError(
                f"signal_keys length ({len(self.signal_keys)}) must match "
                f"weights length ({len(self.weights)})"
            )


class NeurotransmitterModule(ABC):
    """
    Abstract base for per-NT behavior specification.

    Subclasses define WHAT drives a specific NT (release logic,
    oscillation coupling). The engine defines HOW to integrate
    (EM stepping, state management).

    Methods with default implementations can be overridden for
    NTs that need custom nonlinear logic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """NT identifier matching configs.py key (e.g., 'DA', '5HT')."""
        ...

    @property
    @abstractmethod
    def release_spec(self) -> ReleaseDriveSpec:
        """Release drive specification for this NT."""
        ...

    @property
    @abstractmethod
    def oscillation_rules(self) -> List[OscillationCouplingRule]:
        """Oscillation coupling rules for this NT."""
        ...

    def compute_release_drive(
        self,
        modulation_signals: Dict[str, float],
    ) -> float:
        """
        Compute total release drive from modulation signals.

        Default implementation: weighted sum of signal values,
        then subtract threshold and clamp to non-negative.

        Override for NTs that need nonlinear drive computation
        (e.g., DA's RPE can be negative).

        Parameters
        ----------
        modulation_signals : dict
            Signal values for this NT, e.g. {"novelty": 0.5, "rpe": 0.3}

        Returns
        -------
        float
            Total release drive (non-negative by default)
        """
        spec = self.release_spec
        total = 0.0
        for key, weight in zip(spec.signal_keys, spec.weights):
            total += weight * modulation_signals.get(key, 0.0)
        return max(0.0, total - spec.threshold)

    def apply_oscillation_coupling(
        self,
        params: Dict[str, float],
        oscillation_amplitudes: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply oscillation coupling rules to kinetic parameters.

        Parameters
        ----------
        params : dict
            Kinetic parameters dict, e.g. {"u_base": 0.1, "sigma_tonic": 0.05, ...}
        oscillation_amplitudes : dict
            Band amplitudes, e.g. {"delta": 0.2, "theta": 0.5, ...}
            May also include cross-frequency products like "theta_gamma".

        Returns
        -------
        dict
            Modified copy of params with coupling applied.
        """
        result = dict(params)
        for rule in self.oscillation_rules:
            phi = oscillation_amplitudes.get(rule.band, 0.0)
            current = result.get(rule.target, 0.0)

            if rule.formula == "multiplicative":
                result[rule.target] = current * (1.0 + rule.coefficient * phi)
            else:  # additive
                result[rule.target] = current + rule.coefficient * phi

        return result

    def get_primary_release_band(self) -> Optional[str]:
        """
        Return the primary oscillation band that gates release for this NT.

        Looks for the first oscillation rule targeting "release".
        Returns None if no release-targeting rules exist.
        """
        for rule in self.oscillation_rules:
            if rule.target == "release":
                return rule.band
        return None

    def get_primary_release_coefficient(self) -> float:
        """
        Return the coefficient of the primary release oscillation rule.

        Returns 0.0 if no release-targeting rules exist.
        """
        for rule in self.oscillation_rules:
            if rule.target == "release":
                return rule.coefficient
        return 0.0
