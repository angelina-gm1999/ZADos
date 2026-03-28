from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import math


@dataclass
class OscillationState:
    """
    State container for neural oscillation bands.
    
    Tracks amplitude envelopes φ_k(t) for each frequency band
    as defined in Appendix H/I.
    
    Bands:
    - delta (δ): 0.5-4 Hz - recovery/reset, slow inhibition
    - theta (θ): 4-8 Hz - simulation/narrative, temporal binding
    - alpha (α): 8-12 Hz - inhibitory gating, noise suppression
    - beta (β): 13-30 Hz - precision/contradiction detection
    - gamma (γ): 30-100+ Hz - binding/integration
    - sigma (σ): 12-15 Hz - sleep spindle / thalamocortical replay (sleep-only)

    Attributes
    ----------
    delta : float
        Delta band envelope amplitude φ_δ(t) ∈ [0, 1]
    theta : float
        Theta band envelope amplitude φ_θ(t) ∈ [0, 1]
    alpha : float
        Alpha band envelope amplitude φ_α(t) ∈ [0, 1]
    beta : float
        Beta band envelope amplitude φ_β(t) ∈ [0, 1]
    gamma : float
        Gamma band envelope amplitude φ_γ(t) ∈ [0, 1]
    sigma : float
        Sigma band envelope amplitude φ_σ(t) ∈ [0, 1].
        Sleep-only band (0.0 during waking).
    phase : Dict[str, float]
        Phase angles for each band (radians)
    """
    
    delta: float = 0.0
    theta: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    sigma: float = 0.0
    phase: Dict[str, float] = field(default_factory=lambda: {
        "delta": 0.0,
        "theta": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "gamma": 0.0,
        "sigma": 0.0,
    })

    def __post_init__(self):
        """Enforce bounds on initialization."""
        self.delta = max(0.0, min(1.0, self.delta))
        self.theta = max(0.0, min(1.0, self.theta))
        self.alpha = max(0.0, min(1.0, self.alpha))
        self.beta = max(0.0, min(1.0, self.beta))
        self.gamma = max(0.0, min(1.0, self.gamma))
        self.sigma = max(0.0, min(1.0, self.sigma))
    
    def set_band(self, band: str, amplitude: float):
        """
        Set amplitude for a specific band.
        
        Parameters
        ----------
        band : str
            Band name ('delta', 'theta', 'alpha', 'beta', 'gamma')
        amplitude : float
            Amplitude value (will be clamped to [0, 1])
        """
        amplitude = max(0.0, min(1.0, amplitude))
        
        if band == "delta":
            self.delta = amplitude
        elif band == "theta":
            self.theta = amplitude
        elif band == "alpha":
            self.alpha = amplitude
        elif band == "beta":
            self.beta = amplitude
        elif band == "gamma":
            self.gamma = amplitude
        elif band == "sigma":
            self.sigma = amplitude
        else:
            raise ValueError(f"Unknown band: {band}")
    
    def get_band(self, band: str) -> float:
        """
        Get amplitude for a specific band.
        
        Parameters
        ----------
        band : str
            Band name
            
        Returns
        -------
        float
            Amplitude value
        """
        if band == "delta":
            return self.delta
        elif band == "theta":
            return self.theta
        elif band == "alpha":
            return self.alpha
        elif band == "beta":
            return self.beta
        elif band == "gamma":
            return self.gamma
        elif band == "sigma":
            return self.sigma
        else:
            raise ValueError(f"Unknown band: {band}")
    
    def set_phase(self, band: str, phase: float):
        """
        Set phase for a specific band.
        
        Parameters
        ----------
        band : str
            Band name
        phase : float
            Phase in radians
        """
        if band not in self.phase:
            raise ValueError(f"Unknown band: {band}")
        self.phase[band] = phase
    
    def get_phase(self, band: str) -> float:
        """
        Get phase for a specific band.
        
        Parameters
        ----------
        band : str
            Band name
            
        Returns
        -------
        float
            Phase in radians
        """
        if band not in self.phase:
            raise ValueError(f"Unknown band: {band}")
        return self.phase[band]
    
    def theta_gamma_coupling(self) -> float:
        """
        Compute theta-gamma cross-frequency coupling.
        
        φ_θγ(t) = φ_θ(t) · φ_γ(t)
        
        Returns
        -------
        float
            Coupling index ∈ [0, 1]
        """
        return self.theta * self.gamma
    
    def alpha_beta_coupling(self) -> float:
        """
        Compute alpha-beta cross-frequency coupling.

        φ_αβ(t) = φ_α(t) · φ_β(t)

        Returns
        -------
        float
            Coupling index ∈ [0, 1]
        """
        return self.alpha * self.beta

    def delta_sigma_coupling(self) -> float:
        """
        Compute delta-sigma cross-frequency coupling (NREM consolidation).

        φ_δσ(t) = φ_δ(t) · φ_σ(t)

        Spindles nested in slow oscillation Up-states. High φ_δσ → active
        memory replay window during NREM sleep processing.

        Returns
        -------
        float
            Coupling index ∈ [0, 1]
        """
        return self.delta * self.sigma
    
    def normalize(self):
        """
        Normalize band amplitudes so they sum to 1.

        Useful for ensuring total oscillatory power is bounded.
        """
        total = self.delta + self.theta + self.alpha + self.beta + self.gamma + self.sigma
        if total > 0.0:
            self.delta /= total
            self.theta /= total
            self.alpha /= total
            self.beta /= total
            self.gamma /= total
            self.sigma /= total
    
    def as_dict(self) -> dict:
        """
        Export state as dictionary.
        
        Returns
        -------
        dict
            Dictionary with all band amplitudes and phases
        """
        return {
            "delta": self.delta,
            "theta": self.theta,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "sigma": self.sigma,
            "phase": dict(self.phase),
            "theta_gamma_coupling": self.theta_gamma_coupling(),
            "alpha_beta_coupling": self.alpha_beta_coupling(),
            "delta_sigma_coupling": self.delta_sigma_coupling(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> OscillationState:
        """
        Create state from dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary with band amplitude and phase values
            
        Returns
        -------
        OscillationState
            New state instance
        """
        phase = data.get("phase", {
            "delta": 0.0,
            "theta": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
            "sigma": 0.0,
        })

        return cls(
            delta=data.get("delta", 0.0),
            theta=data.get("theta", 0.0),
            alpha=data.get("alpha", 0.0),
            beta=data.get("beta", 0.0),
            gamma=data.get("gamma", 0.0),
            sigma=data.get("sigma", 0.0),
            phase=phase,
        )
    
    def copy(self) -> OscillationState:
        """
        Create a deep copy of this state.
        
        Returns
        -------
        OscillationState
            Independent copy
        """
        return OscillationState(
            delta=self.delta,
            theta=self.theta,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            sigma=self.sigma,
            phase=dict(self.phase),
        )
    
    @classmethod
    def bands(cls) -> list[str]:
        """
        Return list of all band names.
        
        Returns
        -------
        list[str]
            Band names
        """
        return ["delta", "theta", "alpha", "beta", "gamma", "sigma"]