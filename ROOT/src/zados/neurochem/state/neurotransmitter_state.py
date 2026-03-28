from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NeurotransmitterState:
    """
    State container for a single neurotransmitter system.
    
    Represents the dynamical variables that evolve during simulation,
    as distinct from static parameters (stored separately in configs).
    
    Follows the concentration decomposition from Appendix F:
        C(t) = C_tonic(t) + C_phasic(t)
    
    Attributes
    ----------
    C_tonic : float
        Baseline (tonic) concentration component
    C_phasic : float
        Burst (phasic) concentration component
    F : float
        Fatigue level (0 to 1), affects reuptake efficiency
    eta_u : float
        Transporter efficiency scaling factor (affected by fatigue)
    """
    
    C_tonic: float = 0.5
    C_phasic: float = 0.0
    F: float = 0.0
    eta_u: float = 1.0
    
    def __post_init__(self):
        """Enforce bounds on initialization."""
        self.C_tonic = max(0.0, self.C_tonic)
        self.C_phasic = max(0.0, self.C_phasic)
        self.F = max(0.0, min(1.0, self.F))
        self.eta_u = max(0.0, min(1.0, self.eta_u))
    
    @property
    def C(self) -> float:
        """
        Total concentration: C(t) = C_tonic(t) + C_phasic(t).
        
        Returns
        -------
        float
            Total neurotransmitter concentration
        """
        return self.C_tonic + self.C_phasic
    
    def update_concentration(self, delta_C: float, is_phasic: bool = False):
        """
        Update concentration component by a delta.
        
        Parameters
        ----------
        delta_C : float
            Change in concentration
        is_phasic : bool, default=False
            If True, update phasic component; otherwise update tonic
        """
        if is_phasic:
            self.C_phasic = max(0.0, self.C_phasic + delta_C)
        else:
            self.C_tonic = max(0.0, self.C_tonic + delta_C)
    
    def set_concentration(self, C_total: float):
        """
        Set total concentration (allocates to tonic, zeroes phasic).
        
        Parameters
        ----------
        C_total : float
            New total concentration value
        """
        self.C_tonic = max(0.0, C_total)
        self.C_phasic = 0.0
    
    def update_fatigue(self, delta_F: float):
        """
        Update fatigue level.
        
        Parameters
        ----------
        delta_F : float
            Change in fatigue
        """
        self.F = max(0.0, min(1.0, self.F + delta_F))
    
    def update_transporter_efficiency(self, delta_eta: float):
        """
        Update transporter efficiency.
        
        Parameters
        ----------
        delta_eta : float
            Change in transporter efficiency
        """
        self.eta_u = max(0.0, min(1.0, self.eta_u + delta_eta))
    
    def as_dict(self) -> dict:
        """
        Export state as dictionary.
        
        Returns
        -------
        dict
            Dictionary with all state variables
        """
        return {
            "C_tonic": self.C_tonic,
            "C_phasic": self.C_phasic,
            "C": self.C,
            "F": self.F,
            "eta_u": self.eta_u,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> NeurotransmitterState:
        """
        Create state from dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary with state variable values
            
        Returns
        -------
        NeurotransmitterState
            New state instance
        """
        return cls(
            C_tonic=data.get("C_tonic", 0.5),
            C_phasic=data.get("C_phasic", 0.0),
            F=data.get("F", 0.0),
            eta_u=data.get("eta_u", 1.0),
        )
    
    def copy(self) -> NeurotransmitterState:
        """
        Create a deep copy of this state.
        
        Returns
        -------
        NeurotransmitterState
            Independent copy
        """
        return NeurotransmitterState(
            C_tonic=self.C_tonic,
            C_phasic=self.C_phasic,
            F=self.F,
            eta_u=self.eta_u,
        )