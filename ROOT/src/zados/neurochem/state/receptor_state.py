from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ReceptorFunctionalState(Enum):
    """
    Functional state of a receptor (from Appendix D).
    
    Models the CTMC (continuous-time Markov chain) transitions:
    active ⇄ desensitized ⇄ internalized
           ↕
      upregulated
    """
    ACTIVE = "active"
    DESENSITIZED = "desensitized"
    INTERNALIZED = "internalized"
    UPREGULATED = "upregulated"


@dataclass
class ReceptorState:
    """
    State container for a single receptor subtype.
    
    Tracks density, sensitivity, functional state, and exposure history
    for plasticity mechanisms (Appendix D).
    
    Attributes
    ----------
    receptor_id : str
        Unique identifier (e.g., "DA_D1", "5HT_2A")
    rho : float
        Receptor density (normalized, 0-1)
    sigma : float
        Receptor sensitivity/affinity scaling (0-1)
    lambda_loc : float
        Localization bias: 0=presynaptic, 0.5=synaptic, 1=extrasynaptic
    gamma_gprotein : float
        G-protein coupling efficacy (0-1)
    chi : ReceptorFunctionalState
        Current functional state (active/desensitized/internalized/upregulated)
    exposure_trace : float
        Slow-decaying trace of ligand exposure (for plasticity rules)
    time_in_state : float
        Duration in current functional state (for hysteresis)
    """
    
    receptor_id: str
    rho: float = 1.0
    sigma: float = 1.0
    lambda_loc: float = 0.5
    gamma_gprotein: float = 1.0
    chi: ReceptorFunctionalState = ReceptorFunctionalState.ACTIVE
    exposure_trace: float = 0.0
    time_in_state: float = 0.0
    
    def __post_init__(self):
        """Enforce bounds on initialization."""
        self.rho = max(0.0, min(1.0, self.rho))
        self.sigma = max(0.0, min(1.0, self.sigma))
        self.lambda_loc = max(0.0, min(1.0, self.lambda_loc))
        self.gamma_gprotein = max(0.0, min(1.0, self.gamma_gprotein))
        self.exposure_trace = max(0.0, self.exposure_trace)
        self.time_in_state = max(0.0, self.time_in_state)
    
    def update_density(self, delta_rho: float):
        """
        Update receptor density.
        
        Parameters
        ----------
        delta_rho : float
            Change in density
        """
        self.rho = max(0.0, min(1.0, self.rho + delta_rho))
    
    def update_sensitivity(self, delta_sigma: float):
        """
        Update receptor sensitivity.
        
        Parameters
        ----------
        delta_sigma : float
            Change in sensitivity
        """
        self.sigma = max(0.0, min(1.0, self.sigma + delta_sigma))

    def update_gamma_gprotein(self, delta_gamma: float):
        """
        Update G-protein coupling efficacy.

        Parameters
        ----------
        delta_gamma : float
            Change in gamma
        """
        self.gamma_gprotein = max(0.0, min(1.0, self.gamma_gprotein + delta_gamma))

    def set_functional_state(self, new_state: ReceptorFunctionalState):
        """
        Transition to a new functional state.
        
        Parameters
        ----------
        new_state : ReceptorFunctionalState
            Target state
        """
        if new_state != self.chi:
            self.chi = new_state
            self.time_in_state = 0.0
    
    def update_exposure_trace(self, saturation: float, dt: float, tau: float = 10.0):
        """
        Update exposure trace with exponential decay.
        
        E(t+dt) = E(t) * exp(-dt/tau) + S(t) * dt
        
        Parameters
        ----------
        saturation : float
            Current receptor saturation S_ij(t)
        dt : float
            Time step
        tau : float, default=10.0
            Decay time constant
        """
        import math
        decay_factor = math.exp(-dt / tau)
        self.exposure_trace = self.exposure_trace * decay_factor + saturation * dt
    
    def increment_time_in_state(self, dt: float):
        """
        Increment the time spent in current state.
        
        Parameters
        ----------
        dt : float
            Time step
        """
        self.time_in_state += dt
    
    def saturation(self, concentration: float, K_d: float) -> float:
        """
        Compute receptor saturation given ligand concentration.
        
        S_ij(t) = C_i(t) / (C_i(t) + K_d)
        
        Parameters
        ----------
        concentration : float
            Ligand concentration C_i(t)
        K_d : float
            Dissociation constant (half-saturation)
            
        Returns
        -------
        float
            Saturation level in [0, 1]
        """
        if concentration <= 0.0 or K_d <= 0.0:
            return 0.0
        return concentration / (concentration + K_d)
    
    def as_dict(self) -> dict:
        """
        Export state as dictionary.
        
        Returns
        -------
        dict
            Dictionary with all state variables
        """
        return {
            "receptor_id": self.receptor_id,
            "rho": self.rho,
            "sigma": self.sigma,
            "lambda_loc": self.lambda_loc,
            "gamma_gprotein": self.gamma_gprotein,
            "chi": self.chi.value,
            "exposure_trace": self.exposure_trace,
            "time_in_state": self.time_in_state,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> ReceptorState:
        """
        Create state from dictionary.
        
        Parameters
        ----------
        data : dict
            Dictionary with state variable values
            
        Returns
        -------
        ReceptorState
            New state instance
        """
        chi_str = data.get("chi", "active")
        chi_state = ReceptorFunctionalState(chi_str)
        
        return cls(
            receptor_id=data["receptor_id"],
            rho=data.get("rho", 1.0),
            sigma=data.get("sigma", 1.0),
            lambda_loc=data.get("lambda_loc", 0.5),
            gamma_gprotein=data.get("gamma_gprotein", 1.0),
            chi=chi_state,
            exposure_trace=data.get("exposure_trace", 0.0),
            time_in_state=data.get("time_in_state", 0.0),
        )
    
    def copy(self) -> ReceptorState:
        """
        Create a deep copy of this state.
        
        Returns
        -------
        ReceptorState
            Independent copy
        """
        return ReceptorState(
            receptor_id=self.receptor_id,
            rho=self.rho,
            sigma=self.sigma,
            lambda_loc=self.lambda_loc,
            gamma_gprotein=self.gamma_gprotein,
            chi=self.chi,
            exposure_trace=self.exposure_trace,
            time_in_state=self.time_in_state,
        )