from __future__ import annotations

from typing import Dict, Optional
import math


def compute_reuptake_loss(
    C: float,
    eta_u: float,
    u_base: float = 0.1,
) -> float:
    """
    Compute concentration loss due to reuptake.
    
    L_u(t) = u_base · η_u(t) · C(t)
    
    Reuptake is proportional to current concentration and transporter efficiency.
    
    Parameters
    ----------
    C : float
        Current concentration
    eta_u : float
        Transporter efficiency ∈ [0, 1]
    u_base : float, default=0.1
        Base reuptake rate constant
        
    Returns
    -------
    float
        Reuptake loss rate (positive value)
    """
    return u_base * eta_u * C


def compute_degradation_loss(
    C: float,
    d_base: float = 0.05,
) -> float:
    """
    Compute concentration loss due to enzymatic degradation.
    
    L_d(t) = d_base · C(t)
    
    Degradation is first-order kinetics (proportional to concentration).
    
    Parameters
    ----------
    C : float
        Current concentration
    d_base : float, default=0.05
        Base degradation rate constant
        
    Returns
    -------
    float
        Degradation loss rate (positive value)
    """
    return d_base * C


def compute_clearance_loss(
    C: float,
    c_base: float = 0.02,
    C_extracell: float = 0.0,
) -> float:
    """
    Compute concentration loss due to diffusion/clearance (Fick's law).

    d_i(t) = c_base · (C(t) - C_extracell)

    Diffusion is driven by the gradient between synaptic concentration
    and the extracellular (ambient) concentration.  When C < C_extracell
    the gradient reverses, but we clamp to >= 0 because loss functions
    are always non-negative.

    Parameters
    ----------
    C : float
        Current synaptic concentration
    c_base : float, default=0.02
        Base diffusion rate constant (k_d in §3)
    C_extracell : float, default=0.0
        Extracellular (ambient) concentration.
        Reserved for future multi-compartment modeling where ambient
        NT levels (e.g., tonic volume transmission) influence local
        clearance gradients. Currently defaults to 0.0 (single
        compartment).

    Returns
    -------
    float
        Clearance loss rate (non-negative)
    """
    return max(0.0, c_base * (C - C_extracell))


def compute_total_loss(
    C: float,
    eta_u: float,
    u_base: float = 0.1,
    d_base: float = 0.05,
    c_base: float = 0.02,
    C_extracell: float = 0.0,
) -> float:
    """
    Compute total concentration loss from all mechanisms.

    L_total(t) = L_u(t) + L_d(t) + L_c(t)

    Parameters
    ----------
    C : float
        Current concentration
    eta_u : float
        Transporter efficiency
    u_base : float, default=0.1
        Base reuptake rate
    d_base : float, default=0.05
        Base degradation rate
    c_base : float, default=0.02
        Base clearance rate
    C_extracell : float, default=0.0
        Extracellular concentration for Fick's law diffusion

    Returns
    -------
    float
        Total loss rate (positive value)
    """
    L_u = compute_reuptake_loss(C, eta_u, u_base)
    L_d = compute_degradation_loss(C, d_base)
    L_c = compute_clearance_loss(C, c_base, C_extracell)

    return L_u + L_d + L_c


def compute_drift_term(
    C: float,
    C_baseline: float,
    theta: float,
    eta_u: float,
    u_base: float = 0.1,
    d_base: float = 0.05,
    c_base: float = 0.02,
    C_extracell: float = 0.0,
) -> float:
    """
    Compute drift term for concentration SDE (Ornstein-Uhlenbeck with losses).

    μ(C, t) = -θ(C - C_baseline) - L_total(t)

    The OU mean-reversion term ``-θ(C - C_baseline)`` is an intentional
    extension beyond the bare mass-balance equation in Appendix §3.  It
    provides exponential convergence toward the homeostatic baseline,
    improving dynamic stability and preventing unconstrained drift under
    sustained input.  The loss term follows §3 exactly:
    ``L_total = u·η_u·C + d·C + c·(C - C_extracell)``.

    Parameters
    ----------
    C : float
        Current concentration
    C_baseline : float
        Baseline/homeostatic concentration
    theta : float
        Mean reversion rate (how quickly it returns to baseline)
    eta_u : float
        Transporter efficiency
    u_base : float, default=0.1
        Base reuptake rate
    d_base : float, default=0.05
        Base degradation rate
    c_base : float, default=0.02
        Base clearance rate
    C_extracell : float, default=0.0
        Extracellular concentration for Fick's law diffusion

    Returns
    -------
    float
        Drift term (can be positive or negative)

    Notes
    -----
    Positive drift → concentration increases
    Negative drift → concentration decreases
    """
    # Mean reversion component (OU extension — see §3 note in docstring)
    reversion = -theta * (C - C_baseline)

    # Loss component (always negative drift)
    loss = -compute_total_loss(C, eta_u, u_base, d_base, c_base, C_extracell)

    return reversion + loss


def compute_diffusion_term(
    C: float,
    sigma: float = 0.05,
    multiplicative: bool = True,
) -> float:
    """
    Compute diffusion term for concentration SDE.
    
    σ(C, t) = σ_base · √C   (multiplicative noise, default)
    σ(C, t) = σ_base        (additive noise)
    
    Multiplicative noise naturally enforces positivity (noise → 0 as C → 0).
    
    Parameters
    ----------
    C : float
        Current concentration
    sigma : float, default=0.05
        Base volatility/noise scaling
    multiplicative : bool, default=True
        If True, use multiplicative noise √C; if False, use additive
        
    Returns
    -------
    float
        Diffusion coefficient (non-negative)
    """
    if multiplicative:
        return sigma * math.sqrt(max(0.0, C))
    else:
        return sigma


def compute_mass_balance_drift(
    C_tonic: float,
    C_phasic: float,
    C_baseline: float,
    theta_tonic: float,
    theta_phasic: float,
    eta_u: float,
    u_base: float = 0.1,
    d_base: float = 0.05,
    c_base: float = 0.02,
    C_extracell: float = 0.0,
) -> tuple[float, float]:
    """
    Compute drift terms for tonic and phasic components separately.

    Tonic: slow baseline concentration, weak mean reversion
    Phasic: burst component, strong mean reversion (rapid decay)

    Parameters
    ----------
    C_tonic : float
        Current tonic concentration
    C_phasic : float
        Current phasic concentration
    C_baseline : float
        Baseline concentration (target for tonic)
    theta_tonic : float
        Tonic mean reversion rate (typically low, ~0.1)
    theta_phasic : float
        Phasic mean reversion rate (typically high, ~1.0)
    eta_u : float
        Transporter efficiency
    u_base : float, default=0.1
        Base reuptake rate
    d_base : float, default=0.05
        Base degradation rate
    c_base : float, default=0.02
        Base clearance rate
    C_extracell : float, default=0.0
        Extracellular concentration for Fick's law diffusion

    Returns
    -------
    tuple[float, float]
        (drift_tonic, drift_phasic)
    """
    # Tonic drift: slow return to baseline
    drift_tonic = compute_drift_term(
        C_tonic,
        C_baseline,
        theta_tonic,
        eta_u,
        u_base,
        d_base,
        c_base,
        C_extracell,
    )

    # Phasic drift: rapid decay to zero
    drift_phasic = compute_drift_term(
        C_phasic,
        0.0,  # Phasic decays to zero, not to baseline
        theta_phasic,
        eta_u,
        u_base,
        d_base,
        c_base,
        C_extracell,
    )

    return drift_tonic, drift_phasic


def compute_effective_reversion_rate(
    theta_base: float,
    fatigue: float,
    fatigue_scaling: float = 0.5,
) -> float:
    """
    Compute effective mean reversion rate modulated by fatigue.
    
    θ_eff = θ_base · (1 - fatigue_scaling · F)
    
    High fatigue → slower recovery (lower reversion rate).
    
    Parameters
    ----------
    theta_base : float
        Base mean reversion rate
    fatigue : float
        Current fatigue level ∈ [0, 1]
    fatigue_scaling : float, default=0.5
        How much fatigue reduces reversion rate
        
    Returns
    -------
    float
        Effective reversion rate (non-negative)
    """
    scaling_factor = 1.0 - fatigue_scaling * fatigue
    return max(0.0, theta_base * scaling_factor)