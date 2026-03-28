"""
Pure functions for oscillatory modulation of neurochemical parameters.

Implements the PDF-specified oscillation → kinetics coupling equations:
- K_d modulation (theta lowers receptor affinity)
- Release modulation (gamma boosts phasic release)
- Noise modulation (alpha suppresses noise)
- Effective signaling proxy (A_ij = rho * sigma * gamma * g(chi) * S)

All functions are stateless and side-effect-free.

Usage
-----
>>> from zados.neurochem.oscillations.oscillation_modulation import (
...     modulate_K_d, modulate_release, modulate_noise,
...     compute_effective_signaling_proxy, compute_g_chi,
... )
>>> K_d_eff = modulate_K_d(0.3, phi_theta=0.7)
>>> release_eff = modulate_release(0.5, phi_gamma=0.8)
"""

from __future__ import annotations

from typing import Dict


def modulate_K_d(
    K_d_base: float,
    phi_theta: float,
    kd_coefficient: float = 0.3,
) -> float:
    """
    Oscillatory modulation of receptor binding affinity.

    K_d(t) = K_d_base * (1 - kd_coefficient * phi_theta(t))

    Higher theta -> lower K_d -> higher binding affinity.

    From PDF Appendix E.2.1, H.3.

    Parameters
    ----------
    K_d_base : float
        Baseline dissociation constant
    phi_theta : float
        Theta band oscillation amplitude in [0, 1]
    kd_coefficient : float, default=0.3
        K_d modulation strength. Must be in [0, 1) to keep K_d positive.
        (Not to be confused with the alpha oscillation band.)

    Returns
    -------
    float
        Effective K_d (always positive, clamped to >= 0.01)
    """
    K_d_eff = K_d_base * (1.0 - kd_coefficient * phi_theta)
    return max(0.01, K_d_eff)


def modulate_release(
    base_release: float,
    phi_gamma: float,
    coefficient: float = 0.5,
) -> float:
    """
    Gamma band boosts phasic release.

    R_mod = R_base * (1 + coefficient * phi_gamma)

    From PDF Appendix E.2.2, H.4.

    Parameters
    ----------
    base_release : float
        Baseline release drive
    phi_gamma : float
        Gamma band oscillation amplitude in [0, 1]
    coefficient : float, default=0.5
        Release boost strength

    Returns
    -------
    float
        Modulated release drive
    """
    return base_release * (1.0 + coefficient * phi_gamma)


def modulate_noise(
    sigma_base: float,
    phi_alpha: float,
    coefficient: float = 0.4,
) -> float:
    """
    Alpha band suppresses noise.

    sigma_mod = sigma_base * max(0.1, 1 - coefficient * phi_alpha)

    High alpha -> lower noise floor (more stable dynamics).

    From PDF Appendix H.6.

    The ``max(0.1, ...)`` floor is a numerical stability extension that
    prevents the noise coefficient from reaching zero.  A zero-noise
    state would collapse the SDE to a deterministic ODE and could cause
    the system to lock onto a fixed point, losing the stochastic
    exploration that the neurochemical layer relies on.  The 10 % floor
    preserves a minimal level of endogenous variability even under
    maximal alpha suppression.

    Parameters
    ----------
    sigma_base : float
        Baseline noise/volatility coefficient
    phi_alpha : float
        Alpha band oscillation amplitude in [0, 1]
    coefficient : float, default=0.4
        Noise suppression strength

    Returns
    -------
    float
        Modulated sigma (always >= 0.1 * sigma_base to prevent zero noise)
    """
    suppression = max(0.1, 1.0 - coefficient * phi_alpha)
    return sigma_base * suppression


def modulate_reuptake(
    u_base: float,
    phi_beta: float,
    coefficient: float = 0.3,
) -> float:
    """
    Beta band modulates reuptake rate.

    u_mod = u_base * (1 + coefficient * phi_beta)

    High beta -> faster reuptake (tighter regulation).

    Parameters
    ----------
    u_base : float
        Baseline reuptake rate
    phi_beta : float
        Beta band oscillation amplitude in [0, 1]
    coefficient : float, default=0.3
        Reuptake modulation strength

    Returns
    -------
    float
        Modulated reuptake rate
    """
    return u_base * (1.0 + coefficient * phi_beta)


def modulate_tonic_baseline(
    C_baseline: float,
    phi_delta: float,
    coefficient: float = 0.2,
) -> float:
    """
    Delta band modulates tonic baseline.

    C_baseline_mod = C_baseline * (1 - coefficient * phi_delta)

    High delta -> lowered tonic baseline (recovery/rest state).

    Parameters
    ----------
    C_baseline : float
        Baseline tonic concentration target
    phi_delta : float
        Delta band oscillation amplitude in [0, 1]
    coefficient : float, default=0.2
        Baseline modulation strength

    Returns
    -------
    float
        Modulated baseline (clamped to [0.01, 1.0])
    """
    modulated = C_baseline * (1.0 - coefficient * phi_delta)
    return max(0.01, min(1.0, modulated))


def compute_g_chi(functional_state: str) -> float:
    """
    Compute the functional state gating factor g(chi).

    Maps receptor functional state to a signaling efficiency multiplier.

    From PDF Appendix D.1.6.

    Parameters
    ----------
    functional_state : str
        One of: "ACTIVE", "DESENSITIZED", "INTERNALIZED", "UPREGULATED"

    Returns
    -------
    float
        State gate factor in [0, 1.2]
    """
    G_CHI_MAP = {
        "ACTIVE": 1.0,
        "DESENSITIZED": 0.5,
        "INTERNALIZED": 0.1,
        "UPREGULATED": 1.2,
    }
    return G_CHI_MAP.get(functional_state, 1.0)


def compute_effective_signaling_proxy(
    rho: float,
    sigma: float,
    g_chi: float,
    saturation: float,
    gamma_gprotein: float = 1.0,
) -> float:
    """
    Compute effective signaling proxy A_ij.

    A_ij = rho * sigma * gamma * g(chi) * S_ij

    Combines receptor density, sensitivity, G-protein coupling,
    functional state gating, and ligand binding saturation into
    a single effective signaling value.

    From PDF Appendix D.1.6.

    Parameters
    ----------
    rho : float
        Receptor density in [0, 1]
    sigma : float
        Receptor sensitivity in [0, 1]
    g_chi : float
        Functional state gate (from compute_g_chi)
    saturation : float
        Ligand binding saturation S_ij in [0, 1]
    gamma_gprotein : float, default=1.0
        G-protein coupling efficacy in [0, 1]

    Returns
    -------
    float
        Effective signaling proxy (non-negative, typically in [0, ~1.2])
    """
    return rho * sigma * gamma_gprotein * g_chi * saturation


def modulate_K_d_multiband(
    K_d_base: float,
    osc_amplitudes: Dict[str, float],
    band_coefficients: Dict[str, float],
) -> float:
    """
    Multi-band oscillatory modulation of receptor binding affinity.

    K_d(t) = K_d_base * (1 - Σ α_k * φ_k(t))

    Generalizes single-band theta modulation to allow any combination
    of oscillation bands to influence K_d.

    From PDF Appendix H.3.

    Parameters
    ----------
    K_d_base : float
        Baseline dissociation constant
    osc_amplitudes : dict
        Map of band name -> amplitude φ_k(t) ∈ [0, 1]
    band_coefficients : dict
        Map of band name -> modulation coefficient α_k.
        Positive coefficients decrease K_d (increase affinity) when
        the band is active.

    Returns
    -------
    float
        Effective K_d (clamped to >= 0.01)
    """
    total_mod = sum(
        band_coefficients.get(b, 0.0) * osc_amplitudes.get(b, 0.0)
        for b in osc_amplitudes
    )
    return max(0.01, K_d_base * (1.0 - total_mod))


def modulate_noise_multiband(
    sigma_base: float,
    osc_amplitudes: Dict[str, float],
    suppression_coefficients: Dict[str, float],
    amplification_coefficients: Dict[str, float],
    floor: float = 0.1,
) -> float:
    """
    Multi-band oscillatory modulation of noise amplitude.

    sigma_mod = sigma_base * max(floor, 1 - Σ s_k * φ_k + Σ a_k * φ_k)

    Suppression channels (e.g., alpha) reduce noise; amplification
    channels (e.g., gamma) increase it.

    From PDF Appendix H.6.

    Parameters
    ----------
    sigma_base : float
        Baseline noise/volatility coefficient
    osc_amplitudes : dict
        Map of band name -> amplitude φ_k(t) ∈ [0, 1]
    suppression_coefficients : dict
        Map of band name -> suppression coefficient s_k (positive values reduce noise)
    amplification_coefficients : dict
        Map of band name -> amplification coefficient a_k (positive values increase noise)
    floor : float, default=0.1
        Minimum scaling factor (prevents zero noise)

    Returns
    -------
    float
        Modulated sigma (non-negative)
    """
    total_suppression = sum(
        suppression_coefficients.get(b, 0.0) * osc_amplitudes.get(b, 0.0)
        for b in osc_amplitudes
    )
    total_amplification = sum(
        amplification_coefficients.get(b, 0.0) * osc_amplitudes.get(b, 0.0)
        for b in osc_amplitudes
    )
    scale = max(floor, 1.0 - total_suppression + total_amplification)
    return sigma_base * scale
