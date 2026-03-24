from __future__ import annotations

from typing import Optional
import math


def compute_novelty_drive(
    stimulus_novelty: float,
    sensitivity: float = 1.0,
    threshold: float = 0.3,
) -> float:
    """
    Compute release drive from novelty detection.
    
    N(t) = sensitivity · max(0, stimulus_novelty - threshold)
    
    Novelty above threshold triggers phasic release.
    
    Parameters
    ----------
    stimulus_novelty : float
        Novelty signal ∈ [0, 1] from perception/attention systems
    sensitivity : float, default=1.0
        Novelty sensitivity scaling
    threshold : float, default=0.3
        Minimum novelty required to trigger release
        
    Returns
    -------
    float
        Novelty drive (non-negative)
    """
    return sensitivity * max(0.0, stimulus_novelty - threshold)


def compute_rpe_drive(
    reward_prediction_error: float,
    gain: float = 1.0,
) -> float:
    """
    Compute release drive from reward prediction error.
    
    RPE(t) = gain · reward_prediction_error
    
    Positive RPE (better than expected) → increased release
    Negative RPE (worse than expected) → decreased release (can be negative)
    
    Parameters
    ----------
    reward_prediction_error : float
        RPE signal: actual - predicted reward
    gain : float, default=1.0
        RPE gain scaling
        
    Returns
    -------
    float
        RPE drive (can be positive or negative)
    """
    return gain * reward_prediction_error


def compute_effort_drive(
    task_demand: float,
    willingness: float = 1.0,
    threshold: float = 0.2,
) -> float:
    """
    Compute release drive from effort/task demands.
    
    E(t) = willingness · max(0, task_demand - threshold)
    
    High task demands can trigger compensatory release.
    
    Parameters
    ----------
    task_demand : float
        Task difficulty/complexity ∈ [0, 1]
    willingness : float, default=1.0
        Effort willingness (modulated by motivation state)
    threshold : float, default=0.2
        Minimum demand required to trigger drive
        
    Returns
    -------
    float
        Effort drive (non-negative)
    """
    return willingness * max(0.0, task_demand - threshold)


def compute_combined_release_drive(
    novelty_drive: float,
    rpe_drive: float,
    effort_drive: float = 0.0,
    baseline_release: float = 0.0,
) -> float:
    """
    Combine multiple release drives into total phasic release signal.
    
    R_total(t) = baseline + novelty + RPE + effort
    
    Parameters
    ----------
    novelty_drive : float
        Novelty-driven release
    rpe_drive : float
        RPE-driven release (can be negative)
    effort_drive : float, default=0.0
        Effort-driven release
    baseline_release : float, default=0.0
        Tonic baseline release rate
        
    Returns
    -------
    float
        Total release drive (can be negative if RPE is strongly negative)
    """
    return baseline_release + novelty_drive + rpe_drive + effort_drive


def apply_fatigue_gating(
    release_drive: float,
    fatigue: float,
    fatigue_threshold: float = 0.7,
    suppression_factor: float = 0.5,
) -> float:
    """
    Gate release drive by fatigue state.
    
    High fatigue suppresses release capacity.
    
    R_gated(t) = R(t) · (1 - suppression · max(0, F - threshold))
    
    Parameters
    ----------
    release_drive : float
        Base release drive
    fatigue : float
        Current fatigue level ∈ [0, 1]
    fatigue_threshold : float, default=0.7
        Fatigue level at which suppression begins
    suppression_factor : float, default=0.5
        Maximum suppression (0=none, 1=complete)
        
    Returns
    -------
    float
        Fatigue-gated release drive
    """
    if fatigue_threshold >= 1.0:
        # Threshold at or above maximum — fatigue never suppresses
        return release_drive
    if fatigue <= fatigue_threshold:
        return release_drive

    excess_fatigue = fatigue - fatigue_threshold
    suppression = suppression_factor * excess_fatigue / (1.0 - fatigue_threshold)
    gating = 1.0 - suppression

    return release_drive * max(0.0, gating)


def apply_oscillatory_gating(
    release_drive: float,
    oscillation_amplitude: float,
    band_preference: float = 1.0,
) -> float:
    """
    Modulate release drive by oscillatory state.
    
    R_osc(t) = R(t) · (1 + band_preference · φ_k(t))
    
    Parameters
    ----------
    release_drive : float
        Base release drive
    oscillation_amplitude : float
        Relevant oscillation band amplitude ∈ [0, 1]
    band_preference : float, default=1.0
        How strongly this release is coupled to the oscillation
        
    Returns
    -------
    float
        Oscillation-modulated release drive
    """
    modulation = 1.0 + band_preference * oscillation_amplitude
    return release_drive * modulation


def compute_phasic_burst_amplitude(
    release_drive: float,
    receptor_sensitivity: float = 1.0,
    max_burst: float = 1.0,
) -> float:
    """
    Convert release drive to actual phasic burst amplitude.
    
    Applies saturation to prevent unrealistic bursts.
    
    A_burst = max_burst · (1 - exp(-sensitivity · drive))
    
    Parameters
    ----------
    release_drive : float
        Total release drive
    receptor_sensitivity : float, default=1.0
        Sensitivity of release machinery
    max_burst : float, default=1.0
        Maximum burst amplitude
        
    Returns
    -------
    float
        Phasic burst amplitude ∈ [0, max_burst]
    """
    if release_drive <= 0.0:
        return 0.0
    
    # Saturating exponential (prevents unbounded bursts)
    normalized = 1.0 - math.exp(-receptor_sensitivity * release_drive)
    return max_burst * normalized


def compute_adaptive_threshold(
    baseline_threshold: float,
    recent_activity: float,
    adaptation_rate: float = 0.1,
) -> float:
    """
    Compute adaptive novelty/RPE threshold based on recent activity.
    
    θ_adaptive = θ_baseline + adaptation_rate · recent_activity
    
    High recent activity → higher threshold (habituation).
    
    Parameters
    ----------
    baseline_threshold : float
        Base threshold value
    recent_activity : float
        Exponential moving average of recent drive
    adaptation_rate : float, default=0.1
        How quickly threshold adapts
        
    Returns
    -------
    float
        Adapted threshold (non-negative)
    """
    return baseline_threshold + adaptation_rate * recent_activity


def update_recent_activity_trace(
    current_trace: float,
    current_drive: float,
    dt: float,
    tau: float = 10.0,
) -> float:
    """
    Update exponential moving average of recent release activity.
    
    E(t+dt) = E(t) · exp(-dt/tau) + drive(t) · dt
    
    Used for adaptive thresholding and habituation.
    
    Parameters
    ----------
    current_trace : float
        Current activity trace
    current_drive : float
        Current release drive
    dt : float
        Time step
    tau : float, default=10.0
        Decay time constant
        
    Returns
    -------
    float
        Updated activity trace
    """
    decay_factor = math.exp(-dt / tau)
    return current_trace * decay_factor + current_drive * dt