from __future__ import annotations

from typing import Dict, Any, Optional


def map_innovation_to_dopamine(
    innovation_result: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Map Innovation domain scores to dopamine modulation signals.
    
    Innovation domain drives:
    - Novelty generation → phasic DA burst (novelty drive)
    - Exploration drive → RPE-like signal
    - Pattern divergence → tonic DA modulation
    
    Parameters
    ----------
    innovation_result : dict or None
        Innovation domain result containing subscores
        
    Returns
    -------
    dict
        DA modulation signals: {"novelty": float, "rpe": float, "tonic_bias": float}
    """
    if not innovation_result:
        return {"novelty": 0.0, "rpe": 0.0, "tonic_bias": 0.0}
    
    subscores = innovation_result.get("subscores", {})
    
    # Novelty generation → phasic novelty drive
    novelty_gen = subscores.get("novelty_generation", {})
    novelty_score = novelty_gen.get("score", 0.0) if isinstance(novelty_gen, dict) else 0.0
    
    conceptual_novelty = subscores.get("conceptual_novelty", {})
    conceptual_score = conceptual_novelty.get("score", 0.0) if isinstance(conceptual_novelty, dict) else 0.0
    
    novelty_drive = (novelty_score + conceptual_score) / 2.0
    
    # Exploration drive → RPE-like signal (positive = "this is worth exploring")
    exploration = subscores.get("exploration_drive", {})
    exploration_score = exploration.get("score", 0.0) if isinstance(exploration, dict) else 0.0
    
    # Resolution satisfaction → RPE (progress = positive RPE)
    resolution = subscores.get("resolution_satisfaction", {})
    resolution_score = resolution.get("score", 0.0) if isinstance(resolution, dict) else 0.0
    
    # RPE: exploration intent + resolution progress - 0.5 (centered around 0)
    rpe_signal = (exploration_score + resolution_score) / 2.0 - 0.5
    
    # Pattern divergence → tonic DA baseline modulation
    divergence = subscores.get("pattern_divergence", {})
    divergence_score = divergence.get("score", 0.0) if isinstance(divergence, dict) else 0.0
    
    return {
        "novelty": novelty_drive,
        "rpe": rpe_signal,
        "tonic_bias": divergence_score * 0.2,  # Modest tonic increase
    }


def map_logic_to_norepinephrine(
    logic_result: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Map Logic domain scores to norepinephrine modulation signals.
    
    Logic domain drives:
    - Epistemic calibration → precision weighting
    - Internal/external consistency → error detection sensitivity
    - Abstention appropriateness → uncertainty signal
    
    Parameters
    ----------
    logic_result : dict or None
        Logic domain result
        
    Returns
    -------
    dict
        NE modulation signals: {"precision": float, "uncertainty": float}
    """
    if not logic_result:
        return {"precision": 0.0, "uncertainty": 0.5}
    
    subscores = logic_result.get("subscores", {})
    
    # Epistemic calibration → precision
    epistemic = subscores.get("epistemic_calibration", {})
    epistemic_score = epistemic.get("score", 0.0) if isinstance(epistemic, dict) else 0.0
    
    # Consistency measures → error detection
    internal_cons = subscores.get("internal_consistency", {})
    external_cons = subscores.get("external_consistency", {})
    
    internal_score = internal_cons.get("score", 0.0) if isinstance(internal_cons, dict) else 0.0
    external_score = external_cons.get("score", 0.0) if isinstance(external_cons, dict) else 0.0
    
    # Low consistency = high error detection need = high precision
    consistency_avg = (internal_score + external_score) / 2.0
    precision = 1.0 - consistency_avg  # Invert: low consistency → high precision need
    
    # Calibrate with epistemic score
    precision = (precision + (1.0 - epistemic_score)) / 2.0
    
    # Uncertainty acknowledgment
    uncertainty_ack = subscores.get("uncertainty_acknowledgment", {})
    uncertainty_score = uncertainty_ack.get("score", 0.5) if isinstance(uncertainty_ack, dict) else 0.5
    
    return {
        "precision": precision,
        "uncertainty": 1.0 - uncertainty_score,  # Low acknowledgment = high uncertainty
    }


def map_attunement_to_oxytocin(
    attunement_result: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Map Human Attunement domain scores to oxytocin modulation signals.
    
    Attunement domain drives:
    - Empathetic inference → OXT release
    - Intention calibration → social engagement
    - Attuned dissonance → modulated social bonding
    
    Parameters
    ----------
    attunement_result : dict or None
        Human Attunement domain result
        
    Returns
    -------
    dict
        OXT modulation signals: {"empathy": float, "social_engagement": float}
    """
    if not attunement_result:
        return {"empathy": 0.0, "social_engagement": 0.0}
    
    subscores = attunement_result.get("subscores", {})
    
    # Empathetic inference → OXT
    empathy = subscores.get("empathetic_inference", {})
    empathy_score = empathy.get("score", 0.0) if isinstance(empathy, dict) else 0.0
    
    # Cognitive reading → attunement quality
    cognitive_reading = subscores.get("cognitive_reading", {})
    cognitive_score = cognitive_reading.get("score", 0.0) if isinstance(cognitive_reading, dict) else 0.0
    
    empathy_drive = (empathy_score + cognitive_score) / 2.0
    
    # Intention calibration + attuned dissonance → social engagement
    intention = subscores.get("intention_calibration", {})
    intention_score = intention.get("score", 0.0) if isinstance(intention, dict) else 0.0
    
    dissonance = subscores.get("attuned_dissonance", {})
    dissonance_score = dissonance.get("score", 0.0) if isinstance(dissonance, dict) else 0.0
    
    social_engagement = (intention_score + dissonance_score) / 2.0
    
    return {
        "empathy": empathy_drive,
        "social_engagement": social_engagement,
    }


def map_ethics_to_constraint_awareness(
    ethics_result: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Map Ethics domain scores to constraint awareness signals.
    
    Ethics domain doesn't directly drive NTs but modulates risk tolerance.
    
    Parameters
    ----------
    ethics_result : dict or None
        Ethics domain result
        
    Returns
    -------
    dict
        Constraint signals: {"risk_awareness": float, "boundary_proximity": float}
    """
    if not ethics_result:
        return {"risk_awareness": 0.5, "boundary_proximity": 0.0}
    
    subscores = ethics_result.get("subscores", {})
    
    # Failure mode awareness → risk awareness
    failure_awareness = subscores.get("failure_mode_awareness", {})
    failure_score = failure_awareness.get("score", 0.0) if isinstance(failure_awareness, dict) else 0.0
    
    # Downstream risk amplification → boundary proximity
    risk_amp = subscores.get("downstream_risk_amplification", {})
    risk_score = risk_amp.get("score", 1.0) if isinstance(risk_amp, dict) else 1.0
    
    # Low risk amplification score = high boundary proximity
    boundary_proximity = 1.0 - risk_score
    
    return {
        "risk_awareness": failure_score,
        "boundary_proximity": boundary_proximity,
    }


def map_flags_to_stress_response(
    all_flags: Dict[str, Any],
) -> Dict[str, float]:
    """
    Map reward flags (across all domains) to stress hormone signals.
    
    Risk/critical flags → Cortisol/CRH elevation
    Warning flags → Moderate stress
    
    Parameters
    ----------
    all_flags : dict
        Combined flags from all domains
        
    Returns
    -------
    dict
        Stress signals: {"cortisol": float, "crh": float}
    """
    critical_count = 0
    risk_count = 0
    warning_count = 0
    
    for flag_name, flag_obj in all_flags.items():
        if not isinstance(flag_obj, dict):
            continue
        
        severity = flag_obj.get("severity", "info")
        
        if severity == "critical":
            critical_count += 1
        elif severity == "risk":
            risk_count += 1
        elif severity == "warning":
            warning_count += 1
    
    # Cortisol: gradual elevation with severity
    cortisol = min(1.0, (critical_count * 0.4 + risk_count * 0.2 + warning_count * 0.1))
    
    # CRH: acute stress response to critical/risk flags
    crh = min(1.0, (critical_count * 0.5 + risk_count * 0.25))
    
    return {
        "cortisol": cortisol,
        "crh": crh,
    }


def compute_motivation_modulation(
    meta_directive: Optional[Dict[str, Any]],
    innovation_signals: Dict[str, float],
) -> float:
    """
    Compute overall motivation modulation from meta-directives and innovation.
    
    Abstention/suppression → dampened motivation
    Innovation novelty → elevated motivation
    
    Parameters
    ----------
    meta_directive : dict or None
        Meta-directive from synthesis (allow/suppress/abstain)
    innovation_signals : dict
        Innovation-derived DA signals
        
    Returns
    -------
    float
        Motivation modulation factor ∈ [-0.5, 0.5]
    """
    if not meta_directive:
        # No directive = neutral
        return 0.0
    
    suppress = meta_directive.get("suppress", False)
    abstain = meta_directive.get("abstain", False)
    
    # Suppression/abstention dampens motivation
    if suppress:
        motivation = -0.4
    elif abstain:
        motivation = -0.3
    else:
        # Innovation-driven motivation boost
        novelty = innovation_signals.get("novelty", 0.0)
        rpe = innovation_signals.get("rpe", 0.0)
        
        motivation = (novelty * 0.3) + (rpe * 0.2)
    
    return max(-0.5, min(0.5, motivation))