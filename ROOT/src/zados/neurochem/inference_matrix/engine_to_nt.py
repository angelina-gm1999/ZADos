"""
Cognitive engine evaluation results → NT modulation signals.

Maps evaluation outcomes back into the neurochemical system as
modulation signals, closing the bidirectional loop.

Evaluation outcomes include:
    - Domain scores (quality of reasoning in each domain)
    - Confidence levels (certainty of evaluation)
    - Error/contradiction detection (found issues)
    - Social resonance (attunement quality)
    - Risk assessment (safety evaluation)
"""

from __future__ import annotations

from typing import Dict, Any, Optional


def compute_nt_modulation_from_evaluation(
    evaluation_results: Dict[str, Any],
    current_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute NT modulation signals from cognitive engine evaluation results.

    This closes the NT→engine→NT bidirectional loop. Evaluation quality
    feeds back into the neurochemical state to adjust future processing.

    Parameters
    ----------
    evaluation_results : dict
        Results from cognitive engine evaluation. Expected keys:
            domain_scores: {domain_name: float} — domain-level quality
            confidence: float — overall evaluation confidence
            contradictions_found: int — number of contradictions detected
            social_resonance: float — attunement quality [0, 1]
            risk_detected: float — risk level [0, 1]
            novelty_detected: float — novel elements found [0, 1]
    current_metrics : dict, optional
        Current neurosymbolic metrics for adaptive feedback

    Returns
    -------
    dict
        {nt_name: {signal_key: value}} modulation signals for the engine
    """
    signals: Dict[str, Dict[str, float]] = {}

    # Extract evaluation features
    confidence = evaluation_results.get("confidence", 0.5)
    contradictions = evaluation_results.get("contradictions_found", 0)
    social_resonance = evaluation_results.get("social_resonance", 0.0)
    risk_detected = evaluation_results.get("risk_detected", 0.0)
    novelty_detected = evaluation_results.get("novelty_detected", 0.0)
    domain_scores = evaluation_results.get("domain_scores", {})

    # ── DA modulation: novelty and reward prediction ────────────────
    # High novelty → DA burst; confidence acts as RPE proxy
    da_novelty = novelty_detected * 0.8
    da_rpe = (confidence - 0.5) * 0.6  # Centered around 0
    signals["DA"] = {
        "novelty": da_novelty,
        "rpe": da_rpe,
        "emotion_drive": _compute_overall_quality(domain_scores) * 0.3,
    }

    # ── NE modulation: precision and contradiction ──────────────────
    # Contradictions drive NE precision; low confidence → uncertainty
    ne_contradiction = min(1.0, contradictions * 0.3)
    ne_precision = ne_contradiction * 0.5 + (1.0 - confidence) * 0.5
    ne_uncertainty = 1.0 - confidence
    signals["NE"] = {
        "precision": ne_precision,
        "uncertainty": ne_uncertainty,
        "contradiction": ne_contradiction,
    }

    # ── OXT modulation: social resonance ────────────────────────────
    signals["OXT"] = {
        "empathy": social_resonance * 0.7,
        "social_engagement": social_resonance * 0.6,
    }

    # ── 5HT modulation: evaluation quality → mood stability ─────────
    overall_quality = _compute_overall_quality(domain_scores)
    signals["5HT"] = {
        "mood_stability": overall_quality * 0.5,
        "emotion_drive": overall_quality * 0.2,
    }

    # ── GABA modulation: risk → inhibition ──────────────────────────
    signals["GABA"] = {
        "inhibition": risk_detected * 0.6,
        "boundary_proximity": risk_detected * 0.4,
    }

    # ── Cortisol/CRH modulation: risk and contradictions ────────────
    stress_signal = (risk_detected * 0.6 + min(1.0, contradictions * 0.2) * 0.4)
    signals["cortisol"] = {
        "stress_level": stress_signal * 0.5,
    }
    signals["CRH"] = {
        "acute_stress": risk_detected * 0.4,
    }

    # ── ACh modulation: confidence → attention demand ───────────────
    # Low confidence → need more attention
    signals["ACh"] = {
        "attention_demand": (1.0 - confidence) * 0.6,
        "rule_fidelity": _get_domain_score(domain_scores, "logic") * 0.5,
    }

    # ── MOR modulation: social + quality → comfort ──────────────────
    signals["MOR"] = {
        "hedonic_tone": (social_resonance * 0.3 + overall_quality * 0.3),
        "comfort": social_resonance * 0.4,
    }

    # ── CB1 modulation: novelty → flexibility ───────────────────────
    signals["CB1"] = {
        "flexibility": novelty_detected * 0.5,
    }

    # ── GLU modulation: contradictions → integration demand ─────────
    signals["GLU"] = {
        "integration_demand": min(1.0, contradictions * 0.25),
        "excitation": (1.0 - confidence) * 0.3,
    }

    return signals


def _compute_overall_quality(domain_scores: Dict[str, Any]) -> float:
    """Compute average domain quality score."""
    if not domain_scores:
        return 0.5
    scores = []
    for v in domain_scores.values():
        if isinstance(v, (int, float)):
            scores.append(float(v))
        elif isinstance(v, dict):
            scores.append(v.get("score", 0.5))
    if not scores:
        return 0.5
    return sum(scores) / len(scores)


def _get_domain_score(domain_scores: Dict[str, Any], domain: str) -> float:
    """Get score for a specific domain."""
    val = domain_scores.get(domain, 0.5)
    if isinstance(val, dict):
        return val.get("score", 0.5)
    return float(val)
