from __future__ import annotations

from typing import Dict, Any, Optional

from zados.reward.base.types import RewardDomainResult, RewardMetaDirective, RewardSubscore


# ---------------------------------------------------------------------------
# Default gain parameters (conservative for stability)
# ---------------------------------------------------------------------------

DEFAULT_FEEDBACK_GAINS: Dict[str, float] = {
    "baseline_gain": 0.05,        # max ±5% baseline shift per cycle
    "baseline_center": 0.5,       # feedback is zero at this score
    "reuptake_gain": 0.3,         # up to ±30% reuptake modification
    "affinity_gain": 0.2,         # up to ±20% affinity modification
}


# ---------------------------------------------------------------------------
# Signal extraction from domain subscores
# ---------------------------------------------------------------------------

def extract_contradiction_load(
    logic_result: Optional[RewardDomainResult],
) -> float:
    """
    Extract contradiction load signal from logic domain subscores.

    ContradictionLoad = 1.0 - internal_consistency_score

    High contradiction load (near 1.0) means logic is detecting many
    internal contradictions, triggering NE reuptake modulation.

    Parameters
    ----------
    logic_result : RewardDomainResult or None
        Logic domain evaluation result.

    Returns
    -------
    float
        Contradiction load in [0.0, 1.0].  Returns 0.0 if absent.
    """
    if logic_result is None:
        return 0.0

    subscore = logic_result.subscores.get("internal_consistency")
    if subscore is None:
        return 0.0

    # Handle both RewardSubscore objects and dicts
    if isinstance(subscore, RewardSubscore):
        score = subscore.score
    elif isinstance(subscore, dict):
        score = subscore.get("score", 1.0)
    else:
        return 0.0

    return max(0.0, min(1.0, 1.0 - score))


def extract_timeline_mismatch(
    ethics_result: Optional[RewardDomainResult],
) -> float:
    """
    Extract timeline mismatch signal from ethics domain subscores.

    TimelineMismatch = 1.0 - timeline_reflection_score

    High timeline mismatch (near 1.0) means short-term vs long-term
    ethical reasoning is misaligned, triggering GABA-B affinity modulation.

    Parameters
    ----------
    ethics_result : RewardDomainResult or None
        Ethics domain evaluation result.

    Returns
    -------
    float
        Timeline mismatch in [0.0, 1.0].  Returns 0.0 if absent.
    """
    if ethics_result is None:
        return 0.0

    subscore = ethics_result.subscores.get("timeline_reflection")
    if subscore is None:
        return 0.0

    if isinstance(subscore, RewardSubscore):
        score = subscore.score
    elif isinstance(subscore, dict):
        score = subscore.get("score", 1.0)
    else:
        return 0.0

    return max(0.0, min(1.0, 1.0 - score))


# ---------------------------------------------------------------------------
# Elementary feedback computations
# ---------------------------------------------------------------------------

def compute_baseline_feedback(
    weighted_score: float,
    gain: float = 0.05,
    center: float = 0.5,
) -> float:
    """
    Compute a baseline concentration delta from a weighted domain score.

    delta = (weighted_score - center) * gain

    Positive delta when score > center (increase baseline),
    negative delta when score < center (decrease baseline).

    Parameters
    ----------
    weighted_score : float
        Weighted domain score (w_d * R_d).
    gain : float, default=0.05
        Maximum magnitude of the delta.
    center : float, default=0.5
        Neutral point (no feedback when score equals center).

    Returns
    -------
    float
        Baseline delta, clamped to [-gain, +gain].
    """
    # The *2.0 factor rescales the (score - center) range [−0.5, +0.5]
    # to [−gain, +gain], making gain a direct cap on the output magnitude.
    # This is a direct linear mapping, unlike the regulatory_modulator.py
    # pathways which use leaky integrators for τ-smoothed temporal filtering.
    delta = (weighted_score - center) * gain * 2.0
    return max(-gain, min(gain, delta))


def compute_reuptake_feedback(
    weighted_score: float,
    load: float,
    gain: float = 0.3,
) -> float:
    """
    Compute a multiplicative reuptake modifier.

    multiplier = 1.0 + weighted_score * load * gain

    Higher score and higher load → faster reuptake (multiplier > 1).
    Zero load → multiplier stays at 1.0 (no effect).

    Parameters
    ----------
    weighted_score : float
        Weighted domain score (w_d * R_d).
    load : float
        Secondary signal (e.g., ContradictionLoad) in [0, 1].
    gain : float, default=0.3
        Maximum deviation from 1.0.

    Returns
    -------
    float
        Reuptake multiplier, clamped to [1 - gain, 1 + gain].
    """
    multiplier = 1.0 + weighted_score * load * gain
    return max(1.0 - gain, min(1.0 + gain, multiplier))


def compute_affinity_feedback(
    weighted_score: float,
    mismatch: float,
    gain: float = 0.2,
) -> float:
    """
    Compute a multiplicative K_d (affinity) modifier.

    multiplier = 1.0 - weighted_score * mismatch * gain

    Higher score and higher mismatch → lower K_d → higher affinity
    (multiplier < 1).  Zero mismatch → multiplier stays at 1.0.

    Parameters
    ----------
    weighted_score : float
        Weighted domain score (w_d * R_d).
    mismatch : float
        Secondary signal (e.g., TimelineMismatch) in [0, 1].
    gain : float, default=0.2
        Maximum deviation from 1.0.

    Returns
    -------
    float
        K_d multiplier, clamped to [1 - gain, 1 + gain].
    """
    multiplier = 1.0 - weighted_score * mismatch * gain
    return max(1.0 - gain, min(1.0 + gain, multiplier))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_reward_feedback(
    meta_directive: RewardMetaDirective,
    domain_results: Dict[str, RewardDomainResult],
    gains: Optional[Dict[str, float]] = None,
    sleep_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute neurochemical baseline feedback from synthesis output.

    Implements the masterdoc's reward-conditioned secondary gradients:

    - OXT baseline  ←  R_Attunement   (social congruence)
    - CB1 baseline  ←  R_Innovation   (symbolic identity)
    - NE  reuptake  ←  R_Logic × ContradictionLoad
    - GABA-B K_d    ←  R_Ethics × TimelineMismatch

    Parameters
    ----------
    meta_directive : RewardMetaDirective
        Output of SynthesisEngine.synthesize().
    domain_results : dict
        Map of domain_name → RewardDomainResult (same input given to
        the synthesis engine).
    gains : dict, optional
        Override default gain parameters.

    Returns
    -------
    dict
        Feedback parameters structured as::

            {
                "neurotransmitters": {
                    "OXT": {"C_baseline_delta": float},
                    "CB1": {"C_baseline_delta": float},
                    "NE":  {"u_base_multiplier": float},
                },
                "receptors": {
                    "GABA_B": {"K_d_multiplier": float},
                },
            }
    """
    g = dict(DEFAULT_FEEDBACK_GAINS)
    if gains:
        g.update(gains)

    # Extract per-domain weighted scores from synthesis meta
    per_domain = meta_directive.meta.get("per_domain_weighted_scores", {})
    R_attunement = per_domain.get("human_attunement", 0.0)
    R_innovation = per_domain.get("innovation", 0.0)
    R_logic = per_domain.get("logic", 0.0)
    R_ethics = per_domain.get("ethics", 0.0)

    # Extract secondary signals from domain subscores
    contradiction_load = extract_contradiction_load(
        domain_results.get("logic"),
    )
    timeline_mismatch = extract_timeline_mismatch(
        domain_results.get("ethics"),
    )

    # Compute feedback for each pathway
    oxt_delta = compute_baseline_feedback(
        R_attunement, gain=g["baseline_gain"], center=g["baseline_center"],
    )
    cb1_delta = compute_baseline_feedback(
        R_innovation, gain=g["baseline_gain"], center=g["baseline_center"],
    )
    ne_multiplier = compute_reuptake_feedback(
        R_logic, contradiction_load, gain=g["reuptake_gain"],
    )
    gaba_b_multiplier = compute_affinity_feedback(
        R_ethics, timeline_mismatch, gain=g["affinity_gain"],
    )

    # Consolidation gate: during deep memory consolidation, reduce feedback
    # strength to avoid disrupting replay dynamics.
    sm = sleep_metrics or {}
    consolidation = sm.get("consolidation_depth", 0.0)
    if consolidation > 0.5:
        gate = 1.0 - consolidation  # e.g. consolidation=0.8 → gate=0.2
        oxt_delta *= gate
        cb1_delta *= gate
        # Pull reuptake/affinity multipliers toward neutral (1.0)
        ne_multiplier = 1.0 + (ne_multiplier - 1.0) * gate
        gaba_b_multiplier = 1.0 + (gaba_b_multiplier - 1.0) * gate

    return {
        "neurotransmitters": {
            "OXT": {"C_baseline_delta": oxt_delta},
            "CB1": {"C_baseline_delta": cb1_delta},
            "NE": {"u_base_multiplier": ne_multiplier},
        },
        "receptors": {
            "GABA_B": {"K_d_multiplier": gaba_b_multiplier},
        },
    }
