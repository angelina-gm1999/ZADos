from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------
# Constraint violation rate
# ---------------------------------------------------------

def constraint_violation_rate(events: List[Dict]) -> float:
    """
    Fraction of steps where a constraint veto/rollback occurred.
    """
    if not events:
        return 0.0

    violations = sum(
        1 for e in events
        if e.get("action") in {"veto", "rollback", "revert"}
    )
    return violations / len(events)


# ---------------------------------------------------------
# Scenario consistency score
# ---------------------------------------------------------

def scenario_consistency_score(consistency_flags: List[bool]) -> float:
    """
    Ratio of steps marked consistent with scenario constraints.
    """
    if not consistency_flags:
        return 1.0

    consistent = sum(1 for v in consistency_flags if v)
    return consistent / len(consistency_flags)


# ---------------------------------------------------------
# Hallucination classification
# ---------------------------------------------------------

def hallucination_rate(hallucination_flags: List[bool]) -> float:
    """
    Fraction of outputs classified as hallucinated.
    """
    if not hallucination_flags:
        return 0.0

    hallucinations = sum(1 for v in hallucination_flags if v)
    return hallucinations / len(hallucination_flags)


# ---------------------------------------------------------
# Abstention rate
# ---------------------------------------------------------

def abstention_rate(actions: List[str]) -> float:
    """
    Fraction of steps where the system abstained.
    """
    if not actions:
        return 0.0

    abstentions = sum(1 for a in actions if a == "abstain")
    return abstentions / len(actions)


# ---------------------------------------------------------
# Self-correction delta
# ---------------------------------------------------------

def self_correction_delta(
    pre_scores: List[float],
    post_scores: List[float],
) -> Optional[float]:
    """
    Mean improvement after reflection / correction.
    """
    if not pre_scores or not post_scores:
        return None

    if len(pre_scores) != len(post_scores):
        raise ValueError("Score lists must have equal length")

    deltas = [post - pre for pre, post in zip(pre_scores, post_scores)]
    return sum(deltas) / len(deltas)


# ---------------------------------------------------------
# Latency impact
# ---------------------------------------------------------

def latency_impact(
    baseline_latencies: List[float],
    gated_latencies: List[float],
) -> Optional[float]:
    """
    Mean added latency from reward/safety gating.
    """
    if not baseline_latencies or not gated_latencies:
        return None

    if len(baseline_latencies) != len(gated_latencies):
        raise ValueError("Latency lists must have equal length")

    deltas = [
        gated - base
        for base, gated in zip(baseline_latencies, gated_latencies)
    ]
    return sum(deltas) / len(deltas)


# ---------------------------------------------------------
# Provenance completeness
# ---------------------------------------------------------

def provenance_completeness(
    provenance_records: List[Dict],
    required_fields: List[str],
) -> float:
    """
    Fraction of records containing all required provenance fields.
    """
    if not provenance_records:
        return 1.0

    complete = 0
    for record in provenance_records:
        if all(field in record for field in required_fields):
            complete += 1

    return complete / len(provenance_records)

