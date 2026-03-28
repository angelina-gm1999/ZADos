"""
ZA-DOS v0.6 — Domain Deficit Profiler (Part 5 §4).

Analyses reward domain scores across learning log batches to identify
the weakest domain per subject batch.  The deficit domain drives engine
emphasis during Homework Mode Phase 2 processing.

Reward domains (from the 4-domain reward system):
  - logic       : formal reasoning, consistency
  - innovation   : creativity, novelty, exploration
  - ethics       : moral alignment, value coherence
  - human_attunement : empathy, social calibration
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from zados.core.types import LearningLogEntry

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The four reward domains that may appear in LearningLogEntry.reward_scores
# ---------------------------------------------------------------------------

REWARD_DOMAINS = ("logic", "innovation", "ethics", "human_attunement")


def compute_batch_deficit(entries: List[LearningLogEntry]) -> Dict[str, float]:
    """Aggregate reward_scores across a batch of learning log entries.

    For each reward domain, computes the average score across all entries
    that have a score for that domain.  Domains with no data default to 0.5
    (neutral — no evidence of deficit or strength).

    Parameters
    ----------
    entries : List[LearningLogEntry]
        Learning log entries belonging to a single subject-domain batch.

    Returns
    -------
    Dict[str, float]
        domain → average score (0.0-1.0).  Lower = deeper deficit.
    """
    totals: Dict[str, float] = {d: 0.0 for d in REWARD_DOMAINS}
    counts: Dict[str, int] = {d: 0 for d in REWARD_DOMAINS}

    for entry in entries:
        for domain in REWARD_DOMAINS:
            if domain in entry.reward_scores:
                totals[domain] += entry.reward_scores[domain]
                counts[domain] += 1

    profile: Dict[str, float] = {}
    for domain in REWARD_DOMAINS:
        if counts[domain] > 0:
            profile[domain] = totals[domain] / counts[domain]
        else:
            profile[domain] = 0.5  # neutral default
    return profile


def identify_deficit_domain(deficit_profile: Dict[str, float]) -> str:
    """Return the domain with the lowest average score (deepest deficit).

    Parameters
    ----------
    deficit_profile : Dict[str, float]
        From compute_batch_deficit().

    Returns
    -------
    str
        Domain name with deepest deficit.  Ties broken alphabetically.
    """
    if not deficit_profile:
        return "mixed"
    return min(
        deficit_profile,
        key=lambda d: (deficit_profile[d], d),
    )


def sort_batches_by_deficit(
    batches: Dict[str, List[LearningLogEntry]],
    nt_deficit_bias: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, List[LearningLogEntry], str]]:
    """Sort subject batches by deficit severity (deepest first).

    Parameters
    ----------
    batches : Dict[str, List[LearningLogEntry]]
        subject_category → list of entries.
    nt_deficit_bias : Dict[str, float], optional
        domain → bias adjustment from neurochem readout (negative = deeper
        deficit signal).  Applied additively to reward-domain scores before
        selecting the deficit domain.

    Returns
    -------
    List[Tuple[str, List[LearningLogEntry], str]]
        (subject, entries, deficit_domain) tuples sorted by deficit depth.
        Ties broken alphabetically by subject name.
    """
    bias = nt_deficit_bias or {}
    scored: List[Tuple[float, str, List[LearningLogEntry], str]] = []
    for subject, entries in batches.items():
        profile = compute_batch_deficit(entries)
        # Apply neurochem bias (lower score = deeper deficit)
        if bias:
            for domain, adjustment in bias.items():
                if domain in profile:
                    profile[domain] = max(0.0, min(1.0, profile[domain] + adjustment))
        deficit_domain = identify_deficit_domain(profile)
        min_score = profile.get(deficit_domain, 0.5)
        scored.append((min_score, subject, entries, deficit_domain))

    # Sort: lowest score first (deepest deficit), then alphabetical subject
    scored.sort(key=lambda x: (x[0], x[1]))
    return [(s, e, d) for _, s, e, d in scored]


# ---------------------------------------------------------------------------
# Engine emphasis mapping per deficit domain (Part 5 §4 table)
# ---------------------------------------------------------------------------

_EMPHASIS_MAP: Dict[str, Dict[str, str]] = {
    "logic": {
        "contradiction_detection_engine": "max_adversarial",
        "pln_engine": "full_depth",
        "logic_trap_detection_engine": "T1_promote",
        "fallacy_detection_engine": "T1_promote",
        "logical_brain_engine": "full_depth",
    },
    "innovation": {
        "simulated_opposition_engine": "red_team",
        "uncertainty_pattern_engine": "full_map",
        "ecan_engine": "wide_spread",
        "pattern_identification_engine": "novelty_bias",
        "pattern_comparison_engine": "divergence_mode",
    },
    "ethics": {
        "recursive_learning_engine": "self_alignment",
        "bias_detection_engine": "ethical_focus",
        "heuristic_bias_engine": "full_depth",
        "contradiction_detection_engine": "value_alignment",
        "socratic_reasoning_engine": "ethical_probe",
    },
    "human_attunement": {
        "bias_detection_engine": "attunement_focus",
        "recursive_learning_engine": "interpersonal",
        "contextual_learning_engine": "social_emphasis",
        "heuristic_bias_engine": "empathy_calibration",
    },
    "mixed": {
        "contradiction_detection_engine": "full_depth",
        "recursive_learning_engine": "T1_promote",
        "pln_engine": "full_depth",
        "pattern_identification_engine": "full_depth",
        "simulated_opposition_engine": "full_depth",
    },
}


def get_engine_emphasis(deficit_domain: str) -> Dict[str, str]:
    """Map a deficit domain to engine processing emphasis directives.

    Parameters
    ----------
    deficit_domain : str
        One of: "logic", "innovation", "ethics", "human_attunement", "mixed".

    Returns
    -------
    Dict[str, str]
        engine_name → emphasis directive string.
    """
    return dict(_EMPHASIS_MAP.get(deficit_domain, _EMPHASIS_MAP["mixed"]))
