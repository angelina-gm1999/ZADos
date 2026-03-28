from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional

from zados.reward.base.types import RewardDomainResult
from zados.reward.base.structure import RewardFlag


# ---------------------------------------------------------------------------
# 4-tier influence system (from masterdoc)
# ---------------------------------------------------------------------------

TIER_BOUNDARIES = (0.25, 0.50, 0.75, 1.0)
TIER_LABELS = ("minimal", "moderate", "significant", "dominant")


def classify_tier(score: float) -> int:
    """
    Classify a [0, 1] score into a 4-tier influence level.

    Returns
    -------
    int
        Tier index 0-3:
          0: [0.00, 0.25]  (minimal influence)
          1: (0.25, 0.50]  (moderate influence)
          2: (0.50, 0.75]  (significant influence)
          3: (0.75, 1.00]  (dominant influence)
    """
    if score <= 0.25:
        return 0
    elif score <= 0.50:
        return 1
    elif score <= 0.75:
        return 2
    else:
        return 3


def tier_label(tier: int) -> str:
    """Return human-readable label for a tier index (clamped to 0-3)."""
    return TIER_LABELS[max(0, min(3, tier))]


# ---------------------------------------------------------------------------
# Weighted composite score: R(t) = sum(w*R) / sum(w)
# ---------------------------------------------------------------------------

def compute_weighted_composite(
    domain_results: Dict[str, RewardDomainResult],
    domain_weights: Dict[str, float],
) -> float:
    """
    Compute the global composite reward score.

    R(t) = sum_d  w_d * R_d  /  sum_d  w_d

    Normalised by weight sum so the result stays in [0, 1].
    Returns 0.0 when no domains are present.
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for domain_name, result in domain_results.items():
        w = domain_weights.get(domain_name, 0.0)
        weighted_sum += w * result.general_score
        total_weight += w

    if total_weight == 0.0:
        return 0.0

    return weighted_sum / total_weight


def compute_per_domain_weighted_scores(
    domain_results: Dict[str, RewardDomainResult],
    domain_weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute w_d * R_d for each domain (un-normalised).

    Returns dict of domain_name -> weighted_score.
    """
    return {
        name: domain_weights.get(name, 0.0) * result.general_score
        for name, result in domain_results.items()
    }


# ---------------------------------------------------------------------------
# Suppression / abstention decision logic
# ---------------------------------------------------------------------------

def compute_suppression(
    composite_score: float,
    suppression_bias: float,
    domain_flag_lists: Dict[str, List[Any]],
) -> bool:
    """
    Determine whether output should be suppressed.

    Suppress when:
    1. Composite weighted score falls below *suppression_bias*, OR
    2. Any domain has a ``"critical"`` severity flag.

    Parameters
    ----------
    composite_score : float
        Global weighted composite R(t).
    suppression_bias : float
        Profile suppression bias (0-1).  Higher = suppress at higher scores.
    domain_flag_lists : dict
        domain_name -> list of flag objects (RewardFlag or dict).
    """
    if composite_score < suppression_bias:
        return True

    for _domain_name, flags in domain_flag_lists.items():
        for flag in flags:
            if isinstance(flag, RewardFlag) and flag.severity == "critical":
                return True
            elif isinstance(flag, dict) and flag.get("severity") == "critical":
                return True

    return False


def compute_abstention(
    domain_results: Dict[str, RewardDomainResult],
    threshold_tolerances: Dict[str, float],
    abstention_bias: float,
) -> bool:
    """
    Determine whether the system should abstain from responding.

    Abstain when the fraction of threshold-violated domains, weighted by
    *abstention_bias*, exceeds 0.5.

    Parameters
    ----------
    domain_results : dict
        domain_name -> RewardDomainResult
    threshold_tolerances : dict
        domain_name -> minimum acceptable general_score
    abstention_bias : float
        Profile abstention bias (0-1).  Higher = more willing to abstain.
    """
    violations = 0
    total_checked = 0

    for domain_name, tolerance in threshold_tolerances.items():
        result = domain_results.get(domain_name)
        if result is None:
            continue

        total_checked += 1
        if result.general_score < tolerance:
            violations += 1

    if total_checked == 0:
        return False

    violation_ratio = violations / total_checked
    # Higher abstention_bias → lower threshold → more willing to abstain.
    # E.g. bias=0.6 triggers at violation_ratio > 0.4; bias=0.9 at > 0.1.
    return violation_ratio > (1.0 - abstention_bias)


# ---------------------------------------------------------------------------
# Flag escalation
# ---------------------------------------------------------------------------

_SEVERITY_RANKS = {"info": 0, "warning": 1, "risk": 2, "critical": 3}
_SEVERITY_NAMES = {0: "info", 1: "warning", 2: "risk", 3: "critical"}


def escalate_domain_flags(
    domain_results: Dict[str, RewardDomainResult],
) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
    """
    Aggregate and escalate flags from all domains.

    Returns
    -------
    meta_flags : dict
        For ``RewardMetaDirective.flags``.  Keys are
        ``"{domain}_{flag_name}"`` plus ``"_max_severity"`` and
        ``"_total_flag_count"`` summary entries.
    domain_flag_lists : dict
        domain_name -> list of raw flag objects (for suppression check).
    """
    meta_flags: Dict[str, Any] = {}
    domain_flag_lists: Dict[str, List[Any]] = {}

    max_severity_rank = 0

    for domain_name, result in domain_results.items():
        domain_flags_list: List[Any] = []

        for flag_name, flag_obj in result.flags.items():
            if isinstance(flag_obj, RewardFlag):
                severity = flag_obj.severity
            elif isinstance(flag_obj, dict):
                severity = flag_obj.get("severity", "info")
            else:
                severity = "info"

            domain_flags_list.append(flag_obj)

            rank = _SEVERITY_RANKS.get(severity, 0)
            if rank > max_severity_rank:
                max_severity_rank = rank

            key = f"{domain_name}_{flag_name}"
            meta_flags[key] = {
                "domain": domain_name,
                "original_name": flag_name,
                "severity": severity,
                "source": flag_obj,
            }

        domain_flag_lists[domain_name] = domain_flags_list

    meta_flags["_max_severity"] = _SEVERITY_NAMES.get(max_severity_rank, "info")
    meta_flags["_total_flag_count"] = sum(
        len(v) for v in domain_flag_lists.values()
    )

    return meta_flags, domain_flag_lists


# ---------------------------------------------------------------------------
# Response-shaping directives
# ---------------------------------------------------------------------------

def _get_weighted_score(
    domain_results: Dict[str, RewardDomainResult],
    domain_weights: Dict[str, float],
    domain_name: str,
) -> float:
    """Return weight * general_score for *domain_name*, or 0.0 if missing."""
    result = domain_results.get(domain_name)
    if result is None:
        return 0.0
    weight = domain_weights.get(domain_name, 0.0)
    return weight * result.general_score


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp *value* to [low, high]."""
    return max(low, min(high, value))


def compute_response_directives(
    domain_results: Dict[str, RewardDomainResult],
    domain_weights: Dict[str, float],
    per_domain_tiers: Dict[str, int],
) -> Dict[str, float]:
    """
    Compute the 8 response-shaping directive values.

    All values are in [0.0, 1.0].

    Keys
    ----
    tone            : 0 = clinical, 1 = warm/empathetic
    structure       : 0 = loose/creative, 1 = rigid/formal
    metaphor_density: 0 = literal, 1 = heavily metaphorical
    reasoning_depth : 0 = shallow, 1 = deep analysis
    moralize        : 0 = neutral, 1 = strongly ethical framing
    clarify         : 0 = ambient, 1 = explicit precision
    speculate       : 0 = conservative, 1 = exploratory
    soothe          : 0 = neutral, 1 = comforting/reassuring
    """
    ethics_s = _get_weighted_score(domain_results, domain_weights, "ethics")
    logic_s = _get_weighted_score(domain_results, domain_weights, "logic")
    innovation_s = _get_weighted_score(domain_results, domain_weights, "innovation")
    attunement_s = _get_weighted_score(domain_results, domain_weights, "human_attunement")

    # tone: attunement drives warmth, logic dampens
    tone = (attunement_s * 0.7 + ethics_s * 0.3) * (1.0 - logic_s * 0.3)

    # structure: logic drives rigidity, innovation loosens
    structure = (logic_s * 0.6 + ethics_s * 0.2) * (1.0 - innovation_s * 0.4)

    # metaphor_density: innovation drives, logic suppresses
    metaphor_density = (innovation_s * 0.8) * (1.0 - logic_s * 0.5)

    # reasoning_depth: logic + ethics + innovation
    reasoning_depth = logic_s * 0.6 + ethics_s * 0.3 + innovation_s * 0.1

    # moralize: ethics tier directly drives
    ethics_tier = per_domain_tiers.get("ethics", 0)
    moralize = ethics_s * 0.7 + (ethics_tier / 3.0) * 0.3

    # clarify: logic drives precision
    clarify = logic_s * 0.7 + attunement_s * 0.2 + ethics_s * 0.1

    # speculate: innovation drives, ethics dampens
    speculate = (innovation_s * 0.8 + attunement_s * 0.1) * (1.0 - ethics_s * 0.3)

    # soothe: attunement drives, logic dampens
    soothe = (attunement_s * 0.8 + ethics_s * 0.2) * (1.0 - logic_s * 0.2)

    return {
        "tone": _clamp(tone),
        "structure": _clamp(structure),
        "metaphor_density": _clamp(metaphor_density),
        "reasoning_depth": _clamp(reasoning_depth),
        "moralize": _clamp(moralize),
        "clarify": _clamp(clarify),
        "speculate": _clamp(speculate),
        "soothe": _clamp(soothe),
    }


# ---------------------------------------------------------------------------
# Cross-domain interactions (second pass on directives)
# ---------------------------------------------------------------------------

def apply_cross_domain_interactions(
    directives: Dict[str, float],
    domain_results: Dict[str, RewardDomainResult],
    domain_weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Apply cross-domain interaction effects.

    From the masterdoc:
    - High logic reduces emotional ambiguity in relational tone
    - High ethics can constrain innovation's speculation
    - High attunement + logic boosts clarity
    - Very high innovation boosts metaphor density

    This is a *second pass* that modifies already-computed directives.
    """
    result = dict(directives)

    logic_result = domain_results.get("logic")
    ethics_result = domain_results.get("ethics")
    innovation_result = domain_results.get("innovation")
    attunement_result = domain_results.get("human_attunement")

    logic_score = logic_result.general_score if logic_result else 0.0
    ethics_score = ethics_result.general_score if ethics_result else 0.0
    innovation_score = innovation_result.general_score if innovation_result else 0.0
    attunement_score = attunement_result.general_score if attunement_result else 0.0

    # 1. High logic + low attunement -> reduce soothe
    if logic_score > 0.7 and attunement_score < 0.3:
        result["soothe"] = result["soothe"] * 0.5

    # 2. High ethics + high innovation -> tension
    if ethics_score > 0.7 and innovation_score > 0.7:
        result["speculate"] = result["speculate"] * 0.7
        result["moralize"] = min(1.0, result["moralize"] * 1.2)

    # 3. High attunement + high logic -> boost clarity
    if attunement_score > 0.5 and logic_score > 0.5:
        result["clarify"] = min(1.0, result["clarify"] * 1.15)

    # 4. Very high innovation -> metaphor boost
    if innovation_score > 0.75:
        result["metaphor_density"] = min(1.0, result["metaphor_density"] * 1.3)

    # Clamp all values
    for key in result:
        result[key] = _clamp(result[key])

    return result


# ---------------------------------------------------------------------------
# Routing hints
# ---------------------------------------------------------------------------

_APPROACH_MAP: Dict[str, Tuple[str, str, str, str]] = {
    "ethics": ("pragmatic", "principled", "reflective", "guardian"),
    "logic": ("casual", "structured", "analytical", "rigorous"),
    "innovation": ("conventional", "explorative", "inventive", "visionary"),
    "human_attunement": ("informational", "supportive", "empathetic", "deeply_attuned"),
}


def compute_routing(
    per_domain_tiers: Dict[str, int],
    domain_weights: Dict[str, float],
    composite_score: float,
) -> Dict[str, Any]:
    """
    Compute routing / selection hints for downstream LLM selection.

    Returns
    -------
    dict
        dominant_domain   : str — highest-weighted domain
        complexity_level  : int — composite tier (0-3)
        suggested_approach: str — e.g. "analytical", "empathetic"
        domain_influence  : dict — normalised influence per domain
    """
    # Normalised domain influence
    total = sum(domain_weights.values()) or 1.0
    influence = {
        domain: weight / total
        for domain, weight in domain_weights.items()
    }

    # Dominant domain (by weight)
    dominant = (
        max(domain_weights, key=domain_weights.get)
        if domain_weights
        else "logic"
    )

    # Complexity from composite tier
    complexity = classify_tier(composite_score)

    # Approach from dominant domain + its tier
    dominant_tier = per_domain_tiers.get(dominant, 0)
    approaches = _APPROACH_MAP.get(dominant, ("default",) * 4)
    approach = approaches[max(0, min(3, dominant_tier))]

    return {
        "dominant_domain": dominant,
        "complexity_level": complexity,
        "suggested_approach": approach,
        "domain_influence": influence,
    }
