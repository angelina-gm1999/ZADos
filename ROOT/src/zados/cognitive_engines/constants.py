"""
ZADOS Cognitive Engines — Shared Constants & Canonical Notation
================================================================

Defines the canonical naming conventions derived from the
**Master Neurochemical Appendix** so that every engine uses
identical keys, field names, and oscillatory band labels.

Notation reference
------------------
- NT set  N = {Glu, GABA, DA, 5HT, NE, ACh, OXT, MOR, CB1, CRH,
               Cortisol, Histamine}
- Receptor  R_{ij}  where j enumerates subtypes of NT i
- Oscillatory bands  phi_k(t), k in {delta, theta, alpha, beta, gamma}
- Cross-frequency    phi_{theta-gamma}, phi_{alpha-beta}
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

# =====================================================================
# 1. Canonical NT Key Names  (used as dict keys in update_neurochem_state)
# =====================================================================
# The pipeline orchestrator pushes a Dict[str, float] whose keys are
# drawn from this set.  Every engine must accept these keys (not
# ``_level`` suffixed variants, not named positional params).

NT_KEYS: FrozenSet[str] = frozenset({
    "glu",        # Glutamate
    "gaba",       # GABA
    "da",         # Dopamine
    "5ht",        # Serotonin  (dict key — starts with digit is OK in strings)
    "ne",         # Norepinephrine
    "ach",        # Acetylcholine
    "oxt",        # Oxytocin
    "mor",        # mu-Opioid
    "cb1",        # Endocannabinoid (CB1)
    "crh",        # Corticotropin-Releasing Hormone
    "cor",        # Cortisol  (canonical abbreviation; maps to "Cortisol" in N)
    "histamine",  # Histamine
})

# Full canonical name mapping  (abbreviation -> formal name in N)
NT_CANONICAL_NAMES: Dict[str, str] = {
    "glu":       "Glutamate",
    "gaba":      "GABA",
    "da":        "Dopamine",
    "5ht":       "5-HT (Serotonin)",
    "ne":        "Norepinephrine",
    "ach":       "Acetylcholine",
    "oxt":       "Oxytocin",
    "mor":       "mu-Opioid",
    "cb1":       "CB1 (Endocannabinoid)",
    "crh":       "CRH",
    "cor":       "Cortisol",
    "histamine": "Histamine",
}

# =====================================================================
# 2. Canonical State Field Names
# =====================================================================
# Python identifiers can't start with a digit, so 5-HT becomes
# ``_5ht_level`` in state dataclasses.  All others use ``<nt>_level``.

NT_STATE_FIELD: Dict[str, str] = {
    "glu":       "glu_level",
    "gaba":      "gaba_level",
    "da":        "da_level",
    "5ht":       "_5ht_level",     # leading underscore — canonical
    "ne":        "ne_level",
    "ach":       "ach_level",
    "oxt":       "oxt_level",
    "mor":       "mor_level",
    "cb1":       "cb1_level",
    "crh":       "crh_level",
    "cor":       "cor_level",
    "histamine": "histamine_level",
}

# =====================================================================
# 3. Canonical Oscillatory Band Labels
# =====================================================================
# phi_k(t) in [0, 1],  k in OSCILLATORY_BANDS

OSCILLATORY_BANDS: Tuple[str, ...] = (
    "delta",   # delta  (0.5-4 Hz)
    "theta",   # theta  (4-8 Hz)
    "alpha",   # alpha  (8-12 Hz)
    "beta",    # beta   (12-30 Hz)
    "gamma",   # gamma  (30-80 Hz)
    "sigma",   # sigma  (12-15 Hz) — sleep spindle / thalamocortical replay
)

# Cross-frequency coupling products (from Master Appendix sec. 7)
CROSS_FREQUENCY_COUPLINGS: Tuple[str, ...] = (
    "theta_gamma",   # phi_theta * phi_gamma
    "alpha_beta",    # phi_alpha * phi_beta
    "delta_sigma",   # phi_delta * phi_sigma — NREM consolidation coupling
)

# =====================================================================
# 4. Canonical Oscillatory Signal Names (for neurochem output)
# =====================================================================
# Engines emit oscillatory modulation signals using these names.
# ``_boost`` means positive modulation (increase band power).
# ``_suppress`` means negative modulation (decrease band power).
#
# Single-band signals:
#   delta_boost / delta_suppress
#   theta_boost / theta_suppress
#   alpha_boost / alpha_suppress
#   beta_boost  / beta_suppress
#   gamma_boost / gamma_suppress
#
# Cross-frequency signals:
#   theta_gamma_boost / theta_gamma_suppress
#   alpha_beta_boost  / alpha_beta_suppress

OSCILLATORY_BOOST_SIGNALS: Tuple[str, ...] = tuple(
    f"{band}_boost" for band in OSCILLATORY_BANDS
) + tuple(
    f"{cfc}_boost" for cfc in CROSS_FREQUENCY_COUPLINGS
)

OSCILLATORY_SUPPRESS_SIGNALS: Tuple[str, ...] = tuple(
    f"{band}_suppress" for band in OSCILLATORY_BANDS
) + tuple(
    f"{cfc}_suppress" for cfc in CROSS_FREQUENCY_COUPLINGS
)

# =====================================================================
# 5. Canonical NT-Oscillatory Associations (Appendix sec. 9)
# =====================================================================

NT_BAND_ASSOCIATIONS: Dict[str, Tuple[str, ...]] = {
    "da":        ("gamma", "theta"),
    "5ht":       ("theta", "alpha"),
    "glu":       ("gamma", "theta_gamma"),   # NMDA-gated
    "gaba":      ("alpha", "delta"),
    "ach":       ("beta",),
    "ne":        ("beta",),
    "oxt":       ("theta",),
    "cb1":       ("delta", "alpha_beta"),
    "mor":       ("delta",),
    "crh":       ("beta",),
    "cor":       ("beta",),
    "histamine": ("beta", "gamma"),
}

# =====================================================================
# 6. Engine Cluster Canonical Names
# =====================================================================

ENGINE_CLUSTERS: Tuple[str, ...] = (
    "detection",            # Engines 1, 2, 4, 5, 6
    "dialectic",            # Engines 7, 14
    "executive_control",    # Engine 3
    "knowledge_substrate",  # Engines 9, 10, 16
    "pattern_analysis",     # Engines 8, 11, 18, 19, 20, 23
    "evaluation",           # Engine 12
    "reasoning",            # Engines 13, 15, 21
    "metacognition",        # Engine 24
    "meta_self_awareness",  # Engine 26
    "homeostasis",          # Engines 27, 29
    "emotional_processing", # Engine 28
    "alignment",            # Engine 30
    "learning",             # Engines 17, 22, 25
)

# =====================================================================
# 7. Canonical Engine ID Mapping
# =====================================================================

ENGINE_IDS: Dict[int, str] = {
    1:  "contradiction_detection_engine",
    2:  "paradox_detection_engine",
    3:  "soar_production_engine",
    4:  "fallacy_detection_engine",
    5:  "bias_detection_engine",
    6:  "logic_trap_detection_engine",
    7:  "simulated_opposition_engine",
    8:  "relevance_scoring_engine",
    9:  "atomspace_engine",
    10: "pln_engine",
    11: "input_relevance_evaluation_engine",
    12: "logical_brain_engine",
    13: "simulation_brain_engine",
    14: "socratic_reasoning_engine",
    15: "decision_making_engine",
    16: "ecan_engine",
    17: "reward_based_learning_engine",
    18: "data_analysis_engine",
    19: "pattern_identification_engine",
    20: "pattern_comparison_engine",
    21: "strategic_decision_engine",
    22: "contextual_learning_engine",
    23: "intention_map_engine",
    24: "heuristic_bias_engine",
    25: "recursive_learning_engine",
    26: "uncertainty_pattern_engine",
    27: "neurochemical_homeostatic_engine",
    28: "emotional_detection_engine",
    29: "memory_compression_engine",
    30: "retroactive_alignment_engine",
    31: "reflective_learning_engine",
    32: "reflective_identity_engine",
}

ENGINE_CLUSTER_MAP: Dict[int, str] = {
    1:  "detection",
    2:  "detection",
    3:  "executive_control",
    4:  "detection",
    5:  "detection",
    6:  "detection",
    7:  "dialectic",
    8:  "pattern_analysis",
    9:  "knowledge_substrate",
    10: "knowledge_substrate",
    11: "pattern_analysis",
    12: "evaluation",
    13: "reasoning",
    14: "dialectic",
    15: "reasoning",
    16: "knowledge_substrate",
    17: "learning",
    18: "pattern_analysis",
    19: "pattern_analysis",
    20: "pattern_analysis",
    21: "reasoning",
    22: "learning",
    23: "pattern_analysis",
    24: "metacognition",
    25: "learning",
    26: "meta_self_awareness",
    27: "homeostasis",
    28: "emotional_processing",
    29: "homeostasis",
    30: "alignment",
    31: "metacognition",
    32: "metacognition",
}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Canonical bounded clamp — shared across all engines."""
    return max(lo, min(hi, v))


# =====================================================================
# 8. NT Key Normalisation
# =====================================================================
# The cognitive-engine layer uses lowercase keys (NT_KEYS above).
# The neurochemical-engine layer uses uppercase keys ("DA", "5HT", …).
# This mapping lets any boundary translate any known variant into the
# canonical *lowercase* key defined in NT_KEYS, or into the canonical
# *uppercase* key used by the neurochem layer.
#
# Known variant families (case-insensitive, punctuation-stripped):
#   "DA", "da"                          → da / DA
#   "5HT", "5ht", "5-HT", "5-ht"      → 5ht / 5HT
#   "NE", "ne"                          → ne / NE
#   "ACh", "ach", "ACH"                 → ach / ACh
#   "OXT", "oxt"                        → oxt / OXT
#   "MOR", "mor"                        → mor / MOR
#   "CB1", "cb1"                        → cb1 / CB1
#   "CRH", "crh"                        → crh / CRH
#   "GABA", "gaba"                      → gaba / GABA
#   "GLU", "glu"                        → glu / GLU
#   "cortisol", "Cortisol", "cor"       → cor / cortisol
#   "histamine", "Histamine"            → histamine / histamine

# _variant → canonical lowercase key
_NT_KEY_ALIASES: Dict[str, str] = {}
for _key in NT_KEYS:
    _NT_KEY_ALIASES[_key] = _key                 # identity
    _NT_KEY_ALIASES[_key.upper()] = _key          # "DA" → "da", "GABA" → "gaba"

# Extra variant aliases that aren't covered by simple upper/lower
_NT_KEY_ALIASES.update({
    "5-HT":      "5ht",
    "5-ht":      "5ht",
    "5HT":       "5ht",
    "ACh":       "ach",
    "ACH":       "ach",
    "Cortisol":  "cor",
    "cortisol":  "cor",
    "CORTISOL":  "cor",
    "Histamine": "histamine",
    "HISTAMINE": "histamine",
})

# canonical lowercase → canonical uppercase (neurochem layer convention)
NT_KEY_TO_UPPER: Dict[str, str] = {
    "glu":       "GLU",
    "gaba":      "GABA",
    "da":        "DA",
    "5ht":       "5HT",
    "ne":        "NE",
    "ach":       "ACh",
    "oxt":       "OXT",
    "mor":       "MOR",
    "cb1":       "CB1",
    "crh":       "CRH",
    "cor":       "cortisol",   # registry key is lowercase (full word, not abbreviation)
    "histamine": "histamine",  # registry key is lowercase (full word, not abbreviation)
}


def normalize_nt_key(key: str, target: str = "lower") -> str:
    """Map any known NT key variant to the canonical form.

    Parameters
    ----------
    key : str
        Any NT key variant (e.g. ``"DA"``, ``"5-HT"``, ``"cortisol"``).
    target : str
        ``"lower"`` → cognitive-engine convention (``"da"``, ``"5ht"``).
        ``"upper"`` → neurochem-engine convention (``"DA"``, ``"5HT"``).

    Returns
    -------
    str
        Canonical key in the requested convention, or *key* unchanged
        if it is not a recognised NT variant.
    """
    lower = _NT_KEY_ALIASES.get(key)
    if lower is None:
        return key  # unknown variant — pass through unchanged
    if target == "upper":
        return NT_KEY_TO_UPPER.get(lower, lower)
    return lower
