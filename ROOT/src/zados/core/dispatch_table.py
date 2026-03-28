"""
ZA-DOS Core Pipeline — Engine Dispatch Table (spec §6.2).

Maps intent archetypes to sets of engine numbers that should be dispatched.
Engine numbers match ENGINE_IDS in cognitive_engines/constants.py.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Set

# ------------------------------------------------------------------
# Archetype → engine numbers  (spec §6.2 table)
# ------------------------------------------------------------------

ARCHETYPE_ENGINE_TABLE: Dict[str, Set[int]] = {
    "ANALYTICAL": {
        23, 8, 11, 24, 18, 19,
        1, 2, 4, 5, 6,
        9, 10, 16,
        12, 7, 14, 20,
    },
    "CREATIVE": {
        23, 8, 11, 24, 19,
        5, 9, 13, 16,
    },
    "EMPATHIC": {
        23, 8, 11, 24,
        5, 14,
    },
    "STRATEGIC": {
        23, 8, 11, 24, 18, 19,
        1, 2, 4, 5, 6,
        9, 10, 16,
        12, 13, 15, 7, 14, 21, 20,
    },
    "REFLECTIVE": {
        23, 8, 11, 24, 18, 19,
        1, 2, 4, 5, 6,
        9, 10,
        12, 15, 7, 14, 21,
    },
    "GENERATIVE": {
        23, 8, 11, 24, 19,
        1, 5, 9,
        16, 12, 20,
    },
    "SOCIAL": {
        23, 8, 11, 24,
    },
}

# ------------------------------------------------------------------
# Always-run guardrail engines (detection cluster)
# ------------------------------------------------------------------

GUARDRAIL_ENGINES: FrozenSet[int] = frozenset({1, 2, 4, 5, 6})

# ------------------------------------------------------------------
# Perception engines (always run in Phase 1, before dispatch)
# ------------------------------------------------------------------

PERCEPTION_ENGINES: List[int] = [23, 8, 11, 18, 19]

# ------------------------------------------------------------------
# Post-processing engines (Phase 7 learning cluster)
# ------------------------------------------------------------------

POSTPROCESS_ENGINES: List[int] = [29, 17, 22, 25]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

DEFAULT_ARCHETYPE = "REFLECTIVE"


def get_dispatch_list(
    archetype: str,
    engine_weights: Dict[str, float],
) -> List[int]:
    """Return sorted engine numbers for *archetype*, filtered by weight > 0.

    Guardrail engines (E1, E2, E4, E5, E6) are always included if their
    weight is > 0, regardless of archetype.

    Parameters
    ----------
    archetype : str
        Intent archetype (e.g. "ANALYTICAL").  Falls back to
        DEFAULT_ARCHETYPE if not found.
    engine_weights : dict
        Map of engine_id_str → weight.  Engines with weight <= 0 are
        excluded from dispatch.

    Returns
    -------
    List[int]
        Engine numbers, sorted ascending (lower = runs first).
    """
    archetype_upper = archetype.upper() if archetype else DEFAULT_ARCHETYPE
    base = ARCHETYPE_ENGINE_TABLE.get(archetype_upper, ARCHETYPE_ENGINE_TABLE[DEFAULT_ARCHETYPE])

    # Merge guardrails (always eligible if weight > 0)
    candidates = base | GUARDRAIL_ENGINES

    # Exclude perception engines (they already ran in Phase 1)
    perception_set = set(PERCEPTION_ENGINES)

    # Exclude post-processing engines (they run in Phase 7)
    postprocess_set = set(POSTPROCESS_ENGINES)

    # Filter by engine weight > 0
    def _weight(eng_num: int) -> float:
        return engine_weights.get(str(eng_num), 1.0)

    dispatched = sorted(
        e for e in candidates
        if _weight(e) > 0 and e not in perception_set and e not in postprocess_set
    )
    return dispatched
