"""
Cross-cutting tag taxonomy for all LTMM entry types.

Tags are plain strings.  The constants below are reserved system tags used
by pipeline code for routing and filtering.  Custom tags are allowed
alongside these — validation only checks *system* tag format.
"""
from __future__ import annotations

from typing import FrozenSet, List

# ── Identity tags ────────────────────────────────────────────────────────
IDENTITY_TAGS: FrozenSet[str] = frozenset({
    "identity:core",
    "identity:values",
    "identity:boundary",
    "identity:drift",
    "identity:update_proposed",
    "identity:update_validated",
})

# ── Cognitive state tags ─────────────────────────────────────────────────
COGNITIVE_TAGS: FrozenSet[str] = frozenset({
    "cognitive:interrupt",
    "cognitive:high_salience",
    "cognitive:unresolved",
    "cognitive:contradiction",
    "cognitive:paradox",
    "cognitive:novel",
})

# ── Pipeline source tags ─────────────────────────────────────────────────
PIPELINE_TAGS: FrozenSet[str] = frozenset({
    "pipeline:reflective",
    "pipeline:peer_review",
    "pipeline:rem",
    "pipeline:dream",
    "pipeline:homework",
    "pipeline:m1",
    "pipeline:m2",
    "pipeline:m3",
    "pipeline:m4",
    "pipeline:m5",
})

# ── Knowledge domain tags ────────────────────────────────────────────────
DOMAIN_TAGS: FrozenSet[str] = frozenset({
    "domain:technical",
    "domain:philosophical",
    "domain:creative",
    "domain:social",
    "domain:historical",
    "domain:practical",
    "domain:linguistic",
})

# All reserved system tags (union)
ALL_SYSTEM_TAGS: FrozenSet[str] = IDENTITY_TAGS | COGNITIVE_TAGS | PIPELINE_TAGS | DOMAIN_TAGS

# Valid prefixes for system tags
TAG_PREFIXES: FrozenSet[str] = frozenset({
    "identity:", "cognitive:", "pipeline:", "domain:",
})


def validate_tags(tags: List[str]) -> List[str]:
    """Return list of invalid system-prefixed tags (empty = all valid).

    Custom tags (no system prefix) are always accepted.
    """
    invalid: List[str] = []
    for tag in tags:
        for prefix in TAG_PREFIXES:
            if tag.startswith(prefix) and tag not in ALL_SYSTEM_TAGS:
                invalid.append(tag)
                break
    return invalid
