"""
ScopeFilter — restricts memory contrast and retrieval to specific LTMM subspaces.

A frozen dataclass that declares which folders (namespace stores) to search,
which tags are required/excluded, and optional subject filtering.

When passed to MemoryContrast.contrast(), the search is routed to the
specified namespaced stores instead of the flat LTMM scan.
When scope_filter is None, existing flat behaviour is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple


@dataclass(frozen=True)
class ScopeFilter:
    """
    Immutable scope declaration for namespaced memory queries.

    Parameters
    ----------
    folders : frozenset of str
        Which LTMM namespace folders to include.
        Valid values: "identity/hardcoded", "identity/core",
        "identity/conclusions", "identity/journal",
        "thoughts/overview_logs", "thoughts/held_blocks",
        "thoughts/unsolved_buffer", "thoughts/general_questions",
        "knowledge/library", "knowledge/lessons",
        "knowledge/academic_buffer", "knowledge/academic_questions",
        "knowledge/knowledge_maps", "knowledge/notebook",
        "knowledge/cognitools_data"
    required_tags : frozenset of str
        Only include entries whose tags contain ALL of these.
    excluded_tags : frozenset of str
        Exclude entries whose tags contain ANY of these.
    subject_filter : str or None
        If set, only include entries with matching subject_category.
    max_results : int
        Maximum results across all targeted stores.
    include_cold : bool
        Whether to include cold-storage entries (LTMM flat store only).
    """
    folders:        FrozenSet[str] = field(default_factory=frozenset)
    required_tags:  FrozenSet[str] = field(default_factory=frozenset)
    excluded_tags:  FrozenSet[str] = field(default_factory=frozenset)
    subject_filter: Optional[str]  = None
    max_results:    int            = 10
    include_cold:   bool           = False


# ---------------------------------------------------------------------------
# Pre-built pipeline scope constants (Part D of spec)
# ---------------------------------------------------------------------------

REGULAR_SCOPE = ScopeFilter(
    folders=frozenset({
        "thoughts/overview_logs",
        "thoughts/general_questions",
        "knowledge/lessons",
        "knowledge/library",
        "thoughts/held_blocks",   # pulled during Phase 3 scoped contrast
    }),
    max_results=12,
)

M1_M5_SCOPE = ScopeFilter(
    folders=frozenset({
        "knowledge/lessons",
        "knowledge/library",
        "knowledge/academic_questions",
        "knowledge/notebook",
    }),
    required_tags=frozenset({"pipeline:m1"}),
    max_results=10,
)

M2_SCOPE = ScopeFilter(
    folders=frozenset({
        "identity/core",
        "identity/conclusions",
        "identity/journal",
    }),
    required_tags=frozenset({"pipeline:m2"}),
    max_results=8,
)

M3_SCOPE = ScopeFilter(
    folders=frozenset({
        "identity/core",
        "identity/conclusions",
        "thoughts/general_questions",
    }),
    max_results=8,
)

HOMEWORK_SCOPE = ScopeFilter(
    folders=frozenset({
        "knowledge/lessons",
        "knowledge/library",
        "knowledge/academic_questions",
        "knowledge/knowledge_maps",
        "knowledge/notebook",
        # Part 5 §5.1 — expanded reads for offline processing
        "thoughts/overview_logs",
        "thoughts/unsolved_buffer",
        "thoughts/general_questions",
    }),
    max_results=20,  # higher than learning modes — processing all accumulated material
)

REFLECTIVE_SCOPE = ScopeFilter(
    folders=frozenset({
        "identity/core",
        "identity/conclusions",
        "identity/journal",
        "thoughts/held_blocks",
        "thoughts/general_questions",
    }),
    max_results=10,
)

REM_SCOPE = ScopeFilter(
    folders=frozenset({
        "thoughts/unsolved_buffer",
        "thoughts/held_blocks",
        "knowledge/academic_buffer",
    }),
    max_results=10,
    include_cold=True,
)

DREAM_SCOPE = ScopeFilter(
    folders=frozenset({
        "thoughts/unsolved_buffer",
        "thoughts/held_blocks",
        "knowledge/academic_buffer",
        "knowledge/lessons",
    }),
    max_results=15,
    include_cold=True,
)
