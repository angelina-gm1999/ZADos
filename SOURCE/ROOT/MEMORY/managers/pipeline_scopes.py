"""
Pipeline scope declarations (Part C of spec).

Each processing pipeline has explicit READ and WRITE scope matrices
that declare which LTMM folders it can access.  This prevents identity
content from polluting knowledge retrieval and vice versa.

Usage:
    scope = get_pipeline_scope("regular")
    result = contrast.contrast(current=..., query_type="concept",
                               scope_filter=scope.read_scope)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from zados.memory.managers.scope_filter import (
    ScopeFilter,
    REGULAR_SCOPE, M1_M5_SCOPE, M2_SCOPE, M3_SCOPE,
    HOMEWORK_SCOPE, REFLECTIVE_SCOPE, REM_SCOPE, DREAM_SCOPE,
)


@dataclass(frozen=True)
class PipelineScope:
    """Read and write scope for a specific processing pipeline."""
    pipeline_name: str
    read_scope:    ScopeFilter
    write_scope:   ScopeFilter


# ---------------------------------------------------------------------------
# Pipeline scope constants
# ---------------------------------------------------------------------------

PIPELINE_REGULAR = PipelineScope(
    pipeline_name="regular",
    read_scope=REGULAR_SCOPE,
    write_scope=ScopeFilter(
        folders=frozenset({
            "thoughts/overview_logs",
            "thoughts/general_questions",
            "knowledge/lessons",
        }),
    ),
)

PIPELINE_M1 = PipelineScope(
    pipeline_name="m1_academic",
    read_scope=M1_M5_SCOPE,
    write_scope=ScopeFilter(
        folders=frozenset({
            "knowledge/lessons",
            "knowledge/notebook",
            "knowledge/academic_questions",
            "knowledge/knowledge_maps",
            # Part 4 §2.3
            "thoughts/general_questions",
            "thoughts/held_blocks",
        }),
    ),
)

PIPELINE_M2 = PipelineScope(
    pipeline_name="m2_peer_review",
    read_scope=M2_SCOPE,
    write_scope=ScopeFilter(
        folders=frozenset({
            "identity/core",
            "identity/conclusions",
            "identity/journal",
            # Part 4 §3.3
            "knowledge/lessons",
            "thoughts/held_blocks",
        }),
    ),
)

PIPELINE_M3 = PipelineScope(
    pipeline_name="m3_learn_together",
    read_scope=M3_SCOPE,
    write_scope=ScopeFilter(
        folders=frozenset({
            "identity/conclusions",
            "thoughts/general_questions",
            # Part 4 §4.3
            "knowledge/lessons",
            "knowledge/knowledge_maps",
            "knowledge/academic_questions",
            "knowledge/notebook",
            "thoughts/held_blocks",
        }),
    ),
)

PIPELINE_M4 = PipelineScope(
    pipeline_name="m4_knowledge_review",
    read_scope=ScopeFilter(
        folders=frozenset({
            "knowledge/lessons",
            "knowledge/knowledge_maps",
            "knowledge/academic_questions",
            # Part 4 §5.2 — question surfacing reads
            "thoughts/unsolved_buffer",
            "thoughts/general_questions",
        }),
        max_results=10,
    ),
    write_scope=ScopeFilter(
        folders=frozenset({
            "knowledge/lessons",
            "knowledge/knowledge_maps",
            # Part 4 §5.2
            "thoughts/unsolved_buffer",
        }),
    ),
)

PIPELINE_M5 = PipelineScope(
    pipeline_name="m5_integration",
    read_scope=ScopeFilter(
        folders=frozenset({
            "knowledge/lessons",
            "knowledge/library",
            "knowledge/academic_questions",
            "knowledge/notebook",
            "knowledge/knowledge_maps",
        }),
        max_results=15,
    ),
    write_scope=ScopeFilter(
        folders=frozenset({
            "knowledge/lessons",
            "knowledge/library",
            # Part 4 §6.2
            "knowledge/notebook",
            "knowledge/academic_questions",
        }),
    ),
)

PIPELINE_HOMEWORK = PipelineScope(
    pipeline_name="homework",
    read_scope=HOMEWORK_SCOPE,
    write_scope=ScopeFilter(
        folders=frozenset({
            "knowledge/lessons",
            "knowledge/notebook",
            "knowledge/academic_questions",
            "knowledge/knowledge_maps",
            # Part 5 §5.2 — expanded writes for offline integration
            "identity/core",               # via CoreMemoryUpdateGate only
            "thoughts/unsolved_buffer",
            "thoughts/general_questions",
            "thoughts/overview_logs",       # HomeworkRunSummary
        }),
    ),
)

PIPELINE_SELF_REFLECTIVE = PipelineScope(
    pipeline_name="self_reflective",
    read_scope=REFLECTIVE_SCOPE,
    write_scope=ScopeFilter(
        folders=frozenset({
            "identity/conclusions",
            "identity/journal",
            "thoughts/general_questions",
        }),
    ),
)

PIPELINE_REM = PipelineScope(
    pipeline_name="rem_sleep",
    read_scope=REM_SCOPE,
    write_scope=ScopeFilter(
        folders=frozenset({
            "thoughts/held_blocks",
            "thoughts/unsolved_buffer",
            "knowledge/academic_buffer",
        }),
    ),
)

PIPELINE_DREAM = PipelineScope(
    pipeline_name="dream",
    read_scope=DREAM_SCOPE,
    write_scope=ScopeFilter(
        folders=frozenset({
            "thoughts/unsolved_buffer",
            "knowledge/lessons",
            "knowledge/academic_buffer",
        }),
    ),
)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

_PIPELINE_REGISTRY: Dict[str, PipelineScope] = {
    ps.pipeline_name: ps
    for ps in [
        PIPELINE_REGULAR, PIPELINE_M1, PIPELINE_M2, PIPELINE_M3,
        PIPELINE_M4, PIPELINE_M5, PIPELINE_HOMEWORK,
        PIPELINE_SELF_REFLECTIVE, PIPELINE_REM, PIPELINE_DREAM,
    ]
}


def get_pipeline_scope(name: str) -> Optional[PipelineScope]:
    """Look up a pipeline scope by name. Returns None if not found."""
    return _PIPELINE_REGISTRY.get(name)


def get_all_pipeline_names() -> list:
    """Return sorted list of all registered pipeline names."""
    return sorted(_PIPELINE_REGISTRY.keys())
