"""
RetrievalRouter — query-type-based routing to namespaced stores.

Routes retrieval queries to the appropriate LTMM namespace stores based
on query_type semantics:

  "knowledge"  → knowledge/ stores
  "identity"   → identity/ stores
  "thought"    → thoughts/ stores
  "general"    → thoughts/ + knowledge/ (subject-filtered)
  unknown      → flat LTMMStore fallback

Each route returns aggregated (score, entry) tuples sorted by similarity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from zados.memory.long_term.store import LTMMStore
from zados.memory.managers.scope_filter import ScopeFilter


@dataclass
class RetrievalContext:
    """Encapsulates a retrieval query with routing metadata."""
    query_text:     str
    query_type:     str           = "general"  # "knowledge" | "identity" | "thought" | "general"
    pipeline_name:  str           = ""
    scope_filter:   Optional[ScopeFilter] = None
    limit:          int           = 5
    tags:           List[str]     = field(default_factory=list)


# ---------------------------------------------------------------------------
# Query type → folder routing rules
# ---------------------------------------------------------------------------

_ROUTE_MAP: Dict[str, List[str]] = {
    "knowledge": [
        "knowledge/lessons",
        "knowledge/library",
        "knowledge/academic_questions",
        "knowledge/notebook",
        "knowledge/knowledge_maps",
    ],
    "identity": [
        "identity/core",
        "identity/conclusions",
        "identity/journal",
    ],
    "thought": [
        "thoughts/overview_logs",
        "thoughts/held_blocks",
        "thoughts/general_questions",
    ],
    "general": [
        "thoughts/general_questions",
        "thoughts/overview_logs",
        "knowledge/lessons",
        "knowledge/library",
    ],
}

# Folder → (namespace_attr, store_attr)
_FOLDER_STORE_MAP: Dict[str, Tuple[str, str]] = {
    "identity/hardcoded":            ("identity", "hardcoded"),
    "identity/core":                 ("identity", "core"),
    "identity/conclusions":          ("identity", "conclusions"),
    "identity/journal":              ("identity", "journal"),
    "thoughts/overview_logs":        ("thoughts", "overview_logs"),
    "thoughts/held_blocks":          ("thoughts", "held_blocks"),
    "thoughts/unsolved_buffer":      ("thoughts", "unsolved_buffer"),
    "thoughts/general_questions":    ("thoughts", "general_questions"),
    "knowledge/library":             ("knowledge", "library"),
    "knowledge/lessons":             ("knowledge", "lessons"),
    "knowledge/academic_buffer":     ("knowledge", "academic_buffer"),
    "knowledge/academic_questions":  ("knowledge", "academic_questions"),
    "knowledge/knowledge_maps":      ("knowledge", "knowledge_maps"),
    "knowledge/notebook":            ("knowledge", "notebook"),
    "knowledge/cognitools_data":     ("knowledge", "cognitools_data"),
}


class RetrievalRouter:
    """
    Routes retrieval queries to namespaced stores by query_type.

    Falls back to flat LTMMStore for unknown query types or when
    namespaces are not available.
    """

    def __init__(
        self,
        ltmm: LTMMStore,
        namespaces: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._ltmm = ltmm
        self._namespaces = namespaces or {}

    def route(self, ctx: RetrievalContext) -> List[Tuple[float, Any]]:
        """
        Route a retrieval context to the appropriate stores.

        If ``ctx.scope_filter`` is set, its folders take precedence
        over query_type routing.

        Returns (similarity, entry) tuples sorted descending.
        """
        # Priority 1: explicit scope_filter
        if ctx.scope_filter is not None:
            folders = list(ctx.scope_filter.folders)
        else:
            # Priority 2: query_type routing
            folders = _ROUTE_MAP.get(ctx.query_type, [])

        if not folders or not self._namespaces:
            # Fallback: flat LTMM
            return self._ltmm.search(ctx.query_text, limit=ctx.limit)

        # Search across targeted stores
        all_hits: List[Tuple[float, Any]] = []
        for folder in folders:
            mapping = _FOLDER_STORE_MAP.get(folder)
            if mapping is None:
                continue
            ns_name, store_attr = mapping
            ns = self._namespaces.get(ns_name)
            if ns is None:
                continue
            store = getattr(ns, store_attr, None)
            if store is None or not hasattr(store, "search"):
                continue

            results = store.search(ctx.query_text, limit=ctx.limit)
            all_hits.extend(results)

        # Tag filtering
        if ctx.tags:
            tag_set = set(ctx.tags)
            filtered = []
            for sim, entry in all_hits:
                entry_tags = set(getattr(entry, "tags", []))
                if tag_set.issubset(entry_tags):
                    filtered.append((sim, entry))
            all_hits = filtered

        # Subject filtering from scope_filter
        if ctx.scope_filter and ctx.scope_filter.subject_filter:
            subj = ctx.scope_filter.subject_filter
            all_hits = [
                (sim, entry) for sim, entry in all_hits
                if getattr(entry, "subject_category", None) in (None, "", subj)
            ]

        # Sort and limit
        all_hits.sort(key=lambda x: x[0], reverse=True)
        return all_hits[:ctx.limit]
