"""
Concrete implementation of MemoryContrastPort.

This is the bridge between the Memory Layer and the Logic Domain submodules.
It implements the Protocol defined in reward/domains/logic/ports.py using
the live MTMM and LTMM stores.

Query type routing:
  'context'          → context/scope fidelity (MTMM: last N turns)
  'concept'          → concept continuity     (MTMM + LTMM)
  'concept_fidelity' → definition adherence   (LTMM primary)
  'semantic'         → semantic drift         (MTMM: recent trajectory)
  'internal'         → internal contradiction (STMM via MTMM last-turn)
  'external'         → cross-turn contradiction (MTMM + LTMM)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from zados.memory.mid_term.store import MTMMStore
from zados.memory.long_term.store import LTMMStore
from zados.memory.managers.scope_filter import ScopeFilter
from zados.reward.domains.logic.ports import ContrastResult


# ---------------------------------------------------------------------------
# Folder → store attribute path mapping for scoped search
# ---------------------------------------------------------------------------

_FOLDER_ATTR_MAP: Dict[str, Tuple[str, str]] = {
    "identity/hardcoded":            ("identity", "hardcoded"),
    "identity/core":                 ("identity", "core"),
    "identity/conclusions":          ("identity", "conclusions"),
    "identity/journal":              ("identity", "journal"),
    "identity/correlation":          ("identity", "correlation"),
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


class MemoryContrast:
    """
    Concrete MemoryContrastPort.

    Plug this into LogicDomain, InternalConsistencySubmodule, etc.:

        logic_domain = LogicDomain(memory_contrast=MemoryContrast(mtmm, ltmm))

    When ``scope_filter`` is provided to ``contrast()``, the search is
    routed to namespaced stores instead of (or in addition to) the flat
    LTMM scan.  When ``scope_filter is None``, behaviour is identical to
    the pre-namespace implementation.
    """

    def __init__(
        self,
        mtmm: MTMMStore,
        ltmm: LTMMStore,
        mtmm_weight: float = 0.6,
        ltmm_weight: float = 0.4,
        namespaces: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._mtmm        = mtmm
        self._ltmm        = ltmm
        self._mtmm_weight = mtmm_weight
        self._ltmm_weight = ltmm_weight
        self._namespaces   = namespaces  # {"identity": ..., "thoughts": ..., "knowledge": ...}

    # -----------------------------------------------------------------------
    # Protocol implementation
    # -----------------------------------------------------------------------

    def contrast(
        self,
        *,
        current: Dict[str, Any],
        query_type: str,
        ctx_id: Optional[str] = None,
        limit: int = 5,
        meta: Optional[Dict[str, Any]] = None,
        scope_filter: Optional[ScopeFilter] = None,
    ) -> ContrastResult:
        query_text = self._extract_query_text(current)

        # If a scope_filter is provided and namespaces are wired, use
        # scoped search over namespaced stores instead of flat LTMM.
        if scope_filter is not None and self._namespaces is not None:
            return self._scoped_search(query_text, scope_filter, query_type)

        if query_type in ("context", "semantic", "internal"):
            return self._mtmm_search(query_text, limit, query_type)
        else:
            return self._combined_search(query_text, limit, query_type)

    # -----------------------------------------------------------------------
    # Search strategies
    # -----------------------------------------------------------------------

    def _mtmm_search(
        self, query_text: str, limit: int, query_type: str
    ) -> ContrastResult:
        hits = self._mtmm.search(query_text, limit=limit)
        if not hits:
            return ContrastResult(similarity=0.0, divergence=0.0)

        best_sim  = hits[0][0]
        avg_sim   = sum(s for s, _ in hits) / len(hits)
        # Divergence = 1 - similarity (simple complement)
        divergence = max(0.0, 1.0 - best_sim)

        references = [
            {
                "packet_id":   pkt.packet_id,
                "turn_index":  pkt.turn_index,
                "similarity":  round(sim, 3),
                "source":      "MTMM",
                "query_type":  query_type,
                "summary":     pkt.user_message[:120],
            }
            for sim, pkt in hits
        ]

        return ContrastResult(
            similarity=round(best_sim, 3),
            divergence=round(divergence, 3),
            references=references,
            meta={"avg_similarity": round(avg_sim, 3), "hit_count": len(hits)},
        )

    def _ltmm_search(
        self, query_text: str, limit: int, query_type: str
    ) -> ContrastResult:
        # internal=True prevents inflating retrieval_count on entries
        # that are only being checked for consistency, not user-requested.
        hits = self._ltmm.search(query_text, limit=limit, internal=True)
        if not hits:
            return ContrastResult(similarity=0.0, divergence=0.0)

        best_sim   = hits[0][0]
        divergence = max(0.0, 1.0 - best_sim)

        references = [
            {
                "packet_id":   entry.packet.packet_id,
                "relevance":   round(entry.relevance_score, 3),
                "similarity":  round(sim, 3),
                "source":      "LTMM",
                "query_type":  query_type,
                "granularity": entry.granularity,
                "summary":     entry.packet.user_message[:120],
            }
            for sim, entry in hits
        ]

        return ContrastResult(
            similarity=round(best_sim, 3),
            divergence=round(divergence, 3),
            references=references,
        )

    def _combined_search(
        self, query_text: str, limit: int, query_type: str
    ) -> ContrastResult:
        m_res = self._mtmm_search(query_text, limit, query_type)
        l_res = self._ltmm_search(query_text, limit, query_type)

        # Weighted blend
        sim = (
            self._mtmm_weight * m_res.similarity +
            self._ltmm_weight * l_res.similarity
        )
        div = (
            self._mtmm_weight * m_res.divergence +
            self._ltmm_weight * l_res.divergence
        )

        references = list(m_res.references) + list(l_res.references)
        references.sort(key=lambda r: r.get("similarity", 0.0), reverse=True)

        return ContrastResult(
            similarity=round(sim, 3),
            divergence=round(div, 3),
            references=references[:limit],
            meta={
                "mtmm_sim": m_res.similarity,
                "ltmm_sim": l_res.similarity,
                "query_type": query_type,
            },
        )

    # -----------------------------------------------------------------------
    # Scoped search (namespaced stores)
    # -----------------------------------------------------------------------

    def _scoped_search(
        self,
        query_text: str,
        scope_filter: ScopeFilter,
        query_type: str,
    ) -> ContrastResult:
        """
        Search across namespaced stores matching the scope_filter.folders.

        Aggregates results from all matching stores, applies tag filters,
        and returns a ContrastResult.
        """
        all_hits: List[Tuple[float, Any, str]] = []  # (sim, entry, folder)

        for folder in scope_filter.folders:
            attr_path = _FOLDER_ATTR_MAP.get(folder)
            if attr_path is None:
                continue
            ns_name, store_name = attr_path
            ns = self._namespaces.get(ns_name)
            if ns is None:
                continue
            store = getattr(ns, store_name, None)
            if store is None:
                continue

            # Skip stores without a callable search() (e.g. HardcodedStore)
            if not callable(getattr(store, "search", None)):
                continue

            results = store.search(query_text, limit=scope_filter.max_results)
            for sim, entry in results:
                all_hits.append((sim, entry, folder))

        # Apply tag filters
        filtered: List[Tuple[float, Any, str]] = []
        for sim, entry, folder in all_hits:
            tags = getattr(entry, "tags", [])
            tag_set = set(tags) if tags else set()

            # required_tags: entry must have ALL
            if scope_filter.required_tags and not scope_filter.required_tags.issubset(tag_set):
                continue
            # excluded_tags: entry must have NONE
            if scope_filter.excluded_tags and scope_filter.excluded_tags & tag_set:
                continue
            # subject_filter
            if scope_filter.subject_filter:
                entry_subject = getattr(entry, "subject_category", None)
                if entry_subject and entry_subject != scope_filter.subject_filter:
                    continue

            filtered.append((sim, entry, folder))

        # Sort by similarity descending
        filtered.sort(key=lambda x: x[0], reverse=True)
        top = filtered[:scope_filter.max_results]

        if not top:
            return ContrastResult(similarity=0.0, divergence=0.0)

        best_sim = top[0][0]
        divergence = max(0.0, 1.0 - best_sim)

        references = []
        for sim, entry, folder in top:
            entry_id = ""
            for _id_attr in ("entry_id", "memory_id", "lesson_id", "block_id",
                             "log_id", "question_id", "conclusion_id",
                             "note_id", "map_id"):
                _val = getattr(entry, _id_attr, None)
                if _val is not None:
                    entry_id = _val
                    break
            summary = ""
            for _sum_attr in ("content", "formulation", "summary", "title",
                              "thought_fragment"):
                _sval = getattr(entry, _sum_attr, None)
                if _sval is not None:
                    summary = _sval
                    break
            references.append({
                "entry_id":   entry_id,
                "folder":     folder,
                "similarity": round(sim, 3),
                "source":     "LTMM_SCOPED",
                "query_type": query_type,
                "summary":    summary[:120],
            })

        return ContrastResult(
            similarity=round(best_sim, 3),
            divergence=round(divergence, 3),
            references=references,
            meta={
                "scoped": True,
                "folders_searched": list(scope_filter.folders),
                "hit_count": len(top),
            },
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_query_text(current: Dict[str, Any]) -> str:
        """
        Pull a searchable text string from the `current` dict passed by
        the Logic submodules.  They typically pass output/statements/concepts.
        """
        parts: List[str] = []
        for key in ("output", "statement", "text", "concept", "query", "content"):
            val = current.get(key)
            if isinstance(val, str) and val:
                parts.append(val)
            elif isinstance(val, list):
                parts.extend(str(v) for v in val if v)
        return " ".join(parts) if parts else str(current)
