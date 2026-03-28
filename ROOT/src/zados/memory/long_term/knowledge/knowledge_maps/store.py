"""
KnowledgeMapStore — human-readable semantic graphs per subject domain.

No TF-IDF search — uses to_search_text() for basic text matching.
Primary access is by map_id or subject_category.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from zados.memory.long_term.knowledge.types import KnowledgeMap
from zados.memory.long_term.search_utils import (
    tokenize as _tokenize,
    term_freq as _term_freq,
    cosine as _cosine,
)


class KnowledgeMapStore:
    """Store for knowledge maps with basic text search over titles/descriptions."""

    def __init__(self) -> None:
        self._storage: Dict[str, KnowledgeMap]       = {}
        self._index:   Dict[str, Dict[str, float]]   = {}

    def write(self, entry: KnowledgeMap) -> None:
        self._storage[entry.map_id] = entry
        self._index[entry.map_id] = _term_freq(
            _tokenize(entry.to_search_text())
        )

    def search(
        self, query_text: str, limit: int = 5,
    ) -> List[Tuple[float, KnowledgeMap]]:
        q_vec = _term_freq(_tokenize(query_text))
        scored: List[Tuple[float, str]] = []
        for mid, t_vec in self._index.items():
            sim = _cosine(q_vec, t_vec)
            if sim > 0.0:
                scored.append((sim, mid))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(sim, self._storage[mid]) for sim, mid in scored[:limit]]

    def get_by_id(self, map_id: str) -> Optional[KnowledgeMap]:
        return self._storage.get(map_id)

    def get_all(self) -> List[KnowledgeMap]:
        return list(self._storage.values())

    def get_by_subject(self, subject_category: str) -> List[KnowledgeMap]:
        return [
            m for m in self._storage.values()
            if m.subject_category == subject_category
        ]

    def __len__(self) -> int:
        return len(self._storage)
