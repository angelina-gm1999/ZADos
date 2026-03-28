"""
NotebookStore — academic journaling about knowledge-domain learning.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from zados.memory.long_term.knowledge.types import NotebookEntry
from zados.memory.long_term.search_utils import (
    tokenize as _tokenize,
    term_freq as _term_freq,
    cosine as _cosine,
)


class NotebookStore:
    """TF-IDF searchable store for academic notebook entries."""

    def __init__(self) -> None:
        self._storage: Dict[str, NotebookEntry]      = {}
        self._index:   Dict[str, Dict[str, float]]   = {}

    def write(self, entry: NotebookEntry) -> None:
        self._storage[entry.note_id] = entry
        self._index[entry.note_id] = _term_freq(
            _tokenize(entry.to_search_text())
        )

    def search(
        self, query_text: str, limit: int = 5,
    ) -> List[Tuple[float, NotebookEntry]]:
        q_vec = _term_freq(_tokenize(query_text))
        scored: List[Tuple[float, str]] = []
        for nid, t_vec in self._index.items():
            sim = _cosine(q_vec, t_vec)
            if sim > 0.0:
                scored.append((sim, nid))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(sim, self._storage[nid]) for sim, nid in scored[:limit]]

    def get_by_id(self, note_id: str) -> Optional[NotebookEntry]:
        return self._storage.get(note_id)

    def get_all(self) -> List[NotebookEntry]:
        return list(self._storage.values())

    def get_by_subject(self, subject_category: str) -> List[NotebookEntry]:
        return [
            n for n in self._storage.values()
            if n.subject_category == subject_category
        ]

    def __len__(self) -> int:
        return len(self._storage)
