"""
LessonStore — validated academic insights from learning modes.

Supports validation lifecycle (pending → validated / contradicted)
and reinforcement (repeated supporting evidence bumps confidence).
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from zados.memory.long_term.knowledge.types import LessonEntry
from zados.memory.long_term.search_utils import (
    tokenize as _tokenize,
    term_freq as _term_freq,
    cosine as _cosine,
)


class LessonStore:
    """TF-IDF searchable store for validated lessons."""

    def __init__(self) -> None:
        self._storage: Dict[str, LessonEntry]        = {}
        self._index:   Dict[str, Dict[str, float]]   = {}

    def write(self, entry: LessonEntry) -> None:
        self._storage[entry.lesson_id] = entry
        self._index[entry.lesson_id] = _term_freq(
            _tokenize(entry.to_search_text())
        )

    def search(
        self, query_text: str, limit: int = 5,
    ) -> List[Tuple[float, LessonEntry]]:
        q_vec = _term_freq(_tokenize(query_text))
        scored: List[Tuple[float, str]] = []
        for lid, t_vec in self._index.items():
            sim = _cosine(q_vec, t_vec)
            if sim > 0.0:
                scored.append((sim, lid))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(sim, self._storage[lid]) for sim, lid in scored[:limit]]

    def get_by_id(self, lesson_id: str) -> Optional[LessonEntry]:
        return self._storage.get(lesson_id)

    def get_all(self) -> List[LessonEntry]:
        return list(self._storage.values())

    def get_validated(self) -> List[LessonEntry]:
        return [e for e in self._storage.values() if e.validation_status == "validated"]

    def validate(self, lesson_id: str) -> bool:
        entry = self._storage.get(lesson_id)
        if entry is None:
            return False
        entry.validation_status = "validated"
        return True

    def contradict(self, lesson_id: str) -> bool:
        entry = self._storage.get(lesson_id)
        if entry is None:
            return False
        entry.validation_status = "contradicted"
        return True

    def reinforce(self, lesson_id: str) -> bool:
        entry = self._storage.get(lesson_id)
        if entry is None:
            return False
        entry.reinforcement_count += 1
        entry.last_reinforced = datetime.utcnow()
        return True

    def __len__(self) -> int:
        return len(self._storage)
