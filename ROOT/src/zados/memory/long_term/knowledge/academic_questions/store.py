"""
AcademicQuestionStore — domain-specific knowledge gap questions.

Mirrors GeneralQuestionStore but for academic/subject-specific questions
with additional subject_category and domain fields.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from zados.memory.long_term.knowledge.types import AcademicQuestion
from zados.memory.long_term.search_utils import (
    tokenize as _tokenize,
    term_freq as _term_freq,
    cosine as _cosine,
)


class AcademicQuestionStore:
    """TF-IDF searchable store for academic questions."""

    def __init__(self) -> None:
        self._storage: Dict[str, AcademicQuestion]   = {}
        self._index:   Dict[str, Dict[str, float]]   = {}

    def write(self, entry: AcademicQuestion) -> None:
        self._storage[entry.question_id] = entry
        self._index[entry.question_id] = _term_freq(
            _tokenize(entry.to_search_text())
        )

    def search(
        self, query_text: str, limit: int = 5,
    ) -> List[Tuple[float, AcademicQuestion]]:
        q_vec = _term_freq(_tokenize(query_text))
        scored: List[Tuple[float, str]] = []
        for qid, t_vec in self._index.items():
            sim = _cosine(q_vec, t_vec)
            if sim > 0.0:
                scored.append((sim, qid))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(sim, self._storage[qid]) for sim, qid in scored[:limit]]

    def get_unresolved(self) -> List[AcademicQuestion]:
        return [q for q in self._storage.values() if not q.resolved]

    def resolve(self, question_id: str, resolution_note: str = "") -> bool:
        entry = self._storage.get(question_id)
        if entry is None:
            return False
        entry.resolved = True
        entry.resolution_note = resolution_note
        entry.last_checked = datetime.utcnow()
        return True

    def tick_stagnation(self, question_id: str) -> bool:
        entry = self._storage.get(question_id)
        if entry is None:
            return False
        entry.stagnation_count += 1
        entry.last_checked = datetime.utcnow()
        return True

    def get_by_id(self, question_id: str) -> Optional[AcademicQuestion]:
        return self._storage.get(question_id)

    def get_all(self) -> List[AcademicQuestion]:
        return list(self._storage.values())

    def __len__(self) -> int:
        return len(self._storage)
