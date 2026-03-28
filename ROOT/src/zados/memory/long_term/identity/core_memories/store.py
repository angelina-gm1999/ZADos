"""
CoreMemoryStore — peer-review-gated core identity beliefs.

Core memories are never deleted.  Updates go through a PendingUpdateQueue
and require peer-review approval (M2 pipeline) before being applied.
Each update appends an UpdateRecord to the memory's history.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from zados.memory.long_term.identity.types import CoreMemory, UpdateRecord
from zados.memory.long_term.search_utils import (
    tokenize as _tokenize,
    term_freq as _term_freq,
    cosine as _cosine,
)


class CoreMemoryStore:
    """
    Searchable store for core identity beliefs.

    All writes are idempotent on memory_id.
    """

    def __init__(self) -> None:
        self._storage: Dict[str, CoreMemory]        = {}
        self._index:   Dict[str, Dict[str, float]]  = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, entry: CoreMemory) -> None:
        """Store or overwrite a core memory and (re)index it."""
        self._storage[entry.memory_id] = entry
        self._index[entry.memory_id] = _term_freq(
            _tokenize(self._entry_text(entry))
        )

    def apply_update(
        self,
        memory_id: str,
        new_content: str,
        peer_review_ref: str = "",
    ) -> bool:
        """
        Apply an approved update to an existing core memory.

        Returns True if the update was applied, False if the memory_id
        was not found.
        """
        entry = self._storage.get(memory_id)
        if entry is None:
            return False

        record = UpdateRecord(
            previous_content=entry.content,
            updated_at=datetime.utcnow(),
            peer_review_ref=peer_review_ref,
        )
        entry.update_history.append(record)
        entry.content = new_content
        entry.version += 1
        entry.updated_at = datetime.utcnow()

        # Re-index
        self._index[memory_id] = _term_freq(
            _tokenize(self._entry_text(entry))
        )
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        limit: int = 5,
    ) -> List[Tuple[float, CoreMemory]]:
        """Semantic search over core memories."""
        q_vec = _term_freq(_tokenize(query_text))
        scored: List[Tuple[float, str]] = []

        for mid, t_vec in self._index.items():
            sim = _cosine(q_vec, t_vec)
            if sim > 0.0:
                scored.append((sim, mid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(sim, self._storage[mid]) for sim, mid in scored[:limit]]

    # ------------------------------------------------------------------
    # Targeted retrieval
    # ------------------------------------------------------------------

    def get_by_id(self, memory_id: str) -> Optional[CoreMemory]:
        return self._storage.get(memory_id)

    def get_all(self) -> List[CoreMemory]:
        return list(self._storage.values())

    def get_by_type(self, memory_type: str) -> List[CoreMemory]:
        """Return all core memories with a given memory_type."""
        return [
            m for m in self._storage.values()
            if m.memory_type == memory_type
        ]

    def __len__(self) -> int:
        return len(self._storage)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_text(m: CoreMemory) -> str:
        parts = [m.content, m.memory_type]
        parts.extend(m.tags)
        return " ".join(parts)
