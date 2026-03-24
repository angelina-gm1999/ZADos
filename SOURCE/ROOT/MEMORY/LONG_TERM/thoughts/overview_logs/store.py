"""
OverviewLogStore — session-level overview summaries.

Each OverviewLogEntry captures a high-level session summary including
mode sequence, dominant emotions, and open threads for continuity.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from zados.memory.long_term.thoughts.types import OverviewLogEntry
from zados.memory.long_term.search_utils import (
    tokenize as _tokenize,
    term_freq as _term_freq,
    cosine as _cosine,
)


class OverviewLogStore:
    """Searchable store for session overview logs."""

    def __init__(self) -> None:
        self._storage: Dict[str, OverviewLogEntry]   = {}
        self._index:   Dict[str, Dict[str, float]]   = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, entry: OverviewLogEntry) -> None:
        self._storage[entry.log_id] = entry
        self._index[entry.log_id] = _term_freq(
            _tokenize(entry.to_search_text())
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        limit: int = 5,
    ) -> List[Tuple[float, OverviewLogEntry]]:
        q_vec = _term_freq(_tokenize(query_text))
        scored: List[Tuple[float, str]] = []

        for lid, t_vec in self._index.items():
            sim = _cosine(q_vec, t_vec)
            if sim > 0.0:
                scored.append((sim, lid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(sim, self._storage[lid]) for sim, lid in scored[:limit]]

    # ------------------------------------------------------------------
    # Targeted retrieval
    # ------------------------------------------------------------------

    def get_by_id(self, log_id: str) -> Optional[OverviewLogEntry]:
        return self._storage.get(log_id)

    def get_all(self) -> List[OverviewLogEntry]:
        return list(self._storage.values())

    def get_by_session(self, session_id: str) -> Optional[OverviewLogEntry]:
        """Find the overview log for a specific session."""
        for entry in self._storage.values():
            if entry.session_id == session_id:
                return entry
        return None

    def __len__(self) -> int:
        return len(self._storage)
