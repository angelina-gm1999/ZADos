"""
IdentityJournalStore — identity-specific reflective journaling.

Distinct from the general cognitive JournalStore: entries here focus on
identity reflection, self-commentary, and identity-related observations.
Supports threaded replies via parent_entry_id.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from zados.memory.long_term.identity.types import (
    IdentityJournalEntry,
    IdentityJournalEntryType,
)
from zados.memory.long_term.search_utils import (
    tokenize as _tokenize,
    term_freq as _term_freq,
    cosine as _cosine,
)


class IdentityJournalStore:
    """
    Searchable store for identity journal entries.

    Supports filtering by entry type and retrieval of reply threads.
    """

    def __init__(self) -> None:
        self._storage: Dict[str, IdentityJournalEntry]  = {}
        self._index:   Dict[str, Dict[str, float]]      = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, entry: IdentityJournalEntry) -> None:
        """Store or overwrite an entry and (re)index it."""
        self._storage[entry.entry_id] = entry
        self._index[entry.entry_id] = _term_freq(
            _tokenize(entry.to_search_text())
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        limit: int = 5,
        entry_type_filter: Optional[IdentityJournalEntryType] = None,
    ) -> List[Tuple[float, IdentityJournalEntry]]:
        """Semantic search over identity journal entries."""
        q_vec = _term_freq(_tokenize(query_text))
        scored: List[Tuple[float, str]] = []

        for eid, t_vec in self._index.items():
            entry = self._storage[eid]
            if entry_type_filter and entry.entry_type != entry_type_filter:
                continue
            sim = _cosine(q_vec, t_vec)
            if sim > 0.0:
                scored.append((sim, eid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(sim, self._storage[eid]) for sim, eid in scored[:limit]]

    # ------------------------------------------------------------------
    # Targeted retrieval
    # ------------------------------------------------------------------

    def get_by_id(self, entry_id: str) -> Optional[IdentityJournalEntry]:
        return self._storage.get(entry_id)

    def get_all(self) -> List[IdentityJournalEntry]:
        return list(self._storage.values())

    def get_by_type(
        self, entry_type: IdentityJournalEntryType,
    ) -> List[IdentityJournalEntry]:
        return [
            e for e in self._storage.values()
            if e.entry_type == entry_type
        ]

    def get_replies(self, parent_entry_id: str) -> List[IdentityJournalEntry]:
        """All entries that are direct replies to the given entry."""
        return [
            e for e in self._storage.values()
            if e.parent_entry_id == parent_entry_id
        ]

    def __len__(self) -> int:
        return len(self._storage)
