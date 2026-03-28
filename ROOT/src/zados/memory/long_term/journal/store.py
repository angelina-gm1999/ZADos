"""
JournalStore — persistent storage for JournalEntry objects.

Architecture mirrors LTMMStore: in-memory dict with term-vector semantic
index, backend-agnostic interface.  Phase 3: swap _storage for SQLite /
embedded vector DB without changing callers.

Entries are indexed by entry_id and searchable by:
  - semantic similarity (prose + reflection_prompts + tags)
  - trigger type
  - review status
  - turn range overlap
  - linked entry graph traversal
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from zados.memory.long_term.journal.entry import JournalEntry, JournalTrigger, ReviewStatus
from zados.memory.long_term.search_utils import tokenize as _tokenize, term_freq as _term_freq, cosine as _cosine


# ---------------------------------------------------------------------------
# JournalStore
# ---------------------------------------------------------------------------

class JournalStore:
    """
    Stores and retrieves JournalEntry objects.

    All writes are idempotent on entry_id.  Updates replace the existing
    entry and rebuild its index vector.
    """

    def __init__(self) -> None:
        self._storage: Dict[str, JournalEntry]        = {}
        self._index:   Dict[str, Dict[str, float]]    = {}

    # ------------------------------------------------------------------
    # Write / update
    # ------------------------------------------------------------------

    def write(self, entry: JournalEntry) -> None:
        """Store or overwrite an entry and (re)index it."""
        self._storage[entry.entry_id] = entry
        self._index[entry.entry_id]   = _term_freq(_tokenize(entry.to_search_text()))

    def update_review_status(self, entry_id: str, status: ReviewStatus) -> None:
        if entry_id in self._storage:
            self._storage[entry_id].review_status = status

    def link_entries(self, entry_id_a: str, entry_id_b: str) -> None:
        """Bidirectional link between two entries."""
        if entry_id_a in self._storage:
            self._storage[entry_id_a].link(entry_id_b)
        if entry_id_b in self._storage:
            self._storage[entry_id_b].link(entry_id_a)

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        limit: int = 5,
        trigger_filter: Optional[JournalTrigger] = None,
        status_filter: Optional[ReviewStatus] = None,
    ) -> List[Tuple[float, JournalEntry]]:
        """
        Semantic search over prose + prompts + tags.

        Parameters
        ----------
        query_text     : free-text query
        limit          : max results
        trigger_filter : if set, only return entries with this trigger
        status_filter  : if set, only return entries with this review_status
        """
        q_vec = _term_freq(_tokenize(query_text))
        scored: List[Tuple[float, str]] = []

        for eid, t_vec in self._index.items():
            entry = self._storage[eid]
            if trigger_filter and entry.trigger != trigger_filter:
                continue
            if status_filter and entry.review_status != status_filter:
                continue
            sim = _cosine(q_vec, t_vec)
            if sim > 0.0:
                scored.append((sim, eid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(sim, self._storage[eid]) for sim, eid in scored[:limit]]

    # ------------------------------------------------------------------
    # Targeted retrieval
    # ------------------------------------------------------------------

    def get_by_id(self, entry_id: str) -> Optional[JournalEntry]:
        return self._storage.get(entry_id)

    def get_unreviewed(self) -> List[JournalEntry]:
        """All entries with reflection prompts not yet reviewed."""
        return [
            e for e in self._storage.values()
            if e.review_status == ReviewStatus.UNREVIEWED
            and e.reflection_prompts
        ]

    def get_by_trigger(self, trigger: JournalTrigger) -> List[JournalEntry]:
        return [e for e in self._storage.values() if e.trigger == trigger]

    def get_recent(self, n: int = 10) -> List[JournalEntry]:
        """Most recent n entries by timestamp."""
        return sorted(
            self._storage.values(),
            key=lambda e: e.timestamp,
            reverse=True,
        )[:n]

    def get_linked(self, entry_id: str) -> List[JournalEntry]:
        """All entries directly linked to the given entry."""
        entry = self._storage.get(entry_id)
        if not entry:
            return []
        return [
            self._storage[eid]
            for eid in entry.linked_entry_ids
            if eid in self._storage
        ]

    def get_all_patterns(self) -> List[str]:
        """
        Aggregate all identified_patterns across all entries.
        Used by E20 to build its cross-session comparison template library.
        """
        patterns: List[str] = []
        for entry in self._storage.values():
            patterns.extend(entry.annotations.identified_patterns)
        return patterns

    def get_all_tags(self) -> List[str]:
        """Flat list of all tags across all entries (for tag-based retrieval)."""
        tags: List[str] = []
        for entry in self._storage.values():
            tags.extend(entry.tags)
        return tags

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._storage)

    def stats(self) -> dict:
        entries = list(self._storage.values())
        return {
            "total":       len(entries),
            "unreviewed":  sum(1 for e in entries if e.review_status == ReviewStatus.UNREVIEWED),
            "in_review":   sum(1 for e in entries if e.review_status == ReviewStatus.IN_REVIEW),
            "resolved":    sum(1 for e in entries if e.review_status == ReviewStatus.RESOLVED),
            "by_trigger":  {t.value: sum(1 for e in entries if e.trigger == t) for t in JournalTrigger},
        }
