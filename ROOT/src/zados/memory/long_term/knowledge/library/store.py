"""
LibraryStore — ingested reference material (books, articles, documents).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from zados.memory.long_term.knowledge.types import LibraryEntry
from zados.memory.long_term.search_utils import (
    tokenize as _tokenize,
    term_freq as _term_freq,
    cosine as _cosine,
)


class LibraryStore:
    """TF-IDF searchable store for ingested reference material."""

    def __init__(self) -> None:
        self._storage: Dict[str, LibraryEntry]       = {}
        self._index:   Dict[str, Dict[str, float]]   = {}

    def write(self, entry: LibraryEntry) -> None:
        self._storage[entry.entry_id] = entry
        self._index[entry.entry_id] = _term_freq(
            _tokenize(entry.to_search_text())
        )

    def search(
        self, query_text: str, limit: int = 5,
    ) -> List[Tuple[float, LibraryEntry]]:
        q_vec = _term_freq(_tokenize(query_text))
        scored: List[Tuple[float, str]] = []
        for eid, t_vec in self._index.items():
            sim = _cosine(q_vec, t_vec)
            if sim > 0.0:
                scored.append((sim, eid))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(sim, self._storage[eid]) for sim, eid in scored[:limit]]

    def get_by_id(self, entry_id: str) -> Optional[LibraryEntry]:
        return self._storage.get(entry_id)

    def get_all(self) -> List[LibraryEntry]:
        return list(self._storage.values())

    def get_by_domain(self, domain: str) -> List[LibraryEntry]:
        return [e for e in self._storage.values() if e.domain == domain]

    def __len__(self) -> int:
        return len(self._storage)

    def ingest(
        self,
        title: str,
        content: str,
        source_type: str = "document",
        domain: str = "",
        tags: Optional[List[str]] = None,
        nt_snapshot: Optional[Dict[str, float]] = None,
    ) -> LibraryEntry:
        """Convenience: create and store a LibraryEntry from raw parameters.

        Parameters
        ----------
        title : str
        content : str
        source_type : str
            "book" | "article" | "document" | "upload"
        domain : str
        tags : list, optional
        nt_snapshot : dict, optional

        Returns
        -------
        LibraryEntry
        """
        entry = LibraryEntry(
            title=title,
            content=content,
            source_type=source_type,
            domain=domain,
            tags=tags or [],
            nt_snapshot=nt_snapshot or {},
        )
        self.write(entry)
        return entry
