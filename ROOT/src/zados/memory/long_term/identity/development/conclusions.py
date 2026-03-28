"""
IdentityConclusionStore — conclusions drawn from identity self-reflection.

Conclusions represent stable identity beliefs derived from repeated
experiences.  They can be reinforced (increasing confidence) when
supporting evidence is encountered.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from zados.memory.long_term.identity.types import IdentityConclusion
from zados.memory.long_term.search_utils import (
    tokenize as _tokenize,
    term_freq as _term_freq,
    cosine as _cosine,
)


class IdentityConclusionStore:
    """
    Searchable store for identity conclusions.

    Supports reinforcement: repeated supporting evidence bumps confidence.
    """

    def __init__(self) -> None:
        self._storage: Dict[str, IdentityConclusion]  = {}
        self._index:   Dict[str, Dict[str, float]]    = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, entry: IdentityConclusion) -> None:
        """Store or overwrite a conclusion and (re)index it."""
        self._storage[entry.conclusion_id] = entry
        self._index[entry.conclusion_id] = _term_freq(
            _tokenize(self._entry_text(entry))
        )

    def reinforce(self, conclusion_id: str) -> bool:
        """
        Increment reinforcement count and update last_reinforced timestamp.

        Returns True if found, False otherwise.
        """
        entry = self._storage.get(conclusion_id)
        if entry is None:
            return False
        entry.reinforcement_count += 1
        entry.last_reinforced = datetime.utcnow()
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        limit: int = 5,
    ) -> List[Tuple[float, IdentityConclusion]]:
        """Semantic search over conclusions."""
        q_vec = _term_freq(_tokenize(query_text))
        scored: List[Tuple[float, str]] = []

        for cid, t_vec in self._index.items():
            sim = _cosine(q_vec, t_vec)
            if sim > 0.0:
                scored.append((sim, cid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(sim, self._storage[cid]) for sim, cid in scored[:limit]]

    # ------------------------------------------------------------------
    # Targeted retrieval
    # ------------------------------------------------------------------

    def get_by_id(self, conclusion_id: str) -> Optional[IdentityConclusion]:
        return self._storage.get(conclusion_id)

    def get_all(self) -> List[IdentityConclusion]:
        return list(self._storage.values())

    def get_by_type(self, conclusion_type: str) -> List[IdentityConclusion]:
        return [
            c for c in self._storage.values()
            if c.conclusion_type == conclusion_type
        ]

    def __len__(self) -> int:
        return len(self._storage)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _entry_text(c: IdentityConclusion) -> str:
        parts = [c.content, c.conclusion_type]
        parts.extend(c.tags)
        return " ".join(parts)
