"""
HeldThinkingBlockStore — emotion-interrupted thinking fragments.

When a thinking phase is interrupted by an emotion spike (threshold > 0.6),
the current thought fragment is captured and stored here for later review.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from zados.memory.long_term.thoughts.types import HeldThinkingBlock
from zados.memory.long_term.search_utils import (
    tokenize as _tokenize,
    term_freq as _term_freq,
    cosine as _cosine,
)


class HeldThinkingBlockStore:
    """Searchable store for held thinking blocks with review tracking."""

    def __init__(self) -> None:
        self._storage: Dict[str, HeldThinkingBlock]  = {}
        self._index:   Dict[str, Dict[str, float]]   = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, entry: HeldThinkingBlock) -> None:
        self._storage[entry.block_id] = entry
        self._index[entry.block_id] = _term_freq(
            _tokenize(entry.to_search_text())
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        limit: int = 5,
    ) -> List[Tuple[float, HeldThinkingBlock]]:
        q_vec = _term_freq(_tokenize(query_text))
        scored: List[Tuple[float, str]] = []

        for bid, t_vec in self._index.items():
            sim = _cosine(q_vec, t_vec)
            if sim > 0.0:
                scored.append((sim, bid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [(sim, self._storage[bid]) for sim, bid in scored[:limit]]

    # ------------------------------------------------------------------
    # Review tracking
    # ------------------------------------------------------------------

    def get_unreviewed(self) -> List[HeldThinkingBlock]:
        """All blocks not yet reviewed by a reflective pipeline."""
        return [b for b in self._storage.values() if not b.reviewed]

    def mark_reviewed(self, block_id: str) -> bool:
        """Mark a block as reviewed. Returns True if found."""
        entry = self._storage.get(block_id)
        if entry is None:
            return False
        entry.reviewed = True
        return True

    # ------------------------------------------------------------------
    # Targeted retrieval
    # ------------------------------------------------------------------

    def get_by_id(self, block_id: str) -> Optional[HeldThinkingBlock]:
        return self._storage.get(block_id)

    def get_all(self) -> List[HeldThinkingBlock]:
        return list(self._storage.values())

    def __len__(self) -> int:
        return len(self._storage)
