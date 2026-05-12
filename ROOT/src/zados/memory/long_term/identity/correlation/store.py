"""
IdentityCorrelationStore — maps relations between fixed and developmental identity.

ZADOS cannot modify or delete hardcoded (fixed) identity entries.
Instead, it creates correlations that describe how developmental identity
elements (conclusions, core memories, journal entries) relate to the
immutable hardcoded foundations.

This store is the bridge between the static identity floor and the
evolving identity ceiling — enabling self-reflection, journaling, and
memory contrasts that reference hardcoded values without modifying them.

Supported operations:
  - write / remove correlations (developmental side is mutable)
  - validate (bump validation_count + timestamp on re-confirmation)
  - query by hardcoded entry, developmental entry, relation type, or text search
  - get_web() for full correlation graph snapshot
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from zados.memory.long_term.identity.types import (
    IdentityCorrelation,
    CorrelationRelationType,
)
from zados.memory.long_term.search_utils import (
    tokenize as _tokenize,
    term_freq as _term_freq,
    cosine as _cosine,
)


class IdentityCorrelationStore:
    """
    Searchable store for identity correlations.

    Correlations link hardcoded_entry_id → developmental_id with a
    typed relation.  The hardcoded side is read-only; only the
    correlation records themselves can be created, updated, or removed.
    """

    def __init__(self) -> None:
        self._storage: Dict[str, IdentityCorrelation] = {}
        self._index: Dict[str, Dict[str, float]] = {}

        # Secondary indexes for fast lookup
        self._by_hardcoded: Dict[str, List[str]] = {}   # hc_id → [corr_ids]
        self._by_developmental: Dict[str, List[str]] = {}  # dev_id → [corr_ids]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, correlation: IdentityCorrelation) -> None:
        """Store or overwrite a correlation and update indexes."""
        cid = correlation.correlation_id
        old = self._storage.get(cid)
        if old is not None:
            self._remove_secondary(old)

        self._storage[cid] = correlation
        self._index[cid] = _term_freq(
            _tokenize(correlation.to_search_text())
        )
        self._add_secondary(correlation)

    def remove(self, correlation_id: str) -> bool:
        """Remove a correlation. Returns True if found."""
        corr = self._storage.pop(correlation_id, None)
        if corr is None:
            return False
        self._index.pop(correlation_id, None)
        self._remove_secondary(corr)
        return True

    def validate(self, correlation_id: str) -> bool:
        """Re-confirm a correlation: bump validation count + timestamp."""
        corr = self._storage.get(correlation_id)
        if corr is None:
            return False
        corr.validation_count += 1
        corr.last_validated = datetime.utcnow()
        return True

    def update_confidence(self, correlation_id: str, new_confidence: float) -> bool:
        """Update confidence score on an existing correlation."""
        corr = self._storage.get(correlation_id)
        if corr is None:
            return False
        corr.confidence = max(0.0, min(1.0, new_confidence))
        return True

    # ------------------------------------------------------------------
    # Query: by hardcoded entry
    # ------------------------------------------------------------------

    def get_by_hardcoded(self, hardcoded_entry_id: str) -> List[IdentityCorrelation]:
        """All correlations linked to a specific hardcoded entry."""
        cids = self._by_hardcoded.get(hardcoded_entry_id, [])
        return [self._storage[cid] for cid in cids if cid in self._storage]

    # ------------------------------------------------------------------
    # Query: by developmental entry
    # ------------------------------------------------------------------

    def get_by_developmental(self, developmental_id: str) -> List[IdentityCorrelation]:
        """All correlations linked to a specific developmental entry."""
        cids = self._by_developmental.get(developmental_id, [])
        return [self._storage[cid] for cid in cids if cid in self._storage]

    # ------------------------------------------------------------------
    # Query: by relation type
    # ------------------------------------------------------------------

    def get_by_relation(self, relation_type: str) -> List[IdentityCorrelation]:
        """All correlations of a given relation type."""
        return [
            c for c in self._storage.values()
            if c.relation_type == relation_type
        ]

    # ------------------------------------------------------------------
    # Query: text search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        limit: int = 5,
    ) -> List[Tuple[float, IdentityCorrelation]]:
        """TF-IDF similarity search over correlation descriptions."""
        q_vec = _term_freq(_tokenize(query_text))
        scored: List[Tuple[float, str]] = []
        for cid, t_vec in self._index.items():
            sim = _cosine(q_vec, t_vec)
            if sim > 0.0:
                scored.append((sim, cid))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(sim, self._storage[cid]) for sim, cid in scored[:limit]]

    # ------------------------------------------------------------------
    # Query: full graph snapshot
    # ------------------------------------------------------------------

    def get_web(self) -> Dict[str, Any]:
        """Return full correlation web as structured dict for introspection.

        Returns
        -------
        dict with keys:
            correlations : list of serialised correlations
            hardcoded_fanout : dict mapping hc_id → count of correlations
            relation_distribution : dict mapping relation_type → count
            total : int
        """
        relation_counts: Dict[str, int] = {}
        hc_counts: Dict[str, int] = {}
        serialised: List[Dict[str, Any]] = []

        for corr in self._storage.values():
            serialised.append({
                "correlation_id": corr.correlation_id,
                "hardcoded_entry_id": corr.hardcoded_entry_id,
                "developmental_id": corr.developmental_id,
                "developmental_type": corr.developmental_type,
                "relation_type": corr.relation_type,
                "description": corr.description,
                "confidence": corr.confidence,
                "validation_count": corr.validation_count,
            })
            relation_counts[corr.relation_type] = (
                relation_counts.get(corr.relation_type, 0) + 1
            )
            hc_counts[corr.hardcoded_entry_id] = (
                hc_counts.get(corr.hardcoded_entry_id, 0) + 1
            )

        return {
            "correlations": serialised,
            "hardcoded_fanout": hc_counts,
            "relation_distribution": relation_counts,
            "total": len(serialised),
        }

    # ------------------------------------------------------------------
    # Targeted retrieval
    # ------------------------------------------------------------------

    def get_by_id(self, correlation_id: str) -> Optional[IdentityCorrelation]:
        return self._storage.get(correlation_id)

    def get_all(self) -> List[IdentityCorrelation]:
        return list(self._storage.values())

    def get_tensions(self) -> List[IdentityCorrelation]:
        """Return all correlations with relation_type == 'tensions_with'."""
        return self.get_by_relation(CorrelationRelationType.TENSIONS)

    def __len__(self) -> int:
        return len(self._storage)

    # ------------------------------------------------------------------
    # Secondary index helpers
    # ------------------------------------------------------------------

    def _add_secondary(self, corr: IdentityCorrelation) -> None:
        cid = corr.correlation_id
        self._by_hardcoded.setdefault(corr.hardcoded_entry_id, []).append(cid)
        self._by_developmental.setdefault(corr.developmental_id, []).append(cid)

    def _remove_secondary(self, corr: IdentityCorrelation) -> None:
        cid = corr.correlation_id
        for hc_list in self._by_hardcoded.values():
            if cid in hc_list:
                hc_list.remove(cid)
                break
        for dev_list in self._by_developmental.values():
            if cid in dev_list:
                dev_list.remove(cid)
                break
