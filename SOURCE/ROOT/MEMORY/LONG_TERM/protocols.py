"""
Structural protocols for LTMM stores.

All new namespaced stores and existing LTMMStore / JournalStore satisfy
``SearchableStore`` without explicit inheritance (duck-typed via Protocol).
"""
from __future__ import annotations

from typing import Any, List, Optional, Protocol, Tuple, runtime_checkable


@runtime_checkable
class SearchableStore(Protocol):
    """Store that supports TF-IDF semantic search."""

    def write(self, entry: Any) -> None: ...
    def search(self, query_text: str, limit: int = 5, **kwargs: Any) -> List[Tuple[float, Any]]: ...
    def get_by_id(self, entry_id: str) -> Optional[Any]: ...
    def get_all(self) -> List[Any]: ...


@runtime_checkable
class ReadOnlyStore(Protocol):
    """Store that is read-only after initialisation (e.g. hardcoded identity)."""

    def get_by_id(self, entry_id: str) -> Optional[Any]: ...
    def get_all(self) -> List[Any]: ...
