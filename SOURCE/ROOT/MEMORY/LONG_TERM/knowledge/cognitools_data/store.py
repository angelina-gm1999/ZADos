"""
CognitoolsDataStore — per-engine persistent state.

Simple key-value store keyed by engine_id.  No search capability.
Each engine can persist arbitrary dicts of state data here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class CognitoolsDataStore:
    """KV store for cognitool engine persistent data. No search."""

    def __init__(self) -> None:
        self._storage: Dict[str, Dict[str, Any]] = {}

    def write(self, engine_id: str, data: Dict[str, Any]) -> None:
        """Store or overwrite data for an engine."""
        self._storage[engine_id] = data

    def get_by_id(self, engine_id: str) -> Optional[Dict[str, Any]]:
        return self._storage.get(engine_id)

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self._storage.values())

    def get_all_engine_ids(self) -> List[str]:
        return list(self._storage.keys())

    def __len__(self) -> int:
        return len(self._storage)
