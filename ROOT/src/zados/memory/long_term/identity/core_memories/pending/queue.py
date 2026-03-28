"""
PendingUpdateQueue — holds proposed core-memory updates awaiting peer review.

A PendingUpdate is submitted when an engine or pipeline wants to modify a
core memory.  The update remains in "pending" status until the M2 (peer-review)
pipeline approves or rejects it.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from zados.memory.long_term.identity.types import PendingUpdate


class PendingUpdateQueue:
    """
    FIFO queue for pending core-memory updates.

    Keyed by update_id for O(1) lookup.
    """

    def __init__(self) -> None:
        self._queue: Dict[str, PendingUpdate] = {}

    def submit(self, update: PendingUpdate) -> None:
        """Add a new pending update to the queue."""
        self._queue[update.update_id] = update

    def approve(self, update_id: str, peer_review_ref: str = "") -> Optional[PendingUpdate]:
        """
        Mark an update as approved and return it.

        Returns None if the update_id is not found or already resolved.
        """
        upd = self._queue.get(update_id)
        if upd is None or upd.status != "pending":
            return None
        upd.status = "approved"
        upd.peer_review_ref = peer_review_ref
        return upd

    def reject(self, update_id: str, peer_review_ref: str = "") -> Optional[PendingUpdate]:
        """
        Mark an update as rejected and return it.

        Returns None if the update_id is not found or already resolved.
        """
        upd = self._queue.get(update_id)
        if upd is None or upd.status != "pending":
            return None
        upd.status = "rejected"
        upd.peer_review_ref = peer_review_ref
        return upd

    def get_pending(self) -> List[PendingUpdate]:
        """All updates still awaiting review."""
        return [u for u in self._queue.values() if u.status == "pending"]

    def get_by_id(self, update_id: str) -> Optional[PendingUpdate]:
        return self._queue.get(update_id)

    def get_all(self) -> List[PendingUpdate]:
        return list(self._queue.values())

    def __len__(self) -> int:
        return len(self._queue)
