"""
ZA-DOS v0.6 — Context Anchor Manager (spec §2.5).

Manages context anchors for learning sessions.  An anchor is a snapshot
of the topic / intent at the start of a learning segment.  Drift
detection compares current input against the anchor to decide whether
the conversation has drifted off-topic (requiring re-anchoring or
E23 re-run).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from zados.core.types import ContextAnchor

log = logging.getLogger(__name__)

# Drift threshold — divergence above this triggers re-anchoring
DRIFT_THRESHOLD = 0.5


class ContextAnchorManager:
    """Create, query, and deactivate context anchors.

    Parameters
    ----------
    memory_contrast : MemoryContrast, optional
        If provided, drift detection uses MemoryContrast.contrast()
        for semantic comparison.  Otherwise falls back to simple
        keyword-overlap heuristic.
    """

    def __init__(self, memory_contrast: Any = None) -> None:
        self._contrast = memory_contrast
        self._anchors: List[ContextAnchor] = []
        self._active: Optional[ContextAnchor] = None

    @property
    def active_anchor(self) -> Optional[ContextAnchor]:
        """Return the currently active anchor, or None."""
        return self._active

    def create_anchor(
        self,
        raw_text: str,
        subject_hint: str = "",
        intent_prior: str = "",
    ) -> ContextAnchor:
        """Create a new context anchor and make it active.

        Deactivates any previous anchor.

        Parameters
        ----------
        raw_text : str
            The input text at anchor time.
        subject_hint : str
            SubjectCategory value hint.
        intent_prior : str
            Dominant intent at anchor time (from E23).

        Returns
        -------
        ContextAnchor
        """
        # Deactivate previous
        if self._active is not None:
            self._active.active = False

        # Build drift reference (simple word-set fingerprint)
        words = set(raw_text.lower().split())
        drift_ref = {w: 1.0 for w in words if len(w) > 2}

        anchor = ContextAnchor(
            raw_text=raw_text,
            subject_hint=subject_hint,
            intent_prior=intent_prior,
            drift_reference=drift_ref,
            timestamp=time.time(),
            active=True,
        )
        self._anchors.append(anchor)
        self._active = anchor
        log.debug("Created context anchor: subject=%s, intent=%s", subject_hint, intent_prior)
        return anchor

    def deactivate(self) -> None:
        """Deactivate the current anchor."""
        if self._active is not None:
            self._active.active = False
            self._active = None

    def check_drift(self, current_text: str) -> float:
        """Check how much the current input has drifted from the anchor.

        Parameters
        ----------
        current_text : str
            Current user input.

        Returns
        -------
        float
            Divergence score in [0.0, 1.0].  >DRIFT_THRESHOLD means
            significant drift.
        """
        if self._active is None:
            return 0.0

        # Try MemoryContrast first
        if self._contrast is not None:
            try:
                result = self._contrast.contrast(
                    current={"text": current_text, "content": current_text},
                    query_type="context",
                )
                divergence = getattr(result, "divergence", 0.0)
                return float(divergence)
            except Exception:
                log.debug("MemoryContrast drift check failed, using keyword fallback.")

        # Fallback: Jaccard distance on word sets
        anchor_words = set(self._active.drift_reference.keys())
        current_words = {w for w in current_text.lower().split() if len(w) > 2}

        if not anchor_words or not current_words:
            return 0.0

        intersection = anchor_words & current_words
        union = anchor_words | current_words
        jaccard = len(intersection) / len(union) if union else 0.0

        # Divergence = 1 - similarity
        return 1.0 - jaccard

    def has_drifted(self, current_text: str) -> bool:
        """Check if drift exceeds the threshold.

        Parameters
        ----------
        current_text : str

        Returns
        -------
        bool
        """
        return self.check_drift(current_text) > DRIFT_THRESHOLD

    @property
    def anchor_history(self) -> List[ContextAnchor]:
        """Return all anchors created in this session."""
        return list(self._anchors)
