"""
ZA-DOS v0.6 — Learning Log Pipeline (spec §2.9).

Records learning events from each turn in learning modes.  Harvests
data from cognitive engine results (E19 patterns, E20 comparisons,
E17 RPE, E25 meta-learning) and MemoryContrast deltas.

Used by Homework Mode to review accumulated learning and by the
meta-learning reflective pipeline to identify trends.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from zados.core.types import LearningLogEntry

log = logging.getLogger(__name__)


class LearningLogPipeline:
    """Records and queries learning log entries.

    Parameters
    ----------
    ltmm_store : LTMMStore, optional
        If provided, entries are persisted to LTMM for cross-session
        retrieval.  Otherwise kept in-memory only.
    """

    def __init__(self, ltmm_store: Any = None) -> None:
        self._ltmm = ltmm_store
        self._entries: List[LearningLogEntry] = []

    def record_turn(
        self,
        mode: str,
        subject: str,
        session_id: str,
        engine_results: Dict[int, Dict[str, Any]],
        contrast_result: Any = None,
        reward_result: Any = None,
    ) -> LearningLogEntry:
        """Record learning events from a single turn.

        Parameters
        ----------
        mode : str
            "M1".."M5".
        subject : str
            SubjectCategory value.
        session_id : str
        engine_results : dict
            engine_number → engine result dict (from Phase 3/7).
        contrast_result : ContrastResult, optional
            From MemoryContrast.contrast() if available.
        reward_result : Phase5Result, optional
            From reward evaluation Phase 5.  If provided, domain scores
            are extracted and stored in ``entry.reward_scores``.

        Returns
        -------
        LearningLogEntry
        """
        entry = LearningLogEntry(
            timestamp=time.time(),
            mode=mode,
            subject=subject,
            session_id=session_id,
        )

        # Harvest E19 — Pattern Identification
        e19 = engine_results.get(19, {})
        patterns = e19.get("patterns", [])
        entry.e19_patterns = patterns
        entry.patterns_detected = len(patterns)

        # Harvest E20 — Pattern Comparison
        e20 = engine_results.get(20, {})
        comparisons = e20.get("comparisons", [])
        entry.e20_comparisons = comparisons

        # Harvest E17 — Reward-Based Learning
        e17 = engine_results.get(17, {})
        rpe_events = e17.get("rpe_events", [])
        entry.e17_rewards = rpe_events

        # Harvest E25 — Recursive Learning
        e25 = engine_results.get(25, {})
        meta_updates = e25.get("meta_updates", [])
        entry.e25_meta_updates = meta_updates

        # Harvest MemoryContrast deltas
        if contrast_result is not None:
            entry.contrast_deltas = {
                "divergence": getattr(contrast_result, "divergence", 0.0),
            }
            # Count learning event types from contrast
            changes = getattr(contrast_result, "changes", {})
            if isinstance(changes, dict):
                entry.confirmations = changes.get("confirmations", 0)
                entry.contradictions = changes.get("contradictions", 0)
                entry.extensions = changes.get("extensions", 0)
                entry.novel_entries = changes.get("novel_entries", 0)

        # Harvest reward domain scores from Phase 5 result
        if reward_result is not None:
            domain_results = getattr(reward_result, "domain_results", None)
            if domain_results and isinstance(domain_results, dict):
                for domain_name, domain_res in domain_results.items():
                    score = getattr(domain_res, "general_score", None)
                    if score is None:
                        score = domain_res.get("general_score", 0.0) if isinstance(domain_res, dict) else 0.0
                    entry.reward_scores[domain_name] = float(score)

        self._entries.append(entry)

        # Persist to LTMM if available
        if self._ltmm is not None:
            try:
                self._ltmm.write(
                    content=f"learning_log:{entry.turn_id}",
                    metadata={
                        "type": "learning_log",
                        "mode": mode,
                        "subject": subject,
                        "session_id": session_id,
                        "timestamp": entry.timestamp,
                        "patterns_detected": entry.patterns_detected,
                        "contradictions": entry.contradictions,
                        "novel_entries": entry.novel_entries,
                    },
                )
            except Exception:
                log.debug("Failed to persist learning log entry to LTMM.")

        log.debug(
            "Recorded learning log: mode=%s, patterns=%d, contradictions=%d, novel=%d",
            mode, entry.patterns_detected, entry.contradictions, entry.novel_entries,
        )
        return entry

    def get_unprocessed_logs(self) -> List[LearningLogEntry]:
        """Return all entries not yet processed by Homework Mode.

        Returns
        -------
        List[LearningLogEntry]
        """
        return [e for e in self._entries if not e.processed]

    def mark_processed(self, turn_ids: List[str]) -> int:
        """Mark entries as processed.

        Parameters
        ----------
        turn_ids : List[str]

        Returns
        -------
        int
            Number of entries marked.
        """
        count = 0
        id_set = set(turn_ids)
        for entry in self._entries:
            if entry.turn_id in id_set and not entry.processed:
                entry.processed = True
                count += 1
        return count

    def get_session_logs(self, session_id: str) -> List[LearningLogEntry]:
        """Return all entries for a given session."""
        return [e for e in self._entries if e.session_id == session_id]

    def get_mode_logs(self, mode: str) -> List[LearningLogEntry]:
        """Return all entries for a given learning mode."""
        return [e for e in self._entries if e.mode == mode]

    @property
    def all_entries(self) -> List[LearningLogEntry]:
        """Return all log entries."""
        return list(self._entries)
