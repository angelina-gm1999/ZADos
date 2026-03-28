"""
ZA-DOS v0.6 — Unsolved Buffer (spec §2.10).

Maintains a priority queue of unresolved questions that emerged during
learning sessions.  Used by M4 (Learned Questions) and the
Self-Reflective Query pipeline to select questions for further
exploration.

Priority cascade:  urgency → creation date → stagnation time.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from zados.core.types import UnsolvedQuestion

log = logging.getLogger(__name__)


class UnsolvedBuffer:
    """Buffer of unresolved questions with priority-based selection.

    Parameters
    ----------
    ltmm_store : LTMMStore, optional
        If provided, questions are persisted to LTMM for cross-session
        retrieval.
    """

    def __init__(self, ltmm_store: Any = None) -> None:
        self._ltmm = ltmm_store
        self._questions: List[UnsolvedQuestion] = []

    def add(
        self,
        question_text: str,
        source_mode: str = "",
        source_context: str = "",
        urgency_score: float = 0.5,
        tags: Optional[List[str]] = None,
    ) -> UnsolvedQuestion:
        """Add a new unsolved question to the buffer.

        Parameters
        ----------
        question_text : str
        source_mode : str
            "M1".."M5" or "self_ref".
        source_context : str
            Brief context snippet.
        urgency_score : float
            0.0 - 1.0.
        tags : List[str], optional

        Returns
        -------
        UnsolvedQuestion
        """
        q = UnsolvedQuestion(
            question_text=question_text,
            source_mode=source_mode,
            source_context=source_context,
            urgency_score=urgency_score,
            tags=tags or [],
        )
        self._questions.append(q)

        # Persist to LTMM
        if self._ltmm is not None:
            try:
                self._ltmm.write(
                    content=f"unsolved_question:{q.question_id}:{question_text}",
                    metadata={
                        "type": "unsolved_question",
                        "question_id": q.question_id,
                        "source_mode": source_mode,
                        "urgency_score": urgency_score,
                        "tags": tags or [],
                    },
                )
            except Exception:
                log.debug("Failed to persist unsolved question to LTMM.")

        log.debug("Added unsolved question: urgency=%.2f, tags=%s", urgency_score, tags)
        return q

    def select_next(self) -> Optional[UnsolvedQuestion]:
        """Select the highest-priority unresolved question.

        Priority cascade:
          1. Highest urgency_score
          2. Oldest creation_date (FIFO among equal urgency)
          3. Longest stagnation_time

        Returns
        -------
        UnsolvedQuestion or None
        """
        active = [q for q in self._questions if not q.resolved]
        if not active:
            return None

        # Update stagnation times
        now = time.time()
        for q in active:
            q.stagnation_time = now - q.last_modified

        # Sort: urgency DESC, creation_date ASC, stagnation DESC
        active.sort(
            key=lambda q: (-q.urgency_score, q.creation_date, -q.stagnation_time),
        )
        return active[0]

    def mark_attempted(self, question_id: str, partial_answer: str = "") -> bool:
        """Record an attempt to answer a question.

        Parameters
        ----------
        question_id : str
        partial_answer : str, optional

        Returns
        -------
        bool
            True if found and updated.
        """
        for q in self._questions:
            if q.question_id == question_id:
                q.resolution_attempts += 1
                q.last_modified = time.time()
                q.stagnation_time = 0.0
                if partial_answer:
                    q.partial_answers.append(partial_answer)
                return True
        return False

    def resolve(self, question_id: str) -> bool:
        """Mark a question as resolved.

        Parameters
        ----------
        question_id : str

        Returns
        -------
        bool
            True if found and resolved.
        """
        for q in self._questions:
            if q.question_id == question_id:
                q.resolved = True
                q.last_modified = time.time()
                log.debug("Resolved question: %s", question_id)
                return True
        return False

    def get_active(self) -> List[UnsolvedQuestion]:
        """Return all unresolved questions."""
        return [q for q in self._questions if not q.resolved]

    def get_by_tags(self, tags: List[str]) -> List[UnsolvedQuestion]:
        """Return unresolved questions matching any of the given tags."""
        tag_set = set(tags)
        return [
            q for q in self._questions
            if not q.resolved and tag_set & set(q.tags)
        ]

    def cluster_questions(self) -> Dict[str, List[UnsolvedQuestion]]:
        """Group unresolved questions by thematic similarity.

        Uses a simple tag-based clustering.  Returns tag → questions.
        Questions with no tags appear under "__untagged".

        Returns
        -------
        Dict[str, List[UnsolvedQuestion]]
        """
        clusters: Dict[str, List[UnsolvedQuestion]] = {}
        for q in self.get_active():
            if q.tags:
                for tag in q.tags:
                    clusters.setdefault(tag, []).append(q)
            else:
                clusters.setdefault("__untagged", []).append(q)
        return clusters

    @property
    def size(self) -> int:
        """Number of active (unresolved) questions."""
        return len(self.get_active())

    def is_empty(self) -> bool:
        """True if no unresolved questions remain."""
        return self.size == 0

    def load_from_ltmm(self, general_question_store: Any) -> int:
        """Restore unresolved questions from LTMM GeneralQuestionStore at session start.

        Returns number of questions loaded.
        """
        if general_question_store is None:
            return 0
        loaded = 0
        try:
            for gq in general_question_store.get_all():
                if getattr(gq, "resolved", False):
                    continue
                # Avoid duplicates
                existing_texts = {q.question_text for q in self._questions}
                formulation = getattr(gq, "formulation", "")
                if not formulation or formulation in existing_texts:
                    continue
                q = UnsolvedQuestion(
                    question_text=formulation,
                    source_mode="restored",
                    source_context="",
                    urgency_score=getattr(gq, "priority", 0.5),
                    tags=getattr(gq, "tags", []),
                )
                self._questions.append(q)
                loaded += 1
        except Exception:
            log.debug("Failed to load questions from LTMM GeneralQuestionStore.")
        log.debug("Loaded %d unresolved questions from LTMM.", loaded)
        return loaded

    def sync_resolved_to_ltmm(self, general_question_store: Any) -> int:
        """Sync resolved status back to LTMM GeneralQuestionStore at session end.

        Returns number of questions synced.
        """
        if general_question_store is None:
            return 0
        synced = 0
        try:
            resolved = [q for q in self._questions if q.resolved]
            for q in resolved:
                # Try to find and resolve matching question in LTMM
                try:
                    if hasattr(general_question_store, "resolve"):
                        general_question_store.resolve(q.question_text)
                        synced += 1
                except Exception:
                    pass
        except Exception:
            log.debug("Failed to sync resolved questions to LTMM.")
        return synced
