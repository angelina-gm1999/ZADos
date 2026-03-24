"""
ZA-DOS v0.6 — Self-Reflective Query Pipeline (spec §3.4).

Selects an unsolved question from the buffer, gathers context via
MemoryContrast, builds a synthetic InputBundle, and delegates to
AnswerPipeline in M3 (Learn Together) mode for deep exploration.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from zados.core.processes.context_anchor import ContextAnchorManager
from zados.core.processes.engine_toolkit import EngineToolkit
from zados.core.processes.subject_classifier import classify_subject_from_text
from zados.core.processes.unsolved_buffer import UnsolvedBuffer
from zados.core.types import (
    InputBundle,
    PipelineResult,
    SelfRefResult,
    SessionState,
    UnsolvedQuestion,
)

log = logging.getLogger(__name__)


class SelfReflectiveQueryPipeline:
    """Self-reflective question exploration pipeline.

    Selects from the unsolved buffer, gathers context, then delegates
    to AnswerPipeline configured for M3 (dialectic) processing.

    Also pulls unreviewed HeldThinkingBlocks from LTMM and merges them
    into the question pool.  Held blocks are marked reviewed after use
    so they do not resurface in subsequent self-ref turns.

    Parameters
    ----------
    answer_pipeline : AnswerPipeline
    unsolved_buffer : UnsolvedBuffer
    memory_contrast : MemoryContrast, optional
    context_manager : ContextAnchorManager, optional
    ltmm : memory layer, optional
        If provided, held thinking blocks are queried and merged.
    """

    def __init__(
        self,
        answer_pipeline: Any,
        unsolved_buffer: UnsolvedBuffer,
        memory_contrast: Any = None,
        context_manager: Optional[ContextAnchorManager] = None,
        ltmm: Any = None,
        identity_journal_store: Any = None,
    ) -> None:
        self._pipeline = answer_pipeline
        self._unsolved = unsolved_buffer
        self._contrast = memory_contrast
        self._context = context_manager or ContextAnchorManager()
        self._toolkit = EngineToolkit()
        self._ltmm = ltmm
        self._identity_journal_store = identity_journal_store

    def process_turn(
        self,
        bundle: InputBundle,
        session: SessionState,
    ) -> SelfRefResult:
        """Process a self-reflective query turn.

        Steps:
          1. Select question from unsolved buffer (or use provided text).
          2. Gather context via MemoryContrast.
          3. Build synthetic bundle with M3 engine tiers.
          4. Delegate to AnswerPipeline.
          5. Update unsolved buffer.

        Parameters
        ----------
        bundle : InputBundle
        session : SessionState

        Returns
        -------
        SelfRefResult
        """
        # Step 0: Pull unreviewed held thinking blocks and inject into unsolved buffer
        held_blocks_used: list = []
        if self._ltmm is not None:
            held_blocks_used = self._inject_held_blocks_into_buffer()

        # Step 1: Select question (may now include held-block questions)
        selected = self._unsolved.select_next()
        if selected is None:
            log.info("No unsolved questions — self-reflective mode has nothing to explore.")
            return SelfRefResult(
                synthesis="No unsolved questions available for self-reflection.",
            )

        log.info("Self-reflective mode selected question: %s", selected.question_id)

        # Step 2: Gather context
        context = self._gather_context(selected)

        # Step 3: Build synthetic bundle configured for M3
        synthetic_text = self._build_synthetic_prompt(selected, context)
        bundle.raw_text = synthetic_text
        bundle.active_mode = "LearningMode_M3"

        subject = classify_subject_from_text(selected.question_text)
        tiers = self._toolkit.resolve("M3", subject)
        weights = self._toolkit.tiers_to_weights_by_id(tiers)
        bundle.engine_weights.update(weights)

        # Step 4: Delegate to AnswerPipeline
        result = self._pipeline.process_turn(bundle, session)

        # Step 5: Update unsolved buffer
        self._unsolved.mark_attempted(
            selected.question_id,
            partial_answer=result.final_answer[:200] if result.final_answer else "",
        )

        # Step 6: Mark held blocks as reviewed now that they've been processed
        self._mark_held_blocks_reviewed(held_blocks_used)

        # Step 7: Write identity journal entry (REFLECTION type)
        self._write_identity_journal_entry(result, selected, session)

        return SelfRefResult(
            selected_question=selected,
            context_gathered=context,
            synthesis=result.final_answer,
            rerouted_to_m3=True,
            pipeline_result=result,
        )

    def _write_identity_journal_entry(
        self,
        result: Any,
        selected: Any,
        session: Any,
    ) -> None:
        """Write a REFLECTION entry to the IdentityJournalStore.

        Called after each self-reflective turn.  The synthesis answer becomes
        the entry content, tagged with the selected question's context.
        """
        if self._identity_journal_store is None:
            return
        try:
            from zados.memory.long_term.identity.types import (
                IdentityJournalEntry,
                IdentityJournalEntryType,
            )
            synthesis = getattr(result, "final_answer", "") or ""
            content = synthesis[:800]
            tags = ["self_reflective"]
            if selected is not None:
                q_tags = getattr(selected, "tags", [])
                tags.extend([t for t in q_tags[:3] if t])
                source_mode = getattr(selected, "source_mode", "")
                if source_mode and source_mode != "held_block":
                    tags.append(f"source:{source_mode}")

            # NT snapshot from pipeline state if available
            nt_snapshot: dict = {}
            state = getattr(result, "state", None)
            if state is not None:
                stmm = getattr(state, "stmm", None)
                if stmm is not None:
                    cll = getattr(stmm, "cephalic_liquid_logger", None)
                    if cll is not None:
                        nt_snapshot = dict(getattr(cll, "nt_concentrations", {}))

            # Emotion tags from STMM
            emotion_tags: list = []
            if state is not None:
                stmm = getattr(state, "stmm", None)
                if stmm is not None:
                    ed = getattr(stmm, "emotion_detection", None)
                    if ed is not None:
                        ue = getattr(ed, "user_emotion_signals", {}) or {}
                        emotion_tags = [
                            k for k, v in ue.items()
                            if isinstance(v, float) and v > 0.3
                        ][:5]

            entry = IdentityJournalEntry(
                entry_type=IdentityJournalEntryType.REFLECTION,
                content=content,
                source_pipeline="self_reflective",
                tags=tags,
                nt_snapshot=nt_snapshot,
                emotion_tags=emotion_tags,
            )
            self._identity_journal_store.write(entry)
        except Exception:
            log.debug("Identity journal write failed in self-reflective pipeline.")

    def _inject_held_blocks_into_buffer(self) -> list:
        """Query unreviewed held thinking blocks from LTMM and add to unsolved buffer.

        Returns list of HeldBlock objects that were injected (for marking reviewed later).
        """
        injected = []
        try:
            store = None
            if hasattr(self._ltmm, "namespace_store"):
                store = self._ltmm.namespace_store
            elif hasattr(self._ltmm, "get_namespace_store"):
                store = self._ltmm.get_namespace_store("thoughts/held_blocks")
            if store is None:
                return []

            results = []
            if hasattr(store, "search_folder"):
                results = store.search_folder(
                    "thoughts/held_blocks",
                    tags_required=frozenset(),
                    tags_excluded=frozenset({"reviewed"}),
                    max_results=5,
                )
            elif hasattr(store, "search"):
                results = store.search(
                    query="",
                    folder="thoughts/held_blocks",
                    max_results=5,
                    exclude_tags=["reviewed"],
                )

            for entry in results:
                content = (
                    getattr(entry, "content", "")
                    if not isinstance(entry, dict)
                    else entry.get("content", "")
                )
                metadata = (
                    getattr(entry, "metadata", {})
                    if not isinstance(entry, dict)
                    else entry.get("metadata", {})
                )
                trigger = ""
                if isinstance(metadata, dict):
                    trigger = metadata.get("trigger_summary", metadata.get("trigger", ""))
                eid = (
                    getattr(entry, "entry_id", "")
                    if not isinstance(entry, dict)
                    else entry.get("entry_id", "")
                )

                # Build a synthetic UnsolvedQuestion from the held block
                q = UnsolvedQuestion(
                    question_text=str(content)[:500],
                    source_mode="held_block",
                    source_context=f"Trigger: {trigger}" if trigger else "held thinking block",
                    tags=["held_block", f"block_id:{eid}"],
                    urgency_score=0.6,
                )
                try:
                    self._unsolved.add(q)
                    injected.append({"entry": entry, "entry_id": str(eid)})
                except Exception:
                    pass

        except Exception:
            log.debug("Failed to inject held blocks into unsolved buffer.")

        return injected

    def _mark_held_blocks_reviewed(self, held_blocks_used: list) -> None:
        """Mark held thinking blocks as reviewed in LTMM so they won't resurface."""
        if not held_blocks_used or self._ltmm is None:
            return
        for item in held_blocks_used:
            try:
                entry = item.get("entry")
                eid = item.get("entry_id", "")
                if entry is None or not eid:
                    continue
                # Add "reviewed" tag
                tags = list(getattr(entry, "tags", []) if not isinstance(entry, dict) else entry.get("tags", []))
                if "reviewed" not in tags:
                    tags.append("reviewed")
                if hasattr(entry, "tags"):
                    entry.tags = tags
                elif isinstance(entry, dict):
                    entry["tags"] = tags
                # Persist if store supports update
                store = getattr(self._ltmm, "namespace_store", None)
                if store and hasattr(store, "update"):
                    store.update(entry)
            except Exception:
                log.debug("Failed to mark held block %s as reviewed.", item.get("entry_id", "?"))

    def _gather_context(self, question: UnsolvedQuestion) -> Dict[str, Any]:
        """Gather context for the selected question via MemoryContrast.

        Parameters
        ----------
        question : UnsolvedQuestion

        Returns
        -------
        Dict[str, Any]
        """
        context: Dict[str, Any] = {
            "question_text": question.question_text,
            "source_mode": question.source_mode,
            "attempts": question.resolution_attempts,
            "partial_answers": question.partial_answers,
        }

        if self._contrast is not None:
            try:
                result = self._contrast.contrast(
                    current={"text": question.question_text, "content": question.question_text},
                    query_type="context",
                )
                context["contrast_divergence"] = getattr(result, "divergence", 0.0)
                context["related_memories"] = getattr(result, "related", [])
            except Exception:
                log.debug("MemoryContrast context gathering failed for self-ref query.")

        return context

    def _build_synthetic_prompt(
        self,
        question: UnsolvedQuestion,
        context: Dict[str, Any],
    ) -> str:
        """Build a synthetic prompt for the AnswerPipeline.

        Parameters
        ----------
        question : UnsolvedQuestion
        context : dict

        Returns
        -------
        str
        """
        parts = [question.question_text]

        if question.partial_answers:
            parts.append(
                f"Previous attempts have explored: {'; '.join(question.partial_answers[-3:])}"
            )

        if context.get("related_memories"):
            parts.append("Related context is available from memory.")

        return " ".join(parts)
