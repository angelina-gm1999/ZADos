"""
ZA-DOS v0.6 — Homework Pipeline (Part 5 spec).

Six-phase offline processing pipeline that integrates accumulated learning
material:

  Phase 0  Input Assembly & Triage — batch by subject, compute deficits
  Phase 1  Analysis Stage          — content decomposition, memory contrast
  Phase 2  Processing Stage        — contradiction resolution, dialectic stress
  Phase 3  Question Resolution     — buffer updates, dream candidate flagging
  Phase 4  Synthesis & Integration — lesson finalization, knowledge map update
  Phase 5  Output & Storage        — LTMM writes, summary generation

Homework Mode operates WITHOUT a user present — no emotional feedback loop,
no E28 emotion detection, no response generation.  NT layer is read-only
(used for diagnostic deficit profiling, not actively modulated).

Triggered by ``/homework`` command.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from zados.core.commanded.meta_learning_mode.homework_mode.deficit_profiler import (
    compute_batch_deficit,
    get_engine_emphasis,
    identify_deficit_domain,
    sort_batches_by_deficit,
)
from zados.memory.long_term.knowledge.academic_buffer.store import (
    AcademicBufferEntry,
)
from zados.core.processes.engine_toolkit import EngineToolkit
from zados.core.processes.learning_log import LearningLogPipeline
from zados.core.processes.subject_classifier import classify_subject_from_text
from zados.core.processes.unsolved_buffer import UnsolvedBuffer
from zados.core.types import (
    HomeworkRunSummary,
    InputBundle,
    LearningLogEntry,
    PendingCoreMemoryUpdate,
    PipelineResult,
    ReflectiveModeInput,
    SessionState,
    SubjectCategory,
    UnsolvedQuestion,
)

log = logging.getLogger(__name__)

# Minimum confidence for automatic lesson validation  [SPEC NEEDED]
_MIN_VALIDATION_CONFIDENCE = 0.5

# Maximum stagnation attempts before dream candidate flagging
_DREAM_STAGNATION_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Intermediate batch data structure
# ---------------------------------------------------------------------------

@dataclass
class ProcessedBatch:
    """Intermediate result from Phase 1 analysis of a subject batch."""

    subject: str = ""
    entries: List[LearningLogEntry] = field(default_factory=list)
    deficit_domain: str = "mixed"
    deficit_profile: Dict[str, float] = field(default_factory=dict)

    # Phase 1 analysis outputs
    contrast_deltas: List[Dict[str, Any]] = field(default_factory=list)
    novel_patterns: List[Dict[str, Any]] = field(default_factory=list)
    pattern_reinforcements: List[Dict[str, Any]] = field(default_factory=list)
    contradiction_candidates: List[Dict[str, Any]] = field(default_factory=list)
    relevance_scored_entries: List[Tuple[float, LearningLogEntry]] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Phase 2 processing output
# ---------------------------------------------------------------------------

@dataclass
class ProcessingOutput:
    """Aggregated output from Phase 2 processing of a single batch."""

    validated_lessons: List[Dict[str, Any]] = field(default_factory=list)
    contradictions_resolved: List[Dict[str, Any]] = field(default_factory=list)
    contradictions_unresolved: List[Dict[str, Any]] = field(default_factory=list)
    fallacy_flags: List[Dict[str, Any]] = field(default_factory=list)
    bias_flags: List[Dict[str, Any]] = field(default_factory=list)
    paradox_flags: List[Dict[str, Any]] = field(default_factory=list)
    pln_confidence_scores: Dict[str, float] = field(default_factory=dict)
    pipeline_result: Optional[PipelineResult] = None


# ===========================================================================
# HomeworkPipeline
# ===========================================================================

class HomeworkPipeline:
    """Homework mode — 6-phase offline processing and integration.

    Parameters
    ----------
    answer_pipeline : AnswerPipeline
        Reused for individual processing sub-tasks via process_turn().
    learning_log : LearningLogPipeline
        Source of unprocessed learning log entries.
    unsolved_buffer : UnsolvedBuffer
        Active unsolved questions queue.
    memory_layer : MemoryLayer, optional
        Full namespaced memory access (knowledge, thoughts, identity stores).
    specialized_logs : SpecializedLogs, optional
        Cross-cutting logs (ContradictionLog, SelfReflectionLog, etc.).
    """

    def __init__(
        self,
        answer_pipeline: Any,
        learning_log: LearningLogPipeline,
        unsolved_buffer: UnsolvedBuffer,
        memory_layer: Any = None,
        specialized_logs: Any = None,
        neurochem_engine: Any = None,
    ) -> None:
        self._pipeline = answer_pipeline
        self._learning_log = learning_log
        self._unsolved = unsolved_buffer
        self._memory = memory_layer
        self._spec_logs = specialized_logs
        self._toolkit = EngineToolkit()
        # Read-only neurochem access for diagnostic deficit profiling
        self._neurochem = neurochem_engine
        self._journal_store: Any = None  # set via set_journal_store() after init

    # ===================================================================
    # Main entry point
    # ===================================================================

    def process(self, session: SessionState) -> Dict[str, Any]:
        """Run the full 6-phase homework pipeline.

        Parameters
        ----------
        session : SessionState

        Returns
        -------
        Dict[str, Any]
            Summary of homework run results.
        """
        log.info("Homework Pipeline activated for session %s.", session.session_id)
        start_time = time.time()

        summary = HomeworkRunSummary(session_id=session.session_id)

        # --- Phase 0: Input Assembly & Triage ---
        batches, unprocessed = self._phase0_input_assembly()
        if not batches and not self._unsolved.get_active():
            log.info("Homework: nothing to process.")
            return self._build_return(summary, unprocessed, start_time)

        # --- Phase 1-2: Per-batch Analysis + Processing ---
        all_batch_outputs: List[Tuple[ProcessedBatch, ProcessingOutput]] = []
        for subject, entries, deficit_domain in batches:
            summary.batches_processed += 1
            summary.processing_emphasis[subject] = deficit_domain

            # Phase 1: Analysis
            processed_batch = self._phase1_analysis(subject, entries, deficit_domain)

            # Phase 2: Processing
            processing_output = self._phase2_processing(processed_batch)

            all_batch_outputs.append((processed_batch, processing_output))

            # Accumulate stats
            summary.lessons_validated += len(processing_output.validated_lessons)
            summary.contradictions_resolved += len(processing_output.contradictions_resolved)
            summary.contradictions_unresolved += len(processing_output.contradictions_unresolved)
            summary.fallacy_bias_flags.extend(processing_output.fallacy_flags)
            summary.fallacy_bias_flags.extend(processing_output.bias_flags)

        # --- Phase 3: Question Resolution ---
        questions = self._unsolved.get_active()
        q_result = self._phase3_question_resolution(all_batch_outputs, questions, session)
        summary.questions_resolved += q_result.get("resolved", 0)
        summary.questions_new += q_result.get("new_questions", 0)
        summary.dream_candidates_flagged += q_result.get("dream_candidates", 0)

        # --- Phase 4: Synthesis & Knowledge Integration ---
        synthesis = self._phase4_synthesis(all_batch_outputs, session)
        summary.core_memory_updates_applied += synthesis.get("core_updates_applied", 0)
        summary.meta_patterns = synthesis.get("meta_patterns", [])
        summary.lessons_pending = synthesis.get("lessons_pending", 0)

        # --- Phase 5: Output & Storage ---
        self._phase5_output(summary, unprocessed, all_batch_outputs, session)

        # --- Tick academic buffer stagnation counters ---
        self._tick_academic_buffer()

        # --- Journal write ---
        self._write_journal(summary, session)

        # Write identity journal entry if identity-relevant findings
        self._write_identity_journal_if_relevant(session, summary)

        return self._build_return(summary, unprocessed, start_time)

    # ===================================================================
    # Phase 0: Input Assembly & Triage
    # ===================================================================

    def _phase0_input_assembly(
        self,
    ) -> Tuple[List[Tuple[str, List[LearningLogEntry], str]], List[LearningLogEntry]]:
        """Fetch unprocessed logs, group by subject, compute deficits.

        Returns
        -------
        Tuple
            (sorted_batches, unprocessed_entries)
        """
        unprocessed = self._learning_log.get_unprocessed_logs()
        log.info("Phase 0: %d unprocessed learning log entries.", len(unprocessed))

        if not unprocessed:
            return [], []

        # Group by subject_category
        subject_groups: Dict[str, List[LearningLogEntry]] = {}
        for entry in unprocessed:
            subj = entry.subject or "mixed"
            subject_groups.setdefault(subj, []).append(entry)

        log.info("Phase 0: %d subject batches: %s",
                 len(subject_groups), list(subject_groups.keys()))

        # Incorporate neurochem metrics into deficit profiling (read-only)
        nt_deficit_bias = self._read_neurochem_deficit_bias()

        # Sort by deficit severity (deepest deficit processed first)
        sorted_batches = sort_batches_by_deficit(subject_groups, nt_deficit_bias)

        return sorted_batches, unprocessed

    # ===================================================================
    # Phase 1: Analysis Stage (per batch)
    # ===================================================================

    def _phase1_analysis(
        self,
        subject: str,
        entries: List[LearningLogEntry],
        deficit_domain: str,
    ) -> ProcessedBatch:
        """Analyse a single subject batch.

        Steps:
          1.1 Content decomposition (aggregate E19 patterns from logs)
          1.2 Relevance scoring (score entries by information density)
          1.3 Memory contrast — compare incoming vs existing knowledge
          1.4 Pattern analysis (gather E19/E20 results)
          1.5 Contradiction candidate identification

        Returns
        -------
        ProcessedBatch
        """
        log.info("Phase 1: Analysing batch '%s' (%d entries, deficit=%s)",
                 subject, len(entries), deficit_domain)

        batch = ProcessedBatch(
            subject=subject,
            entries=entries,
            deficit_domain=deficit_domain,
            deficit_profile=compute_batch_deficit(entries),
        )

        # 1.1-1.2: Aggregate patterns and score entries
        for entry in entries:
            score = self._compute_entry_relevance(entry)
            batch.relevance_scored_entries.append((score, entry))

            # Collect patterns
            for pat in entry.e19_patterns:
                if pat.get("status") == "CONFIRMED":
                    batch.pattern_reinforcements.append(pat)
                else:
                    batch.novel_patterns.append(pat)

            # Collect contrast deltas
            if entry.contrast_deltas:
                batch.contrast_deltas.append({
                    "turn_id": entry.turn_id,
                    "deltas": entry.contrast_deltas,
                })

        # Sort entries by relevance (highest first)
        batch.relevance_scored_entries.sort(key=lambda x: x[0], reverse=True)

        # 1.3: Memory contrast — check for contradictions with existing knowledge
        if self._memory is not None:
            batch.contradiction_candidates = self._contrast_against_knowledge(
                entries, subject
            )

        # 1.4: Aggregate E20 comparison results
        for entry in entries:
            for comp in entry.e20_comparisons:
                if comp.get("divergence", 0.0) > 0.5:
                    batch.contradiction_candidates.append({
                        "type": "pattern_divergence",
                        "source_turn": entry.turn_id,
                        "comparison": comp,
                    })

        log.info("Phase 1 complete: %d novel patterns, %d reinforcements, "
                 "%d contradiction candidates",
                 len(batch.novel_patterns), len(batch.pattern_reinforcements),
                 len(batch.contradiction_candidates))

        return batch

    # ===================================================================
    # Phase 2: Processing Stage (per batch)
    # ===================================================================

    def _phase2_processing(self, batch: ProcessedBatch) -> ProcessingOutput:
        """Process a batch: contradiction resolution, dialectic stress-test.

        Steps:
          2.1 Contradiction resolution via AnswerPipeline
          2.2 Paradox detection (flag productive vs unproductive)
          2.3 Fallacy/bias sweep
          2.4 Dialectic stress-testing (SimOpp + Socratic)
          2.5 PLN confidence weighting

        Returns
        -------
        ProcessingOutput
        """
        log.info("Phase 2: Processing batch '%s' (deficit=%s)",
                 batch.subject, batch.deficit_domain)

        output = ProcessingOutput()

        # Resolve subject category for engine toolkit
        try:
            subject_cat = SubjectCategory(batch.subject)
        except ValueError:
            subject_cat = SubjectCategory.MIXED

        # 2.1-2.4: Run through pipeline with homework tiers + deficit emphasis
        emphasis = get_engine_emphasis(batch.deficit_domain)

        # Build synthetic input from batch's top entries for processing
        top_entries = batch.relevance_scored_entries[:5]
        if not top_entries:
            return output

        # Compose processing context
        context_parts = []
        for _score, entry in top_entries:
            context_parts.append(entry.mode + ": " + str(entry.contrast_deltas))
            if entry.e19_patterns:
                context_parts.append(f"Patterns: {len(entry.e19_patterns)}")

        # Add contradiction candidates if any
        for cc in batch.contradiction_candidates[:3]:
            context_parts.append(f"Contradiction: {cc.get('type', 'unknown')}")

        processing_text = (
            f"[HOMEWORK PROCESSING] Subject: {batch.subject}, "
            f"Deficit: {batch.deficit_domain}. "
            + " | ".join(context_parts[:10])
        )

        bundle = InputBundle(
            raw_text=processing_text,
            active_mode="homework",
        )

        # Apply homework engine tiers with subject adjustment
        tiers = self._toolkit.resolve("homework", subject_cat)
        weights = self._toolkit.tiers_to_weights_by_id(tiers)
        bundle.engine_weights.update(weights)

        # Store emphasis metadata in context_flags for engines to read
        for eng_name, directive in emphasis.items():
            bundle.context_flags[f"emphasis:{eng_name}"] = True

        try:
            result = self._pipeline.process_turn(bundle, SessionState(
                session_id=f"homework_{batch.subject}",
                session_mode="learning",
                initial_mode="homework",
            ))
            output.pipeline_result = result

            # Extract engine results for analysis
            if result.state and result.state.dispatch:
                engine_results = result.state.dispatch.engine_results

                # 2.1: Contradiction detection results
                e1_result = engine_results.get(1, {})
                contradictions = e1_result.get("contradictions", [])
                for c in contradictions:
                    if c.get("resolved", False):
                        output.contradictions_resolved.append(c)
                    else:
                        output.contradictions_unresolved.append(c)

                # 2.2: Paradox detection
                e2_result = engine_results.get(2, {})
                paradoxes = e2_result.get("paradoxes", [])
                for p in paradoxes:
                    output.paradox_flags.append({
                        "classification": p.get("classification", "unproductive"),
                        "formulation": p.get("formulation", ""),
                    })

                # 2.3: Fallacy/bias sweep
                e4_result = engine_results.get(4, {})
                fallacies = e4_result.get("fallacies", [])
                for f in fallacies:
                    output.fallacy_flags.append({
                        "type": "fallacy",
                        "name": f.get("name", ""),
                        "severity": f.get("severity", "low"),
                        "context": f.get("context", ""),
                    })

                e5_result = engine_results.get(5, {})
                biases = e5_result.get("biases", [])
                for b in biases:
                    output.bias_flags.append({
                        "type": "bias",
                        "name": b.get("name", ""),
                        "severity": b.get("severity", "low"),
                        "context": b.get("context", ""),
                    })

                # 2.5: PLN confidence scores
                e10_result = engine_results.get(10, {})
                pln_scores = e10_result.get("confidence_scores", {})
                output.pln_confidence_scores.update(pln_scores)

        except Exception as e:
            log.warning("Phase 2 pipeline processing failed for '%s': %s",
                        batch.subject, e)

        # Log fallacy/bias flags to SelfReflectionLog if available
        if self._spec_logs is not None and (output.fallacy_flags or output.bias_flags):
            self._log_fallacy_bias(output.fallacy_flags, output.bias_flags)

        # Record contradictions to ContradictionLog if available
        if self._spec_logs is not None:
            self._log_contradictions(
                output.contradictions_resolved,
                output.contradictions_unresolved,
            )

        # Determine validated lessons from entries with high confidence
        for _score, entry in top_entries:
            if entry.confirmations > 0 and entry.contradictions == 0:
                output.validated_lessons.append({
                    "turn_id": entry.turn_id,
                    "mode": entry.mode,
                    "subject": entry.subject,
                    "confirmations": entry.confirmations,
                })

        log.info("Phase 2 complete: %d validated, %d contradictions resolved, "
                 "%d unresolved, %d fallacy/bias flags",
                 len(output.validated_lessons),
                 len(output.contradictions_resolved),
                 len(output.contradictions_unresolved),
                 len(output.fallacy_flags) + len(output.bias_flags))

        return output

    # ===================================================================
    # Phase 3: Question Resolution & Buffer Update
    # ===================================================================

    def _phase3_question_resolution(
        self,
        batch_outputs: List[Tuple[ProcessedBatch, ProcessingOutput]],
        questions: List[UnsolvedQuestion],
        session: SessionState,
    ) -> Dict[str, Any]:
        """Resolve questions, flag dream candidates, route cross-domain.

        Steps:
          3.1 Check if Phase 2 resolved any buffered questions
          3.2 Capture new questions from dialectic engines
          3.3 Dream candidate check (stagnation >= threshold)
          3.4 Process top questions directly

        Returns
        -------
        Dict[str, Any]
            Resolution statistics.
        """
        log.info("Phase 3: %d active unsolved questions.", len(questions))

        resolved_count = 0
        new_question_count = 0
        dream_candidate_count = 0

        # 3.1: Check if any pipeline results contain answers to questions
        for batch, proc in batch_outputs:
            if proc.pipeline_result and proc.pipeline_result.final_answer:
                answer = proc.pipeline_result.final_answer
                for q in questions:
                    if not q.resolved and self._answer_matches_question(q, answer, batch.subject):
                        self._unsolved.mark_attempted(
                            q.question_id,
                            partial_answer=answer[:200],
                        )
                        # If answer is substantive, mark resolved
                        if len(answer) > 50:
                            self._unsolved.resolve(q.question_id)
                            resolved_count += 1

        # 3.2: Capture new questions from contradiction/paradox analysis
        for batch, proc in batch_outputs:
            for cc in proc.contradictions_unresolved:
                question_text = cc.get("formulation", cc.get("context", ""))
                if question_text:
                    self._unsolved.add(
                        question_text=f"Unresolved contradiction ({batch.subject}): {question_text}",
                        source_mode="homework",
                        source_context=batch.subject,
                        urgency_score=0.6,
                        tags=[batch.subject, "contradiction", "homework"],
                    )
                    new_question_count += 1

        # 3.3: Write unresolved contradictions to AcademicBufferStore as concepts
        academic_buf = self._get_academic_buffer()
        if academic_buf is not None:
            for batch, proc in batch_outputs:
                for cc in proc.contradictions_unresolved:
                    formulation = cc.get("formulation", cc.get("context", ""))
                    if not formulation:
                        continue
                    concept_text = (
                        f"Unresolved contradiction ({batch.subject}): "
                        f"{formulation}"
                    )
                    already_buffered = any(
                        e.concept_formulation == concept_text and not e.resolved
                        for e in academic_buf.get_all()
                    )
                    if not already_buffered:
                        academic_buf.add(AcademicBufferEntry(
                            concept_formulation=concept_text,
                            subject_category=batch.subject,
                            source_engine="homework_phase2_dialectic",
                            blocking_reason=cc.get("severity", "unresolved contradiction"),
                        ))

        # 3.4: Dream candidate flagging + origin tagging
        from zados.core.tags import T
        for q in self._unsolved.get_active():
            if q.resolution_attempts >= _DREAM_STAGNATION_THRESHOLD:
                dream_candidate_count += 1
                if "dream_candidate" not in q.tags:
                    q.tags.append("dream_candidate")
                # Tag with origin:academic so sleep pipelines can
                # differentiate processing (REM boosts logic, Dream deprioritises)
                origin_tag = T.origin("academic")
                if origin_tag not in q.tags:
                    q.tags.append(origin_tag)
                    q.scope_tag = q.scope_tag or "academic"

        # 3.4: Write unresolvable conceptual items to AcademicBufferStore
        academic_buf = self._get_academic_buffer()
        if academic_buf is not None:
            for q in self._unsolved.get_active():
                if (
                    not q.resolved
                    and q.resolution_attempts >= _DREAM_STAGNATION_THRESHOLD
                ):
                    # Avoid duplicates: check if concept already buffered
                    already_buffered = any(
                        e.concept_formulation == q.question_text
                        and not e.resolved
                        for e in academic_buf.get_all()
                    )
                    if not already_buffered:
                        subject = classify_subject_from_text(q.question_text)
                        academic_buf.add(AcademicBufferEntry(
                            concept_formulation=q.question_text,
                            subject_category=subject,
                            source_engine="homework_phase3",
                            blocking_reason=(
                                f"Stagnated after {q.resolution_attempts} "
                                "resolution attempts in UnsolvedBuffer"
                            ),
                        ))
                        log.info(
                            "Phase 3: Wrote stagnated question %s to "
                            "AcademicBufferStore.",
                            q.question_id,
                        )

        # 3.5: Process top remaining questions directly through pipeline
        remaining = [q for q in self._unsolved.get_active() if not q.resolved]
        for q in remaining[:5]:
            try:
                subject = classify_subject_from_text(q.question_text)
                bundle = InputBundle(
                    raw_text=q.question_text,
                    active_mode="homework",
                )
                tiers = self._toolkit.resolve("homework", subject)
                weights = self._toolkit.tiers_to_weights_by_id(tiers)
                bundle.engine_weights.update(weights)

                result = self._pipeline.process_turn(bundle, session)
                partial = result.final_answer[:200] if result.final_answer else ""
                self._unsolved.mark_attempted(q.question_id, partial_answer=partial)

                if result.final_answer and len(result.final_answer) > 50:
                    self._unsolved.resolve(q.question_id)
                    resolved_count += 1
            except Exception as e:
                log.warning("Phase 3: Failed to process question %s: %s",
                            q.question_id, e)

        log.info("Phase 3 complete: %d resolved, %d new, %d dream candidates",
                 resolved_count, new_question_count, dream_candidate_count)

        return {
            "resolved": resolved_count,
            "new_questions": new_question_count,
            "dream_candidates": dream_candidate_count,
        }

    # ===================================================================
    # Phase 4: Synthesis & Knowledge Integration
    # ===================================================================

    def _phase4_synthesis(
        self,
        batch_outputs: List[Tuple[ProcessedBatch, ProcessingOutput]],
        session: SessionState,
    ) -> Dict[str, Any]:
        """Finalize lessons, update knowledge maps, gate core memory.

        Steps:
          4.1 Lesson finalization — validate confirmed lessons, reinforce
          4.2 Knowledge map update
          4.3 Core memory update gate
          4.4 Cross-batch pattern synthesis

        Returns
        -------
        Dict[str, Any]
        """
        log.info("Phase 4: Synthesis across %d batches.", len(batch_outputs))

        core_updates_applied = 0
        lessons_pending = 0
        meta_patterns: List[Dict[str, Any]] = []

        # 4.1: Lesson finalization
        if self._memory is not None:
            lesson_store = getattr(
                getattr(self._memory, "knowledge", None), "lessons", None
            )
            if lesson_store is not None:
                for batch, proc in batch_outputs:
                    for lesson_info in proc.validated_lessons:
                        # Try to find matching lesson in store
                        results = lesson_store.search(
                            lesson_info.get("subject", batch.subject), limit=3
                        )
                        for _sim, lesson in results:
                            if lesson.validation_status == "pending":
                                if _sim >= _MIN_VALIDATION_CONFIDENCE:
                                    lesson_store.validate(lesson.lesson_id)
                                else:
                                    lessons_pending += 1
                            elif lesson.validation_status == "validated":
                                lesson_store.reinforce(lesson.lesson_id)

        # 4.2: Knowledge map update
        if self._memory is not None:
            km_store = getattr(
                getattr(self._memory, "knowledge", None), "knowledge_maps", None
            )
            if km_store is not None:
                for batch, proc in batch_outputs:
                    if batch.novel_patterns or batch.pattern_reinforcements:
                        existing_maps = km_store.get_by_subject(batch.subject)
                        if existing_maps:
                            # Update existing map with new patterns
                            km = existing_maps[0]
                            for pat in batch.novel_patterns:
                                km.nodes.append({
                                    "label": pat.get("name", "pattern"),
                                    "type": "concept",
                                    "source": "homework",
                                })
                            km_store.write(km)

        # 4.3: Core memory update gate
        if self._memory is not None:
            pending_queue = getattr(
                getattr(
                    getattr(self._memory, "identity", None),
                    "core",
                    None
                ),
                "pending_queue",
                None,
            )
            core_store = getattr(
                getattr(self._memory, "identity", None), "core", None
            )
            # Also check for PendingUpdateQueue as separate attribute
            if pending_queue is None:
                pending_queue = getattr(
                    getattr(self._memory, "identity", None),
                    "pending_updates",
                    None,
                )

            if pending_queue is not None and core_store is not None:
                pending = pending_queue.get_pending()
                for upd in pending:
                    # Gate: only apply if peer_review_ref is present
                    if upd.peer_review_ref:
                        approved = pending_queue.approve(
                            upd.update_id,
                            peer_review_ref=upd.peer_review_ref,
                        )
                        if approved:
                            core_store.apply_update(
                                memory_id=upd.target_memory_id,
                                new_content=upd.proposed_content,
                                peer_review_ref=upd.peer_review_ref,
                            )
                            core_updates_applied += 1

        # 4.4: Cross-batch pattern synthesis
        all_novel = []
        for batch, proc in batch_outputs:
            for pat in batch.novel_patterns:
                pat["source_subject"] = batch.subject
                all_novel.append(pat)

        # Detect cross-domain patterns (shared pattern names across subjects)
        name_to_subjects: Dict[str, List[str]] = {}
        for pat in all_novel:
            name = pat.get("name", "")
            if name:
                name_to_subjects.setdefault(name, []).append(
                    pat.get("source_subject", "")
                )
        for name, subjects in name_to_subjects.items():
            unique_subjects = list(set(subjects))
            if len(unique_subjects) > 1:
                meta_patterns.append({
                    "type": "cross_domain",
                    "pattern_name": name,
                    "subjects": unique_subjects,
                })

        log.info("Phase 4 complete: %d core updates applied, %d lessons pending, "
                 "%d meta-patterns",
                 core_updates_applied, lessons_pending, len(meta_patterns))

        return {
            "core_updates_applied": core_updates_applied,
            "lessons_pending": lessons_pending,
            "meta_patterns": meta_patterns,
        }

    # ===================================================================
    # Phase 5: Output & Storage
    # ===================================================================

    def _phase5_output(
        self,
        summary: HomeworkRunSummary,
        unprocessed: List[LearningLogEntry],
        batch_outputs: List[Tuple[ProcessedBatch, ProcessingOutput]],
        session: SessionState,
    ) -> None:
        """Write results to LTMM and mark logs processed.

        Steps:
          5.1 Mark processed learning logs
          5.2 Write homework run summary to overview_logs
          5.3 Reflective Mode handoff (if fallacy/bias flags present)
        """
        log.info("Phase 5: Output & storage.")

        # 5.1: Mark all unprocessed logs as processed
        if unprocessed:
            processed_ids = [e.turn_id for e in unprocessed]
            count = self._learning_log.mark_processed(processed_ids)
            log.info("Phase 5: Marked %d learning log entries as processed.", count)

        # 5.2: Write summary to overview_logs
        if self._memory is not None:
            overview_store = getattr(
                getattr(self._memory, "thoughts", None), "overview_logs", None
            )
            if overview_store is not None:
                try:
                    from zados.memory.long_term.thoughts.types import OverviewLogEntry
                    overview_entry = OverviewLogEntry(
                        session_id=session.session_id,
                        summary=(
                            f"Homework run: {summary.batches_processed} batches, "
                            f"{summary.lessons_validated} validated, "
                            f"{summary.contradictions_resolved} contradictions resolved, "
                            f"{summary.questions_resolved} questions resolved, "
                            f"{summary.core_memory_updates_applied} core updates"
                        ),
                        tags=["homework", "meta_learning"],
                    )
                    overview_store.write(overview_entry)
                except Exception as e:
                    log.debug("Failed to write homework overview: %s", e)

        # 5.3: Reflective Mode handoff
        if summary.fallacy_bias_flags or summary.meta_patterns:
            reflective_input = ReflectiveModeInput(
                fallacy_flags=[
                    f for f in summary.fallacy_bias_flags
                    if f.get("type") == "fallacy"
                ],
                bias_flags=[
                    f for f in summary.fallacy_bias_flags
                    if f.get("type") == "bias"
                ],
                meta_patterns=summary.meta_patterns,
                source_homework_session=session.session_id,
            )
            # Store for Reflective Mode to pick up [SPEC NEEDED — delivery mechanism]
            log.info("Phase 5: Prepared ReflectiveModeInput with %d fallacy flags, "
                     "%d bias flags, %d meta-patterns.",
                     len(reflective_input.fallacy_flags),
                     len(reflective_input.bias_flags),
                     len(reflective_input.meta_patterns))

    # ===================================================================
    # Neurochem read-only diagnostics
    # ===================================================================

    def _read_neurochem_deficit_bias(self) -> Dict[str, float]:
        """Read current NT metrics to bias deficit profiling (read-only).

        Maps neurosymbolic metrics to reward domain adjustments:
          - Low empathy / social_engagement → human_attunement deficit
          - High cognitive_rigidity → innovation deficit
          - High anxiety → ethics deficit (risk-avoidant)
          - Low precision → logic deficit

        Returns
        -------
        Dict[str, float]
            domain → bias adjustment (negative = deeper deficit signal).
            Empty dict if neurochem not available.
        """
        if self._neurochem is None:
            return {}

        try:
            readout = self._neurochem.get_neurosymbolic_readout()
            if isinstance(readout, dict):
                metrics = readout
            elif hasattr(readout, "as_dict"):
                metrics = readout.as_dict()
            else:
                return {}

            bias: Dict[str, float] = {}

            # Low empathy/social_engagement → human_attunement deficit
            empathy = metrics.get("empathy", 0.5)
            social = metrics.get("social_engagement", 0.5)
            attunement_signal = (empathy + social) / 2.0
            if attunement_signal < 0.4:
                bias["human_attunement"] = -(0.4 - attunement_signal)

            # High cognitive_rigidity → innovation deficit
            rigidity = metrics.get("cognitive_rigidity", 0.5)
            if rigidity > 0.6:
                bias["innovation"] = -(rigidity - 0.6)

            # High anxiety → ethics deficit (risk-avoidant reasoning)
            anxiety = metrics.get("anxiety", 0.5)
            if anxiety > 0.6:
                bias["ethics"] = -(anxiety - 0.6) * 0.5

            # Low precision → logic deficit
            precision = metrics.get("precision", 0.5)
            if precision < 0.4:
                bias["logic"] = -(0.4 - precision)

            if bias:
                log.info("NT deficit bias: %s", {k: round(v, 3) for k, v in bias.items()})

            return bias

        except Exception:
            log.debug("Neurochem deficit readout failed (non-critical).", exc_info=True)
            return {}

    # ===================================================================
    # Helper methods
    # ===================================================================

    @staticmethod
    def _compute_entry_relevance(entry: LearningLogEntry) -> float:
        """Score an entry by information density for processing priority.

        Factors: patterns detected, novel entries, contradictions.
        """
        score = 0.0
        score += entry.patterns_detected * 0.3
        score += entry.novel_entries * 0.4
        score += entry.contradictions * 0.5
        score += entry.confirmations * 0.1
        score += entry.extensions * 0.2
        # Penalize empty entries
        if score == 0.0:
            score = 0.01
        return min(score, 5.0)  # cap at 5.0

    def _contrast_against_knowledge(
        self,
        entries: List[LearningLogEntry],
        subject: str,
    ) -> List[Dict[str, Any]]:
        """Run Memory Contrast to find contradictions with existing knowledge.

        Also searches the LibraryStore for relevant reference material to
        enrich the contrast with book/document context.

        Returns list of contradiction candidates.
        """
        candidates: List[Dict[str, Any]] = []
        contrast = getattr(self._memory, "contrast", None)
        if contrast is None:
            return candidates

        # Build query from entry content
        query_parts = [subject]
        for entry in entries[:3]:
            if entry.contrast_deltas:
                query_parts.append(str(entry.contrast_deltas))
            if entry.e19_patterns:
                for pat in entry.e19_patterns[:2]:
                    query_parts.append(pat.get("name", ""))

        query_text = " ".join(query_parts)

        # --- Library reference enrichment ---
        # Search the library for relevant material that can inform contrast
        library = getattr(
            getattr(self._memory, "knowledge", None), "library", None
        )
        library_refs: List[Dict[str, Any]] = []
        if library is not None:
            try:
                lib_results = library.search(query_text, limit=3)
                for score, entry in lib_results:
                    if score > 0.1:
                        library_refs.append({
                            "entry_id": entry.entry_id,
                            "title":    entry.title,
                            "content":  entry.content[:300],
                            "score":    round(score, 4),
                        })
            except Exception:
                log.debug("Library search during homework contrast failed.",
                          exc_info=True)

        if library_refs:
            candidates.append({
                "type": "library_reference",
                "references": library_refs,
            })

        try:
            from zados.memory.managers.scope_filter import HOMEWORK_SCOPE
            result = contrast.contrast(
                current={"text": query_text},
                query_type="concept",
                scope_filter=HOMEWORK_SCOPE,
            )
            if result.divergence > 0.5:
                candidates.append({
                    "type": "knowledge_divergence",
                    "divergence": result.divergence,
                    "references": [
                        r.get("content", "")[:100] for r in result.references[:3]
                    ],
                })
        except Exception as e:
            log.debug("Memory contrast during homework failed: %s", e)

        return candidates

    @staticmethod
    def _answer_matches_question(
        question: UnsolvedQuestion,
        answer: str,
        batch_subject: str,
    ) -> bool:
        """Heuristic check if an answer is relevant to a question."""
        q_words = set(question.question_text.lower().split())
        a_words = set(answer.lower().split())
        overlap = len(q_words & a_words)
        # Check subject match too
        subject_match = batch_subject.lower() in question.question_text.lower()
        return overlap >= 3 or subject_match

    def _log_fallacy_bias(
        self,
        fallacy_flags: List[Dict[str, Any]],
        bias_flags: List[Dict[str, Any]],
    ) -> None:
        """Record fallacy/bias detections in SelfReflectionLog."""
        self_ref_log = getattr(self._spec_logs, "self_reflection", None)
        if self_ref_log is None:
            return

        try:
            from zados.memory.long_term.specialized_logs import SelfReflectionEntry
            for f in fallacy_flags:
                self_ref_log.record(SelfReflectionEntry(
                    observation_type="fallacy_detected",
                    severity=f.get("severity", "low"),
                    description=f"Homework detected fallacy: {f.get('name', '')}",
                ))
            for b in bias_flags:
                self_ref_log.record(SelfReflectionEntry(
                    observation_type="bias_detected",
                    severity=b.get("severity", "low"),
                    description=f"Homework detected bias: {b.get('name', '')}",
                ))
        except Exception as e:
            log.debug("Failed to log fallacy/bias to SelfReflectionLog: %s", e)

    def _log_contradictions(
        self,
        resolved: List[Dict[str, Any]],
        unresolved: List[Dict[str, Any]],
    ) -> None:
        """Record contradiction analysis in ContradictionLog."""
        contradiction_log = getattr(self._spec_logs, "contradictions", None)
        if contradiction_log is None:
            return

        try:
            from zados.memory.long_term.specialized_logs import ContradictionEntry
            for c in resolved:
                contradiction_log.record(ContradictionEntry(
                    statement_a=c.get("statement_a", ""),
                    statement_b=c.get("statement_b", ""),
                    severity=c.get("severity", "low"),
                    resolution_status="resolved",
                    resolution_method=c.get("method", "homework_processing"),
                ))
            for c in unresolved:
                contradiction_log.record(ContradictionEntry(
                    statement_a=c.get("statement_a", c.get("formulation", "")),
                    statement_b=c.get("statement_b", ""),
                    severity=c.get("severity", "medium"),
                    resolution_status="unresolved",
                ))
        except Exception as e:
            log.debug("Failed to log contradictions: %s", e)

    # ===================================================================
    # Academic buffer helpers
    # ===================================================================

    def _get_academic_buffer(self) -> Any:
        """Return AcademicBufferStore via memory.knowledge.academic_buffer, or None."""
        if self._memory is None:
            return None
        return getattr(
            getattr(self._memory, "knowledge", None), "academic_buffer", None
        )

    def _tick_academic_buffer(self) -> None:
        """Increment stagnation counters on all unresolved academic buffer entries."""
        buf = self._get_academic_buffer()
        if buf is not None:
            buf.tick_all()
            log.info("Academic buffer: ticked %d total entries.", len(buf))

    # ===================================================================
    # Journal write
    # ===================================================================

    def set_journal_store(self, store: Any) -> None:
        """Wire the JournalStore after construction (avoids __init__ breaking change)."""
        self._journal_store = store

    def _write_journal(self, summary: HomeworkRunSummary, session: SessionState) -> None:
        """Write a PERIODIC journal entry summarising the homework run."""
        if self._journal_store is None:
            return
        try:
            from zados.memory.long_term.journal.entry import JournalEntry, JournalTrigger
            from zados.core.tags import T

            emphasis_subjects = list(summary.processing_emphasis.keys())[:5]
            notes = [
                "pipeline:homework",
                f"batches_processed:{summary.batches_processed}",
                f"lessons_validated:{summary.lessons_validated}",
                f"contradictions_resolved:{summary.contradictions_resolved}",
                f"questions_resolved:{summary.questions_resolved}",
                f"core_updates:{summary.core_memory_updates_applied}",
                f"dream_candidates_flagged:{summary.dream_candidates_flagged}",
            ] + [f"subject:{s}" for s in emphasis_subjects]

            prose = (
                f"Homework processing complete. "
                f"{summary.batches_processed} subject batches processed "
                f"({', '.join(emphasis_subjects) or 'no subjects'}). "
                f"{summary.lessons_validated} lessons validated, "
                f"{summary.contradictions_resolved} contradictions resolved, "
                f"{summary.questions_resolved} questions resolved. "
            )
            if summary.core_memory_updates_applied:
                prose += (
                    f"{summary.core_memory_updates_applied} core memory updates applied. "
                )
            if summary.dream_candidates_flagged:
                prose += (
                    f"{summary.dream_candidates_flagged} items flagged for dream processing. "
                )
            if summary.fallacy_bias_flags:
                prose += (
                    f"{len(summary.fallacy_bias_flags)} fallacy/bias issues detected — "
                    "queued for reflective processing."
                )

            entry = JournalEntry(
                session_id=session.session_id,
                trigger=JournalTrigger.PERIODIC,
                trigger_source="homework_pipeline",
                prose=prose,
                pipeline_notes=notes,
                tags=[
                    T.pipeline("homework"),
                    T.mode("homework"),
                    T.content("academic"),
                ] + [f"subject:{s}" for s in emphasis_subjects],
            )
            self._journal_store.write(entry)
            log.info("Homework: journal entry written (trigger=PERIODIC).")
        except Exception:
            log.debug("Homework: journal write failed.", exc_info=True)

    def _write_identity_journal_if_relevant(
        self,
        session: SessionState,
        run_summary: HomeworkRunSummary,
    ) -> None:
        """Write to IdentityJournalStore if homework found identity-relevant patterns."""
        if self._memory is None:
            return
        identity = getattr(self._memory, "identity", None)
        if identity is None:
            return
        journal_store = getattr(identity, "journal", None)
        if journal_store is None:
            return

        # Check if any contradictions or identity-touching patterns were found
        identity_relevant = []
        for batch in getattr(run_summary, "processed_batches", []):
            for contradiction in getattr(batch, "resolved_contradictions", []):
                if isinstance(contradiction, dict) and contradiction.get("identity_relevant"):
                    identity_relevant.append(contradiction)

        # Also check if reflective mode input was generated (signals identity tension)
        reflective_input = getattr(run_summary, "reflective_input", None)
        if reflective_input is not None:
            identity_relevant.append({"type": "reflective_handoff", "content": "Homework generated reflective mode input"})

        if not identity_relevant:
            return

        try:
            from zados.memory.long_term.identity.types import (
                IdentityJournalEntry,
                IdentityJournalEntryType,
            )
            content_parts = [f"Homework session findings ({len(identity_relevant)} identity-relevant items):"]
            for item in identity_relevant[:5]:
                if isinstance(item, dict):
                    content_parts.append(f"  - {item.get('type', 'finding')}: {item.get('content', '')[:200]}")
            entry = IdentityJournalEntry(
                entry_type=IdentityJournalEntryType.REGULAR,
                content="\n".join(content_parts),
                source_pipeline="homework",
                tags=["homework", "identity_relevant"],
            )
            journal_store.write(entry)
            log.debug("Homework pipeline wrote identity journal entry: %s", entry.entry_id)
        except Exception:
            log.debug("Identity journal write failed in homework pipeline.", exc_info=True)

    @staticmethod
    def _build_return(
        summary: HomeworkRunSummary,
        unprocessed: List[LearningLogEntry],
        start_time: float,
    ) -> Dict[str, Any]:
        """Build the return dict from the homework run."""
        return {
            "status": "completed",
            "session_id": summary.session_id,
            "processing_time_s": round(time.time() - start_time, 2),
            "batches_processed": summary.batches_processed,
            "logs_processed": len(unprocessed),
            "lessons_validated": summary.lessons_validated,
            "lessons_pending": summary.lessons_pending,
            "contradictions_resolved": summary.contradictions_resolved,
            "contradictions_unresolved": summary.contradictions_unresolved,
            "questions_resolved": summary.questions_resolved,
            "questions_new": summary.questions_new,
            "dream_candidates_flagged": summary.dream_candidates_flagged,
            "core_memory_updates_applied": summary.core_memory_updates_applied,
            "fallacy_bias_flags": len(summary.fallacy_bias_flags),
            "meta_patterns": summary.meta_patterns,
            "processing_emphasis": summary.processing_emphasis,
        }
