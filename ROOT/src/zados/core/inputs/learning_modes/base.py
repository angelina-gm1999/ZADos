"""
ZA-DOS v0.6 — Learning Mode Pipeline ABC (spec §3.3 + Part 2 §§2-6 + Part 4 §1.1).

Abstract base for all 5 learning mode pipelines.  Provides shared
infrastructure including:

A) 9-stage pipeline skeleton (Part 4 §1.1):
  Stage 0: Setup — preset, scope, drift check, engine resolve
  Stage 1: Memory contrast (scoped read via PipelineScope)
  Stage 2: Engine dispatch (tier-filtered)
  Stage 3: VT thinking pass (with held-thinking-block check)
  Stage 4: Mode-specific processing (abstract — subclass hook)
  Stage 5: LTMM write (scoped write via PipelineScope)
  Stage 6: Unsolved/question extraction
  Stage 7: Response generation
  Stage 8: NT feedback + homeostatic + MIM write

B) 10-step neurochemical emotional feedback loop (Part 2 §2.1):
  Step  1: Apply EmotionalPreset for current mode
  Step  2: Run EmotionalDetection (E28) on user input
  Step  3: Translate emotions → NT signals (dual path: speed 12 + full 46)
  Step  4: Update EmotionTracker (leaky integrators)
  Step  5: Compute NeurochemicalMetrics from updated state
  Step  6: Metrics → Engine priority weights
  Step  7: Combine engine weights (toolkit tiers + NT weights + intent)
  Step  8: Evaluation → NT feedback (closes loop)
  Step  9: Homeostatic bounds check (E27)
  Step 10: Risk emotion check against mode thresholds

C) Held Thinking Block detection — emotion-interrupted thought
   fragments stored directly to LTMM when any single emotion from
   the 46-taxonomy exceeds 0.6, or any identity-relevant emotion is
   detected at any intensity.

All neurochem dependencies are OPTIONAL — the pipeline degrades gracefully
if they aren't wired in.  This ensures backward compatibility with
existing code that constructs LearningModePipeline subclasses without
neurochem references.
"""
from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from zados.core.mode_profiles import profile_for_learning_mode
from zados.core.processes.context_anchor import ContextAnchorManager
from zados.core.processes.emotional_landscape import (
    apply_oscillatory_bias,
    apply_preset_to_bundle,
    apply_preset_to_neurochem,
    get_emotional_preset,
)
from zados.core.processes.engine_toolkit import EngineToolkit
from zados.core.processes.learning_log import LearningLogPipeline
from zados.core.processes.subject_classifier import classify_subject_from_text
from zados.core.processes.unsolved_buffer import UnsolvedBuffer
from zados.core.types import (
    EngineTier,
    InputBundle,
    LearningModeConfig,
    LearningModeResult,
    PipelineResult,
    SessionState,
    SubjectCategory,
)

log = logging.getLogger(__name__)

# Default time step for emotion tracker integration (seconds).
_DT = 0.01

# ------------------------------------------------------------------
# Held Thinking Block — emotion threshold for interrupt
# ------------------------------------------------------------------
# Any single emotion from the 46-taxonomy exceeding this triggers
# a direct LTMM write (Part 4 §1.1 Stage 3).
HELD_BLOCK_EMOTION_THRESHOLD = 0.6

# Identity-relevant emotions — any intensity triggers a held block
# because they directly relate to ZA-DOS's developing self-model.
# Drawn from the self_evaluation + trust_relational groups in E28.
IDENTITY_RELEVANT_EMOTIONS: Set[str] = {
    # self_evaluation group
    "ashamed", "guilty", "regret", "critical",
    # trust_relational (identity-touching)
    "betrayal", "rejected", "isolated",
    # existential subset
    "grief", "numb",
    # positive identity-forming
    "proud", "respected", "belonging", "accepted",
}

# ------------------------------------------------------------------
# Per-mode pipeline configuration (Part 4 §1.1)
# ------------------------------------------------------------------

MODE_CONFIGS: Dict[str, LearningModeConfig] = {
    "M1": LearningModeConfig(
        semantic_expansion_max_hops=2,
        pattern_chain_max_depth=2,
        max_questions_per_turn=2,
        response_depth="full",
        generate_response=True,
        use_retroactive_contrast=False,
        contradiction_mode="learning",
    ),
    "M2": LearningModeConfig(
        semantic_expansion_max_hops=3,
        pattern_chain_max_depth=3,
        max_questions_per_turn=0,           # M2 doesn't generate questions
        response_depth="full",
        generate_response=True,
        use_retroactive_contrast=True,      # Two-pass contrast (§3.2)
        contradiction_mode="soft",          # Soft contradiction — no adversarial
    ),
    "M3": LearningModeConfig(
        semantic_expansion_max_hops=-1,     # Unlimited
        pattern_chain_max_depth=-1,         # Unlimited
        max_questions_per_turn=-1,          # Unlimited
        response_depth="full",
        generate_response=True,
        use_retroactive_contrast=False,
        contradiction_mode="learning",
    ),
    "M4": LearningModeConfig(
        semantic_expansion_max_hops=3,
        pattern_chain_max_depth=2,
        max_questions_per_turn=1,           # One focused question per turn
        response_depth="abbreviated",       # Question-oriented, not full
        generate_response=True,
        use_retroactive_contrast=False,
        contradiction_mode="learning",
    ),
    "M5": LearningModeConfig(
        semantic_expansion_max_hops=3,
        pattern_chain_max_depth=3,
        max_questions_per_turn=2,
        response_depth="none",              # Autonomous — no response to human
        generate_response=False,            # M5 does not produce output text
        use_retroactive_contrast=False,
        contradiction_mode="learning",
    ),
}


class LearningModePipeline(abc.ABC):
    """Abstract base for learning mode pipelines M1-M5.

    Parameters
    ----------
    answer_pipeline : AnswerPipeline
        The v0.5 pipeline to delegate to.
    learning_log : LearningLogPipeline
    unsolved_buffer : UnsolvedBuffer
    context_manager : ContextAnchorManager, optional
    engines : dict, optional
        Cognitive engine instances keyed by engine_id.
    neurochem_engine : NeurochemicalEngine, optional
        Live neurochemical simulation engine (Part 2 §2.1 steps 1/3/8).
    extractor_orchestrator : ExtractorOrchestrator, optional
        Full 9-step stochastic pathway (Part 2 §3.3 — M3 primarily).
    emotion_tracker_state : EmotionTrackerState, optional
        Leaky-integrator state for emotion saturation tracking.
    """

    # Subclasses must set these
    mode_id: str = ""           # "M1".."M5"
    mode_number: int = 0        # 1..5

    def __init__(
        self,
        answer_pipeline: Any,
        learning_log: LearningLogPipeline,
        unsolved_buffer: UnsolvedBuffer,
        context_manager: Optional[ContextAnchorManager] = None,
        engines: Optional[Dict[int, Any]] = None,
        neurochem_engine: Any = None,
        extractor_orchestrator: Any = None,
        emotion_tracker_state: Any = None,
        held_block_store: Any = None,
        pipeline_scope: Any = None,
        journal_writer: Any = None,
        memory: Any = None,
    ) -> None:
        self._pipeline = answer_pipeline
        self._learning_log = learning_log
        self._unsolved_buffer = unsolved_buffer
        self._context = context_manager or ContextAnchorManager()
        self._engines = engines or {}
        self._toolkit = EngineToolkit()

        # Part 2 neurochem wiring (all optional)
        self._neurochem = neurochem_engine
        self._extractor = extractor_orchestrator
        self._emotion_tracker = emotion_tracker_state

        # Part 4 wiring (all optional)
        self._held_block_store = held_block_store  # HeldThinkingBlockStore
        self._pipeline_scope = pipeline_scope      # PipelineScope (read/write scopes)
        self._journal_writer = journal_writer      # JournalWriter for event-log writes

        # LTMM knowledge/thoughts stores (resolved from memory layer)
        self._memory = memory
        self._lesson_store = None
        self._notebook_store = None
        self._academic_question_store = None
        self._general_question_store = None
        self._knowledge_map_store = None
        self._library_store = None
        if self._memory is not None:
            knowledge = getattr(self._memory, "knowledge", None)
            thoughts = getattr(self._memory, "thoughts", None)
            if knowledge is not None:
                self._lesson_store = getattr(knowledge, "lessons", None)
                self._notebook_store = getattr(knowledge, "notebook", None)
                self._library_store = getattr(knowledge, "library", None)
                self._academic_question_store = getattr(
                    knowledge, "academic_questions", None
                )
                self._knowledge_map_store = getattr(
                    knowledge, "knowledge_maps", None
                )
            if thoughts is not None:
                self._general_question_store = getattr(
                    thoughts, "general_questions", None
                )

        # Per-mode config (Part 4 §1.1)
        self._config: LearningModeConfig = MODE_CONFIGS.get(
            self.mode_id, LearningModeConfig()
        )

        # Convenience references to specific engines (from engines dict)
        self._e28 = self._engines.get(28)    # EmotionalDetectionEngine
        self._e27 = self._engines.get(27)    # NeurochemHomeostaticEngine
        self._e17 = self._engines.get(17)    # RewardBasedLearningEngine

    @abc.abstractmethod
    def process_turn(
        self,
        bundle: InputBundle,
        session: SessionState,
    ) -> LearningModeResult:
        """Process a learning mode turn.  Must be overridden."""
        ...

    # ==================================================================
    # 9-STAGE PIPELINE SKELETON (Part 4 §1.1)
    # ==================================================================

    def _run_pipeline_skeleton(
        self,
        bundle: InputBundle,
        session: SessionState,
    ) -> LearningModeResult:
        """Shared 9-stage pipeline template (Part 4 §1.1).

        Subclass ``process_turn()`` may call this directly or override
        individual stages.  The skeleton is designed so subclasses can
        override specific ``_stage_N_*`` hooks while keeping the rest.

        Stages
        ------
        0. Setup — emotional preset, engine resolve, scope, drift
        1. Memory contrast (scoped read)
        2. Engine dispatch
        3. VT thinking pass + held-thinking-block check
        4. Mode-specific processing (abstract hook)
        5. LTMM write (scoped write)
        6. Unsolved / question extraction
        7. Response generation
        8. NT feedback + homeostatic + MIM write

        Returns
        -------
        LearningModeResult
        """
        # ---- Stage 0: Setup ----
        subject = classify_subject_from_text(bundle.raw_text)
        bundle = self._stage0_setup(bundle, session, subject)

        # ---- Stage 1: Memory contrast (scoped) ----
        contrast_result = self._stage1_memory_contrast(bundle)

        # ---- Stage 2: Engine dispatch ----
        result = self._stage2_engine_dispatch(bundle, session)

        # ---- Stage 3: VT thinking + held block check ----
        emotion_profile = self._step2_detect_emotions(bundle, result)
        held_block_ids = self._check_held_thinking_block(
            emotion_profile=emotion_profile,
            thinking_trace=getattr(
                result.state.thinking if result.state else None,
                "thinking_trace", "",
            ),
            bundle=bundle,
            session=session,
        )

        # ---- Stage 4: Mode-specific processing (subclass hook) ----
        mode_data = self._stage4_mode_specific(
            bundle, session, result, contrast_result, emotion_profile,
        )

        # ---- Stage 5: LTMM write (scoped) ----
        self._stage5_ltmm_write(bundle, session, result, mode_data)

        # ---- Stage 6: Unsolved / question extraction ----
        unsolved = self._stage6_extract_questions(
            bundle, session, result, mode_data,
        )

        # ---- Stage 7: Response generation ----
        if not self._config.generate_response:
            # M5 autonomous mode — suppress response
            if result.state and result.state.thinking:
                result.response = ""
            log.debug("%s: response generation suppressed (autonomous mode).", self.mode_id)

        # ---- Stage 8: NT feedback + homeostatic + MIM write ----
        feedback = self._run_feedback_loop(bundle, result, session)
        self._record_learning(session, result, subject=subject.value)

        # ---- Stage 8b: Journal write (learning event log) ----
        self._stage8b_journal_write(bundle, session, result, subject)

        # Build result
        return LearningModeResult(
            mode_number=self.mode_number,
            pipeline_result=result,
            learning_entries=self._learning_log.get_recent(session.session_id),
            unsolved_questions=unsolved,
            held_thinking_blocks=held_block_ids,
            **mode_data.get("result_extras", {}),
        )

    # ------------------------------------------------------------------
    # Per-stage hooks (override in subclasses)
    # ------------------------------------------------------------------

    def _stage0_setup(
        self,
        bundle: InputBundle,
        session: SessionState,
        subject: SubjectCategory,
    ) -> InputBundle:
        """Stage 0: Preset, scope, engine resolve, drift check.

        Parameters
        ----------
        bundle : InputBundle
        session : SessionState
        subject : SubjectCategory

        Returns
        -------
        InputBundle (mutated)
        """
        # Apply emotional preset (NT baseline for this mode)
        bundle = self._apply_emotional_preset(bundle)

        # Resolve engine tiers for mode × subject
        bundle = self._resolve_engines(bundle, subject)

        # Attach pipeline scope info to bundle context flags
        if self._pipeline_scope is not None:
            bundle.context_flags["has_pipeline_scope"] = True
            bundle.context_flags["pipeline_name"] = self._pipeline_scope.pipeline_name

        # Drift check — re-anchor if topic shifted
        self._check_drift(bundle)

        return bundle

    def _stage1_memory_contrast(
        self, bundle: InputBundle,
    ) -> Optional[Any]:
        """Stage 1: Scoped memory contrast read.

        If a PipelineScope is wired, passes the read_scope to
        MemoryContrast for namespaced retrieval.  Otherwise falls back
        to the flat LTMM scan via the AnswerPipeline.

        Returns
        -------
        contrast result or None
        """
        # Scoped contrast is handled by the AnswerPipeline when it
        # calls MemoryContrast internally.  We pass scope via bundle.
        if self._pipeline_scope is not None:
            # Store scope on bundle for downstream contrast use
            bundle_scope = getattr(bundle, "_pipeline_read_scope", None)
            if bundle_scope is None:
                # Attach as ad-hoc attribute (backward-compatible)
                bundle._pipeline_read_scope = self._pipeline_scope.read_scope  # type: ignore[attr-defined]

        return None  # Actual contrast happens inside AnswerPipeline

    def _stage2_engine_dispatch(
        self,
        bundle: InputBundle,
        session: SessionState,
    ) -> PipelineResult:
        """Stage 2: Run the answer pipeline (includes engine dispatch).

        Returns
        -------
        PipelineResult
        """
        return self._pipeline.process(bundle)

    def _stage4_mode_specific(
        self,
        bundle: InputBundle,
        session: SessionState,
        result: PipelineResult,
        contrast_result: Any,
        emotion_profile: Dict[str, float],
    ) -> Dict[str, Any]:
        """Stage 4: Mode-specific processing hook.

        Subclasses MUST override this to implement mode-specific logic
        (e.g., M2 peer-review contrast, M3 challenge, M4 question
        routing, M5 boredom detection).

        Returns
        -------
        Dict[str, Any]
            Mode-specific data.  May include "result_extras" key with
            a dict of extra kwargs for LearningModeResult.
        """
        return {}

    def _stage8b_journal_write(
        self,
        bundle: InputBundle,
        session: SessionState,
        result: PipelineResult,
        subject: Any,
    ) -> None:
        """Stage 8b: Write an event-log journal entry for this learning turn.

        Uses PERIODIC trigger for every learning turn.  Notes carry the mode
        ID and academic subject so journal entries are traceable to the
        learning context.  Writes to the shared JournalStore (general
        cognitive journal, not identity journal).
        """
        if self._journal_writer is None:
            return
        try:
            from zados.memory.long_term.journal.entry import JournalContext, JournalTrigger

            stmm = None
            state = getattr(result, "state", None)
            if state is not None:
                stmm = getattr(state, "stmm", None)
            if stmm is None:
                return

            subject_str = (
                subject.value if hasattr(subject, "value") else str(subject)
            )
            turn_index = getattr(session, "turn_index", 0)

            ctx = JournalContext(
                trigger=JournalTrigger.PERIODIC,
                trigger_source=f"learning_mode_{self.mode_id.lower()}",
                stmm=stmm,
                notes=[
                    f"mode:{self.mode_id}",
                    f"subject:{subject_str}",
                ],
                turn_range=(turn_index, turn_index),
                session_id=session.session_id,
            )
            self._journal_writer.write(ctx)
        except Exception:
            log.debug("Journal write failed in learning mode %s.", self.mode_id)

    def _stage5_ltmm_write(
        self,
        bundle: InputBundle,
        session: SessionState,
        result: PipelineResult,
        mode_data: Dict[str, Any],
    ) -> None:
        """Stage 5: Scoped LTMM write.

        Default implementation writes:
        - LessonEntry to LessonStore (if validated insights found)
        - NotebookEntry to NotebookStore (academic journaling)
        Subclasses may override for mode-specific writes.
        """
        subject = mode_data.get("subject", "")
        if isinstance(subject, str):
            subject_str = subject
        else:
            subject_str = getattr(subject, "value", str(subject))

        # --- Write validated lessons ---
        validated = mode_data.get("validated_lessons", [])
        if validated and self._lesson_store is not None:
            try:
                from zados.memory.long_term.knowledge.types import LessonEntry
                for lesson_info in validated:
                    content = (
                        lesson_info.get("content", "")
                        if isinstance(lesson_info, dict)
                        else str(lesson_info)
                    )
                    if not content:
                        continue
                    confidence = (
                        lesson_info.get("confidence", 0.5)
                        if isinstance(lesson_info, dict)
                        else 0.5
                    )
                    lesson = LessonEntry(
                        content=content,
                        subject_category=subject_str,
                        source_mode=self.mode_id,
                        confidence=confidence,
                        validation_status="pending",
                        tags=[f"mode:{self.mode_id}", f"subject:{subject_str}"],
                    )
                    self._lesson_store.write(lesson)
                    log.info("%s: wrote lesson to LessonStore: %s",
                             self.mode_id, lesson.lesson_id)
            except Exception:
                log.debug("%s: LessonStore write failed.", self.mode_id, exc_info=True)

        # --- Bootstrap KnowledgeMap if new subject ---
        if validated and self._knowledge_map_store is not None and subject_str:
            try:
                from zados.memory.long_term.knowledge.types import (
                    KnowledgeMap,
                    KnowledgeNode,
                )
                # Check if a map already exists for this subject
                existing = self._knowledge_map_store.search(subject_str, limit=1)
                if not existing:
                    # Create initial map with root concept node
                    root_node = KnowledgeNode(
                        label=subject_str,
                        node_type="concept",
                        confidence=0.3,  # Low initial confidence
                    )
                    lesson_ids = []
                    for lesson_info in validated:
                        if isinstance(lesson_info, dict) and lesson_info.get("lesson_id"):
                            lesson_ids.append(lesson_info["lesson_id"])

                    kmap = KnowledgeMap(
                        title=f"{subject_str} — Knowledge Map",
                        subject_category=subject_str,
                        description=f"Auto-created from {self.mode_id} learning session.",
                        nodes=[root_node],
                        contributing_lessons=lesson_ids,
                        tags=[f"mode:{self.mode_id}", f"subject:{subject_str}", "auto_bootstrap"],
                    )
                    self._knowledge_map_store.write(kmap)
                    log.info(
                        "%s: bootstrapped KnowledgeMap for subject '%s': %s",
                        self.mode_id, subject_str, kmap.map_id,
                    )
            except Exception:
                log.debug("%s: KnowledgeMap bootstrap failed.", self.mode_id, exc_info=True)

        # --- Write notebook entry (academic journaling) ---
        if self._notebook_store is not None and result.response:
            try:
                from zados.memory.long_term.knowledge.types import NotebookEntry
                nt_snap = self._build_nt_snapshot_for_held_block()
                note = NotebookEntry(
                    content=(
                        f"[{self.mode_id}] Input: {bundle.raw_text[:150]}... "
                        f"→ {result.response[:300]}"
                    ),
                    subject_category=subject_str,
                    source_mode=self.mode_id,
                    nt_snapshot=nt_snap,
                    tags=[f"mode:{self.mode_id}", f"subject:{subject_str}"],
                )
                self._notebook_store.write(note)
            except Exception:
                log.debug("%s: NotebookStore write failed.", self.mode_id, exc_info=True)

        # --- Write identity journal entry if identity-relevant emotions detected ---
        emotion_profile = mode_data.get("emotion_profile", {})
        if emotion_profile and self._memory is not None:
            identity_emotions = {
                k: v for k, v in emotion_profile.items()
                if k in IDENTITY_RELEVANT_EMOTIONS and v > 0.0
            }
            if identity_emotions:
                identity = getattr(self._memory, "identity", None)
                ij_store = getattr(identity, "journal", None) if identity else None
                if ij_store is not None:
                    try:
                        from zados.memory.long_term.identity.types import (
                            IdentityJournalEntry,
                            IdentityJournalEntryType,
                        )
                        dominant_emotion = max(identity_emotions, key=identity_emotions.get)
                        entry = IdentityJournalEntry(
                            entry_type=IdentityJournalEntryType.REGULAR,
                            content=(
                                f"[{self.mode_id}] Identity-relevant emotion detected: "
                                f"{dominant_emotion} ({identity_emotions[dominant_emotion]:.2f}). "
                                f"Context: {bundle.raw_text[:200]}"
                            ),
                            emotion_tags=list(identity_emotions.keys()),
                            source_pipeline=f"learning_mode_{self.mode_id.lower()}",
                            nt_snapshot=self._build_nt_snapshot_for_held_block(),
                            tags=[f"mode:{self.mode_id}", f"emotion:{dominant_emotion}"],
                        )
                        ij_store.write(entry)
                        log.debug(
                            "%s: wrote identity journal entry for emotion %s",
                            self.mode_id, dominant_emotion,
                        )
                    except Exception:
                        log.debug(
                            "%s: IdentityJournalStore write failed.",
                            self.mode_id, exc_info=True,
                        )

    def _stage6_extract_questions(
        self,
        bundle: InputBundle,
        session: SessionState,
        result: PipelineResult,
        mode_data: Dict[str, Any],
    ) -> List[Any]:
        """Stage 6: Extract unsolved questions from engine results.

        Writes to three targets:
          1. UnsolvedBuffer (in-session priority queue — always available)
          2. GeneralQuestionStore (LTMM — non-academic / identity / relational)
          3. AcademicQuestionStore (LTMM — domain-specific knowledge gaps)

        Question sources:
          - mode_data["open_questions"]: explicit questions from mode-specific logic
          - mode_data["knowledge_gaps"]: domain gaps identified during processing
          - result.state.dispatch.unsolved_flags: engine-flagged unsolved items

        Returns list of UnsolvedQuestion instances added to the buffer.
        """
        from zados.core.tags import T

        questions_added: List[Any] = []
        subject = mode_data.get("subject", "")
        if not isinstance(subject, str):
            subject = getattr(subject, "value", str(subject))

        max_q = self._config.max_questions_per_turn
        if max_q == 0:
            # Mode explicitly suppresses question generation (e.g. M2)
            return questions_added

        # --- Collect raw question candidates ---
        candidates: List[Dict[str, Any]] = []

        # Source A: mode-specific open questions
        for q in mode_data.get("open_questions", []):
            if isinstance(q, dict):
                candidates.append(q)
            elif isinstance(q, str):
                candidates.append({"text": q, "source": "mode_generated"})

        # Source B: knowledge gaps
        for gap in mode_data.get("knowledge_gaps", []):
            if isinstance(gap, dict):
                gap.setdefault("source", "knowledge_gap")
                candidates.append(gap)
            elif isinstance(gap, str):
                candidates.append({"text": gap, "source": "knowledge_gap"})

        # Source C: engine-flagged unsolved items
        state = getattr(result, "state", None)
        dispatch = getattr(state, "dispatch", None) if state else None
        unsolved_flags = getattr(dispatch, "unsolved_flags", []) if dispatch else []
        for flag_item in unsolved_flags:
            if isinstance(flag_item, dict):
                flag_item.setdefault("source", "engine_flagged")
                candidates.append(flag_item)
            elif isinstance(flag_item, str):
                candidates.append({"text": flag_item, "source": "engine_flagged"})

        # Enforce max per turn (unless unlimited = -1)
        if max_q > 0:
            candidates = candidates[:max_q]

        # --- Write each candidate to appropriate stores ---
        for cand in candidates:
            q_text = cand.get("text", cand.get("formulation", ""))
            if not q_text:
                continue

            q_source = cand.get("source", "mode_generated")
            q_domain = cand.get("domain", subject)
            q_priority = cand.get("priority", 0.5)
            is_academic = cand.get("academic", bool(q_domain))
            q_tags = [T.pipeline(f"learning_m{self.mode_number}"), T.mode("learning")]

            # Always write to in-session UnsolvedBuffer
            uq = self._unsolved_buffer.add(
                question_text=q_text,
                source_mode=self.mode_id,
                source_context=bundle.raw_text[:150] if bundle.raw_text else "",
                urgency_score=q_priority,
                tags=q_tags,
            )
            questions_added.append(uq)

            # Write to LTMM AcademicQuestionStore (domain-scoped gaps)
            if is_academic and self._academic_question_store is not None:
                try:
                    from zados.memory.long_term.knowledge.types import AcademicQuestion
                    aq = AcademicQuestion(
                        formulation=q_text,
                        source=q_source,
                        subject_category=subject,
                        domain=q_domain,
                        priority=q_priority,
                        tags=q_tags + [T.origin("academic")],
                    )
                    self._academic_question_store.write(aq)
                    log.debug(
                        "%s: wrote AcademicQuestion %s to LTMM",
                        self.mode_id, aq.question_id,
                    )
                except Exception:
                    log.debug(
                        "%s: AcademicQuestionStore write failed.",
                        self.mode_id, exc_info=True,
                    )

            # Write non-academic questions to GeneralQuestionStore
            if not is_academic and self._general_question_store is not None:
                try:
                    from zados.memory.long_term.thoughts.types import GeneralQuestion
                    gq = GeneralQuestion(
                        formulation=q_text,
                        source=q_source,
                        domain_hint=q_domain or None,
                        priority=q_priority,
                        tags=q_tags + [T.origin("general")],
                    )
                    self._general_question_store.write(gq)
                    log.debug(
                        "%s: wrote GeneralQuestion %s to LTMM",
                        self.mode_id, gq.question_id,
                    )
                except Exception:
                    log.debug(
                        "%s: GeneralQuestionStore write failed.",
                        self.mode_id, exc_info=True,
                    )

        return questions_added

    # ==================================================================
    # HELD THINKING BLOCK DETECTION (Part 4 §1.1 Stage 3)
    # ==================================================================

    def _check_held_thinking_block(
        self,
        emotion_profile: Dict[str, float],
        thinking_trace: str,
        bundle: InputBundle,
        session: SessionState,
    ) -> List[str]:
        """Check if any emotion crosses the held-block threshold.

        Triggers on:
          1. Any single emotion from the 46-taxonomy with intensity > 0.6
          2. Any identity-relevant emotion at any positive intensity

        When triggered, captures the current thinking fragment and writes
        it directly to the HeldThinkingBlockStore (direct LTMM write,
        bypassing STMM/MTMM staging).

        Parameters
        ----------
        emotion_profile : dict
            emotion_name → intensity (0.0-1.0)
        thinking_trace : str
            Current VT thinking trace text.
        bundle : InputBundle
        session : SessionState

        Returns
        -------
        List[str]
            Block IDs of any held thinking blocks captured this turn.
        """
        if not emotion_profile or not thinking_trace:
            return []

        if self._held_block_store is None:
            return []

        captured_ids: List[str] = []

        for emotion_name, intensity in emotion_profile.items():
            if intensity <= 0.0:
                continue

            # Check trigger conditions
            is_identity_relevant = emotion_name in IDENTITY_RELEVANT_EMOTIONS
            exceeds_threshold = intensity > HELD_BLOCK_EMOTION_THRESHOLD

            if not (is_identity_relevant or exceeds_threshold):
                continue

            # Determine trigger type
            if is_identity_relevant and exceeds_threshold:
                trigger_type = "identity_and_threshold"
            elif is_identity_relevant:
                trigger_type = "identity_relevant"
            else:
                trigger_type = "threshold_exceeded"

            # Build NT snapshot (4 derived metrics if available)
            nt_snapshot = self._build_nt_snapshot_for_held_block()

            # Create the HeldThinkingBlock
            try:
                from zados.memory.long_term.thoughts.types import HeldThinkingBlock

                block = HeldThinkingBlock(
                    thought_fragment=thinking_trace,
                    emotion_tag=emotion_name,
                    emotion_trigger_type=trigger_type,
                    nt_snapshot=nt_snapshot,
                    context_summary=bundle.raw_text[:200] if bundle.raw_text else "",
                    pipeline_phase="phase4_thinking",
                    source_turn_ref=getattr(session, "turn_id", ""),
                    session_id=getattr(session, "session_id", ""),
                    tags=[f"mode:{self.mode_id}", f"emotion:{emotion_name}"],
                )

                self._held_block_store.write(block)
                captured_ids.append(block.block_id)

                log.info(
                    "Held thinking block captured in %s: emotion=%s (%.2f), "
                    "trigger=%s, block_id=%s",
                    self.mode_id, emotion_name, intensity,
                    trigger_type, block.block_id,
                )
            except Exception:
                log.warning(
                    "Failed to write held thinking block for emotion '%s'.",
                    emotion_name, exc_info=True,
                )

            # Only capture ONE block per turn per emotion — the most
            # significant trigger is recorded.  We don't break because
            # multiple emotions can each generate a block.

        return captured_ids

    def _build_nt_snapshot_for_held_block(self) -> Dict[str, float]:
        """Build the 4-metric NT snapshot for a held thinking block.

        Returns dict with keys: motivation, empathy, rigidity, fatigue.
        All 0.0 if neurochem engine is not available.
        """
        if self._neurochem is None:
            return {}

        try:
            metrics = self._step5_compute_metrics()
            if metrics is not None:
                return {
                    "motivation": getattr(metrics, "motivation", 0.0),
                    "empathy": getattr(metrics, "empathy", 0.0),
                    "rigidity": getattr(metrics, "cognitive_rigidity", 0.0),
                    "fatigue": getattr(metrics, "fatigue", 0.0),
                }
        except Exception:
            log.debug("NT snapshot for held block failed.", exc_info=True)

        return {}

    # ------------------------------------------------------------------
    # Library access helpers
    # ------------------------------------------------------------------

    def search_library(
        self, query: str, limit: int = 5,
    ) -> List[Any]:
        """Search the LibraryStore for reference material.

        Returns list of (score, LibraryEntry) tuples, or [] if the
        library store is not available.
        """
        if self._library_store is None:
            return []
        try:
            return self._library_store.search(query, limit=limit)
        except Exception:
            log.debug("Library search failed.", exc_info=True)
            return []

    # ==================================================================
    # 10-STEP NEUROCHEM FEEDBACK LOOP (Part 2 §2.1)
    # ==================================================================

    def _run_feedback_loop(
        self,
        bundle: InputBundle,
        result: PipelineResult,
        session: SessionState,
    ) -> Dict[str, Any]:
        """Run the full 10-step neurochem feedback loop post-pipeline.

        This is the primary integration point for Part 2.  Subclass
        ``process_turn()`` should call this AFTER the AnswerPipeline
        produces its result, but BEFORE returning the LearningModeResult.

        The loop degrades gracefully: if neurochem dependencies are
        absent, individual steps are skipped and logged.

        Parameters
        ----------
        bundle : InputBundle
        result : PipelineResult
        session : SessionState

        Returns
        -------
        Dict[str, Any]
            Feedback summary with keys: emotion_profile, dominant_emotion,
            metrics, nt_weights, risk_emotions, homeostatic_result, etc.
        """
        summary: Dict[str, Any] = {}

        # Step 1 — Emotional preset (NT injection)
        self._step1_apply_preset()

        # Step 2 — E28 emotion detection
        emotion_profile = self._step2_detect_emotions(bundle, result)
        summary["emotion_profile"] = emotion_profile

        # Step 3 — Emotion → NT translation (dual path)
        self._step3_translate_emotions(emotion_profile)

        # Step 4 — EmotionTracker update
        dominant = self._step4_update_tracker(emotion_profile)
        summary["dominant_emotion"] = dominant

        # Step 5 — Compute NeurochemicalMetrics
        metrics = self._step5_compute_metrics()
        summary["metrics"] = metrics

        # Step 6 — Metrics → Engine weights
        nt_weights = self._step6_metrics_to_weights(metrics)
        summary["nt_weights"] = nt_weights

        # Step 7 is handled during dispatch (combined weights already set)

        # Step 8 — Evaluation → NT feedback
        eval_results = self._build_eval_results(result)
        self._step8_feedback_to_neurochem(eval_results)
        summary["eval_results"] = eval_results

        # Step 9 — Homeostatic check (E27)
        homeostatic_result = self._step9_homeostatic_check()
        summary["homeostatic_result"] = homeostatic_result

        # Step 10 — Risk emotion check
        risk_emotions = self._step10_check_risk_emotions(emotion_profile)
        summary["risk_emotions"] = risk_emotions

        return summary

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------

    def _step1_apply_preset(self) -> None:
        """Step 1: Apply mode-specific NT baseline adjustments.

        Injects the EmotionalPreset's receptor-specific NT adjustments
        directly into the NeurochemicalEngine, then applies oscillatory
        bias to the current oscillation state.
        """
        preset = get_emotional_preset(self.mode_id)
        if preset is None:
            return

        # Direct NT injection (Part 2 §2.1 step 1)
        if self._neurochem is not None:
            apply_preset_to_neurochem(preset, self._neurochem)

            # Oscillatory bias (Part 2 §5)
            if preset.oscillatory_bias:
                try:
                    osc_state = self._neurochem.get_oscillation_state()
                    apply_oscillatory_bias(osc_state, preset.oscillatory_bias)
                except Exception:
                    log.debug("Could not apply oscillatory bias (no osc_state).")
        else:
            log.debug("Step 1 skipped — no neurochem engine available.")

    def _step2_detect_emotions(
        self,
        bundle: InputBundle,
        result: PipelineResult,
    ) -> Dict[str, float]:
        """Step 2: Run E28 emotional detection on user input.

        E28 reads current NT state for bias adjustment:
          OXT → warmth bias, NE → threat bias, DA → optimism bias

        Falls back to the emotion_profile already on the bundle if E28
        is not available (or was already run during pipeline dispatch).

        Returns
        -------
        Dict[str, float]
            Emotion profile (emotion_name → strength 0.0-1.0).
        """
        # If E28 ran during Phase 3 dispatch, use its result directly
        if result.state and result.state.dispatch and result.state.dispatch.e28_result:
            e28_result = result.state.dispatch.e28_result
            profile = getattr(e28_result, "emotion_profile", None)
            if profile:
                return dict(profile)

        # Try to run E28 directly if we have it
        if self._e28 is not None and self._neurochem is not None:
            try:
                # Update E28's view of the NT state for bias adjustment
                nt_state_for_e28 = self._extract_nt_state_for_e28()
                self._e28.update_neurochem_state(nt_state_for_e28)
                # E28 needs the input text — we pass via a minimal process call
                e28_out = self._e28.process(bundle.raw_text)
                profile = getattr(e28_out, "emotion_profile", {})
                if profile:
                    return dict(profile)
            except Exception:
                log.debug("E28 direct call failed, using bundle fallback.", exc_info=True)

        # Fallback: whatever is already on the bundle
        return dict(bundle.emotion_profile) if bundle.emotion_profile else {}

    def _step3_translate_emotions(self, emotion_profile: Dict[str, float]) -> None:
        """Step 3: Translate emotions → NT signals (dual pathway).

        Path A (Speed — 12 emotions): uses DEFAULT_EMOTION_RECIPES
        Path B (Full — 46 emotions):  uses EMOTION_NT_PROFILES from E28

        Both paths feed into NeurochemicalEngine.step().
        """
        if self._neurochem is None or not emotion_profile:
            return

        try:
            from zados.neurochem.utils.emotion_interface import (
                DEFAULT_EMOTION_RECIPES,
                emotion_profile_to_signals,
            )

            # Path A: Speed path (12 core emotions)
            speed_signals = emotion_profile_to_signals(
                emotion_profile, DEFAULT_EMOTION_RECIPES
            )
            if speed_signals:
                self._neurochem.step(speed_signals)

            # Path B: Full path (46 emotions via E28 profiles)
            try:
                from zados.cognitive_engines.py_engines.emotional_detection_engine import (
                    EMOTION_NT_PROFILES,
                )
                full_signals = emotion_profile_to_signals(
                    emotion_profile, EMOTION_NT_PROFILES
                )
                if full_signals:
                    self._neurochem.step(full_signals)
            except ImportError:
                log.debug("EMOTION_NT_PROFILES not available for full pathway.")

        except ImportError:
            log.debug("emotion_interface not available — step 3 skipped.")

    def _step4_update_tracker(
        self,
        emotion_profile: Dict[str, float],
    ) -> Tuple[str, float]:
        """Step 4: Update EmotionTracker leaky integrators.

        Returns the dominant emotion (name, strength).
        """
        if self._emotion_tracker is None:
            # Return best from raw profile as fallback
            if emotion_profile:
                best = max(emotion_profile.items(), key=lambda kv: kv[1])
                return best
            return ("neutral", 0.0)

        try:
            from zados.neurochem.extractors.emotion_tracker import (
                get_dominant_emotion,
                step_emotion_tracker,
            )

            self._emotion_tracker = step_emotion_tracker(
                self._emotion_tracker, emotion_profile, dt=_DT
            )
            return get_dominant_emotion(self._emotion_tracker)

        except ImportError:
            log.debug("emotion_tracker not importable — step 4 skipped.")
            if emotion_profile:
                best = max(emotion_profile.items(), key=lambda kv: kv[1])
                return best
            return ("neutral", 0.0)

    def _step5_compute_metrics(self) -> Optional[Any]:
        """Step 5: Compute NeurochemicalMetrics from updated NT state.

        Returns NeurochemicalMetrics or None if unavailable.
        """
        if self._neurochem is None:
            return None

        try:
            from zados.neurochem.neurosymbolic.readout import (
                compute_neurosymbolic_readout,
            )

            nt_states = self._neurochem.get_neurotransmitter_states()
            receptor_states = self._neurochem.get_receptor_states()
            osc_state = self._neurochem.get_oscillation_state()
            return compute_neurosymbolic_readout(
                nt_states, receptor_states, osc_state
            )
        except Exception:
            log.debug("Step 5 metrics computation failed.", exc_info=True)
            return None

    def _step6_metrics_to_weights(self, metrics: Any) -> Optional[Any]:
        """Step 6: NeurochemicalMetrics → engine priority weights.

        Returns EnginePriorityWeights or None.
        """
        if metrics is None:
            return None

        try:
            from zados.neurochem.inference_matrix.nt_to_engine import (
                compute_engine_weights,
            )

            return compute_engine_weights(metrics)
        except Exception:
            log.debug("Step 6 engine weight computation failed.", exc_info=True)
            return None

    def _step8_feedback_to_neurochem(
        self, eval_results: Dict[str, Any]
    ) -> None:
        """Step 8: Engine evaluation results → NT feedback signals.

        Closes the bidirectional loop: engine outputs feed back into the
        neurochemical system via engine_to_nt mapping.
        """
        if self._neurochem is None or not eval_results:
            return

        try:
            from zados.neurochem.inference_matrix.engine_to_nt import (
                compute_nt_modulation_from_evaluation,
            )

            feedback = compute_nt_modulation_from_evaluation(eval_results)
            if feedback:
                self._neurochem.step(feedback)
        except Exception:
            log.debug("Step 8 NT feedback failed.", exc_info=True)

    def _step9_homeostatic_check(self) -> Optional[Any]:
        """Step 9: Run E27 homeostatic bounds check.

        Returns E27 result or None.  If violations are detected, applies
        correction signals to the neurochem engine.
        """
        if self._e27 is None or self._neurochem is None:
            return None

        try:
            nt_state_dict = self._neurochem.get_state()
            self._e27.update_neurochem_state(nt_state_dict)
            result = self._e27.process()

            # Apply correction signals if any
            corrections = getattr(result, "correction_signals", None)
            if corrections:
                self._neurochem.step(corrections)
                log.debug("E27 homeostatic corrections applied.")

            return result
        except Exception:
            log.debug("Step 9 homeostatic check failed.", exc_info=True)
            return None

    def _step10_check_risk_emotions(
        self, emotion_profile: Dict[str, float]
    ) -> List[str]:
        """Step 10: Check risk emotions against mode thresholds.

        Returns list of triggered risk emotion names.
        """
        preset = get_emotional_preset(self.mode_id)
        if preset is None:
            return []

        triggered: List[str] = []
        for emotion in preset.risk_emotions:
            threshold = preset.risk_thresholds.get(emotion, 0.5)
            current = emotion_profile.get(emotion, 0.0)
            if current > threshold:
                triggered.append(emotion)
                log.warning(
                    "Risk emotion '%s' exceeded threshold in %s: %.2f > %.2f",
                    emotion, self.mode_id, current, threshold,
                )

        return triggered

    # ==================================================================
    # SHARED HELPERS
    # ==================================================================

    def _resolve_engines(
        self,
        bundle: InputBundle,
        subject: Optional[SubjectCategory] = None,
    ) -> InputBundle:
        """Resolve engine tiers for this mode and apply as weights.

        Parameters
        ----------
        bundle : InputBundle
        subject : SubjectCategory, optional
            If not provided, inferred from raw_text.

        Returns
        -------
        InputBundle (mutated)
        """
        if subject is None:
            subject = classify_subject_from_text(bundle.raw_text)

        tiers = self._toolkit.resolve(self.mode_id, subject)
        weights = self._toolkit.tiers_to_weights_by_id(tiers)
        bundle.engine_weights.update(weights)

        # Set reward profile
        bundle.active_mode = f"LearningMode_{self.mode_id}"

        return bundle

    def _apply_emotional_preset(self, bundle: InputBundle) -> InputBundle:
        """Apply the mode-specific emotional preset to the bundle.

        Sets bundle-level NT signals for deferred application, AND
        applies directly to neurochem engine if available (step 1).

        Parameters
        ----------
        bundle : InputBundle

        Returns
        -------
        InputBundle (mutated)
        """
        preset = get_emotional_preset(self.mode_id)
        if preset is not None:
            apply_preset_to_bundle(preset, bundle)
        return bundle

    def _combine_weights_with_nt(
        self,
        bundle: InputBundle,
        nt_weights: Any,
    ) -> InputBundle:
        """Combine toolkit tier weights with NT-derived priority weights.

        Part 2 §6 — three-source combination:
          1. Toolkit tiers (hard constraint — T4 stays off)
          2. NT weights (soft modulation within tier)
          3. Intent config (depth control — handled separately)

        Parameters
        ----------
        bundle : InputBundle
            Already has toolkit weights applied.
        nt_weights : EnginePriorityWeights or None
            NT-state-derived weight modulations.

        Returns
        -------
        InputBundle (mutated)
        """
        if nt_weights is None:
            return bundle

        # EnginePriorityWeights has: exploration, verification, attunement,
        # safety, integration — we map these to engine clusters
        try:
            exploration = getattr(nt_weights, "exploration", 0.0)
            verification = getattr(nt_weights, "verification", 0.0)
            attunement = getattr(nt_weights, "attunement", 0.0)
            safety = getattr(nt_weights, "safety", 0.0)
            integration = getattr(nt_weights, "integration", 0.0)

            # Modulate existing weights (don't override T4=0.0)
            for engine_key, current_weight in bundle.engine_weights.items():
                if current_weight <= 0.0:
                    continue  # T4 — stays disabled

                # Apply cluster-based modulation
                modulation = self._cluster_modulation(
                    engine_key, exploration, verification,
                    attunement, safety, integration,
                )
                if modulation != 0.0:
                    new_weight = max(0.0, min(1.0, current_weight + modulation * 0.2))
                    bundle.engine_weights[engine_key] = new_weight

        except Exception:
            log.debug("NT weight combination failed.", exc_info=True)

        return bundle

    def _cluster_modulation(
        self,
        engine_key: str,
        exploration: float,
        verification: float,
        attunement: float,
        safety: float,
        integration: float,
    ) -> float:
        """Map engine cluster to the relevant NT priority axis."""
        # Import here to avoid circular dependency
        try:
            from zados.cognitive_engines.constants import ENGINE_CLUSTER_MAP
            cluster = ENGINE_CLUSTER_MAP.get(engine_key, "")
        except ImportError:
            return 0.0

        if cluster == "detection":
            return verification
        elif cluster == "dialectic":
            return exploration * 0.5 + verification * 0.5
        elif cluster == "pattern_analysis":
            return exploration * 0.7 + integration * 0.3
        elif cluster == "reasoning":
            return integration * 0.5 + exploration * 0.5
        elif cluster == "evaluation":
            return verification * 0.7 + safety * 0.3
        elif cluster == "metacognition":
            return safety * 0.5 + integration * 0.5
        elif cluster == "learning":
            return exploration * 0.5 + attunement * 0.3 + integration * 0.2
        elif cluster == "knowledge_substrate":
            return integration
        elif cluster == "executive_control":
            return safety * 0.5 + verification * 0.5
        return 0.0

    def _record_learning(
        self,
        session: SessionState,
        result: PipelineResult,
        subject: str = "",
    ) -> None:
        """Record learning events from the pipeline result.

        Parameters
        ----------
        session : SessionState
        result : PipelineResult
        subject : str
        """
        engine_results: Dict[int, Dict[str, Any]] = {}
        if result.state and result.state.dispatch:
            engine_results = result.state.dispatch.engine_results

        # Get MemoryContrast result if available
        contrast_result = None
        if result.state and result.state.postprocess:
            contrast_result = getattr(
                result.state.postprocess, "contrast_result", None
            )

        self._learning_log.record_turn(
            mode=self.mode_id,
            subject=subject,
            session_id=session.session_id,
            engine_results=engine_results,
            contrast_result=contrast_result,
        )

    def _check_risk_emotions(
        self,
        bundle: InputBundle,
    ) -> List[str]:
        """Check if any risk emotions exceed their thresholds.

        Parameters
        ----------
        bundle : InputBundle

        Returns
        -------
        List[str]
            Names of risk emotions that exceeded their threshold.
        """
        preset = get_emotional_preset(self.mode_id)
        if preset is None:
            return []

        triggered: List[str] = []
        for emotion in preset.risk_emotions:
            threshold = preset.risk_thresholds.get(emotion, 0.5)
            current = bundle.emotion_profile.get(emotion, 0.0)
            if current > threshold:
                triggered.append(emotion)
                log.warning(
                    "Risk emotion '%s' exceeded threshold in %s: %.2f > %.2f",
                    emotion, self.mode_id, current, threshold,
                )

        return triggered

    def _check_drift(self, bundle: InputBundle) -> bool:
        """Check for context drift.

        Returns True if drift was detected and anchor was re-set.
        """
        if self._context.active_anchor is None:
            # Create initial anchor
            self._context.create_anchor(
                raw_text=bundle.raw_text,
                subject_hint=classify_subject_from_text(bundle.raw_text).value,
            )
            return False

        if self._context.has_drifted(bundle.raw_text):
            log.info("Context drift detected in %s — re-anchoring.", self.mode_id)
            self._context.create_anchor(
                raw_text=bundle.raw_text,
                subject_hint=classify_subject_from_text(bundle.raw_text).value,
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Neurochem helper methods
    # ------------------------------------------------------------------

    def _extract_nt_state_for_e28(self) -> Dict[str, float]:
        """Extract a simple NT concentration dict for E28's bias adjustment.

        E28 reads: OXT (warmth bias), NE (threat bias), DA (optimism bias),
        5-HT (stability), COR (stress), GABA (inhibition).

        Returns dict with lowercase keys (e28 expects lowercase).
        """
        if self._neurochem is None:
            return {}

        try:
            state = self._neurochem.get_state()
            # Extract total concentrations — state format may vary
            result: Dict[str, float] = {}
            for nt_key in ("oxt", "ne", "da", "5ht", "cor", "gaba"):
                nt_data = state.get(nt_key, state.get(nt_key.upper(), {}))
                if isinstance(nt_data, dict):
                    result[nt_key] = nt_data.get(
                        "C_total", nt_data.get("total_concentration", 0.0)
                    )
                elif isinstance(nt_data, (int, float)):
                    result[nt_key] = float(nt_data)
            return result
        except Exception:
            return {}

    def _build_eval_results(self, result: PipelineResult) -> Dict[str, Any]:
        """Build evaluation results dict from pipeline output for step 8.

        Maps engine outputs to the evaluation axes used by
        engine_to_nt.compute_nt_modulation_from_evaluation():
          - confidence → DA RPE signal
          - contradictions_found → NE reuptake modulation
          - novelty_detected → DA novelty drive
          - social_resonance → OXT modulation

        Parameters
        ----------
        result : PipelineResult

        Returns
        -------
        Dict[str, Any]
        """
        eval_dict: Dict[str, Any] = {
            "confidence": 0.5,
            "contradictions_found": 0,
            "novelty_detected": 0.0,
            "social_resonance": 0.0,
        }

        if result.state is None:
            return eval_dict

        # Extract from dispatch results
        if result.state.dispatch:
            er = result.state.dispatch.engine_results

            # E1 — Contradiction detection
            e1 = er.get(1, {})
            contradictions = e1.get("contradictions", [])
            eval_dict["contradictions_found"] = len(contradictions) if isinstance(contradictions, list) else 0

            # E19 — Pattern identification (novelty)
            e19 = er.get(19, {})
            novel_patterns = [
                p for p in e19.get("patterns", [])
                if isinstance(p, dict) and p.get("status") == "CANDIDATE"
            ]
            eval_dict["novelty_detected"] = min(1.0, len(novel_patterns) * 0.2)

            # E23 — Intention map (social resonance / attunement)
            e23 = er.get(23, {})
            eval_dict["social_resonance"] = e23.get("attunement_score", 0.0)

        # Extract from reward evaluation
        if result.state.reward and result.state.reward.phase5_result:
            p5 = result.state.reward.phase5_result
            eval_dict["confidence"] = getattr(p5, "confidence", 0.5)

        return eval_dict

    def _get_receptor_saturation(self, nt_key: str, receptor_key: str) -> float:
        """Get a specific receptor's saturation from the neurochem engine.

        Parameters
        ----------
        nt_key : str
            NT key (uppercase, e.g. "DA", "CB1").
        receptor_key : str
            Receptor subtype key (e.g. "D3", "CB1").

        Returns
        -------
        float
            Saturation value (0.0-1.0), 0.0 if unavailable.
        """
        if self._neurochem is None:
            return 0.0

        try:
            receptor_states = self._neurochem.get_receptor_states()
            nt_receptors = receptor_states.get(nt_key, {})
            receptor = nt_receptors.get(receptor_key, None)
            if receptor is not None:
                return getattr(receptor, "saturation", 0.0)
        except Exception:
            pass
        return 0.0

    def _get_metrics_dict(self, metrics: Any) -> Dict[str, float]:
        """Convert NeurochemicalMetrics to a simple dict.

        Returns dict with keys: motivation, empathy, cognitive_rigidity,
        fatigue, precision, openness, anxiety, social_engagement.
        """
        if metrics is None:
            return {}

        result: Dict[str, float] = {}
        for attr in (
            "motivation", "empathy", "cognitive_rigidity", "fatigue",
            "precision", "openness", "anxiety", "social_engagement",
        ):
            val = getattr(metrics, attr, None)
            if val is not None:
                result[attr] = float(val)
        return result
