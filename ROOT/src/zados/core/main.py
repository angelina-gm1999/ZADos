"""
ZA-DOS v0.6 — InputClassifier (spec §3.1 + Part 2 §§2-6).

Top-level Matrioshka outer layer.  Classifies incoming RawInput and
routes it to the correct sub-pipeline:

  MESSAGE:
    - REGULAR           → RegularInputPipeline
    - LEARNING_MODE     → LearningModePipeline M1-M5
    - SELF_REFLECTIVE   → SelfReflectiveQueryPipeline

  FUNCTION:
    - SLEEP (REM/Dream) → REMPipeline / DreamPipeline
    - META_LEARNING     → HomeworkPipeline / ReflectivePipeline

Classification priority:
  1. Command prefix (/sleep, /homework, /reflective, /dream) → FUNCTION
  2. Session mode check (already in learning → stay)
  3. Self-reflective markers + unsolved buffer non-empty → SELF_REFLECTIVE
  4. Learning mode markers → LEARNING_MODE (M1-M5)
  5. Default → REGULAR

Part 2 wiring:
  Learning mode pipelines receive optional neurochem dependencies
  (neurochem_engine, extractor_orchestrator, emotion_tracker_state)
  extracted from the session orchestrator.  If not available, pipelines
  degrade gracefully (all neurochem steps are skipped).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from zados.core.commanded.meta_learning_mode.homework_mode.pipeline import (
    HomeworkPipeline,
)
from zados.core.commanded.meta_learning_mode.reflective_mode.pipeline import (
    ReflectivePipeline,
)
from zados.core.commanded.sleep_mode.dream_mode.pipeline import DreamPipeline
from zados.core.commanded.sleep_mode.rem_mode.pipeline import REMPipeline
from zados.core.inputs.learning_modes.human_teaches import HumanTeachesPipeline
from zados.core.inputs.learning_modes.independent_study import (
    IndependentStudyPipeline,
)
from zados.core.inputs.learning_modes.learn_together import LearnTogetherPipeline
from zados.core.inputs.learning_modes.learned_questions import (
    LearnedQuestionsPipeline,
)
from zados.core.inputs.learning_modes.peer_review import PeerReviewPipeline
from zados.core.inputs.regular_input_mode.pipeline import RegularInputPipeline
from zados.core.inputs.self_ref_query_mode.pipeline import (
    SelfReflectiveQueryPipeline,
)
from zados.core.processes.context_anchor import ContextAnchorManager
from zados.core.processes.learning_log import LearningLogPipeline
from zados.core.processes.unsolved_buffer import UnsolvedBuffer
from zados.core.types import (
    ClassificationResult,
    FunctionSubType,
    InputBundle,
    InputType,
    MessageSubType,
    MetaLearningVariant,
    RawInput,
    SessionState,
    SleepVariant,
)

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Command prefix patterns
# ------------------------------------------------------------------

_COMMAND_PATTERNS: Dict[str, Any] = {
    r"^/sleep\s+rem\b":         (FunctionSubType.SLEEP, SleepVariant.REM),
    r"^/sleep\s+dream\b":       (FunctionSubType.SLEEP, SleepVariant.DREAM),
    r"^/sleep\b":               (FunctionSubType.SLEEP, SleepVariant.REM),  # default to REM
    r"^/homework\b":            (FunctionSubType.META_LEARNING, MetaLearningVariant.HOMEWORK),
    r"^/reflective\b":          (FunctionSubType.META_LEARNING, MetaLearningVariant.REFLECTIVE),
    r"^/dream\b":               (FunctionSubType.SLEEP, SleepVariant.DREAM),
}

# ------------------------------------------------------------------
# Learning mode markers
# ------------------------------------------------------------------

_LEARNING_MODE_MARKERS: Dict[int, list] = {
    1: ["teach me", "explain to me", "show me how", "i want to learn", "help me understand"],
    2: ["review this", "check my work", "find errors", "critique", "analyze this"],
    3: ["let's explore", "let's figure out", "work together", "discuss this", "what do you think about"],
    4: ["what questions", "what haven't we", "unresolved", "open questions"],
    5: ["i'll study", "independent", "self-study", "on my own", "let me explore"],
}

_SELF_REF_MARKERS = [
    "what do i think", "how do i feel about", "reflect on",
    "my understanding", "what have i learned", "self-reflect",
    "introspect", "examine my", "review my thinking",
]


class InputClassifier:
    """Top-level Matrioshka classifier and router.

    Usage
    -----
    >>> classifier = InputClassifier(session_orchestrator)
    >>> result = classifier.process(RawInput(text="Hello!"))

    Parameters
    ----------
    session_orchestrator : SessionOrchestrator
        The v0.5 session manager (holds AnswerPipeline reference).
    learning_log : LearningLogPipeline, optional
    unsolved_buffer : UnsolvedBuffer, optional
    context_manager : ContextAnchorManager, optional
    neurochem_engine : NeurochemicalEngine, optional
        Live neurochemical simulation engine (Part 2 §2.1).
    extractor_orchestrator : ExtractorOrchestrator, optional
        Full 9-step stochastic pathway (Part 2 §3.3).
    emotion_tracker_state : EmotionTrackerState, optional
        Leaky-integrator state for emotion saturation tracking.
    """

    def __init__(
        self,
        session_orchestrator: Any,
        learning_log: Optional[LearningLogPipeline] = None,
        unsolved_buffer: Optional[UnsolvedBuffer] = None,
        context_manager: Optional[ContextAnchorManager] = None,
        neurochem_engine: Any = None,
        extractor_orchestrator: Any = None,
        emotion_tracker_state: Any = None,
    ) -> None:
        self._orchestrator = session_orchestrator
        self._learning_log = learning_log or LearningLogPipeline()
        self._unsolved = unsolved_buffer or UnsolvedBuffer()
        self._context = context_manager or ContextAnchorManager()

        # Extract pipeline from orchestrator
        pipeline = session_orchestrator.pipeline
        engines = session_orchestrator.engines
        memory = session_orchestrator.memory

        # Part 2: Resolve neurochem dependencies
        # Try to extract from orchestrator if not explicitly provided
        self._neurochem = neurochem_engine or getattr(
            session_orchestrator, "neurochem_engine", None
        )
        self._extractor = extractor_orchestrator or getattr(
            session_orchestrator, "extractor_orchestrator", None
        )
        self._emotion_tracker = emotion_tracker_state or getattr(
            session_orchestrator, "emotion_tracker_state", None
        )

        # Resolve GeneralQuestionStore for regular pipeline
        _general_questions = getattr(
            getattr(memory, "thoughts", None), "general_questions", None
        ) if memory else None

        # Restore persisted unsolved questions from LTMM at init
        if _general_questions is not None:
            self._unsolved.load_from_ltmm(_general_questions)

        # Create sub-pipelines
        self._regular = RegularInputPipeline(
            answer_pipeline=pipeline,
            context_manager=self._context,
            engines=engines,
            general_question_store=_general_questions,
        )

        self._self_ref = SelfReflectiveQueryPipeline(
            answer_pipeline=pipeline,
            unsolved_buffer=self._unsolved,
            memory_contrast=getattr(memory, "contrast", None),
            context_manager=self._context,
        )

        # Learning mode pipelines — with Part 2 neurochem + LTMM store wiring
        _held_blocks = getattr(
            getattr(memory, "thoughts", None), "held_blocks", None
        ) if memory else None

        common_kwargs = dict(
            answer_pipeline=pipeline,
            learning_log=self._learning_log,
            unsolved_buffer=self._unsolved,
            context_manager=self._context,
            engines=engines,
            neurochem_engine=self._neurochem,
            extractor_orchestrator=self._extractor,
            emotion_tracker_state=self._emotion_tracker,
            held_block_store=_held_blocks,
            memory=memory,
        )
        self._learning_pipelines: Dict[int, Any] = {
            1: HumanTeachesPipeline(**common_kwargs),
            2: PeerReviewPipeline(**common_kwargs),
            3: LearnTogetherPipeline(**common_kwargs),
            4: LearnedQuestionsPipeline(**common_kwargs),
            5: IndependentStudyPipeline(**common_kwargs),
        }

        # Commanded mode pipelines
        _journal = getattr(memory, "journal_store", None) if memory else None

        self._rem = REMPipeline(
            answer_pipeline=pipeline,
            memory=memory,
            journal_store=_journal,
        )
        self._dream = DreamPipeline(
            answer_pipeline=pipeline,
            memory=memory,
            unsolved_buffer=self._unsolved,
            journal_store=_journal,
        )
        self._homework = HomeworkPipeline(
            answer_pipeline=pipeline,
            learning_log=self._learning_log,
            unsolved_buffer=self._unsolved,
            memory_layer=memory,
            specialized_logs=getattr(
                getattr(memory, "manager", None), "logs", None
            ) if memory is not None else None,
        )
        if _journal is not None:
            self._homework.set_journal_store(_journal)
        self._reflective = ReflectivePipeline(
            answer_pipeline=pipeline,
            learning_log=self._learning_log,
            memory_layer=memory,
            unsolved_buffer=self._unsolved,
            neurochem_engine=self._neurochem,
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, raw_input: RawInput) -> ClassificationResult:
        """Classify a raw input into a route target.

        Parameters
        ----------
        raw_input : RawInput

        Returns
        -------
        ClassificationResult
        """
        text = raw_input.text.strip()
        lower = text.lower()

        # Priority 1: Command prefix
        for pattern, (sub_type, variant) in _COMMAND_PATTERNS.items():
            if re.match(pattern, lower):
                return ClassificationResult(
                    input_type=InputType.FUNCTION,
                    sub_type=sub_type,
                    variant=variant,
                    route_target=f"{sub_type.value}_{variant.value}",
                    confidence=1.0,
                    raw_input=raw_input,
                )

        # Priority 2: Session mode continuity
        session = self._orchestrator.session
        if session is not None and session.active_learning_mode:
            mode_num = int(session.active_learning_mode[1])  # "M1" → 1
            return ClassificationResult(
                input_type=InputType.MESSAGE,
                sub_type=MessageSubType.LEARNING_MODE,
                route_target=f"learning_M{mode_num}",
                confidence=0.9,
                raw_input=raw_input,
                learning_mode_number=mode_num,
            )

        # Priority 3: Self-reflective markers
        if not self._unsolved.is_empty():
            for marker in _SELF_REF_MARKERS:
                if marker in lower:
                    return ClassificationResult(
                        input_type=InputType.MESSAGE,
                        sub_type=MessageSubType.SELF_REFLECTIVE,
                        route_target="self_reflective",
                        confidence=0.8,
                        raw_input=raw_input,
                    )

        # Priority 4: Learning mode markers
        for mode_num, markers in _LEARNING_MODE_MARKERS.items():
            for marker in markers:
                if marker in lower:
                    return ClassificationResult(
                        input_type=InputType.MESSAGE,
                        sub_type=MessageSubType.LEARNING_MODE,
                        route_target=f"learning_M{mode_num}",
                        confidence=0.7,
                        raw_input=raw_input,
                        learning_mode_number=mode_num,
                    )

        # Priority 5: Default — regular
        return ClassificationResult(
            input_type=InputType.MESSAGE,
            sub_type=MessageSubType.REGULAR,
            route_target="regular",
            confidence=1.0,
            raw_input=raw_input,
        )

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self, raw_input: RawInput) -> Any:
        """Classify input and route to the correct sub-pipeline.

        Parameters
        ----------
        raw_input : RawInput

        Returns
        -------
        Any
            PipelineResult, LearningModeResult, SelfRefResult,
            or dict (for commanded modes).
        """
        # Ensure session is open
        if self._orchestrator.session is None:
            self._orchestrator.open_session()
        session = self._orchestrator.session

        classification = self.classify(raw_input)
        log.info(
            "InputClassifier: type=%s, sub=%s, route=%s, confidence=%.2f",
            classification.input_type.value,
            classification.sub_type.value if hasattr(classification.sub_type, 'value') else str(classification.sub_type),
            classification.route_target,
            classification.confidence,
        )

        # Route to correct pipeline
        if classification.input_type == InputType.FUNCTION:
            return self._process_function(classification, session)
        else:
            return self._process_message(classification, raw_input, session)

    def _process_message(
        self,
        classification: ClassificationResult,
        raw_input: RawInput,
        session: SessionState,
    ) -> Any:
        """Route message-type inputs."""
        bundle = InputBundle(raw_text=raw_input.text)

        if classification.sub_type == MessageSubType.REGULAR:
            return self._regular.process_turn(bundle, session)

        elif classification.sub_type == MessageSubType.LEARNING_MODE:
            mode_num = classification.learning_mode_number
            pipeline = self._learning_pipelines.get(mode_num)
            if pipeline is None:
                log.warning("Unknown learning mode %d, falling back to regular.", mode_num)
                return self._regular.process_turn(bundle, session)

            # Track active learning mode in session
            session.active_learning_mode = f"M{mode_num}"
            session.session_mode = "learning"
            return pipeline.process_turn(bundle, session)

        elif classification.sub_type == MessageSubType.SELF_REFLECTIVE:
            return self._self_ref.process_turn(bundle, session)

        # Fallback
        return self._regular.process_turn(bundle, session)

    def _process_function(
        self,
        classification: ClassificationResult,
        session: SessionState,
    ) -> Any:
        """Route function-type (command) inputs."""
        if classification.sub_type == FunctionSubType.SLEEP:
            # Clear learning mode
            session.active_learning_mode = None
            session.session_mode = "sleep"

            if classification.variant == SleepVariant.DREAM:
                return self._dream.process(session)
            return self._rem.process(session)

        elif classification.sub_type == FunctionSubType.META_LEARNING:
            session.session_mode = "meta"

            if classification.variant == MetaLearningVariant.REFLECTIVE:
                return self._reflective.process(session)
            return self._homework.process(session)

        log.warning("Unknown function sub_type: %s", classification.sub_type)
        return {"status": "error", "message": "Unknown command."}

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def process_text(self, text: str) -> Any:
        """Convenience: classify and process raw text.

        Parameters
        ----------
        text : str

        Returns
        -------
        Any
        """
        return self.process(RawInput(text=text))

    def close_session(self) -> Dict[str, Any]:
        """Close the current session and trigger end-of-session processing."""
        return self._orchestrator.close_session()

    @property
    def unsolved_buffer(self) -> UnsolvedBuffer:
        """Access the unsolved question buffer."""
        return self._unsolved

    @property
    def learning_log(self) -> LearningLogPipeline:
        """Access the learning log."""
        return self._learning_log
