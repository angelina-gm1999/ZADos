"""
Tests for learning mode pipelines — Part 4 implementation.

Covers:
  - Pipeline skeleton (base.py): MODE_CONFIGS, held-thinking-block detection
  - M1 (human_teaches.py): question generation, held block check
  - M2 (peer_review.py): core memory update gate, relief tracking
  - M3 (learn_together.py): human challenge logic
  - M4 (learned_questions.py): sub-mode routing, dream threshold
  - M5 (independent_study.py): response suppression, E28 OFF
  - Phantom engine stubs
"""
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Lightweight mock types matching real pipeline types
# ---------------------------------------------------------------------------

@dataclass
class MockDispatchResult:
    engine_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    engines_run: List[int] = field(default_factory=list)
    engines_skipped: List[int] = field(default_factory=list)
    e28_result: Any = None


@dataclass
class MockThinkingResult:
    thinking_trace: str = ""
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class MockRewardResult:
    phase5_result: Any = None


@dataclass
class MockPipelineState:
    dispatch: Optional[MockDispatchResult] = None
    thinking: Optional[MockThinkingResult] = None
    reward: Optional[MockRewardResult] = None
    postprocess: Any = None


@dataclass
class MockPipelineResult:
    state: Optional[MockPipelineState] = None
    final_answer: str = ""
    response: str = ""


@dataclass
class MockInputBundle:
    raw_text: str = "test input"
    intent_archetype: str = ""
    intent_vector: Dict[str, float] = field(default_factory=dict)
    nt_signals: Dict[str, Dict[str, float]] = field(default_factory=dict)
    emotion_profile: Dict[str, float] = field(default_factory=dict)
    active_mode: str = ""
    engine_weights: Dict[str, float] = field(default_factory=dict)
    context_flags: Dict[str, Any] = field(default_factory=dict)
    safety_tier: str = "NORMAL"
    mtmm_context_window: List[Any] = field(default_factory=list)
    mission_briefing: Any = None
    osc_state: Any = None
    extractor_state: Any = None


@dataclass
class MockSessionState:
    session_id: str = "test_session_001"
    turn_id: str = "turn_001"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Mock stores
# ---------------------------------------------------------------------------

class MockHeldBlockStore:
    def __init__(self):
        self.blocks = []

    def write(self, block):
        self.blocks.append(block)

    def get_all(self):
        return list(self.blocks)

    def __len__(self):
        return len(self.blocks)


class MockLearningLog:
    def __init__(self):
        self.entries = []

    def record_turn(self, **kwargs):
        self.entries.append(kwargs)

    def get_recent(self, session_id):
        return [e for e in self.entries if e.get("session_id") == session_id]


class MockUnsolvedBuffer:
    def __init__(self):
        self.questions = []
        self._next_question = None

    def add(self, *args, **kwargs):
        from zados.core.types import UnsolvedQuestion
        q = UnsolvedQuestion(**kwargs)
        self.questions.append(q)
        return q

    def select_next(self):
        return self._next_question

    def mark_attempted(self, qid, partial_answer=""):
        pass

    def get_all(self):
        return list(self.questions)


class MockAnswerPipeline:
    def __init__(self, result=None):
        self._result = result or MockPipelineResult(
            state=MockPipelineState(
                dispatch=MockDispatchResult(),
                thinking=MockThinkingResult(thinking_trace="I am thinking about this topic."),
            ),
            final_answer="Test answer.",
        )

    def process_turn(self, bundle, session):
        return self._result

    def process(self, bundle):
        return self._result


# ===========================================================================
# Tests: Base Pipeline — MODE_CONFIGS
# ===========================================================================

class TestModeConfigs:
    def test_all_5_modes_have_configs(self):
        from zados.core.inputs.learning_modes.base import MODE_CONFIGS
        for mode_id in ("M1", "M2", "M3", "M4", "M5"):
            assert mode_id in MODE_CONFIGS, f"Missing config for {mode_id}"

    def test_m1_config_values(self):
        from zados.core.inputs.learning_modes.base import MODE_CONFIGS
        c = MODE_CONFIGS["M1"]
        assert c.semantic_expansion_max_hops == 2
        assert c.max_questions_per_turn == 2
        assert c.generate_response is True
        assert c.contradiction_mode == "learning"

    def test_m2_config_retroactive(self):
        from zados.core.inputs.learning_modes.base import MODE_CONFIGS
        c = MODE_CONFIGS["M2"]
        assert c.use_retroactive_contrast is True
        assert c.contradiction_mode == "soft"
        assert c.max_questions_per_turn == 0

    def test_m3_config_unlimited(self):
        from zados.core.inputs.learning_modes.base import MODE_CONFIGS
        c = MODE_CONFIGS["M3"]
        assert c.semantic_expansion_max_hops == -1  # Unlimited
        assert c.pattern_chain_max_depth == -1
        assert c.max_questions_per_turn == -1

    def test_m4_config_abbreviated(self):
        from zados.core.inputs.learning_modes.base import MODE_CONFIGS
        c = MODE_CONFIGS["M4"]
        assert c.response_depth == "abbreviated"
        assert c.max_questions_per_turn == 1

    def test_m5_config_autonomous(self):
        from zados.core.inputs.learning_modes.base import MODE_CONFIGS
        c = MODE_CONFIGS["M5"]
        assert c.generate_response is False
        assert c.response_depth == "none"


# ===========================================================================
# Tests: Identity-relevant emotions + threshold constants
# ===========================================================================

class TestHeldBlockConstants:
    def test_threshold_is_0_6(self):
        from zados.core.inputs.learning_modes.base import HELD_BLOCK_EMOTION_THRESHOLD
        assert HELD_BLOCK_EMOTION_THRESHOLD == 0.6

    def test_identity_relevant_emotions_set(self):
        from zados.core.inputs.learning_modes.base import IDENTITY_RELEVANT_EMOTIONS
        expected_subset = {"ashamed", "guilty", "betrayal", "rejected", "grief", "proud"}
        assert expected_subset.issubset(IDENTITY_RELEVANT_EMOTIONS)

    def test_identity_relevant_contains_self_evaluation(self):
        from zados.core.inputs.learning_modes.base import IDENTITY_RELEVANT_EMOTIONS
        # From self_evaluation group in E28
        for em in ("ashamed", "guilty", "regret", "critical"):
            assert em in IDENTITY_RELEVANT_EMOTIONS


# ===========================================================================
# Tests: Held Thinking Block detection
# ===========================================================================

class TestHeldThinkingBlock:
    """Test the _check_held_thinking_block method from base.py."""

    def _make_pipeline(self, held_store=None):
        """Create a minimal concrete pipeline for testing."""
        from zados.core.inputs.learning_modes.human_teaches import HumanTeachesPipeline
        return HumanTeachesPipeline(
            answer_pipeline=MockAnswerPipeline(),
            learning_log=MockLearningLog(),
            unsolved_buffer=MockUnsolvedBuffer(),
            held_block_store=held_store,
        )

    def test_no_block_when_below_threshold(self):
        store = MockHeldBlockStore()
        pipe = self._make_pipeline(held_store=store)
        ids = pipe._check_held_thinking_block(
            emotion_profile={"joy": 0.4, "curious": 0.3},
            thinking_trace="Some thinking",
            bundle=MockInputBundle(),
            session=MockSessionState(),
        )
        assert ids == []
        assert len(store) == 0

    def test_block_written_above_threshold(self):
        store = MockHeldBlockStore()
        pipe = self._make_pipeline(held_store=store)
        ids = pipe._check_held_thinking_block(
            emotion_profile={"grief": 0.8},  # > 0.6 threshold
            thinking_trace="Deep thinking about loss",
            bundle=MockInputBundle(),
            session=MockSessionState(),
        )
        assert len(ids) == 1
        assert len(store) == 1
        block = store.blocks[0]
        assert block.emotion_tag == "grief"
        assert "threshold" in block.emotion_trigger_type

    def test_identity_emotion_triggers_at_any_intensity(self):
        store = MockHeldBlockStore()
        pipe = self._make_pipeline(held_store=store)
        ids = pipe._check_held_thinking_block(
            emotion_profile={"ashamed": 0.2},  # Below 0.6 but identity-relevant
            thinking_trace="Reflecting on mistake",
            bundle=MockInputBundle(),
            session=MockSessionState(),
        )
        assert len(ids) == 1
        block = store.blocks[0]
        assert block.emotion_tag == "ashamed"
        assert block.emotion_trigger_type == "identity_relevant"

    def test_identity_and_threshold_combined(self):
        store = MockHeldBlockStore()
        pipe = self._make_pipeline(held_store=store)
        ids = pipe._check_held_thinking_block(
            emotion_profile={"guilty": 0.8},  # Identity-relevant AND > 0.6
            thinking_trace="Feeling guilty",
            bundle=MockInputBundle(),
            session=MockSessionState(),
        )
        assert len(ids) == 1
        block = store.blocks[0]
        assert block.emotion_trigger_type == "identity_and_threshold"

    def test_no_block_without_store(self):
        pipe = self._make_pipeline(held_store=None)
        ids = pipe._check_held_thinking_block(
            emotion_profile={"grief": 0.9},
            thinking_trace="Deep grief",
            bundle=MockInputBundle(),
            session=MockSessionState(),
        )
        assert ids == []

    def test_no_block_without_thinking_trace(self):
        store = MockHeldBlockStore()
        pipe = self._make_pipeline(held_store=store)
        ids = pipe._check_held_thinking_block(
            emotion_profile={"grief": 0.9},
            thinking_trace="",  # Empty — nothing to capture
            bundle=MockInputBundle(),
            session=MockSessionState(),
        )
        assert ids == []

    def test_multiple_emotions_generate_multiple_blocks(self):
        store = MockHeldBlockStore()
        pipe = self._make_pipeline(held_store=store)
        ids = pipe._check_held_thinking_block(
            emotion_profile={
                "grief": 0.8,       # > threshold
                "ashamed": 0.3,     # identity-relevant
                "joy": 0.2,         # below threshold, not identity
            },
            thinking_trace="Complex emotional moment",
            bundle=MockInputBundle(),
            session=MockSessionState(),
        )
        assert len(ids) == 2  # grief + ashamed, not joy
        tags = {store.blocks[0].emotion_tag, store.blocks[1].emotion_tag}
        assert tags == {"grief", "ashamed"}

    def test_block_has_mode_tag(self):
        store = MockHeldBlockStore()
        pipe = self._make_pipeline(held_store=store)
        pipe._check_held_thinking_block(
            emotion_profile={"betrayal": 0.5},  # identity-relevant
            thinking_trace="Feeling betrayed",
            bundle=MockInputBundle(),
            session=MockSessionState(),
        )
        block = store.blocks[0]
        assert "mode:M1" in block.tags
        assert "emotion:betrayal" in block.tags


# ===========================================================================
# Tests: M1 Question Generation
# ===========================================================================

class TestM1QuestionGeneration:
    def _make_m1(self):
        from zados.core.inputs.learning_modes.human_teaches import HumanTeachesPipeline
        buf = MockUnsolvedBuffer()
        return HumanTeachesPipeline(
            answer_pipeline=MockAnswerPipeline(),
            learning_log=MockLearningLog(),
            unsolved_buffer=buf,
        ), buf

    def test_confusion_generates_question(self):
        pipe, buf = self._make_m1()
        # Override config to ensure max_questions_per_turn > 0
        questions = pipe._generate_clarifying_questions(
            bundle=MockInputBundle(raw_text="What is quantum entanglement?"),
            result=MockPipelineResult(
                state=MockPipelineState(dispatch=MockDispatchResult()),
            ),
            feedback={"emotion_profile": {"confused": 0.5}},
        )
        assert len(questions) >= 1
        assert questions[0].source_mode == "M1"
        assert len(buf.questions) >= 1

    def test_no_question_without_confusion(self):
        pipe, buf = self._make_m1()
        questions = pipe._generate_clarifying_questions(
            bundle=MockInputBundle(),
            result=MockPipelineResult(
                state=MockPipelineState(dispatch=MockDispatchResult()),
            ),
            feedback={"emotion_profile": {"joy": 0.8}},
        )
        assert len(questions) == 0


# ===========================================================================
# Tests: M2 Core Memory Update Gate
# ===========================================================================

class TestM2CoreMemoryGate:
    def _make_m2(self):
        from zados.core.inputs.learning_modes.peer_review import PeerReviewPipeline
        return PeerReviewPipeline(
            answer_pipeline=MockAnswerPipeline(),
            learning_log=MockLearningLog(),
            unsolved_buffer=MockUnsolvedBuffer(),
        )

    def test_no_updates_without_contradictions(self):
        pipe = self._make_m2()
        result = MockPipelineResult(
            state=MockPipelineState(dispatch=MockDispatchResult()),
        )
        updates = pipe._check_core_memory_corrections(
            bundle=MockInputBundle(),
            result=result,
            feedback={"emotion_profile": {}},
            session=MockSessionState(),
        )
        assert updates == []

    def test_updates_from_identity_contradictions(self):
        pipe = self._make_m2()
        result = MockPipelineResult(
            state=MockPipelineState(
                dispatch=MockDispatchResult(
                    engine_results={
                        1: {
                            "contradictions": [
                                {
                                    "target_source": "identity/core",
                                    "target_id": "core_001",
                                    "existing_content": "I believe X",
                                    "correction_content": "Actually Y",
                                    "confidence": 0.7,
                                }
                            ]
                        }
                    }
                ),
            ),
        )
        updates = pipe._check_core_memory_corrections(
            bundle=MockInputBundle(raw_text="I think Y is correct"),
            result=result,
            feedback={"emotion_profile": {"regret": 0.3}},
            session=MockSessionState(),
        )
        assert len(updates) == 1
        assert updates[0].core_memory_key == "core_001"
        assert updates[0].confidence == 0.7
        assert not updates[0].applied  # NOT applied mid-conversation

    def test_non_identity_contradictions_ignored(self):
        pipe = self._make_m2()
        result = MockPipelineResult(
            state=MockPipelineState(
                dispatch=MockDispatchResult(
                    engine_results={
                        1: {
                            "contradictions": [
                                {
                                    "target_source": "knowledge/lessons",
                                    "target_id": "lesson_001",
                                }
                            ]
                        }
                    }
                ),
            ),
        )
        updates = pipe._check_core_memory_corrections(
            bundle=MockInputBundle(),
            result=result,
            feedback={"emotion_profile": {}},
            session=MockSessionState(),
        )
        assert updates == []


# ===========================================================================
# Tests: M3 Human Challenge Logic
# ===========================================================================

class TestM3HumanChallenge:
    def _make_m3(self):
        from zados.core.inputs.learning_modes.learn_together import LearnTogetherPipeline
        return LearnTogetherPipeline(
            answer_pipeline=MockAnswerPipeline(),
            learning_log=MockLearningLog(),
            unsolved_buffer=MockUnsolvedBuffer(),
        )

    def test_no_challenges_without_contradictions(self):
        pipe = self._make_m3()
        result = MockPipelineResult(
            state=MockPipelineState(dispatch=MockDispatchResult()),
        )
        challenges = pipe._check_human_claims(
            bundle=MockInputBundle(),
            result=result,
            feedback={"emotion_profile": {}},
        )
        assert challenges == []

    def test_contradiction_creates_challenge(self):
        pipe = self._make_m3()
        result = MockPipelineResult(
            state=MockPipelineState(
                dispatch=MockDispatchResult(
                    engine_results={
                        1: {
                            "contradictions": [
                                {
                                    "claim": "The sky is green",
                                    "existing_content": "The sky is blue",
                                    "source": "knowledge/lessons",
                                    "confidence": 0.9,
                                }
                            ]
                        }
                    }
                ),
            ),
        )
        challenges = pipe._check_human_claims(
            bundle=MockInputBundle(raw_text="The sky is green"),
            result=result,
            feedback={"emotion_profile": {}},
        )
        assert len(challenges) == 1
        assert challenges[0]["engine"] == "E1_contradiction"
        assert challenges[0]["confidence"] == 0.9

    def test_fallacy_creates_challenge(self):
        pipe = self._make_m3()
        result = MockPipelineResult(
            state=MockPipelineState(
                dispatch=MockDispatchResult(
                    engine_results={
                        4: {
                            "fallacies": [
                                {
                                    "statement": "All X are Y because Z",
                                    "fallacy_type": "hasty_generalization",
                                    "confidence": 0.6,
                                }
                            ]
                        }
                    }
                ),
            ),
        )
        challenges = pipe._check_human_claims(
            bundle=MockInputBundle(),
            result=result,
            feedback={"emotion_profile": {}},
        )
        assert len(challenges) == 1
        assert challenges[0]["engine"] == "E4_fallacy"


# ===========================================================================
# Tests: M4 Dream Threshold
# ===========================================================================

class TestM4DreamThreshold:
    def _make_m4(self):
        from zados.core.inputs.learning_modes.learned_questions import LearnedQuestionsPipeline
        buf = MockUnsolvedBuffer()
        return LearnedQuestionsPipeline(
            answer_pipeline=MockAnswerPipeline(),
            learning_log=MockLearningLog(),
            unsolved_buffer=buf,
        ), buf

    def test_no_candidates_below_threshold(self):
        from zados.core.types import UnsolvedQuestion
        pipe, buf = self._make_m4()
        q = UnsolvedQuestion(
            question_text="What is X?",
            source_mode="M3",
            resolution_attempts=3,  # Below threshold of 5
        )
        buf.questions = [q]
        candidates = pipe._flag_dream_candidates()
        assert candidates == []

    def test_candidates_at_threshold(self):
        from zados.core.types import UnsolvedQuestion
        pipe, buf = self._make_m4()
        q = UnsolvedQuestion(
            question_text="Deep unsolved question",
            source_mode="M3",
            resolution_attempts=5,  # At threshold
        )
        buf.questions = [q]
        candidates = pipe._flag_dream_candidates()
        assert len(candidates) == 1
        assert candidates[0] == q.question_id

    def test_resolved_not_flagged(self):
        from zados.core.types import UnsolvedQuestion
        pipe, buf = self._make_m4()
        q = UnsolvedQuestion(
            question_text="Resolved question",
            source_mode="M3",
            resolution_attempts=10,
            resolved=True,
        )
        buf.questions = [q]
        candidates = pipe._flag_dream_candidates()
        assert candidates == []


# ===========================================================================
# Tests: M4 Sub-mode routing
# ===========================================================================

class TestM4SubModeRouting:
    def _make_m4(self):
        from zados.core.inputs.learning_modes.learned_questions import LearnedQuestionsPipeline
        buf = MockUnsolvedBuffer()
        return LearnedQuestionsPipeline(
            answer_pipeline=MockAnswerPipeline(),
            learning_log=MockLearningLog(),
            unsolved_buffer=buf,
        ), buf

    def test_auto_mode_selects_from_buffer(self):
        from zados.core.types import UnsolvedQuestion
        pipe, buf = self._make_m4()
        q = UnsolvedQuestion(question_text="Auto question", source_mode="M3")
        buf._next_question = q
        bundle = MockInputBundle(raw_text="next")
        selected = pipe._submode_route_question(bundle, MockSessionState())
        assert selected is q
        assert bundle.raw_text == "Auto question"

    def test_prompted_mode_returns_none(self):
        pipe, buf = self._make_m4()
        bundle = MockInputBundle(raw_text="My specific question about X")
        selected = pipe._submode_route_question(bundle, MockSessionState())
        assert selected is None


# ===========================================================================
# Tests: M5 Response Suppression
# ===========================================================================

class TestM5ResponseSuppression:
    def test_m5_config_disables_response(self):
        from zados.core.inputs.learning_modes.base import MODE_CONFIGS
        assert MODE_CONFIGS["M5"].generate_response is False

    def test_m5_e28_disabled_flag(self):
        """M5 should set e28_disabled context flag."""
        from zados.core.inputs.learning_modes.independent_study import IndependentStudyPipeline
        pipe = IndependentStudyPipeline(
            answer_pipeline=MockAnswerPipeline(),
            learning_log=MockLearningLog(),
            unsolved_buffer=MockUnsolvedBuffer(),
        )
        bundle = MockInputBundle(raw_text="Study material about ML")
        session = MockSessionState()
        result = pipe.process_turn(bundle, session)
        assert bundle.context_flags.get("e28_disabled") is True
        assert bundle.context_flags.get("autonomous_mode") is True


# ===========================================================================
# Tests: Phantom Engine Stubs
# ===========================================================================

class TestPhantomStubs:
    def test_all_stubs_importable(self):
        from zados.cognitive_engines.py_engines.phantom_stubs import PHANTOM_ENGINE_STUBS
        assert len(PHANTOM_ENGINE_STUBS) == 7

    def test_held_block_writer_stub(self):
        from zados.cognitive_engines.py_engines.phantom_stubs import HeldThinkingBlockWriterEngine
        engine = HeldThinkingBlockWriterEngine()
        assert engine.engine_id == 34
        status = engine.get_status()
        assert status["stub"] is True
        result = engine.process()
        assert result["held_blocks_written"] == 0

    def test_core_memory_gate_stub(self):
        from zados.cognitive_engines.py_engines.phantom_stubs import CoreMemoryUpdateGateEngine
        engine = CoreMemoryUpdateGateEngine()
        assert engine.engine_id == 40
        result = engine.process()
        assert result["updates_approved"] == 0

    def test_all_stubs_have_correct_ids(self):
        from zados.cognitive_engines.py_engines.phantom_stubs import PHANTOM_ENGINE_STUBS
        expected_ids = {34, 35, 36, 37, 38, 39, 40}
        assert set(PHANTOM_ENGINE_STUBS.keys()) == expected_ids

    def test_all_stubs_implement_pattern_a(self):
        from zados.cognitive_engines.py_engines.phantom_stubs import PHANTOM_ENGINE_STUBS
        for eid, cls in PHANTOM_ENGINE_STUBS.items():
            engine = cls()
            # Pattern A: update_neurochem_state, process, get_status
            engine.update_neurochem_state({"da": 0.5, "ne": 0.3})
            result = engine.process()
            assert isinstance(result, dict)
            status = engine.get_status()
            assert "engine_id" in status


# ===========================================================================
# Tests: Pipeline scope wiring
# ===========================================================================

class TestPipelineScopeWiring:
    def test_m1_sets_scope_on_bundle(self):
        from zados.core.inputs.learning_modes.human_teaches import HumanTeachesPipeline
        from zados.memory.managers.pipeline_scopes import PIPELINE_M1
        pipe = HumanTeachesPipeline(
            answer_pipeline=MockAnswerPipeline(),
            learning_log=MockLearningLog(),
            unsolved_buffer=MockUnsolvedBuffer(),
            pipeline_scope=PIPELINE_M1,
        )
        bundle = MockInputBundle(raw_text="Teach me about X")
        session = MockSessionState()
        result = pipe.process_turn(bundle, session)
        # Verify scope was attached to bundle
        assert hasattr(bundle, "_pipeline_read_scope")

    def test_m2_sets_retroactive_flag(self):
        from zados.core.inputs.learning_modes.peer_review import PeerReviewPipeline
        from zados.memory.managers.pipeline_scopes import PIPELINE_M2
        pipe = PeerReviewPipeline(
            answer_pipeline=MockAnswerPipeline(),
            learning_log=MockLearningLog(),
            unsolved_buffer=MockUnsolvedBuffer(),
            pipeline_scope=PIPELINE_M2,
        )
        bundle = MockInputBundle(raw_text="Review this claim")
        session = MockSessionState()
        result = pipe.process_turn(bundle, session)
        assert bundle.context_flags.get("retroactive_contrast") is True


# ===========================================================================
# Tests: LearningModeConfig dataclass
# ===========================================================================

class TestLearningModeConfigType:
    def test_default_values(self):
        from zados.core.types import LearningModeConfig
        c = LearningModeConfig()
        assert c.semantic_expansion_max_hops == 3
        assert c.generate_response is True
        assert c.use_retroactive_contrast is False

    def test_custom_values(self):
        from zados.core.types import LearningModeConfig
        c = LearningModeConfig(
            generate_response=False,
            response_depth="none",
        )
        assert c.generate_response is False
        assert c.response_depth == "none"


class TestPendingCoreMemoryUpdate:
    def test_default_not_applied(self):
        from zados.core.types import PendingCoreMemoryUpdate
        u = PendingCoreMemoryUpdate()
        assert u.applied is False
        assert u.update_id  # Auto-generated

    def test_custom_values(self):
        from zados.core.types import PendingCoreMemoryUpdate
        u = PendingCoreMemoryUpdate(
            core_memory_key="core_001",
            proposed_value="New value",
            confidence=0.8,
        )
        assert u.core_memory_key == "core_001"
        assert u.confidence == 0.8
