"""Tests for Homework Mode Pipeline (Part 5).

Covers:
  - Phase 0: batch grouping, deficit profiling, sort order
  - Phase 1: analysis (relevance scoring, pattern aggregation, contrast)
  - Phase 2: processing (contradiction resolution, fallacy/bias logging)
  - Phase 3: question resolution, dream candidate flagging
  - Phase 4: synthesis (lesson finalization, knowledge map update, core memory gate)
  - Phase 5: output (LTMM writes, summary generation, reflective handoff)
  - Engine tier matrix: homework base tiers, budget cap 22, subject promotions
  - Deficit profiler: deficit computation, batch sorting, emphasis mapping
  - Data types: HomeworkRunSummary, ReflectiveModeInput, LearningLogEntry.reward_scores
"""
import time
from unittest.mock import MagicMock, patch

import pytest

from zados.core.commanded.meta_learning_mode.homework_mode.deficit_profiler import (
    REWARD_DOMAINS,
    compute_batch_deficit,
    get_engine_emphasis,
    identify_deficit_domain,
    sort_batches_by_deficit,
)
from zados.core.commanded.meta_learning_mode.homework_mode.pipeline import (
    HomeworkPipeline,
    ProcessedBatch,
    ProcessingOutput,
    _DREAM_STAGNATION_THRESHOLD,
    _MIN_VALIDATION_CONFIDENCE,
)
from zados.core.processes.engine_toolkit import (
    BASE_TIERS,
    BUDGET_CAPS,
    EngineToolkit,
)
from zados.core.processes.learning_log import LearningLogPipeline
from zados.core.processes.unsolved_buffer import UnsolvedBuffer
from zados.core.types import (
    EngineTier,
    HomeworkRunSummary,
    InputBundle,
    LearningLogEntry,
    LearningModeResult,
    PendingCoreMemoryUpdate,
    PipelineResult,
    PipelineState,
    EngineDispatchResult,
    ReflectiveModeInput,
    SessionState,
    SubjectCategory,
    UnsolvedQuestion,
)


# -----------------------------------------------------------------------
# Helper factories
# -----------------------------------------------------------------------

def _make_entry(
    mode="M1",
    subject="technical",
    confirmations=0,
    contradictions=0,
    novel_entries=0,
    patterns_detected=0,
    extensions=0,
    reward_scores=None,
    e19_patterns=None,
    e20_comparisons=None,
) -> LearningLogEntry:
    e = LearningLogEntry(
        mode=mode,
        subject=subject,
        session_id="test_session",
        confirmations=confirmations,
        contradictions=contradictions,
        novel_entries=novel_entries,
        patterns_detected=patterns_detected,
        extensions=extensions,
    )
    if reward_scores:
        e.reward_scores = reward_scores
    if e19_patterns:
        e.e19_patterns = e19_patterns
    if e20_comparisons:
        e.e20_comparisons = e20_comparisons
    return e


def _make_pipeline(
    answer_pipeline=None,
    learning_log=None,
    unsolved_buffer=None,
    memory_layer=None,
    specialized_logs=None,
) -> HomeworkPipeline:
    if answer_pipeline is None:
        answer_pipeline = MagicMock()
        answer_pipeline.process_turn.return_value = PipelineResult(
            final_answer="Test answer from pipeline",
            state=PipelineState(
                bundle=InputBundle(raw_text=""),
                dispatch=EngineDispatchResult(
                    engine_results={
                        1: {"contradictions": []},
                        2: {"paradoxes": []},
                        4: {"fallacies": []},
                        5: {"biases": []},
                        10: {"confidence_scores": {}},
                    },
                    engines_run=[1, 2, 4, 5, 10],
                ),
            ),
        )
    return HomeworkPipeline(
        answer_pipeline=answer_pipeline,
        learning_log=learning_log or LearningLogPipeline(),
        unsolved_buffer=unsolved_buffer or UnsolvedBuffer(),
        memory_layer=memory_layer,
        specialized_logs=specialized_logs,
    )


# =======================================================================
# Data Types
# =======================================================================

class TestHomeworkRunSummary:
    def test_default_values(self):
        s = HomeworkRunSummary()
        assert s.batches_processed == 0
        assert s.lessons_validated == 0
        assert s.contradictions_resolved == 0
        assert s.questions_resolved == 0
        assert s.core_memory_updates_applied == 0
        assert s.fallacy_bias_flags == []
        assert s.meta_patterns == []
        assert s.processing_emphasis == {}

    def test_custom_values(self):
        s = HomeworkRunSummary(
            session_id="hw1",
            batches_processed=3,
            lessons_validated=5,
            contradictions_resolved=2,
            questions_resolved=1,
            core_memory_updates_applied=1,
            processing_emphasis={"technical": "logic"},
        )
        assert s.session_id == "hw1"
        assert s.batches_processed == 3
        assert s.processing_emphasis == {"technical": "logic"}


class TestReflectiveModeInput:
    def test_default_values(self):
        r = ReflectiveModeInput()
        assert r.fallacy_flags == []
        assert r.bias_flags == []
        assert r.identity_contradiction_resolutions == []
        assert r.meta_patterns == []
        assert r.source_homework_session == ""

    def test_custom_values(self):
        r = ReflectiveModeInput(
            fallacy_flags=[{"name": "straw_man"}],
            source_homework_session="hw_session_1",
        )
        assert len(r.fallacy_flags) == 1
        assert r.source_homework_session == "hw_session_1"


class TestLearningLogEntryRewardScores:
    def test_default_empty(self):
        e = LearningLogEntry()
        assert e.reward_scores == {}

    def test_custom_scores(self):
        e = LearningLogEntry(
            reward_scores={"logic": 0.8, "innovation": 0.3},
        )
        assert e.reward_scores["logic"] == 0.8
        assert e.reward_scores["innovation"] == 0.3

    def test_reward_scores_independent(self):
        e1 = LearningLogEntry()
        e2 = LearningLogEntry()
        e1.reward_scores["logic"] = 0.5
        assert "logic" not in e2.reward_scores


class TestProcessedBatch:
    def test_default_values(self):
        b = ProcessedBatch()
        assert b.subject == ""
        assert b.deficit_domain == "mixed"
        assert b.entries == []
        assert b.contrast_deltas == []
        assert b.novel_patterns == []

    def test_custom_values(self):
        entries = [_make_entry()]
        b = ProcessedBatch(
            subject="technical",
            entries=entries,
            deficit_domain="logic",
            novel_patterns=[{"name": "pattern_a"}],
        )
        assert b.subject == "technical"
        assert len(b.entries) == 1
        assert b.deficit_domain == "logic"


class TestProcessingOutput:
    def test_default_values(self):
        p = ProcessingOutput()
        assert p.validated_lessons == []
        assert p.contradictions_resolved == []
        assert p.fallacy_flags == []

    def test_custom_values(self):
        p = ProcessingOutput(
            validated_lessons=[{"turn_id": "t1"}],
            fallacy_flags=[{"name": "ad_hominem"}],
        )
        assert len(p.validated_lessons) == 1
        assert len(p.fallacy_flags) == 1


# =======================================================================
# Deficit Profiler
# =======================================================================

class TestDeficitProfiler:
    def test_compute_batch_deficit_empty(self):
        profile = compute_batch_deficit([])
        for domain in REWARD_DOMAINS:
            assert profile[domain] == 0.5  # neutral default

    def test_compute_batch_deficit_single_entry(self):
        e = _make_entry(reward_scores={
            "logic": 0.8, "innovation": 0.2, "ethics": 0.6, "human_attunement": 0.5,
        })
        profile = compute_batch_deficit([e])
        assert profile["logic"] == 0.8
        assert profile["innovation"] == 0.2

    def test_compute_batch_deficit_averages(self):
        e1 = _make_entry(reward_scores={"logic": 0.8, "innovation": 0.4})
        e2 = _make_entry(reward_scores={"logic": 0.6, "innovation": 0.6})
        profile = compute_batch_deficit([e1, e2])
        assert abs(profile["logic"] - 0.7) < 0.01
        assert abs(profile["innovation"] - 0.5) < 0.01
        # Domains without data default to 0.5
        assert profile["ethics"] == 0.5

    def test_identify_deficit_domain_basic(self):
        profile = {"logic": 0.8, "innovation": 0.2, "ethics": 0.6, "human_attunement": 0.5}
        assert identify_deficit_domain(profile) == "innovation"

    def test_identify_deficit_domain_tie_alphabetical(self):
        profile = {"logic": 0.3, "innovation": 0.3, "ethics": 0.5, "human_attunement": 0.5}
        # Both at 0.3, alphabetical tie-break: "innovation" < "logic"
        assert identify_deficit_domain(profile) == "innovation"

    def test_identify_deficit_domain_empty(self):
        assert identify_deficit_domain({}) == "mixed"

    def test_sort_batches_by_deficit(self):
        entries_tech = [_make_entry(subject="technical", reward_scores={"logic": 0.8})]
        entries_phil = [_make_entry(subject="philosophical", reward_scores={"logic": 0.2})]
        batches = {
            "technical": entries_tech,
            "philosophical": entries_phil,
        }
        sorted_b = sort_batches_by_deficit(batches)
        # Philosophical has deeper deficit (0.2 vs 0.5 neutral in other domains)
        assert sorted_b[0][0] == "philosophical"
        assert sorted_b[1][0] == "technical"

    def test_sort_batches_returns_deficit_domain(self):
        entries = [_make_entry(reward_scores={"logic": 0.1, "innovation": 0.9})]
        batches = {"tech": entries}
        result = sort_batches_by_deficit(batches)
        assert result[0][2] == "logic"  # deepest deficit

    def test_get_engine_emphasis_logic(self):
        emphasis = get_engine_emphasis("logic")
        assert "contradiction_detection_engine" in emphasis
        assert "pln_engine" in emphasis

    def test_get_engine_emphasis_innovation(self):
        emphasis = get_engine_emphasis("innovation")
        assert "simulated_opposition_engine" in emphasis

    def test_get_engine_emphasis_ethics(self):
        emphasis = get_engine_emphasis("ethics")
        assert "bias_detection_engine" in emphasis

    def test_get_engine_emphasis_human_attunement(self):
        emphasis = get_engine_emphasis("human_attunement")
        assert "contextual_learning_engine" in emphasis

    def test_get_engine_emphasis_mixed(self):
        emphasis = get_engine_emphasis("mixed")
        assert "contradiction_detection_engine" in emphasis

    def test_get_engine_emphasis_unknown_defaults_to_mixed(self):
        emphasis = get_engine_emphasis("nonexistent")
        assert emphasis == get_engine_emphasis("mixed")


# =======================================================================
# Engine Toolkit — Homework Mode
# =======================================================================

class TestHomeworkEngineTiers:
    def test_homework_mode_exists(self):
        assert "homework" in BASE_TIERS

    def test_homework_budget_cap_22(self):
        assert BUDGET_CAPS["homework"] == 22

    def test_homework_t1_count(self):
        tiers = BASE_TIERS["homework"]
        t1_count = sum(1 for t in tiers.values() if t == EngineTier.T1)
        assert t1_count >= 15  # spec says 18 T1, but phantoms forced to T4

    def test_homework_t2_count(self):
        tiers = BASE_TIERS["homework"]
        t2_count = sum(1 for t in tiers.values() if t == EngineTier.T2)
        assert t2_count >= 6

    def test_homework_input_relevance_t4(self):
        tiers = BASE_TIERS["homework"]
        assert tiers["input_relevance_evaluation_engine"] == EngineTier.T4

    def test_homework_emotional_detection_t3(self):
        tiers = BASE_TIERS["homework"]
        assert tiers["emotional_detection_engine"] == EngineTier.T3

    def test_homework_contradiction_t1(self):
        tiers = BASE_TIERS["homework"]
        assert tiers["contradiction_detection_engine"] == EngineTier.T1

    def test_homework_paradox_t1(self):
        tiers = BASE_TIERS["homework"]
        assert tiers["paradox_detection_engine"] == EngineTier.T1

    def test_homework_pln_t1(self):
        tiers = BASE_TIERS["homework"]
        assert tiers["pln_engine"] == EngineTier.T1

    def test_homework_fallacy_t2(self):
        tiers = BASE_TIERS["homework"]
        assert tiers["fallacy_detection_engine"] == EngineTier.T2

    def test_homework_simulated_opposition_t2(self):
        tiers = BASE_TIERS["homework"]
        assert tiers["simulated_opposition_engine"] == EngineTier.T2

    def test_homework_resolve_respects_budget(self):
        tk = EngineToolkit()
        tiers = tk.resolve("homework", SubjectCategory.MIXED)
        active = sum(1 for t in tiers.values() if t in (EngineTier.T1, EngineTier.T2))
        assert active <= BUDGET_CAPS["homework"]

    def test_homework_resolve_with_subject_promotions(self):
        tk = EngineToolkit()
        tiers = tk.resolve("homework", SubjectCategory.PHILOSOPHICAL)
        # Socratic should be promoted to at least T1 for philosophical
        assert tiers["socratic_reasoning_engine"] in (EngineTier.T1, EngineTier.T2)

    def test_homework_nt_engine_t1_for_diagnostics(self):
        tiers = BASE_TIERS["homework"]
        assert tiers["neurochemical_homeostatic_engine"] == EngineTier.T1

    def test_homework_highest_budget(self):
        """Homework should have the highest budget cap of any mode."""
        assert BUDGET_CAPS["homework"] == max(BUDGET_CAPS.values())


# =======================================================================
# Learning Log — reward_scores integration
# =======================================================================

class TestLearningLogRewardScores:
    def test_record_turn_without_reward_result(self):
        ll = LearningLogPipeline()
        entry = ll.record_turn(
            mode="M1", subject="technical", session_id="s1",
            engine_results={},
        )
        assert entry.reward_scores == {}

    def test_record_turn_with_reward_result(self):
        ll = LearningLogPipeline()

        # Mock Phase5Result with domain_results
        reward_result = MagicMock()
        domain_logic = MagicMock()
        domain_logic.general_score = 0.75
        domain_innovation = MagicMock()
        domain_innovation.general_score = 0.4
        reward_result.domain_results = {
            "logic": domain_logic,
            "innovation": domain_innovation,
        }

        entry = ll.record_turn(
            mode="M1", subject="technical", session_id="s1",
            engine_results={},
            reward_result=reward_result,
        )
        assert entry.reward_scores["logic"] == 0.75
        assert entry.reward_scores["innovation"] == 0.4

    def test_record_turn_reward_result_dict_fallback(self):
        ll = LearningLogPipeline()

        # dict-style domain results
        reward_result = MagicMock()
        reward_result.domain_results = {
            "logic": {"general_score": 0.6},
        }
        # getattr returns None for general_score since dict doesn't have attrs
        entry = ll.record_turn(
            mode="M1", subject="technical", session_id="s1",
            engine_results={},
            reward_result=reward_result,
        )
        assert entry.reward_scores["logic"] == 0.6


# =======================================================================
# Homework Pipeline — Phase 0
# =======================================================================

class TestHomeworkPhase0:
    def test_empty_logs_returns_empty_batches(self):
        hp = _make_pipeline()
        batches, unprocessed = hp._phase0_input_assembly()
        assert batches == []
        assert unprocessed == []

    def test_groups_by_subject(self):
        ll = LearningLogPipeline()
        ll.record_turn("M1", "technical", "s1", {})
        ll.record_turn("M1", "philosophical", "s1", {})
        ll.record_turn("M1", "technical", "s1", {})

        hp = _make_pipeline(learning_log=ll)
        batches, unprocessed = hp._phase0_input_assembly()
        assert len(unprocessed) == 3
        subjects = [b[0] for b in batches]
        assert "technical" in subjects
        assert "philosophical" in subjects

    def test_batches_sorted_by_deficit(self):
        ll = LearningLogPipeline()
        e1 = ll.record_turn("M1", "technical", "s1", {})
        e1.reward_scores = {"logic": 0.9}
        e2 = ll.record_turn("M1", "philosophical", "s1", {})
        e2.reward_scores = {"logic": 0.1}

        hp = _make_pipeline(learning_log=ll)
        batches, _ = hp._phase0_input_assembly()
        # Philosophical has deeper deficit — processed first
        assert batches[0][0] == "philosophical"


# =======================================================================
# Homework Pipeline — Phase 1
# =======================================================================

class TestHomeworkPhase1:
    def test_analysis_creates_processed_batch(self):
        entries = [_make_entry(patterns_detected=3, novel_entries=2)]
        hp = _make_pipeline()
        batch = hp._phase1_analysis("technical", entries, "logic")
        assert isinstance(batch, ProcessedBatch)
        assert batch.subject == "technical"
        assert batch.deficit_domain == "logic"
        assert len(batch.relevance_scored_entries) == 1

    def test_relevance_scoring(self):
        e = _make_entry(
            patterns_detected=5, novel_entries=3, contradictions=2,
            confirmations=1, extensions=1,
        )
        score = HomeworkPipeline._compute_entry_relevance(e)
        assert score > 0.0
        assert score <= 5.0

    def test_relevance_scoring_empty_entry(self):
        e = _make_entry()
        score = HomeworkPipeline._compute_entry_relevance(e)
        assert score == 0.01  # penalised empty

    def test_patterns_categorized(self):
        entries = [_make_entry(
            e19_patterns=[
                {"name": "p1", "status": "CONFIRMED"},
                {"name": "p2", "status": "CANDIDATE"},
            ]
        )]
        hp = _make_pipeline()
        batch = hp._phase1_analysis("technical", entries, "mixed")
        assert len(batch.pattern_reinforcements) == 1
        assert len(batch.novel_patterns) == 1

    def test_contrast_deltas_collected(self):
        e = _make_entry()
        e.contrast_deltas = {"divergence": 0.7}
        hp = _make_pipeline()
        batch = hp._phase1_analysis("technical", [e], "mixed")
        assert len(batch.contrast_deltas) == 1

    def test_e20_divergence_flags_contradiction(self):
        e = _make_entry(e20_comparisons=[{"divergence": 0.8, "detail": "mismatch"}])
        hp = _make_pipeline()
        batch = hp._phase1_analysis("technical", [e], "mixed")
        assert len(batch.contradiction_candidates) >= 1


# =======================================================================
# Homework Pipeline — Phase 2
# =======================================================================

class TestHomeworkPhase2:
    def test_processing_returns_output(self):
        batch = ProcessedBatch(
            subject="technical",
            entries=[_make_entry(confirmations=2)],
            deficit_domain="logic",
            relevance_scored_entries=[(1.0, _make_entry(confirmations=2))],
        )
        hp = _make_pipeline()
        output = hp._phase2_processing(batch)
        assert isinstance(output, ProcessingOutput)

    def test_validated_lessons_from_confirmations(self):
        e = _make_entry(confirmations=3, contradictions=0)
        batch = ProcessedBatch(
            subject="technical",
            entries=[e],
            deficit_domain="logic",
            relevance_scored_entries=[(1.0, e)],
        )
        hp = _make_pipeline()
        output = hp._phase2_processing(batch)
        assert len(output.validated_lessons) >= 1

    def test_no_validation_with_contradictions(self):
        e = _make_entry(confirmations=3, contradictions=1)
        batch = ProcessedBatch(
            subject="technical",
            entries=[e],
            deficit_domain="logic",
            relevance_scored_entries=[(1.0, e)],
        )
        hp = _make_pipeline()
        output = hp._phase2_processing(batch)
        assert len(output.validated_lessons) == 0

    def test_pipeline_called_with_homework_mode(self):
        mock_pipeline = MagicMock()
        mock_pipeline.process_turn.return_value = PipelineResult(
            final_answer="",
            state=PipelineState(
                bundle=InputBundle(raw_text=""),
                dispatch=EngineDispatchResult(engine_results={}, engines_run=[]),
            ),
        )
        e = _make_entry(confirmations=1)
        batch = ProcessedBatch(
            subject="technical",
            entries=[e],
            deficit_domain="logic",
            relevance_scored_entries=[(1.0, e)],
        )
        hp = _make_pipeline(answer_pipeline=mock_pipeline)
        hp._phase2_processing(batch)
        assert mock_pipeline.process_turn.called
        call_args = mock_pipeline.process_turn.call_args
        bundle = call_args[0][0]
        assert bundle.active_mode == "homework"

    def test_fallacy_bias_logged_to_spec_logs(self):
        mock_pipeline = MagicMock()
        mock_pipeline.process_turn.return_value = PipelineResult(
            final_answer="",
            state=PipelineState(
                bundle=InputBundle(raw_text=""),
                dispatch=EngineDispatchResult(
                    engine_results={
                        1: {"contradictions": []},
                        2: {"paradoxes": []},
                        4: {"fallacies": [{"name": "straw_man", "severity": "medium"}]},
                        5: {"biases": [{"name": "confirmation", "severity": "low"}]},
                        10: {},
                    },
                    engines_run=[1, 2, 4, 5, 10],
                ),
            ),
        )
        spec_logs = MagicMock()
        spec_logs.self_reflection = MagicMock()
        spec_logs.contradictions = MagicMock()

        e = _make_entry(confirmations=0)
        batch = ProcessedBatch(
            subject="technical",
            entries=[e],
            deficit_domain="logic",
            relevance_scored_entries=[(1.0, e)],
        )
        hp = _make_pipeline(answer_pipeline=mock_pipeline, specialized_logs=spec_logs)
        output = hp._phase2_processing(batch)
        assert len(output.fallacy_flags) >= 1
        assert len(output.bias_flags) >= 1
        # SelfReflectionLog should have been called
        assert spec_logs.self_reflection.record.called


# =======================================================================
# Homework Pipeline — Phase 3
# =======================================================================

class TestHomeworkPhase3:
    def test_dream_candidate_flagging(self):
        ub = UnsolvedBuffer()
        q = ub.add("What is X?", source_mode="M1", urgency_score=0.5)
        # Simulate many failed attempts
        for _ in range(_DREAM_STAGNATION_THRESHOLD):
            ub.mark_attempted(q.question_id, partial_answer="partial")

        hp = _make_pipeline(unsolved_buffer=ub)
        result = hp._phase3_question_resolution([], ub.get_active(), SessionState())
        assert result["dream_candidates"] >= 1
        assert "dream_candidate" in q.tags

    def test_new_questions_from_unresolved_contradictions(self):
        ub = UnsolvedBuffer()
        batch = ProcessedBatch(subject="technical")
        proc = ProcessingOutput(
            contradictions_unresolved=[{"formulation": "A vs B conflict"}],
        )
        hp = _make_pipeline(unsolved_buffer=ub)
        result = hp._phase3_question_resolution(
            [(batch, proc)], [], SessionState()
        )
        assert result["new_questions"] >= 1
        assert ub.size >= 1

    def test_question_resolution_via_pipeline(self):
        ub = UnsolvedBuffer()
        ub.add("What is quantum entanglement?", source_mode="M3")

        mock_pipeline = MagicMock()
        mock_pipeline.process_turn.return_value = PipelineResult(
            final_answer="Quantum entanglement is a phenomenon where particles become correlated in such a way that measuring one instantly affects the other regardless of distance.",
        )
        hp = _make_pipeline(answer_pipeline=mock_pipeline, unsolved_buffer=ub)
        result = hp._phase3_question_resolution([], ub.get_active(), SessionState())
        assert result["resolved"] >= 1

    def test_answer_matches_question_overlap(self):
        q = UnsolvedQuestion(question_text="What is force in physics?")
        assert HomeworkPipeline._answer_matches_question(
            q, "Force is mass times acceleration in physics", "technical"
        )

    def test_answer_matches_question_no_overlap(self):
        q = UnsolvedQuestion(question_text="What is quantum entanglement?")
        assert not HomeworkPipeline._answer_matches_question(
            q, "The recipe calls for sugar", "cooking"
        )


# =======================================================================
# Homework Pipeline — Phase 4
# =======================================================================

class TestHomeworkPhase4:
    def test_cross_batch_pattern_synthesis(self):
        batch1 = ProcessedBatch(
            subject="technical",
            novel_patterns=[{"name": "recursion"}],
        )
        batch2 = ProcessedBatch(
            subject="philosophical",
            novel_patterns=[{"name": "recursion"}],
        )
        proc = ProcessingOutput()
        hp = _make_pipeline()
        result = hp._phase4_synthesis(
            [(batch1, proc), (batch2, proc)], SessionState()
        )
        assert len(result["meta_patterns"]) >= 1
        mp = result["meta_patterns"][0]
        assert mp["type"] == "cross_domain"
        assert "technical" in mp["subjects"]
        assert "philosophical" in mp["subjects"]

    def test_no_meta_patterns_when_no_overlap(self):
        batch1 = ProcessedBatch(
            subject="technical",
            novel_patterns=[{"name": "recursion"}],
        )
        batch2 = ProcessedBatch(
            subject="philosophical",
            novel_patterns=[{"name": "dialectic"}],
        )
        proc = ProcessingOutput()
        hp = _make_pipeline()
        result = hp._phase4_synthesis(
            [(batch1, proc), (batch2, proc)], SessionState()
        )
        assert len(result["meta_patterns"]) == 0


# =======================================================================
# Homework Pipeline — Phase 5
# =======================================================================

class TestHomeworkPhase5:
    def test_marks_logs_processed(self):
        ll = LearningLogPipeline()
        e1 = ll.record_turn("M1", "technical", "s1", {})
        e2 = ll.record_turn("M1", "philosophical", "s1", {})

        hp = _make_pipeline(learning_log=ll)
        summary = HomeworkRunSummary(session_id="hw1")
        hp._phase5_output(summary, [e1, e2], [], SessionState())
        assert e1.processed is True
        assert e2.processed is True

    def test_reflective_handoff_prepared(self):
        hp = _make_pipeline()
        summary = HomeworkRunSummary(
            session_id="hw1",
            fallacy_bias_flags=[
                {"type": "fallacy", "name": "straw_man"},
                {"type": "bias", "name": "confirmation"},
            ],
            meta_patterns=[{"type": "cross_domain"}],
        )
        # Phase 5 should log the reflective handoff (no crash)
        hp._phase5_output(summary, [], [], SessionState())


# =======================================================================
# Homework Pipeline — Full integration
# =======================================================================

class TestHomeworkFullProcess:
    def test_empty_run(self):
        hp = _make_pipeline()
        result = hp.process(SessionState())
        assert result["status"] == "completed"
        assert result["batches_processed"] == 0
        assert result["logs_processed"] == 0

    def test_basic_run_with_logs(self):
        ll = LearningLogPipeline()
        e1 = ll.record_turn("M1", "technical", "s1", {})
        e1.confirmations = 2
        e1.reward_scores = {"logic": 0.7}

        hp = _make_pipeline(learning_log=ll)
        result = hp.process(SessionState(session_id="hw_test"))
        assert result["status"] == "completed"
        assert result["batches_processed"] >= 1
        assert result["logs_processed"] >= 1
        assert "processing_emphasis" in result

    def test_run_with_unsolved_questions(self):
        ub = UnsolvedBuffer()
        ub.add("What is recursion?", source_mode="M1")

        mock_pipeline = MagicMock()
        mock_pipeline.process_turn.return_value = PipelineResult(
            final_answer="Recursion is a method where the solution depends on solutions to smaller instances of the same problem.",
            state=PipelineState(
                bundle=InputBundle(raw_text=""),
                dispatch=EngineDispatchResult(engine_results={}, engines_run=[]),
            ),
        )

        hp = _make_pipeline(answer_pipeline=mock_pipeline, unsolved_buffer=ub)
        result = hp.process(SessionState())
        assert result["status"] == "completed"

    def test_return_dict_has_all_fields(self):
        hp = _make_pipeline()
        result = hp.process(SessionState())
        expected_keys = [
            "status", "session_id", "processing_time_s",
            "batches_processed", "logs_processed",
            "lessons_validated", "lessons_pending",
            "contradictions_resolved", "contradictions_unresolved",
            "questions_resolved", "questions_new",
            "dream_candidates_flagged", "core_memory_updates_applied",
            "fallacy_bias_flags", "meta_patterns", "processing_emphasis",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_multi_subject_processing(self):
        ll = LearningLogPipeline()
        for subj in ["technical", "philosophical", "creative"]:
            e = ll.record_turn("M1", subj, "s1", {})
            e.confirmations = 1

        hp = _make_pipeline(learning_log=ll)
        result = hp.process(SessionState())
        assert result["batches_processed"] == 3
        assert len(result["processing_emphasis"]) == 3
