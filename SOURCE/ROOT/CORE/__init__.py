"""
ZA-DOS Core Pipeline (v0.6).

Public API
----------
AnswerPipeline
    Single-turn orchestrator.  Sequences Phases 0-7.

SessionOrchestrator
    Session lifecycle manager.  Handles boot, per-turn, drift detection.

InputClassifier
    v0.6 Matrioshka outer layer.  Classifies input and routes to
    the correct sub-pipeline.  Import from ``zados.core.main``.

InputBundle
    Pipeline 1 output / Pipeline 2 input.

SessionState
    Persistent state across turns.

PipelineResult
    Returned by AnswerPipeline.process_turn().
"""
from zados.core.pipeline import AnswerPipeline
from zados.core.session import SessionOrchestrator
from zados.core.types import (
    AnswerResult,
    ClassificationResult,
    ContextAnchor,
    EmotionalPreset,
    EngineDispatchResult,
    EngineTier,
    FunctionSubType,
    InputBundle,
    InputType,
    LearningLogEntry,
    LearningModeResult,
    MessageSubType,
    MetaLearningVariant,
    NTModulationResult,
    PerceptionSnapshot,
    PipelineDepthConfig,
    PipelineResult,
    PipelineState,
    PipelineValidationError,
    PostProcessResult,
    RawInput,
    ReflectiveModeInput,
    ReflectiveModeResult,
    RewardEvaluationResult,
    SelfRefResult,
    SessionState,
    SleepVariant,
    SubjectCategory,
    ThinkingResult,
    UnsolvedQuestion,
)

__all__ = [
    # v0.5 — core pipeline
    "AnswerPipeline",
    "SessionOrchestrator",
    "InputBundle",
    "SessionState",
    "PipelineResult",
    "PipelineState",
    "PipelineValidationError",
    "PerceptionSnapshot",
    "NTModulationResult",
    "EngineDispatchResult",
    "ThinkingResult",
    "RewardEvaluationResult",
    "AnswerResult",
    "PostProcessResult",
    # v0.6 — Matrioshka types
    "InputType",
    "MessageSubType",
    "FunctionSubType",
    "SleepVariant",
    "MetaLearningVariant",
    "EngineTier",
    "SubjectCategory",
    "RawInput",
    "ClassificationResult",
    "PipelineDepthConfig",
    "LearningLogEntry",
    "UnsolvedQuestion",
    "EmotionalPreset",
    "ContextAnchor",
    "LearningModeResult",
    "SelfRefResult",
    "ReflectiveModeInput",
    "ReflectiveModeResult",
]
