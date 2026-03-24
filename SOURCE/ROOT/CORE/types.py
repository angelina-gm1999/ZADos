"""
ZA-DOS Core Pipeline — Data Types (v0.6).

All pipeline data-transfer objects live here so every phase module and the
AnswerPipeline/SessionOrchestrator can import from a single place.

v0.5 types: InputBundle through SessionState  (unchanged)
v0.6 types: InputType through SelfRefResult   (Matrioshka layer)
"""
from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------

class PipelineValidationError(Exception):
    """Raised by Phase 0 when the InputBundle fails validation."""


# ------------------------------------------------------------------
# Pipeline input
# ------------------------------------------------------------------

@dataclass
class InputBundle:
    """Spec §3.1 — Pipeline 1 output / Pipeline 2 input."""

    raw_text: str
    intent_archetype: str = ""
    intent_vector: Dict[str, float] = field(default_factory=dict)
    nt_signals: Dict[str, Dict[str, float]] = field(default_factory=dict)
    emotion_profile: Dict[str, float] = field(default_factory=dict)
    active_mode: str = ""
    engine_weights: Dict[str, float] = field(default_factory=dict)
    context_flags: Dict[str, bool] = field(default_factory=dict)
    safety_tier: str = "NORMAL"
    mtmm_context_window: List[Any] = field(default_factory=list)
    mission_briefing: Any = None      # MemoryPacket or str
    osc_state: Any = None             # OscillationState
    extractor_state: Any = None       # ExtractorState
    time_context: Dict[str, Any] = field(default_factory=dict)  # TimeContextSnapshot.to_dict()


# ------------------------------------------------------------------
# Per-phase result types
# ------------------------------------------------------------------

@dataclass
class PerceptionSnapshot:
    """Phase 1 output — perception layer results."""

    intent_archetype: str = ""
    intent_vector: Dict[str, float] = field(default_factory=dict)
    intent_confidence: float = 0.0
    intent_result: Any = None                # IntentionMapResult (E23)
    ranked_facets: List[Dict[str, Any]] = field(default_factory=list)   # E8 scored items
    filtered_facets: List[Dict[str, Any]] = field(default_factory=list) # E11 filtered
    entity_triples: List[Tuple[str, str, str]] = field(default_factory=list)  # E18
    pattern_list: List[Dict[str, Any]] = field(default_factory=list)    # E19
    engine_statuses: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class NTModulationResult:
    """Phase 2 output — NT modulation + mode selection."""

    mode_token: str = "Normal"
    reward_profile_name: str = "regular_input"
    engine_weights: Dict[str, float] = field(default_factory=dict)
    metrics: Any = None                # NeurochemicalMetrics
    metrics_dict: Dict[str, float] = field(default_factory=dict)
    nt_snapshot: Dict[str, float] = field(default_factory=dict)     # lowercase NT→C
    osc_snapshot: Dict[str, float] = field(default_factory=dict)    # band→amplitude
    mode_selection_result: Any = None  # ModeSelectionResult
    extractor_result: Any = None       # ExtractorResult from sub-component run
    updated_extractor_state: Any = None  # ExtractorState after this turn


@dataclass
class EngineDispatchResult:
    """Phase 3 output — engine dispatch."""

    engine_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    engines_run: List[int] = field(default_factory=list)
    engines_skipped: List[int] = field(default_factory=list)
    e28_result: Any = None             # EmotionalDetectionResult


@dataclass
class ThinkingResult:
    """Phase 4 output — verbalized thinking (VT / LLM pass 1)."""

    thinking_trace: str = ""
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class RewardEvaluationResult:
    """Phase 5 output — two-pathway reward evaluation."""

    phase5_result: Any = None          # Phase5Result
    tonic_applied: bool = False
    phasic_applied: bool = False


@dataclass
class AnswerResult:
    """Phase 6 output — final answer (RG / LLM pass 2)."""

    final_answer: str = ""
    directive_applied: str = "allow"   # allow / suppress / abstain


@dataclass
class PostProcessResult:
    """Phase 7 output — post-processing & memory loop."""

    memory_packet: Any = None          # MemoryPacket
    compression_policy: str = ""
    learning_updates: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Pipeline-level aggregates
# ------------------------------------------------------------------

@dataclass
class PipelineState:
    """Accumulates across all phases for a single turn."""

    bundle: InputBundle = field(default_factory=lambda: InputBundle(raw_text=""))
    stmm: Any = None                   # STMMStore — central state bridge
    perception: Optional[PerceptionSnapshot] = None
    modulation: Optional[NTModulationResult] = None
    dispatch: Optional[EngineDispatchResult] = None
    thinking: Optional[ThinkingResult] = None
    reward: Optional[RewardEvaluationResult] = None
    answer: Optional[AnswerResult] = None
    postprocess: Optional[PostProcessResult] = None
    turn_index: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class PipelineResult:
    """Returned by AnswerPipeline.process_turn()."""

    final_answer: str = ""
    state: Optional[PipelineState] = None
    directive: str = "allow"
    phase5_result: Any = None          # Phase5Result


# ------------------------------------------------------------------
# Session-level state
# ------------------------------------------------------------------

@dataclass
class SessionState:
    """Persistent state across turns within a session."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    branch: str = "C"                  # A / B / C (time-delta branching)
    mission_briefing: Any = None       # MemoryPacket or str
    turn_count: int = 0
    session_start_time: float = field(default_factory=time.time)  # Unix epoch when session opened
    last_interaction_timestamp: float = 0.0
    extractor_state: Any = None        # ExtractorState
    osc_state: Any = None              # OscillationState
    initial_mode: str = "Normal"
    reward_profile_name: str = "regular_input"
    # Accumulated E17 reward-learning adjustments to domain weights.
    # Keys: "logic_weight", "ethics_weight", "innovation_weight", "attunement_weight"
    # Values: current learned weight [0.0, 1.0], initially empty (static profile used).
    learned_domain_weights: Dict[str, float] = field(default_factory=dict)
    # v0.6 — Matrioshka extensions
    active_learning_mode: Optional[str] = None   # "M1".."M5" while in learning
    session_mode: str = "regular"                 # regular / learning / sleep / meta


# ======================================================================
# v0.6  —  Matrioshka Pipeline Layer Types
# ======================================================================

# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------

class InputType(enum.Enum):
    """Top-level classification of incoming input."""
    MESSAGE = "message"
    FUNCTION = "function"


class MessageSubType(enum.Enum):
    """Sub-classification for MESSAGE-type inputs."""
    REGULAR = "regular"
    LEARNING_MODE = "learning_mode"
    SELF_REFLECTIVE = "self_reflective"


class FunctionSubType(enum.Enum):
    """Sub-classification for FUNCTION-type (command) inputs."""
    SLEEP = "sleep"
    META_LEARNING = "meta_learning"


class SleepVariant(enum.Enum):
    """Variants within Sleep mode."""
    REM = "rem"
    DREAM = "dream"


class MetaLearningVariant(enum.Enum):
    """Variants within Meta-Learning mode."""
    HOMEWORK = "homework"
    REFLECTIVE = "reflective"


class EngineTier(enum.Enum):
    """Engine activation tier — controls weight in dispatch.

    T1 = critical  (weight 1.0 — always runs)
    T2 = important (weight 1.0 — runs when budget allows)
    T3 = optional  (weight 0.5 — runs if capacity remains)
    T4 = disabled  (weight 0.0 — suppressed in this mode)
    """
    T1 = 1
    T2 = 2
    T3 = 3
    T4 = 4


class SubjectCategory(enum.Enum):
    """7 broad subject domains for engine tier adjustments."""
    TECHNICAL = "technical"
    SCIENTIFIC = "scientific"
    PHILOSOPHICAL = "philosophical"
    SOCIAL = "social"
    CREATIVE = "creative"
    PRACTICAL = "practical"
    MIXED = "mixed"


# ------------------------------------------------------------------
# v0.6 Dataclasses
# ------------------------------------------------------------------

@dataclass
class RawInput:
    """Outermost input to the Matrioshka layer (pre-classification)."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ClassificationResult:
    """Result of InputClassifier.classify()."""

    input_type: InputType = InputType.MESSAGE
    sub_type: Any = MessageSubType.REGULAR         # MessageSubType | FunctionSubType
    variant: Any = None                             # SleepVariant | MetaLearningVariant
    route_target: str = "regular"                   # human-readable route label
    confidence: float = 1.0
    raw_input: Optional[RawInput] = None
    learning_mode_number: int = 0                   # 1-5 if learning mode detected


@dataclass
class PipelineDepthConfig:
    """Per-intent depth tuning for the answer pipeline.

    Each field controls a dimension of processing depth / resource
    allocation.  Values are normalised 0.0-1.0 unless otherwise noted.
    """

    perception_depth: float = 0.7
    semiotics_depth: float = 0.5
    emotion_detection_sensitivity: float = 0.6
    phase1_depth: float = 0.7                      # how many perception engines to run
    phase3_engine_count_cap: int = 20              # max engines dispatched
    phase4_thinking_token_budget: int = 512        # max tokens for thinking trace
    phase5_reward_thoroughness: float = 0.7        # 0=fast, 1=exhaustive
    phase6_response_style: str = "balanced"        # balanced / concise / elaborate


@dataclass
class LearningLogEntry:
    """One turn's learning harvest (recorded by LearningLogPipeline)."""

    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    mode: str = ""                                  # "M1".."M5"
    subject: str = ""                               # SubjectCategory value
    session_id: str = ""

    # MemoryContrast deltas
    contrast_deltas: Dict[str, float] = field(default_factory=dict)

    # Learning event counters
    confirmations: int = 0
    contradictions: int = 0
    extensions: int = 0
    novel_entries: int = 0
    patterns_detected: int = 0

    # Engine harvest
    e19_patterns: List[Dict[str, Any]] = field(default_factory=list)    # PatternID results
    e20_comparisons: List[Dict[str, Any]] = field(default_factory=list) # PatternComparison
    e17_rewards: List[Dict[str, Any]] = field(default_factory=list)     # RPE events
    e25_meta_updates: List[Dict[str, Any]] = field(default_factory=list)  # Meta-learning

    # Reward domain scores (populated from Phase 5 result)
    reward_scores: Dict[str, float] = field(default_factory=dict)  # domain→score

    # Processing flag
    processed: bool = False


@dataclass
class UnsolvedQuestion:
    """A question that remains unresolved, held in the unsolved buffer."""

    question_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    question_text: str = ""
    source_mode: str = ""                           # "M1".."M5" or "self_ref"
    source_context: str = ""                        # brief context snippet
    creation_date: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)
    urgency_score: float = 0.5                      # 0.0 - 1.0
    stagnation_time: float = 0.0                    # seconds since last attempt
    resolution_attempts: int = 0
    partial_answers: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    resolved: bool = False
    scope_tag: str = ""                             # "academic" / "general" / "identity"


@dataclass
class EmotionalPreset:
    """Mode-specific emotional landscape configuration (spec §2.7).

    Applied to InputBundle before pipeline entry to bias the
    neurochemical + oscillatory state toward a mode-appropriate profile.
    """

    nt_adjustments: Dict[str, Any] = field(default_factory=dict)
    oscillatory_bias: Dict[str, float] = field(default_factory=dict)
    reward_weight_overrides: Dict[str, float] = field(default_factory=dict)
    domain_weight_overrides: Dict[str, float] = field(default_factory=dict)
    risk_emotions: List[str] = field(default_factory=list)
    risk_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class ContextAnchor:
    """Anchor point for drift detection within a learning session."""

    raw_text: str = ""
    subject_hint: str = ""                          # SubjectCategory value
    intent_prior: str = ""                          # dominant intent at anchor time
    drift_reference: Dict[str, float] = field(default_factory=dict)  # embedding / hash
    timestamp: float = field(default_factory=time.time)
    active: bool = True


@dataclass
class StudyAction:
    """Action recommendation from M5 risk state detection."""

    action: str = ""               # "switch_material" / "study_break" / "mode_switch"
    reason: str = ""               # human-readable reason
    duration_minutes: int = 0      # for study_break
    suggest_mode: str = ""         # for mode_switch


@dataclass
class LearningModeConfig:
    """Per-mode pipeline depth and behaviour controls (Part 4 §1.1).

    Controls how deep each stage runs for a specific learning mode,
    and whether to generate a response (M5 autonomous = False).
    """
    semantic_expansion_max_hops: int = 3          # M1=2, M3=-1(unlimited), M5=3
    pattern_chain_max_depth: int = 3              # M1=2, M3=-1(unlimited)
    max_questions_per_turn: int = 2               # M1=2, M4=1, M3=-1(unlimited)
    response_depth: str = "full"                  # "abbreviated" / "full" / "none"
    generate_response: bool = True                # False for M5 autonomous mode
    use_retroactive_contrast: bool = False        # True for M2 (checks own prior outputs)
    contradiction_mode: str = "learning"          # "learning" / "adversarial" / "soft"


@dataclass
class PendingCoreMemoryUpdate:
    """Staged core memory update from M2 peer review (Part 4 §3.2 Stage 5b).

    NOT applied immediately — queued for Homework/Reflective Mode to
    apply via CoreMemoryUpdateGate.  Cannot be applied mid-conversation.
    """
    update_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    core_memory_key: str = ""                      # which core memory to update
    current_value: str = ""                        # existing value snapshot
    proposed_value: str = ""                       # corrected value from review
    correction_source: str = ""                    # brief context of the correction
    correction_session_id: str = ""
    emotion_snapshot: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0                        # reviewer confidence in correction
    timestamp: float = field(default_factory=time.time)
    applied: bool = False


@dataclass
class LearningModeResult:
    """Returned by any LearningModePipeline.process_turn()."""

    mode_number: int = 0                            # 1-5
    pipeline_result: Optional[PipelineResult] = None
    learning_entries: List[LearningLogEntry] = field(default_factory=list)
    unsolved_questions: List[UnsolvedQuestion] = field(default_factory=list)
    suggest_mode_change: str = ""                   # if non-empty, recommend switching
    study_action: Optional[StudyAction] = None      # M5 risk response
    # Part 4 additions
    pending_core_updates: List[PendingCoreMemoryUpdate] = field(default_factory=list)
    dream_candidates: List[str] = field(default_factory=list)  # question IDs flagged for dream
    contrast_challenges: List[Dict[str, Any]] = field(default_factory=list)  # M3 contradictions
    held_thinking_blocks: List[str] = field(default_factory=list)  # block IDs captured this turn


@dataclass
class SelfRefResult:
    """Returned by SelfReflectiveQueryPipeline.process_turn()."""

    selected_question: Optional[UnsolvedQuestion] = None
    context_gathered: Dict[str, Any] = field(default_factory=dict)
    synthesis: str = ""
    rerouted_to_m3: bool = False
    pipeline_result: Optional[PipelineResult] = None


# ======================================================================
# v0.6  —  Homework Mode Types (Part 5)
# ======================================================================

@dataclass
class HomeworkRunSummary:
    """Summary of a single homework pipeline run (Part 5 §5.3).

    Written to OverviewLogStore at the end of each homework session.
    Contains aggregate statistics across all processed batches.
    """

    session_id: str = ""
    timestamp: float = field(default_factory=time.time)

    # Batch processing stats
    batches_processed: int = 0

    # Lesson stats
    lessons_validated: int = 0
    lessons_pending: int = 0

    # Contradiction stats
    contradictions_resolved: int = 0
    contradictions_unresolved: int = 0

    # Question stats
    questions_resolved: int = 0
    questions_new: int = 0
    dream_candidates_flagged: int = 0

    # Core memory
    core_memory_updates_applied: int = 0

    # Fallacy/bias flags for Reflective Mode handoff
    fallacy_bias_flags: List[Dict[str, Any]] = field(default_factory=list)

    # Cross-batch meta-patterns
    meta_patterns: List[Dict[str, Any]] = field(default_factory=list)

    # Per-batch processing emphasis: batch_subject → deficit_domain
    processing_emphasis: Dict[str, str] = field(default_factory=dict)


@dataclass
class ReflectiveModeInput:
    """Packaged input for Reflective Mode handoff from Homework (Part 5 §4.4).

    Contains fallacy/bias flags and identity contradiction resolutions
    discovered during homework processing.  [SPEC NEEDED] — minimal stub.
    """

    fallacy_flags: List[Dict[str, Any]] = field(default_factory=list)
    bias_flags: List[Dict[str, Any]] = field(default_factory=list)
    identity_contradiction_resolutions: List[Dict[str, Any]] = field(default_factory=list)
    meta_patterns: List[Dict[str, Any]] = field(default_factory=list)
    source_homework_session: str = ""


# ======================================================================
# v0.6  —  Reflective Mode Types (Part 6)
# ======================================================================

@dataclass
class ReflectiveModeResult:
    """Summary of a single reflective pipeline run.

    Produced by ``ReflectivePipeline.process()`` and contains the outputs
    from both E31 (meta-learning analysis) and E32 (identity coherence),
    plus any identity store mutations applied during the run.
    """

    session_id: str = ""
    timestamp: float = field(default_factory=time.time)

    # E31 — meta-learning analysis
    learning_patterns: List[Dict[str, Any]] = field(default_factory=list)
    recurring_failures: List[Dict[str, Any]] = field(default_factory=list)
    mode_effectiveness: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    subject_proficiencies: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    style_preferences: List[Dict[str, Any]] = field(default_factory=list)
    learning_recommendations: List[Dict[str, Any]] = field(default_factory=list)

    # E32 — identity coherence analysis
    identity_coherence_status: str = "coherent"
    coherence_score: float = 1.0
    core_contradictions: List[Dict[str, Any]] = field(default_factory=list)
    fragile_conclusions: List[Dict[str, Any]] = field(default_factory=list)
    alignment_issues: List[Dict[str, Any]] = field(default_factory=list)
    identity_themes: List[Dict[str, Any]] = field(default_factory=list)

    # Identity store mutations applied
    conclusions_reinforced: int = 0
    conclusions_created: int = 0
    conclusions_recommended_for_update: int = 0
    journal_entries_created: int = 0
    pending_updates_analysed: int = 0

    # Cross-referencing (E31 × E32)
    cross_references: List[Dict[str, Any]] = field(default_factory=list)

    # Input summary
    fallacy_flags_processed: int = 0
    bias_flags_processed: int = 0
    meta_patterns_processed: int = 0
    learning_logs_analysed: int = 0
