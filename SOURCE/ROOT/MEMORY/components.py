"""
STMM Component dataclasses.

Each section maps directly to the spec (Part I §2.x).
All are mutable dataclasses — STMM is overwritten each cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from zados.memory.types import SpeakerID


# ---------------------------------------------------------------------------
# 2.1 Active Message Buffer
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single user/system message in the active buffer."""
    text:        str
    timestamp:   datetime
    speaker:     SpeakerID
    turn_index:  int


@dataclass
class ActiveMessageBuffer:
    """
    FIFO buffer: last 2 user messages + last 2 system responses.
    Eviction is handled by STMMStore.update_message().
    """
    messages: List[Message] = field(default_factory=list)

    def add(self, message: Message) -> None:
        speaker_msgs = [m for m in self.messages if m.speaker == message.speaker]
        if len(speaker_msgs) >= 2:
            oldest = speaker_msgs[0]
            self.messages.remove(oldest)
        self.messages.append(message)

    def get_by_speaker(self, speaker: SpeakerID) -> List[Message]:
        return [m for m in self.messages if m.speaker == speaker]

    def latest_user(self) -> Optional[Message]:
        user = self.get_by_speaker(SpeakerID.USER)
        return user[-1] if user else None

    def latest_system(self) -> Optional[Message]:
        sys = self.get_by_speaker(SpeakerID.SYSTEM)
        return sys[-1] if sys else None


# ---------------------------------------------------------------------------
# 2.2 Fractal Decomposition Engine Results
# ---------------------------------------------------------------------------

@dataclass
class FractalDecompositionResults:
    """Output of pipeline step (a) — semantic decomposition of current input."""
    tokenized_input:     List[str]              = field(default_factory=list)
    semantic_expansion:  Dict[str, List[str]]   = field(default_factory=dict)
    pattern_matches:     List[str]              = field(default_factory=list)
    fractal_depth:       int                    = 0
    raw_input:           str                    = ""


# ---------------------------------------------------------------------------
# 2.3 Intention Analysis Results
# ---------------------------------------------------------------------------

@dataclass
class IntentionAnalysisResults:
    """Output of pipeline step (c) — mapped intention of current input."""
    primary_intention:   str            = ""
    confidence:          float          = 0.0
    sub_intentions:      List[str]      = field(default_factory=list)
    pressure_type:       str            = "stable"   # stable | escalating | de-escalating
    engagement_expectation: str         = ""
    stability_passed:    bool           = True
    primary_archetype:   str            = ""         # E23 archetype label (e.g. "Guide", "Architect")


# ---------------------------------------------------------------------------
# 2.4 Emotion Detection & Structural Emotion State
# ---------------------------------------------------------------------------

@dataclass
class EmotionDetectionResults:
    """Current emotional state: detected user emotions + system structural state."""
    user_emotion_signals: Dict[str, float]  = field(default_factory=dict)
    system_emotion_state: Dict[str, float]  = field(default_factory=dict)
    emotional_delta:      Dict[str, float]  = field(default_factory=dict)
    saturation_levels:    Dict[str, float]  = field(default_factory=dict)
    # ToneVector fields — written by E28; consumed by LLM layer RG conditioning
    tone_valence:  float = 0.0   # [-1, 1]  overall emotional valence
    tone_coherence: float = 0.5  # [0, 1]   emotional coherence / consistency
    tone_warmth:   float = 0.0   # [-1, 1]  warmth toward user
    tone_discord:  float = 0.0   # [0, 1]   internal emotional conflict


# ---------------------------------------------------------------------------
# 2.5 Memory Contrast Results
# ---------------------------------------------------------------------------

@dataclass
class MemoryMatch:
    """A single entry returned by the memory contrast search."""
    entry_id:        str
    source_tier:     str        # "MTMM" | "LTMM" | "specialized"
    similarity:      float
    content_summary: str
    metadata:        Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContrastResults:
    """Output of pipeline step (d) — MTMM/LTMM retrieval results."""
    matched_entries:           List[MemoryMatch]   = field(default_factory=list)
    detected_echoes:           List[str]           = field(default_factory=list)
    potential_contradictions:  List[str]           = field(default_factory=list)
    unresolved_query_matches:  List[str]           = field(default_factory=list)
    delta_align:               float               = 0.0
    delta_R:                   float               = 0.0
    delta_C:                   float               = 0.0
    delta_B:                   float               = 0.0
    delta_E:                   float               = 0.0


# ---------------------------------------------------------------------------
# 2.6 Cortical Reflection Logger
# ---------------------------------------------------------------------------

@dataclass
class CorticalReflectionLog:
    """Metacognitive observations about the current processing cycle."""
    active_mode:              str            = "Normal"
    processing_anomalies:     List[str]      = field(default_factory=list)
    confidence_calibration:   str            = "calibrated"
    heuristic_bias_alerts:    List[str]      = field(default_factory=list)
    identity_coherence_status: str           = "coherent"
    notes:                    List[str]      = field(default_factory=list)
    # LLM layer outputs — written after Verbalized Thinking call
    verbal_reflection:        str            = ""        # full VT monologue text
    verbal_emotion_labels:    List[str]      = field(default_factory=list)  # top-5 active emotions


# ---------------------------------------------------------------------------
# 2.7 Brain Process Tracker
# ---------------------------------------------------------------------------

@dataclass
class EngineExecution:
    """Record of a single engine run within the current cycle."""
    engine_id:       str
    timing_ms:       float
    output_summary:  str
    skipped:         bool           = False
    skip_reason:     str            = ""


@dataclass
class BrainProcessTracker:
    """Execution log for all engines in the current cycle."""
    executions:            List[EngineExecution] = field(default_factory=list)
    pipeline_stage_flags:  Dict[str, bool]       = field(default_factory=dict)

    def record(self, execution: EngineExecution) -> None:
        self.executions.append(execution)

    def mark_stage(self, stage: str, completed: bool = True) -> None:
        self.pipeline_stage_flags[stage] = completed

    def engine_ids_run(self) -> List[str]:
        return [e.engine_id for e in self.executions if not e.skipped]


# ---------------------------------------------------------------------------
# 2.8 Reward Evaluation Results
# ---------------------------------------------------------------------------

@dataclass
class RewardEvaluationResults:
    """Output of pipeline step (f) — full reward system evaluation."""
    per_domain_results:   Dict[str, Any]   = field(default_factory=dict)
    per_domain_subscores: Dict[str, Any]   = field(default_factory=dict)
    meta_directive:       Dict[str, Any]   = field(default_factory=dict)
    flags:                List[str]        = field(default_factory=list)
    composite_score:      float            = 0.0


# ---------------------------------------------------------------------------
# 2.9 Cephalic Liquid Logger (Neurochemical State Snapshot)
# ---------------------------------------------------------------------------

@dataclass
class CephalicLiquidLogger:
    """Snapshot of the neurochemical layer at current cycle."""
    nt_concentrations:     Dict[str, float]  = field(default_factory=dict)
    receptor_occupancy:    Dict[str, float]  = field(default_factory=dict)
    oscillatory_bands:     Dict[str, float]  = field(default_factory=dict)
    neurosymbolic_metrics: Dict[str, float]  = field(default_factory=dict)
    modulation_signals:    Dict[str, Any]    = field(default_factory=dict)
    extractor_state:       Dict[str, Any]    = field(default_factory=dict)
