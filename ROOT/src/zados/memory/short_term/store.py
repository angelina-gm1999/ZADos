"""
STMMStore — active working memory for a single processing cycle.

Holds all 10 STMM components. Overwritten at the start of each new cycle.
Exposes a reset() that clears all analysis results while preserving the
message buffer (the buffer manages its own FIFO eviction).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from zados.memory.types import SpeakerID
from zados.memory.short_term.components import (
    ActiveMessageBuffer,
    BrainProcessTracker,
    CephalicLiquidLogger,
    CorticalReflectionLog,
    EmotionDetectionResults,
    FractalDecompositionResults,
    IntentionAnalysisResults,
    Message,
    MemoryContrastResults,
    RewardEvaluationResults,
)


@dataclass
class STMMStore:
    """
    The system's active working memory.

    All engines READ from and WRITE to this object during a processing cycle.
    At cycle end, MemoryExitCompressor converts this into a MemoryPacket for MTMM.
    """
    # 2.1
    active_message_buffer:      ActiveMessageBuffer       = field(default_factory=ActiveMessageBuffer)
    # 2.2
    fractal_decomposition:      FractalDecompositionResults = field(default_factory=FractalDecompositionResults)
    # 2.3
    intention_analysis:         IntentionAnalysisResults   = field(default_factory=IntentionAnalysisResults)
    # 2.4
    emotion_detection:          EmotionDetectionResults    = field(default_factory=EmotionDetectionResults)
    # 2.5
    memory_contrast:            MemoryContrastResults      = field(default_factory=MemoryContrastResults)
    # 2.6
    cortical_reflection:        CorticalReflectionLog      = field(default_factory=CorticalReflectionLog)
    # 2.7
    brain_process_tracker:      BrainProcessTracker        = field(default_factory=BrainProcessTracker)
    # 2.8
    reward_evaluation:          RewardEvaluationResults    = field(default_factory=RewardEvaluationResults)
    # 2.9
    cephalic_liquid_logger:     CephalicLiquidLogger       = field(default_factory=CephalicLiquidLogger)

    # Internal cycle counter
    _turn_index: int = field(default=0, repr=False)

    # -----------------------------------------------------------------------
    # Message ingestion (FIFO, maintained across cycles)
    # -----------------------------------------------------------------------

    def add_user_message(self, text: str) -> None:
        self._turn_index += 1
        msg = Message(
            text=text,
            timestamp=datetime.utcnow(),
            speaker=SpeakerID.USER,
            turn_index=self._turn_index,
        )
        self.active_message_buffer.add(msg)

    def add_system_response(self, text: str) -> None:
        msg = Message(
            text=text,
            timestamp=datetime.utcnow(),
            speaker=SpeakerID.SYSTEM,
            turn_index=self._turn_index,
        )
        self.active_message_buffer.add(msg)

    # -----------------------------------------------------------------------
    # Cycle management — reset analysis fields, keep message buffer + counter
    # -----------------------------------------------------------------------

    def begin_cycle(self) -> None:
        """
        Called at the start of a new processing cycle.
        Clears all analysis results; message buffer is NOT cleared here
        (it accumulates across cycles with FIFO eviction in add_user_message).
        """
        self.fractal_decomposition  = FractalDecompositionResults()
        self.intention_analysis     = IntentionAnalysisResults()
        self.emotion_detection      = EmotionDetectionResults()
        self.memory_contrast        = MemoryContrastResults()
        self.cortical_reflection    = CorticalReflectionLog()
        self.brain_process_tracker  = BrainProcessTracker()
        self.reward_evaluation      = RewardEvaluationResults()
        self.cephalic_liquid_logger = CephalicLiquidLogger()

    @property
    def turn_index(self) -> int:
        return self._turn_index

    def snapshot(self) -> dict:
        """Return a lightweight dict summary (for logging / inspection)."""
        return {
            "turn_index":               self._turn_index,
            "user_messages":            len(self.active_message_buffer.get_by_speaker(SpeakerID.USER)),
            "system_messages":          len(self.active_message_buffer.get_by_speaker(SpeakerID.SYSTEM)),
            "intention":                self.intention_analysis.primary_intention,
            "stability_passed":         self.intention_analysis.stability_passed,
            "composite_reward":         self.reward_evaluation.composite_score,
            "engines_run":              self.brain_process_tracker.engine_ids_run(),
            "active_mode":              self.cortical_reflection.active_mode,
            "contradictions":           len(self.memory_contrast.potential_contradictions),
        }
