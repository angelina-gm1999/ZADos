"""
Shared types for the ZADOS Memory Layer.

MemoryPacket is the canonical transfer format between tiers.
All tier modules import from here — no circular deps.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MemoryTier(str, Enum):
    STMM = "STMM"
    MTMM = "MTMM"
    LTMM = "LTMM"


class CompressionLevel(str, Enum):
    FULL     = "full"
    SEMANTIC = "semantic"
    SYMBOLIC = "symbolic"


class SpeakerID(str, Enum):
    USER   = "user"
    SYSTEM = "system"


# ---------------------------------------------------------------------------
# Core transfer object
# ---------------------------------------------------------------------------

@dataclass
class MemoryPacket:
    """
    Canonical transfer format between memory tiers.
    Produced by Memory Exit Compressor; consumed by MTMM / LTMM.
    """
    # Identity
    packet_id:         str            = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:         datetime       = field(default_factory=datetime.utcnow)
    source_tier:       MemoryTier     = MemoryTier.STMM
    destination_tier:  MemoryTier     = MemoryTier.MTMM

    # Core content (lossless)
    user_message:      str            = ""
    system_response:   str            = ""
    turn_index:        int            = 0

    # Compressed analysis (lossy)
    intention:         str            = ""          # primary intention label
    emotion_vector:    Dict[str, float] = field(default_factory=dict)
    neurochemical_snapshot: Dict[str, float] = field(default_factory=dict)
    reward_scores:     Dict[str, float] = field(default_factory=dict)

    # Flags and signals
    flags:                  List[str]  = field(default_factory=list)
    contradictions_detected: int       = 0
    paradoxes_detected:      int       = 0
    unsolved_items_matched:  List[str] = field(default_factory=list)

    # Metadata
    compression_level:    CompressionLevel = CompressionLevel.SEMANTIC
    trust_weight:         float            = 1.0   # [0, 1]
    emotional_significance: float          = 0.0   # [0, 1]

    # Semantic embedding (set at MTMM write time)
    embedding: Optional[List[float]] = None

    # LLM layer outputs — written by Memory Exit Compressor after VT call
    verbal_summary:        str       = ""        # ~100-word extractive compression of VT monologue
    verbal_emotion_labels: List[str] = field(default_factory=list)  # top-5 active emotion names

    # Temporal context — stamped at turn entry (TimeContextSnapshot.to_dict())
    time_context:  Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id":               self.packet_id,
            "timestamp":               self.timestamp.isoformat(),
            "source_tier":             self.source_tier.value,
            "destination_tier":        self.destination_tier.value,
            "user_message":            self.user_message,
            "system_response":         self.system_response,
            "turn_index":              self.turn_index,
            "intention":               self.intention,
            "emotion_vector":          self.emotion_vector,
            "neurochemical_snapshot":  self.neurochemical_snapshot,
            "reward_scores":           self.reward_scores,
            "flags":                   self.flags,
            "contradictions_detected": self.contradictions_detected,
            "paradoxes_detected":      self.paradoxes_detected,
            "unsolved_items_matched":  self.unsolved_items_matched,
            "compression_level":       self.compression_level.value,
            "trust_weight":            self.trust_weight,
            "emotional_significance":  self.emotional_significance,
            "time_context":            dict(self.time_context),
        }
