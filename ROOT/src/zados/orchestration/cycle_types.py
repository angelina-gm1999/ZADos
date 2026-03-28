"""
Shared types for the orchestration layer.

CycleContext    — immutable context passed to every engine adapter during a cycle.
EngineSlot      — registry entry for one engine in the dispatch table.
CycleResult     — output of one full processing cycle.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class CycleContext:
    """Immutable context passed to every engine adapter during a cycle."""
    cycle_id: int
    user_message: str
    system_response: str
    nt_state: Dict[str, float]
    oscillatory_state: Dict[str, float]
    active_mode: str                            # "normal", "dev", "learning", etc.
    memory_contrast: Optional[Any] = None       # MemoryContrast instance
    timestamp: float = field(default_factory=time.time)


@dataclass
class EngineSlot:
    """Registry entry for one engine in the dispatch table.

    Attributes:
        engine_id:        Canonical engine identifier (e.g. "emotional_detection_engine").
        engine_number:    Engine number (1..30).
        instance:         The engine object (has update_neurochem_state / process / get_status).
        adapter:          Function: (engine, CycleContext, STMMStore) -> str summary.
        cluster:          Engine cluster name (e.g. "detection", "evaluation").
        priority:         Dispatch order — lower numbers run first.
        requires_memory:  If True, engine needs memory_contrast on CycleContext.
        depends_on:       Engine IDs that must run before this one.
    """
    engine_id: str
    engine_number: int
    instance: Any
    adapter: Callable[..., str]
    cluster: str
    priority: int = 50
    requires_memory: bool = False
    depends_on: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CycleResult:
    """Output of one full processing cycle."""
    cycle_id: int
    packet: Any                     # MemoryPacket
    engines_run: Tuple[str, ...]
    engines_skipped: Tuple[str, ...]
    timing_ms: float
    stmm_snapshot: Dict[str, Any] = field(default_factory=dict)
