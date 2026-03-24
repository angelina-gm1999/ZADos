"""
ZADOS Orchestration Layer — connects Memory, Cognitive Engines, and Neurochem.

The CycleManager sequences one processing cycle:
  1. STMM begin (reset analysis slots, keep message buffer)
  2. Memory contrast query (populate STMM.memory_contrast)
  3. Engine dispatch (adapters marshal input, call engine.process, write STMM)
  4. STMM compress → MemoryPacket → MTMM write

Usage:
    from zados.orchestration import CycleManager
    from zados.memory import MemoryLayer

    ml = MemoryLayer()
    cm = CycleManager(memory=ml)
    cm.register(slot)
    result = cm.run_cycle("Hello", "Hi there!")
"""
from zados.orchestration.cycle_types import (
    CycleContext,
    CycleResult,
    EngineSlot,
)
from zados.orchestration.cycle_manager import CycleManager

__all__ = [
    "CycleContext",
    "CycleResult",
    "CycleManager",
    "EngineSlot",
]
