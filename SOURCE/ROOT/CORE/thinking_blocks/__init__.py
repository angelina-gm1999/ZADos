"""
core/thinking_blocks — Compressed context block for LLM thinking pass.

Sits between Phase 3 (engine dispatch) and Phase 4 (VT / LLM pass 1).
Aggregates engine flags, memory contrast cross-notes, last MTMM turns,
held thinking blocks, and mission briefing into a single ThinkingContext
object consumed by VTPromptBuilder.
"""
from zados.core.thinking_blocks.types import ThinkingContext
from zados.core.thinking_blocks.builder import ThinkingBlockBuilder

__all__ = ["ThinkingContext", "ThinkingBlockBuilder"]
