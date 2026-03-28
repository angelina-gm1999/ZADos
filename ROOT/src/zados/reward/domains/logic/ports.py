from __future__ import annotations


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol




@dataclass(frozen=True)
class ContrastResult:
    """
    Output of a memory contrast query: similarity/divergence metrics + optional refs.
    """
    similarity: float  # 0..1 (higher = more similar)
    divergence: float  # 0..1 (higher = more drift/contradiction)
    meta: Dict[str, Any] = field(default_factory=dict)
    references: List[Dict[str, Any]] = field(default_factory=list)




class MemoryContrastPort(Protocol):
    """
    Placeholder contract for anything that can compare current representations
    with memory references (context/concept/semantic continuity).
    """


    def contrast(
        self,
        *,
        current: Dict[str, Any],
        query_type: str,
        ctx_id: Optional[str] = None,
        limit: int = 5,
        meta: Optional[Dict[str, Any]] = None,
    ) -> ContrastResult:
        """
        query_type examples: 'context', 'concept', 'semantic', 'internal', 'external'
        """
        ...




@dataclass(frozen=True)
class TraceResult:
    """
    Output of a cognitive trace request (parse trees, inference steps, tool usage).
    """
    trace: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)




class CognitiveTracePort(Protocol):
    """
    Placeholder contract for extracting reasoning traces from downstream engines.
    """


    def get_trace(
        self,
        *,
        request: Dict[str, Any],
        trace_type: str,
        ctx_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> TraceResult:
        """
        trace_type examples: 'parse', 'analysis', 'scientific_rigor', 'socratic'
        """
        ...
