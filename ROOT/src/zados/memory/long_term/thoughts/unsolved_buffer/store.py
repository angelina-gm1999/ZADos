"""
UnsolvedBufferStore — thin wrapper re-exporting UnsolvedConceptsBuffer.

The original UnsolvedConceptsBuffer lives in specialized_logs.py and is
used directly by existing code.  This wrapper exposes it under the
Thoughts namespace without moving or duplicating the implementation.
"""
from __future__ import annotations

from zados.memory.long_term.specialized_logs import (
    UnsolvedConceptsBuffer,
    UnsolvedConceptEntry,
)

# Re-export so that thoughts.unsolved_buffer.store.UnsolvedBufferStore
# is importable, but it IS the same class.
UnsolvedBufferStore = UnsolvedConceptsBuffer

__all__ = ["UnsolvedBufferStore", "UnsolvedConceptEntry"]
