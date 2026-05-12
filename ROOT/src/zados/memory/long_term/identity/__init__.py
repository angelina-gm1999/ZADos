"""
Identity namespace — stores for who ZA-DOS is.

Sub-stores
----------
hardcoded        Immutable baseline identity (system prompt fragments, axioms,
                 core values, core identity).
core_memories    Peer-review-gated core identity beliefs.
development      Conclusions and identity journal from self-reflection.
correlation      Maps relations between fixed (hardcoded) and developmental identity.
"""
from zados.memory.long_term.identity.hardcoded.store import HardcodedStore
from zados.memory.long_term.identity.core_memories.store import CoreMemoryStore
from zados.memory.long_term.identity.core_memories.pending.queue import PendingUpdateQueue
from zados.memory.long_term.identity.development.conclusions import IdentityConclusionStore
from zados.memory.long_term.identity.development.identity_journal.store import IdentityJournalStore
from zados.memory.long_term.identity.correlation.store import IdentityCorrelationStore

__all__ = [
    "HardcodedStore",
    "CoreMemoryStore",
    "PendingUpdateQueue",
    "IdentityConclusionStore",
    "IdentityJournalStore",
    "IdentityCorrelationStore",
]
