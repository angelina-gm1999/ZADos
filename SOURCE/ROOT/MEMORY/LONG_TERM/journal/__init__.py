"""
ZA-DOS Journal Module — identity/journal/ tier of LTMM.

Public API
----------
JournalWriter   — the plugin. Instantiate once, call .write(JournalContext) from any pipeline.
JournalStore    — persistent store for JournalEntry objects.
JournalEntry    — the full journal artifact.
JournalContext  — input payload for JournalWriter.write().
JournalTrigger  — five trigger conditions (PERIODIC, LTMM_THRESHOLD, REM_COMPLETE,
                  INNOVATION_FLAG, DEV).
ReviewStatus    — lifecycle of reflection prompts (UNREVIEWED, IN_REVIEW, RESOLVED).
"""
from zados.memory.long_term.journal.entry import (
    EngineAnnotations,
    JournalContext,
    JournalEntry,
    JournalTrigger,
    ReviewStatus,
)
from zados.memory.long_term.journal.store import JournalStore
from zados.memory.long_term.journal.writer import JournalWriter

__all__ = [
    "JournalWriter",
    "JournalStore",
    "JournalEntry",
    "JournalContext",
    "JournalTrigger",
    "ReviewStatus",
    "EngineAnnotations",
]
