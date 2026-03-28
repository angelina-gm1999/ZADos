"""
Part V — Specialized Log Subsystems.

Eight typed logs that maintain domain-specific records cross-cutting the
temporal tiers.  All are in-memory (backed by the same parent tier store
at the spec level; here they are standalone containers that the Memory
Implementation Manager reads/writes).

Logs:
  1. LearningSystemLog
  2. SandboxLog
  3. ParadoxLog
  4. ContradictionLog
  5. UnsolvedConceptsBuffer  (Unanswered Question Log — highest priority)
  6. SelfReflectionLog
  7. IdentityMemoryLog
  8. DreamLog
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ===========================================================================
# 1. Learning System Log
# ===========================================================================

@dataclass
class LearningEntry:
    entry_id:         str   = field(default_factory=lambda: str(uuid.uuid4()))
    learning_type:    str   = ""    # contextual | reward | reflective | recursive
    source_ref:       str   = ""    # interaction reference (packet_id or turn)
    learned_content:  str   = ""    # the pattern / rule / adjustment
    confidence:       float = 0.5
    validation_status: str  = "pending"   # validated | pending | invalidated
    invalidation_note: str  = ""
    timestamp:        datetime = field(default_factory=datetime.utcnow)


class LearningSystemLog:
    def __init__(self) -> None:
        self._entries: List[LearningEntry] = []

    def record(self, entry: LearningEntry) -> str:
        self._entries.append(entry)
        return entry.entry_id

    def validate(self, entry_id: str) -> None:
        for e in self._entries:
            if e.entry_id == entry_id:
                e.validation_status = "validated"

    def invalidate(self, entry_id: str, note: str = "") -> None:
        for e in self._entries:
            if e.entry_id == entry_id:
                e.validation_status = "invalidated"
                e.invalidation_note = note

    def get_pending(self) -> List[LearningEntry]:
        return [e for e in self._entries if e.validation_status == "pending"]

    def get_validated(self) -> List[LearningEntry]:
        return [e for e in self._entries if e.validation_status == "validated"]

    def __len__(self) -> int:
        return len(self._entries)


# ===========================================================================
# 2. Sandbox Log
# ===========================================================================

@dataclass
class SandboxEntry:
    entry_id:             str   = field(default_factory=lambda: str(uuid.uuid4()))
    trigger:              str   = ""    # what initiated experimental processing
    experimental_params:  Dict[str, Any] = field(default_factory=dict)
    results:              str   = ""
    outcome_assessment:   str   = ""
    promoted:             bool  = False  # promoted to standard operation?
    timestamp:            datetime = field(default_factory=datetime.utcnow)


class SandboxLog:
    def __init__(self) -> None:
        self._entries: List[SandboxEntry] = []

    def record(self, entry: SandboxEntry) -> str:
        self._entries.append(entry)
        return entry.entry_id

    def promote(self, entry_id: str) -> None:
        for e in self._entries:
            if e.entry_id == entry_id:
                e.promoted = True

    def get_all(self) -> List[SandboxEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


# ===========================================================================
# 3. Paradox Log
# ===========================================================================

@dataclass
class ParadoxEntry:
    entry_id:          str   = field(default_factory=lambda: str(uuid.uuid4()))
    formulation:       str   = ""    # the paradoxical statements
    classification:    str   = "unproductive"   # productive | unproductive
    resolution_status: str   = "unresolved"     # resolved | unresolved | in-progress
    resolution_attempts: List[str] = field(default_factory=list)
    unsolved_buffer_id:  Optional[str] = None
    source_ref:        str   = ""
    timestamp:         datetime = field(default_factory=datetime.utcnow)


class ParadoxLog:
    def __init__(self) -> None:
        self._entries: List[ParadoxEntry] = []

    def record(self, entry: ParadoxEntry) -> str:
        self._entries.append(entry)
        return entry.entry_id

    def resolve(self, entry_id: str) -> None:
        for e in self._entries:
            if e.entry_id == entry_id:
                e.resolution_status = "resolved"

    def get_unresolved(self) -> List[ParadoxEntry]:
        return [e for e in self._entries if e.resolution_status != "resolved"]

    def __len__(self) -> int:
        return len(self._entries)


# ===========================================================================
# 4. Contradiction Log
# ===========================================================================

@dataclass
class ContradictionEntry:
    entry_id:          str   = field(default_factory=lambda: str(uuid.uuid4()))
    statement_a:       str   = ""    # first contradicting statement
    statement_b:       str   = ""    # second contradicting statement
    source_a_ref:      str   = ""
    source_b_ref:      str   = ""
    severity:          str   = "low"    # low | medium | high | critical
    contradiction_type: str  = "direct"  # direct | pragmatic | presuppositional
    resolution_status: str   = "unresolved"
    resolution_method: str   = ""
    associated_memory_ids: List[str] = field(default_factory=list)
    timestamp:         datetime = field(default_factory=datetime.utcnow)


class ContradictionLog:
    def __init__(self) -> None:
        self._entries: List[ContradictionEntry] = []

    def record(self, entry: ContradictionEntry) -> str:
        self._entries.append(entry)
        return entry.entry_id

    def resolve(self, entry_id: str, method: str = "") -> None:
        for e in self._entries:
            if e.entry_id == entry_id:
                e.resolution_status = "resolved"
                e.resolution_method = method

    def get_unresolved(self) -> List[ContradictionEntry]:
        return [e for e in self._entries if e.resolution_status != "resolved"]

    def get_by_severity(self, severity: str) -> List[ContradictionEntry]:
        return [e for e in self._entries if e.severity == severity]

    def __len__(self) -> int:
        return len(self._entries)


# ===========================================================================
# 5. Unsolved Concepts Buffer (Unanswered Question Log)
# ===========================================================================

@dataclass
class UnsolvedConceptEntry:
    """
    The system's active learning frontier.  Persists until resolved.
    Checked during every Memory Contrast pass.
    Stagnated entries become REM Dream candidates.
    """
    entry_id:          str   = field(default_factory=lambda: str(uuid.uuid4()))
    concept_formulation: str = ""
    source_engine:     str   = ""
    blocking_reason:   str   = ""
    evidence_accumulated: List[str] = field(default_factory=list)
    resolution_attempts: List[str] = field(default_factory=list)
    stagnation_cycles: int   = 0
    resolved:          bool  = False
    resolution_note:   str   = ""
    associated_memory_ids: List[str] = field(default_factory=list)
    timestamp:         datetime = field(default_factory=datetime.utcnow)
    last_checked:      datetime = field(default_factory=datetime.utcnow)

    def tick_stagnation(self) -> None:
        self.stagnation_cycles += 1
        self.last_checked = datetime.utcnow()

    def is_dream_candidate(self, threshold: int = 5) -> bool:
        return not self.resolved and self.stagnation_cycles >= threshold


class UnsolvedConceptsBuffer:
    """
    Highest-priority specialized log.
    All unresolvable issues end up here and are actively monitored.
    """
    def __init__(self) -> None:
        self._entries: Dict[str, UnsolvedConceptEntry] = {}

    def add(self, entry: UnsolvedConceptEntry) -> str:
        self._entries[entry.entry_id] = entry
        return entry.entry_id

    def resolve(self, entry_id: str, note: str = "") -> None:
        if entry_id in self._entries:
            self._entries[entry_id].resolved      = True
            self._entries[entry_id].resolution_note = note

    def get_all_active(self) -> List[UnsolvedConceptEntry]:
        return [e for e in self._entries.values() if not e.resolved]

    def get_dream_candidates(self, threshold: int = 5) -> List[UnsolvedConceptEntry]:
        return [e for e in self._entries.values() if e.is_dream_candidate(threshold)]

    def tick_all(self) -> None:
        """Called each cycle to increment stagnation counters."""
        for e in self._entries.values():
            if not e.resolved:
                e.tick_stagnation()

    def get_by_id(self, entry_id: str) -> Optional[UnsolvedConceptEntry]:
        return self._entries.get(entry_id)

    def __len__(self) -> int:
        return len(self._entries)


# ===========================================================================
# 6. Self-Reflection Log
# ===========================================================================

@dataclass
class SelfReflectionEntry:
    entry_id:          str   = field(default_factory=lambda: str(uuid.uuid4()))
    observation_type:  str   = ""   # bias_detected | identity_drift | confidence_miscalibration | alignment_error
    severity:          str   = "low"
    description:       str   = ""
    corrective_action: str   = ""
    correction_outcome: str  = ""
    timestamp:         datetime = field(default_factory=datetime.utcnow)


class SelfReflectionLog:
    def __init__(self) -> None:
        self._entries: List[SelfReflectionEntry] = []

    def record(self, entry: SelfReflectionEntry) -> str:
        self._entries.append(entry)
        return entry.entry_id

    def get_by_type(self, observation_type: str) -> List[SelfReflectionEntry]:
        return [e for e in self._entries if e.observation_type == observation_type]

    def get_all(self) -> List[SelfReflectionEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


# ===========================================================================
# 7. Identity Memory Log
# ===========================================================================

@dataclass
class IdentityMemoryEntry:
    """
    Permanent LTMM.  Never demoted to cold storage.
    Records identity-relevant events.
    """
    entry_id:          str   = field(default_factory=lambda: str(uuid.uuid4()))
    identity_aspect:   str   = ""   # values | capabilities | limitations | personality
    trigger_event:     str   = ""
    pre_state:         Dict[str, Any] = field(default_factory=dict)
    post_state:        Dict[str, Any] = field(default_factory=dict)
    stability_assessment: str = "stable"   # stable | challenged | updated
    timestamp:         datetime = field(default_factory=datetime.utcnow)


class IdentityMemoryLog:
    def __init__(self) -> None:
        self._entries: List[IdentityMemoryEntry] = []

    def record(self, entry: IdentityMemoryEntry) -> str:
        self._entries.append(entry)
        return entry.entry_id

    def get_by_aspect(self, aspect: str) -> List[IdentityMemoryEntry]:
        return [e for e in self._entries if e.identity_aspect == aspect]

    def get_all(self) -> List[IdentityMemoryEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


# ===========================================================================
# 8. Dream Log
# ===========================================================================

@dataclass
class DreamEntry:
    """
    Records REM Dream processing episodes.
    All episodes persist in LTMM regardless of outcome.
    """
    entry_id:          str   = field(default_factory=lambda: str(uuid.uuid4()))
    dream_trigger_id:  str   = ""    # UnsolvedConceptEntry.entry_id
    dream_content:     str   = ""    # the creative exploration
    resolution_status: str   = "no_candidate"  # no_candidate | candidate_generated | validated
    resolution_candidate: str = ""
    validated:         bool  = False
    cross_link_id:     Optional[str] = None    # resolved UnsolvedConceptEntry ID
    timestamp:         datetime = field(default_factory=datetime.utcnow)


class DreamLog:
    def __init__(self) -> None:
        self._entries: List[DreamEntry] = []

    def record(self, entry: DreamEntry) -> str:
        self._entries.append(entry)
        return entry.entry_id

    def validate_candidate(self, entry_id: str, resolved_concept_id: str) -> None:
        for e in self._entries:
            if e.entry_id == entry_id:
                e.validated      = True
                e.resolution_status = "validated"
                e.cross_link_id  = resolved_concept_id

    def get_all(self) -> List[DreamEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


# ===========================================================================
# Convenience bundle
# ===========================================================================

@dataclass
class SpecializedLogs:
    """Single container holding all 8 specialized logs."""
    learning:      LearningSystemLog     = field(default_factory=LearningSystemLog)
    sandbox:       SandboxLog            = field(default_factory=SandboxLog)
    paradox:       ParadoxLog            = field(default_factory=ParadoxLog)
    contradiction: ContradictionLog      = field(default_factory=ContradictionLog)
    unsolved:      UnsolvedConceptsBuffer = field(default_factory=UnsolvedConceptsBuffer)
    self_reflect:  SelfReflectionLog     = field(default_factory=SelfReflectionLog)
    identity:      IdentityMemoryLog     = field(default_factory=IdentityMemoryLog)
    dream:         DreamLog              = field(default_factory=DreamLog)
