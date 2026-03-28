from __future__ import annotations


from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import time
import uuid




# ---------------------------------------------------------------------
# Threshold specifications
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdSpec:
    """
    Defines numeric thresholds and optional hysteresis for regime switching.
    """
    lower: float
    upper: float
    hysteresis: float = 0.0
    label: Optional[str] = None


    def in_range(self, value: float) -> bool:
        return self.lower <= value <= self.upper




# ---------------------------------------------------------------------
# Structured reward flags
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RewardFlag:
    """
    Represents a single reward-related flag.
    """
    name: str
    severity: str = "info"  # info | warning | risk | critical
    message: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)




@dataclass(frozen=True)
class RewardFlagSet:
    """
    Container for structured reward flags.
    """
    flags: Tuple[RewardFlag, ...] = ()


    def has_severity(self, severity: str) -> bool:
        return any(f.severity == severity for f in self.flags)


    def names(self) -> Tuple[str, ...]:
        return tuple(f.name for f in self.flags)




# ---------------------------------------------------------------------
# Provenance and audit records
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceRecord:
    """
    Minimal provenance record for auditability.
    """
    provenance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=lambda: time.time())
    source: Optional[str] = None
    notes: Dict[str, Any] = field(default_factory=dict)
