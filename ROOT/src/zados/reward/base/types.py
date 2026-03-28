from __future__ import annotations


from dataclasses import dataclass, field
from typing import Dict, Any, Optional




@dataclass(frozen=True)
class RewardContext:
    """
    Minimal, model-agnostic context container passed to reward evaluators.


    Keep this deliberately generic in Phase 0.
    You can add structured fields later without breaking the interface.
    """
    reward_profile: str = "default"
    timestamp: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)




@dataclass(frozen=True)
class RewardSubscore:
    """
    Output of a single reward submodule evaluation.
    """
    name: str
    score: float  # recommended normalized range: 0.0 to 1.0
    flags: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)




@dataclass(frozen=True)
class RewardDomainResult:
    """
    Aggregated output of a reward domain.
    """
    domain: str
    general_score: float  # recommended normalized range: 0.0 to 1.0
    subscores: Dict[str, RewardSubscore] = field(default_factory=dict)
    flags: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)




@dataclass(frozen=True)
class RewardWeights:
    """
    Domain weights (static in v1).
    """
    weights: Dict[str, float] = field(default_factory=dict)


    def get(self, domain: str, default: float = 0.0) -> float:
        return float(self.weights.get(domain, default))




@dataclass(frozen=True)
class RewardMetaDirective:
    """
    Output of synthesis/arbitration layers.
    Phase 0: just define the container.
    """
    allow_output: bool = True
    abstain: bool = False
    suppress: bool = False


    # Response shaping knobs (kept generic and model-agnostic)
    directives: Dict[str, Any] = field(default_factory=dict)


    # Routing/selection hints for downstream components
    routing: Dict[str, Any] = field(default_factory=dict)


    # Risk and audit flags
    flags: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for downstream consumers."""
        return {
            "allow_output": self.allow_output,
            "abstain": self.abstain,
            "suppress": self.suppress,
            "directives": dict(self.directives),
            "routing": dict(self.routing),
            "flags": dict(self.flags),
            "meta": dict(self.meta),
        }
