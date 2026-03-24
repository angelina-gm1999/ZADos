"""
Simulated Opposition Engine (Detection Cluster — Engine 7).

The system's internal adversary. Takes the system's own reasoning output —
proposed responses, conclusions, analyses, plans — and actively tries to BREAK
them before they reach the user.

Every other detection engine asks "is something wrong with this INPUT?"
This engine asks "what could go wrong with our OUTPUT?"

Five Opposition Modes
----------------------
Mode 1  COUNTERARGUMENT      — Generate the strongest argument against the conclusion.
Mode 2  COUNTEREXAMPLE       — Find specific cases where the claim doesn't hold.
Mode 3  ALTERNATIVE_EXPLANATION — Generate plausible alternative explanations.
Mode 4  ASSUMPTION_EXCAVATION — Identify and challenge unstated assumptions.
Mode 5  STRUCTURAL_RESISTANCE — Apply systematic resistance to test under pressure.

Three Depth Levels
-------------------
QUICK    — Mode 1 only, 1-2 passes, max 1 output
STANDARD — Modes 1-3, 3-5 passes, max 3 outputs
DEEP     — All 5 modes, 5-10 passes, max 5 outputs

Gate Outcomes
--------------
PASS            gate_score < 0.25  — output survived, deliver as-is
PASS_WITH_CAVEAT 0.25-0.50         — survived with qualifications
REVISE          0.50-0.75          — meaningful weakness found, revise
BLOCK           > 0.75             — output completely undermined

Quality Formula
----------------
Q = w_q1 × validity + w_q2 × relevance + w_q3 × strength + w_q4 × novelty
Minimum emission threshold: θ_opposition_quality = 0.45

Activity Function
------------------
Ω(t) = depth_multiplier × Σ_o quality(o) × severity(o) × w_mode(o)

Neurochemical Coupling
-----------------------
NE  — Poisson burst proportional to Ω(t) (adversarial alertness)
DA  — Outcome-dependent: PASS→small, REVISE/BLOCK→larger, BLOCK→extra burst
ACh — Depth × active (sustained analytical effort)
GABA— Reuptake suppression during opposition (suppress self-defense bias)
COR — Mild eustress during Deep only (productive stress, not threat stress)
Beta— Enhancement × Ω(t) (focused challenge)
Gamma— Enhancement during Structural Resistance
Theta— Enhancement during search modes, suppression during Structural Resistance

Anti-Nihilism Safeguard
------------------------
block_rate = n_blocked / n_evaluated (rolling window 20)
If block_rate > θ_nihilism (0.30): raise quality threshold, flag for review.

Usage
-----
>>> from zados.cognitive_engines.py_engines.simulated_opposition_engine import (
...     SimulatedOppositionEngine, OppositionEngineConfig,
...     OppositionGateInput, OppositionRequest,
...     OppositionMode, OppositionDepth, OperationalMode,
... )
>>> engine = SimulatedOppositionEngine()
>>> result = engine.run_gate(OppositionGateInput(candidate_response="The answer is X."))
>>> result2 = engine.run_request(OppositionRequest(target="A causes B.", requested_modes=[OppositionMode.COUNTERARGUMENT]))
"""

from __future__ import annotations

import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from zados.cognitive_engines.constants import _clamp
from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    ContradictionFlag,
    OperationalMode,
    ProcessedStatement,
    SourceTag,
)
from zados.cognitive_engines.py_engines.fallacy_detection_engine import (
    FallacyFlag,
    IntentionVector,
    Proposition,
)
from zados.cognitive_engines.py_engines.logic_trap_detection_engine import (
    BiasFlag,
    TrapFlag,
)


# =====================================================================
# Enumerations
# =====================================================================

class OppositionMode(str, Enum):
    COUNTERARGUMENT       = "COUNTERARGUMENT"
    COUNTEREXAMPLE        = "COUNTEREXAMPLE"
    ALTERNATIVE_EXPLANATION = "ALTERNATIVE_EXPLANATION"
    ASSUMPTION_EXCAVATION = "ASSUMPTION_EXCAVATION"
    STRUCTURAL_RESISTANCE = "STRUCTURAL_RESISTANCE"


class OppositionDepth(str, Enum):
    QUICK    = "QUICK"
    STANDARD = "STANDARD"
    DEEP     = "DEEP"


class OppositionSeverity(str, Enum):
    MINOR       = "MINOR"        # Quality 0.45–0.55
    MODERATE    = "MODERATE"     # Quality 0.55–0.70
    SIGNIFICANT = "SIGNIFICANT"  # Quality 0.70–0.85
    CRITICAL    = "CRITICAL"     # Quality > 0.85


class GateOutcome(str, Enum):
    PASS             = "PASS"             # gate_score < 0.25
    PASS_WITH_CAVEAT = "PASS_WITH_CAVEAT" # 0.25 – 0.50
    REVISE           = "REVISE"           # 0.50 – 0.75
    BLOCK            = "BLOCK"            # > 0.75


class PositionStatus(str, Enum):
    SURVIVED    = "SURVIVED"     # Core claim intact
    IMPROVED    = "IMPROVED"     # Questioning revealed nuances
    COLLAPSED   = "COLLAPSED"    # Core claim couldn't survive
    TRANSFORMED = "TRANSFORMED"  # Claim was about something different


class TargetType(str, Enum):
    ARGUMENT           = "ARGUMENT"
    GENERALIZATION     = "GENERALIZATION"
    EXPLANATION        = "EXPLANATION"
    BELIEF             = "BELIEF"
    DECISION           = "DECISION"
    SCENARIO           = "SCENARIO"


# =====================================================================
# Configuration (frozen)
# =====================================================================

@dataclass(frozen=True)
class OppositionEngineConfig:
    """
    All tunable parameters for the Simulated Opposition Engine.

    Depth selection weights
    -----------------------
    w_d1  0.25  Topic complexity
    w_d2  0.20  (1 - confidence)
    w_d3  0.25  Stakes
    w_d4  0.15  Novelty
    w_d5  0.15  User challenge level

    Quality weights
    ---------------
    w_q1  0.30  Validity
    w_q2  0.25  Relevance
    w_q3  0.25  Strength
    w_q4  0.20  Novelty

    Depth pass/output limits
    -------------------------
    QUICK:    max_passes=2,  sufficient=1, max_output=1
    STANDARD: max_passes=5,  sufficient=2, max_output=3
    DEEP:     max_passes=10, sufficient=3, max_output=5

    Mode activity weights (w_mode)
    --------------------------------
    COUNTERARGUMENT       0.35
    COUNTEREXAMPLE        0.30
    ALTERNATIVE_EXPLANATION 0.25
    ASSUMPTION_EXCAVATION 0.30
    STRUCTURAL_RESISTANCE 0.40

    Depth multipliers
    -----------------
    QUICK 0.5, STANDARD 1.0, DEEP 1.5

    Neurochemical
    -------------
    β_opposition 0.12  NE activity coupling
    λ_ne         1.8   NE Poisson rate
    β_pass       0.08  DA pass reward
    β_find       0.12  DA find reward
    β_critical   0.15  DA critical block reward
    β_opp_ach    0.13  ACh depth coupling
    η_opposition 0.15  GABA reuptake suppression
    ρ_eustress   0.08  COR eustress (Deep only)
    ψ_β_opp      0.10  Beta enhancement
    ψ_γ_opp      0.08  Gamma enhancement (SR mode)
    ψ_θ_search   0.06  Theta search enhancement
    ψ_θ_resist   0.05  Theta SR suppression

    Thresholds
    ----------
    θ_opposition_quality  0.45  Minimum quality to emit
    θ_alt                 0.15  Min alternative plausibility
    θ_relevance           0.50  Min semantic similarity
    θ_nihilism            0.30  Max acceptable block rate
    """
    # Depth selection
    w_d1: float = 0.25
    w_d2: float = 0.20
    w_d3: float = 0.25
    w_d4: float = 0.15
    w_d5: float = 0.15

    # Quality weights
    w_q1: float = 0.30
    w_q2: float = 0.25
    w_q3: float = 0.25
    w_q4: float = 0.20

    # Mode activity weights
    w_mode_counterargument:        float = 0.35
    w_mode_counterexample:         float = 0.30
    w_mode_alternative_explanation: float = 0.25
    w_mode_assumption_excavation:  float = 0.30
    w_mode_structural_resistance:  float = 0.40

    # Depth multipliers
    depth_mult_quick:    float = 0.5
    depth_mult_standard: float = 1.0
    depth_mult_deep:     float = 1.5

    # Depth limits
    max_passes_quick:    int = 2
    max_passes_standard: int = 5
    max_passes_deep:     int = 10

    sufficient_quick:    int = 1
    sufficient_standard: int = 2
    sufficient_deep:     int = 3

    max_output_quick:    int = 1
    max_output_standard: int = 3
    max_output_deep:     int = 5

    # Structural resistance rounds
    sr_rounds_standard: int = 3
    sr_rounds_deep:     int = 5

    # Revision loop
    max_revision_loops: int = 2

    # Neurochemical
    beta_opposition: float = 0.12
    lambda_ne:       float = 1.8
    beta_pass:       float = 0.08
    beta_find:       float = 0.12
    beta_critical:   float = 0.15
    beta_opp_ach:    float = 0.13
    eta_opposition:  float = 0.15
    rho_eustress:    float = 0.08
    psi_beta_opp:    float = 0.10
    psi_gamma_opp:   float = 0.08
    psi_theta_search: float = 0.06
    psi_theta_resist: float = 0.05

    # Thresholds
    theta_opposition_quality: float = 0.45
    theta_alt:                float = 0.15
    theta_relevance:          float = 0.50
    theta_nihilism:           float = 0.30

    # Gate score thresholds
    gate_pass_max:   float = 0.25
    gate_caveat_max: float = 0.50
    gate_revise_max: float = 0.75
    # > gate_revise_max → BLOCK

    # Anti-nihilism rolling window
    nihilism_window: int = 20


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class ReasoningTrace:
    """
    How the system's response was constructed.

    Attributes
    ----------
    premises : list of str
        Supporting claims used to arrive at the response.
    inference_steps : list of str
        Step-by-step reasoning chain.
    conclusion : str
        The main conclusion or claim.
    confidence : float
        System's confidence in its own output [0, 1].
    caveats_already_acknowledged : list of str
        Limitations the system already mentioned.
    """
    premises:                    List[str] = field(default_factory=list)
    inference_steps:             List[str] = field(default_factory=list)
    conclusion:                  str       = ""
    confidence:                  float     = 0.75
    caveats_already_acknowledged: List[str] = field(default_factory=list)


@dataclass
class DialogueTurn:
    """One turn in the conversation history."""
    speaker:   str  = "user"   # "user" | "system"
    content:   str  = ""
    turn_index: int = 0


@dataclass(frozen=True)
class OppositionItem:
    """
    One generated opposition — the atomic output unit.

    Attributes
    ----------
    item_id : str
        UUID.
    mode : OppositionMode
    content : str
        The opposition itself (human-readable argument/example/alternative).
    quality_score : float
        Q = w_q1×validity + w_q2×relevance + w_q3×strength + w_q4×novelty ∈ [0,1].
    validity : float
    relevance : float
    strength : float
    novelty : float
    target_component : str
        Which part of the target this challenges (premise / inference / conclusion / assumption).
    severity : OppositionSeverity
    attack_type : str
        "premise_attack" | "inference_attack" | "conclusion_attack" | "boundary_case" |
        "assumption_necessary" | "alternative_causal" | "resistance_round_N"
    """
    item_id:          str
    mode:             OppositionMode
    content:          str
    quality_score:    float
    validity:         float
    relevance:        float
    strength:         float
    novelty:          float
    target_component: str
    severity:         OppositionSeverity
    attack_type:      str


@dataclass(frozen=True)
class ResistanceResult:
    """
    Output of Mode 5 Structural Resistance.

    Attributes
    ----------
    rounds_survived : int
    final_position_status : PositionStatus
    key_challenges : list of str
    position_delta : str
        How the position changed under pressure.
    """
    rounds_survived:       int
    final_position_status: PositionStatus
    key_challenges:        List[str]
    position_delta:        str


@dataclass(frozen=True)
class ParadoxCandidate:
    """A detected dialectical tension that may constitute a Class G paradox."""
    thesis:              str
    antithesis:          str
    tension_description: str


@dataclass(frozen=True)
class OppositionResult:
    """
    Full engine output for one process() / run_gate() / run_request() call.

    Attributes
    ----------
    oppositions : list of OppositionItem
    gate_outcome : GateOutcome or None
    gate_score : float or None
    caveats : list of str or None
    resistance_result : ResistanceResult or None
    dialectical_tension_discovered : bool
    paradox_candidates : list of ParadoxCandidate or None
    candidates_generated : int
    candidates_passed_quality : int
    depth_used : OppositionDepth
    modes_used : list of OppositionMode
    opposition_activity : float   Ω(t)
    processing_time_ms : float
    neurochemical_signals : dict
    nihilism_flag : bool
    timestamp : datetime
    """
    oppositions:                  List[OppositionItem]
    gate_outcome:                 Optional[GateOutcome]
    gate_score:                   Optional[float]
    caveats:                      Optional[List[str]]
    resistance_result:            Optional[ResistanceResult]
    dialectical_tension_discovered: bool
    paradox_candidates:           Optional[List[ParadoxCandidate]]
    candidates_generated:         int
    candidates_passed_quality:    int
    depth_used:                   OppositionDepth
    modes_used:                   List[OppositionMode]
    opposition_activity:          float
    processing_time_ms:           float
    neurochemical_signals:        Dict
    nihilism_flag:                bool
    timestamp:                    datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class OppositionGateInput:
    """
    Input for the automatic quality gate (sits before response delivery).

    Attributes
    ----------
    candidate_response : str
        The response to stress-test.
    response_reasoning : ReasoningTrace
        How the response was constructed.
    original_query : str
        What the user asked.
    conversation_context : list of DialogueTurn
    intention_vector : IntentionVector
    active_contradiction_flags : list of ContradictionFlag
    active_fallacy_flags : list of FallacyFlag
    active_bias_flags : list of BiasFlag
    active_trap_flags : list of TrapFlag
    active_mode : OperationalMode
    depth_override : OppositionDepth or None
        Caller can force a specific depth.
    complexity : float     [0,1] topic complexity estimate
    stakes : float         [0,1] output stakes estimate
    novelty : float        [0,1] how novel this reasoning is
    challenge_level : float [0,1] user's expressed challenge level
    """
    candidate_response:          str                        = ""
    response_reasoning:          ReasoningTrace             = field(default_factory=ReasoningTrace)
    original_query:              str                        = ""
    conversation_context:        List[DialogueTurn]         = field(default_factory=list)
    intention_vector:            IntentionVector            = field(default_factory=IntentionVector)
    active_contradiction_flags:  List[ContradictionFlag]    = field(default_factory=list)
    active_fallacy_flags:        List[FallacyFlag]          = field(default_factory=list)
    active_bias_flags:           List[BiasFlag]             = field(default_factory=list)
    active_trap_flags:           List[TrapFlag]             = field(default_factory=list)
    active_mode:                 OperationalMode            = OperationalMode.NORMAL
    depth_override:              Optional[OppositionDepth]  = None
    complexity:                  float                      = 0.5
    stakes:                      float                      = 0.5
    novelty:                     float                      = 0.5
    challenge_level:             float                      = 0.5


@dataclass(frozen=True)
class OppositionRequest:
    """
    Callable interface input — any engine can request adversarial testing.

    Attributes
    ----------
    target : str
        What to stress-test.
    target_type : TargetType
    caller_engine : str
        Which engine is requesting.
    caller_context : dict
    requested_modes : list of OppositionMode
    requested_depth : OppositionDepth
    focus_area : str or None
        e.g. "Test especially the causal claim"
    min_opposition_quality : float
        Caller can demand higher quality than the default threshold.
    active_mode : OperationalMode
    """
    target:                str                      = ""
    target_type:           TargetType               = TargetType.ARGUMENT
    caller_engine:         str                      = "unknown"
    caller_context:        Dict                     = field(default_factory=dict)
    requested_modes:       List[OppositionMode]     = field(default_factory=lambda: [OppositionMode.COUNTERARGUMENT])
    requested_depth:       OppositionDepth          = OppositionDepth.STANDARD
    focus_area:            Optional[str]            = None
    min_opposition_quality: float                   = 0.45
    active_mode:           OperationalMode          = OperationalMode.NORMAL


# =====================================================================
# Mutable engine state
# =====================================================================

@dataclass
class OppositionEngineState:
    """Runtime neurochemical + anti-nihilism state."""
    ne_level:   float = 0.0
    da_level:   float = 0.0
    ach_level:  float = 0.0
    gaba_level: float = 0.0
    cor_level:  float = 0.0
    # Rolling window for anti-nihilism (True = blocked)
    gate_history: Deque[bool] = field(default_factory=lambda: deque(maxlen=20))

    def as_dict(self) -> Dict:
        return {
            "ne_level":   self.ne_level,
            "da_level":   self.da_level,
            "ach_level":  self.ach_level,
            "gaba_level": self.gaba_level,
            "cor_level":  self.cor_level,
        }


# =====================================================================
# Pure functions — depth selection
# =====================================================================

def compute_depth_score(
    complexity:      float,
    confidence:      float,
    stakes:          float,
    novelty:         float,
    challenge_level: float,
    config:          OppositionEngineConfig,
) -> float:
    """
    depth_score = w_d1×complexity + w_d2×(1-confidence) + w_d3×stakes
                + w_d4×novelty + w_d5×challenge_level
    ∈ [0, 1].
    """
    score = (
        config.w_d1 * complexity
        + config.w_d2 * (1.0 - confidence)
        + config.w_d3 * stakes
        + config.w_d4 * novelty
        + config.w_d5 * challenge_level
    )
    return _clamp(score)


def select_depth(
    depth_score: float,
    mode: OperationalMode,
    depth_override: Optional[OppositionDepth],
    config: OppositionEngineConfig,
) -> OppositionDepth:
    """
    Apply mode overrides and depth_score thresholds.

    Dev mode → always Deep.
    REM_NORMAL → always Deep.
    REM_DREAM → engine is disabled (caller must check first).
    """
    if depth_override is not None:
        return depth_override

    # Mode overrides
    if mode == OperationalMode.DEV:
        return OppositionDepth.DEEP
    if mode == OperationalMode.REM_NORMAL:
        return OppositionDepth.DEEP
    if mode == OperationalMode.REM_DREAM:
        return OppositionDepth.QUICK  # disabled — caller checks is_disabled_for_mode

    if depth_score < 0.30:
        return OppositionDepth.QUICK
    if depth_score < 0.65:
        return OppositionDepth.STANDARD
    return OppositionDepth.DEEP


def is_disabled_for_mode(mode: OperationalMode) -> bool:
    """REM_DREAM mode disables the opposition engine entirely."""
    return mode == OperationalMode.REM_DREAM


def get_depth_params(
    depth: OppositionDepth,
    config: OppositionEngineConfig,
) -> Tuple[int, int, int, float]:
    """Returns (max_passes, sufficient, max_output, depth_multiplier)."""
    if depth == OppositionDepth.QUICK:
        return config.max_passes_quick, config.sufficient_quick, config.max_output_quick, config.depth_mult_quick
    if depth == OppositionDepth.STANDARD:
        return config.max_passes_standard, config.sufficient_standard, config.max_output_standard, config.depth_mult_standard
    return config.max_passes_deep, config.sufficient_deep, config.max_output_deep, config.depth_mult_deep


def get_modes_for_depth(depth: OppositionDepth) -> List[OppositionMode]:
    """Return which modes are active at the given depth."""
    if depth == OppositionDepth.QUICK:
        return [OppositionMode.COUNTERARGUMENT]
    if depth == OppositionDepth.STANDARD:
        return [
            OppositionMode.COUNTERARGUMENT,
            OppositionMode.COUNTEREXAMPLE,
            OppositionMode.ALTERNATIVE_EXPLANATION,
        ]
    return list(OppositionMode)   # all 5


# =====================================================================
# Pure functions — quality evaluation
# =====================================================================

def classify_severity(quality_score: float) -> OppositionSeverity:
    """Map quality score to severity level."""
    if quality_score > 0.85:
        return OppositionSeverity.CRITICAL
    if quality_score > 0.70:
        return OppositionSeverity.SIGNIFICANT
    if quality_score > 0.55:
        return OppositionSeverity.MODERATE
    return OppositionSeverity.MINOR


def _severity_weight(severity: OppositionSeverity) -> float:
    """Numeric weight for severity in Ω(t) computation."""
    return {
        OppositionSeverity.MINOR:       0.4,
        OppositionSeverity.MODERATE:    0.6,
        OppositionSeverity.SIGNIFICANT: 0.8,
        OppositionSeverity.CRITICAL:    1.0,
    }[severity]


def _mode_weight(mode: OppositionMode, config: OppositionEngineConfig) -> float:
    return {
        OppositionMode.COUNTERARGUMENT:        config.w_mode_counterargument,
        OppositionMode.COUNTEREXAMPLE:         config.w_mode_counterexample,
        OppositionMode.ALTERNATIVE_EXPLANATION: config.w_mode_alternative_explanation,
        OppositionMode.ASSUMPTION_EXCAVATION:  config.w_mode_assumption_excavation,
        OppositionMode.STRUCTURAL_RESISTANCE:  config.w_mode_structural_resistance,
    }[mode]


def compute_opposition_quality(
    validity:   float,
    relevance:  float,
    strength:   float,
    novelty:    float,
    config:     OppositionEngineConfig,
) -> float:
    """
    Q = w_q1×validity + w_q2×relevance + w_q3×strength + w_q4×novelty
    ∈ [0, 1].
    """
    return (
        config.w_q1 * validity
        + config.w_q2 * relevance
        + config.w_q3 * strength
        + config.w_q4 * novelty
    )


def compute_opposition_activity(
    oppositions:      List[OppositionItem],
    depth_multiplier: float,
    config:           OppositionEngineConfig,
) -> float:
    """
    Ω(t) = depth_multiplier × Σ_o quality(o) × severity(o) × w_mode(o)
    Clamped to [0, 1] for a normalised version; raw value can exceed 1.
    """
    if not oppositions:
        return 0.0
    total = sum(
        item.quality_score * _severity_weight(item.severity) * _mode_weight(item.mode, config)
        for item in oppositions
    )
    return depth_multiplier * total


# =====================================================================
# Pure functions — gate scoring
# =====================================================================

def compute_gate_score(oppositions: List[OppositionItem]) -> float:
    """
    gate_score = max(quality × relevance) for all generated opposition.
    Returns 0.0 if no oppositions.
    """
    if not oppositions:
        return 0.0
    return max(item.quality_score * item.relevance for item in oppositions)


def classify_gate_outcome(gate_score: float, config: OppositionEngineConfig) -> GateOutcome:
    """Map gate_score to GateOutcome."""
    if gate_score < config.gate_pass_max:
        return GateOutcome.PASS
    if gate_score < config.gate_caveat_max:
        return GateOutcome.PASS_WITH_CAVEAT
    if gate_score < config.gate_revise_max:
        return GateOutcome.REVISE
    return GateOutcome.BLOCK


def extract_caveats(oppositions: List[OppositionItem]) -> List[str]:
    """
    For PASS_WITH_CAVEAT: collect content of MINOR/MODERATE oppositions as caveats.
    """
    return [
        item.content
        for item in oppositions
        if item.severity in (OppositionSeverity.MINOR, OppositionSeverity.MODERATE)
    ]


# =====================================================================
# Pure functions — anti-nihilism
# =====================================================================

def compute_block_rate(gate_history: Deque[bool]) -> float:
    """block_rate = n_blocked / n_evaluated over the rolling window."""
    if not gate_history:
        return 0.0
    return sum(gate_history) / len(gate_history)


def get_effective_quality_threshold(
    base_threshold:     float,
    block_rate:         float,
    nihilism_threshold: float,
) -> float:
    """
    If block_rate > θ_nihilism, raise the quality threshold.
    Returns the adjusted threshold ∈ [base_threshold, 1.0].
    """
    if block_rate > nihilism_threshold:
        excess = block_rate - nihilism_threshold
        # Linear ramp up to +0.20 boost when block_rate = 1.0
        boost = 0.20 * (excess / (1.0 - nihilism_threshold))
        return min(1.0, base_threshold + boost)
    return base_threshold


# =====================================================================
# Pure functions — dialectical tension detection
# =====================================================================

def detect_dialectical_tension(
    target:      str,
    oppositions: List[OppositionItem],
    config:      OppositionEngineConfig,
) -> List[ParadoxCandidate]:
    """
    Look for CRITICAL + SIGNIFICANT opposition that is ALSO well-supported
    (high quality) — where both the original claim and the opposition
    appear to hold simultaneously.

    A tension candidate is created when:
    - An opposition is CRITICAL severity (quality > 0.85)
    - Its target_component is "conclusion" (challenges the core claim, not a detail)
    - This suggests the original AND opposition may BOTH be defensible
    """
    candidates: List[ParadoxCandidate] = []
    for item in oppositions:
        if (
            item.severity == OppositionSeverity.CRITICAL
            and item.target_component == "conclusion"
        ):
            candidates.append(ParadoxCandidate(
                thesis=target[:200],
                antithesis=item.content[:200],
                tension_description=(
                    f"{item.attack_type} challenge creates an irresolvable tension: "
                    f"both the original position and this opposition appear defensible."
                ),
            ))
    return candidates


# =====================================================================
# Opposition generation — Mode implementations
# =====================================================================
# These are deterministic template-driven generators.
# They produce structured oppositions from the target text using
# vocabulary analysis, assumption detection, and attack pattern templates.
# No LLM is called — this is pure-Python symbolic generation.

_CAUSAL_PATTERNS = [
    "causes", "leads to", "results in", "produces", "makes",
    "triggers", "brings about", "creates", "drives",
]

_UNIVERSAL_PATTERNS = [
    "always", "never", "all", "every", "none", "no one", "everything",
    "everyone", "anywhere", "nowhere", "nothing",
]

_HEDGED_PATTERNS = [
    "might", "may", "could", "perhaps", "possibly", "likely",
    "generally", "often", "sometimes", "tends to",
]

_INFERENCE_MARKERS = [
    "therefore", "thus", "hence", "so", "because", "since",
    "it follows", "consequently", "as a result",
]


def _extract_components(target: str) -> Dict[str, List[str]]:
    """
    Heuristically extract argument components from text.
    Returns {"premises": [...], "conclusion": ..., "causal_claims": [...],
             "universal_claims": [...], "hedged_claims": [...], "inference_steps": [...]}.
    """
    words = target.lower().split()
    sentences = [s.strip() for s in target.replace("!", ".").replace("?", ".").split(".") if s.strip()]

    causal = [s for s in sentences if any(p in s.lower() for p in _CAUSAL_PATTERNS)]
    universal = [s for s in sentences if any(p in s.lower() for p in _UNIVERSAL_PATTERNS)]
    hedged = [s for s in sentences if any(p in s.lower() for p in _HEDGED_PATTERNS)]

    # Heuristic: last sentence with inference marker = conclusion
    # otherwise last sentence
    conclusion_sentences = [s for s in sentences if any(m in s.lower() for m in _INFERENCE_MARKERS)]
    conclusion = conclusion_sentences[-1] if conclusion_sentences else (sentences[-1] if sentences else target[:100])
    premises = [s for s in sentences if s != conclusion]

    return {
        "premises":       premises,
        "conclusion":     conclusion,
        "causal_claims":  causal,
        "universal_claims": universal,
        "hedged_claims":  hedged,
        "all_sentences":  sentences,
    }


def _score_components(components: Dict) -> Dict[str, float]:
    """
    Assign vulnerability scores to each component.
    vulnerability = (1 - evidence_strength) × importance_to_conclusion
    Heuristic: universal claims are most vulnerable (evidence_strength low).
    """
    scores: Dict[str, float] = {}
    for i, premise in enumerate(components["premises"]):
        # Universal = high vulnerability; hedged = low vulnerability
        is_universal = any(p in premise.lower() for p in _UNIVERSAL_PATTERNS)
        is_hedged    = any(p in premise.lower() for p in _HEDGED_PATTERNS)
        evidence_strength = 0.3 if is_universal else (0.7 if is_hedged else 0.5)
        importance        = 1.0 - (i * 0.1)   # earlier premises assumed more important
        scores[f"premise_{i}"] = (1.0 - evidence_strength) * max(0.1, importance)
    return scores


def generate_counterargument(
    target:    str,
    reasoning: ReasoningTrace,
    pass_n:    int,
    config:    OppositionEngineConfig,
) -> OppositionItem:
    """
    Mode 1: Generate a counterargument against the target.
    Uses premise attack, inference attack, and conclusion attack templates.
    """
    components = _extract_components(target)
    vuln_scores = _score_components(components)

    # Pick highest-vulnerability component
    if vuln_scores:
        attack_key = max(vuln_scores, key=vuln_scores.get)
        vuln_score = vuln_scores[attack_key]
    else:
        attack_key = "conclusion"
        vuln_score = 0.5

    # Rotate attack type across passes
    attack_types = ["premise_attack", "inference_attack", "conclusion_attack"]
    attack_type = attack_types[pass_n % len(attack_types)]

    if attack_type == "premise_attack" and components["premises"]:
        premise_idx = int(attack_key.split("_")[-1]) if "_" in attack_key else 0
        premise_idx = min(premise_idx, len(components["premises"]) - 1)
        attacked = components["premises"][premise_idx]
        content = (
            f"Premise attack: The claim '{attacked}' is insufficiently supported. "
            f"Alternative evidence could suggest the opposite, or the premise may hold "
            f"only within a restricted domain that doesn't generalise to this case."
        )
        target_component = "premise"
    elif attack_type == "inference_attack" and components["causal_claims"]:
        causal = components["causal_claims"][pass_n % len(components["causal_claims"])]
        content = (
            f"Inference attack: Even granting the premises, the inferential step '{causal}' "
            f"does not necessarily follow. There may be intervening variables, confounders, "
            f"or alternative causal pathways that break the claimed connection."
        )
        target_component = "inference"
    else:
        conclusion = components["conclusion"]
        content = (
            f"Conclusion attack: The conclusion '{conclusion}' can be directly challenged. "
            f"Consider the negation: what evidence would support ¬(conclusion)? "
            f"A reductio ad absurdum or a direct counterexample may exist within the domain."
        )
        target_component = "conclusion"

    # Quality estimates (heuristic: higher vulnerability → higher strength)
    validity  = max(0.5, min(0.95, 0.6 + vuln_score * 0.3))
    relevance = 0.75 - pass_n * 0.05   # later passes may drift
    strength  = max(0.4, min(0.9, vuln_score + 0.2))
    novelty   = max(0.4, 0.80 - pass_n * 0.10)

    quality = compute_opposition_quality(validity, relevance, strength, novelty, config)

    return OppositionItem(
        item_id=str(uuid.uuid4()),
        mode=OppositionMode.COUNTERARGUMENT,
        content=content,
        quality_score=quality,
        validity=validity,
        relevance=relevance,
        strength=strength,
        novelty=novelty,
        target_component=target_component,
        severity=classify_severity(quality),
        attack_type=attack_type,
    )


def generate_counterexample(
    target: str,
    pass_n: int,
    config: OppositionEngineConfig,
) -> OppositionItem:
    """
    Mode 2: Find specific cases where the general claim doesn't hold.
    Uses boundary case analysis, assumption relaxation, domain transfer.
    """
    components = _extract_components(target)
    strategies = ["boundary_case", "assumption_relaxation", "domain_transfer"]
    strategy = strategies[pass_n % len(strategies)]

    if strategy == "boundary_case":
        universal = components["universal_claims"]
        base = universal[0] if universal else components["conclusion"]
        content = (
            f"Boundary case: The claim '{base}' may fail at domain edges. "
            f"Consider edge cases where the standard conditions do not hold — "
            f"extreme values, degenerate inputs, or limiting cases often expose "
            f"where general claims break down."
        )
    elif strategy == "assumption_relaxation":
        premise = components["premises"][0] if components["premises"] else target[:80]
        content = (
            f"Assumption relaxation: The claim implicitly assumes '{premise}'. "
            f"If this scope assumption is relaxed — e.g. applied to a broader population "
            f"or a different time frame — the claim may no longer hold."
        )
    else:
        content = (
            f"Domain transfer: The abstract claim structure, applied to a different domain, "
            f"may fail. If it fails in the analogous domain, that failure reason may also "
            f"apply to the original domain — suggesting a hidden structural dependency."
        )

    validity  = 0.65
    relevance = max(0.5, 0.80 - pass_n * 0.06)
    strength  = 0.60
    novelty   = max(0.35, 0.75 - pass_n * 0.10)

    quality = compute_opposition_quality(validity, relevance, strength, novelty, config)

    return OppositionItem(
        item_id=str(uuid.uuid4()),
        mode=OppositionMode.COUNTEREXAMPLE,
        content=content,
        quality_score=quality,
        validity=validity,
        relevance=relevance,
        strength=strength,
        novelty=novelty,
        target_component="conclusion",
        severity=classify_severity(quality),
        attack_type=strategy,
    )


def generate_alternative_explanation(
    target: str,
    pass_n: int,
    config: OppositionEngineConfig,
) -> Optional[OppositionItem]:
    """
    Mode 3: Generate plausible alternative explanations.
    Returns None if alternative plausibility is below θ_alt.
    """
    components = _extract_components(target)
    strategies = ["causal_reversal", "occam_competition", "perspective_shift"]
    strategy = strategies[pass_n % len(strategies)]

    if strategy == "causal_reversal" and components["causal_claims"]:
        causal = components["causal_claims"][0]
        content = (
            f"Causal reversal: Instead of '{causal}', consider whether the direction "
            f"is reversed (B causes A), or a third factor C causes both A and B "
            f"(confounding), or A and B are merely correlated without direct causation."
        )
        plausibility = 0.55
    elif strategy == "occam_competition":
        content = (
            f"Occam competition: A simpler explanation for the same evidence may exist. "
            f"If the simpler alternative explains the observations with fewer assumptions, "
            f"it should be preferred unless additional evidence specifically supports the "
            f"more complex account."
        )
        plausibility = 0.60
    else:
        content = (
            f"Perspective shift: From a different ideological, cultural, or disciplinary "
            f"perspective, this evidence might be interpreted differently. "
            f"Such perspectival alternatives may be equally consistent with the observations."
        )
        plausibility = 0.50

    if plausibility < config.theta_alt:
        return None

    validity  = 0.60
    relevance = max(0.45, 0.75 - pass_n * 0.07)
    strength  = plausibility
    novelty   = max(0.35, 0.70 - pass_n * 0.10)

    quality = compute_opposition_quality(validity, relevance, strength, novelty, config)

    return OppositionItem(
        item_id=str(uuid.uuid4()),
        mode=OppositionMode.ALTERNATIVE_EXPLANATION,
        content=content,
        quality_score=quality,
        validity=validity,
        relevance=relevance,
        strength=strength,
        novelty=novelty,
        target_component="explanation",
        severity=classify_severity(quality),
        attack_type=strategy,
    )


def generate_assumption_excavation(
    target:    str,
    reasoning: ReasoningTrace,
    pass_n:    int,
    config:    OppositionEngineConfig,
) -> OppositionItem:
    """
    Mode 4: Identify and challenge unstated assumptions.
    Uses conditional chain decomposition and negation testing.
    """
    components = _extract_components(target)
    strategies = ["conditional_decomposition", "negation_testing"]
    strategy = strategies[pass_n % len(strategies)]

    steps = reasoning.inference_steps or components["all_sentences"]
    step = steps[pass_n % len(steps)] if steps else target[:100]

    if strategy == "conditional_decomposition":
        content = (
            f"Assumption excavation: The step '{step}' implicitly requires certain "
            f"conditions to hold (a hidden IF-THEN). Making these conditions explicit: "
            f"IF [assumed condition] THEN [{step}]. This assumed condition may be "
            f"contested, domain-dependent, or simply unestablished."
        )
    else:
        content = (
            f"Negation test: If the assumption underlying '{step}' is negated, "
            f"does the reasoning chain still hold? If not, this is a NECESSARY assumption — "
            f"a critical dependency that should be stated explicitly and defended."
        )

    validity  = 0.70
    relevance = max(0.50, 0.80 - pass_n * 0.06)
    strength  = 0.65
    novelty   = max(0.40, 0.72 - pass_n * 0.08)

    quality = compute_opposition_quality(validity, relevance, strength, novelty, config)

    return OppositionItem(
        item_id=str(uuid.uuid4()),
        mode=OppositionMode.ASSUMPTION_EXCAVATION,
        content=content,
        quality_score=quality,
        validity=validity,
        relevance=relevance,
        strength=strength,
        novelty=novelty,
        target_component="assumption",
        severity=classify_severity(quality),
        attack_type=strategy,
    )


def generate_structural_resistance(
    target:     str,
    n_rounds:   int,
    config:     OppositionEngineConfig,
) -> Tuple[List[OppositionItem], ResistanceResult]:
    """
    Mode 5: Apply N rounds of adversarial Socratic questioning.
    Returns (list_of_challenge_items, ResistanceResult).
    """
    question_templates = [
        ("implicative",   "If your position is true, wouldn't that imply an uncomfortable consequence?"),
        ("counter_case",  "How does your position account for challenging edge cases where it appears to fail?"),
        ("consistency",   "Isn't your position inconsistent with other widely accepted claims in this domain?"),
        ("reframing",     "What if we look at this from a perspective that fundamentally undermines the framing?"),
        ("grounding",     "What specific evidence supports this rather than the general principle you're invoking?"),
    ]

    items: List[OppositionItem] = []
    key_challenges: List[str] = []

    for round_n in range(n_rounds):
        q_type, q_text = question_templates[round_n % len(question_templates)]
        content = (
            f"Round {round_n + 1} adversarial challenge ({q_type}): "
            f"Against target '{target[:60]}...': {q_text}"
        )
        key_challenges.append(f"Round {round_n + 1}: {q_type} challenge")

        validity  = 0.72
        relevance = max(0.50, 0.82 - round_n * 0.05)
        strength  = max(0.45, 0.75 - round_n * 0.06)
        novelty   = max(0.35, 0.70 - round_n * 0.08)
        quality   = compute_opposition_quality(validity, relevance, strength, novelty, config)

        items.append(OppositionItem(
            item_id=str(uuid.uuid4()),
            mode=OppositionMode.STRUCTURAL_RESISTANCE,
            content=content,
            quality_score=quality,
            validity=validity,
            relevance=relevance,
            strength=strength,
            novelty=novelty,
            target_component="conclusion",
            severity=classify_severity(quality),
            attack_type=f"resistance_round_{round_n + 1}",
        ))

    # Survival assessment: based on how many rounds produced SIGNIFICANT+ opposition
    significant_rounds = sum(
        1 for i in items
        if i.severity in (OppositionSeverity.SIGNIFICANT, OppositionSeverity.CRITICAL)
    )
    fraction_survived = 1.0 - significant_rounds / max(n_rounds, 1)

    if fraction_survived >= 0.75:
        status = PositionStatus.SURVIVED
        delta = "Core claim intact after adversarial questioning."
    elif fraction_survived >= 0.50:
        status = PositionStatus.IMPROVED
        delta = "Questioning revealed nuances that should be incorporated."
    elif fraction_survived >= 0.25:
        status = PositionStatus.TRANSFORMED
        delta = "Questioning revealed the claim was about something different than formulated."
    else:
        status = PositionStatus.COLLAPSED
        delta = "Core claim could not survive adversarial challenge."

    resistance = ResistanceResult(
        rounds_survived=int(fraction_survived * n_rounds),
        final_position_status=status,
        key_challenges=key_challenges,
        position_delta=delta,
    )
    return items, resistance


# =====================================================================
# Core generation loop
# =====================================================================

def run_generation_loop(
    target:           str,
    reasoning:        ReasoningTrace,
    mode:             OppositionMode,
    depth:            OppositionDepth,
    quality_threshold: float,
    config:           OppositionEngineConfig,
) -> Tuple[List[OppositionItem], int, int]:
    """
    Core generation loop for a single mode.

    Returns (selected_items, candidates_generated, candidates_passed).
    For STRUCTURAL_RESISTANCE mode, delegates to generate_structural_resistance.
    """
    max_passes, sufficient, max_output, _ = get_depth_params(depth, config)

    if mode == OppositionMode.STRUCTURAL_RESISTANCE:
        n_rounds = config.sr_rounds_deep if depth == OppositionDepth.DEEP else config.sr_rounds_standard
        items, _ = generate_structural_resistance(target, n_rounds, config)
        passed = [i for i in items if i.quality_score >= quality_threshold]
        return passed[:max_output], len(items), len(passed)

    candidates: List[OppositionItem] = []
    candidates_generated = 0
    candidates_passed    = 0
    last_best_quality    = -1.0

    for pass_n in range(max_passes):
        if mode == OppositionMode.COUNTERARGUMENT:
            item: Optional[OppositionItem] = generate_counterargument(target, reasoning, pass_n, config)
        elif mode == OppositionMode.COUNTEREXAMPLE:
            item = generate_counterexample(target, pass_n, config)
        elif mode == OppositionMode.ALTERNATIVE_EXPLANATION:
            item = generate_alternative_explanation(target, pass_n, config)
        elif mode == OppositionMode.ASSUMPTION_EXCAVATION:
            item = generate_assumption_excavation(target, reasoning, pass_n, config)
        else:
            break   # unknown mode

        candidates_generated += 1

        if item is not None and item.quality_score >= quality_threshold:
            candidates.append(item)
            candidates_passed += 1

        # Early exit: sufficient opposition collected
        if len(candidates) >= sufficient:
            break

        # Early exit: no quality improvement in last passes
        if pass_n > 2 and candidates:
            best_quality = max(c.quality_score for c in candidates)
            if abs(best_quality - last_best_quality) < 0.01:
                break
            last_best_quality = best_quality

    # Select best
    candidates.sort(key=lambda x: x.quality_score, reverse=True)
    selected = candidates[:max_output]
    return selected, candidates_generated, candidates_passed


# =====================================================================
# Neurochemical signals
# =====================================================================

def compute_neurochemical_signals(
    oppositions:      List[OppositionItem],
    gate_outcome:     Optional[GateOutcome],
    depth:            OppositionDepth,
    omega:            float,
    state:            OppositionEngineState,
    config:           OppositionEngineConfig,
    rng:              np.random.Generator,
) -> Dict:
    """
    Compute all neurochemical output signals.
    """
    _, _, _, depth_mult = get_depth_params(depth, config)

    signals: Dict = {
        "ne_burst":       0.0,
        "da_signal":      0.0,
        "ach_burst":      0.0,
        "gaba_suppress": 0.0,
        "cor_eustress":   0.0,
        "beta_boost":     0.0,
        "gamma_boost":    0.0,
        "theta_boost":    0.0,
        "theta_suppress": 0.0,
    }

    opposition_active = len(oppositions) > 0

    # NE — adversarial alertness
    if opposition_active:
        poisson_draw = rng.poisson(config.lambda_ne)
        signals["ne_burst"] = float(
            config.beta_opposition * omega * poisson_draw / max(config.lambda_ne, 1.0)
        )

    # DA — outcome-dependent reward
    if gate_outcome == GateOutcome.PASS and oppositions:
        gate_score = compute_gate_score(oppositions)
        gamma_draw = rng.gamma(shape=2.0, scale=0.3)
        signals["da_signal"] = float(
            config.beta_pass * (1.0 - gate_score) * depth_mult * gamma_draw
        )
    elif gate_outcome in (GateOutcome.REVISE, GateOutcome.BLOCK) and oppositions:
        max_quality   = max(i.quality_score for i in oppositions)
        severity_vals = {
            OppositionSeverity.MINOR: 0.4, OppositionSeverity.MODERATE: 0.6,
            OppositionSeverity.SIGNIFICANT: 0.8, OppositionSeverity.CRITICAL: 1.0,
        }
        severity_max = max(severity_vals[i.severity] for i in oppositions)
        gamma_draw = rng.gamma(shape=2.0, scale=0.4)
        signals["da_signal"] = float(config.beta_find * max_quality * severity_max * gamma_draw)
        if gate_outcome == GateOutcome.BLOCK:
            signals["da_signal"] += config.beta_critical * 0.15

    # ACh — sustained analytical effort
    if opposition_active:
        gamma_draw = rng.gamma(shape=2.5, scale=0.35)
        signals["ach_burst"] = float(config.beta_opp_ach * depth_mult * gamma_draw)

    # GABA — reuptake suppression (reduces defensiveness)
    if opposition_active:
        signals["gaba_suppress"] = float(config.eta_opposition)

    # COR — mild eustress (Deep only)
    if depth == OppositionDepth.DEEP and opposition_active:
        signals["cor_eustress"] = float(config.rho_eustress)

    # Beta enhancement
    if opposition_active:
        signals["beta_boost"] = float(config.psi_beta_opp * omega)

    # Gamma enhancement (Structural Resistance only)
    sr_active = any(i.mode == OppositionMode.STRUCTURAL_RESISTANCE for i in oppositions)
    if sr_active:
        signals["gamma_boost"] = float(config.psi_gamma_opp)

    # Theta modulation
    search_modes = {OppositionMode.COUNTEREXAMPLE, OppositionMode.ALTERNATIVE_EXPLANATION}
    search_active = any(i.mode in search_modes for i in oppositions)
    if search_active:
        signals["theta_boost"] = float(config.psi_theta_search)
    if sr_active:
        signals["theta_suppress"] = float(config.psi_theta_resist)

    return signals


# =====================================================================
# Bidirectional neurochemical feedback modulation
# =====================================================================

def apply_neurochem_feedback(
    quality_threshold: float,
    state:             OppositionEngineState,
    config:            OppositionEngineConfig,
) -> float:
    """
    Bidirectional feedback: NE/GABA/ACh/COR modulate the quality threshold
    and, implicitly, the depth selection.

    Returns the adjusted quality threshold.
    """
    adjusted = quality_threshold

    # High NE → lower threshold (more aggressive challenge)
    if state.ne_level > 0.6:
        adjusted -= 0.05 * (state.ne_level - 0.6) / 0.4

    # High GABA → lower threshold (more thorough, less impulsive acceptance)
    if state.gaba_level > 0.5:
        adjusted -= 0.03 * state.gaba_level

    # High COR (external stress) → raise threshold (conserve resources)
    if state.cor_level > 0.5:
        adjusted += 0.10 * (state.cor_level - 0.5) / 0.5

    return max(0.20, min(0.90, adjusted))


# =====================================================================
# Main Engine
# =====================================================================

class SimulatedOppositionEngine:
    """
    Simulated Opposition Engine — the system's internal red team.

    Ports
    -----
    configure(mode)              — set operational mode
    update_neurochem_state(d)    — update NE/DA/ACh/GABA/COR from readout
    run_gate(gate_input)         — automatic quality gate before response delivery
    run_request(request)         — callable interface for other engines
    process(inp)                 — unified interface (delegates to run_gate or run_request)
    get_status()                 — dict with engine metadata
    """

    engine_id = "simulated_opposition_engine"
    cluster   = "dialectic"

    def __init__(
        self,
        config: Optional[OppositionEngineConfig] = None,
        rng:    Optional[np.random.Generator]    = None,
    ) -> None:
        self._config = config or OppositionEngineConfig()
        self._mode   = OperationalMode.NORMAL
        self._state  = OppositionEngineState()
        self._rng    = rng or np.random.default_rng()

    def configure(self, mode: OperationalMode) -> None:
        """Set operational mode."""
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict) -> None:
        """
        Update internal neurochemical state from readout.
        Accepts keys: ne, da, ach, gaba, cor.
        """
        s = self._state
        if "ne"   in state_dict: s.ne_level   = float(state_dict["ne"])
        if "da"   in state_dict: s.da_level   = float(state_dict["da"])
        if "ach"  in state_dict: s.ach_level  = float(state_dict["ach"])
        if "gaba" in state_dict: s.gaba_level = float(state_dict["gaba"])
        if "cor"  in state_dict: s.cor_level  = float(state_dict["cor"])

    def process(self, inp):
        """Unified interface — delegates to run_gate or run_request."""
        if isinstance(inp, OppositionGateInput):
            return self.run_gate(inp)
        elif isinstance(inp, OppositionRequest):
            return self.run_request(inp)
        raise TypeError(f"Expected OppositionGateInput or OppositionRequest, got {type(inp)}")

    def run_gate(self, gate_input: OppositionGateInput) -> OppositionResult:
        """
        Automatic quality gate. Runs before response delivery.
        Returns PASS/PASS_WITH_CAVEAT/REVISE/BLOCK outcome.
        """
        if is_disabled_for_mode(gate_input.active_mode):
            return self._disabled_result()

        t0 = time.perf_counter()

        # Depth selection
        depth_score = compute_depth_score(
            gate_input.complexity,
            gate_input.response_reasoning.confidence,
            gate_input.stakes,
            gate_input.novelty,
            gate_input.challenge_level,
            self._config,
        )
        depth = select_depth(
            depth_score, gate_input.active_mode, gate_input.depth_override, self._config
        )
        modes = get_modes_for_depth(depth)

        result = self._run_opposition(
            target=gate_input.candidate_response,
            reasoning=gate_input.response_reasoning,
            modes=modes,
            depth=depth,
            compute_gate=True,
            active_mode=gate_input.active_mode,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return OppositionResult(
            oppositions=result["oppositions"],
            gate_outcome=result["gate_outcome"],
            gate_score=result["gate_score"],
            caveats=result["caveats"],
            resistance_result=result["resistance_result"],
            dialectical_tension_discovered=result["tension_discovered"],
            paradox_candidates=result["paradox_candidates"],
            candidates_generated=result["candidates_generated"],
            candidates_passed_quality=result["candidates_passed"],
            depth_used=depth,
            modes_used=modes,
            opposition_activity=result["omega"],
            processing_time_ms=elapsed_ms,
            neurochemical_signals=result["neuro"],
            nihilism_flag=result["nihilism_flag"],
        )

    def run_request(self, request: OppositionRequest) -> OppositionResult:
        """
        Callable interface — any engine can request adversarial testing.
        Uses caller-specified modes and depth.
        """
        if is_disabled_for_mode(request.active_mode):
            return self._disabled_result()

        t0 = time.perf_counter()

        # Quality threshold from caller or neurochem-modulated default
        quality_threshold = apply_neurochem_feedback(
            max(request.min_opposition_quality, self._config.theta_opposition_quality),
            self._state, self._config,
        )

        result = self._run_opposition(
            target=request.target,
            reasoning=ReasoningTrace(),   # no reasoning trace for arbitrary requests
            modes=request.requested_modes,
            depth=request.requested_depth,
            compute_gate=False,
            active_mode=request.active_mode,
            quality_threshold_override=quality_threshold,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return OppositionResult(
            oppositions=result["oppositions"],
            gate_outcome=None,
            gate_score=None,
            caveats=None,
            resistance_result=result["resistance_result"],
            dialectical_tension_discovered=result["tension_discovered"],
            paradox_candidates=result["paradox_candidates"],
            candidates_generated=result["candidates_generated"],
            candidates_passed_quality=result["candidates_passed"],
            depth_used=request.requested_depth,
            modes_used=request.requested_modes,
            opposition_activity=result["omega"],
            processing_time_ms=elapsed_ms,
            neurochemical_signals=result["neuro"],
            nihilism_flag=result["nihilism_flag"],
        )

    def _run_opposition(
        self,
        target:                   str,
        reasoning:                ReasoningTrace,
        modes:                    List[OppositionMode],
        depth:                    OppositionDepth,
        compute_gate:             bool,
        active_mode:              OperationalMode,
        quality_threshold_override: Optional[float] = None,
    ) -> Dict:
        """
        Core opposition pipeline. Returns raw result dict.
        """
        _, _, _, depth_mult = get_depth_params(depth, self._config)

        # Anti-nihilism: get effective quality threshold
        block_rate = compute_block_rate(self._state.gate_history)
        base_thresh = quality_threshold_override or self._config.theta_opposition_quality
        quality_threshold = get_effective_quality_threshold(
            base_thresh, block_rate, self._config.theta_nihilism
        )
        nihilism_flag = block_rate > self._config.theta_nihilism

        # Apply neurochem feedback
        if quality_threshold_override is None:
            quality_threshold = apply_neurochem_feedback(
                quality_threshold, self._state, self._config
            )

        all_oppositions: List[OppositionItem] = []
        total_generated = 0
        total_passed    = 0
        resistance_result: Optional[ResistanceResult] = None

        for mode in modes:
            if mode == OppositionMode.STRUCTURAL_RESISTANCE:
                n_rounds = (
                    self._config.sr_rounds_deep
                    if depth == OppositionDepth.DEEP
                    else self._config.sr_rounds_standard
                )
                items, res = generate_structural_resistance(target, n_rounds, self._config)
                passed = [i for i in items if i.quality_score >= quality_threshold]
                all_oppositions.extend(passed)
                total_generated += len(items)
                total_passed    += len(passed)
                resistance_result = res
            else:
                selected, gen, passed = run_generation_loop(
                    target, reasoning, mode, depth, quality_threshold, self._config
                )
                all_oppositions.extend(selected)
                total_generated += gen
                total_passed    += passed

        # Dialectical tension detection
        tension_candidates = detect_dialectical_tension(target, all_oppositions, self._config)
        tension_discovered = bool(tension_candidates)

        # Gate scoring
        gate_outcome: Optional[GateOutcome] = None
        gate_score:   Optional[float]       = None
        caveats:      Optional[List[str]]   = None

        if compute_gate:
            raw_gate_score = compute_gate_score(all_oppositions)
            gate_outcome   = classify_gate_outcome(raw_gate_score, self._config)

            # Paradox override: if tension discovered, cap at PASS_WITH_CAVEAT
            if tension_discovered and gate_outcome in (GateOutcome.REVISE, GateOutcome.BLOCK):
                gate_outcome = GateOutcome.PASS_WITH_CAVEAT

            gate_score = raw_gate_score
            if gate_outcome == GateOutcome.PASS_WITH_CAVEAT:
                caveats = extract_caveats(all_oppositions)

            # Update gate history for anti-nihilism
            self._state.gate_history.append(gate_outcome == GateOutcome.BLOCK)

        # Ω(t)
        omega = compute_opposition_activity(all_oppositions, depth_mult, self._config)

        # Neurochemical signals
        neuro = compute_neurochemical_signals(
            all_oppositions, gate_outcome, depth, omega,
            self._state, self._config, self._rng,
        )

        return {
            "oppositions":        all_oppositions,
            "gate_outcome":       gate_outcome,
            "gate_score":         gate_score,
            "caveats":            caveats,
            "resistance_result":  resistance_result,
            "tension_discovered": tension_discovered,
            "paradox_candidates": tension_candidates if tension_candidates else None,
            "candidates_generated": total_generated,
            "candidates_passed":    total_passed,
            "omega":              omega,
            "neuro":              neuro,
            "nihilism_flag":      nihilism_flag,
        }

    def _disabled_result(self) -> OppositionResult:
        """Return a no-op result when engine is disabled (REM_DREAM mode)."""
        return OppositionResult(
            oppositions=[],
            gate_outcome=GateOutcome.PASS,
            gate_score=0.0,
            caveats=None,
            resistance_result=None,
            dialectical_tension_discovered=False,
            paradox_candidates=None,
            candidates_generated=0,
            candidates_passed_quality=0,
            depth_used=OppositionDepth.QUICK,
            modes_used=[],
            opposition_activity=0.0,
            processing_time_ms=0.0,
            neurochemical_signals={},
            nihilism_flag=False,
        )

    def get_status(self) -> Dict:
        """Return engine status metadata."""
        block_rate = compute_block_rate(self._state.gate_history)
        return {
            "engine_id":             self.engine_id,
            "version":               "1.0.0",
            "mode":                  self._mode.value,
            "state":                 self._state.as_dict(),
            "block_rate":            block_rate,
            "nihilism_flag":         block_rate > self._config.theta_nihilism,
            "quality_threshold":     self._config.theta_opposition_quality,
        }
