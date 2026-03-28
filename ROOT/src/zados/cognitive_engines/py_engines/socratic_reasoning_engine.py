"""
Socratic Reasoning Engine (Reasoning Cluster — Engine 14).

Implements reasoning through questioning rather than assertion. Two functions:
  Function 1 — External Socratic Dialogue (User-Directed):
      Guide the user toward deeper understanding through progressively structured
      questions that surface assumptions, expose contradictions, and crystallize insight.
  Function 2 — Internal Self-Inquiry (REM / Reflective):
      Apply Socratic method to the system's own unresolved concepts and past
      reasoning, probing knowledge gaps, testing conclusions, identifying hidden
      assumptions.

Dialectical State Machine
--------------------------
States: PROBING → ELENCHUS → APORIA → EXPLORING → MAIEUTICS → EXIT
Transitions driven by: c(t), a(t), u(t), κ(t), f(t) feature signals.

Activation (External)
----------------------
All 4 gates must pass:
  1. Mode gate: mode ∈ {NORMAL, LEARNING, REFLECTIVE, REM_NORMAL}
  2. Intention gate: socratic_score(E_intent) > θ_socratic (default 0.35)
  3. Topic gate: topic_depth(input) > θ_depth (default 0.40)
  4. Fatigue gate: consecutive_socratic_turns < max_socratic_turns (default 5)

Question Generation
--------------------
18 question types across 6 categories.
Template-based (Phase 1) with target selection per state.

Neurochemical Coupling
-----------------------
- ACh Gamma burst ← Socratic processing signal Σ(t)
- DA tonic ← Σ(t)×(1−κ)  (curiosity; drops as insight approaches)
- DA phasic ← insight event × dialectical_distance × difficulty
- OXT drift ← engagement (collaborative); OXT drop ← frustration
- NE Poisson burst ← ELENCHUS state × c(t)
- GABA reuptake suppression ← APORIA state × u(t)
- Theta × (1+ψ) ← EXPLORING / APORIA states
- Beta × (1+ψ) ← ELENCHUS state × c(t)
- Θγ coherence boost ← insight event

Usage
-----
>>> from zados.cognitive_engines.py_engines.socratic_reasoning_engine import (
...     SocraticReasoningEngine, SocraticEngineConfig,
...     SocraticInput, DialogueState, OperationalMode,
... )
>>> engine = SocraticReasoningEngine()
>>> result = engine.process(SocraticInput(user_input=ProcessedStatement(...), ...))
"""

from __future__ import annotations

import math
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    IntentionVector,
    Proposition,
)
from zados.cognitive_engines.py_engines.paradox_detection_engine import (
    ParadoxFlag,
)


# =====================================================================
# Enumerations
# =====================================================================

class DialogueState(str, Enum):
    PROBING      = "PROBING"
    ELENCHUS     = "ELENCHUS"
    APORIA       = "APORIA"
    EXPLORING    = "EXPLORING"
    MAIEUTICS    = "MAIEUTICS"
    EXIT         = "EXIT"


class QuestionType(str, Enum):
    # PROBING types
    CLARIFICATION  = "CLARIFICATION"
    FOUNDATIONAL   = "FOUNDATIONAL"
    DEFINITIONAL   = "DEFINITIONAL"
    SCOPE          = "SCOPE"
    # ELENCHUS types
    IMPLICATIVE    = "IMPLICATIVE"
    COUNTER_CASE   = "COUNTER_CASE"
    CONSISTENCY    = "CONSISTENCY"
    # APORIA types
    REFRAMING      = "REFRAMING"
    ANALOGICAL     = "ANALOGICAL"
    ABSTRACTING    = "ABSTRACTING"
    # EXPLORING types
    GROUNDING      = "GROUNDING"
    EXTENDING      = "EXTENDING"
    CONNECTING     = "CONNECTING"
    TESTING        = "TESTING"
    # MAIEUTICS types
    CRYSTALLIZING  = "CRYSTALLIZING"
    NAMING         = "NAMING"
    INTEGRATING    = "INTEGRATING"
    APPLYING       = "APPLYING"
    # Internal self-inquiry types
    FALSIFICATION  = "FALSIFICATION"
    PROVENANCE     = "PROVENANCE"
    ALTERNATIVE    = "ALTERNATIVE"
    DEPENDENCY     = "DEPENDENCY"
    STABILITY      = "STABILITY"


class ExpectedEffect(str, Enum):
    CLARIFY_AMBIGUITY   = "CLARIFY_AMBIGUITY"
    EXPOSE_ASSUMPTION   = "EXPOSE_ASSUMPTION"
    SURFACE_CONTRADICTION = "SURFACE_CONTRADICTION"
    PRODUCE_CONFUSION   = "PRODUCE_CONFUSION"     # aporia
    BROADEN_SEARCH      = "BROADEN_SEARCH"
    DEEPEN_ANALYSIS     = "DEEPEN_ANALYSIS"
    CRYSTALLIZE_INSIGHT = "CRYSTALLIZE_INSIGHT"
    TEST_ROBUSTNESS     = "TEST_ROBUSTNESS"
    INTEGRATE_INSIGHT   = "INTEGRATE_INSIGHT"


class InsightSource(str, Enum):
    USER_ARTICULATED    = "USER_ARTICULATED"
    SYSTEM_CRYSTALLIZED = "SYSTEM_CRYSTALLIZED"
    COLLABORATIVE       = "COLLABORATIVE"


class YieldReason(str, Enum):
    FATIGUE        = "FATIGUE"
    FRUSTRATION    = "FRUSTRATION"
    TOPIC_MISMATCH = "TOPIC_MISMATCH"
    APORIA_LIMIT   = "APORIA_LIMIT"
    INSIGHT_ACHIEVED = "INSIGHT_ACHIEVED"


# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True)
class SocraticEngineConfig:
    # Activation thresholds
    theta_socratic: float = 0.35
    theta_depth: float = 0.40
    max_socratic_turns: int = 5
    max_aporia_turns: int = 3
    theta_maieutics: float = 0.55
    theta_frustration: float = 0.60
    entailment_confidence: float = 0.60

    # Socratic score weights
    w_s1: float = 0.35   # exploration
    w_s2: float = 0.25   # challenge
    w_s3: float = 0.20   # symbolism
    w_s4: float = 0.30   # pragmatism (penalty)
    w_s5: float = 0.25   # discharge (penalty)
    w_s6: float = 0.25   # defensiveness (penalty)

    # Convergence weights
    w_k1: float = 0.40   # semantic narrowing
    w_k2: float = 0.30   # vocabulary stability
    w_k3: float = 0.30   # proposition strengthening

    # Neurochemical coupling constants
    beta_socratic: float = 0.20      # ACh
    beta_curiosity_socratic: float = 0.06  # tonic DA
    beta_insight: float = 0.25       # phasic DA
    alpha_difficulty: float = 0.10   # DA turn bonus
    rho_collaborative: float = 0.12  # OXT coupling
    gamma_oxt: float = 0.08          # OXT decay
    beta_elenchus: float = 0.12      # NE
    lambda_ne: float = 2.0           # NE Poisson λ
    eta_aporia: float = 0.15         # GABA aporia coupling
    psi_theta_explore: float = 0.12  # Theta exploration
    psi_beta_elenchus: float = 0.10  # Beta elenchus
    delta_insight_coherence: float = 0.10  # Θγ coherence on insight

    # Transition thresholds
    probing_to_elenchus_c: float = 0.50
    probing_to_elenchus_a: float = 0.30
    probing_to_maieutics_k: float = 0.40
    probing_to_maieutics_u: float = 0.20
    elenchus_to_aporia_u: float = 0.40
    aporia_to_exploring_u: float = 0.30
    exploring_to_maieutics_k: float = 0.30
    exploring_to_elenchus_c: float = 0.50

    # Topic depth markers
    procedural_markers: Tuple[str, ...] = ("how do i", "what is the", "how to", "what are the steps")
    opinion_markers: Tuple[str, ...] = ("i think", "i believe", "should", "ought", "must", "perhaps", "maybe")

    # Hedging / uncertainty markers
    hedging_markers: Tuple[str, ...] = ("maybe", "perhaps", "i'm not sure", "i think", "possibly", "might", "could be")
    epistemic_verbs: Tuple[str, ...] = ("believe", "suppose", "wonder", "guess", "assume", "suspect")
    frustration_markers: Tuple[str, ...] = ("just tell me", "what's the answer", "stop asking", "give me", "directly")
    direct_request_markers: Tuple[str, ...] = ("just tell me", "what's the answer", "answer directly", "give me the answer")


# =====================================================================
# Data types
# =====================================================================

@dataclass(frozen=True)
class Assumption:
    """A hidden assumption surfaced from user propositions."""
    assumption_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    source_proposition: str = ""
    confidence: float = 0.5


@dataclass(frozen=True)
class Entailment:
    """A logical consequence generated from a proposition (for elenchus)."""
    entailment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    antecedent: str = ""
    consequent: str = ""
    confidence: float = 0.6
    contradicts_belief: Optional[str] = None


@dataclass(frozen=True)
class SocraticQuestion:
    question_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question_text: str = ""
    question_type: QuestionType = QuestionType.CLARIFICATION
    dialectical_state: DialogueState = DialogueState.PROBING
    target_proposition: Optional[str] = None
    target_assumption: Optional[str] = None
    expected_effect: ExpectedEffect = ExpectedEffect.CLARIFY_AMBIGUITY
    abstraction_direction: int = 0       # -1=grounding, 0=same, +1=ascending
    turn_number: int = 0
    previous_question_id: Optional[str] = None


@dataclass(frozen=True)
class SocraticInsight:
    insight_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    source: InsightSource = InsightSource.USER_ARTICULATED
    starting_position: str = ""
    dialectical_distance: float = 0.0
    question_chain: Tuple[str, ...] = ()
    resolved_paradoxes: Tuple[str, ...] = ()
    turn_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ConvergenceMetrics:
    narrowing: float = 0.0
    stabilization: float = 0.0
    strengthening: float = 0.0
    kappa_overall: float = 0.0
    turns_tracked: int = 0


@dataclass(frozen=True)
class UnsolvedEntry:
    """Minimal stub matching UnsolvedConceptsBuffer entry structure."""
    concept_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    concept_text: str = ""
    attempt_count: int = 0
    motivational_salience: float = 0.5
    accumulated_evidence: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SocraticInput:
    # Activation context
    intention_vector: IntentionVector = field(default_factory=IntentionVector)
    active_mode: OperationalMode = OperationalMode.NORMAL

    # Content
    user_input: ProcessedStatement = field(
        default_factory=ProcessedStatement
    )
    semantic_expansion: Dict[str, List[str]] = field(default_factory=dict)
    user_propositions: List[Proposition] = field(default_factory=list)

    # Detection results
    contradiction_flags: List[ContradictionFlag] = field(default_factory=list)
    paradox_flags: List[ParadoxFlag] = field(default_factory=list)

    # History
    dialogue_history: List[Dict] = field(default_factory=list)
    current_dialogue_state: Optional[DialogueState] = None
    user_beliefs: List[ProcessedStatement] = field(default_factory=list)
    unsolved_buffer: List[UnsolvedEntry] = field(default_factory=list)


@dataclass(frozen=True)
class SocraticOutput:
    # Primary
    generated_question: Optional[SocraticQuestion]
    # State management
    new_dialogue_state: DialogueState
    socratic_active: bool
    yield_to_direct: bool
    yield_reason: Optional[YieldReason]
    # Metadata
    identified_assumptions: List[Assumption]
    identified_entailments: List[Entailment]
    convergence_state: ConvergenceMetrics
    # Insights
    insights: List[SocraticInsight]
    # Processing metadata
    activation_score: float
    topic_depth_score: float
    processing_time_ms: float
    neurochemical_signals: Dict[str, float]


# =====================================================================
# Mutable State
# =====================================================================

@dataclass
class SocraticDialogueState:
    """Persisted per active Socratic sequence (cleared on EXIT)."""
    active: bool = False
    current_state: DialogueState = DialogueState.PROBING
    turn_count: int = 0
    question_history: List[SocraticQuestion] = field(default_factory=list)
    proposition_trajectory: List[List[str]] = field(default_factory=list)
    embedding_trajectory: List[List[float]] = field(default_factory=list)
    assumptions_surfaced: List[Assumption] = field(default_factory=list)
    starting_position: str = ""
    insights_generated: List[SocraticInsight] = field(default_factory=list)
    consecutive_same_state: int = 0
    turns_in_aporia: int = 0
    consecutive_socratic_turns: int = 0
    yield_cooldown: int = 0  # turns before re-engagement


@dataclass
class SocraticEngineState:
    dialogue: SocraticDialogueState = field(default_factory=SocraticDialogueState)
    # Bidirectional neurochem read
    ach_level: float = 0.5
    da_tonic_level: float = 0.5
    oxt_level: float = 0.5
    ne_level: float = 0.5
    gaba_level: float = 0.5


# =====================================================================
# Pure Functions — Activation
# =====================================================================

def compute_socratic_score(intention: IntentionVector, config: SocraticEngineConfig) -> float:
    """
    socratic_score = w_s1×e_exploration + w_s2×e_challenge + w_s3×e_symbolism
                   - w_s4×e_pragmatism - w_s5×e_discharge - w_s6×e_defensiveness
    Clamped [0, 1].

    IntentionVector has direct fields: e_defensiveness, e_challenge.
    Extended fields (e_exploration, e_symbolism, e_pragmatism, e_discharge) are
    stored in intent_scores dict if present.
    """
    scores = intention.intent_scores if hasattr(intention, "intent_scores") else {}
    e_exploration = float(scores.get("e_exploration", 0.0))
    e_symbolism   = float(scores.get("e_symbolism",   0.0))
    e_pragmatism  = float(scores.get("e_pragmatism",  0.0))
    e_discharge   = float(scores.get("e_discharge",   0.0))
    e_challenge    = float(getattr(intention, "e_challenge",    0.0))
    e_defensiveness = float(getattr(intention, "e_defensiveness", 0.0))

    score = (
        config.w_s1 * e_exploration
        + config.w_s2 * e_challenge
        + config.w_s3 * e_symbolism
        - config.w_s4 * e_pragmatism
        - config.w_s5 * e_discharge
        - config.w_s6 * e_defensiveness
    )
    return float(np.clip(score, 0.0, 1.0))


def compute_topic_depth(text: str, config: SocraticEngineConfig) -> float:
    """
    Estimate topic depth:
      - Average hypernym chain depth proxy: avg token length (normalized)
      - Presence of opinion/belief markers
      - Absence of procedural markers
    Returns value in [0, 1].
    """
    text_lower = text.lower()

    # Procedural penalty
    procedural_hit = any(m in text_lower for m in config.procedural_markers)
    if procedural_hit:
        return 0.10

    tokens = text_lower.split()
    if not tokens:
        return 0.0

    # Token length proxy for lexical depth (longer words → more abstract)
    avg_len = sum(len(t) for t in tokens) / len(tokens)
    depth_lex = min(avg_len / 10.0, 1.0)

    # Opinion/belief marker bonus
    opinion_count = sum(1 for m in config.opinion_markers if m in text_lower)
    depth_opinion = min(opinion_count * 0.15, 0.45)

    # Sentence length proxy (longer → more complex)
    depth_length = min(len(tokens) / 40.0, 0.30)

    return float(np.clip(depth_lex * 0.40 + depth_opinion + depth_length, 0.0, 1.0))


def mode_allows_socratic(mode: OperationalMode) -> bool:
    """Gate 1: Only activate in non-dev, non-dream modes."""
    return mode in (
        OperationalMode.NORMAL,
        OperationalMode.LEARNING,
        OperationalMode.REFLECTIVE,
        OperationalMode.REM_NORMAL,
    )


def check_all_activation_gates(
    socratic_input: SocraticInput,
    dialogue: SocraticDialogueState,
    config: SocraticEngineConfig,
    neurochem: SocraticEngineState,
) -> Tuple[bool, float, float]:
    """
    Returns (activated, activation_score, topic_depth_score).
    Checks mode, intention, topic, and fatigue gates.
    """
    if dialogue.yield_cooldown > 0:
        return False, 0.0, 0.0

    # Gate 1: mode
    if not mode_allows_socratic(socratic_input.active_mode):
        return False, 0.0, 0.0

    # Neurochem-modulated threshold: low OXT → raise θ_socratic; high DA tonic → lower it
    effective_theta = config.theta_socratic
    effective_theta += 0.05 * (0.5 - neurochem.oxt_level)   # low OXT → harder to activate
    effective_theta -= 0.04 * (neurochem.da_tonic_level - 0.5)
    effective_theta = float(np.clip(effective_theta, 0.10, 0.80))

    # Gate 2: intention
    activation_score = compute_socratic_score(socratic_input.intention_vector, config)
    if activation_score <= effective_theta:
        return False, activation_score, 0.0

    # Gate 3: topic depth
    topic_text = socratic_input.user_input.raw_text
    topic_depth = compute_topic_depth(topic_text, config)
    if topic_depth <= config.theta_depth:
        return False, activation_score, topic_depth

    # Neurochem-modulated max turns: high ACh → sustain longer
    effective_max_turns = config.max_socratic_turns + round(2 * (neurochem.ach_level - 0.5))
    effective_max_turns = max(2, effective_max_turns)

    # Gate 4: fatigue
    if dialogue.consecutive_socratic_turns >= effective_max_turns:
        return False, activation_score, topic_depth

    return True, activation_score, topic_depth


# =====================================================================
# Pure Functions — Signal Features
# =====================================================================

def compute_contradiction_signal(contradiction_flags: List[ContradictionFlag]) -> float:
    """c(t) = max(contradiction_flags.confidence) if flags exist, else 0."""
    if not contradiction_flags:
        return 0.0
    return float(np.clip(max(f.confidence for f in contradiction_flags), 0.0, 1.0))


def compute_assumption_signal(
    propositions: List[Proposition],
    assumptions: List[Assumption],
) -> float:
    """a(t) = count(implicit_assumptions) / total_propositions."""
    total = len(propositions)
    if total == 0:
        return 0.0
    return float(np.clip(len(assumptions) / total, 0.0, 1.0))


def compute_uncertainty_signal(text: str, config: SocraticEngineConfig) -> float:
    """
    u(t) = (hedging_markers + question_marks + epistemic_verbs) / total_tokens.
    Clamped [0, 1].
    """
    if not text:
        return 0.0
    text_lower = text.lower()
    tokens = text_lower.split()
    if not tokens:
        return 0.0

    hedging_count = sum(1 for m in config.hedging_markers if m in text_lower)
    epistemic_count = sum(1 for v in config.epistemic_verbs if v in text_lower)
    question_marks = text.count("?")
    signal_sum = hedging_count + epistemic_count + question_marks
    return float(np.clip(signal_sum / len(tokens), 0.0, 1.0))


def compute_frustration_signal(
    text: str,
    response_length: int,
    expected_response_length: int,
    config: SocraticEngineConfig,
) -> float:
    """
    f(t) = 0.40×brevity + 0.35×direct_request + 0.25×discord_placeholder
    Brevity = 1 - (response_length / expected_response_length)
    """
    text_lower = text.lower()
    direct_request = 1.0 if any(m in text_lower for m in config.direct_request_markers) else 0.0
    if expected_response_length > 0:
        brevity = 1.0 - min(response_length / expected_response_length, 1.0)
    else:
        brevity = 0.0
    # Frustration markers as additional signal
    frustration_hit = any(m in text_lower for m in config.frustration_markers)
    discord_proxy = 0.5 if frustration_hit else 0.0
    f = 0.40 * brevity + 0.35 * direct_request + 0.25 * discord_proxy
    return float(np.clip(f, 0.0, 1.0))


def compute_convergence_metrics(
    proposition_trajectory: List[List[str]],
    config: SocraticEngineConfig,
) -> ConvergenceMetrics:
    """
    Requires ≥ 3 turns of proposition trajectory.
    Returns ConvergenceMetrics with kappa_overall.
    """
    if len(proposition_trajectory) < 3:
        return ConvergenceMetrics(turns_tracked=len(proposition_trajectory))

    # Use last 3 turns
    t2 = set(proposition_trajectory[-1])
    t1 = set(proposition_trajectory[-2])
    t0 = set(proposition_trajectory[-3])

    # Jaccard-based vocabulary stabilization between t-1 and t
    union_12 = t1 | t2
    inter_12 = t1 & t2
    stabilization = len(inter_12) / len(union_12) if union_12 else 0.0

    # Semantic narrowing proxy: proportion of t-2 words retained in t-1, then t
    dist_01 = len(t0.symmetric_difference(t1))
    dist_12 = len(t1.symmetric_difference(t2))
    if dist_01 > 0:
        narrowing = 1.0 - (dist_12 / dist_01)
    else:
        narrowing = 1.0 if dist_12 == 0 else 0.0
    narrowing = float(np.clip(narrowing, -1.0, 1.0))

    # Proposition strengthening proxy: cannot compute u(t) directly here; stub = 0
    strengthening = 0.0

    kappa = (
        config.w_k1 * max(narrowing, 0.0)
        + config.w_k2 * stabilization
        + config.w_k3 * strengthening
    )
    kappa = float(np.clip(kappa, 0.0, 1.0))
    return ConvergenceMetrics(
        narrowing=float(narrowing),
        stabilization=float(stabilization),
        strengthening=float(strengthening),
        kappa_overall=kappa,
        turns_tracked=len(proposition_trajectory),
    )


# =====================================================================
# Pure Functions — Assumption Detection
# =====================================================================

_ASSUMPTION_PATTERNS: List[Tuple[str, str]] = [
    # (regex pattern, description template)
    (r"\ball\b", "assumes universal scope over all instances"),
    (r"\bnever\b", "assumes universal negative"),
    (r"\balways\b", "assumes invariant recurrence"),
    (r"\beveryone\b", "assumes universal human scope"),
    (r"\bnobody\b|\bno one\b", "assumes universal absence"),
    (r"\bobviously\b|\bclearly\b", "assumes shared understanding without justification"),
    (r"\bof course\b", "assumes given self-evidence"),
    (r"\bjust\b|\bsimply\b", "assumes low complexity or effort"),
    (r"\bnaturally\b", "assumes normative alignment"),
    (r"\binherently\b", "assumes essential property"),
]


def detect_implicit_assumptions(
    text: str,
    propositions: List[Proposition],
) -> List[Assumption]:
    """
    Detect hidden assumptions via pattern matching on the text.
    Returns list of Assumption objects.
    """
    assumptions: List[Assumption] = []
    text_lower = text.lower()
    for pattern, desc in _ASSUMPTION_PATTERNS:
        if re.search(pattern, text_lower):
            # Try to associate with a proposition
            source_prop = propositions[0].text if propositions else text[:80]
            assumptions.append(Assumption(
                text=desc,
                source_proposition=source_prop[:200],
                confidence=0.60,
            ))
    return assumptions


def generate_entailments_from_propositions(
    propositions: List[Proposition],
    user_beliefs: List[ProcessedStatement],
    contradiction_flags: List[ContradictionFlag],
) -> List[Entailment]:
    """
    Phase 1 (template): Generate simple modus-ponens-style entailments for elenchus.
    Checks if any entailment contradicts known beliefs.
    """
    entailments: List[Entailment] = []
    for prop in propositions[:3]:  # cap to avoid explosion
        text = prop.text.strip()
        if not text:
            continue
        # Simple forward entailment: if P → then other conditions hold
        consequent = f"the opposite could also hold under different conditions"
        # Check contradiction flags for conflicts
        contradicts = None
        if contradiction_flags:
            contradicts = contradiction_flags[0].semantic_description
        entailments.append(Entailment(
            antecedent=text[:150],
            consequent=consequent,
            confidence=0.65,
            contradicts_belief=contradicts,
        ))
    return entailments


# =====================================================================
# Pure Functions — Question Generation
# =====================================================================

# Templates for all 18 (+ 5 internal) question types
_QUESTION_TEMPLATES: Dict[QuestionType, List[Tuple[str, ExpectedEffect, int]]] = {
    # (template, expected_effect, abstraction_direction)
    QuestionType.CLARIFICATION: [
        ("What exactly do you mean by '{target}'?", ExpectedEffect.CLARIFY_AMBIGUITY, 0),
        ("Could you unpack what '{target}' refers to in this context?", ExpectedEffect.CLARIFY_AMBIGUITY, 0),
        ("How are you using the term '{target}' here?", ExpectedEffect.CLARIFY_AMBIGUITY, 0),
    ],
    QuestionType.FOUNDATIONAL: [
        ("What leads you to believe that?", ExpectedEffect.EXPOSE_ASSUMPTION, 0),
        ("What's the basis for that claim?", ExpectedEffect.EXPOSE_ASSUMPTION, 0),
        ("What evidence or reasoning supports that?", ExpectedEffect.EXPOSE_ASSUMPTION, 0),
    ],
    QuestionType.DEFINITIONAL: [
        ("How would you define '{target}'?", ExpectedEffect.CLARIFY_AMBIGUITY, 0),
        ("What would count as a clear example of '{target}'?", ExpectedEffect.CLARIFY_AMBIGUITY, -1),
        ("What distinguishes '{target}' from related concepts?", ExpectedEffect.CLARIFY_AMBIGUITY, 0),
    ],
    QuestionType.SCOPE: [
        ("Does that apply in every case, or are there exceptions?", ExpectedEffect.EXPOSE_ASSUMPTION, 0),
        ("How broadly do you think this holds — always, usually, or sometimes?", ExpectedEffect.EXPOSE_ASSUMPTION, 0),
        ("Are there situations where this wouldn't be true?", ExpectedEffect.TEST_ROBUSTNESS, 0),
    ],
    QuestionType.IMPLICATIVE: [
        ("If that's the case, wouldn't it follow that '{consequent}'?", ExpectedEffect.SURFACE_CONTRADICTION, +1),
        ("What would that imply about '{consequent}'?", ExpectedEffect.SURFACE_CONTRADICTION, +1),
        ("If '{target}' is true, what else must be true?", ExpectedEffect.SURFACE_CONTRADICTION, +1),
    ],
    QuestionType.COUNTER_CASE: [
        ("What about a situation where '{target}' leads to the opposite outcome?", ExpectedEffect.SURFACE_CONTRADICTION, 0),
        ("Can you think of a case where this reasoning breaks down?", ExpectedEffect.SURFACE_CONTRADICTION, 0),
        ("How would you account for scenarios where '{target}' doesn't hold?", ExpectedEffect.SURFACE_CONTRADICTION, 0),
    ],
    QuestionType.CONSISTENCY: [
        ("How does that fit with what you said earlier?", ExpectedEffect.SURFACE_CONTRADICTION, 0),
        ("That seems to sit in tension with your earlier point — how do you reconcile the two?", ExpectedEffect.SURFACE_CONTRADICTION, 0),
        ("Do you see a tension between that and '{target}'?", ExpectedEffect.SURFACE_CONTRADICTION, 0),
    ],
    QuestionType.REFRAMING: [
        ("What if we looked at this from a completely different angle?", ExpectedEffect.BROADEN_SEARCH, 0),
        ("Is there another way to frame what you're experiencing here?", ExpectedEffect.BROADEN_SEARCH, 0),
        ("What would this look like from the opposing perspective?", ExpectedEffect.BROADEN_SEARCH, 0),
    ],
    QuestionType.ANALOGICAL: [
        ("Where else have you seen a similar tension play out?", ExpectedEffect.BROADEN_SEARCH, 0),
        ("Does this remind you of a pattern in a different domain?", ExpectedEffect.BROADEN_SEARCH, +1),
        ("Is there an analogy that might illuminate this?", ExpectedEffect.BROADEN_SEARCH, +1),
    ],
    QuestionType.ABSTRACTING: [
        ("What's the deeper principle underneath this specific case?", ExpectedEffect.DEEPEN_ANALYSIS, +1),
        ("If you had to generalize from this, what would you say?", ExpectedEffect.DEEPEN_ANALYSIS, +1),
        ("What does this tell us at a more fundamental level?", ExpectedEffect.DEEPEN_ANALYSIS, +1),
    ],
    QuestionType.GROUNDING: [
        ("Can you give me a concrete example of what you mean?", ExpectedEffect.CLARIFY_AMBIGUITY, -1),
        ("What would that look like in practice?", ExpectedEffect.CLARIFY_AMBIGUITY, -1),
        ("What's a specific case that illustrates this?", ExpectedEffect.CLARIFY_AMBIGUITY, -1),
    ],
    QuestionType.EXTENDING: [
        ("What would follow from that idea if you pushed it further?", ExpectedEffect.DEEPEN_ANALYSIS, +1),
        ("Where does that reasoning lead if you take it to its conclusion?", ExpectedEffect.DEEPEN_ANALYSIS, +1),
        ("What are the downstream implications of that?", ExpectedEffect.DEEPEN_ANALYSIS, +1),
    ],
    QuestionType.CONNECTING: [
        ("How does that relate to what you said about '{target}'?", ExpectedEffect.BROADEN_SEARCH, 0),
        ("Is there a connection between this and '{target}'?", ExpectedEffect.BROADEN_SEARCH, 0),
        ("How does this fit into the broader picture we've been building?", ExpectedEffect.BROADEN_SEARCH, 0),
    ],
    QuestionType.TESTING: [
        ("Is there a case where this wouldn't hold?", ExpectedEffect.TEST_ROBUSTNESS, 0),
        ("What would have to be true for this to be wrong?", ExpectedEffect.TEST_ROBUSTNESS, 0),
        ("How would you stress-test that idea?", ExpectedEffect.TEST_ROBUSTNESS, 0),
    ],
    QuestionType.CRYSTALLIZING: [
        ("So would you say that '{target}'?", ExpectedEffect.CRYSTALLIZE_INSIGHT, 0),
        ("It sounds like you're arriving at — is that right?", ExpectedEffect.CRYSTALLIZE_INSIGHT, 0),
        ("Are you saying that the core insight here is '{target}'?", ExpectedEffect.CRYSTALLIZE_INSIGHT, 0),
    ],
    QuestionType.NAMING: [
        ("What would you call this principle you've just articulated?", ExpectedEffect.CRYSTALLIZE_INSIGHT, +1),
        ("Is there a name or concept that captures what you're describing?", ExpectedEffect.CRYSTALLIZE_INSIGHT, +1),
        ("How would you label this idea so you could recognize it again?", ExpectedEffect.CRYSTALLIZE_INSIGHT, +1),
    ],
    QuestionType.INTEGRATING: [
        ("How does this change your original view?", ExpectedEffect.INTEGRATE_INSIGHT, 0),
        ("What shifts in your thinking after working through this?", ExpectedEffect.INTEGRATE_INSIGHT, 0),
        ("How does this insight connect back to where we started?", ExpectedEffect.INTEGRATE_INSIGHT, 0),
    ],
    QuestionType.APPLYING: [
        ("Where else might this insight apply?", ExpectedEffect.INTEGRATE_INSIGHT, +1),
        ("What would change if you applied this reasoning to '{target}'?", ExpectedEffect.INTEGRATE_INSIGHT, +1),
        ("How could you put this understanding to use?", ExpectedEffect.INTEGRATE_INSIGHT, 0),
    ],
    # Internal types
    QuestionType.FALSIFICATION: [
        ("What evidence would disprove this conclusion?", ExpectedEffect.TEST_ROBUSTNESS, 0),
        ("Under what conditions would this stored belief be false?", ExpectedEffect.TEST_ROBUSTNESS, 0),
    ],
    QuestionType.PROVENANCE: [
        ("Where did I learn this, and was the source reliable?", ExpectedEffect.EXPOSE_ASSUMPTION, 0),
        ("What was the origin of this belief and how was it validated?", ExpectedEffect.EXPOSE_ASSUMPTION, 0),
    ],
    QuestionType.ALTERNATIVE: [
        ("What other explanation fits the same evidence?", ExpectedEffect.BROADEN_SEARCH, 0),
        ("Is there an alternative interpretation that also accounts for these observations?", ExpectedEffect.BROADEN_SEARCH, 0),
    ],
    QuestionType.DEPENDENCY: [
        ("What does this conclusion depend on?", ExpectedEffect.EXPOSE_ASSUMPTION, +1),
        ("Which assumptions does this reasoning chain rest on?", ExpectedEffect.EXPOSE_ASSUMPTION, +1),
    ],
    QuestionType.STABILITY: [
        ("Would I reach the same conclusion today given what I now know?", ExpectedEffect.TEST_ROBUSTNESS, 0),
        ("Has the evidence base changed enough to revisit this?", ExpectedEffect.TEST_ROBUSTNESS, 0),
    ],
}

# State → preferred question types (ordered by preference)
_STATE_QUESTION_TYPES: Dict[DialogueState, List[QuestionType]] = {
    DialogueState.PROBING: [
        QuestionType.CLARIFICATION, QuestionType.FOUNDATIONAL,
        QuestionType.DEFINITIONAL, QuestionType.SCOPE,
    ],
    DialogueState.ELENCHUS: [
        QuestionType.IMPLICATIVE, QuestionType.COUNTER_CASE, QuestionType.CONSISTENCY,
    ],
    DialogueState.APORIA: [
        QuestionType.REFRAMING, QuestionType.ANALOGICAL, QuestionType.ABSTRACTING,
    ],
    DialogueState.EXPLORING: [
        QuestionType.GROUNDING, QuestionType.EXTENDING,
        QuestionType.CONNECTING, QuestionType.TESTING,
        QuestionType.ANALOGICAL,
    ],
    DialogueState.MAIEUTICS: [
        QuestionType.CRYSTALLIZING, QuestionType.NAMING,
        QuestionType.INTEGRATING, QuestionType.APPLYING,
    ],
    DialogueState.EXIT: [],
}


def select_target_proposition(
    state: DialogueState,
    propositions: List[Proposition],
    contradiction_flags: List[ContradictionFlag],
    assumptions: List[Assumption],
    convergence: ConvergenceMetrics,
) -> Optional[str]:
    """Step 1 of question generation: select the target proposition."""
    if state == DialogueState.PROBING:
        # most ambiguous = shortest proposition (proxy for least-elaborated)
        if propositions:
            return min(propositions, key=lambda p: len(p.text)).text
        return None
    elif state == DialogueState.ELENCHUS:
        # strongest contradiction
        if contradiction_flags:
            return contradiction_flags[0].semantic_description
        if assumptions:
            return assumptions[0].text
        return None
    elif state == DialogueState.APORIA:
        return None  # open, not targeted
    elif state == DialogueState.EXPLORING:
        if propositions:
            return max(propositions, key=lambda p: len(p.text)).text
        return None
    elif state == DialogueState.MAIEUTICS:
        if propositions:
            return propositions[-1].text
        return None
    return None


def select_question_type(
    state: DialogueState,
    history: List[SocraticQuestion],
    config: SocraticEngineConfig,
) -> QuestionType:
    """Step 2: Select question type, avoiding recent repeats."""
    candidates = _STATE_QUESTION_TYPES.get(state, [QuestionType.CLARIFICATION])
    if not candidates:
        return QuestionType.CLARIFICATION

    recent_types = {q.question_type for q in history[-3:]}
    # Prefer types not recently used
    for qt in candidates:
        if qt not in recent_types:
            return qt
    return candidates[0]  # fallback to first if all recently used


def formulate_question(
    qt: QuestionType,
    state: DialogueState,
    target: Optional[str],
    turn_number: int,
    previous_id: Optional[str],
    rng: np.random.Generator,
) -> SocraticQuestion:
    """Step 3: Fill template with target, validate."""
    templates = _QUESTION_TEMPLATES.get(qt, [])
    if not templates:
        text = "Can you say more about that?"
        return SocraticQuestion(
            question_text=text,
            question_type=qt,
            dialectical_state=state,
            expected_effect=ExpectedEffect.CLARIFY_AMBIGUITY,
            abstraction_direction=0,
            turn_number=turn_number,
            previous_question_id=previous_id,
        )

    tmpl, effect, abst = templates[int(rng.integers(0, len(templates)))]
    target_str = target or "that"
    # For entailment-based types, we'd fill {consequent} — use a default for Phase 1
    text = tmpl.replace("{target}", target_str).replace("{consequent}", "something else must also be true")

    # Validation: ensure it ends with "?" or is a question
    if not text.endswith("?"):
        text = text.rstrip(".") + "?"

    return SocraticQuestion(
        question_text=text,
        question_type=qt,
        dialectical_state=state,
        target_proposition=target,
        expected_effect=effect,
        abstraction_direction=abst,
        turn_number=turn_number,
        previous_question_id=previous_id,
    )


# =====================================================================
# Pure Functions — State Transitions
# =====================================================================

def compute_next_state(
    current: DialogueState,
    c: float,
    a: float,
    u: float,
    kappa: ConvergenceMetrics,
    f: float,
    turns_in_aporia: int,
    new_propositions: bool,
    config: SocraticEngineConfig,
) -> Tuple[DialogueState, Optional[YieldReason]]:
    """
    Apply transition table from Appendix B.
    Returns (next_state, yield_reason if EXIT).
    """
    k = kappa.kappa_overall

    if current == DialogueState.PROBING:
        if f >= config.theta_frustration:
            return DialogueState.EXIT, YieldReason.FRUSTRATION
        if c > config.probing_to_elenchus_c or a > config.probing_to_elenchus_a:
            return DialogueState.ELENCHUS, None
        if k < config.probing_to_maieutics_k and u < config.probing_to_maieutics_u and kappa.turns_tracked >= 3:
            return DialogueState.MAIEUTICS, None
        return DialogueState.PROBING, None

    elif current == DialogueState.ELENCHUS:
        if f >= config.theta_frustration:
            return DialogueState.EXIT, YieldReason.FRUSTRATION
        if u > config.elenchus_to_aporia_u:
            return DialogueState.APORIA, None
        if c <= 0.2:  # contradiction trivially resolved
            return DialogueState.PROBING, None
        return DialogueState.ELENCHUS, None

    elif current == DialogueState.APORIA:
        if turns_in_aporia > config.max_aporia_turns:
            return DialogueState.EXIT, YieldReason.APORIA_LIMIT
        if new_propositions and u > config.aporia_to_exploring_u:
            return DialogueState.EXPLORING, None
        if k < config.probing_to_maieutics_k and kappa.turns_tracked >= 3:
            return DialogueState.MAIEUTICS, None
        return DialogueState.APORIA, None

    elif current == DialogueState.EXPLORING:
        if f >= config.theta_frustration:
            return DialogueState.EXIT, YieldReason.FRUSTRATION
        if k < config.exploring_to_maieutics_k and kappa.turns_tracked >= 3:
            return DialogueState.MAIEUTICS, None
        if c > config.exploring_to_elenchus_c:
            return DialogueState.ELENCHUS, None
        return DialogueState.EXPLORING, None

    elif current == DialogueState.MAIEUTICS:
        if f >= config.theta_frustration:
            return DialogueState.EXIT, YieldReason.FRUSTRATION
        # Insight crystallized: check if kappa is very high and low uncertainty
        if k >= config.theta_maieutics and u < 0.15 and kappa.turns_tracked >= 3:
            return DialogueState.EXIT, YieldReason.INSIGHT_ACHIEVED
        if k < 0.30 and kappa.turns_tracked >= 3:
            return DialogueState.EXPLORING, None
        return DialogueState.MAIEUTICS, None

    return DialogueState.EXIT, YieldReason.TOPIC_MISMATCH


# =====================================================================
# Pure Functions — Insight Extraction
# =====================================================================

def extract_insight(
    propositions: List[Proposition],
    dialogue: SocraticDialogueState,
    paradox_flags: List[ParadoxFlag],
    source: InsightSource = InsightSource.USER_ARTICULATED,
) -> Optional[SocraticInsight]:
    """
    If convergence is high and propositions are present, extract insight.
    """
    if not propositions:
        return None

    content = propositions[-1].text if propositions else ""
    if not content:
        return None

    # Compute dialectical distance: semantic distance proxy via word set diff
    starting = set(dialogue.starting_position.lower().split())
    current_words = set(content.lower().split())
    union_size = len(starting | current_words)
    inter_size = len(starting & current_words)
    distance = 1.0 - (inter_size / union_size) if union_size > 0 else 0.0

    # Check paradox resolution
    resolved = tuple(str(p.paradox_id) for p in paradox_flags if p.symbolic_tension_score >= 0.6)

    question_chain = tuple(q.question_id for q in dialogue.question_history)

    return SocraticInsight(
        content=content,
        source=source,
        starting_position=dialogue.starting_position,
        dialectical_distance=float(distance),
        question_chain=question_chain,
        resolved_paradoxes=resolved,
        turn_count=dialogue.turn_count,
    )


# =====================================================================
# Pure Functions — Neurochemical Coupling
# =====================================================================

def compute_socratic_processing_signal(
    turn_count: int,
    max_turns: int,
    assumptions: List[Assumption],
    total_propositions: int,
    paradox_flags: List[ParadoxFlag],
    config: SocraticEngineConfig,
) -> float:
    """
    Σ(t) = w_Σ1×dialectical_depth + w_Σ2×assumption_exposure_rate + w_Σ3×productive_tension
    """
    dialectical_depth = min(turn_count / max(max_turns, 1), 1.0)
    assumption_rate = (len(assumptions) / max(total_propositions, 1))
    productive_tension = (
        max((p.symbolic_tension_score for p in paradox_flags), default=0.0)
        if paradox_flags else 0.0
    )
    sigma = 0.30 * dialectical_depth + 0.35 * assumption_rate + 0.35 * productive_tension
    return float(np.clip(sigma, 0.0, 1.0))


def compute_neurochemical_signals(
    state: DialogueState,
    sigma: float,
    kappa: ConvergenceMetrics,
    c: float,
    u: float,
    engagement: float,
    f: float,
    insight_generated: bool,
    dialectical_distance: float,
    turn_count: int,
    config: SocraticEngineConfig,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Compute all neurochemical deltas for one Socratic turn.
    Returns dict with signal keys matching engine apply_feedback interface.
    """
    signals: Dict[str, float] = {
        "ach_burst": 0.0,
        "da_tonic": 0.0,
        "da_phasic": 0.0,
        "oxt_drift": 0.0,
        "ne_burst": 0.0,
        "gaba_suppress": 0.0,
        "theta_boost": 0.0,
        "beta_boost": 0.0,
        "theta_gamma_boost": 0.0,
    }

    if sigma <= 0.0 and not insight_generated:
        return signals

    # ACh — sustained attention (Gamma burst)
    if sigma > 0.0:
        ach = config.beta_socratic * sigma * float(rng.gamma(2.5, 0.4))
        signals["ach_burst"] = float(np.clip(ach, 0.0, 1.0))

    # DA tonic — curiosity (inversely modulated by convergence)
    anti_convergence = 1.0 - kappa.kappa_overall
    signals["da_tonic"] = float(np.clip(
        config.beta_curiosity_socratic * sigma * anti_convergence, 0.0, 1.0
    ))

    # DA phasic — insight reward (event-driven)
    if insight_generated and dialectical_distance > 0.0:
        difficulty_bonus = 1.0 + config.alpha_difficulty * turn_count
        da_phasic = (
            config.beta_insight
            * dialectical_distance
            * difficulty_bonus
            * float(rng.gamma(3.0, 0.5))
        )
        signals["da_phasic"] = float(np.clip(da_phasic, 0.0, 2.0))

    # OXT — collaborative engagement
    if f < 0.5:
        # Positive drift: engagement
        oxt_drift = config.rho_collaborative * engagement
        signals["oxt_drift"] = float(np.clip(oxt_drift, 0.0, 0.5))
    else:
        # Frustration: OXT suppression
        signals["oxt_drift"] = float(np.clip(-0.10 * f, -0.5, 0.0))

    # NE — elenchus vigilance (Poisson burst)
    if state == DialogueState.ELENCHUS and c > 0.0:
        ne_count = rng.poisson(config.lambda_ne)
        ne = config.beta_elenchus * c * float(ne_count) / config.lambda_ne
        signals["ne_burst"] = float(np.clip(ne, 0.0, 1.0))

    # GABA — aporia regulation (reuptake suppression)
    if state == DialogueState.APORIA and u > 0.0:
        signals["gaba_suppress"] = float(np.clip(
            config.eta_aporia * u, 0.0, 0.5
        ))

    # Theta — exploration / aporia
    if state in (DialogueState.EXPLORING, DialogueState.APORIA):
        signals["theta_boost"] = float(np.clip(
            config.psi_theta_explore * sigma, 0.0, 0.5
        ))

    # Beta — elenchus focused analysis
    if state == DialogueState.ELENCHUS and c > 0.0:
        signals["beta_boost"] = float(np.clip(
            config.psi_beta_elenchus * c, 0.0, 0.5
        ))

    # Θγ coherence — insight integration
    if insight_generated:
        signals["theta_gamma_boost"] = config.delta_insight_coherence

    return signals


def apply_neurochem_feedback(
    config: SocraticEngineConfig,
    neurochem: SocraticEngineState,
) -> SocraticEngineConfig:
    """
    Bidirectional: neurochem state modulates config thresholds.
    Returns adjusted effective config (does not mutate passed config).
    """
    # High ACh → sustain longer (handled in activation gates inline)
    # High DA tonic → lower θ_socratic (lowers activation barrier)
    # Low OXT → raise θ_socratic
    # High NE → elenchus more targeted; handled in signal computation
    # High GABA → questions gentler (handled in question type selection)
    # We return the same config object (thresholds adjusted inline in activation)
    return config


# =====================================================================
# Pure Functions — Internal Self-Inquiry
# =====================================================================

def generate_internal_questions(
    unsolved: List[UnsolvedEntry],
    rng: np.random.Generator,
    config: SocraticEngineConfig,
) -> List[SocraticQuestion]:
    """
    Generate internal Socratic questions for REM processing.
    Selects concept with highest motivational salience.
    """
    if not unsolved:
        return []

    target_entry = max(unsolved, key=lambda e: e.motivational_salience)
    concept = target_entry.concept_text or "this unresolved concept"

    internal_types = [
        QuestionType.FALSIFICATION,
        QuestionType.PROVENANCE,
        QuestionType.ALTERNATIVE,
        QuestionType.DEPENDENCY,
        QuestionType.STABILITY,
    ]
    questions: List[SocraticQuestion] = []
    for qt in internal_types:
        templates = _QUESTION_TEMPLATES.get(qt, [])
        if templates:
            tmpl, effect, abst = templates[0]
            text = tmpl.replace("{target}", concept).replace("{consequent}", concept)
            if not text.endswith("?"):
                text = text.rstrip(".") + "?"
            questions.append(SocraticQuestion(
                question_text=text,
                question_type=qt,
                dialectical_state=DialogueState.PROBING,
                target_proposition=concept,
                expected_effect=effect,
                abstraction_direction=abst,
                turn_number=0,
            ))
    return questions


# =====================================================================
# Engine
# =====================================================================

class SocraticReasoningEngine:
    """
    Socratic Reasoning Engine — Engine 14 (Reasoning Cluster).

    Public API:
        configure(mode)                         — set operational mode
        update_neurochem_state(state_dict)       — update read-port NT levels
        process(input_: SocraticInput)           — main per-turn call
        get_status()                             — engine state snapshot
    """

    engine_id = "socratic_reasoning_engine"
    cluster   = "dialectic"

    def __init__(
        self,
        config: SocraticEngineConfig = SocraticEngineConfig(),
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._config = config
        self._mode = OperationalMode.NORMAL
        self._rng = rng if rng is not None else np.random.default_rng()
        self._state = SocraticEngineState()

    # ------------------------------------------------------------------
    # Configuration port
    # ------------------------------------------------------------------
    def configure(self, mode: OperationalMode) -> None:
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """Update NT read-port levels from pipeline dict."""
        if "ach" in state_dict:
            self._state.ach_level = _clamp(state_dict["ach"])
        if "da" in state_dict:
            self._state.da_tonic_level = _clamp(state_dict["da"])
        if "oxt" in state_dict:
            self._state.oxt_level = _clamp(state_dict["oxt"])
        if "ne" in state_dict:
            self._state.ne_level = _clamp(state_dict["ne"])
        if "gaba" in state_dict:
            self._state.gaba_level = _clamp(state_dict["gaba"])

    # ------------------------------------------------------------------
    # Main process port
    # ------------------------------------------------------------------
    def process(self, input_: SocraticInput) -> SocraticOutput:
        t0 = time.perf_counter()

        dialogue = self._state.dialogue

        # Decrement yield cooldown
        if dialogue.yield_cooldown > 0:
            dialogue.yield_cooldown -= 1

        # Override mode from engine state (not input, for consistency with other engines)
        effective_mode = self._mode

        # Remap input mode if provided (input mode takes priority for this turn)
        if input_.active_mode != OperationalMode.NORMAL:
            effective_mode = input_.active_mode

        # Handle internal inquiry for REM_NORMAL/REFLECTIVE
        if effective_mode in (OperationalMode.REM_NORMAL, OperationalMode.REFLECTIVE):
            if input_.unsolved_buffer:
                return self._run_internal_inquiry(input_, t0)

        # Activation gates
        activated, activation_score, topic_depth = check_all_activation_gates(
            input_, dialogue, self._config, self._state
        )

        if not activated:
            # Not active — yield to direct response
            return self._yield_output(
                activation_score=activation_score,
                topic_depth=topic_depth,
                reason=YieldReason.FATIGUE if dialogue.yield_cooldown == 0 and dialogue.consecutive_socratic_turns >= self._config.max_socratic_turns else YieldReason.TOPIC_MISMATCH,
                t0=t0,
            )

        # Initialize or continue dialogue session
        if not dialogue.active:
            dialogue.active = True
            dialogue.current_state = DialogueState.PROBING
            dialogue.turn_count = 0
            dialogue.question_history = []
            dialogue.proposition_trajectory = []
            dialogue.embedding_trajectory = []
            dialogue.assumptions_surfaced = []
            dialogue.insights_generated = []
            dialogue.consecutive_same_state = 0
            dialogue.turns_in_aporia = 0
            dialogue.starting_position = input_.user_input.raw_text[:500]

        dialogue.turn_count += 1
        dialogue.consecutive_socratic_turns += 1

        # Extract propositions from input
        propositions = input_.user_propositions
        text = input_.user_input.raw_text

        # Update proposition trajectory
        prop_words = [w for p in propositions for w in p.text.lower().split()]
        if prop_words:
            dialogue.proposition_trajectory.append(prop_words)

        # Compute transition features
        c = compute_contradiction_signal(input_.contradiction_flags)
        u = compute_uncertainty_signal(text, self._config)

        # Detect assumptions
        new_assumptions = detect_implicit_assumptions(text, propositions)
        for a_item in new_assumptions:
            dialogue.assumptions_surfaced.append(a_item)

        a_signal = compute_assumption_signal(propositions, dialogue.assumptions_surfaced)

        # Convergence
        convergence = compute_convergence_metrics(dialogue.proposition_trajectory, self._config)

        # Frustration
        expected_len = max(
            sum(len(turn.get("text", "").split()) for turn in input_.dialogue_history[-3:]) // max(len(input_.dialogue_history[-3:]), 1),
            20,
        )
        f = compute_frustration_signal(text, len(text.split()), expected_len, self._config)

        # New propositions signal (for APORIA → EXPLORING)
        new_propositions = bool(propositions)

        # State transition
        prev_state = dialogue.current_state
        next_state, yield_reason = compute_next_state(
            prev_state, c, a_signal, u, convergence, f,
            dialogue.turns_in_aporia, new_propositions, self._config
        )

        # Update aporia counter
        if next_state == DialogueState.APORIA:
            if prev_state == DialogueState.APORIA:
                dialogue.turns_in_aporia += 1
        else:
            dialogue.turns_in_aporia = 0

        # Consecutive same state
        if next_state == prev_state:
            dialogue.consecutive_same_state += 1
        else:
            dialogue.consecutive_same_state = 0

        dialogue.current_state = next_state

        # Handle EXIT
        if next_state == DialogueState.EXIT:
            dialogue.active = False
            dialogue.consecutive_socratic_turns = 0
            dialogue.yield_cooldown = 2  # mandatory 2-turn cooldown

            # Extract insight if INSIGHT_ACHIEVED
            insights: List[SocraticInsight] = []
            if yield_reason == YieldReason.INSIGHT_ACHIEVED:
                insight = extract_insight(
                    propositions, dialogue, input_.paradox_flags,
                    InsightSource.USER_ARTICULATED
                )
                if insight:
                    insights.append(insight)
                    dialogue.insights_generated.append(insight)

            return self._build_output(
                question=None,
                new_state=DialogueState.EXIT,
                active=False,
                yield_to_direct=True,
                yield_reason=yield_reason,
                assumptions=list(dialogue.assumptions_surfaced),
                entailments=[],
                convergence=convergence,
                insights=insights,
                activation_score=activation_score,
                topic_depth=topic_depth,
                state=next_state,
                sigma=0.0,
                kappa=convergence,
                c=c,
                u=u,
                engagement=1.0 - f,
                f=f,
                insight_generated=bool(insights),
                dialectical_distance=insights[0].dialectical_distance if insights else 0.0,
                t0=t0,
            )

        # Entailment generation for ELENCHUS
        entailments: List[Entailment] = []
        if next_state == DialogueState.ELENCHUS:
            entailments = generate_entailments_from_propositions(
                propositions, input_.user_beliefs, input_.contradiction_flags
            )

        # Question generation
        target = select_target_proposition(
            next_state, propositions, input_.contradiction_flags,
            dialogue.assumptions_surfaced, convergence
        )
        qt = select_question_type(next_state, dialogue.question_history, self._config)
        prev_q_id = dialogue.question_history[-1].question_id if dialogue.question_history else None
        question = formulate_question(qt, next_state, target, dialogue.turn_count, prev_q_id, self._rng)
        dialogue.question_history.append(question)

        # Neurochemical signals
        sigma = compute_socratic_processing_signal(
            dialogue.turn_count, self._config.max_socratic_turns,
            dialogue.assumptions_surfaced, max(len(propositions), 1),
            input_.paradox_flags, self._config
        )
        engagement = 1.0 - f

        return self._build_output(
            question=question,
            new_state=next_state,
            active=True,
            yield_to_direct=False,
            yield_reason=None,
            assumptions=list(dialogue.assumptions_surfaced),
            entailments=entailments,
            convergence=convergence,
            insights=[],
            activation_score=activation_score,
            topic_depth=topic_depth,
            state=next_state,
            sigma=sigma,
            kappa=convergence,
            c=c,
            u=u,
            engagement=engagement,
            f=f,
            insight_generated=False,
            dialectical_distance=0.0,
            t0=t0,
        )

    # ------------------------------------------------------------------
    # Internal self-inquiry
    # ------------------------------------------------------------------
    def _run_internal_inquiry(
        self, input_: SocraticInput, t0: float
    ) -> SocraticOutput:
        questions = generate_internal_questions(input_.unsolved_buffer, self._rng, self._config)
        primary_q = questions[0] if questions else None
        dt_ms = (time.perf_counter() - t0) * 1000.0
        signals: Dict[str, float] = {
            "ach_burst": 0.10, "da_tonic": 0.05, "da_phasic": 0.0,
            "oxt_drift": 0.0, "ne_burst": 0.0, "gaba_suppress": 0.0,
            "theta_boost": 0.08, "beta_boost": 0.0, "theta_gamma_boost": 0.0,
        }
        return SocraticOutput(
            generated_question=primary_q,
            new_dialogue_state=DialogueState.PROBING,
            socratic_active=True,
            yield_to_direct=False,
            yield_reason=None,
            identified_assumptions=[],
            identified_entailments=[],
            convergence_state=ConvergenceMetrics(),
            insights=[],
            activation_score=0.5,
            topic_depth_score=0.5,
            processing_time_ms=dt_ms,
            neurochemical_signals=signals,
        )

    # ------------------------------------------------------------------
    # Yield (not activated)
    # ------------------------------------------------------------------
    def _yield_output(
        self,
        activation_score: float,
        topic_depth: float,
        reason: Optional[YieldReason],
        t0: float,
    ) -> SocraticOutput:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return SocraticOutput(
            generated_question=None,
            new_dialogue_state=DialogueState.EXIT,
            socratic_active=False,
            yield_to_direct=True,
            yield_reason=reason,
            identified_assumptions=[],
            identified_entailments=[],
            convergence_state=ConvergenceMetrics(),
            insights=[],
            activation_score=activation_score,
            topic_depth_score=topic_depth,
            processing_time_ms=dt_ms,
            neurochemical_signals={
                "ach_burst": 0.0, "da_tonic": 0.0, "da_phasic": 0.0,
                "oxt_drift": 0.0, "ne_burst": 0.0, "gaba_suppress": 0.0,
                "theta_boost": 0.0, "beta_boost": 0.0, "theta_gamma_boost": 0.0,
            },
        )

    # ------------------------------------------------------------------
    # Output builder
    # ------------------------------------------------------------------
    def _build_output(
        self,
        question: Optional[SocraticQuestion],
        new_state: DialogueState,
        active: bool,
        yield_to_direct: bool,
        yield_reason: Optional[YieldReason],
        assumptions: List[Assumption],
        entailments: List[Entailment],
        convergence: ConvergenceMetrics,
        insights: List[SocraticInsight],
        activation_score: float,
        topic_depth: float,
        state: DialogueState,
        sigma: float,
        kappa: ConvergenceMetrics,
        c: float,
        u: float,
        engagement: float,
        f: float,
        insight_generated: bool,
        dialectical_distance: float,
        t0: float,
    ) -> SocraticOutput:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        signals = compute_neurochemical_signals(
            state=state,
            sigma=sigma,
            kappa=kappa,
            c=c,
            u=u,
            engagement=engagement,
            f=f,
            insight_generated=insight_generated,
            dialectical_distance=dialectical_distance,
            turn_count=self._state.dialogue.turn_count,
            config=self._config,
            rng=self._rng,
        )
        return SocraticOutput(
            generated_question=question,
            new_dialogue_state=new_state,
            socratic_active=active,
            yield_to_direct=yield_to_direct,
            yield_reason=yield_reason,
            identified_assumptions=assumptions,
            identified_entailments=entailments,
            convergence_state=convergence,
            insights=insights,
            activation_score=activation_score,
            topic_depth_score=topic_depth,
            processing_time_ms=dt_ms,
            neurochemical_signals=signals,
        )

    # ------------------------------------------------------------------
    # Status port
    # ------------------------------------------------------------------
    def get_status(self) -> Dict:
        d = self._state.dialogue
        return {
            "engine_id": self.engine_id,
            "mode": self._mode.value,
            "socratic_active": d.active,
            "current_state": d.current_state.value,
            "turn_count": d.turn_count,
            "consecutive_socratic_turns": d.consecutive_socratic_turns,
            "yield_cooldown": d.yield_cooldown,
            "assumptions_surfaced": len(d.assumptions_surfaced),
            "insights_generated": len(d.insights_generated),
            "question_history_len": len(d.question_history),
        }
