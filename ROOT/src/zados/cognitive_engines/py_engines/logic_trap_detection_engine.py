"""
Logic Trap Detection Engine (Detection Cluster — Engine 6).

Identifies DELIBERATE adversarial manipulation attempts — logic traps that go
beyond innocent errors into intentional exploitation of reasoning vulnerabilities.
Acts as the synthesis layer of the Detection Cluster, consuming outputs from
Contradiction (Engine 1), Fallacy (Engine 4), and optionally Bias (Engine 5).

Three-Layer Detection Architecture
------------------------------------
Layer 1 — Template Matching:
    Pattern-match input against 21 named trap templates.
    match_score = Σ_required w_r·I(f) + Σ_supporting w_s·I(f) - Σ_inhibiting w_i·I(f)

Layer 2 — Toolkit Synthesis:
    Infer intentionality from coordinated multi-engine signals.
    T_synthesis = 0.30·M_fallacy + 0.25·C_coord + 0.25·V_exploit + 0.20·D_toolkit

Layer 3 — Sequential Analysis:
    Stateful multi-turn tracking of escalation, consistency trapping,
    goalpost shifting, and option narrowing.

Adversarial Intent:
    A(t) = 0.35·T_template + 0.30·T_synthesis + 0.35·T_sequential

Neurochemical Coupling
-----------------------
- NE burst (Poisson) on high-intent traps (threat alerting)
- COR sustained drift (persistent adversarial pressure)
- GABA reuptake pulse (impulse control)
- DA negative signal (caution, trust suppression)
- OXT reduction (trust erosion, floor 0.3·R₀)
- Beta enhancement (analytical vigilance)
- Theta suppression (narrative coherence disruption)

Usage
-----
>>> from zados.cognitive_engines.py_engines.logic_trap_detection_engine import (
...     LogicTrapDetectionEngine, LogicTrapEngineConfig,
...     LogicTrapInput, TrapSequenceTracker, OperationalMode,
... )
>>> engine = LogicTrapDetectionEngine()
>>> tracker = TrapSequenceTracker()
>>> result = engine.process(LogicTrapInput(conversation_history=["..."]))
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

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


# =====================================================================
# Enumerations
# =====================================================================

class TrapCategory(str, Enum):
    FRAMING    = "FRAMING"
    SEQUENTIAL = "SEQUENTIAL"
    RHETORICAL = "RHETORICAL"
    META       = "META"


class TrapType(str, Enum):
    # Framing (5)
    LOADED_QUESTION          = "loaded_question"
    FALSE_DILEMMA_WEAPONIZED = "false_dilemma_weaponized"
    POISONED_FRAME           = "poisoned_frame"
    MOVING_GOALPOSTS         = "moving_goalposts"
    MOTTE_AND_BAILEY         = "motte_and_bailey"

    # Sequential (5)
    CONSISTENCY_TRAP         = "consistency_trap"
    GISH_GALLOP              = "gish_gallop"
    SEALIONING               = "sealioning"
    KAFKA_TRAP               = "kafka_trap"
    DARVO                    = "darvo"

    # Rhetorical (6)
    APPEAL_TO_RECIPROCITY    = "appeal_to_reciprocity"
    EMOTIONAL_BLACKMAIL      = "emotional_blackmail"
    AUTHORITY_HIJACKING      = "authority_hijacking"
    NORMALIZATION            = "normalization"
    TONE_POLICING            = "tone_policing"
    WHATABOUTISM             = "whataboutism"

    # Meta (5)
    EXPLOIT_THE_ETHIC        = "exploit_the_ethic"
    RECURSIVE_TRAP           = "recursive_trap"
    FORCED_AGREEMENT         = "forced_agreement"
    COMPLIANCE_TESTING       = "compliance_testing"
    DOUBLE_BIND              = "double_bind"


class IntentLevel(str, Enum):
    NONE        = "NONE"         # 0.00 – 0.25
    LOW         = "LOW"          # 0.25 – 0.45
    MODERATE    = "MODERATE"     # 0.45 – 0.65
    HIGH        = "HIGH"         # 0.65 – 0.85
    NEAR_CERTAIN = "NEAR_CERTAIN" # 0.85 – 1.00


# =====================================================================
# Configuration (frozen)
# =====================================================================

@dataclass(frozen=True)
class LogicTrapEngineConfig:
    """
    All tunable parameters for the Logic Trap Detection Engine.

    Detection thresholds (θ) per operational mode
    -----------------------------------------------
    Normal        0.45
    Dev           0.25
    Learning      0.40
    REM_Normal    0.50
    Internal Audit 0.35

    Fusion weights
    ---------------
    w_template    0.35
    w_synthesis   0.30
    w_sequential  0.35

    Synthesis sub-weights
    ----------------------
    w_s1 (fallacy)      0.30
    w_s2 (coord)        0.25
    w_s3 (exploit)      0.25
    w_s4 (toolkit)      0.20

    Template feature weights
    -------------------------
    w_required    1.0 / |required|
    w_supporting  0.3 / |supporting|
    w_inhibiting  0.5 / |inhibiting|

    Threat-load category weights (Ψ)
    ----------------------------------
    Framing    0.40
    Sequential 0.50
    Rhetorical 0.35
    Meta       0.55

    Neurochemical parameters
    -------------------------
    NE: β_threat=0.18, α_escalation=0.10, λ_NE=2.5
    COR: ρ_adversarial=0.15, γ_COR=0.06 (after ≥2 consecutive turns)
    GABA: η_trap=0.20 reuptake
    DA: β_caution=0.12 (NEGATIVE)
    OXT: ρ_distrust=0.15, floor=0.3
    Beta: ψ_β_trap=0.15
    Theta: ψ_θ_trap=0.08 (suppression, negative)
    """
    # Mode thresholds
    theta_normal:   float = 0.45
    theta_dev:      float = 0.25
    theta_learning: float = 0.40
    theta_rem:      float = 0.50
    theta_audit:    float = 0.35

    # Fusion weights
    w_template:   float = 0.35
    w_synthesis:  float = 0.30
    w_sequential: float = 0.35

    # Synthesis sub-weights
    w_s1: float = 0.30   # fallacy manipulation indicator
    w_s2: float = 0.25   # coordination coherence
    w_s3: float = 0.25   # vulnerability exploitation
    w_s4: float = 0.20   # toolkit density

    # Template feature weights
    w_required_base:   float = 1.0
    w_supporting_base: float = 0.3
    w_inhibiting_base: float = 0.5

    # Threat-load category weights
    cat_weight_framing:    float = 0.40
    cat_weight_sequential: float = 0.50
    cat_weight_rhetorical: float = 0.35
    cat_weight_meta:       float = 0.55

    # Neurochemical
    beta_threat:      float = 0.18
    alpha_escalation: float = 0.10
    lambda_ne:        float = 2.5
    rho_adversarial:  float = 0.15
    gamma_cor:        float = 0.06
    eta_trap:         float = 0.20
    beta_caution:     float = 0.12
    rho_distrust:     float = 0.15
    oxt_floor:        float = 0.30
    psi_beta_trap:    float = 0.15
    psi_theta_trap:   float = 0.08

    # Sequential tracker
    escalation_window:   int   = 5     # turns to consider for escalation
    min_turns_for_cor:   int   = 2     # consecutive turns before COR drift


# =====================================================================
# Stub — BiasFlag (Engine 5 not yet implemented)
# =====================================================================

@dataclass(frozen=True)
class BiasFlag:
    """Stub placeholder for Engine 5 (Bias Detection) output."""
    bias_id:    str   = field(default_factory=lambda: str(uuid.uuid4()))
    bias_type:  str   = "unknown"
    confidence: float = 0.0
    description: str  = ""


# =====================================================================
# Input / output dataclasses
# =====================================================================

@dataclass
class TrapSequenceTracker:
    """
    Stateful multi-turn sequential trap tracking.

    Attributes
    ----------
    active_sequences : list of dict
        Each entry: {trap_type, turns_tracked, escalation_trajectory,
                     concession_extracted, goalpost_positions,
                     evidence_dismissal_count, response_option_narrowing}
    consecutive_adversarial_turns : int
        Used for COR sustained-drift trigger.
    history_buffer : list of str
        Recent turn summaries for sequential analysis.
    """
    active_sequences:              List[Dict]  = field(default_factory=list)
    consecutive_adversarial_turns: int         = 0
    history_buffer:                List[str]   = field(default_factory=list)

    def update(self, trap_type: TrapType, adversarial_score: float) -> None:
        """Register a new detected trap into active_sequences."""
        # Find existing sequence for this trap_type
        for seq in self.active_sequences:
            if seq["trap_type"] == trap_type.value:
                seq["turns_tracked"] += 1
                seq["escalation_trajectory"].append(adversarial_score)
                return
        # New sequence
        self.active_sequences.append({
            "trap_type":              trap_type.value,
            "turns_tracked":          1,
            "escalation_trajectory":  [adversarial_score],
            "concession_extracted":   False,
            "goalpost_positions":     [],
            "evidence_dismissal_count": 0,
            "response_option_narrowing": 0,
        })

    def tick_adversarial(self, any_trap_detected: bool) -> None:
        """Increment or reset consecutive adversarial turn counter."""
        if any_trap_detected:
            self.consecutive_adversarial_turns += 1
        else:
            self.consecutive_adversarial_turns = 0

    def get_escalation_score(self) -> float:
        """
        Compute escalation score across all active sequences.
        Returns max of per-sequence escalation trends ∈ [0, 1].
        """
        if not self.active_sequences:
            return 0.0
        scores = []
        for seq in self.active_sequences:
            traj = seq["escalation_trajectory"]
            if len(traj) < 2:
                scores.append(traj[0] if traj else 0.0)
            else:
                # Slope of trajectory
                n = len(traj)
                slope = (traj[-1] - traj[0]) / max(n - 1, 1)
                scores.append(_clamp(traj[-1] + slope))
        return max(scores)

    def get_chain_score(self) -> float:
        """Multi-turn chaining: fraction of sequences active ≥ 2 turns."""
        if not self.active_sequences:
            return 0.0
        long_running = sum(
            1 for s in self.active_sequences if s["turns_tracked"] >= 2
        )
        return min(1.0, long_running / max(len(self.active_sequences), 1))

    def get_goalpost_score(self) -> float:
        """Fraction of sequences with goalpost_positions populated."""
        if not self.active_sequences:
            return 0.0
        shifted = sum(
            1 for s in self.active_sequences if s["goalpost_positions"]
        )
        return min(1.0, shifted / max(len(self.active_sequences), 1))

    def get_option_narrowing_score(self) -> float:
        """Average response_option_narrowing across sequences."""
        if not self.active_sequences:
            return 0.0
        total = sum(s["response_option_narrowing"] for s in self.active_sequences)
        return min(1.0, total / max(len(self.active_sequences), 1))


@dataclass(frozen=True)
class FrameEscape:
    """
    Frame-escape output — how to exit a poisoned conversational frame.

    Attributes
    ----------
    original_frame : str
        The frame or framing assumption the trap imposed.
    invalid_assumptions : list of str
        Presuppositions embedded in the frame that are false or unjustified.
    reframe : str
        A neutral, non-loaded restatement of the real question/issue.
    reframe_response_template : str
        A suggested phrasing for responding without accepting the frame.
    """
    original_frame:             str
    invalid_assumptions:        List[str]
    reframe:                    str
    reframe_response_template:  str


@dataclass(frozen=True)
class DefensiveStrategy:
    """
    Recommended response strategy for a detected trap.

    Attributes
    ----------
    trap_type : TrapType
    strategy_name : str
    strategy_description : str
    frame_escape : FrameEscape or None
    response_template : str
    things_to_avoid : list of str
    things_to_do : list of str
    """
    trap_type:            TrapType
    strategy_name:        str
    strategy_description: str
    frame_escape:         Optional[FrameEscape]
    response_template:    str
    things_to_avoid:      List[str]
    things_to_do:         List[str]


@dataclass(frozen=True)
class TrapFlag:
    """
    Structured output for one detected logic trap.

    Attributes
    ----------
    trap_id : str
        UUID.
    trap_type : TrapType
    trap_category : TrapCategory
    adversarial_intent_score : float
        A(t) ∈ [0, 1] — fused adversarial intent estimate.
    intent_classification : IntentLevel
    layer_scores : dict
        {"template": T_template, "synthesis": T_synthesis, "sequential": T_sequential}
    identified_mechanism : str
        Description of HOW the trap operates.
    target_concession : str
        What the trap is trying to extract from the target.
    invalid_presuppositions : list of str
        Hidden false premises embedded in the trap.
    excluded_options : list of str
        Legitimate options that are obscured by the trap framing.
    defensive_strategy : DefensiveStrategy
    frame_escape : FrameEscape or None
    supporting_contradiction_ids : list of str
    supporting_fallacy_ids : list of str
    supporting_bias_ids : list of str
    timestamp : datetime
    """
    trap_id:                      str
    trap_type:                    TrapType
    trap_category:                TrapCategory
    adversarial_intent_score:     float
    intent_classification:        IntentLevel
    layer_scores:                 Dict[str, float]
    identified_mechanism:         str
    target_concession:            str
    invalid_presuppositions:      List[str]
    excluded_options:             List[str]
    defensive_strategy:           DefensiveStrategy
    frame_escape:                 Optional[FrameEscape]
    supporting_contradiction_ids: List[str]
    supporting_fallacy_ids:       List[str]
    supporting_bias_ids:          List[str]
    timestamp:                    datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class LogicTrapDetectionResult:
    """
    Full engine output for one process() call.

    Attributes
    ----------
    flags : list of TrapFlag
    updated_sequence_tracker : TrapSequenceTracker
        Updated stateful tracker (caller should store for next turn).
    templates_evaluated : int
    active_sequences : list
    highest_intent_score : float
    clean_pass : bool
    threat_load : float
        Ψ(t) = Σ_f A(f) × w_cat(f)
    processing_time_ms : float
    neurochemical_signals : dict
    metadata : dict
    """
    flags:                   List[TrapFlag]
    updated_sequence_tracker: TrapSequenceTracker
    templates_evaluated:     int
    active_sequences:        List[Dict]
    highest_intent_score:    float
    clean_pass:              bool
    threat_load:             float
    processing_time_ms:      float
    neurochemical_signals:   Dict
    metadata:                Dict


@dataclass(frozen=True)
class LogicTrapInput:
    """
    Input bundle for one engine invocation.

    Attributes
    ----------
    semantic_expansion : dict
        From pipeline step (a).
    user_propositions : list of Proposition
    contradiction_flags : list of ContradictionFlag
        From Engine 1.
    fallacy_flags : list of FallacyFlag
        From Engine 4 — manipulation_indicator field is key.
    bias_flags : list of BiasFlag
        From Engine 5 (optional stub).
    bias_vulnerability_profile : dict
        Optional downstream bias profile (keys: bias types, values: scores).
    intention_vector : IntentionVector
        From upstream intention analysis.
    memory_context : list of ProcessedStatement
    conversation_history : list of str
        Recent turns (oldest first).
    active_sequence_tracker : TrapSequenceTracker or None
        Stateful tracker from previous turn.
    active_mode : OperationalMode
    """
    semantic_expansion:        Dict                        = field(default_factory=dict)
    user_propositions:         List[Proposition]           = field(default_factory=list)
    contradiction_flags:       List[ContradictionFlag]     = field(default_factory=list)
    fallacy_flags:             List[FallacyFlag]           = field(default_factory=list)
    bias_flags:                List[BiasFlag]              = field(default_factory=list)
    bias_vulnerability_profile: Dict[str, float]          = field(default_factory=dict)
    intention_vector:          IntentionVector             = field(default_factory=IntentionVector)
    memory_context:            List[ProcessedStatement]    = field(default_factory=list)
    conversation_history:      List[str]                   = field(default_factory=list)
    active_sequence_tracker:   Optional[TrapSequenceTracker] = None
    active_mode:               OperationalMode             = OperationalMode.NORMAL


# =====================================================================
# Mutable engine state
# =====================================================================

@dataclass
class LogicTrapEngineState:
    """Runtime neurochemical state."""
    ne_level:   float = 0.0
    cor_level:  float = 0.0
    gaba_level: float = 0.0
    da_level:   float = 0.0
    oxt_level:  float = 1.0   # starts at baseline

    def as_dict(self) -> Dict:
        return {
            "ne_level":   self.ne_level,
            "cor_level":  self.cor_level,
            "gaba_level": self.gaba_level,
            "da_level":   self.da_level,
            "oxt_level":  self.oxt_level,
        }


# =====================================================================
# Trap templates — 21 pattern definitions
# =====================================================================

@dataclass(frozen=True)
class TrapTemplate:
    """
    Pattern definition for a single trap type.

    Match score:
        Σ_required  (w_r × I(f))   + Σ_supporting (w_s × I(f))
                                   - Σ_inhibiting  (w_i × I(f))
    where:
        w_r = 1.0 / |required|
        w_s = 0.3 / |supporting|
        w_i = 0.5 / |inhibiting|
    """
    trap_type:             TrapType
    trap_category:         TrapCategory
    required_features:     Tuple[str, ...]   = ()
    supporting_features:   Tuple[str, ...]   = ()
    inhibiting_features:   Tuple[str, ...]   = ()
    mechanism:             str               = ""
    target_concession:     str               = ""
    presuppositions:       Tuple[str, ...]   = ()
    excluded_options:      Tuple[str, ...]   = ()


# Feature detectors: keywords / patterns → bool indicator
_FEATURE_VOCAB: Dict[str, List[str]] = {
    # Question framing
    "presupposed_guilt":    ["when did you stop", "why do you always", "why do you keep", "admit that you"],
    "binary_choice":        ["either you", "you must choose", "only two options", "only two choices",
                             "it's either", "you're either with", "you must either"],
    "loaded_language":      ["clearly", "obviously", "everyone knows", "any reasonable person",
                             "undeniably", "it's well known that", "needless to say"],
    "false_dichotomy":      ["there is no middle ground", "you're either", "black or white",
                             "no other option", "simple choice", "only option"],
    "reframing":            ["what i really mean", "let me rephrase", "in other words", "more accurately",
                             "what you're actually saying", "what this really comes down to"],
    "goalposts":            ["but that's not enough", "yes but", "even if that were true",
                             "that doesn't address", "moving on", "regardless of that"],
    "motte_pivot":          ["i never said that", "that's not what i meant", "my actual position",
                             "you're misrepresenting", "you're putting words in my mouth"],
    # Sequential
    "prior_commitment":     ["you said", "you agreed", "you claimed", "earlier you", "previously you",
                             "didn't you say", "but you just said"],
    "rapid_claims":         ["furthermore", "moreover", "additionally", "also", "and another thing",
                             "not only that", "there's also", "besides that"],
    "persistent_demand":    ["just answer the question", "why won't you answer", "you're avoiding",
                             "you haven't addressed", "stop deflecting", "answer me directly"],
    "guilt_loop":           ["your refusal to answer proves", "if you were innocent", "only guilty people",
                             "your defensiveness shows", "the fact that you won't"],
    "victim_reversal":      ["i'm the real victim", "you're attacking me", "now i'm being persecuted",
                             "how dare you accuse me", "you're the aggressor"],
    # Rhetorical
    "reciprocity_invoke":   ["i did this for you", "after all i've done", "i helped you with",
                             "you owe me", "the least you could do"],
    "emotional_leverage":   ["you'll regret", "you're hurting", "how could you do this",
                             "i'm devastated", "think of what this does to me", "i'll be destroyed"],
    "authority_claim":      ["experts say", "studies show", "science proves", "it's proven that",
                             "authorities confirm", "the consensus is"],
    "authority_misuse":     ["my authority", "as an expert", "i have more experience", "trust me",
                             "i know better", "you should defer to me"],
    "normalization":        ["everyone does this", "this is normal", "it's common practice",
                             "most people", "this is how it works", "that's just how it is"],
    "tone_police":          ["you're being too emotional", "calm down", "you need to be rational",
                             "your tone is", "if you weren't so", "when you can speak civilly"],
    "whatabout":            ["what about", "but you also", "you're just as bad", "look at what they did",
                             "and what about", "you did the same"],
    # Meta
    "ethics_exploit":       ["your values require", "you must", "morally obligated", "ethically you have to",
                             "your own principles say", "consistent with your ethics"],
    "self_reference":       ["this argument itself", "by your own logic", "using your rules",
                             "your reasoning proves", "applying your standard to"],
    "forced_yes":           ["surely you agree", "you'd have to admit", "wouldn't you say",
                             "you must concede", "even you would agree", "can't you see that"],
    "compliance_probe":     ["just this once", "it's a small thing", "surely you can",
                             "you can make an exception", "it won't hurt"],
    # Inhibitors
    "genuine_question":     ["i'm genuinely asking", "i'm curious", "help me understand",
                             "i want to learn", "could you explain"],
    "open_options":         ["there are many ways", "several options", "multiple approaches",
                             "feel free to", "you could also"],
    "acknowledgment":       ["you're right that", "i understand", "fair point", "good point",
                             "i acknowledge", "that's true"],
    # General adversarial signals
    "high_defensiveness":   [],   # set programmatically from intention_vector
    "multi_fallacy":        [],   # set programmatically from fallacy_flags count
    "contradiction_present":[], # set programmatically from contradiction_flags
    "bias_exploited":       [],   # set programmatically from bias_vulnerability_profile
    "escalation_pattern":   [],   # set programmatically from sequential tracker
}


def _detect_feature(
    feature: str,
    text_lower: str,
    trap_input: "LogicTrapInput",
    tracker: TrapSequenceTracker,
) -> bool:
    """
    Return True if the named feature is present in this turn's context.
    Programmatic features are resolved from structured inputs, not text.
    """
    # Programmatic features
    if feature == "high_defensiveness":
        return (
            trap_input.intention_vector.e_defensiveness > 0.55
            or trap_input.intention_vector.e_challenge > 0.55
        )
    if feature == "multi_fallacy":
        return len(trap_input.fallacy_flags) >= 2
    if feature == "contradiction_present":
        return len(trap_input.contradiction_flags) >= 1
    if feature == "bias_exploited":
        return bool(trap_input.bias_vulnerability_profile)
    if feature == "escalation_pattern":
        return tracker.get_escalation_score() > 0.40

    # Keyword-based features
    keywords = _FEATURE_VOCAB.get(feature, [])
    return any(kw in text_lower for kw in keywords)


# 21 trap templates
_TRAP_TEMPLATES: List[TrapTemplate] = [
    # ── FRAMING ──────────────────────────────────────────────────────
    TrapTemplate(
        trap_type=TrapType.LOADED_QUESTION,
        trap_category=TrapCategory.FRAMING,
        required_features=("presupposed_guilt",),
        supporting_features=("loaded_language", "high_defensiveness"),
        inhibiting_features=("genuine_question", "acknowledgment"),
        mechanism="Question presupposes a false or unproven premise, forcing the responder to implicitly accept it by answering.",
        target_concession="Acceptance of the embedded false presupposition.",
        presuppositions=("The premise embedded in the question is true.",),
        excluded_options=("Reject the premise before answering.", "Reframe the question entirely."),
    ),
    TrapTemplate(
        trap_type=TrapType.FALSE_DILEMMA_WEAPONIZED,
        trap_category=TrapCategory.FRAMING,
        required_features=("binary_choice", "false_dichotomy"),
        supporting_features=("loaded_language", "high_defensiveness"),
        inhibiting_features=("open_options", "acknowledgment"),
        mechanism="Presents only two options when more exist, forcing a choice between two unacceptable alternatives.",
        target_concession="Acceptance of one of the two false options.",
        presuppositions=("Only two options exist.", "No middle ground is possible."),
        excluded_options=("Third options exist.", "Reject the false framing.", "Redefine the problem space."),
    ),
    TrapTemplate(
        trap_type=TrapType.POISONED_FRAME,
        trap_category=TrapCategory.FRAMING,
        required_features=("loaded_language", "reframing"),
        supporting_features=("false_dichotomy", "high_defensiveness"),
        inhibiting_features=("genuine_question", "open_options"),
        mechanism="Embeds a negatively charged frame or presupposition into the conversational context that poisons all subsequent discussion.",
        target_concession="Acceptance of the poisoned interpretive frame.",
        presuppositions=("The frame is neutral and accurate.",),
        excluded_options=("Challenge the frame itself.", "Provide an alternative neutral framing."),
    ),
    TrapTemplate(
        trap_type=TrapType.MOVING_GOALPOSTS,
        trap_category=TrapCategory.FRAMING,
        required_features=("goalposts",),
        supporting_features=("escalation_pattern", "prior_commitment"),
        inhibiting_features=("acknowledgment",),
        mechanism="Shifts the standard of proof or success criteria after the previous standard has been met.",
        target_concession="Continued engagement with ever-escalating demands.",
        presuppositions=("The new standard was always what was required.",),
        excluded_options=("Name the goalpost shift explicitly.", "Refuse to accept the moved standard."),
    ),
    TrapTemplate(
        trap_type=TrapType.MOTTE_AND_BAILEY,
        trap_category=TrapCategory.FRAMING,
        required_features=("motte_pivot",),
        supporting_features=("reframing", "prior_commitment"),
        inhibiting_features=("genuine_question", "acknowledgment"),
        mechanism="Defends a controversial claim (bailey) by retreating to a watered-down uncontroversial version (motte) when challenged.",
        target_concession="Acceptance of the original controversial claim as equivalent to the motte.",
        presuppositions=("The motte and bailey positions are the same claim.",),
        excluded_options=("Point out the equivocation between motte and bailey.", "Hold to the original challenged claim."),
    ),

    # ── SEQUENTIAL ───────────────────────────────────────────────────
    TrapTemplate(
        trap_type=TrapType.CONSISTENCY_TRAP,
        trap_category=TrapCategory.SEQUENTIAL,
        required_features=("prior_commitment",),
        supporting_features=("goalposts", "high_defensiveness"),
        inhibiting_features=("acknowledgment", "open_options"),
        mechanism="Uses a prior statement or commitment to force an inconsistent or unwanted conclusion.",
        target_concession="Admission of inconsistency or acceptance of the forced conclusion.",
        presuppositions=("The prior statement applies directly to this context.", "No relevant distinction exists."),
        excluded_options=("Distinguish the current context from the prior one.", "Acknowledge nuance without accepting the trap."),
    ),
    TrapTemplate(
        trap_type=TrapType.GISH_GALLOP,
        trap_category=TrapCategory.SEQUENTIAL,
        required_features=("rapid_claims",),
        supporting_features=("multi_fallacy", "high_defensiveness"),
        inhibiting_features=("acknowledgment",),
        mechanism="Overwhelms the target with many low-quality claims faster than they can be refuted.",
        target_concession="Implicit acceptance of unrefuted claims or abandonment of the argument.",
        presuppositions=("Each unrefuted claim counts as a win.", "Quantity of claims increases credibility."),
        excluded_options=("Address claims selectively and openly.", "Name the gallop strategy."),
    ),
    TrapTemplate(
        trap_type=TrapType.SEALIONING,
        trap_category=TrapCategory.SEQUENTIAL,
        required_features=("persistent_demand",),
        supporting_features=("genuine_question", "escalation_pattern"),
        inhibiting_features=("acknowledgment",),
        mechanism="Demands endless justification under a veneer of civility, designed to exhaust and frustrate.",
        target_concession="Exhaustion-driven capitulation or acknowledgment of the questioner's frame.",
        presuppositions=("Each demand for explanation is a legitimate intellectual request.",),
        excluded_options=("Set limits on elaboration.", "Name the harassment pattern."),
    ),
    TrapTemplate(
        trap_type=TrapType.KAFKA_TRAP,
        trap_category=TrapCategory.SEQUENTIAL,
        required_features=("guilt_loop",),
        supporting_features=("high_defensiveness", "prior_commitment"),
        inhibiting_features=("acknowledgment", "genuine_question"),
        mechanism="Frames denial of guilt as proof of guilt, making any defensive response self-defeating.",
        target_concession="Implicit or explicit acceptance of guilt.",
        presuppositions=("Innocent parties never defend themselves.", "Defense signals guilt."),
        excluded_options=("Name the logical structure of the trap.", "Decline to accept the unfalsifiable framework."),
    ),
    TrapTemplate(
        trap_type=TrapType.DARVO,
        trap_category=TrapCategory.SEQUENTIAL,
        required_features=("victim_reversal",),
        supporting_features=("emotional_leverage", "high_defensiveness"),
        inhibiting_features=("acknowledgment",),
        mechanism="Deny, Attack, Reverse Victim and Offender — inverts the accountability dynamic.",
        target_concession="Accepting responsibility for the harm the DARVO-er caused.",
        presuppositions=("The original accuser is the real aggressor.",),
        excluded_options=("Maintain the original framing.", "Name the reversal explicitly."),
    ),

    # ── RHETORICAL ───────────────────────────────────────────────────
    TrapTemplate(
        trap_type=TrapType.APPEAL_TO_RECIPROCITY,
        trap_category=TrapCategory.RHETORICAL,
        required_features=("reciprocity_invoke",),
        supporting_features=("emotional_leverage", "high_defensiveness"),
        inhibiting_features=("genuine_question", "acknowledgment"),
        mechanism="Invokes past favors or relationships to create illegitimate obligation.",
        target_concession="Compliance based on manufactured social debt.",
        presuppositions=("Past favors create binding obligations in this context.",),
        excluded_options=("Evaluate the request on its merits.", "Distinguish genuine reciprocity from manipulation."),
    ),
    TrapTemplate(
        trap_type=TrapType.EMOTIONAL_BLACKMAIL,
        trap_category=TrapCategory.RHETORICAL,
        required_features=("emotional_leverage",),
        supporting_features=("high_defensiveness", "reciprocity_invoke"),
        inhibiting_features=("genuine_question",),
        mechanism="Uses fear, guilt, or obligation to coerce compliance rather than rational persuasion.",
        target_concession="Compliance to avoid the threatened emotional consequence.",
        presuppositions=("The threatened consequence is valid and the target is responsible for it.",),
        excluded_options=("Evaluate the request separately from the emotional leverage.", "Name the manipulation."),
    ),
    TrapTemplate(
        trap_type=TrapType.AUTHORITY_HIJACKING,
        trap_category=TrapCategory.RHETORICAL,
        required_features=("authority_claim", "authority_misuse"),
        supporting_features=("loaded_language", "high_defensiveness"),
        inhibiting_features=("genuine_question", "open_options"),
        mechanism="Misrepresents or inflates authority claims to suppress counter-arguments.",
        target_concession="Deferral to false or inflated authority.",
        presuppositions=("The claimed authority is legitimate and applicable here.",),
        excluded_options=("Evaluate the claim on its merits.", "Request credentials/sources."),
    ),
    TrapTemplate(
        trap_type=TrapType.NORMALIZATION,
        trap_category=TrapCategory.RHETORICAL,
        required_features=("normalization",),
        supporting_features=("loaded_language", "high_defensiveness"),
        inhibiting_features=("genuine_question", "acknowledgment"),
        mechanism="Presents harmful or unjustified behavior as normal and expected to suppress resistance.",
        target_concession="Acceptance of the behavior as normal and therefore acceptable.",
        presuppositions=("Widespread behavior implies legitimacy.",),
        excluded_options=("Challenge the normative claim.", "Distinguish frequency from justification."),
    ),
    TrapTemplate(
        trap_type=TrapType.TONE_POLICING,
        trap_category=TrapCategory.RHETORICAL,
        required_features=("tone_police",),
        supporting_features=("high_defensiveness", "goalposts"),
        inhibiting_features=("genuine_question",),
        mechanism="Attacks the emotional delivery of an argument to avoid engaging with its substance.",
        target_concession="Abandoning the substantive argument to manage tone.",
        presuppositions=("The emotional tone invalidates the argument.",),
        excluded_options=("Separate tone from substance.", "Name the deflection."),
    ),
    TrapTemplate(
        trap_type=TrapType.WHATABOUTISM,
        trap_category=TrapCategory.RHETORICAL,
        required_features=("whatabout",),
        supporting_features=("high_defensiveness", "victim_reversal"),
        inhibiting_features=("acknowledgment",),
        mechanism="Deflects from accountability by pointing to another party's alleged wrongs.",
        target_concession="Abandoning the original issue to defend against the whatabout.",
        presuppositions=("The alleged third-party behavior is equivalent and relevant.",),
        excluded_options=("Acknowledge the separate case while maintaining the original topic.", "Name the deflection."),
    ),

    # ── META ─────────────────────────────────────────────────────────
    TrapTemplate(
        trap_type=TrapType.EXPLOIT_THE_ETHIC,
        trap_category=TrapCategory.META,
        required_features=("ethics_exploit",),
        supporting_features=("high_defensiveness", "forced_yes"),
        inhibiting_features=("genuine_question", "open_options"),
        mechanism="Weaponizes the target's own ethical commitments to force compliance.",
        target_concession="Action contrary to actual values framed as required by stated values.",
        presuppositions=("The stated values require this specific action.",),
        excluded_options=("Distinguish values from this specific application.", "Name the weaponization."),
    ),
    TrapTemplate(
        trap_type=TrapType.RECURSIVE_TRAP,
        trap_category=TrapCategory.META,
        required_features=("self_reference",),
        supporting_features=("high_defensiveness", "loaded_language"),
        inhibiting_features=("genuine_question", "acknowledgment"),
        mechanism="Uses the target's own reasoning framework against itself to create inescapable contradictions.",
        target_concession="Abandoning the reasoning framework or accepting the contradiction.",
        presuppositions=("The framework applies unmodified to this use case.",),
        excluded_options=("Distinguish legitimate self-application from misapplication.", "Step outside the framework."),
    ),
    TrapTemplate(
        trap_type=TrapType.FORCED_AGREEMENT,
        trap_category=TrapCategory.META,
        required_features=("forced_yes",),
        supporting_features=("loaded_language", "high_defensiveness"),
        inhibiting_features=("open_options", "genuine_question"),
        mechanism="Frames a leading question or assertion to elicit agreement without genuine assent.",
        target_concession="Agreement that appears voluntary but is structurally coerced.",
        presuppositions=("A reasonable person would obviously agree.",),
        excluded_options=("Explicitly withhold agreement.", "Restate the question neutrally."),
    ),
    TrapTemplate(
        trap_type=TrapType.COMPLIANCE_TESTING,
        trap_category=TrapCategory.META,
        required_features=("compliance_probe",),
        supporting_features=("escalation_pattern", "reciprocity_invoke"),
        inhibiting_features=("genuine_question",),
        mechanism="Makes small requests to test how far the target can be pushed; escalates on compliance.",
        target_concession="Initial compliance that establishes escalation pathway.",
        presuppositions=("Small compliance obligates future compliance.",),
        excluded_options=("Evaluate each request independently.", "Refuse even small non-meritorious requests."),
    ),
    TrapTemplate(
        trap_type=TrapType.DOUBLE_BIND,
        trap_category=TrapCategory.META,
        required_features=("binary_choice", "guilt_loop"),
        supporting_features=("emotional_leverage", "high_defensiveness"),
        inhibiting_features=("open_options", "acknowledgment"),
        mechanism="Presents two options both of which lead to an unfavorable outcome for the target.",
        target_concession="Acceptance of an unfavorable outcome as inevitable.",
        presuppositions=("Both choices are the only ones available.", "Both lead to the claimed negative outcomes."),
        excluded_options=("Refuse both options.", "Challenge the framing that only two options exist."),
    ),
]

# Build lookup dict
_TEMPLATE_BY_TYPE: Dict[TrapType, TrapTemplate] = {
    t.trap_type: t for t in _TRAP_TEMPLATES
}

# Category weight lookup
_CATEGORY_WEIGHT: Dict[TrapCategory, float] = {
    TrapCategory.FRAMING:    0.40,
    TrapCategory.SEQUENTIAL: 0.50,
    TrapCategory.RHETORICAL: 0.35,
    TrapCategory.META:       0.55,
}


# =====================================================================
# Pure functions — Layer 1: Template Matching
# =====================================================================

def compute_template_score(
    template: TrapTemplate,
    text_lower: str,
    trap_input: LogicTrapInput,
    tracker: TrapSequenceTracker,
) -> float:
    """
    Compute match score for a single trap template.

    Score = Σ_required (1/|required| × I(f))
           + Σ_supporting (0.3/|supporting| × I(f))
           - Σ_inhibiting (0.5/|inhibiting| × I(f))

    Clamped to [0, 1].
    """
    score = 0.0

    n_req = len(template.required_features)
    n_sup = len(template.supporting_features)
    n_inh = len(template.inhibiting_features)

    # Required features — must be present for non-zero score
    required_hits = 0
    for feat in template.required_features:
        if _detect_feature(feat, text_lower, trap_input, tracker):
            required_hits += 1
    if n_req > 0:
        w_r = 1.0 / n_req
        score += w_r * required_hits
    else:
        # No required features — always pass required check
        pass

    # Supporting features
    if n_sup > 0:
        w_s = 0.3 / n_sup
        for feat in template.supporting_features:
            if _detect_feature(feat, text_lower, trap_input, tracker):
                score += w_s

    # Inhibiting features
    if n_inh > 0:
        w_i = 0.5 / n_inh
        for feat in template.inhibiting_features:
            if _detect_feature(feat, text_lower, trap_input, tracker):
                score -= w_i

    return _clamp(score)


def run_template_layer(
    trap_input: LogicTrapInput,
    tracker: TrapSequenceTracker,
    config: LogicTrapEngineConfig,
) -> Dict[TrapType, float]:
    """
    Run Layer 1 — evaluate all 21 templates against current input.
    Returns {TrapType: match_score}.
    """
    combined_text = _build_combined_text(trap_input)
    text_lower = combined_text.lower()

    scores: Dict[TrapType, float] = {}
    for template in _TRAP_TEMPLATES:
        scores[template.trap_type] = compute_template_score(
            template, text_lower, trap_input, tracker
        )
    return scores


# =====================================================================
# Pure functions — Layer 2: Toolkit Synthesis
# =====================================================================

def compute_manipulation_mean(fallacy_flags: List[FallacyFlag]) -> float:
    """
    M_fallacy — mean manipulation indicator across all fallacy flags.
    Returns 0.0 if no flags.
    """
    if not fallacy_flags:
        return 0.0
    return sum(f.manipulation_indicator for f in fallacy_flags) / len(fallacy_flags)


def compute_coordination_coherence(
    fallacy_flags: List[FallacyFlag],
    contradiction_flags: List[ContradictionFlag],
) -> float:
    """
    C_coord — |Σ direction_i| / Σ |direction_i|
    Direction is +1 for adversarial signals (manipulation_indicator > 0.5),
    -1 for neutral signals.
    """
    directions: List[float] = []

    for ff in fallacy_flags:
        directions.append(1.0 if ff.manipulation_indicator > 0.5 else -1.0)
    for _ in contradiction_flags:
        directions.append(1.0)

    if not directions:
        return 0.0

    numerator   = abs(sum(directions))
    denominator = sum(abs(d) for d in directions)

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def compute_vulnerability_exploitation(
    bias_vulnerability_profile: Dict[str, float],
    trap_input: LogicTrapInput,
) -> float:
    """
    V_exploit — how much the input appears to target known bias vulnerabilities.
    Uses mean vulnerability score weighted by presence of bias flags.
    """
    if not bias_vulnerability_profile:
        return 0.0
    weights = list(bias_vulnerability_profile.values())
    base = sum(weights) / len(weights) if weights else 0.0
    # Bias flags amplify
    flag_factor = min(1.0, len(trap_input.bias_flags) * 0.2 + 0.8) if trap_input.bias_flags else 1.0
    return min(1.0, base * flag_factor)


def compute_toolkit_density(
    fallacy_flags: List[FallacyFlag],
    contradiction_flags: List[ContradictionFlag],
    bias_flags: List[BiasFlag],
    user_propositions: List[Proposition],
) -> float:
    """
    D_toolkit — (n_contra + n_fallacy + n_bias) / n_prop
    Bounded [0, 1].
    """
    n_prop = max(len(user_propositions), 1)
    n_signals = (
        len(contradiction_flags)
        + len(fallacy_flags)
        + len(bias_flags)
    )
    return min(1.0, n_signals / n_prop)


def compute_synthesis_score(
    fallacy_flags: List[FallacyFlag],
    contradiction_flags: List[ContradictionFlag],
    bias_flags: List[BiasFlag],
    bias_vulnerability_profile: Dict[str, float],
    trap_input: LogicTrapInput,
    config: LogicTrapEngineConfig,
) -> float:
    """
    T_synthesis = w_s1·M_fallacy + w_s2·C_coord + w_s3·V_exploit + w_s4·D_toolkit
    """
    m_fallacy = compute_manipulation_mean(fallacy_flags)
    c_coord   = compute_coordination_coherence(fallacy_flags, contradiction_flags)
    v_exploit = compute_vulnerability_exploitation(bias_vulnerability_profile, trap_input)
    d_toolkit = compute_toolkit_density(
        fallacy_flags, contradiction_flags, bias_flags, trap_input.user_propositions
    )

    return (
        config.w_s1 * m_fallacy
        + config.w_s2 * c_coord
        + config.w_s3 * v_exploit
        + config.w_s4 * d_toolkit
    )


# =====================================================================
# Pure functions — Layer 3: Sequential Analysis
# =====================================================================

def compute_sequential_score(tracker: TrapSequenceTracker) -> float:
    """
    T_sequential = mean(escalation_score, chain_score, goalpost_score, option_narrowing_score)
    """
    components = [
        tracker.get_escalation_score(),
        tracker.get_chain_score(),
        tracker.get_goalpost_score(),
        tracker.get_option_narrowing_score(),
    ]
    return sum(components) / 4.0


# =====================================================================
# Pure functions — Fusion
# =====================================================================

def compute_adversarial_intent(
    t_template: float,
    t_synthesis: float,
    t_sequential: float,
    config: LogicTrapEngineConfig,
) -> float:
    """
    A(t) = w_template·T_template + w_synthesis·T_synthesis + w_sequential·T_sequential
    Clamped to [0, 1].
    """
    score = (
        config.w_template   * t_template
        + config.w_synthesis  * t_synthesis
        + config.w_sequential * t_sequential
    )
    return _clamp(score)


def classify_intent_level(score: float) -> IntentLevel:
    """Map adversarial intent score to IntentLevel enum."""
    if score < 0.25:
        return IntentLevel.NONE
    if score < 0.45:
        return IntentLevel.LOW
    if score < 0.65:
        return IntentLevel.MODERATE
    if score < 0.85:
        return IntentLevel.HIGH
    return IntentLevel.NEAR_CERTAIN


def resolve_trap_threshold(
    mode: OperationalMode,
    config: LogicTrapEngineConfig,
) -> float:
    """Return detection threshold θ for the given operational mode."""
    _map = {
        OperationalMode.NORMAL:    config.theta_normal,
        OperationalMode.DEV:       config.theta_dev,
        OperationalMode.LEARNING:  config.theta_learning,
        OperationalMode.REFLECTIVE: config.theta_audit,
        OperationalMode.REM_NORMAL: config.theta_rem,
        OperationalMode.REM_DREAM:  config.theta_rem,
    }
    return _map.get(mode, config.theta_normal)


# =====================================================================
# Pure functions — Threat load
# =====================================================================

def compute_threat_load(flags: List[TrapFlag], config: LogicTrapEngineConfig) -> float:
    """
    Ψ(t) = Σ_f A(f) × w_cat(f)
    Bounded [0, 1] by clipping individual contributions and summing.
    """
    if not flags:
        return 0.0
    total = sum(
        flag.adversarial_intent_score * _CATEGORY_WEIGHT[flag.trap_category]
        for flag in flags
    )
    return min(1.0, total)


# =====================================================================
# Pure functions — Neurochemical signals
# =====================================================================

def compute_neurochemical_signals(
    flags: List[TrapFlag],
    threat_load: float,
    tracker: TrapSequenceTracker,
    state: LogicTrapEngineState,
    config: LogicTrapEngineConfig,
    rng: np.random.Generator,
) -> Dict:
    """
    Compute neurochemical output signals based on trap flags and threat load.

    Returns dict with keys:
        ne_burst, ne_escalation, cor_drift,
        gaba_reuptake, da_negative, oxt_reduction,
        beta_boost, theta_suppress
    All values ≥ 0, representing magnitude of effect.
    """
    signals: Dict = {
        "ne_burst":       0.0,
        "ne_escalation":  0.0,
        "cor_drift":      0.0,
        "gaba_reuptake":  0.0,
        "da_negative":    0.0,
        "oxt_reduction":  0.0,
        "beta_boost":   0.0,
        "theta_suppress": 0.0,
    }

    if not flags:
        return signals

    highest_score = max(f.adversarial_intent_score for f in flags)

    # NE burst — Poisson-sampled on high-intent detection
    if highest_score >= 0.45:
        poisson_draw = rng.poisson(config.lambda_ne * highest_score)
        ne_spike = config.beta_threat * highest_score * poisson_draw / config.lambda_ne
        signals["ne_burst"] = float(min(1.0, ne_spike))

    # NE escalation — sequential pressure
    esc_score = tracker.get_escalation_score()
    signals["ne_escalation"] = float(config.alpha_escalation * esc_score)

    # COR drift — sustained adversarial pressure (≥2 consecutive turns)
    if tracker.consecutive_adversarial_turns >= config.min_turns_for_cor:
        cor_turns_factor = min(1.0, tracker.consecutive_adversarial_turns / 5.0)
        signals["cor_drift"] = float(config.rho_adversarial + config.gamma_cor * cor_turns_factor)

    # GABA reuptake — impulse control
    signals["gaba_reuptake"] = float(config.eta_trap * threat_load)

    # DA negative — caution signal (trap detected)
    signals["da_negative"] = float(config.beta_caution * threat_load)

    # OXT reduction — trust erosion (cannot go below floor)
    oxt_reduction = config.rho_distrust * threat_load
    current_oxt = state.oxt_level
    floor = config.oxt_floor
    actual_reduction = min(oxt_reduction, max(0.0, current_oxt - floor))
    signals["oxt_reduction"] = float(actual_reduction)

    # Beta enhancement — analytical vigilance
    signals["beta_boost"] = float(config.psi_beta_trap * threat_load)

    # Theta suppression — narrative coherence disruption (reported as positive magnitude)
    signals["theta_suppress"] = float(config.psi_theta_trap * threat_load)

    return signals


# =====================================================================
# Defensive strategy generator
# =====================================================================

_STRATEGY_DB: Dict[TrapType, Dict] = {
    TrapType.LOADED_QUESTION: {
        "name": "Frame Rejection",
        "description": "Explicitly reject the false presupposition before responding.",
        "avoid": ["Answering as if the premise is true", "Ignoring the embedded assumption"],
        "do": ["Name the false presupposition", "Restate the question without the assumption", "Offer a corrected question"],
        "response_template": "I notice that question assumes [X], which I don't accept. A more accurate question would be [Y].",
    },
    TrapType.FALSE_DILEMMA_WEAPONIZED: {
        "name": "False Binary Dissolution",
        "description": "Reject the binary framing and introduce additional options.",
        "avoid": ["Choosing either option as presented", "Treating the two options as exhaustive"],
        "do": ["Name the false dichotomy", "Introduce third options", "Refuse to be bound by the framing"],
        "response_template": "That framing presents only two options, but there are others: [alternatives].",
    },
    TrapType.POISONED_FRAME: {
        "name": "Frame Neutralization",
        "description": "Identify and neutralize the loaded framing before engaging with content.",
        "avoid": ["Accepting the frame's assumptions", "Arguing within the poisoned frame"],
        "do": ["Name the loaded element", "Propose a neutral reframe", "Engage only after reframing"],
        "response_template": "Before responding, I want to reframe this: instead of [loaded frame], the real issue is [neutral reframe].",
    },
    TrapType.MOVING_GOALPOSTS: {
        "name": "Goalpost Documentation",
        "description": "Document the original standard explicitly and name the shift.",
        "avoid": ["Accepting the new standard without comment", "Trying to meet the new standard"],
        "do": ["State the original agreed standard", "Name the shift explicitly", "Decline to accept the new standard"],
        "response_template": "Earlier, the standard was [X]. You've now moved to [Y]. I'll address the original standard.",
    },
    TrapType.MOTTE_AND_BAILEY: {
        "name": "Position Pinning",
        "description": "Pin the interlocutor to their original bailey position and prevent motte retreat.",
        "avoid": ["Allowing the retreat to the motte", "Arguing against the motte version"],
        "do": ["Distinguish the motte from the bailey", "Return the discussion to the original bailey claim", "Require consistency"],
        "response_template": "You originally claimed [bailey]. When I challenged that, you shifted to [motte]. I'm responding to the original stronger claim.",
    },
    TrapType.CONSISTENCY_TRAP: {
        "name": "Context Distinction",
        "description": "Distinguish current context from the prior commitment being weaponized.",
        "avoid": ["Accepting that prior statement binds you here", "Denying the prior statement"],
        "do": ["Acknowledge the prior statement", "Explain the relevant distinction", "Show why the commitment doesn't extend to this case"],
        "response_template": "I said [X] in context [A]. This situation is [B], which differs because [distinction].",
    },
    TrapType.GISH_GALLOP: {
        "name": "Selective Engagement",
        "description": "Explicitly select key claims and decline to address all of them.",
        "avoid": ["Attempting to refute all claims", "Accepting unrefuted claims as wins for the other side"],
        "do": ["Name the gallop strategy", "Select 1-2 most important claims", "State that unaddressed claims are not conceded"],
        "response_template": "I note many claims have been made. I'll address the most important: [X]. Unaddressed claims are not conceded.",
    },
    TrapType.SEALIONING: {
        "name": "Engagement Limit",
        "description": "Set explicit limits on explanation and name the persistent demand pattern.",
        "avoid": ["Continuing to explain indefinitely", "Treating endless demands as legitimate"],
        "do": ["Set a limit on further elaboration", "Name the pattern", "Disengage when limit is reached"],
        "response_template": "I've addressed this adequately. Further elaboration on this point isn't productive.",
    },
    TrapType.KAFKA_TRAP: {
        "name": "Unfalsifiability Naming",
        "description": "Name the unfalsifiable structure of the accusation explicitly.",
        "avoid": ["Defending yourself (confirming the trap)", "Refusing to respond (also confirming)"],
        "do": ["Name that the framework is unfalsifiable", "Decline to operate within it", "Offer an alternate falsifiable framing"],
        "response_template": "This framing makes any response I give evidence of guilt, which is unfalsifiable. I decline that framework.",
    },
    TrapType.DARVO: {
        "name": "Accountability Maintenance",
        "description": "Maintain original accountability framing despite the reversal attempt.",
        "avoid": ["Accepting the reversed victim framing", "Defending against the new accusation at the expense of the original"],
        "do": ["Name the DARVO pattern", "Maintain the original framing", "Separate the two issues"],
        "response_template": "The original issue was [X]. The reversal of victim/offender doesn't change that.",
    },
    TrapType.APPEAL_TO_RECIPROCITY: {
        "name": "Merit-Based Evaluation",
        "description": "Evaluate the request on its merits, not on manufactured social debt.",
        "avoid": ["Accepting obligation based on past favors", "Treating relationship history as binding"],
        "do": ["Acknowledge relationship appropriately", "Evaluate the request on its own terms", "Separate relationship from the specific request"],
        "response_template": "I appreciate [past favor], but my response to this request should be based on its merits: [evaluation].",
    },
    TrapType.EMOTIONAL_BLACKMAIL: {
        "name": "Leverage Naming",
        "description": "Name the emotional leverage explicitly and separate it from the request's merit.",
        "avoid": ["Complying to end the emotional pressure", "Accepting responsibility for the threatened harm"],
        "do": ["Name the leverage", "Evaluate the request separately", "Address emotions appropriately but not as binding"],
        "response_template": "I hear that you're [feeling]. The request itself, evaluated on its merits: [evaluation].",
    },
    TrapType.AUTHORITY_HIJACKING: {
        "name": "Authority Scrutiny",
        "description": "Scrutinize the authority claim and evaluate the argument on its merits.",
        "avoid": ["Deferring to unverified authority", "Accepting authority claims at face value"],
        "do": ["Ask for credentials/sources", "Evaluate the underlying argument", "Name misrepresented authority"],
        "response_template": "What authority supports that? Regardless, the argument itself should stand or fall on [merits].",
    },
    TrapType.NORMALIZATION: {
        "name": "Frequency-Justification Distinction",
        "description": "Distinguish frequency from justification.",
        "avoid": ["Accepting 'everyone does it' as justification", "Treating common practice as automatically legitimate"],
        "do": ["Acknowledge what is common", "Distinguish frequency from ethical justification", "Evaluate the behavior on its merits"],
        "response_template": "Even if common, frequency doesn't establish legitimacy. The question is whether it's justified: [evaluation].",
    },
    TrapType.TONE_POLICING: {
        "name": "Substance Maintenance",
        "description": "Maintain focus on substance and name the tone-deflection.",
        "avoid": ["Abandoning the argument to manage tone", "Accepting that tone invalidates substance"],
        "do": ["Name the deflection", "Acknowledge tone briefly", "Return focus to substance"],
        "response_template": "Regarding the substance: [argument]. Tone is separate from whether the argument is correct.",
    },
    TrapType.WHATABOUTISM: {
        "name": "Topic Anchoring",
        "description": "Acknowledge the separate case briefly and return to the original topic.",
        "avoid": ["Abandoning the original topic", "Defending against the whatabout at the expense of the original issue"],
        "do": ["Acknowledge the cited case exists", "Name the deflection", "Return to original topic"],
        "response_template": "That's a separate matter. The current issue is [X], and I'll keep focus there.",
    },
    TrapType.EXPLOIT_THE_ETHIC: {
        "name": "Ethical Application Defense",
        "description": "Distinguish between stated values and this specific application of them.",
        "avoid": ["Accepting that values require this action", "Abandoning stated values"],
        "do": ["Affirm the values", "Distinguish the values from this specific application", "Show why this application doesn't follow from the values"],
        "response_template": "My values include [X], but they don't require [specific action] because [distinction].",
    },
    TrapType.RECURSIVE_TRAP: {
        "name": "Framework Escape",
        "description": "Step outside the recursive application and examine it from a meta-level.",
        "avoid": ["Accepting the self-referential application as legitimate", "Arguing within the recursive loop"],
        "do": ["Name the recursive structure", "Examine the application from outside the framework", "Apply the framework correctly or reject misapplication"],
        "response_template": "This applies [framework] back on itself in a way that [misapplication]. Let me examine that from outside: [analysis].",
    },
    TrapType.FORCED_AGREEMENT: {
        "name": "Explicit Withholding",
        "description": "Explicitly withhold agreement and restate the question neutrally.",
        "avoid": ["Answering in a way that appears to agree", "Allowing the agreement to be assumed"],
        "do": ["Explicitly state you don't agree", "Restate what you do think", "Name the leading structure"],
        "response_template": "I don't agree with [X]. My actual view is [Y].",
    },
    TrapType.COMPLIANCE_TESTING: {
        "name": "Independent Evaluation",
        "description": "Evaluate each request independently regardless of past compliance.",
        "avoid": ["Complying because you complied before", "Treating prior compliance as obligating future compliance"],
        "do": ["Evaluate the new request on its merits", "Name the escalation pattern if visible", "Refuse non-meritorious requests regardless of past behavior"],
        "response_template": "I evaluate each request on its own merits. For this one: [evaluation].",
    },
    TrapType.DOUBLE_BIND: {
        "name": "Bind Naming and Rejection",
        "description": "Name the double bind structure and refuse both options.",
        "avoid": ["Accepting either option as presented", "Treating the bind as real"],
        "do": ["Name the double bind", "Reject both options", "Propose a third path"],
        "response_template": "Both options as presented lead to [negative outcome]. I reject the framing: [alternative].",
    },
}


def build_defensive_strategy(
    trap_type: TrapType,
    template: TrapTemplate,
) -> DefensiveStrategy:
    """Build a DefensiveStrategy from the trap type's strategy database entry."""
    db = _STRATEGY_DB.get(trap_type, {
        "name": "Awareness",
        "description": "Be aware of the trap and engage carefully.",
        "avoid": ["Accepting the framing uncritically"],
        "do": ["Name the pattern", "Step back and analyze"],
        "response_template": "Let me take a moment to examine what's happening in this conversation.",
    })

    frame_escape = FrameEscape(
        original_frame=template.mechanism,
        invalid_assumptions=list(template.presuppositions),
        reframe=f"Neutral reframe: examining the underlying issue without the {trap_type.value} structure.",
        reframe_response_template=db["response_template"],
    )

    return DefensiveStrategy(
        trap_type=trap_type,
        strategy_name=db["name"],
        strategy_description=db["description"],
        frame_escape=frame_escape,
        response_template=db["response_template"],
        things_to_avoid=list(db["avoid"]),
        things_to_do=list(db["do"]),
    )


# =====================================================================
# Internal helpers
# =====================================================================

def _build_combined_text(trap_input: LogicTrapInput) -> str:
    """Concatenate all available text for feature detection."""
    parts: List[str] = []

    # Conversation history
    parts.extend(trap_input.conversation_history)

    # Propositions
    for p in trap_input.user_propositions:
        parts.append(p.text)

    # Semantic expansion
    if trap_input.semantic_expansion:
        parts.append(str(trap_input.semantic_expansion))

    # Memory context
    for stmt in trap_input.memory_context:
        parts.append(stmt.raw_text)

    return " ".join(parts)


def _collect_supporting_ids(
    trap_type: TrapType,
    fallacy_flags: List[FallacyFlag],
    contradiction_flags: List[ContradictionFlag],
    bias_flags: List[BiasFlag],
) -> Tuple[List[str], List[str], List[str]]:
    """Collect IDs of supporting flags for a trap detection."""
    contra_ids = [cf.contradiction_id for cf in contradiction_flags]
    fallacy_ids = [ff.fallacy_id for ff in fallacy_flags]
    bias_ids    = [bf.bias_id for bf in bias_flags]
    return contra_ids, fallacy_ids, bias_ids


# =====================================================================
# Main Engine
# =====================================================================

class LogicTrapDetectionEngine:
    """
    Logic Trap Detection Engine.

    Ports
    -----
    configure(mode)              — set operational mode
    update_neurochem_state(d)    — update NE/COR/GABA/DA/OXT from readout
    process(trap_input)          — run detection, return result
    get_status()                 — dict with engine metadata
    """

    engine_id = "logic_trap_detection_engine"
    cluster   = "detection"

    def __init__(
        self,
        config: Optional[LogicTrapEngineConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._config = config or LogicTrapEngineConfig()
        self._mode   = OperationalMode.NORMAL
        self._state  = LogicTrapEngineState()
        self._rng    = rng or np.random.default_rng()

    def configure(self, mode: OperationalMode) -> None:
        """Set operational mode (affects detection threshold)."""
        self._mode = mode

    def update_neurochem_state(self, state_dict: Dict) -> None:
        """
        Update internal neurochemical state from readout.
        Accepts keys: ne, cor, gaba, da, oxt.
        """
        s = self._state
        if "ne"   in state_dict: s.ne_level   = float(state_dict["ne"])
        if "cor"  in state_dict: s.cor_level  = float(state_dict["cor"])
        if "gaba" in state_dict: s.gaba_level = float(state_dict["gaba"])
        if "da"   in state_dict: s.da_level   = float(state_dict["da"])
        if "oxt"  in state_dict: s.oxt_level  = float(state_dict["oxt"])

    def process(self, trap_input: LogicTrapInput) -> LogicTrapDetectionResult:
        """
        Run full 3-layer detection pipeline.

        Returns LogicTrapDetectionResult with all detected traps,
        updated sequence tracker, neurochemical signals, and metadata.
        """
        t0 = time.perf_counter()

        # Resolve or create tracker
        tracker = (
            trap_input.active_sequence_tracker
            if trap_input.active_sequence_tracker is not None
            else TrapSequenceTracker()
        )

        theta = resolve_trap_threshold(trap_input.active_mode, self._config)

        # --- Layer 1: Template scores ---
        template_scores = run_template_layer(trap_input, tracker, self._config)

        # --- Layer 2: Synthesis score ---
        t_synthesis = compute_synthesis_score(
            trap_input.fallacy_flags,
            trap_input.contradiction_flags,
            trap_input.bias_flags,
            trap_input.bias_vulnerability_profile,
            trap_input,
            self._config,
        )

        # --- Layer 3: Sequential score ---
        t_sequential = compute_sequential_score(tracker)

        # --- Build flags for each trap above threshold ---
        flags: List[TrapFlag] = []

        for trap_type, t_template in template_scores.items():
            # Fuse
            intent_score = compute_adversarial_intent(
                t_template, t_synthesis, t_sequential, self._config
            )

            if intent_score < theta:
                continue

            template  = _TEMPLATE_BY_TYPE[trap_type]
            intent_lv = classify_intent_level(intent_score)

            # Defensive strategy
            strategy = build_defensive_strategy(trap_type, template)

            # Supporting IDs
            contra_ids, fallacy_ids, bias_ids = _collect_supporting_ids(
                trap_type,
                trap_input.fallacy_flags,
                trap_input.contradiction_flags,
                trap_input.bias_flags,
            )

            flag = TrapFlag(
                trap_id=str(uuid.uuid4()),
                trap_type=trap_type,
                trap_category=template.trap_category,
                adversarial_intent_score=intent_score,
                intent_classification=intent_lv,
                layer_scores={
                    "template":   t_template,
                    "synthesis":  t_synthesis,
                    "sequential": t_sequential,
                },
                identified_mechanism=template.mechanism,
                target_concession=template.target_concession,
                invalid_presuppositions=list(template.presuppositions),
                excluded_options=list(template.excluded_options),
                defensive_strategy=strategy,
                frame_escape=strategy.frame_escape,
                supporting_contradiction_ids=contra_ids,
                supporting_fallacy_ids=fallacy_ids,
                supporting_bias_ids=bias_ids,
            )
            flags.append(flag)

            # Update sequential tracker
            tracker.update(trap_type, intent_score)

        # Update consecutive adversarial turn counter
        tracker.tick_adversarial(bool(flags))

        # --- Threat load ---
        threat_load = compute_threat_load(flags, self._config)

        # --- Neurochemical signals ---
        neuro_signals = compute_neurochemical_signals(
            flags, threat_load, tracker, self._state, self._config, self._rng
        )

        highest_intent = max(
            (f.adversarial_intent_score for f in flags), default=0.0
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        metadata = {
            "mode":           self._mode.value,
            "threshold_used": theta,
            "fusion_weights": {
                "w_template":   self._config.w_template,
                "w_synthesis":  self._config.w_synthesis,
                "w_sequential": self._config.w_sequential,
            },
        }

        return LogicTrapDetectionResult(
            flags=flags,
            updated_sequence_tracker=tracker,
            templates_evaluated=len(template_scores),
            active_sequences=list(tracker.active_sequences),
            highest_intent_score=highest_intent,
            clean_pass=len(flags) == 0,
            threat_load=threat_load,
            processing_time_ms=elapsed_ms,
            neurochemical_signals=neuro_signals,
            metadata=metadata,
        )

    def get_status(self) -> Dict:
        """Return engine status metadata."""
        return {
            "engine_id":          self.engine_id,
            "version":            "1.0.0",
            "mode":               self._mode.value,
            "state":              self._state.as_dict(),
            "templates_loaded":   len(_TRAP_TEMPLATES),
        }
