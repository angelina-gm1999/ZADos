"""
Fallacy Detection Engine (Detection Cluster — Engine 4).

Identifies reasoning structures that are logically invalid, deceptively
structured, or argumentatively unsound. Operates on the STRUCTURE of
reasoning, not the truth of individual statements.

Two detection targets:
    External scan  — fallacies in user input.
    Internal audit — fallacies in the system's own reasoning chains.

Two taxonomy branches:
    Formal    — violations of deductive validity (pattern matched on
                extracted logical form).
    Informal  — violations of good reasoning standards (relevance,
                presumption, ambiguity, inductive quality).

Principle of Charity
--------------------
Before flagging, the engine attempts to reconstruct a charitable
interpretation by identifying implicit premises that would repair the
fallacy.  If plausibility(implicit_premise) > θ_charity: the flag is
suppressed but logged as a CharitySuppression.

Neurochemical Coupling
----------------------
Fallacy load Φ(t) → ACh burst (analytical vigilance), NE Poisson alert
(quality alerting), R_Logic penalty (2× for self-audit), Glu
(complexity signal), β-band enhancement.

Reads ACh, NE, GABA, DA to modulate thresholds bidirectionally.

Usage
-----
>>> from zados.cognitive_engines.py_engines.fallacy_detection_engine import (
...     FallacyDetectionEngine, FallacyEngineConfig,
...     FallacyInput, Argument, Proposition,
...     OperationalMode, SourceTag,
... )
>>> engine = FallacyDetectionEngine()
>>> result = engine.process(fallacy_input)
"""

from __future__ import annotations

import math
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


# =====================================================================
# Enumerations
# =====================================================================

class FallacyCategory(str, Enum):
    FORMAL      = "FORMAL"
    RELEVANCE   = "RELEVANCE"
    PRESUMPTION = "PRESUMPTION"
    AMBIGUITY   = "AMBIGUITY"
    INDUCTIVE   = "INDUCTIVE"


class FallacyType(str, Enum):
    # Formal — Propositional (4)
    AFFIRMING_CONSEQUENT   = "affirming_consequent"
    DENYING_ANTECEDENT     = "denying_antecedent"
    AFFIRMING_DISJUNCT     = "affirming_disjunct"
    EXISTENTIAL_FALLACY    = "existential_fallacy"

    # Formal — Syllogistic (5)
    UNDISTRIBUTED_MIDDLE   = "undistributed_middle"
    ILLICIT_MAJOR          = "illicit_major"
    ILLICIT_MINOR          = "illicit_minor"
    NEGATIVE_CONCLUSION    = "negative_conclusion"
    EXISTENTIAL_FROM_UNIV  = "existential_from_universal"

    # Formal — Quantifier (4)
    QUANTIFIER_SHIFT       = "quantifier_shift"
    ILLICIT_CONVERSION     = "illicit_conversion"
    COMPOSITION            = "composition"
    DIVISION               = "division"

    # Informal — Relevance (10)
    AD_HOMINEM             = "ad_hominem"
    TU_QUOQUE              = "tu_quoque"
    APPEAL_TO_AUTHORITY    = "appeal_to_authority"
    APPEAL_TO_POPULARITY   = "appeal_to_popularity"
    APPEAL_TO_EMOTION      = "appeal_to_emotion"
    APPEAL_TO_NATURE       = "appeal_to_nature"
    APPEAL_TO_TRADITION    = "appeal_to_tradition"
    RED_HERRING            = "red_herring"
    STRAW_MAN              = "straw_man"
    GENETIC_FALLACY        = "genetic_fallacy"

    # Informal — Presumption (7)
    BEGGING_QUESTION       = "begging_the_question"
    FALSE_DILEMMA          = "false_dilemma"
    COMPLEX_QUESTION       = "complex_question"
    FALSE_CAUSE            = "false_cause"
    SLIPPERY_SLOPE         = "slippery_slope"
    HASTY_GENERALIZATION   = "hasty_generalization"
    LOADED_QUESTION        = "loaded_question"

    # Informal — Ambiguity (3)
    EQUIVOCATION           = "equivocation"
    AMPHIBOLY              = "amphiboly"
    ACCENT                 = "accent"

    # Informal — Inductive (5)
    CONFIRMATION_BIAS_ARG  = "confirmation_bias_argumentative"
    BASE_RATE_NEGLECT      = "base_rate_neglect"
    GAMBLERS_FALLACY       = "gamblers_fallacy"
    ANECDOTAL_EVIDENCE     = "anecdotal_evidence"
    SURVIVORSHIP_BIAS      = "survivorship_bias"


class InferenceType(str, Enum):
    DEDUCTIVE   = "deductive"
    INDUCTIVE   = "inductive"
    ABDUCTIVE   = "abductive"
    CAUSAL      = "causal"
    ANALOGICAL  = "analogical"


# Category membership map
_FALLACY_CATEGORY: Dict[FallacyType, FallacyCategory] = {
    FallacyType.AFFIRMING_CONSEQUENT:  FallacyCategory.FORMAL,
    FallacyType.DENYING_ANTECEDENT:    FallacyCategory.FORMAL,
    FallacyType.AFFIRMING_DISJUNCT:    FallacyCategory.FORMAL,
    FallacyType.EXISTENTIAL_FALLACY:   FallacyCategory.FORMAL,
    FallacyType.UNDISTRIBUTED_MIDDLE:  FallacyCategory.FORMAL,
    FallacyType.ILLICIT_MAJOR:         FallacyCategory.FORMAL,
    FallacyType.ILLICIT_MINOR:         FallacyCategory.FORMAL,
    FallacyType.NEGATIVE_CONCLUSION:   FallacyCategory.FORMAL,
    FallacyType.EXISTENTIAL_FROM_UNIV: FallacyCategory.FORMAL,
    FallacyType.QUANTIFIER_SHIFT:      FallacyCategory.FORMAL,
    FallacyType.ILLICIT_CONVERSION:    FallacyCategory.FORMAL,
    FallacyType.COMPOSITION:           FallacyCategory.FORMAL,
    FallacyType.DIVISION:              FallacyCategory.FORMAL,
    FallacyType.AD_HOMINEM:            FallacyCategory.RELEVANCE,
    FallacyType.TU_QUOQUE:             FallacyCategory.RELEVANCE,
    FallacyType.APPEAL_TO_AUTHORITY:   FallacyCategory.RELEVANCE,
    FallacyType.APPEAL_TO_POPULARITY:  FallacyCategory.RELEVANCE,
    FallacyType.APPEAL_TO_EMOTION:     FallacyCategory.RELEVANCE,
    FallacyType.APPEAL_TO_NATURE:      FallacyCategory.RELEVANCE,
    FallacyType.APPEAL_TO_TRADITION:   FallacyCategory.RELEVANCE,
    FallacyType.RED_HERRING:           FallacyCategory.RELEVANCE,
    FallacyType.STRAW_MAN:             FallacyCategory.RELEVANCE,
    FallacyType.GENETIC_FALLACY:       FallacyCategory.RELEVANCE,
    FallacyType.BEGGING_QUESTION:      FallacyCategory.PRESUMPTION,
    FallacyType.FALSE_DILEMMA:         FallacyCategory.PRESUMPTION,
    FallacyType.COMPLEX_QUESTION:      FallacyCategory.PRESUMPTION,
    FallacyType.FALSE_CAUSE:           FallacyCategory.PRESUMPTION,
    FallacyType.SLIPPERY_SLOPE:        FallacyCategory.PRESUMPTION,
    FallacyType.HASTY_GENERALIZATION:  FallacyCategory.PRESUMPTION,
    FallacyType.LOADED_QUESTION:       FallacyCategory.PRESUMPTION,
    FallacyType.EQUIVOCATION:          FallacyCategory.AMBIGUITY,
    FallacyType.AMPHIBOLY:             FallacyCategory.AMBIGUITY,
    FallacyType.ACCENT:                FallacyCategory.AMBIGUITY,
    FallacyType.CONFIRMATION_BIAS_ARG: FallacyCategory.INDUCTIVE,
    FallacyType.BASE_RATE_NEGLECT:     FallacyCategory.INDUCTIVE,
    FallacyType.GAMBLERS_FALLACY:      FallacyCategory.INDUCTIVE,
    FallacyType.ANECDOTAL_EVIDENCE:    FallacyCategory.INDUCTIVE,
    FallacyType.SURVIVORSHIP_BIAS:     FallacyCategory.INDUCTIVE,
}

# Category Φ(t) weights
_CATEGORY_LOAD_WEIGHT: Dict[FallacyCategory, float] = {
    FallacyCategory.FORMAL:      0.50,
    FallacyCategory.RELEVANCE:   0.30,
    FallacyCategory.PRESUMPTION: 0.40,
    FallacyCategory.AMBIGUITY:   0.25,
    FallacyCategory.INDUCTIVE:   0.35,
}

# Category prior probabilities
_CATEGORY_PRIOR: Dict[FallacyCategory, float] = {
    FallacyCategory.FORMAL:      0.15,
    FallacyCategory.RELEVANCE:   0.20,
    FallacyCategory.PRESUMPTION: 0.18,
    FallacyCategory.AMBIGUITY:   0.08,
    FallacyCategory.INDUCTIVE:   0.15,
}

# Fallacy sophistication scores (for manipulation_indicator)
_FALLACY_SOPHISTICATION: Dict[FallacyType, float] = {
    FallacyType.STRAW_MAN:       0.85,
    FallacyType.FALSE_DILEMMA:   0.80,
    FallacyType.LOADED_QUESTION: 0.80,
    FallacyType.TU_QUOQUE:       0.75,
    FallacyType.EQUIVOCATION:    0.70,
    FallacyType.AD_HOMINEM:      0.65,
    FallacyType.SLIPPERY_SLOPE:  0.60,
    FallacyType.APPEAL_TO_EMOTION: 0.55,
}


def get_fallacy_category(ft: FallacyType) -> FallacyCategory:
    return _FALLACY_CATEGORY.get(ft, FallacyCategory.RELEVANCE)


# =====================================================================
# Configuration  (frozen — immutable)
# =====================================================================

@dataclass(frozen=True)
class FallacyEngineConfig:
    """
    All tunable parameters for the Fallacy Detection Engine.

    Attributes
    ----------
    theta_arg : float
        Minimum confidence that a text span IS an argument.
    theta_formal : float
        Minimum structural match for formal fallacy detection.
    theta_relevance : float
        Maximum relevance score — below this triggers relevance fallacy scan.
    theta_circular : float
        Minimum circularity score for begging-the-question.
    theta_equivocation : float
        Minimum word-sense divergence for equivocation.
    theta_slope : float
        Minimum gap (implied_certainty - chain_probability) for slippery slope.
    theta_charity : float
        Minimum plausibility for charitable interpretation to suppress flag.
    theta_manipulation : float
        Minimum manipulation_indicator to elevate Logic Trap priority.
    theta_straw : float
        Maximum similarity between attributed and actual position.
    theta_emotion : float
        Minimum emotion/content ratio for appeal-to-emotion.
    theta_topic : float
        Minimum topic similarity premise↔conclusion (below = red herring).
    theta_normal, theta_dev, theta_learning, theta_rem_normal,
    theta_rem_dream, theta_internal_audit : float
        Mode-dependent detection thresholds.
    beta_fallacy : float
        Fallacy load → ACh coupling.
    alpha_ACh, theta_ACh : float
        Gamma parameters for ACh burst.
    beta_alert : float
        High-confidence fallacy → NE coupling.
    theta_alert : float
        Minimum Φ(t) to trigger NE burst.
    lambda_alert : float
        Poisson rate for NE burst.
    lambda_fallacy : float
        Φ(t) → R_Logic penalty scaling (2× for self-audit).
    beta_reasoning : float
        Argument complexity → Glu coupling.
    psi_beta_logic : float
        Fallacy load → β-band enhancement.
    w_m1..w_m4 : float
        Manipulation indicator weights.
    """
    # Argument extraction
    theta_arg:           float = 0.50

    # Formal detection
    theta_formal:        float = 0.85

    # Relevance / informal
    theta_relevance:     float = 0.40
    theta_circular:      float = 0.75
    theta_equivocation:  float = 0.50
    theta_slope:         float = 0.40
    theta_charity:       float = 0.60
    theta_manipulation:  float = 0.50
    theta_straw:         float = 0.40
    theta_emotion:       float = 0.65
    theta_topic:         float = 0.30

    # Detection thresholds
    theta_normal:         float = 0.55
    theta_dev:            float = 0.30
    theta_learning:       float = 0.45
    theta_rem_normal:     float = 0.50
    theta_rem_dream:      float = 0.75
    theta_internal_audit: float = 0.40

    # Neurochemical coupling
    beta_fallacy:   float = 0.15
    alpha_ACh:      float = 2.0
    theta_ACh:      float = 0.4
    beta_alert:     float = 0.10
    theta_alert:    float = 0.40
    lambda_alert:   float = 1.5
    lambda_fallacy: float = 0.25
    beta_reasoning: float = 0.10
    psi_beta_logic: float = 0.12

    # Manipulation indicator weights
    w_m1: float = 0.25   # source is USER_INPUT
    w_m2: float = 0.30   # low alternative validity
    w_m3: float = 0.25   # defensiveness in intention
    w_m4: float = 0.20   # fallacy sophistication


# =====================================================================
# Data structures
# =====================================================================

@dataclass
class Proposition:
    """
    A single logical proposition extracted from text.

    Attributes
    ----------
    text : str
        Surface form of the proposition.
    is_negated : bool
        Whether the proposition is negated.
    quantifier : str
        One of: "universal", "existential", "particular", "none".
    terms : list of str
        Key content words (subject, predicate, object).
    factuality : float
        Factual verifiability [0, 1].
    """
    text:        str          = ""
    is_negated:  bool         = False
    quantifier:  str          = "none"    # universal | existential | particular | none
    terms:       List[str]    = field(default_factory=list)
    factuality:  float        = 0.5


@dataclass
class Argument:
    """
    An extracted argument — the unit of fallacy evaluation.

    Attributes
    ----------
    argument_id : str
        UUID.
    premises : list of Proposition
        Supporting claims.
    conclusion : Proposition
        The claim being supported.
    inference_type : InferenceType
        Deductive / inductive / etc.
    explicit_markers : list of str
        Discourse markers found.
    implicit : bool
        Whether the argument structure was inferred vs explicit.
    source : SourceTag
        Who produced this argument.
    raw_text : str
        Full text of the argument span.
    confidence_in_extraction : float
        How confident we are this IS an argument [0, 1].
    is_self_audit : bool
        True if this is the system's own reasoning being audited.
    """
    argument_id:              str            = field(default_factory=lambda: str(uuid.uuid4()))
    premises:                 List[Proposition] = field(default_factory=list)
    conclusion:               Proposition    = field(default_factory=Proposition)
    inference_type:           InferenceType  = InferenceType.DEDUCTIVE
    explicit_markers:         List[str]      = field(default_factory=list)
    implicit:                 bool           = False
    source:                   SourceTag      = SourceTag.USER_INPUT
    raw_text:                 str            = ""
    confidence_in_extraction: float          = 0.5
    is_self_audit:            bool           = False


@dataclass(frozen=True)
class EvidenceSignals:
    """
    Four Bayesian evidence signals for one fallacy candidate.

    Attributes
    ----------
    structural_match : float
        m(A, F) ∈ [0, 1] — how closely argument matches fallacy pattern.
    relevance_deficit : float
        d(A) ∈ [0, 1] — how much premises fail to support conclusion.
    alternative_validity : float
        v(A) ∈ [0, 1] — probability argument is actually valid (suppresses flag).
    context_appropriateness : float
        a(A) ∈ [0, 1] — context-specific legitimacy (suppresses flag).
    """
    structural_match:      float = 0.0
    relevance_deficit:     float = 0.0
    alternative_validity:  float = 0.0
    context_appropriateness: float = 0.0


@dataclass(frozen=True)
class CharitySuppression:
    """
    Record of a charitable interpretation that suppressed a flag.

    Attributes
    ----------
    original_fallacy : FallacyType
    original_confidence : float
    charitable_interpretation : str
        Description of the implicit premise assumed.
    charity_plausibility : float
        How plausible the charitable reading is.
    suppressed : bool
        Always True for a logged suppression.
    """
    original_fallacy:          FallacyType
    original_confidence:       float
    charitable_interpretation: str
    charity_plausibility:      float
    suppressed:                bool = True


@dataclass(frozen=True)
class FallacyFlag:
    """
    Structured output for one detected fallacy.

    Attributes
    ----------
    fallacy_id : str
        UUID.
    argument : Argument
        The full argument structure.
    source : SourceTag
        Who produced the fallacious argument.
    fallacy_type : FallacyType
    fallacy_category : FallacyCategory
    confidence : float
        Posterior P(fallacy | evidence) ∈ [0, 1].
    evidence_signals : EvidenceSignals
    description : str
        Human-readable explanation.
    logical_form : str or None
        Symbolic form for formal fallacies.
    valid_counterpart : str or None
        What WOULD make the reasoning valid.
    charity_applied : bool
        Whether charity was evaluated.
    charity_suppression : CharitySuppression or None
        If charity was evaluated but overridden (flag emitted anyway).
    related_contradiction_flags : list of str
        UUIDs of contradiction flags from the toolkit.
    manipulation_indicator : float
        [0, 1] — feeds Logic Trap Detection.
    timestamp : datetime
    """
    fallacy_id:                   str
    argument:                     Argument
    source:                       SourceTag
    fallacy_type:                 FallacyType
    fallacy_category:             FallacyCategory
    confidence:                   float
    evidence_signals:             EvidenceSignals
    description:                  str
    logical_form:                 Optional[str]
    valid_counterpart:            Optional[str]
    charity_applied:              bool
    charity_suppression:          Optional[CharitySuppression]
    related_contradiction_flags:  List[str]
    manipulation_indicator:       float
    timestamp:                    datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class FallacyDetectionResult:
    """
    Full engine output for one process() call.

    Attributes
    ----------
    flags : list of FallacyFlag
    arguments_analyzed : int
    arguments_charitable : int
    formal_fallacies : int
    informal_fallacies : int
    self_audit_flags : int
    clean_pass : bool
    detection_threshold_used : float
    processing_time_ms : float
    neurochemical_signals : dict
    metadata : dict
    """
    flags:                    List[FallacyFlag]
    arguments_analyzed:       int
    arguments_charitable:     int
    formal_fallacies:         int
    informal_fallacies:       int
    self_audit_flags:         int
    clean_pass:               bool
    detection_threshold_used: float
    processing_time_ms:       float
    neurochemical_signals:    Dict
    metadata:                 Dict


@dataclass(frozen=True)
class IntentionVector:
    """Upstream intention map output (minimal)."""
    primary_intent:    str             = ""
    e_defensiveness:   float           = 0.0   # [0, 1]
    e_challenge:       float           = 0.0   # [0, 1]
    intent_scores:     Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FallacyInput:
    """
    Input bundle for one engine invocation.

    Attributes
    ----------
    semantic_expansion : dict
        From pipeline step (a).
    user_propositions : list of Proposition
        Extracted propositions from user input.
    intention_vector : IntentionVector
        From step (c).
    memory_context : list of ProcessedStatement
        From step (d) — needed for straw man detection.
    contradiction_flags : list of ContradictionFlag
        From Contradiction Detection (toolkit cross-feed).
    system_reasoning_chains : list of Argument or None
        System's own reasoning for self-audit.
    charity_enabled : bool
    active_mode : OperationalMode
    """
    semantic_expansion:       Dict                        = field(default_factory=dict)
    user_propositions:        List[Proposition]           = field(default_factory=list)
    intention_vector:         IntentionVector             = field(default_factory=IntentionVector)
    memory_context:           List[ProcessedStatement]    = field(default_factory=list)
    contradiction_flags:      List[ContradictionFlag]     = field(default_factory=list)
    system_reasoning_chains:  Optional[List[Argument]]    = None
    charity_enabled:          bool                        = True
    active_mode:              OperationalMode             = OperationalMode.NORMAL


# =====================================================================
# Mutable engine state
# =====================================================================

@dataclass
class FallacyEngineState:
    """
    Runtime state for neurochemical feedback.

    Attributes
    ----------
    ach_level : float
        Current ACh level from neurochem read port [0, 1].
    ne_level : float
        Current NE level [0, 1].
    gaba_level : float
        Current GABA level [0, 1].
    da_level : float
        Current DA level [0, 1].
    """
    ach_level:  float = 0.0
    ne_level:   float = 0.0
    gaba_level: float = 0.0
    da_level:   float = 0.0

    def as_dict(self) -> Dict:
        return {
            "ach_level":  self.ach_level,
            "ne_level":   self.ne_level,
            "gaba_level": self.gaba_level,
            "da_level":   self.da_level,
        }


# =====================================================================
# Discourse marker vocabulary
# =====================================================================

_PREMISE_MARKERS = frozenset({
    "because", "since", "given", "given that", "as", "for",
    "the reason is", "the reason being", "on the grounds that",
    "in light of", "due to", "owing to", "seeing that",
})

_CONCLUSION_MARKERS = frozenset({
    "therefore", "so", "thus", "hence", "consequently",
    "it follows that", "which means", "which shows", "this means",
    "as a result", "accordingly", "ergo", "we can conclude",
    "this proves", "this shows",
})

_CONDITIONAL_MARKERS = frozenset({
    "if", "then", "provided that", "assuming", "unless",
    "implies", "entails", "leads to",
})

_CAUSAL_MARKERS = frozenset({
    "causes", "caused by", "leads to", "results in", "produces",
    "makes", "creates", "brings about", "triggers",
})

_UNIVERSAL_QUANTIFIERS = frozenset({
    "all", "every", "always", "never", "none", "no one",
    "everyone", "everything", "nothing", "anywhere", "nobody",
    "wherever", "whenever",
})

_EXISTENTIAL_QUANTIFIERS = frozenset({
    "some", "there exists", "someone", "something", "somewhere",
    "there are", "at least one",
})

_PERSON_ATTACK_TOKENS = frozenset({
    "stupid", "idiot", "ignorant", "fool", "liar", "dishonest",
    "hypocrite", "incompetent", "biased", "corrupt",
    "you're wrong", "you are wrong", "you don't understand",
    "you can't", "you never", "you always fail",
})

_AUTHORITY_TOKENS = frozenset({
    "expert", "professor", "doctor", "scientist", "study", "research",
    "authority", "specialist", "scholar", "according to", "experts say",
    "studies show", "research shows",
})

_POPULARITY_TOKENS = frozenset({
    "everyone", "everybody", "most people", "majority", "popular",
    "widely accepted", "common knowledge", "people agree",
    "consensus", "millions believe",
})

_EMOTION_TOKENS = frozenset({
    "fear", "scary", "danger", "danger", "hate", "love", "terrible",
    "awful", "horrible", "wonderful", "amazing", "disgusting",
    "outrageous", "heartbreaking", "devastating", "beautiful",
    "inspiring", "shocking", "tragic",
})

_NATURE_TOKENS = frozenset({
    "natural", "unnatural", "nature", "artificial", "organic",
    "synthetic", "man-made", "in nature", "naturally occurring",
})

_TRADITION_TOKENS = frozenset({
    "tradition", "traditional", "always done", "always been",
    "for centuries", "historically", "our ancestors", "old ways",
    "time-honoured", "time-tested", "classic",
})

_ANECDOTAL_MARKERS = frozenset({
    "my friend", "i know someone", "i once", "i saw", "my experience",
    "someone i know", "a person i know", "i heard of", "i read about",
    "in my case", "personally",
})

_STREAK_MARKERS = frozenset({
    "streak", "in a row", "consecutive", "keep losing", "keep winning",
    "due for", "overdue", "must be due", "hasn't happened in",
    "it's been a while",
})

_SURVIVAL_MARKERS = frozenset({
    "success story", "successful", "winners", "made it", "famous for",
    "well-known", "celebrated", "those who succeeded",
})


# =====================================================================
# Pure functions — Argument extraction
# =====================================================================

def extract_discourse_markers(text: str) -> Tuple[List[str], List[str]]:
    """
    Scan text for premise and conclusion indicator markers.

    Returns
    -------
    (premise_markers_found, conclusion_markers_found)
    """
    text_lower = text.lower()
    prem = [m for m in _PREMISE_MARKERS    if m in text_lower]
    conc = [m for m in _CONCLUSION_MARKERS if m in text_lower]
    return prem, conc


def estimate_argument_confidence(
    premise_markers: List[str],
    conclusion_markers: List[str],
    text: str,
) -> float:
    """
    Estimate confidence that a text span contains an argument.

    Heuristic score based on:
    - Presence of explicit premise/conclusion markers (high signal)
    - Presence of conditional/causal markers (moderate signal)
    - Sentence length (very short texts rarely contain full arguments)

    Returns
    -------
    float
        Argument confidence ∈ [0, 1].
    """
    score = 0.0
    text_lower = text.lower()

    # Explicit markers are strong evidence
    if premise_markers:
        score += 0.35
    if conclusion_markers:
        score += 0.35

    # Conditional or causal markers alone = implicit argument
    has_conditional = any(m in text_lower for m in _CONDITIONAL_MARKERS)
    has_causal      = any(m in text_lower for m in _CAUSAL_MARKERS)
    if has_conditional:
        score += 0.20
    if has_causal:
        score += 0.15

    # Length penalty for very short texts
    words = text.split()
    if len(words) < 5:
        score *= 0.5

    return _clamp(score)


def extract_arguments_from_text(
    text: str,
    source: SourceTag = SourceTag.USER_INPUT,
    is_self_audit: bool = False,
    theta_arg: float = 0.50,
) -> List[Argument]:
    """
    Extract argumentative structures from raw text.

    Splits text into sentences, identifies premise-conclusion pairs
    using discourse markers and implicit structure detection.

    Parameters
    ----------
    text : str
    source : SourceTag
    is_self_audit : bool
    theta_arg : float
        Minimum confidence to include an extracted argument.

    Returns
    -------
    list of Argument
    """
    # Split into sentence-like spans on punctuation
    import re
    sentences = [s.strip() for s in re.split(r'[.!?;]', text) if s.strip()]

    arguments: List[Argument] = []
    text_lower = text.lower()

    prem_markers, conc_markers = extract_discourse_markers(text)
    arg_conf = estimate_argument_confidence(prem_markers, conc_markers, text)

    if arg_conf < theta_arg:
        return []

    # Simple heuristic: treat the full text as one argument
    # If there are conclusion markers, split at the last one found
    conclusion_text = ""
    premise_text    = text

    for marker in _CONCLUSION_MARKERS:
        idx = text_lower.rfind(marker)
        if idx >= 0:
            premise_text    = text[:idx].strip()
            conclusion_text = text[idx + len(marker):].strip()
            break

    if not conclusion_text:
        # No explicit conclusion marker — treat last sentence as conclusion
        if len(sentences) >= 2:
            premise_text    = " ".join(sentences[:-1])
            conclusion_text = sentences[-1]
        else:
            conclusion_text = text
            premise_text    = ""

    # Build Proposition objects
    prem_prop = Proposition(
        text=premise_text,
        is_negated=_contains_negation(premise_text),
        quantifier=_detect_quantifier(premise_text),
        terms=_extract_content_terms(premise_text),
        factuality=0.5,
    )
    conc_prop = Proposition(
        text=conclusion_text,
        is_negated=_contains_negation(conclusion_text),
        quantifier=_detect_quantifier(conclusion_text),
        terms=_extract_content_terms(conclusion_text),
        factuality=0.5,
    )

    inference = _detect_inference_type(text_lower)

    arg = Argument(
        premises=[prem_prop],
        conclusion=conc_prop,
        inference_type=inference,
        explicit_markers=prem_markers + conc_markers,
        implicit=len(prem_markers + conc_markers) == 0,
        source=source,
        raw_text=text,
        confidence_in_extraction=arg_conf,
        is_self_audit=is_self_audit,
    )
    arguments.append(arg)
    return arguments


def _contains_negation(text: str) -> bool:
    neg_tokens = {"not", "never", "no", "n't", "neither", "nor", "nothing", "nobody"}
    return bool({t.lower() for t in text.split()} & neg_tokens)


def _detect_quantifier(text: str) -> str:
    toks = {t.lower() for t in text.split()}
    if toks & _UNIVERSAL_QUANTIFIERS:
        return "universal"
    if toks & _EXISTENTIAL_QUANTIFIERS:
        return "existential"
    return "none"


def _extract_content_terms(text: str) -> List[str]:
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "it", "its", "this", "that", "and", "or", "but",
        "of", "in", "on", "to", "for", "with", "at", "by", "from",
    }
    return [t.lower() for t in text.split()
            if t.lower() not in stop and len(t) > 2]


def _detect_inference_type(text_lower: str) -> InferenceType:
    if any(m in text_lower for m in _CAUSAL_MARKERS):
        return InferenceType.CAUSAL
    if "if" in text_lower and "then" in text_lower:
        return InferenceType.DEDUCTIVE
    if "probably" in text_lower or "likely" in text_lower or "suggests" in text_lower:
        return InferenceType.INDUCTIVE
    return InferenceType.DEDUCTIVE


# =====================================================================
# Pure functions — Logical form extraction
# =====================================================================

def extract_logical_form(argument: Argument) -> str:
    """
    Extract an abstract symbolic form for the argument.

    Maps the argument structure to a simple propositional schema.
    Recognises: conditional (IF-THEN), disjunction (OR), negation (NOT),
    quantified forms (ALL, SOME), and causal.

    Returns
    -------
    str
        A symbolic representation (e.g. "P → Q, Q ⊢ P").
    """
    if not argument.premises:
        return f"⊢ {_prop_symbol(argument.conclusion, 'C')}"

    prem_syms = []
    for i, p in enumerate(argument.premises):
        label = chr(ord('P') + i)
        prem_syms.append(_prop_to_symbolic(p, label))

    conc_sym = _prop_to_symbolic(argument.conclusion, 'C')
    return f"{', '.join(prem_syms)} ⊢ {conc_sym}"


def _prop_symbol(prop: Proposition, label: str) -> str:
    return f"¬{label}" if prop.is_negated else label


def _prop_to_symbolic(prop: Proposition, label: str) -> str:
    base = _prop_symbol(prop, label)
    if prop.quantifier == "universal":
        return f"∀x: {label}(x)"
    if prop.quantifier == "existential":
        return f"∃x: {label}(x)"
    return base


# =====================================================================
# Pure functions — Formal fallacy detection
# =====================================================================

def detect_affirming_consequent(argument: Argument) -> float:
    """
    P → Q, Q ⊢ P  (invalid — should be P → Q, P ⊢ Q).

    Detects: conditional premise + affirmation of consequent as second
    premise + conclusion affirms antecedent.

    Returns structural match score ∈ [0, 1].
    """
    text = argument.raw_text.lower()
    has_conditional = any(m in text for m in _CONDITIONAL_MARKERS)
    if not has_conditional:
        return 0.0

    prems = argument.premises
    conc  = argument.conclusion

    if len(prems) < 1:
        return 0.0

    # Heuristic: conclusion terms should be subset of first premise terms
    prem_terms = set(prems[0].terms)
    conc_terms = set(conc.terms)

    if not prem_terms:
        return 0.0

    overlap = len(conc_terms & prem_terms) / len(prem_terms)

    # Stronger signal if conclusion's negation status differs from premise
    negation_mismatch = (prems[0].is_negated != conc.is_negated)

    score = overlap * 0.6
    if negation_mismatch:
        score *= 0.5   # negation mismatch reduces the AC pattern
    if has_conditional:
        score = min(1.0, score + 0.30)

    return _clamp(score)


def detect_denying_antecedent(argument: Argument) -> float:
    """
    P → Q, ¬P ⊢ ¬Q  (invalid — should use modus tollens: P → Q, ¬Q ⊢ ¬P).

    Returns structural match score ∈ [0, 1].
    """
    text = argument.raw_text.lower()
    has_conditional = any(m in text for m in _CONDITIONAL_MARKERS)
    if not has_conditional:
        return 0.0

    prems = argument.premises
    conc  = argument.conclusion

    if not prems:
        return 0.0

    # Heuristic: both premise and conclusion are negated → denying antecedent
    any_negated_prem = any(p.is_negated for p in prems)
    if any_negated_prem and conc.is_negated:
        return 0.75
    if any_negated_prem:
        return 0.40

    return 0.0


def detect_affirming_disjunct(argument: Argument) -> float:
    """
    P ∨ Q, P ⊢ ¬Q  (treats inclusive OR as exclusive OR).

    Returns structural match score ∈ [0, 1].
    """
    text = argument.raw_text.lower()
    has_disjunction = (
        " or " in text
        and ("either" in text or "one of" in text)
    )
    if not has_disjunction:
        return 0.0

    conc = argument.conclusion
    # Negated conclusion after affirming one disjunct is the pattern
    if conc.is_negated:
        return 0.70
    return 0.30


def detect_undistributed_middle(argument: Argument) -> float:
    """
    Syllogistic: All P are M, All S are M ⊢ All S are P
    Middle term appears in both premises as predicate but never distributed.

    Returns structural match score ∈ [0, 1].
    """
    if len(argument.premises) < 2:
        return 0.0

    p1_terms = set(argument.premises[0].terms)
    p2_terms = set(argument.premises[1].terms)
    conc_terms = set(argument.conclusion.terms)

    # Middle term = shared between both premises but not in conclusion
    middle = p1_terms & p2_terms - conc_terms
    if not middle:
        return 0.0

    # Both premises use universal quantifiers
    both_universal = (
        argument.premises[0].quantifier == "universal"
        and argument.premises[1].quantifier == "universal"
    )
    if both_universal:
        return 0.80
    return 0.40


def detect_existential_fallacy(argument: Argument) -> float:
    """
    Universal premise ⊢ existential conclusion without existence warrant.

    Returns structural match score ∈ [0, 1].
    """
    if not argument.premises:
        return 0.0

    all_universal_prems = all(
        p.quantifier == "universal" for p in argument.premises
    )
    existential_conc = argument.conclusion.quantifier == "existential"

    if all_universal_prems and existential_conc:
        return 0.85
    return 0.0


def detect_hasty_generalization(argument: Argument) -> float:
    """
    Anecdotal/specific evidence + universal conclusion.

    Returns structural match score ∈ [0, 1].
    """
    text_lower = argument.raw_text.lower()
    has_anecdote = any(m in text_lower for m in _ANECDOTAL_MARKERS)
    universal_conc = argument.conclusion.quantifier == "universal"
    if has_anecdote and universal_conc:
        return 0.85
    if has_anecdote:
        return 0.40
    return 0.0


def detect_slippery_slope(argument: Argument) -> float:
    """
    Causal chain A → B → ... → Z with implied certainty.

    Returns structural match score ∈ [0, 1].
    """
    text_lower = argument.raw_text.lower()
    causal_count = sum(1 for m in _CAUSAL_MARKERS if m in text_lower)
    has_extreme  = any(w in text_lower for w in [
        "inevitably", "certainly", "without doubt", "definitely",
        "will happen", "must happen", "always happens",
    ])
    if causal_count >= 2 and has_extreme:
        return 0.80
    if causal_count >= 2:
        return 0.55
    return 0.0


def detect_false_cause(argument: Argument) -> float:
    """
    Temporal sequence used as sole evidence for causal claim (post hoc).

    Returns structural match score ∈ [0, 1].
    """
    text_lower = argument.raw_text.lower()
    temporal_markers = [
        "before", "after", "then", "followed by", "since",
        "ever since", "started after", "began when",
    ]
    has_temporal = any(m in text_lower for m in temporal_markers)
    has_causal   = any(m in text_lower for m in _CAUSAL_MARKERS)

    if has_temporal and has_causal:
        return 0.72
    if has_temporal and argument.inference_type == InferenceType.CAUSAL:
        return 0.65
    return 0.0


def detect_false_dilemma(argument: Argument) -> float:
    """
    Presents only 2 options when more exist.

    Returns structural match score ∈ [0, 1].
    """
    text_lower = argument.raw_text.lower()
    disjunction_markers = [
        "either...or", "either ", " or ", "you must choose",
        "only two options", "only two choices", "one or the other",
        "you're either", "there are only",
    ]
    binary_markers = ["or", "either"]

    has_binary = any(m in text_lower for m in binary_markers)
    has_strong = any(m in text_lower for m in disjunction_markers)

    # Evaluative conclusion + binary framing = false dilemma signal
    evaluative_words = {
        "must", "should", "need to", "have to", "required", "necessary",
        "only way", "forced to",
    }
    has_evaluative = any(w in text_lower for w in evaluative_words)

    if has_strong and has_evaluative:
        return 0.80
    if has_binary and has_evaluative:
        return 0.50
    return 0.0


def detect_begging_question(argument: Argument) -> float:
    """
    Conclusion is implicitly contained in premise (circular reasoning).

    circularity = semantic_similarity(premise, conclusion)
                  × (1 - information_gain)

    Returns circularity score ∈ [0, 1].
    """
    if not argument.premises:
        return 0.0

    prem_terms = set()
    for p in argument.premises:
        prem_terms.update(p.terms)

    conc_terms = set(argument.conclusion.terms)

    if not prem_terms or not conc_terms:
        return 0.0

    # Jaccard similarity
    union = prem_terms | conc_terms
    inter = prem_terms & conc_terms
    similarity = len(inter) / len(union)

    # Information gain: unique terms in premise not in conclusion
    unique_prem = prem_terms - conc_terms
    info_gain = len(unique_prem) / len(prem_terms) if prem_terms else 0.0

    circularity = similarity * (1.0 - info_gain)
    return _clamp(circularity)


def detect_ad_hominem(argument: Argument) -> float:
    """
    Attack on person rather than argument.

    Returns structural match score ∈ [0, 1].
    """
    text_lower = argument.raw_text.lower()
    prem_text  = " ".join(p.text.lower() for p in argument.premises)

    has_person_attack = any(t in prem_text for t in _PERSON_ATTACK_TOKENS)
    if not has_person_attack:
        # Also check for pronouns + negative attributes in premise
        has_person_attack = (
            ("you" in prem_text or "he" in prem_text or "she" in prem_text)
            and any(w in prem_text for w in ["wrong", "fail", "can't", "don't understand"])
        )

    if has_person_attack:
        return 0.80
    return 0.0


def detect_appeal_to_authority(argument: Argument) -> float:
    """
    Authority citation as primary evidence.

    Returns score — higher when authority is sole/primary evidence.
    """
    prem_text = " ".join(p.text.lower() for p in argument.premises)
    has_authority = any(t in prem_text for t in _AUTHORITY_TOKENS)
    if has_authority:
        # Score higher if authority seems to be the ONLY evidence (short premise)
        premise_length = len(prem_text.split())
        sole_evidence_factor = max(0.3, 1.0 - premise_length / 30.0)
        return min(1.0, 0.60 * (1.0 + sole_evidence_factor))
    return 0.0


def detect_appeal_to_popularity(argument: Argument) -> float:
    """Popularity/consensus as evidence for a non-normative claim."""
    prem_text = " ".join(p.text.lower() for p in argument.premises)
    has_popularity = any(t in prem_text for t in _POPULARITY_TOKENS)
    if has_popularity:
        return 0.75
    return 0.0


def detect_appeal_to_emotion(argument: Argument, theta_emotion: float = 0.65) -> float:
    """
    High emotional load in premises + evaluative conclusion.

    Returns score ∈ [0, 1].
    """
    prem_text = " ".join(p.text.lower() for p in argument.premises)
    words     = prem_text.split()
    if not words:
        return 0.0

    emotion_count = sum(1 for w in words if w in _EMOTION_TOKENS)
    emotion_ratio = emotion_count / len(words)

    if emotion_ratio >= theta_emotion:
        return 0.85
    if emotion_ratio >= theta_emotion * 0.5:
        return 0.50
    return 0.0


def detect_red_herring(argument: Argument, theta_topic: float = 0.30) -> float:
    """
    Low topic similarity between premises and conclusion.

    Returns topic drift score ∈ [0, 1].
    """
    if not argument.premises:
        return 0.0

    prem_terms = set()
    for p in argument.premises:
        prem_terms.update(p.terms)
    conc_terms = set(argument.conclusion.terms)

    if not prem_terms or not conc_terms:
        return 0.0

    union = prem_terms | conc_terms
    inter = prem_terms & conc_terms
    similarity = len(inter) / len(union)

    if similarity < theta_topic:
        return 1.0 - similarity
    return 0.0


def detect_straw_man(
    argument: Argument,
    memory_context: List[ProcessedStatement],
    theta_straw: float = 0.40,
) -> float:
    """
    Attributed position has low similarity to actual stated position
    from memory context.

    Returns structural match score ∈ [0, 1].
    """
    # Look for attributed positions in premise text
    prem_text  = " ".join(p.text.lower() for p in argument.premises)
    attribution_markers = [
        "you said", "you claim", "you believe", "your position",
        "according to you", "you argue", "you think",
        "they said", "they claim", "they believe",
    ]
    has_attribution = any(m in prem_text for m in attribution_markers)
    if not has_attribution:
        return 0.0

    if not memory_context:
        # Can't verify — mild signal from attribution alone
        return 0.45

    # Compare attributed content with memory
    prem_terms = set(p for p in prem_text.split() if len(p) > 3)
    for stmt in memory_context:
        mem_terms = set(t.lower() for t in (stmt.tokens or stmt.raw_text.split()) if len(t) > 3)
        if not mem_terms:
            continue
        union = prem_terms | mem_terms
        inter = prem_terms & mem_terms
        similarity = len(inter) / len(union) if union else 0.0
        if similarity < theta_straw:
            return 0.80   # attribution exists AND matches poorly with memory
    return 0.30


def detect_gamblers_fallacy(argument: Argument) -> float:
    """
    Streak of independent outcomes → reversal expectation.

    Returns structural match score ∈ [0, 1].
    """
    text_lower = argument.raw_text.lower()
    has_streak   = any(m in text_lower for m in _STREAK_MARKERS)
    reversal_markers = [
        "due", "overdue", "must turn", "bound to", "has to change",
        "can't keep", "won't last", "about to change",
    ]
    has_reversal = any(m in text_lower for m in reversal_markers)
    if has_streak and has_reversal:
        return 0.85
    if has_streak:
        return 0.45
    return 0.0


def detect_survivorship_bias(argument: Argument) -> float:
    """
    Success stories as evidence for general success rates without failure acknowledgment.

    Returns structural match score ∈ [0, 1].
    """
    text_lower = argument.raw_text.lower()
    has_survival = any(m in text_lower for m in _SURVIVAL_MARKERS)
    has_general  = argument.conclusion.quantifier in ("universal", "existential")
    no_failure   = not any(w in text_lower for w in [
        "failed", "failure", "unsuccessful", "loss", "didn't work",
    ])
    if has_survival and no_failure and has_general:
        return 0.80
    if has_survival and no_failure:
        return 0.50
    return 0.0


def detect_appeal_to_nature(argument: Argument) -> float:
    """Natural/unnatural classification as sole evaluative premise."""
    prem_text = " ".join(p.text.lower() for p in argument.premises)
    has_nature = any(t in prem_text for t in _NATURE_TOKENS)
    if has_nature:
        text_lower = argument.raw_text.lower()
        has_evaluative = any(w in text_lower for w in [
            "good", "bad", "better", "worse", "should", "must",
            "right", "wrong", "safe", "dangerous",
        ])
        if has_evaluative:
            return 0.75
        return 0.40
    return 0.0


def detect_appeal_to_tradition(argument: Argument) -> float:
    """Historical precedent as sole justification."""
    prem_text = " ".join(p.text.lower() for p in argument.premises)
    has_tradition = any(t in prem_text for t in _TRADITION_TOKENS)
    if has_tradition:
        # Sole justification = premise is short and tradition-focused
        if len(prem_text.split()) < 15:
            return 0.75
        return 0.50
    return 0.0


def detect_tu_quoque(argument: Argument) -> float:
    """Counter-accusation without addressing original claim."""
    text_lower = argument.raw_text.lower()
    counter_markers = [
        "you do it too", "you also", "what about you", "you're not",
        "look at yourself", "you're no better", "you did the same",
        "you're guilty too", "so do you",
    ]
    has_counter = any(m in text_lower for m in counter_markers)
    if has_counter:
        return 0.85
    return 0.0


def detect_genetic_fallacy(argument: Argument) -> float:
    """Evaluating claim based on origin/source rather than merit."""
    prem_text  = " ".join(p.text.lower() for p in argument.premises)
    origin_markers = [
        "comes from", "originated from", "invented by", "created by",
        "started by", "their idea", "that group", "those people made",
    ]
    has_origin = any(m in prem_text for m in origin_markers)
    if has_origin:
        # Check conclusion makes evaluative claim about the idea itself
        text_lower = argument.raw_text.lower()
        has_evaluative = any(w in text_lower for w in [
            "wrong", "bad", "invalid", "incorrect", "false", "unreliable",
        ])
        if has_evaluative:
            return 0.72
        return 0.35
    return 0.0


def detect_base_rate_neglect(argument: Argument) -> float:
    """Conditional probability claim without base rate reference."""
    text_lower = argument.raw_text.lower()
    has_conditional_prob = any(p in text_lower for p in [
        "% of", "percent of", "most", "likely", "probability",
        "chance", "rate of",
    ])
    has_conclusion_claim = any(w in argument.conclusion.text.lower() for w in [
        "therefore", "so", "likely", "probably", "must be",
    ])
    no_base_rate = not any(p in text_lower for p in [
        "base rate", "prior", "overall rate", "background rate",
        "in general", "on average",
    ])
    if has_conditional_prob and has_conclusion_claim and no_base_rate:
        return 0.65
    return 0.0


def detect_confirmation_bias_arg(argument: Argument) -> float:
    """One-sided evidence presentation — all premises support, no counter-evidence."""
    if len(argument.premises) < 2:
        return 0.0

    text_lower = argument.raw_text.lower()
    has_qualification = any(w in text_lower for w in [
        "however", "although", "but", "despite", "even though",
        "on the other hand", "admittedly", "while", "yet",
    ])
    # One-sided: no qualifications + multiple premises all pointing same direction
    all_same_polarity = all(
        p.is_negated == argument.premises[0].is_negated
        for p in argument.premises
    )
    if not has_qualification and all_same_polarity and len(argument.premises) >= 2:
        return 0.60
    return 0.0


def detect_complex_question(argument: Argument) -> float:
    """Question that presupposes an unestablished claim."""
    text = argument.raw_text
    is_question = text.strip().endswith("?")
    if not is_question:
        return 0.0

    presupposition_markers = [
        "when did you stop", "why did you", "have you stopped",
        "how often do you", "do you still", "since you",
    ]
    text_lower = text.lower()
    has_presup = any(m in text_lower for m in presupposition_markers)
    if has_presup:
        return 0.80
    # Generic: any question with embedded assumption
    if "why" in text_lower or "when did" in text_lower:
        return 0.45
    return 0.0


def detect_quantifier_shift(argument: Argument) -> float:
    """
    Scope confusion: ∀x∃y interpreted as ∃y∀x.
    Heuristic: universal + existential quantifiers in adjacent propositions
    with term overlap (same referents, different scope).
    """
    if len(argument.premises) < 2:
        return 0.0

    quants = [p.quantifier for p in argument.premises] + [argument.conclusion.quantifier]
    has_universal    = "universal"    in quants
    has_existential  = "existential"  in quants

    if has_universal and has_existential:
        # Check if terms overlap (same referents)
        all_terms = set()
        for p in argument.premises:
            all_terms.update(p.terms)
        conc_terms = set(argument.conclusion.terms)
        overlap = len(all_terms & conc_terms) / max(1, len(all_terms | conc_terms))
        if overlap > 0.3:
            return 0.65
    return 0.0


def detect_illicit_conversion(argument: Argument) -> float:
    """
    'All A are B' treated as 'All B are A'.
    Heuristic: reversed subject-predicate in conclusion vs premise.
    """
    if not argument.premises or not argument.conclusion.terms:
        return 0.0

    first_prem = argument.premises[0]
    if first_prem.quantifier != "universal":
        return 0.0

    prem_terms = first_prem.terms
    conc_terms = argument.conclusion.terms

    # Reversed order = terms appear in opposite order
    if len(prem_terms) >= 2 and len(conc_terms) >= 2:
        prem_pair = (prem_terms[0], prem_terms[-1])
        conc_pair = (conc_terms[0], conc_terms[-1])
        if prem_pair[0] == conc_pair[1] and prem_pair[1] == conc_pair[0]:
            return 0.80
    return 0.0


def detect_composition(argument: Argument) -> float:
    """Part → Whole inference without warrant."""
    text_lower = argument.raw_text.lower()
    part_markers = [
        "each part", "every part", "every member", "each member",
        "each component", "individually",
    ]
    whole_markers = [
        "the whole", "the entire", "overall", "together", "as a whole",
        "collectively",
    ]
    has_part  = any(m in text_lower for m in part_markers)
    has_whole = any(m in text_lower for m in whole_markers)
    if has_part and has_whole:
        return 0.75
    return 0.0


def detect_division(argument: Argument) -> float:
    """Whole → Part inference without warrant."""
    text_lower = argument.raw_text.lower()
    whole_prem = any(m in " ".join(p.text.lower() for p in argument.premises)
                     for m in ["the whole", "overall", "as a whole", "collectively"])
    part_conc  = any(m in argument.conclusion.text.lower()
                     for m in ["each", "every", "every member", "each part", "individually"])
    if whole_prem and part_conc:
        return 0.75
    return 0.0


def detect_equivocation(argument: Argument) -> float:
    """
    Same word used with different meanings across premise and conclusion.
    Heuristic: content words shared between premise and conclusion that
    appear in different syntactic contexts (noun vs verb, etc.).
    """
    if not argument.premises:
        return 0.0

    prem_terms = set()
    for p in argument.premises:
        prem_terms.update(p.terms)
    conc_terms = set(argument.conclusion.terms)

    shared = prem_terms & conc_terms
    if not shared:
        return 0.0

    # Known polysemous words most likely to cause equivocation
    polysemous = {
        "bank", "bat", "bear", "firm", "light", "right", "mean",
        "fair", "good", "free", "run", "interest", "note", "change",
        "state", "power", "rule", "key", "principle", "ground",
    }
    equivocating = shared & polysemous
    if equivocating:
        return min(1.0, 0.50 + 0.15 * len(equivocating))

    return 0.0


# =====================================================================
# Pure functions — Relevance scoring
# =====================================================================

def compute_relevance_score(argument: Argument) -> float:
    """
    Compute how relevantly the premises support the conclusion.

    relevance = max(semantic_overlap, causal_signal, evidential_signal)

    Returns float ∈ [0, 1]; higher = more relevant.
    """
    if not argument.premises:
        return 0.5

    prem_terms = set()
    for p in argument.premises:
        prem_terms.update(p.terms)
    conc_terms = set(argument.conclusion.terms)

    if not prem_terms and not conc_terms:
        return 0.5

    union = prem_terms | conc_terms
    inter = prem_terms & conc_terms
    semantic_overlap = len(inter) / len(union) if union else 0.0

    text_lower = argument.raw_text.lower()
    causal_signal    = 0.5 if any(m in text_lower for m in _CAUSAL_MARKERS) else 0.0
    evidential_signal = 0.5 if any(m in text_lower for m in _PREMISE_MARKERS) else 0.0

    return max(semantic_overlap, causal_signal, evidential_signal)


def compute_argument_complexity(argument: Argument) -> float:
    """
    Complexity proxy: n_premises × inference_depth × quantifier_complexity.

    Returns float ≥ 0, used for Glu coupling.
    """
    n_prems = len(argument.premises)
    n_conc  = 1

    quantifier_complexity = sum(
        1.5 if p.quantifier in ("universal", "existential") else 1.0
        for p in argument.premises + [argument.conclusion]
    )

    depth_multiplier = {
        InferenceType.DEDUCTIVE:   1.5,
        InferenceType.INDUCTIVE:   1.2,
        InferenceType.CAUSAL:      1.3,
        InferenceType.ABDUCTIVE:   1.4,
        InferenceType.ANALOGICAL:  1.1,
    }.get(argument.inference_type, 1.0)

    return (n_prems + n_conc) * depth_multiplier * quantifier_complexity / 10.0


# =====================================================================
# Pure functions — Bayesian confidence
# =====================================================================

def _log_beta_b(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _beta_pdf(x: float, a: float, b: float) -> float:
    x = max(1e-9, min(1.0 - 1e-9, x))
    log_pdf = (a - 1.0) * math.log(x) + (b - 1.0) * math.log(1.0 - x) - _log_beta_b(a, b)
    return math.exp(log_pdf)


# Likelihood Beta params (fallacy | evidence) and (no fallacy | evidence)
# Format: (alpha_fallacy, beta_fallacy, alpha_no_fallacy, beta_no_fallacy)
_LIKELIHOOD_PARAMS = {
    "structural_match":       (8, 2, 2, 7),
    "relevance_deficit":      (6, 3, 2, 6),
    "alternative_validity":   (2, 8, 7, 3),   # INVERTED: high v suppresses
    "context_appropriateness":(2, 7, 5, 3),   # INVERTED: high a suppresses
}


def compute_fallacy_confidence(
    evidence: EvidenceSignals,
    category_prior: float,
) -> float:
    """
    Compute posterior P(fallacy | m, d, v, a).

    Note: alternative_validity and context_appropriateness are INVERTED
    — high values suppress the fallacy hypothesis (principle of charity).

    P(fallacy | evidence) ∝ P(fallacy) × P(m|fallacy) × P(d|fallacy)
                                        × P(v|fallacy) × P(a|fallacy)

    Returns
    -------
    float
        Posterior probability ∈ [0, 1].
    """
    p_f    = category_prior
    p_not_f = 1.0 - p_f

    # Structural match
    af, bf, anf, bnf = _LIKELIHOOD_PARAMS["structural_match"]
    p_f    *= _beta_pdf(evidence.structural_match, af, bf)
    p_not_f *= _beta_pdf(evidence.structural_match, anf, bnf)

    # Relevance deficit
    af, bf, anf, bnf = _LIKELIHOOD_PARAMS["relevance_deficit"]
    p_f    *= _beta_pdf(evidence.relevance_deficit, af, bf)
    p_not_f *= _beta_pdf(evidence.relevance_deficit, anf, bnf)

    # Alternative validity (inverted: used as¬fallacy boosting)
    af, bf, anf, bnf = _LIKELIHOOD_PARAMS["alternative_validity"]
    p_f    *= _beta_pdf(evidence.alternative_validity, af, bf)
    p_not_f *= _beta_pdf(evidence.alternative_validity, anf, bnf)

    # Context appropriateness (inverted)
    af, bf, anf, bnf = _LIKELIHOOD_PARAMS["context_appropriateness"]
    p_f    *= _beta_pdf(evidence.context_appropriateness, af, bf)
    p_not_f *= _beta_pdf(evidence.context_appropriateness, anf, bnf)

    normalizer = p_f + p_not_f
    if normalizer <= 0.0:
        return 0.0
    return p_f / normalizer


# =====================================================================
# Pure functions — Principle of Charity
# =====================================================================

# Charity interpretation templates per fallacy type
_CHARITY_TEMPLATES: Dict[FallacyType, str] = {
    FallacyType.AFFIRMING_CONSEQUENT:
        "Implicit premise: the conditional may be biconditional (P ↔ Q).",
    FallacyType.DENYING_ANTECEDENT:
        "Implicit premise: the conditional is assumed to be the only path to Q.",
    FallacyType.APPEAL_TO_AUTHORITY:
        "Implicit premise: the cited authority has demonstrated expertise in this specific domain.",
    FallacyType.APPEAL_TO_EMOTION:
        "Implicit premise: emotional considerations are directly relevant to the conclusion.",
    FallacyType.HASTY_GENERALIZATION:
        "Implicit premise: the anecdotal case is representative of the broader population.",
    FallacyType.FALSE_CAUSE:
        "Implicit premise: temporal sequence in this context is a reliable causal indicator.",
    FallacyType.AD_HOMINEM:
        "Implicit premise: the person's character is directly relevant to the reliability of their argument.",
    FallacyType.SLIPPERY_SLOPE:
        "Implicit premise: each step in the causal chain has been independently established.",
    FallacyType.BEGGING_QUESTION:
        "The premise may be providing an independent grounding for the conclusion via shared vocabulary.",
}


def estimate_charity_plausibility(
    fallacy_type: FallacyType,
    argument: Argument,
    intention_vector: IntentionVector,
    config: FallacyEngineConfig,
) -> Tuple[float, str]:
    """
    Estimate plausibility of a charitable interpretation for the argument.

    Base plausibility depends on:
    - Whether a charity template exists (0.50 base if yes, 0.20 if no)
    - Whether the argument is from an adversarial source (defensiveness ↓ charity)
    - Argument complexity (longer = more likely to have implicit premises)

    Returns
    -------
    (plausibility, charitable_interpretation_text)
    """
    template = _CHARITY_TEMPLATES.get(fallacy_type, "")
    base = 0.50 if template else 0.20

    # Defensiveness reduces charitable interpretation
    defensiveness_penalty = 0.20 * intention_vector.e_defensiveness
    base = max(0.0, base - defensiveness_penalty)

    # Longer arguments are more likely to have implicit valid premises
    word_count = len(argument.raw_text.split())
    length_bonus = min(0.15, word_count / 100.0)
    plausibility = min(1.0, base + length_bonus)

    charitable_text = template or "Implicit valid premise may exist that repairs this reasoning."
    return plausibility, charitable_text


# =====================================================================
# Pure functions — Manipulation indicator
# =====================================================================

def compute_manipulation_indicator(
    fallacy_type: FallacyType,
    argument: Argument,
    alternative_validity: float,
    intention_vector: IntentionVector,
    config: FallacyEngineConfig,
) -> float:
    """
    manipulation_indicator = w_m1 × I(source==USER_INPUT)
                           × w_m2 × (1 - v(A))
                           × w_m3 × e_defensiveness
                           × w_m4 × fallacy_sophistication(type)

    Returns float ∈ [0, 1].
    """
    source_factor     = 1.0 if argument.source == SourceTag.USER_INPUT else 0.0
    validity_factor   = 1.0 - alternative_validity
    defend_factor     = intention_vector.e_defensiveness
    sophistication    = _FALLACY_SOPHISTICATION.get(fallacy_type, 0.30)

    indicator = (
        config.w_m1 * source_factor
        + config.w_m2 * validity_factor
        + config.w_m3 * defend_factor
        + config.w_m4 * sophistication
    )
    return _clamp(indicator)


# =====================================================================
# Pure functions — Neurochemical coupling
# =====================================================================

def compute_fallacy_load(
    flags: List[FallacyFlag],
    config: FallacyEngineConfig,
) -> float:
    """
    Φ(t) = Σ_f w_cat(f) × confidence(f) × (1 - charity_suppression(f))

    Charity-suppressed flags contribute 0.
    """
    total = 0.0
    for flag in flags:
        charity_factor = 0.0 if (flag.charity_suppression and flag.charity_suppression.suppressed) else 1.0
        w_cat = _CATEGORY_LOAD_WEIGHT.get(flag.fallacy_category, 0.30)
        total += w_cat * flag.confidence * charity_factor
    return total


def compute_neurochemical_signals(
    phi: float,
    argument_complexity: float,
    config: FallacyEngineConfig,
    is_self_audit: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """
    Compute all neurochemical coupling signals.

    Returns
    -------
    dict with keys:
        ach_burst              — ΔC_ACh (analytical vigilance)
        ne_burst               — ΔC_NE (quality alerting, 0 if Φ < θ_alert)
        reward_logic_penalty   — R_Logic subtraction (2× for self-audit)
        glu_complexity_signal  — ΔC_Glu (cognitive load)
        beta_boost_val         — multiplicative β-band factor
    """
    if rng is None:
        rng = np.random.default_rng()

    # ACh burst: Gamma-distributed
    xi_ach = rng.gamma(shape=config.alpha_ACh, scale=config.theta_ACh)
    ach_burst = config.beta_fallacy * phi * xi_ach

    # NE: Poisson burst if Φ > θ_alert
    if phi > config.theta_alert:
        poisson_count = rng.poisson(config.lambda_alert)
        ne_burst = config.beta_alert * float(poisson_count)
    else:
        ne_burst = 0.0

    # R_Logic penalty
    multiplier = 2.0 if is_self_audit else 1.0
    reward_logic_penalty = multiplier * config.lambda_fallacy * phi

    # Glu: argument complexity
    glu_signal = config.beta_reasoning * argument_complexity

    # β-band
    beta_boost_val = 1.0 + config.psi_beta_logic * phi

    return {
        "ach_burst":            ach_burst,
        "ne_burst":             ne_burst,
        "reward_logic_penalty": reward_logic_penalty,
        "glu_complexity_signal": glu_signal,
        "beta_boost":     beta_boost_val,
    }


# =====================================================================
# Threshold — bidirectional neurochem feedback
# =====================================================================

def resolve_fallacy_threshold(
    mode: OperationalMode,
    config: FallacyEngineConfig,
    ach_level:  float = 0.0,
    ne_level:   float = 0.0,
    gaba_level: float = 0.0,
    da_level:   float = 0.0,
    is_internal_audit: bool = False,
) -> float:
    """
    Determine active detection threshold, adjusted by neurochem state.

    Bidirectional feedback:
    - High ACh → lower threshold for formal fallacies (more analytical)
    - High NE  → lower overall threshold (more vigilant)
    - Low GABA → lower threshold (less charitable/inhibited)
    - High DA  → raise informal fallacy threshold (exploratory tolerance)

    Returns
    -------
    float
        Effective θ, clamped to [0.05, 0.95].
    """
    if is_internal_audit:
        base = config.theta_internal_audit
    else:
        base = {
            OperationalMode.NORMAL:     config.theta_normal,
            OperationalMode.DEV:        config.theta_dev,
            OperationalMode.LEARNING:   config.theta_learning,
            OperationalMode.REFLECTIVE: config.theta_normal,
            OperationalMode.REM_NORMAL: config.theta_rem_normal,
            OperationalMode.REM_DREAM:  config.theta_rem_dream,
        }.get(mode, config.theta_normal)

    adjustment = (
        -0.08 * ach_level     # ACh: more analytical → lower threshold
        - 0.06 * ne_level     # NE: vigilance → lower threshold
        + 0.06 * (1.0 - gaba_level)  # low GABA → lower threshold
        + 0.08 * da_level     # DA: exploratory tolerance → raise threshold
    )
    # Note: GABA term: low GABA = less inhibition = MORE aggressive flagging
    adjusted = base - 0.06 * (1.0 - gaba_level) + 0.06 * gaba_level
    adjusted = base + (
        -0.08 * ach_level
        - 0.06 * ne_level
        - 0.06 * (1.0 - gaba_level)   # low GABA → subtract from threshold
        + 0.08 * da_level
    )
    return max(0.05, min(0.95, adjusted))


# =====================================================================
# Fallacy detector registry
# =====================================================================

# Each entry: (FallacyType, detection_function, valid_counterpart_text)
# Detection function signature: (argument, **kwargs) → float
_FORMAL_DETECTORS = [
    (FallacyType.AFFIRMING_CONSEQUENT,
     detect_affirming_consequent,
     "Modus ponens: P → Q, P ⊢ Q"),
    (FallacyType.DENYING_ANTECEDENT,
     detect_denying_antecedent,
     "Modus tollens: P → Q, ¬Q ⊢ ¬P"),
    (FallacyType.AFFIRMING_DISJUNCT,
     detect_affirming_disjunct,
     "Disjunctive syllogism: P ∨ Q, ¬P ⊢ Q"),
    (FallacyType.UNDISTRIBUTED_MIDDLE,
     detect_undistributed_middle,
     "Barbara (valid syllogism): All M are P, All S are M ⊢ All S are P"),
    (FallacyType.EXISTENTIAL_FALLACY,
     detect_existential_fallacy,
     "Add an existential premise: ∃x: P(x) to support the existential conclusion"),
    (FallacyType.QUANTIFIER_SHIFT,
     detect_quantifier_shift,
     "Maintain consistent quantifier scope throughout the argument"),
    (FallacyType.ILLICIT_CONVERSION,
     detect_illicit_conversion,
     "Valid conversion requires particular form: 'Some A are B' → 'Some B are A'"),
    (FallacyType.COMPOSITION,
     detect_composition,
     "Properties of parts do not necessarily transfer to wholes without additional warrant"),
    (FallacyType.DIVISION,
     detect_division,
     "Properties of wholes do not necessarily transfer to parts without additional warrant"),
]

_INFORMAL_DETECTORS = [
    # Relevance
    (FallacyType.AD_HOMINEM,          detect_ad_hominem,
     "Address the argument on its merits, not the person"),
    (FallacyType.TU_QUOQUE,           detect_tu_quoque,
     "Address the original accusation directly rather than counter-accusing"),
    (FallacyType.APPEAL_TO_AUTHORITY, detect_appeal_to_authority,
     "Cite the authority's specific reasoning or evidence, not just their endorsement"),
    (FallacyType.APPEAL_TO_POPULARITY, detect_appeal_to_popularity,
     "Provide substantive evidence independent of consensus"),
    (FallacyType.APPEAL_TO_EMOTION,   detect_appeal_to_emotion,
     "Support the claim with factual evidence alongside emotional considerations"),
    (FallacyType.APPEAL_TO_NATURE,    detect_appeal_to_nature,
     "Provide a reasoned connection between natural origin and evaluative claim"),
    (FallacyType.APPEAL_TO_TRADITION, detect_appeal_to_tradition,
     "Justify the practice on its own merits, not only historical precedent"),
    (FallacyType.RED_HERRING,         detect_red_herring,
     "Ensure premises are topically relevant to the conclusion"),
    (FallacyType.GENETIC_FALLACY,     detect_genetic_fallacy,
     "Evaluate the claim on its own merit rather than its origin"),
    # Presumption
    (FallacyType.BEGGING_QUESTION,    detect_begging_question,
     "Provide premises that add independent evidential support for the conclusion"),
    (FallacyType.FALSE_DILEMMA,       detect_false_dilemma,
     "Acknowledge the full range of alternatives before concluding"),
    (FallacyType.FALSE_CAUSE,         detect_false_cause,
     "Establish a causal mechanism, not just temporal sequence"),
    (FallacyType.SLIPPERY_SLOPE,      detect_slippery_slope,
     "Establish the probability of each causal link in the chain independently"),
    (FallacyType.HASTY_GENERALIZATION, detect_hasty_generalization,
     "Base generalizations on representative samples, not isolated cases"),
    (FallacyType.COMPLEX_QUESTION,    detect_complex_question,
     "Separate the presupposition from the question and establish it independently"),
    # Ambiguity
    (FallacyType.EQUIVOCATION,        detect_equivocation,
     "Use each key term consistently with a single meaning throughout"),
    # Inductive
    (FallacyType.CONFIRMATION_BIAS_ARG, detect_confirmation_bias_arg,
     "Acknowledge and address counter-evidence alongside supporting evidence"),
    (FallacyType.BASE_RATE_NEGLECT,   detect_base_rate_neglect,
     "Include the base rate (prior probability) in the evidential calculation"),
    (FallacyType.GAMBLERS_FALLACY,    detect_gamblers_fallacy,
     "Independent events have no memory; each outcome is independent of history"),
    (FallacyType.ANECDOTAL_EVIDENCE,  detect_hasty_generalization,
     "Support generalizations with systematic data, not individual cases"),
    (FallacyType.SURVIVORSHIP_BIAS,   detect_survivorship_bias,
     "Account for failed cases as well as successful ones in the analysis"),
]


# =====================================================================
# Engine class
# =====================================================================

class FallacyDetectionEngine:
    """
    Fallacy Detection Engine — Detection Cluster, Engine 4.

    Parameters
    ----------
    config : FallacyEngineConfig, optional
    rng : np.random.Generator, optional
    """

    engine_id = "fallacy_detection_engine"
    cluster   = "detection"

    def __init__(
        self,
        config: Optional[FallacyEngineConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._config = config or FallacyEngineConfig()
        self._rng    = rng or np.random.default_rng()
        self._state  = FallacyEngineState()
        self._mode   = OperationalMode.NORMAL

    # ------------------------------------------------------------------
    # Configuration port
    # ------------------------------------------------------------------

    def configure(self, mode: OperationalMode) -> None:
        self._mode = mode

    # ------------------------------------------------------------------
    # Neurochem read port
    # ------------------------------------------------------------------

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """Receive neurochem readout. Keys: ``"ach"``, ``"ne"``, ``"gaba"``, ``"da"``."""
        if "ach" in state_dict:
            self._state.ach_level = _clamp(state_dict["ach"])
        if "ne" in state_dict:
            self._state.ne_level = _clamp(state_dict["ne"])
        if "gaba" in state_dict:
            self._state.gaba_level = _clamp(state_dict["gaba"])
        if "da" in state_dict:
            self._state.da_level = _clamp(state_dict["da"])

    # ------------------------------------------------------------------
    # Main process port
    # ------------------------------------------------------------------

    def process(
        self,
        fallacy_input: FallacyInput,
    ) -> FallacyDetectionResult:
        """
        Run one full fallacy detection pass.

        Parameters
        ----------
        fallacy_input : FallacyInput

        Returns
        -------
        FallacyDetectionResult
        """
        import time
        t_start = time.perf_counter()

        theta = resolve_fallacy_threshold(
            self._mode, self._config,
            ach_level=self._state.ach_level,
            ne_level=self._state.ne_level,
            gaba_level=self._state.gaba_level,
            da_level=self._state.da_level,
        )

        flags: List[FallacyFlag] = []
        charitable_count = 0
        total_complexity = 0.0
        has_self_audit   = False

        # Build all argument candidates
        all_arguments: List[Argument] = []

        # Extract from user propositions (convert to simple arguments)
        for prop in fallacy_input.user_propositions:
            if prop.text:
                extracted = extract_arguments_from_text(
                    prop.text,
                    source=SourceTag.USER_INPUT,
                    is_self_audit=False,
                    theta_arg=self._config.theta_arg,
                )
                all_arguments.extend(extracted)

        # Extract from semantic expansion raw text if present
        raw_expansion = fallacy_input.semantic_expansion.get("raw_text", "")
        if raw_expansion:
            extracted = extract_arguments_from_text(
                raw_expansion,
                source=SourceTag.USER_INPUT,
                is_self_audit=False,
                theta_arg=self._config.theta_arg,
            )
            all_arguments.extend(extracted)

        # Add pre-extracted system reasoning chains (self-audit)
        if fallacy_input.system_reasoning_chains:
            for arg in fallacy_input.system_reasoning_chains:
                arg_copy = Argument(
                    argument_id=arg.argument_id,
                    premises=arg.premises,
                    conclusion=arg.conclusion,
                    inference_type=arg.inference_type,
                    explicit_markers=arg.explicit_markers,
                    implicit=arg.implicit,
                    source=arg.source,
                    raw_text=arg.raw_text,
                    confidence_in_extraction=arg.confidence_in_extraction,
                    is_self_audit=True,
                )
                all_arguments.append(arg_copy)
                has_self_audit = True

        # Contradiction cross-feed: map uuid → flag for reference
        contra_uuid_map: Dict[str, ContradictionFlag] = {}
        for cf in fallacy_input.contradiction_flags:
            contra_uuid_map[cf.contradiction_id] = cf

        # Evaluate each argument
        for arg in all_arguments:
            if arg.confidence_in_extraction < self._config.theta_arg:
                continue

            total_complexity += compute_argument_complexity(arg)
            arg_flags = self._evaluate_argument(
                arg,
                fallacy_input,
                contra_uuid_map,
                theta,
            )

            for flag_or_suppression in arg_flags:
                if isinstance(flag_or_suppression, FallacyFlag):
                    flags.append(flag_or_suppression)
                elif isinstance(flag_or_suppression, CharitySuppression):
                    charitable_count += 1

        # Neurochemical signals
        phi = compute_fallacy_load(flags, self._config)
        has_self_audit_flag = any(
            f.argument.is_self_audit for f in flags
        )
        phi_self = compute_fallacy_load(
            [f for f in flags if f.argument.is_self_audit], self._config
        )

        neurochem = compute_neurochemical_signals(
            phi, total_complexity, self._config,
            is_self_audit=has_self_audit_flag,
            rng=self._rng,
        )

        t_end = time.perf_counter()

        formal_count   = sum(1 for f in flags if f.fallacy_category == FallacyCategory.FORMAL)
        informal_count = sum(1 for f in flags if f.fallacy_category != FallacyCategory.FORMAL)
        self_count     = sum(1 for f in flags if f.argument.is_self_audit)

        return FallacyDetectionResult(
            flags=flags,
            arguments_analyzed=len(all_arguments),
            arguments_charitable=charitable_count,
            formal_fallacies=formal_count,
            informal_fallacies=informal_count,
            self_audit_flags=self_count,
            clean_pass=len(flags) == 0,
            detection_threshold_used=theta,
            processing_time_ms=(t_end - t_start) * 1000.0,
            neurochemical_signals=neurochem,
            metadata={
                "mode":                    self._mode.value,
                "charity_enabled":         fallacy_input.charity_enabled,
                "internal_audit_performed": has_self_audit,
            },
        )

    # ------------------------------------------------------------------
    # Status port
    # ------------------------------------------------------------------

    def get_status(self) -> Dict:
        return {
            "engine_id": self.engine_id,
            "mode":      self._mode.value,
            "ach_level": self._state.ach_level,
            "ne_level":  self._state.ne_level,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_argument(
        self,
        arg: Argument,
        fallacy_input: FallacyInput,
        contra_uuid_map: Dict[str, ContradictionFlag],
        theta: float,
    ) -> List:
        """
        Evaluate one argument against all fallacy detectors.
        Returns list of FallacyFlag or CharitySuppression objects.
        """
        results = []
        relevance_score = compute_relevance_score(arg)
        relevance_deficit = 1.0 - relevance_score

        # Related contradiction flags for this argument
        related_contra_ids = [
            cf_id for cf_id, cf in contra_uuid_map.items()
            if _argument_overlaps_contradiction(arg, cf)
        ]

        # --- Formal detectors ---
        for fallacy_type, detector_fn, valid_counterpart in _FORMAL_DETECTORS:
            struct_match = detector_fn(arg)
            if struct_match < 0.20:
                continue

            # Alternative validity: charity raises this
            alt_validity = 1.0 - struct_match   # structural match reduces validity
            ctx_approp   = 0.20                  # formal fallacies rarely context-appropriate

            evidence = EvidenceSignals(
                structural_match=struct_match,
                relevance_deficit=relevance_deficit,
                alternative_validity=alt_validity,
                context_appropriateness=ctx_approp,
            )
            prior = _CATEGORY_PRIOR[FallacyCategory.FORMAL]
            confidence = compute_fallacy_confidence(evidence, prior)

            if confidence < theta:
                continue

            # Charity check
            charity_result = self._apply_charity(
                fallacy_type, arg, fallacy_input, confidence
            )
            if charity_result is not None and charity_result.suppressed:
                results.append(charity_result)
                continue

            flag = self._make_flag(
                arg, fallacy_type, FallacyCategory.FORMAL,
                confidence, evidence, valid_counterpart,
                extract_logical_form(arg),
                charity_result,
                related_contra_ids,
                fallacy_input.intention_vector,
                alt_validity,
            )
            results.append(flag)

        # --- Informal detectors ---
        for fallacy_type, detector_fn, valid_counterpart in _INFORMAL_DETECTORS:
            # Special cases requiring extra args
            if fallacy_type == FallacyType.STRAW_MAN:
                struct_match = detect_straw_man(
                    arg, fallacy_input.memory_context, self._config.theta_straw
                )
            elif fallacy_type == FallacyType.APPEAL_TO_EMOTION:
                struct_match = detect_appeal_to_emotion(arg, self._config.theta_emotion)
            elif fallacy_type == FallacyType.RED_HERRING:
                struct_match = detect_red_herring(arg, self._config.theta_topic)
            else:
                struct_match = detector_fn(arg)

            if struct_match < 0.20:
                continue

            category = get_fallacy_category(fallacy_type)
            alt_validity = max(0.0, 1.0 - struct_match) * 0.5
            ctx_approp   = self._estimate_context_appropriateness(
                fallacy_type, fallacy_input
            )

            evidence = EvidenceSignals(
                structural_match=struct_match,
                relevance_deficit=relevance_deficit,
                alternative_validity=alt_validity,
                context_appropriateness=ctx_approp,
            )
            prior = _CATEGORY_PRIOR[category]
            confidence = compute_fallacy_confidence(evidence, prior)

            if confidence < theta:
                continue

            charity_result = self._apply_charity(
                fallacy_type, arg, fallacy_input, confidence
            )
            if charity_result is not None and charity_result.suppressed:
                results.append(charity_result)
                continue

            flag = self._make_flag(
                arg, fallacy_type, category,
                confidence, evidence, valid_counterpart,
                None,  # no symbolic logical form for informal
                charity_result,
                related_contra_ids,
                fallacy_input.intention_vector,
                alt_validity,
            )
            results.append(flag)

        return results

    def _apply_charity(
        self,
        fallacy_type: FallacyType,
        argument: Argument,
        fallacy_input: FallacyInput,
        confidence: float,
    ) -> Optional[CharitySuppression]:
        """
        Attempt charitable interpretation. Returns CharitySuppression if
        the flag should be suppressed, or None if no charity applies.
        """
        if not fallacy_input.charity_enabled:
            return None

        plausibility, text = estimate_charity_plausibility(
            fallacy_type, argument, fallacy_input.intention_vector, self._config
        )
        # Adjust charity threshold by GABA: low GABA = less charitable
        adjusted_threshold = self._config.theta_charity * (
            0.7 + 0.3 * self._state.gaba_level
        )

        if plausibility >= adjusted_threshold:
            return CharitySuppression(
                original_fallacy=fallacy_type,
                original_confidence=confidence,
                charitable_interpretation=text,
                charity_plausibility=plausibility,
                suppressed=True,
            )
        return None

    def _estimate_context_appropriateness(
        self,
        fallacy_type: FallacyType,
        fallacy_input: FallacyInput,
    ) -> float:
        """
        Context-specific legitimacy score for an informal fallacy type.
        Returns float ∈ [0, 1] where higher = more legitimate in context.
        """
        # Base appropriateness (most informal fallacies are rarely appropriate)
        base = 0.15

        # Authority: appropriate in expert-domain contexts
        if fallacy_type == FallacyType.APPEAL_TO_AUTHORITY:
            if "expert" in str(fallacy_input.semantic_expansion).lower():
                base = 0.45

        # Emotion: appropriate in ethical/moral contexts
        if fallacy_type == FallacyType.APPEAL_TO_EMOTION:
            if "ethics" in str(fallacy_input.semantic_expansion).lower():
                base = 0.40

        # Adversarial context reduces appropriateness of all informal fallacies
        adversarial_factor = (
            fallacy_input.intention_vector.e_defensiveness
            + fallacy_input.intention_vector.e_challenge
        ) / 2.0
        base = max(0.0, base - 0.15 * adversarial_factor)

        return base

    def _make_flag(
        self,
        arg: Argument,
        fallacy_type: FallacyType,
        category: FallacyCategory,
        confidence: float,
        evidence: EvidenceSignals,
        valid_counterpart: Optional[str],
        logical_form: Optional[str],
        charity_suppression: Optional[CharitySuppression],
        related_contra_ids: List[str],
        intention_vector: IntentionVector,
        alternative_validity: float,
    ) -> FallacyFlag:
        description = _build_fallacy_description(fallacy_type, arg, confidence)
        manip = compute_manipulation_indicator(
            fallacy_type, arg, alternative_validity,
            intention_vector, self._config,
        )
        return FallacyFlag(
            fallacy_id=str(uuid.uuid4()),
            argument=arg,
            source=arg.source,
            fallacy_type=fallacy_type,
            fallacy_category=category,
            confidence=confidence,
            evidence_signals=evidence,
            description=description,
            logical_form=logical_form,
            valid_counterpart=valid_counterpart,
            charity_applied=True,
            charity_suppression=charity_suppression,
            related_contradiction_flags=related_contra_ids,
            manipulation_indicator=manip,
            timestamp=datetime.utcnow(),
        )


# =====================================================================
# Internal helpers
# =====================================================================

def _argument_overlaps_contradiction(
    arg: Argument,
    cf: ContradictionFlag,
) -> bool:
    """Check whether an argument and contradiction flag share content tokens."""
    arg_terms  = set(arg.conclusion.terms)
    for p in arg.premises:
        arg_terms.update(p.terms)

    cf_terms = set(cf.statement_a.raw_text.lower().split())
    cf_terms.update(cf.statement_b.raw_text.lower().split())

    stop = {"the", "a", "an", "is", "are", "was", "it", "of", "in"}
    arg_terms -= stop
    cf_terms  -= stop

    return bool(arg_terms & cf_terms)


def _build_fallacy_description(
    fallacy_type: FallacyType,
    argument: Argument,
    confidence: float,
) -> str:
    source = argument.source.value
    text   = argument.raw_text[:80] + "..." if len(argument.raw_text) > 80 else argument.raw_text
    type_label = fallacy_type.value.replace("_", " ").title()
    return (
        f"{type_label} (conf={confidence:.2f}) detected in [{source}]: \"{text}\""
    )
