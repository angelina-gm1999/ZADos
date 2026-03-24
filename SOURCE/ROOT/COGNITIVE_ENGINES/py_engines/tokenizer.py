"""
Tokenizer — Pipeline Infrastructure (Pattern & Analysis Cluster).

Inward decomposition: breaks raw text into structural units, extracts
per-token isolated meaning, computes linguistic features.

Pipeline Position: Fractal Brain Step (a), fires FIRST.

Processing Pipeline (4 Stages)
-------------------------------
Stage 1 — NLP Pipeline:  tokenization, POS, lemma, dep parse, NER, morphology
Stage 2 — Per-Token Meaning: WordNet senses, morphological decomposition, lexical
           profiles (emotion, concreteness, hedging, discourse markers)
Stage 3 — Sentence-Level Analysis: sentence type, question classification, clause
           structure, modality, tense-aspect
Stage 4 — Aggregate Features:  pronoun distribution, emotional vocab density,
           hedging density, coherence, type-token ratio, polysemy load, etc.

External NLP dependencies (spaCy, NLTK WordNet, NRC Emotion Lexicon, Brysbaert
concreteness) are accessed through injectable *provider* objects.  The default
``BuiltinNLPProvider`` uses pure-Python regex / heuristic fallbacks so that the
entire engine is testable without installing multi-GB model files.

Neurochemical Coupling
-----------------------
- ACh Gamma burst ← parsing load L_tok(t)
- NE Poisson burst ← structural anomaly
- Beta oscillatory boost (0.03 while active)

Usage
-----
>>> from zados.cognitive_engines.py_engines.tokenizer import (
...     Tokenizer, TokenizerConfig, TokenizerResult,
... )
>>> tok = Tokenizer()
>>> result = tok.process("I believe freedom is more important than security.")
"""

from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from zados.cognitive_engines.py_engines.contradiction_detection_engine import (
    OperationalMode,
)


# =====================================================================
# Enumerations
# =====================================================================

class SentenceType(str, Enum):
    DECLARATIVE   = "declarative"
    INTERROGATIVE = "interrogative"
    IMPERATIVE    = "imperative"
    EXCLAMATORY   = "exclamatory"
    CONDITIONAL   = "conditional"


class QuestionClassification(str, Enum):
    OPEN       = "open"
    CLOSED     = "closed"
    RHETORICAL = "rhetorical"
    LEADING    = "leading"
    TAG        = "tag"
    INDIRECT   = "indirect"


class DiscourseMarkerType(str, Enum):
    ADDITIVE     = "additive"
    CONTRASTIVE  = "contrastive"
    CAUSAL       = "causal"
    TEMPORAL     = "temporal"


# =====================================================================
# Configuration
# =====================================================================

@dataclass(frozen=True)
class TokenizerConfig:
    # Performance
    n_baseline: int = 20               # typical message length for load normalization
    lambda_complexity: float = 0.10    # clause-depth complexity multiplier

    # Neurochemical coupling
    ach_coupling: float = 0.04         # parsing attention
    ach_gamma_alpha: float = 3.0
    ach_gamma_theta: float = 0.2
    ne_anomaly_coupling: float = 0.03  # structural anomaly
    ne_poisson_lambda: float = 1.0
    beta_boost: float = 0.03           # oscillatory

    # Lexicon markers (built-in)
    hedging_markers: Tuple[str, ...] = (
        "maybe", "perhaps", "possibly", "might", "could", "somewhat",
        "sort of", "kind of", "i think", "i guess", "i suppose",
        "it seems", "arguably", "likely", "unlikely", "probably",
        "in my opinion", "i'm not sure", "roughly", "about",
    )
    discourse_markers_additive: Tuple[str, ...] = (
        "also", "moreover", "furthermore", "in addition", "besides",
        "additionally", "and", "too", "likewise",
    )
    discourse_markers_contrastive: Tuple[str, ...] = (
        "but", "however", "although", "nevertheless", "on the other hand",
        "yet", "still", "though", "whereas", "despite", "instead",
    )
    discourse_markers_causal: Tuple[str, ...] = (
        "because", "since", "therefore", "thus", "so", "consequently",
        "as a result", "hence", "due to", "owing to",
    )
    discourse_markers_temporal: Tuple[str, ...] = (
        "then", "next", "finally", "meanwhile", "afterwards", "before",
        "after", "later", "previously", "while", "during",
    )

    # Emotion lexicon (built-in minimal subset)
    # Keys: anger, fear, joy, sadness, trust, disgust, surprise, anticipation
    emotion_lexicon: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Abstract threshold
    concreteness_abstract_threshold: float = 2.5


# =====================================================================
# Data Types
# =====================================================================

@dataclass(frozen=True)
class LexicalProfile:
    emotion_associations: Dict[str, float] = field(default_factory=dict)
    valence: float = 0.0
    concreteness: float = 3.0
    is_hedging: bool = False
    is_discourse_marker: bool = False
    discourse_marker_type: Optional[str] = None
    is_action_verb: bool = False
    is_stative_verb: bool = False


@dataclass(frozen=True)
class MorphologyEntry:
    token: str = ""
    root: str = ""
    prefixes: Tuple[str, ...] = ()
    suffixes: Tuple[str, ...] = ()
    derivation_type: Optional[str] = None
    inflectional_features: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class WordNetEntry:
    lemma: str = ""
    pos: str = ""
    synset_count: int = 0
    polysemy_score: int = 0
    primary_sense: str = ""


@dataclass(frozen=True)
class TokenData:
    text: str = ""
    lemma: str = ""
    pos_coarse: str = ""
    pos_fine: str = ""
    dep_relation: str = ""
    head_index: int = 0
    wordnet: Optional[WordNetEntry] = None
    morphology: MorphologyEntry = field(default_factory=MorphologyEntry)
    lexical_profile: LexicalProfile = field(default_factory=LexicalProfile)
    is_content_word: bool = False
    is_stop_word: bool = False
    is_entity_part: bool = False
    entity_type: Optional[str] = None
    token_index: int = 0
    sentence_index: int = 0
    char_start: int = 0
    char_end: int = 0


@dataclass(frozen=True)
class SentenceAnalysis:
    sentence_text: str = ""
    sentence_type: str = SentenceType.DECLARATIVE.value
    question_type: Optional[str] = None
    clause_count: int = 1
    nesting_depth: int = 0
    has_conditional: bool = False
    has_negation: bool = False
    modality: Optional[str] = None
    tense_aspect: str = "present_simple"


@dataclass(frozen=True)
class EntitySpan:
    text: str = ""
    entity_type: str = ""
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class AggregateFeatures:
    # Vocabulary metrics
    pronoun_distribution: Dict[str, float] = field(default_factory=dict)
    emotional_vocab_density: float = 0.0
    action_verb_density: float = 0.0
    hedging_density: float = 0.0
    abstract_noun_ratio: float = 0.0
    discourse_marker_density: float = 0.0
    inter_sentence_coherence: float = 0.0

    # Structural metrics
    imperative_count: int = 0
    conditional_count: int = 0
    question_type_distribution: Dict[str, int] = field(default_factory=dict)
    mean_sentence_length: float = 0.0
    sentence_count: int = 0
    clause_nesting_depth_max: int = 0
    negation_count: int = 0

    # Punctuation metrics
    exclamation_density: float = 0.0
    question_density: float = 0.0
    ellipsis_density: float = 0.0

    # Meta-linguistic markers
    meta_linguistic_count: int = 0

    # Complexity metrics
    type_token_ratio: float = 0.0
    mean_word_length: float = 0.0
    polysemy_load: float = 0.0

    # Message trajectory
    message_length_ratio: Optional[float] = None


@dataclass(frozen=True)
class TokenizerResult:
    tokens: List[TokenData] = field(default_factory=list)
    sentences: List[SentenceAnalysis] = field(default_factory=list)
    aggregate_features: AggregateFeatures = field(default_factory=AggregateFeatures)
    entities: List[EntitySpan] = field(default_factory=list)
    raw_text: str = ""
    processing_time_ms: float = 0.0
    token_count: int = 0
    sentence_count: int = 0
    neurochemical_signals: Dict[str, float] = field(default_factory=dict)


# =====================================================================
# Pure Functions — Stage 1: Tokenization (heuristic fallback)
# =====================================================================

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "about",
    "through", "after", "above", "below", "between", "and", "but", "or",
    "nor", "not", "so", "if", "then", "than", "that", "this", "it",
    "its", "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "they", "them", "their", "who", "which", "what", "how", "when",
    "where", "there", "here",
})

_POS_PATTERNS: List[Tuple[str, str, str]] = [
    # (regex, coarse POS, fine POS)
    (r"^(I|me|my|mine|myself)$", "PRON", "PRP"),
    (r"^(you|your|yours|yourself)$", "PRON", "PRP"),
    (r"^(he|she|it|him|her|his|its|they|them|their|theirs|we|us|our|ours)$", "PRON", "PRP"),
    (r"^(who|what|which|whom|whose|where|when|how|why)$", "PRON", "WP"),
    (r"^(is|am|are|was|were|be|been|being)$", "AUX", "VBZ"),
    (r"^(have|has|had)$", "AUX", "VBZ"),
    (r"^(will|would|could|should|may|might|shall|can|must)$", "AUX", "MD"),
    (r"^(not|n't)$", "PART", "RB"),
    (r"^(the|a|an)$", "DET", "DT"),
    (r"^(and|but|or|nor|yet|so)$", "CCONJ", "CC"),
    (r"^(if|because|since|although|while|when|after|before|unless|until|though)$", "SCONJ", "IN"),
    (r"^(in|on|at|to|for|of|with|by|from|about|into|through|during|between|after|before|above|below)$", "ADP", "IN"),
    (r"^(very|really|quite|rather|extremely|just|already|still|always|never|often|sometimes)$", "ADV", "RB"),
    (r"\w+ly$", "ADV", "RB"),
    (r"\w+ing$", "VERB", "VBG"),
    (r"\w+ed$", "VERB", "VBD"),
    (r"\w+tion$|.*ment$|.*ness$|.*ity$|.*ence$|.*ance$", "NOUN", "NN"),
    (r"\w+ous$|.*ive$|.*ful$|.*less$|.*able$|.*ible$", "ADJ", "JJ"),
]

_STATIVE_VERBS = frozenset({
    "be", "is", "am", "are", "was", "were", "been", "being",
    "have", "has", "had", "know", "believe", "think", "understand",
    "feel", "seem", "appear", "like", "love", "hate", "want",
    "need", "prefer", "doubt", "suppose", "mean", "contain",
    "consist", "belong", "own", "possess", "exist", "deserve",
    "matter", "depend", "involve", "resemble", "lack",
})


def _simple_tokenize(text: str) -> List[Tuple[str, int, int]]:
    """Split text into (token, start, end) triples."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _heuristic_pos(token_text: str) -> Tuple[str, str]:
    """Assign POS using regex patterns. Returns (coarse, fine)."""
    lower = token_text.lower().rstrip(".,;:!?\"'")
    for pattern, coarse, fine in _POS_PATTERNS:
        if re.match(pattern, lower, re.IGNORECASE):
            return coarse, fine
    # Default: if starts with uppercase and not first word → proper noun
    if token_text[0].isupper():
        return "PROPN", "NNP"
    return "NOUN", "NN"


def _simple_lemma(token_text: str) -> str:
    """Very basic lemmatization via suffix stripping."""
    lower = token_text.lower().rstrip(".,;:!?\"'")
    if lower.endswith("ing") and len(lower) > 5:
        return lower[:-3]
    if lower.endswith("ed") and len(lower) > 4:
        return lower[:-2]
    if lower.endswith("s") and not lower.endswith("ss") and len(lower) > 3:
        return lower[:-1]
    return lower


def _detect_negation(text: str) -> bool:
    return bool(re.search(r"\bnot\b|\bn't\b|\bnever\b|\bno\b|\bnone\b|\bnor\b", text, re.I))


def _detect_conditional(text: str) -> bool:
    return bool(re.search(r"\bif\b|\bunless\b|\bprovided\b|\bassuming\b", text, re.I))


def _detect_modality(text: str) -> Optional[str]:
    t = text.lower()
    if any(w in t for w in ("must", "should", "ought", "need to", "have to")):
        return "deontic"
    if any(w in t for w in ("might", "may", "could", "possibly", "perhaps")):
        return "epistemic"
    if any(w in t for w in ("can", "able to", "capable")):
        return "dynamic"
    return None


# =====================================================================
# Pure Functions — Stage 2: Per-Token Meaning
# =====================================================================

def build_lexical_profile(
    token_text: str,
    pos_coarse: str,
    config: TokenizerConfig,
) -> LexicalProfile:
    """Build lexical profile using built-in heuristics."""
    lower = token_text.lower().rstrip(".,;:!?\"'")

    # Hedging
    is_hedging = any(m in lower for m in config.hedging_markers if len(m.split()) == 1)
    # Multi-word hedging checked at sentence level

    # Discourse marker detection
    dm_type: Optional[str] = None
    is_dm = False
    if lower in config.discourse_markers_additive:
        is_dm, dm_type = True, DiscourseMarkerType.ADDITIVE.value
    elif lower in config.discourse_markers_contrastive:
        is_dm, dm_type = True, DiscourseMarkerType.CONTRASTIVE.value
    elif lower in config.discourse_markers_causal:
        is_dm, dm_type = True, DiscourseMarkerType.CAUSAL.value
    elif lower in config.discourse_markers_temporal:
        is_dm, dm_type = True, DiscourseMarkerType.TEMPORAL.value

    # Action vs stative (for verbs)
    is_action = pos_coarse == "VERB" and lower not in _STATIVE_VERBS
    is_stative = pos_coarse in ("VERB", "AUX") and lower in _STATIVE_VERBS

    # Concreteness heuristic: shorter content words tend more concrete
    concreteness = min(5.0, max(1.0, 5.0 - len(lower) * 0.30))

    # Emotion from built-in lexicon
    emotions = config.emotion_lexicon.get(lower, {})
    valence = emotions.get("valence", 0.0)

    return LexicalProfile(
        emotion_associations=emotions,
        valence=valence,
        concreteness=concreteness,
        is_hedging=is_hedging,
        is_discourse_marker=is_dm,
        discourse_marker_type=dm_type,
        is_action_verb=is_action,
        is_stative_verb=is_stative,
    )


def build_morphology(token_text: str) -> MorphologyEntry:
    """Basic morphological decomposition."""
    lower = token_text.lower().rstrip(".,;:!?\"'")
    prefixes: List[str] = []
    suffixes: List[str] = []
    root = lower

    # Common prefixes
    for pfx in ("un", "re", "pre", "dis", "mis", "non", "over", "under", "out"):
        if lower.startswith(pfx) and len(lower) > len(pfx) + 2:
            prefixes.append(pfx)
            root = lower[len(pfx):]
            break

    # Common suffixes
    for sfx in ("tion", "ment", "ness", "ity", "ence", "ance", "ous", "ive", "ful", "less", "able", "ible", "ing", "ed", "er", "ly"):
        if root.endswith(sfx) and len(root) > len(sfx) + 2:
            suffixes.append(sfx)
            root = root[:-len(sfx)]
            break

    derivation = None
    if suffixes:
        sfx = suffixes[0]
        if sfx in ("tion", "ment", "ness", "ity", "ence", "ance"):
            derivation = "nominalization"
        elif sfx in ("er",):
            derivation = "agentive"
        elif sfx in ("ous", "ive", "ful", "less", "able", "ible"):
            derivation = "adjectival"
        elif sfx in ("ly",):
            derivation = "adverbial"

    return MorphologyEntry(
        token=token_text,
        root=root,
        prefixes=tuple(prefixes),
        suffixes=tuple(suffixes),
        derivation_type=derivation,
    )


def build_wordnet_entry(lemma: str, pos_coarse: str) -> WordNetEntry:
    """
    Stub WordNet entry — polysemy estimated heuristically.
    In Phase 2+, would use NLTK WordNet for real synset lookups.
    """
    # Heuristic polysemy: common short words → higher polysemy
    poly = max(1, 8 - len(lemma)) if len(lemma) < 8 else 1
    return WordNetEntry(
        lemma=lemma,
        pos=pos_coarse[:1].lower(),
        synset_count=poly,
        polysemy_score=poly,
        primary_sense=f"{lemma}.01",
    )


# =====================================================================
# Pure Functions — Stage 3: Sentence-Level Analysis
# =====================================================================

def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter on .!? boundaries."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sents if s.strip()]


def classify_sentence(text: str) -> SentenceAnalysis:
    """Classify a single sentence."""
    stripped = text.strip()
    # Sentence type
    if stripped.endswith("?"):
        stype = SentenceType.INTERROGATIVE
    elif stripped.endswith("!"):
        stype = SentenceType.EXCLAMATORY
    elif re.match(r"^(Do|Don't|Please|Stop|Go|Tell|Let|Make|Give|Take|Show)\b", stripped, re.I):
        stype = SentenceType.IMPERATIVE
    elif _detect_conditional(stripped):
        stype = SentenceType.CONDITIONAL
    else:
        stype = SentenceType.DECLARATIVE

    # Question classification
    qtype: Optional[str] = None
    if stype == SentenceType.INTERROGATIVE:
        if stripped.lower().startswith(("who", "what", "where", "when", "why", "how")):
            qtype = QuestionClassification.OPEN.value
        elif stripped.lower().endswith(("right?", "isn't it?", "don't you?", "won't you?")):
            qtype = QuestionClassification.TAG.value
        elif re.search(r"\bdon't you think\b|\bisn't it obvious\b", stripped, re.I):
            qtype = QuestionClassification.LEADING.value
        else:
            qtype = QuestionClassification.CLOSED.value

    # Clause count (heuristic: commas + conjunctions + subordinators)
    clause_count = 1
    clause_count += len(re.findall(r",\s*(?:and|but|or|because|since|although|while|if|when|though)\b", stripped, re.I))
    clause_count += len(re.findall(r"\b(?:because|since|although|while|if|when|though|unless)\b", stripped, re.I))
    clause_count = min(clause_count, 8)

    nesting = max(0, clause_count - 1)

    return SentenceAnalysis(
        sentence_text=stripped,
        sentence_type=stype.value,
        question_type=qtype,
        clause_count=clause_count,
        nesting_depth=nesting,
        has_conditional=_detect_conditional(stripped),
        has_negation=_detect_negation(stripped),
        modality=_detect_modality(stripped),
    )


# =====================================================================
# Pure Functions — Stage 4: Aggregate Features
# =====================================================================

def compute_aggregate_features(
    tokens: List[TokenData],
    sentences: List[SentenceAnalysis],
    config: TokenizerConfig,
    previous_message_length: Optional[int] = None,
) -> AggregateFeatures:
    """Compute all aggregate linguistic features."""
    total_tokens = len(tokens)
    if total_tokens == 0:
        return AggregateFeatures(sentence_count=len(sentences))

    # Pronoun distribution
    pronoun_groups = {"I": 0, "you": 0, "we": 0, "they": 0, "it": 0, "he_she": 0}
    for t in tokens:
        lower = t.text.lower().rstrip(".,;:!?\"'")
        if lower in ("i", "me", "my", "mine", "myself"):
            pronoun_groups["I"] += 1
        elif lower in ("you", "your", "yours", "yourself"):
            pronoun_groups["you"] += 1
        elif lower in ("we", "us", "our", "ours"):
            pronoun_groups["we"] += 1
        elif lower in ("they", "them", "their", "theirs"):
            pronoun_groups["they"] += 1
        elif lower in ("it", "its"):
            pronoun_groups["it"] += 1
        elif lower in ("he", "him", "his", "she", "her", "hers"):
            pronoun_groups["he_she"] += 1
    total_pronouns = sum(pronoun_groups.values())
    if total_pronouns > 0:
        pronoun_dist = {k: v / total_pronouns for k, v in pronoun_groups.items()}
    else:
        pronoun_dist = {k: 0.0 for k in pronoun_groups}

    # Content words
    content_words = [t for t in tokens if t.is_content_word]
    content_count = max(len(content_words), 1)

    # Emotional vocab density
    emotion_count = sum(1 for t in content_words if t.lexical_profile.emotion_associations)
    emotional_vocab_density = emotion_count / content_count

    # Action verb density
    verbs = [t for t in tokens if t.pos_coarse in ("VERB",)]
    action_count = sum(1 for t in tokens if t.lexical_profile.is_action_verb)
    verb_count = max(len(verbs), 1)
    action_verb_density = action_count / verb_count if verbs else 0.0

    # Hedging density
    hedging_count = sum(1 for t in tokens if t.lexical_profile.is_hedging)
    hedging_density = hedging_count / total_tokens

    # Abstract noun ratio
    nouns = [t for t in tokens if t.pos_coarse in ("NOUN",)]
    abstract_nouns = sum(1 for t in nouns if t.lexical_profile.concreteness < config.concreteness_abstract_threshold)
    abstract_noun_ratio = abstract_nouns / max(len(nouns), 1)

    # Discourse marker density
    dm_count = sum(1 for t in tokens if t.lexical_profile.is_discourse_marker)
    discourse_marker_density = dm_count / total_tokens

    # Structural metrics from sentences
    imperative_count = sum(1 for s in sentences if s.sentence_type == SentenceType.IMPERATIVE.value)
    conditional_count = sum(1 for s in sentences if s.has_conditional)
    negation_count = sum(1 for s in sentences if s.has_negation)
    max_nesting = max((s.nesting_depth for s in sentences), default=0)

    # Question type distribution
    q_dist: Dict[str, int] = {}
    for s in sentences:
        if s.question_type:
            q_dist[s.question_type] = q_dist.get(s.question_type, 0) + 1

    # Mean sentence length
    if sentences:
        mean_sent_len = sum(len(s.sentence_text.split()) for s in sentences) / len(sentences)
    else:
        mean_sent_len = total_tokens

    # Punctuation densities
    full_text = " ".join(t.text for t in tokens)
    total_punct = full_text.count("!") + full_text.count("?") + full_text.count(".")
    total_punct = max(total_punct, 1)
    excl_density = full_text.count("!") / total_punct
    q_density = full_text.count("?") / total_punct
    ellipsis_density = full_text.count("...") / max(total_tokens, 1)

    # Complexity
    unique_tokens = set(t.lemma.lower() for t in tokens)
    ttr = len(unique_tokens) / total_tokens
    mean_word_len = sum(len(t.text) for t in tokens) / total_tokens

    # Polysemy load
    polysemy_scores = [t.wordnet.polysemy_score for t in content_words if t.wordnet]
    polysemy_load = sum(polysemy_scores) / max(len(polysemy_scores), 1) if polysemy_scores else 0.0

    # Message length ratio
    ml_ratio = None
    if previous_message_length is not None and previous_message_length > 0:
        ml_ratio = total_tokens / previous_message_length

    return AggregateFeatures(
        pronoun_distribution=pronoun_dist,
        emotional_vocab_density=emotional_vocab_density,
        action_verb_density=action_verb_density,
        hedging_density=hedging_density,
        abstract_noun_ratio=abstract_noun_ratio,
        discourse_marker_density=discourse_marker_density,
        inter_sentence_coherence=0.0,  # requires embeddings
        imperative_count=imperative_count,
        conditional_count=conditional_count,
        question_type_distribution=q_dist,
        mean_sentence_length=mean_sent_len,
        sentence_count=len(sentences),
        clause_nesting_depth_max=max_nesting,
        negation_count=negation_count,
        exclamation_density=excl_density,
        question_density=q_density,
        ellipsis_density=ellipsis_density,
        meta_linguistic_count=0,
        type_token_ratio=ttr,
        mean_word_length=mean_word_len,
        polysemy_load=polysemy_load,
        message_length_ratio=ml_ratio,
    )


# =====================================================================
# Pure Functions — Neurochemical Coupling
# =====================================================================

def compute_parsing_load(
    token_count: int,
    max_nesting: int,
    config: TokenizerConfig,
) -> float:
    """L_tok(t) = token_count × (1 + λ_complexity × nesting_max) / N_baseline."""
    load = token_count * (1 + config.lambda_complexity * max_nesting) / max(config.n_baseline, 1)
    return float(np.clip(load, 0.0, 10.0))


def compute_neurochemical_signals(
    load: float,
    anomaly_detected: bool,
    config: TokenizerConfig,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Compute tokenizer neurochemical signals."""
    signals: Dict[str, float] = {
        "ach_burst": 0.0,
        "ne_burst": 0.0,
        "beta_boost": config.beta_boost,
    }
    # ACh — parsing attention
    if load > 0.0:
        ach = config.ach_coupling * load * float(rng.gamma(config.ach_gamma_alpha, config.ach_gamma_theta))
        signals["ach_burst"] = float(np.clip(ach, 0.0, 1.0))
    # NE — structural anomaly
    if anomaly_detected:
        ne_count = rng.poisson(config.ne_poisson_lambda)
        ne = config.ne_anomaly_coupling * float(ne_count) / max(config.ne_poisson_lambda, 0.01)
        signals["ne_burst"] = float(np.clip(ne, 0.0, 1.0))
    return signals


# =====================================================================
# Engine
# =====================================================================

class Tokenizer:
    """
    Tokenizer — Pipeline Infrastructure.

    Public API:
        configure(mode)          — set operational mode
        process(text)            — main tokenization call
        get_status()             — engine state snapshot
    """

    def __init__(
        self,
        config: TokenizerConfig = TokenizerConfig(),
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._config = config
        self._mode = OperationalMode.NORMAL
        self._rng = rng if rng is not None else np.random.default_rng()
        self._previous_message_length: Optional[int] = None

    def configure(self, mode: OperationalMode) -> None:
        self._mode = mode

    def process(self, text: str) -> TokenizerResult:
        if self._mode == OperationalMode.REM_DREAM:
            return TokenizerResult(raw_text=text)

        t0 = time.perf_counter()

        # Stage 1: Tokenize
        raw_tokens = _simple_tokenize(text)

        # Stage 2: Per-token meaning
        token_list: List[TokenData] = []
        for idx, (tok_text, start, end) in enumerate(raw_tokens):
            clean = tok_text.rstrip(".,;:!?\"'")
            if not clean:
                continue
            pos_c, pos_f = _heuristic_pos(tok_text)
            lemma = _simple_lemma(tok_text)
            lex = build_lexical_profile(tok_text, pos_c, self._config)
            morph = build_morphology(tok_text)
            wn = build_wordnet_entry(lemma, pos_c)
            is_content = pos_c in ("NOUN", "VERB", "ADJ", "ADV", "PROPN") and clean.lower() not in _STOP_WORDS
            is_stop = clean.lower() in _STOP_WORDS

            token_list.append(TokenData(
                text=tok_text,
                lemma=lemma,
                pos_coarse=pos_c,
                pos_fine=pos_f,
                dep_relation="",
                head_index=max(0, idx - 1),
                wordnet=wn,
                morphology=morph,
                lexical_profile=lex,
                is_content_word=is_content,
                is_stop_word=is_stop,
                is_entity_part=False,
                entity_type=None,
                token_index=idx,
                sentence_index=0,
                char_start=start,
                char_end=end,
            ))

        # Stage 3: Sentence analysis
        raw_sents = _split_sentences(text)
        if not raw_sents and text.strip():
            raw_sents = [text.strip()]
        sentences = [classify_sentence(s) for s in raw_sents]

        # Assign sentence indices to tokens
        # (simplified: map by char position)
        sent_boundaries = []
        offset = 0
        for s in raw_sents:
            s_start = text.find(s, offset)
            s_end = s_start + len(s)
            sent_boundaries.append((s_start, s_end))
            offset = s_end

        for t in token_list:
            for si, (sb_start, sb_end) in enumerate(sent_boundaries):
                if sb_start <= t.char_start < sb_end:
                    # Need to create new frozen instance
                    token_list[t.token_index] = TokenData(
                        text=t.text, lemma=t.lemma, pos_coarse=t.pos_coarse,
                        pos_fine=t.pos_fine, dep_relation=t.dep_relation,
                        head_index=t.head_index, wordnet=t.wordnet,
                        morphology=t.morphology, lexical_profile=t.lexical_profile,
                        is_content_word=t.is_content_word, is_stop_word=t.is_stop_word,
                        is_entity_part=t.is_entity_part, entity_type=t.entity_type,
                        token_index=t.token_index, sentence_index=si,
                        char_start=t.char_start, char_end=t.char_end,
                    )
                    break

        # Stage 4: Aggregate features
        agg = compute_aggregate_features(
            token_list, sentences, self._config, self._previous_message_length
        )

        # Update previous message length
        self._previous_message_length = len(token_list)

        # Neurochemical signals
        max_nest = agg.clause_nesting_depth_max
        load = compute_parsing_load(len(token_list), max_nest, self._config)
        anomaly = max_nest > 4 or agg.polysemy_load > 5.0
        signals = compute_neurochemical_signals(load, anomaly, self._config, self._rng)

        dt_ms = (time.perf_counter() - t0) * 1000.0

        return TokenizerResult(
            tokens=token_list,
            sentences=sentences,
            aggregate_features=agg,
            entities=[],
            raw_text=text,
            processing_time_ms=dt_ms,
            token_count=len(token_list),
            sentence_count=len(sentences),
            neurochemical_signals=signals,
        )

    def get_status(self) -> Dict:
        return {
            "mode": self._mode.value,
            "previous_message_length": self._previous_message_length,
        }
