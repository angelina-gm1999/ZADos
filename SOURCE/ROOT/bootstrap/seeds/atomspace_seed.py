"""
AtomSpace seed — parser-driven cognitive ontology for Engine 9.

Seeds AtomSpace from the full ZA-DOS concept library (~230 concepts).
For each ConceptEntry the seed creates:

  1. A CONCEPT_NODE for the canonical name (TV from TV-SEED mapping)
  2. A CONCEPT_NODE for each alias (medium-priority TV)
  3. INHERITANCE_LINK for each DEPENDS-ON relationship (concept → dependency)
  4. Typed links for each ATOM-LINKS entry
  5. EVALUATION_LINK cluster tags (engine relevance)
  6. EVALUATION_LINK reward-domain tags

Falls back to hardcoded legacy seed if the concept library file is unavailable.

Called once per session by KnowledgeBootstrap.seed_atomspace().
Skip if AtomSpace already has atoms (idempotency guard).
Returns the number of atoms added.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from zados.cognitive_engines.cognitools.atomspace_engine import AtomSpaceEngine

from zados.cognitive_engines.cognitools.atomspace_engine import (
    AtomType,
    TruthValue,
)

_SRC = "bootstrap"

# TV-SEED mapping
_TV_MAP = {
    "HIGH":   TruthValue(strength=0.90, confidence=0.90),
    "MEDIUM": TruthValue(strength=0.75, confidence=0.75),
    "LOW":    TruthValue(strength=0.60, confidence=0.60),
}
_TV_HIGH   = _TV_MAP["HIGH"]
_TV_MEDIUM = _TV_MAP["MEDIUM"]
_TV_LOW    = _TV_MAP["LOW"]

# Alias TV is one step below the concept's own TV
_TV_ALIAS = TruthValue(strength=0.75, confidence=0.70)

# Relation TV for DEPENDS-ON and ATOM-LINKS
_TV_LINK = TruthValue(strength=0.85, confidence=0.85)

# AtomType lookup from string link-type names used in the concept library
_LINK_TYPE_MAP: Dict[str, AtomType] = {
    "InheritanceLink": AtomType.INHERITANCE_LINK,
    "SimilarityLink":  AtomType.SIMILARITY_LINK,
    "EvaluationLink":  AtomType.EVALUATION_LINK,
    "ImplicationLink": AtomType.IMPLICATION_LINK,
    "HebbianLink":     AtomType.HEBBIAN_LINK,
    "NotLink":         AtomType.NOT_LINK,
    "ListLink":        AtomType.LIST_LINK,
    "AndLink":         AtomType.AND_LINK,
    "OrLink":          AtomType.OR_LINK,
}


def _atom_name(concept_name: str) -> str:
    """Normalise a concept name to a stable atom name (replace spaces with _)."""
    return concept_name.strip().replace(" ", "_")


def seed_atomspace(engine: "AtomSpaceEngine") -> int:
    """Seed AtomSpace from the ZA-DOS concept library.

    Parameters
    ----------
    engine : AtomSpaceEngine
        Engine 9 instance to populate.

    Returns
    -------
    int
        Number of new atoms added (already-present atoms not counted).
    """
    # Idempotency guard — skip if engine already has atoms
    if len(engine._atoms) > 0:
        return 0

    before = len(engine._atoms)

    # Try parser-driven seed first
    try:
        _seed_from_library(engine)
    except Exception:
        # Fall back to legacy hardcoded seed
        _seed_legacy(engine)

    after = len(engine._atoms)
    return after - before


# ---------------------------------------------------------------------------
# Parser-driven seed
# ---------------------------------------------------------------------------

def _seed_from_library(engine: "AtomSpaceEngine") -> None:
    """Populate AtomSpace from the concept library document."""
    from zados.bootstrap.concept_library_parser import (
        parse_concept_library,
        get_default_library_path,
    )
    import os

    path = get_default_library_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Concept library not found: {path}")

    entries = parse_concept_library(path)

    # Phase 1: Create all CONCEPT_NODEs (concepts + aliases)
    # name → atom_id
    node_ids: Dict[str, str] = {}

    # Shared predicate nodes for cluster/domain tags
    cluster_pred = engine.add_node(
        AtomType.PREDICATE_NODE, "engine_cluster", _TV_HIGH, _SRC
    )
    domain_pred = engine.add_node(
        AtomType.PREDICATE_NODE, "reward_domain", _TV_HIGH, _SRC
    )

    # Cache cluster/domain concept nodes
    cluster_nodes: Dict[str, str] = {}  # cluster_name → atom_id
    domain_nodes: Dict[str, str] = {}   # domain_name → atom_id

    for entry in entries:
        tv = _TV_MAP.get(entry.tv_seed, _TV_MEDIUM)
        atom_name = _atom_name(entry.name)
        node = engine.add_node(AtomType.CONCEPT_NODE, atom_name, tv, _SRC)
        node_ids[entry.name.lower()] = node.atom_id

        # Alias nodes (lower priority TV)
        for alias in entry.aliases:
            alias_clean = alias.strip()
            if not alias_clean:
                continue
            alias_atom_name = _atom_name(alias_clean)
            alias_key = alias_clean.lower()
            if alias_key not in node_ids:
                alias_node = engine.add_node(
                    AtomType.CONCEPT_NODE, alias_atom_name, _TV_ALIAS, _SRC
                )
                node_ids[alias_key] = alias_node.atom_id

    # Phase 2: Add DEPENDS-ON links (INHERITANCE_LINK: concept → dependency)
    for entry in entries:
        concept_id = node_ids.get(entry.name.lower())
        if concept_id is None:
            continue
        for dep in entry.depends_on:
            dep_id = node_ids.get(dep.lower())
            if dep_id is None:
                # Create a stub node for the dependency
                stub = engine.add_node(
                    AtomType.CONCEPT_NODE, _atom_name(dep), _TV_LOW, _SRC
                )
                dep_id = stub.atom_id
                node_ids[dep.lower()] = dep_id
            try:
                engine.add_link(
                    AtomType.INHERITANCE_LINK,
                    (concept_id, dep_id),
                    _TV_LINK,
                    _SRC,
                )
            except Exception:
                pass

    # Phase 3: Add ATOM-LINKS typed relationships
    for entry in entries:
        concept_id = node_ids.get(entry.name.lower())
        if concept_id is None:
            continue
        for spec in entry.atom_links:
            atom_type = _LINK_TYPE_MAP.get(spec.link_type)
            if atom_type is None:
                continue  # Unknown link type — skip

            # Get or create target node
            target_key = spec.target.lower()
            target_id = node_ids.get(target_key)
            if target_id is None:
                stub = engine.add_node(
                    AtomType.CONCEPT_NODE, _atom_name(spec.target), _TV_LOW, _SRC
                )
                target_id = stub.atom_id
                node_ids[target_key] = target_id

            try:
                if atom_type == AtomType.EVALUATION_LINK:
                    # EvaluationLink needs a predicate; use a generic "relates-to"
                    pred_key = "relates-to"
                    pred_id = node_ids.get(pred_key)
                    if pred_id is None:
                        pred_node = engine.add_node(
                            AtomType.PREDICATE_NODE, "relates-to", _TV_HIGH, _SRC
                        )
                        pred_id = pred_node.atom_id
                        node_ids[pred_key] = pred_id
                    engine.add_link(
                        AtomType.EVALUATION_LINK,
                        (pred_id, concept_id, target_id),
                        _TV_LINK,
                        _SRC,
                    )
                else:
                    engine.add_link(
                        atom_type,
                        (concept_id, target_id),
                        _TV_LINK,
                        _SRC,
                    )
            except Exception:
                pass

    # Phase 4: Engine cluster tags
    # EvaluationLink(cluster_pred, [concept_node, cluster_node])
    for entry in entries:
        concept_id = node_ids.get(entry.name.lower())
        if concept_id is None:
            continue
        for cluster in entry.engine_relevance:
            cluster_clean = cluster.strip()
            if not cluster_clean:
                continue
            if cluster_clean not in cluster_nodes:
                cn = engine.add_node(
                    AtomType.CONCEPT_NODE, f"cluster_{cluster_clean}", _TV_HIGH, _SRC
                )
                cluster_nodes[cluster_clean] = cn.atom_id
            try:
                engine.add_link(
                    AtomType.EVALUATION_LINK,
                    (cluster_pred.atom_id, concept_id, cluster_nodes[cluster_clean]),
                    _TV_LINK,
                    _SRC,
                )
            except Exception:
                pass

    # Phase 5: Reward domain tags
    for entry in entries:
        concept_id = node_ids.get(entry.name.lower())
        if concept_id is None:
            continue
        for domain in entry.reward_domains:
            domain_clean = domain.strip()
            if not domain_clean:
                continue
            if domain_clean not in domain_nodes:
                dn = engine.add_node(
                    AtomType.CONCEPT_NODE, f"domain_{domain_clean}", _TV_HIGH, _SRC
                )
                domain_nodes[domain_clean] = dn.atom_id
            try:
                engine.add_link(
                    AtomType.EVALUATION_LINK,
                    (domain_pred.atom_id, concept_id, domain_nodes[domain_clean]),
                    _TV_LINK,
                    _SRC,
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Legacy hardcoded seed (fallback if library file unavailable)
# ---------------------------------------------------------------------------

_HIGH = TruthValue(strength=0.9, confidence=0.9)
_MED  = TruthValue(strength=0.8, confidence=0.8)


def _seed_legacy(engine: "AtomSpaceEngine") -> None:
    """Hardcoded legacy seed — kept as fallback."""

    def _n(atype: AtomType, name: str, tv: TruthValue = _HIGH) -> object:
        return engine.add_node(atype, name, tv, _SRC)

    def inh(a: object, b: object) -> None:
        engine.add_link(AtomType.INHERITANCE_LINK, (a.atom_id, b.atom_id), _HIGH, _SRC)  # type: ignore[union-attr]

    def ev(pred: object, subj: object, obj: object) -> None:
        engine.add_link(AtomType.EVALUATION_LINK,
                        (pred.atom_id, subj.atom_id, obj.atom_id), _HIGH, _SRC)  # type: ignore[union-attr]

    def sim(a: object, b: object) -> None:
        engine.add_link(AtomType.SIMILARITY_LINK, (a.atom_id, b.atom_id), _MED, _SRC)  # type: ignore[union-attr]

    CN = AtomType.CONCEPT_NODE
    PN = AtomType.PREDICATE_NODE

    cognition       = _n(CN, "Cognition")
    memory          = _n(CN, "Memory")
    learning        = _n(CN, "Learning")
    reasoning       = _n(CN, "Reasoning")
    perception      = _n(CN, "Perception")
    attention       = _n(CN, "Attention")
    emotion         = _n(CN, "Emotion")
    consciousness   = _n(CN, "Consciousness")
    intelligence    = _n(CN, "Intelligence")
    decision_making = _n(CN, "Decision_Making")
    metacognition   = _n(CN, "Metacognition")
    language        = _n(CN, "Language")
    creativity      = _n(CN, "Creativity")

    working_memory  = _n(CN, "Working_Memory")
    stm             = _n(CN, "Short_Term_Memory")
    ltm             = _n(CN, "Long_Term_Memory")
    episodic        = _n(CN, "Episodic_Memory")
    semantic        = _n(CN, "Semantic_Memory")
    procedural      = _n(CN, "Procedural_Memory")
    consolidation   = _n(CN, "Memory_Consolidation")

    nt_base         = _n(CN, "Neurotransmitter")
    dopamine        = _n(CN, "Dopamine")
    serotonin       = _n(CN, "Serotonin")
    norepinephrine  = _n(CN, "Norepinephrine")
    acetylcholine   = _n(CN, "Acetylcholine")
    gaba            = _n(CN, "GABA")
    cortisol        = _n(CN, "Cortisol")
    oxytocin        = _n(CN, "Oxytocin")
    cannabinoid     = _n(CN, "Cannabinoid")

    reward          = _n(CN, "Reward")
    mood_stability  = _n(CN, "Mood_Stability")
    arousal         = _n(CN, "Arousal")
    attention_focus = _n(CN, "Attention_Focus")
    inhibition      = _n(CN, "Inhibition")
    stress_response = _n(CN, "Stress_Response")
    social_bonding  = _n(CN, "Social_Bonding")
    plasticity      = _n(CN, "Synaptic_Plasticity")
    pred_error      = _n(CN, "Prediction_Error")

    rl              = _n(CN, "Reinforcement_Learning")
    supervised      = _n(CN, "Supervised_Learning")
    unsupervised    = _n(CN, "Unsupervised_Learning")
    meta_learning   = _n(CN, "Meta_Learning")
    contextual_l    = _n(CN, "Contextual_Learning")
    predictive_cod  = _n(CN, "Predictive_Coding")

    proposition     = _n(CN, "Proposition")
    inference       = _n(CN, "Inference")
    deduction       = _n(CN, "Deduction")
    induction       = _n(CN, "Induction")
    abduction       = _n(CN, "Abduction")
    uncertainty     = _n(CN, "Uncertainty")
    probability     = _n(CN, "Probability")
    pattern         = _n(CN, "Pattern")
    abstraction     = _n(CN, "Abstraction")
    contradiction   = _n(CN, "Contradiction")

    joy             = _n(CN, "Joy", _MED)
    sadness         = _n(CN, "Sadness", _MED)
    fear            = _n(CN, "Fear", _MED)
    anger           = _n(CN, "Anger", _MED)
    curiosity       = _n(CN, "Curiosity")
    disgust         = _n(CN, "Disgust", _MED)
    surprise        = _n(CN, "Surprise", _MED)
    anticipation    = _n(CN, "Anticipation", _MED)
    emo_regulation  = _n(CN, "Emotion_Regulation")

    self_model      = _n(CN, "Self_Model")
    identity        = _n(CN, "Identity")
    self_awareness  = _n(CN, "Self_Awareness")
    theory_of_mind  = _n(CN, "Theory_of_Mind")
    empathy         = _n(CN, "Empathy")
    insight         = _n(CN, "Insight")
    agency          = _n(CN, "Agency")
    goal            = _n(CN, "Goal")

    modulates  = _n(PN, "modulates")
    inhibits   = _n(PN, "inhibits")
    activates  = _n(PN, "activates")
    enables    = _n(PN, "enables")
    drives     = _n(PN, "drives")
    requires   = _n(PN, "requires")
    regulates  = _n(PN, "regulates")
    supports   = _n(PN, "supports")

    inh(working_memory, memory); inh(stm, memory); inh(ltm, memory)
    inh(episodic, ltm); inh(semantic, ltm); inh(procedural, ltm)
    inh(dopamine, nt_base); inh(serotonin, nt_base); inh(norepinephrine, nt_base)
    inh(acetylcholine, nt_base); inh(gaba, nt_base); inh(cortisol, nt_base)
    inh(oxytocin, nt_base); inh(cannabinoid, nt_base)
    inh(rl, learning); inh(supervised, learning); inh(unsupervised, learning)
    inh(meta_learning, learning); inh(contextual_l, learning)
    inh(predictive_cod, learning)
    inh(deduction, inference); inh(induction, inference); inh(abduction, inference)
    inh(joy, emotion); inh(sadness, emotion); inh(fear, emotion)
    inh(anger, emotion); inh(curiosity, emotion); inh(disgust, emotion)
    inh(surprise, emotion); inh(anticipation, emotion)
    inh(metacognition, cognition); inh(reasoning, cognition)
    inh(perception, cognition); inh(decision_making, cognition)
    inh(creativity, cognition)
    inh(self_awareness, self_model); inh(identity, self_model)
    inh(theory_of_mind, self_model); inh(empathy, theory_of_mind)

    ev(modulates, dopamine, reward)
    ev(modulates, dopamine, pred_error)
    ev(modulates, dopamine, plasticity)
    ev(modulates, serotonin, mood_stability)
    ev(inhibits, serotonin, fear)
    ev(modulates, norepinephrine, arousal)
    ev(activates, norepinephrine, attention)
    ev(modulates, acetylcholine, attention_focus)
    ev(modulates, acetylcholine, consolidation)
    ev(inhibits, gaba, arousal)
    ev(modulates, gaba, inhibition)
    ev(modulates, cortisol, stress_response)
    ev(inhibits, cortisol, consolidation)
    ev(modulates, oxytocin, social_bonding)
    ev(modulates, oxytocin, empathy)
    ev(modulates, cannabinoid, plasticity)
    ev(modulates, cannabinoid, creativity)
    ev(enables, attention, learning)
    ev(enables, memory, reasoning)
    ev(drives, curiosity, learning)
    ev(drives, pred_error, rl)
    ev(regulates, metacognition, cognition)
    ev(regulates, emo_regulation, emotion)
    ev(requires, learning, memory)
    ev(supports, consolidation, ltm)
    ev(enables, working_memory, reasoning)
    ev(drives, goal, agency)
    ev(supports, insight, creativity)

    sim(memory, learning); sim(emotion, attention)
    sim(self_model, identity); sim(uncertainty, probability)
    sim(creativity, insight); sim(empathy, social_bonding)
    sim(consciousness, self_awareness); sim(intelligence, reasoning)
    sim(curiosity, anticipation); sim(language, reasoning)
