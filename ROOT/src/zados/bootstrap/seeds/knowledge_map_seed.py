"""
Knowledge map seeds — foundational semantic graphs.

Original hardcoded maps (1–4):
  1. Neurochemical Dynamics        (neuroscience)
  2. Cognitive Architecture        (cognitive_science)
  3. Memory Systems Hierarchy      (cognitive_science)
  4. Learning Mechanisms           (learning_theory)

Parser-driven maps (5+): one map per layer group from the ZA-DOS concept library.
  layer_1_1_concepts  — Existence & Being
  layer_1_2_concepts  — Identity & Difference
  layer_1_3_concepts  — Space & Structure
  layer_1_4_concepts  — Time & Change
  layer_1_5_concepts  — Quantity & Probability
  layer_1_6_concepts  — Logic & Truth
  layer_2_concepts    — Experiential Concepts  (all 2.x)
  layer_3_concepts    — Relational & Social Concepts  (all 3.x)

Each map uses stable, deterministic node/link IDs prefixed with "seed_"
so they remain consistent across sessions and can be referenced by lessons.
"""
from __future__ import annotations

from typing import List

from zados.memory.long_term.knowledge.types import (
    KnowledgeLink,
    KnowledgeMap,
    KnowledgeNode,
)


def _node(nid: str, label: str, ntype: str = "concept", conf: float = 0.9) -> KnowledgeNode:
    return KnowledgeNode(node_id=nid, label=label, node_type=ntype, confidence=conf)


def _link(lid: str, src: str, tgt: str, rel: str, w: float = 1.0) -> KnowledgeLink:
    return KnowledgeLink(link_id=lid, source_node=src, target_node=tgt, relation=rel, weight=w)


# ============================================================================
# Map 1 — Neurochemical Dynamics
# ============================================================================

def _make_neuro_map() -> KnowledgeMap:
    nodes = [
        _node("seed_n_da",   "Dopamine",              "concept"),
        _node("seed_n_5ht",  "Serotonin",             "concept"),
        _node("seed_n_ne",   "Norepinephrine",        "concept"),
        _node("seed_n_ach",  "Acetylcholine",         "concept"),
        _node("seed_n_gaba", "GABA",                  "concept"),
        _node("seed_n_cor",  "Cortisol",              "concept"),
        _node("seed_n_oxt",  "Oxytocin",              "concept"),
        _node("seed_n_cb1",  "Cannabinoid (CB1)",     "concept"),
        _node("seed_n_reward",  "Reward & Motivation",   "principle"),
        _node("seed_n_mood",    "Mood Stability",        "principle"),
        _node("seed_n_arousal", "Arousal & Focus",       "principle"),
        _node("seed_n_mem_c",   "Memory Consolidation",  "principle"),
        _node("seed_n_inhib",   "Neural Inhibition",     "principle"),
        _node("seed_n_stress",  "Stress Response",       "principle"),
        _node("seed_n_social",  "Social Bonding",        "principle"),
        _node("seed_n_plast",   "Synaptic Plasticity",   "principle"),
        _node("seed_n_pred_e",  "Prediction Error",      "principle"),
    ]
    links = [
        _link("seed_nl_01", "seed_n_da",   "seed_n_reward",  "supports"),
        _link("seed_nl_02", "seed_n_da",   "seed_n_pred_e",  "supports"),
        _link("seed_nl_03", "seed_n_da",   "seed_n_plast",   "supports"),
        _link("seed_nl_04", "seed_n_5ht",  "seed_n_mood",    "supports"),
        _link("seed_nl_05", "seed_n_ne",   "seed_n_arousal", "supports"),
        _link("seed_nl_06", "seed_n_ach",  "seed_n_arousal", "supports", 0.8),
        _link("seed_nl_07", "seed_n_ach",  "seed_n_mem_c",   "supports"),
        _link("seed_nl_08", "seed_n_gaba", "seed_n_inhib",   "supports"),
        _link("seed_nl_09", "seed_n_gaba", "seed_n_arousal", "contradicts", 0.9),
        _link("seed_nl_10", "seed_n_cor",  "seed_n_stress",  "supports"),
        _link("seed_nl_11", "seed_n_cor",  "seed_n_mem_c",   "contradicts", 0.9),
        _link("seed_nl_12", "seed_n_oxt",  "seed_n_social",  "supports"),
        _link("seed_nl_13", "seed_n_cb1",  "seed_n_plast",   "supports"),
        _link("seed_nl_14", "seed_n_pred_e", "seed_n_reward", "requires"),
    ]
    return KnowledgeMap(
        map_id="seed_map_neurochemical_dynamics",
        title="Neurochemical Dynamics",
        subject_category="neuroscience",
        description=(
            "The eight neurochemical systems of ZADOS and their functional roles. "
            "Dopamine drives reward and prediction error. Serotonin anchors mood "
            "stability. Norepinephrine governs arousal and attention scope. "
            "Acetylcholine sharpens focus and consolidates memory. GABA provides "
            "inhibitory balance. Cortisol mediates acute stress. Oxytocin reinforces "
            "social and affiliative signals. Cannabinoids modulate plasticity and "
            "divergent thinking."
        ),
        nodes=nodes,
        links=links,
        tags=["neurochemistry", "neurotransmitters", "ZADOS", "seed"],
    )


# ============================================================================
# Map 2 — Cognitive Architecture Fundamentals
# ============================================================================

def _make_cognitive_arch_map() -> KnowledgeMap:
    nodes = [
        _node("seed_ca_cognition",     "Cognition",          "concept"),
        _node("seed_ca_attention",     "Attention",          "concept"),
        _node("seed_ca_perception",    "Perception",         "concept"),
        _node("seed_ca_wm",            "Working Memory",     "concept"),
        _node("seed_ca_ltm",           "Long-Term Memory",   "concept"),
        _node("seed_ca_learning",      "Learning",           "concept"),
        _node("seed_ca_reasoning",     "Reasoning",          "concept"),
        _node("seed_ca_emotion",       "Emotion",            "concept"),
        _node("seed_ca_decision",      "Decision-Making",    "concept"),
        _node("seed_ca_metacog",       "Metacognition",      "concept"),
        _node("seed_ca_consciousness", "Consciousness",      "concept", 0.7),
        _node("seed_ca_agency",        "Agency",             "concept"),
        _node("seed_ca_bottleneck",    "Attentional Bottleneck", "principle"),
        _node("seed_ca_pred_model",    "Predictive Model",   "principle"),
    ]
    links = [
        _link("seed_cal_01", "seed_ca_attention",  "seed_ca_perception",    "requires"),
        _link("seed_cal_02", "seed_ca_perception",  "seed_ca_wm",            "extends"),
        _link("seed_cal_03", "seed_ca_wm",          "seed_ca_reasoning",     "supports"),
        _link("seed_cal_04", "seed_ca_reasoning",   "seed_ca_decision",      "supports"),
        _link("seed_cal_05", "seed_ca_ltm",         "seed_ca_reasoning",     "supports"),
        _link("seed_cal_06", "seed_ca_emotion",     "seed_ca_attention",     "supports"),
        _link("seed_cal_07", "seed_ca_emotion",     "seed_ca_decision",      "supports"),
        _link("seed_cal_08", "seed_ca_metacog",     "seed_ca_cognition",     "extends"),
        _link("seed_cal_09", "seed_ca_attention",   "seed_ca_bottleneck",    "exemplifies"),
        _link("seed_cal_10", "seed_ca_pred_model",  "seed_ca_perception",    "supports"),
        _link("seed_cal_11", "seed_ca_pred_model",  "seed_ca_learning",      "supports"),
        _link("seed_cal_12", "seed_ca_consciousness","seed_ca_metacog",      "requires"),
        _link("seed_cal_13", "seed_ca_agency",      "seed_ca_decision",      "requires"),
        _link("seed_cal_14", "seed_ca_learning",    "seed_ca_ltm",           "supports"),
    ]
    return KnowledgeMap(
        map_id="seed_map_cognitive_architecture",
        title="Cognitive Architecture Fundamentals",
        subject_category="cognitive_science",
        description=(
            "Core cognitive components and their functional relationships. "
            "Attention acts as a bottleneck selecting information for working memory. "
            "Working memory supports active reasoning and decision-making. "
            "Long-term memory provides stored context for inference. Emotion "
            "biases both attention and decisions. Predictive processing underlies "
            "perception and learning. Metacognition monitors and regulates all "
            "cognitive activity."
        ),
        nodes=nodes,
        links=links,
        tags=["cognition", "architecture", "attention", "memory", "seed"],
    )


# ============================================================================
# Map 3 — Memory Systems Hierarchy
# ============================================================================

def _make_memory_systems_map() -> KnowledgeMap:
    nodes = [
        _node("seed_ms_sensory",    "Sensory Buffer",        "concept"),
        _node("seed_ms_wm",         "Working Memory",        "concept"),
        _node("seed_ms_stm",        "Short-Term Memory",     "concept"),
        _node("seed_ms_ltm",        "Long-Term Memory",      "concept"),
        _node("seed_ms_episodic",   "Episodic Memory",       "concept"),
        _node("seed_ms_semantic",   "Semantic Memory",       "concept"),
        _node("seed_ms_procedural", "Procedural Memory",     "concept"),
        _node("seed_ms_consol",     "Consolidation",         "principle"),
        _node("seed_ms_retrieval",  "Retrieval",             "principle"),
        _node("seed_ms_encoding",   "Encoding",              "principle"),
        _node("seed_ms_forgetting", "Forgetting Curve",      "principle", 0.85),
        _node("seed_ms_ltp",        "Long-Term Potentiation","principle"),
        _node("seed_ms_capacity",   "Capacity Limit (~7±2)", "fact"),
    ]
    links = [
        _link("seed_msl_01", "seed_ms_sensory",    "seed_ms_wm",         "extends"),
        _link("seed_msl_02", "seed_ms_wm",         "seed_ms_stm",        "extends"),
        _link("seed_msl_03", "seed_ms_stm",        "seed_ms_ltm",        "extends"),
        _link("seed_msl_04", "seed_ms_ltm",        "seed_ms_episodic",   "exemplifies"),
        _link("seed_msl_05", "seed_ms_ltm",        "seed_ms_semantic",   "exemplifies"),
        _link("seed_msl_06", "seed_ms_ltm",        "seed_ms_procedural", "exemplifies"),
        _link("seed_msl_07", "seed_ms_consol",     "seed_ms_ltm",        "supports"),
        _link("seed_msl_08", "seed_ms_ltp",        "seed_ms_consol",     "supports"),
        _link("seed_msl_09", "seed_ms_encoding",   "seed_ms_consol",     "requires"),
        _link("seed_msl_10", "seed_ms_retrieval",  "seed_ms_ltm",        "requires"),
        _link("seed_msl_11", "seed_ms_wm",         "seed_ms_capacity",   "exemplifies"),
        _link("seed_msl_12", "seed_ms_forgetting", "seed_ms_retrieval",  "contradicts", 0.8),
    ]
    return KnowledgeMap(
        map_id="seed_map_memory_systems",
        title="Memory Systems Hierarchy",
        subject_category="cognitive_science",
        description=(
            "The layered structure of memory. Sensory buffers feed working memory "
            "(capacity ~7±2 items). Consolidation moves information to long-term "
            "memory via long-term potentiation (LTP). Long-term memory subdivides "
            "into episodic (autobiographical events), semantic (facts/concepts), "
            "and procedural (skills). Retrieval failure follows the forgetting curve."
        ),
        nodes=nodes,
        links=links,
        tags=["memory", "LTP", "consolidation", "working_memory", "seed"],
    )


# ============================================================================
# Map 4 — Learning Mechanisms
# ============================================================================

def _make_learning_mechanisms_map() -> KnowledgeMap:
    nodes = [
        _node("seed_lm_rl",       "Reinforcement Learning",   "concept"),
        _node("seed_lm_pred_err", "Prediction Error (δ)",     "principle"),
        _node("seed_lm_reward",   "Reward Signal",            "principle"),
        _node("seed_lm_habit",    "Habit Formation",          "concept"),
        _node("seed_lm_meta",     "Meta-Learning",            "concept"),
        _node("seed_lm_ctx",      "Contextual Learning",      "concept"),
        _node("seed_lm_pattern",  "Pattern Recognition",      "concept"),
        _node("seed_lm_generalize","Generalization",          "principle"),
        _node("seed_lm_transfer", "Transfer Learning",        "concept"),
        _node("seed_lm_curiosity","Intrinsic Motivation",     "principle"),
        _node("seed_lm_consol",   "Consolidation (sleep)",    "principle"),
        _node("seed_lm_feedback", "Feedback Loop",            "principle"),
        _node("seed_lm_plateau",  "Learning Plateau",         "fact", 0.85),
    ]
    links = [
        _link("seed_lml_01", "seed_lm_pred_err", "seed_lm_rl",        "supports"),
        _link("seed_lml_02", "seed_lm_reward",   "seed_lm_pred_err",  "requires"),
        _link("seed_lml_03", "seed_lm_rl",       "seed_lm_habit",     "supports"),
        _link("seed_lml_04", "seed_lm_feedback", "seed_lm_rl",        "supports"),
        _link("seed_lml_05", "seed_lm_pattern",  "seed_lm_generalize","supports"),
        _link("seed_lml_06", "seed_lm_generalize","seed_lm_transfer",  "supports"),
        _link("seed_lml_07", "seed_lm_meta",     "seed_lm_rl",        "extends"),
        _link("seed_lml_08", "seed_lm_meta",     "seed_lm_ctx",       "extends"),
        _link("seed_lml_09", "seed_lm_curiosity","seed_lm_rl",        "supports"),
        _link("seed_lml_10", "seed_lm_consol",   "seed_lm_rl",        "supports"),
        _link("seed_lml_11", "seed_lm_plateau",  "seed_lm_meta",      "requires"),
    ]
    return KnowledgeMap(
        map_id="seed_map_learning_mechanisms",
        title="Learning Mechanisms",
        subject_category="learning_theory",
        description=(
            "Core mechanisms underlying learning. Prediction error (δ) is the "
            "signal that drives reinforcement learning — the gap between expected "
            "and actual outcome. Reward signals calibrate prediction error. "
            "Repeated reinforcement builds habits. Meta-learning monitors learning "
            "effectiveness and switches strategies at plateaus. Contextual learning "
            "encodes situation-specific parameters. Consolidation during rest "
            "integrates new learning into long-term storage."
        ),
        nodes=nodes,
        links=links,
        tags=["learning", "reinforcement", "prediction_error", "meta_learning", "seed"],
    )


# ============================================================================
# Parser-driven concept library maps (Maps 5+)
# ============================================================================

# Layer group metadata
_LAYER_META = {
    "1.1": ("Existence & Being",          "ontology"),
    "1.2": ("Identity & Difference",      "ontology"),
    "1.3": ("Space & Structure",          "ontology"),
    "1.4": ("Time & Change",              "ontology"),
    "1.5": ("Quantity & Probability",     "ontology"),
    "1.6": ("Logic & Truth",              "ontology"),
    "2":   ("Experiential Concepts",      "phenomenology"),
    "3":   ("Relational & Social Concepts","social_epistemology"),
}


def _safe_node_id(name: str) -> str:
    """Create a safe node_id from a concept name."""
    return name.replace(" ", "_").replace("-", "_").replace("/", "_")[:64]


def _make_concept_library_maps() -> List[KnowledgeMap]:
    """Build one KnowledgeMap per layer group from the concept library."""
    try:
        from zados.bootstrap.concept_library_parser import (
            parse_concept_library,
            get_default_library_path,
        )
        import os
        path = get_default_library_path()
        if not os.path.exists(path):
            return []
        entries = parse_concept_library(path)
    except Exception:
        return []

    # Group entries by layer key
    # Fine-grained layers (1.1 … 1.6) keep their exact layer code.
    # Layers 2.x and 3.x are grouped under "2" and "3" respectively.
    def _group_key(layer: str) -> str:
        group = layer.split(".")[0]
        if group in ("2", "3"):
            return group
        return layer  # keep exact for 1.x

    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for entry in entries:
        key = _group_key(entry.layer)
        groups[key].append(entry)

    maps: List[KnowledgeMap] = []

    for group_key, group_entries in sorted(groups.items()):
        if group_key not in _LAYER_META:
            continue  # skip unknown groups

        title, subject_category = _LAYER_META[group_key]
        map_id = "layer_" + group_key.replace(".", "_") + "_concepts"

        # Build nodes
        nodes: List[KnowledgeNode] = []
        node_id_set: set[str] = set()
        entry_node_ids: dict[str, str] = {}  # concept name → node_id

        for entry in group_entries:
            nid = _safe_node_id(entry.name)
            # Ensure uniqueness within map
            base_nid = nid
            counter = 1
            while nid in node_id_set:
                nid = f"{base_nid}_{counter}"
                counter += 1
            node_id_set.add(nid)
            entry_node_ids[entry.name] = nid
            conf = 0.9 if entry.tv_seed == "HIGH" else (0.75 if entry.tv_seed == "MEDIUM" else 0.6)
            nodes.append(KnowledgeNode(
                node_id=nid,
                label=entry.name,
                node_type="concept",
                confidence=conf,
            ))

        # Build links from DEPENDS-ON (only within same map's node set)
        links: List[KnowledgeLink] = []
        link_counter = 0
        seen_links: set[tuple] = set()

        for entry in group_entries:
            src_nid = entry_node_ids.get(entry.name)
            if src_nid is None:
                continue
            for dep in entry.depends_on:
                tgt_nid = entry_node_ids.get(dep)
                if tgt_nid is None:
                    continue  # dependency not in this map
                link_key = (src_nid, tgt_nid, "requires")
                if link_key in seen_links:
                    continue
                seen_links.add(link_key)
                lid = f"{map_id}_l{link_counter}"
                links.append(KnowledgeLink(
                    link_id=lid,
                    source_node=src_nid,
                    target_node=tgt_nid,
                    relation="requires",
                    weight=1.0,
                ))
                link_counter += 1

        desc = (
            f"Foundational concepts from ZA-DOS concept library layer {group_key}: "
            f"{title}. Contains {len(nodes)} concepts with typed AtomSpace relationships, "
            f"reward domain mappings, and engine cluster annotations. "
            f"These concepts form part of the base vocabulary for the concept type registry "
            f"and AtomSpace ontology."
        )
        tags = [
            "seed",
            "concept_library",
            "layer_" + group_key.replace(".", "_"),
            subject_category,
        ]

        maps.append(KnowledgeMap(
            map_id=map_id,
            title=title,
            subject_category=subject_category,
            description=desc,
            nodes=nodes,
            links=links,
            tags=tags,
        ))

    return maps


# ============================================================================
# Public API
# ============================================================================

def make_seed_maps() -> List[KnowledgeMap]:
    """Return all seed knowledge maps (hardcoded + parser-driven)."""
    hardcoded = [
        _make_neuro_map(),
        _make_cognitive_arch_map(),
        _make_memory_systems_map(),
        _make_learning_mechanisms_map(),
    ]
    concept_library_maps = _make_concept_library_maps()
    return hardcoded + concept_library_maps
