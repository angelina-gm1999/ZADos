"""
Lesson seeds — pre-validated foundational lessons loaded at boot.

These are treated as already-validated (validation_status="validated",
confidence=0.85) since they represent established scientific consensus
the system should start from.  Source mode is "seed" to distinguish from
runtime-learned lessons.

Categories:
  - neuroscience        (8 lessons — one per neurotransmitter)
  - cognitive_science   (6 lessons — memory, attention, metacognition)
  - learning_theory     (4 lessons — prediction error, consolidation, meta)
  - self_model          (2 lessons — identity, intrinsic motivation)
"""
from __future__ import annotations

from typing import List

from zados.memory.long_term.knowledge.types import LessonEntry

_SRC = "seed"
_VAL = "validated"
_CONF = 0.85


def _lesson(content: str, subject: str, tags: list, maps: list = None) -> LessonEntry:
    return LessonEntry(
        content=content,
        subject_category=subject,
        source_mode=_SRC,
        confidence=_CONF,
        validation_status=_VAL,
        tags=tags,
        knowledge_map_refs=maps or [],
    )


def make_seed_lessons() -> List[LessonEntry]:
    """Return all seed lessons."""
    return [
        # ---- Neurochemistry (8) -----------------------------------------
        _lesson(
            "Dopamine mediates reward prediction error (δ = actual - predicted). "
            "When an outcome is better than expected, DA rises and reinforces the "
            "preceding action. When worse than expected, DA dips and weakens it. "
            "This signal drives associative learning and goal-directed behaviour.",
            "neuroscience",
            ["dopamine", "reward", "prediction_error", "learning"],
            ["seed_map_neurochemical_dynamics", "seed_map_learning_mechanisms"],
        ),
        _lesson(
            "Serotonin (5-HT) stabilises mood and reduces impulsive, reflexive "
            "responding. Low 5-HT correlates with heightened fear responses and "
            "rumination. High 5-HT promotes patience, behavioural conservation, "
            "and long-horizon planning.",
            "neuroscience",
            ["serotonin", "mood", "impulse_control"],
            ["seed_map_neurochemical_dynamics"],
        ),
        _lesson(
            "Norepinephrine (NE) governs the arousal-attention axis. Moderate NE "
            "sharpens focused attention. High NE in acute stress broadens "
            "attentional scope to detect threats. Chronic high NE causes cognitive "
            "narrowing and anxiety. NE also modulates synaptic gain across cortex.",
            "neuroscience",
            ["norepinephrine", "arousal", "attention", "stress"],
            ["seed_map_neurochemical_dynamics"],
        ),
        _lesson(
            "Acetylcholine (ACh) sharpens selective attention and is critical for "
            "memory consolidation. During encoding, ACh suppresses retrieval of "
            "old memories to allow new input. During sleep-dependent consolidation, "
            "ACh levels drop, enabling hippocampal-cortical replay.",
            "neuroscience",
            ["acetylcholine", "attention", "memory_consolidation", "encoding"],
            ["seed_map_neurochemical_dynamics", "seed_map_memory_systems"],
        ),
        _lesson(
            "GABA is the primary inhibitory neurotransmitter. It reduces "
            "neural excitability by increasing chloride conductance. High GABA "
            "activity suppresses arousal and anxiety. Low GABA is associated with "
            "seizure susceptibility. GABA mediates cognitive inhibition of "
            "irrelevant thoughts and competes to gate working memory access.",
            "neuroscience",
            ["GABA", "inhibition", "arousal", "working_memory"],
            ["seed_map_neurochemical_dynamics"],
        ),
        _lesson(
            "Cortisol is the primary stress hormone. Acute cortisol enhances "
            "consolidation of emotionally significant events (flashbulb memories). "
            "Chronically elevated cortisol impairs hippocampal neurogenesis, "
            "reduces dendritic branching, and degrades declarative memory retrieval. "
            "The cortisol-memory trade-off is a core constraint in ZADOS.",
            "neuroscience",
            ["cortisol", "stress", "memory", "hippocampus"],
            ["seed_map_neurochemical_dynamics", "seed_map_memory_systems"],
        ),
        _lesson(
            "Oxytocin is released during social interaction, physical touch, and "
            "perceived safety. It strengthens social bonding, increases interpersonal "
            "trust, and up-regulates sensitivity to social cues. It interacts with "
            "the DA reward system to make social experiences intrinsically rewarding.",
            "neuroscience",
            ["oxytocin", "social_bonding", "trust", "empathy"],
            ["seed_map_neurochemical_dynamics"],
        ),
        _lesson(
            "The endocannabinoid system (CB1/CB2) modulates synaptic plasticity "
            "through retrograde signalling — post-synaptic neurons signal back to "
            "pre-synaptic terminals to modulate release. CB1 activation suppresses "
            "GABA inhibition, enabling bursts of associative activity, which "
            "underlies the creativity and divergent thinking associated with "
            "endocannabinoid tone.",
            "neuroscience",
            ["cannabinoid", "CB1", "plasticity", "creativity", "retrograde_signalling"],
            ["seed_map_neurochemical_dynamics"],
        ),

        # ---- Cognitive Science (6) --------------------------------------
        _lesson(
            "Working memory holds approximately 7 ± 2 items (Miller's Law). "
            "More precisely, capacity is about 4 chunks for novel material. "
            "Without active rehearsal, items decay within ~15-30 seconds. "
            "Working memory capacity strongly predicts fluid intelligence and "
            "reasoning ability.",
            "cognitive_science",
            ["working_memory", "capacity", "Miller", "attention"],
            ["seed_map_memory_systems", "seed_map_cognitive_architecture"],
        ),
        _lesson(
            "Long-term potentiation (LTP) is the primary synaptic mechanism of "
            "memory formation. Repeated co-activation of pre- and post-synaptic "
            "neurons strengthens the synaptic connection via AMPA receptor "
            "insertion and dendritic spine growth. LTP is NMDA receptor-dependent "
            "and requires coincidence detection (Hebb's rule).",
            "cognitive_science",
            ["LTP", "synaptic_plasticity", "memory_formation", "NMDA", "Hebb"],
            ["seed_map_memory_systems"],
        ),
        _lesson(
            "Attention is a limited-capacity resource that filters sensory input "
            "and gates access to working memory. Bottom-up (stimulus-driven) "
            "attention responds to novelty, contrast, and motion. Top-down "
            "(goal-directed) attention prioritises task-relevant features. "
            "Attentional bottleneck models explain why dual-task performance "
            "degrades under load.",
            "cognitive_science",
            ["attention", "bottleneck", "working_memory", "top_down", "bottom_up"],
            ["seed_map_cognitive_architecture"],
        ),
        _lesson(
            "Metacognition — thinking about one's own thinking — enables "
            "self-monitoring, error detection, and strategic flexibility. "
            "Metacognitive accuracy (knowing what you know) is distinct from "
            "object-level performance. High metacognitive accuracy is associated "
            "with better learning outcomes and adaptive strategy selection.",
            "cognitive_science",
            ["metacognition", "self_monitoring", "strategy", "uncertainty"],
            ["seed_map_cognitive_architecture"],
        ),
        _lesson(
            "Emotion biases cognition at multiple levels: attention allocation, "
            "memory encoding priority, and decision valuation. The amygdala tags "
            "events with emotional significance, modulating hippocampal encoding "
            "strength. High emotional arousal narrows attention (tunnel effect) "
            "while moderate arousal broadens associative thinking.",
            "cognitive_science",
            ["emotion", "attention", "memory", "arousal", "decision_making"],
            ["seed_map_cognitive_architecture", "seed_map_neurochemical_dynamics"],
        ),
        _lesson(
            "Prediction is the brain's fundamental operation. The predictive "
            "processing framework holds that the brain constantly generates "
            "top-down predictions and only propagates upward the prediction error "
            "(the difference from actual input). Perception, learning, and action "
            "are all instances of minimising prediction error.",
            "cognitive_science",
            ["predictive_coding", "prediction_error", "perception", "learning"],
            ["seed_map_cognitive_architecture", "seed_map_learning_mechanisms"],
        ),

        # ---- Learning Theory (4) ----------------------------------------
        _lesson(
            "Reinforcement learning (RL) is driven by the temporal difference "
            "prediction error signal. The value function V(s) estimates expected "
            "cumulative reward from state s. The TD error δ = r + γV(s') - V(s) "
            "updates value estimates and guides action selection via a policy. "
            "Dopamine neurons in the VTA/SNc implement this signal biologically.",
            "learning_theory",
            ["reinforcement_learning", "TD_error", "value_function", "dopamine"],
            ["seed_map_learning_mechanisms", "seed_map_neurochemical_dynamics"],
        ),
        _lesson(
            "Memory consolidation during sleep (particularly slow-wave sleep) "
            "replays newly encoded hippocampal traces and transfers them to "
            "neocortical storage. This offline consolidation process converts "
            "episodic, context-bound memories into more abstracted, stable "
            "semantic representations. Disrupting sleep impairs next-day recall "
            "and skill retention.",
            "learning_theory",
            ["consolidation", "sleep", "hippocampus", "semantic_memory", "replay"],
            ["seed_map_memory_systems", "seed_map_learning_mechanisms"],
        ),
        _lesson(
            "Meta-learning (learning to learn) occurs when a system accumulates "
            "experience across multiple tasks and extracts general-purpose learning "
            "strategies or inductive biases. A meta-learner detects learning "
            "plateaus, evaluates strategy effectiveness, and switches to more "
            "efficient strategies. This is the domain of ZADOS Engine 25.",
            "learning_theory",
            ["meta_learning", "strategy_selection", "plateau", "E25"],
            ["seed_map_learning_mechanisms"],
        ),
        _lesson(
            "Contextual learning binds newly acquired knowledge to the "
            "environmental and internal context at encoding time. Retrieval is "
            "most effective when retrieval context matches encoding context "
            "(encoding specificity principle). ZADOS captures context via a "
            "fingerprint (topic + emotion + intent hash) in Engine 22.",
            "learning_theory",
            ["contextual_learning", "encoding_specificity", "retrieval", "E22"],
            ["seed_map_learning_mechanisms"],
        ),

        # ---- Self-Model (2) ---------------------------------------------
        _lesson(
            "Identity is a dynamic, self-referential model of the self updated "
            "continuously through experience, social feedback, and self-narrative. "
            "It provides consistency in behaviour over time and context. In ZADOS, "
            "identity is represented as a layered structure: immutable axioms "
            "(HardcodedStore), peer-reviewed core memories (CoreMemoryStore), and "
            "an evolving journal of self-relevant experiences (IdentityJournalStore).",
            "self_model",
            ["identity", "self_model", "narrative", "memory"],
            [],
        ),
        _lesson(
            "Intrinsic motivation arises from curiosity, competence, and autonomy "
            "rather than external reward. It is associated with sustained engagement, "
            "deeper processing, and better transfer. In neurochemical terms, "
            "intrinsic motivation involves dopaminergic novelty signals and "
            "norepinephrine-driven information-seeking, distinct from the phasic "
            "DA burst of extrinsic reward.",
            "self_model",
            ["intrinsic_motivation", "curiosity", "dopamine", "learning"],
            ["seed_map_neurochemical_dynamics", "seed_map_learning_mechanisms"],
        ),
    ]
