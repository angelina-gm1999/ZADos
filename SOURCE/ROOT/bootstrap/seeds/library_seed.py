"""
Library seeds — initial reference material loaded into LibraryStore at boot.

These entries represent foundational documents the system "knows it has read".
They give the LLM retrieval context and the LibraryStore TF-IDF index a
starting corpus for semantic search.

When the user drops new documents into knowledge_sources/books/ and they are
processed, new LibraryEntry seeds are added here.
"""
from __future__ import annotations

from typing import List

from zados.memory.long_term.knowledge.types import LibraryEntry

_SRC_DOC = "document"
_SRC_BOOK = "book"


def make_seed_library_entries() -> List[LibraryEntry]:
    """Return all seed library entries."""
    return [
        LibraryEntry(
            entry_id="seed_lib_zados_architecture",
            title="ZADOS System Architecture Overview",
            source_type=_SRC_DOC,
            domain="cognitive_architecture",
            content=(
                "ZADOS (Zonal Adaptive Dynamics Operating System) is a cognitive "
                "architecture that models intelligent behaviour through layered "
                "neurochemical dynamics, structured memory, and a pipeline of 29 "
                "cognitive engines.\n\n"
                "NEUROCHEMISTRY\n"
                "Eight neurotransmitter axes modulate every cognitive process: "
                "Dopamine (DA) — reward, motivation, learning rate; "
                "Serotonin (5-HT) — mood stability, behavioural conservation; "
                "Norepinephrine (NE) — arousal, attentional scope; "
                "Acetylcholine (ACh) — encoding depth, attention focus; "
                "GABA — inhibitory tone, cognitive gating; "
                "Cortisol (COR) — stress response, risk aversion; "
                "Oxytocin (OXT) — social sensitivity, affiliative bias; "
                "Cannabinoid (CB1) — creative relaxation, synaptic plasticity.\n\n"
                "MEMORY ARCHITECTURE\n"
                "Memory is organised into three tiers: "
                "STMM (Short-Term Memory Manager) — active working context, "
                "decays within a session; "
                "MTMM (Mid-Term Memory Manager) — session-level context, persists "
                "across recent sessions; "
                "LTMM (Long-Term Memory Manager) — durable knowledge base with "
                "16 specialised stores including KnowledgeMap, Lessons, Library, "
                "AcademicBuffer, AcademicQuestions, Notebook, CognitoolsData, "
                "GeneralQuestions, UnsolvedBuffer, HeldThinkingBlocks, OverviewLogs, "
                "Journal, CoreMemories, Hardcoded, and IdentityJournal.\n\n"
                "COGNITIVE ENGINES (29 total)\n"
                "Detection cluster (E1-E6): Contradiction, Paradox, SOAR, "
                "Fallacy, Bias, LogicTrap.\n"
                "Knowledge substrate (E9, E10, E16): AtomSpace (typed hypergraph), "
                "PLN (probabilistic logic networks), ECAN (attention economy).\n"
                "Pattern analysis (E8, E11, E18, E19, E20, E23): Relevance Scoring, "
                "Input Relevance, Data Analysis, Pattern Identification, "
                "Pattern Comparison, Intention Map.\n"
                "Reasoning (E13, E15, E21): Simulation Brain, Decision Making, "
                "Strategic Decision.\n"
                "Learning cluster (E17, E22, E25): Reward-Based Learning, "
                "Contextual Learning, Recursive (Meta) Learning.\n"
                "Homeostasis (E27, E29): Neurochemical Homeostatic, Memory Compression.\n"
                "Meta-awareness (E24, E26): Heuristic Bias, Uncertainty Pattern.\n"
                "Alignment (E30): Retroactive Alignment.\n"
                "Emotional (E28): Emotional Detection.\n\n"
                "PROCESSING PIPELINE\n"
                "Each user input passes through 7 phases: "
                "Phase 1 Extraction (tokenise, mode detect, intent), "
                "Phase 2 Memory Retrieval (STMM + MTMM + LTMM), "
                "Phase 3 Engine Processing (relevant engines fire), "
                "Phase 4 Synthesis (integrate engine outputs into ThinkingContext), "
                "Phase 5 Evaluation (alignment check, reward scoring), "
                "Phase 6 Generation (LLM response), "
                "Phase 7 Post-processing (memory write, NT update, journal).\n\n"
                "MODES\n"
                "Input is classified into modes before processing: "
                "Normal (regular conversation), Learning (academic intake), "
                "Reflective, Creative, Sleep, Dream, Homework, Deep Analysis."
            ),
            tags=["ZADOS", "architecture", "overview", "neurochemistry", "memory",
                  "cognitive_engines", "seed"],
        ),

        LibraryEntry(
            entry_id="seed_lib_neurochem_foundations",
            title="Neurochemical Foundations of Adaptive Cognition",
            source_type=_SRC_DOC,
            domain="neuroscience",
            content=(
                "This document summarises the functional roles of the eight "
                "neuromodulatory systems that ZADOS models and their implications "
                "for adaptive cognition.\n\n"
                "DOPAMINE (DA)\n"
                "Dopamine is the prediction error signal. VTA/SNc DA neurons fire "
                "phasically when actual reward exceeds prediction (positive PE), "
                "are suppressed when outcome is worse (negative PE), and show no "
                "change when outcome exactly matches prediction. This delta signal "
                "updates value representations and drives associative learning. "
                "DA also modulates prefrontal working memory via D1 receptors "
                "(inverted-U dose-response curve) and motivational salience.\n\n"
                "SEROTONIN (5-HT)\n"
                "5-HT from raphe nuclei projects widely to cortex, limbic system, "
                "and basal ganglia. It promotes behavioural patience (wait for "
                "larger-later vs smaller-sooner), reduces threat reactivity, and "
                "opposes impulsive reward-seeking. Chronic 5-HT depletion produces "
                "depressive and anxiety phenotypes. 5-HT2A receptor activation "
                "mediates broad cognitive flexibility and is implicated in creative "
                "and psychedelic states.\n\n"
                "NOREPINEPHRINE (NE)\n"
                "Locus coeruleus NE implements a gain-modulation signal across "
                "cortex. The LC-NE system operates in two modes: tonic (baseline "
                "arousal, broad exploration) and phasic (target-evoked burst, "
                "focused exploitation). Optimal NE levels produce the Yerkes-Dodson "
                "performance peak. NE also regulates synaptic noise and gates "
                "prefrontal-hippocampal interactions during arousal.\n\n"
                "ACETYLCHOLINE (ACh)\n"
                "Cholinergic projections from the basal forebrain (nucleus basalis, "
                "septal area) modulate hippocampal theta rhythms and cortical "
                "desynchronisation. High ACh during waking suppresses "
                "cortico-cortical feedback to prioritise encoding of new "
                "hippocampal input. Low ACh during slow-wave sleep allows "
                "hippocampal-neocortical consolidation replay.\n\n"
                "GABA\n"
                "GABAergic interneurons are the primary inhibitory regulators of "
                "cortical circuits. Parvalbumin-positive fast-spiking interneurons "
                "generate gamma oscillations and sharpen spatial selectivity. "
                "Somatostatin interneurons gate dendritic input. GABA-A "
                "chloride-channel-mediated IPSPs provide rapid inhibition; GABA-B "
                "K+ channel-mediated IPSPs provide slow inhibition. Cognitive "
                "inhibition — suppressing task-irrelevant representations — depends "
                "critically on prefrontal GABAergic interneuron circuits.\n\n"
                "CORTISOL\n"
                "Glucocorticoids act on mineralocorticoid receptors (low-affinity, "
                "tonic binding) and glucocorticoid receptors (high-affinity, stress "
                "response). Acute stress-level cortisol enhances amygdala-dependent "
                "emotional memory consolidation while impairing PFC-mediated "
                "executive function. Chronic hypercortisolaemia causes hippocampal "
                "volume loss and retrieval impairment. Cortisol rhythmicity (diurnal "
                "peak at waking) also modulates cognition across the day.\n\n"
                "OXYTOCIN\n"
                "Hypothalamic OXT release during social contact activates VTA DA "
                "neurons, making social interaction rewarding. OXT reduces amygdala "
                "reactivity to social threats, increases gaze toward faces, "
                "and enhances mentalising (theory of mind). It shows group-specific "
                "effects: increasing in-group trust and, sometimes, out-group "
                "wariness.\n\n"
                "ENDOCANNABINOIDS (CB1)\n"
                "2-AG and anandamide are synthesised on-demand post-synaptically "
                "and travel retrogradely to suppress pre-synaptic release. CB1 "
                "receptors on GABAergic terminals disinhibit DA and glutamate "
                "release, enabling associative bursting. This mechanism underlies "
                "the pro-creative, pro-associative effects of cannabinoid tone and "
                "the extinction of conditioned fear memories."
            ),
            tags=["dopamine", "serotonin", "norepinephrine", "acetylcholine",
                  "GABA", "cortisol", "oxytocin", "cannabinoid", "neuroscience",
                  "neuromodulation", "seed"],
        ),

        LibraryEntry(
            entry_id="seed_lib_concept_library",
            title="ZA-DOS Concept Library — Ontological, Experiential & Relational Primitives",
            source_type=_SRC_DOC,
            domain="ontology",
            content=(
                "The ZA-DOS Concept Library is the foundational vocabulary for the entire "
                "ZADOS cognitive architecture. It defines approximately 230 foundational "
                "concepts organised across three major sections and eight layer groups:\n\n"
                "Layer 1 — ONTOLOGICAL PRIMITIVES\n"
                "  Layer 1.1 Existence & Being: the ground conditions of any predication — "
                "exists, does-not-exist, unknown, real, abstract, possible, impossible, "
                "thing, object, property, state, event, process, relation, instance, type, "
                "token, and context.\n"
                "  Layer 1.2 Identity & Difference: what makes things the same or distinct — "
                "identity, difference, same, other, distinction, individuation, uniqueness, "
                "and boundary.\n"
                "  Layer 1.3 Space & Structure: spatial and structural primitives — "
                "space, location, place, part, whole, structure, container, boundary, "
                "inside, outside, and adjacency.\n"
                "  Layer 1.4 Time & Change: temporal primitives — "
                "time, now, before, after, duration, change, persist, sequence, "
                "cause, and effect.\n"
                "  Layer 1.5 Quantity & Probability: measurement and uncertainty — "
                "quantity, number, measure, more, less, some, all, none, probability, "
                "and distribution.\n"
                "  Layer 1.6 Logic & Truth: epistemic and logical foundations — "
                "true, false, belief, knowledge, inference, implication, consistency, "
                "contradiction, and evidence.\n\n"
                "Layer 2 — EXPERIENTIAL CONCEPTS\n"
                "Covers the full range of subjective and cognitive experience: action, "
                "intention, attention, perception, emotion, affect, desire, motivation, "
                "memory, learning, creativity, language, self, agency, consciousness, "
                "and effort. This layer grounds ZADOS's phenomenological model.\n\n"
                "Layer 3 — RELATIONAL & SOCIAL CONCEPTS\n"
                "Covers inter-agent and social primitives: communication, trust, "
                "responsibility, autonomy, fairness, harm, consent, authority, "
                "collaboration, role, norm, value, identity-coherence, and introspection. "
                "This layer grounds ZADOS's ethical and social-epistemological reasoning.\n\n"
                "Each concept entry includes: typed AtomSpace relationships (InheritanceLink, "
                "SimilarityLink, EvaluationLink, ImplicationLink, HebbianLink, etc.), "
                "a TV-SEED trust value (HIGH/MEDIUM/LOW), DEPENDS-ON dependency chain, "
                "REWARD-DOMAIN mapping (ethics, logic, innovation, human_attunement), "
                "and ENGINE-RELEVANCE cluster annotations (detection, dialectic, "
                "executive_control, knowledge_substrate, pattern_analysis, evaluation, "
                "reasoning, metacognition, meta_self_awareness, homeostasis, "
                "emotional_processing, alignment, learning).\n\n"
                "This library is the base vocabulary for the ConceptTypeRegistry and the "
                "primary seed source for the AtomSpace ontology at session boot."
            ),
            tags=[
                "seed",
                "ontology",
                "concept_library",
                "atomspace",
                "type_system",
                "knowledge_graph",
                "layers_1_through_3",
                "foundational_concepts",
            ],
        ),
    ]
