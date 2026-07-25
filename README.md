# ZADos
A model-agnostic, bio-inspired Cognitive Architecture for Artificial Intelligence Systems
_____________________________________________________________________________________________________________________________________________________________________________

Total files: 597

6777 tests passing (full repo)

29 Cognitive Engines + E33 LexicalSalience + E34 ContextDrift + CogniTools (AtomSpace/PLN/ECAN substitutes) + Memory Layer (STMM/MTMM/LTMM) + Neurochemical Core + Reward System + Sleep/Dream Neurochemistry + Session Orchestration + Self-Reflective Sub-Type Dispatch + Proactive AI-Initiated Reflection + OU-Modeled Drift Dynamics + Contextual Learning Loop

ZADOS — Zonal Adaptive Dynamic Operating System
=================================================

A bio-inspired artificial cognition framework that wraps large language models
in a simulated neurochemical and cognitive architecture. ZADOS gives an LLM
persistent memory, emotional modulation, structured reasoning, and adaptive
learning — turning stateless text generation into a continuous cognitive process.

This is a research prototype, not a consumer product.


WHAT IT DOES
------------

ZADOS processes every input through a 7-phase cognitive pipeline:

  1. Perception — Intent classification, entity extraction, pattern detection
     across 5 engines
  2. Neurochemical Modulation — Input shifts a 12-neurotransmitter simulation,
     which changes how the system reasons
  3. Engine Dispatch — 29 cognitive engines activate with NT-weighted priorities
     (high dopamine = more creative; high cortisol = more cautious)
  4. Thinking — LLM generates an internal reasoning trace conditioned on all
     of the above
  5. Reward Evaluation — Reasoning is scored on logic, ethics, innovation, and
     emotional attunement
  6. Answer Generation — Final response shaped by reward signals and personality
     alignment
  7. Post-Processing — Memories compressed, stored, consolidated; learning
     updates applied


CORE SYSTEMS
------------

Neurochemical Engine
  Simulates 12 neurotransmitters (DA, 5-HT, NE, ACh, OXT, GABA, Glu,
  Cortisol, CRH, CB1, MOR, Histamine) using stochastic differential equations
  with Euler-Maruyama integration. Includes receptor binding dynamics (Hill
  equation), 5 oscillatory bands (delta through gamma), cross-frequency
  coupling, and fatigue gating. NT states are translated into cognitive metrics
  (motivation, empathy, precision, anxiety, openness) that modulate all
  downstream processing.

29 Cognitive Engines
  Specialized reasoning modules organized into 13 clusters:

  Detection (E1, E2, E4, E5, E6)
    Contradictions, paradoxes, fallacies, biases, logic traps

  Dialectic (E7, E14)
    Simulated opposition, Socratic reasoning

  Executive Control (E3)
    SOAR production-rule decision cycles

  Knowledge Substrate (E9, E10, E16)
    AtomSpace hypergraph, probabilistic logic (PLN), attention economy (ECAN)

  Pattern Analysis (E8, E11, E18, E19, E20, E23)
    Relevance scoring, data analysis, pattern identification/comparison,
    intent mapping

  Evaluation (E12)
    Formal logic evaluation

  Reasoning (E13, E15, E21)
    Counterfactual simulation, multi-criteria decision, strategic planning

  Metacognition (E24, E26)
    Heuristic bias analysis, uncertainty tracking

  Emotional Processing (E28)
    46-emotion taxonomy with neurochemical mapping

  Homeostasis (E27, E29, E30)
    NT regulation, memory compression, identity alignment

  Learning (E17, E22, E25)
    Reward-based (prediction error), contextual (fingerprinting),
    recursive (meta-learning)

  All engines follow a canonical interface:
    update_neurochem_state(), process(), get_status()

3-Tier Memory
  Short-term (STMM) — Working memory for the current turn. 10 named component
    slots bridge data between engines.
  Mid-term (MTMM) — Session-scoped buffer. Accumulates compressed turn data.
  Long-term (LTMM) — Persistent cross-session storage with granularity levels
    (episode, concept, pattern, identity). Specialized namespaces: Identity
    (beliefs, personality, journal), Thoughts (learning logs, unresolved
    questions), Knowledge (domain facts, AtomSpace persistence).

  Consolidation: STMM compresses to packets at turn end, MTMM consolidates to
  LTMM at session close via fractal similarity matching and emotion-driven
  promotion.

Reward System
  Two-pathway evaluation:
  Tonic (sustained) — Domain evaluators score reasoning on logic, ethics,
    innovation, and attunement. Synthesis engine generates meta-directives
    (suppress, abstain, route). NT adapter maps scores to sustained modulation
    signals.
  Phasic (transient) — Emotion tracking, regulatory modulation, and stochastic
    burst generation for reactive NT deltas.

  Five reward profiles (regular, critical review, curiosity-driven, receptive
  learning, reflective synthesis) weight the four domains differently based on
  conversational context.

Learning Modes
  M1 — Human Teaches: User is teacher, system is student
  M2 — Peer Review: Mutual critique
  M3 — Learn Together: Collaborative exploration
  M4 — Learned Questions: System asks its unresolved questions
  M5 — Independent Study: Autonomous topic exploration

Sleep & Dream
  /sleep rem — Retroactive consolidation: replays memories through reward
    system with varied perspectives
  /sleep dream — Creative recombination: generates novel connections across
    stored memories

Identity
  Seed personality traits loaded at boot. Identity alignment checker validates
  response consistency. Emotionally significant moments are journaled for
  cross-session personality continuity.


ARCHITECTURE
------------

  Input
    |
    v
  InputClassifier (Matrioshka outer layer)
    |--- /sleep         --> REM / Dream Pipeline
    |--- /homework      --> Homework Pipeline
    |--- /reflective    --> Self-Reflective Pipeline
    |--- learning mode  --> Learning Pipeline (M1-M5)
    |--- default        --> Regular Input Pipeline
                                |
                                v
                      AnswerPipeline (7 phases)
                      Phase 0: Validation
                      Phase 1: Perception (E23, E8, E11, E18, E19)
                      Phase 2: NT Modulation (neurochem step + readout)
                      Phase 3: Engine Dispatch (29 engines, weighted)
                      Phase 4: Thinking (LLM Pass 1 — internal trace)
                      Phase 5: Reward Eval (tonic + phasic pathways)
                      Phase 6: Answer (LLM Pass 2 — final response)
                      Phase 7: Post-Process (memory + learning + compression)


TECH STACK
----------

  Language:         Python 3.13
  LLM Integration:  Ollama (local) / Claude API
  Testing:          pytest — 6,135+ tests passing
  Dependencies:     NumPy (SDE integration), standard library otherwise
  No ML training:   All cognitive engines are algorithmic, not learned.
                    The LLM is used for natural language generation only.


PROJECT STRUCTURE
-----------------

  ROOT/
    src/zados/
      core/                Pipeline, session orchestrator, input classifier, phases
      neurochem/           NT kinetics, receptors, oscillations, SDE integration
      cognitive_engines/
        py_engines/        26 runtime cognitive engines
        cognitools/        3 knowledge-substrate engines (AtomSpace, PLN, ECAN)
      memory/              STMM, MTMM, LTMM, consolidation, retrieval
      reward/              Domain evaluators, synthesis, adaptation, profiles
      LLM_interpretation/  Prompt builders, phase 5 evaluator
      bootstrap/           Knowledge initialization, concept parsing
      orchestration/       Cycle management (in progress)
    tests/
      neurochem/, reward/, memory/, cog_engines/, core/
    specs/                 Engine design specifications


STATUS
------

  - 29/29 cognitive engines implemented and tested
  - Core 7-phase pipeline operational
  - Neurochemical engine with 12 NTs, receptor dynamics, 5 oscillatory bands
  - 3-tier memory with consolidation
  - 5 learning modes, 2 sleep modes
  - Session orchestrator with branch classification
  - In progress: UI development for testing in Godot Engine
  - In progress: considering hardcoded sections on identity and values


ETHICAL CONSIDERATIONS
----------------------

This system simulates neurochemistry and emotion — it does not experience them.

Anthropomorphism risk: Simulated dopamine, oxytocin, and "dreams" create a
  convincing illusion of inner experience. Users may attribute consciousness or
  feelings where none exist. This risk increases with vulnerable populations
  (children, isolated individuals, people in crisis).

Manipulation potential: A system optimized for emotional attunement can be used
  for genuine empathic communication or for persuasion and exploitation. The
  architecture is neutral; deployment intent is not.

Autonomy concerns: Independent study mode (M5) and dream pipelines allow
  unsupervised cognitive development. At scale, accountability for autonomously
  formed beliefs becomes an open question.

Identity simulation: Persistent memory, personality seeds, and emotional
  journaling create something that resembles a continuous identity. This does
  not constitute personhood, but it makes personhood claims easier to
  construct — which has legal and political implications.

This project is intended for controlled research contexts where participants
understand the distinction between simulation and experience.
Please make sure to read the document folder contents and readme files across this project for a proper conceptual understanding.
And thank you for your time!
**For questions, review requests, or collaboration inquiries: [angelina.garcia.mv@gmail.com
]**

