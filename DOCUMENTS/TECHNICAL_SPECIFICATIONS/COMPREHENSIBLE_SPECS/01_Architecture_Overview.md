# ZADOS Architecture Overview

## What is ZADOS?

ZADOS (Zonal Adaptive Dynamics Operating System) is a bio-inspired cognitive architecture that wraps large language models in a simulated neurochemical and cognitive processing layer. Instead of treating an LLM as a stateless text generator, ZADOS gives it something resembling a mind: persistent memory across conversations, emotional responses that shape how it thinks, a reward system that guides its reasoning, and 29 specialized cognitive engines that analyze, reason, and learn.

Think of it this way: if a standard LLM is a brain in a jar, ZADOS is the entire nervous system — complete with neurotransmitters, emotions, memory consolidation (even sleep), and the ability to learn from its own mistakes.

---

## The Four Layers

ZADOS is built from four interconnected layers, each responsible for a different aspect of cognition:

```
┌─────────────────────────────────────────────────────┐
│                    CORE LAYER                        │
│        (Orchestration, Pipelines, Session Mgmt)      │
│                                                      │
│   ┌──────────┐  ┌──────────────┐  ┌──────────────┐  │
│   │ NEUROCHEM│←→│   REWARD     │←→│  COGNITIVE   │  │
│   │  LAYER   │  │   SYSTEM     │  │  ENGINES     │  │
│   └────┬─────┘  └──────┬───────┘  └──────┬───────┘  │
│        │               │                 │           │
│        └───────────────┼─────────────────┘           │
│                        │                             │
│              ┌─────────┴─────────┐                   │
│              │   MEMORY LAYER    │                   │
│              │  (STMM/MTMM/LTMM)│                   │
│              └───────────────────┘                   │
└─────────────────────────────────────────────────────┘
```

### 1. Neurochemical Layer
Simulates 12 neurotransmitters (dopamine, serotonin, norepinephrine, etc.) using stochastic differential equations. These aren't decorative — they actively modulate how every other part of the system behaves. High dopamine makes the system more exploratory and novelty-seeking. Elevated cortisol makes it more cautious. Serotonin stabilizes mood and reasoning. The neurochemistry creates a continuous internal state that colors all cognition, just as it does in biological brains.

### 2. Reward System
A four-domain evaluation framework (Logic, Ethics, Innovation, Human Attunement) that scores every response the system produces. These scores drive 8 response-shaping directives (tone, structure, metaphor density, reasoning depth, etc.) and feed back into the neurochemical layer as reward signals. The reward system decides whether the system should speak, stay silent, or abstain — and shapes *how* it speaks when it does.

### 3. Cognitive Engines (29 total)
Specialized processing modules organized into 13 functional clusters. These range from contradiction detection and fallacy identification to probabilistic logic networks, attention economies, and meta-learning systems. Each engine receives the current neurochemical state and produces domain-specific analysis. Together, they form the system's analytical and reasoning backbone.

### 4. Memory Layer
A three-tier temporal hierarchy:
- **STMM (Short-Term)**: Working memory for the current turn — holds perception results, emotion states, engine outputs
- **MTMM (Mid-Term)**: Session-scoped memory that accumulates compressed turn data and tracks trends
- **LTMM (Long-Term)**: Persistent cross-session storage with 16+ specialized stores covering identity, knowledge, thoughts, and learning history

Memory consolidation happens automatically: important experiences are promoted from short-term to long-term based on emotional significance, reward scores, and pattern detection.

---

## How a Conversation Turn Works

Every user message flows through a **7-phase pipeline**. Here's what happens, step by step:

### Phase 0 — Input Validation
The system checks that the input is well-formed, not blocked by safety constraints, and ready for processing. Fast and simple.

### Phase 1 — Perception
Five cognitive engines analyze the raw input:
- **E23 (Intention Map)**: Classifies the user's intent into 8 categories (connection, challenge, exploration, pragmatic, etc.)
- **E8 (Relevance Scoring)**: Scores every concept on 6 axes (recency, frequency, semantic proximity, attention weight, contextual fit, novelty)
- **E11 (Input Relevance)**: Filters out low-relevance content
- **E18 (Data Analysis)**: Extracts entities and relationships (who did what to whom)
- **E19 (Pattern Identification)**: Detects recurring patterns across the conversation

### Phase 3 — Engine Dispatch (runs before Phase 2)
Based on the intent classification, the system activates a subset of its 29 cognitive engines. An analytical question might activate 18 engines; a social greeting might only need 4. Each active engine receives the current neurochemical state and produces its analysis.

**Why Phase 3 before Phase 2?** Because the emotional detection engine (E28) runs in this phase, and its output is needed for the neurochemical modulation that happens next.

### Phase 2 — Neurochemical Modulation
The emotion detection results from Phase 3 are translated into neurochemical signals. The system's 12 neurotransmitters are updated via stochastic differential equations, producing a new internal state. This state determines cognitive metrics like motivation, empathy, precision, anxiety, and openness — which in turn influence how the LLM will be prompted.

### Phase 4 — Internal Thinking (LLM Pass 1)
The system assembles a rich context block and sends it to the LLM for an internal reasoning pass (called "Verbalized Thinking" or VT). This is a 150-300 word internal monologue that the user never sees. It includes:
- The current emotional and neurochemical state
- Relevant memory matches from previous conversations
- Engine analysis flags and patterns
- Identity anchors and personality context

### Phase 5 — Reward Evaluation
The internal thinking is evaluated through two parallel pathways:
- **Tonic pathway**: Domain evaluators score the reasoning on logic, ethics, innovation, and human attunement. These produce 8 response-shaping directives.
- **Phasic pathway**: Emotion-driven signals generate fast neurochemical bursts that capture reactive emotional responses.

The reward evaluation can decide to suppress the response entirely (if it fails quality thresholds) or to abstain (if the system recognizes it shouldn't answer).

### Phase 6 — Response Generation (LLM Pass 2)
Using the reward directives, emotional context, and thinking trace, the system prompts the LLM for the final user-facing response. The prompt is shaped by the 8 directives — for example, high human attunement scores increase warmth and soothing language, while high logic scores increase precision and structure.

### Phase 7 — Post-Processing & Learning
After the response is generated:
1. The turn is compressed into a memory packet and stored in MTMM
2. **E29** assigns a compression policy (verbatim, semantic, symbolic, or prune)
3. **E17** computes reward prediction errors and adjusts learning parameters
4. **E22** fingerprints the conversational context for future recognition
5. **E25** monitors overall learning effectiveness and adjusts meta-parameters
6. Reward feedback closes the neurochemical loop
7. High-significance experiences are flagged for LTMM promotion

---

## Beyond Regular Conversation

ZADOS doesn't just handle normal back-and-forth. It supports several specialized modes:

### Learning Modes (M1-M5)
Five structured learning modes where ZADOS can be taught, review knowledge with a peer, explore topics collaboratively, investigate its own questions, or study independently. Each mode has its own neurochemical preset, engine configuration, and question-handling rules.

### Sleep Modes
- **REM**: Consolidates session memories by scanning for emotionally significant experiences, promoting them to long-term storage, and adjusting domain weights based on learning signals
- **Dream**: Pulls stagnated unresolved questions and attempts creative recombination — generating novel connections that might break through impasses

### Meta-Learning
- **Homework**: Offline batch processing of accumulated learning logs — decomposes content, resolves contradictions, finalizes lessons, updates knowledge maps
- **Reflective**: Self-assessment and strategic planning based on learning history and neurochemical patterns

### Self-Reflective Queries
When the system detects self-reflective markers in the conversation and has unresolved questions in its buffer, it can enter a self-examination mode — probing its own held thoughts and uncertainties.

---

## How the Layers Talk to Each Other

The four layers are deeply interconnected. Here are the key feedback loops:

### Neurochem ←→ Reward
The reward system evaluates outputs and produces domain scores. These scores are translated into neurochemical signals (dopamine for novelty/reward, norepinephrine for precision needs, oxytocin for social attunement). The neurochemical state then influences how the next evaluation is conducted — creating a continuous feedback loop.

### Neurochem ←→ Cognitive Engines
Every cognitive engine receives the current neurochemical state before processing. High acetylcholine makes detection engines more vigilant. High dopamine makes pattern engines more novelty-seeking. High GABA suppresses impulsive responses. The engines also emit signals back — a contradiction detection creates a norepinephrine spike; a novel pattern triggers a dopamine burst.

### Reward ←→ Memory
Reward scores determine which experiences get promoted to long-term memory. High emotional significance or reward scores trigger LTMM promotion. During sleep modes, the reward system relaxes its thresholds to allow more creative consolidation.

### Memory ←→ Cognitive Engines
Engines query memory during processing (E8 uses retrieval frequency for relevance scoring, E20 compares against stored pattern templates). After processing, engines write results back (E19 stores confirmed patterns, E9 persists its knowledge graph). The learning engines (E17, E22, E25) explicitly read and write learning state.

### Core ←→ Everything
The Core layer orchestrates all of this. The SessionOrchestrator manages the lifecycle (open → process turns → close). The InputClassifier routes messages to the right pipeline. The AnswerPipeline sequences the 7 phases. And the MemoryImplementationManager ensures consistent writes across all memory tiers.

---

## System at a Glance

| Aspect | Details |
|--------|---------|
| **Neurotransmitters** | 12 (DA, 5-HT, NE, ACh, OXT, GABA, Glu, Cortisol, CRH, CB1, MOR, Histamine) |
| **Cognitive Engines** | 29, organized in 13 clusters |
| **Reward Domains** | 4 (Logic, Ethics, Innovation, Human Attunement) with 37 submodules |
| **Memory Tiers** | 3 (STMM, MTMM, LTMM) with 16+ specialized stores |
| **Response Directives** | 8 (tone, structure, metaphor density, reasoning depth, moralize, clarify, speculate, soothe) |
| **Emotion Taxonomy** | 46 emotions across 7 functional groups |
| **Learning Modes** | 5 structured modes (M1-M5) + Sleep (REM/Dream) + Meta-Learning (Homework/Reflective) |
| **LLM Passes per Turn** | 2 (internal thinking + final response) |
| **Oscillatory Bands** | 6 (Delta, Theta, Alpha, Beta, Gamma, Sigma) with cross-frequency coupling |
| **Reward Profiles** | 17 presets for different operational contexts |
| **Tests** | 6,015+ passing |

---

## What Makes ZADOS Different

1. **Continuous internal state**: Unlike stateless LLMs, ZADOS maintains a rich neurochemical state that evolves over time and colors all cognition
2. **Multi-pass reasoning**: Every response goes through internal thinking first, gets evaluated by the reward system, and only then generates the user-facing answer
3. **Persistent learning**: The system genuinely learns across sessions — tracking what works, what doesn't, and adjusting its approach
4. **Emotional grounding**: Emotions aren't simulated for show; they drive real computational changes in how the system processes information
5. **Self-monitoring**: Multiple engines (E24, E26, E30, E32) constantly audit the system's own reasoning for biases, uncertainty patterns, and identity drift
6. **Sleep and dreams**: Dedicated consolidation modes that process accumulated experience, resolve stagnated questions, and generate creative connections

---

*For detailed documentation on each layer, see:*
- [Reward System](02_Reward_System.md)
- [Neurochemical Layer](03_Neurochemical_Layer.md)
- [Memory System](04_Memory_System.md)
- [Cognitive Engines](05_Cognitive_Engines.md)
