# ZADOS — Current State, Ongoing Work & Future Implementations

**Zonal Adaptive Dynamics Operating System**
March 2026

---

## Overview

ZADOS is a biologically-inspired cognitive architecture built in Python. It wraps large language models in a simulated neurochemical and cognitive processing layer — providing persistent memory, emotional modulation, structured reasoning, reward-conditioned cognition, identity persistence, and adaptive learning. The LLM handles natural language generation only; all cognitive processing happens in the algorithmic layer.

As of Session 36 (2026-03-18): 6,135+ tests passing, 370 source files + 176 test files, 6,135 tests collected, ~56,000 lines of test code), 0 regressions. LLM-agnostic. Runs on consumer hardware.

---

## 1. Architecture

### 1.1 Neurochemical Engine (Foundation)

- 12 neurotransmitters: DA, 5-HT, NE, ACh, OXT, MOR, CB1, Cortisol, CRH, GABA, GLU, Histamine
- 30 receptors mapped via NT_RECEPTOR_MAP
- Stochastic differential equations (SDE) with Euler-Maruyama integration
- Oscillatory bands modulate noise and receptor binding
- Pure-function kinetics: mass balance, release drives, receptor dynamics
- Stochastic extractors: evaluation vector, reactivity matrix, regulatory modulator, emotion tracker, urgency forecast — sequenced by ExtractorOrchestrator

### 1.2 Reward System

- 4 reward domains producing domain results
- SynthesisEngine merges domain results into a RewardMetaDirective
- NeurochemicalAdapter transforms directives into NT signals
- 4 feedback pathways (OXT, CB1, NE, GABA_B) close the loop back into the neurochemical engine
- Reward profiles with taxonomy and presets

### 1.3 Memory (3-Tier)

| Tier | Scope | Key Features |
|------|-------|-------------|
| STMM (Short-Term) | Current cycle | 10 components (active buffer, fractal decomposition, emotion, intent, etc.), FIFO 2+2 messages |
| MTMM (Mid-Term) | Session | Raw interaction log, TF-IDF cosine search, trend analysis (contradiction/emotion/reward/intent) |
| LTMM (Long-Term) | Persistent | Consolidation engine, relevance decay (1-week half-life), 8 specialized logs, identity entries never demoted |

**Specialized Logs**: Learning, Sandbox, Paradox, Contradiction, Unsolved Concepts, Self-Reflection, Identity Memory, Dream Log

**Knowledge Stores**: LibraryStore, KnowledgeMap, GeneralQuestionStore, AcademicQuestionStore, OverviewLogStore, IdentityJournalStore, CognitoolsDataStore, PendingUpdateQueue

### 1.4 Cognitive Engines (29 total, all complete)

| Cluster | Engines |
|---------|---------|
| Detection (5) | E1 Contradiction, E2 Paradox, E4 Fallacy, E5 Bias, E6 Logic Trap |
| Dialectic (2) | E7 Simulated Opposition, E14 Socratic Reasoning |
| Executive Control (1) | E3 SOAR Production (5-phase decision cycle) |
| Knowledge Substrate (3) | E9 AtomSpace, E10 PLN, E16 ECAN |
| Pattern Analysis (6) | E8 Relevance, E11 Input Relevance, E18 Data Analysis, E19 Pattern ID, E20 Pattern Comparison, E23 Intention Map |
| Evaluation (1) | E12 Logical Brain |
| Reasoning (3) | E13 Simulation Brain, E15 Decision Making, E21 Strategic Decision |
| Metacognition (1) | E24 Heuristic Bias |
| Meta Self-Awareness (1) | E26 Uncertainty Pattern |
| Homeostasis (2) | E27 Neurochemical Homeostatic, E29 Memory Compression |
| Emotional Processing (1) | E28 Emotional Detection |
| Alignment (1) | E30 Retroactive Alignment |
| Learning (3) | E17 Reward-Based, E22 Contextual, E25 Recursive (meta-learning) |

All 29 engines follow a unified interface: `update_neurochem_state(Dict[str, float])`, `process()`, `get_status()`.

### 1.5 Core Pipelines

- **Regular input pipeline** — 7-phase answer pipeline with LLM interpretation integration
- **Learning modes** — multi-stage pipeline with question extraction, KnowledgeMap bootstrap, identity journal writes (M1-M5: Human Teaches, Peer Review, Learn Together, Learned Questions, Independent Study)
- **Reflective mode** — meta-learning with PendingUpdateQueue
- **Homework mode** — identity-relevant emotion detection
- **Sleep & Dream modes** — neurochemical layer extension with retroactive learning
- **SessionOrchestrator** — manages full lifecycle including close_session() (overview write, consolidate, tick unsolved, end cycle, persist cognitools, cleanup)

### 1.6 Emotion Framework

- Verified against 56-page emotion framework spec
- 12 emotions tracked with per-emotion leaky integrators
- 4M/4R split: modulatory (tonic) + reactive (phasic) pathways
- Emotion detection engine (E28) feeds into neurochemical modulation

### 1.7 Identity System

- Hardcoded immutable axioms, values, and constraints in a read-only store
- Developmental identity through journal and conclusions stores
- Identity Alignment Checker runs per-turn as a soft advisory check
- Personality entries shape tone without altering factual content or ethical constraints

### 1.8 User Interface

- Development-oriented interface drafted in Godot
- Designed for development use, not public-facing
- Early versions in progress

---

## 2. Development History (Recent Sessions)

| Session | Focus | Tests Added |
|---------|-------|-------------|
| 30 | Emotion framework audit (56-page spec verification) | fixes only |
| 31 | Sleep & Dream neurochemical extension | +266 |
| 32 | Reward Profile refactoring | +180 |
| 33 | Reward → Learning Engine wiring | fixes |
| 34 | Regular input pipeline + LLM integration | +132 |
| 35 | Journal integration + TimeContext + Sleep retroactive learning | 0 (wiring) |
| 36 | LTMM Store Wiring Sweep (14+ orphaned stores connected) | +46 |

---

## 3. Tech Stack

- Python 3.13.5, pytest 9.0.1, NumPy (PCG64 RNG)
- LLM integration: Ollama (local) / Claude API
- No external ML dependencies — TF-IDF cosine search is hand-rolled
- Pure-function kinetics + mutable dataclass state containers
- All OpenCog/Hyperon components replaced with native Python (cognitools)
- Interface: Godot (development builds)

---

## 4. Current Status

All core subsystems (neurochemistry, reward, memory, cognition, emotion, learning) are implemented, tested, and wired together. Recent work has been integration-focused — connecting stores, closing lifecycle gaps, ensuring data flows end-to-end. The orchestration layer (CycleManager) formalizing engine-to-memory dispatch with dependency resolution is partially implemented.

---

## 5. Ongoing Work

### Currently Active

- **Engine and learning iteration** — refining cognitive engine behavior, learning pipeline quality, and self-reflective processes
- **Identity and self-reflection research** — ongoing investigation into identity architecture; also waiting for external support and review on classification of identity dimensions (immutable vs. developmental vs. emergent)
- **User interface** — background development in Godot; development-oriented tooling, not public-facing; early drafts exist
- **Logic and bias tagging** - implementing a tag system for optimization of logic and bias detection and processing.

### On Hold (Waiting for Support & Review)

- **Knowledge and learning library definition** — defining the specific content the system should learn, curriculum design, domain priorities, and structuring knowledge for the memory and knowledge substrate systems
- **Testing requirements definition** — designing evaluation methodology for cognitive output quality, LLM-agnostic testing framework, adaptive system regression testing, per-model alignment validation

### Current Priority

Review, training, teaching, and testing. The architecture is built; the next phase is filling it with knowledge, validating its cognitive output, and submitting it to external review. This is the reason for current outreach.

---

## 6. Future Implementations

- **Reward system expansion** — extending reward heuristics and domain evaluation depth
- **Cognitive process iteration** — further refinement of engine behavior and inter-engine dynamics
- **Modular pipeline interface** — interface functions allowing personalized processing pipelines by using cognitive engines as composable blocks
- **LLM interpretation adaptation** — extending the LLM interpretation layer to support additional models beyond Ollama and Claude API, making the LLM-agnostic design practically operational across a wider range of backends

---

## 7. Acknowledgments

The knowledge substrate engines (E9 AtomSpace, E10 PLN, E16 ECAN) are built on theoretical frameworks from the OpenCog cognitive architecture developed under the SingularityNET ecosystem (Ben Goertzel et al.). All implementations are native Python replacements, not direct ports. A detailed attribution mapping original versus derived components is forthcoming.
