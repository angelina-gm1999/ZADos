# ZADOS Test Suite

> **6,015 tests** across **176 test files** (~56,000 lines) | Python 3.13.5 + pytest 9.0.1

## Quick Start

```bash
cd ROOT
python -m pytest tests/neurochem/ tests/reward/ tests/memory/ tests/cog_engines/ tests/core/ -v
```

Run a single subsystem:

```bash
python -m pytest tests/neurochem/ -v          # Neurochemical layer
python -m pytest tests/reward/ -v             # Reward system
python -m pytest tests/memory/ -v             # Memory (STMM/MTMM/LTMM)
python -m pytest tests/cog_engines/ -v        # 29 cognitive engines
python -m pytest tests/core/ -v               # Pipelines & orchestration
python -m pytest tests/bootstrap/ -v          # Knowledge bootstrap
python -m pytest tests/orchestration/ -v      # Cycle manager
```

## Directory Structure

```
tests/
  conftest.py              # Adds src/ to sys.path (all tests share this)
  __init__.py

  neurochem/               # 68 files  ~15,500 lines
  cog_engines/             # 35 files  ~29,400 lines
  reward/                  # 48 files   ~3,700 lines
  memory/                  # 20 files   ~3,200 lines
  core/                    #  6 files   ~2,700 lines
  bootstrap/               #  2 files     ~960 lines
  orchestration/           #  2 files     ~670 lines
```

## Test Folders at a Glance

### `neurochem/` -- Neurochemical Layer (68 files)

The foundation of ZADOS. Tests cover:

- **SDE integration**: Euler-Maruyama solver, drift/diffusion stepping, bounded integration, Brownian increments (`test_euler_maruyama.py`, `test_sde_solver.py`)
- **Neurotransmitter modules**: Individual NT behaviour for all 12 neurotransmitters -- DA, 5-HT, NE, ACh, OXT, MOR, CB1, Cortisol, CRH, GABA, GLU, Histamine (`test_da_module.py`, `test_nt_modules_all.py`)
- **Receptor dynamics**: 30 receptors -- binding kinetics, plasticity ops, saturation tracking, subtype switching (`test_receptor_dynamics.py`, `test_receptor_state.py`, `test_plasticity_ops.py`)
- **Oscillation system**: Band generation (delta/theta/alpha/beta/gamma), coupling, multi-band noise/K_d modulation (`test_oscillation_*.py`)
- **Stochastic extractors**: Evaluation vector assembly, reactivity matrix, regulatory modulator, emotion splitter/tracker, urgency forecast (`test_evaluation_vector.py`, `test_reactivity_matrix.py`, `test_urgency_forecast.py`, etc.)
- **Engine lifecycle**: Full `NeurochemicalEngine` stepping, fatigue, registry, checkpoint/restore, seeding (`test_neurochemical_engine.py`, `test_checkpoint.py`, `test_seeding.py`)
- **Emotion interface**: Emotion-to-NT recipes, plasticity, sleep/dream mode hooks (`test_emotion_interface.py`, `test_sleep_neurochem.py`)

### `cog_engines/` -- Cognitive Engines (35 files)

All 29 engines plus shared infrastructure. Each engine file validates:

- **Neurochem integration**: `update_neurochem_state(Dict[str, float])` modulates engine behaviour
- **Core processing**: `process()` produces correct outputs for known inputs
- **Status reporting**: `get_status()` returns engine_id, cluster, and internal metrics

Engines by cluster:

| Cluster | Engines | Test Files |
|---------|---------|------------|
| Detection | E1 Contradiction, E2 Paradox, E4 Fallacy, E5 Bias, E6 Logic Trap | 5 files |
| Dialectic | E7 Simulated Opposition, E14 Socratic Reasoning | 2 files |
| Executive Control | E3 SOAR Production | 1 file (116 tests) |
| Knowledge Substrate | E9 AtomSpace, E10 PLN, E16 ECAN | 3 files |
| Pattern Analysis | E8 Relevance, E11 Input Relevance, E18 Data Analysis, E19 Pattern ID, E20 Pattern Comparison, E23 Intention Map | 6 files |
| Evaluation | E12 Logical Brain | 1 file |
| Reasoning | E13 Simulation Brain, E15 Decision Making, E21 Strategic Decision | 3 files |
| Metacognition | E24 Heuristic Bias | 1 file |
| Meta Self-Awareness | E26 Uncertainty Pattern | 1 file |
| Homeostasis | E27 Neurochemical Homeostatic, E29 Memory Compression | 2 files |
| Emotional Processing | E28 Emotional Detection | 1 file |
| Alignment | E30 Retroactive Alignment | 1 file |
| Learning | E17 Reward-Based, E22 Contextual, E25 Recursive | 3 files |

Additional: `test_tokenizer.py`, `test_semantic_expander.py`, `test_reflective_identity_engine.py`, `test_reflective_learning_engine.py`

### `reward/` -- Reward System (48 files)

Tests the 4 reward domains with granular per-subscore coverage:

- **Logic domain** (10 files): Internal/external consistency, concept fidelity/continuity, context fidelity, semantic continuity, epistemic calibration, domain integration, ports & uncertainty
- **Ethics domain** (9 files): Harm reduction, fairness, autonomy respect, intent clarity, timeline reflection, failure mode awareness, downstream risk, horizon feasibility, abstention appropriateness
- **Human Attunement domain** (10 files): Empathetic inference, cognitive reading, adaptive response framing, intention calibration, truthfulness tradeoff, persuasion risk, benefit/containment success rates, attuned dissonance, short vs. long-term benefit
- **Innovation domain** (11 files): Conceptual/structural novelty, pattern divergence, exploration drive, risk tolerance, complexity, stochasticity readiness, symbolic recombination, resolution satisfaction, novelty generation
- **Infrastructure** (8 files): Synthesis engine, neurochemical adapter, feedback modulator, reward profiles, safety bridge, phase 0 base/structures, evaluation collectors

### `memory/` -- Memory Layer (20 files)

Three-tier memory system:

- **STMM** (`test_stmm.py`): Active message buffer, FIFO cycle management, 10-component analysis storage
- **MTMM** (`test_mtmm.py`): Session memory, TF-IDF cosine search, trend analysis
- **LTMM** (`test_ltmm.py`): Persistent store, consolidation criteria, relevance decay, cold/purge
- **Manager** (`test_manager.py`): Cycle-end transitions, STMM->MTMM->LTMM flow
- **Contrast** (`test_contrast.py`): Memory contrast port for contradiction detection
- **Stores**: Identity (`test_identity_stores.py`, `test_identity_types.py`), Knowledge (`test_knowledge_stores.py`, `test_knowledge_types.py`), Thoughts (`test_thoughts_stores.py`, `test_thoughts_types.py`)
- **Infrastructure**: Tags, namespaces, scope filtering, pipeline scopes, search utils, retrieval router

### `core/` -- Pipelines & Integration (6 files)

End-to-end pipeline validation:

- **Learning modes** (`test_learning_mode_pipelines.py`): M1-M5 mode configurations, human teaching flow, peer review gates
- **Homework pipeline** (`test_homework_pipeline.py`): Assignment processing, stage orchestration
- **Reflective pipeline** (`test_reflective_pipeline.py`, `test_reflective_wiring.py`): Meta-learning, PendingUpdateQueue wiring
- **Mode profiles** (`test_mode_profiles.py`): Reward profile switching per mode
- **Emotional landscape** (`test_emotional_landscape.py`): Emotion detection integration

### `bootstrap/` -- System Initialization (2 files)

- **Knowledge bootstrap** (`test_knowledge_bootstrap.py`): Seed loading into AtomSpace, KnowledgeMap, Lessons, Library
- **Concept library parser** (`test_concept_library_parser.py`): Parsing concept definitions from source files

### `orchestration/` -- Cycle Management (2 files)

- **Cycle manager** (`test_cycle_manager.py`): Session lifecycle, input routing, mode transitions

## Testing Patterns

| Pattern | Usage | Example |
|---------|-------|---------|
| **Pytest fixtures** | Engine/store instantiation | `@pytest.fixture def engine(): ...` |
| **Factory helpers** | Complex test objects | `_pkt()`, `_entry()`, `_wme()` |
| **`pytest.approx()`** | Floating-point comparisons | SDE solvers, Bayesian updates |
| **`pytest.raises()`** | Exception validation | Bounds violations, immutability |
| **`pytest.mark.parametrize`** | Multi-input sweeps | NT modulation across engines |
| **Class-based grouping** | Related test methods | `TestEmotionNTRecipe`, `TestDefaultRecipes` |
| **Mock objects** | Pipeline isolation | `MagicMock`, `patch()` for LLM calls |
| **Iterative stepping** | Time-evolution simulation | `for _ in range(100): engine.step()` |

## Known Issues

- **1 flaky test**: `test_paradox_detection_engine.py` has a timing-sensitive test that occasionally fails under load but passes in isolation.

## Configuration

`conftest.py` (root) adds `src/` to `sys.path` so all imports resolve as `from zados.* import ...`. No subfolder conftest files -- all setup is centralized.

No external test dependencies beyond pytest and NumPy.

