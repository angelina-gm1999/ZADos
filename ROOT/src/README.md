[README.md](https://github.com/user-attachments/files/26317973/README.md)
# ZADOS Source Code

> **Zonal Adaptive Dynamics Operating System** — a biologically-inspired cognitive architecture
>
> Python 3.13.5 | ~370 source files | 6,135 tests passing

## Package Map

```
src/zados/
  neurochem/            Neurochemical simulation layer (SDE, NTs, receptors, oscillations)
  reward/               4-domain reward evaluation + synthesis + feedback loop
  memory/               3-tier memory (STMM → MTMM → LTMM) + specialized stores
  cognitive_engines/    29 cognitive engines across 13 functional clusters
  core/                 Session lifecycle, 8-phase answer pipeline, input modes
  orchestration/        Cycle manager, mode transitions
  LLM_interpretation/   LLM bridge layer (Phase 5 evaluator)
  bootstrap/            Knowledge seeding on first boot
```

### Dependency Flow (top → bottom)

```
              core/
          (SessionOrchestrator, AnswerPipeline)
           ┌────┼────┬────────┐
           ▼    ▼    ▼        ▼
     memory/  cognitive_engines/  LLM_interpretation/
           │    │
           ▼    ▼
         reward/
           │
           ▼
       neurochem/     ← foundation, no upward imports
```

The neurochemical layer is the foundation — it has zero imports from other ZADOS packages. The reward system depends only on neurochem. Memory and cognitive engines depend on reward and neurochem. The core layer sits on top and orchestrates everything.

---

## Architecture Overview

### Layer 1: Neurochemical Simulation (`neurochem/`)

The mathematical foundation. Simulates 12 neurotransmitters and 30 receptors using stochastic differential equations (Euler-Maruyama integration), producing real-time signals that modulate every other layer.

```
neurochem/
  core/                    Engine lifecycle
    engine.py              NeurochemicalEngine — online real-time stepper
    simulation.py          SimulationRunner — batch offline experiments
    registry.py            Component registry (NTs, receptors, configs)
    scheduler.py           SparseUpdateScheduler for performance

  state/                   State containers (mutable dataclasses)
    neurotransmitter_state.py   C_tonic, C_phasic, F (fatigue), eta_u (transporter)

  kinetics/                Pure functions — reaction kinetics
    mass_balance.py        Drift, diffusion, Fick's law clearance, OU reversion
    release_drives.py      Tonic/phasic release computation, fatigue gating
    receptor_dynamics.py   Binding kinetics, saturation, effective concentration
    reuptake.py            Transporter kinetics
    fatigue.py             Activity-dependent fatigue accumulation

  stochastic_modulation/   SDE solvers
    euler_maruyama.py      euler_maruyama_step_bounded() — bounded EM integration
    noise_models.py        Wiener process, colored noise generators

  neurotransmitters/       Per-NT concrete modules (12 total)
    base.py                NeurotransmitterModule ABC + ReleaseDriveSpec + OscillationCouplingRule
    dopamine.py            DAModule — DA-specific release drives & oscillation coupling
    serotonin.py           SerotoninModule (5-HT)
    ...                    (one file per NT)
    configs.py             DEFAULT_NT_CONFIGS, DEFAULT_RECEPTOR_CONFIGS, NT_RECEPTOR_MAP

  receptors/               Per-receptor family modules (30 receptors)
    base.py                ReceptorFamilyModule ABC + ReceptorSpec

  oscillations/            Oscillatory band system
    oscillation_generator.py   Waveform production (delta/theta/alpha/beta/gamma/sigma)
    oscillation_modulation.py  modulate_noise(), modulate_K_d(), compute_g_chi()
    ...

  extractors/              Stochastic extraction pipeline
    evaluation_vector.py   8-axis E(t) assembly from reward domains
    reactivity_matrix.py   20 NT-axis coupling entries with threshold gating
    regulatory_modulator.py  4 tau-smoothed pathways (OXT, CB1, NE, GABA_B)
    emotion_tracker.py     12 per-emotion leaky integrators
    emotion_splitter.py    4M/4R modulatory-reactive split
    urgency_forecast.py    5-axis threshold prediction + NE/DA reactive bursts
    stochastic_impulse.py  Gamma/Poisson/Lognormal generators
    leaky_integrator.py    Generic leaky integrator + EMA
    extractor_orchestrator.py  Sequences all extractors in one step() call

  neurosymbolic/           Readout & symbolic encoding
    readout.py             compute_neurosymbolic_readout() — NT state → cognitive metrics
    tags.py                Neurosymbolic triplet encoding (K.3: NT• phasic, NT~ tonic)

  sleep/                   Sleep/dream mode neurochemistry
  inference_matrix/        NT inference matrix
  optimization/            Seeding, checkpointing, timescale, logging, batch runner, analysis
  config/                  NTConfig, ReceptorConfig typed wrappers
```

### Layer 2: Reward System (`reward/`)

Evaluates system outputs across 4 domains, synthesizes a meta-directive, and feeds back into the neurochemical engine.

```
reward/
  base/
    types.py               RewardContext, RewardSubscore, RewardDomainResult,
                            RewardWeights, RewardMetaDirective (all frozen dataclasses)

  domains/                 4 evaluation domains
    ethics/                9 evaluators (harm reduction, fairness, autonomy, ...)
    logic/                 10 evaluators (consistency, fidelity, calibration, ...)
    human_attunement/      10 evaluators (empathy, cognitive reading, ...)
    innovation/            11 evaluators (novelty, divergence, exploration, ...)

  synthesis/
    engine.py              SynthesisEngine — merges domain results into RewardMetaDirective
    directives.py          Pure functions: classify_tier, compute_weighted_composite, ...

  adapter/
    neurochemical_adapter.py  Transforms RewardMetaDirective → NT modulation signals

  feedback/
    modulator.py           4 feedback pathways (OXT←Attunement, CB1←Innovation,
                            NE←Logic×ContradictionLoad, GABA_B←Ethics×TimelineMismatch)

  profile/                 Reward profile taxonomy and presets
  safety/                  Safety bridge — hard stops on ethical violations
  evaluation/              Multi-evaluator aggregation utilities
```

### Layer 3: Memory (`memory/`)

Three temporal tiers with consolidation, relevance decay, and specialized persistent stores.

```
memory/
  types.py                 MemoryPacket, MemoryTier, CompressionLevel, SpeakerID
  __init__.py              MemoryLayer facade — wires all tiers together

  short_term/              STMM — per-cycle working memory
    store.py               STMMStore (FIFO 2+2 messages, 10 analysis components)
    components.py          ActiveMessageBuffer, FractalDecompositionResults,
                            IntentionAnalysisResults, EmotionDetectionResults, ...
    compressor.py          MemoryExitCompressor → MemoryPacket

  mid_term/                MTMM — session-scoped memory
    store.py               MTMMStore — write() + search() + trends + validate()
    logger.py              RawInteractionLogger — MemoryPacket log + re-compression
    trends.py              Trend analysis (contradiction, emotion, reward, intention)
    context_processor.py   TF-IDF cosine search (hand-rolled, no external ML dep)

  long_term/               LTMM — persistent cross-session memory
    store.py               LTMMStore + LTMMEntry + Granularity enum
    consolidation.py       MTMM→LTMM promotion criteria
    relevance.py           Relevance decay: exp(-λ·hours), half-life 1 week
    fractal_comparator.py  Deduplication, merge, reinforce, cross-link
    specialized_logs.py    8 specialized logs (see below)
    retrieval_router.py    Query-type-based retrieval routing
    search_utils.py        Shared search utilities
    tags.py                Tagging system
    namespaces.py          IdentityNamespace, ThoughtsNamespace, KnowledgeNamespace

    identity/              Identity stores (never demoted)
    knowledge/             Library, KnowledgeMap, question stores
    thoughts/              Overview logs, pending updates
    journal/               Cognitive journal
    unsolved_buffer/       Unsolved concepts frontier

  managers/
    implementation.py      MemoryImplementationManager — on_cycle_end, consolidate
    contrast.py            MemoryContrast — contradiction cross-referencing port
```

### Layer 4: Cognitive Engines (`cognitive_engines/`)

29 engines organized into 13 clusters. All follow a unified interface.

```
cognitive_engines/
  constants.py             Canonical NT keys, oscillatory bands, engine IDs, clusters

  py_engines/              26 runtime engines (one .py per engine)
    contradiction_detection_engine.py    E1  — detection cluster
    paradox_detection_engine.py          E2
    soar_production_engine.py            E3  — executive control
    fallacy_detection_engine.py          E4
    bias_detection_engine.py             E5
    logic_trap_detection_engine.py       E6
    simulated_opposition_engine.py       E7  — dialectic
    relevance_scoring_engine.py          E8  — pattern analysis
    input_relevance_evaluation_engine.py E11
    logical_brain_engine.py              E12 — evaluation
    simulation_brain_engine.py           E13 — reasoning
    socratic_reasoning_engine.py         E14 — dialectic
    decision_making_engine.py            E15
    data_analysis_engine.py              E18
    pattern_identification_engine.py     E19
    pattern_comparison_engine.py         E20
    strategic_decision_engine.py         E21
    contextual_learning_engine.py        E22
    intention_map_engine.py              E23
    heuristic_bias_engine.py             E24
    recursive_learning_engine.py         E25
    uncertainty_pattern_engine.py        E26
    neurochemical_homeostatic_engine.py  E27
    emotional_detection_engine.py        E28
    memory_compression_engine.py         E29
    retroactive_alignment_engine.py      E30
    reward_based_learning_engine.py      E17

  cognitools/              3 knowledge-substrate engines (OpenCog replacements)
    atomspace_engine.py    E9  — typed hypergraph
    pln_engine.py          E10 — probabilistic logic network
    ecan_engine.py         E16 — attention economy

  SOAR_engines/            Reserved / legacy SOAR reference
  hyperon_engines/         Reserved / legacy Hyperon reference
```

### Layer 5: Core Pipeline (`core/`)

Session lifecycle and the 8-phase answer pipeline.

```
core/
  session.py               SessionOrchestrator — boot, per-turn, close lifecycle
  pipeline.py              AnswerPipeline — Phase 0→7 sequencer
  types.py                 InputBundle, PipelineResult, PipelineState, SessionState, ...
  mode_profiles.py         Reward profile selection per processing mode
  dispatch_table.py        Engine dispatch routing
  time_context.py          TimeContext + temporal awareness
  tags.py                  Pipeline-level tagging

  phases/                  8 processing phases (one file each)
    phase0_reception.py    Input validation
    phase1_perception.py   Intent, relevance, entity extraction, pattern detection
    phase2_modulation.py   NT modulation + mode selection
    phase3_dispatch.py     Engine dispatch (which engines fire this turn)
    phase4_thinking.py     Multi-engine thinking pass
    phase5_reward.py       Reward evaluation (4 domains)
    phase6_answer.py       Answer generation
    phase7_postprocess.py  Memory writes, journal, learning updates, STMM→MTMM

  inputs/                  Input mode pipelines
    regular_input_mode/    Standard conversation processing
    learning_modes/        Structured learning (M1-M5: human teaching, peer review, ...)
    self_ref_query_mode/   Self-referential query handling

  commanded/               Commanded mode pipelines
    meta_learning_mode/
      reflective_mode/     Meta-learning + pending update queue
      homework_mode/       Assignment processing

  processes/               Long-running processes
  thinking_blocks/         Thinking block builder for introspection output
```

---

## Design Decisions

### 1. Pure Functions + Mutable State Containers

The codebase enforces a strict separation between **computation** (pure functions) and **state** (mutable dataclasses).

**State containers** are mutable dataclasses with bounds enforcement in `__post_init__`:

```python
@dataclass
class NeurotransmitterState:
    C_tonic: float = 0.5       # Baseline concentration
    C_phasic: float = 0.0      # Burst concentration
    F: float = 0.0             # Fatigue [0, 1]
    eta_u: float = 1.0         # Transporter efficiency [0, 1]

    def __post_init__(self):
        self.C_tonic = max(0.0, self.C_tonic)
        self.F = max(0.0, min(1.0, self.F))
        ...
```

**Kinetics functions** are pure — they take values, return values, never mutate:

```python
def compute_reuptake_loss(C: float, eta_u: float, u_base: float = 0.1) -> float:
    """L_u(t) = u_base · η_u(t) · C(t)"""
    return u_base * eta_u * C
```

**Why**: Pure functions are trivially testable (68 test files in neurochem alone). State containers are inspectable and serializable. The engine orchestrates the interplay between the two.

### 2. Frozen Dataclasses for Outputs

All reward types, configurations, and engine configs are `frozen=True`:

```python
@dataclass(frozen=True)
class RewardMetaDirective:
    allow_output: bool = True
    abstain: bool = False
    suppress: bool = False
    directives: Dict[str, Any] = field(default_factory=dict)
    ...
```

```python
@dataclass(frozen=True)
class ContradictionEngineConfig:
    p_base: float = 0.1
    alpha_mc: float = 0.3
    ...
```

**Why**: Frozen dataclasses prevent accidental mutation of configuration and evaluation results as they flow through the pipeline. If you need different config, construct a new instance.

### 3. Unified Engine Interface (Pattern A)

All 29 cognitive engines implement the same three-method interface:

```python
class AnyEngine:
    def update_neurochem_state(self, nt_levels: Dict[str, float]) -> None:
        """Receive current NT concentrations. Keys from constants.NT_KEYS."""

    def process(self, input_data) -> result:
        """Run the engine's core logic. Input/output types are engine-specific."""

    def get_status(self) -> Dict[str, Any]:
        """Return engine_id, cluster, and internal metrics."""
```

The `nt_levels` dict uses canonical keys from `constants.py` (`"da"`, `"5ht"`, `"ne"`, `"ach"`, etc.). Two coherence audits (Sessions 24, 28) enforced this across all 29 engines.

**Why**: The pipeline orchestrator (`phase3_dispatch.py`) can treat all engines uniformly. NT modulation is always a dict — no positional args, no `**kwargs`, no `_level` suffixed variants.

### 4. No External ML Dependencies

All mathematical operations are hand-implemented:

- **TF-IDF cosine similarity** in `memory/mid_term/context_processor.py` — for memory search
- **Bayesian inference** in cognitive engines — for confidence tracking
- **Euler-Maruyama SDE integration** in `stochastic_modulation/` — for neurochemical dynamics
- **Leaky integrators** in `extractors/leaky_integrator.py` — for temporal smoothing

The only external dependency is **NumPy** (for RNG, array math, and the SDE solver). No TensorFlow, PyTorch, scikit-learn, or any ML framework.

**Why**: ZADOS is a cognitive architecture, not a neural network. The "intelligence" comes from the interplay of 29 engines modulated by neurochemistry, not from trained weights. Keeping dependencies minimal makes the system fully inspectable and deterministic under seeded RNG.

### 5. Facade Pattern for Complex Subsystems

Major subsystems expose a facade class that wires internals together:

```python
class MemoryLayer:
    """Convenience facade that wires all three tiers together."""
    def __init__(self):
        self.stmm     = STMMStore()
        self.mtmm     = MTMMStore()
        self.ltmm     = LTMMStore()
        self.identity, self.thoughts, self.knowledge = build_namespaces()
        self.manager   = MemoryImplementationManager(self.mtmm, self.ltmm, ...)
        self.contrast  = MemoryContrast(self.mtmm, self.ltmm, ...)
        self.router    = RetrievalRouter(self.ltmm, ...)
```

**Why**: Consumers (like `SessionOrchestrator`) interact with `MemoryLayer`, not 15 individual stores. Internal wiring is encapsulated. Tests can still target individual components directly.

### 6. Stateless Orchestrators with Pure-Function Delegation

Orchestrator classes (like `SynthesisEngine`) are stateless thin shells that delegate to pure functions:

```python
class SynthesisEngine:
    """Stateless orchestrator — all computation delegated to pure functions
    in zados.reward.synthesis.directives."""

    def synthesize(self, domain_results, ...):
        per_domain_tiers = {name: classify_tier(r.general_score) for ...}
        composite = compute_weighted_composite(domain_results, weights)
        ...
```

**Why**: The orchestrator defines *sequence* (what happens in what order). The pure functions define *computation* (how each step works). This makes each step independently testable and the overall flow easy to follow.

### 7. ABC + Concrete Module Pattern

Abstract base classes define contracts; concrete subclasses implement per-entity behavior:

```python
# Abstract
class NeurotransmitterModule(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def release_spec(self) -> ReleaseDriveSpec: ...

# Concrete (one per NT)
class DAModule(NeurotransmitterModule):
    @property
    def name(self) -> str: return "DA"
    @property
    def release_spec(self) -> ReleaseDriveSpec:
        return ReleaseDriveSpec(drives=[...], weights=[...])
```

**Why**: Each NT has unique release drive weights and oscillation coupling rules, but the engine treats them all through the same abstract interface. Same pattern for receptor family modules.

### 8. Validated Frozen Configs with `__post_init__`

Frozen dataclasses with `__post_init__` validation enforce domain invariants at construction time:

```python
@dataclass(frozen=True)
class OscillationCouplingRule:
    target: str        # must be in {"release", "reuptake", "sigma_tonic", ...}
    band: str          # must be in {"delta", "theta", "alpha", ...}
    coefficient: float
    formula: str = "multiplicative"  # must be "multiplicative" or "additive"

    def __post_init__(self):
        if self.target not in valid_targets:
            raise ValueError(f"Invalid target {self.target!r}")
        ...
```

**Why**: Invalid configurations fail immediately at construction, not at runtime deep in a simulation loop.

### 9. Canonical Constants as Single Source of Truth

`cognitive_engines/constants.py` defines all canonical names, preventing drift:

```python
NT_KEYS = frozenset({"glu", "gaba", "da", "5ht", "ne", "ach", ...})
NT_STATE_FIELD = {"da": "da_level", "5ht": "_5ht_level", ...}
OSCILLATORY_BANDS = ("delta", "theta", "alpha", "beta", "gamma", "sigma")
ENGINE_IDS = {1: "contradiction_detection", 2: "paradox_detection", ...}
ENGINE_CLUSTER_MAP = {1: "detection", 2: "detection", 3: "executive_control", ...}
```

**Why**: Before this existed (Session 24), engines used inconsistent key names (`sht_level` vs `_5ht_level` vs `sht1a_level`). The constants file + coherence audit tests prevent this from recurring.

### 10. Phase-Based Pipeline Architecture

The answer pipeline processes each turn through 8 sequential phases:

```
Phase 0 — Reception       Validate InputBundle
Phase 1 — Perception      Intent detection, relevance scoring, entity extraction, patterns
Phase 2 — Modulation      NT modulation, mode selection, reward profile assignment
Phase 3 — Dispatch        Route to relevant engine subset based on mode + NT state
Phase 4 — Thinking        Multi-engine thinking pass (engines process in parallel)
Phase 5 — Reward          4-domain reward evaluation
Phase 6 — Answer          Answer generation
Phase 7 — Postprocess     Memory writes, journal, learning updates, STMM→MTMM flush
```

Each phase is a standalone function (`run_perception`, `run_nt_modulation`, etc.) that takes a `PipelineState` and returns updated results. The `AnswerPipeline` class calls them in sequence.

**Why**: Each phase is independently testable. The pipeline is a simple sequence — no complex control flow. Adding a new phase means adding one function and one call in the pipeline.

---

## Coding Conventions

| Convention | Example | Rationale |
|---|---|---|
| `from __future__ import annotations` | Top of every file | Enables forward references, consistent across Python 3.13 |
| Type hints on all public APIs | `def step(self, signals: Dict[str, float]) -> None` | Self-documenting, IDE support |
| NumPy-style docstrings | `Parameters / Returns / Examples` sections | Consistent format across ~370 files |
| `_clamp(value, lo, hi)` utility | Used across all engines | Consistent bounds enforcement |
| `str` Enums | `class SourceTag(str, Enum)` | JSON-serializable, readable in logs |
| Factory `field(default_factory=dict)` | All dict/list defaults in dataclasses | Avoids mutable default argument bug |
| Explicit `__all__` exports | In every `__init__.py` | Clean public surface, IDE auto-import support |
| Section separators | `# ====...` blocks | Visual structure in large files |
| Spec cross-references | `# (spec §2.2, steps B.1-B.9)` | Traceability to design documents |

> **Where are the specs?** Spec cross-references (e.g. `§2.2`, `steps B.1-B.9`) point to design documents in `ROOT/specs_docs/`. Each major subsystem has a corresponding specification: `Neurochemical Layer — Consolidated Spec v2.0.docx`, `Cognitive Engines — Technical Specification v1.0.docx`, `Memory Layer — Technical Specification v2.0.docx`, `Reward System — Technical Specification v1.0.docx`, and `ZA-DOS_LLM_Layer_Spec_v0.5.docx.txt`.

## Running the System

The development interface is a Godot 4.6 application documented in [`frontend/README.md`](../../frontend/README.md). ZADOS requires a local LLM backend (Ollama with LLaMA) or alternative LLM credentials — see the interface documentation for setup and first-run instructions.

To run the test suite without booting the full system:

```bash
cd ROOT
python -m pytest tests/ -v
```

See the [Test Suite README](../tests/README.md) for per-subsystem commands and testing patterns.

## How to Navigate

**Starting from a user message**, trace the flow:

1. `core/session.py` → `SessionOrchestrator.process_turn()` — entry point
2. `core/pipeline.py` → `AnswerPipeline.process_turn()` — 8-phase sequencer
3. `core/phases/phase1_perception.py` → perception engines fire
4. `core/phases/phase2_modulation.py` → neurochemical engine steps
5. `core/phases/phase3_dispatch.py` → selects which engines to run
6. `core/phases/phase4_thinking.py` → multi-engine thinking
7. `core/phases/phase5_reward.py` → reward evaluation
8. `core/phases/phase7_postprocess.py` → memory writes, learning

**Starting from neurochemistry**, trace upward:

1. `neurochem/kinetics/mass_balance.py` → pure kinetics math
2. `neurochem/core/engine.py` → `NeurochemicalEngine.step()` — applies kinetics
3. `neurochem/extractors/extractor_orchestrator.py` → reward→NT bridge
4. `reward/adapter/neurochemical_adapter.py` → transforms directives to signals
5. `core/phases/phase2_modulation.py` → pipeline calls `engine.step()`

**Starting from memory**, trace the lifecycle:

1. `memory/short_term/store.py` → `STMMStore` — current cycle
2. `memory/short_term/compressor.py` → `MemoryExitCompressor` — STMM→packet
3. `memory/mid_term/store.py` → `MTMMStore` — session accumulation
4. `memory/long_term/consolidation.py` → `MemoryConsolidationEngine` — MTMM→LTMM
5. `memory/long_term/relevance.py` → decay scoring
6. `memory/managers/implementation.py` → `on_cycle_end()` orchestrates the flow
