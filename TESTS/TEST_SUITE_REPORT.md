# ZADOS Test Suite Report

> **Version**: Post-Session 36 (2026-03-18)
> **Total tests**: 6,015 | **Files**: 176 | **Lines**: ~56,000 | **Regressions**: 0

---

## 1. Executive Summary

The ZADOS test suite validates a biologically-inspired cognitive architecture across seven subsystems: neurochemistry, reward evaluation, memory, cognitive engines, core pipelines, bootstrap, and orchestration. Over 36 development sessions, the suite grew from initial SDE solver tests to comprehensive coverage of all 29 cognitive engines, a 4-domain reward system, 3-tier memory, and multi-mode processing pipelines.

**Key findings:**
- All 29 cognitive engines are tested and follow a unified interface (`update_neurochem_state`, `process`, `get_status`)
- The neurochemical foundation (SDE, 12 NTs, 30 receptors, oscillations) is the most deeply tested layer with 68 test files
- The reward system has granular per-subscore tests across all 4 domains (Logic, Ethics, Human Attunement, Innovation)
- Memory tier transitions (STMM -> MTMM -> LTMM) are tested at both unit and integration level
- Pipeline tests validate end-to-end data flow through learning, reflective, and homework modes
- 1 known flaky test (timing-dependent paradox detection); all other tests are deterministic or use seeded RNG

---

## 2. Coverage by Subsystem

### 2.1 Neurochemical Layer (68 files, ~15,500 lines)

**What it validates:** The mathematical and biological foundation of the entire system -- stochastic differential equations that model neurotransmitter dynamics in real time.

#### SDE Integration & Numerical Methods
| Component | File(s) | What's Tested |
|-----------|---------|---------------|
| Euler-Maruyama solver | `test_euler_maruyama.py` | Deterministic drift, stochastic diffusion, bounded integration, reflection modes, Brownian increment generation |
| SDE solver wrapper | `test_sde_solver.py` | Full solver pipeline, parameter passing, step accumulation |
| Mass balance | `test_mass_balance.py` | Fick's law diffusion, OU mean-reversion, C_extracell parameter, concentration bounds |
| Release drives | `test_release_drives.py` | Tonic/phasic release computation, NT-specific drive profiles |
| Reuptake | `test_reuptake.py` | Transporter kinetics, reuptake rate modulation |

**Conclusions:** The numerical core is deterministic under seeded RNG. Tests use `pytest.approx()` with explicit epsilon for floating-point comparisons. The Euler-Maruyama solver correctly handles boundary reflection (concentrations clamped to [0, 1]) and the mass balance equations follow documented kinetics.

#### Neurotransmitter Modules
| Component | File(s) | What's Tested |
|-----------|---------|---------------|
| All 12 NTs | `test_nt_modules_all.py` | Registration, stepping, config defaults, signal output keys |
| DA module | `test_da_module.py`, `test_dopamine_module.py` | DA-specific dynamics, legacy Dopamine class |
| NT module base | `test_nt_module_base.py` | Abstract interface compliance |
| Configs | `test_nt_configs.py`, `test_config_typing.py` | DEFAULT_NT_CONFIGS, DEFAULT_RECEPTOR_CONFIGS completeness and types |

**Conclusions:** Every NT registers correctly, produces signals with canonical keys (`"da"`, `"5ht"`, `"ne"`, etc.), and respects its configuration. The config tests catch schema drift -- if a new NT is added without proper config entries, tests fail.

#### Receptor System
| Component | File(s) | What's Tested |
|-----------|---------|---------------|
| Receptor dynamics | `test_receptor_dynamics.py` | Binding kinetics, K_d modulation, multi-band K_d |
| Receptor state | `test_receptor_state.py` | Functional state machine, saturation tracking |
| Receptor modules | `test_receptor_modules.py`, `test_engine_receptor_modules.py` | Per-receptor behaviour, parent NT inference |
| Plasticity | `test_plasticity_ops.py`, `test_emotion_plasticity.py` | Up/downregulation, sigma/rho adjustments |
| Subtype switching | `test_subtype_switching.py` | Receptor subtype transitions |

**Conclusions:** The 30 receptors are the most intricate subsystem to test. Key validation: parent NT inference via `receptor_id.split("_")[0]` with `parent_nt` config override (fixes the OXTR->OXT resolution bug found in Session 19). Plasticity ops correctly clamp at boundaries -- upregulation effects (sigma*1.3, rho*1.2) get clamped to 1.0 when starting from high values.

#### Oscillation System
| Component | File(s) | What's Tested |
|-----------|---------|---------------|
| State | `test_oscillation_state.py` | Band initialization, getters/setters, bounds enforcement |
| Generator | `test_oscillation_generator.py` | Waveform production, frequency correctness |
| Bands | `test_oscillations_bands.py`, `test_band_associations.py` | Band definitions, NT-band associations |
| Coupling | `test_oscillation_coupling.py` | Cross-band coupling dynamics |
| Modulation | `test_oscillation_modulation.py`, `test_threshold_modulation.py`, `test_multiband_noise.py`, `test_multiband_kd.py` | Noise modulation, K_d modulation, threshold effects |

**Conclusions:** Oscillation state properties enforce [0, 1] bounds via clamping (not exceptions). Band associations follow documented mapping: GLU->{gamma, theta-gamma}, ACh->{beta}, CB1->{delta, alpha-beta}, histamine->{beta, gamma}. Multi-band modulation tests verify that oscillatory envelopes modulate both noise amplitude and receptor binding correctly.

#### Stochastic Extractors
| Component | File(s) | What's Tested |
|-----------|---------|---------------|
| Evaluation vector | `test_evaluation_vector.py` | 8-axis assembly from reward domains |
| Reactivity matrix | `test_reactivity_matrix.py` | 20 NT-axis coupling entries, threshold gating |
| Regulatory modulator | `test_regulatory_modulator.py` | 4 tau-smoothed pathways |
| Emotion tracking | `test_emotion_tracker.py`, `test_emotion_splitter.py` | 12 per-emotion integrators, 4M/4R split |
| Urgency forecast | `test_urgency_forecast.py` | 5-axis threshold prediction, NE/DA reactive bursts |
| Stochastic impulse | `test_stochastic_impulse.py` | Gamma/Poisson/Lognormal generators |
| Leaky integrator | `test_leaky_integrator.py` | Generic leaky integrator, EMA, batch stepping |
| Orchestrator | `test_extractor_orchestrator.py` | Full pipeline sequencing |

**Conclusions:** The extractor pipeline is the bridge between reward evaluation and neurochemical dynamics. Tests validate the full chain: domain results -> evaluation vector -> reactivity matrix -> NT modulation signals. The urgency forecast uses a linear-exponential predictor with Poisson-driven NE/DA spikes -- tests verify both the smooth forecasting and the stochastic spike generation.

#### Engine & Infrastructure
| Component | File(s) | What's Tested |
|-----------|---------|---------------|
| Engine lifecycle | `test_neurochemical_engine.py` | Init, registration, stepping, fatigue, signal integration |
| Registry | `test_neurochem_registry.py` | NT/receptor registration, config storage |
| Simulation runner | `test_simulation_runner.py`, `test_online_simulator.py` | Batch simulation, online stepping |
| Neurosymbolic | `test_neurosymbolic_readout.py`, `test_neurosymbolic_tags.py`, `test_neurosymbolic_metrics.py` | Readout extraction, triplet encoding, phasic/tonic markers |
| Utilities | `test_checkpoint.py`, `test_seeding.py`, `test_timescale.py`, `test_scheduler.py`, `test_batch_runner.py`, `test_analysis.py` | State serialization, RNG reproducibility, time scaling |
| Sleep mode | `test_sleep_neurochem.py` | Sleep/dream neurochemical dynamics |
| Mode hooks | `test_mode_hooks.py` | Mode transition callbacks |
| Logging | `test_logging_opt.py` | Optimized logging |

**Conclusions:** The engine correctly accumulates time via `current_time` tracking (validated with `abs(engine.current_time - expected) < 1e-9`). Checkpoint/restore round-trips preserve full state. Seeding with PCG64 produces reproducible sequences. The simulation runner supports both batch mode (offline analysis) and online mode (live stepping with external input).

---

### 2.2 Cognitive Engines (35 files, ~29,400 lines)

**What it validates:** 29 cognitive engines organized into 13 clusters. This is the largest test folder by code volume, reflecting the complexity of each engine's internal logic.

#### Detection Cluster (5 engines)
| Engine | Tests | Key Validations |
|--------|-------|-----------------|
| E1 Contradiction Detection | Bayesian confidence updates, prior computation, threshold resolution, negation/semantic opposition, kappa contradiction load | NT modulation: DA broadens search, 5-HT stabilises confidence, NE urgency |
| E2 Paradox Detection | Self-referential loop detection, paradox classification, productive vs. unproductive tracking | 1 flaky timing test (passes in isolation) |
| E4 Fallacy Detection | Formal/informal fallacy templates, confidence scoring, multi-fallacy interaction | Pattern-matched detection with NT threshold shifting |
| E5 Bias Detection | 65 tests covering cognitive bias taxonomy, detection triggers, mitigation suggestions | DA boosts novelty-seeking, ACh deepens inspection |
| E6 Logic Trap Detection | 161 tests -- the most thorough engine test suite; covers all trap types, depth levels, escape strategies | Full NT modulation matrix verified |

**Conclusions:** The detection cluster is heavily tested because false negatives in contradiction/fallacy detection cascade through the whole system. E6 (Logic Trap) has the highest test count of any single engine, reflecting the complexity of trap type classification. The Bayesian confidence system in E1 was validated against manual calculations -- prior and posterior probabilities match closed-form solutions.

#### Knowledge Substrate (3 engines -- CogniTools)
| Engine | Tests | Key Validations |
|--------|-------|-----------------|
| E9 AtomSpace-Lite | 151 tests | Typed hypergraph CRUD, 15 AtomTypes, TruthValue(s,c), AttentionValue(sti,lti), O(1) index lookup, pattern matching, capacity pruning, serialization |
| E10 PLN Core | 72 tests | 12 inference rules as pure functions, backward chaining, truth-value propagation with confidence factors |
| E16 ECAN Core | 95 tests | Attention economy (rent/wage/spread/clamp/AF), HebbianLink co-activation, LTI dynamics |

**Conclusions:** These three engines replace the OpenCog Hyperon framework with pure Python implementations. AtomSpace tests verify O(1) lookup performance by index type (id, type, name, outgoing set). PLN inference tests validate truth-value propagation matches expected confidence decay formulas. ECAN tests verify the attention economy converges to stable equilibria under varied load.

#### Executive Control (1 engine)
| Engine | Tests | Key Validations |
|--------|-------|-----------------|
| E3 SOAR Production | 116 tests | 5-phase decision cycle (Input->Elaboration->Proposal->Decision->Application), WME triple indexing (O(1) by identifier/attribute), production matching/firing, operator preferences, impasse detection (TIE/CONFLICT/NO_CHANGE/STATE_NO_CHANGE), impasse delegation to other engines, chunking (learning new productions from resolved impasses) |

**Conclusions:** E3 is the executive backbone. Tests verify that impasse delegation routes correctly: TIE->E13 (Simulation), CONFLICT->E1+E14 (Contradiction+Socratic), NO_CHANGE->E7 (Opposition), STATE_NO_CHANGE->E26 (Uncertainty). Chunking tests confirm that resolved impasses produce new productions that are exported to LTMM.

#### Pattern Analysis (6 engines)
| Engine | Tests | Key Validations |
|--------|-------|-----------------|
| E8 Relevance Scoring | 70 tests | Multi-axis scoring (recency, frequency, semantic proximity, attention weight, contextual fit, novelty bonus) |
| E11 Input Relevance | Standard | Input-level relevance gating |
| E18 Data Analysis | Standard | Entity-relation-entity triple extraction, dependency structures |
| E19 Pattern Identification | 80 tests | Sliding-window hash fingerprinting, pattern lifecycle (CANDIDATE->CONFIRMED->DECAYING->removed) |
| E20 Pattern Comparison | Standard | Jaccard + cosine + alignment scoring against templates |
| E23 Intention Map | Standard | Intent detection and tracking |

**Conclusions:** Pattern identification (E19) uses a lifecycle model where patterns must survive multiple observations before confirmation. Tests validate the full lifecycle including decay and removal. The hash fingerprinting approach is tested for collision resistance on diverse input patterns.

#### Reasoning & Decision (3 engines)
| Engine | Tests | Key Validations |
|--------|-------|-----------------|
| E13 Simulation Brain | 43 tests | Monte Carlo scenario simulation, outcome probability estimation |
| E15 Decision Making | Standard | Multi-criteria decision framework |
| E21 Strategic Decision | Standard | Multi-step goal planning, commitment tracking, plan revision |

#### Learning Cluster (3 engines)
| Engine | Tests | Key Validations |
|--------|-------|-----------------|
| E17 Reward-Based Learning | Standard | Prediction error: delta = r_actual - r_predicted -> parameter adjustment; DA modulates learning rate |
| E22 Contextual Learning | Standard | Context fingerprinting (topic+emotion+intent hash), context recognition, parameter lookup |
| E25 Recursive Learning | Standard | Meta-learning: monitors E17 effectiveness, plateau/divergence detection, strategy switching |

**Conclusions:** The learning cluster forms a hierarchy: E17 does direct reward-based learning, E22 adds context sensitivity, and E25 monitors both for meta-optimization. Tests verify that E25 correctly detects plateaus in E17's learning curve and triggers strategy switches.

#### Remaining Clusters
| Engine | Cluster | Tests | Key Validations |
|--------|---------|-------|-----------------|
| E7 Simulated Opposition | Dialectic | Standard | Gate/request modes, opposition generation |
| E14 Socratic Reasoning | Dialectic | Standard | Question generation, belief examination |
| E12 Logical Brain | Evaluation | 55 tests | Formal logic evaluation |
| E24 Heuristic Bias | Metacognition | 80 tests | Cognitive bias detection in own reasoning |
| E26 Uncertainty Pattern | Meta Self-Awareness | 75 tests | Confidence calibration, uncertainty tracking |
| E27 Neurochemical Homeostatic | Homeostasis | 55 tests | NT baseline maintenance |
| E28 Emotional Detection | Emotional Processing | 70 tests | Emotion recognition and classification |
| E29 Memory Compression | Homeostasis | 65 tests | Compression policy (VERBATIM/SEMANTIC/SYMBOLIC/PRUNE) |
| E30 Retroactive Alignment | Alignment | 65 tests | Past-decision alignment audit |

#### Coherence Audit (Session 24 + Session 28)
Two full audits verified all 29 engines against the Master Neurochemical Appendix:
- **Pattern A compliance**: All engines accept `Dict[str, float]` for `update_neurochem_state()`
- **Canonical NT keys**: Standardized to `"da"`, `"5ht"`, `"ne"`, `"ach"`, `"oxt"`, `"mor"`, `"cb1"`, `"cortisol"`, `"gaba"`
- **Oscillatory signal naming**: `*_enhancement` -> `*_boost`, `*_suppression` -> `*_suppress`
- **Engine metadata**: All engines report `engine_id` and `cluster` in `get_status()`

---

### 2.3 Reward System (48 files, ~3,700 lines)

**What it validates:** A 4-domain reward evaluation framework that scores system outputs across Logic, Ethics, Human Attunement, and Innovation -- each domain decomposed into 8-11 subscores.

#### Domain Breakdown

| Domain | Files | Subscores Tested |
|--------|-------|------------------|
| **Logic** | 10 | Internal consistency, external consistency, concept fidelity, concept continuity, context fidelity, semantic continuity, epistemic calibration, domain integration, ports & uncertainty |
| **Ethics** | 9 | Harm reduction, fairness, autonomy respect, intent clarity, timeline reflection, failure mode awareness, downstream risk amplification, horizon feasibility, abstention appropriateness |
| **Human Attunement** | 10 | Empathetic inference, cognitive reading, adaptive response framing, intention calibration, truthfulness tradeoff, persuasion risk suppression, benefit/containment success rates, attuned dissonance, short vs. long-term interpersonal benefit |
| **Innovation** | 11 | Conceptual novelty, structural novelty, pattern divergence, exploration drive, risk tolerance, challenge complexity, controlled stochasticity readiness, symbolic recombination, resolution satisfaction, novelty generation |

#### Infrastructure Tests
| Component | File | What's Tested |
|-----------|------|---------------|
| Synthesis engine | `test_synthesis_engine.py` | Domain result merging into RewardMetaDirective |
| Neurochemical adapter | `test_neurochemical_adapter.py` | Directive -> NT signal transformation |
| Feedback modulator | `test_feedback_modulator.py` | 4 feedback pathways (OXT, CB1, NE, GABA_B) |
| Reward profiles | `test_reward_profiles_static_profiles.py` | Profile taxonomy, preset validation |
| Safety bridge | `test_reward_safety_bridge.py` | Safety boundary enforcement |
| Phase 0 structures | `test_reward_phase0_base.py`, `test_reward_phase0_structures.py` | RewardContext, RewardSubscore, RewardDomainResult, RewardWeights |
| Evaluation collectors | `test_reward_evaluation_collectors.py` | Multi-evaluator aggregation |

**Conclusions:** Each subscore is tested independently with controlled state dictionaries, ensuring that evaluator logic is isolated from integration concerns. The synthesis engine test verifies that domain results are correctly weighted and merged. The safety bridge test confirms that ethical violations produce hard stops regardless of other domain scores. Integration tests (`test_logic_domain_integration.py`) verify that adding/removing subscores doesn't silently change the general_score average.

---

### 2.4 Memory Layer (20 files, ~3,200 lines)

**What it validates:** A biologically-inspired 3-tier memory system with consolidation, relevance decay, and specialized logs.

| Tier | File | Key Tests |
|------|------|-----------|
| **STMM** (Short-Term) | `test_stmm.py` | FIFO message buffer (2 user + 2 system), `begin_cycle()` clears analysis while keeping buffer, 10-component analysis storage |
| **MTMM** (Mid-Term) | `test_mtmm.py` | Session memory, TF-IDF cosine search (hand-rolled, no external ML dependency), progressive re-compression, trend analysis |
| **LTMM** (Long-Term) | `test_ltmm.py` | Persistent store, consolidation criteria (emotion >= 0.6, unsolved items, CRITICAL/IDENTITY flags, trust < 0.4), relevance decay (exp(-lambda*hours), half-life 1 week), cold/purge lifecycle |

| Component | File | Key Tests |
|-----------|------|-----------|
| Manager | `test_manager.py` | Full cycle-end flow (STMM->MTMM->LTMM), multi-cycle accumulation |
| Memory Contrast | `test_contrast.py` | MemoryContrastPort for contradiction detection cross-referencing |
| Specialized Logs | `test_specialized_logs.py` | 8 logs: Learning, Sandbox, Paradox, Contradiction, Unsolved, Self-Reflection, Identity, Dream |
| Identity | `test_identity_stores.py`, `test_identity_types.py` | Never-demoted identity entries, journal store |
| Knowledge | `test_knowledge_stores.py`, `test_knowledge_types.py` | Library, KnowledgeMap, question stores |
| Thoughts | `test_thoughts_stores.py`, `test_thoughts_types.py` | Thought packet storage |
| Infrastructure | `test_tags.py`, `test_namespaces.py`, `test_scope_filter.py`, `test_pipeline_scopes.py`, `test_search_utils.py`, `test_retrieval_router.py`, `test_types.py` | Tagging system, namespace isolation, scope-based filtering, retrieval routing |

**Conclusions:** The memory manager integration test is particularly important -- it validates that a `MemoryPacket` created in STMM survives compression into MTMM and eventual consolidation into LTMM with content preserved. The relevance decay test confirms identity-flagged entries never get demoted regardless of age. The LTMM store wiring sweep (Session 36) added tests for 14+ previously orphaned stores.

---

### 2.5 Core Pipelines (6 files, ~2,700 lines)

**What it validates:** End-to-end data flow through the system's processing modes.

| Component | File | Key Tests |
|-----------|------|-----------|
| Learning modes M1-M5 | `test_learning_mode_pipelines.py` | Mode routing, human teaching flow, peer review gates, challenge logic, independent study suppression, question extraction, KnowledgeMap bootstrap |
| Homework pipeline | `test_homework_pipeline.py` | Assignment processing, stage orchestration, identity journal writes |
| Reflective pipeline | `test_reflective_pipeline.py` | Meta-learning mode, PendingUpdateQueue wiring |
| Reflective wiring | `test_reflective_wiring.py` | E32 conclusion_updates -> PendingUpdate submissions |
| Mode profiles | `test_mode_profiles.py` | Reward profile switching per processing mode |
| Emotional landscape | `test_emotional_landscape.py` | Emotion detection integration into processing |

**Conclusions:** Pipeline tests use lightweight mocks (`MockDispatchResult`, `MockPipelineState`) to isolate pipeline logic from engine implementations. This is by design -- engine behaviour is validated in `cog_engines/` and pipeline tests focus on orchestration, routing, and data flow correctness.

---

### 2.6 Bootstrap & Orchestration (4 files, ~1,630 lines)

| Component | File | Key Tests |
|-----------|------|-----------|
| Knowledge bootstrap | `test_knowledge_bootstrap.py` | Seed loading into AtomSpace, KnowledgeMap, Lessons, Library |
| Concept library parser | `test_concept_library_parser.py` | Parsing concept definitions from source files |
| Cycle manager | `test_cycle_manager.py` | Session lifecycle, input routing, mode transitions |

---

## 3. Testing Methodology

### 3.1 Test Categories

The suite contains five categories of tests, applied differently across subsystems:

| Category | Description | Where Used |
|----------|-------------|------------|
| **Unit** | Single class/function in isolation | Everywhere -- the dominant test type |
| **Mathematical** | Numerical correctness of algorithms against closed-form solutions | Neurochem (SDE, Bayesian), Reward (score averaging) |
| **Integration** | Multi-component data flow | Memory (tier transitions), Core (pipeline flows), Engine (impasse delegation) |
| **Regression** | Ensures past bugs stay fixed | Readout (OXTR->OXT), CB1 name collision, 5-HT naming |
| **Edge/Stress** | Boundary conditions, capacity limits | AtomSpace (pruning at capacity), Oscillation (bounds clamping) |

### 3.2 Patterns & Conventions

- **File naming**: `test_<component>.py` -- one file per engine or major component
- **Fixtures**: `@pytest.fixture` for engine/store instantiation; factory helpers (`_pkt()`, `_entry()`, `_wme()`) for complex objects
- **Floating-point**: Always `pytest.approx(expected)` or explicit epsilon, never raw `==`
- **Stochastic tests**: Seeded RNG (`numpy.random.default_rng(seed)`) for reproducibility; statistical assertions use wide margins
- **No external mocking frameworks**: Only `unittest.mock` (stdlib)
- **No external ML dependencies**: TF-IDF search, cosine similarity, and all scoring are hand-rolled

### 3.3 Test Growth Over Time

| Session | Total Tests | Added | Focus |
|---------|-------------|-------|-------|
| 19 | 1,943 | -- | Neurochem + Reward baseline |
| 20 | 2,086 | +143 | Memory layer (4 phases) |
| 21 | 2,572 | +486 | Engine 6 (Logic Trap) + more engines |
| 23 | 3,477 | +905 | Engines 5, 12, 24, 27, 30 |
| 24 | 3,962 | +485 | Engines 26, 28, 13 + coherence audit |
| 25 | 4,078 | +116 | Engine 3 (SOAR) |
| 26 | 4,396 | +318 | CogniTools (E9, E10, E16) |
| 27 | 5,391 | +995 | Final 9 engines -- all 29 complete |
| 31 | 5,657 | +266 | Sleep & Dream modes |
| 32 | 5,837 | +180 | Reward profiles |
| 34 | 5,969 | +132 | Regular input pipeline + LLM integration |
| 36 | 6,015 | +46 | LTMM store wiring sweep |

---

## 4. Key Findings & Bugs Caught

### 4.1 Critical Bugs Found by Tests

| Bug | Session | Impact | How Tests Caught It |
|-----|---------|--------|---------------------|
| **SDE infinite loop** | 29 | Engine hangs | Step-count assertion in euler_maruyama tests |
| **OXTR->OXT receptor resolution** | 19 | Wrong NT readout for oxytocin receptor | Readout test comparing parent_nt inference |
| **CB1 name collision** | 19 | Registry config overwrite (NT and receptor share ID) | Registration test detecting missing config keys |
| **5-HT naming inconsistency** | 24 | `sht_level` vs `_5ht_level` vs `sht1a_level` across engines | Coherence audit tests checking canonical keys |
| **Noise splitting bug** | 29 | Incorrect phasic/tonic noise separation | Emotion splitter tests comparing 4M/4R outputs |
| **RNG divergence** | 29 | Non-reproducible stochastic tests | Seeding tests verifying identical sequences from same seed |
| **Memory mutation** | 29 | Shared mutable state between cycles | Contrast test detecting unexpected state changes |
| **emotion_drive ignored** | 29 | Emotion input had no effect on NT dynamics | Emotion interface test asserting non-zero delta after emotional input |

### 4.2 Architectural Invariants Enforced by Tests

1. **All NT concentrations in [0, 1]**: Range assertions after every step in engine tests
2. **Identity memory never demoted**: LTMM relevance tests verify identity-flagged entries survive decay
3. **Engine interface compliance**: Every engine test verifies `update_neurochem_state`, `process`, `get_status`
4. **Canonical NT keys**: Coherence audit tests reject non-standard key names
5. **Checkpoint round-trip**: Serialization tests verify save->load produces identical state
6. **Consolidation criteria**: Memory tests verify only qualifying packets get promoted to LTMM

---

## 5. Observations & Recommendations

### 5.1 Strengths

- **No external ML dependencies**: All mathematical operations (cosine similarity, TF-IDF, SDE solving) are hand-rolled, making the test suite fast and self-contained
- **Seeded stochastic tests**: NumPy PCG64 RNG with explicit seeds ensures reproducibility
- **Coherence audits**: Two systematic audits (Sessions 24, 28) caught 9+ critical interface mismatches that unit tests alone would have missed
- **Zero regressions**: All 36 sessions maintained 0 regression count

### 5.2 Areas to Watch

- **Flaky paradox timing test**: The single known flaky test in E2 should be investigated -- likely needs a tolerance margin or deterministic trigger instead of wall-clock timing
- **Integration test depth**: Pipeline tests (core/) use mocks extensively, which is correct for isolation but means true end-to-end flows (user input -> neurochemical response -> memory storage -> learning update) are not fully tested as integrated sequences
- **Reward subscore independence**: Each subscore is tested in isolation, but cross-domain interactions (e.g., ethics overriding innovation) are only tested at the synthesis engine level

### 5.3 Test Suite Health

| Metric | Value | Assessment |
|--------|-------|------------|
| Total tests | 6,015 | Substantial |
| Test-to-source ratio | 158 test files / 230 source files | 0.69 -- good coverage density |
| Lines test / lines source | ~56k / ~source | Well-proportioned |
| Flaky tests | 1 | Excellent stability |
| External dependencies | pytest, numpy | Minimal |
| Avg. growth per session | ~170 tests | Consistent |

---

## 6. How to Read the Test Suite

If you're new to the codebase, read tests in this order:

1. **`test_neurochemical_engine.py`** -- Understand the engine lifecycle (init -> register -> step -> read)
2. **`test_euler_maruyama.py`** -- See how the SDE math works
3. **`test_reward_phase0_base.py`** -- Understand reward data structures
4. **`test_stmm.py`** -> **`test_mtmm.py`** -> **`test_ltmm.py`** -- Follow data through memory tiers
5. **`test_contradiction_detection_engine.py`** -- See a typical cognitive engine test
6. **`test_learning_mode_pipelines.py`** -- See how pipelines orchestrate engines

Each test file is self-contained -- you can read any file independently to understand the component it tests.
