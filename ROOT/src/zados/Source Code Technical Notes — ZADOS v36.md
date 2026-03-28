# ZADOS Source Code — Technical Notes

> Design choices, patterns, trade-offs, and lessons learned across 36 development sessions.

---

## 1. Why This Architecture

ZADOS is not a neural network. It's a **symbolic-stochastic hybrid** — 29 deterministic cognitive engines modulated by a stochastic neurochemical substrate. The "intelligence" comes from the dynamic interaction between engines, not from trained weights.

This has direct consequences for the codebase:

- **No training loop, no gradient descent.** The neurochemical layer uses SDEs (stochastic differential equations) to model neurotransmitter dynamics — these are physics equations, not loss functions.
- **No external ML framework.** All math is hand-rolled with NumPy. TF-IDF cosine similarity for memory search, Bayesian inference for contradiction detection, Euler-Maruyama for SDE integration.
- **Fully inspectable.** Every decision the system makes can be traced through engine outputs, NT concentrations, reward scores, and memory states. There are no opaque embedding layers.

---

## 2. Separation Principles

### 2.1 Pure Functions vs. Stateful Engines

The strictest rule in the codebase: **kinetics are pure functions, engines are stateful orchestrators.**

```
neurochem/kinetics/mass_balance.py     ← pure functions
neurochem/core/engine.py               ← stateful orchestrator
```

The engine calls pure functions, collects their results, and updates state. The pure functions have no access to state — they receive values and return values.

**Trade-off**: This means the engine has to pass a lot of parameters around (C_tonic, C_phasic, F, eta_u, configs...). We accepted the verbosity because:
- Pure functions have zero test setup cost (no fixtures, no mocks)
- State bugs are localized to the engine, never in kinetics
- Functions can be composed freely (e.g., `compute_drift_term` is used by both the online engine and the batch simulator)

### 2.2 Frozen vs. Mutable Dataclasses

The codebase uses both, with a clear rule:

| Mutable (`@dataclass`) | Frozen (`@dataclass(frozen=True)`) |
|---|---|
| State that evolves over time | Configuration, evaluation results, directives |
| `NeurotransmitterState`, `STMMStore` | `ContradictionEngineConfig`, `RewardMetaDirective`, `RewardSubscore` |
| Modified by engines during processing | Created once, passed through pipeline, never changed |

**Why not all frozen?** State containers like `NeurotransmitterState` are updated thousands of times per simulation step. Creating a new frozen instance each time would be wasteful and obscure the mutation semantics.

**Why not all mutable?** Configurations and results flowing through the pipeline should never be accidentally modified. Freezing them turns accidental mutation into an immediate `FrozenInstanceError`.

### 2.3 Composition over Inheritance

The codebase uses inheritance in exactly two places:
1. `NeurotransmitterModule` ABC → 12 concrete NT modules (DA, 5-HT, NE, ...)
2. `ReceptorFamilyModule` ABC → concrete receptor family modules

Everything else is composition. The `MemoryLayer` facade composes STMM + MTMM + LTMM + managers. The `SessionOrchestrator` composes `AnswerPipeline` + `MemoryLayer` + engines. The `SynthesisEngine` composes pure functions from `directives.py`.

**Why**: Deep inheritance hierarchies make it hard to understand what code runs when. With composition, you can read `MemoryLayer.__init__` and see exactly what gets wired to what.

---

## 3. The Engine Interface Problem (and Solution)

### The Problem

By Session 21, we had 11 engines with inconsistent interfaces:
- Some accepted `**kwargs` for NT levels
- Some used positional arguments (`ne_level, da_level, ...`)
- Some used `_level` suffixes, some didn't
- 5-HT was variously `sht_level`, `_5ht_level`, `sht1a_level`, or via `"5ht"` key

### The Solution (Sessions 24 + 28)

Two coherence audits established **Pattern A**:

```python
def update_neurochem_state(self, nt_levels: Dict[str, float]) -> None:
```

One dict. Canonical keys from `constants.py`. No `_level` suffixes in keys. No `**kwargs`.

We created `cognitive_engines/constants.py` as the single source of truth:

```python
NT_KEYS = frozenset({"glu", "gaba", "da", "5ht", "ne", "ach", "oxt", "mor", "cb1", ...})
```

And added tests that verify every engine only uses keys from `NT_KEYS`. If someone adds a new engine with `sht_level`, the tests catch it.

**Lesson**: Interface consistency doesn't happen organically when you're building 29 engines over months. You need a canonical constants file and automated enforcement.

---

## 4. The Neurochemical Pipeline (Technical Detail)

### 4.1 SDE Integration

Each neurotransmitter follows:

```
dC(t) = μ(C, t) dt + σ(C, t) dW(t)
```

Where:
- `μ` (drift) = release drives − reuptake − degradation − clearance + mean reversion
- `σ` (diffusion) = noise amplitude, modulated by oscillatory bands
- `dW(t)` = Wiener process increment

The solver (`euler_maruyama_step_bounded`) uses bounded integration with reflection — if a step would push C below 0 or above a cap, the value is reflected back.

**Why Euler-Maruyama (not Milstein or RK)?** EM is first-order, but:
- Our noise is additive (σ doesn't depend on C in practice), so EM and Milstein converge at the same rate
- EM is simpler to implement correctly with bounded reflection
- The dt is small enough (0.01) that higher-order methods don't improve accuracy meaningfully

### 4.2 Concentration Decomposition

Every NT has two concentration components:

```
C(t) = C_tonic(t) + C_phasic(t)
```

- **C_tonic**: Baseline level, evolves slowly, mean-reverts to `C_baseline`
- **C_phasic**: Burst component, decays rapidly after phasic release events

This decomposition lets the system distinguish between "dopamine is generally elevated" (high C_tonic) and "dopamine just spiked" (high C_phasic). The emotion splitter uses this split for its 4M (modulatory/tonic) and 4R (reactive/phasic) pathways.

**Implementation note**: `NeurotransmitterState.C` is a property that returns `C_tonic + C_phasic`. This can exceed 1.0 because each component is individually bounded to [0, 1]. This is intentional — the total concentration can temporarily overshoot during simultaneous tonic elevation and phasic burst.

### 4.3 Oscillatory Modulation

Five oscillatory bands (delta through gamma, plus sigma for sleep) modulate two things:

1. **Noise amplitude** (`modulate_noise`): Some bands amplify stochastic noise, others suppress it
2. **Receptor binding** (`modulate_K_d`): Bands shift the dissociation constant of receptors

Each NT has a set of `OscillationCouplingRule`s that define which bands affect which kinetic parameters. For example, DA's rules might say "gamma band multiplicatively modulates release by +0.3" — meaning when gamma is high, DA release is boosted.

### 4.4 The Extractor Pipeline

The extractor pipeline bridges reward evaluation → neurochemical modulation:

```
Domain results → Evaluation vector (8 axes)
                          ↓
                  Reactivity matrix (20 entries)
                          ↓
                  NT modulation signals
                          ↓
                  engine.step(signals)
```

The reactivity matrix maps evaluation axes to NT changes through threshold-gated stochastic distributions. For example, high `logical_conflict` might trigger a NE burst via a Poisson-distributed impulse — but only if the conflict exceeds a threshold.

**Why stochastic?** Deterministic mappings (high conflict → always NE spike) produce robotic responses. Adding Poisson/Gamma/Lognormal noise creates natural variability — sometimes high conflict triggers a big NE response, sometimes a small one, occasionally none at all.

---

## 5. Memory Architecture Decisions

### 5.1 Three Tiers (Not Two, Not Four)

STMM (current cycle) → MTMM (session) → LTMM (persistent)

We considered a two-tier model (short + long) but session-scoped memory is genuinely different from both:
- Short-term is FIFO (latest 2+2 messages). It's a buffer, not searchable.
- Session memory accumulates all interactions within a conversation. It's searchable by TF-IDF. But it gets cleared between sessions.
- Long-term stores information across sessions with relevance decay.

### 5.2 Hand-Rolled TF-IDF (Not a Vector DB)

Memory search uses TF-IDF cosine similarity implemented from scratch in `context_processor.py`. No FAISS, no ChromaDB, no embeddings.

**Why**:
- ZADOS already has semantic engines (E8 Relevance Scoring, E18 Data Analysis) that do deep semantic analysis. Memory search doesn't need to duplicate that.
- TF-IDF is fast, deterministic, and dependency-free
- The memory layer's job is *storage and retrieval*, not *understanding*. Understanding happens in the cognitive engines.

### 5.3 Specialized Logs as First-Class Citizens

The 8 specialized logs (Learning, Sandbox, Paradox, Contradiction, Unsolved, Self-Reflection, Identity, Dream) are not just filters on a general store — they're separate data structures with their own semantics.

**Why**: An `IdentityMemoryLog` entry must never be demoted by relevance decay. An `UnsolvedConceptsBuffer` entry has a stagnation counter that ticks each session. A `DreamLog` entry records creative exploration episodes with their own lifecycle. These semantics can't be expressed as filters on a generic store.

### 5.4 The LTMM Store Wiring Problem

By Session 35, we had 14+ LTMM stores... and some had no writers. The `OverviewLogStore` had a `write_session_overview()` method, but nothing called it. The `GeneralQuestionStore` existed but no pipeline ever wrote to it.

Session 36 was a dedicated wiring sweep that connected every orphaned store to its proper pipeline. The key insight was adding `close_session()` to `SessionOrchestrator` — many stores should be written to at session end, not during turn processing.

**Lesson**: Stores are easy to design. Wiring them into the pipeline at the right lifecycle point is the hard part.

---

## 6. Reward System Design

### 6.1 Four Domains, Not One Score

The reward system produces four independent domain scores:

| Domain | What It Measures |
|---|---|
| **Logic** | Internal consistency, concept fidelity, epistemic calibration |
| **Ethics** | Harm reduction, fairness, autonomy respect, failure awareness |
| **Human Attunement** | Empathy, cognitive reading, intention calibration |
| **Innovation** | Novelty, divergence, exploration, symbolic recombination |

These are combined by the `SynthesisEngine` using weighted composition with a `RewardProfile`.

**Why four?** A single reward score creates Goodhart's Law problems — optimize for one thing and everything else degrades. Four domains with configurable weights let the system shift focus (e.g., `REFLECTIVE_PROFILE` emphasizes logic and ethics; `CREATIVE_PROFILE` emphasizes innovation).

### 6.2 Frozen Outputs, Stateless Evaluation

Every evaluator produces a `RewardSubscore(frozen=True)`. Every domain produces a `RewardDomainResult(frozen=True)`. The synthesis engine produces a `RewardMetaDirective(frozen=True)`.

The evaluators themselves are stateless — they receive a context dict and return a score. No evaluator remembers previous evaluations.

**Why stateless?** Statefulness in evaluators creates ordering dependencies (evaluator A's result depends on whether evaluator B ran first). Stateless evaluators can run in any order, be parallelized, or be skipped without side effects.

### 6.3 Safety Bridge

The safety bridge (`reward/safety/`) enforces hard stops on ethical violations. It runs *after* domain evaluation but *before* the synthesis engine can produce a permissive directive.

```
Domain results → Safety bridge check → SynthesisEngine → RewardMetaDirective
                     ↓ (if violation)
                 Hard stop (suppress=True, allow_output=False)
```

**Why separate from ethics domain?** The ethics domain produces a *score*. The safety bridge produces a *binary decision*. A low ethics score means "this response could be better ethically." A safety bridge trigger means "this response must not be generated."

---

## 7. Cognitive Engine Patterns

### 7.1 Engines Detect, Don't Resolve

Detection engines (E1 Contradiction, E2 Paradox, E4 Fallacy, E5 Bias, E6 Logic Trap) follow a strict rule: **detect, flag, and log — do NOT resolve or act.**

Resolution happens elsewhere:
- E3 (SOAR) resolves impasses by delegating to other engines
- E14 (Socratic) resolves by questioning
- E21 (Strategic Decision) resolves by planning

**Why**: Single-responsibility. A contradiction detector that also tries to resolve contradictions becomes untestable — you can't tell if it failed to detect or failed to resolve.

### 7.2 SOAR as Executive Backbone

Engine 3 (SOAR Production Rule Engine) implements the SOAR cognitive architecture's 5-phase decision cycle:

```
Input → Elaboration → Proposal → Decision → Application
```

When it hits an impasse (can't decide), it delegates to specific engines:
- TIE → E13 (Simulation Brain): run scenarios to break the tie
- CONFLICT → E1 + E14 (Contradiction + Socratic): examine the conflict
- NO_CHANGE → E7 (Simulated Opposition): generate alternative perspectives
- STATE_NO_CHANGE → E26 (Uncertainty Pattern): analyze why progress stalled

**Why SOAR?** SOAR provides a principled decision-making framework with well-studied impasse handling. Rather than building ad-hoc routing logic, impasses are classified into known types with known resolution strategies.

### 7.3 CogniTools (OpenCog Replacements)

Engines 9 (AtomSpace), 10 (PLN), and 16 (ECAN) are Python replacements for OpenCog Hyperon components:

- **AtomSpace-Lite**: Typed hypergraph with 15 AtomTypes, TruthValues, AttentionValues, O(1) indexes
- **PLN Core**: 12 inference rules as pure functions, backward chaining
- **ECAN Core**: Attention economy with rent/wage/spread/clamp dynamics

These live in `cognitools/` (not `py_engines/`) to distinguish them as development-time knowledge infrastructure, not runtime processing engines.

**Why replace OpenCog?** OpenCog Hyperon is a C++ framework with Python bindings. It adds significant dependency weight, build complexity, and debugging difficulty. Our AtomSpace-Lite handles the same operations for our scale with pure Python and zero external dependencies.

### 7.4 Learning Cluster Hierarchy

Three engines form a learning hierarchy:

```
E25 Recursive Learning (meta-learning)
  ↓ monitors
E17 Reward-Based Learning (prediction error)
  ↓ contextualized by
E22 Contextual Learning (context fingerprinting)
```

E17 does basic prediction-error learning (δ = actual − predicted → adjust parameters). E22 adds context sensitivity (different parameters for different contexts, identified by topic+emotion+intent fingerprints). E25 monitors E17's effectiveness and switches strategies when it detects plateaus or divergence.

---

## 8. Cross-Cutting Patterns

### 8.1 The `_clamp` Utility

Every cognitive engine uses `_clamp(value, lo, hi)` from `constants.py` for bounds enforcement:

```python
def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
```

This tiny function appears hundreds of times across the codebase. It's the primary defense against out-of-range values propagating through the system.

### 8.2 Leaky Integrators Everywhere

The leaky integrator pattern appears in multiple subsystems:

- **Urgency forecast**: 5 leaky integrators for urgency axes
- **Emotion tracker**: 12 leaky integrators (one per emotion)
- **Regulatory modulator**: 4 leaky integrators for feedback pathways
- **Contradiction load**: Leaky integral of κ_contradiction in E1

All share the same `LeakyIntegrator` class from `extractors/leaky_integrator.py`:

```
x(t+dt) = x(t) · exp(-dt/τ) + input · gain
```

**Why leaky integrators?** They provide temporal smoothing with configurable memory (τ = time constant). High τ means slow response (stable but laggy). Low τ means fast response (reactive but noisy). Different subsystems need different τ values, but the math is identical.

### 8.3 Spec Cross-References

Comments throughout the codebase reference the design specifications:

```python
# (spec §2.2, steps B.1-B.9)
# From Appendix F: C(t) = C_tonic(t) + C_phasic(t)
# (from Master Appendix sec. 7)
```

The specs live in `specs_docs/`:
- Neurochemical Layer — Consolidated Spec v2.0
- Cognitive Engines — Technical Specification v1.0
- Memory Layer — Technical Specification v2.0
- Reward System — Technical Specification v1.0
- ZA-DOS LLM Layer Spec v0.5

### 8.4 Error Handling Philosophy

The codebase prefers **fail-fast at construction, fail-safe at runtime**:

- **Construction**: Frozen dataclasses with `__post_init__` validation raise `ValueError` immediately on invalid configs
- **Runtime**: `_clamp()` silently bounds values. Missing dict keys return defaults via `.get()`. Unknown engine IDs are skipped in dispatch.

**Why?** A bad config should crash immediately — before any simulation runs. But during runtime, a slightly out-of-range NT concentration shouldn't crash the system. Clamping it to [0, 1] and continuing is safer than raising an exception mid-conversation.

---

## 9. Performance Considerations

### 9.1 NumPy PCG64 RNG

All stochastic operations use `numpy.random.default_rng(seed)` with the PCG64 generator. This provides:
- Reproducibility under seeding
- Better statistical properties than Mersenne Twister
- Independent streams via `SeedSequence` for parallel engines

### 9.2 Hash-Indexed Data Structures

Performance-sensitive stores use hash-indexed lookups:
- AtomSpace (E9): O(1) lookup by id, type, name, or outgoing set
- SOAR Working Memory (E3): O(1) triple lookup by identifier, attribute, or (identifier, attribute)
- Memory search: TF-IDF vectors are precomputed at write time, search is dot-product

### 9.3 SparseUpdateScheduler

Not every NT needs updating every step. The `SparseUpdateScheduler` skips NTs whose concentration hasn't changed significantly, reducing computation in the common case where most NTs are near baseline.

---

## 10. Known Technical Debt

1. **`Any` type hints in core/**: `SessionOrchestrator.__init__` uses `Any` for most parameters. This was pragmatic during rapid development but should be tightened to concrete types.

2. **Legacy directories**: `SOAR_engines/` and `hyperon_engines/` in `cognitive_engines/` are reserved/empty — holdovers from early design that considered external engine integrations.

3. **Single flaky test**: E2 (Paradox Detection) has a timing-sensitive test that fails under load. Needs deterministic triggering instead of wall-clock timing.

4. **Phase 5 evaluator bridge**: `LLM_interpretation/phase5_evaluator.py` bridges to an external LLM. This is the only component with an external runtime dependency beyond NumPy.
