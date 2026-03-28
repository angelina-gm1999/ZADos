# Cognitive Engines Layer — Technical Specification

**Project:** ZADOS  
**Version:** 1.0  
**Date:** March 23, 2026  
**Status:** Code-Verified (6015 tests, 32 engines)

---

## 1. Consolidated Summary

The Cognitive Engines Layer contains 32 engines organized into 13 functional clusters. All engines follow Pattern A (canonical interface): `update_neurochem_state(Dict[str, float])`, `process(input) → output`, `get_status() → Dict`. Engines are divided into two packages: `py_engines/` (29 runtime engines) and `cognitools/` (3 knowledge substrate engines).

| Aspect | Value |
|--------|-------|
| Source package | `src/zados/cognitive_engines/` |
| Total engines | 32 (29 py_engines + 3 cognitools) |
| Functional clusters | 13 |
| Canonical interface | Pattern A: `update_neurochem_state` / `process` / `get_status` |
| NT keys | 12 canonical: `glu`, `gaba`, `da`, `5ht`, `ne`, `ach`, `oxt`, `mor`, `cb1`, `crh`, `cor`, `histamine` |
| Oscillatory bands | 6: `delta`, `theta`, `alpha`, `beta`, `gamma`, `sigma` |
| Cross-frequency couplings | `theta_gamma`, `alpha_beta`, `delta_sigma` |
| Config system | Per-engine frozen `@dataclass` with mode overrides |
| State management | Mutable `@dataclass` per engine |
| No inheritance | Duck-typed Pattern A; no ABC or Protocol superclass |
| Constants module | `constants.py`: `ENGINE_IDS`, `ENGINE_CLUSTER_MAP`, `NT_KEYS`, `_clamp()` |
| Journal events | `journal_stub.py`: `emit(engine_id, event_type, data)` |

---

## 2. Architecture Overview

### 2.1 Pattern A — Canonical Engine Interface

```python
class EngineClass:
    engine_id: str = "engine_short_name"  # class attribute
    cluster: str = "cluster_name"          # class attribute

    def __init__(self, config=None, rng_seed=None):
        ...

    def update_neurochem_state(self, state_dict: Dict[str, float]) -> None:
        """Inject external NT levels. Keys from NT_KEYS (lowercase)."""

    def process(self, input_data: <InputType>) -> <OutputType>:
        """Execute engine core logic. Types vary per engine."""

    def get_status(self) -> Dict[str, Any]:
        """Introspection. Must include 'engine_id' and 'cluster'."""
```

**Key characteristics:**

- All state via `@dataclass` (frozen for config, mutable for working state)
- No abstract base class — duck-typed to Pattern A
- Neurochemical state injected externally via `Dict[str, float]`
- Engines emit oscillatory signals in their neurochem outputs
- No dependencies on orchestrator internals

### 2.2 Cluster Taxonomy

| Cluster | Engine IDs | Role |
|---------|-----------|------|
| Detection | E1, E2, E4, E5, E6 | Anomaly and error finding: contradictions, paradoxes, fallacies, biases, logic traps |
| Dialectic | E7, E14 | Generate opposing views and Socratic questioning |
| Executive Control | E3 | SOAR 5-phase decision cycle with chunking and impasse delegation |
| Knowledge Substrate | E9, E10, E16 | AtomSpace hypergraph, PLN inference, ECAN attention economy (OpenCog replacements) |
| Pattern Analysis | E8, E11, E18, E19, E20, E23 | Extract, score, identify, compare patterns and map intentions |
| Evaluation | E12 | Comprehensive logical coherence assessment |
| Reasoning | E13, E15, E21 | Simulation, decision-making, and strategic planning |
| Metacognition | E24, E31, E32 | Monitor own reasoning, learning effectiveness, and identity coherence |
| Meta-Self-Awareness | E26 | Uncertainty pattern detection and propagation analysis |
| Homeostasis | E27, E29 | Neurochemical balance monitoring and memory compression |
| Emotional Processing | E28 | Emotion detection, classification, and tone calibration |
| Alignment | E30 | Retroactive alignment to goals and temporal coherence |
| Learning | E17, E22, E25 | Reward-based, contextual, and recursive meta-learning |

### 2.3 py_engines vs cognitools

| Package | Purpose | Engines | Count |
|---------|---------|---------|-------|
| `py_engines/` | Runtime AI pipeline engines | E1–E8, E11–E15, E17–E32 | 29 |
| `cognitools/` | Dev-time cognitive toolkit (OpenCog replacements) | E9, E10, E16 | 3 |

---

## 3. Constants & Shared Infrastructure

### 3.1 Canonical NT Keys

| Key | Full Name | State Field |
|-----|-----------|------------|
| `glu` | Glutamate | `glu_level` |
| `gaba` | GABA | `gaba_level` |
| `da` | Dopamine | `da_level` |
| `5ht` | 5-HT (Serotonin) | `_5ht_level` |
| `ne` | Norepinephrine | `ne_level` |
| `ach` | Acetylcholine | `ach_level` |
| `oxt` | Oxytocin | `oxt_level` |
| `mor` | mu-Opioid | `mor_level` |
| `cb1` | CB1 (Endocannabinoid) | `cb1_level` |
| `crh` | CRH | `crh_level` |
| `cor` | Cortisol | `cor_level` |
| `histamine` | Histamine | `histamine_level` |

### 3.2 NT-Band Associations

| NT | Associated Bands |
|----|----------------|
| `da` | gamma, theta |
| `5ht` | theta, alpha |
| `glu` | gamma, theta_gamma |
| `gaba` | alpha, delta |
| `ach` | beta |
| `ne` | beta |
| `oxt` | theta |
| `cb1` | delta, alpha_beta |
| `mor` | delta |
| `crh` | beta |
| `cor` | beta |
| `histamine` | beta, gamma |

### 3.3 Utility Functions

- `_clamp(v, lo=0.0, hi=1.0)` — Canonical bounded clamp, shared across all engines.
- `normalize_nt_key(key, target='lower')` — Map any variant to canonical form (e.g., `'5-HT'` → `'5ht'`).

### 3.4 Journal Event Stub

Singleton `JournalEventStub` for decoupled event logging. Engines call `emit(engine_id, event_type, data)`. Orchestrator wires callback via `register(callback)`. Event types: `innovation_flag`, `pattern_detected`, `socratic_question`, `identity_observation`.

```python
# Engine-side usage:
journal_event_stub.emit(engine_id="E19", event_type="innovation_flag",
                        data={"pattern": "recursive_analogy", "confidence": 0.82})

# Orchestrator wiring:
journal_event_stub.register(lambda eid, evt, data: route_to_journal(eid, evt, data))
```

---

## 4. Complete Engine Map

| ID | Engine Name | Class | Cluster | File |
|----|------------|-------|---------|------|
| 1 | Contradiction Detection | `ContradictionDetectionEngine` | detection | `py_engines/contradiction_detection_engine.py` |
| 2 | Paradox Detection | `ParadoxDetectionEngine` | detection | `py_engines/paradox_detection_engine.py` |
| 4 | Fallacy Detection | `FallacyDetectionEngine` | detection | `py_engines/fallacy_detection_engine.py` |
| 5 | Bias Detection | `BiasDetectionEngine` | detection | `py_engines/bias_detection_engine.py` |
| 6 | Logic Trap Detection | `LogicTrapDetectionEngine` | detection | `py_engines/logic_trap_detection_engine.py` |
| 7 | Simulated Opposition | `SimulatedOppositionEngine` | dialectic | `py_engines/simulated_opposition_engine.py` |
| 14 | Socratic Reasoning | `SocraticReasoningEngine` | dialectic | `py_engines/socratic_reasoning_engine.py` |
| 3 | SOAR Production | `SOARProductionEngine` | executive_control | `py_engines/soar_production_engine.py` |
| 9 | AtomSpace-Lite | `AtomSpaceEngine` | knowledge_substrate | `cognitools/atomspace_engine.py` |
| 10 | PLN Core | `PLNEngine` | knowledge_substrate | `cognitools/pln_engine.py` |
| 16 | ECAN Core | `ECANEngine` | knowledge_substrate | `cognitools/ecan_engine.py` |
| 8 | Relevance Scoring | `RelevanceScoringEngine` | pattern_analysis | `py_engines/relevance_scoring_engine.py` |
| 11 | Input Relevance Evaluation | `InputRelevanceEvaluationEngine` | pattern_analysis | `py_engines/input_relevance_evaluation_engine.py` |
| 18 | Data Analysis | `DataAnalysisEngine` | pattern_analysis | `py_engines/data_analysis_engine.py` |
| 19 | Pattern Identification | `PatternIdentificationEngine` | pattern_analysis | `py_engines/pattern_identification_engine.py` |
| 20 | Pattern Comparison | `PatternComparisonEngine` | pattern_analysis | `py_engines/pattern_comparison_engine.py` |
| 23 | Intention Map | `IntentionMapEngine` | pattern_analysis | `py_engines/intention_map_engine.py` |
| 12 | Logical Brain | `LogicalBrainEngine` | evaluation | `py_engines/logical_brain_engine.py` |
| 13 | Simulation Brain | `SimulationBrainEngine` | reasoning | `py_engines/simulation_brain_engine.py` |
| 15 | Decision Making | `DecisionMakingEngine` | reasoning | `py_engines/decision_making_engine.py` |
| 21 | Strategic Decision | `StrategicDecisionEngine` | reasoning | `py_engines/strategic_decision_engine.py` |
| 24 | Heuristic Bias | `HeuristicBiasEngine` | metacognition | `py_engines/heuristic_bias_engine.py` |
| 31 | Reflective Learning | `ReflectiveLearningEngine` | metacognition | `py_engines/reflective_learning_engine.py` |
| 32 | Reflective Identity | `ReflectiveIdentityEngine` | metacognition | `py_engines/reflective_identity_engine.py` |
| 26 | Uncertainty Pattern | `UncertaintyPatternEngine` | meta_self_awareness | `py_engines/uncertainty_pattern_engine.py` |
| 27 | Neurochemical Homeostatic | `NeurochemicalHomeostaticEngine` | homeostasis | `py_engines/neurochemical_homeostatic_engine.py` |
| 29 | Memory Compression | `MemoryCompressionEngine` | homeostasis | `py_engines/memory_compression_engine.py` |
| 28 | Emotional Detection | `EmotionalDetectionEngine` | emotional_processing | `py_engines/emotional_detection_engine.py` |
| 30 | Retroactive Alignment | `RetroactiveAlignmentEngine` | alignment | `py_engines/retroactive_alignment_engine.py` |
| 17 | Reward-Based Learning | `RewardBasedLearningEngine` | learning | `py_engines/reward_based_learning_engine.py` |
| 22 | Contextual Learning | `ContextualLearningEngine` | learning | `py_engines/contextual_learning_engine.py` |
| 25 | Recursive Learning | `RecursiveLearningEngine` | learning | `py_engines/recursive_learning_engine.py` |

---

## 5. Detection Cluster

*Anomaly and error finding: contradictions, paradoxes, fallacies, biases, logic traps* — Engines: E1, E2, E4, E5, E6

### E1 — Contradiction Detection

**Class:** `ContradictionDetectionEngine` | **File:** `py_engines/contradiction_detection_engine.py` | **Cluster:** detection

3-level contradiction detection (lexical negation, semantic opposition, contextual incompatibility) with Bayesian posterior fusion, temporal decay, leaky-integral load tracking, and affective integration.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | NE (threshold -0.10), GABA (threshold +0.10), DA D2 (prior ×1.5) |
| NT Write (output) | DA D2 burst (Gamma α=2.0 θ=0.5), 5-HT2C k_on multiplier, GABA reuptake suppression, β-suppress, θ-boost |
| Key Thresholds | `θ_normal=0.50`, `θ_dev=0.30`, `θ_learning=0.40`, `θ_rem_dream=0.70` |

**Algorithm:** Phase 1: Threshold resolution (NE/GABA bidirectional). Phase 2: Prior computation (DA D2 modulated). Phase 3: Pairwise comparison → Level 1 negation signal → Level 2 semantic opposition → Level 3 contextual incompatibility → Bayesian fusion → Threshold gating. Phase 4: κ composition (λ₁=0.40 semantic + λ₂=0.35 logical + λ₃=0.25 affective). Phase 5: Load update `C(t)` leaky integral (τ_c=10.0s). Phase 6: Neurochemical signal emission.

**Unique features:**

- 3-level detection hierarchy (negation → semantic → contextual)
- Leaky-integral contradiction load `C(t)` with exponential decay
- Affective integral for 5-HT2C coupling via temporal convolution (τ_aff=8.0s)
- Bayesian product-of-likelihood-ratios fusion
- `ComparisonSet` input with `memory_contrast_hit_ratio` adjusting prior
- 28 config fields, 6 state fields

---

### E2 — Paradox Detection

**Class:** `ParadoxDetectionEngine` | **File:** `py_engines/paradox_detection_engine.py` | **Cluster:** detection

4-class paradox classification (Resolvable/Apparent/Genuine/Structural) with Beta-distributed Bayesian classifier, 2-path detection (contradiction-derived + independent), dialectical productivity scoring, and Unsolved Concepts Buffer integration.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | 5-HT2A (threshold -0.05), CB1 (threshold -0.05), DA tonic (+0.08), NE (low → +0.04) |
| NT Write (output) | 5-HT2A k_on (λ=0.25), CB1 baseline drift, DA tonic/phasic (curiosity + resolution burst Gamma α=3.0), θ-boost, γ-suppress |
| Key Thresholds | `θ_normal=0.50`, `θ_dev=0.30`, `θ_learning=0.40`, `θ_rem_dream=0.25` |

**Algorithm:** Path 1 — Contradiction-Derived: Extract features (resolvability, self-reference, abstraction divergence, dialectical productivity) → Beta-distributed Bayesian classifier over 4 classes → Posterior argmax with threshold gating. Path 2 — Independent: Oxymoron scan (semantic opposition > θ_oxy=0.70) + concept-level paradox (π > θ_conceptual=0.60). Load decomposition: Π_G/Π_A/Π_S/Π_R weighted. State updates: 5-HT2A integral + CB1 drift.

**Unique features:**

- Beta-distributed Bayesian classification (4 classes × 4 features)
- Unsolved paradox buffer with motivational salience model `M(t) = m₀ + α_attempt × attempts + λ_age × age`
- Two independent detection paths (contradiction-derived + independent)
- Temporal convolution for 5-HT2A coupling
- CB1 baseline drift for structural stability + identity protection
- Resolution proximity scoring for REM scheduling
- 48 config fields, 6 state fields

---

### E4 — Fallacy Detection

**Class:** `FallacyDetectionEngine` | **File:** `py_engines/fallacy_detection_engine.py` | **Cluster:** detection

38 fallacy types across 5 categories (formal, relevance, presumption, ambiguity, inductive). Charity mechanism, self-audit mode with 2× penalty, manipulation indicator for Logic Trap cross-feed.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | ACh (threshold -0.08), NE (-0.06), GABA (low → -0.06), DA (+0.08) |
| NT Write (output) | ACh burst (Gamma α=2.0 θ=0.4), NE burst (Poisson λ=1.5 if Φ>0.40), Glu complexity, β-boost, R_Logic penalty (2× self-audit) |
| Key Thresholds | `θ_normal=0.55`, `θ_dev=0.30`, `θ_internal_audit=0.40`, `θ_rem_dream=0.75` |

**Algorithm:** Phase 1: Threshold (ACh/NE/GABA/DA bidirectional). Phase 2: Argument extraction (user propositions + semantic expansion + system chains). Phase 3: Multi-category detection (13 formal + 25 informal across 5 categories). Phase 4: Charity suppression (plausibility > θ_charity=0.60). Phase 5: Manipulation indicator `M = Σ w_mᵢ × feature_i`. Phase 6: Fallacy load `Φ(t) = Σ w_cat × confidence × (1-charity)`. Phase 7: Neurochem signals.

**Unique features:**

- 38 distinct fallacy types (13 formal + 25 informal)
- Charity mechanism suppressing borderline fallacies with plausible implicit premises
- Self-audit path (2× R_Logic penalty on own reasoning chains)
- Manipulation indicator for Logic Trap cross-feed (E6)
- Discourse marker vocabularies (premise, conclusion, conditional, causal)
- Category load weights: `FORMAL=0.50`, `RELEVANCE=0.30`, `PRESUMPTION=0.40`, `AMBIGUITY=0.25`, `INDUCTIVE=0.35`
- 27 config fields, 4 state fields

---

### E5 — Bias Detection

**Class:** `BiasDetectionEngine` | **File:** `py_engines/bias_detection_engine.py` | **Cluster:** detection

24 bias types across 8 categories. Template matching (keyword + structural + contextual co-occurrence), Bayesian update, severity classification (LOW/MODERATE/HIGH/CRITICAL).

| Aspect | Detail |
|--------|--------|
| NT Read (input) | ACh (implicit), NE (salience), DA (low → ×0.90 threshold), COR (high → ×0.85 threshold) |
| NT Write (output) | ACh attention (Gamma), NE salience (Poisson λ=1.5), 5-HT2A flexibility, DA novelty (Gamma), COR threat-frame, β-boost |
| Key Thresholds | `θ_normal=0.45`, `θ_dev=0.20`, `θ_reflective=0.30`, `θ_rem_dream=0.60` |

**Algorithm:** Stage 1: Threshold resolution (mode + DA/COR bidirectional). Stage 2: Per-bias template matching (24 types, keyword + structural marker). Stage 3: Contextual reinforcement (co-occurrence within categories). Stage 4: Bayesian update `P(bias|E) = prior + α × evidence × (1-prior)`. Stage 5: Severity classification. Stage 6: Bias load `B(t) = Σ confidence × severity_weight`.

**Unique features:**

- 24 bias types × 8 categories (Kahneman-inspired taxonomy)
- Three-stage composition: keyword → structural → contextual
- Threat-frame detection (special COR coupling for FRAMING + SOCIAL biases)
- Severity levels: `LOW(<0.50)`, `MODERATE(≥0.50)`, `HIGH(≥0.70)`, `CRITICAL(≥0.85)`
- 18 config fields, 4 state fields

---

### E6 — Logic Trap Detection

**Class:** `LogicTrapDetectionEngine` | **File:** `py_engines/logic_trap_detection_engine.py` | **Cluster:** detection

21 trap templates across 4 categories (framing, sequential, rhetorical, meta). 3-layer detection fusion, stateful sequence tracking, defensive strategy database, adversarial intent scoring.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | NE, COR, GABA, DA, OXT (all read; OXT has floor=0.30) |
| NT Write (output) | NE burst (Poisson λ=2.5), COR drift (sustained adversarial), GABA reuptake, DA negative (caution), OXT reduction (floor 0.30), β-boost, θ-suppress |
| Key Thresholds | `θ_normal=0.45`, `θ_dev=0.25`, `θ_audit=0.35`, `θ_rem=0.50` |

**Algorithm:** Layer 1 — Template Matching: 21 TrapTemplates with required/supporting/inhibiting features → match_score per type. Layer 2 — Synthesis: `T_synthesis = w_s1×M_fallacy + w_s2×C_coord + w_s3×V_exploit + w_s4×D_toolkit`. Layer 3 — Sequential Analysis: TrapSequenceTracker (escalation, chaining, goalpost shifting, option narrowing). Fusion: `A(t) = 0.35×T_template + 0.30×T_synthesis + 0.35×T_sequential`. Intent Classification: 5 levels (NONE→NEAR_CERTAIN).

**Unique features:**

- 3-layer detection (template + synthesis + sequential)
- 21 named trap templates with required/supporting/inhibiting features
- Stateful `TrapSequenceTracker` for multi-turn escalation patterns
- Defensive strategy database with concrete response templates
- Cross-feed from E1 (Contradiction), E4 (Fallacy), E5 (Bias) engines
- OXT erosion mechanism (floor at 0.30 to protect trust baseline)
- `IntentLevel`: `NONE(0–0.25)`, `LOW(0.25–0.45)`, `MODERATE(0.45–0.65)`, `HIGH(0.65–0.85)`, `NEAR_CERTAIN(0.85+)`
- 22 config fields, 5 state fields

---

## 6. Dialectic Cluster

*Generate opposing views and Socratic questioning* — Engines: E7, E14

### E7 — Simulated Opposition

**Class:** `SimulatedOppositionEngine` | **File:** `py_engines/simulated_opposition_engine.py` | **Cluster:** dialectic

5 opposition modes (counterargument, counterexample, alternative explanation, assumption excavation, structural resistance). Gate outcomes: PASS/CAVEAT/REVISE/BLOCK. Anti-nihilism safeguard.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | NE (>0.6 lowers threshold), GABA (>0.5 lowers), COR (>0.5 raises), DA, ACh |
| NT Write (output) | NE burst (Poisson λ=1.8), DA outcome-dependent, ACh analytical (Gamma 2.5/0.4), GABA reuptake suppress, COR eustress (Deep only), β/γ/θ osc |
| Key Thresholds | `θ_quality=0.45`, gate: `PASS<0.25`, `CAVEAT<0.50`, `REVISE<0.75`, `BLOCK>0.75` |

**Algorithm:** Compute depth score (complexity + confidence inversion + stakes + novelty) → Select depth (QUICK/STANDARD/DEEP) → Run opposition modes per depth → Quality scoring `Q = w_q1×validity + w_q2×relevance + w_q3×strength + w_q4×novelty` → Gate classification → Anti-nihilism check (`block_rate > θ_nihilism=0.30`). Opposition activity `Ω(t) = depth_mult × Σ quality × severity × w_mode`.

**Unique features:**

- Five opposition modes with mode-specific weights
- Three depth levels: QUICK (1 mode, 1 output), STANDARD (3, 3), DEEP (5, 5)
- Anti-nihilism safeguard (rolling window 20, raises threshold if blocking too much)
- Dialectical tension detection → paradox candidate emission
- Gate outcomes drive downstream pipeline routing
- 47 config fields, 6 state fields

---

### E14 — Socratic Reasoning

**Class:** `SocraticReasoningEngine` | **File:** `py_engines/socratic_reasoning_engine.py` | **Cluster:** dialectic

6-state dialogue machine (PROBING→ELENCHUS→APORIA→EXPLORING→MAIEUTICS→EXIT). 18+ question types, convergence tracking, fatigue gate, internal self-inquiry mode.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | ACh (sustain longer), DA tonic (lower threshold), OXT (low raises threshold), NE (>0.6 lowers), GABA |
| NT Write (output) | ACh γ-burst (Gamma 2.5/0.4), DA tonic (curiosity) + phasic (insight Gamma 3.0/0.5), OXT collaborative drift, NE Poisson (λ=2.0), GABA reuptake, θ/β/θγ osc |
| Key Thresholds | `θ_socratic=0.35`, `θ_maieutics=0.55` (convergence for insight) |

**Algorithm:** Activation gates: mode + intention + topic depth + fatigue → Initialize/continue dialogue → Compute transition features: `c(t)` contradiction, `a(t)` assumption, `u(t)` uncertainty → State transition via lookup table → Generate question (target → type → template) → Convergence tracking κ → Insight extraction.

**Unique features:**

- 18+ question types across 6 dialogue states + 5 internal types
- 6-state dialogue machine with formal transition rules
- Convergence tracking via proposition trajectory (Jaccard vocabulary stability)
- Internal self-inquiry mode (REM/REFLECTIVE: Socratic questions on unsolved concepts)
- Dialectical distance measure (semantic shift from starting position)
- Frustration detection and graceful exit
- 50+ config fields, complex dialogue state

---

## 7. Executive Control Cluster

*SOAR 5-phase decision cycle with chunking and impasse delegation* — Engines: E3

### E3 — SOAR Production

**Class:** `SOARProductionEngine` | **File:** `py_engines/soar_production_engine.py` | **Cluster:** executive_control

5-phase SOAR decision cycle (Input→Elaboration→Proposal→Decision→Application). Hash-indexed triple WM (O(1) lookup), chunking, neurochemically-weighted preferences, impasse delegation.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | DA, 5-HT, NE, COR, ACh, GABA, OXT, CB1 (all 8 NTs modulate preference weights) |
| NT Write (output) | DA reward (confidence), NE conflict (severity), 5-HT consolidation (chunks), COR escalation (depth), ACh attention (WM load), GABA inhibition (suppress ratio), β/θγ/α osc |
| Key Thresholds | `max_wm=500`, `max_productions=50/cycle`, all thresholds mode-dependent |

**Algorithm:** Phase 1 — Input: Populate WMEs from engine outputs + NT state + rewards + memory. Phase 2 — Elaboration: Match state-elaboration productions until quiescence (max rounds mode-dependent). Phase 3 — Proposal: Match proposal productions → create operators + ACCEPTABLE preferences. Phase 4 — Decision: Match comparison productions → NT-modulated preference bias → SOAR preference semantics → Impasse detection. Phase 5 — Application: Fire application productions → output-link → decay o-supported WMEs.

**Unique features:**

- Five-phase SOAR decision cycle
- Hash-indexed triple WM: O(1) lookup by identifier, attribute, (id,attr)
- NT-weighted preferences: DA=explore, 5-HT=conserve, NE=urgency, COR=risk suppress, GABA=inhibit, OXT=social, CB1=creative
- Impasse delegation: `TIE→E13`, `CONFLICT→E1+E14`, `NO_CHANGE→E7`, `STATE_NO_CHANGE→E26`
- Chunking: learns new productions from resolved impasses, exportable to LTMM
- Production types: ELABORATION, PROPOSAL, COMPARISON, APPLICATION
- All thresholds/weights mode-dependent (6 modes × 8 NT weights)

---

## 8. Knowledge Substrate Cluster

*AtomSpace hypergraph, PLN inference, ECAN attention economy (OpenCog replacements)* — Engines: E9, E10, E16

### E9 — AtomSpace-Lite

**Class:** `AtomSpaceEngine` | **File:** `cognitools/atomspace_engine.py` | **Cluster:** knowledge_substrate

Typed hypergraph with 15 AtomTypes (6 node + 9 link). O(1) indexes by id/type/name/outgoing. `TruthValue(s,c)`, `AttentionValue(STI,LTI)`. Persistence via `CognitoolsDataStore`.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | 5-HT (stabilize decay), DA (relax write gate), NE (broaden match scope), ACh (tighten precision), GABA (accelerate pruning), COR (tighten write gate), OXT (protect social atoms), CB1 (relax write gate) |
| NT Write (output) | DA delta (novel atoms), NE delta (match failures), ACh delta (match results), 5-HT delta (merges), γ-boost (pruning), θ-boost (consolidation), α-suppress (expansion) |
| Key Thresholds | `max_atoms=50K`, capacity enforce at 90%→80%, maintenance every 10 ticks |

**Algorithm:** Reset counters → Update NT state → Apply mode config → Execute commands (`add_node`/`add_link`/`remove`/`get`/`query`/`pattern_match`/`decay`) → Run maintenance (TV decay + capacity enforcement) → Compute neurochem signals → Return results + stats.

**Unique features:**

- 15 AtomTypes: 6 nodes (CONCEPT, PREDICATE, NUMBER, VARIABLE, SCHEMA, GROUNDED) + 9 links (INHERITANCE, SIMILARITY, EVALUATION, LIST, AND, OR, NOT, IMPLICATION, HEBBIAN)
- `TruthValue(strength, confidence)` with decay (rate=0.001, floor=0.05, identity immune)
- Pattern matching with variable bindings, NT-modulated scope & precision
- Merging: atoms with same type + similar names (≥0.8 Jaccard) merge via TV revision
- Directional link indices: O(k) lookup for links from/to specific atoms
- Persistence: `export_to_dict`/`import_from_dict` + `CognitoolsDataStore` bridge for LTMM
- Modes: `ANALYTICAL` (strict), `CREATIVE` (lenient), `REM_DREAM` (ultra-creative)

---

### E10 — PLN Core

**Class:** `PLNEngine` | **File:** `cognitools/pln_engine.py` | **Cluster:** knowledge_substrate

12 inference rules as pure functions. Backward chaining with truth-value propagation, confidence factors, and cycle detection. Requires AtomSpace (E9).

| Aspect | Detail |
|--------|--------|
| NT Read (input) | 5-HT (raise `min_confidence` → conservative), DA (lower → bold), NE (extend depth +5), ACh (not direct), GABA (reduce step budget) |
| NT Write (output) | DA delta (novel inferences), ACh delta (successful chains), NE delta (failures), 5-HT delta (revisions), γ-boost (consolidation), θ-boost (deep reasoning ≥3) |
| Key Thresholds | `max_depth=5`, `max_steps=100`, `min_confidence=0.1` |

**Algorithm:** Backward chaining: Check if target exists with sufficient TV → Apply rules in priority order (Revision → Modus Ponens → Deduction → NOT/AND/OR) → Recurse on premises → Return best node with proof tree. Each rule application uses confidence factor (CF 0.5–1.0).

**Unique features:**

- 12 inference rules: `Deduction(0.9)`, `Induction(0.6)`, `Abduction(0.5)`, `Modus Ponens(0.85)`, `Revision(1.0)`, `AND(0.95)`, `OR(0.95)`, `NOT(0.95)`, `Sim→Inh(0.7)`, `Inh→Sim(0.7)`, `Intensional(0.5)`, `Context(0.8)`
- Backward chaining with cycle detection (visited set)
- Proof trees via `InferenceNode` structure
- TV propagation: longer chains naturally erode confidence
- Modes: `ANALYTICAL` (depth=4, conf=0.2), `CREATIVE` (depth=7, conf=0.05), `REM_DREAM` (depth=10, conf=0.01)

---

### E16 — ECAN Core

**Class:** `ECANEngine` | **File:** `cognitools/ecan_engine.py` | **Cluster:** knowledge_substrate

Attention economy: rent/wage/spread/clamp/AF. Hebbian co-activation links with creation threshold, LTI dynamics. Owns STI/LTI values in AtomSpace.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | NE (broaden spreading), ACh (tighten AF threshold), DA (increase wage), GABA (increase rent), 5-HT (stabilize Hebbian decay), CB1 (lower AF threshold), OXT (social atom wage bonus) |
| NT Write (output) | DA delta (AF entries), NE delta (AF size), ACh delta (focused AF 10–20), 5-HT delta (Hebbian strengthened), γ-boost (spread events), β-boost (focused AF) |
| Key Thresholds | `af_threshold=10.0`, STI range `[-200, 200]`, `max AF=100`, Hebbian creation=3 co-activations |

**Algorithm:** Cycle: Rent (all atoms `STI -= eff_rent`) → Spread (HebbianLinks: A↔B proportional to STI × strength) → Wage (accessed atoms `STI += eff_wage`, OXT bonus for social) → Clamp `[floor, ceiling]` → Compute AF (STI > threshold, top max_af) → LTI updates (+0.1 in AF, -0.01 outside) → Hebbian management (strengthen/create/decay/remove) → Neurochem output.

**Unique features:**

- Economic attention model: atoms compete via STI (volatile) + LTI (persistent)
- Rent/Wage cycle: stale atoms decay, accessed atoms rewarded
- Hebbian learning: co-activated atoms create self-organizing memory links
- Social atoms: OXT amplifies wage for atoms with `social_context` metadata
- STI range `[-200, 200]`, AF threshold modulated by ACh/CB1
- Modes: `ANALYTICAL` (AF=15, rent=1.5), `CREATIVE` (AF=5, wage=15), `REM_DREAM` (AF=2, rent=0.5, spread=0.5)

---

## 9. Pattern Analysis Cluster

*Extract, score, identify, compare patterns and map intentions* — Engines: E8, E11, E18, E19, E20, E23

### E8 — Relevance Scoring

**Class:** `RelevanceScoringEngine` | **File:** `py_engines/relevance_scoring_engine.py` | **Cluster:** pattern_analysis

6-axis scoring: recency, frequency, semantic proximity, attention weight, contextual fit, novelty bonus. Per-item tracking with decay.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | ACh (tighten threshold +0.25), NE (lower threshold -0.20), DA (boost novelty weight +0.30), 5-HT (dampen NT effects +0.20), GABA (raise threshold +0.25) |
| NT Write (output) | DA novelty, ACh tighten, NE broaden, γ/β osc |
| Key Thresholds | `θ_relevance=0.30`, weights: recency=0.20, frequency=0.15, semantic=0.20, attention=0.20, context=0.15, novelty=0.10 |

**Algorithm:** For each item: compute 6 axis scores → weighted sum → NT-modulated threshold → classify as relevant/irrelevant. Recency uses exponential half-life decay (50 ticks). Novelty inversely proportional to frequency (cap at 10) and recency (cap at 5 ticks).

**Unique features:**

- 6-axis scoring with configurable weights
- Per-item tracking with `_ItemRecord` (last_access, access_count, context vectors)
- Exponential recency decay (`half_life=50 ticks`)
- Modes: ANALYTICAL (strict), CREATIVE (lenient, novelty-boosted), REM_DREAM

---

### E11 — Input Relevance Evaluation

**Class:** `InputRelevanceEvaluationEngine` | **File:** `py_engines/input_relevance_evaluation_engine.py` | **Cluster:** pattern_analysis

5 relevance dimensions (Contextual Continuity, Topical Alignment, Novelty Value, Emotional Salience, Intent Relevance). 2-phase pipeline with priority fusion. 4 processing depth levels, 4 quadrants.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | ACh (attention), DA (novelty), NE (vigilance), 5-HT (stability), COR (conflict) |
| NT Write (output) | ACh attention, DA novelty, NE vigilance, COR conflict |
| Key Thresholds | depth: SHALLOW→STANDARD→DEEP→CRITICAL |

**Algorithm:** Phase 1: Compute 5 relevance dimensions independently. Phase 2: Priority fusion (weighted sum → processing depth). Depth routing: SHALLOW for routine, CRITICAL for high-stakes. Quadrant classification based on relevance × urgency.

**Unique features:**

- 5-dimension relevance model
- 4 processing depths: SHALLOW, STANDARD, DEEP, CRITICAL
- 4 quadrants for routing decisions
- Priority fusion with mode-dependent weights

---

### E18 — Data Analysis

**Class:** `DataAnalysisEngine` | **File:** `py_engines/data_analysis_engine.py` | **Cluster:** pattern_analysis

Entity-relation-entity triple extraction (8 entity types, 9 relation types), dependency structures, co-occurrence windows.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | ACh (depth), NE (scope), DA (novelty), 5-HT (stability), GABA (filtering) |
| NT Write (output) | DA novelty, ACh depth, NE scope, γ-boost |
| Key Thresholds | `θ_entity=0.30`, 8 entity types, 9 relation types |

**Algorithm:** Extract entities (PERSON, ORG, CONCEPT, LOCATION, EVENT, QUANTITY, TIME, ABSTRACT) → Detect relations (IS_A, HAS, CAUSES, PART_OF, SIMILAR, OPPOSITE, TEMPORAL, SPATIAL, DEPENDS_ON) → Build dependency graph → Compute co-occurrence within sliding window → NT-modulated depth and scope.

**Unique features:**

- 8 entity types × 9 relation types
- Triple extraction: entity-relation-entity
- Co-occurrence sliding window analysis
- Dependency structure building
- NT-modulated depth (ACh) and scope (NE)

---

### E19 — Pattern Identification

**Class:** `PatternIdentificationEngine` | **File:** `py_engines/pattern_identification_engine.py` | **Cluster:** pattern_analysis

4 pattern types (temporal, structural, semantic, behavioral). Sliding-window hash fingerprinting. Lifecycle: CANDIDATE→CONFIRMED→DECAYING→removed.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | DA (reward novel discovery), 5-HT (stabilise patterns), ACh (tighten confirmation), NE (broaden scan), GABA (accelerate decay) |
| NT Write (output) | DA discovery, 5-HT stabilise, ACh tighten, γ/θ osc |
| Key Thresholds | `confirm_threshold=3` observations, lifecycle: CANDIDATE→CONFIRMED→DECAYING |

**Algorithm:** Detect patterns via sliding-window hash fingerprinting → Classify (temporal/structural/semantic/behavioral) → Lifecycle management: new matches → CANDIDATE, threshold crossings → CONFIRMED, decay below threshold → DECAYING → removed. NT modulation adjusts confirmation threshold and decay rates.

**Unique features:**

- 4 pattern types with distinct detection algorithms
- Sliding-window hash fingerprinting for efficient matching
- Pattern lifecycle with configurable confirmation threshold
- DA rewards novel pattern discovery, 5-HT protects confirmed patterns

---

### E20 — Pattern Comparison

**Class:** `PatternComparisonEngine` | **File:** `py_engines/pattern_comparison_engine.py` | **Cluster:** pattern_analysis

Compares patterns against template library: Jaccard (0.35) + cosine (0.35) + alignment (0.30) scoring. Template lifecycle with decay.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | ACh (tighten match), CB1 (relax match), DA (boost novelty), 5-HT (stability), NE (broaden search) |
| NT Write (output) | ACh tighten, CB1 relax, DA novelty, γ/θ osc |
| Key Thresholds | `θ_match=0.40`, weights: Jaccard=0.35, cosine=0.35, alignment=0.30 |

**Algorithm:** For each pattern: compute Jaccard similarity (set overlap) + cosine similarity (vector space) + alignment score (structural alignment) → weighted fusion → compare against `θ_match` → classify match quality. Template library with lifecycle and decay.

**Unique features:**

- 3-metric comparison: Jaccard + cosine + alignment
- Template library with lifecycle management and decay
- ACh tightens match criteria, CB1 relaxes them
- DA rewards novel non-matching patterns

---

### E23 — Intention Map

**Class:** `IntentionMapEngine` | **File:** `py_engines/intention_map_engine.py` | **Cluster:** pattern_analysis

8 intent categories + 8 archetypes. 3-stage Bayesian classification. 11-channel NT burst matrix, 5-band oscillatory modulation, cross-frequency coupling.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | DA, NE, 5-HT, ACh, OXT, GABA + 5 more (11 total) |
| NT Write (output) | 11-channel B_intent matrix, 5-band Φ_intent, CFC (θγ, αβ) |
| Key Thresholds | `θ_disintegration=0.20`, 8 intent categories, 8 archetypes |

**Algorithm:** Stage 1: Feature extraction (linguistic, emotional, contextual). Stage 2: Bayesian intent classification (8 categories: SEEK_INFO, SHARE_OPINION, REQUEST_ACTION, EXPRESS_EMOTION, CHALLENGE, COLLABORATE, META, CREATIVE). Stage 3: Archetype mapping. 11-channel NT burst matrix per intent category. 5-band oscillatory modulation with cross-frequency coupling.

**Unique features:**

- 8 intent categories × 8 archetypes
- 3-stage Bayesian classification pipeline
- 11-channel NT burst matrix `B_intent` (per intent → multi-channel emission)
- 5-band oscillatory modulation `Φ_intent` + CFC (θγ, αβ)
- Disintegration threshold for low-confidence inputs

---

## 10. Evaluation Cluster

*Comprehensive logical coherence assessment* — Engines: E12

### E12 — Logical Brain

**Class:** `LogicalBrainEngine` | **File:** `py_engines/logical_brain_engine.py` | **Cluster:** evaluation

4-tier submodule scoring (Core Epistemic, Consistency, Extended Fidelity, Continuity). Verdict levels: EXEMPLARY/ADEQUATE/DEFICIENT/CRITICAL.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | NE (vigilance), ACh (attention), Glu (integration), DA (RPE), COR (stress), GABA-A (inhibition) |
| NT Write (output) | NE vigilance, ACh attention, Glu integration, DA RPE, COR stress, β/γ osc |
| Key Thresholds | verdict: `≥0.85 EXEMPLARY`, `≥0.60 ADEQUATE`, `≥0.35 DEFICIENT`, `<0.35 CRITICAL` |

**Algorithm:** Tier 1 — Core Epistemic: logical consistency, evidence grounding, inference validity. Tier 2 — Consistency: internal coherence, temporal consistency, cross-reference alignment. Tier 3 — Extended Fidelity: source reliability, uncertainty propagation, alternative consideration. Tier 4 — Continuity: narrative coherence, argument flow, conclusion support. Weighted aggregate → verdict classification.

**Unique features:**

- 4-tier scoring (Core Epistemic + Consistency + Extended Fidelity + Continuity)
- Verdict levels: `EXEMPLARY(≥0.85)`, `ADEQUATE(≥0.60)`, `DEFICIENT(≥0.35)`, `CRITICAL(<0.35)`
- NE/ACh bidirectional feedback on scoring thresholds

---

## 11. Reasoning Cluster

*Simulation, decision-making, and strategic planning* — Engines: E13, E15, E21

### E13 — Simulation Brain

**Class:** `SimulationBrainEngine` | **File:** `py_engines/simulation_brain_engine.py` | **Cluster:** reasoning

4-phase scenario simulation: seeding→branching→evaluation→synthesis. Temperature-modulated softmax branching, entropy-controlled pruning, uncertainty-driven recursion depth.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | DA (optimism/branch quality), NE (threat salience), ACh (evaluation precision), 5-HT (stability), CB1 (creative branching), COR (catastrophize), GABA (pruning) |
| NT Write (output) | DA optimism, NE threat, CB1 creative, COR catastrophize, θγ/β/γ osc |
| Key Thresholds | `T₀=0.80` (mode-dependent), `D_max=8–15`, `B_max=6–12` |

**Algorithm:** Phase 1 — Seeding: Select N_seeds scenarios weighted by probability + novelty + risk. Phase 2 — Branching: Temperature-modulated softmax (`T = T₀ + α × uncertainty`), prune low-entropy branches (θ_prune mode-dependent). Phase 3 — Evaluation: Score branches (consistency + coherence + reward + uncertainty). Phase 4 — Synthesis: Aggregate best branches → emit scenario recommendations. Recursion depth `D = δ₀ - δ₁×θγ + δ₂×symbolic_reward`, clamped `[D_min, D_max]`.

**Unique features:**

- Temperature-modulated softmax branching
- Entropy-controlled pruning
- Uncertainty-driven recursion depth with θγ coupling
- NT-modulated scenario weighting: DA=optimism, NE=threat, COR=catastrophize, CB1=creative
- Mode-dependent: `REM_DREAM T₀=2.0`, `D_max=15`, `B_max=12`, `θ_prune=0.02`

---

### E15 — Decision Making

**Class:** `DecisionMakingEngine` | **File:** `py_engines/decision_making_engine.py` | **Cluster:** reasoning

3-stage pipeline: confidence fusion (Bayesian log-odds)→risk assessment→decision routing. 4 quadrants (RESPOND/QUALIFY/DEFER/ESCALATE), 5 certainty levels, hard overrides.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | DA (confident/uncertain), NE (alert), ACh (precision), 5-HT (stability), COR (threat), GABA (calm), OXT (social) |
| NT Write (output) | DA confident/uncertain, NE alert, COR threat, GABA calm, β/θγ/αβ osc |
| Key Thresholds | `θ_confidence=0.55`, quadrants: RESPOND/QUALIFY/DEFER/ESCALATE |

**Algorithm:** Stage 1 — Confidence Fusion: Bayesian log-odds aggregation across engine evidence sources. Stage 2 — Risk Assessment: threat probability × severity, modulated by COR/NE. Stage 3 — Decision Routing: confidence × risk → 4 quadrant classification. 5 certainty levels: CERTAIN/PROBABLE/AMBIGUOUS/DOUBTFUL/IGNORANT. Hard overrides: safety/ethics escalation regardless of confidence.

**Unique features:**

- Bayesian log-odds confidence fusion
- 4 quadrants: `RESPOND` (high conf, low risk), `QUALIFY` (high conf, high risk), `DEFER` (low conf, low risk), `ESCALATE` (low conf, high risk)
- 5 certainty levels
- Hard overrides for safety/ethics (bypass normal routing)
- COR/NE bidirectional risk modulation

---

### E21 — Strategic Decision

**Class:** `StrategicDecisionEngine` | **File:** `py_engines/strategic_decision_engine.py` | **Cluster:** reasoning

Multi-step goal planning with commitment tracking, plan revision. Goal tree (max 64), strategy pool (max 12/goal), stagnation detection, GABA-pruning.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | DA (explore strategies), 5-HT (conserve plan), NE (revision trigger), ACh (depth), COR (risk), GABA (prune), CB1 (flexibility) |
| NT Write (output) | DA explore, 5-HT conserve, NE revision, COR risk, β/θ osc |
| Key Thresholds | `stagnation_threshold=5 cycles`, `max_goals=64`, `max_strategies=12/goal` |

**Algorithm:** Register goals → Generate strategies per goal (DA-modulated exploration) → Evaluate strategies (multi-criteria scoring) → Commit to best strategy per goal → Track progress → Stagnation detection (5 cycles no improvement) → Plan revision (NE-triggered) → GABA-pruning of stale strategies. Commitment tracking prevents oscillating between plans.

**Unique features:**

- Goal tree (max 64 goals) with strategy pools (max 12 per goal)
- Stagnation detection: 5 cycles without improvement triggers revision
- GABA-pruning of stale/low-value strategies
- Commitment tracking prevents strategy oscillation
- DA explores new strategies, 5-HT conserves existing plans

---

## 12. Metacognition Cluster

*Monitor own reasoning, learning effectiveness, and identity coherence* — Engines: E24, E31, E32

### E24 — Heuristic Bias

**Class:** `HeuristicBiasEngine` | **File:** `py_engines/heuristic_bias_engine.py` | **Cluster:** metacognition

22 heuristic bias types across 4 categories (reasoning, memory, evaluation, reward). Correction authority matrix. Reward system health monitor.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | ACh (meta-attention), NE (alert), 5-HT2A (flexibility), DA (correction reward) |
| NT Write (output) | ACh meta-attention, NE alert, 5-HT2A flexibility, DA correction, γ-boost |
| Key Thresholds | `θ_anchoring=0.65`, 22 bias types, 4 categories |

**Algorithm:** Detect heuristic biases in system reasoning → Correction authority matrix (which biases can override which) → Reward system health monitoring (domain balance, prediction calibration) → Emit corrections with authority levels.

**Unique features:**

- 22 heuristic bias types × 4 categories (reasoning, memory, evaluation, reward)
- Correction authority matrix (bias-specific override permissions)
- Reward system health monitoring (domain balance, prediction calibration)
- DA rewards successful corrections

---

### E31 — Reflective Learning

**Class:** `ReflectiveLearningEngine` | **File:** `py_engines/reflective_learning_engine.py` | **Cluster:** metacognition

Meta-analysis of learning effectiveness. Mode effectiveness scoring, subject proficiency tracking, recurring failure detection, meta-pattern identification.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | DA (`salience_threshold 0.3+0.4×DA`), 5-HT (`abstraction_level 0.3+0.4×5HT`), ACh (`precision_factor 0.5+0.5×ACh`), NE (`urgency_weight 0.3+0.7×NE`) |
| NT Write (output) | Output via `nt_modulation` dict (DA salience, 5-HT abstraction, ACh precision, NE urgency) |
| Key Thresholds | `recurring_failure_min=2`, `stagnation_threshold=0.30` |

**Algorithm:** Compute per-mode statistics (turns, confirmations, contradictions) → Per-subject proficiency trending (first half vs second half) → Recurring failure detection (error types ≥ min_count) → Meta-pattern identification (comfort zone, confirmation bias risk, underperforming modes) → Rank mode preferences → Generate recommendations.

**Unique features:**

- Meta-learning across entire learning log history
- Proficiency trend detection (first half vs second half comparison)
- Pattern merging at high abstraction (5-HT > 0.6)
- Precision-based failure detection (ACh lowers `min_count` via `1/precision_factor`)
- DA-modulated salience filtering for meta-patterns

---

### E32 — Reflective Identity

**Class:** `ReflectiveIdentityEngine` | **File:** `py_engines/reflective_identity_engine.py` | **Cluster:** metacognition

Identity coherence monitoring. Core contradiction detection (word overlap + negation asymmetry), fragile conclusion identification, identity-behaviour alignment, emotion assessment, coherence scoring.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | OXT (`social_weight 0.3+0.4×OXT`), 5-HT (`stability_tolerance 0.2+0.3×5HT`), DA (`self_relevance 0.5+0.3×DA`), COR (`threat_bias 0.0+0.15×COR`) |
| NT Write (output) | Output via `nt_modulation` dict (OXT social, 5-HT stability, DA relevance, COR threat) |
| Key Thresholds | `coherence_disrupted=0.40`, `fragmented=0.70` |

**Algorithm:** Detect core contradictions (word overlap > 0.3 + negation asymmetry) → Fragile conclusions (confidence < 0.3 OR reinforcement < 2) → Identity-behaviour alignment (recent journals vs core values) → Identity-relevant emotions (16 types) → Coherence score (`1.0 - penalties + modulations`) → Status: COHERENT/FRAGMENTED/DISRUPTED. Forced disruption if `confused > 0.6`.

**Unique features:**

- 16 identity-relevant emotions (ashamed, guilty, proud, belonging, etc.)
- Coherence scoring: `contradiction_penalty + fragile_penalty + alignment_penalty + emotion_mod + stability_tolerance`
- Negation asymmetry check for contradiction detection
- Alignment: recent 20 journal entries vs core value keywords
- Forced disruption if `confused > 0.6` (overrides score)

---

## 13. Meta-Self-Awareness Cluster

*Uncertainty pattern detection and propagation analysis* — Engines: E26

### E26 — Uncertainty Pattern

**Class:** `UncertaintyPatternEngine` | **File:** `py_engines/uncertainty_pattern_engine.py` | **Cluster:** meta_self_awareness

4 uncertainty types (epistemic, aleatoric, model, linguistic). Propagation analysis (cascade, island, divergence, stagnation). Emotion integration.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | NE (cascade sensitivity), DA (dampen uncertainty), ACh (focus), 5-HT (stability), COR (amplify), GABA (stabilise) |
| NT Write (output) | NE cascade, DA dampen, ACh focus, COR amplify, GABA stabilise |
| Key Thresholds | `θ_alert=0.50–0.85` (type-dependent) |

**Algorithm:** Classify uncertainty by type (epistemic/aleatoric/model/linguistic) → Propagation analysis: cascade (spreading uncertainty), island (isolated), divergence (growing), stagnation (persistent) → Emotion integration (uncertainty → anxiety/curiosity mapping) → Alert if above type-specific threshold.

**Unique features:**

- 4 uncertainty types with distinct detection algorithms
- 4 propagation patterns: cascade, island, divergence, stagnation
- Emotion integration: uncertainty drives anxiety or curiosity depending on context
- NE amplifies cascade detection, DA dampens uncertainty signals

---

## 14. Homeostasis Cluster

*Neurochemical balance monitoring and memory compression* — Engines: E27, E29

### E27 — Neurochemical Homeostatic

**Class:** `NeurochemicalHomeostaticEngine` | **File:** `py_engines/neurochemical_homeostatic_engine.py` | **Cluster:** homeostasis

Cognitive load estimation (sigmoid model), NT bound monitoring, escalating corrections (gradual→aggressive→hard_reset). Dream-mode creative tolerance.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | (monitors all 12 NTs) |
| NT Write (output) | GABA reactive, 5-HT1A containment, CB1 dampening, NE violation, COR chronic stress |
| Key Thresholds | `overload=0.85`, correction levels: gradual→aggressive→hard_reset |

**Algorithm:** Compute cognitive load via sigmoid model → Monitor all 12 NT concentrations against bounds → Classify violations (mild/moderate/severe) → Escalating correction: gradual (small push toward baseline), aggressive (stronger correction), hard_reset (snap to baseline). Dream-mode: creative tolerance (wider bounds, slower corrections).

**Unique features:**

- Monitors all 12 NT concentrations
- Sigmoid cognitive load model
- 3-level escalating corrections: gradual → aggressive → hard_reset
- Dream-mode creative tolerance (wider bounds, slower corrections)
- Tracks chronic violations for COR stress signaling

---

### E29 — Memory Compression

**Class:** `MemoryCompressionEngine` | **File:** `py_engines/memory_compression_engine.py` | **Cluster:** homeostasis

Compression policy assignment (VERBATIM/SEMANTIC/SYMBOLIC/PRUNE) via information-theoretic scoring: entropy, redundancy, salience, recency, access frequency. Override rules for identity/critical/unresolved/emotional content.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | ACh (preserve salience), 5-HT (protect emotional), GABA (accelerate pruning), DA (novelty), COR (stress) |
| NT Write (output) | 5-HT protect, GABA accelerate, ACh salience, α-boost |
| Key Thresholds | `verbatim≥0.75`, `semantic≥0.50`, `symbolic≥0.25`, `prune<0.25` |

**Algorithm:** Compute 5-axis information score (entropy + redundancy + salience + recency + access_freq) → Apply override rules (identity→VERBATIM, critical→VERBATIM, unresolved→SEMANTIC, high_emotion→SEMANTIC) → Assign compression policy. ACh preserves high-salience content, 5-HT protects emotionally tagged memories, GABA accelerates pruning of low-value content.

**Unique features:**

- 4 compression policies: VERBATIM, SEMANTIC, SYMBOLIC, PRUNE
- 5-axis information-theoretic scoring
- Override rules: `identity→VERBATIM`, `critical→VERBATIM`, `unresolved→SEMANTIC`, `high_emotion→SEMANTIC`
- ACh preserves salience, 5-HT protects emotional, GABA accelerates pruning

---

## 15. Emotional Processing Cluster

*Emotion detection, classification, and tone calibration* — Engines: E28

### E28 — Emotional Detection

**Class:** `EmotionalDetectionEngine` | **File:** `py_engines/emotional_detection_engine.py` | **Cluster:** emotional_processing

46-emotion taxonomy in 7 groups. Per-emotion NT profiles. Arousal estimation, tone calibration (valence, coherence, warmth, discord). ENOCH structural override.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | OXT, NE, DA, 5-HT, COR, GABA (6 primary + 7 secondary channels) |
| NT Write (output) | 13+ write channels: tonic (OXT, 5-HT1A, GABA) + phasic (DA, NE, OXT, MOR, COR, ACh, GABA) + oscillatory (α, β, θ, γ) |
| Key Thresholds | `θ_detect=0.25`, 46 emotions, 7 groups |

**Algorithm:** Phase 1: Feature extraction (lexical, semantic, prosodic markers). Phase 2: Multi-label emotion classification (46 types across 7 groups). Phase 3: Per-emotion NT profile lookup → 13+ channel emission. Phase 4: Arousal estimation. Phase 5: Tone calibration (valence + coherence + warmth + discord). ENOCH override for structural boundary emotions.

**Unique features:**

- 46-emotion taxonomy in 7 groups
- Per-emotion NT profiles (13+ channels each)
- ENOCH structural override for boundary emotions
- Tone calibration: valence + coherence + warmth + discord
- Arousal estimation integrated with NT emission
- Matrix burst: multi-channel NT emission from classification

---

## 16. Alignment Cluster

*Retroactive alignment to goals and temporal coherence* — Engines: E30

### E30 — Retroactive Alignment

**Class:** `RetroactiveAlignmentEngine` | **File:** `py_engines/retroactive_alignment_engine.py` | **Cluster:** alignment

4-component state vector `S(t)=[R,C,B,E]`. 3-delta decomposition (δ_sym, δ_aff, δ_rew), collapse probability via sigmoid, EWMA smoothing (τ=4), 4 scan horizons, attribution system.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | COR (temporal threat), 5-HT1A (stability), NE (vigilance, Poisson), DA (RPE ±), OXT (trust decay/repair) |
| NT Write (output) | COR stress (0.12), 5-HT1A calm, NE alert (Poisson λ=1.5), DA RPE (Gamma 2.0/0.3), OXT trust, θ/γ/αθ/β/δ osc |
| Key Thresholds | collapse: `STABLE(<0.15)`, `ELEVATED(0.15–0.30)`, `AT_RISK(0.30–0.50)`, `CRITICAL(0.50–0.70)`, `COLLAPSE_IMMINENT(≥0.70)` |

**Algorithm:** Project past state forward (accounting for acknowledged changes) → Compute 3 deltas: δ_sym(1-cosine), δ_aff(Euclidean), δ_rew(Euclidean) → Collapse probability via sigmoid with interaction term (`sym×aff + sym×rew + aff×rew`) → EWMA smoothing with hysteresis (τ=4, t_hysteresis=2) → Attribution (SELF/OTHER/SYSTEM/UNKNOWN) → Corrections: symbolic_contradiction, affective_bridge, memory_trust, reward_recalibration → NT emission based on attribution type.

**Unique features:**

- 4-component state vector: `[R(t), C(t), B(t), E(t)]`
- 3-delta decomposition: symbolic (cosine), affective (Euclidean), reward (Euclidean)
- Collapse probability via sigmoid with interaction term
- EWMA smoothing (τ=4 cycles) with hysteresis gating (2 cycles)
- 4 scan horizons: `IMMEDIATE(1/cycle)`, `SESSION(5)`, `CROSS_SESSION(15)`, `IDENTITY(100, REM only)`
- Attribution system: SELF/OTHER/SYSTEM/UNKNOWN → determines NT response profile
- 5 collapse states: STABLE → ELEVATED → AT_RISK → CRITICAL → COLLAPSE_IMMINENT
- 10 emotion triggers for affective consequence mapping

---

## 17. Learning Cluster

*Reward-based, contextual, and recursive meta-learning* — Engines: E17, E22, E25

### E17 — Reward-Based Learning

**Class:** `RewardBasedLearningEngine` | **File:** `py_engines/reward_based_learning_engine.py` | **Cluster:** learning

Prediction error learning: `δ=r_actual-r_predicted`. Per-parameter tracking with convergence status (DIVERGING→EXPLORING→CONVERGING→CONVERGED→CONSOLIDATED). LR decay with warmup. Consolidation gate.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | DA (LR boost +0.30), 5-HT (stability damp -0.25), NE (urgency +0.20), ACh (credit depth +0.15), GABA (noise gate +0.20), OXT (social domain +0.15), CB1 (explore in REM +0.20), COR (stress penalty -0.15) |
| NT Write (output) | DA RPE (Gamma 2.0/0.3), 5-HT stability, NE surprise (Poisson λ=1.5 if `|δ|>0.15`), ACh depth, θ/γ/β osc |
| Key Thresholds | `LR_initial=0.10`, convergence=0.02 (window 10), consolidation=0.015 (window 15) |

**Algorithm:** Compute prediction errors per domain (`δ = r_actual - r_predicted`) → Update predictions via EMA (α=0.15) → Effective LR with NT modulation (DA+, 5-HT-, NE+, COR-) + mode multiplier + decay (`0.995^cycles` after 5 warmup) → Per-parameter adjustment = `lr × δ`, noise-gated (GABA) → Convergence tracking (5 statuses) → Consolidation (mean `|δ| < 0.015` for 15 cycles → freeze). Phase: INITIAL→ACTIVE→PLATEAU→CONSOLIDATED.

**Unique features:**

- Per-domain reward predictions via running EMA (α=0.15)
- Per-parameter learning records with individual LR, convergence status, history
- 5 convergence statuses: DIVERGING→EXPLORING→CONVERGING→CONVERGED→CONSOLIDATED
- LR decay schedule: `0.995^cycles` after 5-cycle warmup
- Noise gating: `|δ|` below 0.005 suppressed (GABA-modulated)
- Consolidation: parameters with mean `|δ| < 0.015` for 15 cycles frozen
- 8 NTs modulate LR/gate/depth independently
- Mode multipliers: `LEARNING=1.4×LR`, `REM_DREAM=1.6×LR`

---

### E22 — Contextual Learning

**Class:** `ContextualLearningEngine` | **File:** `py_engines/contextual_learning_engine.py` | **Cluster:** learning

Context fingerprinting (SHA-256 hash of topic+emotion+intent → 32/16/32-dim vectors). Context recognition via weighted cosine similarity. Lifecycle: ACTIVE→DORMANT→DECAYED.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | ACh (strengthen encoding +0.15), OXT (social sensitivity +0.12), CB1 (lower threshold +0.10), DA (novel context reward +0.12), 5-HT (stabilise records -0.08), NE (broaden features +0.10) |
| NT Write (output) | DA novel context, ACh encoding, OXT social, 5-HT stability, θ-boost (encoding), γ-boost (recognition) |
| Key Thresholds | `θ_recognition=0.60`, `max_contexts=512`, `decay_half_life=500 ticks` |

**Algorithm:** Build `ContextFingerprint` (topic→32-dim, emotion→16-dim, intent→32-dim via n-gram hashing) → Match against stored records (weighted cosine: topic=0.45, emotion=0.30, intent=0.25) → Classify match quality (EXACT≥0.95, STRONG≥0.80, MODERATE≥θ) → If match: retrieve parameter adjustments + strengthen record (EMA blend 0.20). If no match: encode new context. Lifecycle decay: `exp(-λ×ticks)`.

**Unique features:**

- 3-component fingerprinting: topic(32-dim) + emotion(16-dim) + intent(32-dim)
- SHA-256 hashing + character n-gram vectorization
- Weighted cosine similarity (topic=0.45, emotion=0.30, intent=0.25)
- Context lifecycle: ACTIVE → DORMANT (conf<0.15) → DECAYED (conf<0.05)
- Confidence decay: `exp(-ln(2)/500 × ticks_since_seen)`
- Parameter adjustments stored per context + EMA-blended on re-encounter
- Modes: `ANALYTICAL(θ=0.75)`, `CREATIVE(θ=0.45)`, `REM_DREAM(θ=0.35, broadening=2.0)`

---

### E25 — Recursive Learning

**Class:** `RecursiveLearningEngine` | **File:** `py_engines/recursive_learning_engine.py` | **Cluster:** learning

Meta-learning: monitors E17 effectiveness via sliding window. Plateau/divergence detection via linear regression. Strategy switching: EXPLOIT/EXPLORE/RESET.

| Aspect | Detail |
|--------|--------|
| NT Read (input) | DA (explore probability +0.15), 5-HT (exploit preference -0.12), NE (urgency amplifies divergence +0.10), ACh (deepen analysis +0.08), GABA (smooth meta-metrics +0.10) |
| NT Write (output) | DA explore/improvement, 5-HT exploit, NE divergence, ACh analysis, GABA smooth, β-boost |
| Key Thresholds | `plateau_var=0.001`, `divergence_slope=0.005`, `explore_probability=0.25` |

**Algorithm:** Update performance window (20 cycles of mean_abs_δ, convergence_ratio, LR) → Linear regression for trend → Detect plateau (low variance + convergence < 0.80 for ≥5 cycles) → Detect divergence (positive slope + mean_δ > 0.30 for ≥3 cycles) → Compute explore probability (DA+, 5-HT-, plateau+0.30, divergence-0.15) → Strategy decision: RESET on divergence, EXPLORE on plateau, EXPLOIT otherwise → Emit meta-parameter adjustments → Meta-LR decay (×0.95/cycle).

**Unique features:**

- Monitors E17 performance via sliding window (20 cycles)
- Trend detection via linear regression (slope, R²)
- Plateau detection: low variance (<0.001) + convergence < 0.80 for ≥5 cycles
- Divergence detection: positive slope (>0.005) + mean_δ > 0.30 for ≥3 cycles
- 3 strategies: `EXPLOIT (LR×0.5, gate=0.02)`, `EXPLORE (LR×2.0, gate=0.10)`, `RESET (LR=0.01)`
- Meta-LR decay: ×0.95/cycle (gradual convergence toward EXPLOIT)
- Cooldown after max switches (10): prevents oscillating

---

## 18. Neurochemical Coupling Reference

Every engine reads NT state via `update_neurochem_state(Dict[str, float])` and emits output signals (NT deltas + oscillatory boosts/suppressions) in `process()` return.

### 18.1 Coupling Patterns

| Pattern | Description | Engines |
|---------|-------------|---------|
| Threshold modulation | NT levels raise/lower detection thresholds | E1, E2, E4, E5, E6, E8, E19, E20 |
| Burst emission | NT delta output proportional to engine activity | E1 (DA D2), E4 (ACh), E5 (ACh/NE), E6 (NE), E28 (phasic) |
| Poisson alert | Stochastic NE burst on high-confidence detections | E4, E5, E6, E14, E17, E25, E26 |
| Drift/decay | Sustained NT shift for persistent conditions | E2 (CB1), E6 (COR), E28 (OXT tonic), E30 (COR) |
| Bidirectional | Both reads NT state AND emits NT deltas | E1, E2, E4, E5, E6, E28, E30 |
| Matrix burst | Multi-channel NT burst from classification | E23 (11-channel B_intent), E28 (13+ channels) |
| Oscillatory | α/β/γ/θ/δ/σ boost/suppress signals | All engines emit at least 1 |

### 18.2 Stochastic Distributions Used

| Distribution | Usage | Typical Parameters |
|-------------|-------|-------------------|
| `Gamma(α, θ)` | NT burst magnitude (ACh, DA, NE) | α=2.0–3.0, θ=0.25–0.50 |
| `Poisson(λ)` | NE alert events | λ=1.5–2.5 |
| `LogNormal(μ, σ)` | Discharge/symbolic intent noise | μ=-0.5–0.0, σ=0.7–1.0 |
| `Beta(α, β)` | Bayesian classification priors | Varies per class |

---

## 19. Integration Points

### 19.1 Pipeline Integration

- Orchestrator calls `engine.update_neurochem_state(nt_state)` then `engine.process(input)`
- Detection cluster (E1, E2, E4, E5, E6) runs during pipeline step (e)
- E11 (Input Relevance) runs during pipeline step (a) for depth routing
- E23 (Intention Map) runs during pipeline step (c)
- E28 (Emotional Detection) populates `EmotionDetectionResults` in STMM
- E12 (Logical Brain) runs during Data Processing step (4)
- E15 (Decision Making) runs during Data Processing step (3)
- E29 (Memory Compression) runs during Data Processing step (6)

### 19.2 Neurochemical Layer Integration

- All engines receive NT state from Neurochemical Engine readout
- Engine NT deltas fed back to neurochem layer via reward/feedback loop
- Oscillatory signals modulate band powers in oscillations module
- E27 (Homeostatic) directly monitors all 12 NT bounds

### 19.3 Reward System Integration

- E17 (Reward-Based Learning) receives prediction errors from reward domains
- E24 (Heuristic Bias) monitors reward system health (domain balance, prediction calibration)
- E15 (Decision Making) fuses reward domain alignment into risk assessment
- E30 (Retroactive Alignment) tracks reward trajectory R(t) in state vector

### 19.4 Memory Layer Integration

- E3 (SOAR) and E12 (Logical Brain) accept `MemoryContrastPort` for retrieval
- E9/E10/E16 persist/restore via `CognitoolsDataStore` in LTMM
- E29 (Memory Compression) evaluates `MemoryPacket` descriptors for policy assignment
- E18/E19/E20 annotate `JournalEntry` via `EngineAnnotations`
- E31 (Reflective Learning) analyses `LearningSystemLog` entries
- E32 (Reflective Identity) analyses `CoreMemory`, `IdentityConclusions`, `IdentityJournal`

### 19.5 Knowledge Substrate Integration

- E9 (AtomSpace) — shared hypergraph; all knowledge-dependent engines can query
- E10 (PLN) — backward chaining over AtomSpace with 12 inference rules
- E16 (ECAN) — manages attention (STI/LTI) for all atoms; AF determines salience
- E8 (Relevance Scoring) reads `AttentionValue` from E16 for scoring

### 19.6 Inter-Engine Delegation

| Source | Target | Condition |
|--------|--------|-----------|
| E3 (SOAR) | E13 (Simulation) | TIE impasse |
| E3 (SOAR) | E1 + E14 (Contradiction + Socratic) | CONFLICT impasse |
| E3 (SOAR) | E7 (Opposition) | NO_CHANGE impasse |
| E3 (SOAR) | E26 (Uncertainty) | STATE_NO_CHANGE impasse |
| E4 (Fallacy) | E6 (Logic Trap) | `manipulation_indicator > 0.50` |
| E2 (Paradox) | Unsolved Concepts Buffer | Genuine paradox (Class G) |
| E7 (Opposition) | E2 (Paradox) | Structural resistance → paradox candidate |

---

## 20. Operational Modes

Most engines support mode-dependent configuration. The `OperationalMode` enum defines:

| Mode | Description | Typical Effect |
|------|-------------|----------------|
| NORMAL | Default conversational processing | Balanced thresholds |
| DEV | Developer/diagnostic mode | Lower thresholds, more verbose |
| LEARNING (M1–M5) | Active learning session | Higher sensitivity, faster LR |
| REFLECTIVE | Self-reflective processing | Deeper analysis, identity focus |
| REM_NORMAL | Normal sleep consolidation | Standard thresholds |
| REM_DREAM | Creative dream processing | Very low thresholds, high noise tolerance |

---

## 21. Constants Reference

### 21.1 ENGINE_IDS

| ID | Engine Name |
|----|-------------|
| 1 | `ContradictionDetectionEngine` |
| 2 | `ParadoxDetectionEngine` |
| 4 | `FallacyDetectionEngine` |
| 5 | `BiasDetectionEngine` |
| 6 | `LogicTrapDetectionEngine` |
| 7 | `SimulatedOppositionEngine` |
| 14 | `SocraticReasoningEngine` |
| 3 | `SOARProductionEngine` |
| 9 | `AtomSpaceEngine` |
| 10 | `PLNEngine` |
| 16 | `ECANEngine` |
| 8 | `RelevanceScoringEngine` |
| 11 | `InputRelevanceEvaluationEngine` |
| 18 | `DataAnalysisEngine` |
| 19 | `PatternIdentificationEngine` |
| 20 | `PatternComparisonEngine` |
| 23 | `IntentionMapEngine` |
| 12 | `LogicalBrainEngine` |
| 13 | `SimulationBrainEngine` |
| 15 | `DecisionMakingEngine` |
| 21 | `StrategicDecisionEngine` |
| 24 | `HeuristicBiasEngine` |
| 31 | `ReflectiveLearningEngine` |
| 32 | `ReflectiveIdentityEngine` |
| 26 | `UncertaintyPatternEngine` |
| 27 | `NeurochemicalHomeostaticEngine` |
| 29 | `MemoryCompressionEngine` |
| 28 | `EmotionalDetectionEngine` |
| 30 | `RetroactiveAlignmentEngine` |
| 17 | `RewardBasedLearningEngine` |
| 22 | `ContextualLearningEngine` |
| 25 | `RecursiveLearningEngine` |

### 21.2 ENGINE_CLUSTER_MAP

| Engine ID | Cluster |
|-----------|---------|
| 1, 2, 4, 5, 6 | detection |
| 7, 14 | dialectic |
| 3 | executive_control |
| 9, 10, 16 | knowledge_substrate |
| 8, 11, 18, 19, 20, 23 | pattern_analysis |
| 12 | evaluation |
| 13, 15, 21 | reasoning |
| 24, 31, 32 | metacognition |
| 26 | meta_self_awareness |
| 27, 29 | homeostasis |
| 28 | emotional_processing |
| 30 | alignment |
| 17, 22, 25 | learning |
