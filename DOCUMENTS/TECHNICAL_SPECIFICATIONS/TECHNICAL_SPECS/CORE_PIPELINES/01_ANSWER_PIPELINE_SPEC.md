# ZADOS Answer Pipeline Spec v1.0

> Source: `ROOT/src/zados/core/pipeline.py` + `ROOT/src/zados/core/phases/`
> Class: `AnswerPipeline`

---

## 1. Purpose

The AnswerPipeline is the single-turn orchestrator at the center of every processing path. All modes (regular, learning, self-reflective, dream) ultimately delegate to this pipeline for the 8-phase cognitive processing cycle.

---

## 2. Phase Execution Order

**Important:** Phase 3 runs BEFORE Phase 2. This is intentional — E28 emotional detection (dispatched in Phase 3) must inform the NT modulation in Phase 2.

```
Phase 0: Input Validation
    ↓
Phase 1: Perception
    ↓
Phase 3: Engine Dispatch  ← runs before Phase 2
    ↓
Phase 2: NT Modulation
    ↓
ThinkingContext build
    ↓
Identity Alignment Check (optional)
    ↓
Phase 4: Thinking (VT / LLM Pass 1)
    ↓
Phase 5: Reward Evaluation
    ↓
Phase 6: Answer (RG / LLM Pass 2)
    ↓
Phase 7: Post-Processing & Memory Loop
```

---

## 3. Phase Details

### Phase 0 — Input Reception (`phase0_reception.py`)

**Function:** `validate_bundle(bundle)`

- Validates InputBundle has non-empty raw_text
- Raises `PipelineValidationError` on failure
- Marks `phase0_validated` on STMM BrainProcessTracker

### Phase 1 — Perception (`phase1_perception.py`)

**Function:** `run_perception(bundle, engines, nt_snapshot, tokenizer, semantic_expander, stmm)`

**Engines dispatched (in order):**

| Engine | ID | Purpose |
|--------|----|---------|
| Intention Map | E23 | Intent classification → archetype |
| Relevance Scoring | E8 | Multi-axis facet scoring |
| Input Relevance | E11 | Filter low-relevance facets |
| Data Analysis | E18 | Entity-relation-entity triple extraction |
| Pattern Identification | E19 | Temporal/structural/semantic pattern detection |

**Output:** `PerceptionSnapshot`
- `intent_archetype` — dominant intent category
- `intent_vector` — intent distribution
- `ranked_facets` — E8 scored items
- `filtered_facets` — E11 filtered items
- `entity_triples` — E18 (subject, relation, object) triples
- `pattern_list` — E19 detected patterns

### Phase 3 — Engine Dispatch (`phase3_dispatch.py`)

**Function:** `run_engine_dispatch(state, engines, nt_snapshot, memory_contrast)`

**Dispatches remaining engines** based on:
1. Archetype from Phase 1 (via `dispatch_table.py`)
2. Engine weights from InputBundle (set by outer pipeline)
3. NT snapshot (modulates engine parameters)

**Archetype → Engine Table (`dispatch_table.py`):**

| Archetype | Engine Set |
|-----------|-----------|
| ANALYTICAL | Logic-heavy: E1, E2, E4, E5, E6, E12, E15 |
| CREATIVE | Divergent: E7, E13, E14, E16 |
| EMPATHIC | Attunement: E28, E23 |
| STRATEGIC | Planning: E3, E15, E21 |
| REFLECTIVE | Meta: E24, E26, E30 |
| GENERATIVE | Pattern: E19, E20 |
| SOCIAL | Relational: E14, E28 |

**Always-on guardrail engines:** E1 (Contradiction), E2 (Paradox), E4 (Fallacy), E5 (Bias), E6 (Logic Trap)

**Perception-only engines (Phase 1):** E23, E8, E11, E18, E19

**Post-process-only engines (Phase 7):** E29, E17, E22, E25

**Output:** `EngineDispatchResult`
- `engine_results` — Dict[engine_id, result_dict]
- `engines_run` — list of dispatched engine IDs
- `e28_result` — EmotionalDetectionResult (if E28 ran)

### Phase 2 — NT Modulation (`phase2_modulation.py`)

**Function:** `run_nt_modulation(bundle, perception, dispatch, engine, stmm, osc_state, extractor_state)`

**Steps:**
1. Read E28 emotional detection from Phase 3 dispatch
2. Compute NT modulation from emotions + context
3. Run mode selection (NeurochemicalMetrics → mode hooks)
4. Map mode token → reward profile name
5. Update oscillatory state
6. Run ExtractorOrchestrator (if available) for full stochastic pathway

**Output:** `NTModulationResult`
- `mode_token` — selected neurosymbolic mode
- `reward_profile_name` — maps to reward evaluation weights
- `nt_snapshot` — updated NT concentrations
- `osc_snapshot` — oscillatory band amplitudes
- `metrics` / `metrics_dict` — NeurochemicalMetrics
- `updated_extractor_state` — persisted to SessionState

### ThinkingContext Build (`thinking_blocks/builder.py`)

**Class:** `ThinkingBlockBuilder`

Assembles compressed context for LLM passes from:
- Mission briefing (session-level context)
- Emotional profile
- Intent vector
- Recent memories (MTMM semantic search)
- Contradiction/uncertainty lists from engine dispatch
- Personality prompts from identity store
- Identity alignment result (if checker available)

### Identity Alignment Check (optional)

If `hardcoded_store` is provided:
- `IdentityAlignmentChecker.check(thinking_context)` runs
- Injects `alignment_result` and `personality_prompts` into ThinkingContext
- If checker unavailable but store exists: extract personality prompts only

### Phase 4 — Thinking (`phase4_thinking.py`)

**Function:** `run_thinking_pass(state, stmm, bundle_dict)`

- **LLM Pass 1 — Verbalized Thinking (VT)**
- Generates thinking trace using ThinkingContext
- `bundle_dict` carries mission_briefing, reward_profile, emotion_profile, oscillations, context_flags
- Output: `ThinkingResult.thinking_trace`

### Phase 5 — Reward Evaluation (`phase5_reward.py`)

**Function:** `run_reward_evaluation(state, stmm, engine, evaluator, bundle_dict)`

- **Dual-pathway reward computation:**
  - **Tonic pathway** — baseline reward from static profile
  - **Phasic pathway** — event-driven reward from engine results
- Uses `Phase5Evaluator` (from `LLM_interpretation.phase5_evaluator`)
- Domain evaluators score across: logic, ethics, innovation, attunement
- Output: `RewardEvaluationResult` wrapping `Phase5Result`

### Phase 6 — Answer (`phase6_answer.py`)

**Function:** `run_answer_pass(state, stmm, phase5_result, bundle_dict)`

- **LLM Pass 2 — Response Generation (RG)**
- Takes Phase 4 thinking trace + Phase 5 reward guidance
- Produces final user-facing answer
- Applies directive (allow / suppress / abstain)
- Output: `AnswerResult.final_answer`

### Phase 7 — Post-Processing (`phase7_postprocess.py`)

**Function:** `run_postprocessing(state, engine, memory, engines, phase5_result, session, journal_writer)`

**Post-processing engines:**

| Engine | ID | Purpose |
|--------|----|---------|
| Memory Compression | E29 | Assign compression policy (VERBATIM/SEMANTIC/SYMBOLIC/PRUNE) |
| Reward-Based Learning | E17 | Prediction error learning → domain weight updates |
| Contextual Learning | E22 | Context fingerprinting → parameter lookup |
| Recursive Learning | E25 | Meta-learning: monitor E17 effectiveness |

**Memory loop:**
1. Create MemoryPacket from turn data
2. Run E29 compression policy assignment
3. Write to STMM (auto-promoted to MTMM at cycle boundary)
4. Feed E17/E22/E25 with reward signals
5. Write journal entry (if journal_writer available)

**Output:** `PostProcessResult`

---

## 4. Bundle Dict (LLM Bridge)

The `bundle_dict` connects pipeline state to LLM prompt builders (VT/RG):

```python
{
    "mission_briefing": str,           # User session context
    "active_reward_profile_name": str, # Current reward profile
    "prior_urgency_risk": float,       # Extractor urgency
    "emotion_profile": dict,           # Current emotional state
    "current_oscillations": Any,       # Oscillatory state
    "extractor_state": Any,            # ExtractorState
    "thinking_context": ThinkingContext, # Compressed context
    "context_flags": dict,             # Pipeline origin / mode flags
}
```

---

## 5. Session State Updates (per turn)

After each turn completes:
1. `session.turn_count += 1`
2. `session.last_interaction_timestamp` updated
3. `session.reward_profile_name` updated from Phase 2 modulation
4. `session.extractor_state` updated from Phase 2 (and Phase 5 if available)

---

## 6. STMM Lifecycle (per turn)

- Created fresh each turn: `STMMStore()`
- User message added: `stmm.add_user_message(raw_text)`
- `BrainProcessTracker` marks each phase completion
- `CephalicLiquidLogger` tracks NT concentrations
- `EmotionDetection` stores emotion signals
- Flushed to MTMM at session close (step 4 of `close_session()`)

---

## 7. Time Context Stamping

Every turn stamps temporal context via `get_time_context()`:

| Field | Description |
|-------|-------------|
| `timestamp` | Unix epoch |
| `hour` | 0-23 |
| `time_of_day` | morning/afternoon/evening/night |
| `day_of_week` | Monday-Sunday |
| `circadian_phase` | waking (05-07) / active (07-18) / wind_down (18-22) / sleep (22-05) |
| `session_elapsed_s` | Seconds since session open |
