# ZADOS Process & Pipeline Architecture — Master Spec v1.0

> Generated from codebase at `ROOT/src/zados/core/` — March 2026
> Supersedes: scattered pipelinespecs/ documents

---

## 1. System Overview

ZADOS uses a **Matrioshka (nested-doll) pipeline architecture** where an outer InputClassifier layer wraps an inner AnswerPipeline core. All cognitive processing flows through this layered structure.

```
┌─────────────────────────────────────────────────────────────┐
│                    InputClassifier (main.py)                │
│  Classifies RawInput → routes to correct sub-pipeline      │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Sub-Pipeline Layer                      │   │
│  │                                                      │   │
│  │  ┌─────────────────────────────────────────────┐     │   │
│  │  │          AnswerPipeline (pipeline.py)        │     │   │
│  │  │          Phase 0 → Phase 7                   │     │   │
│  │  │          (single-turn processor)             │     │   │
│  │  └─────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  SessionOrchestrator (session.py) — lifecycle management    │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Routing Matrix

```
InputClassifier.classify(RawInput)
    │
    ├── InputType.MESSAGE
    │   ├── MessageSubType.REGULAR ──────────────► RegularInputPipeline
    │   │                                            └── AnswerPipeline (Phases 0-7)
    │   │
    │   ├── MessageSubType.LEARNING_MODE ────────► LearningModePipeline (M1-M5)
    │   │   ├── M1: HumanTeachesPipeline              9-stage pipeline
    │   │   ├── M2: PeerReviewPipeline                 └── delegates to AnswerPipeline
    │   │   ├── M3: LearnTogetherPipeline
    │   │   ├── M4: LearnedQuestionsPipeline
    │   │   └── M5: IndependentStudyPipeline
    │   │
    │   └── MessageSubType.SELF_REFLECTIVE ──────► SelfReflectiveQueryPipeline
    │                                                └── AnswerPipeline (M3 mode)
    │
    └── InputType.FUNCTION
        ├── FunctionSubType.SLEEP
        │   ├── SleepVariant.REM ────────────────► REMPipeline
        │   └── SleepVariant.DREAM ──────────────► DreamPipeline
        │
        └── FunctionSubType.META_LEARNING
            ├── MetaLearningVariant.HOMEWORK ────► HomeworkPipeline
            └── MetaLearningVariant.REFLECTIVE ──► ReflectivePipeline
```

---

## 3. Classification Priority (InputClassifier)

| Priority | Condition | Route |
|----------|-----------|-------|
| 1 | Command prefix (`/sleep`, `/homework`, `/reflective`, `/dream`) | FUNCTION |
| 2 | Session already in learning mode (continuity) | LEARNING_MODE (same M#) |
| 3 | Self-reflective markers + unsolved buffer non-empty | SELF_REFLECTIVE |
| 4 | Learning mode markers in text | LEARNING_MODE (M1-M5) |
| 5 | Default | REGULAR |

**Command Patterns:**

| Pattern | Route |
|---------|-------|
| `/sleep rem` | SLEEP → REM |
| `/sleep dream` | SLEEP → DREAM |
| `/sleep` (bare) | SLEEP → REM (default) |
| `/dream` | SLEEP → DREAM |
| `/homework` | META_LEARNING → HOMEWORK |
| `/reflective` | META_LEARNING → REFLECTIVE |

**Learning Mode Markers:**

| Mode | Trigger Phrases |
|------|----------------|
| M1 | "teach me", "explain to me", "show me how", "i want to learn", "help me understand" |
| M2 | "review this", "check my work", "find errors", "critique", "analyze this" |
| M3 | "let's explore", "let's figure out", "work together", "discuss this", "what do you think about" |
| M4 | "what questions", "what haven't we", "unresolved", "open questions" |
| M5 | "i'll study", "independent", "self-study", "on my own", "let me explore" |

**Self-Reflective Markers:**
"what do i think", "how do i feel about", "reflect on", "my understanding", "what have i learned", "self-reflect", "introspect", "examine my", "review my thinking"

---

## 4. Session Lifecycle (SessionOrchestrator)

### 4.1 Boot Sequence — `open_session()`

| Step | Action | Details |
|------|--------|---------|
| B.1 | Time-delta classification | Branch A (<5s), B (<10min), C (>=10min cold start) |
| B.3 | Read NT state | Pharmacodynamically decayed concentrations |
| B.4 | Neurosymbolic readout | Metrics dict from NeurochemicalEngine |
| B.5 | Mode selection | DEFAULT_MODE_HOOKS → initial_mode token |
| B.6 | MTMM context search | Branch C only — search for prior session context |
| B.7 | Mission briefing | Collected via `set_mission_briefing()` after boot |
| KB | Knowledge bootstrap | Seed LTMM stores + AtomSpace (E9) on first run |

### 4.2 Per-Turn Processing — `process_turn()`

1. Auto-open session if needed
2. Build InputBundle (raw_text + mode + mission_briefing + osc/extractor state)
3. Delegate to `AnswerPipeline.process_turn()`
4. Persist turn-level state back (reward_profile, extractor_state)
5. Return final_answer string (or full PipelineResult via `process_turn_full()`)

### 4.3 Session Close — `close_session()`

| Step | Action | Fallback |
|------|--------|----------|
| 1 | Write OverviewLogEntry | Logged, skipped on failure |
| 2 | Consolidate MTMM → LTMM | Logged, skipped on failure |
| 3 | Increment stagnation counters (tick_unsolved) | Logged, skipped on failure |
| 4 | Flush STMM → MTMM (end_cycle) | Logged, skipped on failure |
| 5 | Persist cognitools (AtomSpace E9 → CognitoolsDataStore) | Logged, skipped on failure |
| 6 | Clear session state | Always runs |

---

## 5. Data Flow Types

### Input Types

| Type | Source | Description |
|------|--------|-------------|
| `RawInput` | User / system | Pre-classification: text + metadata |
| `ClassificationResult` | InputClassifier | Route target + confidence |
| `InputBundle` | Pipeline 1 output | Enriched input for AnswerPipeline |

### Per-Phase Output Types

| Phase | Output Type | Key Fields |
|-------|------------|------------|
| 0 | (validated InputBundle) | — |
| 1 | `PerceptionSnapshot` | intent_archetype, ranked_facets, entity_triples, pattern_list |
| 2 | `NTModulationResult` | mode_token, reward_profile_name, nt_snapshot, metrics |
| 3 | `EngineDispatchResult` | engine_results, engines_run, e28_result |
| 4 | `ThinkingResult` | thinking_trace |
| 5 | `RewardEvaluationResult` | phase5_result, tonic/phasic_applied |
| 6 | `AnswerResult` | final_answer, directive_applied |
| 7 | `PostProcessResult` | memory_packet, compression_policy, learning_updates |

### Aggregate Types

| Type | Description |
|------|-------------|
| `PipelineState` | Accumulates all phase outputs for one turn |
| `PipelineResult` | Returned by AnswerPipeline — final_answer + state + directive |
| `SessionState` | Persistent across turns — session_id, branch, turn_count, learned_domain_weights |

### Mode-Specific Return Types

| Mode | Return Type |
|------|-------------|
| Regular | `PipelineResult` |
| Learning M1-M5 | `LearningModeResult` (wraps PipelineResult + learning_entries + unsolved_questions) |
| Self-Reflective | `SelfRefResult` (selected_question + context + synthesis) |
| REM/Dream | `Dict[str, Any]` (status + statistics) |
| Homework | `Dict[str, Any]` (HomeworkRunSummary serialized) |
| Reflective | `Dict[str, Any]` (ReflectiveModeResult serialized) |

---

## 6. Dependency Wiring

### InputClassifier Constructor Dependencies

```python
InputClassifier(
    session_orchestrator,          # Required — holds pipeline, engines, memory
    learning_log=...,              # Optional — LearningLogPipeline
    unsolved_buffer=...,           # Optional — UnsolvedBuffer
    context_manager=...,           # Optional — ContextAnchorManager
    neurochem_engine=...,          # Optional — NeurochemicalEngine
    extractor_orchestrator=...,    # Optional — ExtractorOrchestrator
    emotion_tracker_state=...,     # Optional — EmotionTrackerState
)
```

### What InputClassifier Extracts from SessionOrchestrator

- `pipeline` → AnswerPipeline instance
- `engines` → Dict[int, Engine] (all 29 cognitive engines)
- `memory` → MemoryLayer (STMM, MTMM, LTMM namespaces)
- `neurochem_engine` → NeurochemicalEngine (if available)
- `extractor_orchestrator` → ExtractorOrchestrator (if available)
- `emotion_tracker_state` → EmotionTrackerState (if available)

### Memory Stores Extracted at Init

| Store | Path | Used By |
|-------|------|---------|
| `GeneralQuestionStore` | `memory.thoughts.general_questions` | RegularInputPipeline |
| `HeldThinkingBlockStore` | `memory.thoughts.held_blocks` | Learning modes |
| `JournalStore` | `memory.journal_store` | REM, Dream |
| `SpecializedLogs` | `memory.manager.logs` | Homework |

---

## 7. Cross-Reference: Code → Spec

| Source File | Spec Document |
|-------------|--------------|
| `core/main.py` | 00_ARCHITECTURE_OVERVIEW (this doc) |
| `core/session.py` | 00_ARCHITECTURE_OVERVIEW §4 |
| `core/pipeline.py` | 01_ANSWER_PIPELINE_SPEC |
| `core/inputs/regular_input_mode/pipeline.py` | 02_REGULAR_INPUT_MODE_SPEC |
| `core/inputs/learning_modes/base.py` | 03_LEARNING_MODES_SPEC |
| `core/inputs/learning_modes/*.py` | 03_LEARNING_MODES_SPEC |
| `core/inputs/self_ref_query_mode/pipeline.py` | 04_SELF_REFLECTIVE_MODE_SPEC |
| `core/commanded/sleep_mode/rem_mode/pipeline.py` | 05_SLEEP_MODES_SPEC |
| `core/commanded/sleep_mode/dream_mode/pipeline.py` | 05_SLEEP_MODES_SPEC |
| `core/commanded/meta_learning_mode/homework_mode/pipeline.py` | 06_META_LEARNING_MODES_SPEC |
| `core/commanded/meta_learning_mode/reflective_mode/pipeline.py` | 06_META_LEARNING_MODES_SPEC |
| `core/processes/*.py` | 07_CORE_PROCESSES_SPEC |
| `core/types.py` | 00_ARCHITECTURE_OVERVIEW §5 |
| `core/tags.py` | 07_CORE_PROCESSES_SPEC §7 |
| `core/time_context.py` | 07_CORE_PROCESSES_SPEC §8 |
