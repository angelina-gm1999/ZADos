# ZADOS Meta-Learning Modes Spec v1.0

> Source: `ROOT/src/zados/core/commanded/meta_learning_mode/`
> Classes: `HomeworkPipeline`, `ReflectivePipeline`

---

## 1. Overview

Meta-learning modes consume and integrate the raw data gathered by learning modes (M1-M5). They operate **offline** — no user present, no emotional feedback loop, no live response generation.

| Mode | Command | Purpose |
|------|---------|---------|
| Homework | `/homework` | Process + integrate accumulated learning material |
| Reflective | `/reflective` | Meta-learning analysis + identity coherence check |

**Information flow:**
```
Learning Modes (M1-M5)
    │ LearningLogEntries
    │ UnsolvedQuestions
    │ PendingCoreMemoryUpdates
    ▼
Homework Mode
    │ HomeworkRunSummary
    │ ReflectiveModeInput (fallacy/bias flags)
    ▼
Reflective Mode
    │ ReflectiveModeResult
    │ Identity store mutations
    ▼
LTMM (permanent knowledge)
```

---

## Part A: Homework Pipeline

> Source: `homework_mode/pipeline.py`

### A.1 Purpose

Six-phase offline processing that validates, reconciles contradictions, builds structured knowledge, updates knowledge maps, and integrates conclusions into LTMM.

### A.2 Six-Phase Pipeline

```
Phase 0: Input Assembly & Triage
    ├── Fetch unprocessed LearningLogEntries from LearningLogPipeline
    ├── Batch entries by SubjectCategory
    ├── Compute deficit profiles per batch (NT-based deficit identification)
    ├── Sort batches by deficit severity (worst-first processing)
    └── Identify processing emphasis per subject

Phase 1: Analysis Stage (per batch)
    ├── Content decomposition via relevant engines
    ├── Memory contrast — check against existing LTMM knowledge
    ├── Pattern extraction (E19 novel patterns, E20 reinforcements)
    ├── Contradiction candidate identification
    └── Relevance scoring of entries within batch

Phase 2: Processing Stage (per batch)
    ├── Contradiction resolution (full adversarial weight — unlike learning modes)
    ├── Dialectic stress-testing of validated lessons
    ├── PLN (E10) confidence scoring
    ├── Fallacy detection (E4) + Bias detection (E5)
    └── Paradox identification (E2)

Phase 3: Question Resolution
    ├── Cross-reference batch findings against unsolved buffer
    ├── Resolve questions that batch analysis answered
    ├── Generate new questions from unresolved contradictions
    ├── Flag stagnated questions as dream_candidates (attempts ≥ 5)
    └── Update unsolved buffer

Phase 4: Synthesis & Knowledge Integration
    ├── Build/update KnowledgeMaps from validated lessons
    ├── Apply PendingCoreMemoryUpdates via CoreMemoryUpdateGate
    ├── Detect meta-patterns across batches
    └── Create initial knowledge maps on first lessons per subject

Phase 5: Output & Storage
    ├── Write validated lessons to LTMM
    ├── Write academic buffer entries
    ├── Generate HomeworkRunSummary
    ├── Prepare ReflectiveModeInput (fallacy/bias flags for handoff)
    ├── Mark processed learning log entries
    ├── Write journal entry
    └── Write to OverviewLogStore
```

### A.3 Deficit Profiling (`deficit_profiler.py`)

NT-based identification of learning gaps:

| Function | Purpose |
|----------|---------|
| `compute_batch_deficit(entries)` | Compute deficit profile from batch's reward scores |
| `identify_deficit_domain(deficit_profile)` | Find the weakest domain |
| `get_engine_emphasis(deficit_domain)` | Map deficit → engine emphasis list |
| `sort_batches_by_deficit(batches)` | Process worst-deficit batches first |

### A.4 Processing Output (per batch)

```python
ProcessingOutput:
    validated_lessons: List[Dict]          # Stress-tested and confirmed
    contradictions_resolved: List[Dict]    # Resolved via dialectic
    contradictions_unresolved: List[Dict]  # Flagged for further work
    fallacy_flags: List[Dict]             # → ReflectiveModeInput
    bias_flags: List[Dict]               # → ReflectiveModeInput
    paradox_flags: List[Dict]            # Identified paradoxes
    pln_confidence_scores: Dict[str, float]  # E10 truth-value scores
    pipeline_result: Optional[PipelineResult]  # If sub-task used pipeline
```

### A.5 HomeworkRunSummary

```python
HomeworkRunSummary:
    batches_processed: int
    lessons_validated / lessons_pending: int
    contradictions_resolved / contradictions_unresolved: int
    questions_resolved / questions_new: int
    dream_candidates_flagged: int
    core_memory_updates_applied: int
    fallacy_bias_flags: List[Dict]        # For Reflective handoff
    meta_patterns: List[Dict]             # Cross-batch discoveries
    processing_emphasis: Dict[str, str]   # subject → deficit_domain
```

### A.6 Key Design Decisions

- **Full adversarial engine weights** — unlike learning modes which run soft, homework uses full dialectic pressure
- **No emotional feedback** — NT layer is read-only (diagnostic deficit profiling)
- **Dream candidate threshold** — questions with ≥5 resolution attempts are flagged
- **Core memory gates** — PendingCoreMemoryUpdates from M2 are only applied here (never mid-conversation)
- **Minimum validation confidence** — 0.5 (entries below this remain pending)

---

## Part B: Reflective Pipeline

> Source: `reflective_mode/pipeline.py`

### B.1 Purpose

Six-phase meta-reflective pipeline that analyzes learning patterns and identity coherence. Runs two dedicated engines (E31 + E32) against accumulated learning data and identity stores.

### B.2 Six-Phase Pipeline

```
Phase 0: Input Assembly
    ├── Load learning log entries (all or recent N)
    ├── Load identity stores:
    │   ├── Core memories
    │   ├── Conclusions store
    │   ├── Identity journal
    │   └── Pending core memory updates
    └── Load ReflectiveModeInput from Homework handoff (if available)

Phase 1: Meta-Learning Analysis (E31)
    ├── Feed learning logs to ReflectiveLearningEngine
    ├── Detect recurring failure patterns
    ├── Evaluate mode effectiveness (which modes work for which subjects)
    ├── Assess subject proficiency trends
    ├── Identify style preferences
    └── Generate learning recommendations

Phase 2: Identity Coherence Analysis (E32)
    ├── Feed identity stores to ReflectiveIdentityEngine
    ├── Produce coherence score (0.0-1.0)
    ├── Identify core contradictions
    ├── Flag fragile conclusions (low confidence / frequently challenged)
    ├── Detect alignment issues
    ├── Extract identity themes
    └── Analyze pending core memory updates

Phase 3: Cross-Reference (E31 × E32)
    ├── Correlate E31 recurring failures with E32 identity conclusions
    ├── Detect learning-identity connections
    │   e.g., persistent failure in logic that contradicts self-belief of competence
    └── Correlate meta-patterns from Homework with identity themes

Phase 4: Identity Store Mutations
    ├── Reinforce conclusions aligned with E31 mode effectiveness
    ├── Create new conclusions from E31 meta-patterns
    ├── Recommend updates for conclusions contradicted by evidence
    ├── Write identity journal entries (type=REFLECTION)
    └── Update CorticalReflectionLog.identity_coherence_status

Phase 5: Output & Summary
    └── Build ReflectiveModeResult with all analysis + mutation stats
```

### B.3 Engine Integration

| Engine | ID | Input | Output |
|--------|----|-------|--------|
| ReflectiveLearningEngine | E31 | Learning logs + identity context | Patterns, failures, effectiveness, recommendations |
| ReflectiveIdentityEngine | E32 | Identity stores + emotion state | Coherence score, contradictions, fragility, themes |

Both engines receive NT state via `update_neurochem_state()` if neurochem_engine is available (read-only — the pipeline does NOT inject NT signals).

### B.4 ReflectiveModeResult

```python
ReflectiveModeResult:
    # E31 outputs
    learning_patterns: List[Dict]
    recurring_failures: List[Dict]
    mode_effectiveness: Dict[str, Dict]    # mode → {accuracy, speed, retention}
    subject_proficiencies: Dict[str, Dict] # subject → {level, trend}
    style_preferences: List[Dict]
    learning_recommendations: List[Dict]

    # E32 outputs
    identity_coherence_status: str         # "coherent" / "disrupted" / "fragmented"
    coherence_score: float                 # 0.0-1.0
    core_contradictions: List[Dict]
    fragile_conclusions: List[Dict]
    alignment_issues: List[Dict]
    identity_themes: List[Dict]

    # Mutations applied
    conclusions_reinforced: int
    conclusions_created: int
    conclusions_recommended_for_update: int
    journal_entries_created: int
    pending_updates_analysed: int

    # Cross-referencing
    cross_references: List[Dict]           # E31 × E32 correlations

    # Input stats
    fallacy_flags_processed: int
    bias_flags_processed: int
    meta_patterns_processed: int
    learning_logs_analysed: int
```

### B.5 Key Design Decisions

- **Observational NT** — reads NT state but doesn't inject signals
- **Lazy engine creation** — E31/E32 instantiated on first `process()` call
- **Identity mutation safety** — creates/reinforces conclusions but only *recommends* updates to existing ones (never overwrites)
- **Cross-referencing** — the key value: connecting learning failures to identity beliefs reveals blind spots

---

## 3. Homework → Reflective Handoff

`ReflectiveModeInput` bridges the two pipelines:

| Field | Source | Purpose |
|-------|--------|---------|
| `fallacy_flags` | Phase 2 fallacy detection | E32 checks if fallacies relate to identity beliefs |
| `bias_flags` | Phase 2 bias detection | E32 checks for systematic bias patterns |
| `identity_contradiction_resolutions` | Phase 2 contradiction resolution | E32 validates resolution consistency |
| `meta_patterns` | Phase 4 cross-batch patterns | E31 incorporates into learning trend analysis |
| `source_homework_session` | Session ID | Traceability |
