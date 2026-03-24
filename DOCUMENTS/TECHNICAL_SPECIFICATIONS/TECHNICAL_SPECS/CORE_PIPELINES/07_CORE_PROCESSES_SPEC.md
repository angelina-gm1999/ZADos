# ZADOS Core Processes Spec v1.0

> Source: `ROOT/src/zados/core/processes/` + `ROOT/src/zados/core/tags.py` + `ROOT/src/zados/core/time_context.py`

---

## 1. Engine Toolkit (`engine_toolkit.py`)

### 1.1 Purpose

Central Mode × Subject → EngineTier resolution matrix. Controls which engines run at what weight for every mode/subject combination.

### 1.2 Engine Tiers

| Tier | Weight | Meaning |
|------|--------|---------|
| T1 | 1.0 | Always On — core to mode, runs every turn |
| T2 | 1.0 | Subject-Activated — fires when subject matches; shares budget with T1 |
| T3 | 0.5 | Standby — available if T1/T2 flag a need |
| T4 | 0.0 | Off — not relevant for this mode |

### 1.3 Resolution Algorithm

```
1. Start with BASE_TIERS[mode][engine_id]
2. Apply SUBJECT_PROMOTIONS: if subject matches promotion rule, promote tier
3. Apply SUBJECT_DEMOTIONS: if subject matches demotion rule, demote tier
4. Force phantom/unimplemented engines to T4
5. Apply BUDGET_CAPS: if count(T1 + T2) > cap, demote excess T2 → T3
6. Return final Dict[engine_id, EngineTier]
```

### 1.4 Key Methods

| Method | Purpose |
|--------|---------|
| `resolve(mode, subject)` | Full Mode × Subject → tier resolution |
| `tiers_to_weights_by_id(tiers)` | Convert tier dict to engine_id → weight dict |
| `tier_to_weight(tier)` | T1=1.0, T2=1.0, T3=0.5, T4=0.0 |

### 1.5 Modes Supported

`"regular"`, `"M1"`, `"M2"`, `"M3"`, `"M4"`, `"M5"`, `"homework"`, `"reflective"`, `"rem"`, `"dream"`

---

## 2. Unsolved Buffer (`unsolved_buffer.py`)

### 2.1 Purpose

Priority queue of unresolved questions that accumulates across learning sessions and feeds into M4, Self-Reflective Mode, and Dream Mode.

### 2.2 Priority Cascade

Questions are selected by:
1. `urgency_score` DESC
2. `creation_date` ASC (oldest first)
3. `stagnation_time` DESC (longest since attempt)

### 2.3 Key Methods

| Method | Purpose |
|--------|---------|
| `add(question)` | Add UnsolvedQuestion to buffer |
| `select_next()` | Pop highest-priority unresolved question |
| `mark_attempted(id, partial_answer)` | Track resolution attempt |
| `resolve(id)` | Mark question as resolved |
| `get_active()` | Return all unresolved questions |
| `is_empty()` | Check if buffer has any active questions |
| `cluster_questions()` | Group by tag |
| `load_from_ltmm(store)` | Restore persisted questions on boot |
| `sync_resolved_to_ltmm()` | Persist resolved state on close |

### 2.4 UnsolvedQuestion Fields

| Field | Type | Description |
|-------|------|-------------|
| `question_id` | str | UUID hex (12 chars) |
| `question_text` | str | The question |
| `source_mode` | str | "M1".."M5" or "self_ref" or "held_block" |
| `source_context` | str | Brief context snippet |
| `creation_date` | float | Unix timestamp |
| `last_modified` | float | Unix timestamp |
| `urgency_score` | float | 0.0-1.0 |
| `stagnation_time` | float | Seconds since last attempt |
| `resolution_attempts` | int | How many times attempted |
| `partial_answers` | List[str] | Accumulated partial answers |
| `tags` | List[str] | Classification tags |
| `resolved` | bool | Resolution status |
| `scope_tag` | str | "academic" / "general" / "identity" |

### 2.5 Consumers

| Consumer | Access Pattern |
|----------|---------------|
| M4 (Learned Questions) | `select_next()` → process → `mark_attempted()` |
| Self-Reflective Mode | `select_next()` + inject held blocks |
| Dream Pipeline | `get_active()` filtered by "dream_candidate" tag |
| Homework Pipeline | `get_active()` → cross-reference → `resolve()` |
| Session Close | `tick_unsolved()` → increment stagnation counters |

---

## 3. Learning Log (`learning_log.py`)

### 3.1 Purpose

Records structured learning events from each turn in learning modes. Bridge between knowledge gathering (M1-M5) and integration (Homework).

### 3.2 Key Methods

| Method | Purpose |
|--------|---------|
| `record_turn(mode, subject, session_id, engine_results, contrast_result, reward_result)` | Record one turn |
| `get_unprocessed_logs()` | Return entries not yet processed by Homework |
| `mark_processed(turn_ids)` | Mark entries as processed |
| `get_session_logs(session_id)` | Query by session |
| `get_mode_logs(mode)` | Query by learning mode |

### 3.3 Harvested Data

| Source | Fields Captured |
|--------|----------------|
| E19 (Pattern ID) | Detected patterns |
| E20 (Pattern Comparison) | Template matches |
| E17 (Reward Learning) | RPE events (prediction errors) |
| E25 (Recursive Learning) | Meta-learning strategy updates |
| MemoryContrast | Contrast deltas (confirmations, contradictions, extensions) |
| Phase 5 | Reward domain scores |

---

## 4. Context Anchor (`context_anchor.py`)

### 4.1 Purpose

Maintains context anchors for drift detection. When the user sets a session context (mission briefing), subsequent turns are checked for topic drift.

### 4.2 Key Methods

| Method | Purpose |
|--------|---------|
| `create_anchor(raw_text, subject_hint, intent_prior)` | Create new anchor |
| `check_drift(current_text)` | Compute divergence (0.0-1.0) vs anchor |
| `has_drifted(current_text)` | Check if drift > threshold (0.5) |
| `deactivate()` | Deactivate current anchor |

### 4.3 Drift Detection Strategy

1. **Primary:** MemoryContrast divergence score (if available)
2. **Fallback:** Jaccard distance on word sets

### 4.4 ContextAnchor Fields

| Field | Type | Description |
|-------|------|-------------|
| `raw_text` | str | Anchor text |
| `subject_hint` | str | SubjectCategory value |
| `intent_prior` | str | Dominant intent at anchor time |
| `drift_reference` | Dict | Embedding / hash for comparison |
| `timestamp` | float | When anchor was created |
| `active` | bool | Whether anchor is active |

---

## 5. Subject Classifier (`subject_classifier.py`)

### 5.1 Purpose

Classifies input into 7 broad subject domains for engine tier adjustments.

### 5.2 Categories

| Category | Examples |
|----------|---------|
| TECHNICAL | Programming, engineering, systems |
| SCIENTIFIC | Physics, biology, chemistry, math |
| PHILOSOPHICAL | Ethics, epistemology, metaphysics |
| SOCIAL | Relationships, communication, culture |
| CREATIVE | Art, writing, music, design |
| PRACTICAL | How-to, procedures, daily life |
| MIXED | Multi-domain or ambiguous |

### 5.3 Key Functions

| Function | Purpose |
|----------|---------|
| `classify_subject(tokenizer_result, expansion_result)` | Rich classification using pipeline data |
| `classify_subject_from_text(text)` | Keyword-based fallback |

---

## 6. Emotional Landscape (`emotional_landscape.py`)

### 6.1 Purpose

Applies mode-specific emotional presets to InputBundle before pipeline entry. Biases the neurochemical + oscillatory state toward a mode-appropriate profile.

### 6.2 Key Functions

| Function | Purpose |
|----------|---------|
| `get_emotional_preset(mode_id)` | Retrieve preset for mode |
| `apply_preset_to_bundle(bundle, preset)` | Apply NT + oscillatory adjustments to InputBundle |
| `apply_preset_to_neurochem(neurochem_engine, preset)` | Directly modulate NT layer |
| `apply_oscillatory_bias(preset, osc_state)` | Apply oscillatory band biases |

### 6.3 EmotionalPreset Fields

| Field | Description |
|-------|-------------|
| `nt_adjustments` | Per-NT concentration adjustments |
| `oscillatory_bias` | Per-band amplitude adjustments |
| `reward_weight_overrides` | Override reward domain weights |
| `domain_weight_overrides` | Override domain weights |
| `risk_emotions` | Emotions that trigger risk response in this mode |
| `risk_thresholds` | Per-emotion threshold for risk response |

---

## 7. Tag Taxonomy (`tags.py`)

### 7.1 Purpose

Centralized tag namespace for consistent labeling across all memory, journals, and context. All tags are built via the `T` builder class.

### 7.2 Namespaces

| Namespace | Example | Usage |
|-----------|---------|-------|
| `pipeline:*` | `pipeline:regular_input` | Origin pipeline |
| `mode:*` | `mode:learning` | Operational mode |
| `intent:*` | `intent:question` | User intention |
| `signal:*` | `signal:frustration` | Learning/emotional signal |
| `reward:*` | `reward:logic_high` | Reward domain strength |
| `mem:*` | `mem:high_significance` | Memory salience |
| `flag:*` | `flag:contradiction` | Operational event |
| `content:*` | `content:academic` | Content type |
| `origin:*` | `origin:identity` | Provenance |

### 7.3 Builder Usage

```python
from zados.core.tags import T

T.pipeline("regular_input")    # → "pipeline:regular_input"
T.signal("frustration")        # → "signal:frustration"
T.reward("logic", "high")      # → "reward:logic_high"
T.origin("identity")           # → "origin:identity"
```

---

## 8. Time Context (`time_context.py`)

### 8.1 Purpose

Lightweight temporal context stamping on each turn. Provides circadian awareness and session duration tracking.

### 8.2 TimeContextSnapshot Fields

| Field | Description |
|-------|-------------|
| `timestamp` | Unix epoch |
| `hour` | 0-23 |
| `time_of_day` | morning (06-12) / afternoon (12-18) / evening (18-22) / night (22-06) |
| `day_of_week` | Monday-Sunday |
| `circadian_phase` | waking (05-07) / active (07-18) / wind_down (18-22) / sleep (22-05) |
| `session_elapsed_s` | Seconds since session open |
| `flags` | Special time-related flags |

### 8.3 Usage

```python
tc = get_time_context(session_start=session.session_start_time)
bundle.time_context = tc.to_dict()
```

Stamped once per turn (idempotent — skips if already set).

---

## 9. Dispatch Table (`dispatch_table.py`)

### 9.1 Purpose

Maps intent archetypes to cognitive engine numbers for Phase 3 dispatch.

### 9.2 Engine Groups

| Group | Engines | When |
|-------|---------|------|
| GUARDRAIL_ENGINES | {1, 2, 4, 5, 6} | Always run if weight > 0 |
| PERCEPTION_ENGINES | [23, 8, 11, 18, 19] | Phase 1 only |
| POSTPROCESS_ENGINES | [29, 17, 22, 25] | Phase 7 only |

### 9.3 Key Function

```python
get_dispatch_list(archetype, engine_weights) → List[int]
```

Returns sorted list of engine IDs to dispatch, filtered by weight > 0, excluding perception-only and postprocess-only engines.

---

## 10. Mode Profiles (`mode_profiles.py`)

### 10.1 Purpose

Maps mode tokens to static reward profile names. Bridges neurosymbolic mode selection to reward evaluation configuration.

### 10.2 Key Mappings

| Mode Category | Token Examples | Profile |
|---------------|---------------|---------|
| v0.5 neurosymbolic | EmpathicAttunement | reflective |
| Learning M1 | M1 / receptive | receptive_learning |
| Learning M2 | M2 / critical | critical_review |
| Learning M3 | M3 / dialectic | dialectic_exploration |
| Learning M4 | M4 / curiosity | curiosity_driven |
| Learning M5 | M5 / independent | independent_study |
| Sleep REM | rem / triage | sleep_triage / sleep_deep |
| Sleep Dream | dream | sleep_dream |
| Homework | homework | homework_processing |
| Reflective | reflective | reflective_synthesis |

### 10.3 Key Functions

| Function | Purpose |
|----------|---------|
| `profile_for_mode(mode_token)` | Returns reward profile name for any mode token |
| `profile_for_learning_mode(mode_number)` | Returns profile for M1-M5 by number |
