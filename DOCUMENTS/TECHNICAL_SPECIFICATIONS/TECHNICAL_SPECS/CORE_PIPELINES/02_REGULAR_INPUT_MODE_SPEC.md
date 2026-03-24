# ZADOS Regular Input Mode Spec v1.0

> Source: `ROOT/src/zados/core/inputs/regular_input_mode/pipeline.py`
> Class: `RegularInputPipeline`

---

## 1. Purpose

Standard conversation processing. Wraps the AnswerPipeline with intent-driven depth tuning, subject classification, engine tier resolution, and drift detection. This is the default route for all non-command, non-learning, non-reflective messages.

---

## 2. Processing Steps

```
Step 1:  Run E23 (Intention Map) for intent classification
Step 1b: Derive reward profile from intent + mission briefing
Step 2:  Adapt intent → PipelineDepthConfig
Step 3:  Classify subject → SubjectCategory
Step 4:  Resolve engine tiers → engine_weights on bundle
Step 5:  Check drift via ContextAnchorManager
Step 6:  Delegate to AnswerPipeline.process_turn()
Step 7:  Extract low-confidence answers → GeneralQuestionStore
```

---

## 3. Step Details

### Step 1 — Intent Classification

Runs E23 IntentionMapEngine on `raw_text` (if engine available). Returns `IntentionMapResult` with `intent_category` and `dominant_intent`.

### Step 1b — Reward Profile Selection

Two-stage selection:

**Priority 1: Mission briefing keyword override** (session context takes precedence)

| Keywords | Profile |
|----------|---------|
| study, learn, homework, course | `receptive_learning` |
| review, critique, evaluate, assess | `critical_review` |
| explore, curious, wonder, hypothetical | `curiosity_driven` |
| reflect, think about, introspect, self | `reflective_synthesis` |
| creative, write, story, imagination | `curiosity_driven` |

**Priority 2: Intent category mapping**

| Intent Category | Profile |
|----------------|---------|
| connection | `receptive_learning` |
| challenge | `critical_review` |
| exploration | `curiosity_driven` |
| discharge | `receptive_learning` |
| pragmatic | `regular_input` |
| symbolic | `reflective_synthesis` |
| defensive | `critical_review` |
| disintegration | `regular_input` |

### Step 2 — Depth Config

`adapt_intent_to_depth(intent_result)` → `PipelineDepthConfig`:

| Field | Default | Description |
|-------|---------|-------------|
| `perception_depth` | 0.7 | How many perception engines to run |
| `semiotics_depth` | 0.5 | Fractal semiotic expansion depth |
| `emotion_detection_sensitivity` | 0.6 | E28 sensitivity |
| `phase1_depth` | 0.7 | Perception engine count |
| `phase3_engine_count_cap` | 20 | Max engines dispatched |
| `phase4_thinking_token_budget` | 512 | Max tokens for VT |
| `phase5_reward_thoroughness` | 0.7 | 0=fast, 1=exhaustive |
| `phase6_response_style` | "balanced" | balanced / concise / elaborate |

### Step 3 — Subject Classification

`classify_subject_from_text(raw_text)` → `SubjectCategory`:

| Category | Description |
|----------|-------------|
| TECHNICAL | Programming, engineering, systems |
| SCIENTIFIC | Physics, biology, chemistry, math |
| PHILOSOPHICAL | Ethics, epistemology, metaphysics |
| SOCIAL | Relationships, communication, culture |
| CREATIVE | Art, writing, music, design |
| PRACTICAL | How-to, procedures, daily life |
| MIXED | Multi-domain or ambiguous |

Classification uses keyword overlap scoring with aggregate heuristics.

### Step 4 — Engine Tier Resolution

`EngineToolkit.resolve("regular", subject)` → Dict[engine_id, EngineTier]

Applies Mode × Subject matrix:
1. Start with BASE_TIERS["regular"][engine]
2. Apply SUBJECT_PROMOTIONS if subject matches
3. Apply SUBJECT_DEMOTIONS if subject matches
4. Force phantom engines to T4
5. Apply BUDGET_CAPS (demote excess T2 → T3)

Converted to weights: T1=1.0, T2=1.0, T3=0.5, T4=0.0

### Step 5 — Drift Detection

If a context anchor is active:
- `has_drifted(raw_text)` checks divergence > 0.5 threshold
- On drift: creates new anchor with current text/subject/intent
- Uses MemoryContrast if available, falls back to Jaccard distance

### Step 7 — Low-Confidence Question Extraction

When `result.confidence < 0.4`:
- Creates `GeneralQuestion` from user's input text
- Tags: `pipeline:regular_input`, `mode:normal`, `origin:general`
- Priority: `max(0.3, 1.0 - confidence)` (lower confidence → higher priority)
- Written to `GeneralQuestionStore` for later revisiting in M4 or self-ref mode

---

## 4. Dependencies

| Dependency | Required | Purpose |
|------------|----------|---------|
| `answer_pipeline` | Yes | Core processing delegation |
| `context_manager` | No | Drift detection (defaults to new ContextAnchorManager) |
| `engines` | No | E23 access for intent classification |
| `general_question_store` | No | Low-confidence question capture |

---

## 5. Intent Archetypes (E23 Categories)

The 8 intent categories map to pipeline optimization archetypes:

| Intent | Archetype | Optimization |
|--------|-----------|-------------|
| Connection | Guide | Prioritize attunement, memory, emotional; lighter logic |
| Challenge | Opponent | Full dialectic suite; opposition + Socratic high weight |
| Exploration | Explorer | Pattern engines, knowledge substrate, innovation elevated |
| Discharge | Container | Emotional processing maximized; minimal cognitive load |
| Pragmatic | Architect | Data analysis, decision making; streamlined fast path |
| Symbolic | Oracle | Fractal semiotics full depth; abstract reasoning elevated |
| Defensive | Firewall | Attunement + safety awareness; reduced adversarial pressure |
| Disintegration | Stabilizer | Containment maximized; minimal load; safety escalation |
