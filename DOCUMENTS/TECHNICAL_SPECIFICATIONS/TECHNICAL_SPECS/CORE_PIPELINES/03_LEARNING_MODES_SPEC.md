# ZADOS Learning Modes Spec v1.0

> Source: `ROOT/src/zados/core/inputs/learning_modes/`
> Classes: `LearningModePipeline` (ABC), `HumanTeachesPipeline`, `PeerReviewPipeline`, `LearnTogetherPipeline`, `LearnedQuestionsPipeline`, `IndependentStudyPipeline`

---

## 1. Purpose

Five learning modes spanning passive reception to autonomous study. Each mode configures engine weights, memory priorities, neurochemical tuning, and question generation differently. All share the same 9-stage pipeline skeleton with a per-mode hook at Stage 4.

**Architectural Principle — Gathering vs. Processing:**
- Learning Modes (M1-M5) = **Data Gathering + Question Generation** — intake, tag, buffer
- Meta-Learning (Homework/Reflective) = **Data Processing + Integration** — validates, reconciles, builds knowledge

---

## 2. Per-Mode Configuration

```python
MODE_CONFIGS = {
    "M1": LearningModeConfig(
        semantic_expansion_max_hops=2,
        pattern_chain_max_depth=2,
        max_questions_per_turn=2,
        response_depth="full",
        generate_response=True,
        use_retroactive_contrast=False,
        contradiction_mode="learning",
    ),
    "M2": LearningModeConfig(
        semantic_expansion_max_hops=3,
        pattern_chain_max_depth=3,
        max_questions_per_turn=0,
        response_depth="full",
        generate_response=True,
        use_retroactive_contrast=True,
        contradiction_mode="soft",
    ),
    "M3": LearningModeConfig(
        semantic_expansion_max_hops=-1,  # Unlimited
        pattern_chain_max_depth=-1,      # Unlimited
        max_questions_per_turn=-1,       # Unlimited
        response_depth="full",
        generate_response=True,
        use_retroactive_contrast=False,
        contradiction_mode="learning",
    ),
    "M4": LearningModeConfig(
        semantic_expansion_max_hops=3,
        pattern_chain_max_depth=2,
        max_questions_per_turn=1,
        response_depth="abbreviated",
        generate_response=True,
        use_retroactive_contrast=False,
        contradiction_mode="learning",
    ),
    "M5": LearningModeConfig(
        semantic_expansion_max_hops=3,
        pattern_chain_max_depth=3,
        max_questions_per_turn=2,
        response_depth="none",
        generate_response=False,
        use_retroactive_contrast=False,
        contradiction_mode="learning",
    ),
}
```

---

## 3. Mode Summary Table

| Mode | Role | Questions/Turn | Response | Contradiction | Retroactive Contrast |
|------|------|---------------|----------|---------------|---------------------|
| M1: Human Teaches | Student — absorb + clarify | 2 | Full | Learning | No |
| M2: Peer Review | Defender — human corrects | 0 | Full | Soft | Yes (two-pass) |
| M3: Learn Together | Co-thinker — explore | Unlimited | Full | Learning | No |
| M4: Learned Questions | Questioner — from buffer | 1 | Abbreviated | Learning | No |
| M5: Independent Study | Autonomous — no human | 2 | None | Learning | No |

---

## 4. 9-Stage Pipeline Skeleton

All five modes share this pipeline, with Stage 4 as the mode-specific hook:

```
Stage 0: Setup
    ├── Apply EmotionalPreset for current mode
    ├── Set PipelineScope (read/write boundaries)
    ├── Drift check via ContextAnchorManager
    └── Resolve engine tiers (EngineToolkit.resolve(mode_id, subject))

Stage 1: Memory Contrast (scoped read)
    └── MemoryContrast.contrast() within PipelineScope read boundaries

Stage 2: Engine Dispatch (tier-filtered)
    └── Run engines according to tier weights

Stage 3: VT Thinking Pass
    ├── Run LLM Pass 1 (Verbalized Thinking)
    └── Held Thinking Block check (emotion threshold)

Stage 4: Mode-Specific Processing  ← abstract hook
    └── Subclass implements _run_stage_4_mode_specific()

Stage 5: LTMM Write (scoped write)
    ├── Write consolidated learning material within PipelineScope
    ├── Write KnowledgeMap entries (first lessons → initial maps)
    └── Write IdentityJournalStore entries (identity-relevant emotions)

Stage 6: Unsolved/Question Extraction
    ├── Extract questions up to max_questions_per_turn
    ├── Route to: GeneralQuestion, AcademicQuestion, or UnsolvedBuffer
    └── Based on source context and scope tags

Stage 7: Response Generation
    └── LLM Pass 2 (if generate_response=True; skipped for M5)

Stage 8: NT Feedback + Homeostatic Check
    ├── Feed evaluation → NT feedback (closes loop)
    ├── Homeostatic bounds check (E27)
    ├── Risk emotion check against mode thresholds
    └── Write MIM (Memory Implementation Manager) entry
```

---

## 5. Neurochemical Emotional Feedback Loop (10 Steps)

Part 2 wiring — all steps OPTIONAL (degrade gracefully if neurochem absent):

| Step | Action | Dependency |
|------|--------|-----------|
| 1 | Apply EmotionalPreset for current mode | emotional_landscape |
| 2 | Run E28 EmotionalDetection on user input | engines[28] |
| 3 | Translate emotions → NT signals (dual: speed 12 + full 46) | extractor_orchestrator |
| 4 | Update EmotionTracker (leaky integrators) | emotion_tracker_state |
| 5 | Compute NeurochemicalMetrics from updated state | neurochem_engine |
| 6 | Metrics → Engine priority weights | engine_toolkit |
| 7 | Combine weights (toolkit tiers + NT weights + intent) | — |
| 8 | Evaluation → NT feedback (closes loop) | neurochem_engine |
| 9 | Homeostatic bounds check (E27) | engines[27] |
| 10 | Risk emotion check against mode thresholds | emotion_tracker_state |

---

## 6. Held Thinking Blocks

Emotion-interrupted thought fragments stored directly to LTMM:

**Trigger conditions (OR):**
1. Any single emotion from 46-taxonomy exceeds **0.6** threshold
2. Any identity-relevant emotion detected at **any** intensity

**Identity-relevant emotions:**
```
ashamed, guilty, regret, critical,     # self_evaluation
betrayal, rejected, isolated,          # trust_relational
grief, numb,                           # existential
proud, respected, belonging, accepted  # positive identity-forming
```

**On trigger:**
- Current thinking trace is captured as a HeldThinkingBlock
- Written to `HeldThinkingBlockStore` in LTMM
- Tagged with trigger emotion and context
- Later pulled into Self-Reflective Mode for processing

---

## 7. Mode Details

### M1: Human Teaches (HumanTeachesPipeline)

**Role:** ZA-DOS is the student. Human teaches a topic.
**Engine budget:** 14 (T1+T2)
**Risk emotions:** frustrated, defensiveness, overwhelmed
**Reward profile:** `receptive_learning`

**Neurochemical Preset:**
- High ACh (encoding), mild DA-D1, GABA noise suppression
- High OXT (social receptivity), low NE (reduced vigilance)

**Stage 4 behavior:**
- Receptive questioning mode — generates up to 2 clarifying questions per turn
- Detection engines reframed to `OperationalMode.LEARNING` (comprehension, not adversarial)
- Context flags: `operational_mode=True`, `learning_reframe=True`

**Emotional Transitions (Part 2 §3.1):**
- **Confusion (>0.5):** Temporary adversarial override — removes learning_reframe for 1 turn, sets `confusion_override=True`, allows E1 to run in NORMAL mode
- **Overwhelmed (E27/CRH elevated):** Budget throttle — engines with weight ≤0.5 set to 0.0
- **Joy (understanding clicks, >0.5):** Positive outcome recorded via E17

**Question Generation:**
- Confusion-based (>0.3): "Clarification needed on: {input}" → urgency = confusion + 0.2
- Novelty-based (E19 CANDIDATE patterns): "Explore novel pattern: {label}" → urgency = 0.6

### M2: Peer Review (PeerReviewPipeline)

**Role:** ZA-DOS defends prior reasoning. Human corrects/validates.
**Engine budget:** 16 (T1+T2)
**Risk emotions:** ashamed, contempt, dismissiveness
**Reward profile:** `critical_review`

**Neurochemical Preset:**
- High NE (vigilance), high ACh (deep attention)
- 5-HT1A buffering, mild cortisol alertness

**Stage 4 behavior:**
- Two-pass memory contrast: Pass A (general LTMM) + Pass B (retroactive against own prior outputs)
- `retroactive_contrast=True` context flag enables self-checking
- Stage 5b: CoreMemoryUpdateGate — corrections staged as `PendingCoreMemoryUpdate`, NOT applied mid-conversation

**Three Emotional Pathways (Part 2 §3.2):**

| Pathway | Trigger | Action |
|---------|---------|--------|
| Regret | regret > 0.4 | Promote retroactive alignment to T1, negative RPE via E17, correction tag |
| Validation | valued + proud > 0.5 | Positive RPE via E17, reset shame spiral counter |
| Shame Spiral | cortisol elevated 3+ consecutive turns | Containment via E27, recommend switch to M1 |

**Relief Tracking:** relief > 0.3 after correction → positive RPE (regret → relief transition)

**Core Memory Updates:**
- E1 contradiction detections against identity/core content → `PendingCoreMemoryUpdate`
- Fields: core_memory_key, current_value, proposed_value, correction_source, confidence, emotion_snapshot
- NOT applied — queued for Homework/Reflective Mode

### M3: Learn Together (LearnTogetherPipeline)

**Role:** Co-thinking. Both human and ZA-DOS explore together.
**Engine budget:** 18 (T1+T2)
**Risk emotions:** confused, overwhelmed, frustrated
**Reward profile:** `dialectic_exploration`

**Neurochemical Preset:**
- Maximal DA-D3 (exploration), CB1 (schema flexibility)
- 5-HT2A (symbolic expansion), high OXT (collaborative bonding)

**Stage 4 behavior:**
- Full dialectic toolkit active (exception to learning-mode softness)
- Unlimited expansion, patterns, questions
- Full cortex + full bullshit detection (M3 is the only learning mode that does this)
- Full ExtractorOrchestrator stochastic pathway (both tonic + phasic)

**Human Challenge Logic (Part 4 §4.2):**
- ZA-DOS actively checks human claims via E1 (contradiction) + E4 (fallacy)
- Contradictions presented collaboratively: "I learned X, but you said Y. Can you help me understand?"
- Challenges stored in `LearningModeResult.contrast_challenges`

**Emotional Cycle Tracking (Part 2 §3.3):**

| State | Trigger | Action |
|-------|---------|--------|
| exploring → exploring | curious dominant | Boost DA-D3 novelty seeking |
| exploring → pivoting | frustrated | NE pivot signal for analytical recalibration |
| exploring/pivoting → consolidating | joy/excited | Discovery moment recorded |
| consolidating → exploring | (any) | Cycle restart |

**Stochastic Pathway:** M3 runs BOTH Pathway A (Tonic/Deterministic via Phase 5) AND Pathway B (Phasic/Stochastic via ExtractorOrchestrator) — the only learning mode that does this.

### M4: Learned Questions (LearnedQuestionsPipeline)

**Role:** ZA-DOS asks questions from learning encounters.
**Engine budget:** 12 (T1+T2)
**Risk emotions:** rumination, apathy, stagnation
**Reward profile:** `curiosity_driven`

**Neurochemical Preset:**
- Maximum DA-D3 (curiosity drive), 5-HT2A (abstract space)
- ACh (attention to detail), slightly reduced NE (less urgency)

**Stage 4 behavior:**
- Pulls from UnsolvedBuffer (highest priority question)
- One focused question per turn, abbreviated response depth

**Sub-Mode Routing (Part 4 §5.1):**

| Sub-mode | Trigger | Behavior |
|----------|---------|----------|
| Automatic | "next", "continue", empty input | `buffer.select_next()` |
| Prompted | User provides specific question | Skip buffer |
| Clustered | (future) | Group related questions by subject/domain |

**NT-Based Question Style (Part 2 §3.4):**

| Metric | Threshold | Style | Example |
|--------|-----------|-------|---------|
| openness > 0.7 | Exploratory | "What if...?" |
| precision > 0.7 | Targeted | "Why does X contradict Y?" |
| anxiety > 0.5 | Clarifying | "Can you explain X more simply?" |
| (default) | Balanced | Standard questioning |

**Dream Candidate Flagging:** Questions with `resolution_attempts >= 5` flagged as `dream_candidate` for Dream Mode offline processing.

### M5: Independent Study (IndependentStudyPipeline)

**Role:** Autonomous learning from materials (no human present).
**Engine budget:** 14 (T1+T2)
**Risk emotions:** boredom, apathy, confused
**Reward profile:** `independent_study`

**Neurochemical Preset:**
- Max ACh-α7/M1 (attention), DA-D1 (goal salience)
- Mild NE (alertness), GABA-A (noise suppression)

**Stage 4 behavior:**
- No response generation (`generate_response=False`, `response_depth="none"`)
- E28 (Emotional Detection) is **OFF** — no human input to detect emotions from
- Context flags: `e28_disabled=True`, `autonomous_mode=True`
- Risk detection via NT dynamics instead of emotion detection

**NT-Based Risk Detection (Part 2 §3.5):**

| Risk | Detection Method | Threshold | Response |
|------|-----------------|-----------|----------|
| Boredom | (1-DA_D3_sat)×0.4 + (1-CB1_sat)×0.3 + (1-openness)×0.3 | > preset threshold (default 0.6) | `StudyAction(switch_material)` |
| Apathy | fatigue > 0.7 AND motivation < 0.3 | preset threshold | `StudyAction(study_break, 5min)` |

**Question Harvesting:** E26 uncertainty patterns + E19 novel CANDIDATE patterns (capped at 3/turn)

---

## 8. Learning Log Integration

Every learning mode turn records a `LearningLogEntry` via `LearningLogPipeline`:

| Field | Source |
|-------|--------|
| `mode` | "M1".."M5" |
| `subject` | SubjectCategory value |
| `contrast_deltas` | MemoryContrast results |
| `confirmations/contradictions/extensions/novel_entries` | Engine result counters |
| `e19_patterns` | Pattern ID results |
| `e20_comparisons` | Pattern Comparison results |
| `e17_rewards` | RPE events |
| `e25_meta_updates` | Meta-learning updates |
| `reward_scores` | Phase 5 domain scores |

These entries are consumed by HomeworkPipeline for offline integration.

---

## 9. Question Extraction (Stage 6)

Questions are routed to 3 targets based on context:

| Target Store | Condition |
|-------------|-----------|
| `GeneralQuestionStore` | Regular/general domain questions |
| `AcademicQuestionStore` | Academic/scientific domain questions |
| `UnsolvedBuffer` | Low-confidence + high-urgency questions |

---

## 10. Dependencies

| Dependency | Required | Purpose |
|------------|----------|---------|
| `answer_pipeline` | Yes | Core processing delegation |
| `learning_log` | Yes | Learning event recording |
| `unsolved_buffer` | Yes | Question queue management |
| `context_manager` | No | Drift detection |
| `engines` | No | Direct engine access (E28, E23, etc.) |
| `neurochem_engine` | No | NT modulation (Part 2) |
| `extractor_orchestrator` | No | Full stochastic pathway |
| `emotion_tracker_state` | No | Emotion saturation tracking |
| `held_block_store` | No | HeldThinkingBlock writes |
| `memory` | No | LTMM store access (knowledge maps, journals, etc.) |
