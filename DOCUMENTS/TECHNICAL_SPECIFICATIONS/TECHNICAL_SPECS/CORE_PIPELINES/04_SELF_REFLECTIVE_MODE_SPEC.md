# ZADOS Self-Reflective Query Mode Spec v1.0

> Source: `ROOT/src/zados/core/inputs/self_ref_query_mode/pipeline.py`
> Class: `SelfReflectiveQueryPipeline`

---

## 1. Purpose

Enables ZA-DOS to introspect by selecting unresolved questions from the unsolved buffer and exploring them in M3 (Learn Together / dialectic) mode. This is ZA-DOS thinking about its own thinking — examining what it doesn't know and attempting self-directed resolution.

---

## 2. Activation

- Triggered when self-reflective markers are detected in user input AND the unsolved buffer is non-empty
- Classification priority: 3 (after commands and learning mode continuity)
- Confidence: 0.8

---

## 3. Processing Steps

```
Step 0: Inject held thinking blocks into unsolved buffer
    ├── Query LTMM for unreviewed HeldThinkingBlocks (max 5)
    ├── Convert each to synthetic UnsolvedQuestion
    │   ├── source_mode = "held_block"
    │   ├── urgency_score = 0.6
    │   └── tags = ["held_block", "block_id:{id}"]
    └── Add to unsolved buffer

Step 1: Select question from unsolved buffer
    └── select_next() → highest priority unresolved question
        Priority cascade: urgency DESC → creation_date ASC → stagnation DESC

Step 2: Gather context via MemoryContrast
    ├── Query MemoryContrast with question text
    ├── Retrieve contrast divergence score
    └── Retrieve related memories

Step 3: Build synthetic InputBundle for M3
    ├── Overwrite raw_text with synthetic prompt (question + partial answers + context)
    ├── Set active_mode = "LearningMode_M3"
    ├── Classify subject → SubjectCategory
    ├── Resolve engine tiers for M3
    └── Apply tier weights to bundle

Step 4: Delegate to AnswerPipeline
    └── process_turn(bundle, session) → PipelineResult

Step 5: Update unsolved buffer
    └── mark_attempted(question_id, partial_answer=first 200 chars)

Step 6: Mark held blocks as reviewed
    └── Add "reviewed" tag to each used held block in LTMM

Step 7: Write identity journal entry (REFLECTION type)
    └── IdentityJournalEntry with synthesis answer, question tags, NT snapshot
```

---

## 4. Synthetic Prompt Construction

The prompt sent to AnswerPipeline combines:

```
{question_text}
[Previous attempts have explored: {partial_answer_1}; {partial_answer_2}; ...]
[Related context is available from memory.]
```

This ensures the pipeline has full context of prior exploration and doesn't repeat already-attempted angles.

---

## 5. Held Thinking Block Integration

**Source:** `HeldThinkingBlockStore` in LTMM (`thoughts/held_blocks`)

**Query:** Up to 5 unreviewed blocks (exclude tag "reviewed")

**Conversion to UnsolvedQuestion:**

| Field | Value |
|-------|-------|
| `question_text` | Block content (truncated to 500 chars) |
| `source_mode` | "held_block" |
| `source_context` | "Trigger: {trigger_summary}" or "held thinking block" |
| `urgency_score` | 0.6 |
| `tags` | ["held_block", "block_id:{entry_id}"] |

After processing, blocks are tagged "reviewed" to prevent resurface.

---

## 6. Identity Journal Entry

Written after each self-reflective turn:

| Field | Value |
|-------|-------|
| `entry_type` | REFLECTION |
| `content` | Synthesis answer (truncated to 800 chars) |
| `source_pipeline` | "self_reflective" |
| `tags` | ["self_reflective"] + up to 3 question tags + source mode |
| `nt_snapshot` | From STMM CephalicLiquidLogger |
| `emotion_tags` | Top-5 emotions > 0.3 from STMM EmotionDetection |

---

## 7. Output

Returns `SelfRefResult`:

| Field | Description |
|-------|-------------|
| `selected_question` | The UnsolvedQuestion that was explored |
| `context_gathered` | Dict with question_text, source_mode, attempts, partial_answers, contrast info |
| `synthesis` | Final answer from AnswerPipeline |
| `rerouted_to_m3` | Always True (M3 dialectic mode) |
| `pipeline_result` | Full PipelineResult from AnswerPipeline |

---

## 8. Dependencies

| Dependency | Required | Purpose |
|------------|----------|---------|
| `answer_pipeline` | Yes | Core processing delegation |
| `unsolved_buffer` | Yes | Question selection |
| `memory_contrast` | No | Context gathering |
| `context_manager` | No | Drift detection |
| `ltmm` | No | Held thinking block queries |
| `identity_journal_store` | No | REFLECTION journal entries |
