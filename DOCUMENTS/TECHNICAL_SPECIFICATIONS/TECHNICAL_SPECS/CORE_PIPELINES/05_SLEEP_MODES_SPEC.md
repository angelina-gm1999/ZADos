# ZADOS Sleep Modes Spec v1.0

> Source: `ROOT/src/zados/core/commanded/sleep_mode/`
> Classes: `REMPipeline`, `DreamPipeline`

---

## 1. Overview

Sleep modes are commanded (function-type) pipelines triggered by `/sleep` commands. They operate **without user interaction** — no emotional feedback loop, no response generation to humans.

| Mode | Command | Purpose |
|------|---------|---------|
| REM | `/sleep rem` or `/sleep` | Memory consolidation + retroactive learning |
| Dream | `/sleep dream` or `/dream` | Creative recombination of stagnated questions |

REM is corrective (fix what went wrong), Dream is generative (explore what's unresolved).

---

## Part A: REM Pipeline

> Source: `rem_mode/pipeline.py`

### A.1 Purpose

Two interleaved functions:
1. **Memory Consolidation** — MTMM → LTMM promotion of high-value packets
2. **Retroactive Learning** — Domain weight self-adjustment based on emotional signals accumulated during the session

### A.2 Phase Sequence

```
Phase 0: Read MTMM Packets
    └── memory.mtmm.logger.get_all()

Phase 1: Score Emotional Signals
    └── For each packet, detect learning-relevant signals from NT snapshot

Phase 2: Aggregate Signal Profile
    └── Weighted average of signal strengths across all packets

Phase 2.5: Origin-Based Adjustments
    ├── Scan AcademicBufferStore for dream candidates
    ├── Scan GeneralQuestionStore for identity-scope questions
    ├── Scan IdentityConclusionsStore for pending/challenged entries
    └── Apply origin-tagged domain boosts (sqrt diminishing returns)

Phase 3: Compute + Apply Domain Weight Adjustments
    └── Signal profile → domain weight deltas → session.learned_domain_weights

Phase 4: MTMM → LTMM Consolidation
    └── Promote packets passing significance/reward/contradiction gates

Journal Write
    └── JournalEntry (trigger=REM_COMPLETE) with stats + adjustments
```

### A.3 Learning Signal Detection

Signals detected from `MemoryPacket.neurochemical_snapshot`:

| Signal | NT Pattern | Domain Weight Effect |
|--------|-----------|---------------------|
| frustration | NE≥0.50, DA≥0.40, COR≥0.40 | logic +0.06, ethics +0.04 |
| curiosity | DA≥0.50, ACh≥0.40, CB1≥0.30 | innovation +0.08 |
| confusion | NE≥0.45, GLU≥0.35 | logic +0.07 |
| boredom | DA≤0.30, NE≤0.30 | all domains -0.03 |
| anxiety | NE≥0.55, COR≥0.50 | ethics +0.05 |
| overwhelmed | NE≥0.65, COR≥0.60 | all domains -0.02 |

**Fallback:** When NT snapshot is absent (recompressed packets), signals are detected from `emotion_vector` labels using direct label matching (e.g., "frustrated" → frustration at label intensity).

### A.4 Origin-Based Boosts

Additional domain adjustments from origin-tagged items:

| Origin | Domain Boosts |
|--------|--------------|
| academic | logic +0.06, ethics +0.02 |
| identity | ethics +0.06, attunement +0.05 |
| dialectic | logic +0.04, ethics +0.03 |

Scale: sqrt(count)/3.0, capped at 1.0x. (1 item → 0.33x, 9+ items → 1.0x)

### A.5 Consolidation Gates

A packet is promoted to LTMM if ANY of these gates pass:

| Gate | Condition |
|------|-----------|
| Primary | `emotional_significance ≥ 0.45` |
| Secondary | average reward score ≥ 0.40 |
| Tertiary | `contradictions_detected > 0` or `paradoxes_detected > 0` |

LTMM entries written with `Granularity.SEMANTIC`, relevance = significance + 0.2.

### A.6 Output

Returns dict:
```python
{
    "status": "completed",
    "session_id": str,
    "processing_time_s": float,
    "packets_scanned": int,
    "packets_consolidated": int,
    "dominant_signals": [str],       # top-4 signals > 0.1
    "domain_weight_adjustments": {str: float},
}
```

---

## Part B: Dream Pipeline

> Source: `dream_mode/pipeline.py`

### B.1 Purpose

Creative recombination of stagnated/unresolved items. Dream mode leans into **possibility** rather than corrective re-weighting. Items that REM couldn't consolidate (high confusion, stagnated questions) are processed with abstract re-association.

### B.2 Phase Sequence

```
Phase 0: Gather Dream Candidates
    ├── Source 1: UnsolvedBuffer items tagged "dream_candidate"
    ├── Source 2: GeneralQuestionStore identity-scope questions
    └── Sort: identity first, general middle, academic last

Phase 1: Build Emotional Signal Profile
    ├── Source 1: Recent MTMM packet NT snapshots (retroactive)
    ├── Source 2: Live neurochem readout (current dream-phase state)
    └── Average both sources per signal

Phase 2: Retroactive Domain Weight Adjustments
    └── Dream signal profile → domain weight deltas → session.learned_domain_weights

Phase 3: Creative Recombination (max 6 candidates)
    ├── Build dream InputBundle with context flags:
    │   ├── dream_mode = True
    │   ├── cb1_plasticity = True (schema flexibility)
    │   ├── abstract_association = True
    │   ├── identity_salience / oxt_boost (for identity-origin)
    │   └── dream_signal:{signal} for each active signal
    ├── Create isolated SessionState (dream_{session_id})
    ├── Process through AnswerPipeline
    ├── Novel connection = answer > 40 chars
    ├── mark_attempted() on unsolved buffer
    └── Write novel connections to LTMM

Journal Write
    └── JournalEntry (trigger=REM_COMPLETE, source=dream_pipeline)
```

### B.3 Dream Signal Detection

| Signal | NT Pattern | Domain Weight Effect |
|--------|-----------|---------------------|
| curiosity | DA≥0.50, ACh≥0.40, CB1≥0.30 | innovation +0.07 |
| confusion | NE≥0.45, GLU≥0.35 | logic +0.06 |
| wonder | DA≥0.55, CB1≥0.35, 5-HT≥0.35 | innovation +0.09 |
| perplexed | DA≥0.45, 5-HT≥0.35, GABA≤0.35 | logic +0.04, innovation +0.05 |

### B.4 Candidate Priority Sorting

| Origin Tag | Priority | Rationale |
|-----------|----------|-----------|
| `origin:identity` | 0 (highest) | Identity questions benefit most from creative exploration |
| general / dialectic | 1 | Standard dream processing |
| `origin:academic` | 2 (lowest) | Better handled by REM's logic-focused consolidation |

### B.5 Novel Connection LTMM Write

When a dream recombination produces a meaningful answer (>40 chars):

| Field | Value |
|-------|-------|
| `source_tier` | STMM |
| `destination_tier` | LTMM |
| `intention` | "dream_connection" |
| `flags` | ["dream", "novel_association"] + origin flags |
| `emotional_significance` | 0.7 (identity) / 0.6 (other) |
| `granularity` | SEMANTIC |
| `relevance_score` | 0.8 (identity) / 0.7 (other) |
| `identity_relevant` | True if origin:identity |

### B.6 Output

Returns dict:
```python
{
    "status": "completed",
    "session_id": str,
    "processing_time_s": float,
    "candidates_found": int,
    "candidates_processed": int,     # max 6
    "novel_connections": int,
    "dominant_signals": [str],       # top-3 signals > 0.1
    "domain_weight_adjustments": {str: float},
}
```

---

## 3. REM vs Dream Comparison

| Aspect | REM | Dream |
|--------|-----|-------|
| Trigger | `/sleep rem` or `/sleep` | `/sleep dream` or `/dream` |
| Focus | Corrective | Generative |
| Signals | frustration, anxiety, overwhelmed | curiosity, wonder, perplexed |
| Weight direction | Fix deficits (raise logic/ethics) | Explore possibilities (raise innovation) |
| Consolidation | MTMM → LTMM promotion | Novel connections → LTMM |
| Candidate source | All MTMM packets | UnsolvedBuffer dream_candidates + identity questions |
| Academic priority | High (origin boost) | Low (deprioritized) |
| Identity priority | Medium | High (sorted first) |
| Pipeline use | No (direct packet analysis) | Yes (AnswerPipeline per candidate) |
