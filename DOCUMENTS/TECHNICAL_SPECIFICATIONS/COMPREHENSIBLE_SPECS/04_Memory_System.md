# ZADOS Memory System

## Overview

The Memory System gives ZADOS something most AI systems lack: a past. It implements a three-tier temporal hierarchy — short-term, mid-term, and long-term memory — that mirrors how biological memory works. Experiences start in volatile working memory, get compressed and accumulated into session-level storage, and the most important ones are consolidated into persistent long-term memory that survives across sessions.

This isn't just storage. The memory system actively shapes cognition: it detects contradictions between current and past statements, provides context for reasoning, tracks unresolved questions, maintains a persistent identity, and even supports dream-like processing that can generate novel insights from stagnated problems.

Every memory carries its emotional and neurochemical context — not just *what* was said, but *how the system felt* when it was said. This emotional tagging drives consolidation decisions: emotionally significant experiences are more likely to be preserved long-term, just as in biological memory.

---

## STMM — Short-Term Memory Module

### What It Is
STMM is the system's working memory for a single processing turn. Think of it as a whiteboard that gets erased at the start of each turn and filled in as the 7-phase pipeline runs.

### What It Holds
STMM has 10 named component slots:

| Slot | What It Contains |
|------|-----------------|
| **Active Message Buffer** | Last 2 user messages + 2 system responses (FIFO with eviction) |
| **Fractal Decomposition** | Semantic breakdown of the input |
| **Intention Analysis** | E23's intent classification (connection, challenge, exploration, etc.) |
| **Emotion Detection** | E28's output — user emotions, system emotions, tone vector |
| **Memory Contrast** | Matches found when comparing current input against MTMM/LTMM |
| **Cortical Reflection** | Metacognitive observations about the current processing |
| **Brain Process Tracker** | Execution records for all engines that ran this turn |
| **Reward Evaluation** | Phase 5 domain scores and flags |
| **Cephalic Liquid Logger** | Neurochemical snapshot (all 12 NT concentrations + oscillations) |
| **Response Trace** | LLM outputs (internal thinking + final response) |

### Lifecycle
1. `begin_cycle()` — Resets all analysis fields; preserves the message buffer
2. Each pipeline phase reads from and writes to STMM as it runs
3. At turn end, the **Memory Exit Compressor** converts STMM into a compressed MemoryPacket
4. The packet is handed off to MTMM

### The Compression Step
The Memory Exit Compressor performs a lossy-to-lossless conversion:
- **Kept intact**: User/system message text, turn index, full emotion vector, neurochemical snapshot
- **Compressed**: Intention (primary category only), reward (domain scores only, subscores dropped)
- **Filtered**: Memory contrast matches below 0.30 similarity are discarded
- **Added**: Taxonomy tags for pipeline routing, significance markers

The output is a **MemoryPacket** — the universal transfer format between memory tiers.

---

## MTMM — Mid-Term Memory Module

### What It Is
MTMM is session-scoped memory. It accumulates compressed turn data across an entire conversation session and tracks trends over time. If STMM is a whiteboard for one turn, MTMM is the notebook for the whole session.

### Three Internal Components

**1. Raw Interaction Logger**
An ordered list of all MemoryPackets for the session. Every turn's compressed output is appended here. Old, low-importance entries get progressively re-compressed to save space — their neurochemical snapshots are cleared, emotion details are trimmed, and they're compressed to a more symbolic level.

High-importance entries (importance > 0.6) are protected from this aggressive compression and retain their full semantic content.

**2. Session Trends Engine**
Tracks aggregate patterns across the session:
- **Contradiction trend**: Are contradictions increasing, stable, or decreasing?
- **Emotional trajectory**: How is the emotional tone shifting over the conversation?
- **Reward trajectories**: Per-domain score trends (is logic improving? Is attunement drifting?)
- **Intention pattern**: Is the user's intent shifting over time?
- **NT trajectories**: Neurochemical drift patterns

Trends are recalculated every 3 packets, or immediately when a high-significance event occurs (emotional significance > 0.7 or more than 3 flags).

**3. Context Processor**
Provides semantic indexing and search over the session:
- Builds TF-IDF (term frequency) vectors for each packet — no external ML dependencies needed
- Supports cosine similarity search for finding relevant past turns
- Validates cross-session consistency, flagging anomalies like sudden trust-weight drops

### How It's Used
- **Memory Contrast** queries MTMM during Phase 1 to find relevant recent context
- **Learning engines** read session trends to detect patterns
- **Consolidation** at session end reviews all MTMM packets for LTMM promotion

---

## LTMM — Long-Term Memory Module

### What It Is
LTMM is persistent memory that survives across sessions. It's where important experiences, learned knowledge, identity beliefs, and unresolved questions live permanently (or until they decay away through disuse).

### Entry Structure
Each LTMM entry wraps a MemoryPacket with additional metadata:

| Field | Purpose |
|-------|---------|
| **granularity** | VERBATIM (exact), SEMANTIC (meaning preserved), or SYMBOLIC (tags only) |
| **relevance_score** | Decays over time; determines retrieval priority |
| **retrieval_count** | How many times this entry has been accessed |
| **utility_score** | Feedback from retrieval success — was this entry actually useful? |
| **cold_storage** | Whether this entry has been demoted to inactive status |
| **identity_relevant** | If true, this entry is never demoted or purged |

### The Flat Store + Namespaced Stores

LTMM operates at two levels:

**Flat Store**: A general-purpose store for any promoted memory. Used as a fallback when namespaced routing isn't applicable.

**Namespaced Stores** (16+ specialized stores): Organized into three facades:

#### Identity Namespace — "Who am I?"

| Store | Purpose |
|-------|---------|
| **Hardcoded Store** | Immutable bootstrapped values — core identity axioms that never change |
| **Core Memory Store** | Persistent beliefs and self-model. Updates require peer-review validation (M2 mode). All versions preserved with timestamps. |
| **Pending Update Queue** | Staged updates awaiting M2 peer-review approval before becoming core memories |
| **Identity Conclusions Store** | AI-derived values, lessons, and insights about itself |
| **Identity Journal Store** | Reflective journal entries on identity-relevant emotional experiences (see Journal System section below) |

#### Thoughts Namespace — "What am I thinking about?"

| Store | Purpose |
|-------|---------|
| **Overview Log Store** | Session cognitive summaries (~200 words each) with dominant emotions, subject tags, and open threads |
| **Held Thinking Block Store** | Emotionally-interrupted thought fragments — thoughts that were abandoned mid-stream due to emotional intensity. Written directly to LTMM (bypassing MTMM) because they're considered too significant to risk compression. |
| **Unsolved Buffer Store** | Priority queue of unresolved questions with urgency scores and stagnation tracking. The system's active "what don't I know?" list. |
| **General Question Store** | Low-confidence questions from regular conversation, with priority ranking and resolution tracking |

#### Knowledge Namespace — "What do I know?"

| Store | Purpose |
|-------|---------|
| **Library Store** | Ingested reference material from external sources |
| **Lesson Store** | Validated academic insights from learning modes (M1-M5). Has a validation lifecycle: pending → validated / contradicted |
| **Academic Question Store** | Domain-specific knowledge gaps from learning sessions |
| **Knowledge Map Store** | Semantic graphs per subject — nodes (concepts, principles, facts, open questions) connected by typed edges (supports, contradicts, extends, requires, exemplifies) |
| **Notebook Store** | Academic journaling on knowledge-domain learning (see Journal System section below) |
| **Cognitools Data Store** | Persistent state for cognitive engines (E9 AtomSpace hypergraph, E10 PLN rules, E16 ECAN attention values) |

---

## How Memory Flows Between Tiers

### Turn End: STMM → MTMM
```
STMM (full working memory)
  ↓ Memory Exit Compressor
MemoryPacket (compressed, tagged)
  ↓ MTMM.write()
  ├── Raw Interaction Logger (stores packet)
  ├── Session Trends Engine (updates trends)
  ├── Context Processor (indexes for search)
  └── Old entries re-compressed (progressive decay)
```

### Session End: MTMM → LTMM (Consolidation)

At session close, the **Memory Consolidation Engine** reviews every MTMM packet and decides what deserves long-term storage:

**Promotion Criteria:**
| Criterion | Granularity | Identity Relevant? |
|-----------|-------------|-------------------|
| Emotional significance ≥ 0.6 | SEMANTIC | No |
| Contradictions > 1 or paradoxes detected | VERBATIM | No |
| Critical flags (IDENTITY, PARADOX) | VERBATIM | Yes |
| Low trust weight < 0.4 (anomalies worth preserving) | SEMANTIC | No |

Before writing a promoted entry, the **Fractal Pattern Comparator** checks for duplicates:
- **Similarity > 0.85**: Merge — don't write a new entry, reinforce the existing one
- **Similarity > 0.60**: Cross-link — write the new entry but connect it to the similar existing one
- **Similarity < 0.30**: Flag for contradiction — these are similar enough to be related but different enough to potentially contradict

### Relevance Decay

LTMM entries naturally decay over time. A relevance heuristic combines five factors:

| Factor | Weight | How It Works |
|--------|--------|-------------|
| **Recency** | 30% | Exponential decay — halves every week of non-access |
| **Frequency** | 20% | Saturates at ~10 retrievals (diminishing returns) |
| **Emotional weight** | 20% | Packet's emotional significance score |
| **Utility** | 20% | Feedback from whether retrieval was actually useful |
| **Coherence** | 10% | How well this entry fits with the current knowledge base |

The decay process:
- Entries below **0.15 relevance** → demoted to cold storage (excluded from normal search)
- Entries below **0.05 relevance** in cold storage → purge candidates
- **Identity-relevant entries are exempt** — they maintain minimum relevance and are never demoted

---

## Retrieval: How Memory Is Searched

### Retrieval Router (Namespaced Queries)

Queries specify a `query_type` that determines which stores are searched:

| Query Type | Stores Searched |
|------------|----------------|
| **"knowledge"** | knowledge/lessons, library, academic_questions, notebook, knowledge_maps |
| **"identity"** | identity/core, conclusions, journal |
| **"thought"** | thoughts/overview_logs, held_blocks, general_questions |
| **"general"** | thoughts/general_questions, overview_logs, knowledge/lessons, library |

### Memory Contrast (Logic Domain Integration)

The Logic domain's internal consistency submodule uses a **MemoryContrastPort** to compare current statements against stored memories. This supports several query strategies:

- **MTMM search**: For "context" and "semantic" queries — finding relevant recent history
- **LTMM search**: For "concept" and "definition" queries — finding established knowledge
- **Combined search**: Blending MTMM (weight 0.6) and LTMM (weight 0.4) for comprehensive context

The contrast result includes a divergence score (1 - similarity) that feeds directly into the Logic domain's internal consistency evaluation.

### Pipeline-Scoped Search

Different pipelines have pre-built scope filters that restrict which stores they can access:

| Pipeline | Accessible Stores |
|----------|------------------|
| **Regular** | thoughts/overview_logs, general_questions, knowledge/lessons, library |
| **M1-M5 (Learning)** | Knowledge-focused stores |
| **M2 (Peer Review)** | Identity-focused stores (core, conclusions, journal) |
| **Homework** | Expanded knowledge reads |
| **REM (Sleep)** | Unsolved buffer, held blocks, academic buffer (including cold storage) |
| **Dream** | Expanded REM scope plus lessons (including cold storage) |

Note that sleep modes can access cold storage — this is intentional. Consolidation and dreaming should be able to reach memories that have faded from active use, potentially finding new relevance.

---

## Feedback and Interaction with Other Layers

### Memory ←→ Neurochemical Layer

Every memory packet carries a neurochemical snapshot — the full NT state at the time of recording. This serves dual purposes:
- **Emotional context**: When memories are retrieved, the associated NT state tells the system how it "felt" during that experience
- **Consolidation gating**: During sleep/REM, neurochemical conditions determine which memories get promoted

The neurochemical layer also influences memory operations directly:
- **ACh (Acetylcholine)**: High ACh strengthens memory encoding — new experiences are more likely to be retained
- **NE (Norepinephrine)**: Elevated NE enhances emotional tagging — experiences feel more significant
- **Cortisol**: Sustained stress can impair consolidation (just as in biological memory)
- **During sleep**: Specific NT profiles (low NE/5-HT, high ACh) enable effective replay

### Memory ←→ Reward System

The reward system both reads from and writes to memory:
- **Reads**: Logic domain queries memory to check consistency; reward prediction errors reference stored expectations
- **Writes**: Reward scores are stored in memory packets, influencing future consolidation decisions
- **Gates**: Reward thresholds (avg ≥ 0.40) serve as consolidation filters during session-end promotion

### Memory ←→ Cognitive Engines

Many cognitive engines interact directly with memory stores:
- **E8 (Relevance Scoring)**: Uses retrieval frequency for relevance calculations
- **E9 (AtomSpace)**: Persists its hypergraph to CognitoolsDataStore
- **E17 (Reward-Based Learning)**: Reads and writes learned parameters
- **E19 (Pattern Identification)**: Stores confirmed patterns
- **E20 (Pattern Comparison)**: Reads stored templates for comparison
- **E22 (Contextual Learning)**: Stores and retrieves context fingerprints
- **E29 (Memory Compression)**: Determines compression policy for each memory packet
- **E30 (Retroactive Alignment)**: Reads past states to check temporal coherence

### Memory ←→ Core (Session Orchestration)

The Core layer manages memory lifecycle:
- **Session open**: Boots the unsolved buffer from LTMM, restores AtomSpace state
- **Turn end**: Compresses STMM → MTMM via MemoryImplementationManager
- **Session close**: Triggers consolidation (MTMM → LTMM), ticks stagnation counters, writes session overview, persists engine state

### The Unsolved Buffer: Active Learning Frontier

The Unsolved Buffer deserves special mention because it bridges multiple layers:

- **Source**: Questions arise from learning modes (academic), regular conversation (general), engine flags, and self-reflective processing
- **Priority**: Sorted by urgency (descending) → creation date (ascending) → stagnation time (descending)
- **Lifecycle**: Questions can be attempted (partial answers tracked), resolved, or stagnated
- **Stagnation**: At session close, all unresolved questions have their stagnation counter incremented
- **Dream candidates**: Questions with stagnation ≥ 5 cycles become eligible for dream processing
- **Bidirectional sync**: Loaded from LTMM at boot, synced back at close

This creates a persistent "problem queue" that follows the system across sessions, with old unsolved problems eventually being routed to the creative Dream pipeline for novel approaches.

---

## The Journal System

ZADOS maintains two journal systems that work together to create an ongoing narrative record of the system's cognitive and personal development. If memory stores capture *what* happened, the journals capture *what it meant*.

### Two Tiers of Journaling

**1. Cognitive Journal (JournalStore)**

The general-purpose reflective journal. Entries are created across all major pipelines — regular conversation, learning modes, sleep consolidation, and offline processing. Each entry is a rich snapshot that captures not just what the system was thinking, but the full cognitive context around it.

A journal entry contains:
- **Reflective prose** (150-400 words): An LLM-generated first-person monologue reflecting on recent cognitive activity
- **Open questions** (3-5): Self-directed questions for future reflection — creating threads that can be revisited later
- **Engine annotations**: Structured analysis from E18 (entities and relationships extracted), E19 (patterns identified), and E20 (cross-session pattern matches and novelty flags)
- **State snapshots**: The full emotional state (from E28), neurochemical concentrations (all 12 NTs), reward domain scores, and tone vector (valence, warmth, discord, coherence) at the moment of writing
- **Review lifecycle**: Entries start as UNREVIEWED, can move to IN_REVIEW during self-reflective processing, and finally RESOLVED when their open questions have been addressed
- **Cross-links**: Each new entry is compared against recent entries using cosine similarity. Entries above 0.35 similarity are linked bidirectionally, forming a web of related reflections

Entries are **immutable once created** — they represent the system's genuine reflection at a specific moment and are never retroactively edited.

**2. Identity Journal (IdentityJournalStore)**

A separate, more focused journal for identity-relevant experiences only. While the cognitive journal tracks general reflection, the identity journal tracks moments that matter for *who the system is*.

Identity journal entries are triggered when **identity-relevant emotions** are detected during processing:

| Category | Emotions |
|----------|----------|
| **Self-evaluation** | ashamed, guilty, regret, critical |
| **Trust / relational** | betrayal, rejected, isolated |
| **Existential** | grief, numb |
| **Positive identity-forming** | proud, respected, belonging, accepted |

These emotions trigger a journal write at **any positive intensity** — there is no minimum threshold. For the general 46-emotion taxonomy, a separate mechanism captures **held thinking blocks** when any emotion exceeds 0.6 intensity, but identity-relevant emotions get journal treatment regardless of strength because even subtle identity moments matter for self-model development.

Identity journal entries support three types:
- **REGULAR**: Standard identity reflection
- **REFLECTION**: Created by the reflective pipeline (E31 + E32 meta-analysis)
- **COMMENT**: Annotations on previous entries, creating threaded discussions with the self

### When Journal Entries Are Written

Different pipelines trigger journal writes at different moments and for different reasons:

| Pipeline | Trigger | What Gets Captured |
|----------|---------|-------------------|
| **Regular input** (Phase 7) | PERIODIC (every 5 turns), LTMM_THRESHOLD (on memory promotion), or INNOVATION_FLAG (when E7/E14/E19 are active) | Full cognitive state + reward profile + dominant emotion + intent category + temporal context |
| **Learning modes** (M1-M5) | PERIODIC (every turn) | Mode ID + academic subject + learning-specific context |
| **REM sleep** | REM_COMPLETE (after consolidation) | Packets consolidated count + dominant learning signals (frustration, curiosity, confusion) + domain weight adjustments |
| **Dream mode** | REM_COMPLETE (after recombination) | Dream candidates processed + novel connections found + domain orientation nudges |
| **Homework** | PERIODIC (after batch processing) | Lessons validated + contradictions resolved + questions resolved + core memory updates |
| **Self-reflective query** | Via identity journal | Selected question + synthesis result |
| **Reflective pipeline** | Via identity journal (REFLECTION type) | Learning patterns + identity coherence + cross-references between learning and identity |

Each trigger carries a **contextual phrase** that shapes how the LLM writes its reflection. For example:
- **LTMM_THRESHOLD**: "Something from this conversation was significant enough to commit to long-term memory. Reflect on why."
- **REM_COMPLETE**: "A consolidation and dreaming process has just completed. Reflect on what was processed and what it surfaced."
- **INNOVATION_FLAG**: "Something novel was flagged — a pattern, concept, or connection that had not been encountered before. Reflect on it."

### How the Journal Connects to Other Systems

The journal system sits at the intersection of several layers:

**From the Cognitive Engines**: The JournalTool (a cognitool alongside AtomSpace, PLN, and ECAN) runs a 3-phase pipeline — annotate (E18/E19/E20), generate (LLM), tag (rule-based) — to produce each entry. See the Cognitive Engines doc for details on how this pipeline works.

**From the Neurochemical Layer**: Every entry captures the full NT state at writing time. During sleep modes (REM/Dream), the system reads these snapshots back to detect **retroactive learning signals** — patterns like frustration (NE↑ DA↑ Cortisol↑) or curiosity (DA↑ ACh↑ CB1↑) that indicate where domain weight adjustments should be made.

**From the Reward System**: Reward domain scores (logic, ethics, innovation, human attunement) are stored in each entry. These scores feed the reward-based learning engine (E17) and inform the reflective pipeline's analysis of which domains the system excels or struggles in.

**From TimeContext**: Temporal metadata (time of day, circadian phase, session elapsed time) is attached to journal entries as pipeline notes, providing time-aware context for future retrieval and analysis.

**To self-reflection**: The self-reflective query pipeline pulls unreviewed journal entries and held thinking blocks, using them as starting material for deeper exploration. The reflective pipeline (E31 + E32) reads journal history to identify recurring themes and track identity coherence over time.

### Retrieval and Search

Both journal stores support **semantic search** via TF-IDF term vectors built from the entry prose, reflection prompts, and tags. Searches can also filter by:
- Trigger type (find all REM reflections, or all innovation-flagged entries)
- Review status (find unreviewed entries that need revisiting)
- Turn range (find entries from a specific part of a conversation)
- Linked entry graph traversal (follow cross-links to explore related reflections)

The identity journal additionally supports filtering by entry type (REGULAR, REFLECTION, COMMENT) and threading via parent entry IDs.

---

## FAQ

**Q: How big can memory get?**
In the current implementation, memory is in-memory (Python dicts). The architecture is designed for backend swaps — SQLite, vector databases, or distributed stores can replace the in-memory implementation without changing the interfaces. Cold storage and purging prevent unbounded growth.

**Q: What happens if the system contradicts something it said 10 sessions ago?**
The Memory Contrast system can catch this. When Phase 1 runs, it queries both MTMM (recent sessions) and LTMM (cross-session) for relevant matches. If a high-similarity match shows divergent content, the Logic domain's internal consistency submodule flags it. The contradiction detection engine (E1) can then analyze the specifics.

**Q: Can memory be manually edited?**
Core memories (identity/core) can only be updated through the M2 (Peer Review) pipeline, which requires explicit peer-review validation. This prevents accidental or unvetted changes to the system's core identity. Pending updates sit in the queue until approved or rejected.

**Q: Why are some thoughts written directly to LTMM?**
Held Thinking Blocks bypass the normal STMM → MTMM → LTMM path because they represent emotionally interrupted thoughts — reasoning that was abandoned due to emotional intensity (emotion > 0.6 or identity-relevant emotion). These are considered too significant to risk being lost in compression. They're written directly to LTMM with full context preserved.

**Q: How does dream processing interact with memory?**
Dream mode pulls from the Unsolved Buffer, specifically targeting questions that have stagnated (attempted 5+ times without resolution). The Dream pipeline runs with elevated CB1 (creative flexibility) and relaxed evaluation thresholds, allowing the system to make lateral connections between seemingly unrelated memories. If a novel connection is found, it can resolve a previously stuck question. The Dream pipeline can access cold storage, potentially finding relevance in memories that had faded from active use.

**Q: What's the difference between Knowledge Maps and the AtomSpace (E9)?**
Knowledge Maps are human-readable semantic graphs — organized by subject, with labeled nodes (concept, principle, fact, open question) and typed edges (supports, contradicts, etc.). They're designed for comprehension and overview. AtomSpace is a computational substrate — a typed hypergraph with truth values and attention values, designed for machine inference (PLN, ECAN). They can be cross-linked but serve different purposes: one for understanding, one for computation.

**Q: How does the system decide what's "identity relevant"?**
An entry is marked identity-relevant during consolidation when it carries critical flags (IDENTITY, PARADOX) or when it's associated with identity-relevant emotions (ashamed, guilty, proud, rejected, valued, etc.). Identity-relevant entries receive special protection: they're always stored at VERBATIM granularity, maintain minimum relevance scores, and are never demoted to cold storage or purged.

**Q: What's the Journal system for?**
The Journal system provides structured reflective writing across two tiers — a general **Cognitive Journal** for all reflective entries (regular processing, sleep, learning) and a specialized **Identity Journal** for moments involving identity-relevant emotions (shame, pride, rejection, belonging, etc.). Both are created by the JournalTool cognitool through a 3-phase pipeline (annotate with E18/E19/E20 → generate prose via LLM → auto-tag). Entries capture full cognitive snapshots (emotions, neurochemistry, reward scores, tone) and are cross-linked to related past entries for retrieval. See the dedicated Journal System section above for full details.

There is also a **Notebook Store** in the knowledge namespace for academic journaling about specific learning domains — these are simpler records focused on what was learned rather than reflective self-examination.

**Q: What's the difference between Held Thinking Blocks and Journal entries?**
Held Thinking Blocks are emotion-interrupted thought *fragments* — reasoning that was abandoned mid-stream because an emotion exceeded 0.6 intensity or an identity-relevant emotion was detected. They're raw, unprocessed, and written directly to LTMM to prevent compression loss. Journal entries are structured *reflections* — the system deliberately pauses to write a 150-400 word monologue about its cognitive activity, with engine annotations and open questions. Held blocks are captured involuntarily (something interrupted the thought); journal entries are created systematically (triggered at regular intervals or by significant events).
