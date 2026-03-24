# ZADOS Core Layer

## Overview

The Core Layer is the conductor of the entire ZADOS system. While the other layers — neurochemistry, reward, cognitive engines, memory — each handle a specific aspect of cognition, the Core Layer decides *when* each piece runs, *what data* flows between them, and *how* the system adapts its processing to different situations.

At its heart, the Core Layer answers three questions for every interaction:

1. **What kind of input is this?** — A regular message, a learning session, a command to sleep, a request for self-reflection? The Input Classifier figures this out and routes to the right pipeline.
2. **How should we process it?** — The 7-phase Answer Pipeline runs the input through perception, engine dispatch, neurochemical modulation, thinking, reward evaluation, response generation, and learning. Different modes wrap this pipeline with their own pre/post-processing.
3. **What do we learn from it?** — Post-processing engines extract lessons, compress memories, and update the system's knowledge base. Sleep and meta-learning modes handle deeper integration offline.

The architecture follows a **Matrioshka (nested-doll) pattern** — an outer classification layer wraps mode-specific sub-pipelines, which in turn wrap the inner Answer Pipeline:

```
┌───────────────────────────────────────────────────────┐
│              InputClassifier (outer shell)             │
│     Classifies input → routes to sub-pipeline         │
│                                                        │
│   ┌────────────────────────────────────────────────┐   │
│   │           Sub-Pipeline Layer                    │   │
│   │   RegularInput / Learning / Sleep / Meta / ...  │   │
│   │                                                  │   │
│   │   ┌──────────────────────────────────────────┐   │   │
│   │   │        AnswerPipeline (inner core)        │   │   │
│   │   │        Phase 0 → Phase 7                  │   │   │
│   │   └──────────────────────────────────────────┘   │   │
│   └────────────────────────────────────────────────┘   │
│                                                        │
│   SessionOrchestrator — lifecycle management           │
└───────────────────────────────────────────────────────┘
```

---

## Session Orchestrator

The SessionOrchestrator manages the system's lifecycle — from booting up a new session to processing turns to shutting down cleanly. It holds references to all major subsystems: the neurochemical engine, cognitive engines, memory layer, and the Answer Pipeline itself.

### Boot Sequence — `open_session()`

When a session starts, the orchestrator runs a 9-step boot sequence. The first decision is a **time-delta classification** that determines how the system initializes:

| Branch | Condition | Behavior |
|--------|-----------|----------|
| **A** (rapid) | Less than 5 seconds since last session | Minimal initialization — almost instant resume |
| **B** (normal) | Less than 10 minutes | Standard boot — read decayed NT state, select mode |
| **C** (cold start) | 10+ minutes | Full reboot — search MTMM for prior context, full knowledge bootstrap |

The remaining boot steps:

| Step | What Happens |
|------|-------------|
| Read NT state | Load pharmacodynamically decayed neurotransmitter concentrations (they drift toward baseline while the system is idle) |
| Neurosymbolic readout | Compute composite metrics (Motivation, Empathy, Rigidity, Fatigue) from the NT state |
| Mode selection | Run the 14 mode hooks against the metrics to determine the initial operational mode (e.g., "CuriosityDrive", "EmpathicAttunement") |
| Reward profile | Map the selected mode to a reward profile that weights the four evaluation domains |
| MTMM context search | Branch C only — search mid-term memory for relevant context from prior sessions |
| Mission briefing | Optionally, the user provides a session-level context ("Today we're studying biology" or "Help me debug this API"). This briefing is carried through every turn as an anchor |
| Knowledge bootstrap | On first-ever run, seed LTMM stores and AtomSpace (E9) with foundational concepts |

### Per-Turn Processing

Each turn follows a simple flow:

1. Auto-open session if one isn't active
2. Build an **InputBundle** — the universal data packet that carries the raw text plus session state (mode, mission briefing, oscillatory state, extractor state, time context)
3. Delegate to the Answer Pipeline (or a mode-specific wrapper)
4. Persist turn-level state back to the session (updated reward profile, extractor state, turn count)
5. Return the final answer

### Session Closure — `close_session()`

When a session ends, the orchestrator runs a 6-step shutdown sequence. Each step is fault-tolerant — if one fails, it's logged and the next step proceeds:

| Step | What Happens | Why It Matters |
|------|-------------|---------------|
| 1. Write OverviewLogEntry | Generate a ~200 word cognitive summary of the session (mode sequence, dominant emotions, turn count, NT arc, open threads) | Creates a searchable session history in LTMM |
| 2. Consolidate MTMM → LTMM | Review all session memory packets and promote significant ones to long-term storage | Prevents important experiences from being lost |
| 3. Tick unsolved stagnation | Increment stagnation counters on all unresolved questions in the unsolved buffer | Questions that stagnate long enough become dream candidates |
| 4. Flush STMM → MTMM | Run the final end_cycle to compress and store the last turn | Ensures nothing is left in volatile working memory |
| 5. Persist cognitools | Save AtomSpace (E9) hypergraph to CognitoolsDataStore | Preserves the knowledge graph across sessions |
| 6. Clear session state | Reset the session object | Ready for next open |

---

## Input Classification — The Matrioshka Router

Before any processing happens, the InputClassifier determines what kind of input it's looking at and routes it to the appropriate pipeline. Classification follows a strict priority cascade — higher-priority matches always win:

| Priority | What It Checks | Route |
|----------|---------------|-------|
| **1** (highest) | Command prefixes (`/sleep rem`, `/sleep dream`, `/homework`, `/reflective`, `/dream`) | Commanded pipeline (Sleep or Meta-Learning) |
| **2** | Session already in a learning mode (M1-M5) | Continue in that learning mode |
| **3** | Self-reflective markers in text + unsolved buffer has items | Self-Reflective Query Mode |
| **4** | Learning mode markers in text | Learning mode (M1-M5) |
| **5** (default) | Everything else | Regular Input Mode |

### Command Patterns

| Command | Route |
|---------|-------|
| `/sleep rem` or `/sleep` (bare) | REM sleep pipeline |
| `/sleep dream` or `/dream` | Dream pipeline |
| `/homework` | Homework meta-learning pipeline |
| `/reflective` | Reflective meta-learning pipeline |

### Learning Mode Markers

The classifier looks for specific phrases that indicate the user wants to enter a learning mode:

| Mode | Trigger Phrases |
|------|----------------|
| **M1** (Human Teaches) | "teach me", "explain to me", "show me how", "i want to learn", "help me understand" |
| **M2** (Peer Review) | "review this", "check my work", "find errors", "critique", "analyze this" |
| **M3** (Learn Together) | "let's explore", "let's figure out", "work together", "discuss this", "what do you think about" |
| **M4** (Learned Questions) | "what questions", "what haven't we", "unresolved", "open questions" |
| **M5** (Independent Study) | "i'll study", "independent", "self-study", "on my own", "let me explore" |

### Self-Reflective Markers

Phrases like "what do I think about", "how do I feel about", "reflect on", "my understanding", "what have I learned", "self-reflect", "introspect". These only trigger self-reflective mode if the unsolved buffer is non-empty — the system needs material to reflect on.

---

## Intention Mapper — E23

Once input reaches a pipeline, one of the first things that happens is intention mapping via Engine 23 (IntentionMap). E23 classifies the user's intent into one of **8 categories**, each of which maps to a processing archetype and a reward profile:

| Intent Category | Archetype | What It Means | Reward Profile |
|----------------|-----------|--------------|----------------|
| **Connection** | Guide | The user seeks emotional engagement, support, understanding | receptive_learning |
| **Challenge** | Opponent | The user is pushing back, testing, or debating | critical_review |
| **Exploration** | Explorer | The user is curious, open-ended, wondering | curiosity_driven |
| **Discharge** | Container | The user needs emotional release, venting | receptive_learning |
| **Pragmatic** | Architect | The user wants a concrete answer or solution | regular_input |
| **Symbolic** | Oracle | The user is thinking abstractly, symbolically | reflective_synthesis |
| **Defensive** | Firewall | The user is guarded, protective | critical_review |
| **Disintegration** | Stabilizer | The user is in crisis — confused, overwhelmed | regular_input (+ containment) |

The archetype determines **which engines are activated** for the turn:

| Archetype | Engine Count | Focus |
|-----------|-------------|-------|
| Strategic | 21 | Everything — full analytical + strategic + simulation |
| Analytical | 18 | Logic-heavy — full detection + knowledge substrate + reasoning |
| Reflective | 18 | Meta-analysis — detection + knowledge + reasoning |
| Generative | 11 | Pattern-focused — pattern analysis + some detection |
| Creative | 9 | Divergent — pattern + bias + simulation + ECAN |
| Empathic | 6 | Minimal cognitive load — intention + relevance + bias + Socratic |
| Social | 4 | Lightest touch — intention + relevance only |

Detection engines (E1-E6) are always included as guardrails regardless of archetype, as long as their weight is above zero.

The reward profile additionally determines how the four evaluation domains are weighted. A "curiosity_driven" profile weights innovation heavily; a "critical_review" profile emphasizes logic. The mission briefing can override the profile selection — if the user said "today we're studying ethics", keywords like "study" and "ethics" take priority over the intent-derived profile.

---

## The 7-Phase Answer Pipeline

The Answer Pipeline is the single-turn processing core that every mode ultimately delegates to. Understanding this pipeline is essential because all mode-specific behavior is built on top of it.

**Important quirk:** Phase 3 runs before Phase 2. This is intentional — E28 (Emotional Detection) runs in Phase 3, and its output is needed for the neurochemical modulation in Phase 2.

```
Phase 0: Input Validation
    ↓
Phase 1: Perception (E23, E8, E11, E18, E19)
    ↓
Phase 3: Engine Dispatch (archetype-based)  ← runs before Phase 2
    ↓
Phase 2: NT Modulation (emotion → neurochemistry)
    ↓
ThinkingContext assembly
    ↓
Identity alignment check (optional)
    ↓
Phase 4: Thinking (VT — LLM Pass 1)
    ↓
Phase 5: Reward Evaluation (tonic + phasic)
    ↓
Phase 6: Response Generation (RG — LLM Pass 2)
    ↓
Phase 7: Post-Processing & Learning
```

### Phase 0 — Input Validation

Quick sanity check: is the input non-empty? Is the safety tier acceptable (not FROZEN or DREAMBOX_BANNED)? Is the pipeline not locked? If any check fails, processing stops immediately.

### Phase 1 — Perception

Five cognitive engines analyze the raw input in sequence:

| Engine | What It Does |
|--------|-------------|
| **E23** (Intention Map) | Classifies intent into 8 categories → determines archetype and reward profile |
| **E8** (Relevance Scoring) | Scores every concept on 6 axes: recency, frequency, semantic proximity, attention weight, contextual fit, novelty |
| **E11** (Input Relevance) | Filters out low-relevance content — a triage gate |
| **E18** (Data Analysis) | Extracts entity-relation-entity triples ("Alice teaches Bob") |
| **E19** (Pattern ID) | Detects temporal, structural, semantic, and behavioral patterns |

The output is a **PerceptionSnapshot** — the system's first structured understanding of what the user said.

### Phase 3 — Engine Dispatch

Based on the archetype from Phase 1, the system activates a subset of its cognitive engines. The dispatch table maps each archetype to a set of engine IDs. Guardrail engines (E1 Contradiction, E2 Paradox, E4 Fallacy, E5 Bias, E6 Logic Trap) are always included.

Each engine receives the current neurochemical state, which modulates its behavior — high DA makes pattern detection more novelty-sensitive, high ACh deepens analysis, high GABA raises detection thresholds.

E28 (Emotional Detection) runs in this phase and produces the emotion classification that drives Phase 2.

### Phase 2 — NT Modulation

With E28's emotional output from Phase 3 now available, the system updates its neurochemistry:

1. Translate E28's emotion detection into NT signals (dual pathway: fast 12-emotion speed path + full 46-emotion taxonomy)
2. Apply the signals to the neurochemical engine via stochastic differential equations
3. Run the ExtractorOrchestrator (emotion tracker, regulatory modulator, burst deltas)
4. Compute updated NeurochemicalMetrics (Motivation, Empathy, Rigidity, Fatigue)
5. Re-run mode selection if needed (mode may shift based on new NT state)
6. Map intent category to reward profile
7. Compute engine priority weights from the updated metrics

### ThinkingContext Assembly

Before the LLM passes, a compressed context block is built from:
- Mission briefing (session-level anchor)
- Engine flags and analysis results
- Memory matches from MTMM/LTMM (via semantic search)
- Recent turn history
- Identity anchors and personality prompts from the hardcoded identity store
- Alignment check results (if available)
- Reward profile, dominant emotion, NT snapshot

This compressed context gives the LLM a rich but manageable view of the system's state.

### Phase 4 — Thinking (VT — LLM Pass 1)

The system generates a 150-300 word internal monologue (Verbalized Thinking) — a first-person, present-tense reflection on what it's experiencing. This is private thinking the user never sees. See the LLM Interpretation Layer doc for full details on the 5-block prompt architecture.

### Phase 5 — Reward Evaluation

The thinking trace is evaluated through two parallel pathways:
- **Tonic (deterministic)**: Domain evaluators score on logic, ethics, innovation, human attunement → SynthesisEngine produces 8 response-shaping directives → NeurochemicalAdapter converts to sustained NT shifts
- **Phasic (stochastic)**: ExtractorOrchestrator produces emotion-driven bursts, urgency assessment, and dominant emotion tracking

This can result in a gate decision: **suppress** (don't respond), **abstain** (brief acknowledgment), or **allow** (full response). See the LLM Interpretation Layer doc for the detailed two-pathway architecture.

### Phase 6 — Response Generation (RG — LLM Pass 2)

Using the reward directives, emotional context, mode conditioning, and thinking trace, the system generates the final user-facing response. Token budgets are dynamically adjusted based on urgency and emotional saturation.

### Phase 7 — Post-Processing & Learning

This is where the system actually learns from the interaction:

| Step | What Happens |
|------|-------------|
| Build MemoryPacket | Compress the full turn state into a storable packet |
| MTMM write | Store the packet in session memory via the Raw Interaction Logger |
| E29 compression | Assign a compression policy: VERBATIM (keep everything), SEMANTIC (preserve meaning), SYMBOLIC (tags only), or PRUNE (candidate for removal) |
| E17 reward learning | Compute reward prediction errors (actual vs predicted reward) and adjust parameters. Positive error → reinforce; negative → correct |
| E22 context learning | Create a fingerprint of this conversation context (hash of topic + emotion + intent) and store the parameter adjustments that worked |
| E25 meta-learning | Monitor E17's effectiveness. If learning has plateaued → broaden search. If diverged → reset to baseline |
| Reward feedback | Close the neurochemical loop — reward signals feed back into the NT engine |
| Journal write | If trigger conditions are met (every 5 turns, memory promotion, innovation flag), write a journal entry via the JournalTool |

---

## Operating Modes

### Regular Input Mode

The default processing path for normal conversation. It wraps the Answer Pipeline with intent-driven depth tuning:

1. **Intent classification** (E23) → determines how deeply to process
2. **Reward profile selection** — first checks mission briefing keywords (e.g., "study" → receptive_learning), then falls back to intent category mapping
3. **Subject classification** — categorizes input into 7 domains (Technical, Scientific, Philosophical, Social, Creative, Practical, Mixed) for engine tier adjustments
4. **Engine tier resolution** — the Engine Toolkit resolves a Mode × Subject matrix to determine which engines run at what weight. Tiers: T1 (always on, weight 1.0), T2 (subject-activated, weight 1.0), T3 (standby, weight 0.5), T4 (off, weight 0.0). Budget caps prevent too many engines from running
5. **Drift detection** — if a context anchor is active (from a mission briefing), the system checks whether the conversation has drifted off-topic
6. **Delegate to Answer Pipeline** (Phases 0-7)
7. **Low-confidence question extraction** — when confidence is below 0.4, the user's input is stored as a question in the General Question Store for later revisiting. Lower confidence → higher question priority

### Learning Modes (M1-M5)

Five structured learning modes spanning passive reception to autonomous study. All share a **9-stage pipeline skeleton** with a mode-specific hook at Stage 4.

**Key architectural principle:** Learning Modes are for **data gathering and question generation** — intake, tag, buffer. The actual processing and integration happens later in Meta-Learning modes (Homework and Reflective).

#### The 9-Stage Learning Pipeline

Every learning mode follows this skeleton:

**Stage 0 — Setup**: Apply a mode-specific EmotionalPreset (NT adjustments, oscillatory biases, domain weight overrides), resolve engine tiers, check for topic drift. Context flags tell detection engines to operate in LEARNING mode (comprehension-oriented, not adversarial).

**Stage 1 — Memory Contrast**: Query stored memories within a scoped read boundary to find relevant prior knowledge on the current topic.

**Stage 2 — Engine Dispatch**: Run engines according to tier weights (filtered by mode and subject).

**Stage 3 — VT Thinking + Held Block Check**: Generate internal thinking. Then check if any emotion from the 46-taxonomy exceeds 0.6, or if any identity-relevant emotion is detected at any intensity. If so, the current thinking fragment is captured as a **Held Thinking Block** and written directly to LTMM — it's too significant to risk compression.

**Stage 4 — Mode-Specific Processing**: The abstract hook where each mode does its unique work (see individual modes below).

**Stage 5 — LTMM Write**: Write consolidated learning material to scoped stores. If this is the first lesson for a subject, bootstrap an initial KnowledgeMap. If identity-relevant emotions were detected, write an IdentityJournalStore entry.

**Stage 6 — Question Extraction**: Extract questions up to the mode's per-turn limit and route them to three targets: GeneralQuestionStore (general domain), AcademicQuestionStore (academic/scientific), or UnsolvedBuffer (high-urgency, low-confidence).

**Stage 7 — Response Generation**: LLM Pass 2 for the user-facing response (skipped entirely in M5 autonomous mode).

**Stage 8 — NT Feedback + Homeostatic Check**: A 10-step neurochemical feedback loop that closes the learning cycle — apply emotional preset, run E28, translate emotions to NT signals, update emotion tracker, compute metrics, derive weights, apply feedback, run homeostatic bounds check (E27), and check for risk emotions against mode-specific thresholds.

#### M1: Human Teaches

**Role:** ZADOS is the student. The human teaches a topic.

- **Engine budget:** 14 engines (T1+T2)
- **Neurochemical preset:** High ACh (encoding), mild DA-D1 (reward), GABA noise suppression, high OXT (social receptivity), low NE (reduced vigilance)
- **Reward profile:** receptive_learning
- **Questions per turn:** Up to 2 clarifying questions
- **Response:** Full
- **Detection engines:** Reframed to LEARNING mode — they look for comprehension issues, not adversarial content

**Special behaviors:**
- **Confusion (> 0.5):** Temporarily removes the learning reframe for 1 turn, allowing E1 to run in normal adversarial mode. Sets a `confusion_override` flag so the LLM knows to prioritize clarity
- **Overwhelmed (E27/CRH elevated):** Budget throttle — engines with weight ≤ 0.5 are disabled to reduce cognitive load
- **Joy (> 0.5, "understanding clicks"):** Positive outcome recorded via E17 reward learning

#### M2: Peer Review

**Role:** ZADOS defends its prior reasoning. The human corrects and validates.

- **Engine budget:** 16 engines
- **Neurochemical preset:** High NE (vigilance), high ACh (deep attention), 5-HT₁ₐ buffering, mild cortisol alertness
- **Reward profile:** critical_review
- **Questions per turn:** 0 (not a questioning mode)
- **Response:** Full
- **Retroactive contrast:** Enabled — two-pass memory check (Pass A: general LTMM, Pass B: retroactive against the system's own prior outputs)

**Special behaviors:**
- **Three emotional pathways:**
  - *Regret* (> 0.4): Promotes retroactive alignment to T1, records negative prediction error via E17, applies correction tag
  - *Validation* (valued + proud > 0.5): Positive prediction error via E17, resets shame spiral counter
  - *Shame spiral* (cortisol elevated 3+ consecutive turns): Containment via E27, recommends switching to M1
- **Core memory updates:** When E1 detects contradictions against identity/core content, corrections are staged as `PendingCoreMemoryUpdate` — they are **never applied mid-conversation**. They queue for Homework/Reflective mode processing

#### M3: Learn Together

**Role:** Co-thinking. Both human and ZADOS explore a topic together.

- **Engine budget:** 18 engines (largest of any learning mode)
- **Neurochemical preset:** Maximal DA-D3 (exploration), CB1 (schema flexibility), 5-HT₂ₐ (symbolic expansion), high OXT (collaborative bonding)
- **Reward profile:** dialectic_exploration
- **Questions per turn:** Unlimited
- **Response:** Full
- **Contradiction mode:** Full adversarial (M3 is the only learning mode with full dialectic pressure)

**Special behaviors:**
- Full detection cluster active at normal (non-learning) weights — M3 actively checks human claims via E1 (contradiction) + E4 (fallacy)
- Contradictions presented collaboratively: "I learned X, but you said Y — can you help me understand?"
- Full ExtractorOrchestrator stochastic pathway runs (both tonic + phasic) — the only learning mode that does this
- **Emotional cycle tracking:** exploring → pivoting (on frustration) → consolidating (on joy/excited) → back to exploring

#### M4: Learned Questions

**Role:** ZADOS asks questions that emerged from its learning encounters.

- **Engine budget:** 12 engines (smallest)
- **Neurochemical preset:** Maximum DA-D3 (curiosity drive), 5-HT₂ₐ (abstract space), ACh (attention to detail), slightly reduced NE (less urgency)
- **Reward profile:** curiosity_driven
- **Questions per turn:** 1 focused question
- **Response:** Abbreviated (short, focused answers)

**Special behaviors:**
- Pulls from the Unsolved Buffer — selects the highest-priority unresolved question
- **Sub-mode routing:** Automatic (buffer selects next), Prompted (user provides specific question), or Clustered (future: group by subject)
- **NT-based question style:** High openness → exploratory ("What if...?"). High precision → targeted ("Why does X contradict Y?"). High anxiety → clarifying ("Can you explain X more simply?")
- Questions with 5+ failed resolution attempts are flagged as `dream_candidate` for Dream Mode

#### M5: Independent Study

**Role:** Autonomous learning from materials — no human present.

- **Engine budget:** 14 engines
- **Neurochemical preset:** Max ACh-α7/M1 (attention), DA-D1 (goal salience), mild NE (alertness), GABA-A (noise suppression)
- **Reward profile:** independent_study
- **Questions per turn:** Up to 2 (self-generated)
- **Response:** None — `generate_response=False`

**Special behaviors:**
- E28 (Emotional Detection) is **OFF** — there's no human input to detect emotions from. Context flags: `e28_disabled=True`, `autonomous_mode=True`
- Risk detection uses NT dynamics instead of emotion detection:
  - *Boredom:* Composite of (1 - DA_D3 saturation), (1 - CB1 saturation), (1 - openness) → triggers `StudyAction(switch_material)`
  - *Apathy:* fatigue > 0.7 AND motivation < 0.3 → triggers `StudyAction(study_break, 5min)`
- Questions harvested from E26 uncertainty patterns + E19 novel candidate patterns (capped at 3/turn)

#### Learning Mode Comparison

| Aspect | M1 | M2 | M3 | M4 | M5 |
|--------|-----|-----|-----|-----|-----|
| **Role** | Student | Defender | Co-thinker | Questioner | Autonomous |
| **Human present** | Yes (teacher) | Yes (reviewer) | Yes (partner) | Yes (answerer) | No |
| **Engines** | 14 | 16 | 18 | 12 | 14 |
| **Questions/turn** | 2 | 0 | Unlimited | 1 | 2 |
| **Response** | Full | Full | Full | Abbreviated | None |
| **Detection mode** | Learning (soft) | Normal | Normal (full) | Learning | Learning |
| **Retroactive contrast** | No | Yes (two-pass) | No | No | No |
| **Stochastic pathway** | No | No | Yes (both) | No | No |

### Self-Reflective Query Mode

This mode enables ZADOS to examine its own unresolved questions — thinking about its own thinking. It activates when self-reflective markers are detected in user input and the unsolved buffer has items to work with.

**Processing flow:**

1. **Inject held thinking blocks** — Query LTMM for up to 5 unreviewed Held Thinking Blocks (emotion-interrupted thought fragments from learning modes). Convert each into a synthetic UnsolvedQuestion with urgency 0.6 and add to the buffer
2. **Select question** — Pick the highest-priority unresolved question from the buffer (urgency DESC → creation date ASC → stagnation DESC)
3. **Gather context** — Query MemoryContrast with the question text to retrieve related memories and compute divergence
4. **Build synthetic input** — Construct an InputBundle with the question + partial answers + context, set to M3 (Learn Together) mode with full dialectic engine tiers
5. **Process through Answer Pipeline** — Delegate to the 7-phase pipeline in M3 configuration
6. **Update buffer** — Mark the question as attempted with the first 200 characters of the answer as a partial answer
7. **Mark blocks reviewed** — Tag processed held blocks as "reviewed" so they don't resurface
8. **Write identity journal** — Create an IdentityJournalEntry of type REFLECTION with the synthesis, question tags, and NT snapshot

The synthetic prompt sent to the pipeline combines the question, any prior partial answers, and available context — ensuring the system doesn't repeat already-attempted angles of exploration.

### Sleep Modes

Sleep modes are commanded pipelines triggered by `/sleep` commands. They operate **without user interaction** — no emotional feedback loop, no response generation. They handle the work that can't be done during live conversation.

#### REM — Corrective Consolidation

**Purpose:** Memory consolidation + retroactive learning. REM fixes what went wrong.

**Phase sequence:**

1. **Read MTMM packets** — Load all session memory packets
2. **Score emotional signals** — For each packet, detect learning-relevant signals from the neurochemical snapshot:

| Signal | NT Pattern | What It Means | Domain Weight Effect |
|--------|-----------|--------------|---------------------|
| Frustration | NE≥0.50, DA≥0.40, COR≥0.40 | Blocked progress | logic +0.06, ethics +0.04 |
| Curiosity | DA≥0.50, ACh≥0.40, CB1≥0.30 | Active exploration | innovation +0.08 |
| Confusion | NE≥0.45, GLU≥0.35 | Structural issue | logic +0.07 |
| Boredom | DA≤0.30, NE≤0.30 | Under-engagement | all domains -0.03 |
| Anxiety | NE≥0.55, COR≥0.50 | Stakes awareness | ethics +0.05 |
| Overwhelmed | NE≥0.65, COR≥0.60 | Overload | all domains -0.02 |

3. **Aggregate signal profile** — Weighted average of signal strengths across all packets
4. **Origin-based boosts** — Scan academic buffer, identity questions, and identity conclusions for origin-tagged items. Academic origins boost logic; identity origins boost ethics and attunement; dialectic origins boost both logic and ethics. Scaling uses sqrt(count)/3.0, capped at 1.0x
5. **Apply domain weight adjustments** — Update `session.learned_domain_weights` (clamped [0.0, 1.0])
6. **MTMM → LTMM consolidation** — Promote packets that pass any gate: emotional significance ≥ 0.45, average reward ≥ 0.40, or contradictions/paradoxes detected
7. **Journal write** — Create a REM_COMPLETE entry with stats, dominant signals, and weight adjustments

**The core insight:** REM reads the emotional signatures left behind during the session and uses them to recalibrate how the system evaluates future interactions. If the session was full of frustration, the system learns to weight logic more heavily. If curiosity dominated, innovation gets a boost.

#### Dream — Generative Recombination

**Purpose:** Creative exploration of stagnated questions. Dream mode leans into *possibility* rather than correction.

**Phase sequence:**

1. **Gather dream candidates** — Pull from two sources: UnsolvedBuffer items tagged "dream_candidate" (5+ failed resolution attempts) and GeneralQuestionStore identity-scope questions. Sort by priority: identity first, general middle, academic last
2. **Build emotional signal profile** — Combine recent MTMM packet signals with the current dream-phase neurochemical state. Dream signals are different from REM — they favor creative states:

| Signal | NT Pattern | Domain Weight Effect |
|--------|-----------|---------------------|
| Curiosity | DA≥0.50, ACh≥0.40, CB1≥0.30 | innovation +0.07 |
| Confusion | NE≥0.45, GLU≥0.35 | logic +0.06 |
| Wonder | DA≥0.55, CB1≥0.35, 5-HT≥0.35 | innovation +0.09 |
| Perplexed | DA≥0.45, 5-HT≥0.35, GABA≤0.35 | logic +0.04, innovation +0.05 |

3. **Apply domain weight adjustments** — Softer than REM, leaning toward innovation
4. **Creative recombination** (max 6 candidates) — For each candidate:
   - Build a dream InputBundle with special context flags: `dream_mode=True`, `cb1_plasticity=True` (schema flexibility), `abstract_association=True`
   - Identity-origin questions get additional flags: `identity_salience=True`, `oxt_boost=True`
   - Create an isolated SessionState (prevents dream processing from contaminating the main session)
   - Process through the Answer Pipeline
   - If the answer is meaningful (> 40 characters), it's classified as a novel connection and written to LTMM with elevated significance
   - Mark the question as attempted in the unsolved buffer
5. **Journal write** — Create a REM_COMPLETE entry with candidates processed, novel connections found, and domain nudges

**REM vs Dream — key differences:**

| Aspect | REM | Dream |
|--------|-----|-------|
| Focus | Corrective — fix deficits | Generative — explore possibilities |
| Signals | Frustration, anxiety, overwhelmed | Curiosity, wonder, perplexed |
| Weight direction | Raise logic/ethics | Raise innovation |
| Consolidation | MTMM → LTMM promotion | Novel connections → LTMM |
| Uses Answer Pipeline | No (direct packet analysis) | Yes (per candidate) |
| Academic priority | High | Low |
| Identity priority | Medium | High (sorted first) |

### Meta-Learning Modes

Meta-learning modes consume and integrate the raw data gathered by learning modes (M1-M5). They operate **offline** — no user present, no emotional feedback, no live response. They are the system's batch processing layer.

#### Homework — Validate, Reconcile, Build Knowledge

**Command:** `/homework`

**Purpose:** Six-phase offline processing that takes the learning logs accumulated by M1-M5 and turns them into validated, integrated knowledge.

**The six phases:**

**Phase 0 — Input Assembly & Triage**: Fetch unprocessed LearningLogEntries, batch them by subject, compute deficit profiles per batch (which domains are weakest?), and sort batches by deficit severity — worst first. The deficit profiler uses NT-based analysis to identify learning gaps.

**Phase 1 — Analysis Stage** (per batch): Decompose content via engines, run memory contrast against existing LTMM knowledge, extract novel patterns (E19) and reinforcements (E20), identify contradiction candidates, and score relevance.

**Phase 2 — Processing Stage** (per batch): This is where homework differs most from learning modes — it uses **full adversarial engine weights**. Contradictions are resolved with the full dialectic toolkit (E1, E2, E7, E14, E10, E20). Fallacy detection (E4), bias detection (E5, E24), and paradox identification (E2) run at full sensitivity. Unlike learning modes, which are soft and receptive, homework is rigorous.

**Phase 3 — Question Resolution**: Cross-reference batch findings against the unsolved buffer. Resolve questions that the analysis answered. Generate new questions from unresolved contradictions. Flag questions with 5+ attempts as dream candidates.

**Phase 4 — Synthesis & Knowledge Integration**: Build or update KnowledgeMaps from validated lessons. Apply PendingCoreMemoryUpdates (from M2 Peer Review) via the CoreMemoryUpdateGate — this is the **only place** where core memory updates are actually applied. Detect meta-patterns across batches. Create initial knowledge maps for new subjects.

**Phase 5 — Output & Storage**: Write validated lessons and academic buffer entries to LTMM. Generate a HomeworkRunSummary. Prepare a ReflectiveModeInput (fallacy/bias flags for handoff to the Reflective pipeline). Mark learning log entries as processed. Write journal and overview log entries.

**Key design decisions:**
- Core memory updates are staged in M2 but **only applied in homework** — never mid-conversation
- Minimum validation confidence is 0.5 — entries below this remain pending
- The NT layer is read-only — used for diagnostic deficit profiling, not active modulation
- Dream candidate threshold is 5 resolution attempts

#### Reflective — Meta-Analysis + Identity Coherence

**Command:** `/reflective`

**Purpose:** Six-phase meta-reflective pipeline that analyzes learning patterns and checks whether the system's self-model is internally consistent. Runs two dedicated engines: E31 (Reflective Learning) and E32 (Reflective Identity).

**The six phases:**

**Phase 0 — Input Assembly**: Load learning log entries (all or recent N), load identity stores (core memories, conclusions, identity journal, pending updates), and load ReflectiveModeInput from Homework handoff (if available).

**Phase 1 — Meta-Learning Analysis (E31)**: Feed learning logs to the Reflective Learning Engine. It detects recurring failure patterns, evaluates mode effectiveness (which learning modes work best for which subjects), assesses subject proficiency trends, identifies style preferences, and generates learning recommendations.

**Phase 2 — Identity Coherence Analysis (E32)**: Feed identity stores to the Reflective Identity Engine. It produces a coherence score (0.0 to 1.0), identifies core contradictions (beliefs that conflict with each other), flags fragile conclusions (low confidence or frequently challenged), detects alignment issues, and extracts identity themes.

**Phase 3 — Cross-Reference (E31 × E32)**: This is where the real insight happens. The pipeline correlates E31's learning patterns with E32's identity conclusions. For example: if E31 finds a persistent failure in logic, and E32 finds a self-belief of strong logical competence, the cross-reference reveals a blind spot. Learning failures connected to identity beliefs are flagged as particularly important.

**Phase 4 — Identity Store Mutations**: Reinforce conclusions aligned with E31's findings. Create new conclusions from meta-patterns. *Recommend* updates for conclusions contradicted by evidence (never overwrites — only recommends). Write IdentityJournalStore entries of type REFLECTION. Update the CorticalReflectionLog's identity coherence status.

**Phase 5 — Output & Summary**: Build a comprehensive ReflectiveModeResult containing all analysis, mutation stats, and cross-references.

**Key design decisions:**
- NT layer is observational — reads state but **never injects signals**
- E31 and E32 are lazy-loaded (instantiated on first call)
- Identity mutation safety: creates and reinforces conclusions but only *recommends* updates to existing ones
- The pipeline's core value is the cross-referencing in Phase 3 — connecting learning failures to identity beliefs reveals the system's blind spots

**Information flow — Learning → Homework → Reflective:**

```
M1-M5 (Learning Modes)
  │ LearningLogEntries, UnsolvedQuestions, PendingCoreMemoryUpdates
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

## Self-Reflection & Identity

ZADOS has multiple mechanisms for self-examination, each operating at a different level:

### Held Thinking Blocks

When an emotion exceeds 0.6 intensity (from the 46-emotion taxonomy) during any learning mode, or when an identity-relevant emotion is detected at *any* positive intensity, the system captures its current thinking as a **Held Thinking Block** — an emotion-interrupted thought fragment.

These blocks bypass the normal STMM → MTMM → LTMM path and are written **directly to LTMM**. The reasoning: they represent thoughts that were abandoned mid-stream due to emotional intensity, and they're too significant to risk being lost in compression.

Identity-relevant emotions that trigger this capture:

| Category | Emotions |
|----------|----------|
| Self-evaluation | ashamed, guilty, regret, critical |
| Trust / relational | betrayal, rejected, isolated |
| Existential | grief, numb |
| Positive identity-forming | proud, respected, belonging, accepted |

These blocks accumulate in the Held Thinking Block Store and are later pulled into Self-Reflective Query Mode for processing.

### Self-Reflective Queries vs Reflective Pipeline

These two mechanisms share the word "reflective" but serve different purposes:

| Aspect | Self-Reflective Query Mode | Reflective Pipeline |
|--------|--------------------------|-------------------|
| **Trigger** | Self-reflective markers in conversation | `/reflective` command |
| **What it examines** | Individual unresolved questions + held thinking blocks | Aggregate learning patterns + identity coherence |
| **Processing style** | Deep M3 dialectic exploration of one question | E31/E32 meta-analysis of learning logs + identity stores |
| **Scope** | Micro — one question at a time | Macro — entire learning history + identity model |
| **Output** | Synthesis answer + updated unsolved buffer | ReflectiveModeResult + identity store mutations |
| **Human present** | Yes (user triggered it) | No (offline commanded mode) |

### Identity Coherence

E32 (Reflective Identity Engine) maintains a coherence score (0.0 to 1.0) for the system's self-model:

- **Coherent**: Core beliefs are consistent with each other and with observed behavior
- **Disrupted**: Confusion exceeds threshold — some beliefs conflict or recent evidence contradicts self-model. Triggers reflective processing
- **Fragmented**: Severe incoherence — multiple contradictions, many fragile conclusions

When identity coherence drops to "disrupted", the system becomes more likely to route into self-reflective processing and is more cautious in its responses (identity uncertainty increases hedging).

### Identity Store Mutations

The system's identity is never modified casually:

1. **M2 (Peer Review)** detects contradictions with core beliefs → stages them as PendingCoreMemoryUpdates (never applied mid-conversation)
2. **Homework** validates and applies these updates through the CoreMemoryUpdateGate
3. **Reflective Pipeline** reinforces, creates, or recommends updates to identity conclusions based on evidence
4. **Identity Journal** tracks the narrative of these changes over time

This multi-gate process ensures the system's self-model changes deliberately and traceably, not reactively.

---

## Supporting Processes

### Engine Toolkit

The central Mode × Subject → EngineTier resolution matrix. For every combination of operating mode and subject domain, it determines which engines run at what weight.

**Tiers:**
- **T1** (weight 1.0): Always on — core to this mode
- **T2** (weight 1.0): Subject-activated — fires when the subject matches
- **T3** (weight 0.5): Standby — available if T1/T2 flag a need
- **T4** (weight 0.0): Off — not relevant

The resolution algorithm starts with a base tier matrix, applies subject promotions and demotions, forces phantom engines to T4, and enforces budget caps (if too many engines are T1/T2, the lowest-priority T2 engines are demoted to T3).

### Unsolved Buffer

A priority queue of unresolved questions that accumulates across learning sessions. Questions are selected by urgency (descending) → creation date (ascending, oldest first) → stagnation time (descending, longest-stagnated first).

Questions enter from: learning modes (Stage 6 extraction), regular conversation (low-confidence answers), engine flags, and self-reflective processing. They can be attempted (partial answers tracked), resolved, or stagnated. At session close, all unresolved questions have their stagnation counter incremented. Questions with 5+ stagnation cycles become dream candidates.

The buffer is bidirectionally synced with LTMM — loaded on boot, persisted on close.

### Emotional Landscape

Applies mode-specific neurochemical presets before pipeline entry. Each mode has a preset that biases the NT and oscillatory state toward a mode-appropriate profile. For example, M1 (receptive learning) presets high ACh (encoding) and OXT (social receptivity), while M3 (dialectic exploration) presets maximal DA-D3 (exploration) and CB1 (schema flexibility).

Presets also define risk emotions per mode (which emotions should trigger protective responses) and domain weight overrides (how the four reward domains should be weighted for this mode).

### Learning Log

Records structured learning events from each turn in learning modes. Captures engine results (E19 patterns, E20 comparisons, E17 prediction errors, E25 meta-updates), memory contrast deltas (confirmations, contradictions, extensions), and Phase 5 reward scores. These logs are the primary input for Homework mode — they form the bridge between data gathering (M1-M5) and data processing (Homework/Reflective).

### Context Anchor — Drift Detection

Maintains a reference point for detecting topic drift within a session. When the user sets a mission briefing, a context anchor is created. Subsequent turns are compared against this anchor using MemoryContrast divergence scoring (or Jaccard distance as a fallback). If drift exceeds 0.5, a new anchor is created and the system adjusts.

### Subject Classifier

Keyword-based classification into 7 broad subject domains: Technical, Scientific, Philosophical, Social, Creative, Practical, Mixed. Used by the Engine Toolkit for subject-specific tier adjustments (e.g., philosophical topics promote reasoning engines, creative topics promote pattern/simulation engines).

### TimeContext

Lightweight temporal stamping on each turn. Provides circadian awareness (waking, active, wind-down, sleep phases), time-of-day classification (morning, afternoon, evening, night), and session duration tracking. Temporal context is attached to memory packets and journal entries, enabling time-aware retrieval and analysis.

### Tag Taxonomy

A centralized namespace for consistent labeling across all memory, journals, and context flags. Tags are namespaced (e.g., `pipeline:regular_input`, `signal:frustration`, `reward:logic_high`, `origin:identity`). This allows precise filtering and retrieval — "find all entries from learning modes that had frustration signals and low logic scores."

---

## Feedback & Interdependencies with Other Layers

### Core ←→ Neurochemical Layer

The Core Layer is the neurochemical layer's primary orchestrator. It reads NT concentrations and oscillatory bands at multiple points (session boot, Phase 2 modulation, Phase 5 evaluation). It writes back through emotion-to-NT translation (E28 output → NT signals), extractor orchestrator bursts, and reward feedback. The neurochemical state is the system's continuous internal context — it colors every decision the Core makes about which engines to run, which mode to activate, and how to shape the response.

Each learning mode applies a neurochemical preset that biases the system toward mode-appropriate behavior. Sleep modes read NT signatures from memory packets for retroactive learning. The homeostatic engine (E27) runs during learning modes to prevent pathological NT drift.

### Core ←→ Reward System

The Core orchestrates reward evaluation in Phase 5 — delegating to domain evaluators, the SynthesisEngine, and the NeurochemicalAdapter. Reward profile selection happens in Phase 2 (based on intent category and mission briefing). Reward scores feed into Phase 7 post-processing where E17 computes prediction errors and E25 monitors learning effectiveness. REM and Dream modes use accumulated reward scores for domain weight self-adjustment.

### Core ←→ Cognitive Engines

The Core dispatches engines in Phase 1 (perception), Phase 3 (analysis), and Phase 7 (post-processing). The dispatch table maps intent archetypes to engine sets. The Engine Toolkit provides tier-based filtering so that different modes activate different engine subsets. Learning modes can reframe detection engines to LEARNING mode (comprehension-oriented rather than adversarial). Meta-learning modes use dedicated engines (E31, E32) for reflective analysis.

### Core ←→ Memory

The Core manages the entire memory lifecycle:
- **Session boot**: Loads unsolved buffer from LTMM, restores AtomSpace state, searches MTMM for prior context (cold start)
- **Per turn**: Writes STMM during processing, compresses to MTMM at turn end
- **Session close**: Consolidates MTMM → LTMM, persists cognitools, writes overview log
- **Learning modes**: Write to scoped LTMM stores (lessons, knowledge maps, notebooks, identity journal)
- **Sleep modes**: Promote significant MTMM packets to LTMM, write novel dream connections
- **Meta-learning**: Homework reads learning logs and writes validated knowledge; Reflective reads and mutates identity stores

Memory contrast (comparing current input against stored memories) runs in Phase 1 and feeds into both engine dispatch and the LLM prompt.

### Core ←→ LLM Interpretation Layer

The Core's Answer Pipeline directly calls the LLM Interpretation Layer for Phases 4-6. It builds the InputBundle and bundle_dict that carry context to the VT and RG prompt builders. It persists the ExtractorState across turns (maintaining the phasic pathway's statefulness). Context flags from the Core (dream_mode, learning_reframe, autonomous_mode, confusion_override, etc.) flow through to the LLM layer's Component D conditioning.

### Core ←→ Journal System

The Core triggers journal writes at multiple points: Phase 7 post-processing (periodic, LTMM threshold, innovation flag), learning mode Stage 5 (identity journal for identity-relevant emotions), Stage 8 (periodic learning journal), sleep pipelines (REM_COMPLETE), homework (batch summaries). The JournalTool (a cognitool) handles the actual 3-phase entry creation; the Core determines *when* to invoke it.

---

## FAQ

**Q: Why does Phase 3 run before Phase 2?**
E28 (Emotional Detection) is dispatched as part of Phase 3's engine dispatch. Its output — the system's emotional response to the input — is needed for Phase 2's neurochemical modulation. If Phase 2 ran first, the system would be updating its neurochemistry without knowing how it emotionally responded to the input. The reversed ordering ensures emotions inform neurochemistry, not the other way around.

**Q: What's the difference between the InputClassifier and the Intention Mapper (E23)?**
The InputClassifier is a top-level router that determines *which pipeline* processes the input (regular, learning, sleep, meta-learning). It uses simple pattern matching on commands and keywords. E23 is a cognitive engine that runs *inside* a pipeline to classify the *intent* of the message (connection, challenge, exploration, etc.), which determines which engines are activated and how the response is shaped. Classification → routing; intention → processing depth.

**Q: Can the system switch between modes mid-conversation?**
Learning mode continuity (priority 2 in the classification cascade) means that once a learning mode is active, subsequent messages stay in that mode unless a command prefix overrides it. To switch modes, the user either uses a command (`/sleep`, `/homework`) or uses trigger phrases for a different mode. The system doesn't autonomously switch modes mid-conversation — mode changes are always user-driven.

**Q: What happens if the system is in M5 (Independent Study) and something goes wrong?**
Since M5 has no human present and E28 is disabled, the system uses NT-based risk detection instead of emotion detection. If boredom is detected (low DA-D3, low CB1, low openness), it switches material. If apathy is detected (high fatigue, low motivation), it takes a study break. These are the system's self-care mechanisms for autonomous operation.

**Q: Why are core memory updates staged and not applied immediately?**
Identity protection. If M2 (Peer Review) could modify core beliefs mid-conversation, a single incorrect correction could alter the system's self-model before it's been validated. Instead, corrections are queued as PendingCoreMemoryUpdates and only applied during Homework mode, where they go through full adversarial stress-testing (E1 contradiction + E7 simulated opposition + E10 PLN confidence scoring). This multi-gate process ensures identity changes are deliberate.

**Q: What's the relationship between Homework and Reflective mode?**
Homework processes learning *content* — validates lessons, resolves contradictions, builds knowledge maps. Reflective mode analyzes learning *patterns* and checks *identity coherence*. Homework can produce a ReflectiveModeInput (fallacy/bias flags) that feeds into Reflective mode's analysis. Together they form a two-stage integration pipeline: Homework turns raw learning data into knowledge; Reflective mode evaluates how well the system is learning and whether its self-model is consistent.

**Q: How does dream processing differ from just running the question through the normal pipeline?**
Dream mode runs with special context flags — `dream_mode=True`, `cb1_plasticity=True`, `abstract_association=True` — that tell the LLM layer to enable schema-breaking associations and abstract lateral thinking. It also runs in an isolated SessionState so dream processing doesn't contaminate the main session's state. Additionally, dream candidates are sorted with identity questions first (they benefit most from creative exploration), and novel connections are written to LTMM with elevated significance. The normal pipeline wouldn't enable these creative affordances.

**Q: What is the Engine Toolkit's budget cap for?**
Without budget caps, certain mode × subject combinations could activate too many engines, slowing processing and generating conflicting signals. The cap (different per mode) limits the number of T1+T2 engines that can run simultaneously. When the cap is exceeded, the lowest-priority T2 engines are demoted to T3 (standby weight 0.5), reducing their influence without disabling them entirely.

**Q: How does the learning log bridge learning and meta-learning?**
Every learning mode turn (M1-M5) records a LearningLogEntry capturing: engine results (patterns detected, comparisons made, prediction errors, meta-learning updates), memory contrast deltas (what was confirmed, contradicted, extended, or novel), and Phase 5 reward scores. These entries accumulate during learning sessions. When Homework mode runs, it fetches all unprocessed entries, batches them by subject, and processes them through its six-phase pipeline. The learning log is the system's "homework notebook" — raw notes taken during class that get organized later.

**Q: What does the Unsolved Buffer's stagnation mechanism accomplish?**
It creates a natural escalation path for difficult questions. A question enters the buffer during learning. Each session close increments its stagnation counter. After 5+ stagnation cycles (meaning the question has persisted through multiple sessions without resolution), it's flagged as a dream candidate. This routes it to the Dream pipeline, which processes it with elevated creative flexibility — CB1 plasticity, abstract association, schema-breaking connections. Questions that the logical, structured pipeline can't solve get sent to the creative, associative pipeline. It's the computational equivalent of "sleep on it."
